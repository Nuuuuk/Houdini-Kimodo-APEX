"""the node side of things: generate, status, stored poses, the cooks."""

import os
import time

import hou

from . import client, constraints, doc, skeleton

CACHE_DIR = "$HIP/kimodo"
_running = {}
_status = {}
ACTIVE_STAGES = ("starting", "loading", "generating", "canceling")
_motions = {}


def read_doc(node):
    return doc.loads(node.parm("kimodo_doc").unexpandedString())


def write_doc(node, d):
    node.parm("kimodo_doc").set(doc.dumps(d))


def set_status(node, text, stage="idle", progress=None, started=None, detail=""):
    sid = node.sessionId()
    previous = _status.get(sid, {}).get("stage")
    elapsed = None if started is None or stage in ACTIVE_STAGES else time.monotonic() - started
    _status[sid] = {
        "stage": stage,
        "text": text,
        "progress": progress,
        "started": started,
        "elapsed": elapsed,
        "detail": detail,
        "log": client.log_path() if stage == "error" else "",
    }
    with hou.undos.disabler():
        node.parm("status").set(text)
        if stage == "error":
            node.parm("error").set(text if not detail else f"{text}\n{detail}")
    if stage != previous and hou.isUIAvailable():
        severity = hou.severityType.Error if stage == "error" else hou.severityType.Message
        hou.ui.setStatusMessage(f"Kimodo Motion ({node.name()}): {text}", severity)


def get_status(node):
    """after a hip load only the saved text and error are known"""
    st = _status.get(node.sessionId())
    if st is None:
        error = node.parm("error").unexpandedString()
        st = {
            "stage": "error" if error else "idle",
            "text": error.split("\n", 1)[0] or node.evalParm("status"),
            "progress": None,
            "started": None,
            "elapsed": None,
            "detail": error.partition("\n")[2],
            "log": client.log_path() if error else "",
        }
    return st


def clear_error(node):
    had_error = bool(node.parm("error").unexpandedString())
    with hou.undos.disabler():
        node.parm("error").set("")
    if had_error:
        set_status(node, "Error dismissed.")


def doc_from_parms(node):
    """the doc with the model and generation settings from the parms"""
    d = read_doc(node)
    d["model"] = node.evalParm("model")
    frames = int(round(node.evalParm("duration") * doc.FPS))
    d["prompts"] = [{"id": "p1", "text": node.parm("prompt").unexpandedString(), "start_frame": 0, "end_frame": frames}]
    d["generation"] = {
        "seed": node.evalParm("seed"),
        "steps": node.evalParm("steps"),
        "cfg_text": node.evalParm("cfg_text"),
        "cfg_constraint": node.evalParm("cfg_constraint"),
        "num_samples": node.evalParm("num_samples"),
        "postprocess": bool(node.evalParm("postprocess")),
        "root_margin": node.evalParm("root_margin"),
    }
    return d


def constraints_json(d):
    if d["constraints"]:
        raise hou.NodeError("Constraint editing isn't available yet, clear the constraints first.")
    return constraints.to_kimodo_json(d["constraints"], None)


class _GenerateJob:
    """one press of generate, step() moves it on without blocking"""

    def __init__(self, node):
        self.node, self.client = node, client.Client(timeout=2.0)
        self.doc = doc_from_parms(node)
        doc.validate(self.doc)
        self.request = client.make_request(self.doc, constraints_json(self.doc))
        key = client.cache_key(self.request)
        self.paths = [f"{CACHE_DIR}/{key}/sample_{i}.npz" for i in range(self.request["num_samples"])]
        self.request["output_dir"] = os.path.normpath(hou.text.expandString(f"{CACHE_DIR}/{key}"))
        self.job_id, self.started, self.canceled, self.proc = None, time.monotonic(), False, None
        self.warning = "" if abs(hou.fps() - doc.FPS) < 1e-6 else f"Scene is {hou.fps():g} fps, Kimodo is 30 fps. "

    def status(self, text, stage, progress=None, detail=""):
        set_status(self.node, self.warning + text, stage, progress, self.started, detail)

    def fail(self, text, detail=""):
        self.status(text, "error", detail=detail or f"See {client.log_path()}")

    def finish(self, text):
        self.doc["result"] = {"npz": self.paths, "selected": min(self.node.evalParm("sample"), len(self.paths) - 1)}
        write_doc(self.node, self.doc)
        self.status(text, "done")

    def step(self):
        elapsed = time.monotonic() - self.started
        if self.job_id is None:
            if all(os.path.isfile(hou.text.expandString(p)) for p in self.paths):
                self.finish("Loaded cached result.")
                return True
            if self.canceled:
                self.status("Canceled.", "canceled")
                return True
            health = self.client.health()
            if health is None:
                if self.proc is None:
                    self.proc = client.start_service()
                client.check_exited(self.proc)
                self.status(f"Starting the Kimodo service... ({elapsed:.0f} s)", "starting")
                return False
            if health["status"] == "error":
                self.fail("The Kimodo service failed to load its models.", health["error"])
                return True
            if health["status"] != "ready":
                self.status(f"Loading models... ({elapsed:.0f} s)", "loading")
                return False
            self.client.register_session()
            self.job_id = self.client.generate(self.request)
        if self.canceled:
            self.client.cancel(self.job_id)
        job = self.client.job(self.job_id)
        if job["status"] in ("queued", "running"):
            stage = "canceling" if self.canceled else "generating"
            self.status(f"Generating... {job['progress'] * 100:.0f}%", stage, job["progress"])
            return False
        if job["status"] == "done":
            self.finish(f"Done in {elapsed:.1f} s.")
        elif job["status"] == "canceled":
            self.status("Canceled.", "canceled")
        else:
            self.fail("Generation failed.", job["error"])
        return True


def generate(node, wait=False, poll=0.25, timeout=600.0):
    """start generating, returns the job or None"""
    sid = node.sessionId()
    if sid in _running:
        set_status(node, "Already generating.")
        return None
    clear_error(node)
    try:
        gen = _GenerateJob(node)
    except (hou.Error, doc.DocError, client.ServiceError) as e:
        message = e.instanceMessage() if isinstance(e, hou.Error) else str(e)
        set_status(node, "Cannot generate.", "error", detail=message)
        return None
    _running[sid] = gen
    if gen.warning and hou.isUIAvailable():
        hou.ui.setStatusMessage(f"Kimodo Motion ({node.name()}): {gen.warning.strip()}", hou.severityType.Warning)
    set_busy(node, True)

    def finished():
        _running.pop(sid, None)
        try:
            set_busy(node, False)
        except hou.ObjectWasDeleted:
            pass

    if wait or not hou.isUIAvailable():
        deadline = time.monotonic() + timeout
        try:
            while not _safe_step(gen):
                if time.monotonic() > deadline:
                    gen.fail("Timed out waiting for the Kimodo service.")
                    break
                time.sleep(poll)
        finally:
            finished()
        return gen

    last = [0.0]

    def tick():
        if time.monotonic() - last[0] < poll:
            return
        last[0] = time.monotonic()
        try:
            done = _safe_step(gen)
        except hou.ObjectWasDeleted:
            done = True
        if done:
            hou.ui.removeEventLoopCallback(tick)
            finished()

    hou.ui.addEventLoopCallback(tick)
    return gen


def set_busy(node, busy):
    with hou.undos.disabler():
        node.parm("busy").set(int(busy))


def _safe_step(gen):
    try:
        return gen.step()
    except client.PortInUseError as e:
        gen.fail(str(e), f"{gen.client.base}/health did not answer as the Kimodo service.")
        return True
    except (client.ServiceError, OSError) as e:
        gen.fail("Kimodo service error.", str(e))
        return True


def cancel(node):
    """returns right away, the job stops at the next denoising step"""
    gen = _running.get(node.sessionId())
    if gen is None:
        set_busy(node, False)
        set_status(node, "Nothing to cancel.")
    else:
        gen.canceled = True
        gen.status("Canceling...", "canceling")


ANIMATE_NODE = "animate"
ANIMATE_STATE = "apexanimate"
_edit_restore = {}


def scene_viewer():
    return hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)


def is_editing(node):
    if not hou.isUIAvailable():
        return False
    viewer = scene_viewer()
    return (viewer is not None and viewer.currentState() == ANIMATE_STATE and viewer.currentNode() == node.node(ANIMATE_NODE))


def edit_pose(node):
    """start posing in the shown scene view"""
    if not read_doc(node)["result"]["npz"]:
        set_status(node, "Generate a motion first; poses are edited on top of it.")
        return False
    viewer = scene_viewer()
    if viewer is None:
        set_status(node, "Open a Scene Viewer to edit poses.")
        return False
    inner = node.node(ANIMATE_NODE)
    if not inner.isDisplayFlagSet():
        set_status(
            node,
            "This node was made with an older version of the asset; create a new Kimodo Motion node to edit poses.",
        )
        return False
    _edit_restore[node.sessionId()] = viewer.isPin()
    viewer.setPin(True)
    viewer.setPwd(node)
    viewer.setCurrentNode(inner, False)
    viewer.setCurrentState(ANIMATE_STATE)
    node.setCurrent(True, clear_all_selected=True)
    return is_editing(node)


def end_edit(node):
    """leave posing and show the node's own output again"""
    viewer = scene_viewer()
    if viewer is None:
        return
    viewer.setCurrentState("select")
    viewer.setPwd(node.parent())
    viewer.setPin(_edit_restore.pop(node.sessionId(), False))
    node.setCurrent(True, clear_all_selected=True)


def _motion(path):
    mtime = os.path.getmtime(path)
    cached = _motions.get(path)
    if cached is None or cached[0] != mtime:
        cached = _motions[path] = (mtime, skeleton.load_motion(path))
    return cached[1]


def cook(sop):
    """python sop: the shown sample at the current frame"""
    node = sop.parent()
    geo = sop.geometry()
    frame = hou.frame()
    d = read_doc(node)
    error = node.parm("error").unexpandedString()
    paths = d["result"]["npz"]
    path = hou.text.expandString(paths[min(max(node.evalParm("sample"), 0), len(paths) - 1)]) if paths else ""
    if not os.path.isfile(path):
        if error:
            raise hou.NodeError(error)
        if path:
            raise hou.NodeWarning(f"Result file is missing: {path}. Press Generate to recreate it.")
        return
    skeleton.build_geometry(geo, _motion(path), int(round(frame)) - 1)
    if error:
        raise hou.NodeWarning(f"The last Generate failed; this is the previous result.\n{error}")


def cook_rest(sop):
    """python sop: the rest skeleton the rig is built from"""
    skeleton.build_rest_geometry(sop.geometry())
