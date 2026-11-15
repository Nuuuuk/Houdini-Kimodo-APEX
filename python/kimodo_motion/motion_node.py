"""the node side of things: generate, status, stored poses, the cooks."""

import os
import time

import hou

from . import client, constraints, doc, skeleton

CACHE_DIR = "$HIP/kimodo"
_running = {}
_motions = {}


def read_doc(node):
    return doc.loads(node.parm("kimodo_doc").unexpandedString())


def write_doc(node, d):
    node.parm("kimodo_doc").set(doc.dumps(d))


def set_status(node, text):
    with hou.undos.disabler():
        node.parm("status").set(text)


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

    def status(self, text):
        set_status(self.node, self.warning + text)

    def finish(self, text):
        self.doc["result"] = {"npz": self.paths, "selected": min(self.node.evalParm("sample"), len(self.paths) - 1)}
        write_doc(self.node, self.doc)
        self.status(text)

    def step(self):
        elapsed = time.monotonic() - self.started
        if self.job_id is None:
            if all(os.path.isfile(hou.text.expandString(p)) for p in self.paths):
                self.finish("Loaded cached result.")
                return True
            if self.canceled:
                self.status("Canceled.")
                return True
            health = self.client.health()
            if health is None:
                if self.proc is None:
                    self.proc = client.start_service()
                client.check_exited(self.proc)
                self.status(f"Starting the Kimodo service... ({elapsed:.0f} s)")
                return False
            if health["status"] == "error":
                self.status(f"Service failed to load: {health['error']} (see {client.log_path()})")
                return True
            if health["status"] != "ready":
                self.status(f"Loading models... ({elapsed:.0f} s)")
                return False
            self.client.register_session()
            self.job_id = self.client.generate(self.request)
        if self.canceled:
            self.client.cancel(self.job_id)
        job = self.client.job(self.job_id)
        if job["status"] in ("queued", "running"):
            self.status(f"Generating... {job['progress'] * 100:.0f}%")
            return False
        if job["status"] == "done":
            self.finish(f"Done in {elapsed:.1f} s.")
        elif job["status"] == "canceled":
            self.status("Canceled.")
        else:
            self.status(f"Generation failed: {job['error']}")
        return True


def generate(node, wait=False, poll=0.25, timeout=600.0):
    """start generating, returns the job or None"""
    sid = node.sessionId()
    if sid in _running:
        set_status(node, "Already generating.")
        return None
    try:
        gen = _GenerateJob(node)
    except (hou.Error, doc.DocError, client.ServiceError) as e:
        set_status(node, f"Cannot generate: {e}")
        return None
    _running[sid] = gen
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
                    gen.status("Timed out waiting for the service.")
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
    except (client.ServiceError, OSError) as e:
        gen.status(f"Service error: {e}")
        return True


def cancel(node):
    """returns right away, the job stops at the next denoising step"""
    gen = _running.get(node.sessionId())
    if gen is None:
        set_busy(node, False)
        set_status(node, "Nothing to cancel.")
    else:
        gen.canceled = True
        gen.status("Canceling...")


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
    paths = d["result"]["npz"]
    if not paths:
        return
    sample = min(max(node.evalParm("sample"), 0), len(paths) - 1)
    path = hou.text.expandString(paths[sample])
    if not os.path.isfile(path):
        raise hou.NodeWarning(f"Result file is missing: {path}. Press Generate to recreate it.")
    skeleton.build_geometry(geo, _motion(path), int(round(frame)) - 1)
