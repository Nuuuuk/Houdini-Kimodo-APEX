"""the node side of things: generate, status, stored poses, the cooks."""

import json
import os
import time

import hou
import numpy as np

from . import client, constraints, doc, rig, skeleton

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


_poses_cache = {}


def read_poses(node):
    """{frame: Pose}, a root only pose has local_rots None"""
    text = node.parm("kimodo_poses").unexpandedString()
    cached = _poses_cache.get(text)
    if cached is None:
        raw = json.loads(text) if text.strip() else {}
        cached = {
            int(f): constraints.Pose(
                None if p["rot"] is None else constraints.axis_angle_to_matrix(np.array(p["rot"])),
                np.array([np.nan if v is None else v for v in p["root"]], dtype=np.float64),
            )
            for f, p in raw.items()
        }
        _poses_cache.clear()
        _poses_cache[text] = cached
    return cached


def write_poses(node, poses):
    raw = {
        str(f): {
            "rot": None if p.local_rots is None else constraints.matrix_to_axis_angle(p.local_rots).tolist(),
            "root": [None if np.isnan(v) else float(v) for v in p.root_position],
        }
        for f, p in sorted(poses.items())
    }
    node.parm("kimodo_poses").set(json.dumps(raw))


def constraint_frames(d):
    return sorted({f for c in d["constraints"] for f in c["frames"]})


def pose_source(node, frames):
    """pose of each constraint frame: stored while unedited, else the rig's"""
    stored = read_poses(node)
    edited = set(rig.edited_frames(node, only=[f for f in frames if f in stored]))

    def pose(frame):
        s = stored.get(frame)
        if s is not None and frame not in edited and s.local_rots is not None:
            return s
        p = rig.pose(node, frame)
        if s is not None and frame not in edited:
            return constraints.Pose(
                p.local_rots, np.array([s.root_position[0], p.root_position[1], s.root_position[2]])
            )
        return p

    return pose


def constraints_json(node, d):
    return constraints.to_kimodo_json(d["constraints"], pose_source(node, constraint_frames(d)))


def _new_id(d):
    used = {c["id"] for c in d["constraints"]}
    return next(f"c{n}" for n in range(1, len(used) + 2) if f"c{n}" not in used)


def add_constraint(node, ctype, frame, auto=False):
    d = read_doc(node)
    if not 0 <= frame < doc.total_frames(doc_from_parms(node)):
        set_status(node, f"Frame {frame + 1} is outside the motion.")
        return None
    if any(c["type"] == ctype and frame in c["frames"] for c in d["constraints"]):
        return None
    entry = {"id": _new_id(d), "type": ctype, "frames": [frame]}
    if auto:
        entry["auto"] = True
    d["constraints"].append(entry)
    write_doc(node, d)
    set_status(node, f"{TYPE_LABELS[ctype]} keyframe added at frame {frame + 1}.")
    return entry["id"]


def delete_constraint(node, cid):
    d = read_doc(node)
    gone = [c for c in d["constraints"] if c["id"] == cid]
    if not gone:
        return
    d["constraints"] = [c for c in d["constraints"] if c["id"] != cid]
    freed = sorted(set(gone[0]["frames"]) - set(constraint_frames(d)))

    def apply():
        with hou.undos.group("Delete Kimodo constraint"):
            write_doc(node, d)
            poses = read_poses(node)
            if any(f in poses for f in freed):
                write_poses(node, {f: p for f, p in poses.items() if f not in freed})
            rig.snap(node, freed)

    _outside_edit(node, apply)
    set_status(node, f"{TYPE_LABELS[gone[0]['type']]} constraint deleted.")


def set_constraint_type(node, cid, ctype):
    d = read_doc(node)
    for c in d["constraints"]:
        if c["id"] == cid:
            c["type"] = ctype
            c.pop("auto", None)
    write_doc(node, d)
    set_status(node, f"Constraint set to {TYPE_LABELS[ctype]}.")


TYPE_LABELS = {
    "fullbody": "Full-Body",
    "root2d": "2D Root",
    "left-hand": "Left Hand",
    "right-hand": "Right Hand",
    "left-foot": "Left Foot",
    "right-foot": "Right Foot",
}


def update_auto_constraints(node, only=None):
    """auto constraints follow the edited frames"""
    d = read_doc(node)
    total = doc.total_frames(doc_from_parms(node))
    edited = {f for f in rig.edited_frames(node, only) if 0 <= f < total}
    constrained = set(constraint_frames(d))
    added = sorted(edited - constrained)
    removed = [
        c
        for c in d["constraints"]
        if c.get("auto") and c["frames"][0] not in edited and (only is None or c["frames"][0] in only)
    ]
    if not added and not removed:
        return
    d["constraints"] = [c for c in d["constraints"] if c not in removed]
    for f in added:
        d["constraints"].append({"id": _new_id(d), "type": "fullbody", "frames": [f], "auto": True})
    with hou.undos.disabler():
        write_doc(node, d)
    if added:
        set_status(
            node,
            f"Full-Body keyframe added at frame {added[0] + 1}."
            if len(added) == 1
            else f"Full-Body keyframes added at {len(added)} edited frames.",
        )


def sync_rig(node):
    """rebuild the rig's animation if it was made from another result"""
    if rig.stash(node) is None:
        return False
    tag, current = rig.source_tag(node), rig.stash_tag(node)
    if current == tag:
        return False

    def apply():
        with hou.undos.disabler():
            if rig.stash_tag(node) is None:
                rig.retag(node, tag)
            else:
                rig.merge(node, constraint_frames(read_doc(node)), tag)

    _outside_edit(node, apply)
    return current is not None


def _outside_edit(node, fn):
    editing = is_editing(node)
    if editing:
        end_edit(node)
    try:
        fn()
    finally:
        if editing:
            edit_pose(node)


def import_constraints(node, path):
    with open(path) as f:
        cons, poses = constraints.from_kimodo_json(json.load(f))
    d = doc_from_parms(node)
    last = max((c["frames"][-1] for c in cons), default=0)
    if last >= doc.total_frames(d):
        raise hou.Error(
            f"{os.path.basename(path)} has constraints up to frame {last + 1}, the motion has "
            f"{doc.total_frames(d)} frames. Increase the duration first."
        )
    stored = {}
    for c in cons:
        for frame, pose in poses[c["id"]].items():
            if pose.local_rots is not None or frame not in stored:
                stored[frame] = pose
    d["constraints"] = cons

    def apply():
        with hou.undos.group("Import Kimodo constraints"):
            write_doc(node, d)
            write_poses(node, stored)
            node.node(rig.ANIMATE).parm("animation").set(hou.Geometry())

    _outside_edit(node, apply)
    set_status(node, f"Imported {len(cons)} constraints from {os.path.basename(path)}.")


def export_constraints(node, path):
    d = read_doc(node)
    update_auto_constraints(node)
    with open(path, "w") as f:
        json.dump(constraints_json(node, read_doc(node)), f)
    set_status(node, f"Exported {len(d['constraints'])} constraints to {os.path.basename(path)}.")


def import_constraints_dialog(node):
    path = hou.ui.selectFile(title="Import Kimodo constraints", pattern="*.json", chooser_mode=hou.fileChooserMode.Read)
    if path:
        import_constraints(node, hou.text.expandString(path))


def export_constraints_dialog(node):
    path = hou.ui.selectFile(
        title="Export Kimodo constraints",
        pattern="*.json",
        default_value="constraints.json",
        chooser_mode=hou.fileChooserMode.Write,
    )
    if path:
        export_constraints(node, hou.text.expandString(path))


class _GenerateJob:
    """one press of generate, step() moves it on without blocking"""

    def __init__(self, node):
        self.node, self.client = node, client.Client(timeout=2.0)
        sync_rig(node)
        update_auto_constraints(node)
        self.doc = doc_from_parms(node)
        doc.validate(self.doc)
        self.request = client.make_request(self.doc, constraints_json(node, self.doc))
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
        d = read_doc(self.node)
        d.update({k: self.doc[k] for k in ("model", "prompts", "generation")})
        d["result"] = {"npz": self.paths, "selected": min(self.node.evalParm("sample"), len(self.paths) - 1)}
        with hou.undos.disabler():
            write_doc(self.node, d)
        sync_rig(self.node)
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
    if is_editing(node):
        end_edit(node)
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
    if rig.stash(node) is None:
        with hou.undos.disabler():
            rig.merge(node, [], rig.source_tag(node))
    sync_rig(node)
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
    update_auto_constraints(node)
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


def cook_rig_source(sop):
    """python sop: the motion the rig follows"""
    node = sop.parent()
    frame = int(round(hou.frame())) - 1
    d = read_doc(node)
    paths = d["result"]["npz"]
    path = hou.text.expandString(paths[min(max(node.evalParm("sample"), 0), len(paths) - 1)]) if paths else ""
    stored = read_poses(node).get(frame)
    if os.path.isfile(path):
        motion = _motion(path)
        i = min(max(frame, 0), motion.num_frames - 1)
        if stored is None:
            return skeleton.build_geometry(sop.geometry(), motion, i)
        local, root = motion.local_rots[i], motion.positions[i, 0].astype(np.float64)
    else:
        local, root = np.broadcast_to(np.eye(3), (skeleton.NUM_JOINTS, 3, 3)), skeleton.rest_positions()[0]
    if stored is not None:
        if stored.local_rots is not None:
            local = stored.local_rots
        root = np.where(np.isnan(stored.root_position), root, stored.root_position)
    positions, world = skeleton.fk(local, root)
    skeleton.write_geometry(sop.geometry(), positions, np.swapaxes(world, -1, -2).reshape(skeleton.NUM_JOINTS, 9))
