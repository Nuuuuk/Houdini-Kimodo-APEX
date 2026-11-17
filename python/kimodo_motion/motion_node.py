"""the node side of things: generate, status, stored poses, the cooks."""

import base64
import hashlib
import json
import os
import time
import zlib

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
    return _parse_poses(node.parm("kimodo_poses").unexpandedString())


def _parse_poses(text):
    cached = _poses_cache.get(text)
    if cached is None:
        raw = json.loads(text) if text.strip() else {}
        cached = {
            int(f): constraints.Pose(
                None if p["rot"] is None else constraints.axis_angle_to_matrix(np.array(p["rot"])),
                np.array([np.nan if v is None else v for v in p["root"]], dtype=np.float64),
                None if p.get("root2d") is None else tuple(p["root2d"]),
            )
            for f, p in raw.items()
        }
        _poses_cache.clear()
        _poses_cache[text] = cached
    return cached


def write_poses(node, poses):
    old_text = node.parm("kimodo_poses").unexpandedString()
    old, old_raw = _parse_poses(old_text), (json.loads(old_text) if old_text.strip() else {})
    raw = {}
    for f, p in sorted(poses.items()):
        if old.get(f) is p:
            raw[str(f)] = old_raw[str(f)]
            continue
        raw[str(f)] = {
            "rot": None if p.local_rots is None else constraints.matrix_to_axis_angle(p.local_rots).tolist(),
            "root": [None if np.isnan(v) else float(v) for v in p.root_position],
        }
        if p.root_2d is not None:
            raw[str(f)]["root2d"] = [float(v) for v in p.root_2d]
    text = json.dumps(raw) if raw else ""
    if old_text != text:
        node.parm("kimodo_poses").set(text)
    return _parse_poses(text)


_captured = {}


def capture_poses(node, d, reference=None, record=False):
    """store the pose each constraint frame sends now"""
    frames = constraint_frames(d)
    stored = dict(read_poses(node))
    top = rig.stash(node)
    if top is None:
        edited = set()
    elif reference is not None:
        edited = rig._differing_frames(rig.channels(top), rig.channels(reference), frames)
    elif rig.stash_tag(node) in (None, rig.source_tag(node)):
        edited = set(rig.edited_frames(node, only=frames))
    else:
        edited = set(frames)
    mine = rig.channels(top) if top is not None else None
    taken = _captured.setdefault(node.sessionId(), {})
    for frame in frames:
        if frame in stored and frame not in edited:
            continue
        keys = None if mine is None else np.array([prim.eval(float(frame + 1)) for prim in mine.values()]).tobytes()
        if frame in stored and keys is not None and taken.get(frame) == keys:
            continue
        pose = rig.pose(node, frame)
        stored[frame] = constraints.Pose(pose.local_rots, pose.root_position, pose.root_2d)
        taken[frame] = keys
    stored = {frame: pose for frame, pose in stored.items() if frame in frames}
    if record:
        stored = write_poses(node, stored)
        write_rig_keep(node, d, record=True)
    else:
        with hou.undos.disabler():
            stored = write_poses(node, stored)
            write_rig_keep(node, d)
    return stored


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


def constraints_key(cjson):
    return hashlib.sha256(json.dumps(cjson, sort_keys=True).encode()).hexdigest()[:16]


NO_CONSTRAINTS = constraints_key([])


def kept_keys(d):
    """{frame: channels whose keys are ours there, None for all}"""
    types = {}
    for c in d["constraints"]:
        for frame in c["frames"]:
            types.setdefault(frame, set()).add(c["type"])
    return {frame: rig.kept_channels(found) for frame, found in types.items()}


def pinned_at(d, frame):
    """rig controls the constraints on a frame pin, None for all of them"""
    types = {c["type"] for c in d["constraints"] if frame in c["frames"]}
    return rig.pinned_controls(types) if types else None


def pinned_note(d, frame):
    types = [c["type"] for c in d["constraints"] if frame in c["frames"]]
    if not types or "fullbody" in types:
        return ""
    labels = ", ".join(TYPE_LABELS[t] for t in doc.CONSTRAINT_TYPES if t in types)
    return f"Frame {frame + 1} pins {labels}: only those controls are shown. Add a Full-Body constraint to pose more."


def release_unpinned(node, d):
    """what a frame doesn't pin goes back to the generated motion"""
    keep = {frame: names for frame, names in kept_keys(d).items() if names is not None}
    if keep and not is_editing(node):
        with hou.undos.disabler():
            rig.snap(node, list(keep), keep)


def _rig_keep_text(d, stored):
    keep = {}
    for c in d["constraints"]:
        for frame in c["frames"]:
            if frame in stored:
                keep.setdefault(frame, set()).add(c["type"])
    text = json.dumps({str(f): sorted(t) for f, t in sorted(keep.items())})
    return "" if text == "{}" else text


def write_rig_keep(node, d, record=False):
    text = _rig_keep_text(d, read_poses(node))
    if node.parm("rig_keep").unexpandedString() != text:
        if record:
            node.parm("rig_keep").set(text)
        else:
            with hou.undos.disabler():
                node.parm("rig_keep").set(text)


def _change_constraints(node, before, d, label, freed=(), force=False):
    stored = read_poses(node)
    frames = set(constraint_frames(d))
    left = {frame: pose for frame, pose in stored.items() if frame in frames}
    rebuild = force or len(left) != len(stored) or _rig_keep_text(d, left) != node.parm("rig_keep").unexpandedString()

    def apply():
        with hou.undos.group(label):
            if rebuild:
                captured = capture_poses(node, before, record=True)
                write_poses(node, {frame: pose for frame, pose in captured.items() if frame in frames})
            write_doc(node, d)
            write_rig_keep(node, d, record=True)
            if rig.stash(node) is not None:
                if rig.stash_tag(node) != rig.source_tag(node):
                    rig.merge(node, {}, rig.source_tag(node))
                else:
                    rig.snap(node, list(freed))

    _outside_edit(node, apply)


def update_unapplied(node):
    """unapplied: do the constraints differ from the ones the result was made with"""
    d = read_doc(node)
    try:
        unapplied = int(constraints_key(constraints_json(node, d)) != d["result"].get("applied", NO_CONSTRAINTS))
    except rig.NotReady:
        _retry.add(node.sessionId())
        return False
    if node.evalParm("unapplied") != unapplied:
        with hou.undos.disabler():
            node.parm("unapplied").set(unapplied)
    return True


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
    before = json.loads(json.dumps(d))
    entry = {"id": _new_id(d), "type": ctype, "frames": [frame]}
    if auto:
        entry["auto"] = True
    d["constraints"].append(entry)
    if frame in read_poses(node):
        _change_constraints(node, before, d, "Add Kimodo constraint")
    else:
        write_doc(node, d)
    update_unapplied(node)
    set_status(node, f"{TYPE_LABELS[ctype]} keyframe added at frame {frame + 1}.")
    return entry["id"]


def ensure_stash(node):
    sync_rig(node)
    if rig.stash(node) is None:
        with hou.undos.disabler():
            rig.merge(node, {}, rig.source_tag(node))


def delete_constraint(node, cid):
    before = read_doc(node)
    gone = [c for c in before["constraints"] if c["id"] == cid]
    if not gone:
        return
    d = dict(before, constraints=[c for c in before["constraints"] if c["id"] != cid])
    freed = sorted(set(gone[0]["frames"]) - set(constraint_frames(d)))
    _change_constraints(node, before, d, "Delete Kimodo constraint", freed)
    update_unapplied(node)
    set_status(node, f"{TYPE_LABELS[gone[0]['type']]} constraint deleted.")


def set_constraint_type(node, cid, ctype):
    before = read_doc(node)
    d = json.loads(json.dumps(before))
    for c in d["constraints"]:
        if c["id"] == cid:
            c["type"] = ctype
            c.pop("auto", None)
    _change_constraints(node, before, d, "Change Kimodo constraint type", force=True)
    update_unapplied(node)
    set_status(node, f"Constraint set to {TYPE_LABELS[ctype]}.")


_copied = {}


def copy_pose(node, frame):
    ensure_stash(node)
    _copied.update(keys=rig.keys_at(node, frame), frame=frame)
    set_status(node, f"Pose of frame {frame + 1} copied.")


def has_copied_pose():
    return bool(_copied)


def paste_pose(node, ctype, frame, keep_root=False):
    """new constraint of this type at a frame, with the copied pose"""
    if not _copied:
        set_status(node, "No pose copied yet: right-click a marker, Copy Pose.")
        return None
    if not 0 <= frame < doc.total_frames(doc_from_parms(node)):
        set_status(node, f"Frame {frame + 1} is outside the motion.")
        return None
    ensure_stash(node)
    names = rig.pasted_channels(ctype, keep_root)
    keys = _copied["keys"]
    if isinstance(names, tuple):
        keys = {name: value for name, value in keys.items() if name not in names[1]}
    elif names is not None:
        keys = {name: value for name, value in keys.items() if name in names}
    d = read_doc(node)
    entry = next((c for c in d["constraints"] if c["type"] == ctype and frame in c["frames"]), None)
    if entry is None:
        entry = {"id": _new_id(d), "type": ctype, "frames": [frame]}
        d["constraints"].append(entry)
    entry.pop("auto", None)

    def apply():
        with hou.undos.group("Paste Kimodo pose"):
            rig.set_keys(node, frame, keys)
            write_doc(node, d)
            write_rig_keep(node, d)

    _outside_edit(node, apply)
    update_unapplied(node)
    where = " at its own root position" if keep_root else ""
    set_status(node, f"Pose of frame {_copied['frame'] + 1} pasted as a {TYPE_LABELS[ctype]} keyframe at frame {frame + 1}{where}.")
    return entry["id"]


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
    try:
        edited = {f for f in rig.edited_frames(node, only) if 0 <= f < total}
    except rig.NotReady:
        _retry.add(node.sessionId())
        return False
    constrained = set(constraint_frames(d))
    added = sorted(edited - constrained)
    stored = read_poses(node)
    removed = [
        c
        for c in d["constraints"]
        if c.get("auto")
        and c["frames"][0] not in edited
        and c["frames"][0] not in stored
        and (only is None or c["frames"][0] in only)
    ]
    if not added and not removed:
        return True
    d["constraints"] = [c for c in d["constraints"] if c not in removed]
    for f in added:
        d["constraints"].append({"id": _new_id(d), "type": "fullbody", "frames": [f], "auto": True})
    with hou.undos.disabler():
        write_doc(node, d)
    update_unapplied(node)
    if added:
        set_status(
            node,
            f"Full-Body keyframe added at frame {added[0] + 1}."
            if len(added) == 1
            else f"Full-Body keyframes added at {len(added)} edited frames.",
        )
    return True


def embed_results(node, paths):
    """compressed copy of the result files in the node"""
    stored = {}
    for path in paths:
        with open(hou.text.expandString(path), "rb") as f:
            stored[path] = base64.b64encode(zlib.compress(f.read(), 6)).decode("ascii")
    with hou.undos.disabler():
        node.parm("kimodo_results").set(json.dumps(stored))


_embedded = {}


def result_file(node, path):
    """the result file, put back from the node's copy if it's missing"""
    expanded = hou.text.expandString(path)
    if not path or os.path.isfile(expanded):
        return expanded
    text = node.parm("kimodo_results").unexpandedString()
    if text not in _embedded:
        _embedded.clear()
        _embedded[text] = json.loads(text) if text else {}
    stored = _embedded[text].get(path)
    if stored is not None:
        try:
            os.makedirs(os.path.dirname(expanded), exist_ok=True)
            with open(expanded + ".restoring", "wb") as f:
                f.write(zlib.decompress(base64.b64decode(stored)))
            os.replace(expanded + ".restoring", expanded)
        except OSError:
            pass
    return expanded


def shown_result(node):
    """'<request key>:<sample>' of the shown sample, '' while there is none"""
    paths = read_doc(node)["result"]["npz"]
    return paths[min(max(node.evalParm("sample"), 0), len(paths) - 1)] if paths else ""


def sync_rig(node):
    """rebuild the rig's animation if it was made from another result"""
    d = read_doc(node)
    sid = node.sessionId()
    top = rig.stash(node)
    current = rig.stash_tag(node)
    made_from = rig._bake_cache.get(sid)
    reference = made_from[1] if made_from is not None and made_from[0] == current else None
    shown = shown_result(node)
    if node.parm("rig_npz").unexpandedString() != shown:
        with hou.undos.disabler():
            node.parm("rig_npz").set(shown)
    write_rig_keep(node, d)
    if top is None:
        return False
    tag, damaged = rig.source_tag(node), rig.damaged(top)
    if current == tag and not damaged:
        return False
    if current is None and not damaged and is_editing(node):
        return False

    def apply():
        with hou.undos.disabler():
            if damaged:
                rig.repair(node, current if current is not None else tag)
                set_status(node, "The stored poses listed the character as removed; that was repaired.")
            if rig.stash_tag(node) is None:
                rig.retag(node, rig.source_tag(node))
            if rig.stash_tag(node) != rig.source_tag(node):
                capture_poses(node, d, reference)
                rig.merge(node, {}, rig.source_tag(node))
                _captured.pop(sid, None)

    _outside_edit(node, apply)
    return current is not None


def _outside_edit(node, fn):
    viewer = editing_viewer(node)
    sid = node.sessionId()
    held = sid in _switching
    _switching.add(sid)
    try:
        leave_state(node)
        fn()
    finally:
        if viewer is not None and viewer.isCurrentTab():
            viewer.setCurrentState(ANIMATE_STATE)
        if not held:
            _switching.discard(sid)


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
            write_rig_keep(node, d)
            node.parm(rig.STASH).set(hou.Geometry())

    _outside_edit(node, apply)
    update_unapplied(node)
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
        stored = capture_poses(node, self.doc)
        if rig.stash(node) is not None and rig.stash_tag(node) != rig.source_tag(node):
            with hou.undos.disabler():
                rig.merge(node, {}, rig.source_tag(node))
            _captured.pop(node.sessionId(), None)
        cjson = constraints.to_kimodo_json(self.doc["constraints"], stored.__getitem__)
        self.request = client.make_request(self.doc, cjson)
        self.applied = constraints_key(cjson)
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
        d["result"] = {
            "npz": self.paths,
            "selected": min(self.node.evalParm("sample"), len(self.paths) - 1),
            "applied": self.applied,
        }
        with hou.undos.disabler():
            write_doc(self.node, d)
        embed_results(self.node, self.paths)
        sync_rig(self.node)
        update_unapplied(self.node)
        self.status(text, "done")

    def step(self):
        elapsed = time.monotonic() - self.started
        if self.job_id is None:
            if all(os.path.isfile(result_file(self.node, p)) for p in self.paths):
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
    leave_state(node)
    clear_error(node)
    try:
        gen = _GenerateJob(node)
    except (hou.Error, doc.DocError, client.ServiceError, rig.NotReady) as e:
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
    except client.ServiceExited as e:
        gen.fail("Kimodo service stopped while loading. Press Generate to retry.", str(e))
        return True
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


ANIMATE_STATE = "apexanimate"
_switching = set()
_retry = set()
_changes = {}
_watched = {}
_shown = {}
WATCHED_PARMS = ("animation", "kimodo_doc", "kimodo_poses", "sample")
WATCH_INTERVAL = 0.2


def scene_viewers():
    return [p for p in hou.ui.paneTabs() if p.type() == hou.paneTabType.SceneViewer]


def editing_viewers(node):
    """scene viewers posing this node, shown ones first"""
    if not hou.isUIAvailable():
        return []
    found = [v for v in scene_viewers() if v.currentState() == ANIMATE_STATE and v.currentNode() == node]
    return sorted(found, key=lambda v: not v.isCurrentTab())


def editing_viewer(node):
    return next(iter(editing_viewers(node)), None)


def _leave(viewer):
    pane = viewer.pane()
    front = None if viewer.isCurrentTab() or pane is None else pane.currentTab()
    if front is not None:
        viewer.setIsCurrentTab()
    try:
        viewer.enterViewState()
    finally:
        if front is not None:
            front.setIsCurrentTab()


def leave_state(node):
    for viewer in editing_viewers(node):
        _leave(viewer)


def is_editing(node):
    return editing_viewer(node) is not None


def edit_pose(node):
    """start posing in the shown scene view"""
    if is_editing(node):
        return True
    viewer = next((v for v in scene_viewers() if v.isCurrentTab() and v.pwd() == node.parent()), None)
    if viewer is None:
        set_status(node, f"Show {node.parent().path()} in a Scene View to edit poses.")
        return False
    if not node.isDisplayFlagSet():
        set_status(node, "Set the display flag on this node to edit poses.")
        return False
    try:
        ensure_stash(node)
    except rig.NotReady as e:
        set_status(node, f"The rig is not ready: {e}. Try again.")
        return False
    if viewer.currentNode() != node:
        node.setCurrent(True, clear_all_selected=True)
    viewer.setCurrentState(ANIMATE_STATE)
    watch(node)
    return is_editing(node)


def end_edit(node):
    """leave posing and show the node's own output again"""
    leave_state(node)
    finish_edit(node)


def finish_edit(node):
    """bring the node up to date with its poses"""
    sid = node.sessionId()
    if sid in _switching:
        return False
    _retry.discard(sid)
    try:
        if not node.evalParm("busy"):
            sync_rig(node)
        update_auto_constraints(node)
        release_unpinned(node, read_doc(node))
        update_unapplied(node)
    except rig.NotReady:
        _retry.add(sid)
    except (hou.Error, doc.DocError) as e:
        message = e.instanceMessage() if isinstance(e, hou.Error) else str(e)
        set_status(node, f"Could not read the edited poses: {message}")
    return sid not in _retry


INTERRUPTED = "The generation that ran when this scene was saved did not finish. Press Generate."


def node_ready(node):
    """OnCreated / OnLoaded"""
    if node.evalParm("busy") and node.sessionId() not in _running:
        set_busy(node, False)
        with hou.undos.disabler():
            node.parm("status").set(INTERRUPTED)
    watch(node)


def watch(node):
    """follow the node while the ui runs"""
    if not hou.isUIAvailable():
        return
    sid = node.sessionId()
    if sid in _watched:
        return
    _watched[sid] = [False, -1]
    _changes.setdefault(sid, 0)
    node.addEventCallback((hou.nodeEventType.ParmTupleChanged,), _on_parm_changed)
    if _poll not in hou.ui.eventLoopCallbacks():
        hou.ui.addEventLoopCallback(_poll)


def _on_parm_changed(node, event_type, **kwargs):
    parm_tuple = kwargs.get("parm_tuple")
    if parm_tuple is None or parm_tuple.name() in WATCHED_PARMS:
        _changes[node.sessionId()] = _changes.get(node.sessionId(), 0) + 1


_last_poll = [0.0]


def _poll():
    """one look at a watched node"""
    now = time.monotonic()
    if now - _last_poll[0] < WATCH_INTERVAL:
        return
    _last_poll[0] = now
    for sid in list(_watched):
        node = hou.nodeBySessionId(sid)
        if node is None:
            del _watched[sid]
            continue
        try:
            poll(node)
        except hou.ObjectWasDeleted:
            del _watched[sid]


def show_pinned_controls(node, viewer):
    """offer only the controls the current frame pins"""
    sid = node.sessionId()
    wanted = pinned_at(read_doc(node), int(round(hou.frame())) - 1)
    key = None if wanted is None else frozenset(wanted)
    if sid in _shown and _shown[sid] == key:
        return
    names = rig.pose_controls() if wanted is None else wanted
    viewer.runStateCommand(
        "showControls",
        {
            "controls": [rig.CONTROLS_PATH + name for name in sorted(names)],
            "controls_in_tree_format": False,
            "pick_modifier": hou.pickModifier.Replace,
        },
    )
    if wanted is not None:
        import apex.ui.statecommandutils as scu

        selected = list(scu.getAnimateState(viewer).control_selection)
        still = [path for path in selected if path.rsplit("/", 1)[-1] in wanted]
        if len(still) != len(selected):
            viewer.runStateCommand(
                "selectControls",
                {"controls": still, "controls_in_tree_format": False, "pick_modifier": hou.pickModifier.Replace},
            )
    _shown[sid] = key


def poll(node):
    """one look at a watched node"""
    sid = node.sessionId()
    seen = _watched[sid]
    if sid in _switching:
        return
    viewers = editing_viewers(node)
    displayed = node.isDisplayFlagSet()
    for viewer in [v for v in viewers if not displayed or not v.isCurrentTab()]:
        _leave(viewer)
        viewers.remove(viewer)
        set_status(
            node,
            "Pose editing ended: "
            + ("its Scene View tab is no longer shown." if displayed else "the node is no longer displayed."),
        )
    viewer = viewers[0] if viewers else None
    editing, changes = viewer is not None, _changes.get(sid, 0)
    if editing:
        try:
            show_pinned_controls(node, viewer)
        except hou.Error:
            pass
    else:
        _shown.pop(sid, None)
    if (seen[0] and not editing) or changes != seen[1] or sid in _retry:
        if finish_edit(node):
            seen[1] = _changes.get(sid, 0)
    seen[0] = editing


def _motion(path):
    mtime = os.path.getmtime(path)
    cached = _motions.get(path)
    if cached is None or cached[0] != mtime:
        cached = _motions[path] = (mtime, skeleton.load_motion(path))
    return cached[1]


def _shown_path(node):
    paths = read_doc(node)["result"]["npz"]
    return result_file(node, paths[min(max(node.evalParm("sample"), 0), len(paths) - 1)]) if paths else ""


def cook(sop):
    """python sop: the shown sample at the current frame"""
    frame = hou.frame()
    path = _shown_path(sop.parent())
    if os.path.isfile(path):
        skeleton.build_geometry(sop.geometry(), _motion(path), int(round(frame)) - 1)


UNAPPLIED_WARNING = "Edited poses not applied. Press Generate."


def cook_notice(sop):
    """python sop at the end, raises the asset's warning or error"""
    node = sop.parent()
    error = node.parm("error").unexpandedString()
    has_output = len(sop.geometry().points()) > 0
    if node.evalParm("unapplied"):
        raise hou.NodeWarning(
            UNAPPLIED_WARNING if not error else f"{UNAPPLIED_WARNING}\nThe last Generate failed.\n{error}"
        )
    if not has_output:
        if error:
            raise hou.NodeError(error)
        path = _shown_path(node)
        if path:
            raise hou.NodeWarning(f"Result file is missing: {path}. Press Generate to recreate it.")
    elif error:
        raise hou.NodeWarning(f"The last Generate failed; this is the previous result.\n{error}")


def cook_rest(sop):
    """python sop: the rest skeleton the rig is built from"""
    skeleton.build_rest_geometry(sop.geometry())


_keep_cache = {}


def _rig_keep(text):
    if text not in _keep_cache:
        _keep_cache.clear()
        _keep_cache[text] = {int(f): types for f, types in json.loads(text or "{}").items()}
    return _keep_cache[text]


def cook_rig_source(sop):
    """python sop: the motion the rig follows"""
    node = sop.parent()
    frame = int(round(hou.frame())) - 1
    node.evalParm("rig_npz")
    path = result_file(node, node.parm("rig_npz").unexpandedString())
    stored = read_poses(node).get(frame)
    types = _rig_keep(node.parm("rig_keep").unexpandedString()).get(frame)
    if os.path.isfile(path):
        motion = _motion(path)
        i = min(max(frame, 0), motion.num_frames - 1)
        if stored is None:
            positions, transform = skeleton.pose_arrays(motion, i)
            return skeleton.write_geometry(sop.geometry(), positions, transform, with_root=True)
        local, root = motion.local_rots[i], motion.positions[i, 0].astype(np.float64)
    else:
        local, root = np.broadcast_to(np.eye(3), (skeleton.NUM_JOINTS, 3, 3)), skeleton.rest_positions()[0]
    if stored is not None:
        positions, world = skeleton.composed_pose(
            local, root, stored.local_rots, stored.root_position, stored.root_2d, types or ()
        )
    else:
        positions, world = skeleton.fk(local, root)
    skeleton.write_geometry(
        sop.geometry(), positions, np.swapaxes(world, -1, -2).reshape(skeleton.NUM_JOINTS, 9), with_root=True
    )


def cook_rig_output(sop):
    """python sop: the rig's pose, shown while constraints are unapplied"""
    source = sop.inputGeometry(0)
    geo = sop.geometry()
    geo.clear()
    if source is not None and len(source.points()):
        pose = rig.pose_from_geometry(source)
        positions, world = skeleton.fk(pose.local_rots, pose.root_position)
        skeleton.write_geometry(geo, positions, np.swapaxes(world, -1, -2).reshape(skeleton.NUM_JOINTS, 9))
