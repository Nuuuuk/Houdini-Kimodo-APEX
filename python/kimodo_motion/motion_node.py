"""the node side of things: generate, status, stored poses, the cooks."""

import base64
import contextlib
import functools
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time

import hou
import numpy as np

from . import body, client, constraints, doc, examples, rig, rootpath, skeleton, timeline


class _NodeState:
    def __init__(self):
        self.running = None
        self.status = None
        self.watched = False
        self.was_editing = False
        self.finished = -1
        self.changes = 0
        self.retry = False
        self.ended = False
        self.watch_error = None
        self.shown = {}
        self.redo_seen = None
        self.drawn_changed = None
        self.before_interruption = None
        self.reported = None
        self.noticed = None
        self.sign = None
        self.undone = []
        # {stash tag: {frame: {rig keys: the stored pose they stand for}}}
        self.pose_of = {}
        self.was_auto = set()
        self.made = {}
        # rig changes not made here (undo, redo, the animate state), None until tracked
        self.foreign = None
        self.rig_motion = None


_nodes = {}


def _of(node):
    return _nodes.setdefault(node.sessionId(), _NodeState())


ACTIVE_STAGES = ("starting", "loading", "generating", "canceling")


def read_doc(node):
    return doc.loads(node.parm("kimodo_doc").unexpandedString())


def write_doc(node, d):
    node.parm("kimodo_doc").set(doc.dumps(d))


def say_once(text):
    """drops sentences or lines that only repeat an earlier one"""
    lines = []
    for line in text.split("\n"):
        kept, parts = [], []
        for part in re.split(r"(?<=[.;])\s+", line):
            words = " ".join(re.findall(r"\w+", part.lower()))
            if words and not any(k == words or k.endswith(" " + words) for k in kept):
                kept.append(words)
                parts.append(part.strip())
        if not parts:
            parts = [""] if lines and lines[-1] else []
        if parts and (not lines or " ".join(parts) != lines[-1]):
            lines.append(" ".join(parts))
    return "\n".join(lines)


_posted = set()


# houdini keeps timers running while it cooks, so node work waits for idle
def when_idle(key, fn):
    """run fn from the event queue, between cooks, once per key"""
    if not hou.isUIAvailable():
        return fn()
    if key in _posted:
        return
    _posted.add(key)

    def run():
        _posted.discard(key)
        fn()

    hou.ui.postEventCallback(run)


def _once_per(key_of):
    """keeps fn(node)'s value until key_of(node) changes"""

    def wrap(fn):
        last = []

        @functools.wraps(fn)
        def call(node):
            key = key_of(node)
            if not last or last[0] != key:
                last[:] = [key, fn(node)]
            return last[1]

        return call

    return wrap


def set_status(node, text, stage="idle", progress=None, started=None, detail=""):
    mine = _of(node)
    text, detail = say_once(text), say_once(detail)
    previous = (mine.status or {}).get("stage")
    elapsed = None if started is None or stage in ACTIVE_STAGES else time.monotonic() - started
    mine.status = {
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
    st = _of(node).status
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


# kimodo frame i is shown at houdini frame i + 1
def current_frame():
    """kimodo frame of houdini's current frame"""
    return int(round(hou.frame())) - 1


def pose_hint(node):
    """the hint line: how to start posing, or what the posed frame pins"""
    if not is_editing(node):
        return "To edit pose: hover the viewer and press Enter, or double-click a keyframe marker."
    frame = current_frame()
    on_frame = _types(node).get(frame, ())
    types = [t for t in doc.CONSTRAINT_TYPES if t in on_frame]
    finish = "Esc to finish."
    if not types:
        return f"Posing frame {frame + 1}. {finish}"
    if "fullbody" in types:
        return f"Posing frame {frame + 1}: Full-Body pins the whole pose. {finish}"
    hands, feet = sum(t.endswith("hand") for t in types), sum(t.endswith("foot") for t in types)
    parts = [("the hand", "both hands")[hands - 1]] if hands else []
    parts += [("the foot", "both feet")[feet - 1]] if feet else []
    parts += ["COG", "heading"] if hands or feet else []
    parts += ["the Root disk's place on the ground"] if "root2d" in types else []
    labels = [TYPE_LABELS[t] for t in types]
    names = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
    pinned = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Posing frame {frame + 1}: {names} {'pins' if len(labels) == 1 else 'pin'} {pinned}. {finish}"


def clear_error(node):
    had_error = bool(node.parm("error").unexpandedString())
    with hou.undos.disabler():
        node.parm("error").set("")
    if had_error:
        set_status(node, "Error dismissed.")


def selected_prompt(node, d):
    """index of the selected clip, always one"""
    return min(max(node.evalParm("selected_prompt"), 0), len(d["prompts"]) - 1)


def motion_frames(node):
    return rig.motion_frames(node)


def doc_from_parms(node):
    """the doc with the model and generation settings from the parms"""
    d = read_doc(node)
    d["model"] = node.evalParm("model")
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


def _listen(node, callback):
    if any(heard == callback for _, heard in node.eventCallbacks()):
        return False
    node.addEventCallback((hou.nodeEventType.ParmTupleChanged,), callback)
    return True


def _track(node):
    link = _of(node)
    if _listen(node, _on_stash_changed) and link.foreign is not None:
        link.foreign += 1
    if link.foreign is None:
        link.foreign = 0


def _on_stash_changed(node, event_type, **kwargs):
    parm_tuple = kwargs.get("parm_tuple")
    if (parm_tuple is None or parm_tuple.name() == rig.STASH) and not rig.writing(node):
        _of(node).foreign = (_of(node).foreign or 0) + 1


def _records(node, tag=None):
    kept = _of(node).pose_of
    tag = tag or rig.tag_for(node, shown_result(node))
    if tag not in kept and len(kept) >= rig.MAX_BAKES:
        kept.pop(next(iter(kept)))
    return kept.setdefault(tag, {})


def _undone_animation(node, table=None):
    link, top = _of(node), rig.stash(node)
    if top is None or rig.damaged(top) or not link.foreign:
        return None
    tag, shown, bases = rig.stash_tag(node), rig.tag_for(node, shown_result(node)), rig.bases(node)
    if tag == shown:
        others = [t for t in bases if t != shown and t in link.pose_of]
        if not others or shown not in bases:
            return None
        names, mine = table if table is not None else rig.stash_table(node)

        def on_bake(t):
            _, base_names, base = bases[t]
            return (
                -1
                if base_names != names or base.shape != mine.shape
                else int((np.abs(mine - base).max(axis=0) <= rig.EPS).sum())
            )

        tag = max(others, key=on_bake)
        if on_bake(tag) <= on_bake(shown):
            return None
    if tag not in link.pose_of or tag not in bases:
        return None
    return tag


def raw_poses(node):
    """{frame: stored pose as written}"""
    return _raw_of(node.parm("kimodo_poses").unexpandedString())


@functools.lru_cache(maxsize=1)
def _raw_of(text):
    return {int(f): p for f, p in (json.loads(text) if text.strip() else {}).items()}


UNHEADED = "unheaded"


def raw_pose(pose):
    raw = {
        "rot": None if pose.local_rots is None else constraints.matrix_to_axis_angle(pose.local_rots).tolist(),
        "root": [None if np.isnan(v) else float(v) for v in pose.root_position],
    }
    if pose.root_2d is not None:
        raw["root2d"] = [float(v) for v in pose.root_2d]
    if pose.heading is not None:
        raw["heading"] = [float(v) for v in pose.heading]
    return raw


def read_poses(node):
    """{frame: Pose}, a root only pose has local_rots None"""
    return _poses_of(node.parm("kimodo_poses").unexpandedString())


@functools.lru_cache(maxsize=1)
def _poses_of(text):
    return {
        f: constraints.Pose(
            None if p["rot"] is None else constraints.axis_angle_to_matrix(np.array(p["rot"])),
            np.array([np.nan if v is None else v for v in p["root"]], dtype=np.float64),
            None if p.get("root2d") is None else tuple(p["root2d"]),
            tuple(p["heading"]) if p.get("heading") else None,
        )
        for f, p in _raw_of(text).items()
    }


def write_raw_poses(node, raw, record=False):
    """text that was read is written back unchanged, digit for digit"""
    text = json.dumps({str(f): raw[f] for f in sorted(raw)}) if raw else ""
    if node.parm("kimodo_poses").unexpandedString() != text:
        if record:
            node.parm("kimodo_poses").set(text)
        else:
            with hou.undos.disabler():
                node.parm("kimodo_poses").set(text)


def constraint_frames(d):
    return sorted({f for c in d["constraints"] for f in c["frames"]})


def types_at(d):
    """{frame: types of the constraints on it}"""
    found = {}
    for c in d["constraints"]:
        for frame in c["frames"]:
            found.setdefault(frame, set()).add(c["type"])
    return found


def headed(node):
    """2d root frames that hold the heading too"""
    return _doc_facts(node.parm("kimodo_doc").unexpandedString())[1]


@functools.lru_cache(maxsize=16)
def _doc_facts(text):
    d = doc.loads(text)
    return types_at(d), {f for c in d["constraints"] if c["type"] == "root2d" and c.get("heading") for f in c["frames"]}


def held_heading(node, frame, pose):
    """(cos, sin) a 2d root keyframe with constrain heading holds"""
    if frame not in headed(node) or (pose.local_rots is None and pose.heading is None):
        return None
    return constraints.heading_of(pose)


def _types(node):
    return _doc_facts(node.parm("kimodo_doc").unexpandedString())[0]


def base_pose(node, frame):
    """pose of the shown result at a frame, rest pose before the first generate"""
    return constraints.Pose(*_frame_pose(shown_motion(node), frame))


def constraints_json(node, d, keyframes=False):
    stored = read_poses(node)

    def pose(frame):
        s = stored.get(frame)
        if s is not None and s.local_rots is not None:
            return s
        b = base_pose(node, frame)
        if s is None:
            return b
        return constraints.Pose(
            b.local_rots, np.array([s.root_position[0], b.root_position[1], s.root_position[2]]), None, s.heading
        )

    cjson = constraints.to_kimodo_json(d["constraints"], pose)
    return rootpath.apply(cjson) if node.evalParm("smooth_root_path") and not keyframes else cjson


def constraints_key(cjson):
    return hashlib.sha256(json.dumps(cjson, sort_keys=True).encode()).hexdigest()[:16]


NO_CONSTRAINTS = constraints_key([])


def pinned_at(d, frame):
    """rig controls the constraints on a frame pin, None for all of them"""
    types = types_at(d).get(frame)
    return rig.pinned_controls(types) if types else None


def taken(types):
    """what a frame's rig keys take from its stored pose"""
    return "all" if "fullbody" in types else frozenset(types)


def flush(node):
    """every viewer posing the node writes its animation to the node"""
    for viewer in editing_viewers(node):
        _write_state(viewer)


def _write_state(viewer):
    import apex.ui.statecommandutils as scu

    with hou.undos.disabler():
        scu.getAnimateState(viewer).writeToSopNode()


def stand_for(node, frames):
    """the rig keys on these frames now stand for their stored poses"""
    source = shown_result(node)
    if not frames or not rig.current(node, source):
        return
    _, mine = rig.stash_table(node)
    raw, known = raw_poses(node), _records(node)
    made, types = _of(node).made, _types(node)
    for frame in frames:
        if frame in raw and frame < mine.shape[1]:
            keys = mine[:, frame].tobytes()
            known.setdefault(frame, {})[keys] = raw[frame]
            made[frame] = (keys, taken(types.get(frame, ())), raw[frame])


@contextlib.contextmanager
def _interruptible(text, long):
    """houdini's progress window (esc interrupts) around a wait"""
    if not hou.isUIAvailable():
        yield
        return
    done = False
    try:
        with hou.InterruptableOperation(text, open_interrupt_dialog=long):
            yield
            done = True
    except hou.OperationInterrupted:
        if not done:
            raise rig.Interrupted("converting was interrupted") from None


def ensure_rig(node):
    """give the rig its animation for what the node shows, if it doesn't have it"""
    _track(node)
    source = shown_result(node)
    if node.evalParm("motion_frames") != motion_frames(node):
        with hou.undos.disabler():
            node.parm("motion_frames").set(motion_frames(node))
    if rig.current(node, source):
        return False
    if _undone_animation(node) is not None:
        sync_poses(node)
    frames = [f for f in constraint_frames(read_doc(node)) if f in raw_poses(node)]

    def build():
        with hou.undos.disabler():
            if node.parm("rig_result").unexpandedString() != source:
                node.parm("rig_result").set(source)
            rig.build(node, source, frames)

    if rig.has_base(node, source):
        build()
    else:
        with _interruptible(
            "Converting motion to rig\u2026", rig.bake_seconds(node, doc.total_frames(doc_from_parms(node))) > 1.0
        ):
            build()
    mine = _of(node)
    mine.made = {}
    # built from the stored poses, nothing in it is houdini's
    mine.foreign = 0
    stand_for(node, list(raw_poses(node)))
    mine.shown = {}
    before, mine.before_interruption = mine.before_interruption, None
    if before is not None and get_status(node)["text"].startswith(INTERRUPTED_BAKE):
        set_status(node, before)
    return True


def sync_poses(node):
    """stored poses and auto constraints follow the rig keys"""
    _track(node)
    source = shown_result(node)
    if rig.current(node, source):
        names, mine = rig.stash_table(node)
        undone = _undone_animation(node, (names, mine))
    else:
        undone = _undone_animation(node)
        if undone is None:
            return
        names, mine = rig.stash_table(node)
    every = mine
    if undone is None:
        _, base_names, base = rig.base(node, source)
        known = _records(node)
    else:
        _, base_names, base = rig.bases(node)[undone]
        known = _records(node, undone)
        if base.shape[1] != mine.shape[1]:
            return
    if names != base_names:
        rows = [names.index(n) for n in base_names if n in names]
        mine, base = mine[rows], base[[base_names.index(names[r]) for r in rows]]
    posed = np.abs(mine - base).max(axis=0) > rig.EPS
    if undone is not None:
        _poses_of_undone(node, undone, mine, posed, known)
    elif not _of(node).foreign:
        _rig_follows_poses(node, source, every, posed)
    else:
        _poses_follow_rig(node, source, every, mine, posed, known)


def _rig_follows_poses(node, source, every, posed):
    """sync_poses when only we changed the rig"""
    types, raw, made = types_at(read_doc(node)), raw_poses(node), _of(node).made
    baked = [
        f
        for f in sorted(types)
        if f in raw and f < every.shape[1] and made.get(f) != (every[:, f].tobytes(), taken(types[f]), raw[f])
    ]
    freed = [f for f in np.nonzero(posed)[0].tolist() if f not in types]
    if baked or freed:
        _rebake(node, source, baked, freed)


def _poses_follow_rig(node, source, every, mine, posed, known):
    """sync_poses when houdini changed the rig: a pose, an undo or a redo"""
    d, link = read_doc(node), _of(node)
    types = types_at(d)
    raw, added, removed = _read_rig(node, d, mine, posed, known)
    write_raw_poses(node, raw)
    link.foreign = 0
    renew = [
        f
        for f in sorted(set(np.nonzero(posed)[0].tolist()) | set(types))
        if f < mine.shape[1]
        and f not in removed
        and posed[f]
        and types.get(f)
        and "fullbody" not in types[f]
        and every[:, f].tobytes() != link.made.get(f, (None,))[0]
    ]
    if renew:
        _rebake(node, source, renew)
    _write_automatic(node, d, added, removed)


def _poses_of_undone(node, undone, mine, posed, known):
    """sync_poses when an undo across a generate brought back an older animation"""
    d, link = read_doc(node), _of(node)
    raw, added, removed = _read_rig(node, d, mine, posed, known, link.was_auto, only_known=True)
    write_raw_poses(node, raw)
    if rig.stash_tag(node) != undone:
        with hou.undos.disabler():
            rig.relabel(node, undone)
    link.foreign = 0
    _write_automatic(node, d, added, removed)


def _read_rig(node, d, mine, posed, known, gone=(), only_known=False):
    types, raw = types_at(d), dict(raw_poses(node))
    added, removed = [], []
    for frame in sorted(set(np.nonzero(posed)[0].tolist()) | set(types)):
        if frame >= mine.shape[1]:
            continue
        on_frame = [c for c in d["constraints"] if frame in c["frames"]]
        if (
            on_frame
            and not posed[frame]
            and all(c.get("auto") or (c["id"], tuple(c["frames"])) in gone for c in on_frame)
        ):
            d["constraints"] = [c for c in d["constraints"] if c not in on_frame]
            raw.pop(frame, None)
            removed.append(frame)
            continue
        keys = mine[:, frame].tobytes()
        if only_known and keys not in known.get(frame, {}):
            continue
        if not on_frame:
            d["constraints"].append({"id": timeline.new_constraint_id(d), "type": "fullbody", "frames": [frame], "auto": True})
            added.append(frame)
        seen = known.setdefault(frame, {})
        if keys not in seen:
            if frame in raw and not seen and not posed[frame]:
                seen[keys] = raw[frame]
            else:
                pose = rig.pose(node, frame)
                if types.get(frame) and types[frame] <= {"root2d"}:
                    pose = _sent_of_2d_root(node, frame, pose)
                seen[keys] = raw_pose(constraints.Pose(pose.local_rots, pose.root_position, pose.root_2d))
                if UNHEADED in raw.get(frame, {}):
                    seen[keys][UNHEADED] = raw[frame][UNHEADED]
        raw[frame] = seen[keys]
    return raw, added, removed


def _write_automatic(node, d, added, removed):
    if added or removed:
        with hou.undos.disabler():
            write_doc(node, d)
        if added:
            set_status(
                node,
                f"Full-Body keyframe added at frame {added[0] + 1}."
                if len(added) == 1
                else f"Full-Body keyframes added at {len(added)} edited frames.",
            )
        else:
            set_status(
                node,
                f"Full-Body keyframe removed at frame {removed[0] + 1}."
                if len(removed) == 1
                else f"Full-Body keyframes removed at {len(removed)} frames.",
            )


def _rebake(node, source, baked, freed=()):
    with hou.undos.disabler():
        rig.put_frames(node, source, baked, freed)
    stand_for(node, baked)
    _of(node).shown = {}


OWN_UNDO = "Kimodo: "


def own_step():
    """an undo step was made here, the watcher must not take it for a redo"""
    _undo.seen.clear()
    _undo.event.clear()
    for mine in _nodes.values():
        mine.noticed = None


def settle(node):
    """before an edit: store a pose made a moment ago in the viewer"""
    flush(node)
    sync_poses(node)


def write_constraints(node, d, raw, label, renew=(), settled=(), also=None):
    """changed constraints and their poses as one undo step"""
    source = shown_result(node)
    live = bool(renew) and rig.current(node, source)
    with hou.undos.group(OWN_UNDO + label):
        write_doc(node, d)
        write_raw_poses(node, raw, record=True)
        if live:
            types = types_at(d)
            rig.put_frames(node, source, [f for f in renew if f in types], [f for f in renew if f not in types])
        if also is not None:
            also()
    own_step()
    if live:
        stand_for(node, [f for f in renew if f in raw])
        _of(node).shown = {}
    stand_for(node, settled)
    update_unapplied(node)


def edit_constraints(node, before, after, mapping, label, poses=None):
    timeline.keep_path_ends(after, bool(node.evalParm("smooth_root_path")))
    was, now = types_at(before), types_at(after)
    old = raw_poses(node)
    raw = {f: p for f, p in old.items() if f in now}
    raw.update({new: old[f] for f, new in mapping.items() if f in old})
    raw.update(poses or {})
    fresh = [f for f in now if f not in raw]
    for f in fresh:
        raw[f] = raw_pose(base_pose(node, f))
    renew = [
        f
        for f in sorted(set(was) | set(now))
        if f not in fresh and (old.get(f) is not raw.get(f) or taken(was.get(f, ())) != taken(now.get(f, ())))
    ]
    write_constraints(node, after, raw, label, renew, settled=fresh)


def update_unapplied(node):
    """unapplied: do the constraints differ from the ones the result was made with"""
    unapplied = int(constraints_key(constraints_json(node, read_doc(node))) != applied_key(node))
    _redraw_posing(node)
    if node.evalParm("unapplied") != unapplied:
        with hou.undos.disabler():
            node.parm("unapplied").set(unapplied)
        if hou.isUIAvailable():
            try:
                node.cook()
            except hou.OperationFailed:
                pass


def _redraw_posing(node):
    if not hou.isUIAvailable():
        return
    changed = changed_keyframes(node)
    if _of(node).drawn_changed == changed:
        return
    _of(node).drawn_changed = changed
    for viewer in editing_viewers(node):
        viewer.curViewport().draw()


def add_constraint(node, ctype, frame):
    if not 0 <= frame < doc.total_frames(doc_from_parms(node)):
        set_status(node, f"Frame {frame + 1} is outside the motion.")
        return None
    settle(node)
    d = read_doc(node)
    before = taken(types_at(d).get(frame, ()))
    cid = timeline.add_keyframe(d, ctype, frame)
    if cid is None:
        return None
    raw = dict(raw_poses(node))
    had = frame in raw
    if not had:
        raw[frame] = raw_pose(base_pose(node, frame))
    write_constraints(
        node,
        d,
        raw,
        f"add {TYPE_LABELS[ctype]} keyframe at frame {frame + 1}",
        [frame] if had and taken(types_at(d)[frame]) != before else [],
        settled=[frame],
    )
    set_status(node, f"{TYPE_LABELS[ctype]} keyframe added at frame {frame + 1}.")
    return cid


_copied = {}


def copy_pose(node, frame):
    ensure_rig(node)
    flush(node)
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
    ensure_rig(node)
    settle(node)
    names = rig.pasted_channels(ctype, keep_root)
    keys = _copied["keys"]
    if isinstance(names, tuple):
        keys = {name: value for name, value in keys.items() if name not in names[1]}
    elif names is not None:
        keys = {name: value for name, value in keys.items() if name in names}
    d = read_doc(node)
    entry = next((c for c in d["constraints"] if c["type"] == ctype and frame in c["frames"]), None)
    if entry is None:
        entry = {"id": timeline.new_constraint_id(d), "type": ctype, "frames": [frame]}
        d["constraints"].append(entry)
    entry.pop("auto", None)
    entry.pop(timeline.LOCKED, None)
    raw = dict(raw_poses(node))
    label = f"paste pose at frame {frame + 1}"
    with hou.undos.group(OWN_UNDO + label):
        rig.set_keys(node, frame, keys)
        pose = rig.pose(node, frame)
        raw[frame] = raw_pose(constraints.Pose(pose.local_rots, pose.root_position, pose.root_2d))
        write_constraints(node, d, raw, label, settled=[frame])
    _of(node).shown = {}
    what = {"fullbody": "Pose only" if keep_root else "Pose and root", "root2d": "Root"}.get(
        ctype, f"{TYPE_LABELS[ctype]} pose"
    )
    set_status(node, f"{what} of frame {_copied['frame'] + 1} pasted at frame {frame + 1}.")
    return entry["id"]


TYPE_LABELS = {
    "fullbody": "Full-Body",
    "root2d": "2D Root",
    "left-hand": "Left Hand",
    "right-hand": "Right Hand",
    "left-foot": "Left Foot",
    "right-foot": "Right Foot",
}


RESULT = "kimodo_result"


def pack_sample(local_rots, root_positions):
    """one sample as stored text"""
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        r=constraints.matrix_to_axis_angle(local_rots).astype(np.float32),
        p=np.asarray(root_positions, dtype=np.float32),
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _result(node):
    return _result_of(node.parm(RESULT).unexpandedString())


@functools.lru_cache(maxsize=1)
def _result_of(text):
    return json.loads(text) if text else {"key": "", "samples": []}


def store_result(node, samples, key, applied, sent, poses):
    result = {
        "key": key,
        "applied": applied,
        "sent": sent,
        "poses": {str(f): p for f, p in sorted(poses.items())},
        "samples": [pack_sample(*s) for s in samples],
    }
    with hou.undos.disabler():
        node.parm(RESULT).set(json.dumps(result) if samples else "")
    _of(node).rig_motion = None
    set_result_samples(node)


def stored_bytes(node):
    """scene file bytes for the result and the poses"""
    return len(node.parm(RESULT).unexpandedString()) + len(node.parm("kimodo_poses").unexpandedString())


def stored_note(node):
    """one line for the node's info window"""
    samples, poses = len(_result(node)["samples"]), len(raw_poses(node))
    result = plural(samples, "sample") if samples else "no result"
    return f"Stored in this node: {result}, {plural(poses, 'pose')}, {stored_bytes(node) / 1e6:.2f} MB."


def set_result_samples(node):
    """tell the parms how many samples there are"""
    count = max(len(_result(node)["samples"]), 1)
    if node.evalParm("result_samples") != count:
        with hou.undos.disabler():
            node.parm("result_samples").set(count)


def _sample_index(node):
    return min(max(node.evalParm("sample"), 0), len(_result(node)["samples"]) - 1)


def _sample_motion(node, index):
    samples = _result(node)["samples"]
    if not 0 <= index < len(samples):
        return None
    return _decoded(samples[index])


@functools.lru_cache(maxsize=16)
def _decoded(text):
    with np.load(io.BytesIO(base64.b64decode(text))) as z:
        return skeleton.motion_from_rotations(
            constraints.axis_angle_to_matrix(z["r"].astype(np.float64)), z["p"].astype(np.float64)
        )


def shown_motion(node):
    """the shown sample as a Motion, None before the first generate"""
    return _sample_motion(node, _sample_index(node)) if _result(node)["samples"] else None


def shown_result(node):
    """'<request key>:<sample>' of the shown sample, '' while there is none"""
    result = _result(node)
    return f"{result['key']}:{_sample_index(node)}" if result["samples"] else ""


def finish_with(node, samples, text="Done."):
    """finish a generate with these motions, no service needed"""
    gen = _GenerateJob(node)
    gen.finish(text, samples)
    return gen


def applied_key(node):
    """key of the constraints the shown result was made with"""
    result = _result(node)
    return result.get("applied", NO_CONSTRAINTS) if result["samples"] else NO_CONSTRAINTS


def _shown_sign(node):
    return node.parm("kimodo_doc").unexpandedString(), node.parm("kimodo_poses").unexpandedString(), shown_result(node)


@_once_per(_shown_sign)
def changed_keyframes(node):
    """{(type, frame)} whose constraint or pose differs from the shown result"""
    result = _result(node)
    now = constraints.frame_entries(constraints_json(node, read_doc(node), keyframes=True))
    if not result["samples"]:
        return set(now)
    sent = constraints.frame_entries(result["sent"])
    return {k for k, v in now.items() if sent.get(k) != v}


def can_reset(node):
    return bool(_result(node)["samples"])


def generated_pose(node, frame):
    """a constraint frame's pose from the last generate"""
    return _result(node)["poses"].get(str(frame)) or raw_pose(base_pose(node, frame))


def changed_frames(node):
    return {frame for _, frame in changed_keyframes(node)}


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


SETTINGS = (
    "model",
    "seed",
    "steps",
    "cfg_text",
    "cfg_constraint",
    "num_samples",
    "postprocess",
    "root_margin",
    "smooth_root_path",
)


def import_folder(node, folder, what=None):
    name = os.path.basename(os.path.normpath(folder))
    what = what or f"import timeline {name}"
    try:
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            clips, settings, notes = examples.read_meta(json.load(f))
        path = os.path.join(folder, "constraints.json")
        data = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        cons, poses = constraints.from_kimodo_json(data)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise hou.Error(f"{name}: {e}") from None
    total = sum(frames for _, frames in clips)
    last = max((c["frames"][-1] for c in cons), default=0)
    if last >= total or min((c["frames"][0] for c in cons), default=0) < 0:
        raise hou.Error(f"{name}: constraints up to frame {last + 1}, its clips last {total} frames.")
    d, start = read_doc(node), 0
    d["prompts"] = []
    for n, (text, frames) in enumerate(clips, 1):
        d["prompts"].append({"id": f"p{n}", "text": text, "start_frame": start, "end_frame": start + frames})
        start += frames
    d["constraints"] = cons
    raw = {}
    for c in cons:
        for frame, pose in poses[c["id"]].items():
            if pose.local_rots is not None or frame not in raw:
                raw[frame] = raw_pose(pose)
    timeline.keep_path_ends(d, bool(settings.get("smooth_root_path", node.evalParm("smooth_root_path"))))
    for frame in constraint_frames(d):
        if frame not in raw:
            raw[frame] = raw_pose(base_pose(node, frame))
    flush(node)

    def rest():
        with rig.own_write(node):
            node.parm(rig.STASH).set(hou.Geometry())
        node.parm("selected_prompt").set(0)
        for parm, value in settings.items():
            node.parm(parm).set(int(value) if isinstance(value, bool) else value)

    write_constraints(node, d, raw, what, also=rest)
    from . import timeline_edit

    timeline_edit.update_root_path_guide(node)
    text = f"{what[0].upper()}{what[1:]}: {plural(len(clips), 'clip')}, {plural(sum(len(c['frames']) for c in cons), 'keyframe')}."
    set_status(node, " ".join([text, "Press Generate."] + notes))


def export_folder(node, folder):
    settle(node)
    d = doc_from_parms(node)
    d["constraints"] = [c for c in d["constraints"] if not c.get(timeline.LOCKED)]
    settings = {name: node.evalParm(name) for name in SETTINGS}
    for name in ("postprocess", "smooth_root_path"):
        settings[name] = bool(settings[name])
    meta = examples.make_meta([(p["text"], p["end_frame"] - p["start_frame"]) for p in d["prompts"]], settings)
    os.makedirs(folder, exist_ok=True)
    for file, data in (("meta.json", meta), ("constraints.json", constraints_json(node, d, keyframes=True))):
        with open(os.path.join(folder, file), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    set_status(
        node,
        f"Timeline exported: {plural(len(d['prompts']), 'clip')} and "
        f"{plural(sum(len(c['frames']) for c in d['constraints']), 'keyframe')} to {folder}.",
    )


def _choose_folder(title):
    path = hou.ui.selectFile(title=title, file_type=hou.fileType.Directory, chooser_mode=hou.fileChooserMode.Read)
    return os.path.normpath(hou.text.expandString(path)) if path else None


def import_dialog(node):
    folder = _choose_folder("Import a timeline: a folder with meta.json and constraints.json")
    if folder:
        import_folder(node, folder)


def export_dialog(node):
    folder = _choose_folder("Export the timeline to a folder (meta.json and constraints.json)")
    if not folder:
        return
    there = [f for f in ("meta.json", "constraints.json") if os.path.exists(os.path.join(folder, f))]
    if there and hou.ui.displayMessage(
        f"{folder} already has {' and '.join(there)}. Replace?",
        buttons=("Replace", "Cancel"),
        default_choice=1,
        close_choice=1,
    ):
        return
    export_folder(node, folder)


def example_folders():
    return examples.soma_examples(hou.getenv("KIMODO_MOTION") or "")


def example_menu():
    """menu script of the example parm"""
    names = list(example_folders())
    if not names:
        return ["", "(Kimodo's examples were not found)"]
    return [x for n in names for x in (n, examples.label(n))]


def load_example(node):
    """load the example picked in the menu, same as import"""
    name = node.evalParm("example")
    folder = example_folders().get(name)
    if folder is None:
        raise hou.Error(f"Kimodo's example {name!r} was not found in the installed Kimodo.")
    import_folder(node, folder, f"load example {examples.label(name)}")


def timeline_buttons(node):
    """callback of the timeline import / export strip"""
    clicked = node.evalParm("timeline")
    with hou.undos.disabler():
        node.parm("timeline").set(0)
    if clicked & 1:
        import_dialog(node)
    elif clicked & 2:
        export_dialog(node)


class _GenerateJob:
    """one press of generate, step() moves it on without blocking"""

    def __init__(self, node):
        self.node, self.client = node, client.Client(timeout=2.0)
        sync_poses(node)
        self.doc = doc_from_parms(node)
        doc.validate(self.doc)
        cjson = constraints_json(node, self.doc)
        self.request = client.make_request(self.doc, cjson)
        self.applied = constraints_key(cjson)
        self.sent = constraints_json(node, self.doc, keyframes=True) if node.evalParm("smooth_root_path") else cjson
        frames = constraint_frames(self.doc)
        self.poses = {f: p for f, p in raw_poses(node).items() if f in frames}
        self.key = client.cache_key(self.request)
        self.folder = None
        self.job_id, self.started, self.canceled, self.proc = None, time.monotonic(), False, None
        self.warning = "" if abs(hou.fps() - doc.FPS) < 1e-6 else f"Scene is {hou.fps():g} fps, Kimodo is 30 fps. "

    def status(self, text, stage, progress=None, detail=""):
        set_status(self.node, self.warning + text, stage, progress, self.started, detail)

    def fail(self, text, detail=""):
        self.status(text, "error", detail=detail or f"See {client.log_path()}")

    def unchanged(self):
        """same request the result was made with?"""
        return bool(_result(self.node)["samples"]) and _result(self.node)["key"] == self.key

    def discard_folder(self):
        if self.folder is not None:
            shutil.rmtree(self.folder, ignore_errors=True)
            self.folder = None

    def finish(self, text, samples):
        d = read_doc(self.node)
        d.update({k: self.doc[k] for k in ("model", "prompts", "generation")})
        d["result"] = {"npz": [], "selected": 0}
        sent = {c["id"] for c in self.doc["constraints"]}
        for c in d["constraints"]:
            if c["id"] in sent and c.pop("auto", None):
                _of(self.node).was_auto.add((c["id"], tuple(c["frames"])))
        with hou.undos.disabler():
            write_doc(self.node, d)
        store_result(self.node, samples, self.key, self.applied, self.sent, self.poses)
        update_unapplied(self.node)
        self.status(text, "done")

    def step(self):
        elapsed = time.monotonic() - self.started
        if self.job_id is None:
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
            self.folder = tempfile.mkdtemp(prefix="kimodo_")
            self.job_id = self.client.generate(dict(self.request, output_dir=self.folder))
        if self.canceled:
            self.client.cancel(self.job_id)
        job = self.client.job(self.job_id)
        if job["status"] in ("queued", "running"):
            stage = "canceling" if self.canceled else "generating"
            self.status(f"Generating... {job['progress'] * 100:.0f}%", stage, job["progress"])
            return False
        if job["status"] == "done":
            samples = []
            for path in job["npz"]:
                with np.load(path) as z:
                    samples.append((z["local_rot_mats"], z["posed_joints"][:, 0]))
            self.finish(f"Done in {elapsed:.1f} s.", samples)
        elif job["status"] == "canceled":
            self.status("Canceled.", "canceled")
        else:
            self.fail("Generation failed.", job["error"])
        self.discard_folder()
        return True


def generate(node, wait=False, poll=0.25, timeout=600.0):
    """start generating, returns the job or None"""
    sid, mine = node.sessionId(), _of(node)
    if mine.running is not None:
        set_status(node, "Already generating.")
        return None
    flush(node)
    clear_error(node)
    try:
        gen = _GenerateJob(node)
    except (hou.Error, doc.DocError, client.ServiceError, rig.NotReady) as e:
        set_status(node, "Cannot generate.", "error", detail=_message(e))
        return None
    if gen.unchanged():
        set_status(node, gen.warning + "Nothing changed since the last Generate.")
        return gen
    mine.running = gen
    if gen.warning and hou.isUIAvailable():
        hou.ui.setStatusMessage(f"Kimodo Motion ({node.name()}): {gen.warning.strip()}", hou.severityType.Warning)
    set_busy(node, True)

    def finished():
        mine.running = None
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

    def step():
        try:
            done = _safe_step(gen)
        except hou.ObjectWasDeleted:
            done = True
        if done and tick in hou.ui.eventLoopCallbacks():
            hou.ui.removeEventLoopCallback(tick)
            finished()

    def tick():
        if time.monotonic() - last[0] >= poll:
            last[0] = time.monotonic()
            when_idle(("generate", sid), step)

    hou.ui.addEventLoopCallback(tick)
    return gen


def _message(e):
    return e.instanceMessage() if isinstance(e, hou.Error) else str(e)


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
        if gen.job_id is not None and gen.client.health() is None:
            gen.fail("Kimodo service stopped while generating. Press Generate to retry.", str(e))
        else:
            gen.fail("Kimodo service error.", str(e))
        gen.discard_folder()
        return True


def cancel(node):
    """returns right away, the job stops at the next denoising step"""
    gen = _of(node).running
    if gen is None:
        set_busy(node, False)
        set_status(node, "Nothing to cancel.")
    else:
        gen.canceled = True
        gen.status("Canceling...", "canceling")


ANIMATE_STATE = "kimodo_pose"
# sidefx's own state, a viewer in it on our node poses it too
STOCK_STATE = "apexanimate"
POSE_STATE_PROBLEM = None


def pose_state():
    """the state that poses a kimodo node: ours, or sidefx's if ours didn't load"""
    return ANIMATE_STATE if not hou.isUIAvailable() or POSE_STATE_PROBLEM is not None else STOCK_STATE


WATCHED_PARMS = ("animation", "kimodo_doc", "kimodo_poses", "sample", "smooth_root_path")
WATCH_INTERVAL = 0.1


def scene_viewers():
    return [p for p in hou.ui.paneTabs() if p.type() == hou.paneTabType.SceneViewer]


def editing_viewers(node):
    """scene viewers posing this node, shown ones first"""
    if not hou.isUIAvailable():
        return []
    found = [v for v in scene_viewers() if v.currentState() in (pose_state(), STOCK_STATE) and v.currentNode() == node]
    return sorted(found, key=lambda v: not v.isCurrentTab())


def is_editing(node):
    return bool(editing_viewers(node))


def edit_pose(node):
    """start posing in the shown scene view"""
    viewer = next((v for v in scene_viewers() if v.isCurrentTab() and v.pwd() == node.parent()), None)
    if viewer is None:
        set_status(node, f"Show {node.parent().path()} in a Scene View to edit poses.")
        return False
    if viewer in editing_viewers(node):
        return True
    _of(node).ended = False
    try:
        ensure_rig(node)
    except rig.Interrupted:
        _say_interrupted(node, INTERRUPTED_BAKE)
        return False
    except rig.NotReady as e:
        set_status(node, f"The rig is not ready: {e}. Try again.")
        return False
    if viewer.currentNode() != node:
        node.setCurrent(True)
    viewer.setCurrentState(pose_state())
    watch(node)
    try:
        show_pinned_controls(node, viewer)
    except hou.Error:
        pass
    return viewer in editing_viewers(node)


def finish_edit(node):
    """bring the node up to date with its poses"""
    mine = _of(node)
    mine.retry = False
    try:
        if is_editing(node):
            ensure_rig(node)
        sync_poses(node)
        update_unapplied(node)
    except rig.Interrupted:
        _end_posing(node)
    except rig.NotReady:
        mine.retry = True
    except (hou.Error, doc.DocError) as e:
        set_status(node, f"Could not read the edited poses: {_message(e)}")
    return not mine.retry


INTERRUPTED_BAKE = "Converting motion to rig was interrupted."


def _say_interrupted(node, text):
    if _of(node).before_interruption is None:
        _of(node).before_interruption = get_status(node)["text"]
    set_status(node, text)


def _end_posing(node):
    _of(node).ended = True
    _say_interrupted(node, f"{INTERRUPTED_BAKE} Posing ended.")


def _on_enter_state(state, kwargs):
    node = kwargs["node"]
    _of(node).ended = False
    if POSE_STATE_PROBLEM or pose_state() == STOCK_STATE:
        why = POSE_STATE_PROBLEM or "the Kimodo pose state could not be registered"
        set_status(node, f"Posing with SideFX's full animate state: {why}.")
    try:
        ensure_rig(node)
    except rig.Interrupted:
        _end_posing(node)
    except (rig.NotReady, hou.Error):
        pass


INTERRUPTED = "The generation that ran when this scene was saved did not finish. Press Generate."


def node_ready(node):
    """OnCreated / OnLoaded"""
    if node.evalParm("busy") and _of(node).running is None:
        set_busy(node, False)
        with hou.undos.disabler():
            node.parm("status").set(INTERRUPTED)
    if node.evalParm("is_posing"):
        with hou.undos.disabler():
            node.parm("is_posing").set(0)
    set_result_samples(node)
    _track(node)
    watch(node)


def watch(node):
    """follow the node while the ui runs"""
    if not hou.isUIAvailable():
        return
    mine = _of(node)
    _track(node)
    _listen(node, _on_parm_changed)
    if node.cachedUserData("apexanimate:onenter") is None:
        node.setCachedUserData("apexanimate:onenter", _on_enter_state)
    if mine.watched:
        return
    mine.watched = True
    mine.was_editing, mine.finished = False, -1
    if _poll not in hou.ui.eventLoopCallbacks():
        hou.ui.addEventLoopCallback(_poll)


def _on_parm_changed(node, event_type, **kwargs):
    parm_tuple = kwargs.get("parm_tuple")
    if parm_tuple is None or parm_tuple.name() in WATCHED_PARMS:
        _of(node).changes += 1


# houdini has no undo event, so the undo and redo lists get watched
class _UndoWatch:
    def __init__(self):
        self.seen = []
        self.event = []
        self.count = 0
        self.last_poll = 0.0


_undo = _UndoWatch()


def _look_at_undo():
    """notice an undo or redo from the lengths of the undo and redo lists"""
    undo, redo = hou.undos.undoLabels(), hou.undos.redoLabels()
    before = list(_undo.seen)
    _undo.seen[:] = [len(undo), len(redo)]
    if _undo.event:
        _undo.event[2] -= 1
        if _undo.event[2] <= 0:
            _undo.event.clear()
    kind = None
    if before and len(redo) > before[1] and len(undo) < before[0]:
        kind, label = "Undone", redo[0]
    elif before and len(redo) < before[1] and len(undo) > before[0]:
        kind, label = "Redone", undo[0]
    if kind:
        _undo.count += 1
        _undo.event[:] = [kind, label, 3, _undo.count]


def _report_undo(node, event):
    mine = _of(node)
    sign = (node.parm("kimodo_doc").unexpandedString(), node.parm("kimodo_poses").unexpandedString())
    before, mine.sign = mine.sign, sign
    if before is None or sign == before:
        return
    history = mine.undone
    if event and mine.reported == event[3]:
        event = None
    if event and event[0] == "Undone":
        history.append(before)
    elif event and event[0] == "Redone" and history and history[-1] == sign:
        history.pop()
    else:
        history.clear()
        return
    label = event[1]
    if label.startswith(OWN_UNDO):
        what = label[len(OWN_UNDO) :]
    else:
        was, now = (json.loads(texts[1]) if texts[1].strip() else {} for texts in (before, sign))
        frames = [int(f) for f in set(was) | set(now) if was.get(f) != now.get(f)]
        what = f"pose at frame {frames[0] + 1}" if len(frames) == 1 else "pose edit"
    mine.reported = event[3]
    set_status(node, f"{event[0]}: {what}.")


def _poll():
    """one look at a watched node"""
    now = time.monotonic()
    if now - _undo.last_poll >= WATCH_INTERVAL:
        _undo.last_poll = now
        when_idle("watch", _poll_watched)


def _poll_watched():
    _look_at_undo()
    for sid, mine in list(_nodes.items()):
        if not mine.watched:
            continue
        node = hou.nodeBySessionId(sid)
        if node is None:
            continue
        try:
            poll(node)
            mine.watch_error = None
        except hou.ObjectWasDeleted:
            pass
        except Exception as e:
            text = f"{type(e).__name__}: {e}"
            if mine.watch_error != text:
                mine.watch_error = text
                set_status(node, f"Kimodo could not follow the pose state ({text}).")


def show_pinned_controls(node, viewer):
    """offer only the controls the current frame pins"""
    import apex.control_2 as ac
    import apex.ui.statecommandutils as scu

    if POSE_STATE_PROBLEM:
        return
    shown = _of(node).shown
    everything = rig.pose_controls()
    pinned = pinned_at(read_doc(node), current_frame())
    pinned = everything if pinned is None else pinned
    state = scu.getAnimateState(viewer)
    if state is None:
        return
    dim = bool(getattr(state, "show_unpinned", True))
    visible = everything if dim else pinned
    disabled = {rig.CONTROLS_PATH + name for name in (everything - pinned if dim else set())}
    scene = state.scene
    for path, control in scene.control_manager.controls.items():
        name = path.rsplit("/", 1)[-1]
        if name in everything and bool(control.node_properties.get("selectable", True)) != (name in pinned):
            control.node_properties["selectable"] = name in pinned
    templates = [t for t in (scene.getData(f"{path}/control_template") for path in scene.findDataPaths("*.rig")) if t is not None]
    current = {path for t in templates for path in ac.getEnabledControlsFromTemplate(t, invert=True)}
    if shown.get(viewer.name()) == frozenset(visible) and current == disabled:
        return
    with hou.undos.disabler():
        if shown.get(viewer.name()) != frozenset(visible):
            viewer.runStateCommand(
                "showControls",
                {
                    "controls": [rig.CONTROLS_PATH + name for name in sorted(visible)],
                    "controls_in_tree_format": False,
                    "pick_modifier": hou.pickModifier.Replace,
                },
            )
        if current != disabled:
            for template in templates:
                ac.setControlTemplateEnabled(template, sorted(current - disabled), True)
                ac.setControlTemplateEnabled(template, sorted(disabled), False)
            state.runSceneCallbacks(force=True)
        selectable = pinned | (
            rig.selected_by_selectors() if not pinned.isdisjoint(skeleton.FINGER_SELECTORS) else set()
        )
        selected = list(state.control_selection)
        still = [path for path in selected if path.rsplit("/", 1)[-1] in selectable]
        if len(still) != len(selected):
            viewer.runStateCommand(
                "selectControls",
                {"controls": still, "controls_in_tree_format": False, "pick_modifier": hou.pickModifier.Replace},
            )
    shown[viewer.name()] = frozenset(visible)


def follow_undo(node, viewer):
    """after an undo or redo while posing the state writes its animation"""
    mine, redo = _of(node), len(hou.undos.redoLabels())
    own = bool(_undo.event) and _undo.event[1].startswith(OWN_UNDO)
    # a redo while posing, the state's own copy is stale
    if mine.redo_seen is not None and mine.redo_seen != redo and not own:
        _write_state(viewer)
    mine.redo_seen = redo


def poll(node):
    """one look at a watched node"""
    mine = _of(node)
    watch(node)
    if pose_state() == STOCK_STATE:
        for viewer in scene_viewers():
            if viewer.isCurrentTab() and viewer.currentState() == ANIMATE_STATE and viewer.currentNode() == node:
                viewer.setCurrentState(STOCK_STATE)
    viewers = editing_viewers(node)
    if mine.ended:
        for viewer in [v for v in viewers if v.isCurrentTab()]:
            viewer.enterViewState()
            viewers.remove(viewer)
        if not viewers:
            mine.ended = False
    editing, changes = bool(viewers), mine.changes
    if bool(node.evalParm("is_posing")) != editing:
        with hou.undos.disabler():
            node.parm("is_posing").set(int(editing))
    front = [v for v in viewers if v.isCurrentTab()]
    try:
        for viewer in front:
            if not viewer.geometryVisibility(node):
                viewer.setGeometryVisibility(node, True)
            if viewer.name() not in mine.shown:
                import apex.ui.statecommandutils as scu

                state = scu.getAnimateState(viewer)
                if state is not None:
                    state.updateHUD()
            show_pinned_controls(node, viewer)
        if front:
            follow_undo(node, front[0])
    except hou.Error:
        pass
    if not editing:
        mine.shown = {}
        mine.redo_seen = None
    stale = editing and not mine.ended and not rig.current(node, shown_result(node))
    if (mine.was_editing and not editing) or changes != mine.finished or mine.retry or stale:
        if _undo.event:
            mine.noticed = list(_undo.event)
        if finish_edit(node):
            mine.finished = mine.changes
            noticed, mine.noticed = mine.noticed, None
            _report_undo(node, noticed)
    mine.was_editing = editing


def cook(sop):
    """python sop: the shown sample at the current frame"""
    frame = hou.frame()
    motion = shown_motion(sop.parent())
    if motion is None:
        skeleton.write_geometry(
            sop.geometry(),
            skeleton.rest_positions(),
            np.tile(np.eye(3).reshape(9), (skeleton.NUM_JOINTS, 1)),
            names=skeleton.OUTPUT_NAMES,
        )
    else:
        skeleton.build_geometry(sop.geometry(), motion, int(round(frame)) - 1, names=skeleton.OUTPUT_NAMES)
    skeleton.write_rest_transform(sop.geometry())


UNAPPLIED_WARNING = "Edited poses not applied. Press Generate."


def cook_notice(sop):
    """python sop at the end, raises the asset's warning or error"""
    node = sop.parent()
    sop.addMessage(stored_note(node))
    shown, asked = pose_frames(read_doc(node))
    if node.evalParm("show_poses") and len(shown) < asked:
        sop.addMessage(f"The viewer shows {len(shown)} of {asked} constraint poses.")
    if node.evalParm("show_body") and _skin() is None:
        sop.addMessage(NO_BODY)
    error = node.parm("error").unexpandedString()
    if node.evalParm("unapplied"):
        raise hou.NodeWarning(
            UNAPPLIED_WARNING if not error else f"{UNAPPLIED_WARNING}\nThe last Generate failed.\n{error}"
        )
    if error:
        shown = "the previous result" if _result(node)["samples"] else "the rest pose, there is no result yet"
        raise hou.NodeWarning(f"The last Generate failed; this is {shown}.\n{error}")


MARKER_RGB, UNAPPLIED_MARKER_RGB = (240, 170, 60), (190, 140, 230)
REST_COLOR = (0.33, 0.33, 0.33)
MAX_POSES = 40
RUN_STEP = 10
CROSS = 0.12
ARROW, ARROW_HEAD = 0.6, 0.06
POSE_FRAME = "pose_frame"


def _pinned_joints(ctype):
    found = {skeleton.JOINT_NAMES.index(skeleton.TIPS[ctype])}
    for j, p in enumerate(skeleton.PARENTS):
        if p in found:
            found.add(j)
    return found


_PINNED_JOINTS = {ctype: _pinned_joints(ctype) for ctype in skeleton.TIPS}
_BONES = [(p, j) for j, p in enumerate(skeleton.PARENTS) if p >= 0]


def marker_color(frame, changed):
    """color of a frame's pose, same as its timeline marker"""
    return tuple(v / 255.0 for v in (UNAPPLIED_MARKER_RGB if frame in changed else MARKER_RGB))


def _pose_colors(types, frame, changed):
    """per joint colors of a constraint frame's pose"""
    ctypes = types.get(frame, set())
    if not ctypes - {"root2d"}:
        return None
    marker = marker_color(frame, changed)
    pinned = (
        set(range(skeleton.NUM_JOINTS))
        if "fullbody" in ctypes
        else set().union(*(_PINNED_JOINTS.get(t, set()) for t in ctypes))
    )
    return [marker if j in pinned else REST_COLOR for j in range(skeleton.NUM_JOINTS)]


def pose_colors(node, frame):
    """per joint colors of a constraint frame's pose"""
    return _pose_colors(_types(node), frame, changed_frames(node))


def posed_positions(node, frame, rig_pose):
    """joint positions of a frame's pose as stored when the rig has rig_pose"""
    local, root = _frame_pose(shown_motion(node), frame)
    return skeleton.composed_pose(
        local,
        root,
        rig_pose.local_rots,
        rig_pose.root_position,
        rig_pose.root_2d,
        _types(node).get(frame, set()),
        held_heading(node, frame, rig_pose),
    )[0]


def pose_frames(d):
    """constraint frames whose pose is drawn, and which get a label"""
    keep = {}
    for c in d["constraints"]:
        if c["type"] == "root2d":
            continue
        for run in timeline.runs(sorted(c["frames"])):
            for i, frame in enumerate(run):
                if i % RUN_STEP == 0 or frame == run[-1]:
                    keep[frame] = keep.get(frame, False) or i == 0
    frames = sorted(keep)
    while len(frames) > MAX_POSES:
        frames = frames[::2]
    return {frame: keep[frame] for frame in frames}, len(keep)


def _pose_root(stored, frame):
    pose = stored.get(frame)
    if pose is None:
        return None
    x, z = (pose.root_position[0], pose.root_position[2]) if pose.root_2d is None else pose.root_2d
    return None if np.isnan(x) or np.isnan(z) else (float(x), 0.0, float(z))


def _root_paths(d, stored):
    for c in d["constraints"]:
        if c["type"] == "root2d":
            for run in timeline.runs(sorted(c["frames"])):
                path = [(frame, _pose_root(stored, frame)) for frame in run if _pose_root(stored, frame) is not None]
                if path:
                    yield c, path


def root_mark(x, z, heading, cross=True):
    """lines of a 2d root keyframe's cross on the ground"""
    c, s = (1.0, 0.0) if heading is None else heading

    def at(ahead, aside):
        return (x + ahead * s + aside * c, 0.0, z + ahead * c - aside * s)

    lines = [[at(0.0, -CROSS), at(0.0, CROSS)]] if cross else []
    if heading is None:
        return lines + [[at(-CROSS, 0.0), at(CROSS, 0.0)]]
    tip = at(ARROW, 0.0)
    head = [[tip, at(ARROW - ARROW_HEAD * 0.866, side * ARROW_HEAD * 0.5)] for side in (-1.0, 1.0)]
    return lines + [[at(-CROSS if cross else 0.0, 0.0), tip]] + head


def posed_root_mark(node, frame, pose):
    """root_mark from the rig's pose instead of the stored one"""
    held = held_heading(node, frame, pose)
    if held is None:
        return None
    frames = next(c["frames"] for c in read_doc(node)["constraints"] if c["type"] == "root2d" and frame in c["frames"])
    x, z = pose.root_2d if pose.root_2d is not None else pose.root_position[[0, 2]]
    return root_mark(float(x), float(z), held, cross=frame - 1 not in frames and frame + 1 not in frames)


def cook_poses(sop):
    """python sop: the pose of every constraint keyframe"""
    node, geo = sop.parent(), sop.geometry()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
    geo.addAttrib(hou.attribType.Point, POSE_FRAME, -1)
    for name in ("kimodo_doc", "kimodo_poses", RESULT, "sample"):
        node.evalParm(name)
    if not node.evalParm("show_poses"):
        return
    d = read_doc(node)
    types, stored, motion, changed = types_at(d), read_poses(node), shown_motion(node), changed_frames(node)

    def color(frame):
        return marker_color(frame, changed)

    points, colors, frames, lines = [], [], [], []
    for frame in pose_frames(d)[0]:
        if stored.get(frame) is None:
            continue
        positions, _ = _composed(node, motion, frame)
        first = len(points)
        points += positions.tolist()
        colors += _pose_colors(types, frame, changed)
        frames += [frame] * skeleton.NUM_JOINTS
        lines += [(first + a, first + b) for a, b in _BONES]

    for c, path in _root_paths(d, stored):
        first = len(points)
        if len(path) > 1:
            points += [p for _, p in path]
            colors += [color(frame) for frame, _ in path]
            frames += [-1] * len(path)
            lines += [tuple(range(first, first + len(path)))]
        for frame, (x, _, z) in path if len(path) == 1 or c.get("heading") else ():
            held = held_heading(node, frame, stored[frame]) if c.get("heading") else None
            for line in root_mark(x, z, held, cross=len(path) == 1):
                lines.append(tuple(range(len(points), len(points) + len(line))))
                points += line
                colors += [color(frame)] * len(line)
                frames += [frame] * len(line)
    if points:
        made = geo.createPoints(points)
        geo.setPointFloatAttribValues("Cd", [v for c in colors for v in c])
        geo.setPointIntAttribValues(POSE_FRAME, frames)
        geo.createPolygons([tuple(made[i] for i in line) for line in lines], False)


NO_BODY = "The installed Kimodo has no SOMA body (skin_standard.npz); no body is shown."


def _skin():
    return body.load(hou.getenv("KIMODO_MOTION") or "")


def cook_body_mesh(sop):
    """python sop: the body mesh at bind pose"""
    geo, skin = sop.geometry(), _skin()
    if skin is None:
        return
    geo.addAttrib(hou.attribType.Point, "N", (0.0, 0.0, 0.0))
    geo.addAttrib(hou.attribType.Point, "Cd", body.COLOR)
    geo.addAttrib(hou.attribType.Point, "Alpha", body.ALPHA)
    geo.createPoints(skin.vertices.tolist())
    geo.setPointFloatAttribValues("N", skin.normals.ravel().tolist())
    # houdini's polygons wind the other way
    geo.createPolygons(skin.faces[:, ::-1].tolist())


def cook_body(sop):
    """python sop: the body on the shown sample"""
    geo, skin = sop.geometry(), _skin()
    frame = hou.frame()
    positions, world = _motion_pose(shown_motion(sop.parent()), int(round(frame)) - 1)
    if skin is None or geo.intrinsicValue("pointcount") != len(skin.vertices):
        return
    vertices, normals = skin.posed(positions, world)
    geo.setPointFloatAttribValuesFromString("P", vertices)
    geo.setPointFloatAttribValuesFromString("N", normals)


def posed_body(node, frame, rig_pose):
    """the body on the pose the pose state shows"""
    skin = _skin()
    if skin is None:
        return None
    types = _types(node).get(frame, set())
    if types - {"root2d"}:
        local, root = _frame_pose(shown_motion(node), frame)
        positions, world = skeleton.composed_pose(
            local,
            root,
            rig_pose.local_rots,
            rig_pose.root_position,
            rig_pose.root_2d,
            types,
            held_heading(node, frame, rig_pose),
        )
    else:
        positions, world = skeleton.fk(rig_pose.local_rots, rig_pose.root_position)
    return skin.posed(positions, world)


def constraint_label(node, frame):
    """'Left Hand @ 90' and so on"""
    types = _types(node).get(frame)
    if not types:
        return ""
    return " + ".join(TYPE_LABELS[t] for t in doc.CONSTRAINT_TYPES if t in types) + f" @ {frame + 1}"


@_once_per(lambda node: (*_shown_sign(node), node.evalParm("show_poses")))
def pose_labels(node):
    found = []
    if not node.evalParm("show_poses"):
        return found
    d = read_doc(node)
    types, stored, motion, changed = types_at(d), read_poses(node), shown_motion(node), changed_frames(node)

    def color(frame):
        return marker_color(frame, changed)

    for frame, labeled in pose_frames(d)[0].items():
        if labeled and stored.get(frame) is not None:
            pelvis = _composed(node, motion, frame)[0][0]
            found.append((constraint_label(node, frame), tuple(float(v) for v in pelvis), frame, color(frame)))
    for _, path in _root_paths(d, stored):
        first, place = path[0]
        if types[first] == {"root2d"}:
            found.append((constraint_label(node, first), place, first if len(path) == 1 else -1, color(first)))
    return found


def cook_rest(sop):
    """python sop: the rest skeleton the rig is built from"""
    skeleton.build_rest_geometry(sop.geometry())


def _frame_pose(motion, frame):
    if motion is None:
        return np.tile(np.eye(3), (skeleton.NUM_JOINTS, 1, 1)), skeleton.rest_positions()[0]
    i = min(max(frame, 0), motion.num_frames - 1)
    return motion.local_rots[i], motion.positions[i, 0].astype(np.float64)


def _motion_pose(motion, frame):
    if motion is None:
        return skeleton.fk(*_frame_pose(None, frame))
    i = min(max(frame, 0), motion.num_frames - 1)
    return motion.positions[i], motion.global_rots[i]


def _rig_motion(node):
    node.evalParm("rig_result")
    source = node.parm("rig_result").unexpandedString()
    kept = _of(node).rig_motion
    if kept is None or kept[0] != source:
        kept = (source, _sample_motion(node, int(source.rpartition(":")[2])) if source else None)
        _of(node).rig_motion = kept
    return kept[1]


def _write_pose(geo, positions, world, with_root, root_rotation=None):
    skeleton.write_geometry(
        geo,
        positions,
        np.swapaxes(world, -1, -2).reshape(skeleton.NUM_JOINTS, 9),
        with_root=with_root,
        root_rotation=root_rotation,
    )


def _sent_of_2d_root(node, frame, pose):
    local, root = _frame_pose(_rig_motion(node), frame)
    held = held_heading(node, frame, pose)
    local = np.array(local, dtype=np.float64)
    if held is not None:
        local[0] = skeleton.facing(local[0], held)
    x, z = (pose.root_position[0], pose.root_position[2]) if pose.root_2d is None else pose.root_2d
    root = np.array(root, dtype=np.float64)
    root[0], root[2] = x, z
    return constraints.Pose(local, root, pose.root_2d)


def _held_turn(node, motion, frame):
    types, stored = _types(node).get(frame, set()), read_poses(node).get(frame)
    held = held_heading(node, frame, stored) if stored is not None and types and types <= {"root2d"} else None
    return None if held is None else skeleton.root_turn(_frame_pose(motion, frame)[0], held)


def _composed(node, motion, frame):
    """the motion with what the frame's constraints pin taken from its stored pose"""
    stored, types = read_poses(node).get(frame), _types(node).get(frame)
    if stored is None or not types:
        return _motion_pose(motion, frame)
    local, root = _frame_pose(motion, frame)
    return skeleton.composed_pose(
        local, root, stored.local_rots, stored.root_position, stored.root_2d, types, held_heading(node, frame, stored)
    )


def cook_rig_source(sop):
    """python sop: the motion the rig follows"""
    node = sop.parent()
    _write_pose(sop.geometry(), *_motion_pose(_rig_motion(node), current_frame()), True)


def cook_pose_source(sop):
    """python sop: the same motion with the stored poses on their frames"""
    node, frame = sop.parent(), current_frame()
    motion = _rig_motion(node)
    _write_pose(sop.geometry(), *_composed(node, motion, frame), True, _held_turn(node, motion, frame))
