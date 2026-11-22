"""timeline edits on the node, between the ui and timeline.py."""

import copy
import json

import hou

from . import doc, motion_node, rootpath, timeline


def current_doc(node):
    """the doc with the clips and settings generate would use now"""
    return motion_node.doc_from_parms(node)


def _seconds(frames):
    return f"{frames / doc.FPS:.1f} s"


def _dropped_text(dropped):
    if not dropped:
        return ""
    return f" {len(dropped)} constraint frame{'s' if len(dropped) > 1 else ''} beyond the end removed."


def _settle(node):
    motion_node._flush(node)
    motion_node.sync_poses(node)


def _write_prompts(node, d, label, selected=None):
    """write a doc whose clips changed, and which one is selected"""
    prompts, total = d["prompts"], doc.total_frames(d)
    ids = [p["id"] for p in prompts]
    if selected is None:
        old = motion_node.read_doc(node)
        i = motion_node.selected_prompt(node, old)
        selected = old["prompts"][i]["id"] if old["prompts"][i]["id"] in ids else ids[min(i, len(ids) - 1)]
    index = ids.index(selected)
    with hou.undos.group(motion_node.OWN_UNDO + label):
        motion_node.write_doc(node, d)
        if node.evalParm("selected_prompt") != index:
            node.parm("selected_prompt").set(index)
        raw = motion_node._raw_poses(node)
        if any(f >= total for f in raw):
            motion_node.write_raw_poses(node, {f: p for f, p in raw.items() if f < total}, record=True)
    motion_node._own_step()
    motion_node.update_unapplied(node)


def add_prompt(node):
    _settle(node)
    d = current_doc(node)
    pid = timeline.add_prompt(d)
    _write_prompts(node, d, "add prompt", selected=pid)
    motion_node.set_status(node, f"Prompt added. The motion is now {_seconds(doc.total_frames(d))}.")
    return pid


def select_prompt(node, pid):
    d = current_doc(node)
    if d["prompts"][motion_node.selected_prompt(node, d)]["id"] == pid:
        return
    with hou.undos.disabler():
        node.parm("selected_prompt").set([p["id"] for p in d["prompts"]].index(pid))


def set_prompt_text(node, pid, text):
    _settle(node)
    d = current_doc(node)
    timeline.set_prompt_text(d, pid, text)
    _write_prompts(node, d, "edit prompt")
    motion_node.set_status(node, "Prompt changed.")


def resize_prompt(node, pid, frames):
    _settle(node)
    d = current_doc(node)
    dropped = timeline.resize_prompt(d, pid, frames)
    _write_prompts(node, d, "resize prompt")
    length = next(p["end_frame"] - p["start_frame"] for p in d["prompts"] if p["id"] == pid)
    motion_node.set_status(
        node, f"Prompt is {_seconds(length)}, the motion {_seconds(doc.total_frames(d))}." + _dropped_text(dropped)
    )


def delete_prompt(node, pid):
    _settle(node)
    d = current_doc(node)
    if len(d["prompts"]) < 2:
        motion_node.set_status(node, "The last prompt cannot be deleted.")
        return
    dropped = timeline.delete_prompt(d, pid)
    _write_prompts(node, d, "delete prompt")
    motion_node.set_status(node, f"Prompt deleted. The motion is now {_seconds(doc.total_frames(d))}." + _dropped_text(dropped))


def _write_constraints(node, before, after, mapping, label, poses=None):
    was, now = motion_node.types_at(before), motion_node.types_at(after)
    old = motion_node._raw_poses(node)
    raw = {f: p for f, p in old.items() if f in now}
    raw.update({new: old[f] for f, new in mapping.items() if f in old})
    raw.update(poses or {})
    fresh = [f for f in now if f not in raw]
    for f in fresh:
        raw[f] = motion_node.raw_pose(motion_node.base_pose(node, f))
    renew = [
        f
        for f in sorted(set(was) | set(now))
        if f not in fresh
        and (old.get(f) is not raw.get(f) or motion_node._taken(was.get(f, ())) != motion_node._taken(now.get(f, ())))
    ]
    motion_node._write_constraints(node, after, raw, motion_node.OWN_UNDO + label, renew)
    motion_node._stand_for(node, fresh)
    motion_node.update_unapplied(node)


def _span_text(frames):
    return f"frame {frames[0] + 1}" if len(frames) == 1 else f"frames {frames[0] + 1} to {frames[-1] + 1}"


def _frames_text(frames):
    return f"frame {frames[0] + 1}" if len(frames) == 1 else f"frames {', '.join(str(f + 1) for f in frames)}"


def move_runs(node, runs, delta):
    _settle(node)
    before = current_doc(node)
    after = copy.deepcopy(before)
    runs = [(cid, list(run)) for cid, run in runs]
    same_type = timeline.move_check(before, runs, delta)[1]
    moved = timeline.move_runs(after, runs, delta)
    if not moved:
        motion_node.set_status(node, "Not moved: " + ("it is" if len(runs) == 1 else "they are") + " at the end of the motion.")
        return {}
    mapping = {}
    for cid, frames in moved.items():
        mapping.update(frames)
        next(c for c in after["constraints"] if c["id"] == cid).pop("auto", None)
    landed = {f for frames in moved.values() for f in frames.values()}
    staying = {f for c in after["constraints"] for f in c["frames"] if c["id"] not in moved or f not in landed}
    replaced = sorted(landed & staying)
    if len(runs) == 1:
        c = next(c for c in after["constraints"] if c["id"] == runs[0][0])
        where = _span_text(sorted(mapping.values()))
        label, text = (
            f"move {motion_node.TYPE_LABELS[c['type']]} constraint to {where}",
            f"{motion_node.TYPE_LABELS[c['type']]} constraint moved to {where}.",
        )
    else:
        label = f"move {len(runs)} markers by {delta:+d} frames"
        text = f"{len(runs)} markers moved by {delta:+d} frames."
    _write_constraints(node, before, after, mapping, label)
    if len(replaced) == 1:
        text += (
            f" {'Its pose' if len(runs) == 1 else 'A moved pose'} replaced the one already on frame {replaced[0] + 1}."
        )
    elif replaced:
        text += f" {'Its' if len(runs) == 1 else 'Moved'} poses replaced the ones already on {len(replaced)} frames ({replaced[0] + 1} to {replaced[-1] + 1})."
    if same_type:
        kinds = sorted({t for t, _ in same_type})
        what = " and ".join(motion_node.TYPE_LABELS[t] for t in kinds)
        frames = sorted(f for _, f in same_type)
        text += (
            f" It replaced the {what} {'constraint' if len(frames) == 1 else 'constraints'} on {_frames_text(frames)}."
        )
    motion_node.set_status(node, text)
    return moved


def reset_runs(node, runs):
    """markers back to their poses from the last generate"""
    _settle(node)
    if not motion_node.can_reset(node):
        motion_node.set_status(node, "Nothing to reset to: press Generate first.")
        return False
    d = current_doc(node)
    frames = sorted({f for _, run in runs for f in run})
    raw = dict(motion_node._raw_poses(node))
    for f in frames:
        raw[f] = motion_node.generated_pose(node, f)
    if len(runs) == 1:
        c = next(c for c in d["constraints"] if c["id"] == runs[0][0])
        what = f"{motion_node.TYPE_LABELS[c['type']]} constraint on {_span_text(frames)}"
    else:
        what = f"{len(runs)} markers"
    motion_node._write_constraints(node, d, raw, motion_node.OWN_UNDO + f"reset {what}", frames)
    motion_node._stand_for(node, frames)
    motion_node.update_unapplied(node)
    motion_node.set_status(node, f"Reset {what} to the last Generate.")
    return True


def clear_all(node):
    _settle(node)
    before = current_doc(node)
    count = len(before["constraints"])
    if not count:
        motion_node.set_status(node, "There are no constraints to clear.")
        return False
    after = copy.deepcopy(before)
    after["constraints"] = []
    _write_constraints(node, before, after, {}, "clear all constraints")
    motion_node.set_status(node, f"All constraints cleared ({count}).")
    return True


def add_range(node, ctype, first, last):
    _settle(node)
    before = current_doc(node)
    after = copy.deepcopy(before)
    cid = timeline.add_range(after, ctype, first, last)
    frames = next(c["frames"] for c in after["constraints"] if c["id"] == cid)
    _write_constraints(node, before, after, {}, f"add {motion_node.TYPE_LABELS[ctype]} range on {_span_text(frames)}")
    motion_node.set_status(node, f"{motion_node.TYPE_LABELS[ctype]} range added on {_span_text(frames)}.")
    return cid


def delete_run(node, cid, run):
    _settle(node)
    before = current_doc(node)
    after = copy.deepcopy(before)
    ctype = next(c["type"] for c in after["constraints"] if c["id"] == cid)
    timeline.delete_run(after, cid, list(run))
    _write_constraints(
        node, before, after, {}, f"delete {motion_node.TYPE_LABELS[ctype]} constraint on {_span_text(list(run))}"
    )
    motion_node.set_status(node, f"{motion_node.TYPE_LABELS[ctype]} constraint on {_span_text(list(run))} deleted.")


def delete_runs(node, runs, verb="delete"):
    runs = [(cid, list(run)) for cid, run in runs]
    if len(runs) == 1 and verb == "delete":
        delete_run(node, *runs[0])
        return True
    _settle(node)
    before = current_doc(node)
    after = copy.deepcopy(before)
    for cid, run in runs:
        if any(c["id"] == cid for c in after["constraints"]):
            timeline.delete_run(after, cid, run)
    if after["constraints"] == before["constraints"]:
        return False
    _write_constraints(node, before, after, {}, f"{verb} {_plural(len(runs), 'marker')}")
    motion_node.set_status(node, f"{_plural(len(runs), 'marker')} {'cut' if verb == 'cut' else 'deleted'}.")
    return True


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


_copied_markers = []


def copy_runs(node, runs, quiet=False):
    """quiet leaves the status alone"""
    _settle(node)
    d, raw = current_doc(node), motion_node._raw_poses(node)
    found = [
        (next(c for c in d["constraints"] if c["id"] == cid), list(run))
        for cid, run in runs
        if any(c["id"] == cid for c in d["constraints"])
    ]
    if not found:
        return 0
    first = min(run[0] for _, run in found)
    _copied_markers[:] = [
        {
            "type": c["type"],
            "heading": bool(c.get("heading")),
            "frames": [f - first for f in run],
            "poses": [raw.get(f) or motion_node.raw_pose(motion_node.base_pose(node, f)) for f in run],
        }
        for c, run in found
    ]
    if not quiet:
        motion_node.set_status(node, f"{_plural(len(found), 'marker')} copied. Ctrl+V pastes at the current frame.")
    return len(found)


def cut_runs(node, runs):
    return copy_runs(node, runs, quiet=True) and delete_runs(node, runs, verb="cut")


def paste_runs(node, frame):
    """the earliest copied marker lands on frame, the rest keep their distance"""
    _settle(node)
    if not _copied_markers:
        motion_node.set_status(node, "Nothing copied yet: select markers and press Ctrl+C.")
        return []
    before = current_doc(node)
    after = copy.deepcopy(before)
    total = doc.total_frames(after)
    poses, pasted, dropped, replaced = {}, [], 0, 0
    for m in _copied_markers:
        keep = [(frame + off, pose) for off, pose in zip(m["frames"], m["poses"]) if 0 <= frame + off < total]
        dropped += len(m["frames"]) - len(keep)
        if not keep:
            continue
        targets = [f for f, _ in keep]
        for c in after["constraints"]:
            if c["type"] == m["type"]:
                replaced += len(set(c["frames"]) & set(targets))
                c["frames"] = [f for f in c["frames"] if f not in targets]
        after["constraints"] = [c for c in after["constraints"] if c["frames"]]
        entry = {"id": timeline._new_id(after, "constraints", "c"), "type": m["type"], "frames": targets}
        if m["heading"]:
            entry["heading"] = True
        after["constraints"].append(entry)
        poses.update(keep)
        pasted.append((entry["id"], targets))
    if not pasted:
        motion_node.set_status(node, f"Not pasted: frame {frame + 1} would put every copied marker past the end of the motion.")
        return []
    _write_constraints(node, before, after, {}, f"paste {_plural(len(pasted), 'marker')} at frame {frame + 1}", poses)
    text = f"{_plural(len(pasted), 'marker')} pasted at frame {frame + 1}."
    if replaced:
        text += f" {_plural(replaced, 'frame')} of the same type replaced."
    if dropped:
        text += f" {_plural(dropped, 'frame')} past the end of the motion dropped."
    motion_node.set_status(node, text)
    return pasted


def set_heading(node, runs, on):
    """constrain heading on or off for 2d root markers"""
    d = current_doc(node)
    runs = timeline.heading_runs(d, [(cid, list(run)) for cid, run in runs], on)
    frames = sorted({f for _, run in runs for f in run})
    if not frames:
        return False
    raw = dict(motion_node._raw_poses(node))
    for f in frames:
        pose = raw.get(f)
        if pose is None:
            continue
        if on:
            raw[f] = {**pose, motion_node.UNHEADED: {k: v for k, v in pose.items() if k != motion_node.UNHEADED}}
        else:
            before = pose.get(motion_node.UNHEADED) or {"rot": None}
            raw[f] = {k: v for k, v in before.items() if k not in ("root", "root2d")}
            raw[f]["root"] = pose["root"] if before.get("rot") is not None else [pose["root"][0], None, pose["root"][2]]
            if "root2d" in pose:
                raw[f]["root2d"] = pose["root2d"]
    what = "2D Root heading" if len(runs) == 1 else f"{len(runs)} 2D Root headings"
    motion_node._write_constraints(
        node, d, raw, motion_node.OWN_UNDO + ("constrain " if on else "free ") + what, frames
    )
    motion_node.update_unapplied(node)
    who = "2D Root constraint" if len(runs) == 1 else f"{len(runs)} 2D Root markers"
    motion_node.set_status(node, f"{who}: heading constrained too." if on else f"{who}: position only.")
    return True


def change_runs(node, runs, ctype):
    _settle(node)
    before = current_doc(node)
    after = copy.deepcopy(before)
    runs = [(cid, list(run)) for cid, run in runs]
    changing = [
        (cid, run) for cid, run in runs if next(c for c in before["constraints"] if c["id"] == cid)["type"] != ctype
    ]
    replaced = timeline.change_runs(after, changing, ctype)
    if not changing:
        return False
    new = motion_node.TYPE_LABELS[ctype]
    if len(changing) == 1:
        old = motion_node.TYPE_LABELS[next(c for c in before["constraints"] if c["id"] == changing[0][0])["type"]]
        where = _span_text(changing[0][1])
        label, text = f"change {old} constraint on {where} to {new}", f"{old} constraint on {where} changed to {new}."
    else:
        label, text = f"change {len(changing)} markers to {new}", f"{len(changing)} markers changed to {new}."
    _write_constraints(node, before, after, {}, label)
    if replaced:
        text += f" It replaced the {new} {'constraint' if len(replaced) == 1 else 'constraints'} on {_frames_text(replaced)}."
    motion_node.set_status(node, text)
    return True


def set_smooth_path(node):
    """callback of smooth 2d root path"""
    on = bool(node.evalParm("smooth_root_path"))
    _settle(node)
    before = current_doc(node)
    d = copy.deepcopy(before)
    if on:
        for frame in (0, doc.total_frames(d) - 1):
            timeline.add_keyframe(d, "root2d", frame)
    _write_constraints(node, before, d, {}, "smooth 2D Root path " + ("on" if on else "off"))
    update_root_path_guide(node)
    frames = sorted(f for c in d["constraints"] if c["type"] == "root2d" for f in c["frames"])
    motion_node.set_status(
        node,
        f"Smooth path on: one 2D Root path through {len(frames)} keyframes, frames {frames[0] + 1} to {frames[-1] + 1}."
        if on
        else "Smooth path off: 2D Root keyframes are sent one by one.",
    )


GUIDE_HEIGHT = 0.005


def update_root_path_guide(node):
    """store what the viewport draws for the smooth path"""
    parm = node.parm("root_path_guide")
    if parm is None:
        return
    d = motion_node.read_doc(node)
    text = ""
    if any(c["type"] == "root2d" for c in d["constraints"]):
        text = json.dumps(rootpath.guide(motion_node.constraints_json(node, d)))
    if parm.unexpandedString() != text:
        with hou.undos.disabler():
            parm.set(text)


def cook_root_path(sop):
    """python sop: the smooth 2d root path on the ground"""
    parm = sop.parent().parm("root_path_guide")
    text = parm.unexpandedString() if parm is not None else ""
    if not text.strip():
        return
    shapes, geo = json.loads(text), sop.geometry()

    def polyline(points):
        poly = geo.createPolygon(is_closed=False)
        for x, z in points:
            pt = geo.createPoint()
            pt.setPosition((x, GUIDE_HEIGHT, z))
            poly.addVertex(pt)

    for line in shapes["lines"]:
        polyline(line)
