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


def _write_prompts(node, d, label):
    """write a doc whose clips changed, and which one is selected"""
    prompts, total = d["prompts"], doc.total_frames(d)
    with hou.undos.group(motion_node.OWN_UNDO + label):
        motion_node.write_doc(node, d)
        if len(prompts) == 1:
            node.parm("prompt").set(prompts[0]["text"])
        node.parm("duration").set(total / doc.FPS)
        if node.parm("num_prompts") is not None:
            node.parm("num_prompts").set(len(prompts))
        raw = motion_node._raw_poses(node)
        if any(f >= total for f in raw):
            motion_node.write_raw_poses(node, {f: p for f, p in raw.items() if f < total}, record=True)
    motion_node._own_step()
    motion_node.update_unapplied(node)


def add_prompt(node):
    _settle(node)
    d = current_doc(node)
    pid = timeline.add_prompt(d)
    _write_prompts(node, d, "add prompt")
    motion_node.set_status(node, f"Prompt added. The motion is now {_seconds(doc.total_frames(d))}.")
    return pid


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


def _write_constraints(node, before, after, mapping, label):
    was, now = motion_node.types_at(before), motion_node.types_at(after)
    old = motion_node._raw_poses(node)
    raw = {f: p for f, p in old.items() if f in now}
    raw.update({new: old[f] for f, new in mapping.items() if f in old})
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


def move_run(node, cid, run, delta):
    _settle(node)
    before = current_doc(node)
    after = copy.deepcopy(before)
    mapping = timeline.move_run(after, cid, list(run), delta)
    if not mapping:
        motion_node.set_status(
            node, "Not moved: a constraint of the same type is in the way, or it is at the end of the motion."
        )
        return False
    c = next(c for c in after["constraints"] if c["id"] == cid)
    c.pop("auto", None)
    had = motion_node.types_at(before)
    replaced = sorted(new for new in mapping.values() if new in had and new not in mapping)
    _write_constraints(
        node,
        before,
        after,
        mapping,
        f"move {motion_node.TYPE_LABELS[c['type']]} constraint to {_span_text(sorted(mapping.values()))}",
    )
    text = f"{motion_node.TYPE_LABELS[c['type']]} constraint moved to {_span_text(sorted(mapping.values()))}."
    if len(replaced) == 1:
        text += f" Its pose replaced the one already on frame {replaced[0] + 1}."
    elif replaced:
        text += (
            f" Its poses replaced the ones already on {len(replaced)} frames ({replaced[0] + 1} to {replaced[-1] + 1})."
        )
    motion_node.set_status(node, text)
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


def set_heading(node, cid, on):
    """constrain heading on or off for 2d root markers"""
    d = current_doc(node)
    c = next(c for c in d["constraints"] if c["id"] == cid)
    if on:
        c["heading"] = True
    else:
        c.pop("heading", None)
    with hou.undos.group(motion_node.OWN_UNDO + ("constrain 2D Root heading" if on else "free 2D Root heading")):
        motion_node.write_doc(node, d)
    motion_node._own_step()
    motion_node.update_unapplied(node)
    motion_node.set_status(
        node, "2D Root constraint: heading constrained too." if on else "2D Root constraint: position only."
    )


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
TICK = 0.25


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
    if not sop.parent().evalParm("show_poses"):
        return
    for x, z, c, s in shapes["headings"]:
        polyline([(x, z), (x + TICK * s, z + TICK * c)])
