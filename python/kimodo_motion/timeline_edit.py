"""timeline edits on the node, between the ui and timeline.py."""

import copy
import json
import math

import hou

from . import doc, motion_node, rig, rootpath, timeline


def current_doc(node):
    """the doc with the clips and settings generate would use now"""
    return motion_node.doc_from_parms(node)


def _seconds(frames):
    return f"{frames / doc.FPS:.1f} s"


def _dropped_text(dropped):
    if not dropped:
        return ""
    return f" {len(dropped)} constraint frame{'s' if len(dropped) > 1 else ''} beyond the end removed."


def _write_prompts(node, d, label):
    """write a doc whose clips changed, and which one is selected"""
    prompts, total = d["prompts"], doc.total_frames(d)
    with hou.undos.group(label):
        motion_node.write_doc(node, d)
        if len(prompts) == 1:
            node.parm("prompt").set(prompts[0]["text"])
        node.parm("duration").set(total / doc.FPS)
        if node.parm("num_prompts") is not None:
            node.parm("num_prompts").set(len(prompts))
        poses = motion_node.read_poses(node)
        if any(f >= total for f in poses):
            motion_node.write_poses(node, {f: p for f, p in poses.items() if f < total})
    motion_node.update_unapplied(node)


def add_prompt(node):
    d = current_doc(node)
    pid = timeline.add_prompt(d)
    _write_prompts(node, d, "Add Kimodo prompt")
    motion_node.set_status(node, f"Prompt added. The motion is now {_seconds(doc.total_frames(d))}.")
    return pid


def set_prompt_text(node, pid, text):
    d = current_doc(node)
    timeline.set_prompt_text(d, pid, text)
    _write_prompts(node, d, "Edit Kimodo prompt")
    motion_node.set_status(node, "Prompt changed.")


def resize_prompt(node, pid, frames):
    d = current_doc(node)
    dropped = timeline.resize_prompt(d, pid, frames)
    _write_prompts(node, d, "Resize Kimodo prompt")
    length = next(p["end_frame"] - p["start_frame"] for p in d["prompts"] if p["id"] == pid)
    motion_node.set_status(
        node, f"Prompt is {_seconds(length)}, the motion {_seconds(doc.total_frames(d))}." + _dropped_text(dropped)
    )


def delete_prompt(node, pid):
    d = current_doc(node)
    if len(d["prompts"]) < 2:
        motion_node.set_status(node, "The last prompt cannot be deleted.")
        return
    dropped = timeline.delete_prompt(d, pid)
    _write_prompts(node, d, "Delete Kimodo prompt")
    motion_node.set_status(node, f"Prompt deleted. The motion is now {_seconds(doc.total_frames(d))}." + _dropped_text(dropped))


def _write_constraints(node, before, after, mapping, label):
    targets = set(mapping.values())
    freed = [f for f in timeline.freed_frames(before, after) if f not in targets]
    used = set(motion_node.constraint_frames(after))
    rig.bake(node)
    poses = motion_node.read_poses(node)
    unedited = set(poses) - set(rig.edited_frames(node, only=[f for f in mapping if f in poses]))
    carried = {f: new for f, new in mapping.items() if f not in unedited}

    def apply():
        with hou.undos.group(label):
            old = rig.stash(node)
            keys = {}
            if carried:
                for name, prim in rig.channels(old if old is not None else rig.bake(node)).items():
                    keys[name] = {
                        new: prim.keyValue(float(f + 1))
                        for f, new in carried.items()
                        if prim.hasKeyAtFrame(float(f + 1))
                    }
            motion_node.write_doc(node, after)
            stored = {f: p for f, p in poses.items() if f not in freed and (f not in mapping or f in used)}
            stored.update({mapping[f]: p for f, p in poses.items() if f in mapping})
            if sorted(stored) != sorted(poses) or any(stored[f] is not poses[f] for f in stored):
                motion_node.write_poses(node, stored)
            if mapping:
                tag = rig.source_tag(node)
                rig.merge(node, [f for f in used if f not in targets], tag)

                def put(geo):
                    for prim in geo.prims():
                        for new, value in keys.get(prim.attribValue("name"), {}).items():
                            if prim.hasKeyAtFrame(float(new + 1)):
                                prim.setKeyValue(float(new + 1), value)

                rig._set_stash(node, rig._edited_copy(rig.stash(node), put), tag)
            elif freed:
                rig.snap(node, freed)

    motion_node._outside_edit(node, apply)
    motion_node.update_unapplied(node)


def _span_text(frames):
    return f"frame {frames[0] + 1}" if len(frames) == 1 else f"frames {frames[0] + 1} to {frames[-1] + 1}"


def move_run(node, cid, run, delta):
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
    _write_constraints(node, before, after, mapping, "Move Kimodo constraint")
    motion_node.set_status(
        node, f"{motion_node.TYPE_LABELS[c['type']]} constraint moved to {_span_text(sorted(mapping.values()))}."
    )
    return True


def add_range(node, ctype, first, last):
    before = current_doc(node)
    after = copy.deepcopy(before)
    cid = timeline.add_range(after, ctype, first, last)
    _write_constraints(node, before, after, {}, "Add Kimodo constraint range")
    frames = next(c["frames"] for c in after["constraints"] if c["id"] == cid)
    motion_node.set_status(node, f"{motion_node.TYPE_LABELS[ctype]} range added on {_span_text(frames)}.")
    return cid


def delete_run(node, cid, run):
    before = current_doc(node)
    after = copy.deepcopy(before)
    ctype = next(c["type"] for c in after["constraints"] if c["id"] == cid)
    timeline.delete_run(after, cid, list(run))
    _write_constraints(node, before, after, {}, "Delete Kimodo constraint")
    motion_node.set_status(node, f"{motion_node.TYPE_LABELS[ctype]} constraint on {_span_text(list(run))} deleted.")


def set_heading(node, cid, on):
    """constrain heading on or off for 2d root markers"""
    d = current_doc(node)
    c = next(c for c in d["constraints"] if c["id"] == cid)
    if on:
        c["heading"] = True
    else:
        c.pop("heading", None)
    with hou.undos.group("Kimodo 2D Root heading"):
        motion_node.write_doc(node, d)
    motion_node.update_unapplied(node)
    motion_node.set_status(
        node, "2D Root constraint: heading constrained too." if on else "2D Root constraint: position only."
    )


def set_smooth_path(node, on):
    """callback of smooth 2d root path"""
    d = current_doc(node)
    if on:
        d["smooth_root_path"] = True
        for frame in (0, doc.total_frames(d) - 1):
            timeline.add_keyframe(d, "root2d", frame)
    else:
        d.pop("smooth_root_path", None)
    with hou.undos.group("Kimodo smooth root path"):
        motion_node.write_doc(node, d)
    motion_node.update_unapplied(node)
    frames = sorted(f for c in d["constraints"] if c["type"] == "root2d" for f in c["frames"])
    motion_node.set_status(
        node,
        f"Smooth path on: one 2D Root path through {len(frames)} keyframes, frames {frames[0] + 1} to {frames[-1] + 1}."
        if on
        else "Smooth path off: 2D Root keyframes are sent one by one.",
    )


GUIDE_HEIGHT = 0.005
MARK, TICK = 0.04, 0.25


def update_root_path_guide(node):
    """store what the viewport draws for the smooth path"""
    parm = node.parm("root_path_guide")
    if parm is None:
        return
    d = motion_node.read_doc(node)
    text = ""
    if any(c["type"] == "root2d" for c in d["constraints"]):
        try:
            text = json.dumps(rootpath.guide(motion_node.constraints_json(node, d)))
        except rig.NotReady:
            return
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

    def polyline(points, closed=False):
        poly = geo.createPolygon(is_closed=closed)
        for x, z in points:
            pt = geo.createPoint()
            pt.setPosition((x, GUIDE_HEIGHT, z))
            poly.addVertex(pt)

    for line in shapes["lines"]:
        polyline(line)
    for x, z in shapes["points"]:
        polyline([(x - MARK, z), (x, z + MARK), (x + MARK, z), (x, z - MARK)], closed=True)
    for x, z, c, s in shapes["headings"]:
        polyline([(x, z), (x + TICK * s, z + TICK * c)])
