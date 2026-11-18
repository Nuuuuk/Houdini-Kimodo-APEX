"""the apex constraint rig inside the asset: baking and reading poses."""

import uuid

import hou
import numpy as np

from . import constraints, doc, skeleton

SOURCE, BAKE, FRAME_BAKE, ANIMATE, POSE = "rig_source", "motion_on_rig", "pose_on_rig", "animate", "rig_pose"
STASH = "animation"
PATCH = "rig_patch"
TAG = "kimodo_source"
SESSION = uuid.uuid4().hex[:8]
MAX_BAKES = 8
REMOVED = "removed_paths"
EPS = 1e-5
_PARENTS = np.array(skeleton.PARENTS)
_PICKS = (lambda n: n == "animation", lambda n: n.endswith(".clip"), lambda n: n != "layer.data")


class NotReady(Exception):
    """the bake is cooking or incomplete, ask again later"""


class Interrupted(NotReady):
    pass


def _prim(geo, pick):
    for p in geo.prims():
        if pick(p.attribValue("name")):
            return p
    raise NotReady("the rig's animation is not there (yet)")


def channels(top):
    geo = top
    for pick in _PICKS:
        geo = _prim(geo, pick).getEmbeddedGeometry()
    return {p.attribValue("name"): p for p in geo.prims()}


def _edited_copy(top, edit):
    chain = [hou.Geometry(top)]
    for pick in _PICKS:
        chain.append(hou.Geometry(_prim(chain[-1], pick).getEmbeddedGeometry()))
    edit(chain[-1])
    for level in (2, 1, 0):
        _prim(chain[level], _PICKS[level]).setEmbeddedGeometry(chain[level + 1])
    return chain[0]


def stash(node):
    """the animation stored on the animate node, None until there is one"""
    geo = node.parm(STASH).eval()
    if geo is None or not any(p.attribValue("name") == "animation" for p in geo.prims()):
        return None
    return geo


_baking = set()
_bases = {}


def tag_for(node, source):
    """tag of a stash built in this session for a source"""
    return f"{source}|{node.evalParm('duration'):g}|{SESSION}"


def _cooked(node, name):
    sid = node.sessionId()
    if sid in _baking:
        raise NotReady("the rig is being evaluated")
    _baking.add(sid)
    inner = node.node(name)
    try:
        geo = inner.geometry()
    except hou.OperationInterrupted:
        raise Interrupted("converting was interrupted") from None
    except hou.OperationFailed as e:
        if "interrupted" in str(e).lower():
            raise Interrupted("converting was interrupted") from None
        raise NotReady(f"the bake failed: {e}") from None
    finally:
        _baking.discard(sid)
    if any("interrupted" in error.lower() for error in inner.errors()):
        raise Interrupted("converting was interrupted")
    if geo is None:
        raise NotReady("the bake has no output")
    return geo


def table(top, frames):
    found = channels(top)
    names = sorted(found)
    values = np.empty((len(names), frames))
    for i, name in enumerate(names):
        prim = found[name]
        keys = prim.keyFrames()
        if len(keys) == frames and keys[0] == 1.0 and keys[-1] == float(frames):
            values[i] = prim.keyValues()
        else:
            values[i] = [prim.eval(float(hf)) for hf in range(1, frames + 1)]
    return names, values


def base(node, source):
    """the bake of the motion rig_result names"""
    sid, tag = node.sessionId(), tag_for(node, source)
    kept = _bases.setdefault(sid, {})
    if tag in kept:
        return kept[tag]
    top = hou.Geometry(_cooked(node, BAKE))
    top.deletePrims([p for p in top.prims() if p.attribValue("name") != "animation"])
    frames = int(round(node.evalParm("duration") * doc.FPS))
    found = channels(top)
    if not found or any(len(prim.keyFrames()) != frames for prim in found.values()):
        raise NotReady("the bake does not have a key on every frame")
    if len(kept) >= MAX_BAKES:
        kept.pop(next(iter(kept)))
    kept[tag] = (top, *table(top, frames))
    return kept[tag]


def has_base(node, source):
    return tag_for(node, source) in _bases.get(node.sessionId(), {})


def bake_seconds(node, frames):
    """rough bake time for this many frames"""
    return frames * 0.012 + (0.0 if _bases.get(node.sessionId()) else 0.9)


def frame_keys(node, frames):
    """{frame: {channel: value}} baked for single frames"""
    frames, out = sorted(frames), {}
    runs = [[f, f] for i, f in enumerate(frames) if i == 0 or f != frames[i - 1] + 1]
    for run in runs:
        while run[1] + 1 in frames:
            run[1] += 1
    for first, last in runs:
        with hou.undos.disabler():
            node.parmTuple(PATCH).set((first + 1, last + 1))
        for name, prim in channels(_cooked(node, FRAME_BAKE)).items():
            at, values = np.array(prim.keyFrames()), prim.keyValues()
            for frame in range(first, last + 1):
                out.setdefault(frame, {})[name] = values[int(np.abs(at - (frame + 1)).argmin())]
    return out


def _with_keys(top, keys):

    def write(geo):
        for prim in geo.prims():
            name = prim.attribValue("name")
            for frame, values in keys.items():
                if name in values and prim.hasKeyAtFrame(float(frame + 1)):
                    prim.setKeyValue(float(frame + 1), values[name])

    return _edited_copy(top, write)


def _set_stash(node, top, tag):
    if top.findGlobalAttrib(TAG) is None:
        top.addAttrib(hou.attribType.Global, TAG, "")
    top.setGlobalAttribValue(TAG, tag)
    node.parm(STASH).set(top)


def stash_tag(node):
    top = stash(node)
    if top is None or top.findGlobalAttrib(TAG) is None:
        return None
    return top.attribValue(TAG)


def damaged(top):
    """the stash lists removed scene paths, can't be evaluated"""
    return top.findGlobalAttrib(REMOVED) is not None and len(top.attribValue(REMOVED)) > 0


def current(node, source):
    """the node has a stash this session built for this source"""
    top = stash(node)
    return top is not None and not damaged(top) and stash_tag(node) == tag_for(node, source)


def build(node, source, frames):
    """the stash for a source: its bake with the keys of these frames"""
    top = base(node, source)[0]
    keys = frame_keys(node, frames) if frames else {}
    _set_stash(node, _with_keys(top, keys) if keys else hou.Geometry(top), tag_for(node, source))


def put_frames(node, source, baked=(), freed=()):
    """renew single frames: baked from stored poses, freed back to the bake"""
    keys = frame_keys(node, baked) if baked else {}
    if freed:
        _, names, values = base(node, source)
        keys.update({frame: dict(zip(names, values[:, frame].tolist())) for frame in freed})
    if keys:
        old = stash(node)
        _set_stash(node, _with_keys(old, keys), old.attribValue(TAG))


def stash_table(node):
    return table(stash(node), int(round(node.evalParm("duration") * doc.FPS)))


_TARGETS = {
    "left-hand": "L_armTarget_ik",
    "right-hand": "R_armTarget_ik",
    "left-foot": "L_legTarget_ik",
    "right-foot": "R_legTarget_ik",
}


def _control_channels(control):
    return {f"{control}_{kind}{axis}" for kind in "trs" for axis in "xyz"}


def pinned_controls(ctypes):
    """controls these constraint types pin, None for all"""
    if "fullbody" in ctypes:
        return None
    names = set()
    for ctype in ctypes:
        names |= (
            {skeleton.ROOT_CONTROL}
            if ctype == "root2d"
            else {_TARGETS[ctype], skeleton.ROOT_CONTROL, skeleton.PELVIS_CONTROL}
        )
    return names


def pasted_channels(ctype, keep_root=False):
    """channels a pasted pose sets, None for all"""
    root = {f"{skeleton.ROOT_CONTROL}_t{axis}" for axis in "xyz"}
    if ctype == "fullbody":
        return ("all but", root) if keep_root else None
    if ctype == "root2d":
        return set() if keep_root else {f"{skeleton.ROOT_CONTROL}_tx", f"{skeleton.ROOT_CONTROL}_tz"}
    return _control_channels(_TARGETS[ctype])


def keys_at(node, frame):
    return {name: prim.eval(float(frame + 1)) for name, prim in channels(stash(node)).items()}


def set_keys(node, frame, values):
    old = stash(node)
    f = float(frame + 1)

    def write(geo):
        for prim in geo.prims():
            name = prim.attribValue("name")
            if name in values and prim.hasKeyAtFrame(f):
                prim.setKeyValue(f, values[name])

    tag = old.attribValue(TAG) if old.findGlobalAttrib(TAG) is not None else ""
    _set_stash(node, _edited_copy(old, write), tag)


CONTROLS_PATH = "/soma.char/Base.rig/"


def pose_controls():
    limbs = [f"{limb}{part}_ik" for limb in skeleton.LIMBS for part in ("Target", "Polevec")]
    return {skeleton.ROOT_CONTROL, skeleton.PELVIS_CONTROL, *skeleton.FK_CONTROLS, *limbs}


def _evaluate(node, name, frame):
    sid = node.sessionId()
    if sid in _baking:
        raise NotReady("the rig is being evaluated")
    _baking.add(sid)
    try:
        geo = node.node(name).geometryAtFrame(frame + 1)
    except hou.OperationInterrupted:
        raise NotReady("the rig's evaluation was interrupted") from None
    finally:
        _baking.discard(sid)
    if geo is None or not len(geo.points()):
        errors = "; ".join(dict.fromkeys(node.node(name).errors()))
        # houdini is cooking the rig for the viewport right now
        if "recursion" in errors.lower():
            raise NotReady("the rig is being evaluated")
        if "interrupted" in errors.lower():
            raise NotReady("the rig's evaluation was interrupted")
        raise hou.Error(f"The rig has no pose at frame {frame + 1}: {errors or 'empty output'}")
    return pose_from_geometry(geo)


def pose(node, frame):
    """the rig's pose at a kimodo frame"""
    return _evaluate(node, POSE, frame)


def pose_from_geometry(geo):
    names = list(geo.pointStringAttribValues("name"))
    index = [names.index(n) for n in skeleton.JOINT_NAMES]
    world = np.array(geo.pointFloatAttribValues("transform")).reshape(-1, 3, 3)[index].transpose(0, 2, 1)
    points = np.array(geo.pointFloatAttribValues("P")).reshape(-1, 3)
    positions = points[index]
    for upper, mid, tip in _LIMB_INDICES:
        _solve_limb(world, positions, upper, mid, tip)
    local = world.copy()
    local[1:] = np.einsum("jba,jbc->jac", world[_PARENTS[1:]], world[1:])
    root_2d = points[names.index(skeleton.ROOT)][[0, 2]] if skeleton.ROOT in names else None
    return constraints.Pose(local, positions[0], root_2d)


_REST = skeleton.rest_positions()
_LIMB_INDICES = [tuple(skeleton.JOINT_NAMES.index(j) for j in limb) for limb in skeleton.LIMBS.values()]


def _rotation_between(a, b):
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _solve_limb(world, positions, upper, mid, tip):
    bone1, bone2 = _REST[mid] - _REST[upper], _REST[tip] - _REST[mid]
    l1, l2 = np.linalg.norm(bone1), np.linalg.norm(bone2)
    start, goal = positions[upper], positions[tip]
    to_goal = goal - start
    dist = np.linalg.norm(to_goal)
    if dist < 1e-9:
        return
    axis = to_goal / dist
    dist = min(max(dist, abs(l1 - l2) + 1e-9), l1 + l2)
    along = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
    out = np.sqrt(max(l1 * l1 - along * along, 0.0))
    side = positions[mid] - start
    side -= axis * np.dot(side, axis)
    if np.linalg.norm(side) < 1e-9:
        side = world[upper] @ bone1
        side -= axis * np.dot(side, axis)
    norm = np.linalg.norm(side)
    side = side / norm if norm > 1e-9 else np.zeros(3)
    middle = start + axis * along + side * out
    end = start + axis * dist
    world[upper] = _rotation_between(world[upper] @ bone1 / l1, (middle - start) / l1) @ world[upper]
    world[mid] = _rotation_between(world[mid] @ bone2 / l2, (end - middle) / l2) @ world[mid]
    positions[mid] = middle
