"""the apex constraint rig inside the asset: baking and reading poses."""

import hashlib

import hou
import numpy as np

from . import constraints, doc, skeleton

SOURCE, BAKE, ANIMATE, POSE = "rig_source", "motion_on_rig", "animate", "rig_pose"
TAG = "kimodo_source"
EPS = 1e-5
_PARENTS = np.array(skeleton.PARENTS)
_PICKS = (lambda n: n == "animation", lambda n: n.endswith(".clip"), lambda n: n != "layer.data")


class NotReady(Exception):
    """the bake is cooking or incomplete, ask again later"""


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
    geo = node.node(ANIMATE).parm("animation").eval()
    if geo is None or not any(p.attribValue("name") == "animation" for p in geo.prims()):
        return None
    return geo


_baking = set()
_bake_cache = {}


def bake(node):
    sid, tag = node.sessionId(), source_tag(node)
    cached = _bake_cache.get(sid)
    if cached is not None and cached[0] == tag:
        return cached[1]
    if sid in _baking:
        raise NotReady("the rig's animation is being baked")
    _baking.add(sid)
    try:
        geo = node.node(BAKE).geometry()
    except hou.OperationFailed as e:
        raise NotReady(f"the bake failed: {e}") from None
    finally:
        _baking.discard(sid)
    if geo is None:
        raise NotReady("the bake has no output")
    top = hou.Geometry(geo)
    top.deletePrims([p for p in top.prims() if p.attribValue("name") != "animation"])
    baked = channels(top)
    frames = int(round(node.evalParm("duration") * doc.FPS))
    if not baked or any(len(prim.keyFrames()) != frames for prim in baked.values()):
        raise NotReady("the bake does not have a key on every frame")
    _bake_cache[sid] = (tag, top)
    return top


def source_tag(node):
    """what the rig should follow now"""
    poses = node.parm("kimodo_poses").unexpandedString()
    return (
        f"{node.parm('rig_npz').unexpandedString()}|{node.evalParm('duration'):g}|"
        f"{hashlib.md5(poses.encode()).hexdigest()[:12]}"
    )


def _set_stash(node, top, tag):
    if top.findGlobalAttrib(TAG) is None:
        top.addAttrib(hou.attribType.Global, TAG, "")
    top.setGlobalAttribValue(TAG, tag)
    node.node(ANIMATE).parm("animation").set(top)


def retag(node, tag):
    _set_stash(node, hou.Geometry(stash(node)), tag)


def stash_tag(node):
    top = stash(node)
    if top is None or top.findGlobalAttrib(TAG) is None:
        return None
    return top.attribValue(TAG)


def merge(node, keep_frames, tag):
    """rebuild the stash from the bake, keeping its own keys on keep_frames"""
    old = stash(node)
    new = hou.Geometry(bake(node))
    if old is not None and keep_frames:
        old_channels = channels(old)
        frames = [float(f + 1) for f in keep_frames]

        def keep(geo):
            for prim in geo.prims():
                source = old_channels.get(prim.attribValue("name"))
                if source is None:
                    continue
                for f in frames:
                    if source.hasKeyAtFrame(f) and prim.hasKeyAtFrame(f):
                        prim.setKeyValue(f, source.keyValue(f))

        new = _edited_copy(new, keep)
    _set_stash(node, new, tag)


def snap(node, frames):
    """the bake's keys back on these frames"""
    old = stash(node)
    if old is None or not frames:
        return
    baked = channels(bake(node))
    wanted = [float(f + 1) for f in frames]

    def reset(geo):
        for prim in geo.prims():
            source = baked.get(prim.attribValue("name"))
            if source is None:
                continue
            for f in wanted:
                if source.hasKeyAtFrame(f) and prim.hasKeyAtFrame(f):
                    prim.setKeyValue(f, source.keyValue(f))

    tag = old.attribValue(TAG) if old.findGlobalAttrib(TAG) is not None else ""
    _set_stash(node, _edited_copy(old, reset), tag)


def edited_frames(node, only=None):
    top = stash(node)
    if top is None:
        return []
    mine, baked = channels(top), channels(bake(node))
    edited = set()
    if only is not None:
        for f in only:
            hf = float(f + 1)
            for name, prim in mine.items():
                other = baked.get(name)
                if (
                    other is not None
                    and prim.hasKeyAtFrame(hf)
                    and other.hasKeyAtFrame(hf)
                    and abs(prim.keyValue(hf) - other.keyValue(hf)) > EPS
                ):
                    edited.add(f)
                    break
        return sorted(edited)
    for name, prim in mine.items():
        other = baked.get(name)
        if other is None:
            continue
        frames, other_frames = prim.keyFrames(), other.keyFrames()
        if frames == other_frames:
            diff = np.abs(np.array(prim.keyValues()) - np.array(other.keyValues())) > EPS
            edited.update(int(round(frames[i])) - 1 for i in np.nonzero(diff)[0])
        else:
            for hf in range(
                int(round(min(frames[0], other_frames[0]))), int(round(max(frames[-1], other_frames[-1]))) + 1
            ):
                if abs(prim.eval(float(hf)) - other.eval(float(hf))) > EPS:
                    edited.add(hf - 1)
    return sorted(edited)


def pose(node, frame):
    """the rig's pose at a kimodo frame"""
    geo = node.node(POSE).geometryAtFrame(frame + 1)
    if geo is None or not len(geo.points()):
        raise hou.Error(
            f"The rig has no pose at frame {frame + 1}: {'; '.join(node.node(POSE).errors()) or 'empty output'}"
        )
    return pose_from_geometry(geo)


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
