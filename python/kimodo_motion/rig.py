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


def _prim(geo, pick):
    return next(p for p in geo.prims() if pick(p.attribValue("name")))


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


def bake(node):
    top = hou.Geometry(node.node(BAKE).geometry())
    top.deletePrims([p for p in top.prims() if p.attribValue("name") != "animation"])
    return top


def source_tag(node):
    """what the rig should follow now"""
    d = doc.loads(node.parm("kimodo_doc").unexpandedString())
    paths = d["result"]["npz"]
    shown = paths[min(max(node.evalParm("sample"), 0), len(paths) - 1)] if paths else ""
    poses = node.parm("kimodo_poses").unexpandedString()
    return f"{shown}|{node.evalParm('duration'):g}|{hashlib.md5(poses.encode()).hexdigest()[:12]}"


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
    new = bake(node)
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
    mine, baked = channels(top), channels(bake_cached(node))
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


_bake_cache = {}


def bake_cached(node):
    """bake(), copied only after the bake node cooked again"""
    inner = node.node(BAKE)
    inner.geometry()
    key = inner.cookCount()
    cached = _bake_cache.get(node.sessionId())
    if cached is None or cached[0] != key:
        cached = _bake_cache[node.sessionId()] = (key, bake(node))
    return cached[1]


def pose(node, frame):
    """the rig's pose at a kimodo frame"""
    geo = node.node(POSE).geometryAtFrame(frame + 1)
    if geo is None or not len(geo.points()):
        raise hou.Error(
            f"The rig has no pose at frame {frame + 1}: {'; '.join(node.node(POSE).errors()) or 'empty output'}"
        )
    names = list(geo.pointStringAttribValues("name"))
    index = [names.index(n) for n in skeleton.JOINT_NAMES]
    world = np.array(geo.pointFloatAttribValues("transform")).reshape(-1, 3, 3)[index].transpose(0, 2, 1)
    local = world.copy()
    local[1:] = np.einsum("jba,jbc->jac", world[_PARENTS[1:]], world[1:])
    root = np.array(geo.pointFloatAttribValues("P")).reshape(-1, 3)[index[0]]
    return constraints.Pose(local, root)
