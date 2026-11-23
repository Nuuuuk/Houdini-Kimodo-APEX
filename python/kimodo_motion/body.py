"""the soma body mesh from the installed kimodo, posed for the viewer."""

import os

import numpy as np

from . import examples, skeleton

SKIN_FILE = os.path.join("assets", "skeletons", "somaskel77", "skin_standard.npz")
COLOR = (0.62, 0.7, 0.78)
ALPHA = 0.3


class Skin:
    def __init__(self, path):
        with np.load(path) as data:
            names = [str(n) for n in data["rig_joint_names"]]
            if names != list(skeleton.JOINT_NAMES):
                raise ValueError(f"{path}: its joints are not the SOMA 77 joints in Kimodo's order")
            self.vertices = np.asarray(data["bind_vertices"], dtype=np.float64)
            self.faces = np.asarray(data["faces"], dtype=np.int64)
            self._bind_inverse = np.linalg.inv(np.asarray(data["bind_rig_transform"], dtype=np.float64))
            indices, weights = data["lbs_indices"], data["lbs_weights"]
        self.normals = _vertex_normals(self.vertices, self.faces)
        used = weights > 0
        if not used.any(axis=1).all():
            raise ValueError(f"{path}: a vertex is bound to no joint")
        self._joints, self._weights = indices[used], weights[used].astype(np.float32)[:, None]
        self._starts = np.concatenate([[0], np.cumsum(used.sum(axis=1))[:-1]])
        self._bind, self._bind_normals = self.vertices.astype(np.float32), self.normals.astype(np.float32)

    def posed(self, positions, world_rots):
        """vertices and normals as float32 for these joint positions and rotations"""
        pose = np.zeros((skeleton.NUM_JOINTS, 4, 4))
        pose[:, :3, :3], pose[:, :3, 3], pose[:, 3, 3] = world_rots, positions, 1.0
        carry = (pose @ self._bind_inverse)[:, :3, :].reshape(skeleton.NUM_JOINTS, 12).astype(np.float32)
        blended = np.add.reduceat(carry[self._joints] * self._weights, self._starts).reshape(-1, 3, 4)
        turn = blended[:, :, :3]
        vertices = (turn * self._bind[:, None, :]).sum(axis=2) + blended[:, :, 3]
        normals = (turn * self._bind_normals[:, None, :]).sum(axis=2)
        return vertices, normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _vertex_normals(vertices, faces):
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def skin_path(root):
    package = examples.kimodo_package(root)
    path = package and os.path.join(package, SKIN_FILE)
    return path if path and os.path.isfile(path) else None


_skins = {}


def load(root):
    path = skin_path(root)
    if path is None:
        return None
    if path not in _skins:
        _skins[path] = Skin(path)
    return _skins[path]
