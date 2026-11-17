"""constraint poses and kimodo's constraints json."""

from collections import namedtuple

import numpy as np

from .skeleton import NUM_JOINTS, fk

Pose = namedtuple("Pose", "local_rots root_position root_2d", defaults=(None,))

SOMA30_IN_77 = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    18,
    28,
    39,
    40,
    41,
    42,
    46,
    56,
    67,
    68,
    69,
    70,
    72,
    73,
    74,
    75,
)

POSE_TYPES = ("fullbody", "left-hand", "right-hand", "left-foot", "right-foot")


def motion_pose_source(motion):

    def pose(frame):
        i = min(max(int(frame), 0), motion.num_frames - 1)
        return Pose(motion.local_rots[i], motion.positions[i, 0])

    return pose


def same_pose(a, b, position_tol=1e-4, angle_tol=1e-3):
    (pa, wa), (pb, wb) = fk(a.local_rots, a.root_position), fk(b.local_rots, b.root_position)
    if np.abs(pa - pb).max() > position_tol:
        return False
    if (a.root_2d is None) != (b.root_2d is None) or (
        a.root_2d is not None and np.abs(np.subtract(a.root_2d, b.root_2d)).max() > position_tol
    ):
        return False
    dot = np.abs(np.sum(_matrix_to_quat(wa) * _matrix_to_quat(wb), axis=-1))
    return bool((2.0 * np.arccos(np.minimum(dot, 1.0))).max() <= angle_tol)


def to_kimodo_json(constraints, pose_source):
    """the doc's constraints as kimodo's json list"""
    out = []
    for c in constraints:
        frames = list(c["frames"])
        poses = [pose_source(f) for f in frames]
        root = np.array([p.root_position for p in poses], dtype=np.float64).reshape(len(frames), 3)
        entry = {"type": c["type"], "frame_indices": frames}
        if c["type"] in POSE_TYPES:
            entry["local_joints_rot"] = np.stack([matrix_to_axis_angle(p.local_rots) for p in poses]).tolist()
            entry["root_positions"] = root.tolist()
        if c["type"] == "root2d":
            entry["smooth_root_2d"] = [
                list(map(float, root[i, [0, 2]] if p.root_2d is None else p.root_2d)) for i, p in enumerate(poses)
            ]
        else:
            entry["smooth_root_2d"] = root[:, [0, 2]].tolist()
        out.append(entry)
    return out


def from_kimodo_json(data):
    """(constraints, poses) from kimodo's json list"""
    constraints, poses = [], {}
    for n, entry in enumerate(data, 1):
        ctype = entry["type"]
        if ctype not in POSE_TYPES + ("root2d",):
            raise ValueError(f"unsupported constraint type {ctype!r}")
        frames = [int(f) for f in entry["frame_indices"]]
        cid = f"c{n}"
        constraints.append({"id": cid, "type": ctype, "frames": sorted(set(frames))})
        xz = np.asarray(entry["smooth_root_2d"], dtype=np.float64).reshape(len(frames), 2)
        if ctype == "root2d":
            poses[cid] = {f: Pose(None, np.array([x, np.nan, z])) for f, (x, z) in zip(frames, xz)}
            continue
        rots = _to_77(axis_angle_to_matrix(np.asarray(entry["local_joints_rot"], dtype=np.float64)))
        root = np.asarray(entry["root_positions"], dtype=np.float64).reshape(len(frames), 3)
        poses[cid] = {f: Pose(r, p) for f, r, p in zip(frames, rots, root)}
    return constraints, poses


def _to_77(rots):
    j = rots.shape[1]
    if j == NUM_JOINTS:
        return rots
    if j != len(SOMA30_IN_77):
        raise ValueError(f"constraint has {j} joints, expected 77 or 30 (SOMA)")
    full = np.broadcast_to(np.eye(3), (len(rots), NUM_JOINTS, 3, 3)).copy()
    full[:, list(SOMA30_IN_77)] = rots
    return full


def axis_angle_to_matrix(v):
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.divide(v, angle, out=np.zeros_like(v), where=angle > 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    k = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(v.shape[:-1] + (3, 3))
    s, c = np.sin(angle)[..., None], np.cos(angle)[..., None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def matrix_to_axis_angle(m):
    m = np.asarray(m, dtype=np.float64)
    q = _matrix_to_quat(m)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sin_half, q[..., :1])
    scale = np.where(sin_half > 1e-12, angle / np.maximum(sin_half, 1e-12), 2.0)
    return xyz * scale


def _matrix_to_quat(m):
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    t = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    cand = np.stack(
        [
            np.stack(
                [t[..., 0], m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1
            ),
            np.stack(
                [m[..., 2, 1] - m[..., 1, 2], t[..., 1], m[..., 1, 0] + m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0]], -1
            ),
            np.stack(
                [m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1], t[..., 2], m[..., 2, 1] + m[..., 1, 2]], -1
            ),
            np.stack(
                [m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] + m[..., 1, 2], t[..., 3]], -1
            ),
        ],
        -2,
    )
    best = np.argmax(t, axis=-1)[..., None, None]
    q = np.take_along_axis(cand, np.broadcast_to(best, best.shape[:-2] + (1, 4)), axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)
