"""kimodo motion to a soma 77 joint kinefx skeleton."""

import json
import os

import numpy as np

JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumbEnd",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumbEnd",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinkyEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)
PARENTS = (
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
)
NUM_JOINTS = len(JOINT_NAMES)


class Motion:
    def __init__(self, positions, global_rots, local_rots):
        self.positions = positions
        self.global_rots = global_rots
        self.local_rots = local_rots

    @property
    def num_frames(self):
        return len(self.positions)


def load_motion(path):
    with np.load(path) as npz:
        missing = {"posed_joints", "global_rot_mats", "local_rot_mats"} - set(npz.files)
        if missing:
            raise ValueError(f"{path}: not a Kimodo motion NPZ, missing {sorted(missing)}")
        m = Motion(npz["posed_joints"], npz["global_rot_mats"], npz["local_rot_mats"])
    t = m.num_frames
    for name, arr, shape in (
        ("posed_joints", m.positions, (t, NUM_JOINTS, 3)),
        ("global_rot_mats", m.global_rots, (t, NUM_JOINTS, 3, 3)),
        ("local_rot_mats", m.local_rots, (t, NUM_JOINTS, 3, 3)),
    ):
        if arr.shape != shape:
            raise ValueError(f"{path}: {name} has shape {arr.shape}, expected {shape} (SOMA 77 joints)")
    return m


def motion_from_rotations(local_rots, root_positions):
    rest = rest_positions()
    world = np.array(local_rots, dtype=np.float64)
    positions = np.zeros(world.shape[:2] + (3,))
    positions[:, 0] = root_positions
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        world[:, j] = world[:, p] @ world[:, j]
        positions[:, j] = positions[:, p] + world[:, p] @ (rest[j] - rest[p])
    return Motion(positions, world, np.asarray(local_rots, dtype=np.float64))


def pose_arrays(motion, index):
    """point positions and kinefx transforms for one frame"""
    i = min(max(int(index), 0), motion.num_frames - 1)
    transform = np.swapaxes(motion.global_rots[i], -1, -2).reshape(NUM_JOINTS, 9)
    return motion.positions[i].astype(np.float32), transform.astype(np.float32)


def build_geometry(geo, motion, index):
    import hou

    positions, transform = pose_arrays(motion, index)
    write_geometry(geo, positions, transform)


def fk(local_rots, root_position):
    """world positions and rotations of a pose"""
    rest = rest_positions()
    world = np.array(local_rots, dtype=np.float64)
    positions = np.zeros((NUM_JOINTS, 3))
    positions[0] = root_position
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        world[j] = world[p] @ world[j]
        positions[j] = positions[p] + world[p] @ (rest[j] - rest[p])
    return positions, world


def _hip_line(rotation):
    rest = rest_positions()
    line = rotation @ (rest[JOINT_NAMES.index("RightLeg")] - rest[JOINT_NAMES.index("LeftLeg")])
    length = np.hypot(line[0], line[2])
    return None if length < 1e-9 else np.array([line[0], line[2]]) / length


def with_heading(rotation, like):
    """rotation turned about y to the heading of like"""
    mine, wanted = _hip_line(rotation), _hip_line(like)
    if mine is None or wanted is None:
        return np.array(rotation, dtype=np.float64)
    c, s = wanted[0] * mine[0] + wanted[1] * mine[1], wanted[0] * mine[1] - wanted[1] * mine[0]
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ rotation


def pinned_pose(base_local, stored_local, stored_root, tips):
    """base pose with only what hand and foot constraints pin from the stored pose"""
    local = np.array(base_local, dtype=np.float64)
    local[0] = with_heading(local[0], stored_local[0])
    positions, world = fk(local, stored_root)
    stored_positions, stored_world = fk(stored_local, stored_root)
    rest = rest_positions()
    moved = set()
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        if JOINT_NAMES[j] in tips:
            positions[j], world[j] = stored_positions[j], stored_world[j]
            moved.add(j)
        elif p in moved:
            world[j] = world[p] @ local[j]
            positions[j] = positions[p] + world[p] @ (rest[j] - rest[p])
            moved.add(j)
    return positions, world


TIPS = {"left-hand": "LeftHand", "right-hand": "RightHand", "left-foot": "LeftFoot", "right-foot": "RightFoot"}


def composed_pose(base_local, base_root, stored_local, stored_root, stored_root_2d, ctypes):
    """the pose the rig follows on a frame with a stored pose"""
    if stored_local is not None and ("fullbody" in ctypes or not ctypes):
        return fk(stored_local, stored_root)
    tips = {TIPS[ctype] for ctype in ctypes if ctype in TIPS}
    if stored_local is not None and tips:
        return pinned_pose(base_local, stored_local, stored_root, tips)
    x, z = (stored_root[0], stored_root[2]) if stored_root_2d is None else stored_root_2d
    root = np.array(base_root, dtype=np.float64)
    if not (np.isnan(x) or np.isnan(z)):
        root[0], root[2] = x, z
    return fk(base_local, root)


def write_geometry(geo, positions, transform, with_root=False, tags=None):
    """kinefx skeleton from joint positions and transforms"""
    import hou

    positions, transform = (
        np.asarray(positions, dtype=np.float64),
        np.asarray(transform, dtype=np.float64).reshape(-1, 9),
    )
    names = list(JOINT_NAMES)
    if with_root:
        names.append(ROOT)
        positions = np.vstack([positions, [positions[0, 0], 0.0, positions[0, 2]]])
        transform = np.vstack([transform, np.eye(3).reshape(1, 9)])
    geo.addAttrib(hou.attribType.Point, "name", "")
    geo.addAttrib(hou.attribType.Point, "transform", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    points = geo.createPoints([tuple(p) for p in positions.tolist()])
    geo.setPointStringAttribValues("name", names)
    geo.setPointFloatAttribValues("transform", transform.ravel().tolist())
    if tags:
        geo.addArrayAttrib(hou.attribType.Point, "tags", hou.attribData.String, 1)
        for joint, joint_tags in tags.items():
            points[names.index(joint)].setAttribValue("tags", list(joint_tags))
    if with_root:
        geo.createPolygons([(points[NUM_JOINTS], points[0])], False)
    geo.createPolygons([(points[p], points[j]) for j, p in enumerate(PARENTS) if p >= 0], False)


ROOT = "Root"

LIMBS = {
    "L_arm": ("LeftArm", "LeftForeArm", "LeftHand"),
    "R_arm": ("RightArm", "RightForeArm", "RightHand"),
    "L_leg": ("LeftLeg", "LeftShin", "LeftFoot"),
    "R_leg": ("RightLeg", "RightShin", "RightFoot"),
}
IK_JOINTS = tuple(j for limb in LIMBS.values() for j in limb)
FK_JOINTS = tuple(j for j in JOINT_NAMES if j != "Hips" and j not in IK_JOINTS)
FK_HIDDEN = tuple(j for j in FK_JOINTS if j.endswith("End") or j in ("Jaw", "LeftEye", "RightEye"))
FK_CONTROLS = tuple(j for j in FK_JOINTS if j not in FK_HIDDEN)
ROOT_CONTROL, COG_CONTROL = "Root_ctrl", "COG_ctrl"


def rest_positions():
    with open(os.path.join(os.path.dirname(__file__), "soma77_rest.json")) as f:
        return np.array(json.load(f))


def build_rest_geometry(geo):
    tags = {joint: [tag] for tag, joints in LIMBS.items() for joint in joints}
    write_geometry(geo, rest_positions(), np.tile(np.eye(3).reshape(9), (NUM_JOINTS, 1)), with_root=True, tags=tags)
