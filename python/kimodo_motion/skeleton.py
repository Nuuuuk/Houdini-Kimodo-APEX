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


def _mixamo_names():
    names = {
        "Hips": "Hips",
        "Spine1": "Spine",
        "Spine2": "Spine1",
        "Chest": "Spine2",
        "Neck1": "Neck",
        "Neck2": "Neck1",
        "Head": "Head",
        "HeadEnd": "HeadTop_End",
        "Jaw": "Jaw",
    }
    for side in ("Left", "Right"):
        names.update(
            {
                f"{side}Eye": f"{side}Eye",
                f"{side}Shoulder": f"{side}Shoulder",
                f"{side}Arm": f"{side}Arm",
                f"{side}ForeArm": f"{side}ForeArm",
                f"{side}Hand": f"{side}Hand",
                f"{side}Leg": f"{side}UpLeg",
                f"{side}Shin": f"{side}Leg",
                f"{side}Foot": f"{side}Foot",
                f"{side}ToeBase": f"{side}ToeBase",
                f"{side}ToeEnd": f"{side}Toe_End",
            }
        )
        thumb = f"{side}HandThumb"
        names.update(
            {f"{thumb}1": f"{thumb}1", f"{thumb}2": f"{thumb}2", f"{thumb}3": f"{thumb}3", f"{thumb}End": f"{thumb}4"}
        )
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            base = f"{side}Hand{finger}"
            names.update({f"{base}{i}": f"{base}{i - 1}" for i in range(1, 5)})
            names[f"{base}End"] = f"{base}4"
    return tuple(names[j] for j in JOINT_NAMES)


OUTPUT_NAMES = _mixamo_names()


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


def build_geometry(geo, motion, index, names=JOINT_NAMES):
    import hou

    positions, transform = pose_arrays(motion, index)
    write_geometry(geo, positions, transform, names=names)


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
    for upper, mid, tip in LIMB_INDICES:
        if JOINT_NAMES[tip] in tips:
            solve_limb(world, positions, upper, mid, tip)
    return positions, world


def rotation_between(a, b):
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def solve_limb(world, positions, upper, mid, tip):
    """exact two bone solve, in place"""
    rest = rest_positions()
    bone1, bone2 = rest[mid] - rest[upper], rest[tip] - rest[mid]
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
    was = (world[upper] @ bone1 / l1, world[mid] @ bone2 / l2)
    now = ((middle - start) / l1, (end - middle) / l2)
    planes = [np.cross(*bones) for bones in (was, now)]
    if min(np.linalg.norm(n) for n in planes) > 1e-6:
        frames = [
            np.column_stack([bones[0], n / np.linalg.norm(n), np.cross(bones[0], n / np.linalg.norm(n))])
            for bones, n in zip((was, now), planes)
        ]
        turn = frames[1] @ frames[0].T
    else:
        turn = rotation_between(was[0], now[0])
    world[upper] = turn @ world[upper]
    world[mid] = rotation_between(turn @ was[1], now[1]) @ turn @ world[mid]
    positions[mid] = middle


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


def write_geometry(geo, positions, transform, with_root=False, tags=None, names=JOINT_NAMES):
    """kinefx skeleton from joint positions and transforms"""
    import hou

    positions, transform = (
        np.asarray(positions, dtype=np.float64),
        np.asarray(transform, dtype=np.float64).reshape(-1, 9),
    )
    names = list(names)
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
LIMB_INDICES = [tuple(JOINT_NAMES.index(j) for j in limb) for limb in LIMBS.values()]
IK_JOINTS = tuple(j for limb in LIMBS.values() for j in limb)
FK_JOINTS = tuple(j for j in JOINT_NAMES if j != "Hips" and j not in IK_JOINTS)
FK_HIDDEN = tuple(j for j in FK_JOINTS if j.endswith("End") or j in ("Jaw", "LeftEye", "RightEye"))


def _fk_control_names():
    names = {
        "Spine1": "C_spine_01",
        "Spine2": "C_spine_02",
        "Chest": "C_chest",
        "Neck1": "C_neck_01",
        "Neck2": "C_neck_02",
        "Head": "C_head",
        "HeadEnd": "C_headEnd",
        "Jaw": "C_jaw",
    }
    for side, s in (("Left", "L"), ("Right", "R")):
        names.update(
            {
                f"{side}Eye": f"{s}_eye",
                f"{side}Shoulder": f"{s}_clavicle",
                f"{side}ToeBase": f"{s}_ball",
                f"{side}ToeEnd": f"{s}_ballEnd",
            }
        )
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for j in JOINT_NAMES:
                if j.startswith(f"{side}Hand{finger}"):
                    part = j[len(f"{side}Hand{finger}") :]
                    names[j] = f"{s}_{finger.lower()}End" if part == "End" else f"{s}_{finger.lower()}_{int(part):02d}"
    return {j: names[j] + "_fk" for j in FK_JOINTS}


FK_CONTROL = _fk_control_names()
FK_CONTROLS = tuple(FK_CONTROL[j] for j in FK_JOINTS if j not in FK_HIDDEN)
ROOT_CONTROL, PELVIS_CONTROL = "C_root_ctrl", "C_pelvis_ctrl"


REST_TRANSFORM = "rest_transform"


def rest_transforms():
    """the rest pose as kinefx rest_transform values"""
    out = np.tile(np.eye(4), (NUM_JOINTS, 1, 1))
    out[:, 3, :3] = rest_positions()
    return out.reshape(NUM_JOINTS, 16)


def write_rest_transform(geo):
    import hou

    geo.addAttrib(hou.attribType.Point, REST_TRANSFORM, tuple(np.eye(4).ravel().tolist()))
    geo.setPointFloatAttribValues(REST_TRANSFORM, rest_transforms().ravel().tolist())


def rest_positions():
    with open(os.path.join(os.path.dirname(__file__), "soma77_rest.json")) as f:
        return np.array(json.load(f))


def build_rest_geometry(geo):
    tags = {joint: [tag] for tag, joints in LIMBS.items() for joint in joints}
    write_geometry(geo, rest_positions(), np.tile(np.eye(3).reshape(9), (NUM_JOINTS, 1)), with_root=True, tags=tags)
