"""Loading the Act-FIGNet real-world episode pack and small rotation helpers.

Conventions used throughout the project
  * one data step = 0.1 s (10 Hz); the ``timestep`` field in the pack is not in seconds
  * pose  = tool body origin position (world, m) + orientation quaternion (w, x, y, z)
  * action a_t = ft_seq[t, 1]  -> commanded wrench (world frame), drives motion t -> t+1
  * observation o_t = wrench_data[t] -> measured F/T (world-aligned axes, about tool origin)
"""
import json
import os

import numpy as np

DT = 0.1
DATA_ROOT = os.environ.get(
    "LCR_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "real_world_datasets_pack"))

TRAIN_SPLITS = ["cylinder_peg/train_200k", "hexagon_peg/train_200k"]
VAL_SPLITS = ["cylinder_peg/val_10k", "hexagon_peg/val_10k"]
SYSID_SPLITS = ["cylinder_peg/sysid_10k", "hexagon_peg/sysid_10k"]
FREE_SPLIT = "cylinder_peg/sysid_no_con_10k"
TEST_SETS = {
    "test_seen": ["cylinder_peg/test_25k", "hexagon_peg/test_25k"],
    "square_unseen": ["square_peg/test_25k"],
    "expert_ood": ["cylinder_peg/expert_10k", "hexagon_peg/expert_10k"],
    "circle_ood": ["cylinder_peg/circle/radius_0.02_numCircles_1_epLen_200_steps_10k",
                   "hexagon_peg/circle/radius_0.02_numCircles_1_epLen_200_steps_10k"],
}


def manifest():
    with open(os.path.join(DATA_ROOT, "manifest.json")) as f:
        return json.load(f)


def model_path(split):
    return os.path.join(DATA_ROOT, manifest()[split]["abstract"])


def load_episodes(split):
    raw = np.load(os.path.join(DATA_ROOT, split, "episode.npz"), allow_pickle=True)
    key = "episodes" if "episodes" in raw else raw.files[0]
    eps = raw[key]
    eps = list(eps.item() if eps.ndim == 0 else eps)
    out = []
    for e in eps:
        out.append({
            "pos": np.asarray(e["xpos"][:, 0], np.float64),
            "quat": np.asarray(e["xquat"][:, 0], np.float64),
            "qpos": np.asarray(e["qpos"], np.float64),
            "action": np.asarray(e["ft_seq"][:, 1], np.float64),
            "obs": np.asarray(e["wrench_data"], np.float64),
        })
    return out


# ---------------------------------------------------------------- rotations (wxyz)
def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_rotvec(q):
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., :1])
    return np.where(s > 1e-12, v / np.maximum(s, 1e-12) * ang, 2.0 * v)


def rotvec_to_quat(r):
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * ang
    k = np.where(ang > 1e-12, np.sin(half) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([np.cos(half), r * k], -1)


def quat_to_mat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot_delta_world(q_from, q_to):
    """World-frame rotation vector taking orientation q_from to q_to."""
    return quat_to_rotvec(quat_mul(q_to, quat_conj(q_from)))


def angle_between(q1, q2):
    """Geodesic angle (rad) between orientations."""
    return np.linalg.norm(rot_delta_world(q1, q2), axis=-1)


def backward_velocity(pos, quat, t):
    """World linear (m/s) and angular (rad/s) velocity from frames t-1 -> t."""
    v = (pos[t] - pos[t - 1]) / DT
    w = rot_delta_world(quat[t - 1], quat[t]) / DT
    return v, w
