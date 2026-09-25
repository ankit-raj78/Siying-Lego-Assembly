"""Glue between the precomputed caches and the skin representation.

The caches store the tool pose (relative to the socket reference), the global kinematics and the
base simulator's per-contact forces; the dense surface features are recomputed on the fly per
batch with `features_torch`, so no new precompute is needed when the representation changes.
"""
import numpy as np
import torch

from . import data as D
from .sim import AdmSim
from .skin import SkinGeometry, features_torch

TOOLS = ["cylinder", "hexagon", "square"]
SPLIT_FOR_TOOL = {"cylinder": "cylinder_peg/test_25k", "hexagon": "hexagon_peg/test_25k", "square": "square_peg/test_25k"}
CACHE_SPLITS = {
    "train": (D.TRAIN_SPLITS, 2, 0), "train_off1": (D.TRAIN_SPLITS, 2, 1), "val": (D.VAL_SPLITS, 2, 0),
    "test_seen": (D.TEST_SETS["test_seen"], 2, 0), "square_unseen": (D.TEST_SETS["square_unseen"], 2, 0),
    "expert_ood": (D.TEST_SETS["expert_ood"], 1, 0), "circle_ood": (D.TEST_SETS["circle_ood"], 1, 0),
}


def tool_of_split(split):
    return next(i for i, k in enumerate(TOOLS) if k in split)


def tool_ids(cache_name, n_rows):
    """Tool id per cache row, reconstructed from the deterministic row order of precompute.py
    (asserts that no row was dropped). Caches written with a 'tool' field don't need this."""
    base = cache_name
    for pre in ("s2_", "s3_"):
        if base.startswith(pre):
            base = base[len(pre):]
    splits, stride, offset = CACHE_SPLITS[base]
    ids = []
    for s in splits:
        T = 200 if ("expert" in s or "circle" in s) else 500
        per_ep = len(range(2 + offset, T - 1, stride))
        ids.append(np.full(D.manifest()[s]["n_episodes"] * per_ep, tool_of_split(s), np.int64))
    ids = np.concatenate(ids)
    assert len(ids) == n_rows, f"{cache_name}: expected {len(ids)} rows, cache has {n_rows}"
    return ids


class GeometryBundle:
    """Sample points for every tool (stacked, indexed by tool id) and the shared environment planes."""

    def __init__(self):
        geos, refs = {}, []
        for tool, split in SPLIT_FOR_TOOL.items():
            sim = AdmSim(split)
            geos[tool] = SkinGeometry(sim)
            refs.append(sim.socket_ref())
        assert np.allclose(refs[0], refs[1]) and np.allclose(refs[0], refs[2]), "socket reference differs between tools"
        self.ref = torch.tensor(refs[0], dtype=torch.float32)
        K = min(g.K for g in geos.values())
        t = lambda key: torch.stack([torch.tensor(np.asarray(getattr(geos[k], key)[:K]), dtype=torch.float32) for k in TOOLS])
        self.pts, self.nrm, self.edge = t("pts"), t("nrm"), t("edge")
        self.A = torch.tensor(geos["cylinder"].A, dtype=torch.float32)
        self.b = torch.tensor(geos["cylinder"].b, dtype=torch.float32)
        self.np_geos = geos

    def batch(self, b):
        """Build the SkinNet input dict from a cache batch (pose, glob, tool, R, r, lam, mask)."""
        pose = b["pose"]
        pos = pose[:, :3] * 0.05 + self.ref
        x, y = pose[:, 3:6], pose[:, 6:9]
        R = torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)          # columns = body axes
        v, w = b["glob"][:, :3] * 5e-3, b["glob"][:, 3:6] * 0.05
        tool = b["tool"].long()
        geo = {"pts": self.pts[tool], "nrm": self.nrm[tool], "edge": self.edge[tool], "A": self.A, "b": self.b}
        feats, mask, r_w, g_w = features_torch(geo, pos, R, v, w)
        f_w = torch.einsum("bkij,bki->bkj", b["R"], b["lam"]) * b["mask"][..., None]   # per-contact sim forces, world
        W_inst = torch.cat([f_w.sum(1), torch.cross(b["r"], f_w, dim=-1).sum(1)], -1)
        return {"feats": feats, "mask": mask, "r": r_w, "g": g_w, "glob": b["glob"], "W0": W_inst}
