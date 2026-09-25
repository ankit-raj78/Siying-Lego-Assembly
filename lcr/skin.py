"""Dense surface-anchored contact field ("virtual tactile skin").

Fixed sample points on the tool's convex collision surfaces (with outward normals and distance to
the nearest hull edge) are transformed with the tool pose every step, and the environment's signed
distance is evaluated at each point (max over planes of each convex piece, min over pieces). Points
closer than `margin` are active. This replaces the solver-selected contact points as the unit the
network reasons about; the base simulator still handles non-penetration and admittance dynamics.

Both a numpy path (rollouts) and a batched torch path (training from cached poses) are provided.
"""
import numpy as np
import torch
from scipy.spatial import ConvexHull

from . import data as D
from .sim import TOOL_BODY, _planes_for_geom

K_POINTS = 384
EDGE_STEP = 4e-3         # one edge sample per 4 mm of hull edge length
TIP_FRACTION = 0.8       # share of points on the tool tip (the part that enters the socket)
MARGIN = 3e-3
N_SKIN_FEAT = 19
ANCHOR_RADIUS = 1.0      # every solver contact is attached to its nearest skin point (radius effectively unlimited)
N_ACTIVE = 64            # points kept per sample (closest to the environment); the rest carry no force


def _sample_hull(V, n, rng):
    """n points on the convex hull of V: all hull vertices and edge midpoints first (contacts of
    polygonal pegs happen at edges and corners), the remainder area-weighted on faces.
    Returns points, outward normals (vertex/edge normals = mean of adjacent faces), edge distance."""
    hull = ConvexHull(V)
    tri = V[hull.simplices]
    fn = hull.equations[:, :3]
    edges_raw = np.concatenate([hull.simplices[:, [0, 1]], hull.simplices[:, [1, 2]], hull.simplices[:, [0, 2]]])
    face_of_edge = np.repeat(np.arange(len(tri)), 3)
    key = np.sort(edges_raw, axis=1)
    edges, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    edge_n = np.zeros((len(edges), 3))
    np.add.at(edge_n, inv, fn[face_of_edge])
    vert_ids = np.unique(hull.simplices)
    vert_n = np.zeros((len(V), 3))
    for f, sim in enumerate(hull.simplices):
        vert_n[sim] += fn[f]
    unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    P = [V[vert_ids]]; N = [unit(vert_n[vert_ids])]
    # interior edge samples, one per EDGE_STEP of length (deepest contact of a polygonal peg lies on an edge)
    p0, p1 = V[edges[:, 0]], V[edges[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    cand_e, cand_t = [], []
    for e, L in enumerate(lengths):
        m = int(L / EDGE_STEP)
        if m > 0:
            ts = np.linspace(0, 1, m + 2)[1:-1]
            cand_e += [e] * len(ts); cand_t += list(ts)
    cand_e, cand_t = np.array(cand_e, int), np.array(cand_t)
    budget = max(n - len(P[0]), 0)
    if len(cand_e) > budget:
        keep = rng.choice(len(cand_e), size=budget, replace=False)
        cand_e, cand_t = cand_e[keep], cand_t[keep]
    P.append(p0[cand_e] + cand_t[:, None] * (p1[cand_e] - p0[cand_e])); N.append(unit(edge_n[cand_e]))
    n_face = max(n - len(P[0]) - len(P[1]), 0)
    if n_face > 0:
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        idx = rng.choice(len(tri), size=n_face, p=areas / areas.sum())
        r1, r2 = np.sqrt(rng.random(n_face)), rng.random(n_face)
        a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
        P.append((1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c)
        N.append(fn[idx])
    pts = np.concatenate(P)[:n]; nrm = np.concatenate(N)[:n]
    p0, p1 = V[edges[:, 0]], V[edges[:, 1]]
    d = p1 - p0
    t = np.clip(((pts[:, None, :] - p0[None]) * d[None]).sum(-1) / (d * d).sum(-1)[None], 0, 1)
    closest = p0[None] + t[..., None] * d[None]
    edge_dist = np.linalg.norm(pts[:, None, :] - closest, axis=-1).min(1)
    return pts, nrm, edge_dist


class SkinGeometry:
    """Static geometry for one scene: tool sample points (body frame) and environment planes (world)."""

    def __init__(self, sim, k=K_POINTS, seed=0):
        m = sim.m
        rng = np.random.default_rng(seed)
        tips = [g for g in sim.tool_geoms if "tool_tip" in m.geom(g).name]
        pts, nrm, edg = [], [], []
        for g in sim.tool_geoms:
            n_g = int(k * TIP_FRACTION / max(len(tips), 1)) if g in tips else \
                int(k * (1 - TIP_FRACTION) / max(len(sim.tool_geoms) - len(tips), 1))
            V = self._verts(m, g)
            p, n, e = _sample_hull(V, n_g, rng)
            Rg = D.quat_to_mat(m.geom_quat[g])
            pts.append(m.geom_pos[g] + p @ Rg.T)
            nrm.append(n @ Rg.T)
            edg.append(e)
        self.pts = np.concatenate(pts)[:k]                 # (K,3) body frame
        self.nrm = np.concatenate(nrm)[:k]
        self.edge = np.concatenate(edg)[:k]
        self.K = len(self.pts)
        # environment pieces as world-frame half-spaces, padded to a common plane count
        mujoco_forward(sim)
        pieces = []
        for g in sim.env_geoms:
            A, b = sim.planes[g]
            R = sim.d.geom_xmat[g].reshape(3, 3)
            Aw = A @ R.T
            pieces.append((Aw, b - Aw @ sim.d.geom_xpos[g]))
        P = max(len(A) for A, _ in pieces)
        self.A = np.zeros((len(pieces), P, 3)); self.b = np.full((len(pieces), P), -1e9)
        for j, (Aw, bw) in enumerate(pieces):
            self.A[j, :len(Aw)] = Aw; self.b[j, :len(bw)] = bw

    @staticmethod
    def _verts(m, g):
        import mujoco
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            return np.asarray(m.mesh_vert[a:a + n], np.float64)
        s = m.geom_size[g]
        return np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])

    # ------------------------------------------------------------ numpy path
    def sdf(self, P):
        """Signed distance and outward gradient of the environment at world points P (N,3)."""
        s = np.einsum("jpk,nk->njp", self.A, P) + self.b[None]      # (N,J,P)
        s_piece = s.max(2); kstar = s.argmax(2)                      # (N,J)
        j = s_piece.argmin(1)
        n = np.arange(len(P))
        return s_piece[n, j], self.A[j, kstar[n, j]]

    def features(self, pos, quat, v, w, cmd, contact_pos=None, contact_force=None, contact_normal=None, margin=MARGIN):
        R = D.quat_to_mat(quat)
        Pw = pos + self.pts @ R.T
        sdf, g_w = self.sdf(Pw)
        mask = (sdf < margin).astype(np.float32)
        r_w = Pw - pos
        vel_b = (v[None] + np.cross(w[None], r_w)) @ R
        g_b = g_w @ R
        f = np.concatenate([np.clip(sdf, -margin, margin)[:, None] / margin, self.nrm, g_b, self.pts / 0.05,
                            vel_b / 5e-3, self.edge[:, None] / 5e-3, (self.nrm * g_b).sum(1, keepdims=True)], 1)
        glob = np.concatenate([v / 5e-3, w / 0.05, cmd[:3] / 10.0, cmd[3:]]).astype(np.float32)
        keep = np.argsort(sdf)[:N_ACTIVE]                       # closest points only
        f, mask, r_w, g_w = f[keep], mask[keep], r_w[keep], g_w[keep]
        # anchor the base simulator's contact forces (world) on the nearest skin points
        lam = np.zeros((len(keep), 3))
        if contact_pos is not None and len(contact_pos):
            d = np.linalg.norm(contact_pos[:, None, :] - (pos + r_w)[None], axis=-1)
            j = np.zeros(len(contact_pos), int)
            for c in range(len(contact_pos)):                                    # distinct nearest point per contact
                j[c] = d[c].argmin(); d[:, j[c]] = np.inf
            g_w = g_w.copy(); r_w = r_w.copy()
            g_w[j] = contact_normal; r_w[j] = contact_pos - pos
            t1, t2 = tangent_basis_np(g_w)
            for c in range(len(contact_pos)):
                fw = contact_force[c]
                lam[j[c]] += [fw @ g_w[j[c]], fw @ t1[j[c]], fw @ t2[j[c]]]
            mask = mask.copy(); mask[j] = 1.0                                    # anchored = active
        f = np.concatenate([f, lam / 10.0, (np.abs(lam).sum(1, keepdims=True) > 0).astype(np.float64)], 1)
        return {"feats": f.astype(np.float32), "mask": mask.astype(np.float32), "r": r_w.astype(np.float32),
                "g": g_w.astype(np.float32), "lam_skin": lam.astype(np.float32), "glob": glob}

    # ------------------------------------------------------------ torch path (batched)
    def torch_tensors(self, device="cpu"):
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
        return {"pts": t(self.pts), "nrm": t(self.nrm), "edge": t(self.edge), "A": t(self.A), "b": t(self.b)}


def tangent_basis_np(g):
    e = np.where(np.abs(g[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(g, e); t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9
    return t1, np.cross(g, t1)


def tangent_basis(g):
    e = torch.where(g[..., :1].abs() < 0.9, torch.tensor([1.0, 0, 0], device=g.device).expand_as(g),
                    torch.tensor([0, 1.0, 0], device=g.device).expand_as(g))
    t1 = torch.cross(g, e, dim=-1); t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-9)
    return t1, torch.cross(g, t1, dim=-1)


def mujoco_forward(sim):
    import mujoco
    mujoco.mj_forward(sim.m, sim.d)


def features_torch(geo, pos, R, v, w, contact_pos=None, contact_force=None, contact_normal=None, contact_mask=None, margin=MARGIN):
    """geo: dict of per-sample tensors (pts (B,K,3), nrm (B,K,3), edge (B,K)) and env (A (J,P,3), b (J,P)).
    pos (B,3), R (B,3,3) body->world, v/w (B,3) world; solver contacts (B,Kc,3) positions/forces (world) + mask.
    Returns feats (B,Ka,19), mask, r_w, g_w, lam_skin (anchored solver force in the point frame)."""
    pts, nrm, edge, A, b = geo["pts"], geo["nrm"], geo["edge"], geo["A"], geo["b"]
    Pw = pos[:, None] + torch.einsum("bij,bkj->bki", R, pts)                 # (B,K,3)
    s = torch.einsum("jpi,bki->bkjp", A, Pw) + b[None, None]                 # (B,K,J,P)
    s_piece, kstar = s.max(-1)                                               # (B,K,J)
    sdf, j = s_piece.min(-1)                                                 # (B,K)
    g_w = A[j, torch.gather(kstar, 2, j[..., None])[..., 0]]                 # (B,K,3)
    mask = (sdf < margin).float()
    r_w = Pw - pos[:, None]
    vel_w = v[:, None] + torch.cross(w[:, None].expand_as(r_w), r_w, dim=-1)
    vel_b = torch.einsum("bki,bij->bkj", vel_w, R)
    g_b = torch.einsum("bki,bij->bkj", g_w, R)
    f = torch.cat([sdf.clamp(-margin, margin)[..., None] / margin, nrm, g_b, pts / 0.05, vel_b / 5e-3,
                   edge[..., None] / 5e-3, (nrm * g_b).sum(-1, keepdim=True)], -1)
    keep = torch.topk(-sdf, min(N_ACTIVE, sdf.shape[1]), dim=1).indices          # closest points only
    gather = lambda x: torch.gather(x, 1, keep[..., None].expand(-1, -1, x.shape[-1])) if x.dim() == 3 else torch.gather(x, 1, keep)
    f, mask, r_w, g_w = gather(f), gather(mask), gather(r_w), gather(g_w)
    # anchor the base simulator's contact forces on the nearest skin points; those points adopt the
    # contact's normal and position so the initial wrench equals the simulator's exactly
    B, Ka = f.shape[:2]
    lam = torch.zeros(B, Ka, 3, device=f.device)
    if contact_pos is not None:
        d = torch.cdist(contact_pos, pos[:, None] + r_w)                          # (B,Kc,Ka)
        taken = torch.zeros(B, Ka, dtype=torch.bool, device=f.device)
        js = []
        for c in range(d.shape[1]):                                              # distinct nearest point per contact
            dc = d[:, c].masked_fill(taken, float("inf"))
            jc = dc.argmin(-1)
            js.append(jc)
            taken[torch.arange(B), jc] |= contact_mask[:, c] > 0
        j = torch.stack(js, 1)
        ok = (contact_mask > 0)[..., None]
        j3 = j[..., None].expand(-1, -1, 3)
        g_w = g_w.scatter(1, j3, torch.where(ok, contact_normal, torch.gather(g_w, 1, j3)))
        r_w = r_w.scatter(1, j3, torch.where(ok, contact_pos - pos[:, None], torch.gather(r_w, 1, j3)))
        t1, t2 = tangent_basis(g_w)
        gj, t1j, t2j = (torch.gather(x, 1, j3) for x in (g_w, t1, t2))
        fw = contact_force * ok
        loc = torch.stack([(fw * gj).sum(-1), (fw * t1j).sum(-1), (fw * t2j).sum(-1)], -1)   # (B,Kc,3)
        lam.scatter_add_(1, j3, loc)
        mask = mask.scatter(1, j, torch.where(ok[..., 0], torch.ones_like(j, dtype=mask.dtype), torch.gather(mask, 1, j)))  # anchored = active
    f = torch.cat([f, lam / 10.0, (lam.abs().sum(-1, keepdim=True) > 0).float()], -1)
    return f, mask, r_w, g_w, lam
