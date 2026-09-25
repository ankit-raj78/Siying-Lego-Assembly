"""MuJoCo-Adm: the exported abstract scene (free tool + hex socket + table) turned into an
admittance-controlled system, plus per-contact feature extraction for LCR.

Physics per 0.1 s data step (quasi-static admittance, see design doc F1/F2):
    D * v = F_cmd + F_contact (+ dW, the learned residual)
MuJoCo integrates this with implicit damping over 0.1 s / sub_dt substeps.
"""
import json
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from . import data as D

TOOL_BODY = 1
MAX_CONTACTS = 10
GRID = np.stack(np.meshgrid(*[np.array([-2e-3, 0.0, 2e-3])] * 3, indexing="ij"), -1).reshape(-1, 3)
N_FEAT = 1 + 3 + 3 + 3 + 3 + 2 * len(GRID)          # = 67
N_GLOBAL = 12


@dataclass
class SimParams:
    # free-space fit: v_next = a*v_prev + b*F per axis (see analysis/free_space_fit.py)
    damping: tuple = (1044.0, 955.0, 1336.0, 13.1, 13.1, 13.4)
    armature: tuple = (403.0, 876.0, 529.0, 0.57, 1.12, 0.30)   # virtual mass / inertia per dof
    mass: float = 1.0
    inertia: float = 1e-3
    friction: float = 0.6               # stage-0 sysid (results/sysid.json)
    solref_tc: float = 0.1
    solref_dr: float = 1.0
    solimp: tuple = (0.95, 0.99, 0.001, 0.5, 2.0)
    margin: float = 3e-3
    sub_dt: float = 0.005                # MuJoCo substep; 0.1 s / sub_dt substeps per data step
    cmd_at_origin: bool = True           # commanded torque acts about the tool origin (fits data better)
    inflate: dict = field(default_factory=dict)      # per tool-tip type ('cylinder','hex','square'): surface inflation [m]
    socket_offset: tuple = (0.0, 0.0, 0.0, 0.0)      # session-level socket pose correction dx, dy, dz [m], dyaw [rad]
    extra: dict = field(default_factory=dict)


def params_from_cfg(c):
    """SimParams from a sysid2-style config (scales relative to the free-space fit)."""
    base = SimParams()
    sd, sm = c.get("damp_scale", 1.0), c.get("mass_scale", 1.0)
    da, ma = np.array(c.get("damp_axis", [1.0] * 6)), np.array(c.get("mass_axis", [1.0] * 6))
    return SimParams(damping=tuple(np.array(base.damping) * np.array([sd, sd, sd, 1, 1, 1]) * da),
                     armature=tuple(np.array(base.armature) * np.array([sm, sm, sm, 1, 1, 1]) * ma),
                     friction=c.get("friction", base.friction), solref_tc=c.get("tc", base.solref_tc),
                     solref_dr=c.get("dr", base.solref_dr),
                     solimp=(c.get("d0", base.solimp[0]),) + tuple(base.solimp[1:]),
                     inflate={k: c[f"inflate_{k}"] for k in ("cylinder", "hexagon", "square") if f"inflate_{k}" in c},
                     socket_offset=(c.get("dx", 0.0), c.get("dy", 0.0), c.get("dz", 0.0), c.get("dyaw", 0.0)))


def default_params():
    """Default base-sim parameters; LCR_SIMPARAMS=<json> selects another identified set."""
    path = os.environ.get("LCR_SIMPARAMS")
    if path:
        with open(path) as f:
            return params_from_cfg(json.load(f))
    return SimParams()


def _planes_for_geom(model, g):
    """Half-space form A x + b <= 0 (inside) of a convex geom, in the geom frame."""
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        V = np.asarray(model.mesh_vert[a:a + n], np.float64)
        eq = ConvexHull(V).equations
        return eq[:, :3], eq[:, 3]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        s = model.geom_size[g]
        A = np.vstack([np.eye(3), -np.eye(3)])
        return A, -np.concatenate([s, s])
    raise ValueError(f"unsupported geom type {t}")


class AdmSim:
    def __init__(self, split, params: SimParams = None):
        params = params or default_params()
        self.split = split
        self.p = params
        m = mujoco.MjModel.from_xml_path(D.model_path(split))
        m.opt.gravity[:] = 0.0
        m.opt.timestep = params.sub_dt
        self.n_sub = int(round(D.DT / params.sub_dt))
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.dof_damping[:] = params.damping
        m.dof_armature[:] = params.armature
        m.body_mass[TOOL_BODY] = params.mass
        m.body_inertia[TOOL_BODY] = params.inertia
        collide = lambda g: (m.geom_contype[g] | m.geom_conaffinity[g]) != 0
        self.tool_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == TOOL_BODY and collide(g)]
        self.env_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and collide(g)
                          and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
        self.tool_type = next((k for k in ("cylinder", "hexagon", "square") if k in split), "unknown")
        for g in self.tool_geoms:            # detect near contacts without creating force
            a = params.inflate.get(self.tool_type, 0.0) if "tool_tip" in m.geom(g).name else 0.0
            m.geom_margin[g] = params.margin + a      # force only when dist < a (surface inflated by a)
            m.geom_gap[g] = params.margin
        dx, dy, dz, dyaw = params.socket_offset
        if any(params.socket_offset):
            socket = [g for g in self.env_geoms if m.geom(g).name.startswith("socket")]
            pivot = m.geom_pos[socket].mean(0)       # compiled origins differ per piece (mesh centroids)
            qy = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
            Ry = D.quat_to_mat(qy)
            for g in socket:                         # rigid yaw about the common pivot, then translate
                m.geom_pos[g] = pivot + Ry @ (m.geom_pos[g] - pivot) + np.array([dx, dy, dz])
                m.geom_quat[g] = D.quat_mul(qy, m.geom_quat[g])
        for g in self.env_geoms:
            m.geom_friction[g, 0] = params.friction
            m.geom_solref[g] = [params.solref_tc, params.solref_dr]
            m.geom_solimp[g] = params.solimp
        self.m = m
        self.d = mujoco.MjData(m)
        self.tool_set = set(self.tool_geoms)
        self.planes = {g: _planes_for_geom(m, g) for g in self.tool_geoms + self.env_geoms}
        self.com_local = m.body_ipos[TOOL_BODY].copy()
        self._c6 = np.zeros(6)

    # ------------------------------------------------------------ state io
    def set_state(self, pos, quat, v_world=None, w_world=None):
        d = self.d
        d.qpos[:3] = pos
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        d.qvel[:] = 0.0
        if v_world is not None:
            R = D.quat_to_mat(quat)
            d.qvel[:3] = v_world
            d.qvel[3:6] = R.T @ w_world          # free-joint angular dofs are in the body frame
        d.xfrc_applied[:] = 0.0

    def socket_ref(self):
        """Reference point of the socket (shared mesh offset of the socket pieces)."""
        socket = [g for g in self.env_geoms if self.m.geom(g).name.startswith("socket")]
        return self.m.geom_pos[socket[0]].copy() if socket else np.zeros(3)

    def pose(self):
        return self.d.qpos[:3].copy(), self.d.qpos[3:7].copy()

    def _apply(self, wrench_origin, cmd):
        """Set xfrc_applied (world, about COM) from wrenches given about the tool origin."""
        d = self.d
        R = d.xmat[TOOL_BODY].reshape(3, 3)
        com = d.xpos[TOOL_BODY] + R @ self.com_local
        arm = d.xpos[TOOL_BODY] - com                # origin - com
        f = cmd[:3] + wrench_origin[:3]
        t_cmd = cmd[3:] + (np.cross(arm, cmd[:3]) if self.p.cmd_at_origin else 0.0)
        t = t_cmd + wrench_origin[3:] + np.cross(arm, wrench_origin[:3])
        d.xfrc_applied[TOOL_BODY, :3] = f
        d.xfrc_applied[TOOL_BODY, 3:] = t

    def contact_wrench(self):
        """Total constraint wrench on the tool: world force, world torque about tool origin."""
        d = self.d
        R = d.xmat[TOOL_BODY].reshape(3, 3)
        return np.concatenate([d.qfrc_constraint[:3], R @ d.qfrc_constraint[3:6]])

    # ------------------------------------------------------------ dynamics
    def forward(self, cmd, dW=np.zeros(6)):
        mujoco.mj_forward(self.m, self.d)          # kinematics first (for COM/arm)
        self._apply(dW, cmd)
        mujoco.mj_forward(self.m, self.d)
        return self.contact_wrench()

    def step(self, cmd, dW=np.zeros(6), chunk=2):
        """Advance one 0.1 s data step with constant command and residual (applied wrench is
        set once per data step; the tool rotates < 1 deg per step). Returns the mean contact
        wrench (sampled every `chunk` substeps) plus dW, and sets self.unstable on blow-up."""
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        self._apply(dW, cmd)
        n_bad = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        Wsum = np.zeros(6)
        n = self.n_sub // chunk
        for _ in range(n):
            mujoco.mj_step(m, d, chunk)
            Wsum += self.contact_wrench()
        self.unstable = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number != n_bad
        return Wsum / n + dW

    # ------------------------------------------------------------ LCR features
    def _sdf(self, P, geoms):
        d = self.d
        best = np.full(len(P), np.inf)
        for g in geoms:
            A, b = self.planes[g]
            R = d.geom_xmat[g].reshape(3, 3)
            local = (P - d.geom_xpos[g]) @ R
            best = np.minimum(best, (local @ A.T + b).max(1))
        return best

    def features(self, cmd, v_world, w_world):
        """Per-contact features at the current (forwarded) state.
        Returns dict with feats (K,67), R (K,3,3 rows n,t1,t2), r (K,3), lam (K,3), mask (K,), glob (12,)."""
        m, d = self.m, self.d
        K = MAX_CONTACTS
        out = {"feats": np.zeros((K, N_FEAT), np.float32), "R": np.zeros((K, 3, 3), np.float32),
               "r": np.zeros((K, 3), np.float32), "lam": np.zeros((K, 3), np.float32),
               "mask": np.zeros(K, np.float32)}
        xo = d.xpos[TOOL_BODY]
        Rt = d.xmat[TOOL_BODY].reshape(3, 3)
        cands = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom2 in self.tool_set and c.geom1 not in self.tool_set:
                sgn = 1.0
            elif c.geom1 in self.tool_set and c.geom2 not in self.tool_set:
                sgn = -1.0
            else:
                continue
            cands.append((c.dist, i, sgn))
        cands.sort()
        rows = []
        for k, (dist, i, sgn) in enumerate(cands[:K]):
            c = d.contact[i]
            fr = c.frame.reshape(3, 3)
            n = sgn * fr[0]                                  # points from environment into tool
            f_world = np.zeros(3)
            if c.efc_address >= 0:
                mujoco.mj_contactForce(m, d, i, self._c6)
                f_world = fr.T @ self._c6[:3]                # force on geom2
                if sgn < 0:
                    f_world = -f_world
            t1 = Rt[:, 2] - n * (n @ Rt[:, 2])
            if np.linalg.norm(t1) < 1e-3:
                t1 = np.array([1.0, 0, 0]) - n * n[0]
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            Rc = np.stack([n, t1, t2])
            r = c.pos - xo
            vp = v_world + np.cross(w_world, r)
            out["R"][k] = Rc
            out["r"][k] = r
            out["lam"][k] = Rc @ f_world
            out["mask"][k] = 1.0
            rows.append((k, dist, Rc, r, vp, c.pos.copy()))
        if rows:
            P = np.concatenate([pc + GRID @ Rc for (_, _, Rc, _, _, pc) in rows])
            s_tool = np.clip(self._sdf(P, self.tool_geoms), -0.01, 0.01).reshape(len(rows), -1)
            s_env = np.clip(self._sdf(P, self.env_geoms), -0.01, 0.01).reshape(len(rows), -1)
            for j, (k, dist, Rc, r, vp, _) in enumerate(rows):
                out["feats"][k] = np.concatenate([
                    [dist / 3e-3], (Rc @ vp) / 5e-3, (Rc @ cmd[:3]) / 10.0,
                    out["lam"][k] / 10.0, (Rc @ r) / 0.05, s_tool[j] / 3e-3, s_env[j] / 3e-3])
        out["glob"] = np.concatenate([v_world / 5e-3, w_world / 0.05, cmd[:3] / 10.0, cmd[3:]]).astype(np.float32)
        return out
