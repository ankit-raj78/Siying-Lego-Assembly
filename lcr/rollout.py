"""Rollouts and metrics shared by system identification, training and evaluation."""
import numpy as np

from . import data as D


def sim_velocity_from_history(P, Q):
    """Backward-difference world velocity from the last two poses of a (predicted) history."""
    v = (P[-1] - P[-2]) / D.DT
    w = D.rot_delta_world(Q[-2], Q[-1]) / D.DT
    return v, w


def rollout(sim, ep, t0, H, residual=None):
    """Open-loop rollout from the true state at t0 using actions a_t0..a_{t0+H-1}.
    residual(sim, cmd, v, w) -> dW (6,) is evaluated at the start of every data step.
    Returns predicted positions/quats for t0+1..t0+H, predicted wrench per step, unstable flag."""
    pos, quat = ep["pos"], ep["quat"]
    v, w = D.backward_velocity(pos, quat, t0)
    sim.set_state(pos[t0], quat[t0], v, w)
    Ph, Qh = [pos[t0 - 1], pos[t0]], [quat[t0 - 1], quat[t0]]
    P, Q, W = [], [], []
    bad = False
    for k in range(H):
        a = ep["action"][t0 + k]
        dW = np.zeros(6)
        if residual is not None:
            vv, ww = sim_velocity_from_history(Ph, Qh)
            sim.forward(a)
            dW = residual(sim, a, vv, ww)
        Wk = sim.step(a, dW)
        bad |= sim.unstable
        p, q = sim.pose()
        P.append(p); Q.append(q); W.append(Wk)
        Ph.append(p); Qh.append(q)
    return np.array(P), np.array(Q), np.array(W), bad


def segment_starts(n_steps, H, n_per_ep, rng):
    hi = n_steps - H - 1
    if hi <= 2:
        return []
    return sorted(rng.choice(np.arange(2, hi), size=min(n_per_ep, hi - 2), replace=False))


def pose_errors(P, Q, ep, t0):
    H = len(P)
    ep_ = np.linalg.norm(P - ep["pos"][t0 + 1:t0 + H + 1], axis=1)
    ea = D.angle_between(Q, ep["quat"][t0 + 1:t0 + H + 1])
    return ep_, ea


def wrench_metrics(pred, true):
    """Per-axis-mean RMSE and vector-norm RMSE for force and torque."""
    E = pred - true
    f_axis = np.sqrt((E[:, :3] ** 2).mean(0)); t_axis = np.sqrt((E[:, 3:] ** 2).mean(0))
    return {"force_axis_mean": float(f_axis.mean()),
            "force_vec": float(np.sqrt((np.linalg.norm(E[:, :3], axis=1) ** 2).mean())),
            "torque_axis_mean": float(t_axis.mean()),
            "torque_vec": float(np.sqrt((np.linalg.norm(E[:, 3:], axis=1) ** 2).mean()))}
