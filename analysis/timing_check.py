import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import load
np.set_printoptions(precision=4, suppress=True, linewidth=160)
eps = load.load_episodes("cylinder_peg/train_200k")
# 1) step duration: compare position increments with recorded linear velocity
dx = np.concatenate([np.diff(e["xpos"][:, 0], axis=0) for e in eps])
v_prev = np.concatenate([e["qvel"][:-1, :3] for e in eps])
v_next = np.concatenate([e["qvel"][1:, :3] for e in eps])
for name, v in [("v_t", v_prev), ("v_t+1", v_next), ("avg", 0.5 * (v_prev + v_next))]:
    dt_hat = (dx * v).sum() / (v * v).sum()
    r = np.corrcoef(dx.ravel(), v.ravel())[0, 1]
    print(f"dx vs {name}: implied dt = {dt_hat:.4f} s, corr {r:.4f}")
# 2) is qvel equal to a finite difference of poses?  (future leakage check)
for name, v in [("forward diff (x_t+1 - x_t)", "fwd"), ("backward diff (x_t - x_t-1)", "bwd")]:
    errs = []
    for e in eps[:50]:
        x = e["xpos"][:, 0]; q = e["qvel"][:, :3]
        if v == "fwd":
            fd = (x[1:] - x[:-1]); qq = q[:-1]
        else:
            fd = (x[1:] - x[:-1]); qq = q[1:]
        s = (fd * qq).sum() / (qq * qq).sum()
        errs.append(np.linalg.norm(fd - s * qq) / np.linalg.norm(fd))
    print(f"relative mismatch qvel vs {name}: {np.mean(errs):.3f}")
# 3) how does the commanded wrench change over time (10 Hz target vs smooth)?
F = eps[0]["ft_seq"][:, 1]
print("first 12 commanded Fx:", F[:12, 0])
chg = np.mean(np.abs(np.diff(F, axis=0)) > 1e-9)
print("fraction of steps where command changes:", chg)
