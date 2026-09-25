import sys, time, numpy as np, mujoco
sys.path.insert(0, sys.argv[1])
import load
model, eps = load.load_split("hexagon_peg/test_25k")
model.opt.gravity[:] = 0
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
model.dof_damping[:] = [915, 898, 1122, 12.4, 13.0, 11.7]
model.body_mass[1] = 5.0; model.body_inertia[1] = [0.05, 0.05, 0.05]
for g in range(model.ngeom):
    if model.geom_bodyid[g] == 1: model.geom_margin[g] = 0.003
data = mujoco.MjData(model)
e = eps[0]
data.qpos[:] = e["qpos"][0]
N = 300; ncons = []
t0 = time.perf_counter()
for t in range(N):
    data.xfrc_applied[1] = e["ft_seq"][t % 500, 1]
    for _ in range(50):
        mujoco.mj_step(model, data)
    ncons.append(data.ncon)
dt = (time.perf_counter() - t0) / N
print(f"one 0.1 s step (50 substeps, 1 CPU core): {dt*1000:.2f} ms;  contacts/step mean {np.mean(ncons):.1f} max {np.max(ncons)}")
# per-contact MLP cost estimate (numpy, 12 contacts, 138->64->64->3)
W1 = np.random.randn(138, 64); W2 = np.random.randn(64, 64); W3 = np.random.randn(64, 3); x = np.random.randn(12, 138)
t0 = time.perf_counter()
for _ in range(2000):
    h = np.tanh(x @ W1); h = np.tanh(h @ W2); y = h @ W3
print(f"per-contact MLP for 12 contacts (numpy): {(time.perf_counter()-t0)/2000*1e3:.3f} ms")
