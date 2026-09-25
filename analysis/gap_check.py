"""When the real F/T says 'pushing', how far apart are tool and environment in the sim?"""
import sys, numpy as np, mujoco
sys.path.insert(0, sys.argv[1])
import load
np.set_printoptions(precision=2, suppress=True, linewidth=160)
split = sys.argv[2]
model, eps = load.load_split(split)
tool = 1
tool_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == tool and model.geom_contype[g] | model.geom_conaffinity[g]]
env_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0 and (model.geom_contype[g] | model.geom_conaffinity[g]) and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
names = lambda gs: [model.geom(g).name for g in gs]
print("tool geoms:", names(tool_geoms)); print("env geoms:", names(env_geoms))
data = mujoco.MjData(model)
fromto = np.zeros(6)
rows = []
for e in eps[:30]:
    for t in range(e["qpos"].shape[0]):
        data.qpos[:] = e["qpos"][t]; mujoco.mj_kinematics(model, data); mujoco.mj_collision(model, data)
        best = (1e9, None)
        for gt in tool_geoms:
            for ge in env_geoms:
                d = mujoco.mj_geomDistance(model, data, gt, ge, 0.02, fromto)
                if d < best[0]:
                    best = (d, ge, fromto.copy())
        Fm = e["wrench_data"][t]
        n = best[2][3:] - best[2][:3] if best[1] is not None else np.zeros(3)
        rows.append([best[0], np.linalg.norm(Fm[:3]), Fm[2], best[1]])
R = np.array(rows, dtype=float)
push = R[:, 1] > 2.0; free = R[:, 1] < 0.3
print(f"steps {len(R)}; measured |F|>2N: {push.mean():.2f}; |F|<0.3N: {free.mean():.2f}")
for name, m in [("measured PUSHING (|F|>2N)", push), ("measured FREE (|F|<0.3N)", free)]:
    d = R[m, 0] * 1000
    print(f"{name:28s} sim min distance [mm]: p10 {np.percentile(d,10):6.2f}  p50 {np.percentile(d,50):6.2f}  p90 {np.percentile(d,90):6.2f}   frac(d<0.1mm) {np.mean(d<0.1):.2f}")
# which env geom is closest when pushing
ge = R[push, 3].astype(int)
u, c = np.unique(ge, return_counts=True)
print("closest env geom when pushing:", {model.geom(int(g)).name: int(k) for g, k in zip(u, c)})
