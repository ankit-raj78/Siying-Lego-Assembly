"""Is the free-space one-step residual predictable (systematic) or noise-like?
Fit v_{t+1} = p v_t + q F_t + c per axis, then look at the lag-1 autocorrelation of the residual
and how much a rich nonlinear regressor (gradient boosting on [v, w, F_t, F_{t-1}, F_{t+1}]) can still explain."""
import sys, numpy as np
sys.path.insert(0, ".")
from lcr import data as D
np.set_printoptions(precision=3, suppress=True)
def collect(splits, max_eps=None):
    rows = []
    for s in splits:
        for ei, e in enumerate(D.load_episodes(s)[:max_eps]):
            P, Q, A, O = e["pos"], e["quat"], e["action"], e["obs"]
            for t in range(3, len(P) - 2):
                if max(np.linalg.norm(O[t-1:t+2, :3], axis=1)) > 0.3: continue
                v0, w0 = D.backward_velocity(P, Q, t); v1, w1 = D.backward_velocity(P, Q, t + 1)
                rows.append((ei, t, np.r_[v0, w0], np.r_[v1, w1], A[t-1], A[t], A[t+1]))
    return rows
tr = collect(D.TRAIN_SPLITS, 200); va = collect(D.VAL_SPLITS)
def arr(rows, i): return np.array([r[i] for r in rows])
res_tr, res_va = np.zeros((len(tr), 3)), np.zeros((len(va), 3))
for k in range(3):
    X = np.c_[arr(tr,2)[:,k], arr(tr,5)[:,k], np.ones(len(tr))]; c, *_ = np.linalg.lstsq(X, arr(tr,3)[:,k], rcond=None)
    res_tr[:,k] = arr(tr,3)[:,k] - X @ c
    Xv = np.c_[arr(va,2)[:,k], arr(va,5)[:,k], np.ones(len(va))]; res_va[:,k] = arr(va,3)[:,k] - Xv @ c
# lag-1 autocorrelation within episodes
ac = []
for k in range(3):
    a, b = [], []
    for i in range(len(va) - 1):
        if va[i][0] == va[i+1][0] and va[i+1][1] == va[i][1] + 1:
            a.append(res_va[i,k]); b.append(res_va[i+1,k])
    ac.append(np.corrcoef(a, b)[0,1])
print("val residual RMSE (m/s):", np.sqrt((res_va**2).mean(0)), " lag-1 autocorrelation per axis:", np.round(ac,3))
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xtr = np.c_[arr(tr,2), arr(tr,4), arr(tr,5), arr(tr,6)]; Xva = np.c_[arr(va,2), arr(va,4), arr(va,5), arr(va,6)]
    out = []
    for k in range(3):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05).fit(Xtr, res_tr[:,k])
        out.append(1 - ((m.predict(Xva) - res_va[:,k])**2).mean() / (res_va[:,k]**2).mean())
    print("fraction of residual variance a nonlinear regressor on [v,w,F_{t-1},F_t,F_{t+1}] explains on val:", np.round(out,3))
except ImportError:
    print("sklearn not installed; skipped nonlinear check")
