"""Stage 2: train LCR and the learned baselines on precomputed samples (no simulator in the loop).

Residual models (lcr*, global) are supervised through the linearised simulator:
    pose error after the step : e0 + J  dW      -> should be 0
    predicted F/T reading     : W0 + JW dW      -> should equal the measured o_t
The same dW enters both terms (coupling).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr.models import LCR, GlobalResidual, BlackBox, WRENCH_SCALE  # noqa: E402

KEYS = ["feats", "R", "r", "lam", "mask", "glob", "pose", "W0", "JW", "e0", "J", "obs", "delta"]


def load(name, frac=1.0, seed=0):
    z = np.load(f"cache/{name}.npz")
    n = len(z["obs"])
    idx = np.arange(n)
    if frac < 1.0:   # subsample whole episodes-worth of steps uniformly
        idx = np.sort(np.random.default_rng(seed).choice(n, int(n * frac), replace=False))
    return {k: torch.from_numpy(z[k][idx]) for k in KEYS}


def build(kind):
    return {"lcr": lambda: LCR(), "lcr_nogeom": lambda: LCR(use_geom=False),
            "lcr_noattn": lambda: LCR(attention=False), "global": lambda: GlobalResidual(),
            "blackbox": lambda: BlackBox()}[kind]()


def losses(model, kind, b, st, beta=1.0, gamma=1e-3):
    if kind == "blackbox":
        out = model(b)
        ld = (((out[:, :6] - b["delta"] / st["sd"]) ** 2)).mean()
        lo = (((out[:, 6:] - b["obs"] / st["so"]) ** 2)).mean()
        return lo + beta * ld, lo, ld
    dW = model(b)
    err = b["e0"] + torch.einsum("bij,bj->bi", b["J"], dW)
    W = b["W0"] + torch.einsum("bij,bj->bi", b["JW"], dW)
    ld = ((err / st["sd"]) ** 2).mean()
    lo = (((W - b["obs"]) / st["so"]) ** 2).mean()
    lr = ((dW / WRENCH_SCALE) ** 2).mean()
    return lo + beta * ld + gamma * lr, lo, ld


def batches(data, bs, shuffle, rng):
    n = len(data["obs"])
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        j = torch.from_numpy(idx[i:i + bs])
        yield {k: v[j] for k, v in data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train", default="train")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tr, va = load(a.train, a.frac), load("val")
    st = {"so": tr["obs"].std(0), "sd": tr["delta"].std(0)}
    model = build(a.model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    name = a.model + (f"_{a.tag}" if a.tag else "")
    best, hist, t0 = 1e9, [], time.time()
    for ep in range(a.epochs):
        model.train()
        tl = []
        for b in batches(tr, a.bs, True, rng):
            loss, _, _ = losses(model, a.model, b, st, beta=a.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            v = [losses(model, a.model, b, st, beta=a.beta) for b in batches(va, 8192, False, rng)]
        vl = float(np.mean([x[0].item() for x in v])); vo = float(np.mean([x[1].item() for x in v]))
        vd = float(np.mean([x[2].item() for x in v]))
        hist.append({"epoch": ep, "train": float(np.mean(tl)), "val": vl, "val_obs": vo, "val_dyn": vd})
        print(f"[{name}] epoch {ep:2d} train {np.mean(tl):.4f}  val {vl:.4f} (obs {vo:.4f}, dyn {vd:.4f})  {time.time()-t0:.0f}s", flush=True)
        if vl < best:
            best = vl
            os.makedirs("results/models", exist_ok=True)
            torch.save({"kind": a.model, "state": model.state_dict(), "stats": st}, f"results/models/{name}.pt")
    json.dump({"name": name, "kind": a.model, "frac": a.frac, "train_seconds": time.time() - t0,
               "n_params": sum(p.numel() for p in model.parameters()), "history": hist},
              open(f"results/models/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
