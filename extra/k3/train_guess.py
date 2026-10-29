#!/usr/bin/env python3
# the engine's guess of layer T's experts is a router applied to layer T-1's router input (the only input there at that time). today it is
# layer T's own router: this fits a linear map per layer instead, ridge pulled towards that router, from the teacher traces
# (router_input-L, logits-L of 98k tokens). reports recall of the true top 16 for both on held out tokens and writes the maps.
# python3 train_guess.py teacher_dir trunk.gguf out_dir [gguf_py]
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else "/home/nimlgen/k3q/atomic-llama/gguf-py")
import gguf

D, OUT = Path(sys.argv[1]), Path(sys.argv[3])
OUT.mkdir(exist_ok=True)
r = gguf.GGUFReader(sys.argv[2])
routers = {int(t.name.split(".")[1]): t for t in r.tensors if t.name.endswith("ffn_gate_inp.weight")}
split, routes, bias = np.load(D / "split.npy"), np.load(D / "routes.npy", mmap_mode="r"), np.load(D / "router_bias.npy")
tr, va, te = (np.nonzero(split == s)[0] for s in (0, 1, 2))
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0)

def recall(logits:np.ndarray, T:int, rows:np.ndarray, k:int) -> float:
  s = 1 / (1 + np.exp(-logits)) + bias[T - 1]
  top = np.argpartition(-s, k, axis=1)[:, :k]
  true = np.asarray(routes[rows, T - 1])
  return float(np.mean((top[:, :, None] == true[:, None, :]).any(-1).sum(-1) / 16))

res = []
for T in range(2, 93):
  st = time.perf_counter()
  X = np.load(D / f"router_input-{T - 1}.npy", mmap_mode="r")
  Y = np.load(D / f"logits-{T}.npy", mmap_mode="r")
  R = np.asarray(routers[T].data, np.float32).reshape(896, -1).T  # (7168, 896): logits = x @ R
  Xt, Yt = np.asarray(X[tr], np.float32), np.asarray(Y[tr], np.float32)
  G, B = Xt.T @ Xt, Xt.T @ Yt
  Xv, Xe = np.asarray(X[va], np.float32), np.asarray(X[te], np.float32)
  scale = np.trace(G) / len(G)
  best = None
  for a in ALPHAS:  # ridge towards the router: min |XW - Y|^2 + lam |W - R|^2
    lam = a * scale
    W = np.linalg.solve(G + lam * np.eye(len(G), dtype=np.float32), B + lam * R)
    rv = recall(Xv @ W, T, va, 16)
    if best is None or rv > best[0]: best = (rv, a, W)
  rv, a, W = best
  row = {"T": T, "alpha": a, "base16": recall(Xe @ R, T, te, 16), "fit16": recall(Xe @ W, T, te, 16),
         "base10": recall(Xe @ R, T, te, 10), "fit10": recall(Xe @ W, T, te, 10)}
  res.append(row)
  np.save(OUT / f"guess-{T}.npy", W.T.astype(np.float16))  # (896, 7168) like the router
  print(f"T {T:2d} alpha {a:g}: recall@16 router {row['base16']:.3f} -> fit {row['fit16']:.3f}, @10 {row['base10']:.3f} -> {row['fit10']:.3f} "
        f"({time.perf_counter() - st:.0f} s)", flush=True)
m = {k: float(np.mean([x[k] for x in res])) for k in ("base16", "fit16", "base10", "fit10")}
print(f"mean over layers 2..92 (test): recall@16 router {m['base16']:.3f} -> fit {m['fit16']:.3f}, "
      f"@10 {m['base10']:.3f} -> {m['fit10']:.3f}")
