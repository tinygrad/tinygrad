#!/usr/bin/env python3
"""
Reduced-bit latent cache formats on REAL C9 rows (glm53_pfchunk.py QDUMP=dir: rows_b<block>_p<pos>.npy, (T, 576) half), numpy only (teeny):
per format the latent rows' error (RMSE / RMS) and a score proxy (q.k for queries = other rows of the block, the rope dims exact).
  python3 extra/glm_kvq_formats.py <dir>
formats: fp8 (fp8_rows), q4 (q4_rows: 4-bit, a 6-bit multiplier a group of 32, a float scale a 128), and the same after a block Hadamard
rotation of the 512 latent dims (blocks of 128: H c is stored, H^T folds into k_b / v_b at load)
"""
import sys, glob, re
import numpy as np

def e4m3(x:np.ndarray) -> np.ndarray:
  # round to nearest e4m3 (3 mantissa bits, max 448, subnormals from 2**-6)
  a = np.abs(x)
  e = np.floor(np.log2(np.maximum(a, 2.0**-9)))
  e = np.maximum(e, -6)
  step = 2.0 ** (e - 3)
  return np.sign(x) * np.minimum(np.round(a / step) * step, 448)

def fp8(c:np.ndarray) -> np.ndarray:
  g = c.reshape(-1, 4, 128)
  s = np.maximum(np.abs(g).max(-1, keepdims=True), 1e-20) / 448
  return (e4m3(g / s) * s).reshape(c.shape)

def q4(c:np.ndarray, gs:int=32) -> np.ndarray:
  g = c.reshape(-1, 4, 128 // gs, gs)
  a = np.abs(g).max(-1)
  sc = np.maximum(a.max(-1, keepdims=True), 1e-20) / (7 * 63)
  m = np.clip(np.ceil(a / (7 * sc)), 1, 63)
  q = np.clip(np.round(g / (m * sc)[..., None]), -8, 7)
  return (q * (m * sc)[..., None]).reshape(c.shape)

def hadamard(n:int) -> np.ndarray:
  h = np.ones((1, 1))
  while h.shape[0] < n: h = np.block([[h, h], [h, -h]])
  return h / np.sqrt(n)

H = hadamard(128)
rng = np.random.default_rng(0)
SIGNS = np.where(rng.random(512) < 0.5, -1.0, 1.0)  # (a random sign a dim before the rotation: the outlier dims spread, not aligned)
def rot(c:np.ndarray) -> np.ndarray: return ((c * SIGNS).reshape(-1, 4, 128) @ H.T).reshape(c.shape)
def unrot(c:np.ndarray) -> np.ndarray: return (c.reshape(-1, 4, 128) @ H).reshape(c.shape) * SIGNS

FORMATS = {"fp8": fp8, "q4": q4, "q4g16": lambda c: q4(c, 16), "fp8+H": lambda c: unrot(fp8(rot(c))), "q4+H": lambda c: unrot(q4(rot(c))),
           "q4g16+H": lambda c: unrot(q4(rot(c), 16))}

def main(d:str) -> None:
  files = sorted(glob.glob(f"{d}/rows_b*_p*.npy"), key=lambda f: int(re.search(r"_b(\d+)_", f).group(1)))
  tot: dict[str, list[float]] = {k: [] for k in FORMATS}
  print(f"{'block':>5} {'rms':>7} {'kurt':>7} " + " ".join(f"{k+' row/score':>18}" for k in FORMATS))
  for f in files:
    x = np.load(f).astype(np.float64)
    c, r = x[:, :512], x[:, 512:]
    kurt = float(np.mean(((c - c.mean()) / c.std())**4))
    qi = rng.choice(len(x), min(256, len(x)), replace=False)
    qs = x[qi]
    ref = qs[:, :512] @ c.T + qs[:, 512:] @ r.T
    cells = []
    for k, fn in FORMATS.items():
      ch = fn(c)
      row = float(np.sqrt(np.mean((ch - c)**2) / np.mean(c**2)))
      sc = qs[:, :512] @ ch.T + qs[:, 512:] @ r.T
      se = float(np.sqrt(np.mean((sc - ref)**2) / np.mean(ref**2)))
      tot[k].append(row)
      cells.append(f"{row:8.4f}/{se:8.4f}")
    blk = re.search(r"_b(\d+)_", f).group(1)
    print(f"{blk:>5} {np.sqrt(np.mean(c**2)):7.3f} {kurt:7.1f} " + " ".join(f"{s:>18}" for s in cells))
  print("mean row error: " + " ".join(f"{k} {np.mean(v):.4f}" for k, v in tot.items()))

if __name__ == "__main__": main(sys.argv[1])
