#!/usr/bin/env python3
"""
GLM-5.3 Flash (glm5next) on one card: every quantized linear class of the first NUM_BLOCKS blocks (and the output head) run on the same input
rows in batches of ROWS rows (the kernels and paddings production picks by row count), rows 0..2 compared with a float64 numpy reference (the
GGUF weights dequantized in float32 (exact), the dot in float64): relative L2 error of a row's output, and whether the row's output changes
with the batch (bits). The MoE experts: the decode path (moe_linear) at every row count (the prefill's grouped path is not probed).
Inputs: fixed random rows, heavy tailed (1% of the features x10), unit rms.
env: MODEL, NUM_BLOCKS (6), ROWS (3,6,12,54,192,768), OUTS (256: the output features compared), SEED (0), LAYERS (all: e.g. blk.0.attn_q_a)
"""
import os
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.glm5next import Transformer, ExpertWeights
from tinygrad.llm.kernels.amd_glm import Linear, QUANT_SIZES
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.gguf import _GGML_QUANT  # noqa: F401  (the types' names below)

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 8192), shard=1)
ROWS = [int(r) for r in str(getenv("ROWS", "3,6,12,54,192,768")).split(",")]
OUTS, rng = getenv("OUTS", 256), np.random.default_rng(getenv("SEED", 0))
WANT = [w for w in str(getenv("LAYERS", "")).split(",") if w]
TYPES = {v: k for k, v in __import__("tinygrad.llm.kernels.amd_glm", fromlist=["x"]).__dict__.items()
         if isinstance(v, int) and k.isupper() and k in ("Q8_0", "Q6_K", "Q4_K", "Q5_K", "IQ2_S", "IQ2_XS", "IQ3_S", "IQ3_XXS", "IQ4_XS", "IQ4_NL")}

def inputs(n:int, d:int) -> np.ndarray:
  x = rng.standard_normal((n, d)) * np.where(rng.random((n, d)) < 0.01, getenv("TAIL", 10.0), 1.0)
  return (x / np.sqrt((x**2).mean(-1, keepdims=True))).astype(np.float32)

def packed_rows(w:Tensor, typ:int, inf:int, row0:int, rows:int) -> np.ndarray:
  """rows row0.. of a packed GGUF weight (rows of inf weights, QUANT_SIZES bytes a 256) dequantized (float32 exact), float64"""
  rb = inf * QUANT_SIZES[typ] // 256
  raw = w.flatten().bitcast(dtypes.uint8)[row0*rb:(row0+rows)*rb].contiguous()
  return ggml_data_to_tensor(raw, rows*inf, typ).reshape(rows, inf).float().numpy().astype(np.float64)

def rel(a:np.ndarray, b:np.ndarray) -> float: return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

def mods():
  for li, b in enumerate(model.blk):
    for name, m in sorted(vars(b).items()):
      for sub, mm in ([(name, m)] + [(f"{name}.{n2}", m2) for n2, m2 in sorted(vars(m).items())] if hasattr(m, "__dict__") and
                      not isinstance(m, (Linear, ExpertWeights, Tensor)) else [(name, m)]):
        if isinstance(mm, (Linear, ExpertWeights)): yield f"blk.{li}.{sub}", mm
  yield "output", model.output

dev = model.token_embd.weight.device
for name, m in mods():
  if WANT and name not in WANT: continue
  if isinstance(m, Linear):
    if m.ggml_type is None and m.use_custom_quant: m.set_quantized(m.weight)
    if m.ggml_type not in QUANT_SIZES or m.shard_axis is not None: continue
    W = packed_rows(m.weight, m.ggml_type, m.in_features, 0, min(OUTS, m.out_features))
    if W.ndim != 2: continue
    X = inputs(max(ROWS), m.in_features)
    ref = X[:3].astype(np.float64) @ W.T
    outs, us = {}, {}
    for n in ROWS:
      y = m(Tensor(X[:n], device=dev)).float().numpy()
      outs[n] = y[:3, :W.shape[0]]
      if getenv("TIME", 0):
        # (TIME=1: the GPU time of the linear's kernels on these rows (extra/kprof), the input already on the card, 5 runs, the min)
        import extra.kprof as kprof
        xt, best = Tensor(X[:n], device=dev).realize(), []
        for _ in range(5):
          Device[dev].synchronize()
          kprof.reset(dev)
          m(xt).realize()
          Device[dev].synchronize()
          best.append(sum(v[1] for v in kprof.times(dev).values()))
        us[n] = min(best)
    tname = next((k for k, v in TYPES.items() if v == m.ggml_type), str(m.ggml_type))
    base = outs[ROWS[0]]
    print(f"{name} {tname} {m.out_features}x{m.in_features}: " + " ".join(
      f"n={n}: err {rel(o, ref):.2e}{'' if np.array_equal(o, base) else ' (bits differ from n=' + str(ROWS[0]) + f': {rel(o, base):.1e})'}"
      for n, o in outs.items()) + (" | us " + " ".join(f"n={n}: {u:.1f}" for n, u in us.items()) if us else ""), flush=True)
  elif isinstance(m, ExpertWeights) and getenv("MOE", 1) and not m.slices:
    K, E, inf = getenv("MOE_K", 8), m.num_experts, m.in_features
    X = inputs(max(ROWS), inf)
    sel = np.stack([rng.choice(E, K, replace=False) for _ in range(max(ROWS))]).astype(np.int32)
    m.set_quantized()
    if m.ggml_type not in QUANT_SIZES: continue
    outf = int(m.weight.numel()) * m.weight.dtype.itemsize * 256 // QUANT_SIZES[m.ggml_type] // inf // E
    W = {e: packed_rows(m.weight, m.ggml_type, inf, e*outf, OUTS) for e in set(sel[:3].flatten().tolist())}
    ref = np.stack([np.stack([W[sel[r, j]] @ X[r] for j in range(K)]) for r in range(3)])  # (3, K, OUTS)
    tname = next((k for k, v in TYPES.items() if v == m.ggml_type), str(m.ggml_type))
    res = []
    for n in ROWS:
      st, xt = Tensor(sel[:n].reshape(1, n, K), device=dev), Tensor(X[:n].reshape(1, n, 1, inf), device=dev)
      y = m(st, xt).float().numpy().reshape(n, K, -1)[:3, :, :OUTS]
      res.append(f"n={n} decode: err {rel(y, ref):.2e}")
    print(f"{name} {tname} experts {E} x {m.weight.shape}: " + " ".join(res), flush=True)
