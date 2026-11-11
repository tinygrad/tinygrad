# seq_scores alone on one device: a device's share of a decoding step of G sequences at 262k (R rows each, LP pools each), the old kernel
# (SEQ_SCORE_U=1 SEQ_SCORE_WG=16 SEQ_SCORE_LDSQ=1) vs configs "U,WG,LDSQ ...": us a call and bit equality of the scores
# usage: DEV=AMD G=28 CFGS="4,64,0 2,32,0" python3 extra/qwen_seqscore_bench.py
import os, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.llm.kernels import amd_qwen
G, R, H, D, KP = int(os.environ.get("G", 28)), 3, 4, 128, 4
LP = int(os.environ.get("LP", 10923))  # (a device's pools of a 262k sequence: 65,536/6)
stride, dev = LP + 64, Device.DEFAULT
rng = np.random.default_rng(0)
q = Tensor(rng.standard_normal((G*R, H, D), dtype=np.float32), device=dev).realize()
pk = Tensor((rng.standard_normal((G*stride, D), dtype=np.float32) * 0.5).astype(np.float16), device=dev).realize()
bases = Tensor(np.arange(G, dtype=np.int32) * stride * KP, device=dev).realize()
positions = Tensor(np.full(G, LP*KP - R, dtype=np.int32), device=dev).realize()
def run(u:int, wg:int, ldsq:int) -> tuple[np.ndarray, float]:
  amd_qwen.SEQ_SCORE_U, amd_qwen.SEQ_SCORE_WG, amd_qwen.SEQ_SCORE_LDSQ = u, wg, ldsq
  amd_qwen._seq_scores_kernel.cache_clear()
  out = amd_qwen.seq_scores(q, pk, bases, positions, R, KP, LP).realize().numpy()
  jit = TinyJit(lambda: amd_qwen.seq_scores(q, pk, bases, positions, R, KP, LP).realize())
  for _ in range(3): jit()
  ts = []
  for _ in range(5):
    Device[dev].synchronize()
    st = time.perf_counter()
    for _ in range(20): jit()
    Device[dev].synchronize()
    ts.append((time.perf_counter() - st) / 20)
  return out, sorted(ts)[2] * 1e6
ref, t0 = run(1, 16, 1)
print(f"old U=1 WG=16 LDSQ=1 {t0:8.1f} us ({G*LP*D*2/t0/1e3:.0f} GB/s of pooled keys)", flush=True)
for cfg in os.environ.get("CFGS", "4,64,0").split():
  u, wg, ldsq = (int(x) for x in cfg.split(","))
  o, t = run(u, wg, ldsq)
  same = np.array_equal(o.view(np.uint32), ref.view(np.uint32))
  print(f"U={u} WG={wg} LDSQ={ldsq} {t:8.1f} us ({G*LP*D*2/t/1e3:.0f} GB/s)  bit-identical={same}", flush=True)
