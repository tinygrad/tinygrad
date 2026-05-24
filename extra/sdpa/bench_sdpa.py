"""Benchmark + reference comparison for the auto-discovered FA SDPA kernel.

Runs the same (Q, K, V) through:
  1. tinygrad's Tensor.scaled_dot_product_attention. With no env: default
     codegen (4 kernels: Q@K, max, sum-exp, P@V with matmul-opt). With
     PCONTIG=3 + CR_* knobs: forces fusion into one kernel via the
     coupled-reduce descriptor in tinygrad/uop/coupled_reduce.py.
  2. PyTorch's F.scaled_dot_product_attention (dispatches to MPS
     sdpa_vector_fast_mps for short N, sdpa_vector_2pass_mps for longer —
     the actual implementations in aten/src/ATen/native/mps/kernels/
     DecodeAttention.h).

Recommended configs:
  # Default + BEAM (best at long N, near-PyTorch at short N):
  BEAM=2 python3 extra/sdpa/bench_sdpa.py

  # Long-N fused single-kernel (algorithmic FA structure):
  PCONTIG=3 CR_LOCAL=32 CR_GROUP=4 CR_TILE_D=32 CR_UNROLL_QK=4 CR_J_UPCAST=8 \\
    python3 extra/sdpa/bench_sdpa.py --shapes 2048,4096
"""
import argparse, io, os, statistics, sys, time
from contextlib import redirect_stdout
from tinygrad import Tensor, TinyJit, dtypes, Device
from tinygrad.helpers import Context, GlobalCounters

SHAPES_DEFAULT = (128, 256, 512, 1024, 2048)
B, H, D = 4, 32, 128   # Llama-class: matches real LLM workload (batch*head*head_dim).
                       # The earlier toy B=1 H=4 D=64 measured noise/overhead, not kernel quality.
RUNS = 200
WARMUP = 20

def _build_jit():
  @TinyJit
  def f(q, k, v): return q.scaled_dot_product_attention(k, v).realize()
  return f

def _time_tinygrad(N:int):
  # JIT path: the discovered FA kernel is single-kernel so JIT timing is fair.
  # Stream RUNS calls back-to-back inside one sync window — measures pure
  # GPU throughput, amortizes Python overhead just like torch.mps.synchronize sweep.
  f = _build_jit()
  q = Tensor.randn(B, H, N, D).realize()
  k = Tensor.randn(B, H, N, D).realize()
  v = Tensor.randn(B, H, N, D).realize()
  for _ in range(WARMUP): f(q, k, v)
  Device[Device.DEFAULT].synchronize()
  ts = []
  for _ in range(RUNS):
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    out = f(q, k, v)
    Device[Device.DEFAULT].synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
  return ts, out

def _time_torch(N:int):
  import torch
  dev = "mps" if torch.backends.mps.is_available() else "cpu"
  q = torch.randn(B, H, N, D, device=dev)
  k = torch.randn(B, H, N, D, device=dev)
  v = torch.randn(B, H, N, D, device=dev)
  import torch.nn.functional as F
  for _ in range(WARMUP): F.scaled_dot_product_attention(q, k, v)
  if dev == "mps": torch.mps.synchronize()
  ts = []
  for _ in range(RUNS):
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    out = F.scaled_dot_product_attention(q, k, v)
    if dev == "mps": torch.mps.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
  return ts, out

def _dump_kernel(N:int) -> str:
  buf = io.StringIO()
  with Context(DEBUG=4), redirect_stdout(buf):
    q = Tensor.empty(B, H, N, D)
    k = Tensor.empty(B, H, N, D)
    v = Tensor.empty(B, H, N, D)
    q.scaled_dot_product_attention(k, v).realize()
  return buf.getvalue()

def _parity(N:int) -> float:
  import torch, numpy as np
  rng = np.random.default_rng(7)
  inputs = [rng.standard_normal((B, H, N, D)).astype(np.float32) * 0.125 for _ in range(3)]
  tiny = Tensor(inputs[0]).scaled_dot_product_attention(Tensor(inputs[1]), Tensor(inputs[2])).numpy()
  ref = torch.nn.functional.scaled_dot_product_attention(
    *(torch.from_numpy(x) for x in inputs)).numpy()
  return float(np.abs(tiny - ref).max())

def _stats(ts): return min(ts), statistics.median(ts), max(ts)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--shapes", default=",".join(str(s) for s in SHAPES_DEFAULT))
  ap.add_argument("--dump-kernel", action="store_true")
  ap.add_argument("--skip-torch", action="store_true")
  args = ap.parse_args()
  shapes = [int(s) for s in args.shapes.split(",")]
  print(f"# tinygrad SDPA bench - device {Device.DEFAULT} - B={B} H={H} D={D}")
  print(f"# {RUNS} runs per shape, wall-clock around realize() with sync")
  print(f"# env: PCONTIG={os.getenv('PCONTIG','')} CR_LOCAL={os.getenv('CR_LOCAL','')} "
        f"CR_GROUP={os.getenv('CR_GROUP','')} CR_TILE_D={os.getenv('CR_TILE_D','')} "
        f"CR_UNROLL_QK={os.getenv('CR_UNROLL_QK','')} CR_J_UPCAST={os.getenv('CR_J_UPCAST','')}")
  print(f"# {'N':>6} | {'tiny min':>10} {'med':>10} | {'torch min':>10} {'med':>10} | {'min ratio':>9} {'med ratio':>9} | parity")
  for N in shapes:
    ts_t, _ = _time_tinygrad(N)
    if args.skip_torch:
      tmin, tmed, _ = _stats(ts_t)
      print(f"  {N:>6} | {tmin:>10.4f} {tmed:>10.4f} |       -          - |        -         - |    -")
      continue
    ts_p, _ = _time_torch(N)
    tmin, tmed, _ = _stats(ts_t)
    pmin, pmed, _ = _stats(ts_p)
    par = _parity(N)
    print(f"  {N:>6} | {tmin:>10.4f} {tmed:>10.4f} | {pmin:>10.4f} {pmed:>10.4f} | "
          f"{pmin/tmin:>8.2f}x {pmed/tmed:>8.2f}x | {par:.2e}")
  if args.dump_kernel:
    for N in shapes:
      path = f"/tmp/tinygrad_sdpa_N{N}.metal"
      with open(path, "w") as fh: fh.write(_dump_kernel(N))
      print(f"# wrote {path}", file=sys.stderr)

if __name__ == "__main__": main()
