"""Six-GPU LLM collective microbenchmark; no model load.

python test/external/external_benchmark_llm_allreduce.py
OP=rs selects sequence-parallel reduce-scatter (Q8 payloads for aligned chunks).
OP=agq8 selects packed-Q8 allgather; OP=ag selects fp16 allgather.
OP=tiny selects the small float32 allgather used by sampling. ROWS and K override sizes/repetitions.
Uses GPU steady clocks, not host submission time; validates every output on every rank.
Allreduce includes a dependent add (~1.3us); other operations use explicit ordering dependencies.
"""
import os, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device, UOp, dtypes
from tinygrad.uop.ops import KernelInfo
from tinygrad.llm.kernels.amd import partial_sum, reduce_scatter, all_gather, all_gather_q8, host_allgather, _sys, FLAG_AR_MAXN


def stamp(out, *deps):
  clock = _sys("__builtin_readsteadycounter()", dtypes.uint64, UOp.const(0, dtypes.uint32))
  return out[0].store(clock).sink(arg=KernelInfo(name="collective_clock", opts_to_apply=()))


def shards(t):
  dtype = np.uint64 if t.dtype == dtypes.uint64 else np.float16 if t.dtype == dtypes.half else np.float32
  return [np.frombuffer(b.as_memoryview(), dtype=dtype).copy() for b in t.uop.base.buffer.bufs]


def operation(x, op):
  if op == "ar": return partial_sum(x)
  if op == "rs": return reduce_scatter(x, 1)
  if op == "ag": return all_gather(x, 1, dtypes.half)
  if op == "agq8": return all_gather_q8(x, 1)
  if op == "tiny":
    result = host_allgather(x)
    assert result is not None
    return result
  raise ValueError(op)


if __name__ == "__main__":
  N, D, K = int(os.environ.get("TEST_GPUS", "6")), int(os.environ.get("D", "2560")), int(os.environ.get("K", "16"))
  op = os.environ.get("OP", "ar")
  devs = tuple(Device.canonicalize(f"AMD:{i}") for i in range(N))
  assert all(Device[d].target[0] == 11 for d in devs), "clock conversion is for gfx11 (100MHz steady clock)"
  rng = np.random.default_rng(123)
  rows = "1,5,32,256" if op == "ar" else "1" if op == "tiny" else "192,768"
  for R in map(int, os.environ.get("ROWS", rows).split(",")):
    started = time.perf_counter()
    q8_rs = op == "rs" and R*D % (N*256) == 0 and R*D//(N*32)*9 <= FLAG_AR_MAXN
    shape = (N, 1, R, D) if op in ("ar", "rs") else (N, 2) if op == "tiny" else (1, R, D)
    inputs = [rng.standard_normal(shape).astype(np.float32) for _ in range(K+1)]
    xs = [Tensor(a).shard(devs, axis=0 if op in ("ar", "rs", "tiny") else 1).contiguous().realize() for a in inputs]
    @TinyJit
    def run(*xs):
      warm = operation(xs[0], op).contiguous()  # synchronize ranks before measuring
      start = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.uint64, device=devs), warm, fxn=stamp)
      y = start[1] if op == "ar" else start[0]
      ys = []
      for x in xs[1:]:
        ordered = (x + y.unsqueeze(0)*1e-30).contiguous() if op == "ar" else Tensor(x.uop.after(y.uop))
        y = operation(ordered, op).contiguous()
        ys.append(y)
      end = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.uint64, device=devs), *ys, fxn=stamp)[0]
      Tensor.realize(start[0], end, *ys)
      return start[0], end, *ys
    for _ in range(2): result = run(*xs)
    times = []
    for _ in range(7):
      result = run(*xs)
      for d in devs: Device[d].synchronize()
      times.append(max(float(b[0]-a[0])/100/K for a, b in zip(shards(result[0]), shards(result[1]))))
    error = 0.0
    for data, y in zip(inputs[1:], result[2:]):
      if q8_rs:
        z = data.reshape(N, -1, 32)
        scale = np.maximum(np.abs(z).max(-1, keepdims=True)/127, 1e-8)
        quant = (np.rint(z/scale)*scale).reshape(N, N, -1)
        refs = []
        for rank in range(N):
          ref = np.zeros(R*D//N, dtype=np.float32)
          for s in range(N):
            if s != rank: ref += quant[s, rank]
          refs.append(ref + data.reshape(N, N, -1)[rank, rank])
      elif op in ("ar", "rs"):
        ref = data.astype(np.float16).astype(np.float32).sum(0)
        refs = np.split(ref, N, axis=1) if op == "rs" else [ref]*N
      elif op == "agq8":
        z = data.reshape(-1, 32)
        scale = np.maximum(np.abs(z).max(-1, keepdims=True)/127, 1e-8)
        refs = [(np.rint(z/scale)*scale).astype(np.float16).reshape(data.shape)]*N
      else: refs = [data if op == "tiny" else data.astype(np.float16)]*N
      for actual, ref in zip(shards(y), refs):
        assert np.isfinite(actual).all(), (op, R, "nonfinite output")
        error = max(error, float(np.abs(actual-ref.ravel()).max()/max(np.abs(ref).max(), 1e-8)))
    # The old ring requantizes running sums; the new transport uses fp16 or independently quantized Q8 partials.
    tolerance = (1e-6 if R*D//(2*N) <= FLAG_AR_MAXN else 0.03) if op == "rs" else 0.003
    if q8_rs: tolerance = 0.003  # NumPy/GPU rounding can differ at quantization boundaries.
    assert error < tolerance, (op, R, error)
    print(f"{op:4s} {R:5d} rows: {min(times):8.2f}us  all-rank error {error:.2e}  elapsed {time.perf_counter()-started:.1f}s", flush=True)
