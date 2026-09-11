import argparse, time
from contextlib import nullcontext
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.compile import load_pickle
from tinygrad.nn.state import get_parameters


def make_inputs(jit, seed=100):
  rng = np.random.default_rng(seed)
  inputs = {}
  for name, (view, _, dtype, device) in zip(jit.captured.expected_names, jit.captured.expected_input_info, strict=True):
    data = rng.standard_normal(view.shape) * 8 if dtypes.is_float(dtype) else rng.integers(0, 2 if dtype == dtypes.bool else 16, view.shape)
    inputs[name] = Tensor(data, dtype=dtype, device=device).realize()
  return inputs


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Benchmark inference from a compiled TinyJit pickle.")
  parser.add_argument('pickle')
  parser.add_argument('--out-of-band', action='store_true')
  parser.add_argument('--runs', type=int, default=20)
  args = parser.parse_args()
  with open(args.pickle, 'rb') as f: run = load_pickle(f, out_of_band=args.out_of_band)
  inputs = make_inputs(run)
  if (log := bool(getenv("BENCHMARK_LOG", ""))): from extra.bench_log import WallTimeEvent, BenchEvent
  times = []
  for _ in range(args.runs):
    start = time.perf_counter()
    with WallTimeEvent(BenchEvent.STEP) if log else nullcontext():
      output = run(**inputs)
      enqueued = time.perf_counter()
      for tensor in get_parameters(output): tensor.numpy()
    times.append((time.perf_counter() - start) * 1e3)
    print(f"enqueue {(enqueued-start)*1e3:6.2f} ms -- total run {times[-1]:6.2f} ms")
  if (limit := getenv("ASSERT_MIN_STEP_TIME", 0.0)):
    assert min(times) < limit, f"Speed regression, expected < {limit} ms but took {min(times)} ms"
