import argparse, time
from contextlib import nullcontext
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn.compile import load_pickle
from tinygrad.nn.state import get_parameters


def make_inputs(variant, seed=100):
  rng = np.random.default_rng(seed)
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in variant['input_specs'].items()}
  views = arrays.copy()
  if variant['packed_specs']:
    packed = views.pop('packed_inputs')
    views.update({name: packed[start:start+int(np.prod(shape))*np.dtype(dtype).itemsize].view(dtype).reshape(shape)
                  for name, (start, shape, dtype) in variant['packed_specs'].items()})
  for value in views.values():
    value[...] = (rng.standard_normal(value.shape) * 8 if np.issubdtype(value.dtype, np.floating) else
                  rng.integers(0, 2 if value.dtype == np.bool_ else 16, value.shape))
  return {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in variant['input_specs'].items()}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Benchmark variants from a compiled model or warp pickle.")
  parser.add_argument('pickle')
  parser.add_argument('--out-of-band', action='store_true')
  parser.add_argument('--runs', type=int, default=20)
  parser.add_argument('--variant', help='benchmark one named variant; defaults to all variants')
  args = parser.parse_args()
  with open(args.pickle, 'rb') as f: variants = load_pickle(f, out_of_band=args.out_of_band)['variants']
  if args.variant is not None: variants = {args.variant: variants[args.variant]}
  if (log := bool(getenv("BENCHMARK_LOG", ""))): from extra.bench_log import WallTimeEvent, BenchEvent
  for name, variant in variants.items():
    print(f"variant {name}")
    inputs = make_inputs(variant)
    times = []
    for _ in range(args.runs):
      start = time.perf_counter()
      with WallTimeEvent(BenchEvent.STEP) if log else nullcontext():
        output = variant['run'](**inputs)
        enqueued = time.perf_counter()
        for tensor in get_parameters(output): tensor.numpy()
      times.append((time.perf_counter() - start) * 1e3)
      print(f"enqueue {(enqueued-start)*1e3:6.2f} ms -- total run {times[-1]:6.2f} ms")
    if (limit := getenv("ASSERT_MIN_STEP_TIME", 0.0)):
      assert min(times) < limit, f"Speed regression, expected < {limit} ms but took {min(times)} ms"
