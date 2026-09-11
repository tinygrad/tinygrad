import argparse, time
from contextlib import nullcontext
import numpy as np
from extra.bench_log import WallTimeEvent, BenchEvent
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_parameters
from .helpers import allocate_inputs, load_pickle


def make_inputs(variant, seed=100):
  rng = np.random.default_rng(seed)
  def initialize(views):
    for value in views.values():
      value[...] = (rng.standard_normal(value.shape) * 8 if np.issubdtype(value.dtype, np.floating) else
                    rng.integers(0, 2 if value.dtype == np.bool_ else 16, value.shape))
  return allocate_inputs(variant['input_specs'], variant['packed_specs'], initialize)[0]


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Benchmark loading or running compiled model and warp pickles.")
  parser.add_argument('pickle', nargs='?', default='/tmp/openpilot.pkl')
  parser.add_argument('--run', action='store_true', help='benchmark inference instead of loading')
  parser.add_argument('--out-of-band', action='store_true', default=bool(getenv('PICKLE_OOB')))
  parser.add_argument('--runs', type=int, help='defaults to 10 loads or 20 inference runs')
  parser.add_argument('--variant', help='run one named variant; defaults to all variants')
  args = parser.parse_args()
  if not args.run:
    load_times = []
    for _ in range(args.runs or 10):
      with WallTimeEvent(BenchEvent.STEP) as wte, open(args.pickle, 'rb') as f: load_pickle(f, out_of_band=args.out_of_band)
      load_times.append(wte.time)
      print(f"pickle load: {wte.time:6.2f} s")
    if (limit := getenv("ASSERT_MIN_LOAD_TIME", 0.0)):
      assert min(load_times) < limit, f"Speed regression, expected < {limit} s but took {min(load_times)} s"
  else:
    with open(args.pickle, 'rb') as f: variants = load_pickle(f, out_of_band=args.out_of_band)['variants']
    if args.variant is not None: variants = {args.variant: variants[args.variant]}
    for name, variant in variants.items():
      print(f"variant {name}")
      inputs = make_inputs(variant)
      times = []
      for _ in range(args.runs or 20):
        start = time.perf_counter()
        with WallTimeEvent(BenchEvent.STEP) if getenv('BENCHMARK_LOG', '') else nullcontext():
          output = variant['run'](**inputs)
          enqueued = time.perf_counter()
          for tensor in get_parameters(output): tensor.numpy()
        times.append((time.perf_counter() - start) * 1e3)
        print(f"enqueue {(enqueued-start)*1e3:6.2f} ms -- total run {times[-1]:6.2f} ms")
      if (limit := getenv("ASSERT_MIN_STEP_TIME", 0.0)):
        assert min(times) < limit, f"Speed regression, expected < {limit} ms but took {min(times)} ms"
