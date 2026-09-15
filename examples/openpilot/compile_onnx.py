#!/usr/bin/env python3
import argparse
import atexit
import os
import tempfile
import time
import shutil

import numpy as np

from examples.openpilot.helpers import dump_oob, load_oob, make_metadata_dict
from tinygrad.helpers import fetch, getenv
from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit


def make_input_queues(input_shapes, device):
  return {name: Tensor(np.zeros(shape, dtype=dtype.fmt), device=device).realize() for name, (shape, dtype) in input_shapes.items()}


def make_run_model(model_runner, state_pairs):
  def run_model(**inputs):
    outputs = {name: value.contiguous() for name, value in model_runner(inputs).items()}
    Tensor.realize(*outputs.values())
    if state_pairs:
      Tensor.realize(*(inputs[name].assign(outputs[next_name]) for name, next_name in state_pairs.items()))
    return tuple(value for name, value in outputs.items() if name not in state_pairs.values())
  return run_model


def run_jit(jit, input_shapes, seed, n_runs, benchmark=False):
  from contextlib import nullcontext
  if benchmark and getenv("BENCHMARK_LOG"): from extra.bench_log import WallTimeEvent, BenchEvent
  input_queues = make_input_queues(input_shapes, Device.DEFAULT)
  rng = np.random.default_rng(seed)
  times = []
  for i in range(n_runs):
    for value in input_queues.values():
      values = rng.standard_normal(value.shape) if np.issubdtype(np.dtype(value.dtype.fmt), np.floating) else rng.integers(0, 256, value.shape)
      value.assign(Tensor(values.astype(value.dtype.fmt), device=Device.DEFAULT)).realize()
    Device.default.synchronize()
    with WallTimeEvent(BenchEvent.STEP) if benchmark and getenv("BENCHMARK_LOG") else nullcontext():
      st = time.perf_counter()
      outs = jit(**input_queues)
      mt = time.perf_counter()
      Device.default.synchronize()
      et = time.perf_counter()
    times.append((et-st)*1e3)
    print(f"  [{i+1}/{n_runs}] enqueue {(mt-st)*1e3:6.2f} ms -- total {times[-1]:6.2f} ms")
    if i == 0:
      val = [v.numpy() for v in outs]
      buffers = [v.numpy() for v in input_queues.values()]
  if benchmark and (limit := getenv("ASSERT_MIN_STEP_TIME", 0.0)):
    assert min(times) < limit, f"Speed regression: {min(times):.2f} ms >= {limit} ms"
  return val, buffers


def compile_jit(jit, input_shapes, benchmark_runs):
  if benchmark_runs < 1: raise ValueError("benchmark_runs must be at least 1")
  print('capture + replay')
  baseline = run_jit(jit, input_shapes, 42, 3)
  print(f'pickle round trip ({benchmark_runs} runs per seed)')
  with tempfile.TemporaryFile(dir=".") as f:
    dump_oob(jit, f)
    f.seek(0)
    loaded_jit = load_oob(f)
  for seed, expect_match in ((42, True), (43, False)):
    result = run_jit(loaded_jit, input_shapes, seed, benchmark_runs, benchmark=True)
    for name, actual, expected in zip(('outputs', 'buffers'), result, baseline, strict=True):
      match = all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
      assert match == expect_match, f"{name} {'differ from' if expect_match else 'match'} baseline (seed={seed})"
  return jit


def read_file_chunked_to_disk(path):
  tmp_path = f'{path}.unchunked'
  manifest = Path(f'{path}.chunkmanifest')
  if not manifest.is_file(): return Path(path) if Path(path).is_file() else fetch(path)
  count = int(manifest.read_text())
  with open(tmp_path, 'wb') as f:
    for i in range(count):
      with open(f'{path}.chunk{i+1:02d}of{count:02d}', 'rb') as src: shutil.copyfileobj(src, f)
  atexit.register(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
  return tmp_path


def compile_onnx(model_path, output, benchmark_runs=1):
  from tinygrad.nn.onnx import OnnxRunner
  model_runner = OnnxRunner(model_path)
  input_shapes = {name: (tuple(s if isinstance(s, int) else 1 for s in spec.shape), spec.dtype)
                  for name, spec in model_runner.graph_inputs.items()}
  state_pairs = {name: f'next_{name}' for name in input_shapes if f'next_{name}' in model_runner.graph_outputs}
  out = {
    'metadata': make_metadata_dict(model_path),
    'input_shapes': input_shapes,
    'state_pairs': state_pairs,
    'input_devices': {'model': Device.DEFAULT},
  }
  out['run_model'] = compile_jit(TinyJit(make_run_model(model_runner, state_pairs), prune=True), input_shapes, benchmark_runs)
  with open(output, "wb") as f: dump_oob(out, f)
  with open(output, "rb") as f:
    load_oob(f)
    assert not f.read(1), "unexpected model buffer data"
  print(f"Saved JIT to {output} ({os.path.getsize(output) / 1e6:.2f} MB)")
  return out


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  source = p.add_mutually_exclusive_group(required=True)
  source.add_argument('--onnx')
  source.add_argument('--run-pickle', help='benchmark an already compiled artifact')
  p.add_argument('--output', default='/tmp/openpilot.pkl')
  p.add_argument('--benchmark-runs', type=int, default=1,
                 help='timed loaded-JIT runs for each correctness seed')
  args = p.parse_args()

  if args.run_pickle:
    with open(args.run_pickle, 'rb') as f: out = load_oob(f)
    run_jit(out['run_model'], out['input_shapes'], 42, args.benchmark_runs, benchmark=True)
  else:
    compile_onnx(read_file_chunked_to_disk(args.onnx), args.output, args.benchmark_runs)
