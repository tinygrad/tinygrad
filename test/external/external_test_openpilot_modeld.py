#!/usr/bin/env python3
import argparse, re, tempfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tinygrad import Device, Tensor
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops
from examples.openpilot.compile_onnx import compile_onnx, make_input_queues, read_file_chunked_to_disk
from examples.openpilot.helpers import MODELS, fetch_model, load_oob


def check_kernels(jit, expected):
  calls = [u for u in jit.captured.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM)
           if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
  print(f'{Device.DEFAULT}: {len(calls)} kernels')
  if expected is not None: assert len(calls) == expected, f'{len(calls)} kernels != {expected}'
  sources = [call.src[0].src[2].arg for call in calls]
  reads = sum(src.count('read_image') for src in sources)
  gated = sum(src.count('?read_image') + sum(bool(re.search(fr'[\?:]{v}\.[xyzw]', src))
              for v in re.findall(r'(val\d+)\s*=\s*read_imagef\(', src)) for src in sources)
  for variable, count in (('ALLOWED_READ_IMAGE', reads), ('ALLOWED_GATED_READ_IMAGE', gated)):
    if (expected_images := getenv(variable, -1)) != -1:
      assert count == expected_images, f'{variable}: {count} != {expected_images}'


def check_onnx(model, path, atol, rtol):
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
  inputs = make_input_queues(model['input_shapes'], Device.DEFAULT)
  reference = {name: np.zeros(shape, dtype=dtype.fmt) for name, (shape, dtype) in model['input_shapes'].items()}
  rng = np.random.default_rng(42)
  output_names = [o.name for o in session.get_outputs() if o.name not in model['state_pairs'].values()]
  for step in range(3):
    for name, (shape, dtype) in model['input_shapes'].items():
      if name in model['state_pairs']: continue
      data = rng.integers(0, 256, shape) if 'img' in name or not np.issubdtype(np.dtype(dtype.fmt), np.floating) else rng.normal(0, .1, shape)
      reference[name] = data.astype(dtype.fmt)
      inputs[name].assign(Tensor(reference[name], device=Device.DEFAULT)).realize()
    expected = dict(zip((o.name for o in session.get_outputs()), session.run(None, reference), strict=True))
    actual = model['run_model'](**inputs)
    for name, value in zip(output_names, actual, strict=True):
      result = value.numpy()
      delta = np.abs(result.astype(np.float32) - expected[name].astype(np.float32))
      print(f'frame {step}, {name}: max error {delta.max():.6f}, mean error {delta.mean():.6f}')
      np.testing.assert_allclose(result, expected[name], atol=atol, rtol=rtol)
    for name, next_name in model['state_pairs'].items():
      np.testing.assert_allclose(inputs[name].numpy(), expected[next_name], atol=atol, rtol=rtol)
      reference[name] = expected[next_name]


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--model', choices=MODELS, default='driving')
  p.add_argument('--onnx', help='override the pinned model with a path or URL')
  p.add_argument('--kernel-count', type=int)
  # Full fp16 networks differ across ONNX Runtime versions; the fp32 CI case uses 1e-4.
  p.add_argument('--atol', type=float, default=2.5)
  p.add_argument('--rtol', type=float, default=.2)
  args = p.parse_args()
  path = read_file_chunked_to_disk(args.onnx) if args.onnx else fetch_model(args.model)
  with tempfile.TemporaryDirectory() as directory:
    output = Path(directory)/'model.pkl'
    compile_onnx(path, output)
    with output.open('rb') as f: model = load_oob(f)
    check_kernels(model['run_model'], args.kernel_count)
    check_onnx(model, path, args.atol, args.rtol)
  print('PASS: compile, pickle replay, state feedback, and ONNX Runtime comparison')
