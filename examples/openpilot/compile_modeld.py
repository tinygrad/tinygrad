#!/usr/bin/env python3
"""Compile fused camera preparation and the openpilot recurrent driving model."""
import argparse, codecs, math, pickle
from functools import partial
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.nn.compile import compile_jit, dump_pickle, onnx_metadata
from tinygrad.nn.warp_compile import make_frame_prepare, parse_frame, parse_size


MODELD_INPUTS = ['img_q', 'big_img_q', 'feat_q', 'desire_q', 'packed_npy_inputs']


def get_policy_npy_shapes(input_shapes):
  dp = input_shapes['desire_pulse']  # (1, 25, 8)
  tc = input_shapes['traffic_convention']  # (1, 2)
  at = input_shapes['action_t']  # (1, 2)
  fb = input_shapes['features_buffer']  # (1, T-1, ...) e.g. (1, 24, 32, 512) with spatial features
  feat_dim = math.prod(fb[2:])
  # TODO prev_feat shouldn't exist and be handled inside the JIT, but corrupt on QCOM for now
  shapes = {'desire': (dp[2],), 'traffic_convention': tuple(tc), 'action_t': tuple(at), 'prev_feat': (fb[0], feat_dim)}
  return shapes, [math.prod(s) for s in shapes.values()]


def make_input_queues(input_shapes, frame_skip, device, frame_copy_size):
  img = input_shapes['img']  # (1, 12, 128, 256)
  fb = input_shapes['features_buffer']  # (1, T-1, ...), past features only; the model appends the current frame's feature
  feat_dim = math.prod(fb[2:])
  dp = input_shapes['desire_pulse']  # (1, 25, 8)
  n_frames = img[1] // 6
  img_buf_shape = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])

  policy_shapes, _ = get_policy_npy_shapes(input_shapes)
  shapes = {'tfm': (3, 3), 'big_tfm': (3, 3)} | policy_shapes
  sizes = [math.prod(s) for s in shapes.values()]
  packed_npy_size = sum(sizes) * np.dtype(np.float32).itemsize
  packed_input = np.zeros(packed_npy_size + 2 * frame_copy_size, dtype=np.uint8)
  packed_npy_inputs = packed_input[:packed_npy_size].view(np.float32)
  frames = packed_input[packed_npy_size:]
  frame_views = {'img': frames[:frame_copy_size], 'big_img': frames[frame_copy_size:]}
  # views into the packed inputs, to be refilled at runtime
  npy = {k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(packed_npy_inputs, np.cumsum(sizes[:-1])), strict=True)}
  input_queues = {
    'img_q': Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'big_img_q': Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    'feat_q': Tensor(np.zeros((frame_skip * fb[1], fb[0], feat_dim), dtype=np.float32), device=device).contiguous().realize(),
    'desire_q': Tensor(np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32), device=device).contiguous().realize(),
    'packed_npy_inputs': Tensor(packed_input, device='NPY').realize(),
  }
  return input_queues, npy, frame_views


def shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)


def sample_skip(buf, frame_skip):
  return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)


def sample_desire(buf, frame_skip):
  return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)


def make_warp(nv12, model_w, model_h):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h)

  def warp(tfm, big_tfm, frame, big_frame):
    tfm = tfm.to(Device.DEFAULT)
    big_tfm = big_tfm.to(Device.DEFAULT)
    frame = frame.to(Device.DEFAULT)
    big_frame = big_frame.to(Device.DEFAULT)
    Tensor.realize(tfm, big_tfm, frame, big_frame)

    warped_frame = frame_prepare(frame, tfm).unsqueeze(0)
    warped_big_frame = frame_prepare(big_frame, big_tfm).unsqueeze(0)
    return Tensor.cat(warped_frame, warped_big_frame)

  return warp


def make_run_policy(model_runner, model_metadata, frame_skip):
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  npy_shapes, npy_sizes = get_policy_npy_shapes(model_metadata['input_shapes'])
  model_input_dtypes = {name: spec.dtype for name, spec in model_runner.graph_inputs.items()}

  def run_policy(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_npy_inputs = packed_npy_inputs.to(Device.DEFAULT)
    Tensor.realize(packed_npy_inputs, warped)

    img = shift_and_sample(img_q, warped[0:1], sample_skip_fn)
    big_img = shift_and_sample(big_img_q, warped[1:2], sample_skip_fn)

    desire, traffic_convention, action_t, prev_feat = (t.reshape(s) for t, s in zip(packed_npy_inputs.split(npy_sizes), npy_shapes.values(), strict=True))
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire_fn)
    feat_buf = shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), sample_skip_fn)

    inputs = {
      'img': img,
      'big_img': big_img,
      'features_buffer': feat_buf.reshape(model_metadata['input_shapes']['features_buffer']),
      'desire_pulse': desire_buf,
      'traffic_convention': traffic_convention,
      'action_t': action_t,
    }
    inputs = {name: value.cast(model_input_dtypes[name]) for name, value in inputs.items()}
    out = next(iter(model_runner(inputs).values())).cast('float32')
    return out,
  return run_policy


def make_run_model(warp, run_policy, model_metadata, frame_copy_size):
  _, policy_sizes = get_policy_npy_shapes(model_metadata['input_shapes'])
  packed_npy_size = (18 + sum(policy_sizes)) * np.dtype(np.float32).itemsize

  def run_model(img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_input = packed_npy_inputs.to(Device.DEFAULT)
    Tensor.realize(packed_input)
    packed_npy_inputs = packed_input[:packed_npy_size].bitcast('float32')
    frame = packed_input[packed_npy_size:packed_npy_size + frame_copy_size]
    big_frame = packed_input[packed_npy_size + frame_copy_size:]
    tfm, big_tfm, policy_inputs = packed_npy_inputs.split([9, 9, sum(policy_sizes)])
    warped = warp(tfm.reshape(3, 3), big_tfm.reshape(3, 3), frame, big_frame)
    return run_policy(warped, img_q, big_img_q, feat_q, desire_q, policy_inputs)
  return run_model


def compile_model(onnx, frames, model_size, frame_skip, benchmark_runs=20):
  runner = OnnxRunner(onnx)
  metadata = onnx_metadata(onnx)
  properties = metadata.pop('metadata')
  metadata.update(model_checkpoint=properties.get('model_checkpoint'),
                  output_slices=pickle.loads(codecs.decode(properties['output_slices'].encode(), 'base64')))
  out = {'metadata': metadata, 'input_devices': {'model': Device.DEFAULT}, 'run_model': {}, 'input_specs': {}}
  out['npy_shapes'] = {'tfm': (3, 3), 'big_tfm': (3, 3)} | get_policy_npy_shapes(metadata['input_shapes'])[0]
  policy = make_run_policy(runner, metadata, frame_skip)
  for frame in frames:
    def make_inputs(seed):
      queues, npy, images = make_input_queues(metadata['input_shapes'], frame_skip, Device.DEFAULT, frame.copy_size)
      rng = np.random.default_rng(seed)
      for v in npy.values(): v[:] = rng.standard_normal(v.shape).astype(v.dtype)
      for v in images.values(): v[:] = rng.integers(0, 256, v.shape, dtype=np.uint8)
      return (), queues
    queues = make_inputs(42)[1]
    out['input_specs'][(frame.width, frame.height)] = {k: (v.shape, v.numpy().dtype.str, v.device) for k, v in queues.items()}
    warp = make_warp(frame, *model_size)
    out['run_model'][(frame.width, frame.height)] = compile_jit(
      make_run_model(warp, policy, metadata, frame.copy_size), make_inputs, benchmark_runs, out_of_band=True)
  return out


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--onnx', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--model-size', type=parse_size, required=True)
  parser.add_argument('--frames', type=parse_frame, nargs='+', required=True, help='width,height,stride,y_height,uv_height,buffer_size')
  parser.add_argument('--frame-skip', type=int, required=True)
  parser.add_argument('--benchmark-runs', type=int, default=20)
  args = parser.parse_args()
  model = compile_model(args.onnx, args.frames, args.model_size, args.frame_skip, args.benchmark_runs)
  with open(args.output, 'wb') as f: dump_pickle(model, f, out_of_band=True)
