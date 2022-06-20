#!/usr/bin/env python
import os
import time
import io
import unittest
import numpy as np
import onnx
from extra.utils import fetch
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod
from tinygrad.nn import batch_normalize

MAX_CONVS = int(os.getenv("MAX_CONVS", -1))

def get_run_onnx(onnx_model):
  def shape_to_tuple(s): return tuple(x.dim_value for x in s.dim)
  def buffer_parse(inp):
    if inp.data_type == 1:
      ret = Tensor(np.frombuffer(inp.raw_data, dtype=np.float32).reshape(inp.dims).copy())
    elif inp.data_type == 7:
      ret = Tensor(np.frombuffer(inp.raw_data, dtype=np.int64).reshape(inp.dims).astype(np.float32).copy())
    else:
      raise Exception(f"bad data type {inp.name} {inp.dims} {inp.data_type}")
    return ret

  def attribute_parse(a):
    if a.type == 7: return tuple([int(x) for x in a.ints])
    elif a.type == 4: return buffer_parse(a.t)  # TENSOR
    elif a.type == 2: return int(a.i)
    elif a.type == 1: return float(a.f)
    else: raise Exception(f"can't parse {a.type} {a}")
  def attribute_to_dict(a): return {x.name:attribute_parse(x) for x in a}

  tensors = {}

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    if len(inp.raw_data) > 0:
      tensors[inp.name] = buffer_parse(inp)
    elif len(inp.float_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims))
    elif len(inp.int64_data) > 0:
      tensors[inp.name] = Tensor(np.array(inp.int64_data, dtype=np.float32).reshape(inp.dims))
    else:
      print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
      print(inp)
      raise Exception("no data")

  def run_onnx(inputs={}, debug=False):
    input_tensors = {}

    # get inputs
    for inp in onnx_model.graph.input:
      if inp.name in tensors: continue
      shape = shape_to_tuple(inp.type.tensor_type.shape)
      if shape[0] == 0: shape = tuple([1]+list(shape[1:]))   # 1 batch size
      if inp.name in inputs:
        input_shape = inputs[inp.name].shape
        assert input_shape == shape, f"wrong shape for input {inp.name}, {input_shape} isn't {shape}"
        input_tensors[inp.name] = Tensor(inputs[inp.name])
      else:
        raise Exception(f"no data for {inp.name} with shape {shape}")

    conv_count = 0
    for num,n in enumerate(onnx_model.graph.node):
      if debug: print(f"{num}: op {n.op_type}")
      inp = [tensors[x] if x in tensors else input_tensors[x] for x in n.input]
      opt = attribute_to_dict(n.attribute)

      # free ones
      if n.op_type == "Relu": ret = inp[0].relu()
      elif n.op_type == "Sigmoid": ret = inp[0].sigmoid()
      elif n.op_type == "Tanh": ret = inp[0].tanh()
      elif n.op_type == "Softmax": ret = inp[0].softmax()
      elif n.op_type == "MatMul": ret = inp[0].matmul(inp[1])
      # one liners
      elif n.op_type == "Elu": ret = inp[0].elu(alpha=opt['alpha'])
      elif n.op_type == "Clip": ret = inp[0].clip(*(inp[1:] if len(inp) > 1 else (opt['min'], opt['max'])))
      elif n.op_type == "Concat": ret = inp[0].cat(*inp[1:], dim=opt['axis'])
      elif n.op_type == "Flatten": ret = inp[0].flatten(opt['axis'] if 'axis' in opt else 0)
      elif n.op_type == "Transpose": ret = inp[0].permute(order=opt['perm'])
      elif n.op_type == "Squeeze": ret = inp[0].reshape([s for i,s in enumerate(inp[0].shape) if i not in opt['axes']])
      elif n.op_type == "GlobalAveragePool": ret = inp[0].mean(axis=tuple(range(2, len(inp[0].shape))), keepdim=True)
      elif n.op_type == "BatchNormalization": ret = batch_normalize(inp[0], inp[1], inp[2], inp[3], inp[4], opt.get('epsilon', 1e-5))
      elif n.op_type == "Gemm": ret = inp[0].linear(inp[1].transpose() if opt.get('transB', 0) == 1 else inp[1], inp[2])
      elif n.op_type == "Conv":
        x,w,b = inp if len(inp) == 3 else (inp[0], inp[1], None)
        assert 'dilations' not in opt or opt['dilations'] == (1,1)
        if opt['pads'][0] == opt['pads'][2] and opt['pads'][1] == opt['pads'][3]:
          # symmetric padding
          ret = x.conv2d(w, b, stride=opt['strides'], groups=opt.get('group', 1), padding=opt['pads'][0:2])
        else:
          x = x.pad2d((opt['pads'][0], opt['pads'][2], opt['pads'][1], opt['pads'][3]))
          ret = x.conv2d(w, b, stride=opt['strides'], groups=opt.get('group', 1))
        conv_count += 1
        if conv_count == MAX_CONVS:
          ret.numpy()
          break
      elif n.op_type in ["Add", "Sub", "Mul"]:
        # TODO: add this to tinygrad? i don't think it's in torch
        if len(inp[0].shape) != len(inp[1].shape) and prod(inp[0].shape) == prod(inp[1].shape):
          inp[1] = inp[1].reshape(inp[0].shape)
        # TODO: is this right?
        if 'broadcast' in opt: inp[1] = inp[1].reshape([-1 if i == opt['broadcast'] else 1 for i in range(len(inp[0].shape))])
        if n.op_type == "Add": ret = inp[0] + inp[1]
        if n.op_type == "Sub": ret = inp[0] - inp[1]
        if n.op_type == "Mul": ret = inp[0] * inp[1]
      elif n.op_type == "Split":
        i = 0
        arg = [(0,x) for x in inp[0].shape]
        for o,s in zip(n.output, opt['split']):
          arg[opt['axis']] = (i,i+s)
          tensors[o] = inp[0].slice(arg=arg)
          i = i+s
        continue
      elif n.op_type == "AveragePool":
        assert opt['kernel_shape'] == opt['strides'] or opt['strides'] == (1,1)
        ret = inp[0].avg_pool2d(opt['kernel_shape'])
      elif n.op_type == "MaxPool":
        assert opt['kernel_shape'] == opt['strides']
        #opt['kernel_shape'] = opt['strides']
        # TODO: this is untested and probably wrong
        ret = inp[0].pad2d(opt['pads'])
        ret = ret.max_pool2d(opt['kernel_shape'])
        # strides aren't supported in max_pool
        #chan = ret.shape[1]
        #w = Tensor.eye(chan).reshape((chan, chan, 1, 1))
        #ret = ret.conv2d(w, stride=opt['strides'])
      else:
        print("UNSUPPORTED", n.op_type, n.input, n.output)
        raise Exception(f"op_type {n.op_type} not supported")
      assert len(n.output) == 1
      if debug: print(ret.shape)
      tensors[n.output[0]] = ret
      #print(ret.numpy().mean())

    return {outp.name:tensors[outp.name] for outp in onnx_model.graph.output}
  return run_onnx

def run_onnx_torch(onnx_model, inputs):
  import torch
  from onnx2torch import convert
  torch_model = convert(onnx_model)
  with torch.no_grad():
    torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
  return torch_out

class TestOnnxModel(unittest.TestCase):
  def test_profile_openpilot_model(self):
    dat = fetch("https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx")
    onnx_model = onnx.load(io.BytesIO(dat))
    run_onnx = get_run_onnx(onnx_model)
    inputs = {
      "input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "desire": np.zeros((1, 8)),
      "traffic_convention": np.array([[1., 0.]]),
      "initial_state": np.zeros((1, 512))
    }
    inputs = {k:v.astype(np.float32) for k,v in inputs.items()}
    tinygrad_out = run_onnx(inputs)['outputs']
    tinygrad_out = tinygrad_out.numpy()

    import cProfile
    import pstats
    pr = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-6)
    pr.enable()
    tinygrad_out = run_onnx(inputs)['outputs']
    tinygrad_out = tinygrad_out.numpy()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    ps.print_stats()

  def test_benchmark_openpilot_model(self):
    dat = fetch("https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx")
    onnx_model = onnx.load(io.BytesIO(dat))
    run_onnx = get_run_onnx(onnx_model)
    for _ in range(5):
      inputs = {
        "input_imgs": np.random.randn(*(1, 12, 128, 256)),
        "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
        "desire": np.zeros((1, 8)),
        "traffic_convention": np.array([[1., 0.]]),
        "initial_state": np.zeros((1, 512))
      }
      inputs = {k:v.astype(np.float32) for k,v in inputs.items()}
      st = time.monotonic()
      tinygrad_out = run_onnx(inputs)['outputs']
      mt = time.monotonic()
      tinygrad_out = tinygrad_out.numpy()
      et = time.monotonic()
      print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(et-mt)*1000.0:.2f} ms for realize")

  def test_openpilot_model(self):
    dat = fetch("https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx")
    onnx_model = onnx.load(io.BytesIO(dat))
    run_onnx = get_run_onnx(onnx_model)
    inputs = {
      "input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "desire": np.zeros((1, 8)),
      "traffic_convention": np.array([[1., 0.]]),
      "initial_state": np.zeros((1, 512))
    }
    inputs = {k:v.astype(np.float32) for k,v in inputs.items()}
    st = time.monotonic()
    tinygrad_out = run_onnx(inputs)['outputs']
    #tinygrad_out.data.realize()
    mt = time.monotonic()
    tinygrad_out = tinygrad_out.numpy()
    et = time.monotonic()
    print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(et-mt)*1000.0:.2f} ms for realize")

    torch_out = run_onnx_torch(onnx_model, inputs).numpy()
    print(tinygrad_out, torch_out)
    np.testing.assert_allclose(torch_out, tinygrad_out, atol=1e-4, rtol=1e-2)

  def test_resnet(self):
    # NOTE: many onnx models can't be run right now due to max pool with strides != kernel_size
    dat = fetch("https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx")
    onnx_model = onnx.load(io.BytesIO(dat))
    from test.test_efficientnet import chicken_img, car_img, preprocess, _LABELS
    run_onnx = get_run_onnx(onnx_model)

    def run(img):
      inputs = {"images:0": preprocess(img, new=True)}
      tinygrad_out = list(run_onnx(inputs, False).values())[0].numpy()
      return tinygrad_out.argmax()
    
    cls = run(chicken_img)
    print(cls, _LABELS[cls])
    assert _LABELS[cls] == "hen"
    cls = run(car_img)
    print(cls, _LABELS[cls])
    assert "car" in _LABELS[cls]

if __name__ == "__main__":
  unittest.main()
