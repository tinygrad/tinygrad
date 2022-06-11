#!/usr/bin/env python
import io
import unittest
import numpy as np
import onnx
from extra.utils import fetch
from tinygrad.tensor import Tensor

def run_onnx(onnx_model, inputs={}, debug=False):
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
    #print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
    if len(inp.raw_data) == 0:
      tensors[inp.name] = Tensor(np.array(inp.float_data, dtype=np.float32).reshape(inp.dims))
    else:
      tensors[inp.name] = buffer_parse(inp)

  # get inputs
  for inp in onnx_model.graph.input:
    if inp.name in tensors: continue
    shape = shape_to_tuple(inp.type.tensor_type.shape)
    if shape[0] == 0: shape = tuple([1]+list(shape[1:]))   # 1 batch size
    if inp.name in inputs:
      input_shape = inputs[inp.name].shape
      assert input_shape == shape, f"wrong shape for input {inp.name}, {input_shape} isn't {shape}"
      tensors[inp.name] = Tensor(inputs[inp.name])
    else:
      print(f"filling {inp.name} shape {shape} with 0")
      tensors[inp.name] = Tensor.zeros(*shape)


  for num,n in enumerate(onnx_model.graph.node):
    if debug: print(f"{num}: op {n.op_type}")
    inp = [tensors[x] for x in n.input]
    opt = attribute_to_dict(n.attribute)
    if n.op_type == "Conv":
      x,w,b = inp if len(inp) == 3 else (inp[0], inp[1], None)
      assert opt['dilations'] == (1,1)
      ret = x.pad2d(opt['pads']).conv2d(w, b, stride=opt['strides'], groups=opt['group'])
    elif n.op_type == "Elu": ret = inp[0].elu(alpha=opt['alpha'])
    elif n.op_type == "Relu": ret = inp[0].relu()
    elif n.op_type == "Sigmoid": ret = inp[0].sigmoid()
    elif n.op_type == "Tanh": ret = inp[0].tanh()
    elif n.op_type == "Add": ret = inp[0] + inp[1]
    elif n.op_type == "Sub": ret = inp[0] - inp[1]
    elif n.op_type == "Mul": ret = inp[0] * inp[1]
    elif n.op_type == "Flatten": ret = inp[0].flatten(opt['axis'] if 'axis' in opt else 0)
    elif n.op_type == "Concat": ret = inp[0].cat(*inp[1:], dim=opt['axis'])
    elif n.op_type == "Clip":
      if 'min' in opt and 'max' in opt: ret = inp[0].clip(opt['min'], opt['max'])
      else: ret = inp[0].clip(inp[1], inp[2])
    elif n.op_type == "GlobalAveragePool": ret = inp[0].mean(axis=tuple(range(2, len(inp[0].shape))), keepdim=True)
    elif n.op_type == "Split":
      i = 0
      arg = [(0,x) for x in inp[0].shape]
      for o,s in zip(n.output, opt['split']):
        arg[opt['axis']] = (i,i+s)
        tensors[o] = inp[0].slice(arg=arg)
        i = i+s
      continue
    elif n.op_type == "Gemm":
      a,w,b = inp
      #print(a.shape, w.shape, b.shape)
      if opt['transB'] == 1: w = w.transpose()
      ret = a.linear(w,b)
    elif n.op_type == "BatchNormalization":
      from tinygrad.nn import batch_normalize
      ret = batch_normalize(inp[0], inp[1], inp[2], inp[3], inp[4], opt['epsilon'])
    elif n.op_type == "MaxPool":
      ret = inp[0].pad2d(opt['pads'])
      ret = ret.max_pool2d(opt['kernel_shape'])
      chan = ret.shape[1]
      # strides aren't supported in max_pool
      w = Tensor.eye(chan).reshape((chan, chan, 1, 1))
      ret = ret.conv2d(w, stride=opt['strides'])
    else:
      print("UNSUPPORTED", n.op_type, n.input, n.output)
      raise Exception(f"op_type {n.op_type} not supported")
    assert len(n.output) == 1
    tensors[n.output[0]] = ret

  return {outp.name:tensors[outp.name] for outp in onnx_model.graph.output}

def run_onnx_torch(onnx_model, inputs):
  import torch
  from onnx2torch import convert
  torch_model = convert(onnx_model)
  with torch.no_grad():
    torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
  return torch_out

class TestOnnxModel(unittest.TestCase):
  def test_openpilot_model(self):
    dat = fetch("https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx")
    onnx_model = onnx.load(io.BytesIO(dat))
    inputs = {
      "input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "desire": np.zeros((1, 8)),
      "traffic_convention": np.array([[1., 0.]]),
      "initial_state": np.zeros((1, 512))
    }
    inputs = {k:v.astype(np.float32) for k,v in inputs.items()}
    tinygrad_out = run_onnx(onnx_model, inputs)['outputs'].numpy()
    torch_out = run_onnx_torch(onnx_model, inputs).numpy()
    print(tinygrad_out, torch_out)
    np.testing.assert_allclose(torch_out, tinygrad_out, atol=1e-4, rtol=1e-2)
  def test_resnet(self):
    # mobilenet requires "Shape", "Gather", "Unsqueeze"
    # googlenet doesn't work without dilated convs
    dat = fetch("https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx")
    onnx_model = onnx.load(io.BytesIO(dat))
    from test.test_efficientnet import chicken_img, car_img, _LABELS
    inputs = {"data": chicken_img}
    tinygrad_out = run_onnx(onnx_model, inputs, False)['resnetv15_dense0_fwd'].numpy()
    cls = tinygrad_out.argmax()
    print(cls, _LABELS[cls])

if __name__ == "__main__":
  unittest.main()
