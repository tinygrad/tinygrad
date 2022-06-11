#!/usr/bin/env python
import io
import unittest
import numpy as np
import onnx
from extra.utils import fetch
from tinygrad.tensor import Tensor

def run_onnx(dat):
  onnx_model = onnx.load_model(dat)

  def shape_to_tuple(s): return tuple(x.dim_value for x in s.dim)
  def attribute_parse(a):
    if a.type == 7: return tuple([int(x) for x in a.ints])
    elif a.type == 2: return int(a.i)
    elif a.type == 1: return float(a.f)
    else: raise Exception(f"can't parse {a.type} {a}")
  def attribute_to_dict(a): return {x.name:attribute_parse(x) for x in a}

  tensors = {}

  # get inputs
  for inp in onnx_model.graph.input:
    tensors[inp.name] = Tensor.zeros(*shape_to_tuple(inp.type.tensor_type.shape))

  # get weights and biases
  for inp in onnx_model.graph.initializer:
    assert inp.data_type == 1
    #print(inp.name, inp.dims, inp.data_type, len(inp.raw_data))
    tensors[inp.name] = Tensor(np.frombuffer(inp.raw_data, dtype=np.float32).reshape(inp.dims).copy())

  for num,n in enumerate(onnx_model.graph.node):
    #print(f"{num}: op {n.op_type}")
    inp = [tensors[x] for x in n.input]
    opt = attribute_to_dict(n.attribute)
    if n.op_type == "Conv":
      x,w,b = inp
      assert opt['dilations'] == (1,1)
      ret = x.pad2d(opt['pads']).conv2d(w, b, stride=opt['strides'], groups=opt['group'])
    elif n.op_type == "Elu": ret = inp[0].elu(alpha=opt['alpha'])
    elif n.op_type == "Relu": ret = inp[0].relu()
    elif n.op_type == "Sigmoid": ret = inp[0].sigmoid()
    elif n.op_type == "Tanh": ret = inp[0].tanh()
    elif n.op_type == "Add": ret = inp[0] + inp[1]
    elif n.op_type == "Sub": ret = inp[0] - inp[1]
    elif n.op_type == "Mul": ret = inp[0] * inp[1]
    elif n.op_type == "Flatten":
      ret = inp[0].flatten(opt['axis'])
    elif n.op_type == "Concat":
      # TODO: add multicat to tinygrad
      ret = inp[0]
      for x in inp[1:]:
        ret = ret.cat(x, dim=opt['axis'])
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
    else:
      print(n.op_type, n.input, n.output)
      print(n)
      raise Exception(f"op_type {n.op_type} not supported")
    assert len(n.output) == 1
    tensors[n.output[0]] = ret

  return {outp.name:tensors[outp.name] for outp in onnx_model.graph.output}

class TestOpenpilotModel(unittest.TestCase):
  def test(self):
    dat = fetch("https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx")
    out = run_onnx(io.BytesIO(dat))
    print(out)

if __name__ == "__main__":
  unittest.main()
