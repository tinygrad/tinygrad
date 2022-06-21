#!/usr/bin/env python3
import os
import io
os.environ['LAZY'] = '1'

import onnx
import numpy as np

from extra.utils import fetch
from test.test_onnx import get_run_onnx, run_onnx_torch
from tinygrad.tensor import Tensor
from tinygrad.ops import set_graph

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx"
#OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/1f2f9ea9c9dc37bdea9c6e32e4cb8f88ea0a34bf/selfdrive/modeld/models/supercombo.onnx"

def get_random_input_tensors():
  np_inputs = {
    "input_imgs": np.random.randn(*(1, 12, 128, 256)),
    "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
    "desire": np.zeros((1, 8)),
    "traffic_convention": np.array([[1., 0.]]),
    "initial_state": np.zeros((1, 512))
    #"initial_state": np.zeros((1, 768))
  }
  np_inputs = {k:v.astype(np.float32) for k,v in np_inputs.items()}
  inputs = {k:Tensor(v.astype(np.float32), requires_grad=False) for k,v in np_inputs.items()}
  for _,v in inputs.items(): v.realize()
  return inputs, np_inputs

if __name__ == "__main__":
  dat = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(io.BytesIO(dat))
  run_onnx = get_run_onnx(onnx_model)
  inputs, _ = get_random_input_tensors()

  # initial run to load weights
  junk = run_onnx(inputs)['outputs'].numpy()

  print("\n***** real compile here *****\n")

  set_graph(True)
  inputs, np_inputs = get_random_input_tensors()
  tinygrad_out = run_onnx(inputs)['outputs']
  tinygrad_out.realize()
  tinygrad_out = tinygrad_out.numpy()

  torch_out = run_onnx_torch(onnx_model, np_inputs).numpy()
  print(tinygrad_out, torch_out)
  np.testing.assert_allclose(torch_out, tinygrad_out, atol=1e-4, rtol=1e-2)





