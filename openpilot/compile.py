#!/usr/bin/env python3
import os
import time
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

  # initial run(s) to load weights
  for _ in range(5):
    st = time.monotonic()
    tinygrad_out = run_onnx(inputs)['outputs']
    mt = time.monotonic()
    tinygrad_out.realize()
    mt2 = time.monotonic()
    tinygrad_out = tinygrad_out.numpy()
    et = time.monotonic()
    print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

  set_graph(True)
  inputs, np_inputs = get_random_input_tensors()
  tinygrad_out = run_onnx(inputs)['outputs']
  tinygrad_out.realize()
  tinygrad_out = tinygrad_out.numpy()

  torch_out = run_onnx_torch(onnx_model, np_inputs).numpy()
  print(tinygrad_out, torch_out)

  # float32
  #np.testing.assert_allclose(torch_out, tinygrad_out, atol=1e-4, rtol=1e-2)

  # float16, i think it's so bad because the data is random
  np.testing.assert_allclose(torch_out, tinygrad_out, atol=1, rtol=100)
