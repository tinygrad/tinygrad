#!/usr/bin/env python
import os
import time
import io
import unittest
import numpy as np
import onnx
from extra.utils import fetch
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor

def run_onnx_torch(onnx_model, inputs):
  import torch
  from onnx2torch import convert
  torch_model = convert(onnx_model).float()
  with torch.no_grad():
    torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
  return torch_out

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/7da48ebdba5e3cf4c0b8078c934bee9a199f0280/selfdrive/modeld/models/supercombo.onnx"
#OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/1f2f9ea9c9dc37bdea9c6e32e4cb8f88ea0a34bf/selfdrive/modeld/models/supercombo.onnx"

np.random.seed(1337)

class TestOnnxModel(unittest.TestCase):
  def test_benchmark_openpilot_model(self):
    dat = fetch(OPENPILOT_MODEL)
    onnx_model = onnx.load(io.BytesIO(dat))
    run_onnx = get_run_onnx(onnx_model)
    def get_inputs():
      np_inputs = {
        "input_imgs": np.random.randn(*(1, 12, 128, 256)),
        "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
        "desire": np.zeros((1, 8)),
        "traffic_convention": np.array([[1., 0.]]),
        "initial_state": np.zeros((1, 512))
        #"initial_state": np.zeros((1, 768))
      }
      inputs = {k:Tensor(v.astype(np.float32), requires_grad=False) for k,v in np_inputs.items()}
      return inputs

    for _ in range(7):
      inputs = get_inputs()
      st = time.monotonic()
      tinygrad_out = run_onnx(inputs)['outputs']
      mt = time.monotonic()
      tinygrad_out.realize()
      mt2 = time.monotonic()
      tinygrad_out = tinygrad_out.numpy()
      et = time.monotonic()
      print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

    import cProfile
    import pstats
    inputs = get_inputs()
    pr = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-6)
    pr.enable()
    tinygrad_out = run_onnx(inputs)['outputs']
    tinygrad_out.realize()
    tinygrad_out = tinygrad_out.numpy()
    pr.disable()
    stats = pstats.Stats(pr)
    stats.dump_stats("/tmp/net.prof")
    os.system("flameprof /tmp/net.prof > /tmp/prof.svg")
    ps = stats.sort_stats(pstats.SortKey.TIME)
    ps.print_stats(30)

  def test_openpilot_model(self):
    dat = fetch(OPENPILOT_MODEL)
    onnx_model = onnx.load(io.BytesIO(dat))
    run_onnx = get_run_onnx(onnx_model)
    print("got run_onnx")
    inputs = {
      "input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
      "desire": np.zeros((1, 8)),
      "traffic_convention": np.array([[1., 0.]]),
      "initial_state": np.zeros((1, 512))
      #"initial_state": np.zeros((1, 768))
    }
    inputs = {k:v.astype(np.float32) for k,v in inputs.items()}

    st = time.monotonic()
    print("****** run onnx ******")
    tinygrad_out = run_onnx(inputs)['outputs']
    mt = time.monotonic()
    print("****** realize ******")
    tinygrad_out.realize()
    mt2 = time.monotonic()
    tinygrad_out = tinygrad_out.numpy()
    et = time.monotonic()
    print(f"ran openpilot model in {(et-st)*1000.0:.2f} ms, waited {(mt2-mt)*1000.0:.2f} ms for realize, {(et-mt2)*1000.0:.2f} ms for GPU queue")

    Tensor.no_grad = True
    torch_out = run_onnx_torch(onnx_model, inputs).numpy()
    Tensor.no_grad = False
    print(tinygrad_out, torch_out)
    np.testing.assert_allclose(torch_out, tinygrad_out, atol=1e-4, rtol=1e-2)

  def test_efficientnet(self):
    dat = fetch("https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx")
    input_name, input_new = "images:0", True
    self._test_model(dat, input_name, input_new)

  @unittest.skip("maxpool not implemented w strides")
  def test_resnet(self):
    # NOTE: many onnx models can't be run right now due to max pool with strides != kernel_size
    dat = fetch("https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v2-7.onnx")
    input_name, input_new = "data", False
    self._test_model(dat, input_name, input_new)

  def _test_model(self, dat, input_name, input_new):
    onnx_model = onnx.load(io.BytesIO(dat))
    from test.test_efficientnet import chicken_img, car_img, preprocess, _LABELS
    run_onnx = get_run_onnx(onnx_model)

    def run(img):
      inputs = {input_name: preprocess(img, new=input_new)}
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
