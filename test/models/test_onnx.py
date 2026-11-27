#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.device import Device
from tinygrad.helpers import fetch, Context

from extra.onnx_helpers import validate
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry

np.random.seed(1337)

class TestOnnxModel(unittest.TestCase):
  @unittest.skip("slow")
  def test_efficientnet(self):
    input_name, input_new = "images:0", True
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"),
      input_name, input_new)

  @unittest.skip("TODO: FIX THIS IT CAUSES SEGFAULT")
  def test_shufflenet(self):
    input_name, input_new = "gpu_0/data_0", False
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx"),
      input_name, input_new)

  @unittest.skip("test is very slow")
  def test_resnet(self):
    # NOTE: many onnx models can't be run right now due to max pool with strides != kernel_size
    input_name, input_new = "data", False
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx"),
      input_name, input_new)

  def _test_model(self, fn, input_name, input_new, debug=False):
    run_onnx = OnnxRunner(fn)
    print("onnx loaded")
    from test.models.test_efficientnet import chicken_img, car_img, preprocess, _LABELS

    def run(img):
      inputs = {input_name: preprocess(img, new=input_new)}
      tinygrad_out = list(run_onnx(inputs, debug=debug).values())[0].numpy()
      return tinygrad_out.argmax()

    cls = run(chicken_img)
    print(cls, _LABELS[cls])
    assert _LABELS[cls] == "hen" or _LABELS[cls] == "cock"
    cls = run(car_img)
    print(cls, _LABELS[cls])
    assert "car" in _LABELS[cls] or _LABELS[cls] == "convertible"

@unittest.skipUnless(Device.DEFAULT == "METAL", "only run on METAL")
class TestHuggingFaceOnnxModels(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.enterClassContext(Context(MAX_BUFFER_SIZE=0))

  def test_xlm_roberta_large(self):
    repo_id = "FacebookAI/xlm-roberta-large"
    model_file = "onnx/model.onnx"
    custom_inputs = {
      "input_ids": np.random.randint(0, 250002, (1, 11), dtype=np.int64),
      "attention_mask": np.ones((1, 11), dtype=np.int64),
    }
    onnx_model_path = snapshot_download_with_retry(
      repo_id=repo_id,
      allow_patterns=["*.onnx", "*.onnx_data"],
      local_dir=DOWNLOADS_DIR / repo_id
    )
    onnx_model_path = onnx_model_path / model_file
    validate(onnx_model_path, custom_inputs, rtol=1e-4, atol=1e-3)

if __name__ == "__main__":
  unittest.main()
