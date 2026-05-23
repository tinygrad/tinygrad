#!/usr/bin/env python3
"""Standalone repro/regression for issue #16335."""
import subprocess
from pathlib import Path
from tinygrad import Tensor, Context
from tinygrad.nn.onnx import OnnxRunner

VISION_ONNX = Path("/tmp/deep_vision.onnx")
URL = "https://raw.githubusercontent.com/haraschax/filedump/master/deep_vision.onnx"

def main():
  if not VISION_ONNX.is_file():
    subprocess.run(["curl", "-L", "--fail", "--silent", "--output", str(VISION_ONNX), URL], check=True)
  with Context(IMAGE=1, FLOAT16=1):
    runner = OnnxRunner(str(VISION_ONNX))
    img = Tensor.zeros(1, 12, 128, 256, dtype="uint8").contiguous().realize()
    big_img = Tensor.zeros(1, 12, 128, 256, dtype="uint8").contiguous().realize()
    out = next(iter(runner({"img": img, "big_img": big_img}).values())).cast("float32")
    out.realize()   # END(STORE) assertion fires here during scheduling, before any kernel executes
  print("PASS: scheduled #16335 model without the END(STORE) assertion")

if __name__ == "__main__":
  main()