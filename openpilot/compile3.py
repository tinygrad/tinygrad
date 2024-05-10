#!/usr/bin/env python3
from typing import Dict
import os, sys, io, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

import onnx
from tinygrad import Tensor, fetch, dtypes, GlobalCounters, TinyJit, Device, Context
from tinygrad.lazy import lazycache
from extra.onnx import get_run_onnx  # TODO: migrate to core tinygrad
Device.DEFAULT = "GPU"

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  onnx_data = fetch(sys.argv[1] if len(sys.argv) > 1 else OPENPILOT_MODEL).read_bytes()
  onnx_model = onnx.load(io.BytesIO(onnx_data))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  def get_inputs(): return {k:Tensor.empty(*shp) for k,shp in input_shapes.items()}
  def run_model(**inputs) -> Tensor: return next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()

  jit_run_model = TinyJit(run_model)
  for i in range(3):
    print(f"jit run {i}")
    with Context(DEBUG=2):
      GlobalCounters.reset()
      jit_run_model(**get_inputs())
