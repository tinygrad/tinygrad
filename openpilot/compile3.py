#!/usr/bin/env python3
import os, sys, io, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from tinygrad.helpers import fetch
import onnx
from extra.onnx import get_run_onnx
from tinygrad import Tensor, dtypes, TinyJit, GlobalCounters

if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

if __name__ == "__main__":
  onnx_data = fetch(sys.argv[1] if len(sys.argv) > 1 else OPENPILOT_MODEL).read_bytes()

  # load the model
  onnx_model = onnx.load(io.BytesIO(onnx_data))
  run_onnx = get_run_onnx(onnx_model)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}

  # run the model
  @TinyJit
  def run(**inputs) -> Tensor: return next(iter(run_onnx(inputs).values())).cast(dtypes.float32).contiguous()
  for _ in range(3):
    inputs = {k:Tensor.empty(*shp).realize() for k,shp in input_shapes.items()}
    GlobalCounters.reset()
    run(**inputs)


