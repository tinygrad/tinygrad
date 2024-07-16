import time
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
import onnxruntime as ort
from extra.onnx import get_run_onnx
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import colored, fetch
from tinygrad.tensor import _from_np_dtype
import numpy as np

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  onnx_model = onnx.load(onnx_path := fetch(OPENPILOT_MODEL))

  Tensor.manual_seed(100)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in input_shapes.items()}
  new_inputs_np = {k:inp.numpy() for k,inp in new_inputs.items()}

  # benchmark
  tms = []
  for _ in range(10):
    st = time.perf_counter_ns()
    ret = next(iter(run_onnx(new_inputs).values())).cast(dtypes.float32).numpy()
    tms.append(time.perf_counter_ns() - st)
  print(f"unjitted: {min(tms)*1e-6:7.2f} ms")

  tms = []
  run_onnx_jit = TinyJit(run_onnx)
  for _ in range(10):
    st = time.perf_counter_ns()
    ret = next(iter(run_onnx_jit(new_inputs).values())).cast(dtypes.float32).numpy()
    tms.append(time.perf_counter_ns() - st)
  print(f"jitted: {min(tms)*1e-6:7.2f} ms")

  # validate
  ort_options = ort.SessionOptions()
  ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  ort_options.log_severity_level = 3
  onnx_session = ort.InferenceSession(onnx_path, ort_options)
  onnx_output = onnx_session.run([onnx_model.graph.output[0].name], new_inputs_np)
  ort_out = onnx_output[0]

  run_onnx = get_run_onnx(onnx_model)
  tinygrad_out = next(iter(run_onnx(new_inputs).values())).cast(dtypes.float32).numpy()

  np.testing.assert_allclose(ort_out, tinygrad_out, atol=2e-3, rtol=1e-2)
  print(colored("outputs validated!", "green"))
