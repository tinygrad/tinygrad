import time, sys
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import get_run_onnx
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import GlobalCounters, fetch
from tinygrad.tensor import _from_np_dtype

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  onnx_model = onnx.load(onnx_path := fetch(sys.argv[1] if len(sys.argv) > 1 else OPENPILOT_MODEL))
  run_onnx = get_run_onnx(onnx_model)

  Tensor.manual_seed(100)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in input_shapes.items()}
  new_inputs_np = {k:inp.numpy() for k,inp in new_inputs.items()}

  # benchmark
  for _ in range(5):
    GlobalCounters.reset()
    st = time.perf_counter_ns()
    ret = next(iter(run_onnx(new_inputs).values())).cast(dtypes.float32).numpy()
    print(f"unjitted: {(time.perf_counter_ns() - st)*1e-6:7.4f} ms")

  run_onnx_jit = TinyJit(run_onnx)
  for _ in range(10):
    GlobalCounters.reset()
    st = time.perf_counter_ns()
    ret = next(iter(run_onnx_jit(new_inputs).values())).cast(dtypes.float32).numpy()
    print(f"jitted:  {(time.perf_counter_ns() - st)*1e-6:7.4f} ms")
