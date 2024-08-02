import time, sys, hashlib
from pathlib import Path
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import get_run_onnx
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import IMAGE, GlobalCounters, fetch, colored, getenv
from tinygrad.tensor import _from_np_dtype
import numpy as np

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  onnx_model = onnx.load(onnx_path := fetch(OPENPILOT_MODEL))
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

  # validate (only if IMAGE=2 and FLOAT16=0)
  if IMAGE.value == 2 and getenv("FLOAT16") == 0:
    tinygrad_out = next(iter(run_onnx_jit(new_inputs).values())).cast(dtypes.float32).numpy()
    if getenv("SAVE_OUTPUT"):
      np.save(Path(__file__).parent / "openpilot" / f"{hashlib.md5(OPENPILOT_MODEL.encode()).hexdigest()}.npy", tinygrad_out)
    else:
      known_good_out = np.load(Path(__file__).parent / "openpilot" / f"{hashlib.md5(OPENPILOT_MODEL.encode()).hexdigest()}.npy")

      np.testing.assert_allclose(known_good_out, tinygrad_out, atol=1e-2, rtol=1e-2)
      print(colored("outputs validated!", "green"))
  else:
    print(colored("skipping validation", "yellow"))
