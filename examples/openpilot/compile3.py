import os, sys, pickle, time
import numpy as np
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "NATIVE_MATH" not in os.environ: os.environ["NATIVE_MATH"] = "1"

from tinygrad import fetch, Tensor, TinyJit, Device, Context, GlobalCounters
from tinygrad.helpers import OSX, DEBUG, Timing
from tinygrad.tensor import _from_np_dtype

import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import get_run_onnx   # TODO: port to main tinygrad

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = "/tmp/openpilot.pkl"

def compile():
  # hack to fix GPU on OSX: max doesn't work on half, see test/external/external_gpu_fail_osx.py
  if OSX:
    from tinygrad.ops import BinaryOps
    from tinygrad.renderer.cstyle import ClangRenderer, CStyleLanguage
    CStyleLanguage.code_for_op[BinaryOps.MAX] = ClangRenderer.code_for_op[BinaryOps.MAX]

  Tensor.no_grad = True
  Tensor.training = False

  onnx_bytes = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(onnx_bytes)
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  Tensor.manual_seed(100)
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
  print("created tensors")

  run_onnx_jit = TinyJit(lambda **kwargs: run_onnx(kwargs), prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = next(iter(run_onnx_jit(**new_inputs).values())).cast('float32').numpy()
    if i == 0: test_val = np.copy(ret)
  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  np.testing.assert_equal(test_val, ret)
  print("jit run validated")

  with open(OUTPUT, "wb") as f:
    pickle.dump(run_onnx_jit, f)
  mdl_sz = os.path.getsize(onnx_bytes)
  pkl_sz = os.path.getsize(OUTPUT)
  print(f"mdl size is {mdl_sz/1e6:.2f}M")
  print(f"pkl size is {pkl_sz/1e6:.2f}M")
  print("**** compile done ****")
  return test_val

def test(test_val):
  with open(OUTPUT, "rb") as f:
    run = pickle.load(f)
  Tensor.manual_seed(100)
  new_inputs = {nm:Tensor.randn(*st.shape, dtype=dtype).mul(8).realize() for nm, (st, _, dtype, _) in
                sorted(zip(run.captured.expected_names, run.captured.expected_st_vars_dtype_device))}
  for _ in range(20):
    st = time.perf_counter()
    out = run(**new_inputs)
    mt = time.perf_counter()
    val = out['outputs'].numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
  print(out, val.shape, val.dtype)
  np.testing.assert_equal(test_val, val)
  print("**** test done ****")

if __name__ == "__main__":
  test_val = compile()
  test(test_val)

