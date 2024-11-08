import os, sys, pickle, time
import numpy as np
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "JIT_BATCH_SIZE" not in os.environ: os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import fetch, Tensor, TinyJit, Context, GlobalCounters
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import _from_np_dtype

import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import get_run_onnx   # TODO: port to main tinygrad

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/openpilot.pkl"

def compile():
  # hack to fix GPU on OSX: max doesn't work on half, see test/external/external_gpu_fail_osx.py
  #if OSX:
  #  from tinygrad.ops import BinaryOps
  #  from tinygrad.renderer.cstyle import ClangRenderer, CStyleLanguage
  #  CStyleLanguage.code_for_op[BinaryOps.MAX] = ClangRenderer.code_for_op[BinaryOps.MAX]

  Tensor.no_grad = True
  Tensor.training = False

  onnx_bytes = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(onnx_bytes)
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: np.float32 for inp in onnx_model.graph.input}
  if 'input_img' in input_shapes:
    input_shapes['input_img'] = (1, 1812, 1928)
    input_types['input_img'] = np.uint8
  else:
    input_types['input_imgs'] = np.uint8
    input_types['big_input_imgs'] = np.uint8
  Tensor.manual_seed(100)
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
  print("created tensors")
  
  # TODO remove this hack from dm
  if 'input_img' in input_shapes: #DM model
    def fun_to_jit(kwargs):
      MODEL_WIDTH = 1440
      MODEL_HEIGHT = 960
      v_offset = kwargs['input_img'].shape[1] * 2 // 3 - MODEL_HEIGHT
      h_offset = (kwargs['input_img'].shape[2] - MODEL_WIDTH) // 2
      kwargs['input_img'] = kwargs['input_img'][:,v_offset:v_offset+MODEL_HEIGHT, h_offset:h_offset+MODEL_WIDTH].reshape((1,-1))
      return run_onnx(kwargs)
  else:
    fun_to_jit = run_onnx

  run_onnx_jit = TinyJit(lambda **kwargs: fun_to_jit(kwargs), prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = next(iter(run_onnx_jit(**new_inputs).values())).cast('float32').numpy()
    # copy i == 1 so use of JITBEAM is okay
    if i == 1: test_val = np.copy(ret)
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

def test(test_val=None):
  with open(OUTPUT, "rb") as f:
    run = pickle.load(f)
  Tensor.manual_seed(100)
  new_inputs = {nm:Tensor.randn(*st.shape, dtype=dtype).mul(8).realize() for nm, (st, _, dtype, _) in
                sorted(zip(run.captured.expected_names, run.captured.expected_st_vars_dtype_device))}
  for _ in range(20):
    inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}
    st = time.perf_counter()
    for k in new_inputs:
      if 'img' not in k: # dont need to init img tensors, those are backed by openCL GPU memory
        new_inputs[k] = Tensor(inputs_numpy[k])
    out = run(**new_inputs)
    mt = time.perf_counter()
    val = out['outputs'].numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
  print(out, val.shape, val.dtype)
  if test_val is not None: np.testing.assert_equal(test_val, val)
  print("**** test done ****")

if __name__ == "__main__":
  test_val = compile() if not getenv("RUN") else None
  test(test_val)

