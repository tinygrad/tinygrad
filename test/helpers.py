import numpy as np
from tinygrad.device import JITRunner
from tinygrad.ops import LazyOp, LoadOps
from tinygrad.device import Device
from tinygrad.helpers import CI, OSX, DType, dtypes, getenv
from tinygrad.nn.state import get_parameters

# for speed
def derandomize(x):
  if isinstance(x, LazyOp):
    new_op = LoadOps.EMPTY if x.op == LoadOps.CUSTOM else x.op
    return LazyOp(new_op, tuple([derandomize(s) for s in x.src]), None if x.op == LoadOps.CUSTOM else x.arg)
  x.op = derandomize(x.op)
  return x

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = derandomize(p.lazydata)
    p.realize()

def assert_jit_cache_len(fxn, expected_len):
  assert len(fxn.jit_cache) > 0
  if issubclass(type(fxn.jit_cache[0].prg), JITRunner):
    assert len(fxn.jit_cache) == expected_len
  else:
    assert len(fxn.jit_cache) == 1
    # until we have a better way of typing the prg in JitItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len

def generate_random(dtype: DType, size=10): return np.random.randint(0, 100, size=size, dtype=dtype.np) if dtypes.is_int(dtype) else np.random.choice([True, False], size=size) if dtype == dtypes.bool else np.random.uniform(0, 1, size=size)

def is_dtype_supported(dtype: DType):
  # for GPU, cl_khr_fp16 isn't supported (except now we don't need it!)
  # for LLVM, it segfaults because it can't link to the casting function
  if dtype == dtypes.half: return not (CI and Device.DEFAULT in ["GPU", "LLVM"]) and Device.DEFAULT != "WEBGPU" and getenv("CUDACPU") != 1
  if dtype == dtypes.bfloat16: return False # numpy doesn't support bf16, tested separately in TestBFloat16DType
  if dtype == dtypes.float64: return Device.DEFAULT not in ["WEBGPU", "METAL"] and not OSX
  if dtype in [dtypes.int8, dtypes.uint8]: return Device.DEFAULT not in ["WEBGPU"]
  if dtype in [dtypes.int16, dtypes.uint16]: return Device.DEFAULT not in ["WEBGPU", "TORCH"]
  if dtype == dtypes.uint32: return Device.DEFAULT not in ["TORCH"]
  if dtype in [dtypes.int64, dtypes.uint64]: return Device.DEFAULT not in ["WEBGPU", "TORCH"]
  if dtype == dtypes.bool: return Device.DEFAULT != "WEBGPU" # host-shareablity is a requirement for storage buffers, but 'bool' type is not host-shareable
  return True
