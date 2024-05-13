import sys
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import Runner
from tinygrad.dtype import DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import Context, CI, OSX, getenv

def derandomize_model(model):
  with Context(GRAPH=0):
    for p in get_parameters(model):
      p.lazydata = Tensor.empty(p.shape, device=p.device, dtype=p.dtype).lazydata
      p.realize()

def assert_jit_cache_len(fxn, expected_len):
  assert len(fxn.jit_cache) > 0
  # until we have a better way of typing the prg in ExecItem
  if issubclass(type(fxn.jit_cache[0].prg), Runner) and not type(fxn.jit_cache[0].prg).__name__.endswith('Graph'):
    assert len(fxn.jit_cache) == expected_len
  else:
    assert len(fxn.jit_cache) == 1
    # until we have a better way of typing the prg in ExecItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len

def is_dtype_supported(dtype: DType, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16:
    # NOTE: this requires bf16 buffer support
    return device in {"RHIP", "HSA", "AMD"} or (device == "CUDA" and not CI and not getenv("PTX"))
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  if device == "CUDA" and getenv("PTX") and dtype in (dtypes.int8, dtypes.uint8): return False
  # for CI GPU and OSX, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CUDACPU architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  # PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if dtype == dtypes.half:
    if device == "GPU": return not CI and not OSX
    if device in ["LLVM", "CUDA"]: return not CI
    if device == "PYTHON": return sys.version_info >= (3, 12)
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

def rand_for_dtype(dt:DType, size:int):
  if dtypes.is_unsigned(dt):
    return np.random.randint(0, 100, size=size, dtype=dt.np)
  elif dtypes.is_int(dt):
    return np.random.randint(-100, 100, size=size, dtype=dt.np)
  elif dt == dtypes.bool:
    return np.random.choice([True, False], size=size)
  return np.random.uniform(-10, 10, size=size).astype(dt.np)
