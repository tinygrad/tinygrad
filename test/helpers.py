import sys, time, logging, difflib
from typing import Callable, Optional, Tuple, TypeVar
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.ops import UOp, UOps, sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.tensor import _to_np_dtype
from tinygrad.engine.realize import Runner
from tinygrad.dtype import ConstType, DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import CI, OSX, getenv, colored

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = Tensor.empty(p.shape, device=p.device, dtype=p.dtype).lazydata
    p.realize()

def assert_jit_cache_len(fxn, expected_len):
  if not fxn.jit_cache:
    assert expected_len == 0, expected_len
    return
  # until we have a better way of typing the prg in ExecItem
  if issubclass(type(fxn.jit_cache[0].prg), Runner) and not type(fxn.jit_cache[0].prg).__name__.endswith('Graph'):
    assert len(fxn.jit_cache) == expected_len, len(fxn.jit_cache)
  else:
    assert len(fxn.jit_cache) == 1, len(fxn.jit_cache)
    # until we have a better way of typing the prg in ExecItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len

def is_dtype_supported(dtype: DType, device: str = Device.DEFAULT):
  if dtype == dtypes.pyint and device != "PYTHON": return False
  if dtype == dtypes.bfloat16:
    # NOTE: this requires bf16 buffer support
    return device in {"AMD"} or (device in {"CUDA", "NV"} and not CI and not getenv("PTX"))
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  # for CI GPU and OSX, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CI CUDA architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  # PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if dtype == dtypes.half:
    if device == "GPU": return not CI and not OSX
    if device in ["LLVM", "CUDA", "NV"]: return not CI
    if device == "PYTHON": return sys.version_info >= (3, 12)
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

def rand_for_dtype(dt:DType, size:int):
  if dtypes.is_unsigned(dt):
    return np.random.randint(0, 100, size=size, dtype=_to_np_dtype(dt))
  elif dtypes.is_int(dt):
    return np.random.randint(-100, 100, size=size, dtype=_to_np_dtype(dt))
  elif dt == dtypes.bool:
    return np.random.choice([True, False], size=size)
  return np.random.uniform(-10, 10, size=size).astype(_to_np_dtype(dt))

def print_diff(s0, s1, unified=getenv("UNIFIED_DIFF",1)):
  if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO, format="%(message)s")
  if unified:
    lines = list(difflib.unified_diff(str(s0).splitlines(), str(s1).splitlines()))
    diff = "\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in lines)
  else:
    import ocdiff
    diff = ocdiff.console_diff(str(s0), str(s1))
  logging.info(diff)

def assert_equiv_uops(u1:UOp, u2:UOp) -> None:
  if u1 is not u2:
    print_diff(u1, u2)
    raise AssertionError("uops aren't equal.")

def ast_const(dtype:DType, val:ConstType, shape:Tuple[sint, ...]=(), st:Optional[ShapeTracker]=None, st_src:Optional[Tuple[UOp]]=None) -> UOp:
  if st_src is None:
    st_src = (st.to_uop() if st is not None else ShapeTracker.from_shape(()).reshape((1,)*len(shape)).expand(shape).to_uop(),)
  return UOp(UOps.VALID, dtypes.bool, st_src).where(UOp.const(dtype, val), UOp.const(dtype, 0))

T = TypeVar("T")
def timeit(fxn:Callable[..., T], *args, **kwargs) -> Tuple[T, float]:
  st = time.perf_counter_ns()
  ret = fxn(*args, **kwargs)
  return ret, (time.perf_counter_ns()-st)*1e-6
