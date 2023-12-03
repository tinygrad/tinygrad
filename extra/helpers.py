import numpy as np
import multiprocessing, subprocess
import cloudpickle
from typing import Any
from tinygrad.device import Device

from tinygrad.helpers import CI, OSX, DType, dtypes, getenv

def _early_exec_process(qin, qout):
  while True:
    path, inp = qin.get()
    try:
      qout.put(subprocess.check_output(path, input=inp))
    except Exception as e:
      qout.put(e)

def enable_early_exec():
  qin: multiprocessing.Queue = multiprocessing.Queue()
  qout: multiprocessing.Queue = multiprocessing.Queue()
  p = multiprocessing.Process(target=_early_exec_process, args=(qin, qout))
  p.daemon = True
  p.start()
  def early_exec(x):
    qin.put(x)
    ret = qout.get()
    if isinstance(ret, Exception): raise ret
    else: return ret
  return early_exec

def proc(itermaker, q) -> None:
  try:
    for x in itermaker(): q.put(x)
  except Exception as e:
    q.put(e)
  finally:
    q.put(None)
    q.close()

class _CloudpickleFunctionWrapper:
  def __init__(self, fn): self.fn = fn
  def __getstate__(self): return cloudpickle.dumps(self.fn)
  def __setstate__(self, pfn): self.fn = cloudpickle.loads(pfn)
  def __call__(self, *args, **kwargs) -> Any:  return self.fn(*args, **kwargs)

def cross_process(itermaker, maxsize=16):
  q: multiprocessing.Queue = multiprocessing.Queue(maxsize)
  # multiprocessing uses pickle which cannot dump lambdas, so use cloudpickle.
  p = multiprocessing.Process(target=proc, args=(_CloudpickleFunctionWrapper(itermaker), q))
  p.start()
  while True:
    ret = q.get()
    if isinstance(ret, Exception): raise ret
    elif ret is None: break
    else: yield ret

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
