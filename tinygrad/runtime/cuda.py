from typing import Optional
import pycuda.autoprimaryctx # type: ignore # pylint: disable=unused-import # noqa: F401
import pycuda.driver as cuda # type: ignore
from pycuda.compiler import compile # type: ignore
import numpy as np
from tinygrad.ops import DEBUG, GlobalCounters

class CLImage:
  def __init__(self, shape): raise NotImplementedError("CUDA runtime doesn't support images")

class CLBuffer:
  def __init__(self, size): self._cl = cuda.mem_alloc(size)
  def copyin(self, b:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._cl, b, stream)
  def copyout(self, a:np.ndarray): cuda.memcpy_dtoh(a, self._cl)

class CLProgram:
  def __init__(self, name:str, prg:str, binary=False, shared=0, op_estimate:int=0, mem_estimate:int=0):
    self.name, self.op_estimate, self.mem_estimate, self.shared = name, op_estimate, mem_estimate, shared
    if DEBUG >= 4 and not binary: print("CUDA compile", prg)
    if not binary: prg = compile(prg, target="ptx").decode('utf-8')
    if DEBUG >= 5: print(prg)
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(name)

  def __call__(self, global_size, local_size, *args):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    if DEBUG >= 2: print("CUDA launch", global_size, local_size)
    self.prg(*args, block=tuple(local_size), grid=tuple(global_size), shared=self.shared)
    GlobalCounters.log_kernel(self.op_estimate, self.mem_estimate)