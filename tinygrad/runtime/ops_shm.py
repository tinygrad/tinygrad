from multiprocessing import shared_memory
from typing import Callable, Dict
from tinygrad.helpers import DType
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, UnaryOps, MovementOps

class RawShmBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, device:str):
    super().__init__(size, dtype, shared_memory.SharedMemory(name=device))
    setattr(self._buf, "shape", (size,))
  def __del__(self): self._buf.close()
  def _buffer(self): return self._buf.buf

shm_fxn_for_op: Dict[Op, Callable] = { UnaryOps.NOOP: lambda x:x, MovementOps.RESHAPE: lambda x,_:x }
ShmBuffer = Interpreted(RawShmBuffer, shm_fxn_for_op, to_underlying=lambda x:x, from_underlying=lambda x:x)
