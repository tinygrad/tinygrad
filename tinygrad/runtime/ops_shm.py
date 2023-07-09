import os, mmap, _posixshmem
from typing import Callable, Dict
from tinygrad.helpers import DType
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, UnaryOps, MovementOps

SHM_CACHE: Dict[str, mmap.mmap] = {}

class RawShmBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, device:str):
    device, cache_id = device.split(",")[0], None if "," not in device else device.split(",")[1]

    if cache_id in SHM_CACHE: shm = SHM_CACHE[cache_id]
    else:
      fd = _posixshmem.shm_open(device, os.O_RDWR, 0o600)
      shm = mmap.mmap(fd, size * dtype.itemsize, flags=mmap.MAP_SHARED | 0x2000 | 0x008000)
      os.close(fd)
      if cache_id is not None: SHM_CACHE[cache_id] = shm

    super().__init__(size, dtype, shm)
  def _buffer(self): return memoryview(self._buf)

shm_fxn_for_op: Dict[Op, Callable] = { UnaryOps.NOOP: lambda x:x, MovementOps.RESHAPE: lambda x,_:x }
ShmBuffer = Interpreted(RawShmBuffer, shm_fxn_for_op, to_underlying=lambda x:x, from_underlying=lambda x:x)
