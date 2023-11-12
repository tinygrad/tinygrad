import os, mmap
try: import _posixshmem
except Exception: pass
from typing import Callable, Dict
from tinygrad.helpers import DType, OSX
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, UnaryOps, MovementOps, BufferOps

class RawShmBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, device:str):
    if OSX:
      with open(f"/tmp/shm_{device}", "w+b") as f:
        f.truncate(size * dtype.itemsize)
        shm = mmap.mmap(f.fileno(), size * dtype.itemsize, flags=mmap.MAP_SHARED)
    else:
      fd = _posixshmem.shm_open(device, os.O_RDWR, 0o600)
      # TODO: these flags are somewhat platform specific, but python doesn't expose the ones we need
      shm = mmap.mmap(fd, size * dtype.itemsize, flags=mmap.MAP_SHARED | 0x2000 | 0x008000)
      shm.madvise(mmap.MADV_HUGEPAGE)
      os.close(fd)

    super().__init__(size, dtype, shm)
  def __del__(self): self._buf.close()
  def _buffer(self): return memoryview(self._buf)

# TODO: is this wrong?
shm_fxn_for_op: Dict[Op, Callable] = { BufferOps.MEM: lambda x: x, UnaryOps.NOOP: lambda x:x, MovementOps.RESHAPE: lambda x,_:x, MovementOps.AS_STRIDED: lambda x,_:x }
ShmBuffer = Interpreted(RawShmBuffer, shm_fxn_for_op, from_underlying=lambda x:x)
