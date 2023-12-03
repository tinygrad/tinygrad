import os, mmap
try: import _posixshmem
except Exception: pass
from typing import Callable, Dict, Tuple
from tinygrad.helpers import prod, DType, OSX, dtypes
from tinygrad.device import Interpreted, Allocator
from tinygrad.ops import Op, MovementOps, UnaryOps
from tinygrad.shape.view import strides_for_shape

class UnderlyingDiskBuffer:
  def __init__(self, fd, mem): self.fd, self.mem = fd, mem
  def __del__(self):
    if self.fd: self.fd.close()

class DiskBuffer:
  def __init__(self, ud:UnderlyingDiskBuffer, size:int, dtype:DType=dtypes.uint8, offset=0): self.ud, self.size, self.dtype, self.offset = ud, size, dtype, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} dtype={self.dtype} offset={self.offset}>"
  def cast(self, arg:Tuple[DType, bool]): return DiskBuffer(self.ud, self.size, arg[0], offset=self.offset)
  def as_strided(self, arg):
    assert strides_for_shape(arg[0]) == arg[1], "disk tensors don't support strides"
    return DiskBuffer(self.ud, prod(arg[0]), self.dtype, offset=self.offset+arg[2]*self.dtype.itemsize)
  def _buf(self) -> memoryview: return memoryview(self.ud.mem)[self.offset:self.offset+self.size*self.dtype.itemsize]

disk_fxn_for_op: Dict[Op, Callable] = { UnaryOps.CAST: DiskBuffer.cast, MovementOps.AS_STRIDED: DiskBuffer.as_strided }

MAP_LOCKED, MAP_POPULATE = 0x2000, 0x008000
class DiskAllocator(Allocator):
  def __init__(self, device): self.device = device
  def _alloc(self, size):
    if str(self.device).startswith("shm:"):
      if OSX:
        with open(f"/tmp/shm_{self.device[4:]}", "w+b") as f:
          f.truncate(size)
          shm = mmap.mmap(f.fileno(), size, flags=mmap.MAP_SHARED)
      else:
        fd = _posixshmem.shm_open(self.device[4:], os.O_RDWR, 0o600)
        # TODO: these flags are somewhat platform specific, but python doesn't expose the ones we need
        shm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED | MAP_LOCKED | MAP_POPULATE)
        shm.madvise(mmap.MADV_HUGEPAGE)     # type: ignore   # not on OSX
        os.close(fd)
      buf = UnderlyingDiskBuffer(None, shm)
    else:
      f = open(self.device, "a+b")
      if os.path.getsize(self.device) < size: os.ftruncate(f.fileno(), size)
      buf = UnderlyingDiskBuffer(f, mmap.mmap(f.fileno(), size))
    return DiskBuffer(buf, size)
  def as_buffer(self, src:DiskBuffer): return src._buf()
  def copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def copyout(self, dest:memoryview, src:DiskBuffer):
    if src.ud.fd is not None:
      src.ud.fd.seek(src.offset)
      src.ud.fd.readinto(dest)
    else:
      dest[:] = src._buf()

class DiskDevice(Interpreted):
  def __init__(self, device): super().__init__(DiskAllocator(device[5:]), disk_fxn_for_op)