import os, mmap, _posixshmem, io
from typing import Callable, Dict, Tuple, Optional
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import prod, OSX
from tinygrad.device import Interpreted, Allocator
from tinygrad.ops import Op, MovementOps, UnaryOps
from tinygrad.shape.view import strides_for_shape

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class UnderlyingDiskBuffer:
  def __init__(self, device:str, size:int, fd:Optional[int]=None, mem:Optional[mmap.mmap]=None):
    self.device, self.size, self._fd, self._mem = device, size, fd, mem
  def __load(self):
    if self._mem is not None: return self
    if str(self.device).startswith("shm:"):
      fd = _posixshmem.shm_open("/"+self.device[4:].lstrip("/"), os.O_RDWR, 0o600)
      self._mem = mmap.mmap(fd, self.size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      self._fd = os.open(self.device, os.O_RDWR|os.O_CREAT) #|(0 if OSX else os.O_DIRECT))
      if os.fstat(fd).st_size < self.size: os.ftruncate(fd, self.size)
      self._mem = mmap.mmap(fd, self.size)
    if (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None: mem.madvise(hp) # type: ignore
    return self
  @property
  def fd(self): return self.__load()._fd
  @property
  def mem(self): return self.__load()._mem
  def __del__(self): os.close(self._fd) if self._fd is not None else None

class DiskBuffer:
  def __init__(self, ud:UnderlyingDiskBuffer, size:int, dtype:DType=dtypes.uint8, offset=0):
    self.ud, self.size, self.dtype, self.offset = ud, size, dtype, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} dtype={self.dtype} offset={self.offset}>"
  def cast(self, arg:Tuple[DType, bool]):
    # TODO: support shape changing bitcast
    #assert arg[1], "DiskTensor only supports bitcast"
    return DiskBuffer(self.ud, self.size, arg[0], offset=self.offset)
  def as_strided(self, arg):
    assert strides_for_shape(arg[0]) == arg[1], "disk tensors don't support strides"
    return DiskBuffer(self.ud, prod(arg[0]), self.dtype, offset=self.offset+arg[2]*self.dtype.itemsize)
  def _buf(self) -> memoryview: return memoryview(self.ud.mem)[self.offset:self.offset+self.size*self.dtype.itemsize]

disk_fxn_for_op: Dict[Op, Callable] = { UnaryOps.CAST: DiskBuffer.cast, MovementOps.AS_STRIDED: DiskBuffer.as_strided }

class DiskAllocator(Allocator):
  def __init__(self, device): self.device = device
  def _alloc(self, size): return DiskBuffer(UnderlyingDiskBuffer(self.device, size), size)
  def as_buffer(self, src:DiskBuffer): return src._buf()
  def copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and src.ud.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(src.ud.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        fo.readinto(dest)
    else:
      dest[:] = src._buf()

class DiskDevice(Interpreted):
  def __init__(self, device): super().__init__(DiskAllocator(device[5:]), disk_fxn_for_op)
