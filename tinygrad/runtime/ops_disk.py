import os, mmap
try: import _posixshmem
except Exception: pass
from typing import Optional
from typing import Callable, Dict, Tuple
from tinygrad.helpers import prod, all_int, DType, OSX
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, MovementOps, UnaryOps, BufferOps
from tinygrad.shape.view import strides_for_shape
MAP_LOCKED, MAP_POPULATE = 0x2000, 0x008000

class UnderlyingDiskBuffer:
  def __init__(self, fd, mem, offset=0): self.fd, self.mem, self.offset = fd, mem, offset
  def __del__(self): self.fd.close()

class RawDiskBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, buf=None, device:Optional[str]=None):  # pylint: disable=super-init-not-called
    assert device is not None or buf is not None, "disk tensor needs a path or a buf"
    if device is not None:
      if str(device).startswith("shm:"):
        if OSX:
          with open(f"/tmp/shm_{device[4:]}", "w+b") as f:
            f.truncate(size * dtype.itemsize)
            shm = mmap.mmap(f.fileno(), size * dtype.itemsize, flags=mmap.MAP_SHARED)
        else:
          fd = _posixshmem.shm_open(device[4:], os.O_RDWR, 0o600)
          # TODO: these flags are somewhat platform specific, but python doesn't expose the ones we need
          shm = mmap.mmap(fd, size * dtype.itemsize, flags=mmap.MAP_SHARED | MAP_LOCKED | MAP_POPULATE)
          shm.madvise(mmap.MADV_HUGEPAGE)     # type: ignore   # not on OSX
          os.close(fd)
        buf = UnderlyingDiskBuffer(None, shm)
      else:
        f = open(device, "a+b")
        if os.path.getsize(device) < size * dtype.itemsize: os.ftruncate(f.fileno(), size * dtype.itemsize)
        buf = UnderlyingDiskBuffer(f, mmap.mmap(f.fileno(), size * dtype.itemsize))
    # NOTE: we don't call super since disk tensors don't use RAM
    self.size, self.dtype, self._buf = size, dtype, buf
  def __del__(self):
    self._buf[2] -= 1
    if self._buf[2] == 0 and self._buf[0] is not None: self._buf[0].close()
  def cast(self, arg:Tuple[DType, bool]):
    return RawDiskBuffer(self.size, arg[0], self._buf)
  def as_strided(self, arg):
    assert strides_for_shape(arg[0]) == arg[1], "disk tensors don't support strides"
    return RawDiskBuffer(prod(arg[0]), self.dtype, UnderlyingDiskBuffer(self._buf.fd, self._buf.mem, offset=self._buf.offset+arg[2]*self.dtype.itemsize))
  def _buffer(self): return memoryview(self._buf.mem)[self._buf.offset:self._buf.offset+self.size*self.dtype.itemsize]
  def readinto(self, buf:memoryview):
    if self._buf.fd is not None:
      self._buf.fd.seek(self._buf.offset)
      self._buf.fd.readinto(buf)
    else:
      buf.cast('B')[:] = self._buffer()
  def transfer(self, cls, shape, dtype, **kwargs):
    assert all_int(shape), "does not support symbolic shape"
    instance = cls(prod(shape), dtype, **kwargs)
    self.readinto(instance._buffer())
    return instance

disk_fxn_for_op: Dict[Op, Callable] = { BufferOps.MEM: lambda x: x, UnaryOps.NOOP: lambda x: x, UnaryOps.CAST: RawDiskBuffer.cast, MovementOps.AS_STRIDED: RawDiskBuffer.as_strided }
DiskBuffer = Interpreted(RawDiskBuffer, disk_fxn_for_op)
