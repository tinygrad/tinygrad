import os, mmap
try: import _posixshmem
except Exception: pass
from typing import Optional
from typing import Callable, Dict, Tuple
from tinygrad.helpers import prod, DType, OSX
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, MovementOps, UnaryOps, BufferOps
from tinygrad.shape.view import strides_for_shape

class RawDiskBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, device:Optional[str]=None, buf=None, shape=None, offset=0):  # pylint: disable=super-init-not-called
    self.shape = (size, ) if shape is None else shape
    self.offset = offset  # this is an offset in bytes
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
          shm = mmap.mmap(fd, size * dtype.itemsize, flags=mmap.MAP_SHARED | 0x2000 | 0x008000)
          shm.madvise(mmap.MADV_HUGEPAGE)     # type: ignore   # not on OSX
          os.close(fd)
        buf = [None, shm, 1]
      else:
        f = open(device, "a+b")
        if os.path.getsize(device) < size * dtype.itemsize: os.ftruncate(f.fileno(), size * dtype.itemsize)
        buf = [f, mmap.mmap(f.fileno(), size * dtype.itemsize), 1]
    else:
      buf[2] += 1
    # NOTE: we don't call super since disk tensors don't use RAM
    self.size, self.dtype, self._buf = size, dtype, buf
  def __del__(self):
    self._buf[2] -= 1
    if self._buf[2] == 0 and self._buf[0] is not None: self._buf[0].close()
  def cast(self, arg:Tuple[DType, bool]): return RawDiskBuffer(self.size, arg[0], buf=self._buf, shape=self.shape, offset=self.offset)
  def as_strided(self, arg):
    assert strides_for_shape(arg[0]) == arg[1], "disk tensors don't support strides"
    return RawDiskBuffer(prod(arg[0]), self.dtype, buf=self._buf, offset=self.offset+arg[2]*self.dtype.itemsize, shape=arg[0])

  def _buffer(self): return memoryview(self._buf[1])[self.offset:self.offset+self.size*self.dtype.itemsize]
  def readinto(self, buf:memoryview):
    if self._buf[0] is not None:
      self._buf[0].seek(self.offset)
      self._buf[0].readinto(buf)
    else:
      buf.cast('B')[:] = self._buffer()

disk_fxn_for_op: Dict[Op, Callable] = { BufferOps.MEM: lambda x: x, UnaryOps.NOOP: lambda x: x, UnaryOps.CAST: RawDiskBuffer.cast, MovementOps.AS_STRIDED: RawDiskBuffer.as_strided }
DiskBuffer = Interpreted(RawDiskBuffer, disk_fxn_for_op)
