import os, mmap, _posixshmem, io
from tinygrad.helpers import OSX
from tinygrad.device import Compiled, Allocator

class DiskBuffer:
  def __init__(self, fd, mem, offset, size): self.fd, self.mem, self.offset, self.size = fd, mem, offset, size
  def _buf(self) -> memoryview: return memoryview(self.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, device:str): self.device = device
  def _alloc(self, size:int, options) -> DiskBuffer:
    if self.device.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+self.device[4:].lstrip("/"), os.O_RDWR, 0o600)
      mem = mmap.mmap(fd, size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
      fd = None
    else:
      try: fd = os.open(self.device, os.O_RDWR|os.O_CREAT|(0 if OSX else os.O_DIRECT))
      except OSError: fd = os.open(self.device, os.O_RDWR|os.O_CREAT)
      if os.fstat(fd).st_size < size: os.ftruncate(fd, size)
      mem = mmap.mmap(fd, size)
    if (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None: mem.madvise(hp) # type: ignore
    return DiskBuffer(fd, mem, 0, size)
  def _free(self, opaque:DiskBuffer, options):
    if opaque.fd: os.close(opaque.fd)
  def as_buffer(self, src:DiskBuffer): return src._buf()
  def copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and src.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(src.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        fo.readinto(dest)
    else:
      dest[:] = src._buf()
  def offset(self, buf:DiskBuffer, offset:int, size:int): return DiskBuffer(buf.fd, buf.mem, buf.offset+offset, size)

class DiskDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, DiskAllocator(device[len("disk:"):]), None, None)
