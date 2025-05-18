import os, sys, mmap, io, ctypes, ctypes.util, contextlib
from typing import Optional, Generator, Callable
from tinygrad.helpers import OSX, round_up
from tinygrad.device import Compiled, Allocator

class DiskDevice(Compiled):
  def __init__(self, device:str):
    self.size: Optional[int] = None
    self.fd: Optional[int] = None
    self.count = 0
    super().__init__(device, DiskAllocator(self), None, None, None)

  def _might_open(self, size):
    self.count += 1
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None: return
    filename = self.device[len("disk:"):]
    self.size = size
    try: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT|getattr(os, "O_DIRECT", 0))
    except OSError: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
    if os.fstat(self.fd).st_size < self.size: os.ftruncate(self.fd, self.size)
    self.mem = mmap.mmap(self.fd, self.size)
    if hasattr(self.mem, 'madvise') and (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None:
      with contextlib.suppress(OSError): self.mem.madvise(hp) # some systems have transparent_hugepage disabled

  def _might_close(self):
    self.count -= 1
    if self.count == 0:
      if self.fd is not None: os.close(self.fd)
      self.size = None

class DiskBuffer:
  def __init__(self, device:DiskDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
  def _buf(self) -> memoryview:
    assert hasattr(self.device, "mem"), f"DiskBuffer wasn't opened: {self.device.device}"
    return memoryview(self.device.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, dev:DiskDevice): self.dev = dev
  def _alloc(self, size:int, options):
    self.dev._might_open(size)
    return DiskBuffer(self.dev, size)
  def _free(self, opaque, options): self.dev._might_close()
  def _as_buffer(self, src:DiskBuffer): return src._buf()
  def _copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def _copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and self.dev.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(self.dev.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        bytes_read = 0
        while (n := fo.readinto(dest[bytes_read:])) is not None and n > 0: bytes_read += n
    else:
      dest[:] = src._buf()
  def _offset(self, buf:DiskBuffer, size:int, offset:int): return DiskBuffer(buf.device, size, offset)
