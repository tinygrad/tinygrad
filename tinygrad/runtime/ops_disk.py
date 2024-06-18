from __future__ import annotations
import os, mmap, _posixshmem, io, ctypes
from typing import Optional
from tinygrad.helpers import OSX
from tinygrad.device import Compiled, Allocator
import tinygrad.runtime.autogen.uring as io_uring

def check(status): assert status == 0

class DiskBuffer:
  def __init__(self, device:DiskDevice, size:int, offset=0):
    self.device, self.size, self.offset = device, size, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} offset={self.offset}>"
  def _buf(self) -> memoryview:
    assert self.device.mem is not None, "DiskBuffer wasn't opened"
    return memoryview(self.device.mem)[self.offset:self.offset+self.size]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, device:DiskDevice): self.device = device
  def _alloc(self, size:int, options):
    self.device._might_open(size)
    return DiskBuffer(self.device, size)
  def _free(self, opaque, options): self.device._might_close()
  def as_buffer(self, src:DiskBuffer): return src._buf()
  def copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and hasattr(self.device, 'fd'):
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(self.device.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        fo.readinto(dest)
    else:
      dest[:] = src._buf()
  def offset(self, buf:DiskBuffer, size:int, offset:int): return DiskBuffer(buf.device, size, offset)

class DiskDevice(Compiled):
  io_uring = None

  def __init__(self, device:str):
    if DiskDevice.io_uring is None:
      check(io_uring.io_uring_queue_init(0x1000, ctypes.byref(ring:=io_uring.struct_io_uring()), 0))
      DiskDevice.io_uring = ring

    self.size: Optional[int] = None
    self.count = 0
    super().__init__(device, DiskAllocator(self), None, None, None)
  def _might_open(self, size):
    self.count += 1
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None: return
    filename = self.dname[len("disk:"):]
    self.size = size

    if filename.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      self.mem = mmap.mmap(fd, self.size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      try: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT|(0 if OSX else os.O_DIRECT))
      except OSError: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
      if os.fstat(self.fd).st_size < self.size: os.ftruncate(self.fd, self.size)
      self.mem = mmap.mmap(self.fd, self.size)
    if (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None: self.mem.madvise(hp) # type: ignore
  def _might_close(self):
    self.count -= 1
    if self.count == 0:
      if hasattr(self, 'fd'): os.close(self.fd)
      self.size = None
