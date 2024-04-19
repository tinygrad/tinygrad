from __future__ import annotations
import os, mmap, _posixshmem, io, functools
from typing import Dict, List, Any, Optional
from tinygrad.helpers import prod, OSX
from tinygrad.device import Compiled, Allocator, Runner, Buffer
from tinygrad.ops import UnaryOps, LazyOp, BufferOps
from tinygrad.shape.view import strides_for_shape

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
  def _free(self, buf, options): self.device._might_close()
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

class DiskRunner(Runner):
  def __init__(self, ast:LazyOp):
    # two ASTs are allowed here.
    assert ast.op is BufferOps.STORE, "output of AST must be store"
    assert ast.arg.st.contiguous, "shapetracker must be contiguous"
    # TODO: there shouldn't actually be casts here, bitcasts should fold into the load
    if ast.src[0].op is UnaryOps.CAST:
      top_src = ast.src[0].src[0]
      assert ast.src[0].arg[1], "disk only supports bitcasts, not normal casts"
      self.new_dtype = ast.src[0].arg[0]
    else:
      top_src = ast.src[0]
      self.new_dtype = top_src.arg.dtype
    assert top_src.op is BufferOps.LOAD, "top of AST must be load"
    assert len(top_src.arg.st.views) == 1, "shapetracker must have 1 view"
    view = top_src.arg.st.views[0]
    assert view.mask is None, "view cannot have a mask"
    assert strides_for_shape(view.shape) == view.strides, "disk tensors don't support strides"
    self.new_size = prod(view.shape)
    self.new_offset = view.offset * top_src.arg.dtype.itemsize
    super().__init__(f"sz 0x{self.new_size:X} offset 0x{self.new_offset:X}", "DISK")
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Any, int], wait=False):
    assert len(rawbufs) == 2
    # TODO: this is a terrible hack that should be moved to lazy.py
    rawbufs[0]._buf.offset = rawbufs[1]._buf.offset+self.new_offset

class DiskDevice(Compiled):
  def __init__(self, device:str):
    self.size: Optional[int] = None
    self.count = 0
    super().__init__(device, DiskAllocator(self), None, None)
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
  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, *ast:LazyOp):
    assert len(ast) == 1, "DiskRunner doesn't support multioutput kernels."
    return DiskRunner(ast[0])
