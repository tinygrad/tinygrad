import os, mmap, _posixshmem, io, functools
from typing import Dict, List, Any
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import prod, OSX
from tinygrad.device import Compiled, Allocator, JITRunner, Buffer
from tinygrad.ops import UnaryOps, LazyOp, BufferOps
from tinygrad.shape.view import strides_for_shape

class UnderlyingDiskBuffer:
  def __init__(self, fd, mem): self.fd, self.mem = fd, mem
  def __del__(self):
    if self.fd is not None: os.close(self.fd)

class DiskBuffer:
  def __init__(self, ud:UnderlyingDiskBuffer, size:int, dtype:DType=dtypes.uint8, offset=0):
    self.ud, self.size, self.dtype, self.offset = ud, size, dtype, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} dtype={self.dtype} offset={self.offset}>"
  def _buf(self) -> memoryview: return memoryview(self.ud.mem)[self.offset:self.offset+self.size*self.dtype.itemsize]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, device:str): self.device = device
  def _alloc(self, size:int):
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
    return DiskBuffer(UnderlyingDiskBuffer(fd, mem), size)
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

class DiskRunner(JITRunner):
  skip_allocation = True
  def __init__(self, ast:LazyOp):
    # two ASTs are allowed here.
    assert ast.op == BufferOps.STORE, "output of AST must be store"
    assert ast.arg.st.contiguous, "shapetracker must be contiguous"
    # TODO: there shouldn't actually be casts here, bitcasts should fold into the load
    if ast.src[0].op == UnaryOps.CAST:
      top_src = ast.src[0].src[0]
      # TODO: assert that this is bitcast
      self.new_dtype = ast.src[0].arg[0]
    else:
      top_src = ast.src[0]
      self.new_dtype = top_src.arg.dtype
    assert top_src.op == BufferOps.LOAD, "top of AST must be load"
    assert len(top_src.arg.st.views) == 1, "shapetracker must have 1 view"
    view = top_src.arg.st.views[0]
    assert view.mask is None, "view cannot have a mask"
    assert strides_for_shape(view.shape) == view.strides, "disk tensors don't support strides"
    self.new_size = prod(view.shape)
    self.new_offset = view.offset
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Any, int], wait=False, jit=False):
    assert len(rawbufs) == 2
    src = rawbufs[1]._buf
    # TODO: src.dtype.itemsize or self.new_dtype.itemsize?
    rawbufs[0]._buf = DiskBuffer(src.ud, self.new_size, self.new_dtype, offset=src.offset+self.new_offset*src.dtype.itemsize)

class DiskDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, DiskAllocator(device[len("disk:"):]), None, None)
  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp): return DiskRunner(ast)
