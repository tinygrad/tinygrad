import os, mmap, _posixshmem, io, functools
from typing import Dict, Tuple, List, Any
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
  def cast(self, arg:Tuple[DType, bool]):
    # TODO: support shape changing bitcast
    #assert arg[1], "DiskTensor only supports bitcast"
    return DiskBuffer(self.ud, self.size, arg[0], offset=self.offset)
  def as_strided(self, arg):
    assert strides_for_shape(arg[0]) == arg[1], "disk tensors don't support strides"
    return DiskBuffer(self.ud, prod(arg[0]), self.dtype, offset=self.offset+arg[2]*self.dtype.itemsize)
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
  def __init__(self, ast:LazyOp):
    # two ASTs are allowed here.
    assert ast.op == BufferOps.STORE, "output of AST must be store"
    assert ast.arg.st.contiguous, "shapetracker must be contiguous"
    if ast.src[0].op == UnaryOps.CAST:
      top_src = ast.src[0].src[0]
      self.should_cast = ast.src[0].arg
    else:
      top_src = ast.src[0]
      self.should_cast = (top_src.arg.dtype, False)
    assert top_src.op == BufferOps.LOAD, "top of AST must be load"
    assert len(top_src.arg.st.views) == 1, "shapetracker must have 1 view"
    self.view = top_src.arg.st.views[0]
    assert self.view.mask is None, "view cannot have a mask"
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Any, int], wait=False, jit=False):
    assert len(rawbufs) == 2
    rawbufs[0]._buf = rawbufs[1]._buf.as_strided((self.view.shape, self.view.strides, self.view.offset))
    if self.should_cast is not None: rawbufs[0]._buf = rawbufs[0]._buf.cast(self.should_cast)

class DiskDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, DiskAllocator(device[len("disk:"):]), None, None)
  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp): return DiskRunner(ast)
