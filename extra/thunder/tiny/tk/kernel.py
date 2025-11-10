from contextlib import AbstractContextManager
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.group import Group

class _tk_range:
  user_rid = 0
  def __init__(self, end:int, axis_type:AxisType): self.end, self.axis_type, self.done = end, axis_type, False
  def __iter__(self): return self
  def __next__(self):
    if not self.done:
      self.done = True
      _tk_range.user_rid += 1
      self._rng = UOp.range(self.end, _tk_range.user_rid-1, axis_type=self.axis_type)
      return self._rng
    raise StopIteration

class Kernel(AbstractContextManager):
  def __init__(self, grid_size:tuple[int, int, int], block_size:int):
    self.blockIdx_x = UOp.special(grid_size[0], "gidx0")
    self.blockIdx_y = UOp.special(grid_size[1], "gidx1")
    self.blockIdx_z = UOp.special(grid_size[2], "gidx2")
    self.threadIdx_x = UOp.special(block_size, "lidx0")

    self.range_stack = []
    self.store_stack = []

  @property
  def warpid(self): return self.threadIdx_x // WARP_THREADS

  def __enter__(self): return self
  def __exit__(self, exc_type, exc_value, traceback): pass

  def group(self, size:int): return Group(size, self)
  @property
  def warp(self): return self.group(1)
  @property
  def warpgroup(self): return self.group(4)

  def range(self, end:int, axis_type:AxisType=AxisType.LOOP, track:bool=True):
    rng = _tk_range(end, axis_type)
    if track: self.range_stack.append(rng)
    return rng

  def push_store(self, store:UOp, uop:UOp): self.store_stack.append((store, uop))

  def finish(self):
    # end all ranges
    rngs = []
    while self.range_stack: rngs.append(self.range_stack.pop(0)._rng)

    return self.store_stack.pop()[0].end(*rngs).sink(arg=KernelInfo(opts_to_apply=())).simplify()

  def endrange(self):
    last_store = self.store_stack.pop()
    last_range = self.range_stack.pop()
    return last_store[1].after(last_store[0].barrier().end(last_range._rng)).reshape(last_store[1].shape)
