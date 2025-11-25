from contextlib import AbstractContextManager
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, AddrSpace
from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.group import Group
from extra.thunder.tiny.tk.tiles import GL, ST_16X16, ST_16X16_SWIZZLED, ST, RT_16X16, RT, RV, TileLayout, VecLayout

class _tk_range:
  user_rid = 0
  def __init__(self, start:int, end:int, step:int, axis_type:AxisType):
    self.start, self.end, self.step = start, end, step
    self.axis_type, self.done = axis_type, False
  def __iter__(self): return self
  def __next__(self):
    if not self.done:
      self.done = True
      _tk_range.user_rid += 1
      self._rng = UOp.range(self.end // self.step, _tk_range.user_rid-1, axis_type=self.axis_type) * self.step + self.start
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

    self.global_slot = 0
    self.shared_slot = 0
    self.register_slot = 0
    self.allocs = {}

  @property
  def warpid(self): return self.threadIdx_x // WARP_THREADS
  @property
  def laneid(self): return self.threadIdx_x % WARP_THREADS

  def __enter__(self): return self
  def __exit__(self, exc_type, exc_value, traceback): pass

  def group(self, size:int): return Group(size, self)
  @property
  def warp(self): return self.group(1)
  @property
  def warpgroup(self): return self.group(4)

  def range(self, start:int, end:int=0, step:int=1, axis_type:AxisType=AxisType.LOOP, track:bool=True):
    if end == 0: start, end = 0, start
    rng = _tk_range(start, end, step, axis_type)
    if track: self.range_stack.append(rng)
    return rng

  def alloc(self, shape, dtype, addrspace:AddrSpace, name:str|None=None):
    match addrspace:
      case AddrSpace.GLOBAL:
        slot = self.global_slot
        self.global_slot += 1
      case AddrSpace.LOCAL:
        slot = self.shared_slot
        self.shared_slot += 1
      case AddrSpace.REG:
        slot = self.register_slot
        self.register_slot += 1

    uop = UOp.placeholder(shape, dtype, slot=slot, addrspace=addrspace)

    if name:
      if (name, shape) in self.allocs: return self.allocs[(name, shape)]
      self.allocs[(name, shape)] = uop

    return uop

  def gl(self, shape, dtype): return GL.create(shape, dtype, self)
  def st(self, shape, dtype, layout=TileLayout.ROW, base_shape=ST_16X16): return ST.create(shape, dtype, layout, base_shape, self)
  def rt(self, shape, dtype, layout=TileLayout.ROW, base_shape=RT_16X16): return RT.create(shape, dtype, layout, base_shape, self)
  def rv(self, length, dtype, layout=VecLayout.ORTHO, rt_base_shape=RT_16X16): return RV.create(length, dtype, layout, rt_base_shape, self)

  def push_store(self, store:UOp, uop:UOp): self.store_stack.append((store, uop))

  def finish(self):
    # end all ranges
    rngs = []
    while self.range_stack: rngs.append(self.range_stack.pop(0)._rng)

    last_store = self.store_stack.pop()[0]
    if hasattr(last_store, '_uop'): uop = last_store._uop
    else: uop = last_store

    return uop.end(*rngs).sink(arg=KernelInfo(opts_to_apply=())).simplify()

  def endrange(self):
    last_store = self.store_stack.pop()
    last_range = self.range_stack.pop()
    return last_store[1].after(last_store[0].end(last_range._rng)).reshape(last_store[1].shape)
