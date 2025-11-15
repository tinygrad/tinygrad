import functools
from dataclasses import dataclass
from tinygrad.dtype import AddrSpace
from tinygrad.mixin import MathMixin
from tinygrad.uop.ops import UOp, Ops

from extra.thunder.tiny.tk import WARP_THREADS

def unwrap(x):
  if hasattr(x, "_uop"): return x._uop
  if isinstance(x, (list, tuple)): return type(x)(unwrap(y) for y in x)
  if isinstance(x, dict): return {k: unwrap(v) for k,v in x.items()}
  return x

def wrap(x, ker, cls):
  if isinstance(x, UOp): return cls(x, ker)
  if isinstance(x, (list, tuple)): return type(x)(wrap(y, ker, cls) for y in x)
  return x

def autowrap(source_cls, blacklist=None):
  if blacklist is None:
    blacklist = {
      "__init__", "__new__", "__str__", "__del__", "__repr__", "__dict__", "__getattribute__",
      "__setattr__", "__delattr__", "__weakref__", "__slots__", "__class__",
      "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__hash__"
    }

  def decorator(cls):
    def __getattr__(self, name):
      uop = object.__getattribute__(self, "_uop")
      val = getattr(uop, name)
      if callable(val):
        @functools.wraps(val)
        def proxy(*args, **kwargs):
          return wrap(val(*unwrap(args), **unwrap(kwargs)), self.ker, cls)
        return proxy
      if name in UOp.__slots__: return val
      return wrap(val, self.ker, cls)
    cls.__getattr__ = __getattr__

    for name in dir(source_cls):
      if name in blacklist or not name.startswith("__"): continue

      for base in cls.mro():
        if base is source_cls: break
        if name in base.__dict__: break
      else:
        original = getattr(source_cls, name)
        if callable(original):
          def make_proxy(op_name, func):
            def proxy(self, *args, **kwargs):
              return wrap(func(self._uop, *unwrap(args), **unwrap(kwargs)), self.ker, cls)
            return proxy
          setattr(cls, name, make_proxy(name, original))

    return cls
  return decorator

class TileMathMixin(MathMixin):
  def alu(self, op, *src, inner_op=lambda x:x):
    assert isinstance(self, (RT, RV))
    if len(src) == 0:
      if self._uop._shape is None: uop = UOp.alu(self._uop, op)
      else: uop = self.ker.warp.map(self._uop, lambda x: UOp.alu(x, op))
    elif len(src) == 1:
      if self._uop._shape is None: uop = UOp.alu(self._uop, op, inner_op(self._uop.ufix(src[0])))
      elif isinstance(src[0], (int,float,bool)): uop = self.ker.warp.map(self._uop, lambda x: UOp.alu(x, op, inner_op(x.ufix(src[0]))))
      elif src[0]._shape is None: uop = UOp.alu(self._uop, op, inner_op(self._uop.ufix(src[0])))
      else:
        if isinstance(self, RT) and isinstance(src[0], RV): uop = self.ker.warp.map(self._uop, lambda x, idx: UOp.alu(x, op, inner_op(src[0]._uop[idx[0], 0, (idx[2]%4)//2])))
        else: uop = self.ker.warp.map(self._uop, lambda x, idx: UOp.alu(x, op, inner_op(src[0]._uop[*idx])))
    else: raise NotImplementedError
    return type(self)(uop, self.ker)
  def const_like(self, b): return b

  # override ops that do compute on the src uop
  def sub(self, x, reverse=False):
    return self.ufix(x).alu(Ops.ADD, self, inner_op=lambda y: -y) if reverse else self.alu(Ops.ADD, self.ufix(x), inner_op=lambda y: -y)
  def div(self, x, reverse=False):
    return self.ufix(x).alu(Ops.MUL, self, inner_op=lambda y: 1/y) if reverse else self.alu(Ops.MUL, self.ufix(x), inner_op=lambda y: 1/y)

@autowrap(UOp)
class GL:
  def __init__(self, uop, ker):
    self._uop, self.ker = uop, ker

  @classmethod
  def create(cls, shape, dtype, ker):
    uop = ker.alloc(shape, dtype, AddrSpace.GLOBAL)
    return cls(uop, ker)

@autowrap(UOp)
class ST:
  def __init__(self, uop, ker):
    self._uop, self.ker = uop, ker

  @classmethod
  def create(cls, shape, dtype, ker):
    uop = ker.alloc(shape, dtype, AddrSpace.LOCAL)
    return cls(uop, ker)

@dataclass(frozen=True)
class RTLayout:
  rows: int
  cols: int
  stride: int

  @property
  def num_elements(self):
    return self.rows * self.cols

  @property
  def elements_per_thread(self):
    return self.num_elements // WARP_THREADS

  @property
  def num_strides(self):
    return self.elements_per_thread // self.stride

RT_16X16 = RTLayout(rows=16, cols=16, stride=4)
RT_32X32 = RTLayout(rows=32, cols=32, stride=4)
RT_32X32_8 = RTLayout(rows=32, cols=32, stride=8)
RT_16X32 = RTLayout(rows=16, cols=32, stride=8)
RT_32X16 = RTLayout(rows=32, cols=16, stride=8)
RT_32X16_4 = RTLayout(rows=32, cols=16, stride=4)
RT_16X32_4 = RTLayout(rows=16, cols=32, stride=4)
RT_16X128 = RTLayout(rows=16, cols=128, stride=16)

@autowrap(UOp)
class RT(TileMathMixin):
  BASE_TILE_ROWS, BASE_TILE_COLS = 16, 16
  BASE_TILE_NE = BASE_TILE_ROWS * BASE_TILE_COLS
  BASE_TILE_NEPT = BASE_TILE_NE // WARP_THREADS

  def __init__(self, uop, ker):
    self._uop, self.ker = uop, ker

  @classmethod
  def create(cls, shape, dtype, layout:RTLayout, ker):
    assert len(shape) == 2
    assert shape[0] % layout.rows == 0
    assert shape[1] % layout.cols == 0

    height = shape[0] // layout.rows
    width = shape[1] // layout.cols

    uop = ker.alloc((height, width, layout.elements_per_thread), dtype, AddrSpace.REG)
    return cls(uop, ker)

@autowrap(UOp)
class RV(TileMathMixin):
  def __init__(self, uop, ker):
    self._uop, self.ker = uop, ker

  @classmethod
  def create(cls, length, dtype, layout, ker):
    tiles = length // RT.BASE_TILE_ROWS

    match layout:
      case "naive":
        inner_dim = 1
        outer_dim = (tiles + 1) // 2
      case "ortho":
        inner_dim = 1
        outer_dim = tiles
      case _: raise NotImplementedError(f"rv layout {layout} not implemented")

    uop = ker.alloc((outer_dim, inner_dim, 2), dtype, AddrSpace.REG)
    return RV(uop, ker)

ALL_TILES = UOp | GL | ST | RT | RV
