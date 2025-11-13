from dataclasses import dataclass
import functools
from tinygrad.dtype import AddrSpace
from tinygrad.mixin import OpMixin
from tinygrad.uop.ops import UOp, UOpMetaClass

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
      val = getattr(self._uop, name)
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
      if name in cls.__dict__: continue

      original = getattr(source_cls, name)
      if callable(original):
        def make_proxy(op_name, func):
          def proxy(self, *args, **kwargs):
            return wrap(func(self._uop, *unwrap(args), **unwrap(kwargs)), self.ker, cls)
          return proxy
        setattr(cls, name, make_proxy(name, original))

    return cls
  return decorator

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

@autowrap(UOp)
class RT(OpMixin):
  TILE_ROW_DIM, TILE_COL_DIM = 16, 16
  BASE_TILE_NE = TILE_ROW_DIM * TILE_COL_DIM
  BASE_TILE_NEPT = BASE_TILE_NE // WARP_THREADS

  def __init__(self, uop, ker):
    self._uop, self.ker = uop, ker

  @classmethod
  def create(cls, shape, dtype, ker):
    assert len(shape) == 2
    assert shape[0] % RT.TILE_ROW_DIM == 0
    assert shape[1] % RT.TILE_COL_DIM == 0

    height = shape[0] // RT.TILE_ROW_DIM
    width = shape[1] // RT.TILE_COL_DIM

    uop = ker.alloc((height, width, RT.BASE_TILE_NEPT), dtype, AddrSpace.REG)
    return cls(uop, ker)

  def alu(self, op, *src, **kwargs):
    assert len(src) == 1
    uop = self.ker.warp.map(self, lambda x, idx: UOp.alu(x, op, src[0][*idx], **kwargs))
    return RT(uop, self.ker)
  def const_like(self, b): return RT(UOp.const_like(self._uop, b), self.ker)
  def _mop(self, op, arg): return RT(UOp._mop(self._uop, op, arg), self.ker)
  @property
  def shape(self): return self._uop.shape

@autowrap(UOp)
class RV(OpMixin):
  def __init__(self, uop, ker):
    self._uop, self.ker = uop, ker

  @classmethod
  def create(cls, length, dtype, layout, ker):
    tiles = length // RT.TILE_ROW_DIM

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

  def alu(self, op, *src, **kwargs):
    assert len(src) == 1
    uop = self.ker.warp.map(self._uop, lambda x, idx: UOp.alu(x, op, src[0].uop[*idx], **kwargs))
    return RV(uop, self.ker)
  def const_like(self, b): return RV(UOp.const_like(self._uop, b), self.ker)
  def _mop(self, op, arg): return RV(UOp._mop(self._uop, op, arg), self.ker)
  @property
  def shape(self): return self._uop.shape
