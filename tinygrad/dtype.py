from typing import NamedTuple, Final, Optional, ClassVar, Set, Tuple, Dict
import numpy as np  # TODO: remove numpy
import functools

# TODO: migrate this from NamedTuple -> dataclass
class DType(NamedTuple):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{INVERSE_DTYPES_DICT[self]}" if self.sz == 1 else f"dtypes._{INVERSE_DTYPES_DICT[self.scalar()]}{self.sz}"
  def vec(self, sz:int):
    assert sz > 1 and self.sz == 1, f"can't vectorize {self} with size {sz}"
    return DType(self.priority, self.itemsize*sz, f"{INVERSE_DTYPES_DICT[self]}{sz}", None, sz)
  def scalar(self): return DTYPES_DICT[self.name[:-len(str(self.sz))]] if self.sz > 1 else self

# dependent typing?
class ImageDType(DType):
  def __new__(cls, priority, itemsize, name, np, shape, base):
    return super().__new__(cls, priority, itemsize, name, np)
  def __init__(self, priority, itemsize, name, np, shape, base):
    self.shape: Tuple[int, ...] = shape  # arbitrary arg for the dtype, used in image for the shape
    self.base: DType = base
    super().__init__()
  def scalar(self): return self.base
  def vec(self, sz:int): return self.base.vec(sz)
  def __repr__(self): return f"dtypes.{self.name}({self.shape})"
  # TODO: fix this to not need these
  def __hash__(self): return hash((super().__hash__(), self.shape))
  def __eq__(self, x): return super().__eq__(x) and self.shape == x.shape
  def __ne__(self, x): return super().__ne__(x) or self.shape != x.shape

class PtrDType(DType):
  def __new__(cls, dt:DType): return super().__new__(cls, dt.priority, dt.itemsize, dt.name, dt.np, dt.sz)
  def __repr__(self): return f"ptr.{super().__repr__()}"

class dtypes:
  @staticmethod
  def is_float(x: DType) -> bool: return x.scalar() in (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64)
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType) -> bool: return x.scalar() in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64) or dtypes.is_unsigned(x)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x.scalar() in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  @staticmethod  # NOTE: isinstance(True, int) is True in python
  def from_py(x) -> DType: return dtypes.default_float if isinstance(x, float) else dtypes.bool if isinstance(x, bool) else dtypes.default_int
  @staticmethod
  def fields() -> Dict[str, DType]: return DTYPES_DICT
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
  int8: Final[DType] = DType(1, 1, "char", np.int8)
  uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)
  int16: Final[DType] = DType(3, 2, "short", np.int16)
  uint16: Final[DType] = DType(4, 2, "unsigned short", np.uint16)
  int32: Final[DType] = DType(5, 4, "int", np.int32)
  uint32: Final[DType] = DType(6, 4, "unsigned int", np.uint32)
  int64: Final[DType] = DType(7, 8, "long", np.int64)
  uint64: Final[DType] = DType(8, 8, "unsigned long", np.uint64)
  float16: Final[DType] = DType(9, 2, "half", np.float16)
  # bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  bfloat16: Final[DType] = DType(10, 2, "__bf16", None)
  float32: Final[DType] = DType(11, 4, "float", np.float32)
  float64: Final[DType] = DType(12, 8, "double", np.float64)

  # dtype aliases
  half = float16; float = float32; double = float64 # noqa: E702
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702
  char = int8; short = int16; int = int32; long = int64 # noqa: E702

  # NOTE: these are image dtypes
  @staticmethod
  def imageh(shp): return ImageDType(100, 2, "imageh", np.float16, shp, dtypes.float32)
  @staticmethod
  def imagef(shp): return ImageDType(100, 4, "imagef", np.float32, shp, dtypes.float32)

  default_float: ClassVar[DType] = float32
  default_int: ClassVar[DType] = int32

# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
# we don't support weak type and complex type
promo_lattice = { dtypes.bool: [dtypes.int8, dtypes.uint8], dtypes.int8: [dtypes.int16], dtypes.int16: [dtypes.int32], dtypes.int32: [dtypes.int64],
  dtypes.int64: [dtypes.float16, dtypes.bfloat16], dtypes.uint8: [dtypes.int16, dtypes.uint16], dtypes.uint16: [dtypes.int32, dtypes.uint32],
  dtypes.uint32: [dtypes.int64, dtypes.uint64], dtypes.uint64: [dtypes.float16, dtypes.bfloat16],
  dtypes.float16: [dtypes.float32], dtypes.bfloat16: [dtypes.float32], dtypes.float32: [dtypes.float64], }

@functools.lru_cache(None)
def _get_recursive_parents(dtype:DType) -> Set[DType]:
  return set.union(*[_get_recursive_parents(d) for d in promo_lattice[dtype]], {dtype}) if dtype != dtypes.float64 else {dtypes.float64}
@functools.lru_cache(None)
def least_upper_dtype(*ds:DType) -> DType:
  return min(set.intersection(*[_get_recursive_parents(d) for d in ds])) if not (images:=[d for d in ds if isinstance(d, ImageDType)]) else images[0]
def least_upper_float(dt:DType) -> DType: return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.float32)

# HACK: staticmethods are not callable in 3.8 so we have to compare the class
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default')) or v.__class__ is staticmethod)}
INVERSE_DTYPES_DICT = {v:k for k,v in DTYPES_DICT.items()}
