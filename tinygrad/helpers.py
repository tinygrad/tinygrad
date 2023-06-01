from __future__ import annotations
from dataclasses import dataclass, asdict
import os, math, functools
import numpy as np
from typing import Tuple, Union, List, NamedTuple, Final, Iterator, ClassVar, Optional, Callable, Any
ShapeType = Tuple[int, ...]
# NOTE: helpers is not allowed to import from anything else in tinygrad

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def prod(x:Union[List[int], Tuple[int, ...]]) -> int: return math.prod(x)
def argfix(*x): return tuple() if len(x) == 0 else tuple(x[0]) if isinstance(x[0], (tuple, list)) else tuple(x)
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items) if len(items) > 0 else True
def colored(st, color, background=False, bright=False): return f"\u001b[{10*background+60*bright+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color)}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line
def partition(lst, fxn): return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def mnum(i) -> str: return str(i) if i >= 0 else f"m{-i}"

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

class Context:
  def __init__(self, **kwargs): self.pvars = kwargs
  def __enter__(self): ContextVar.ctx_stack.append({ **self.pvars, **{ key: ContextVar.ctx_stack[-1][key] for key in ContextVar.ctx_stack[-1].keys() if key not in self.pvars } })
  def __exit__(self, *args): ContextVar.ctx_stack.pop()

class ContextVar:
  ctx_stack: ClassVar[List[dict[str, Any]]] = [{}]
  def __init__(self, key, default_value):
    self.key, self.initial_value = key, getenv(key, default_value)
    if key not in ContextVar.ctx_stack[-1]: ContextVar.ctx_stack[-1][key] = self.initial_value
  def __call__(self, x): ContextVar.ctx_stack[-1][self.key] = x
  def __bool__(self): return self.value != 0
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  @property
  def value(self): return ContextVar.ctx_stack[-1][self.key] if self.key in ContextVar.ctx_stack[-1] else self.initial_value

DEBUG, IMAGE = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0)

# **** tinygrad now supports dtypes! *****

class DType(NamedTuple):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: type  # TODO: someday this will be removed with the "remove numpy" project
  def __repr__(self): return f"dtypes.{self.name}"

# dependent typing?
class ImageDType(DType):
  def __new__(cls, priority, itemsize, name, np, shape):
    return super().__new__(cls, priority, itemsize, name, np)
  def __init__(self, priority, itemsize, name, np, shape):
    self.shape: Tuple[int, ...] = shape  # arbitrary arg for the dtype, used in image for the shape
    super().__init__()
  def __repr__(self): return f"dtypes.{self.name}({self.shape})"

class LazyNumpyArray:
  def __init__(self, fxn, shape, dtype): self.fxn, self.shape, self.dtype = fxn, shape, dtype
  def __call__(self) -> np.ndarray: return np.require(self.fxn(self) if callable(self.fxn) else self.fxn, dtype=self.dtype, requirements='C').reshape(self.shape)
  def reshape(self, new_shape): return LazyNumpyArray(self.fxn, new_shape, self.dtype)
  def copy(self): return self if callable(self.fxn) else LazyNumpyArray(self.fxn, self.shape, self.dtype)
  def astype(self, typ): return LazyNumpyArray(self.fxn, self.shape, typ)


@dataclass
class dtypes:
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.uint8, dtypes.int32, dtypes.int64)
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8)
  @staticmethod
  def from_np(x) -> DType: return asdict(dtypes())[np.dtype(x).name]
  bool: Final[DType] = DType(0, 1, "bool", bool)
  float16: Final[DType] = DType(0, 2, "half", np.float16)
  float32: Final[DType] = DType(1, 4, "float", np.float32)
  int8: Final[DType] = DType(0, 1, "char", np.int8)
  int32: Final[DType] = DType(1, 4, "int", np.int32)
  int64: Final[DType] = DType(2, 8, "int64", np.int64)
  uint8: Final[DType] = DType(0, 1, "uchar", np.uint8)


class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  cache: ClassVar[Optional[List[Tuple[Callable, Any]]]] = None
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count, GlobalCounters.cache = 0,0,0.0,0,None
