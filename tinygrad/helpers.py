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
def colored(st, color, background=False, bright=False): return f"\u001b[{10*background+60*bright+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color)}m{st}\u001b[0m"  # replace the termcolor library with one line
def partition(lst, fxn): return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def mnum(i) -> str: return str(i) if i >= 0 else f"m{-i}"

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG, IMAGE = getenv("DEBUG", 0), getenv("IMAGE", 0)

# **** tinygrad now supports dtypes! *****

class DType(NamedTuple):
  itemsize: int
  name: str
  np: type  # TODO: someday this will be removed with the "remove numpy" project
  arg: Optional[Any] = None  # arbitrary arg for the dtype, used in image for the shape
  def __repr__(self): return f"dtypes.{self.name}" + f"({self.arg})" if self.arg is not None else ""

class LazyNumpyArray:
  def __init__(self, fxn, shape, dtype): self.fxn, self.shape, self.dtype = fxn, shape, dtype
  def __call__(self): return self.fxn(self)
  def reshape(self, new_shape): return LazyNumpyArray(self.fxn, new_shape, self.dtype)
  def copy(self): return self
  def astype(self, typ): return self

class LazyConst:
  def __init__(self, value, dtype): self.value, self.dtype = value, dtype

class dtypes:
  float16: Final[DType] = DType(2, "half", np.float16)
  float32: Final[DType] = DType(4, "float", np.float32)
  @staticmethod
  def from_np(x:Union[LazyNumpyArray, np.ndarray]) -> DType: return {np.dtype(np.float16): dtypes.float16, np.dtype(np.float32): dtypes.float32}[np.dtype(x.dtype)]

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  cache: ClassVar[Optional[List[Tuple[Callable, Any]]]] = None
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count, GlobalCounters.cache = 0,0,0.0,0,None
