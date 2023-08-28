from __future__ import annotations
import os, functools, platform, time, re, contextlib
import numpy as np
from typing import Dict, Tuple, Union, List, NamedTuple, Final, Iterator, ClassVar, Optional, Callable, Any, Iterable
from math import prod # noqa: F401 # pylint:disable=unused-import

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items)
def colored(st, color, background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line
def ansilen(s): return len(re.sub('\x1b\\[(K|.*?m)', '', s))
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def mnum(i) -> str: return str(i) if i >= 0 else f"m{-i}"
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def merge_dicts(ds:Iterable[Dict]) -> Dict:
  kvs = set([(k,v) for d in ds for k,v in d.items()])
  assert len(kvs) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"
  return {k:v for k,v in kvs}
def partition(lst, fxn):
  a: list[Any] = []
  b: list[Any] = []
  for s in lst: (a if fxn(s) else b).append(s)
  return a,b

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

class Context(contextlib.ContextDecorator):
  stack: ClassVar[List[dict[str, int]]] = [{}]
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    Context.stack[-1] = {k:o.value for k,o in ContextVar._cache.items()} # Store current state.
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v # Update to new temporary state.
    Context.stack.append(self.kwargs) # Store the temporary state so we know what to undo later.
  def __exit__(self, *args):
    for k in Context.stack.pop(): ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)

class ContextVar:
  _cache: ClassVar[Dict[str, ContextVar]] = {}
  value: int
  def __new__(cls, key, default_value):
    if key in ContextVar._cache: return ContextVar._cache[key]
    instance = ContextVar._cache[key] = super().__new__(cls)
    instance.value = getenv(key, default_value)
    return instance
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG, IMAGE = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0)
GRAPH, PRUNEGRAPH, GRAPHPATH = getenv("GRAPH", 0), getenv("PRUNEGRAPH", 0), getenv("GRAPHPATH", "/tmp/net")

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

# **** tinygrad now supports dtypes! *****

class DType(NamedTuple):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{self.name}"

# dependent typing?
class ImageDType(DType):
  def __new__(cls, priority, itemsize, name, np, shape):
    return super().__new__(cls, priority, itemsize, name, np)
  def __init__(self, priority, itemsize, name, np, shape):
    self.shape: Tuple[int, ...] = shape  # arbitrary arg for the dtype, used in image for the shape
    super().__init__()
  def __repr__(self): return f"dtypes.{self.name}({self.shape})"

class dtypes:
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes._half4, dtypes._float2, dtypes._float4)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  @staticmethod
  def fields() -> Dict[str, DType]: return DTYPES_DICT
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
  float16: Final[DType] = DType(0, 2, "half", np.float16)
  half = float16
  float32: Final[DType] = DType(4, 4, "float", np.float32)
  float = float32
  float64: Final[DType] = DType(0, 8, "double", np.float64)
  double = float64
  int8: Final[DType] = DType(0, 1, "char", np.int8)
  int16: Final[DType] = DType(1, 2, "short", np.int16)
  int32: Final[DType] = DType(2, 4, "int", np.int32)
  int64: Final[DType] = DType(3, 8, "long", np.int64)
  uint8: Final[DType] = DType(0, 1, "unsigned char", np.uint8)
  uint16: Final[DType] = DType(1, 2, "unsigned short", np.uint16)
  uint32: Final[DType] = DType(2, 4, "unsigned int", np.uint32)
  uint64: Final[DType] = DType(3, 8, "unsigned long", np.uint64)

  # NOTE: bfloat16 isn't supported in numpy
  bfloat16: Final[DType] = DType(0, 2, "__bf16", None)

  # NOTE: these are internal dtypes, should probably check for that
  _half4: Final[DType] = DType(0, 2*4, "half4", None, 4)
  _float2: Final[DType] = DType(4, 4*2, "float2", None, 2)
  _float4: Final[DType] = DType(4, 4*4, "float4", None, 4)
  _arg_int32: Final[DType] = DType(2, 4, "_arg_int32", None)

# HACK: staticmethods are not callable in 3.8 so we have to compare the class
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  mem_cached: ClassVar[int] = 0 # NOTE: this is not reset
  cache: ClassVar[Optional[List[Tuple[Callable, Any, Dict[Any, int]]]]] = None  # List[Tuple[Callable, List[RawBuffer], Dict[Variable, int]]]
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count, GlobalCounters.cache = 0,0,0.0,0,None
