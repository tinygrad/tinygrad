from __future__ import annotations
import platform, time, re
from _weakref import _remove_dead_weakref # type: ignore
import inspect
from typing import Tuple, Union, List, Iterator, ClassVar, Any
from tinygrad.GRAPH_MACROS import getenv
from math import prod # noqa: F401 # pylint:disable=unused-import

ShapeType = Tuple[int, ...]
# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def argfix(*x):
  if x[0].__class__ in {tuple, list}:
    try: return tuple(x[0])
    except IndexError: return tuple()
  return tuple(x)
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items)
def colored(st, color, background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line
def ansilen(s): return len(re.sub('\x1b\\[(K|.*?m)', '', s))
def partition(lst, fxn): return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def mnum(i) -> str: return str(i) if i >= 0 else f"m{-i}"
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)

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
  def __bool__(self) -> bool: return self.value != 0
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

  @property
  def value(self) -> int: return ContextVar.ctx_stack[-1][self.key] if self.key in ContextVar.ctx_stack[-1] else self.initial_value

DEBUG, IMAGE = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0)

class Timing(object):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))
