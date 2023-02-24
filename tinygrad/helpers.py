import os, math, functools, time
from typing import Tuple, Union

def dedup(x): return list(dict.fromkeys(x))   # retains list order
def prod(x): return math.prod(x)
def argfix(*x): return tuple() if len(x) == 0 else tuple(x[0]) if isinstance(x[0], (tuple, list)) else tuple(x)
def argsort(x): return sorted(range(len(x)), key=x.__getitem__) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items): return all(x == items[0] for x in items) if len(items) > 0 else True
def colored(st, color): return f"\u001b[{30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color)}m{st}\u001b[0m"  # replace the termcolor library with one line
def partition(lst, fxn): return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]
def modn(x, a): return -((-x)%a) if x < 0 else x%a
def make_pair(x:Union[int, Tuple[int, ...]]) -> Tuple[int, ...]: return (x,x) if isinstance(x, int) else x

class Timing(object):
  def __enter__(self): self.st = time.monotonic_ns()
  def __exit__(self, exc_type, exc_val, exc_tb): print(f"{(time.monotonic_ns()-self.st)*1e-6:.2f} ms")

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))
DEBUG = getenv("DEBUG", 0)
IMAGE = getenv("IMAGE", 0)

def reduce_shape(shape, axis): return tuple(1 if i in axis else shape[i] for i in range(len(shape)))
def shape_to_axis(old_shape, new_shape):
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple([i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b])
