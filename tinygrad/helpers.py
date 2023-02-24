from collections import namedtuple
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

def shape_to_axis(old_shape, new_shape):
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple([i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b])

ConvArgs = namedtuple('ConvArgs', ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'sy', 'sx', 'bs', 'cout', 'py', 'py_', 'px', 'px_', 'dy', 'dx', 'out_shape'])
def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1, out_shape=None):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  cout,cin,H,W = w_shape
  sy,sx = make_pair(stride)
  px,px_,py,py_ = [padding]*4 if isinstance(padding, int) else (padding if len(padding) == 4 else [padding[1], padding[1], padding[0], padding[0]])
  dy,dx = make_pair(dilation)
  bs,cin_,iy,ix = x_shape

  # this can change px_ and py_ to make the out_shape right
  # TODO: copy padding names from http://nvdla.org/hw/v1/ias/unit_description.html
  if out_shape is not None:
    py_ = (out_shape[2] - 1) * sy + 1 + dy * (H-1) - iy - py
    px_ = (out_shape[3] - 1) * sx + 1 + dx * (W-1) - ix - px

  # TODO: should be easy to support asymmetric padding by changing output size
  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html describes these sizes well
  oy = (iy + py + py_ - dy * (H-1) - 1)//sy + 1
  ox = (ix + px + px_ - dx * (W-1) - 1)//sx + 1
  if cin*groups != cin_:
    raise TypeError(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0 and (out_shape is None or out_shape == (bs, cout, oy, ox))
  return ConvArgs(H, W, groups, cout//groups, cin, oy, ox, iy, ix, sy, sx, bs, cout, py, py_, px, px_, dy, dx, (bs, cout, oy, ox))
