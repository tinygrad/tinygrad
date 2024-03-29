from itertools import chain
from typing import Tuple, Optional, Union
from tinygrad.shape.int_tuple import is_int, is_tuple, compact_strides, crd2idx, product, shape_div, flatten
from tinygrad.shape.symbolic import sint, Variable
import functools
from collections import deque

class View:
  shape: Union[Tuple, sint]
  strides: Union[Tuple, sint]
  
  def __init__(self, shape, strides=None):
    self.shape  = shape
    if strides is None:
      self.strides = compact_strides(self.shape)
    else:
      self.strides = strides

  def __eq__(self, other):
    return self.shape == other.shape and self.strides == other.strides

  def __len__(self):
    return len(self.shape) if is_tuple(self.shape) else 1

  def __call__(self, crd):
    return crd2idx(crd, self.shape, self.strides)

  def __getitem__(self, i):
    if is_tuple(self.shape):
      return View(self.shape[i], self.strides[i])
    else:
      assert i == 0
      return self

  def size(self):
    return product(self.shape)

  def cosize(self):
    return self(self.size() - 1) + 1

  def reshape(self, shape):
    assert product(self.shape) == product(shape), f"Cannot reshape {self.shape} to {shape}"
    return composition(self, View(shape))
  
  def permute(self, perm: Tuple):
    assert len(perm) == len(self.shape)
    new_shape = tuple(self.shape[i] for i in perm)
    new_strides = tuple(self.strides[i] for i in perm)
    return View(new_shape, new_strides)
  
  @property
  def continuous(self) -> bool:
    v = coalesce(self)
    return is_continuous(v.shape, v.strides)
  
  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape:Tuple, strides:Optional[Tuple]=None):
    return View(shape, strides)
   
  def render(self):
    idxs = tuple(Variable(f"idx{i}", 0, product(s)-1) for i,s in enumerate(self.shape))
    return self(idxs).render()
  
  def __str__(self):
    return f"View(shape={self.shape}, strides={self.strides}, continuous={self.continuous})"

  def __repr__(self):
    return f"View(shape={self.shape}, strides={self.strides}, continuous={self.continuous})"

@functools.lru_cache(maxsize=None)
def is_continuous(shape: Union[sint, Tuple], strides: Union[sint, Tuple]) -> bool:
  return strides == compact_strides(shape)
  
def make_view(*views: View) -> View:
  shape, strides = zip(*((a.shape,a.strides) for a in views))
  return View(shape, strides)

def coalesce(view):
  result_shape  = deque([1])
  result_strides = deque([0])
  for (shape,strides) in zip(reversed(flatten(view.shape)),reversed(flatten(view.strides))):
    if shape == 1:
      continue
    elif result_shape[0] == 1:
      result_shape[0]  = shape
      result_strides[0] = strides
    # merge modes if the shape*strides match
    elif result_shape[0] * result_strides[0] == strides:
      result_shape[0] = result_shape[0] * shape
    # append a new mode
    else:
      result_shape.appendleft(shape)
      result_strides.appendleft(strides)

  if len(result_shape) == 1:
    return View(result_shape[0], result_strides[0])
  else:
    return View(tuple(result_shape), tuple(result_strides))

def composition(viewA: View, viewB:View):
  if viewB.strides == 0: return View(viewB.shape, 0)
  
  if is_tuple(viewB.shape):
    return make_view(*tuple(composition(viewA, viewB_i) for viewB_i in viewB))
  else:
    result_shape = deque()
    result_strides = deque()
    rest_shape   = viewB.shape
    rest_strides  = viewB.strides
    for (s, d) in zip(reversed(flatten(viewA.shape)[1:]), reversed(flatten(viewA.strides)[1:])):
      s1 = shape_div(s, rest_strides)
      result_shape.appendleft(min(s1,rest_shape))
      result_strides.appendleft(rest_strides * d)
      rest_shape  = shape_div(rest_shape, abs(s1))
      rest_strides = shape_div(rest_strides, s)

    result_shape.appendleft(rest_shape)
    result_strides.appendleft(rest_strides * flatten(viewA.strides)[0])

    return coalesce(View(tuple(result_shape), tuple(result_strides)))
