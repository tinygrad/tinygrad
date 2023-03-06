#!/usr/bin/env python
import unittest
import functools
from typing import Tuple, Union, List
from tinygrad.lazy import get_contraction
from tinygrad.helpers import prod

def get_contraction_and_arg_old(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...], arg):
  def get_contraction_old(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]):
    out : List[List[int]] = []
    curr : List[int] = []
    for t in old_shape:
      if len(out) >= len(new_shape): break
      if t*prod(curr) <= new_shape[len(out)]:
        curr.append(t)
      else:
        out.append(curr)
        curr = [t]
    out.append(curr)
    if len(new_shape) == len(out) and all(prod(i) == j and len(i) >= 1 for i,j in zip(out, new_shape)):
        return out
  if contraction := get_contraction_old(old_shape, new_shape):
    numbered, start = [], 0
    for c in contraction:
      numbered.append(list(range(start, start+len(c))))
      start += len(c)
    new_arg = []
    for p in arg: new_arg += numbered[p]
    return new_arg

def get_contraction_and_arg_new(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...], arg):
  if shape_idx_groups := get_contraction(old_shape, new_shape):
    new_arg = functools.reduce(lambda r, x: r + shape_idx_groups[x], arg, [])
    return new_arg


class TestContraction(unittest.TestCase):
  def helper_do_contraction(self, old_shape, new_shape):
    arg = tuple(range(len(new_shape)))
    self.assertEqual(get_contraction_and_arg_new(old_shape, new_shape, arg), get_contraction_and_arg_old(old_shape, new_shape, arg))

  def test_old_vs_new(self):
    self.helper_do_contraction((1,2,3,4,5), (1,2,3,4,5))
    self.helper_do_contraction((1,2,3,4,5), (1,2,3,4*5))
    self.helper_do_contraction((1,2,3,4,5), (1,2,3*4*5))
    self.helper_do_contraction((1,2,3,4,5), (1,2*3*4*5))
    self.helper_do_contraction((1,2,3,4,5), (1,2*3,4,5))
    self.helper_do_contraction((1,2,3,4,5), (1*2,3,4,5))
    self.helper_do_contraction((1,2,3,4,5), (1*2*3,5,4))
    self.helper_do_contraction((1,2,3,4,5), (1,2,3,5,4))
    self.helper_do_contraction((1,2,3,4,5), (2,1,3,4,5))
    self.helper_do_contraction((1,2,3,4,5), (2,1*3,4,5))
    self.helper_do_contraction((1,2,3,4,5), (2,1*3*4,5))
    self.helper_do_contraction((1,2,3,4,5), (1,2,15,4))
    self.helper_do_contraction((1,2,3,4,5), (1,2,1,3,4,5))

  
if __name__ == '__main__':
  unittest.main()
