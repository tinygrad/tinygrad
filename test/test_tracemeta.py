#!/usr/bin/env python

import unittest
from tinygrad import Tensor, Context, TRACEMETA
from tinygrad.ops import UOp, all_metadata
from tinygrad.helpers import ContextVar

class TestTraceMeta(unittest.TestCase):
  def test_tracemeta_default(self):
    t = Tensor([1,2,3,4])
    t2 = t + t

    # should be metadata for the add operation
    self.assertTrue(any(all_metadata.values()))

  def test_tracemeta_context_zero(self):
    all_metadata.clear()

    with Context(TRACEMETA=0):
      t = Tensor([1,2,3,4])
      t2 = t + t

      # no metadata for operations inside the context
      self.assertEqual(len(all_metadata), 0)

    # back to default value
    self.assertEqual(TRACEMETA.value, 1)

  def test_tracemeta_global_zero(self):
    all_metadata.clear()

    original_value = TRACEMETA.value
    try:
      # set TRACEMETA to 0 globally
      ContextVar._cache["TRACEMETA"].value = 0

      t = Tensor([1,2,3,4])
      t2 = t + t
      # should be no metadata
      self.assertEqual(len(all_metadata), 0)
    finally:
      # restore original value
      ContextVar._cache["TRACEMETA"].value = original_value

if __name__ == "__main__":
  unittest.main()