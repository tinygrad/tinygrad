#!/usr/bin/env python
import unittest
from tinygrad import Tensor, dtypes
from tinygrad.engine.schedule import create_schedule

class TestAttention(unittest.TestCase):
  def test_half_intermediate_dtypes(self):
    q = Tensor.empty(1, 64, 128, dtype=dtypes.half).realize()
    k = Tensor.empty(1, 64, 128, dtype=dtypes.half).realize()
    v = Tensor.empty(1, 64, 128, dtype=dtypes.half).realize()
    attn = q.scaled_dot_product_attention(k, v)

    sched = create_schedule(attn.lazydata.lbs)
    # TODO: make attention 1 kernel
    self.assertEqual(len(sched), 5)
    # store in half after after matmul
    for buf in sched[0].outputs: self.assertEqual(buf.dtype, dtypes.half)
