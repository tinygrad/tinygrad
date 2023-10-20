#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.renderer.opencl import OpenCLRenderer
from tinygrad.graph import graph_uops

class TestUopsFlopCounter(unittest.TestCase):
  def test_flops_matmul(self):
    N = 1024
    a = Tensor.rand(N,N)
    b = Tensor.rand(N,N)
    si = (a@b).lazydata.schedule()[-1]
    lin = Linearizer(si.ast)
    lin.hand_coded_optimizations()
    print(lin.colored_shape())
    uops = lin.linearize().uops
    graph_uops(uops)
    for u in uops: print(u)
    print(OpenCLRenderer("matmul", uops)[0])

  def test_flops_reduce(self):
    a = Tensor.rand(1024*1024)
    si = a.sum().lazydata.schedule()[-1]
    lin = Linearizer(si.ast)
    lin.hand_coded_optimizations()
    uops = lin.linearize().uops
    graph_uops(uops)
    #print(OpenCLRenderer("reduce", uops)[0])

if __name__ == '__main__':
  unittest.main()
