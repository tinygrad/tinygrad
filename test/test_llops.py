#!/usr/bin/env python3
import time
import unittest
from tqdm import trange
import numpy as np
from tinygrad.llops.opencl import GPUBuffer, sync, unary_op, binary_op, reduce_op
from tinygrad.tensor import Device

def timeit(fxn, its=1000):
  fxn()
  st = time.monotonic()
  for _ in trange(its):
    fxn()
  return its/(time.monotonic() - st)

@unittest.skipUnless(Device.DEFAULT == Device.GPU, "Not Implemented")
class TestBenchmarkCL(unittest.TestCase):
  def test_benchmark_unary_nosync(self):
    shape = (1024,1024)
    buf = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    def fxn():
      unary_op('1.01*a', buf, buf)
    sync()
    its_sec = timeit(fxn, 100000)
    print(f"unary op (no sync) {its_sec:.2f} its/sec")
    self.assertGreater(its_sec, 10000)

  def test_benchmark_unary_tiny(self):
    shape = (1,)
    buf = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    def fxn():
      unary_op('1.01*a', buf, buf)
      sync()
    its_sec = timeit(fxn, 1000)
    print(f"unary op tiny {its_sec:.2f} its/sec")
    self.assertGreater(its_sec, 1000)

  def test_benchmark_unary(self):
    shape = (1024,1024)
    buf = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    def fxn():
      unary_op('1.01*a', buf, buf)
      sync()
    its_sec = timeit(fxn, 1000)
    print(f"unary op {its_sec:.2f} its/sec")
    self.assertGreater(its_sec, 1000)

  def test_benchmark_binary(self):
    shape = (1024,1024)
    buf_a = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    buf_b = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    def fxn():
      binary_op('a+b', buf_a, buf_b, buf_a)
      sync()
    its_sec = timeit(fxn, 1000)
    print(f"binary op {its_sec:.2f} its/sec")
    self.assertGreater(its_sec, 1000)

  def test_benchmark_reduce(self):
    shape = (1024,1024)
    buf = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    buf_out = GPUBuffer((1024,1))
    def fxn():
      reduce_op('out += a', buf, buf_out)
      sync()
    its_sec = timeit(fxn, 100)
    print(f"reduce op {its_sec:.2f} its/sec")
    self.assertGreater(its_sec, 500)

  def test_benchmark_reduce_full(self):
    shape = (1024,1024)
    buf = GPUBuffer(shape, hostbuf=np.ones(shape, dtype=np.float32))
    buf_out = GPUBuffer((1,1))
    def fxn():
      reduce_op('out += a', buf, buf_out)
      sync()
    its_sec = timeit(fxn, 10)
    print(f"reduce op full {its_sec:.2f} its/sec")
    # this is terrible performance. it's a single kernel doing it all
    #self.assertGreater(its_sec, 500)

if __name__ == '__main__':
  unittest.main()
