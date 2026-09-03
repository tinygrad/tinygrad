import unittest
from tinygrad import Tensor, UOp, function
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import KernelInfo, Ops

@function(in_kernel=True)
def add_one(a:UOp) -> UOp: return a + 1

@function(in_kernel=True)
def add_two(a:UOp) -> UOp: return a + 2

def simple_kernel(A:UOp, B:UOp) -> UOp:
  return B.index(0).store(add_one(A[0].load())).sink(arg=KernelInfo(name="test"))

def simple_nested_kernel(A:UOp, B:UOp) -> UOp:
  return B.index(0).store(add_two(add_one(A[0].load()))).sink(arg=KernelInfo(name="test"))

def simple_loop_kernel(A:UOp, B:UOp) -> UOp:
  r = UOp.range(4, 0)
  return B[r].store(add_one(A[r].load())).end(r).sink(arg=KernelInfo(name="test"))

@function(in_kernel=True)
def handmade_reduce(a:UOp, n:UOp) -> UOp:
  acc = UOp.placeholder((1,), dtypes.int, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(0))
  r = UOp.range(n, 0)
  nxt = acc.after(r)[0].load() + add_one(a[r].load())
  return acc.after(acc[0].store(nxt).end(r))[0].load()

def var_loop_kernel(A:UOp, B:UOp) -> UOp:
  n = UOp.variable("n", 1, 4, param=True)
  return B[0].store(handmade_reduce(A, n)).sink(arg=KernelInfo(name="test"))

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "TODO: a called SINK is rendered in C style only")
class TestCallSink(unittest.TestCase):
  def test_simple(self):
    a, b = Tensor([1], dtype=dtypes.int).contiguous(), Tensor.zeros(1, dtype=dtypes.int).contiguous()
    self.assertEqual(Tensor.custom_kernel(a, b, fxn=simple_kernel)[1].tolist(), [2])

  def test_nested(self):
    a, b = Tensor([1], dtype=dtypes.int).contiguous(), Tensor.zeros(1, dtype=dtypes.int).contiguous()
    self.assertEqual(Tensor.custom_kernel(a, b, fxn=simple_nested_kernel)[1].tolist(), [4])

  def test_loop(self):
    a, b = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous(), Tensor.zeros(4, dtype=dtypes.int).contiguous()
    self.assertEqual(Tensor.custom_kernel(a, b, fxn=simple_loop_kernel)[1].tolist(), [2, 3, 4, 5])

  def test_var_loop(self):
    a, b = Tensor([1, 2, 3, 4], dtype=dtypes.int).contiguous().realize(), Tensor.zeros(1, dtype=dtypes.int).contiguous().realize()
    kernel = var_loop_kernel(UOp.param(0, dtypes.int, 4), UOp.param(1, dtypes.int, 1)).call(a.uop.buf_uop, b.uop.buf_uop)
    for n, expected in ((2, 5), (4, 14)):
      run_linear(UOp(Ops.LINEAR, src=(kernel,)), var_vals={"n": n})
      self.assertEqual(b.item(), expected)

if __name__ == "__main__": unittest.main()
