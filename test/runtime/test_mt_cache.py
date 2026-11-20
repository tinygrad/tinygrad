import unittest
from tinygrad import Tensor, Device, Variable
from tinygrad.helpers import Context

@unittest.skipUnless(Device.DEFAULT == 'MT', 'MT cache regressions require MT hardware')
class TestMTHCQCache(unittest.TestCase):
  def test_revisit_with_changing_buffers(self):
    a1, b1 = Tensor([1.,2,3]), Tensor([4.,5,6])
    a2, b2 = Tensor([7.,8,9]), Tensor([10.,11,12])
    self.assertEqual((a1+b1).tolist(), [5.,7.,9.])
    self.assertEqual((a2+b2).tolist(), [17.,19.,21.])
    self.assertEqual((a1+b1).tolist(), [5.,7.,9.])
    self.assertEqual((a2+b2).tolist(), [17.,19.,21.])

  def test_revisit_with_allocations(self):
    a = Tensor([1.,2,3])
    self.assertEqual((a+1).tolist(), [2.,3.,4.])
    for _ in range(3): Tensor.empty(64,64).realize()
    self.assertEqual((a+1).tolist(), [2.,3.,4.])
    b = Tensor([5.,6,7])
    self.assertEqual((b+2).tolist(), [7.,8.,9.])
    self.assertEqual((a+1).tolist(), [2.,3.,4.])

  def test_output_replacement(self):
    a, b = Tensor([1.,2,3]), Tensor([4.,5,6])
    first = (a+b).tolist()
    second = (a+b).tolist()
    self.assertEqual(first, [5.,7.,9.])
    self.assertEqual(second, [5.,7.,9.])
    self.assertEqual((a+b).tolist(), [5.,7.,9.])

  def test_a_b_a_switches(self):
    a, b = Tensor([1.,2,3]), Tensor([4.,5,6])
    self.assertEqual((a+b).tolist(), [5.,7.,9.])
    self.assertEqual((a-b).tolist(), [-3.,-3.,-3.])
    self.assertEqual((a+b).tolist(), [5.,7.,9.])
    self.assertEqual((a-b).tolist(), [-3.,-3.,-3.])
    self.assertEqual((a+b).tolist(), [5.,7.,9.])

  def test_symbolic_reduce_revisits(self):
    i = Variable('i', 1, 10)
    ones = Tensor.ones(10).contiguous()
    for s in [2,5,2,8,5]:
      self.assertEqual(ones[:i.bind(s)].sum().item(), s)

  def test_profiled_revisit(self):
    with Context(DEBUG=2):
      a, b = Tensor([1.,2,3]), Tensor([4.,5,6])
      self.assertEqual((a+b).tolist(), [5.,7.,9.])
      self.assertEqual((a+b).tolist(), [5.,7.,9.])

  def test_gemm_then_gemv(self):
    a, b = Tensor.ones(64,64).contiguous(), Tensor.eye(64).clone()
    self.assertEqual((a@b).tolist(), Tensor.ones(64,64).tolist())
    self.assertEqual((Tensor.ones(1,64)@b).tolist(), Tensor.ones(1,64).tolist())

if __name__ == '__main__': unittest.main()
