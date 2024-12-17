import unittest
from tinygrad import Tensor
from tinygrad.ops import UPat, Ops

realized_pattern = UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),))
def is_pattern(ten:Tensor, pat:UPat): assert pat.match(ten.lazydata, {})

class TestTensorUopRepresentation(unittest.TestCase):
  def test_realized(self):
    a = Tensor([1.,2,3]).realize()
    print(a.lazydata)
    is_pattern(a, realized_pattern)

  def test_add_realized(self):
    a = Tensor([1.,2,3]).realize()
    b = Tensor([4.,5,6]).realize()
    c = a+b
    print(c.lazydata)
    is_pattern(c, UPat(Ops.ADD, src=(realized_pattern, realized_pattern)))

  @unittest.expectedFailure
  def test_copyin(self):
    a = Tensor([1.,2,3]).realize()
    c = a.to("TEST")   # NOTE: this isn't checked
    print(c.lazydata)
    # NOTE: this is wrong, COPY has an extra buffer for some reason
    is_pattern(c, UPat(Ops.COPY, src=(realized_pattern,)))

  @unittest.expectedFailure
  def test_const(self):
    a = Tensor(1).realize()
    is_pattern(a, realized_pattern)

  # NOTE: for VIEW of CONST we have two options:
  # a. realize the base, expand
  # b. realize the view
  # depending on which one we pick you can comment out the other assert

  def _assert_realized_const(self, a:Tensor):
    # a.
    # NOTE: this needs to rewrite a VIEW(BUFFER, <op>) that folded to VIEW(BUFFER, CONST) to a STORE(BUFFER, ShapeTracker.from_shape(()), CONST)
    # while keeping the BUFFER around (to mark the tensor_uop as realized)
    realized_pattern.match(a.lazydata.base, realized_pattern)
    self.assertEqual(a.lazydata.base.realized.size, 1) # NOTE: the BUFFER may resize (eg. if it's a Tensor(4,4)*0, we push the movement op to a VIEW)
    self.assertEqual(a.lazydata.op, Ops.EXPAND)
    # b.
    # NOTE: this option is like calling .contiguous() on all the Tensors passed into realize
    realized_pattern.match(a.lazydata, realized_pattern)

  @unittest.expectedFailure
  def test_const_view(self):
    a = Tensor(1).expand(4, 4).realize()
    self._assert_realized_const(a)

  @unittest.expectedFailure
  def test_late_const_fold_simple(self):
    a = ((Tensor([1, 2, 3])+1) * (1-1)).realize()
    self._assert_realized_const(a)

  # NOTE: this behaves like calling .contiguous() on all the Tensors passed into realize
  @unittest.expectedFailure
  def test_late_const_fold_complex(self):
    a = Tensor.uniform(16, 3, 3, 3).realize()
    is_pattern(a, realized_pattern)
    self.assertEqual(a.lazydata.realized.size, 432)

if __name__ == '__main__':
  unittest.main()
