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

if __name__ == '__main__':
  unittest.main()
