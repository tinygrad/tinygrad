import unittest
from tinygrad.uop import Ops, GroupOp
from tinygrad.renderer.isa import X86Ops, X86GroupOp

class TestX86OpValues(unittest.TestCase):
  def test_values(self):
    assert X86Ops.ADD != Ops.ADD
    assert X86Ops.ADD is not Ops.ADD
    assert not isinstance(Ops.ADD, X86Ops)
    assert isinstance(X86Ops.ADD, Ops)
    assert isinstance(X86Ops.ADD, X86Ops)
    assert Ops.ADD not in X86GroupOp.All
    assert X86Ops.ADD not in GroupOp.All
    assert X86Ops.ADD in X86GroupOp.All
    # this is now possible but is essentially invalid
    assert X86Ops.SINK not in X86GroupOp.All
    assert max(op.value for op in GroupOp.All) + 1 == min(op.value for op in X86GroupOp.All)

if __name__ == "__main__":
  unittest.main()