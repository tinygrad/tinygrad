import os, unittest
os.environ["SQTT"] = "1"
os.environ["AMD"] = "1"
os.environ["VIZ"] = "1"
os.environ["AMD_LLVM"] = "0"

from tinygrad import Tensor
from tinygrad.renderer import ProgramSpec
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.engine.realize import CompiledRunner

def custom_add_one_kernel(B:UOp, A:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert B.size == A.size
  i = UOp.range(A.size, 0)
  return B[i].store(A[i]+1).end(i).sink(arg=KernelInfo(name=f"custom_v_add_{A.size}"))

class TestSQTT(unittest.TestCase):
  def test_v_add(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.empty_like(a)
    c = Tensor.custom_kernel(b, a, fxn=custom_add_one_kernel)[0]
    print(c.numpy())

if __name__ == "__main__":
  unittest.main()
