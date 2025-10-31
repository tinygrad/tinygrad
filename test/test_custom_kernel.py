import unittest
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.schedule.rangeify import Kernel

def custom_elementwise_add_kernel(C:UOp, A:UOp, B:UOp):
  i = UOp.range(C.size, 0)
  return C[i].store(A[i]+B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.size}")).simplify()

class TestCustomKernel(unittest.TestCase):
  def test_simple(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.ones(16, 16).contiguous()
    c = Tensor.empty(16, 16)

    ast = custom_elementwise_add_kernel(*[UOp.placeholder_like(t.uop, slot=i) for i,t in enumerate([c,a,b])])
    kernel = UOp(Ops.KERNEL, src=(c.uop.base, a.uop.base, b.uop.base), arg=Kernel(ast))
    c_modded = Tensor(c.uop.after(kernel))
    out = c_modded.flatten().tolist()

    assert all(x == 2 for x in out), "all 2"

if __name__ == '__main__':
  unittest.main()
