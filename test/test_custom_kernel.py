import unittest
from typing import Callable
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.schedule.rangeify import Kernel

def custom_elementwise_add_kernel(C:UOp, A:UOp, B:UOp):
  i = UOp.range(C.size, 0)
  return C[i].store(A[i]+B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.size}")).simplify()

def custom_elementwise_add_kernel_backward(C:UOp, A:UOp, B:UOp):
  i = UOp.range(C.size, 0)
  store_a = A[i].store(C[i])
  store_b = B[i].store(C[i])
  return UOp.group(store_a, store_b).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.size}_backward")).simplify()

def custom_elementwise_addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp):
  assert C.size == D.size
  i = UOp.range(C.size, 0)
  store_c = C[i].store(A[i]+B[i])
  store_d = D[i].store(A[i]*B[i])
  return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name=f"custom_addmul_kernel_{C.size}")).simplify()

def _kernel(tensors:list[Tensor], fxn:Callable, backward_fxn:Callable|None=None) -> list[Tensor]:
  placeholders = [UOp.placeholder_like(t.uop, slot=i) for i,t in enumerate(tensors)]
  ast = fxn(*placeholders)
  backward_ast = backward_fxn(*placeholders) if backward_fxn else None
  kernel = UOp(Ops.KERNEL, src=tuple(x.uop.base for x in tensors), arg=Kernel(ast, backward_ast=backward_ast))
  return [Tensor(t.uop.after(kernel)) for t in tensors]

class TestCustomKernel(unittest.TestCase):
  def test_simple(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.ones(16, 16).contiguous()
    c = Tensor.empty(16, 16)

    c = _kernel([c,a,b], custom_elementwise_add_kernel)[0]

    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_multioutput(self):
    a = Tensor.full((16, 16), 3.).contiguous()
    b = Tensor.full((16, 16), 3.).contiguous()
    c = Tensor.empty(16, 16)
    d = Tensor.empty(16, 16)

    c,d = _kernel([c,d,a,b], custom_elementwise_addmul_kernel)[:2]
    Tensor.realize(c,d)

    assert all(x == 6 for x in c.flatten().tolist()), "all 6"
    assert all(x == 9 for x in d.flatten().tolist()), "all 9"

  def test_simple_backward(self):
    a = Tensor.ones(16, 16).contiguous().requires_grad_()
    b = Tensor.ones(16, 16).contiguous().requires_grad_()
    c = Tensor.empty(16, 16)

    c = _kernel([c,a,b], custom_elementwise_add_kernel, custom_elementwise_add_kernel_backward)[0]
    c.sum().backward()

    Tensor.realize(c, a.grad, b.grad)

if __name__ == '__main__':
  unittest.main()
