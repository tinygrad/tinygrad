import unittest
from typing import Callable
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import KernelInfo, AxisType

def _kernel(tensors:list[Tensor], fxn:Callable, grad_fxn:Callable|None=None) -> list[Tensor]:
  return [Tensor(u) for u in UOp.custom_kernel(*[t.uop for t in tensors], fxn=fxn, grad_fxn=grad_fxn)]

# **** kernels ****

def custom_arange_kernel(C:UOp):
  i = UOp.range(C.size, 0)
  return C[i].store(i.cast(C.dtype.base)).end(i).sink(arg=KernelInfo(name=f"custom_arange_{C.size}"))

def custom_add_one_kernel(B:UOp, A:UOp):
  assert B.size == A.size
  i = UOp.range(A.size, 0)
  return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.size}"))

def custom_elementwise_add_kernel(C:UOp, A:UOp, B:UOp):
  i = UOp.range(C.size, 0)
  return C[i].store(A[i]+B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.size}")).simplify()

def custom_elementwise_addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp):
  assert C.size == D.size
  i = UOp.range(C.size, 0)
  store_c = C[i].store(A[i]+B[i])
  store_d = D[i].store(A[i]*B[i])
  return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name=f"custom_addmul_kernel_{C.size}")).simplify()

def custom_gemm(C:UOp, A:UOp, B:UOp):
  assert A.shape[1] == B.shape[0]
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
  C = C[i, j].set(0.0)
  C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
  prog = C.end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

# **** backward callbacks ****

def backward_gemm(gradient:UOp, k:UOp) -> tuple[UOp, UOp]:
  out, a, b = k.src
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)

def backward_gemm_custom(gradient:UOp, k:UOp) -> tuple[UOp, UOp]:
  out, a, b = k.src
  grad_a = _kernel([Tensor.empty_like(Tensor(a)), Tensor(gradient), Tensor(b).T], fxn=custom_gemm)[0].uop
  grad_b = _kernel([Tensor.empty_like(Tensor(b)), Tensor(a).T, Tensor(gradient)], fxn=custom_gemm)[0].uop
  return (None, grad_a, grad_b)

# **** tests ****

class TestCustomKernel(unittest.TestCase):
  def test_simple(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.ones(16, 16).contiguous()
    c = Tensor.empty(16, 16)

    c = _kernel([c,a,b], fxn=custom_elementwise_add_kernel)[0]

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

  def test_arange(self):
    ref = Tensor.arange(100)
    tst = Tensor.empty_like(ref)
    tst = _kernel([tst], custom_arange_kernel)[0]
    self.assertTrue((ref == tst).all().item())

  def test_noncontig(self):
    a = Tensor.ones(16, 16).contiguous()
    tst = Tensor.empty_like(a)
    b = a+1
    b_p1 = _kernel([tst, b], custom_add_one_kernel)[0]
    self.assertTrue((b_p1 == 3).all().item())

  def test_gemm(self):
    N = 16
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = Tensor.empty(N, N)

    tst = _kernel([c, a, b], custom_gemm)[0]
    err = (tst - (a@b)).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_gemm_backward_custom(self): self.test_gemm_backward(True)
  def test_gemm_backward(self, custom_backward_gemm=False):
    N = 4
    a_rand = Tensor.randn(N, 8)
    b_rand = Tensor.randn(8, N)
    Tensor.realize(a_rand, b_rand)

    a, b = Tensor(a_rand.numpy(), requires_grad=True), Tensor(b_rand.numpy(), requires_grad=True)
    c = Tensor.empty(N, N)
    tst = _kernel([c, a, b], custom_gemm, backward_gemm_custom if custom_backward_gemm else backward_gemm)[0]
    tst.sum().backward()
    grad_a, grad_b = a.grad, b.grad
    Tensor.realize(tst, grad_a, grad_b)

    a, b = Tensor(a_rand.numpy(), requires_grad=True), Tensor(b_rand.numpy(), requires_grad=True)
    ref = (a@b)
    ref.sum().backward()
    real_grad_a, real_grad_b = a.grad, b.grad
    Tensor.realize(ref, real_grad_a, real_grad_b)

    err = (tst - ref).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_a - real_grad_a).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_b - real_grad_b).square().max()
    self.assertLess(err.item(), 1e-6)

if __name__ == '__main__':
  unittest.main()
