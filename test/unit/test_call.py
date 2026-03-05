import unittest
import numpy as np
from tinygrad import Tensor, function
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg

class TestCall(unittest.TestCase):
  def test_call_plus(self):
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10)
    Tensor.realize(a,b)

    # we define a plus function
    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))

    c = Tensor.call(a, b, fxn=plus_fxn)
    np.testing.assert_equal(c.numpy(), (a+b).numpy())

  def test_call_plus_backward(self):
    a = Tensor.ones(10, 10, requires_grad=True)
    b = Tensor.ones(10, 10, requires_grad=True)

    (a+b).mean().backward()
    gt_a_grad = a.grad.numpy()
    gt_b_grad = b.grad.numpy()
    a.grad, b.grad = None, None

    # this is the gradient for +
    def grad_fxn(grad:UOp, call:UOp): return (grad, grad)

    # we define a plus function
    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))
    c = Tensor.call(a, b, fxn=plus_fxn, grad_fxn=grad_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_plus_backward_auto(self):
    a = Tensor.ones(10, 10, requires_grad=True)
    b = Tensor.ones(10, 10, requires_grad=True)

    (a+b).mean().backward()
    gt_a_grad = a.grad.numpy()
    gt_b_grad = b.grad.numpy()
    a.grad, b.grad = None, None

    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))
    c = Tensor.call(a, b, fxn=plus_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_gemm(self):
    M, K, N = 4, 8, 4
    a = Tensor.randn(M, K)
    b = Tensor.randn(K, N)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0) @ b.as_param(1))
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-6)

  @unittest.skip("needs GEMM on mixins")
  def test_call_gemm_uop(self):
    M, K, N = 4, 8, 4
    a = Tensor.randn(M, K)
    b = Tensor.randn(K, N)
    Tensor.realize(a, b)

    # we define a gemm function
    x = UOp.param(0, dtypes.float, shape=(M, K))
    y = UOp.param(1, dtypes.float, shape=(K, N))
    c = Tensor.call(a, b, fxn=x@y)

    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-6)

  def test_call_complex_backward_auto(self):
    # complex chain: (a*b + a).exp2() * b.reciprocal() - tests mul, add, exp2, reciprocal, param reuse
    a = Tensor.randn(10, 10, requires_grad=True)
    b = Tensor.randn(10, 10, requires_grad=True) + 2  # avoid div by zero
    Tensor.realize(a, b)

    ((a*b + a).exp2() * b.reciprocal()).mean().backward()
    gt_a_grad, gt_b_grad = a.grad.numpy(), b.grad.numpy()
    a.grad, b.grad = None, None

    p0, p1 = UOp.param(0, dtypes.float, (10,10)), UOp.param(1, dtypes.float, (10,10))
    complex_fxn = (p0*p1 + p0).exp2() * p1.reciprocal()
    c = Tensor.call(a, b, fxn=complex_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_plus_sharded(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    b = Tensor.ones(10, 10).shard(devs, axis=0)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0) + b.as_param(1))
    np.testing.assert_equal(c.numpy(), 2 * np.ones((10, 10)))

class TestCallShape(unittest.TestCase):
  def _param_with_shape(self, slot, shape):
    return UOp(Ops.PARAM, dtypes.float, (shape_to_shape_arg(shape), UOp(Ops.NOOP)), arg=slot)

  def test_call_shape_int(self):
    # shape elements that are plain ints pass through unchanged
    p0 = UOp.param(0, dtypes.float, (4, 8))
    buf = UOp(Ops.BUFFER, dtypes.float.ptr(32), (), 32)
    call = p0.call(buf)
    self.assertEqual(call._shape, (4, 8))

  def test_call_shape_param_substitution(self):
    # a PARAM UOp appearing directly as a shape element is substituted with the corresponding arg
    p0 = UOp(Ops.PARAM, dtypes.index, (shape_to_shape_arg(()), UOp(Ops.NOOP)), arg=0)
    p1 = self._param_with_shape(1, (p0, 8))
    size_arg = UOp.const(dtypes.index, 5)
    buf = UOp(Ops.BUFFER, dtypes.float.ptr(40), (), 40)
    call = p1.call(size_arg, buf)
    # first dim should be substituted: PARAM(0) -> size_arg
    self.assertEqual(call._shape[0], size_arg)
    self.assertEqual(call._shape[1], 8)

  def test_call_shape_expr_substitution(self):
    # a shape element that is an expression containing PARAMs gets substituted
    p0 = UOp(Ops.PARAM, dtypes.index, (shape_to_shape_arg(()), UOp(Ops.NOOP)), arg=0)
    p0_times_2 = p0 * UOp.const(dtypes.index, 2)
    p1 = self._param_with_shape(1, (p0_times_2, 4))
    size_arg = UOp.const(dtypes.index, 5)
    buf = UOp(Ops.BUFFER, dtypes.float.ptr(40), (), 40)
    call = p1.call(size_arg, buf)
    # first dim should be const(5) * const(2)
    s = call._shape[0]
    self.assertIsInstance(s, UOp)
    self.assertEqual(s.op, Ops.MUL)
    self.assertEqual(call._shape[1], 4)

  def test_call_shape_no_param_passthrough(self):
    # a UOp shape element with no PARAMs passes through unchanged
    var = UOp(Ops.DEFINE_VAR, dtypes.index, (UOp.const(dtypes.index, 1), UOp.const(dtypes.index, 100)), arg='n')
    p0 = self._param_with_shape(0, (var, 8))
    buf = UOp(Ops.BUFFER, dtypes.float.ptr(800), (), 800)
    call = p0.call(buf)
    self.assertIs(call._shape[0], var)
    self.assertEqual(call._shape[1], 8)

class TestCallSchedule(unittest.TestCase):
  def test_reshape_precompile(self):
    a = Tensor.empty(4, 8).realize()
    a = a.reshape(4,4,2).assign(Tensor.empty(4,4,2)).reshape(8,4)
    @function(precompile=True)
    def s(x): return x.sum(axis=0)
    (s(a)*3).realize()

  def test_call_precompiled(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    (s(a)*3).realize()

  def test_double_call(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    s(s(a)).realize()

  def test_double_call_contiguous(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    s(s(a).contiguous()).realize()

  def test_call_double_gemm(self):
    a = Tensor.randn(4, 8, requires_grad=True)
    b = Tensor.randn(8, 12, requires_grad=True)
    c = Tensor.randn(12, 16, requires_grad=True)
    ref = Tensor.randn(4, 16)
    Tensor.realize(a,b,c,ref)
    @function(precompile=True)
    def gemm(a:Tensor, b:Tensor, c:Tensor) -> Tensor: return (a@b)@c
    out = gemm(a,b,c)
    (out-ref).square().mean().backward()
    out.realize(a.grad, b.grad, c.grad)

if __name__ == '__main__':
  unittest.main()
