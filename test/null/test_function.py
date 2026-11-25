import unittest
from tinygrad.function import function
from tinygrad import Tensor, GlobalCounters
from tinygrad.uop.ops import UOp, KernelInfo

class TestFunction(unittest.TestCase):
  def test_context_restored_on_exception(self):
    from tinygrad.helpers import ALLOW_DEVICE_USAGE, Context
    @function
    def f(a:Tensor) -> Tensor:
      self.assertFalse(ALLOW_DEVICE_USAGE)
      raise ValueError("error")
    with Context(ALLOW_DEVICE_USAGE=1):
      with self.assertRaises(ValueError): f(Tensor([1]))
      self.assertTrue(ALLOW_DEVICE_USAGE)

  def test_name(self):
    @function
    def f(a:Tensor) -> Tensor: return a + 1
    self.assertEqual(f.__name__, "f")

  def test_method_name(self):
    class Foo:
      @function
      def __call__(self, x:Tensor) -> Tensor: return x + 1

  def test_implicit_symbolic_var_lifted(self):
    from tinygrad import Variable
    from tinygrad.uop.ops import Ops
    v = Variable("n", 1, 8).bind(4)
    imp = (Tensor.ones(8).contiguous().realize()[:v] + 1).contiguous()
    @function(allow_implicit=True, precompile=True)
    def g(x:Tensor) -> Tensor: return x + imp.sum()
    call = next(x for x in g(Tensor.ones(1)).uop.toposort() if x.op is Ops.CALL)
    self.assertEqual([x for x in call.body.toposort() if x.is_variable and x.arg.slot == -1], [])

  def test_lower_call_inline_nested_name_collision(self):
    from tinygrad.dtype import dtypes
    from tinygrad.codegen import do_to_program
    from tinygrad.renderer import Target
    from tinygrad.renderer.cstyle import CStyleLanguage
    ren = CStyleLanguage(Target())
    ren.inline_prefix = "static inline"
    p0 = UOp.param(0, dtypes.float32, (1,))
    sub1 = UOp.sink(p0[0].store(p0[0].load() + UOp.const(1.0, dtypes.float32)), arg=KernelInfo("sub", optimized=True))
    sub2 = UOp.sink(p0[0].store(p0[0].load() + UOp.const(2.0, dtypes.float32)), arg=KernelInfo("sub", optimized=True))
    # helper calls sub1; outer1 calls sub2 (claims "sub") then helper (sub1 -> "sub_1"); outer2 calls only helper (sub1 -> "sub")
    helper = UOp.sink(sub1.call(p0), arg=KernelInfo("helper", optimized=True))
    outer1 = UOp.sink(sub2.call(p0), helper.call(p0), arg=KernelInfo("outer1", optimized=True))
    outer2 = UOp.sink(helper.call(p0), arg=KernelInfo("outer2", optimized=True))
    p1 = do_to_program(outer1, ren)
    p2 = do_to_program(outer2, ren)
    # outer2 must call "sub", not outer1's cached "sub_1"
    self.assertIn("sub(", p2.src[2].arg)
    self.assertNotIn("sub_1(", p2.src[2].arg)
    self.assertIn("sub_1(", p1.src[2].arg)

  def test_split_store_dependent_range_renumber(self):
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import Ops
    from tinygrad.codegen.rangeify import split_store
    buf = UOp.new_buffer("NULL", 16, dtypes.float32)
    r0 = UOp.range(4, 5)
    r1 = UOp.range(r0 + 1, 7)
    st = buf.index(r0 * 4 + r1).store(UOp.const(1.0, dtypes.float32)).end(r0, r1)
    call = split_store(st)
    assert call is not None
    ranges = [u for u in call.body.toposort() if u.op is Ops.RANGE]
    # inner range bound must reference the renumbered outer range, not the stale axis_id=5 range
    self.assertEqual({r.axis_id for r in ranges}, {(0,), (1,)})

  def test_custom_kernel_captured_buffer_lifted(self):
    from tinygrad.uop.ops import Ops
    w = Tensor([10., 20., 30., 40.]).contiguous().realize()
    def k(C:UOp, A:UOp) -> UOp:
      i = UOp.range(4, 0)
      return C[i].store(A[i] + w.uop[i]).end(i).sink(arg=KernelInfo(name="ck_closure"))
    out = Tensor.custom_kernel(Tensor.empty(4), Tensor([1., 2., 3., 4.]).contiguous().realize(), fxn=k)[0]
    call = next(x for x in out.uop.toposort() if x.op is Ops.CALL)
    self.assertEqual([x for x in call.body.toposort() if x.op is Ops.BUFFER], [])
    self.assertIn(w.uop.base, call.src[1:])

  def test_simplify_weak_nan_cast(self):
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import Ops
    u = UOp(Ops.CAST, (UOp.const(float("nan")),), dtypes.weakint)
    self.assertIs(u.simplify().op, Ops.CAST)

class TestFunctionMulti(unittest.TestCase):
  devices_2 = ("NULL:0", "NULL:1")

  def test_call_axis(self):
    @function
    def f(x:Tensor, w:Tensor) -> Tensor: return x @ w

    x = Tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]).shard(self.devices_2, axis=0)
    w = Tensor([[1.,2.],[3.,4.]]).shard(self.devices_2, axis=None)
    result = f(x, w)
    # CALL output should inherit axis=0 from the sharded input
    self.assertEqual(result.uop.axis, 0)
    # reduce on the sharded axis should remove it
    self.assertIsNone(result.sum().uop.axis)

class TestFunctionTuple(unittest.TestCase):
  def test_custom_kernel_inplace_output_is_implicit(self):
    # caller-owned storage must be captured, even before its Buffer is bound
    state = Tensor.empty(4)
    def inplace_add(C:UOp, A:UOp) -> UOp:
      i = UOp.range(A.shape[0], 0)
      return C[i].store(C[i].load() + A[i]).end(i).sink(arg=KernelInfo(name="inplace_add"))
    @function(precompile=True, allow_implicit=False)
    def f(a:Tensor): return Tensor.custom_kernel(state, a, fxn=inplace_add)[0]
    with self.assertRaisesRegex(RuntimeError, "implicit buffer"): f(Tensor([1., 2., 3., 4.]).contiguous().realize())

class TestFunctionGrad(unittest.TestCase):
  def test_function_grad_ops(self, precompile=False, precompile_backward=False):
    N = 64
    x = Tensor.ones(N,N).contiguous()
    w1 = Tensor.ones(N,N).contiguous()
    w2 = Tensor.ones(N,N).contiguous()
    w3 = Tensor.ones(N,N).contiguous()
    ref = Tensor.ones(N,N).contiguous()
    Tensor.realize(x, w1, w2, w3, ref)
    @function(precompile=precompile, precompile_backward=precompile_backward)
    def f(x, w1, w2, w3) -> tuple[Tensor, ...]:
      p1 = x@w1
      p2 = p1@w2
      p3 = p2@w3
      return p1, p2, p3, p3.contiguous()
    ret = f(x, w1, w2, w3)[-1]
    loss = (ret-ref).square().mean().backward()
    print("RESET")
    GlobalCounters.reset()
    loss.realize(w1.grad, w2.grad, w3.grad)
    print(GlobalCounters.global_ops, GlobalCounters.global_mem)
    self.assertLessEqual(GlobalCounters.global_ops, 5000000)
  def test_function_grad_ops_precompile(self): self.test_function_grad_ops(precompile=True)
  def test_function_grad_ops_precompile_backward(self):
    self.test_function_grad_ops(precompile=True, precompile_backward=True)

if __name__ == '__main__':
  unittest.main()
