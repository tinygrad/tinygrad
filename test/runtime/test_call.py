import unittest
import numpy as np
from tinygrad import Tensor, function, Device
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, uopfunc
from tinygrad.engine.realize import run_linear, get_runtime
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.schedule import transform_to_call

def sched_key(t:Tensor): return transform_to_call(UOp.sink(t.uop)).body.key

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
    a = Tensor.ones(10, 10)
    b = Tensor.ones(10, 10)

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
    a = Tensor.ones(10, 10)
    b = Tensor.ones(10, 10)

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
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10) + 2  # avoid div by zero
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

class TestCallSchedule(unittest.TestCase):
  def test_precompile_slice_assign(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    a = Tensor.arange(8).float().realize()
    cache = Tensor.zeros(16)
    # the output must land at the slice offset, not at the start of the base buffer
    cache[4:12].assign(f(a)).realize()
    np.testing.assert_equal(cache.numpy(), np.concatenate([np.zeros(4), np.arange(8)*2+1, np.zeros(4)]).astype(np.float32))

  def test_precompile_slice_assign_2d(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + 1
    a = Tensor.arange(8).reshape(2, 4).float().realize()
    big = Tensor.zeros(4, 8)
    big[1:3, 2:6].assign(f(a)).realize()
    ref = np.zeros((4, 8), dtype=np.float32)
    ref[1:3, 2:6] = np.arange(8).reshape(2, 4) + 1
    np.testing.assert_equal(big.numpy(), ref)

  def test_precompile_full_buffer_assign(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    a = Tensor.arange(8).float().realize()
    cache = Tensor.zeros(8).realize()
    cache.assign(f(a)).realize()
    np.testing.assert_equal(cache.numpy(), np.arange(8)*2+1)

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

  def test_contiguous_call_output_realizes_aliases(self):
    def increment(x:UOp):
      i = UOp.range(x.shape[0], 0)
      return x[i].store(x[i].load() + 1).end(i).sink(arg=KernelInfo(name="increment"))

    for precompile in (False, True):
      for reshape in (False, True):
        with self.subTest(precompile=precompile, reshape=reshape):
          @function(precompile=precompile)
          def f(x:Tensor): return x.custom_kernel(fxn=increment)[0]
          state = Tensor([1., 2.]).realize()
          a = f(state)
          alias = a.reshape(1, 2)
          b = (alias if reshape else a).contiguous().realize()
          self.assertEqual(b.flatten().tolist(), [2., 3.])
          a.realize(alias)
          self.assertEqual(state.tolist(), [2., 3.])
          self.assertIs(a.uop.buffer, b.uop.buffer)
          self.assertIs(alias.uop.buffer, b.uop.buffer)
          b.assign([9., 10.]).realize()
          self.assertEqual(a.tolist(), [9., 10.])
          self.assertEqual(alias.tolist(), [[9., 10.]])

  def test_assign_call_output_to_input(self):
    for precompile in (False, True):
      with self.subTest(precompile=precompile):
        @function(precompile=precompile)
        def f(x:Tensor): return x.flip(0).contiguous()
        a = Tensor.arange(1024).clone().realize()
        a.assign(f(a)).realize()
        self.assertEqual(a.tolist(), list(reversed(range(1024))))

  def test_call_double_gemm(self):
    a = Tensor.randn(4, 8)
    b = Tensor.randn(8, 12)
    c = Tensor.randn(12, 16)
    ref = Tensor.randn(4, 16)
    Tensor.realize(a,b,c,ref)
    @function(precompile=True)
    def gemm(a:Tensor, b:Tensor, c:Tensor) -> Tensor: return (a@b)@c
    out = gemm(a,b,c)
    (out-ref).square().mean().backward()
    out.realize(a.grad, b.grad, c.grad)

  def test_precompile_symbolic_shape(self):
    """precompile with a symbolic-shaped input produces correct values and shape"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(5)]
    out = f(a)
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:5].numpy(), [2., 4., 6., 8., 10.])

  def test_precompile_symbolic_shape_contiguous(self):
    """precompile with a .contiguous() inside the function body on a symbolic-shaped input"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return (x * 2).contiguous() + 1
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(3)]
    out = f(a)
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:3].numpy(), [3., 5., 7.])

  def test_precompile_symbolic_shape_chain(self):
    """precompiled symbolic result used in downstream ops (tests AFTER has correct symbolic shape)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(4)]
    out = f(a) + 10  # downstream op on the precompiled result
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:4].numpy(), [12., 14., 16., 18.])

  def test_precompile_bind_arg(self):
    """precompile with a BIND (scalar variable) as a function argument"""
    @function(precompile=True)
    def f(x:Tensor, scale:UOp) -> Tensor: return x * scale
    v = UOp.variable("scale", 1, 100)
    a = Tensor([1., 2., 3.])
    out = f(a, v.bind(5))
    np.testing.assert_allclose(out.numpy(), [5., 10., 15.])

  def test_precompile_scoped_bind_arg(self):
    @function(precompile=True)
    def f(x:Tensor, scale:UOp) -> Tensor: return x * scale
    a = Tensor.ones(3)
    x = f(a, UOp.variable("scale_a", 1, 100).bind(2))
    y = f(a, UOp.variable("scale_b", 1, 100).bind(3))
    self.assertEqual(sched_key(x), sched_key(y))
    np.testing.assert_equal(x.numpy(), [2, 2, 2])
    np.testing.assert_equal(y.numpy(), [3, 3, 3])

  def test_precompile_nested_scope_collision(self):
    # a precompiled function body gets its own positional p{slot} params; they must not be renumbered when the call is
    # scheduled inside an enclosing realize with a different slot ordering. the store must use this call's Variable
    cache = Tensor.zeros(16)
    @function(precompile=True, allow_implicit=True)
    def store(x:Tensor, sp:UOp) -> Tensor:
      # update a cache at a symbolic offset, like an attention KV cache update
      return Tensor(cache.uop.after(cache[sp:sp+x.shape[0]].uop.store(x.uop)))[:sp+x.shape[0]].sum()
    sp_v, nt_v = UOp.variable("sp", 0, 8), UOp.variable("nt", 1, 8)
    t = Tensor.arange(16).float().realize()
    sp, nt = sp_v.bind(0), nt_v.bind(8)
    store(t[sp:sp+nt].clone().realize(), sp).realize()
    np.testing.assert_equal(cache.numpy()[:8], t[:8].numpy())
    np.testing.assert_equal(cache.numpy()[8:], np.zeros(8))

  def test_precompile_nested(self):
    for precompile in (False, True):
      for devices in (None, ("CPU:0", "CPU:1")):
        with self.subTest(precompile=precompile, devices=devices):
          @function(precompile=True, precompile_backward=True)
          def inner(x:Tensor): return x * 2, x + 3
          @function(precompile=precompile, precompile_backward=True)
          def outer(x:Tensor):
            a, b = inner(x)
            return a + b
          x = Tensor([1., 2., 3., 4.]).realize()
          if devices is not None: x = x.shard(devices, axis=0).realize()
          out = outer(x)
          for call in (u for u in out.uop.toposort(enter_calls=True) if u.op is Ops.CALL):
            self.assertFalse(any(b.op is Ops.BUFFER for b in call.body.toposort(enter_calls=True)))
          self.assertEqual(sched_key(out), sched_key(outer(x)))
          out.sum().backward()
          np.testing.assert_equal(out.numpy(), [6., 9., 12., 15.])
          np.testing.assert_equal(x.grad.numpy(), [3., 3., 3., 3.])

  def test_precompile_consumes_call_output(self):
    """a precompiled function consuming the output of a non-precompiled function"""
    @function
    def inner(x:Tensor) -> Tensor: return x * 2
    @function(precompile=True)
    def outer(x:Tensor) -> Tensor: return x + 1
    x = Tensor.arange(8).float().contiguous().realize()
    np.testing.assert_equal(outer(inner(x)).numpy(), np.arange(8, dtype=np.float32) * 2 + 1)

  def test_precompile_symbolic_2d(self):
    """precompile with symbolic shapes in 2D (tests debuf reshape with symbolic PARAM)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    sz = UOp.variable("sz", 1, 16)
    a = Tensor.arange(16*4).reshape(16, 4).float().clone()[:sz.bind(5)]
    out = f(a)
    # result shape should have the symbolic dim, not the max
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:5].numpy(), (np.arange(16*4).reshape(16, 4)[:5] * 2 + 1).astype(np.float32))

  def test_precompile_multi_sharded(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + 1
    devs = ("CPU:0", "CPU:1")
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    out = f(a) + 2
    np.testing.assert_allclose(out.numpy(), np.arange(8, dtype=np.float32).reshape(4, 2) + 3)

class TestArgOrder(unittest.TestCase):
  """outputs can appear anywhere in a call's srcs (output_pos): slots are src positions, nothing reorders"""
  def _dev(self, x): return x.device if isinstance(x.device, str) else (x.device or (Device.DEFAULT,))[0]
  def make_intersperse_call(self, x, precompile=False):
    # the output is at position 0, the input (param slot 1) at position 1 in the call's args
    val = UOp.param(1, x.dtype, x.shape, self._dev(x)).reshape(x.shape) * 2
    return UOp.call_with_outputs((val,), x.uop, name='t', output_pos=(0,), precompile=precompile)

  def test_intersperse_returned(self):
    x = Tensor.arange(3, dtype=dtypes.int).realize()
    outs = self.make_intersperse_call(x)
    out = Tensor(outs[0], device=x.device) + 1
    np.testing.assert_equal(out.numpy(), [1, 3, 5])

  def test_outputs_arbitrary_order(self):
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    x.requires_grad = True
    y.requires_grad = True
    x, y = x.realize(), y.realize()
    dev = self._dev(x)
    # args (out0, in0, out1, in1): outputs at positions 0 and 2, input params slotted at their final positions 1 and 3
    p1, p3 = UOp.param(1, x.dtype, x.shape, dev), UOp.param(3, y.dtype, y.shape, dev)
    outs = UOp.call_with_outputs((p1.reshape(x.shape) * 2, p3.reshape(y.shape) + p1.reshape(y.shape)), x.uop, y.uop,
                                 output_pos=(0, 2))
    np.testing.assert_equal(Tensor(outs[0]).numpy(), [2, 4, 6])
    np.testing.assert_equal(Tensor(outs[1]).numpy(), [5, 7, 9])
    # the auto gradient path (no grad_fxn) resolves outputs and gradients positionally at any position
    (Tensor(outs[0]).sum() + Tensor(outs[1]).sum()).backward()
    np.testing.assert_equal(x.grad.numpy(), [3, 3, 3])
    np.testing.assert_equal(y.grad.numpy(), [1, 1, 1])

  def test_intersperse_returned_gradient(self):
    x = Tensor([1.0, 2.0, 3.0]).realize()
    x.requires_grad = True
    p1 = UOp.param(1, dtypes.float, x.shape, self._dev(x))
    val = p1.reshape(x.shape) * p1.reshape(x.shape)
    outs = UOp.call_with_outputs((val,), x.uop, name='t', output_pos=(0,))
    y = Tensor(outs[0], device=x.device)
    y.sum().backward()
    np.testing.assert_equal(x.grad.numpy(), [2, 4, 6])

class TestCallMultiSharded(unittest.TestCase):
  # TODO: multi-output + sharded needs per-device CALL execution, which requires reworking how MULTI propagates through TUPLE bodies
  def test_tuple_sharded(self):
    """multi-output function with sharded input"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor): return (x + 1, x * 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref + 1)
    np.testing.assert_allclose(t2.numpy(), ref * 2)

  def test_tuple_sharded_precompile(self):
    """multi-output precompiled function with sharded input"""
    devs = ("CPU:0", "CPU:1")
    @function(precompile=True)
    def f(x:Tensor): return (x + 1, x * 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref + 1)
    np.testing.assert_allclose(t2.numpy(), ref * 2)

  def test_tuple_sharded_different_axis(self):
    """multi-output function where outputs have different sharding: one reduces on sharded axis, one doesn't"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor): return (x.sum(axis=0), x.sum(axis=1))
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref.sum(axis=0))
    np.testing.assert_allclose(t2.numpy(), ref.sum(axis=1))

  def test_tuple_sharded_different_ops(self):
    """multi-output function with different operations per output"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor, y:Tensor): return (x + y, x * y)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    b = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0) + 1
    t1, t2 = f(a, b)
    ref_a = np.arange(8, dtype=np.float32).reshape(4, 2)
    ref_b = ref_a + 1
    np.testing.assert_allclose(t1.numpy(), ref_a + ref_b)
    np.testing.assert_allclose(t2.numpy(), ref_a * ref_b)

  def test_tuple_sharded_mixed_use(self):
    """multi-output sharded results used in further computation"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor): return (x + 1, x * 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    out = (t1 + t2).sum()
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(out.numpy(), ((ref + 1) + (ref * 2)).sum())

  def test_tuple_sharded_outputs_different_axis(self):
    """multi-output function where the two outputs are sharded on different axes"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor, y:Tensor): return (x + 1, y + 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    b = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=1)
    t1, t2 = f(a, b)
    ref_a = np.arange(8, dtype=np.float32).reshape(4, 2)
    ref_b = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref_a + 1)
    np.testing.assert_allclose(t2.numpy(), ref_b + 2)

  def test_call_reduce_sharded(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    Tensor.realize(a)
    c = Tensor.call(a, fxn=a.as_param(0).sum(axis=0))
    np.testing.assert_equal(c.numpy(), 10 * np.ones(10))

  def test_call_reduce_sharded_mixed_args(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    b = Tensor.ones(10).shard(devs, axis=None)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0).sum(axis=0) + b.as_param(1))
    np.testing.assert_equal(c.numpy(), 11 * np.ones(10))

  def test_call_reduce_sharded_backward(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.randn(10, 10).shard(devs, axis=0)
    b = Tensor.randn(10, 10).shard(devs, axis=0)
    Tensor.realize(a, b)

    def grad_fxn(grad, call):
      a_arg, b_arg = call.src[1], call.src[2]
      return (grad.expand(a_arg.shape) * b_arg, grad.expand(b_arg.shape) * a_arg)

    body = (a.as_param(0) * b.as_param(1)).sum(axis=0)
    c = Tensor.call(a, b, fxn=body, grad_fxn=grad_fxn)
    c.sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), b.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), a.numpy(), rtol=1e-5)

  def test_symbolic_reshape_shard_axis(self):
    toks = UOp.variable("toks", 1, 2).bind(2)
    devs = ("CPU:0", "CPU:1")
    x = Tensor(np.arange(16, dtype=np.float32).reshape(1, 2, 8)).shard(devs, axis=2).realize()
    @function
    def f(x:Tensor) -> Tensor: return x.reshape(1, x.shape[1], 2, 4)
    out = f(x[:, :toks]).realize()
    self.assertEqual(out.uop.axis, 2)
    np.testing.assert_equal(out[:1, :2].to(devs[0]).numpy(), np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4))

@unittest.skipUnless((Device.DEFAULT == "CPU" and isinstance(Device[Device.DEFAULT].renderer, (CStyleLanguage, LLVMRenderer)))
                     or Device.DEFAULT == "AMD" and isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage),
                     "calls in kernel not implemented in this renderer")
class TestCallInKernel(unittest.TestCase):
  def test_simple(self):
    N = getenv("N", 4)

    @uopfunc
    def add(out:UOp, x:UOp) -> UOp:
      i = UOp.range(x.numel(), 0)
      return out[i].store(x[i].load() + 1).end(i).sink()

    @uopfunc
    def sum_fn(out:UOp, x:UOp) -> UOp:
      i = UOp.range(x.numel(), 0)
      return out[0].store(x[i].load().reduce(i, arg=Ops.ADD)).sink()

    def main(out:UOp, x:UOp) -> UOp:
      tmp0, tmp1 = (UOp.alloc_like(x, addrspace=AddrSpace.REG) for _ in range(2))
      call0 = add(tmp0, x)
      call1 = add(tmp1, tmp0.after(call0))
      return sum_fn(out, tmp1.after(call1)).sink(arg=KernelInfo(name="test_simple"))

    x = Tensor.arange(N).clone().realize()
    out = Tensor.custom_kernel(Tensor.empty_like(x), x, fxn=lambda out,x: add(out, x).sink(arg=KernelInfo(name="add_kernel")))[0]
    out = Tensor.custom_kernel(Tensor.empty(1, dtype=x.dtype), out,
                               fxn=lambda out,x: sum_fn(out, x).sink(arg=KernelInfo(name="sum_kernel")))[0]
    self.assertEqual(out.tolist(), [N*(N+1)//2])
    out2 = Tensor.custom_kernel(Tensor.empty(1, dtype=x.dtype), x, fxn=main)[0]
    self.assertEqual(out2.tolist(), [N*(N+3)//2])

  def test_reduce_over_call_keeps_loop_order(self):
    # a call can hide stores, so the reduce loops a, k, b in order, and the digits of S are the A[a*2+b] it visits
    @uopfunc
    def update(S:UOp, A:UOp) -> UOp: return S[0].store(S[0].load()*10+A[0].load()).sink()
    def kernel(O:UOp, S:UOp, A:UOp) -> UOp:
      a, k, b = [UOp.range(2, n, AxisType.LOOP) for n in range(3)]
      return O[0].store(A[a*2+b].after(update(S.after(k), A[a*2+b])).reduce(a, k, b, arg=Ops.ADD)).sink(arg=KernelInfo(opts_to_apply=()))
    o, s = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), Tensor.zeros(1, dtype=dtypes.int).contiguous(),
                                Tensor([1, 2, 3, 4], dtype=dtypes.int), fxn=kernel)[:2]
    Tensor.realize(o, s)
    self.assertEqual(s.item(), 12123434)

  def test_nested_call(self):
    @uopfunc
    def incr(out:UOp, A:UOp):
      i = UOp.range(A.shape[0], 0)
      return out[i].store(A[i].load()+1).end(i).sink()

    @uopfunc
    def square(out:UOp, A:UOp):
      tmp0, tmp1 = (UOp.alloc_like(A, addrspace=AddrSpace.REG) for _ in range(2))
      call0 = incr(tmp0, A)
      call1 = incr(tmp1, A.after(call0))
      assert call0.body is call1.body
      i = UOp.range(A.shape[0], 1)
      return out[i].store(tmp1.after(call1)[i].load()**2).end(i).sink()

    def kernel(C:UOp, A:UOp):
      tmp = UOp.alloc_like(A, addrspace=AddrSpace.REG)
      call = square(tmp, A)
      i = UOp.range(A.shape[0], 2)
      return C[i].store(tmp.after(call)[i].load()+10).end(i).sink(arg=KernelInfo(name="nested_calls"))

    a = Tensor.arange(4).clone().realize()
    out = Tensor.custom_kernel(Tensor.empty_like(a), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [11, 14, 19, 26])

  def test_call_loop_mini(self, opts=()):
    @uopfunc
    def mul(p:UOp, q:UOp): return p[0].store(q[0].load()*3).sink()

    def kernel(C:UOp, A:UOp):
      i = UOp.range(4, 0)
      call = mul(C[i], A[i])
      return call.end(i).sink(arg=KernelInfo(name="call_loop_mini", opts_to_apply=opts))
    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [3, 6, 9, 12])

  def test_call_scalar_arg(self):
    @uopfunc
    def add(out:UOp, src:UOp, val:UOp) -> UOp:
      return out[0].store(src[0].load()+val).sink()

    def kernel(out:UOp, src:UOp) -> UOp:
      i = UOp.range(4, 0)
      return add(out[i], src[i], i).end(i).sink(arg=KernelInfo(name="call_scalar_arg", opts_to_apply=()))

    x = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    self.assertEqual(Tensor.empty_like(x).custom_kernel(x, fxn=kernel)[0].tolist(), [1, 3, 5, 7])

  def test_call_with_gated_store(self):
    @uopfunc
    def gated_store(out:UOp, idx:UOp):
      i = idx[0].load()
      return out[i.valid((i >= 0) & (i < out.shape[0]))].store(7).sink()

    def kernel(C:UOp, A:UOp):
      tmp = UOp.alloc_like(C, addrspace=AddrSpace.REG)
      i = UOp.range(C.shape[0], 0)
      init = tmp[i].store(0).end(i)
      call = gated_store(tmp.after(init), A)
      j = UOp.range(C.shape[0], 1)
      return C[j].store(tmp.after(call)[j].load() + 1).end(j).sink(arg=KernelInfo(name="call_with_gated_store", opts_to_apply=()))

    a = Tensor([2], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [1, 1, 8, 1])

  def test_call_body_range_is_not_ours(self):
    @uopfunc
    def mul(out:UOp, A:UOp):
      k = UOp.range(4, 0, AxisType.LOOP)
      return out[k].store(A[k].load()*3).end(k).sink()

    def kernel(C:UOp, A:UOp):
      m, i, call = UOp.range(3, 1, AxisType.LOOP), UOp.range(4, 0, AxisType.LOOP), mul(C, A)
      self.assertIn(i, call.body.toposort())
      end_m = C.after(call)[m].store(A[m].load()).end(m)
      return C.after(end_m)[i].store(A[i].load()+1).end(i).sink(arg=KernelInfo(name="call_body_range", opts_to_apply=()))

    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [2, 3, 4, 5])

  @unittest.expectedFailure
  def test_call_loop_mini_opts(self): self.test_call_loop_mini(opts=None)

  @unittest.expectedFailure
  def test_call_loop_split(self): self.test_call_loop_mini((Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),))

  def test_call_loop_pad(self):
    @uopfunc
    def add(p:UOp, q:UOp): return p[0].store(p[0].load()+q[0].load()).sink()

    def kernel(C:UOp, A:UOp):
      i = UOp.range(3, 0)
      call = add(C[i], A[i])
      return call.end(i).sink(arg=KernelInfo(name="call_loop_padto", opts_to_apply=(Opt(OptOps.PADTO, 0, 4),)))

    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    # the CALL body can't be gated, so padding i is refused
    with self.assertRaises(KernelOptError): Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).clone().realize(), a, fxn=kernel)[0].realize()

  def test_uopfunc(self):
    @uopfunc
    def axpy(out:UOp, x:UOp, a:int) -> UOp: # a is a trace-time constant: two bodies, two C functions
      i = UOp.range(x.shape[0], 0, AxisType.LOOP)
      return out[i].store(out[i].load() + x[i].load() * a).end(i).sink()

    @uopfunc
    def triple(out:UOp, x:UOp): return axpy(out, x, 3).sink()

    def kernel(C:UOp, A:UOp) -> UOp:
      C = C.after(triple(C.after(axpy(C, A, 2)), A))
      i = UOp.range(A.shape[0], 1, AxisType.LOOP) # a loop after the calls
      return C[i].store(C[i].load() + 1).end(i).sink(arg=KernelInfo(name="uopfunc", opts_to_apply=()))

    a = Tensor([1, 2, 3, 4], dtype=dtypes.int).realize()
    out = Tensor.custom_kernel(Tensor.zeros(4, dtype=dtypes.int).contiguous().realize(), a, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [6, 11, 16, 21])

  def test_copy_offset_slices(self):
    @uopfunc
    def copy_block(dst:UOp, src:UOp) -> UOp:
      i = UOp.range(4, 0, AxisType.UPCAST)
      return dst[i].store(src[i].load()).end(i).sink()

    def kernel(dst:UOp, src:UOp) -> UOp:
      return copy_block(dst[1:5], src[1:5]).sink(arg=KernelInfo(opts_to_apply=()))

    src = Tensor([0., 1., 2., 3., 4.]).realize()
    dst = Tensor([-1.] * 6).realize()
    out = dst.custom_kernel(src, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [-1., 1., 2., 3., 4., -1.])

  @unittest.skipIf(Device.DEFAULT == "CPU", "test needs to copy from CPU to another device")
  def test_call_to_copy(self):
    @uopfunc
    def copy(out:UOp, A:UOp):
      i = UOp.range(out.shape[0], 0, AxisType.LOOP)
      return out[i].store(A[i].load()).end(i).sink()

    def forward(out1:UOp, out2:UOp, A:UOp):
      call = copy(out1, A)
      i = UOp.range(out2.shape[0], 0, AxisType.LOOP)
      inline = out2[i].store(A[i].load()).end(i).sink()
      return UOp.group(call, inline).sink(arg=KernelInfo(name="forward", opts_to_apply=()))

    x = Tensor([1, 2, 3, 4], device="CPU").realize().to(Device.DEFAULT)
    y, z = x.empty_like(), x.empty_like()
    y, z, _ = Tensor.custom_kernel(y, z, x, fxn=forward)
    y, z = y.to("CPU"), z.to("CPU")
    Tensor.realize(y, z)

    self.assertEqual(y.tolist(), z.tolist())
    self.assertEqual(x.tolist(), z.tolist())

  @unittest.skipIf(Device[Device.DEFAULT].renderer.inline_prefix is not None, "backend inlines calls instead of linking")
  def test_linker(self):
    @uopfunc
    def plus2(out:UOp, A:UOp): return out[0].store(A[0].load()+2).sink()

    @uopfunc
    def copy(out:UOp, A:UOp): return out[0].store(A[0].load()).sink()

    def kernel(A:UOp, extra=False):
      A = A.flatten()
      i = UOp.range(A.shape[0], 0)
      tmp = UOp.alloc_like(A, addrspace=AddrSpace.REG)
      if extra: tmp = tmp.after(copy(tmp[i], A[i]))
      return A[i].store(tmp[i].after(plus2(tmp[i], A[i]))).end(i).sink(arg=KernelInfo(name="test_kernel", opts_to_apply=()))

    x = Tensor.custom_kernel(Tensor.arange(4).clone().realize(), fxn=kernel)[0]
    run_linear(lin:=x.schedule_linear())

    prg = to_program(lin.src[0].body, Device[Device.DEFAULT].renderer)
    assert "+2" not in prg.src[2].arg, f"found callee body in caller kernel {prg.src[2].arg}"
    call = next(u for u in prg.src[1].src if u.op is Ops.CALL)
    first_binary = call.body.src[3]
    call_runtime = get_runtime(Device.DEFAULT, call.body)
    self.assertListEqual(x.tolist(), Tensor.arange(4).add(2).tolist())

    # launch a different caller with the same callee
    x = Tensor.custom_kernel(Tensor.arange(5).clone().realize(), fxn=lambda A: kernel(A, extra=True))[0]
    run_linear(lin:=x.schedule_linear())
    prg2 = to_program(lin.src[0].body, Device[Device.DEFAULT].renderer).src[1]
    self.assertIs(call.body, next(u.body for u in prg2.src if u.op is Ops.CALL and u.arg.name == "plus2"))
    self.assertIs(first_binary, next(u.body.src[3] for u in prg2.src if u.op is Ops.CALL and u.arg.name == "plus2"))
    self.assertIs(call_runtime, get_runtime(Device.DEFAULT, call.body))
    self.assertListEqual(x.tolist(), Tensor.arange(5).add(2).tolist())

if __name__ == '__main__':
  unittest.main()
