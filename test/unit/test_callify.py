import unittest
from unittest.mock import patch
from tinygrad import Tensor, dtypes, function
from tinygrad.tensor import transform_to_call
from tinygrad.uop.ops import UOp, Ops, ParamArg
from tinygrad.uop.render import pyrender
from tinygrad.uop.spec import eval_pyrender

class TestCallify(unittest.TestCase):
  def test_no_buffer_creation_in_callify(self):
    x = Tensor([1., 2.]).realize()
    for precompile in (False, True):
      @function(precompile=precompile)
      def f(x): return x + 1
      called = f(x)
      roots = ((x + 2).uop.materialize(), called.uop, x.clone().uop)
      with patch.object(UOp, "new_buffer", side_effect=AssertionError("callify created storage")), \
           patch.object(UOp, "empty_like", side_effect=AssertionError("callify replaced storage")), \
           patch.object(UOp, "bind_buffer", side_effect=AssertionError("callify bound storage")):
        call, mapped = transform_to_call(UOp.sink(*roots))
      self.assertIs(mapped[called.uop].storage_base, called.uop.storage_base)
      self.assertIn(called.uop.storage_base, call.src[1:])

  def test_unbound_store_binds_original_declaration(self):
    buf = UOp(Ops.BUFFER, arg=ParamArg(next(UOp.unique_num), dtypes.float32, size=2, device="CPU"))
    alias = Tensor(buf)
    t = Tensor(buf.after(buf.store(buf.const_like(7.))))
    t.callify().callify().realize()
    self.assertEqual(t.uop.storage_base.arg.slot, buf.arg.slot)
    self.assertFalse(t.uop.storage_base.is_unbound)
    self.assertIs(alias.uop.buffer, t.uop.buffer)
    self.assertEqual(t.tolist(), [7., 7.])
    self.assertEqual(t.tolist(), [7., 7.])

  def test_symbolic_view_keeps_bindings(self):
    start, size = UOp.variable("start", 0, 4).bind(2), UOp.variable("size", 1, 4).bind(3)
    t = Tensor.arange(8).float().realize()[start:start+size].clone()
    shape = t.shape
    t.callify().realize()
    self.assertEqual(t.shape, shape)
    self.assertEqual(t[:3].tolist(), [2., 3., 4.])
    self.assertEqual(t[:3].tolist(), [2., 3., 4.])

  def test_effect_only_call_body(self):
    # An opaque tensor-level body needs no returned AFTERs to make its root stores execute.
    for shape in ((6,), (2, 3)):
      with self.subTest(shape=shape):
        x = Tensor.arange(6).float().reshape(shape).realize()
        a, b = Tensor.zeros(shape).contiguous().realize(), Tensor.zeros(shape).contiguous().realize()
        a_buf, b_buf = a.uop.buffer, b.uop.buffer
        a, b = Tensor.custom_kernel(a, b, x, fxn=lambda a,b,x: UOp.sink(a.store(x+1), b.store(x*2)))[:2]
        a.realize(b)
        self.assertIs(a.uop.buffer, a_buf)
        self.assertIs(b.uop.buffer, b_buf)
        self.assertEqual(a.flatten().tolist(), [1., 2., 3., 4., 5., 6.])
        self.assertEqual(b.flatten().tolist(), [0., 2., 4., 6., 8., 10.])

  def test_effect_only_slice_store(self):
    x = Tensor.zeros(4, 4).contiguous().realize()
    y = Tensor.ones(2, 2).contiguous().realize()
    out = Tensor.custom_kernel(x, y, fxn=lambda x,y: x.shrink(((1, 3), (1, 3))).store(y).sink())[0]
    self.assertEqual(out.tolist(), [[0., 0., 0., 0.], [0., 1., 1., 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])

  def test_empty_declaration_binds(self):
    buf = UOp(Ops.BUFFER, arg=ParamArg(next(UOp.unique_num), dtypes.float32, size=2, device="CPU"))
    t = Tensor(buf).realize()
    self.assertEqual(t.uop.arg.slot, buf.arg.slot)
    self.assertFalse(t.uop.is_unbound)

  def test_declaration_pyrender(self):
    for size in (None, 2):
      buf = UOp(Ops.BUFFER, arg=ParamArg(next(UOp.unique_num), dtypes.float32, size=size, device="CPU"))
      self.assertIs(eval_pyrender(pyrender(buf)), buf)

  def test_scalar_declaration_binds(self):
    buf = UOp(Ops.BUFFER, arg=ParamArg(next(UOp.unique_num), dtypes.float32, device="CPU"))
    t = Tensor(buf.after(buf.store(buf.const_like(7.)))).realize()
    self.assertEqual(t.shape, ())
    self.assertEqual(t.uop.storage_base.arg.slot, buf.arg.slot)
    self.assertEqual(t.uop.buffer.size, 1)
    self.assertEqual(t.item(), 7.)

  def test_call_output_identity_and_cache(self):
    for precompile in (False, True):
      @function(precompile=precompile)
      def f(x): return x + 1, x * 2
      x = Tensor([1., 2.]).realize()
      a, b = f(x)
      decls = (a.uop.storage_base, b.uop.storage_base)
      a.callify(b).realize(b)
      self.assertEqual((a.uop.storage_base.arg.slot, b.uop.storage_base.arg.slot), tuple(d.arg.slot for d in decls))
      self.assertEqual(a.tolist(), [2., 3.])
      self.assertEqual(b.tolist(), [2., 4.])
      c, d = f(x)
      c.realize(d)
      self.assertIsNot(a.uop.buffer, c.uop.buffer)
      self.assertIsNot(b.uop.buffer, d.uop.buffer)
      self.assertEqual(c.tolist(), [2., 3.])
      self.assertEqual(d.tolist(), [2., 4.])

  def test_call_read_materializes_declared_output(self):
    for precompile in (False, True):
      @function(precompile=precompile)
      def f(x): return x + 1
      x = Tensor([1., 2.]).realize()
      y = f(x)
      slot = y.uop.storage_base.arg.slot
      self.assertEqual(y.tolist(), [2., 3.])
      self.assertEqual(y.uop.storage_base.arg.slot, slot)
      x.assign(0).realize()
      self.assertEqual(y.tolist(), [2., 3.])

  def test_output_aliases_share_materialization(self):
    x = Tensor([1., 2.]).realize() + 1
    y, z = x.contiguous_backward(), x.contiguous()
    x.realize(y, z, x)
    self.assertIs(x.uop.buffer, y.uop.buffer)
    self.assertIs(x.uop.buffer, z.uop.buffer)
    self.assertEqual(x.tolist(), [2., 3.])

  def test_output_slots_survive_binding(self):
    x = Tensor([1., 2.]).realize()
    p = x.uop.param_like(1)
    (out,) = UOp.call_with_outputs((p + 1,), x.uop, output_pos=(0,))
    c = out.src[1]
    bound = c.substitute({out.storage_base: out.storage_base.bind_buffer()})
    self.assertTrue(bound.is_value_call)
    self.assertFalse(bound.has_unbound_outputs)
    self.assertEqual(bound.arg.output_pos, (0,))
    self.assertEqual(Tensor(bound.call_outputs[0]).tolist(), [2., 3.])

  def test_output_scoping_preserves_storage_targets(self):
    x = Tensor([1., 2.]).realize()
    y = x.clone()
    x.assign(0)
    y.realize(x)
    self.assertEqual(y.tolist(), [1., 2.])
    self.assertEqual(x.tolist(), [0., 0.])
    self.assertIsNot(y.uop.buffer, x.uop.buffer)

  def test_shared_output_order(self):
    for reverse in (False, True):
      x = Tensor([1., 2.]).realize()
      a = (x + 1).sum()
      b = a * 2
      roots = (b, a) if reverse else (a, b)
      Tensor.realize(*roots)
      x.assign(0).realize()
      self.assertEqual(a.item(), 5.)
      self.assertEqual(b.item(), 10.)

  def test_transfers_own_storage(self):
    a = Tensor([1., 2.], device="CPU:0")
    self.assertIs(a.uop.op, Ops.AFTER)
    b = a.to("CPU:1")
    self.assertIs(b.uop.op, Ops.AFTER)
    self.assertIsNot(a.uop.storage_base, b.uop.storage_base)
    c = Tensor.empty(2, device="CPU:1").assign(b).realize()
    a.assign(0).realize()
    self.assertEqual(b.tolist(), [1., 2.])
    self.assertEqual(c.tolist(), [1., 2.])
    b.assign(3).realize()
    self.assertEqual(c.tolist(), [1., 2.])

  def test_virtual_output_does_not_allocate(self):
    t = Tensor(2.)
    with patch.object(UOp, "new_buffer", side_effect=AssertionError("virtual storage")):
      t.callify().realize()
    self.assertEqual(t.item(), 2.)

  def test_contiguous_through_wrapper_keeps_copy(self):
    for wrapper in ("detach", "contiguous_backward"):
      with self.subTest(wrapper=wrapper):
        x = Tensor([1., 2.]).realize()
        y = getattr(x.flip(0), wrapper)().contiguous().realize()
        x.assign(0).realize()
        self.assertEqual(y.tolist(), [2., 1.])

  def test_intermediate_contiguous_through_wrapper_is_view(self):
    x = Tensor([1., 2., 3., 4.], device="CPU").realize()
    y = x[:2].contiguous_backward().contiguous() + 1
    self.assertEqual(len(y.schedule_linear().src), 1)

  def test_basic(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    out = a + b
    out.callify()
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_const(self):
    out = Tensor(2.0) + Tensor(3.0)
    out.callify()
    self.assertEqual(out.item(), 5.0)

  def test_sum(self):
    out = Tensor.ones(16).contiguous().sum()
    out.callify()
    self.assertEqual(out.item(), 16.0)

  def test_multi_output(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    c = a + b
    d = a * b
    c.callify(d)
    self.assertListEqual(c.tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(d.tolist(), [4.0, 10.0, 18.0])

  def test_two_callify_independent(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    c = a + b
    c.callify()

    d = Tensor([10.,20,30])
    e = Tensor([1.,1,1])
    f = d - e
    f.callify()

    self.assertListEqual(c.tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(f.tolist(), [9.0, 19.0, 29.0])

  def test_two_callify_shared_input(self):
    a = Tensor([1.,2,3]).contiguous().realize()
    b = a + 1
    b.callify()
    c = a * 2
    c.callify()
    self.assertListEqual(b.tolist(), [2.0, 3.0, 4.0])
    self.assertListEqual(c.tolist(), [2.0, 4.0, 6.0])

  def test_chained_callify(self):
    a = Tensor([1.,2,3])
    b = a + 1
    b.callify()
    b.realize()
    c = b + 1
    c.callify()
    self.assertListEqual(c.tolist(), [3.0, 4.0, 5.0])

  def test_gemm(self):
    a = Tensor.ones(8, 8).contiguous()
    b = Tensor.eye(8).contiguous()
    out = a @ b
    out.callify()
    lst = out.tolist()
    for y in range(8):
      for x in range(8):
        self.assertEqual(lst[y][x], 1.0)

  def test_int_dtype(self):
    a = Tensor([1,2,3], dtype=dtypes.int)
    b = Tensor([4,5,6], dtype=dtypes.int)
    out = a + b
    out.callify()
    self.assertListEqual(out.tolist(), [5, 7, 9])

  def test_reduce(self):
    out = Tensor([1.,2,3,4]).sum()
    out.callify()
    self.assertEqual(out.item(), 10.0)

  def test_multiple_ops(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    out = (a + b) * (a - b)
    out.callify()
    self.assertListEqual(out.tolist(), [-15.0, -21.0, -27.0])

  def test_double_callify(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    out = a + b
    out.callify()
    out.callify()
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_double_callify_multi_output(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    c = a + b
    d = a * b
    c.callify(d)
    c.callify(d)
    self.assertListEqual(c.tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(d.tolist(), [4.0, 10.0, 18.0])

  def test_intermediate_clone_persists(self):
    x = (Tensor([1, 2, 3]).realize() + 1).clone()
    y = (x * 2).realize()
    self.assertTrue(x.uop.has_buffer_identity())
    self.assertEqual(x.tolist(), [2, 3, 4])
    self.assertEqual(y.tolist(), [4, 6, 8])

  def test_zero_size_cat_with_rng(self):
    # Empty outputs must not replay a pending RNG counter update.
    a = Tensor.rand(2, 2)
    b = Tensor.rand(2, 0)
    t = a.cat(b, dim=1).realize()
    self.assertEqual(t.shape, (2, 2))
    self.assertListEqual(t.tolist(), a.tolist())

if __name__ == "__main__":
  unittest.main()
