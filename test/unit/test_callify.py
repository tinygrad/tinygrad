import unittest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.tensor import transform_to_call

class TestCallify(unittest.TestCase):
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

  def test_only_replace_inputs(self):
    x = Tensor.empty(4)
    body = UOp.sink((x.uop + 1).contiguous().copy_to_device("CPU:1"))
    call = transform_to_call(body)
    self.assertEqual(call.src[1:], (x.uop,))
    self.assertIs(call.src[0], body.substitute({x.uop: x.uop.param_like(0)}))

  def test_existing_params_do_not_alias_buffers(self):
    x = Tensor.empty(4)
    param = UOp.param(0, x.dtype, x.shape, device=x.device)
    body = UOp.sink(x.uop + param)
    call = transform_to_call(body)
    self.assertEqual(set(call.src[1:]), {x.uop, param})
    params = [u for u in call.src[0].toposort() if u.op is Ops.PARAM]
    self.assertEqual({u.arg.slot for u in params}, {0, 1})
    self.assertIs(call.src[0].substitute({u: call.src[1+u.arg.slot] for u in params}, walk=True), body)

  def test_scalar_param_binding_survives_renumbering(self):
    from tinygrad.schedule import create_linear_with_vars
    from tinygrad.engine.realize import run_linear
    x = Tensor([1, 2, 3]).realize()
    out = Tensor.empty_like(x)
    binding = UOp.variable("amount", 1, 10, dtypes.int).bind(4)
    param = binding.param_like(7)
    call = transform_to_call(UOp.sink(out.uop.after(out.uop.store(x.uop + param))))
    call = call.replace(src=(call.src[0], *(binding if arg is param else arg for arg in call.src[1:])))
    run_linear(*create_linear_with_vars(call))
    self.assertEqual(out.tolist(), [5, 6, 7])

  def test_nested_params_keep_their_scope(self):
    x = Tensor.empty(4)
    param = UOp.param(7, x.dtype, x.shape, device=x.device)
    nested_body = UOp.sink(param + 1)
    nested = nested_body.call(*([x.uop] * 8))
    call = transform_to_call(UOp.sink(x.uop + param, nested))
    self.assertIs(call.src[0].src[1].src[0], nested_body)
    self.assertEqual(set(call.src[1:]), {x.uop, param})

  def test_unbound_renumbering_preserves_distinct_outputs(self):
    def output(): return UOp.call_with_outputs((Tensor(1., dtype=dtypes.float, device="CPU").uop,))[0]
    canonical = transform_to_call(UOp.sink(output())).src[0].src[0]
    body = UOp.sink(canonical, output())
    call = transform_to_call(body)
    self.assertEqual(len([u for u in call.src[0].toposort() if u.is_unbound]), 2)
    self.assertIs(transform_to_call(call.src[0]).src[0], call.src[0])

  def test_intermediate_contiguous_stays_a_value(self):
    x = (Tensor([1, 2, 3]).realize() + 1).contiguous()
    original = x.uop
    y = (x * 2).realize()
    self.assertIs(x.uop, original)
    self.assertIs(x.uop.op, Ops.CONTIGUOUS)
    self.assertEqual(y.tolist(), [4, 6, 8])

  def test_intermediate_clone_persists(self):
    x = (Tensor([1, 2, 3]).realize() + 1).clone()
    y = (x * 2).realize()
    self.assertTrue(x.uop.has_buffer_identity())
    self.assertEqual(x.tolist(), [2, 3, 4])
    self.assertEqual(y.tolist(), [4, 6, 8])

  def test_creation_copy_has_storage(self):
    x = Tensor([1, 2, 3], device="PYTHON").to("CPU")
    self.assertTrue(x.uop.has_buffer_identity(after_ok=True))
    y = Tensor.empty(3, dtype=dtypes.int, device=x.device).assign(x).realize()
    y.assign(0).realize()
    self.assertEqual(x.tolist(), [1, 2, 3])

  def test_zero_size_cat_with_rng(self):
    # Empty outputs must not replay a pending RNG counter update.
    a = Tensor.rand(2, 2)
    b = Tensor.rand(2, 0)
    t = a.cat(b, dim=1).realize()
    self.assertEqual(t.shape, (2, 2))
    self.assertListEqual(t.tolist(), a.tolist())

if __name__ == "__main__":
  unittest.main()
