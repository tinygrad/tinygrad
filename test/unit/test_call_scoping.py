import unittest

from tinygrad.dtype import dtypes
from tinygrad.gradient import call_gradient
from tinygrad.schedule.rangeify import resolve_call
from tinygrad.uop.ops import Ops, UOp

class TestCallScoping(unittest.TestCase):

  def test_resolve_call_scoping_duplicate_param_slots(self):
    # inner has PARAM(0), outer has PARAM(0) too
    # pre-fix exploded with "params not in order"
    inner_param = UOp.param(0, dtypes.int, (), "CPU")
    inner_call = (inner_param + 1).call(inner_param)
    outer_param = UOp.param(0, dtypes.int, (), "CPU")
    outer_call = (outer_param + inner_call).call(outer_param)
    resolved = resolve_call(outer_call)
    self.assertIsNotNone(resolved)

  def test_resolve_call_scoping_sequential_param_slots(self):
    # inner has PARAM(1), outer has PARAM(0)
    # pre-fix exploded with "expected 2 args, got 1"
    inner_param = UOp.param(1, dtypes.int, (), "CPU")
    inner_call = (inner_param + 1).call(inner_param)
    outer_param = UOp.param(0, dtypes.int, (), "CPU")
    outer_call = (outer_param + inner_call).call(outer_param)
    resolved = resolve_call(outer_call)
    self.assertIsNotNone(resolved)

  def test_resolve_call_only_binds_outer_params(self):
    # when we bind the outer param with a CONST,
    # the outer PARAM should disappear but the inner PARAM should remain
    # NOTE: inner uses slot 1 so ucache doesn't merge it with outer slot 0
    inner_param = UOp.param(0, dtypes.int, (), "CPU")
    inner_call = (inner_param + 1).call(inner_param)

    outer_param = UOp.param(0, dtypes.float, (), "CPU")
    outer_call = (outer_param + inner_call.cast(dtypes.float)).call(UOp.const(dtypes.float, 2, "CPU", ()))

    resolved = resolve_call(outer_call)
    self.assertIsNotNone(resolved)

    ts = set(resolved.toposort())
    self.assertNotIn(outer_param, ts)  # must be substituted away
    self.assertIn(inner_param, ts)     # still exists inside the nested call body

  def test_resolve_call_doesnt_capture_nested_params(self):
    dev = "CPU"
    inner_function = UOp.param(0, dtypes.float, (), dev)
    inner_call = inner_function.call(UOp.const(dtypes.float, 3, dev, ()))
    outer_param = UOp.param(0, dtypes.float, (), dev)
    fxn = UOp.sink(outer_param, inner_call)
    out = resolve_call(fxn.call(UOp.const(dtypes.float, 2, dev, ())))
    self.assertIsNotNone(out)

  def test_call_gradient_doesnt_capture_nested_params(self):
    dev = "CPU"
    inner_function = UOp.param(0, dtypes.float, (), dev)
    inner_call = inner_function.call(UOp.const(dtypes.float, 3, dev, ()))
    outer_param = UOp.param(0, dtypes.float, (), dev)
    fxn = outer_param.alu(Ops.ADD, inner_call)
    ctx = UOp.const(dtypes.float, 1, dev, ())
    grads = call_gradient(ctx, fxn.call(UOp.const(dtypes.float, 2, dev, ())))
    self.assertEqual(len(grads), 2)
    self.assertIsNone(grads[0])
    self.assertIsNotNone(grads[1])

if __name__ == '__main__':
  unittest.main()
