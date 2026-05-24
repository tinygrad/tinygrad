import unittest
from dataclasses import replace

from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.codegen.late.devectorizer import ReduceContext, pm_reduce
from tinygrad.codegen.late.reduce import CoupledReduceLowered, bind_coupled_reduce_descriptors, lower_coupled_reduce_plan
from tinygrad.codegen.opt import KernelOptError, Opt, OptOps
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer
from tinygrad.uop.coupled_reduce import CoupledReduceDescriptor, CoupledReduceField, CoupledReducePlan, CoupledReduceRejectReason, \
  validate_coupled_reduce_plan, rewrite_normalized_weighted_add_reduces
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite
from tinygrad.uop.spec import spec_tensor, type_verify

class TestCoupledReduce(unittest.TestCase):
  def _scalar_descriptor_and_peer(self):
    g = UOp.range(3, 0)
    r = UOp.range(4, 1, AxisType.REDUCE)
    x = UOp.placeholder((3, 4), dtypes.float32, 0)
    y = UOp.placeholder((3, 4), dtypes.float32, 1)
    state0 = UOp.variable("coupled_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("coupled_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_x", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + x[g, r])
    field1 = CoupledReduceField(1, dtypes.float32, UOp.const(dtypes.float32, 1.0), state1, state1 + state0 * y[g, r])
    plan = CoupledReducePlan((field0, field1), (r,), state0 - state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (x[g, r], r), (Ops.ADD, ()))
    scalar_peer = UOp(Ops.REDUCE, dtypes.float32, (y[g, r], r), (Ops.ADD, ()))
    return CoupledReduceDescriptor(target, plan), scalar_peer, g, r

  def _vector_descriptor(self):
    g = UOp.range(2, 0)
    r = UOp.range(3, 1, AxisType.REDUCE)
    vec_dtype = dtypes.float32.vec(4)
    x = UOp.placeholder((2, 3), vec_dtype, 0)
    state0 = UOp.variable("vector_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("vector_state1", -1000.0, 1000.0, dtype=vec_dtype)
    field0 = CoupledReduceField("scalar_sum", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + g.cast(dtypes.float32))
    field1 = CoupledReduceField("vector_sum", vec_dtype, UOp.const(vec_dtype, (0.0, 0.0, 0.0, 0.0)), state1, state1 + x[g, r])
    plan = CoupledReducePlan((field0, field1), (r,), state1)
    target = UOp(Ops.REDUCE, vec_dtype, (x[g, r], r), (Ops.ADD, ()))
    return CoupledReduceDescriptor(target, plan), g, r

  def _codegen_descriptor_and_peer(self):
    r = UOp.range(4, 0, AxisType.REDUCE)
    value = r.cast(dtypes.float32)
    state0 = UOp.variable("codegen_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("codegen_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("uses_sum", dtypes.float32, UOp.const(dtypes.float32, 1.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    scalar_peer = UOp(Ops.REDUCE, dtypes.float32, (value + UOp.const(dtypes.float32, 1.0), r), (Ops.ADD, ()))
    return CoupledReduceDescriptor(target, plan), scalar_peer

  def _indexed_codegen_descriptor(self):
    g = UOp.range(2, 0)
    r = UOp.range(4, 1, AxisType.REDUCE)
    x = UOp.placeholder((2, 4), dtypes.float32, 0)
    value = x[g, r]
    state0 = UOp.variable("indexed_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("indexed_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("uses_sum", dtypes.float32, UOp.const(dtypes.float32, 1.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    return CoupledReduceDescriptor(target, plan), g

  def _normalized_weighted_expr(self, product_order="xw"):
    g = UOp.range(2, 0)
    nr = UOp.range(4, 1, AxisType.REDUCE)
    dr = UOp.range(4, 2, AxisType.REDUCE)
    x = UOp.placeholder((2, 4), dtypes.float32, 0)
    w = UOp.placeholder((2, 4), dtypes.float32, 1)
    xval, nw, dw = x[g, nr], w[g, nr], w[g, dr]
    numerator_value = nw * xval if product_order == "wx" else xval * nw
    numerator = UOp(Ops.REDUCE, dtypes.float32, (numerator_value, nr), (Ops.ADD, ()))
    denominator = UOp(Ops.REDUCE, dtypes.float32, (dw, dr), (Ops.ADD, ()))
    return numerator * denominator.reciprocal(), numerator, denominator, g, nr, dr, x, w

  def _regs(self, root):
    return sorted([u for u in root.toposort() if u.op is Ops.DEFINE_REG and u.dtype.addrspace is AddrSpace.REG], key=lambda u: u.arg)

  def _full_rewrite(self, sink):
    with Context(SPEC=1):
      return full_rewrite_to_sink(sink, Renderer(Target()), optimize=False)

  def test_scalar_fields_lower_to_one_register_tuple(self):
    descriptor, _, g, r = self._scalar_descriptor_and_peer()
    lowered = lower_coupled_reduce_plan(descriptor.plan, slot=7, target=descriptor.target)

    self.assertIsInstance(lowered, CoupledReduceLowered)
    self.assertEqual(len(lowered.accumulators), 1)
    self.assertEqual(lowered.accumulators[0].op, Ops.DEFINE_REG)
    self.assertEqual(lowered.accumulators[0].arg, 7)
    self.assertEqual(lowered.accumulators[0].dtype.addrspace, AddrSpace.REG)
    self.assertEqual(lowered.accumulators[0].dtype.size, 2)
    self.assertEqual(lowered.accumulator_slots, 1)
    self.assertEqual(lowered.input_ranges, (g,))
    self.assertEqual(lowered.end.src[1:], (r,))
    self.assertEqual(tuple(x.dtype for x in lowered.old_fields), (dtypes.float32, dtypes.float32))
    self.assertIn(lowered.final_fields[0], lowered.final.toposort())
    self.assertIn(lowered.final_fields[1], lowered.final.toposort())
    update0, update1 = (store.src[1] for store in lowered.update_group.src)
    self.assertIn(lowered.old_fields[0], update0.toposort())
    self.assertIn(lowered.old_fields[0], update1.toposort())
    self.assertIn(lowered.old_fields[1], update1.toposort())

  def test_vector_field_lowers_to_typed_register_slots(self):
    descriptor, _, _ = self._vector_descriptor()
    lowered = lower_coupled_reduce_plan(descriptor.plan, target=descriptor.target)

    self.assertIsInstance(lowered, CoupledReduceLowered)
    self.assertEqual(lowered.final.dtype, dtypes.float32.vec(4))
    self.assertEqual(tuple(x.dtype for x in lowered.old_fields), (dtypes.float32, dtypes.float32.vec(4)))
    self.assertEqual(lowered.accumulator_slots, 2)
    self.assertEqual([(reg.arg, reg.dtype.base, reg.dtype.size) for reg in lowered.accumulators],
                     [(0, dtypes.float32, 1), (1, dtypes.float32.vec(4), 1)])
    self.assertEqual([(reg.arg, reg.dtype.base, reg.dtype.size) for reg in self._regs(lowered.final)],
                     [(0, dtypes.float32, 1), (1, dtypes.float32.vec(4), 1)])

  def test_mixed_dtype_coupled_reduce_advances_ordinary_reduce_slot(self):
    descriptor, g, r = self._vector_descriptor()
    scalar_peer = UOp(Ops.REDUCE, dtypes.float32, ((g + r).cast(dtypes.float32), r), (Ops.ADD, ()))
    ctx = ReduceContext(coupled={descriptor.target: descriptor.plan})
    rewritten = graph_rewrite(UOp.sink(descriptor.target, scalar_peer), pm_reduce, ctx=ctx, name="remove_reduce")

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual([(reg.arg, reg.dtype.base, reg.dtype.size) for reg in self._regs(rewritten)],
                     [(0, dtypes.float32, 1), (1, dtypes.float32.vec(4), 1), (2, dtypes.float32, 1)])
    self.assertEqual(ctx.acc_num, 3)

  def test_two_reduce_ranges_lower_through_shared_end(self):
    r0 = UOp.range(3, 0, AxisType.REDUCE)
    r1 = UOp.range(4, 1, AxisType.REDUCE)
    value = (r0 + r1).cast(dtypes.float32)
    state0 = UOp.variable("two_range_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("two_range_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r0, r1), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r0, r1), (Ops.ADD, ()))
    lowered = lower_coupled_reduce_plan(plan, target=target)

    self.assertIsInstance(lowered, CoupledReduceLowered)
    self.assertEqual(lowered.end.src[1:], (r0, r1))
    self.assertEqual(lowered.input_ranges, ())

  def test_rejects_target_independent_of_one_reduce_range(self):
    r0 = UOp.range(3, 0, AxisType.REDUCE)
    r1 = UOp.range(4, 1, AxisType.REDUCE)
    value = r0.cast(dtypes.float32)
    state0 = UOp.variable("missing_range_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("missing_range_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r0, r1), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r0, r1), (Ops.ADD, ()))
    rejection = validate_coupled_reduce_plan(plan, target)

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("all reduce ranges", rejection.detail)

  def test_rewrite_normalized_weighted_add_direct_positive(self):
    expr, numerator, _, _, _, _, _, _ = self._normalized_weighted_expr()
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)

    self.assertEqual(len(descriptors), 1)
    descriptor = descriptors[0]
    # the descriptor target is the numerator reduce carrying a coupled-reduce tag (so binding can rebind it)
    self.assertEqual(rewritten, descriptor.target)
    self.assertEqual(descriptor.target.replace(tag=None), numerator)
    self.assertIsNotNone(descriptor.target.tag)
    self.assertEqual(descriptor.plan.reduce_ranges, numerator.src[1:])
    self.assertEqual(tuple(field.name for field in descriptor.plan.fields), ("weighted_sum", "weight_sum"))
    self.assertEqual(tuple(field.state.op for field in descriptor.plan.fields), (Ops.DEFINE_VAR, Ops.DEFINE_VAR))
    self.assertEqual(tuple(field.state.src for field in descriptor.plan.fields), ((), ()))
    self.assertIsNone(validate_coupled_reduce_plan(descriptor.plan, descriptor.target))
    self.assertEqual(descriptor.plan.final.op, Ops.MUL)
    self.assertEqual(descriptor.plan.final.src[1].op, Ops.RECIPROCAL)

  def test_rewrite_normalized_weighted_add_product_order_positive(self):
    expr, numerator, _, _, _, _, _, _ = self._normalized_weighted_expr("wx")
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)

    self.assertEqual(len(descriptors), 1)
    self.assertEqual(rewritten.replace(tag=None), numerator)
    self.assertIsNone(validate_coupled_reduce_plan(descriptors[0].plan, descriptors[0].target))

  def test_rewrite_normalized_weighted_add_preserves_enclosing_graph(self):
    expr, numerator, _, g, _, _, _, _ = self._normalized_weighted_expr()
    out = UOp.placeholder((2,), dtypes.float32, 2)
    root = UOp.sink(out.index(g).store(expr))
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(root)

    self.assertEqual(len(descriptors), 1)
    self.assertEqual(rewritten.op, Ops.SINK)
    self.assertIn(descriptors[0].target, rewritten.toposort())
    self.assertEqual(descriptors[0].target.replace(tag=None), numerator)
    self.assertNotIn(expr, rewritten.toposort())

  def test_rewrite_normalized_weighted_add_rejects_non_matches(self):
    expr, numerator, denominator, g, nr, dr, x, w = self._normalized_weighted_expr()
    y = UOp.placeholder((2, 4), dtypes.float32, 3)
    cases = {
      "same_extent_without_witness": UOp(Ops.REDUCE, dtypes.float32, (x[g, nr] * y[g, nr], nr), (Ops.ADD, ())) *
                                     denominator.reciprocal(),
      "source_mismatch": numerator * UOp(Ops.REDUCE, dtypes.float32, (y[g, dr], dr), (Ops.ADD, ())).reciprocal(),
      "shifted_index": UOp(Ops.REDUCE, dtypes.float32, (x[g, nr] * w[g, nr + UOp.const(dtypes.weakint, 1)], nr),
                           (Ops.ADD, ())) * denominator.reciprocal(),
      "non_add_numerator": numerator.replace(arg=(Ops.MAX, ())) * denominator.reciprocal(),
      "non_add_denominator": numerator * denominator.replace(arg=(Ops.MAX, ())).reciprocal(),
      "non_reduce_denominator": numerator * w[g, nr].reciprocal(),
      "scalar_denominator": numerator * UOp(Ops.REDUCE, dtypes.float32, (UOp.const(dtypes.float32, 1.0), dr), (Ops.ADD, ())).reciprocal(),
      "duplicate_matching_factor": UOp(Ops.REDUCE, dtypes.float32, (x[g, nr] * w[g, nr] * w[g, nr], nr), (Ops.ADD, ())) *
                                   denominator.reciprocal(),
      "fdiv_form": numerator.alu(Ops.FDIV, denominator),
    }
    for name, root in cases.items():
      with self.subTest(name=name):
        rewritten, descriptors = rewrite_normalized_weighted_add_reduces(root)
        self.assertIs(rewritten, root)
        self.assertEqual(descriptors, ())
    cast_expr = expr.cast(dtypes.float16)
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(cast_expr)
    self.assertIs(rewritten, cast_expr)
    self.assertEqual(descriptors, ())

  def test_rewrite_normalized_weighted_add_rejects_permuted_indices(self):
    g = UOp.range(2, 0)
    nr0 = UOp.range(4, 1, AxisType.REDUCE)
    nr1 = UOp.range(4, 2, AxisType.REDUCE)
    dr0 = UOp.range(4, 3, AxisType.REDUCE)
    dr1 = UOp.range(4, 4, AxisType.REDUCE)
    x = UOp.placeholder((2, 4, 4), dtypes.float32, 0)
    w = UOp.placeholder((2, 4, 4), dtypes.float32, 1)
    numerator = UOp(Ops.REDUCE, dtypes.float32, (x[g, nr0, nr1] * w[g, nr1, nr0], nr0, nr1), (Ops.ADD, ()))
    denominator = UOp(Ops.REDUCE, dtypes.float32, (w[g, dr0, dr1], dr0, dr1), (Ops.ADD, ()))
    expr = numerator * denominator.reciprocal()

    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)
    self.assertIs(rewritten, expr)
    self.assertEqual(descriptors, ())

  def test_rewrite_normalized_weighted_add_rejects_dtype_mismatch(self):
    r = UOp.range(4, 0, AxisType.REDUCE)
    x = UOp.placeholder((4,), dtypes.float16, 0)
    w = UOp.placeholder((4,), dtypes.float16, 1)
    numerator = UOp(Ops.REDUCE, dtypes.float16, (x[r] * w[r], r), (Ops.ADD, ()))
    denominator = UOp(Ops.REDUCE, dtypes.float16, (w[r], r), (Ops.ADD, ()))
    expr = numerator * denominator.reciprocal()

    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)
    self.assertIs(rewritten, expr)
    self.assertEqual(descriptors, ())

  def test_rewrite_normalized_weighted_add_rejects_multiple_candidates_and_raw_numerator_consumers(self):
    expr0, numerator0, _, _, _, _, _, _ = self._normalized_weighted_expr()
    expr1, _, _, _, _, _, _, _ = self._normalized_weighted_expr("wx")
    multi = UOp.sink(expr0, expr1)
    raw = UOp.sink(expr0, numerator0)

    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(multi)
    self.assertIs(rewritten, multi)
    self.assertEqual(descriptors, ())
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(raw)
    self.assertIs(rewritten, raw)
    self.assertEqual(descriptors, ())

  def _online_softmax_expr(self, *, D=2, N=4):
    import math
    # mirrors the post-fusion SDPA shape: numerator / denominator each reduce over a distinct range,
    # both sharing one stabilizing max reduce over its own range (all three the same extent).
    g = UOp.range(D, 0, AxisType.GLOBAL)
    j_n = UOp.range(N, 1, AxisType.REDUCE)
    j_d = UOp.range(N, 2, AxisType.REDUCE)
    j_m = UOp.range(N, 3, AxisType.REDUCE)
    s_buf = UOp.placeholder((N,), dtypes.float32, 0)
    v_buf = UOp.placeholder((N, D), dtypes.float32, 1)
    log2_e = UOp.const(dtypes.float32, math.log2(math.e))
    def stable_exp(a, b): return ((a - b).alu(Ops.MUL, log2_e)).alu(Ops.EXP2)
    M = UOp(Ops.REDUCE, dtypes.float32, (s_buf[j_m], j_m), (Ops.MAX, ()))
    numerator = UOp(Ops.REDUCE, dtypes.float32, (stable_exp(s_buf[j_n], M) * v_buf[j_n, g], j_n), (Ops.ADD, ()))
    denominator = UOp(Ops.REDUCE, dtypes.float32, (stable_exp(s_buf[j_d], M), j_d), (Ops.ADD, ()))
    return numerator * denominator.reciprocal(), s_buf, v_buf, g, j_n

  def test_online_softmax_rewriter_emits_three_acc_descriptor(self):
    expr, *_ = self._online_softmax_expr()
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)

    self.assertEqual(len(descriptors), 1)
    descriptor = descriptors[0]
    self.assertEqual(tuple(f.name for f in descriptor.plan.fields), ("softmax_max", "softmax_denom", "softmax_weighted"))
    self.assertEqual(descriptor.plan.final.op, Ops.MUL)
    self.assertEqual(descriptor.plan.final.src[1].op, Ops.RECIPROCAL)
    self.assertIsNone(validate_coupled_reduce_plan(descriptor.plan, descriptor.target))
    self.assertIs(rewritten, descriptor.target)

  def test_online_softmax_executes_to_pytorch_softmax_at_atol(self):
    import numpy as np
    D, N = 2, 4
    expr, s_buf, v_buf, g, _j_n = self._online_softmax_expr(D=D, N=N)
    out_buf = UOp.placeholder((D,), dtypes.float32, 2)
    sink = UOp.sink(out_buf.index(g).store(expr), arg=KernelInfo(name="fa_online_softmax_e2e", opts_to_apply=()))

    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(sink)
    self.assertEqual(len(descriptors), 1)
    rewritten = rewritten.replace(arg=replace(rewritten.arg, coupled_reduce=descriptors))
    with Context(SPEC=1):
      program = to_program(rewritten, PythonRenderer(Target("PYTHON")))
    global_size, local_size = program.arg.launch_dims({})

    np_rng = np.random.default_rng(0)
    s_np = np_rng.standard_normal(N, dtype=np.float32)
    v_np = np_rng.standard_normal((N, D), dtype=np.float32)
    m_np = s_np.max()
    e_np = np.exp(s_np - m_np)
    expected = (e_np[:, None] * v_np).sum(axis=0) / e_np.sum()

    out_bytes = bytearray(D * 4)
    PythonProgram(program.arg.name, program.src[4].arg)(
      memoryview(s_np.tobytes()), memoryview(v_np.tobytes()), memoryview(out_bytes),
      global_size=global_size, local_size=local_size or (1, 1, 1), vals=program.arg.vals({}))
    got = np.frombuffer(out_bytes, dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)

  def test_online_softmax_field_widths_collapse_to_single_register_tuple(self):
    expr, *_ = self._online_softmax_expr()
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)
    lowered = lower_coupled_reduce_plan(descriptors[0].plan, target=rewritten)
    self.assertIsInstance(lowered, CoupledReduceLowered)
    self.assertEqual(lowered.accumulator_slots, 1)
    self.assertEqual(lowered.accumulators[0].dtype.size, 3)

  def test_online_softmax_does_not_match_non_stable_weighted_average(self):
    # plain weighted average without subtracted max should fall back to 2-acc descriptor (NOT the 3-acc one)
    expr, *_ = self._normalized_weighted_expr()
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)
    self.assertEqual(len(descriptors), 1)
    field_names = tuple(f.name for f in descriptors[0].plan.fields)
    self.assertEqual(field_names, ("weighted_sum", "weight_sum"))

  def test_online_softmax_rejects_max_over_different_logits(self):
    # the max reduce should reduce over the SAME logits that appear in (logits - max). Mismatched logits ⇒ no 3-acc.
    import math
    D, N = 2, 4
    g = UOp.range(D, 0, AxisType.GLOBAL)
    j_n = UOp.range(N, 1, AxisType.REDUCE)
    j_d = UOp.range(N, 2, AxisType.REDUCE)
    j_m = UOp.range(N, 3, AxisType.REDUCE)
    s_buf = UOp.placeholder((N,), dtypes.float32, 0)
    other = UOp.placeholder((N,), dtypes.float32, 3)
    v_buf = UOp.placeholder((N, D), dtypes.float32, 1)
    log2_e = UOp.const(dtypes.float32, math.log2(math.e))
    def stable_exp(a, b): return ((a - b).alu(Ops.MUL, log2_e)).alu(Ops.EXP2)
    M_wrong = UOp(Ops.REDUCE, dtypes.float32, (other[j_m], j_m), (Ops.MAX, ()))
    numerator = UOp(Ops.REDUCE, dtypes.float32, (stable_exp(s_buf[j_n], M_wrong) * v_buf[j_n, g], j_n), (Ops.ADD, ()))
    denominator = UOp(Ops.REDUCE, dtypes.float32, (stable_exp(s_buf[j_d], M_wrong), j_d), (Ops.ADD, ()))
    expr = numerator * denominator.reciprocal()
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(expr)
    self.assertEqual(len(descriptors), 1)
    self.assertEqual(tuple(f.name for f in descriptors[0].plan.fields), ("weighted_sum", "weight_sum"))

  def test_rejects_missing_descriptor_target(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    with self.assertRaisesRegex(AssertionError, "missing"):
      bind_coupled_reduce_descriptors(UOp.sink(UOp.const(dtypes.float32, 1.0)), (descriptor,))

  def test_rejects_ambiguous_descriptor_target(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    first = descriptor.target.replace(tag="first_same_key")
    second = descriptor.target.replace(tag="second_same_key")
    with self.assertRaisesRegex(AssertionError, "ambiguous"):
      bind_coupled_reduce_descriptors(UOp.sink(first, second), (descriptor,))

  def test_rejects_duplicate_descriptor_target_binding(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    with self.assertRaisesRegex(AssertionError, "duplicate"):
      bind_coupled_reduce_descriptors(UOp.sink(descriptor.target), (descriptor, descriptor))

  def test_binds_rebound_target_without_uop_truthiness(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    rebound_r = UOp.range(4, 1, AxisType.UNROLL)
    rebound_value = descriptor.target.src[0].substitute({r: rebound_r}, walk=True)
    rebound_target = descriptor.target.replace(src=(rebound_value, rebound_r))

    bound = bind_coupled_reduce_descriptors(UOp.sink(rebound_target), (descriptor,))
    self.assertEqual(tuple(bound), (rebound_target,))
    self.assertEqual(bound[rebound_target].reduce_ranges, (rebound_r,))

  def test_rejects_non_tuple_descriptor_payload_in_binder(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    with self.assertRaisesRegex(TypeError, "tuple"):
      bind_coupled_reduce_descriptors(UOp.sink(descriptor.target), [descriptor])

  def test_rejects_malformed_nominal_descriptor_in_binder(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    bad_descriptor = replace(descriptor, plan="bad")
    with self.assertRaisesRegex(TypeError, "plan"):
      bind_coupled_reduce_descriptors(UOp.sink(descriptor.target), (bad_descriptor,))

  def test_rejects_malformed_nominal_plan_shape_in_binder(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    bad_field = replace(descriptor.plan.fields[0], init="bad")
    cases = (
      (replace(descriptor.plan, fields=[*descriptor.plan.fields]), "fields"),
      (replace(descriptor.plan, fields=("bad", descriptor.plan.fields[1])), "fields"),
      (replace(descriptor.plan, reduce_ranges=list(descriptor.plan.reduce_ranges)), "reduce_ranges"),
      (replace(descriptor.plan, final="bad"), "final"),
      (replace(descriptor.plan, fields=(bad_field, descriptor.plan.fields[1])), "field init"),
    )
    for bad_plan, msg in cases:
      with self.subTest(msg=msg), self.assertRaisesRegex(TypeError, msg):
        bind_coupled_reduce_descriptors(UOp.sink(descriptor.target), (CoupledReduceDescriptor(descriptor.target, bad_plan),))

  def test_rejects_state_placeholder_that_appears_in_live_sink(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    live_state = descriptor.plan.fields[0].state
    with self.assertRaisesRegex(AssertionError, "state placeholders"):
      bind_coupled_reduce_descriptors(UOp.sink(descriptor.target, live_state), (descriptor,))

  def test_rejects_dtype_mismatch(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    x = UOp.placeholder((4,), dtypes.float32.vec(2), 9)
    bad_target = UOp(Ops.REDUCE, dtypes.float32.vec(2), (x[r], r), (Ops.ADD, ()))
    rejection = validate_coupled_reduce_plan(descriptor.plan, bad_target)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.DTYPE_MISMATCH)

  def test_rejects_invalid_target_reduce_arg(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    with Context(SPEC=0):
      bad_target = descriptor.target.replace(arg=(Ops.SUB, ()))
    rejection = validate_coupled_reduce_plan(descriptor.plan, bad_target)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("valid REDUCE arg", rejection.detail)

  def test_rejects_malformed_target_reduce_arg(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    cases = (("singleton_tuple", (Ops.ADD,)), ("tuple_with_int", (Ops.ADD, 0)), ("bare_op", Ops.ADD))
    for name, bad_arg in cases:
      with self.subTest(case=name), Context(SPEC=0):
        bad_target = descriptor.target.replace(arg=bad_arg)
        rejection = validate_coupled_reduce_plan(descriptor.plan, bad_target)
      self.assertIsNotNone(rejection)
      self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
      self.assertIn("valid REDUCE arg", rejection.detail)

  def test_rejects_duplicate_state_handle(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    bad_field = replace(second, state=first.state)
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((first, bad_field), descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("states must be unique", rejection.detail)

  def test_rejects_basic_invalid_plan_invariants(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    cases = (
      (replace(descriptor.plan, fields=(first,)), CoupledReduceRejectReason.INVALID_PLAN, "at least two fields"),
      (replace(descriptor.plan, fields=(first, replace(second, name=first.name))), CoupledReduceRejectReason.INVALID_PLAN, "names must be unique"),
      (replace(descriptor.plan, reduce_ranges=()), CoupledReduceRejectReason.MISSING_REDUCE_RANGE, "requires explicit reduce ranges"),
      (replace(descriptor.plan, reduce_ranges=(UOp.const(dtypes.int, 0),)), CoupledReduceRejectReason.MISSING_REDUCE_RANGE, "concrete RANGE"),
      (replace(descriptor.plan, fields=(replace(first, init=first.init.cast(dtypes.float16)), second)),
       CoupledReduceRejectReason.DTYPE_MISMATCH, "field dtype"),
    )
    for bad_plan, reason, detail in cases:
      with self.subTest(detail=detail):
        rejection = validate_coupled_reduce_plan(bad_plan)
      self.assertIsNotNone(rejection)
      self.assertEqual(rejection.reason, reason)
      self.assertIn(detail, rejection.detail)

  def test_rejects_non_placeholder_state_handle(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    bad_field = replace(first, state=UOp.const(dtypes.float32, 0.0))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("DEFINE_VAR", rejection.detail)

  def test_rejects_undeclared_state_in_update(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    hidden = UOp.variable("hidden_update_state", -1000.0, 1000.0, dtype=dtypes.float32)
    bad_field = replace(second, update=second.update + hidden)
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((first, bad_field), descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("undeclared state", rejection.detail)
    self.assertIn(hidden, rejection.evidence)

  def test_rejects_undeclared_state_in_final(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    hidden = UOp.variable("hidden_final_state", -1000.0, 1000.0, dtype=dtypes.float32)
    bad_plan = CoupledReducePlan(descriptor.plan.fields, descriptor.plan.reduce_ranges, descriptor.plan.final + hidden)
    rejection = validate_coupled_reduce_plan(bad_plan)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("undeclared state", rejection.detail)
    self.assertIn(hidden, rejection.evidence)

  def test_rejects_state_reference_in_init(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    bad_field = replace(first, init=first.state)
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("init", rejection.detail)
    self.assertIn(first.state, rejection.evidence)

  def test_rejects_declared_reduce_range_in_init(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    bad_field = replace(first, init=r.cast(dtypes.float32))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges, descriptor.plan.final))

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.REDUCE_RANGE_IN_INIT)
    self.assertIn(bad_field.init, rejection.evidence)

  def test_rejects_declared_reduce_range_in_final(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    bad_final = descriptor.plan.final + r.cast(dtypes.float32)
    rejection = validate_coupled_reduce_plan(replace(descriptor.plan, final=bad_final))

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.REDUCE_RANGE_IN_FINAL)
    self.assertIn(bad_final, rejection.evidence)

  def test_rejects_range_mismatch(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    mismatch = UOp.range(4, 99, AxisType.REDUCE)
    bad_plan = CoupledReducePlan(descriptor.plan.fields, (mismatch,), descriptor.plan.final)
    rejection = validate_coupled_reduce_plan(bad_plan, descriptor.target)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)

  def test_rejects_target_independent_of_reduce_range(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    bad_target = UOp(Ops.REDUCE, dtypes.float32, (UOp.const(dtypes.float32, 1.0), r), (Ops.ADD, ()))
    rejection = validate_coupled_reduce_plan(descriptor.plan, bad_target)

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("depend on all reduce ranges", rejection.detail)

  def test_rejects_target_with_only_control_dependency_on_reduce_range(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    barrier = UOp.const(dtypes.float32, 0.0).end(r)
    value = UOp.const(dtypes.float32, 1.0).after(barrier)
    bad_target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    rejection = validate_coupled_reduce_plan(descriptor.plan, bad_target)

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("depend on all reduce ranges", rejection.detail)

  def test_rejects_update_with_only_control_dependency_on_reduce_range(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    barrier = UOp.const(dtypes.float32, 0.0).end(r)
    bad_field = replace(first, update=UOp.const(dtypes.float32, 1.0).after(barrier))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges,
                                                               descriptor.plan.final), descriptor.target)

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("control dependencies", rejection.detail)

  def test_full_rewrite_rejects_optimizable_independent_target(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    bad_target = UOp(Ops.REDUCE, dtypes.float32, (UOp.const(dtypes.float32, 1.0), r), (Ops.ADD, ()))
    bad_descriptor = CoupledReduceDescriptor(bad_target, replace(descriptor.plan, reduce_ranges=(r,)))
    with Context(SPEC=0):
      sink = UOp.sink(bad_target, arg=KernelInfo(name="bad_independent", opts_to_apply=(), coupled_reduce=(bad_descriptor,)))

    with Context(SPEC=0), self.assertRaisesRegex(AssertionError, "depend on all reduce ranges"):
      full_rewrite_to_sink(sink, Renderer(Target()))

  def test_full_rewrite_rejects_control_only_target_dependency(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    barrier = UOp.const(dtypes.float32, 0.0).end(r)
    value = UOp.const(dtypes.float32, 1.0).after(barrier)
    bad_target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    bad_descriptor = CoupledReduceDescriptor(bad_target, replace(descriptor.plan, reduce_ranges=(r,)))
    with Context(SPEC=0):
      sink = UOp.sink(bad_target, arg=KernelInfo(name="bad_control_dependency", opts_to_apply=(), coupled_reduce=(bad_descriptor,)))

    with Context(SPEC=0), self.assertRaisesRegex(AssertionError, "depend on all reduce ranges"):
      full_rewrite_to_sink(sink, Renderer(Target()))

  def test_full_rewrite_rejects_control_only_update_dependency(self):
    descriptor, _, _, r = self._scalar_descriptor_and_peer()
    first, second = descriptor.plan.fields
    barrier = UOp.const(dtypes.float32, 0.0).end(r)
    bad_field = replace(first, update=UOp.const(dtypes.float32, 1.0).after(barrier))
    bad_plan = CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges, descriptor.plan.final)
    with Context(SPEC=0):
      sink = UOp.sink(descriptor.target, arg=KernelInfo(name="bad_control_update", opts_to_apply=(),
                                                        coupled_reduce=(CoupledReduceDescriptor(descriptor.target, bad_plan),)))

    with Context(SPEC=0), self.assertRaisesRegex(AssertionError, "control dependencies"):
      full_rewrite_to_sink(sink, Renderer(Target()), optimize=False)

  def test_full_rewrite_rejects_invalid_target_reduce_arg(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    with Context(SPEC=0):
      bad_target = descriptor.target.replace(arg=(Ops.SUB, ()))
      bad_descriptor = CoupledReduceDescriptor(bad_target, descriptor.plan)
      sink = UOp.sink(bad_target, arg=KernelInfo(name="bad_reduce_arg", opts_to_apply=(), coupled_reduce=(bad_descriptor,)))
    with Context(SPEC=0), self.assertRaisesRegex(AssertionError, "valid REDUCE arg"):
      full_rewrite_to_sink(sink, Renderer(Target()))

  def test_full_rewrite_rejects_fake_descriptor_payload_before_rewrite(self):
    descriptor, scalar_peer = self._codegen_descriptor_and_peer()
    fake = type("FakeDescriptor", (), {"target": descriptor.target, "plan": descriptor.plan})()
    with Context(SPEC=0):
      sink = UOp.sink(descriptor.target, scalar_peer, arg=KernelInfo(name="fake_full_rewrite", opts_to_apply=(), coupled_reduce=(fake,)))
    with Context(SPEC=0), self.assertRaisesRegex(TypeError, "CoupledReduceDescriptor"):
      full_rewrite_to_sink(sink, Renderer(Target()), optimize=False)

  def test_rejects_unstable_field_name_type(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    bad_field = replace(descriptor.plan.fields[0], name=("tuple", "name"))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, descriptor.plan.fields[1]),
                                                               descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.INVALID_PLAN)
    self.assertIn("str or int", rejection.detail)

  def test_rejects_foreign_reduce_domain(self):
    descriptor, _, g, _ = self._scalar_descriptor_and_peer()
    bad_plan = CoupledReducePlan(descriptor.plan.fields, (g,), descriptor.plan.final)
    rejection = validate_coupled_reduce_plan(bad_plan)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.UNSUPPORTED_RANGE_KIND)

  def test_rejects_foreign_reduce_domain_in_init(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    foreign = UOp.range(2, 98, AxisType.REDUCE)
    first, second = descriptor.plan.fields
    bad_field = replace(first, init=foreign.cast(dtypes.float32))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)
    self.assertIn("field init uses foreign reduce ranges", rejection.detail)

  def test_rejects_control_only_foreign_reduce_domain_in_init(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    foreign = UOp.range(2, 98, AxisType.REDUCE)
    first, second = descriptor.plan.fields
    barrier = UOp.const(dtypes.float32, 0.0).end(foreign)
    bad_field = replace(first, init=UOp.const(dtypes.float32, 1.0).after(barrier))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((bad_field, second), descriptor.plan.reduce_ranges, descriptor.plan.final))
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)
    self.assertIn("field init uses foreign reduce ranges", rejection.detail)
    self.assertIn(foreign, rejection.evidence)

  def test_rejects_control_only_foreign_reduce_domain_in_update(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    foreign = UOp.range(2, 98, AxisType.REDUCE)
    first, second = descriptor.plan.fields
    barrier = UOp.const(dtypes.float32, 0.0).end(foreign)
    bad_field = replace(second, update=UOp.const(dtypes.float32, 1.0).after(barrier))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((first, bad_field), descriptor.plan.reduce_ranges,
                                                               descriptor.plan.final), descriptor.target)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)
    self.assertIn("field update uses foreign reduce ranges", rejection.detail)
    self.assertIn(foreign, rejection.evidence)

  def test_rejects_control_only_foreign_reduce_domain_in_final(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    foreign = UOp.range(2, 98, AxisType.REDUCE)
    barrier = UOp.const(dtypes.float32, 0.0).end(foreign)
    bad_plan = CoupledReducePlan(descriptor.plan.fields, descriptor.plan.reduce_ranges,
                                 descriptor.plan.final + UOp.const(dtypes.float32, 1.0).after(barrier))
    rejection = validate_coupled_reduce_plan(bad_plan, descriptor.target)
    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)
    self.assertIn("final projection uses foreign reduce ranges", rejection.detail)
    self.assertIn(foreign, rejection.evidence)

  def test_accepts_symbolic_range_extent_define_var(self):
    n = UOp.variable("sym_extent", 1, 8)
    r = UOp.range(n, 0, AxisType.REDUCE)
    value = r.cast(dtypes.float32)
    state0 = UOp.variable("symbolic_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("symbolic_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)

    self.assertIsNone(validate_coupled_reduce_plan(plan, target))

  def test_binds_symbolic_outer_range_descriptor(self):
    n = UOp.variable("sym_outer", 1, 8)
    g = UOp.range(n, 0)
    r = UOp.range(4, 1, AxisType.REDUCE)
    x = UOp.placeholder((n, 4), dtypes.float32, 0)
    value = x[g, r]
    state0 = UOp.variable("symbolic_outer_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("symbolic_outer_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)

    self.assertIsNone(validate_coupled_reduce_plan(plan, target))
    self.assertEqual(bind_coupled_reduce_descriptors(UOp.sink(target), (CoupledReduceDescriptor(target, plan),)), {target:plan})

  def test_rejects_foreign_loop_range_in_update(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    foreign = UOp.range(2, 98, AxisType.LOOP)
    first, second = descriptor.plan.fields
    bad_field = replace(second, update=second.update + foreign.cast(dtypes.float32))
    rejection = validate_coupled_reduce_plan(CoupledReducePlan((first, bad_field), descriptor.plan.reduce_ranges, descriptor.plan.final),
                                             descriptor.target)

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)
    self.assertIn("field update uses foreign ranges", rejection.detail)

  def test_rejects_foreign_loop_range_in_final(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    foreign = UOp.range(2, 98, AxisType.LOOP)
    bad_plan = CoupledReducePlan(descriptor.plan.fields, descriptor.plan.reduce_ranges,
                                 descriptor.plan.final + foreign.cast(dtypes.float32))
    rejection = validate_coupled_reduce_plan(bad_plan, descriptor.target)

    self.assertIsNotNone(rejection)
    self.assertEqual(rejection.reason, CoupledReduceRejectReason.RANGE_MISMATCH)
    self.assertIn("final projection uses foreign ranges", rejection.detail)

  def test_pm_reduce_consumes_only_bound_target(self):
    descriptor, scalar_peer, _, _ = self._scalar_descriptor_and_peer()
    rewritten = graph_rewrite(UOp.sink(descriptor.target, scalar_peer), pm_reduce,
                              ctx=ReduceContext(coupled={descriptor.target:descriptor.plan}), name="remove_reduce")

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual([reg.arg for reg in self._regs(rewritten)], [0, 1])
    self.assertEqual([reg.dtype.size for reg in self._regs(rewritten)], [2, 1])

  def test_pm_reduce_ordinary_reduce_unchanged_without_descriptor(self):
    _, scalar_peer, _, _ = self._scalar_descriptor_and_peer()
    ctx = ReduceContext(acc_num=5)
    rewritten = graph_rewrite(UOp.sink(scalar_peer), pm_reduce, ctx=ctx, name="remove_reduce")
    regs = self._regs(rewritten)

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual(len(regs), 1)
    self.assertEqual(regs[0].arg, 5)
    self.assertEqual(regs[0].dtype.size, 1)
    self.assertEqual(ctx.acc_num, 6)

  def test_full_rewrite_ordinary_reduce_without_descriptor(self):
    _, scalar_peer = self._codegen_descriptor_and_peer()
    sink = UOp.sink(scalar_peer, arg=KernelInfo(name="ordinary_reduce", opts_to_apply=()))
    rewritten = self._full_rewrite(sink)
    regs = self._regs(rewritten)

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertIsInstance(rewritten.arg, KernelInfo)
    self.assertEqual(rewritten.arg.coupled_reduce, ())
    self.assertEqual(len(regs), 1)
    self.assertEqual(regs[0].dtype.size, 1)

  def test_kernelinfo_descriptor_survives_postrange(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    sink = UOp.sink(descriptor.target, arg=KernelInfo(name="coupled_transport", opts_to_apply=(), coupled_reduce=(descriptor,)))
    optimized = apply_opts(sink, Renderer(Target()))

    self.assertIsInstance(optimized.arg, KernelInfo)
    self.assertEqual(optimized.arg.coupled_reduce, (descriptor,))

  def test_auto_normalized_weighted_add_after_apply_opts(self):
    expr, numerator, _, _, _, _, _, _ = self._normalized_weighted_expr()
    sink = UOp.sink(expr, arg=KernelInfo(name="auto_normalized_weighted", opts_to_apply=()))
    optimized = apply_opts(sink, Renderer(Target()))
    rewritten, descriptors = rewrite_normalized_weighted_add_reduces(optimized)

    self.assertEqual(len(descriptors), 1)
    self.assertEqual(descriptors[0].target.replace(tag=None), numerator)
    self.assertIn(descriptors[0].target, rewritten.toposort())
    self.assertNotIn(expr, rewritten.toposort())

  def test_auto_normalized_weighted_add_skips_explicit_descriptor_guard(self):
    expr, numerator, _, _, _, _, _, _ = self._normalized_weighted_expr()
    state0 = UOp.variable("explicit_skip_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("explicit_skip_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    plan = CoupledReducePlan((
      CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + numerator.src[0]),
      CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0),
    ), numerator.src[1:], state0 + state1)
    descriptor = CoupledReduceDescriptor(numerator, plan)
    sink = UOp.sink(expr, arg=KernelInfo(name="explicit_skip_auto", opts_to_apply=(), coupled_reduce=(descriptor,)))
    optimized = apply_opts(sink, Renderer(Target()))

    self.assertIsInstance(optimized.arg, KernelInfo)
    self.assertEqual(optimized.arg.coupled_reduce, (descriptor,))

  def test_full_rewrite_auto_normalized_weighted_add_consumes_descriptor(self):
    expr, _, _, g, _, _, _, _ = self._normalized_weighted_expr()
    out = UOp.placeholder((2,), dtypes.float32, 2)
    sink = UOp.sink(out.index(g).store(expr), arg=KernelInfo(name="full_auto_normalized_weighted", opts_to_apply=()))
    with Context(SPEC=1):
      rewritten = full_rewrite_to_sink(sink, Renderer(Target()))

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual(rewritten.arg.coupled_reduce, ())
    self.assertEqual([reg.dtype.size for reg in self._regs(rewritten)], [2])

  def test_postrange_rejects_unroll_on_descriptor_range(self):
    descriptor, scalar_peer = self._codegen_descriptor_and_peer()
    sink = UOp.sink(descriptor.target, scalar_peer,
                    arg=KernelInfo(name="unroll_coupled", opts_to_apply=(Opt(OptOps.UNROLL, 0, 2),), coupled_reduce=(descriptor,)))
    with Context(SPEC=0), self.assertRaisesRegex(KernelOptError, "rewrite descriptor ranges"):
      full_rewrite_to_sink(sink, Renderer(Target()))

  def test_postrange_rejects_other_descriptor_range_rewrites(self):
    descriptor, _ = self._codegen_descriptor_and_peer()
    indexed_descriptor, g = self._indexed_codegen_descriptor()
    cases = (
      (descriptor, Opt(OptOps.PADTO, 0, 8), "padto_coupled"),
      (indexed_descriptor, Opt(OptOps.UPCAST, 0, 2), "upcast_coupled"),
    )
    for desc, opt, name in cases:
      with self.subTest(name=name):
        sink = UOp.sink(desc.target, arg=KernelInfo(name=name, opts_to_apply=(opt,), coupled_reduce=(desc,)))
        with Context(SPEC=0), self.assertRaisesRegex(KernelOptError, "rewrite descriptor ranges"):
          full_rewrite_to_sink(sink, Renderer(Target()))
    self.assertIn(g, indexed_descriptor.target.ranges)

  def test_postrange_rejects_hidden_descriptor_update_range_rewrite(self):
    g = UOp.range(2, 0)
    r = UOp.range(4, 1, AxisType.REDUCE)
    out = UOp.placeholder((2,), dtypes.float32, 0)
    value = r.cast(dtypes.float32)
    state0 = UOp.variable("hidden_range_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("hidden_range_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("uses_hidden_range", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + g.cast(dtypes.float32))
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    descriptor = CoupledReduceDescriptor(target, plan)
    with Context(SPEC=0):
      sink = UOp.sink(out.index(g).store(target),
                      arg=KernelInfo(name="hidden_range_coupled", opts_to_apply=(Opt(OptOps.UPCAST, 0, 2),),
                                     coupled_reduce=(descriptor,)))

    self.assertNotIn(g, target.toposort())
    self.assertIn(g, field1.update.toposort())
    with Context(SPEC=0), self.assertRaisesRegex(KernelOptError, "rewrite descriptor ranges"):
      apply_opts(sink, Renderer(Target()))

  def test_postrange_rejects_tensor_core_with_descriptor(self):
    descriptor, _ = self._codegen_descriptor_and_peer()
    sink = UOp.sink(descriptor.target, arg=KernelInfo(name="tc_coupled", opts_to_apply=(Opt(OptOps.TC, 0, (0, 0, 1)),),
                                                      coupled_reduce=(descriptor,)))
    with Context(SPEC=0), self.assertRaisesRegex(KernelOptError, "rewrite descriptor ranges"):
      full_rewrite_to_sink(sink, Renderer(Target()))

  def test_full_rewrite_skips_auto_postrange_opts_with_descriptor(self):
    g = UOp.range(8, 0)
    r = UOp.range(4, 1, AxisType.REDUCE)
    x = UOp.placeholder((8, 4), dtypes.float32, 0)
    value = x[g, r]
    state0 = UOp.variable("autoopt_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("autoopt_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    out = UOp.placeholder((8,), dtypes.float32, 1)
    sink = UOp.sink(out.index(g).store(target),
                    arg=KernelInfo(name="autoopt_coupled", coupled_reduce=(CoupledReduceDescriptor(target, plan),)))
    with Context(SPEC=1):
      rewritten = full_rewrite_to_sink(sink, Renderer(Target()))

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual(rewritten.arg.coupled_reduce, ())

  def test_full_rewrite_consumes_kernelinfo_descriptor(self):
    descriptor, scalar_peer = self._codegen_descriptor_and_peer()
    sink = UOp.sink(descriptor.target, scalar_peer,
                    arg=KernelInfo(name="full_coupled", opts_to_apply=(), coupled_reduce=(descriptor,)))
    rewritten = self._full_rewrite(sink)

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertIsInstance(rewritten.arg, KernelInfo)
    self.assertEqual(rewritten.arg.coupled_reduce, ())
    self.assertEqual([reg.dtype.size for reg in self._regs(rewritten)], [2, 1])

  def test_full_rewrite_consumes_descriptor_before_gep_pushing(self):
    r = UOp.range(4, 0, AxisType.REDUCE)
    base = r.cast(dtypes.float32)
    stacked = UOp(Ops.STACK, dtypes.float32.vec(2), (base, base + UOp.const(dtypes.float32, 1.0)))
    value = UOp(Ops.GEP, dtypes.float32, (stacked,), (0,))
    state0 = UOp.variable("gep_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("gep_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    sink = UOp.sink(target, arg=KernelInfo(name="gep_coupled", opts_to_apply=(), coupled_reduce=(CoupledReduceDescriptor(target, plan),)))
    with Context(SPEC=0):
      rewritten = full_rewrite_to_sink(sink, Renderer(Target()), optimize=False)

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual([reg.dtype.size for reg in self._regs(rewritten)], [2])

  def test_full_rewrite_rejects_descriptor_target_removed_before_binding(self):
    r = UOp.range(4, 0, AxisType.REDUCE)
    value = r.cast(dtypes.float32) * UOp.const(dtypes.float32, 0.0) + UOp.const(dtypes.float32, 1.0)
    state0 = UOp.variable("removed_target_state0", -1000.0, 1000.0, dtype=dtypes.float32)
    state1 = UOp.variable("removed_target_state1", -1000.0, 1000.0, dtype=dtypes.float32)
    field0 = CoupledReduceField("sum_value", dtypes.float32, UOp.const(dtypes.float32, 0.0), state0, state0 + value)
    field1 = CoupledReduceField("sum_state", dtypes.float32, UOp.const(dtypes.float32, 0.0), state1, state1 + state0)
    plan = CoupledReducePlan((field0, field1), (r,), state0 + state1)
    target = UOp(Ops.REDUCE, dtypes.float32, (value, r), (Ops.ADD, ()))
    descriptor = CoupledReduceDescriptor(target, plan)
    self.assertIsNone(validate_coupled_reduce_plan(plan, target))

    sink = UOp.sink(target, arg=KernelInfo(name="removed_target_coupled", opts_to_apply=(), coupled_reduce=(descriptor,)))
    for optimize in (False, True):
      with self.subTest(optimize=optimize), Context(SPEC=0), self.assertRaisesRegex(AssertionError, "missing coupled reduce"):
        full_rewrite_to_sink(sink, Renderer(Target()), optimize=optimize)

  def test_full_rewrite_rebinds_descriptor_after_postrange_range_rewrite(self):
    descriptor, g = self._indexed_codegen_descriptor()
    out = UOp.placeholder((2,), dtypes.float32, 1)
    sink = UOp.sink(out.index(g).store(descriptor.target),
                    arg=KernelInfo(name="postrange_coupled", opts_to_apply=(), coupled_reduce=(descriptor,)))
    with Context(SPEC=1):
      rewritten = full_rewrite_to_sink(sink, Renderer(Target()))

    self.assertFalse(any(u.op is Ops.REDUCE for u in rewritten.toposort()))
    self.assertEqual([reg.dtype.size for reg in self._regs(rewritten)], [2])

  def test_to_program_consumes_descriptor_before_linearized_program(self):
    descriptor, scalar_peer = self._codegen_descriptor_and_peer()
    sink = UOp.sink(descriptor.target, scalar_peer,
                    arg=KernelInfo(name="program_coupled", opts_to_apply=(), coupled_reduce=(descriptor,)))
    with Context(SPEC=1):
      program = to_program(sink, PythonRenderer(Target("PYTHON")))

    self.assertEqual(program.op, Ops.PROGRAM)
    self.assertFalse(any(u.op is Ops.REDUCE for u in program.src[0].toposort()))
    self.assertFalse(any(u.op is Ops.REDUCE for u in program.src[2].toposort()))
    self.assertEqual([reg.dtype.size for reg in self._regs(program.src[0])], [2, 1])

  def test_python_program_executes_coupled_recurrence(self):
    descriptor, _ = self._codegen_descriptor_and_peer()
    out = UOp.placeholder((1,), dtypes.float32, 0)
    store = out.index(UOp.const(dtypes.weakint, 0)).store(descriptor.target)
    sink = UOp.sink(store, arg=KernelInfo(name="semantic_coupled", opts_to_apply=(), coupled_reduce=(descriptor,)))
    with Context(SPEC=1):
      program = to_program(sink, PythonRenderer(Target("PYTHON")))
    global_size, local_size = program.arg.launch_dims({})
    raw = bytearray(4)
    PythonProgram(program.arg.name, program.src[4].arg)(memoryview(raw), global_size=global_size, local_size=local_size or (1, 1, 1),
                                                       vals=program.arg.vals({}))
    self.assertAlmostEqual(memoryview(raw).cast("f")[0], 11.0)

  def test_python_program_executes_coupled_and_ordinary_peer(self):
    descriptor, scalar_peer = self._codegen_descriptor_and_peer()
    out = UOp.placeholder((2,), dtypes.float32, 0)
    store0 = out.index(UOp.const(dtypes.weakint, 0)).store(descriptor.target)
    store1 = out.index(UOp.const(dtypes.weakint, 1)).store(scalar_peer)
    sink = UOp.sink(store0, store1, arg=KernelInfo(name="semantic_coupled_peer", opts_to_apply=(), coupled_reduce=(descriptor,)))
    with Context(SPEC=1):
      program = to_program(sink, PythonRenderer(Target("PYTHON")))
    global_size, local_size = program.arg.launch_dims({})
    raw = bytearray(8)
    PythonProgram(program.arg.name, program.src[4].arg)(memoryview(raw), global_size=global_size, local_size=local_size or (1, 1, 1),
                                                       vals=program.arg.vals({}))
    out_vals = memoryview(raw).cast("f")
    self.assertAlmostEqual(out_vals[0], 11.0)
    self.assertAlmostEqual(out_vals[1], 10.0)

  def test_spec_verification_accepts_descriptor_payload(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    sink = UOp.sink(descriptor.target, arg=KernelInfo(name="spec_coupled", coupled_reduce=(descriptor,)))
    with Context(SPEC=2):
      type_verify(sink, spec_tensor)

  def test_spec_verification_rejects_malformed_descriptor_payload(self):
    with Context(SPEC=0):
      sink = UOp.sink(UOp.const(dtypes.float32, 1.0), arg=KernelInfo(name="bad_coupled", coupled_reduce=("bad",)))
    with Context(SPEC=2), self.assertRaisesRegex(RuntimeError, "UOp verification failed"):
      type_verify(sink, spec_tensor)

  def test_spec_verification_rejects_malformed_plan_shape_payload(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    bad_field = replace(descriptor.plan.fields[0], update="bad")
    bad_plan = replace(descriptor.plan, fields=(bad_field, descriptor.plan.fields[1]))
    with Context(SPEC=0):
      sink = UOp.sink(descriptor.target, arg=KernelInfo(name="bad_shape_coupled",
                                                        coupled_reduce=(CoupledReduceDescriptor(descriptor.target, bad_plan),)))
    with Context(SPEC=2), self.assertRaisesRegex(RuntimeError, "UOp verification failed"):
      type_verify(sink, spec_tensor)

  def test_spec_verification_rejects_fake_descriptor_payload(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    fake = type("FakeDescriptor", (), {"target": descriptor.target, "plan": descriptor.plan})()
    with Context(SPEC=0):
      sink = UOp.sink(descriptor.target, arg=KernelInfo(name="fake_coupled", coupled_reduce=(fake,)))
    with Context(SPEC=1), self.assertRaisesRegex(RuntimeError, "UOp verification failed"):
      type_verify(sink, spec_tensor)

  def test_spec_verification_rejects_semantically_invalid_state_payload(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    bad_field = replace(descriptor.plan.fields[0], state=UOp.const(dtypes.float32, 0.0))
    bad_plan = CoupledReducePlan((bad_field, descriptor.plan.fields[1]), descriptor.plan.reduce_ranges, descriptor.plan.final)
    with Context(SPEC=0):
      sink = UOp.sink(descriptor.target, arg=KernelInfo(name="bad_state_coupled",
                                                        coupled_reduce=(CoupledReduceDescriptor(descriptor.target, bad_plan),)))
    with Context(SPEC=2), self.assertRaisesRegex(RuntimeError, "UOp verification failed"):
      type_verify(sink, spec_tensor)

  def test_spec_verification_rejects_semantically_invalid_target_payload(self):
    descriptor, _, _, _ = self._scalar_descriptor_and_peer()
    bad_descriptor = CoupledReduceDescriptor(UOp.const(dtypes.float32, 1.0), descriptor.plan)
    with Context(SPEC=0):
      sink = UOp.sink(descriptor.target, arg=KernelInfo(name="bad_target_coupled", coupled_reduce=(bad_descriptor,)))
    with Context(SPEC=2), self.assertRaisesRegex(RuntimeError, "UOp verification failed"):
      type_verify(sink, spec_tensor)

if __name__ == "__main__":
  unittest.main()
