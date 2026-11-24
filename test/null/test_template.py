import unittest

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.codegen import create_linear, full_rewrite_to_sink, to_program
from tinygrad.dtype import AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import KernelInfo, ProgramInfo, pm_resolve_params, graph_rewrite
from tinygrad.uop.spec import spec_program, spec_tensor, type_verify
from tinygrad.uop.uir import parse_uir, render_uir


class TestTemplateParams(unittest.TestCase):
  def test_template_representation_and_no_separate_op(self):
    self.assertFalse(hasattr(Ops, "TEMPLATE"))
    n = UOp.template("N")
    self.assertIs(n.op, Ops.PARAM)
    self.assertIs(n.addrspace, AddrSpace.TEMPLATE)
    self.assertEqual(n.arg.slot, -1)
    self.assertEqual(n.arg.name, "N")
    self.assertEqual(n.dtype, dtypes.int32)
    self.assertIsNone(n.arg.vmin_vmax)
    self.assertEqual(n.shape, ())
    self.assertTrue(n.is_template)
    self.assertTrue(n.has_templates)
    self.assertEqual(n.templates(), [n])

  def test_scalar_int_and_float_expressions(self):
    n = UOp.template("N", (1, 128), dtype=dtypes.int32)
    m = UOp.template("M", (0, 64), dtype=dtypes.uint32)
    w = UOp.template("W", (1, 16), dtype=dtypes.weakint)
    eps = UOp.template("eps", (0.0, 1.0), dtype=dtypes.float32)
    scale = UOp.template("scale", dtype=dtypes.float16)
    wf = UOp.template("wf", (-10.0, 10.0), dtype=dtypes.weakfloat)

    iexpr = (n + 2) * 3
    self.assertEqual(iexpr.specialize(N=4), UOp.const(18, dtypes.int32))
    self.assertEqual((m + 1).specialize(M=7), UOp.const(8, dtypes.uint32))
    self.assertEqual((w * 2 + 1).specialize(W=5), UOp.const(11))

    fexpr = (eps * 2.0 + 1.0)
    self.assertEqual(fexpr.specialize(eps=0.25), UOp.const(1.5, dtypes.float32))
    self.assertEqual((scale * 2.0).specialize(scale=1.5), UOp.const(3.0, dtypes.float16))
    self.assertEqual((scale * 2.0).specialize(scale=2), UOp.const(4.0, dtypes.float16))
    self.assertEqual((wf + 0.5).specialize(wf=-1.5), UOp.const(-1.0))

    # Single-point template must NOT be folded away before specialize()
    fixed = UOp.template("K", (4, 4), dtype=dtypes.int32)
    self.assertIs(fixed.simplify(), fixed)
    self.assertEqual((fixed + 1).simplify().templates(), [fixed])
    self.assertEqual((fixed + 1).specialize(K=4), UOp.const(5, dtypes.int32))

  def test_param_and_alloc_shapes_exact(self):
    t = UOp.template("T", (1, 64))
    d = UOp.template("D", (32, 4096))
    unbounded = UOp.template("U")

    # 1D PARAM keeps exact symbolic size
    p1 = UOp.param(0, dtypes.float32, (t,))
    self.assertIs(p1.op, Ops.PARAM)
    self.assertEqual(p1.shape, (t,))
    self.assertIs(p1.src[0], t)
    self.assertIs(p1.specialize(T=12), UOp.param(0, dtypes.float32, (12,)))

    # 2D PARAM keeps exact symbolic product size (not vmax*vmax, no SHRINK)
    p2 = UOp.param(1, dtypes.float16, (t, d))
    self.assertIs(p2.op, Ops.RESHAPE)
    self.assertIs(p2.src[0].op, Ops.PARAM)
    self.assertEqual(p2.shape, (t, d))
    self.assertTrue(p2.src[0].src[0].has_templates)
    self.assertEqual(p2.src[0].shape, (t * d,))
    spec_p2 = p2.specialize(T=4, D=128)
    self.assertIs(spec_p2, UOp.param(1, dtypes.float16, (4, 128)))
    self.assertEqual(spec_p2.shape, (4, 128))
    self.assertEqual(spec_p2.src[0].shape, (512,))

    # Derived symbolic shape expression + unbounded template in ALLOC
    a = UOp.alloc((unbounded, d // 32, 8), dtypes.uint32, slot=7, device="NULL")
    self.assertEqual(a.shape, (unbounded, d // 32, 8))
    self.assertTrue(a.src[0].src[0].has_templates)
    spec_a = a.specialize(U=2, D=256)
    self.assertIs(spec_a, UOp.alloc((2, 8, 8), dtypes.uint32, slot=7, device="NULL"))
    self.assertEqual(spec_a.shape, (2, 8, 8))
    self.assertEqual(spec_a.src[0].shape, (128,))

    # Mixed runtime Variable + template: Variable collapses to vmax with SHRINK, template stays exact
    v = UOp.variable("v", 1, 16)
    pmix = UOp.param(0, dtypes.float32, (v, d))
    self.assertEqual(pmix.shape, (v, d))
    self.assertEqual(pmix.variables(), [v])
    self.assertEqual(pmix.templates(), [d])
    spec_mix = pmix.specialize(D=64)
    self.assertIs(spec_mix, UOp.param(0, dtypes.float32, (v, 64)))
    self.assertEqual(spec_mix.variables(), [v])
    self.assertEqual(spec_mix.templates(), [])

    # Valid against spec_tensor before and after specialization
    type_verify(p2, spec_tensor)
    type_verify(a, spec_tensor)
    type_verify(spec_p2, spec_tensor)
    type_verify(spec_a, spec_tensor)

  def test_uir_roundtrip_preserves_templates(self):
    n = UOp.template("N", (1, 128), dtype=dtypes.int32)
    d = UOp.template("D", dtype=dtypes.int32)
    eps = UOp.template("eps", (0.0, 1.0), dtype=dtypes.float32)

    out = UOp.param(0, dtypes.float32, (n, d))
    inp = UOp.param(1, dtypes.float32, (n, d))
    r0, r1 = UOp.range(n, 0), UOp.range(d, 1)
    kern = out[r0, r1].store(inp[r0, r1].load() + eps).end(r0, r1).sink(arg=KernelInfo(name="add_eps", opts_to_apply=()))

    # Outer graph with ALLOC + CALL into templated kernel body
    ext_in = UOp.param(0, dtypes.float32, (n, d), device="NULL")
    ext_out = UOp.alloc((n, d), dtypes.float32, slot=10, device="NULL")
    outer = ext_out.after(kern.call(ext_out, ext_in))

    for g in (n, d, eps, out, kern, outer):
      txt = render_uir(g)
      self.assertIn("addrspace=TEMPLATE", txt)
      self.assertIs(parse_uir(txt), g)

    # Round-tripped graph specializes identically to direct graph
    parsed_kern = parse_uir(render_uir(kern))
    spec_k = parsed_kern.specialize(N=4, D=8, eps=1e-5)
    self.assertFalse(spec_k.has_templates)
    self.assertEqual(spec_k.templates(), [])

    parsed_outer = parse_uir(render_uir(outer))
    spec_outer = parsed_outer.specialize(N=4, D=8, eps=1e-5)
    self.assertFalse(spec_outer.has_templates)
    self.assertEqual(spec_outer.shape, (4, 8))

  def test_excluded_from_runtime_variables(self):
    t = UOp.template("T", (1, 32))
    v = UOp.variable("v", 1, 32)
    expr = t + v
    self.assertFalse(t.is_variable)
    self.assertFalse(t.is_bound_var)
    self.assertEqual(t.variables(), [])
    self.assertEqual(expr.variables(), [v])
    self.assertEqual(expr.templates(), [t])
    with self.assertRaises(AssertionError): t.bind(4)
    # pm_resolve_params only resolves slot >= 0, never touches templates
    resolved = graph_rewrite(expr, pm_resolve_params, ctx=(UOp.const(99),), walk=True)
    self.assertIs(resolved, expr)

  def test_missing_unknown_range_and_type_validation(self):
    # Construction validation
    with self.assertRaises(ValueError): UOp.template("")
    with self.assertRaises(ValueError): UOp.template(123)  # type: ignore
    with self.assertRaises(TypeError): UOp.template("b", dtype=dtypes.bool)
    with self.assertRaises(TypeError): UOp.template("v", dtype=dtypes.void)
    with self.assertRaises(TypeError): UOp.template("x", (False, True), dtype=dtypes.int32)
    with self.assertRaises(TypeError): UOp.template("x", (1.0, 2.0), dtype=dtypes.int32)
    with self.assertRaises(ValueError): UOp.template("x", (10, 5), dtype=dtypes.int32)
    with self.assertRaises(ValueError): UOp.template("x", (-1, 10), dtype=dtypes.uint32)
    with self.assertRaises(ValueError): UOp.template("f", (float("nan"), 1.0), dtype=dtypes.float32)
    with self.assertRaises(ValueError): UOp.template("f", (2.0, 1.0), dtype=dtypes.float32)

    n = UOp.template("N", (1, 16), dtype=dtypes.int32)
    b8 = UOp.template("B8", dtype=dtypes.int8)
    eps = UOp.template("eps", (0.0, 1.0), dtype=dtypes.float32)
    g = n.cast(dtypes.float32) + eps

    # Missing / unknown values
    with self.assertRaises(ValueError): g.specialize(N=4)
    with self.assertRaises(ValueError): g.specialize(N=4, eps=0.5, extra=1)
    self.assertIs(UOp.const(5).specialize(), UOp.const(5))
    with self.assertRaises(ValueError): UOp.const(5).specialize(N=1)

    # Type validation
    with self.assertRaises(TypeError): n.specialize(N=True)
    with self.assertRaises(TypeError): n.specialize(N=2.0)
    with self.assertRaises(TypeError): n.specialize(N="4")  # type: ignore
    with self.assertRaises(TypeError): eps.specialize(eps=False)
    with self.assertRaises(TypeError): eps.specialize(eps="0.1")  # type: ignore

    # Range validation
    with self.assertRaises(ValueError): n.specialize(N=0)
    with self.assertRaises(ValueError): n.specialize(N=17)
    with self.assertRaises(ValueError): b8.specialize(B8=200)
    with self.assertRaises(ValueError): eps.specialize(eps=-0.1)
    with self.assertRaises(ValueError): eps.specialize(eps=1.5)
    with self.assertRaises(ValueError): eps.specialize(eps=float("nan"))

    # Invalid specialized shapes
    u = UOp.template("U")
    p = UOp.param(0, dtypes.float32, (u, 4))
    with self.assertRaises(ValueError): p.specialize(U=-2)
    p_bad_reshape = UOp.param(0, dtypes.float32, (u,)).reshape((4, 8))
    with self.assertRaises(ValueError): p_bad_reshape.specialize(U=16)

  def test_unresolved_templates_rejected_at_codegen_boundary(self):
    n = UOp.template("N", (1, 16))
    eps = UOp.template("eps", (0.0, 1.0), dtype=dtypes.float32)
    out = UOp.param(0, dtypes.float32, (n,))
    inp = UOp.param(1, dtypes.float32, (n,))
    r = UOp.range(n, 0)
    sink = out[r].store(inp[r].load() + eps).end(r).sink(arg=KernelInfo(name="tmpl_add", opts_to_apply=()))

    ren = Device["CPU"].renderer
    with self.assertRaisesRegex(RuntimeError, "unresolved template.*N.*eps"):
      full_rewrite_to_sink(sink, ren, optimize=False)
    with self.assertRaisesRegex(RuntimeError, "unresolved template.*N.*eps"):
      to_program(sink, ren)
    with self.assertRaisesRegex(RuntimeError, "unresolved template"):
      ProgramInfo.from_sink(sink)
    with self.assertRaises(RuntimeError):
      type_verify(sink, spec_program)

    # Schedule / create_linear boundary also rejects unresolved templates (including inside CALL bodies)
    fn_in = UOp.param(0, dtypes.float32, (n,), device="CPU")
    fn_out = UOp.alloc((n,), dtypes.float32, slot=1, device="CPU")
    fn = UOp.sink(fn_out.after(sink.call(fn_out, fn_in)))
    with self.assertRaisesRegex(RuntimeError, "unresolved template"):
      create_linear(fn)

    # After specialization, codegen and execution on CPU succeed cleanly
    spec_sink = parse_uir(render_uir(sink)).specialize(N=4, eps=0.5)
    x = Tensor([1.0, 2.0, 3.0, 4.0], device="CPU").realize()
    y = Tensor.empty(4, dtype=dtypes.float32, device="CPU")
    call = spec_sink.call(y.uop, x.uop)
    got = Tensor(y.uop.after(call)).numpy()
    self.assertEqual(got.tolist(), [1.5, 2.5, 3.5, 4.5])


if __name__ == "__main__":
  unittest.main()
