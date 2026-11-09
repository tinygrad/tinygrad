import unittest, re, math
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.dtype import dtypes, Invalid
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, UPat, graph_rewrite, ParamArg, KernelInfo, ProgramInfo
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.device import Device
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.helpers import Context, ansistrip, Target
from tinygrad.renderer import Estimates

def _strip_buffers(root:UOp) -> UOp:
  # realized BUFFERs carry runtime state the wire can't; a BUFFER with no binding is exactly an ALLOC
  subs = {b: b.replace(op=Ops.ALLOC, arg=replace(b.arg, buffer=None, spec=None))
          for b in root.toposort(enter_calls=True) if b.op is Ops.BUFFER and isinstance(b.arg, ParamArg) and b.arg.buffer is not None}
  return root.substitute(subs, walk=True, enter_calls=True, name="strip buffers for wire format test") if subs else root

def assert_roundtrip(case, root:UOp):
  # UOps are hash-consed, so a lossless round trip gives back the identical UOp
  g = _strip_buffers(root)
  case.assertIs(parse_uir(render_uir(g)), g, render_uir(g))

class TestSSARender(unittest.TestCase):
  def test_gemm_pre_rangeify(self):
    a = Tensor.rand(256, 256, device="CPU").realize()
    b = Tensor.rand(256, 256, device="CPU").realize()
    root = graph_rewrite((a.reshape(256,256,1) * b.reshape(256,1,256)).sum(0).uop, mop_cleanup)
    assert_roundtrip(self, root)

  def _kernel(self, t:Tensor) -> UOp:
    lin, _ = Tensor.linear_with_vars(t)
    return [c for c in lin.src if any(u.op is Ops.REDUCE for u in c.src[0].toposort())][0].src[0]

  def test_conv_kernel_graph(self):
    o = Tensor.rand(1,1,6,6, device="CPU").conv2d(Tensor.rand(1,1,3,3, device="CPU"))
    assert_roundtrip(self, self._kernel(o))

  def test_conv_noopt(self):
    o = Tensor.rand(1,1,6,6, device="CPU").conv2d(Tensor.rand(1,1,3,3, device="CPU"))
    with Context(NOOPT=1): sink = full_rewrite_to_sink(self._kernel(o), Device["CPU"].renderer, optimize=False)
    assert_roundtrip(self, sink)

  def test_gemm_kernel_graph(self):
    o = Tensor.rand(4,4, device="CPU").matmul(Tensor.rand(4,4, device="CPU"))
    assert_roundtrip(self, self._kernel(o))

  def test_symbolic_variable(self):
    n = UOp.variable("n", 1, 64)
    g = UOp.sink(((n + 1) * 2) % 7, (n % 5) + UOp.const(Invalid))
    assert_roundtrip(self, g)

  def test_const_typing(self):
    g = UOp.sink(UOp.const(True), UOp.const(1), UOp.const(-5, dtypes.int32), UOp.const(2.0, dtypes.float32),
                 UOp.const(-0.0, dtypes.float32), UOp.const(0.1, dtypes.float64), UOp.const(Invalid),
                 UOp.const(12345678901234, dtypes.int64))
    assert_roundtrip(self, g)
    # exact float roundtrip: every representable hex float must come back bit-equal
    import random
    random.seed(0)
    vs = [random.uniform(-1e6, 1e6) for _ in range(20)]
    g2 = UOp.sink(*[UOp.const(v, dtypes.float32) for v in vs])
    assert_roundtrip(self, g2)

  def test_where_max_bool(self):
    a = UOp.param(0, dtypes.int32)
    b = (a < UOp.const(3, dtypes.int32)) & (a != 1)   # bool-AND must lower to MUL on render
    g = UOp.sink(b.where(a.maximum(UOp.const(1, dtypes.int32)), a.cast(dtypes.float32).reciprocal()))
    assert_roundtrip(self, g)

  def test_complex_programs(self):
    # softmax over a 256x16 row: reduce-max + reduce-sum + exp2
    x = Tensor.rand(256, 16, device="CPU").realize()
    o = x.softmax(-1)
    assert_roundtrip(self, o.uop)
    assert len(o.uop.toposort()) > 10  # not trivially small

  def test_bound_variable(self):
    v = UOp.variable("n", 1, 10).bind(3)
    assert_roundtrip(self, (Tensor.empty(10, device="CPU")[:v] + 1).uop)

  def test_multi_device(self):
    x = Tensor.empty(4, 4, device="CPU").shard(("CPU:0", "CPU:1"), axis=0)
    assert_roundtrip(self, (x + 1).sum().uop)

  def test_quoted_args(self):
    g = UOp.sink(UOp.param(0, dtypes.int32, name="has space %1"), UOp.param(1, dtypes.int32, name="[x]"))
    self.assertIn("name='has space %1'", ansistrip(render_uir(g)))
    assert_roundtrip(self, g)

  def test_program(self):
    # SINK with applied opts, PROGRAM with params as %refs, SOURCE, BINARY bytes, Target
    prg = to_program(self._kernel((Tensor.empty(16, device="NULL") + 1).sum()), Device["NULL"].renderer)
    self.assertIs(prg.op, Ops.PROGRAM)
    assert_roundtrip(self, prg)

  def test_comments(self):
    g = UOp.sink(UOp.const(1) + UOp.const(2))
    txt = "# hand written\n" + render_uir(g) + " # trailing comment\n\n# last line comment\n"
    assert_roundtrip(self, parse_uir(txt))

class TestFormat(unittest.TestCase):
  def test_exact(self):
    a, r = UOp.param(0, dtypes.float32, (4,)), UOp.range(4, 0)
    sink = UOp(Ops.END, (UOp(Ops.STORE, (UOp(Ops.INDEX, (a, r)), UOp.const(1.0))), r)).sink(arg=KernelInfo())
    self.assertEqual(ansistrip(render_uir(sink)), "%0 = param 4 slot=0 dtype=f32\n%1 = range 4 slot=0\n%2 = index %0 %1\n%3 = store %2 1.0\n"
                                                  "%4 = end %3 %1\n%5 = sink %4 name=test")

class TestPythonObjects(unittest.TestCase):
  def test_uop_in_arg(self):
    # a symbolic Estimates holds UOps that aren't in the graph: they get lines and are %refs in the arg
    n = UOp.variable("n", 1, 64)
    sink = UOp.sink(UOp.const(1), arg=KernelInfo(name="k", estimates=Estimates(ops=n*2, mem=n)))
    self.assertIn("Estimates(ops=%", ansistrip(render_uir(sink)))
    assert_roundtrip(self, sink)
    self.assertEqual(render_uir(list(sink.toposort())), render_uir(sink))

  def test_args(self):
    # an inline root, an all-default arg (not None), and python objects
    n = UOp.variable("n", 1, 64)
    for g in [UOp.const(1), UOp.const(2.5), UOp.const((1, 2)), UOp(Ops.STACK), UOp(Ops.PROGRAM, (UOp.sink(arg=KernelInfo()),), ProgramInfo()),
              UOp(Ops.PROGRAM, (UOp.sink(),)), UOp(Ops.PYLITERAL, arg=UOp.const(1.5)), UOp(Ops.PYLITERAL, arg=frozenset({n})),
              UOp(Ops.PYLITERAL, arg=Target("CPU")), UOp(Ops.PYLITERAL, arg=ProgramInfo(target=Target("CPU", "LLVM")))]:
      assert_roundtrip(self, g)

  def test_ambiguous_names(self):
    # rdna3 and rdna4 define the same instruction names, so the instruction is evaluated in its own module
    from tinygrad.runtime.autogen.amd.rdna3 import ins as rdna3
    from tinygrad.runtime.autogen.amd.rdna4 import ins as rdna4
    for ins in (rdna3, rdna4):
      g = UOp(Ops.INS, arg=(ins.v_mov_b32_e32(ins.v[1], 1.0), dtypes.void))
      self.assertIn(f"`{ins.__name__}:v_mov_b32_e32", ansistrip(render_uir(g)))
      assert_roundtrip(self, g)

  def test_nan(self):
    # a nan literal is a ConstFloat (nan == nan), in a CONST and in the pattern compiler's clauses
    from tinygrad.uop.upat import _get_clause
    assert_roundtrip(self, _get_clause(UPat(Ops.CONST, arg=math.nan), UOp.custom_function("x", fmt=True).call(inline=True)))
    assert_roundtrip(self, UOp.sink(UOp.const(math.nan), UOp.const(-math.inf)))

class TestParseUIR(unittest.TestCase):
  def test_handwritten(self):
    sink = parse_uir("""
      # a+1 over 4 elements
      %0 = param 4 slot=0 dtype=f32
      %1 = range 4 slot=0
      %2 = index %0 %1
      %3 = param 4 slot=1 dtype=f32
      %4 = index %3 %1   # comments can trail a line
      %5 = add %4 1.0
      %6 = store %2 %5
      %7 = end %6 %1
      %8 = sink %7 name=add_one
    """)
    a, b, r = UOp.param(0, dtypes.float32, (4,)), UOp.param(1, dtypes.float32, (4,)), UOp.range(4, 0)
    self.assertIs(sink, a.index(r).store(b.index(r) + 1.0).end(r).sink(arg=KernelInfo(name="add_one")))

  def test_studentgrad_syntax(self):
    # named values, nested (op ...), [stacks] of expressions, {scoped blocks}, hex and exponent literals
    sink = parse_uir("""
      %buf   = param 4 slot=0 dtype=i32   # comment
      %i     = range 4 slot=0
      %inc   = { %one = 0x1
                 %result = (add (index [10 20 30 40] %i) %one) }
      %store = store (index %buf %i) (add %inc (cast 1E2 dtype=i32))
      %end   = end %store %i
    """)
    buf, i = UOp.param(0, dtypes.int32, (4,)), UOp.range(4, 0)
    vals = UOp(Ops.STACK, tuple(UOp.const(v) for v in (10, 20, 30, 40)))
    inc = UOp(Ops.ADD, (UOp(Ops.INDEX, (vals, i)), UOp.const(1)))   # raw UOps: the parser builds exactly what is written
    val = UOp(Ops.ADD, (inc, UOp(Ops.CAST, (UOp.const(100.0),), dtypes.int32)))
    self.assertIs(sink, UOp(Ops.END, (UOp(Ops.STORE, (UOp(Ops.INDEX, (buf, i)), val)), i)))

  def test_errors(self):
    for txt, err in [("%0 = add %1 1", "%1 is not defined"), ("%0 = add 1 2 x=1 4", "srcs must come before key=value"),
                     ("%0 = frob 1", "unknown op 'frob'"), ("%0 = add 1 nope", "not a %name or a literal"), ("", "expected an assignment"),
                     ("%0 = add 1 (mul 2 3", "expected ')'"), ("%0 = [1 2", "expected ']'"), ("%0 = add load 1", "write (load ...)"),
                     ("%x = 1\n%x = 2", "%x is already defined"), ("%x = 1 }", "expected 'end of input'"),
                     ("%x = special name=`12", "unterminated ` in name="), ("%x = special name=)", "expected a value for name="),
                     ("%x = cast 1 dtpye=i32", "cast has no ['dtpye']"), ("%x = cast 1 dtype=i32 dtype=f32", "cast has dtype= twice"),
                     ("%x = { %x = 1 }", "%x is already defined"), ("%x = add %x 1", "%x is not defined")]:
      with self.assertRaisesRegex(ValueError, re.escape(err)): parse_uir(txt)

if __name__ == '__main__': unittest.main()
