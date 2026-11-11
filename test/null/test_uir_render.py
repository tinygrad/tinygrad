import unittest, math
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.dtype import dtypes, Invalid, AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, UPat, graph_rewrite, KernelInfo, ProgramInfo
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.device import Device
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.helpers import Context, ansistrip
from tinygrad.renderer import Estimates

def _strip_buffers(root:UOp) -> UOp:
  # realized BUFFERs carry runtime state the wire can't; a BUFFER with no binding is exactly an ALLOC
  subs = {b: b.replace(op=Ops.ALLOC, arg=replace(b.arg, spec=None))
          for b in root.toposort(enter_calls=True) if b.op is Ops.BUFFER and b.slot_buffer is not None}
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
    assert_roundtrip(self, UOp.sink(UOp.param(0, dtypes.int32, name="has space %1"), UOp.param(1, dtypes.int32, name="[x]")))

  def test_program(self):
    # SINK with applied opts, PROGRAM with params as %refs, SOURCE, BINARY bytes, Target
    prg = to_program(self._kernel((Tensor.empty(16, device="NULL") + 1).sum()), Device["NULL"].renderer)
    self.assertIs(prg.op, Ops.PROGRAM)
    assert_roundtrip(self, prg)

  def test_args(self):
    # an inline root, an all-default arg isn't None, a symbolic Estimates
    n = UOp.variable("n", 1, 64)
    for g in [UOp.const(1), UOp.const((1, 2)), UOp(Ops.STACK), UOp(Ops.PROGRAM, (UOp.sink(arg=KernelInfo()),), ProgramInfo()),
              UOp(Ops.PROGRAM, (UOp.sink(),)), UOp.sink(UOp.const(1), arg=KernelInfo(estimates=Estimates(ops=n*2, mem=n)))]:
      assert_roundtrip(self, g)

  def test_nan_pattern(self):
    # the pattern compiler puts a UPat's float arg in a PYLITERAL, any nan is the same one
    from tinygrad.uop.upat import _get_clause
    assert_roundtrip(self, _get_clause(UPat(Ops.CONST, arg=-math.nan), UOp.custom_function("x", fmt=True).call(inline=True)))

  def test_amd_instructions(self):
    # rdna3 and rdna4 instructions have the same names, a python object is evaluated in its own module
    from tinygrad.runtime.autogen.amd.rdna3 import ins as rdna3
    from tinygrad.runtime.autogen.amd.rdna4 import ins as rdna4
    for ins in (rdna3, rdna4): assert_roundtrip(self, UOp(Ops.INS, arg=(ins.v_mov_b32_e32(ins.v[1], 1.0), dtypes.void)))

class TestParseUIR(unittest.TestCase):
  def test_format(self):
    # the exact text, and hand-written text with comments parses to the same UOp
    a, r = UOp.param(0, dtypes.float32, (4,)), UOp.range(4, 0)
    sink = UOp(Ops.END, (UOp(Ops.STORE, (UOp(Ops.INDEX, (a, r)), UOp.const(1.0))), r)).sink(arg=KernelInfo())
    txt = "%0 = param 4 slot=0 dtype=f32\n%1 = range 4 slot=0\n%2 = index %0 %1\n%3 = store %2 1.0\n%4 = end %3 %1\n%5 = sink %4 name=test"
    self.assertEqual(ansistrip(render_uir(sink)), txt)
    self.assertIs(parse_uir("# a comment\n" + txt.replace("\n", "  # a trailing comment\n")), sink)

  def test_bound_buffer_slot_is_preserved(self):
    live = UOp.new_buffer("NULL", 4, dtypes.float32)
    expected = next(UOp.unique_num) + 1
    self.assertIs(parse_uir(render_uir(live)), live)
    self.assertEqual(next(UOp.unique_num), expected)

  def test_unbound_nonnegative_buffer_slot_is_rejected(self):
    for slot in (next(UOp.unique_num), 10**30):
      for arg in (f"slot={slot} dtype=f32 device=NULL", f"arg=`ParamArg({slot}, dtypes.float32, device='NULL')`"):
        with self.assertRaisesRegex(AssertionError, "negative slot"): parse_uir(f"%0 = buffer 4 {arg}")

  def test_buffer_does_not_capture_future_allocation(self):
    for size, dtype in [(4, dtypes.float32), (8, dtypes.int32)]:
      parsed = parse_uir("%0 = buffer 4 slot=-1 dtype=f32 device=NULL")
      live = UOp.new_buffer("NULL", size, dtype)
      self.assertNotEqual(parsed.arg.slot, live.arg.slot)
      self.assertIsNone(parsed.slot_buffer)
      self.assertIsNotNone(live.buffer)

  def test_buffer_aliases_are_preserved(self):
    text = """
      %a = buffer 4 slot=-7 dtype=f32 device=NULL
      %b = { %view = buffer 4 arg=`ParamArg(-7, dtypes.int32, device='NULL')` }
      %c = buffer 4 slot=-8 dtype=f32 device=NULL
      %d = buffer 4 slot=-7 dtype=f32 device=NULL
      %out = sink %a %b %c %d
    """
    root = parse_uir(text)
    a, b, c, d = root.src
    self.assertEqual(a.arg.slot, b.arg.slot)
    self.assertNotEqual(a.arg.slot, c.arg.slot)
    self.assertIs(a, d)
    for buf in root.src: self.assertIsNone(buf.slot_buffer)
    self.assertIs(parse_uir(text), root)

  def test_negative_buffer_slots_are_preserved(self):
    for slot in (-1, -1234):
      expected = next(UOp.unique_num) + 1
      parsed = parse_uir(f"%0 = buffer 4 slot={slot} dtype=f32 device=NULL")
      self.assertEqual(parsed.arg.slot, slot)
      self.assertEqual(next(UOp.unique_num), expected)
      self.assertIsNone(parsed.slot_buffer)
      self.assertIs(parse_uir(render_uir(parsed)), parsed)

  def test_scratch_and_parameter_slots_are_preserved(self):
    for addrspace in (AddrSpace.LOCAL, AddrSpace.REG):
      buf = UOp._buffer(4, dtypes.float32, 7, addrspace=addrspace)
      self.assertIs(parse_uir(render_uir(buf)), buf)
    for op in ("param", "alloc"):
      parsed = parse_uir(f"%0 = {op} 4 slot=7 dtype=f32 device=NULL")
      self.assertEqual(parsed.arg.slot, 7)
      self.assertIs(parse_uir(render_uir(parsed)), parsed)

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
    for txt in ["", "%0 = add %1 1", "%0 = add 1 2 x=1 4", "%0 = frob 1", "%0 = add 1 nope", "%0 = add 1 (mul 2 3", "%0 = [1 2", "%0 = add load 1",
                "%x = 1\n%x = 2", "%x = 1 }", "%x = special name=`12", "%x = special name=)", "%x = cast 1 dtpye=i32",
                "%x = cast 1 dtype=i32 dtype=f32", "%x = { %x = 1 }", "%x = add %x 1"]:
      with self.assertRaises(ValueError, msg=txt): parse_uir(txt)

if __name__ == '__main__': unittest.main()
