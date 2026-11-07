import unittest, re, ast as pyast
from dataclasses import replace
from typing import Any
from tinygrad import Tensor
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, UOp, graph_rewrite, ParamArg, KernelInfo, CallInfo, CustomFunction
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.render import render_uir
from tinygrad.device import Device
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Context, ansistrip

# ***** prototype parse for the uir wire format: %id = op src src key=value *****

_DTYPES_BY_NAME: dict[str, DType] = {d.name: d for _,v in vars(type(dtypes)).items() if isinstance(v, DType) for d in [v]}
# a token is a key=, a %ref, a bracket, a quoted string, or a bare word. '#' starts a comment
_tok_re = re.compile(r"""#.*|\w+=|%\d+|[\[\]]|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^\s\[\]]+""")
_line_re = re.compile(r"^%(\d+) = (\w+)$")
_named_vals = {"dtype": lambda v: _DTYPES_BY_NAME[v], "addrspace": lambda v: AddrSpace[v], "axis": lambda v: AxisType[v],
               "op": lambda v: Ops[v.upper()]}
_dataclass_args = {Ops.PARAM: ParamArg, Ops.REFPARAM: ParamArg, Ops.ALLOC: ParamArg, Ops.BUFFER: ParamArg,
                   Ops.SINK: KernelInfo, Ops.CALL: CallInfo, Ops.CUSTOM_FUNCTION: CustomFunction}

def _parse_lit(tok:str) -> Any:
  if tok in ("true", "false"): return tok == "true"
  if re.fullmatch(r"-?\d+", tok): return int(tok)
  if re.fullmatch(r"-?(\d+\.\d*(e[-+]?\d+)?|\d+e[-+]?\d+|inf)|nan", tok): return float(tok)
  return tok

def _parse_const(tok:str) -> UOp:
  if tok == "invalid": return UOp(Ops.CONST, src=(), arg=Invalid)
  return UOp(Ops.CONST, src=(), arg=v if isinstance(v:=_parse_lit(tok), (bool, int)) else dtypes.weakfloat.const(v))

def _parse_val(toks:list[str], bare) -> Any:
  if (t:=toks.pop(0)) == "[":
    items = []
    while toks[0] != "]": items.append(_parse_val(toks, bare))
    toks.pop(0)
    return tuple(items)
  return pyast.literal_eval(t) if t[0] in "'\"" else bare(t)

def parse_ssa(text:str) -> UOp:
  # the wire format carries storage declarations without runtime state: `buffer` reconstructs as an ALLOC
  nodes: dict[int, UOp] = {}
  root = None
  def src_tok(t:str) -> UOp: return nodes[int(t[1:])] if t.startswith("%") else _parse_const(t)
  for raw in ansistrip(text).splitlines():   # op names may carry ANSI color from render_uir
    toks = [t for t in _tok_re.findall(raw) if not t.startswith("#")]
    if not toks: continue
    m = _line_re.match(" ".join(toks[:3]))
    assert m, f"unparseable line {raw!r}"
    n, op, toks = int(m.group(1)), Ops[m.group(2).upper()], toks[3:]
    srcs: list[UOp] = []
    while toks and not toks[0].endswith("="):
      v = _parse_val(toks, src_tok)
      srcs.append(UOp(Ops.STACK, src=v) if isinstance(v, tuple) else v)
    kv: dict[str, Any] = {}
    while toks:
      k = toks.pop(0)[:-1]
      kv[k] = _parse_val(toks, _named_vals.get(k, _parse_lit))
    arg: Any = None
    if op is Ops.RANGE: arg = (kv["axis"], *(kv["id"] if isinstance(kv["id"], tuple) else (kv["id"],)))
    elif op is Ops.REDUCE: arg = (kv["op"], kv.get("pop", 0))
    elif op is Ops.ALLREDUCE: arg = (kv["op"], kv["device"])
    elif op in _dataclass_args and (kv or op is Ops.CALL): arg = _dataclass_args[op](**kv)
    elif kv: (arg,) = kv.values()
    if op is Ops.BUFFER: op = Ops.ALLOC   # the wire carries storage declarations without runtime state
    nodes[n] = root = UOp(op, src=tuple(srcs), arg=arg)   # root is always the last node
  assert root is not None, "empty graph"
  return root

def _strip_buffers(root:UOp) -> UOp:
  # realized BUFFERs carry runtime state the wire can't; a BUFFER with no binding is exactly an ALLOC
  subs = {b: b.replace(op=Ops.ALLOC, arg=replace(b.arg, buffer=None, spec=None))
          for b in root.toposort() if b.op is Ops.BUFFER and isinstance(b.arg, ParamArg)}
  return root.substitute(subs, walk=True, name="strip buffers for wire format test") if subs else root

def assert_roundtrip(case, root:UOp):
  txt = render_uir(root)
  parsed = parse_ssa(txt)
  # text + structural equality, both sides stripped: realized BUFFER vs parsed ALLOC converge to the same thing
  g1, g2 = _strip_buffers(root), _strip_buffers(parsed)
  case.assertEqual(render_uir(g1), render_uir(g2))
  case.assertEqual({x.tuplize for x in g1.toposort()}, {x.tuplize for x in g2.toposort()})

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

  def test_continue_parsing(self):
    # lines out of order / comments tolerated, ids sparse
    g = UOp.sink(UOp.const(1) + UOp.const(2))
    txt = "# hand written\n" + render_uir(g) + " # trailing comment\n\n# last line comment\n"
    assert_roundtrip(self, parse_ssa(txt))

if __name__ == '__main__': unittest.main()
