from typing import List, Tuple
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, Ops, UnaryOps, TernaryOps, BinaryOps, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.helpers import strip_parens
import math

def fixup_binops(c,a,b):
  if c.arg == BinaryOps.CMPLT and a.dtype == dtypes.bool: return UOp(c.op, c.dtype, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg)
  if c.arg in (BinaryOps.MAX, BinaryOps.XOR) and c.dtype == dtypes.bool:
    return UOp(c.op, dtypes.int, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg).cast(dtypes.bool)

wgsl_matcher = PatternMatcher([
  (UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL), dtype=dtypes.bool.ptr(), name="a"), lambda a: UOp(a.op, dtypes.int32.ptr(), a.src, a.arg)),
  (UPat(Ops.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("g", dtype=dtypes.bool))),
    lambda root,x,y,g: UOp(root.op, dtypes.int, (x,y.cast(dtypes.int), g), root.arg).cast(dtypes.bool)),
  (UPat(Ops.LOAD, name="root", dtype=dtypes.bool, src=(UPat())), lambda root: UOp(root.op, dtypes.int, root.src, root.arg).cast(dtypes.bool)),
  (UPat(Ops.ALU, src=(UPat(name="a"), UPat(name="b")), name="c"), fixup_binops),
  *[(UPat(Ops.ALU, src=(UPat(name="b", dtype=(dtypes.uint, dtypes.int, dtypes.bool))), arg=a, name="a"),
     lambda a,b: UOp(a.op, dtypes.float, (b.cast(dtypes.float),), a.arg).cast(b.dtype))
    for a in (UnaryOps.EXP2, UnaryOps.SIN, UnaryOps.LOG2, UnaryOps.SQRT)],
  (UPat.store(UPat.var("bidx"), UPat.var("var", dtype=dtypes.bool), UPat.var("gate")),
   lambda bidx,val,gate: UOp.store(bidx, val.cast(dtypes.int), gate)),
  (UPat.store(UPat.var("bidx"), UPat.var("var", dtype=dtypes.bool)), lambda bidx,var: UOp.store(bidx, var.cast(dtypes.int))),
  (UPat(Ops.ALU, name="m", arg=BinaryOps.MAX), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  (UPat(Ops.ALU, name="m", src=(UPat(name="a"), UPat(Ops.ALU, arg=TernaryOps.WHERE, src=(UPat.var("g"), \
    UPat(op=Ops.CONST, name="c1"), UPat(op=Ops.CONST, name="c2")))), dtype=dtypes.ulong,  arg=BinaryOps.MUL), \
    lambda m,a,g,c1,c2: UOp(Ops.ALU, arg=TernaryOps.WHERE, dtype=m.dtype, src=(g, a << 32, UOp.const(dtype=m.dtype, b=0))) \
    if c1.arg == 4294967296 and c2.arg == 0 else None),
  ])

type_map = {dtypes.float: "f32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool", dtypes.ulong: "vec2<u32>"}

def render_load_store(r, bidx):
  sbidx = strip_parens(r[bidx])
  buf, idx = sbidx.split("+")[0], '+'.join(sbidx.split("+")[1:])
  return f"{buf}[{idx}]"

def render_ushift(r, v, am, left):
  v, am = r[v], f"{r[am]}.x" if am.dtype == dtypes.ulong else f"{r[am]}"
  return f"select(vec2<u32>({v}.x << {am}, ({v}.y << {am}) | ({v}.x >> (32u-{am}))), vec2<u32>(0u, {v}.x << ({am}-32u)), {am} >= 32u)" \
  if left else f"select(vec2<u32>(({v}.x >> {am}) | ({v}.y << (32u - {am})), {v}.y >> {am}), vec2<u32>({v}.y >> ({am} - 32u), 0u), {am} >= 32u)"

class WGSLRenderer(CStyleLanguage):
  device = "WEBGPU"
  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  extra_matcher = wgsl_matcher
  external_local_bufs = True
  supports_float4 = False
  barrier = "workgroupBarrier();"
  code_for_op = {**CStyleLanguage.code_for_op, TernaryOps.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})"}
  nan = "nan()"
  type_map = type_map

  string_rewrite = PatternMatcher([
    (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "true" if x.arg else "false"),
    (UPat(Ops.CONST, dtype=dtypes.uint32, name="x"), lambda ctx,x: f"bitcast<u32>({x.arg}i)" if x.arg < 0 else f"{x.arg&0xFFFFFFFF}u"),
    (UPat(Ops.CONST, dtype=dtypes.ulong, name="x"), lambda ctx,x: f"vec2<u32>({x.arg}, 0u)" if x.arg <= 0xFFFFFFFF \
     else  f"vec2<u32>({x.arg&0xFFFFFFFF}, {x.arg>>32})"),
    (UPat(Ops.CONST, arg=math.inf, name="x"), lambda ctx, x: f"{type_map[x.dtype]}({ctx.infinity})"),
    (UPat(Ops.CONST, arg=-math.inf, name="x"), lambda ctx, x: f"{type_map[x.dtype]}(-{ctx.infinity})"),
    (UPat(Ops.CONST, dtype=dtypes.floats, name="x"), lambda ctx,x: f"({type_map[x.dtype]}({ctx.nan}))" if math.isnan(x.arg) else None),
    (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"var<workgroup> {ctx[x]}: array<{type_map[x.dtype.base]}, {x.arg[1]}>;"),
    (UPat(Ops.CAST, name="x"), lambda ctx,x: f"vec2<u32>(({ctx[x.src[0]]})&4294967295, 0u)" if x.dtype == dtypes.uint64 \
      else f"{type_map[x.dtype]}({ctx[x.src[0]]}.x)" if x.src[0].dtype == dtypes.uint64 else f"{type_map[x.dtype]}({ctx[x.src[0]]})"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]})"),
    (UPat(Ops.LOAD, src=(UPat.var("bidx"), UPat.var('var'), UPat.var("gate"))),
      lambda ctx,bidx,var,gate: f"select({ctx[var]}, {render_load_store(ctx, bidx)}, {ctx[gate]})"),
    (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,bidx: f"{render_load_store(ctx, bidx)}"),
    (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True),
     lambda ctx,bidx,var: f"{render_load_store(ctx,bidx)} = {ctx[var]};"), ]) + base_rewrite + PatternMatcher([
    (UPat(Ops.ALU, name="x", dtype=dtypes.ulong, arg=BinaryOps.SHL), lambda ctx,x: render_ushift(ctx, x.src[0], x.src[1], left=True)),
    (UPat(Ops.ALU, name="x", dtype=dtypes.ulong, arg=BinaryOps.SHR), lambda ctx,x: render_ushift(ctx, x.src[0], x.src[1], left=False)),
  ])

  def render_dtype(self, dt:DType, mutable=True) -> str: return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [num for _, num in sorted([u.arg for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == 'l'], key=lambda x: x[0])]
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    prg += "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n"
    prg += "\n".join((prefix or [])+[f"@group(0) @binding({next(bind_it)+1}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype.base]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg
