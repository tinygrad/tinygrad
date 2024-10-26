from typing import List, Tuple
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, UOps, UnaryOps, TernaryOps, BinaryOps, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.helpers import strip_parens

def fixup_binops(c,a,b):
  if c.arg == BinaryOps.CMPLT and a.dtype == dtypes.bool: return UOp(c.op, c.dtype, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg)
  if c.arg in (BinaryOps.MAX, BinaryOps.XOR) and c.dtype == dtypes.bool:
    return UOp(c.op, dtypes.int, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg).cast(dtypes.bool)

wgsl_matcher = PatternMatcher([
  (UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), dtype=dtypes.bool.ptr(), name="a"), lambda a: UOp(a.op, dtypes.int32.ptr(), a.src, a.arg)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("g", dtype=dtypes.bool))),
    lambda root,x,y,g: UOp(root.op, dtypes.int, (x,y.cast(dtypes.int), g), root.arg).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat())), lambda root: UOp(root.op, dtypes.int, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.ALU, src=(UPat(name="a"), UPat(name="b")), name="c"), fixup_binops),
  *[(UPat(UOps.ALU, src=(UPat(name="b", dtype=(dtypes.uint, dtypes.int, dtypes.bool))), arg=a, name="a"),
     lambda a,b: UOp(a.op, dtypes.float, (b.cast(dtypes.float),), a.arg).cast(b.dtype))
    for a in (UnaryOps.EXP2, UnaryOps.SIN, UnaryOps.LOG2, UnaryOps.SQRT)],
  (UPat.store(UPat.var("bidx"), UPat.var("var", dtype=dtypes.bool), UPat.var("gate")),
   lambda bidx,val,gate: UOp.store(bidx, val.cast(dtypes.int), gate)),
  (UPat.store(UPat.var("bidx"), UPat.var("var", dtype=dtypes.bool)), lambda bidx,var: UOp.store(bidx, var.cast(dtypes.int))),
  (UPat(UOps.ALU, name="m", arg=BinaryOps.MAX), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
  # This has to be constrained to just the "a * 2 ** 32", and "a // 2 ** 32" cases
  (UPat(UOps.ALU, name="x", dtype=dtypes.ulong,  arg=BinaryOps.MUL), lambda x: UOp(x.op, x.dtype, \
    (x.src[0], UOp(UOps.CONST, dtypes.uint32, arg=32)), BinaryOps.SHL)),
  (UPat(UOps.ALU, name="x", dtype=dtypes.ulong,  arg=BinaryOps.IDIV), lambda x: UOp(x.op, x.dtype, \
    (x.src[0], UOp(UOps.CONST, dtypes.uint32, arg=32)), BinaryOps.SHR)),
])

type_map = {dtypes.float: "f32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool", dtypes.ulong: "vec2<u32>"}

def gep_ulong(r, val): return f"{r[val]}.x" if val.dtype == dtypes.ulong else f"{r[val]}"
def render_load_store(r, bidx):
  _, buf, idx = {sbidx:=strip_parens(r[bidx])}, sbidx.split("+")[0], '+'.join(sbidx.split("+")[1:])
  return f"{buf}[{idx}]"

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
    (UPat(UOps.CONST, dtype=dtypes.bool, name="x"), lambda r,x: "true" if x.arg else "false"),
    (UPat(UOps.CONST, dtype=dtypes.uint32, name="x"), lambda r,x: f"bitcast<u32>({x.arg}i)" if x.arg < 0 else f"{x.arg&0xFFFFFFFF}u"),
    (UPat(UOps.CONST, dtype=dtypes.ulong, name="x"), lambda r,x: f"vec2<u32>({x.arg}, 0u)" if x.arg < 4294967296 \
     else  f"vec2<u32>({x.arg&4294967295}, {x.arg>>32})"),
    (UPat(UOps.DEFINE_LOCAL, name="x"), lambda r,x: f"var<workgroup> {r[x]}: array<{type_map[x.dtype.base]}, {x.arg[1]}>;"),
    (UPat(UOps.CAST, name="x"), lambda r,x: f"vec2<u32>(({r[x.src[0]]})&4294967295, 0u)" if x.dtype == dtypes.uint64 \
      else f"{type_map[x.dtype]}({r[x.src[0]]}.x)" if x.src[0].dtype == dtypes.uint64 else f"{type_map[x.dtype]}({r[x.src[0]]})"),
    (UPat(UOps.BITCAST, name="x"), lambda r,x: f"bitcast<{type_map[x.dtype]}>({r[x.src[0]]})"),
    (UPat(UOps.LOAD, src=(UPat.var("bidx"), UPat.var('var'), UPat.var("gate"))),
      lambda r,bidx,var,gate: f"select({r[var]}, {render_load_store(r, bidx)}, {r[gate]})"),
    (UPat(UOps.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda r,bidx: f"{render_load_store(r, bidx)}"),
    (UPat(UOps.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda r,bidx,var: f"{render_load_store(r,bidx)} = {r[var]};"),
    ]) + base_rewrite + PatternMatcher([
    (UPat(UOps.ALU, name="x", dtype=dtypes.ulong, arg=BinaryOps.SHL), lambda r,x: f"ushl({r[x.src[0]]},u32({gep_ulong(r, x.src[1])}))"),
    (UPat(UOps.ALU, name="x", dtype=dtypes.ulong, arg=BinaryOps.SHR), lambda r,x: f"ushr({r[x.src[0]]},u32({gep_ulong(r, x.src[1])}))"),
  ])

  def render_dtype(self, dt:DType, mutable=True) -> str: return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [num for _, num in sorted([u.arg for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == 'l'], key=lambda x: x[0])]
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    prg += "fn ushl(v: vec2<u32>, shift: u32) -> vec2<u32> {\n"
    prg += "return select(vec2<u32>(v.x << shift, (v.y << shift) | (v.x >> (32u - shift))), vec2<u32>(0u, v.x << (shift - 32u)), shift >= 32u); }\n"
    prg += "fn ushr(v: vec2<u32>, shift: u32) -> vec2<u32> {\n"
    prg += "return select(vec2<u32>((v.x >> shift) | (v.y << (32u - shift)), v.y >> shift), vec2<u32>(v.y >> (shift - 32u), 0u), shift >= 32u); }\n"
    prg += "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n"
    prg += "\n".join((prefix or [])+[f"@group(0) @binding({next(bind_it)+1}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype.base]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg
