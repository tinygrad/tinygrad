from typing import List, Tuple
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens
import math

wgsl_matcher = PatternMatcher([
  (UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL), dtype=dtypes.bool.ptr(), name="a"), lambda a: UOp(a.op, dtypes.int32.ptr(), a.src, a.arg)),
  (UPat(Ops.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("g", dtype=dtypes.bool))),
    lambda root,x,y,g: UOp(root.op, dtypes.int, (x,y.cast(dtypes.int), g), root.arg).cast(dtypes.bool)),
  (UPat(Ops.LOAD, name="root", dtype=dtypes.bool, src=(UPat())), lambda root: UOp(root.op, dtypes.int, root.src, root.arg).cast(dtypes.bool)),
  (UPat(Ops.CMPLT, src=(UPat(name="a", dtype=dtypes.bool), UPat(name="b")), name="c"),
    lambda a,b,c: UOp(c.op, c.dtype, (a.cast(dtypes.int), b.cast(dtypes.int)))),
  (UPat(Ops.XOR, dtype=dtypes.bool, src=(UPat(name="a"), UPat(name="b")), name="c"),
    lambda a,b,c: UOp(c.op, dtypes.int, (a.cast(dtypes.int), b.cast(dtypes.int))).cast(dtypes.bool)),
  *[(UPat(a, src=(UPat(name="b", dtype=(dtypes.uint, dtypes.int, dtypes.bool))), name="a"),
     lambda a,b: UOp(a, dtypes.float, (b.cast(dtypes.float),)).cast(b.dtype))
    for a in (Ops.EXP2, Ops.SIN, Ops.LOG2, Ops.SQRT)],
  (UPat.store(UPat.var("bidx"), UPat.var("var", dtype=dtypes.bool), UPat.var("gate")),
   lambda bidx,val,gate: UOp.store(bidx, val.cast(dtypes.int), gate)),
  (UPat.store(UPat.var("bidx"), UPat.var("var", dtype=dtypes.bool)), lambda bidx,var: UOp.store(bidx, var.cast(dtypes.int))),
  # fix nan propagation: 'a * select(1, nan, cond) -> select(a, nan, cond)'
  (UPat(Ops.MUL, name="m", src=(UPat(name="a"), UPat(Ops.WHERE, src=(UPat.var("g"), \
    UPat(op=Ops.CONST, name="c1"), UPat(op=Ops.CONST, name="c2"))))), \
    lambda m,a,g,c1,c2: UOp(Ops.WHERE, dtype=m.dtype, src=(g, UOp.const(dtype=dtypes.float, b=float('nan')), a)) \
    if math.isnan(c1.arg) and c2.arg == 1.0 else None),
  ]) + extra_pm

type_map = { dtypes.float: "f32", dtypes.uchar: "u32", dtypes.ushort: "u32", dtypes.short: "i32",
            dtypes.char: "i32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool" }

# convert from pointer style indexing to array style
def render_load_store(r, bidx, sext = False, sext_am = 8):
  sbidx = strip_parens(r[bidx])
  buf, idx = sbidx.split("+")[0], '+'.join(sbidx.split("+")[1:])
  # sign-extend when loading char/short
  return f"bitcast<i32>(select(0u, 0xffffffffu << {sext_am}, (({buf}[{idx}] >> {sext_am-1}) > 0)) | bitcast<u32>({buf}[{idx}]))" \
    if sext else f"{buf}[{idx}]"

class WGSLRenderer(CStyleLanguage):
  device = "WEBGPU"
  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  extra_matcher = wgsl_matcher
  supports_float4 = False
  barrier = "workgroupBarrier();"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})"}
  nan = "nan()"
  type_map = type_map

  string_rewrite = PatternMatcher([
    (UPat(Ops.CONST, dtype=dtypes.bool, name="x"), lambda ctx,x: "true" if x.arg else "false"),
    (UPat(Ops.CONST, dtype=(dtypes.char, dtypes.short), name="x"), lambda ctx,x: f"i32({x.arg})"),
    (UPat(Ops.CONST, dtype=(dtypes.uchar, dtypes.ushort, dtypes.uint32), name="x"), lambda ctx,x: f"bitcast<u32>({x.arg})" \
     if x.arg < 0 else f"{x.arg&0xFFFFFFFF}u"),
    (UPat(Ops.CONST, dtype=dtypes.int32, name="x"), lambda ctx,x: f"bitcast<i32>({x.arg}u)" if x.arg >= 0x80000000 else f"{x.arg}"),
    (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx,x: f"var<workgroup> {ctx[x]}: array<{type_map[x.dtype.base]}, {x.arg[1]}>;"),
    (UPat(Ops.BITCAST, dtype=(dtypes.char, dtypes.uchar), name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]}&0xFF)"),
    (UPat(Ops.BITCAST, dtype=(dtypes.short, dtypes.ushort), name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]}&0xFFFF)"),
    (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast<{type_map[x.dtype]}>({ctx[x.src[0]]})"),
    # sign extended loads for char, short
    (UPat(Ops.LOAD, name="l", src=(UPat.var("bidx"), UPat.var('var'), UPat.var("gate"))),
      lambda ctx,l,bidx,var,gate: f"select({ctx[var]}, "
        f"{render_load_store(ctx, bidx, l.dtype in [dtypes.char, dtypes.short], 8 * l.dtype.itemsize)}, {ctx[gate]})"),
    (UPat(Ops.LOAD, name="l", src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,l, bidx:
     f"{render_load_store(ctx, bidx, l.dtype in [dtypes.char, dtypes.short], 8*l.dtype.itemsize)}"),
    (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True),
     lambda ctx,bidx,var: f"{render_load_store(ctx,bidx)} = {ctx[var]};"),
    # fix nan check: 'a != a -> is_nan()'
    (UPat(Ops.CMPNE, src=(UPat.var("a"), UPat.var("b"))), lambda ctx,a,b: f"is_nan({ctx[a]})" if a == b else None),
  ]) + base_rewrite

  def render_cast(self, dt:DType, val: str) -> str: return f"{self.type_map[dt]}({val})"
  def render_dtype(self, dt:DType, mutable=True) -> str: return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [num for _, num in sorted([u.arg for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == 'l'], key=lambda x: x[0])]
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    external_local_bufs = [line.lstrip() for line in kernel if "var<workgroup>" in line]
    kernel[:] = [line for line in kernel if "var<workgroup>" not in line]
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    # trick to obfuscate compiler so that nan is detected properly
    prg += "fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }\n"
    prg += "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n"
    prg += "\n".join((external_local_bufs or [])+[f"@group(0) @binding({next(bind_it)+1}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype.base]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg
