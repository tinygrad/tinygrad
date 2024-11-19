from typing import List, Tuple
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens
import math

wgsl_matcher = PatternMatcher([
  (UPat(Ops.CMPLT, src=(UPat(name="a", dtype=dtypes.bool), UPat(name="b")), name="c"),
    lambda a,b,c: UOp(c.op, c.dtype, (a.cast(dtypes.int), b.cast(dtypes.int)))),
  (UPat(Ops.XOR, dtype=dtypes.bool, src=(UPat(name="a"), UPat(name="b")), name="c"),
    lambda a,b,c: UOp(c.op, dtypes.int, (a.cast(dtypes.int), b.cast(dtypes.int))).cast(dtypes.bool)),
  *[(UPat(a, src=(UPat(name="b", dtype=(dtypes.uint, dtypes.int, dtypes.bool))), name="a"),
     lambda a,b: UOp(a, dtypes.float, (b.cast(dtypes.float),)).cast(b.dtype)) for a in (Ops.EXP2, Ops.SIN, Ops.LOG2, Ops.SQRT)],
  (UPat(Ops.MUL, name="m", src=(UPat(name="a"), UPat(Ops.WHERE, src=(UPat.var("g"), \
    UPat(op=Ops.CONST, name="c1"), UPat(op=Ops.CONST, name="c2"))))), \
    lambda m,a,g,c1,c2: UOp(Ops.WHERE, dtype=m.dtype, src=(g, UOp.const(dtype=dtypes.float, b=float('nan')), a)) \
    if math.isnan(c1.arg) and c2.arg == 1.0 else None),
  ]) + extra_pm

type_map = { dtypes.float: "f32", dtypes.uchar: "u32", dtypes.ushort: "u32", dtypes.short: "i32",
            dtypes.char: "i32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool" }
buffer_map = { **type_map, dtypes.bool: "i32" }

def sign_extend(val, sext_am): return f"bitcast<i32>(select(0u, 0xffffffffu << {sext_am}, (({val} >> {sext_am-1}) > 0)) | bitcast<u32>({val}))"
def render_shift_am(idx, itemsize): return f"(((u32({idx}))%{4//itemsize}u)*{8*itemsize}u)"

def render_load(r, bidx, dtype):
  sbidx = strip_parens(r[bidx])
  buf, idx = sbidx.split("+")[0], '+'.join(sbidx.split("+")[1:])
  # packed load for bool, char, short
  if dtype.itemsize < 4:
    idx_by_itemsize = f"({idx})/{4//dtype.itemsize}"
    val = f"(({buf}[{idx_by_itemsize}]) >> {render_shift_am(idx, dtype.itemsize)} & { "0xFF" if dtype.itemsize == 1 else "0xFFFF" })"
    val = f"bool({val})" if dtype == dtypes.bool else val
    return sign_extend(val, 8*dtype.itemsize) if dtype in [dtypes.char, dtypes.short] else val
  return f"{buf}[{idx}]"

def render_store(r, bidx, var, dtype):
  sbidx = strip_parens(r[bidx])
  buf, idx = sbidx.split("+")[0], '+'.join(sbidx.split("+")[1:])
  arr = f"{buf}[u32({idx})/{4//dtype.itemsize}u]"
  return f"atomicAdd(&{arr}, (({buffer_map[dtype]}({r[var]}) & { "0xFF" if dtype.itemsize == 1 else "0xFFFF" }) << \
    {render_shift_am(idx, dtype.itemsize)}));" if dtype.itemsize < 4 else f"{arr} = {r[var]};"

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
    (UPat(Ops.LOAD, name="l", src=(UPat.var("b"), UPat.var('v'), UPat.var("g"))),
      lambda ctx,l,b,v,g: f"select({ctx[v]}, "f"{render_load(ctx, b, l.dtype)}, {ctx[g]})"),
    (UPat(Ops.LOAD, name="l", src=(UPat.var('b'),), allow_any_len=True), lambda ctx,l, b: f"{render_load(ctx, b, l.dtype)}"),
    (UPat(Ops.STORE, src=(UPat.var('b'), UPat.var("v")), allow_any_len=True),lambda ctx,b,v: f"{render_store(ctx, b, v, v.dtype)}"),
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
    prg += "\n".join((external_local_bufs or [])+[f"@group(0) @binding({next(bind_it)+1}) {'var<storage,read_write>' if isinstance(dtype, PtrDType)\
      else 'var<uniform>'} {name}: {f'array<{f'atomic<{buffer_map[dtype.base]}>' if rw and (dtype.itemsize < 4) else buffer_map[dtype.base]}>'\
      if isinstance(dtype, PtrDType) else buffer_map[dtype]};" for name,(dtype,rw) in bufs])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>,"
    return prg + "@builtin(local_invocation_id) lindex: vec3<u32>) {\n" + "\n".join(kernel) + "\n}"
