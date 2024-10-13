from typing import List, Tuple
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, UOps, UnaryOps, TernaryOps, BinaryOps, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, _render_index

def fixup_binops(c,a,b):
  if c.arg == BinaryOps.CMPLT and a.dtype == dtypes.bool: return UOp(c.op, c.dtype, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg)
  if c.arg in (BinaryOps.MAX, BinaryOps.XOR) and c.dtype == dtypes.bool:
    return UOp(c.op, dtypes.int, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg).cast(dtypes.bool)
  # test_uops::test_mod_int32 is broken, at least on Vulkan backend, without this
  if c.arg == BinaryOps.MOD and c.dtype == dtypes.int:
    return UOp(c.op, dtypes.float, (a.cast(dtypes.float), b.cast(dtypes.float)), c.arg).cast(dtypes.int)

wgsl_matcher = PatternMatcher([
  (UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), dtype=PtrDType(dtypes.bool), name="a"), lambda a: UOp(a.op, PtrDType(dtypes.int32), a.src, a.arg)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("z"),UPat.var("g", dtype=dtypes.bool))),
    lambda root,x,y,z,g: UOp(root.op, dtypes.int, (x,y,z.cast(dtypes.int), g), root.arg).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var(),UPat.var(),UPat.var(),UPat(UOps.BARRIER))),
    lambda root: UOp(root.op, dtypes.int, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())), lambda root: UOp(root.op, dtypes.int, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.ALU, src=(UPat(name="a"), UPat(name="b")), name="c"), fixup_binops),
  *[(UPat(UOps.ALU, src=(UPat(name="b", dtype=(dtypes.uint, dtypes.int, dtypes.bool))), arg=a, name="a"),
     lambda a,b: UOp(a.op, dtypes.float, (b.cast(dtypes.float),), a.arg).cast(b.dtype))
    for a in (UnaryOps.EXP2, UnaryOps.SIN, UnaryOps.LOG2, UnaryOps.SQRT)],
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.var("val", dtype=dtypes.bool), UPat.var("gate")),
   lambda buf,idx,val,gate: UOp.store(buf, idx, val.cast(dtypes.int), gate)),
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.var("val", dtype=dtypes.bool)), lambda buf,idx,val: UOp.store(buf, idx, val.cast(dtypes.int))),
])

type_map = {dtypes.float: "f32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool"}

class WGSLRenderer(CStyleLanguage):
  device = "WEBGPU"
  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  extra_matcher = wgsl_matcher
  external_local_bufs = True
  supports_float4 = False
  barrier = "workgroupBarrier();"
  code_for_op = {**CStyleLanguage().code_for_op, TernaryOps.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})"}
  infinity = "inf(1.0)"
  nan = "nan()"
  type_map = type_map

  string_rewrite = PatternMatcher([
    (UPat(UOps.CONST, dtype=dtypes.bool, name="x"), lambda r,x: "true" if x.arg else "false"),
    # cstyle uint initialization
    (UPat(UOps.CONST, dtype=dtypes.uint32, name="x"), lambda r,x: f"bitcast<u32>({x.arg}i)" if x.arg < 0 else f"{x.arg}u"),
    (UPat(UOps.DEFINE_LOCAL, name="x"), lambda r,x: f"var<workgroup> {r[x]}: array<{type_map[x.dtype.base]}, {x.arg[1]}>;"),
    (UPat(UOps.CAST, name="x"), lambda r,x: f"{type_map[x.dtype]}({r[x.src[0]]})"),
    (UPat(UOps.BITCAST, name="x"), lambda r,x: f"bitcast<{type_map[x.dtype]}>({r[x.src[0]]})"),
    (UPat(UOps.LOAD, src=(UPat.var("buf"), UPat.var('idx'), UPat.var("var"), UPat.var("gate")), name="load"),
     lambda r,buf,idx,load,var,gate: f"select({r[var]}, {_render_index(r, buf, idx, load.dtype)},{r[gate]})"),
  ]) + base_rewrite

  def render_dtype(self, var_dtype): return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [u.arg[1] for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == 'l']
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\nfn inf(a: f32) -> f32 { return a/0.0; }\n"
    prg += "\n".join((prefix or [])+[f"@group(0) @binding({next(bind_it)}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg
