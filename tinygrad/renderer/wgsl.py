from typing import List, Tuple
import math
from tinygrad.dtype import DType, PtrDType, dtypes, ConstType
from tinygrad.ops import UOp, UOps, UnaryOps, TernaryOps, BinaryOps, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage

def fixup_binops(c,a,b):
  if c.arg == BinaryOps.CMPLT and a.dtype == dtypes.bool: return UOp(c.op, c.dtype, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg)
  if c.arg in (BinaryOps.MAX, BinaryOps.XOR) and c.dtype == dtypes.bool:
    return UOp(c.op, dtypes.int, (a.cast(dtypes.int), b.cast(dtypes.int)), c.arg).cast(dtypes.bool)
  # wgpu int mod is wrong (hw specific?). Returns (-2) % (-3) = 1. test_uops::test_mod_int32 broken on Vulkan without this
  if c.arg == BinaryOps.MOD and c.dtype == dtypes.int:
    return UOp(c.op, dtypes.float, (a.cast(dtypes.float), b.cast(dtypes.float)), c.arg).cast(dtypes.int)

wgsl_matcher = PatternMatcher([
  (UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL), dtype=PtrDType(dtypes.bool), name="a"),
   lambda a: UOp(a.op, PtrDType(dtypes.int32), a.src, a.arg)),
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
   lambda buf,idx,val,gate: UOp.store(buf, idx, val.cast(buf.dtype), gate)),
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.var("val", dtype=dtypes.bool)), lambda buf,idx,val: UOp.store(buf, idx, val.cast(buf.dtype))),
])

class WGSLRenderer(CStyleLanguage):
  device = "WEBGPU"
  global_max = (65535, 65535, 65535)
  local_max = (256, 256, 64)
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  extra_matcher = wgsl_matcher
  external_local_bufs = True
  supports_float4 = False
  barrier = "workgroupBarrier();"
  type_map = {dtypes.float: "f32", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool"}
  code_for_op = {**CStyleLanguage().code_for_op, TernaryOps.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a})"}

  def render_const(self, x:ConstType, dtype: DType) -> str:
    assert dtype.count == 1, f"consts should be scalar, got {dtype}"
    if math.isinf(x): return ("-" if x < 0 else "") + "inf(1.0)"
    if math.isnan(x): return "nan()"
    if dtype == dtypes.bool: return "true" if x else "false"
    if dtypes.is_unsigned(dtype): return f"bitcast<u32>({x}i)" if x < 0 else f"{x}u" # cstyle initialization
    return f"{x}" + ("" if dtypes.is_int(dtype) else "f")

  def render_local(self, name: str, dtype:DType, size: int): return f"var<workgroup> {name}: array<{self.type_map[dtype]},{size}>;"

  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    if self.type_map[var_dtype]: return f"bitcast<{self.type_map[var_dtype]}>({x})" if bitcast else f"{self.type_map[var_dtype]}({x})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  def render_dtype(self, var_dtype): return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [u.arg[1] for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == 'l']
    if not local_size: local_size = [1]
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\nfn inf(a: f32) -> f32 { return a/0.0; }\n"
    prg += "\n".join((prefix or [])+[f"@group(0) @binding({next(bind_it)}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg
