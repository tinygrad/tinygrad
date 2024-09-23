from typing import Dict, List, Optional, Tuple, Union, DefaultDict, Literal, Callable
import os, math
from collections import defaultdict, Counter
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, UOps, UOp
from tinygrad.helpers import strip_parens, getenv, prod, dedup, AMX
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, ConstType
from tinygrad.renderer import Renderer, TensorCore

class CStyleLanguage(Renderer):
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  extra_args: List[str] = []
  float4: Optional[str] = None
  uses_vload: bool = False
  uses_ptr_arithmetic: bool = False
  type_map: Dict[DType, str] = {}
  infinity: str = "INFINITY"
  nan: str = "NAN"
  code_for_op: Dict = {
    UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    UnaryOps.RECIP: lambda x,dtype: f"(1/{x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})",
    BinaryOps.IDIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPNE: lambda a,b,dtype: f"({a}!={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    BinaryOps.AND: lambda a,b,dtype: f"({a}&{b})", BinaryOps.OR: lambda a,b,dtype: f"({a}|{b})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    if bitcast: return f"(*(({self.buffer_prefix}{self.render_dtype(var_dtype)}*)&{x}))"
    return f"({self.render_dtype(var_dtype)})({x})"

  # returns a str expression of the vectorized xs with the given type
  def render_vectorize(self, x:List[str], var_dtype:DType) -> str:
    assert len(x) == var_dtype.count, f"cast is wrong size {len(x)} != {var_dtype.count}"
    assert self.float4 is not None, "vectorized cast is not supported on this platform"
    return f"{self.float4.replace('float4', self.render_dtype(var_dtype))}" + (f"{{{','.join(x)}}}" if self.device == "CLANG" else f"({','.join(x)})")

  # returns a str expression of the const with the given type
  def render_const(self, x:ConstType, dtype:DType) -> str:
    assert dtype.count == 1, f"consts should be scalar, got {dtype}"
    if math.isnan(x): val = self.nan
    elif math.isinf(x): val = ("-" if x < 0 else "") + self.infinity
    elif dtype == dtypes.bool: val = "1" if x else "0"
    elif dtype == dtypes.float: val = f"{x}f"
    elif dtype == dtypes.uint64: val = f"{x}ULL"
    else: val = str(x)
    return (self.render_cast(val, dtype) if dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes.float.vec(4), f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and output_dtype.scalar() != dtypes.float16:
      return f"vload_half{'' if output_dtype.count == 1 else str(output_dtype.count)}(0, {buf_name}+{idx})"
    if output_dtype.count > 1:
      return f"*(({self.smem_prefix if buf_dtype.local and self.smem_prefix_for_cast else self.buffer_prefix}{self.render_dtype(output_dtype)}*)({buf_name}+{idx}))"  # noqa: E501
    return f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"

  def get_kernel_modifier(self, uops:List[UOp]) -> str: return ""
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
    buftypes = [(name,f"{'write_only' if mutable else 'read_only'} image2d_t" if dtype.name.startswith('image') else
                ("" if mutable else "const ")+self.buffer_prefix+self.render_dtype(dtype)+"*"+self.buffer_suffix if isinstance(dtype, PtrDType) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
    prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:Union[ImageDType, PtrDType], var_name:str, var_dtype:DType, idx:str) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes.float.vec(4), f"images must be float4, getting {var_dtype}"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and var_dtype.scalar() != dtypes.float16:
      return f"vstore_half{'' if var_dtype.count == 1 else str(var_dtype.count)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.count > 1:
      prefix = self.smem_prefix if buf_dtype.local and self.smem_prefix_for_cast else self.buffer_prefix
      return f"*(({prefix}{self.render_dtype(var_dtype)}*)({buf_name}+{idx})) = {var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{self.render_dtype(dtype)} {name}[{size}];"
  def render_dtype(self, var_dtype:DType) -> str:
    return self.type_map.get(scalar:=var_dtype.scalar(), scalar.name) + (str(var_dtype.count) if (var_dtype.count) > 1 else "")

  def render(self, name:str, uops:List[UOp]) -> str:
    kernel = []
    bufs: Dict[UOp, Tuple[str, Tuple[DType, bool]]] = {}
    depth = 1
    def kk(s): kernel.append("  "*depth+s)

    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, str] = {}

    def ssa(prefix:str, u:Optional[UOp]=None):
      nonlocal c, r
      ret = f"{prefix}{c[prefix]}"
      if u is not None: r[u] = ret
      c[prefix] += 1
      return ret

    child_count = Counter(v for ru in uops for v in ru.src)

    seen_vars = set()
    for u in uops:
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      # these four uops don't have output dtypes
      if uop is UOps.IF:
        kk(f"if ({r[src[0]]}) {{")
        depth += 1
      elif uop is UOps.BARRIER: kk(self.barrier)
      elif uop in {UOps.ENDRANGE, UOps.ENDIF}:
        depth -= 1
        kk("}")
      elif uop is UOps.STORE:
        # mark DEFINE_GLOBAL buf as writable
        assert isinstance(src[0].dtype, (ImageDType, PtrDType))
        if src[0].op is UOps.DEFINE_GLOBAL: bufs[src[0]] = (bufs[src[0]][0], (bufs[src[0]][1][0], True))
        rendered_store = self.render_store(r[src[0]], src[0].dtype, r[src[2]], src[2].dtype, strip_parens(r[src[1]]))
        kk(f"if ({r[src[3]]}) {{ {rendered_store} }}" if len(src) > 3 and src[3].op is not UOps.IF else rendered_store)
      else:
        if uop is UOps.RANGE:
          kk(f"for (int {(expr := ssa('ridx',u))} = {r[src[0]]}; {expr} < {r[src[1]]}; {expr}++) {{")
          depth += 1
        elif uop is UOps.ALU:
          # remove parens if ALU types are the same. TODO: can do more here
          if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in src]
          elif args is BinaryOps.MAX: operands = [self.render_cast(r[v], v.dtype) if v.op is UOps.CONST else r[v] for v in src]
          else: operands = [r[v] for v in src]
          val = self.code_for_op[args](*operands, dtype)
          assert child_count[u] != 0, f"childless ALU op found {u}"
          # TODO: fix index rendering issue. fix clang nested max macro issue
          if child_count[u] <= 1 and args is not BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val
          else: kk(f"{self.render_dtype(dtype)} {ssa('alu',u)} = {val};")
        elif uop is UOps.SPECIAL:
          kk(f"int {args[0]} = {self.code_for_workitem[args[0][0]](args[0][-1])}; /* {args[1]} */")
          r[u] = args[0]
        elif uop is UOps.DEFINE_VAR:
          assert args[0] not in seen_vars, f"duplicate variable {args[0]}"
          seen_vars.add(args[0])
          bufs[u] = (args[0], (dtype,False))
          r[u] = args[0]
        elif uop is UOps.LOAD:
          val = self.render_load(dtype, r[src[0]], src[0].dtype, strip_parens(r[src[1]]))
          # NOTE: this relies on the load not happening if it's in the unselected branch
          if len(src) > 3 and src[3].op is UOps.ALU: val = self.code_for_op[TernaryOps.WHERE](r[src[3]], val, r[src[2]], dtype)
          kk(f"{self.render_dtype(dtype)} {ssa('val',u)} = {val};")
        elif uop is UOps.ASSIGN:
          kk(f"{r[src[0]]} = {r[src[1]]};")
          r[u] = r[src[0]]
        elif uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}:
          assert len(src) == 1 or (uop is UOps.VECTORIZE and len(src) > 1), "Invalid source length for operation"
          if uop is UOps.BITCAST:
            precast = ssa('precast')
            kk(f"{self.render_dtype(src[0].dtype)} {precast} = {r[src[0]]};")
            val = self.render_cast(precast, dtype, bitcast=True)
          elif uop is UOps.CAST: val = self.render_cast(r[src[0]], dtype, bitcast=False)
          else: val = self.render_vectorize([r[x] for x in src], dtype)
          if child_count[u] <= 1: r[u] = val
          else: kk(f"{self.render_dtype(dtype)} {ssa('cast',u)} = {val};")
        elif uop is UOps.DEFINE_LOCAL:
          kk(self.render_local(args[0], dtype, args[1]))
          r[u] = args[0]
        elif uop is UOps.DEFINE_GLOBAL:
          bufs[u] = (nm:=f"data{args}", (dtype, False))
          r[u] = nm
        elif uop is UOps.WMMA: kk(f"{self.render_dtype(dtype)} {ssa('wmma',u)} = __{args[0]}({r[src[0]]}, {r[src[1]]}, {r[src[2]]});")
        elif uop is UOps.DEFINE_ACC: kk(f"{self.render_dtype(dtype)} {ssa('acc',u)} = {r[src[0]]};")
        elif uop is UOps.CONST: r[u] = self.render_const(args, dtype) if args >= 0 else f"({self.render_const(args, dtype)})"
        elif uop is UOps.GEP:
          assert len(args) == 1
          from_ssa = src[0].op in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
          r[u] = (r[src[0]] if from_ssa else f"{(r[src[0]])}") + \
            (f"[{args[0]}]" if src[0].dtype.count > (8 if self.device in {"CUDA", "NV"} else 4) \
             or self.device == 'CLANG' else f".{'xyzwabcd'[args[0]]}")
        else: raise RuntimeError(f"failed to render {u}")

    # NOTE: this relies on bufs dict preserving order
    return self.render_kernel(name, kernel, list(bufs.values()), uops)

class ClangRenderer(CStyleLanguage):
  device = "CLANG"
  float4 = "(float4)"
  has_local = False
  global_max = None
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**({k:v for k,v in CStyleLanguage().code_for_op.items() if k not in [UnaryOps.EXP2, UnaryOps.SIN, UnaryOps.LOG2]}),
                 UnaryOps.SQRT: lambda x,dtype: f"__builtin_sqrtl({x})" if dtype == dtypes.float64 else f"__builtin_sqrtf({x})",
                 BinaryOps.MAX: lambda a,b,dtype: f"(({a}>{b})?{a}:{b})"}

  if AMX:
    tc_types = [(dtype, amx_size//dtype.itemsize) for dtype, amx_size in zip([dtypes.float], [64])]
    tensor_cores = [TensorCore(dims=(sz,sz,1), threads=[], dtype_in=dtype, dtype_out=dtype) for dtype, sz in tc_types]

  def render_vector_prefix(self, dt:DType) -> str:
    return f"typedef {self.render_dtype(dt.scalar())} {self.render_dtype(dt)} __attribute__((aligned({(sz:=dt.itemsize)}),vector_size({sz})));"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix, macros = [self.render_vector_prefix(dt) for dt in dedup(uop.dtype for uop in uops if uop.dtype.count>1)], []
    # https://github.com/corsix/amx
    for name, (N, M, _), dtype_in, _, _, _, _, _ in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]):
      macros = [
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      prefix += [f"""{(out := self.render_dtype(dtype_in.vec(N*N)))} __{name}({self.render_dtype(dtype_in.vec(N))} data1, {self.render_dtype(dtype_in.vec(M))} data2, {out} data0){{
  AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}"""] # noqa: E501
    return super().render_kernel(function_name, kernel, bufs, uops, macros + prefix)

class OpenCLRenderer(CStyleLanguage):
  device = "GPU"

  code_for_op = {**CStyleLanguage().code_for_op,
    #UnaryOps.SQRT: lambda x,dtype: f"native_sqrt({x})", UnaryOps.RECIP: lambda x,dtype: f"native_recip({x})",
    #UnaryOps.EXP2: lambda x,dtype: f"native_exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"native_log2({x})",
    UnaryOps.SIN: lambda x,dtype: f"native_sin({x})"} if getenv("NATIVE_MATH") else CStyleLanguage().code_for_op

  # language options
  kernel_prefix = "__kernel "
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  code_for_workitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}
  uses_vload = True
  type_map = { dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong", dtypes.bfloat16: "ushort" }
  def render_cast(self, x, var_dtype, bitcast=False) -> str:
    return f"as_{self.render_dtype(var_dtype)}({x})" if bitcast else super().render_cast(x, var_dtype)

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    if any(uop.dtype == dtypes.half for uop in uops): prefix = (["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"] + (prefix or []))
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

class IntelRenderer(OpenCLRenderer):
  device, suffix, kernel_prefix = "GPU", "INTEL", "__attribute__((intel_reqd_sub_group_size(8)))\n" + "__kernel "
  tensor_cores = [TensorCore(dims=(8,8,16), threads=[(0,8)], dtype_in=di, dtype_out=do) for di, do in [(dtypes.half, dtypes.float), (dtypes.bfloat16, dtypes.float)]]  # noqa: E501
  def render_dtype(self, var_dtype:DType) -> str:
    return f"ushort{var_dtype.count}" if "bfloat16" in var_dtype.name else super().render_dtype(var_dtype)
  def render_cast(self, x, var_dtype, bitcast=False, from_dtype=None) -> str:
    return f"intel_convert_bfloat16_as_ushort({x[0]})" if (var_dtype, from_dtype) == (dtypes.bfloat16, dtypes.float) else \
      (f"intel_convert_as_bfloat16_float({x[0]})" if (var_dtype, from_dtype) == (dtypes.float, dtypes.bfloat16) else \
      super().render_cast(x, var_dtype, bitcast))

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = []
    for arg in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]):
      dt_in = ("ushort", "bf16") if arg[2] == dtypes.bfloat16 else (arg[2].name, "f16")
      prefix.append(f"""{arg[3].name}8 __{arg[0]}({dt_in[0]}16 a, {dt_in[0]}16 b, {arg[3].name}8 c) {{
    return intel_sub_group_{dt_in[1]}_{dt_in[1]}_matrix_mad_k16(as_int8(a), as_int8(b), c);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix or None)

class MetalRenderer(CStyleLanguage):
  device = "METAL"
  shared_max = 32768
  tensor_cores = [TensorCore(dims=(8,8,8), threads=[(0,2),(1,4),(0,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in [(dtypes.float, dtypes.float), (dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]] # noqa: E501
  def __init__(self): self.tensor_cores = MetalRenderer.tensor_cores if os.uname().machine == "arm64" else []

  # language options
  kernel_prefix = "kernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic = True
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}
  # uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  type_map = {dtypes.bfloat16: "bfloat"}
  code_for_op = {**CStyleLanguage().code_for_op,
    BinaryOps.MAX: lambda a,b,dtype: f"(bfloat)max((float){a},(float){b})" if dtype == dtypes.bfloat16 else f"max({a},{b})",
    UnaryOps.SQRT: lambda x,dtype: f"(bfloat)sqrt({x})" if dtype == dtypes.bfloat16 else f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"(bfloat)exp2({x})" if dtype == dtypes.bfloat16 else f"exp2({x})",
    UnaryOps.LOG2: lambda x,dtype: f"(bfloat)log2({x})" if dtype == dtypes.bfloat16 else f"log2({x})",
    UnaryOps.SIN: lambda x,dtype: f"(bfloat)precise::sin({x})" if dtype == dtypes.bfloat16 else f"precise::sin({x})",}

  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    return f"as_type<{self.render_dtype(var_dtype)}>({x})" if bitcast else super().render_cast(x, var_dtype)

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix, wmma_args = ["#include <metal_stdlib>","using namespace metal;"], set([uop.arg for uop in uops if uop.op is UOps.WMMA])
    for arg in wmma_args: prefix.append(f"""{arg[3].name}2 __{arg[0]}({arg[2].name}2 m, {arg[2].name}2 n, {arg[3].name}2 o) {{
  simdgroup_{arg[3].name}8x8 a,b,c; a.thread_elements()[0] = m.x; a.thread_elements()[1] = m.y; b.thread_elements()[0] = n.x;
  b.thread_elements()[1] = n.y; c.thread_elements()[0] = o.x; c.thread_elements()[1] = o.y; simdgroup_multiply_accumulate(c, a, b, c);
  return {arg[3].name}2(c.thread_elements()[0], c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

code_for_op_half = {UnaryOps.RECIP: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"1/{x}",
                    BinaryOps.MAX: lambda a,b,dtype: f"__hmax({a},{b})" if dtype in (dtypes.half, dtypes.bfloat16) else f"max({a},{b})",
                    UnaryOps.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
                    UnaryOps.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
                    UnaryOps.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
                    UnaryOps.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",}

_nms = "xyzwabcdefghijkl"

class CUDARenderer(CStyleLanguage):
  device = "CUDA"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float), (dtypes.bfloat16, dtypes.float)])]  # noqa: E501
  def __init__(self, arch:str): self.tensor_cores = CUDARenderer.tensor_cores if int(arch[3:]) >= 80 else []

  # language options
  kernel_prefix = "extern \"C\" __global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+int(x))}", "l": lambda x: f"threadIdx.{chr(120+int(x))}",
                       "i": lambda x: f"(blockIdx.{chr(120+int(x))}*blockDim.{chr(120+x)}+threadIdx.{chr(120+int(x))})"}
  code_for_op = {**CStyleLanguage().code_for_op, **code_for_op_half}
  type_map = {dtypes.bfloat16: "nv_bfloat16"}

  def render_vector_prefix(self, dt:DType) -> str:
    vec, scal = self.render_dtype(dt), self.render_dtype(dt.scalar()),
    elems, header = ', '.join(_nms[:dt.count]), ', '.join([f"{scal} {x}" for x in _nms[:dt.count]])
    return f"struct __align__({dt.itemsize}) {vec} {{ {scal} {elems}; }}; __device__ {vec} make_{vec}({header}) {{ {vec} r={{{elems}}}; return r; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    # TODO: why is dtypes.bfloat16.name == "__bf16"? would be easier not override dtypes.name
    prefix = ["#define INFINITY (__int_as_float(0x7f800000))","#define NAN (__int_as_float(0x7fffffff))"]

    for dtype in dedup(uop.dtype for uop in uops if uop.dtype in {dtypes.half, dtypes.bfloat16}):
      prefix += [f"#include <cuda_{'fp' if dtype == dtypes.half else 'bf'}16.h>"] + [self.render_vector_prefix(dtype.vec(sz)) for sz in [4, 8]]

    # TODO: this has to be way better to generate for arbitrary M,N,K: use arg[1] for MNK, use arg[4] for vec sizes, encode register packing
    dt_map = { dtypes.float: "f32", dtypes.half: "f16", dtypes.bfloat16: "bf16" }
    for arg in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]):
      fn, ti, to, ci, co = arg[0], self.render_dtype(arg[2]), self.render_dtype(arg[3]), dt_map[arg[2]], dt_map[arg[3]]
      prefix.append(f"""__device__ {to}4 __{fn}({ti}8 a, {ti}4 b, {to}4 c) {{ int *a_pk = (int *) (&a), *b_pk = (int *) (&b);
asm( "mma.sync.aligned.m16n8k16.row.col.{co}.{ci}.{ci}.{co} {{ %0, %1, %2, %3 }}, {{ %4, %5, %6, %7 }}, {{ %8, %9 }}, {{ %0, %1, %2, %3 }};"
  : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w) : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]),  "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]) );
return c;}}""")

    return super().render_kernel(function_name, kernel, bufs, uops, prefix=prefix)

  def get_kernel_modifier(self, uops:List[UOp]) -> str:
    maxThreadsPerBlock = prod(u.arg[1] for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == "l")
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    return f"__launch_bounds__({maxThreadsPerBlock}) "

code_for_op_hip = { UnaryOps.SQRT: lambda x,dtype: f"__ocml_sqrt_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.SIN: lambda x,dtype: f"__ocml_sin_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.LOG2: lambda x,dtype: f"__ocml_log2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.EXP2: lambda x,dtype: f"__ocml_exp2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    # TODO: MAX with int uses fmax_f32?
                    BinaryOps.MAX: lambda a,b,dtype: f"__ocml_fmax_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32) }({a},{b})",}

def _make_hip_code_for_op():
  def wrapper(key, func):
    def cast_bf16(*args):
      if args[-1] == dtypes.bfloat16:
        operands = tuple(f"(float)({arg})" for arg in (args[1:-1] if key is TernaryOps.WHERE else args[:-1]))
        return f"(hip_bfloat16)({func(*(((args[0],) if key is TernaryOps.WHERE else ()) + operands), dtypes.float)})"
      return func(*args)
    return cast_bf16
  return { k:wrapper(k,v) for k,v in {**CStyleLanguage().code_for_op, **code_for_op_hip}.items() }

class AMDRenderer(CStyleLanguage):
  device = "AMD"
  shared_max = 65536
  tensor_cores = [TensorCore(dims=(16,16,16), threads=[(0,8),(0,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in [(dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]] # noqa: E501

  # language options
  ockl = [(f"__ockl_get_{name}", "unsigned int", "size_t", "const") for name in ["local_id", "group_id", "local_size"]]
  ocml = [(f"__ocml_{name}_f{n}", f"{dt}, {dt}" if "fmax" == name else dt, dt, atr)
            for dt, n in [(dtype.name, dtype.itemsize * 8) for dtype in [dtypes.float, dtypes.double, dtypes.half]]
            for name, atr in [("fmax", "const"), ("exp2", "pure"), ("log2", "pure"), ("sqrt", "const"), ("sin", "")]]

  kernel_prefix = "\n".join(f'extern "C" __attribute__((device{f", {atr}" if atr else ""})) {dto} {meth}({dti});' for meth,dti,dto,atr in ockl+ocml)
  kernel_prefix += '\nextern "C" __attribute__((global))'
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = _make_hip_code_for_op()
  smem_prefix = "__attribute__((shared))"
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  float4 = "make_float4"
  uses_ptr_arithmetic = False  # NOTE: this fixes TestLinearizerOverflowAlt
  type_map = {dtypes.bfloat16: "hip_bfloat16"}

  def render_vector_prefix(self, dtype:DType) -> str:
    vec, scal = self.render_dtype(dtype), self.render_dtype(dtype.scalar())
    return f"typedef {scal} {vec} __attribute__((ext_vector_type({dtype.count})));\nstatic inline __attribute__((device)) "+ \
           f"{vec} make_{vec}({', '.join([f'{scal} {x}' for x in _nms[:dtype.count]])}) {{ return {{ {', '.join(_nms[:dtype.count])} }}; }}"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = ["#define INFINITY (__builtin_inff())","#define NAN (__builtin_nanf(\"\"))","typedef long unsigned int size_t;","#define half _Float16"]

    # TODO: add BF16 vec dts
    if any(uop.dtype == dtypes.bfloat16 for uop in uops): prefix.append("""
struct hip_bfloat16 {
  unsigned short data;
  inline __attribute__((device)) hip_bfloat16(float val) {
    union { float fp32; unsigned int u32; } u = {val};
    if (~u.u32 & 0x7f800000) { u.u32 += 0x7fff + ((u.u32 >> 16) & 1); } else if (u.u32 & 0xffff) { u.u32 |= 0x10000; }
    data = (u.u32 >> 16);
  }
  inline __attribute__((device)) operator float() const {
    unsigned int uval = data << 16;
    return *reinterpret_cast<float*>(&uval);
  }
};
static inline __attribute__((device)) bool operator<(hip_bfloat16 a, hip_bfloat16 b) { return ((float)a) < ((float)b); }
static inline __attribute__((device)) bool operator==(hip_bfloat16 a, hip_bfloat16 b) { return ((float)a) == ((float)b); }
""")

    for dtype in dedup(uop.dtype for uop in uops if uop.dtype.count > 1): prefix.append(self.render_vector_prefix(dtype))

    for arg in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]): # TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
      if arg[3] == dtypes.float: prefix.append(f"#define __{arg[0]} __builtin_amdgcn_wmma_f32_16x16x16_f16_w32")
      else: prefix.append(f"static inline __attribute__((device)) half8 __{arg[0]}"+"""(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def get_kernel_modifier(self, uops:List[UOp]) -> str:
    requiredMaxThreadsPerBlock = prod(u.arg[1] for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == "l")
    # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
    # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
    return f"__attribute__((amdgpu_flat_work_group_size(1, {requiredMaxThreadsPerBlock})))"

class DSPRenderer(ClangRenderer):
  device = "DSP"
  supports_float4 = False
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_prefix = "__attribute__((noinline)) "
  type_map = { **ClangRenderer().type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {**ClangRenderer().code_for_op, UnaryOps.SIN: lambda x,dtype: f"__builtin_sin({x})",
                 UnaryOps.LOG2: lambda x,dtype: f"__builtin_log2l({x})" if dtype == dtypes.float64 else f"__builtin_log2f({x})",
                 UnaryOps.EXP2: lambda x,dtype: f"__builtin_exp2l({x})" if dtype == dtypes.float64 else f"__builtin_exp2f({x})"}

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    ret = super().render_kernel(function_name, kernel, bufs, uops, prefix)
    msrc = ['''struct dcvs_v2_req { int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency; _Bool set_dcvs_params;
                 short _pad2; char target_corner; char min_corner; char max_corner; int _pad3[3]; };''', 'int HAP_power_set(void*, void*);',
            'typedef union { struct { void *pv; unsigned int len; } buf; struct { int fd; unsigned int offset; } dma; } remote_arg;',
            'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);', 'int HAP_munmap(void *addr, int len);',
            'unsigned long long HAP_perf_get_time_us(void);', 'int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'struct dcvs_v2_req req = {.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner = 6 /* TURBO */};',
            'HAP_power_set((void*)handle, (void*)&req);']
    msrc += ['if ((sc>>24) != 2) return 0;']
    msrc += [f'int sz_or_val_{i} = ((int*)pra[0].buf.pv)[{i}];' for i,b in enumerate(bufs)]
    msrc += [f'int off{i} = ((int*)pra[1].buf.pv)[{i}];' for i,b in enumerate(bufs) if isinstance(b[1][0], PtrDType)]
    msrc += [f'void *buf_{i} = HAP_mmap(0,sz_or_val_{i},3,0,pra[{i+3}].dma.fd,0)+off{i};' for i,b in enumerate(bufs) if isinstance(b[1][0], PtrDType)]
    msrc += ["unsigned long long start = HAP_perf_get_time_us();"]
    msrc += [f"{function_name}({', '.join([(f'buf_{i}' if isinstance(b[1][0], PtrDType) else f'sz_or_val_{i}') for i,b in enumerate(bufs)])});"]
    msrc += ["*(unsigned long long *)(pra[2].buf.pv) = HAP_perf_get_time_us() - start;"]
    msrc += [f'HAP_munmap(buf_{i}, sz_or_val_{i});' for i,b in enumerate(bufs) if isinstance(b[1][0], PtrDType)]
    msrc += ["return 0; }"]
    return ret + '\n' + '\n'.join(msrc)

class NVRenderer(CUDARenderer): device = "NV"
class HIPRenderer(AMDRenderer): device = "HIP"
class QCOMRenderer(OpenCLRenderer): device = "QCOM"
