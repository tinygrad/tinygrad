from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast, Literal, Callable
import math, functools
from collections import defaultdict, Counter
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import prod, strip_parens, getenv
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType

class CStyleLanguage(NamedTuple):
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  half_prekernel: Optional[str] = None
  uses_vload: bool = False
  uses_ptr_arithmetic: bool = False
  launch_bounds: bool = False
  type_map: Dict[DType, str] = {}
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})",
    BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPEQ: lambda a,b,dtype: f"({a}=={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    TernaryOps.MULACC: lambda a,b,c,dtype: f"(({a}*{b})+{c})", TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"
  }

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], var_dtype:DType, bitcast=False) -> str:
    if bitcast: return f"(*(({self.buffer_prefix}{self.render_dtype(var_dtype)}*)&{x[0]}))"
    if len(x) == 1: return f"({self.render_dtype(var_dtype)})({x[0]})"
    assert len(x) == var_dtype.sz, f"cast is wrong size {len(x)} != {var_dtype.sz}"
    assert self.float4 is not None, "vectorized cast is not supported on this platform"
    return f"{self.float4.replace('float4', self.render_dtype(var_dtype))}({','.join(x)})"

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int,bool], var_dtype) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    else: val = f"{float(x)}f" if dtypes.is_float(var_dtype) else f"{int(x)}" if dtypes.is_int(var_dtype) else f"{bool(x)}".lower()
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 or var_dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes.float.vec(4), f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and output_dtype.scalar() != dtypes.float16:
      return f"vload_half{'' if output_dtype.sz == 1 else str(output_dtype.sz)}(0, {buf_name}+{idx})"
    if output_dtype.sz > 1:
      out_val = f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{output_dtype.sz}*)({buf_name}+{idx}))"  # noqa: E501
    else:
      out_val = f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"
    return self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int]) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,dtype in bufs) else ""  # noqa: E501
    buftypes = [(name,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if dtype.name.startswith('image') else
                ("const " if i > 0 else "")+self.buffer_prefix+self.render_dtype(dtype)+"*"+self.buffer_suffix if isinstance(dtype, PtrDType) else
                self.arg_int_prefix if dtype == dtypes.int else None) for i,(name,dtype) in enumerate(bufs)]
    prg = ''.join([f"{self.kernel_prefix}void {f'__launch_bounds__ ({prod(local_size)}, 1) ' if self.launch_bounds else ''}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    if self.half_prekernel and any(dtype in [dtypes.float16, dtypes.bfloat16] for _,dtype in bufs): prg = ''.join((self.half_prekernel, "\n", prg))
    return prg

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes.float.vec(4), f"images must be float4, getting {var_dtype}"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and var_dtype.scalar() != dtypes.float16:
      return f"vstore_half{'' if var_dtype.sz == 1 else str(var_dtype.sz)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.sz > 1:
      return f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{var_dtype.sz}*)({buf_name}+{idx})) = ({buf_dtype.name}{var_dtype.sz}){var_name};"  # noqa: E501
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{dtype.name} {name}[{size}];"
  def render_dtype(self, var_dtype:DType) -> str: return self.type_map[var_dtype] if var_dtype in self.type_map else var_dtype.name

def uops_to_cstyle(lang:CStyleLanguage, function_name:str, uops:List[UOp]) -> str:
  local_size: List[int] = []
  kernel,bufs = [],[]
  #pend_close = None
  depth = 1
  def kk(s): kernel.append("  "*depth+s)

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  def ssa(u, prefix="t"):
    nonlocal c, r
    ret = f"{prefix}{c[prefix]}"
    if u is not None: r[u] = ret
    c[prefix] += 1
    return ret

  child_count = Counter(v for ru in uops for v in ru.vin)

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    # these four uops don't have output dtypes
    if uop == UOps.IF:
      kk(f"if ({r[vin[0]]}) {{")
      depth += 1
    elif uop == UOps.BARRIER: kk(lang.barrier)
    elif uop == UOps.END:
      depth -= 1
      kk("}")
    elif uop == UOps.STORE:
      assert vin[0].dtype is not None and vin[2].dtype is not None
      if len(vin) > 3: kk(f"if ({r[vin[3]]}) {{")
      kk(lang.render_store(r[vin[0]], vin[0].dtype, r[vin[2]], vin[2].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL))
      if len(vin) > 3: kk("}")
    else:
      assert dtype is not None, f"None dtype for uop {uop}"
      if uop == UOps.LOOP:
        kk(f"for (int {(expr := ssa(u,'ridx'))} = {r[vin[0]]}; {expr} < {r[vin[1]]}; {expr}++) {{")
        depth += 1
      elif uop == UOps.ALU:
        # remove parens if ALU types are the same. TODO: can do more here
        if vin[0].uop == UOps.ALU and vin[0].arg == args and args in {BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL, BinaryOps.XOR}:
          val = lang.code_for_op[args](strip_parens(r[vin[0]]), *[r[x] for x in vin[1:]], dtype)
        else:
          val = lang.code_for_op[args](*[r[x] for x in vin] + [dtype])
        assert child_count[u] != 0, f"childless ALU op found {u}"
        # TODO: fix index rendering issue. fix clang nested max macro issue
        if child_count[u] <= 1 and args != BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val
        else: kk(f"{dtype.name} {ssa(u,'alu')} = {val};")
      elif uop == UOps.SPECIAL:
        kk(f"int {args[1]} = {lang.code_for_workitem[args[1][0]](args[0])}; /* {args[2]} */")
        if args[1].startswith("l"): local_size.append(args[2])
        r[u] = args[1]
      elif uop == UOps.LOAD:
        val = lang.render_load(dtype, r[vin[0]], vin[0].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL)
        # NOTE: this relies on the load not happening if it's in the unselected branch
        if len(vin) > 3: val = lang.code_for_op[TernaryOps.WHERE](r[vin[2]], val, r[vin[3]], dtype)
        kk(f"{lang.render_dtype(dtype)} {ssa(u,'val')} = {val};")
      elif uop == UOps.PHI:
        kk(f"{r[vin[0]]} = {r[vin[1]]};")
        r[u] = r[vin[0]]
      elif uop == UOps.CAST:
        if isinstance(args, tuple) and args[1]:  # bitcast
          assert len(vin) == 1
          precast = ssa(None,'precast')
          kk(f"{lang.render_dtype(cast(DType, vin[0].dtype))} {precast} = {r[vin[0]]};")
          val = lang.render_cast([precast], dtype, bitcast=True)
        else:
          val = lang.render_cast([r[x] for x in vin], dtype, bitcast=False)
        if child_count[u] <= 1: r[u] = val
        else: kk(f"{dtype.name} {ssa(u,'cast')} = {val};")
      elif uop == UOps.DEFINE_LOCAL:
        kk(lang.render_local(args[0], dtype, args[1]))
        r[u] = args[0]
      elif uop == UOps.DEFINE_GLOBAL:
        bufs.append((args, dtype))
        r[u] = args
      elif uop == UOps.WMMA: kk(f"{dtype.name} {ssa(u, 'wmma')} = {args}({r[vin[0]]}, {r[vin[1]]}, {r[vin[2]]});")
      elif uop == UOps.DEFINE_ACC: kk(f"{dtype.name} {ssa(u,'acc')} = {lang.render_const(args, dtype)};")
      elif uop == UOps.CONST: r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
      elif uop == UOps.GEP: r[u] = f"({r[vin[0]]})[{args}]" if cast(DType, vin[0].dtype).sz > 4 else f"({r[vin[0]]}).{'xyzw'[args]}"
      else: raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, local_size)

class OpenCLLanguage(CStyleLanguage):
  kernel_prefix = "__kernel "
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  half_prekernel = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  code_for_workitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}
  uses_vload = True
  # NOTE: mad is used so the loads aren't reordered into the math on 845
  code_for_op = {**CStyleLanguage().code_for_op,
                 TernaryOps.MULACC: lambda a,b,c,dtype: f"mad({a},{b},{c})" if dtypes.is_float(dtype) else f"(({a}*{b})+{c})"}
  type_map = { dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong" }
  def render_cast(self, x, var_dtype, bitcast=False) -> str:
    return f"as_{self.type_map.get(var_dtype) or var_dtype.name}({x[0]})" if bitcast else super().render_cast(x, var_dtype)
OpenCLRenderer = functools.partial(uops_to_cstyle, OpenCLLanguage())

class MetalLanguage(CStyleLanguage):
  kernel_prefix = """#include <metal_stdlib>\nusing namespace metal;\ntemplate<typename T, typename S, typename U> U __metal_wmma(T m, T n, U o) {
  S a,b,c; a.thread_elements()[0] = m.x; a.thread_elements()[1] = m.y; b.thread_elements()[0] = n.x; b.thread_elements()[1] = n.y;
  c.thread_elements()[0] = o.x; c.thread_elements()[1] = o.y; simdgroup_multiply_accumulate(c, a, b, c);
  return U(c.thread_elements()[0], c.thread_elements()[1]);\n}\nkernel """
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic = True
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+x)}", "l": lambda x: f"lid.{chr(120+x)}"}
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  def render_cast(self, x: List[str], var_dtype: DType, bitcast=False) -> str:
    return f"as_type<{var_dtype.name}>({x[0]})" if bitcast else super().render_cast(x, var_dtype)
MetalRenderer = functools.partial(uops_to_cstyle, MetalLanguage())

code_for_op_half = {
  BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})" if dtype != dtypes.half else f"__hmax({a},{b})",
  UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})" if dtype != dtypes.half else f"hsqrt({x})",
  UnaryOps.SIN: lambda x,dtype: f"sin({x})" if dtype != dtypes.half else f"hsin({x})",
  UnaryOps.LOG2: lambda x,dtype: f"log2({x})" if dtype != dtypes.half else f"hlog2({x})",
  UnaryOps.EXP2: lambda x,dtype: f"exp2({x})" if dtype != dtypes.half else f"hexp2({x})",
}

class CUDALanguage(CStyleLanguage):
  kernel_prefix = "#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\nextern \"C\" __global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+x)}", "l": lambda x: f"threadIdx.{chr(120+x)}",
                       "i": lambda x: f"(blockIdx.{chr(120+x)}*blockDim.{chr(120+x)}+threadIdx.{chr(120+x)})"}
  code_for_op = {**CStyleLanguage().code_for_op, **code_for_op_half}
  half_prekernel = "#include <cuda_fp16.h>\n"+"#include <cuda_bf16.h>\n"+"""
    struct half4 { half x, y, z, w; };
    __device__ half4 make_half4(half x, half y, half z, half w) { half4 ret; ret.x = x; ret.y = y; ret.z = z; ret.w = w; return ret; }
  """
  type_map = {dtypes.bfloat16: "nv_bfloat16"}
CUDARenderer = functools.partial(uops_to_cstyle, CUDALanguage())

code_for_op_hip = {
  BinaryOps.MAX: lambda a,b,dtype: f"__ocml_fmax_f32({a},{b})" if dtype != dtypes.half else f"__ocml_fmax_f16({a},{b})",
  UnaryOps.SQRT: lambda x,dtype: f"__ocml_sqrt_f32({x})" if dtype != dtypes.half else f"__ocml_sqrt_f16({x})",
  UnaryOps.SIN: lambda x,dtype: f"__ocml_sin_f32({x})" if dtype != dtypes.half else f"__ocml_sin_f16({x})",
  UnaryOps.LOG2: lambda x,dtype: f"__ocml_log2_f32({x})" if dtype != dtypes.half else f"__ocml_log2_f16({x})",
  UnaryOps.EXP2: lambda x,dtype: f"__ocml_exp2_f32({x})" if dtype != dtypes.half else f"__ocml_exp2_f16({x})",
}

def _make_hip_dtype(base_type, name, cnt):
  nms = "xyzwabcdefghijkl"[:cnt]
  return f"typedef {base_type} {name}{cnt} __attribute__((ext_vector_type({cnt})));\n" + \
         f"static inline __attribute__((device)) {name}{cnt} make_{name}{cnt}(" + ', '.join([f"{base_type} {x}" for x in nms]) + \
         ") { return {" + ', '.join(nms) + "}; }"

class HIPLanguage(CStyleLanguage):
  kernel_prefix = "#include <hip/hip_common.h>\n#define INFINITY (__builtin_inff())\n#define NAN (__builtin_nanf(\"\"))" + """
  #define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                       \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
  #define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)           \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock), amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
  #define select_impl_(_1, _2, impl_, ...) impl_
  #define __launch_bounds__(...) select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)
  typedef long unsigned int size_t;
  #define half _Float16
  struct hip_bfloat16 { unsigned short data; };

  extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
  extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
  extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);

  extern "C" {
  __attribute__((device)) __attribute__((const)) float __ocml_fmax_f32(float, float);
  __attribute__((device)) __attribute__((pure)) float __ocml_exp2_f32(float);
  __attribute__((device)) __attribute__((pure)) float __ocml_log2_f32(float);
  __attribute__((device)) float __ocml_sin_f32(float);
  __attribute__((device)) __attribute__((const)) float __ocml_sqrt_f32(float);
  __attribute__((device)) __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
  __attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
  __attribute__((device)) __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
  __attribute__((device)) _Float16 __ocml_sin_f16(_Float16);
  __attribute__((device)) __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
  }\n""" + '\n'.join([_make_hip_dtype(*x) for x in [("signed int", "int", 2),
                     ("_Float16", "half", 2), ("_Float16", "half", 4), ("_Float16", "half", 8), ("_Float16", "half", 16),
                     ("float", "float", 2), ("float", "float", 4), ("float", "float", 8)]]) + """
  static __attribute__((device)) half8 __hip_wmma_f16_f16(half16 a, half16 b, half8 c) {
    half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
    for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;
  }\nextern "C" __attribute__((global))"""
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = {**CStyleLanguage().code_for_op, **code_for_op_hip}
  smem_prefix = "__attribute__((shared))"
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  float4 = "make_float4"
  launch_bounds = True
  uses_ptr_arithmetic = True
  type_map = {dtypes.bfloat16: "hip_bfloat16"}
HIPRenderer = functools.partial(uops_to_cstyle, HIPLanguage())