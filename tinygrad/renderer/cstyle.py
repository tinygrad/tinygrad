from typing import Dict, List, Optional, NamedTuple, Tuple, Union, DefaultDict, cast
import math, functools
from collections import defaultdict
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import ImageDType, dtypes, prod, DType, strip_parens

class CStyleLanguage(NamedTuple):
  size_prefix: str = "int"
  generic_var_prefix: str = ""
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = ""
  barrier: str = ""
  xid: List[str] = []
  gid: List[str] = []
  lid: List[str] = []
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  half_prekernel: Optional[str] = None
  uses_vload: bool = False
  external_local_bufs: bool = False
  uses_ptr_arithmetic: bool = False
  launch_bounds: bool = False
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(-{x})" if dtype != dtypes.bool else f"(!{x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})",
    UnaryOps.SIN: lambda x,dtype: f"sin({x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.SUB: lambda a,b,dtype: f"({a}-{b})",
    BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})", BinaryOps.DIV: lambda a,b,dtype: f"({a}/{b})",
    BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", TernaryOps.MULACC: lambda a,b,c,dtype: f"(({a}*{b})+{c})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}!=0?{b}:{c})"
  }

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], var_dtype:DType) -> str:
    if len(x) == 1: return f"({var_dtype.name})({x[0]})"
    assert len(x) == var_dtype.sz, f"cast is wrong size {len(x)} != {var_dtype.sz}"
    assert self.float4 is not None, "vectorized cast is not supported on this platform"
    return f"{self.float4.replace('float4', var_dtype.name)}({','.join(x)})"

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    else: val = f"{float(x)}f" if dtypes.is_float(var_dtype) else f"{int(x)}"
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 else val

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes.float.vec(4), f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vload_half{'' if output_dtype.sz == 1 else str(output_dtype.sz)}(0, {buf_name}+{idx})"
    if output_dtype.sz > 1:
      out_val = f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{output_dtype.sz}*)({buf_name}+{idx}))"
    else:
      out_val = f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"

    return self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val

  def render_local(self, name:str, size:int):
    return self.smem_align + self.smem_prefix + f"float {name}[{size}];"

  def render_for(self, expr: str, _min:Union[int,str], _max:Union[int,str]) -> str:
    return f"for (int {expr} = {_min}; {expr} < {_max}; ++{expr}) {{"

  def render_if(self, cond: str):
    return f"if ({cond}) {{"

  def render_conditional(self, cond: str, x:str, y:str) -> str:
    return f"({cond})?({x}):{y}"

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,dtype in bufs) else ""
    buftypes = [(name,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if dtype.name.startswith('image') else
                self.arg_int_prefix if dtype == dtypes._arg_int32 else
                ("const " if i > 0 else "")+self.buffer_prefix+dtype.name+"*"+self.buffer_suffix) for i,(name,dtype) in enumerate(bufs)]
    prg = ''.join([f"{self.kernel_prefix}void {f'__launch_bounds__ ({prod(local_size)}, 1) ' if self.launch_bounds else ''}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    if self.half_prekernel and any(dtype == dtypes.float16 for _,dtype in bufs): prg = ''.join([f"{self.half_prekernel}", "\n", prg])
    return prg

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes.float.vec(4), "images must be float4"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype == dtypes.float16 and var_dtype != dtypes.float16:
      return f"vstore_half{'' if var_dtype.sz == 1 else str(var_dtype.sz)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.sz > 1:
      return f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{var_dtype.sz}*)({buf_name}+{idx})) = ({buf_dtype.name}{var_dtype.sz}){var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

def uops_to_cstyle(lang:CStyleLanguage, function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
  local_size: List[int] = []
  kernel,prekernel,bufs = [],[],[]
  #pend_close = None
  depth = 1
  def kk(s): kernel.append("  "*depth+s)

  c: DefaultDict[str, int] = defaultdict(int)
  r: Dict[UOp, str] = {}
  def ssa(u, prefix="t"):
    nonlocal c, r
    c[prefix] += 1
    r[u]=f"{prefix}{c[prefix]-1}"
    return r[u]

  child_count: DefaultDict[UOp, int] = defaultdict(int)
  for ru in uops:
    for v in ru.vin:
      child_count[v] += 1

  for u in uops:
    uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
    if uop == UOps.LOOP:
      kk(lang.render_for(ssa(u,'ridx'), r[vin[0]], r[vin[1]]))
      depth += 1
    elif uop == UOps.IF:
      kk(lang.render_if(r[vin[0]]))
      depth += 1
    elif uop == UOps.BARRIER:
      kk(lang.barrier)
    elif uop == UOps.END:
      depth -= 1
      kk("}")
    elif uop == UOps.WMMA:
      if args[0] == "METAL":
        assert dtype == dtypes.float.vec(2), "output dtype of METAL TC is _float2"
        # ((lidx2*32)+(lidx3*4)+(lidx4*16)+(lidx5*8)+(lidx6*2))
        output = ssa(u, 'wmma')
        kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {output};")
        kk("{ simdgroup_float8x8 a,b,c;")
        kk(f"a.thread_elements()[0] = {r[vin[0]]}; a.thread_elements()[1] = {r[vin[1]]};")
        kk(f"b.thread_elements()[0] = {r[vin[2]]}; b.thread_elements()[1] = {r[vin[3]]};")
        kk(f"c.thread_elements()[0] = {r[vin[4]]}; c.thread_elements()[1] = {r[vin[5]]};")
        kk("simdgroup_multiply_accumulate(c, a, b, c);")
        kk(f"{output}.x = c.thread_elements()[0]; {output}.y = c.thread_elements()[1]; }}")
      elif args[0] == "HIP":
        assert dtype == dtypes.float.vec(8), "output dtype of HIP TC is _float8"
        kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u, 'wmma')} = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32({r[vin[0]]}, {r[vin[1]]}, {r[vin[2]]});")
      else:
        raise NotImplementedError(f"WMMA not implemented for {args}")
    elif uop == UOps.ALU:
      assert dtype is not None
      # remove parens if ALU types are the same. TODO: can do more here
      if vin[0].uop == UOps.ALU and vin[0].arg == args and args in {BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL}:
        val = lang.code_for_op[args](strip_parens(r[vin[0]]), *[r[x] for x in vin[1:]], dtype)
      elif args == BinaryOps.MAX:
        val = lang.code_for_op[args](*[lang.render_cast([r[x]], dtype) if x.dtype != dtype else r[x] for x in vin] + [dtype])
      else:
        val = lang.code_for_op[args](*[r[x] for x in vin] + [dtype])
      assert child_count[u] != 0, f"childless ALU op found {u}"
      if (child_count[u] <= 1 or dtypes.is_int(dtype)) and args != BinaryOps.MAX:  # fix index rendering issue. fix clang nested max macro issue
        r[u] = val
      else:
        kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'alu')} = {val};")
    elif uop == UOps.DEFINE_ACC:
      assert dtype is not None
      kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'acc')} = {lang.render_const(args, dtype)};")
    elif uop == UOps.SPECIAL:
      xid = lang.gid if args[1].startswith("g") else (lang.xid if args[1].startswith("i") else lang.lid)
      kk(f"{lang.size_prefix} {args[1]} = {xid[args[0]]}; /* {args[2]} */")
      if args[1].startswith("l"): local_size.append(args[2])
      r[u] = args[1]
    elif uop == UOps.CONST:
      r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
    elif uop == UOps.LOAD:
      assert dtype is not None
      val = lang.render_load(dtype, r[vin[0]], vin[0].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL)
      if len(vin) > 3: val = lang.render_conditional(r[vin[2]], val, r[vin[3]])
      kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'val')} = {val};")
    elif uop == UOps.PHI:
      kk(f"{r[vin[0]]} = {r[vin[1]]};")
      r[u] = r[vin[0]]
    elif uop == UOps.STORE:
      assert vin[0].dtype is not None and vin[2].dtype is not None
      if len(vin) > 3: kk(lang.render_if(r[vin[3]]))
      kk(lang.render_store(r[vin[0]], vin[0].dtype, r[vin[2]], vin[2].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL))
      if len(vin) > 3: kk("}")
    elif uop == UOps.CAST and dtype is not None:
      val = lang.render_cast([r[x] for x in vin], dtype)
      if child_count[u] <= 1: r[u] = val
      else: kk(f"{lang.generic_var_prefix if lang.generic_var_prefix else dtype.name} {ssa(u,'cast')} = {val};")
    elif uop == UOps.DEFINE_LOCAL:
      if lang.external_local_bufs:
        prekernel.append(lang.render_local(args[0], args[1]))
      else:
        kk(lang.render_local(args[0], args[1]))
      r[u] = args[0]
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
      r[u] = args[0]
    elif uop == UOps.GEP:
      if cast(DType, vin[0].dtype).sz > 4:
        r[u] = f"({r[vin[0]]})[{args}]"  # this is correct for HIP
      else:
        r[u] = f"({r[vin[0]]}).{'xyzw'[args]}"
    else:
      raise RuntimeError(f"failed to render {uop}")

  return lang.render_kernel(function_name, kernel, bufs, local_size, prekernel), {}

class OpenCLLanguage(CStyleLanguage):
  kernel_prefix = "__kernel "
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  arg_int_prefix = "const int"
  half_prekernel = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  gid = [f'get_group_id({i})' for i in range(3)]
  lid = [f'get_local_id({i})' for i in range(3)]
  xid = [f'get_global_id({i})' for i in range(3)]
  uses_vload = True
  # NOTE: mad is used so the loads aren't reordered into the math on 845
  code_for_op = {**CStyleLanguage().code_for_op, TernaryOps.MULACC: lambda a,b,c,dtype: f"mad({a},{b},{c})"}
OpenCLRenderer = functools.partial(uops_to_cstyle, OpenCLLanguage())

class MetalLanguage(CStyleLanguage):
  kernel_prefix = "#include <metal_stdlib>\nusing namespace metal;\nkernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic=True
  gid = [f"gid.{chr(120+i)}" for i in range(3)]
  lid = [f"lid.{chr(120+i)}" for i in range(3)]
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
MetalRenderer = functools.partial(uops_to_cstyle, MetalLanguage())

class CUDALanguage(CStyleLanguage):
  kernel_prefix = "#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\nextern \"C\" __global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  arg_int_prefix = "const int"
  barrier = "__syncthreads();"
  float4 = "make_float4"
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)]
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)]
  xid = [f'(blockIdx.{chr(120+i)}*blockDim.{chr(120+i)}+threadIdx.{chr(120+i)})' for i in range(3)]
  half_prekernel = """
    #include <cuda_fp16.h>
    #include <mma.h>
    using namespace nvcuda;
    struct __align__(8) half4 {
      half2 x, y;
      __device__ __forceinline__ explicit half4(const float4& a): x(make_half2(__float2half(a.x), __float2half(a.y))), y(make_half2(__float2half(a.z),__float2half(a.w))) {}
      __device__ __forceinline__ explicit operator float4() const {return make_float4(__half2float(x.x), __half2float(x.y), __half2float(y.x), __half2float(y.y)); }
    };
    """
CUDARenderer = functools.partial(uops_to_cstyle, CUDALanguage())

class HIPLanguage(CStyleLanguage):
  kernel_prefix = "#include <hip/hip_common.h>\n#define INFINITY (__builtin_inff())\n#define NAN (__builtin_nanf(\"\"))" + """
  __device__ float4 max(float4 x, float4 y) { return float4(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w)); }
  __device__ float4 pow(float x, float4 y) { return float4(pow(x, y.x), pow(x, y.y), pow(x, y.z), pow(x, y.w)); }
  __device__ float4 pow(float4 x, float4 y) { return float4(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w)); }
  __device__ float4 log2(float4 x) { return float4(log2(x.x), log2(x.y), log2(x.z), log2(x.w)); }
  __device__ float4 exp2(float4 x) { return float4(exp2(x.x), exp2(x.y), exp2(x.z), exp2(x.w)); }
  __device__ float4 sin(float4 x) { return float4(sin(x.x), sin(x.y), sin(x.z), sin(x.w)); }
  typedef float float8 __attribute__((ext_vector_type(8))); __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  """
  launch_bounds = True
  smem_prefix = "__shared__ "
  smem_prefix_for_cast=False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  uses_vload=True
  uses_ptr_arithmetic=True
  arg_int_prefix = "const int"
  half_prekernel = "#include <hip/hip_fp16.h>\n" + """
typedef union { struct { half x, y, z, w; } __attribute__((aligned(8))); half data[4]; } half4; __device__ half4 make_half4(half x, half y, half z, half w) { return {x, y, z, w}; }
typedef union { struct { half x, y, z, w, a, b, c, d; } __attribute__((aligned(16))); half data[8]; } half8; __device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { return {x, y, z, w, a, b, c, d}; }
 typedef _Float16 half16 __attribute__((ext_vector_type(16))); __device__ half16 make_half16(half x, half y, half z, half w, half a, half b, half c, half d, half e, half f, half g, half h, half i, half j, half k, half l) { return {x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; }
__device__ float vload_half(size_t offset, const half *p) { return (float)*(p + offset); }
__device__ float2 vload_half2(size_t offset, const half *p) { return make_float2((float)*(p + offset*2), (float)*(p + offset*2 + 1)); }
__device__ float4 vload_half4(size_t offset, const half *p) { return make_float4((float)*(p + offset*4), (float)*(p + offset*4 + 1), (float)*(p + offset*4 + 2), (float)*(p + offset*4 + 3)); }
__device__ void vstore_half(float data, size_t offset, half *p) { *(p + offset) = (half)data; }
__device__ void vstore_half2(float2 data, size_t offset, half *p) { *(p + offset*2) = (half)data.x; *(p + offset*2 + 1) = (half)data.y; }
__device__ void vstore_half4(float4 data, size_t offset, half *p) { *(p + offset*4) = (half)data.x; *(p + offset*4 + 1) = (half)data.y; *(p + offset*4 + 2) = (half)data.z; *(p + offset*4 + 3) = (half)data.w; }
  """
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)]
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)]
  xid = [f'(blockIdx.{chr(120+i)}*blockDim.{chr(120+i)}+threadIdx.{chr(120+i)})' for i in range(3)]
HIPRenderer = functools.partial(uops_to_cstyle, HIPLanguage())

# TODO: how much of this can be merged with above?
class WGSLLanguage(CStyleLanguage):
  gid = [f"i32(gindex.{'xyz'[x]})" for x in range(3)]
  lid = [f"i32(lindex.{'xyz'[x]})" for x in range(3)]
  size_prefix = "let"
  barrier="workgroupBarrier();"
  generic_var_prefix = "var "
  external_local_bufs = True
  code_for_op = { **CStyleLanguage().code_for_op, BinaryOps.CMPLT: lambda x,y,dtype: f"f32({x}<{y})", TernaryOps.MULACC: lambda x,y,z,dtype: f"fma({x},{y},{z})", TernaryOps.WHERE: lambda a,b,c,dtype: f"select({c},{b},{a}!=0.)" }
  type_map = {dtypes.float: "f32", dtypes.half: "f16", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool"}

  def render_local(self, name: str, size: int):
    return f"var<workgroup> {name}: array<f32,{size}>;"

  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): val = "nan()"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "0x1.fffffep+127f"
    else: val = f"({x}" + ("" if dtypes.is_int(var_dtype) else "f") + ")"
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 else val

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> str:
    local_size = local_size[::-1] if local_size else [1]
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n"
    prg += "\n".join(prekernel+[f"@group(0) @binding({next(bind_it)}) var<storage,read_write> {name}: array<{self.type_map[dtype]}>;" for name,dtype in bufs])
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"
    return prg

  def render_for(self, expr:str, _min:Union[int,str], _max:Union[int,str]) -> str:
    return f"for(var {expr} = {_min}; {expr} < {_max}; {expr}++) {{"

  def render_if(self, cond: str):
    return f"if (bool({cond})) {{"

  def render_conditional(self, cond:str, x:str, y:str) -> str:
    return f"select(f32({y}), {x}, bool({cond}))"

  def render_cast(self, x:List[str], var_dtype:DType) -> str:
    if self.type_map[var_dtype]: return f"{self.type_map[var_dtype]}({x[0]})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    return f"f32({super().render_load(output_dtype, buf_name, buf_dtype, idx, local)})"

  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx, local=False) -> str:
    if buf_dtype != var_dtype:
      var_name = f"{self.type_map[buf_dtype]}({var_name})"
    return f"{buf_name}[{idx}] = {var_name};"
WGSLRenderer = functools.partial(uops_to_cstyle, WGSLLanguage())
