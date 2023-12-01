import functools
from tinygrad.renderer.cstyle import CStyleLanguage, uops_to_cstyle

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
