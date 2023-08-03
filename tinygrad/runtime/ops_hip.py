import numpy as np
import ctypes
from typing import List, Tuple
from tinygrad.helpers import DEBUG, getenv, DType
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

HIP_CPU=getenv("HIP_CPU")
if HIP_CPU: 
  import extra.hip_cpu_wrapper as hip #type: ignore
else:
  import extra.hip_wrapper as hip #type: ignore

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 5:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.

class RawHIPBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype): super().__init__(size, dtype, hip.hipMalloc(size * dtype.itemsize))
  def __del__(self): hip.hipFree(self._buf)
  def _copyin(self, x:np.ndarray): hip.hipMemcpy(self._buf, x.ctypes.data_as(ctypes.c_void_p), self.size * self.dtype.itemsize, hip.hipMemcpyHostToDevice)
  def _copyout(self, x:np.ndarray): hip.hipMemcpy(x.ctypes.data_as(ctypes.c_void_p), self._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToHost)

class HIPProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if not binary:
        if HIP_CPU:
          prg = "#include <hip/hip_runtime.h>\n" +  prg
        else:
          prog = hip.hiprtcCreateProgram(prg, name, [], [])
          device_properties = hip.hipGetDeviceProperties(hip.hipGetDevice())
          hip.hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}'])
          prg = hip.hiprtcGetCode(prog)
    except Exception as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
    if HIP_CPU:
      import tempfile, hashlib, os, subprocess
      fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.so"
      if not os.path.exists(fn):
        subprocess.check_output(args=('clang -g -std=c++20 --stdlib=libstdc++ -shared -O2 -Wall -Werror -x c++ -ltbb -ltbbmalloc -fPIC --rtlib=compiler-rt - -o '+fn+'.tmp').split(), input=prg.encode('utf-8'))
        os.rename(fn+'.tmp', fn)
      self.lib = ctypes.CDLL(fn)
      self.prg = self.lib[f"launch_kernel_{name}"]
    else:
      module = hip.hipModuleLoadData(prg)
      self.prg = hip.hipModuleGetFunction(module, name)

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    if HIP_CPU:
      local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
      global_size = global_size + [1] * (3 - len(global_size))
      self.prg(global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, ctypes.c_void_p(0), *[data._buf for data in args])
    else:
      class PackageStruct(ctypes.Structure):
        _fields_ = [(f'field{idx}', ctypes.c_void_p) for idx in range(len(args))]
      struct = PackageStruct(*[data._buf for data in args])
      hip.hipModuleLaunchKernel(self.prg, global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      return hip.hipEventElapsedTime(start, end)*1e-3


class HIPLanguage(CStyleLanguage):
  def render_kernel(self, kernel: List[str], bufs: List[Tuple[str, DType]], global_size: List[int], local_size: List[int], prekernel: List[str]) -> Tuple[str, List[int], List[int]]:
    prg, gs, ls = super().render_kernel(kernel, bufs, global_size, local_size, prekernel)
    return prg + (f"""
extern "C" void launch_kernel_KERNEL_NAME_PLACEHOLDER(
  std::uint32_t grid_dim_x,
  std::uint32_t grid_dim_y,
  std::uint32_t grid_dim_z,
  std::uint32_t block_dim_x,
  std::uint32_t block_dim_y,
  std::uint32_t block_dim_z,
  std::uint32_t shared_mem_bytes,
  hipStream_t stream,
  {",".join([f"{buf[1].name}* {buf[0]}" for buf in bufs])}
  ) {{ 
    hipLaunchKernelGGL(KERNEL_NAME_PLACEHOLDER, dim3(grid_dim_x, grid_dim_y, grid_dim_z), dim3(block_dim_x, block_dim_y, block_dim_z), shared_mem_bytes, stream, {", ".join([buf[0] for buf in bufs])});
    hipStreamSynchronize(stream);
}}\n""" if HIP_CPU else ""), gs, ls
class HIPCodegen(CStyleCodegen):
  lang = HIPLanguage(
    kernel_prefix = (
      "#include <hip/hip_common.h>\n#define INFINITY (__builtin_inff())\n#define NAN (__builtin_nanf(\"\"))" if not HIP_CPU else "#include <math.h>\ntypedef unsigned char uchar;\ninline float max(float x, float y) { return fmax(x, y); }") + 
"""
__device__ float4 max(float4 x, float4 y) { return float4(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w)); }
__device__ float4 pow(float x, float4 y) { return float4(pow(x, y.x), pow(x, y.y), pow(x, y.z), pow(x, y.w)); }
__device__ float4 pow(float4 x, float4 y) { return float4(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w)); }
__device__ float4 log2(float4 x) { return float4(log2(x.x), log2(x.y), log2(x.z), log2(x.w)); }
__device__ float4 exp2(float4 x) { return float4(exp2(x.x), exp2(x.y), exp2(x.z), exp2(x.w)); }
__device__ float4 sin(float4 x) { return float4(sin(x.x), sin(x.y), sin(x.z), sin(x.w)); }
extern "C" __global__
    """, smem_prefix = "__shared__ ", barrier = "__syncthreads();" if not HIP_CPU else "", float4 = "make_float4", uses_vload=True,
    half_prekernel = ("#include <hip/hip_fp16.h>\nusing half4 = HIP_vector_type<half, 4>;\n" if not HIP_CPU else "using half_float::half; MAKE_VECTOR_TYPE(half, half);\n") + 
"""
__device__ float vload_half(size_t offset, const half *p) { return (float)*(p + offset); }
__device__ float2 vload_half2(size_t offset, const half *p) { return make_float2((float)*(p + offset*2), (float)*(p + offset*2 + 1)); }
__device__ float4 vload_half4(size_t offset, const half *p) { return make_float4((float)*(p + offset*4), (float)*(p + offset*4 + 1), (float)*(p + offset*4 + 2), (float)*(p + offset*4 + 3)); }
__device__ void vstore_half(float data, size_t offset, half *p) { *(p + offset) = (half)data; }
__device__ void vstore_half2(float2 data, size_t offset, half *p) { *(p + offset*2) = (half)data.x; *(p + offset*2 + 1) = (half)data.y; }
__device__ void vstore_half4(float4 data, size_t offset, half *p) { *(p + offset*4) = (half)data.x; *(p + offset*4 + 1) = (half)data.y; *(p + offset*4 + 2) = (half)data.z; *(p + offset*4 + 3) = (half)data.w; }
    """,
    gid = [f'blockIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

HIPBuffer = Compiled(RawHIPBuffer, HIPCodegen, HIPProgram, hip.hipDeviceSynchronize)
