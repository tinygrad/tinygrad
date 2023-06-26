import numpy as np
import ctypes, subprocess, tempfile, hashlib, os
from tinygrad.helpers import DEBUG, getenv
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 5:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

HIP_CPU = getenv('HIP_CPU', 0)
if HIP_CPU:
  import extra.hip_cpu_wrapper as hip
else:
  import extra.hip_wrapper as hip # type: ignore

# The default HIP stream is used for everything.

class RawHIPBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype):
    self.buf_sz = size * dtype.itemsize
    super().__init__(size, dtype, hip.hipMalloc(self.buf_sz))
  def _copyin(self, x:np.ndarray): hip.hipMemcpyAsync_htod(self._buf, x.ctypes.data_as(ctypes.c_void_p), self.buf_sz, ctypes.c_void_p(0))
  def _copyout(self, x:np.ndarray): hip.hipMemcpyAsync_dtoh(x.ctypes.data_as(ctypes.c_void_p), self._buf, self.buf_sz, ctypes.c_void_p(0))

class RawHIPBufferCPU(RawBufferCopyInOut):
  def __init__(self, size, dtype):
    self.buf_sz = size * dtype.itemsize
    super().__init__(size, dtype, hip.hipMalloc(self.buf_sz))
  def __del__(self): hip.hipFree(self._buf)
  # hipMemcpyAsync_htod/hipMemcpyAsync_dtoh will crash on HIP CPU, so we use synchronized versions of hipMemcpy instead.
  def _copyin(self, x:np.ndarray): hip.hipMemcpy(self._buf, x.ctypes.data_as(ctypes.c_void_p), self.buf_sz, hip.hipMemcpyHostToDevice)
  def _copyout(self, x:np.ndarray): hip.hipMemcpy(x.ctypes.data_as(ctypes.c_void_p), self._buf, self.buf_sz, hip.hipMemcpyDeviceToHost)

class HIPProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if not binary:
        prog = hip.hiprtcCreateProgram(prg, name, [], [])
        device_properties = hip.hipGetDeviceProperties(0)
        hip.hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}'])
        prg = hip.hiprtcGetCode(prog)
    except Exception as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    module = hip.hipModuleLoadData(prg)
    self.prg = hip.hipModuleGetFunction(module, name)

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    class PackageStruct(ctypes.Structure):
      _fields_ = [(f'field{idx}', ctypes.c_void_p) for idx in range(len(args))]
    struct = PackageStruct(*[data._buf for data in args])
    hip.hipModuleLaunchKernel(self.prg, global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      return hip.hipEventElapsedTime(start, end)*1e-3

class HIPProgramCPU:
  def __init__(self, name:str, prg:str, binary=False):
    prg = '#include <hip/hip_runtime.h>\n' + prg
    # TODO: is there a way to not write this to disk?
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.so"
    if not os.path.exists(fn):
      subprocess.check_output(args=('clang -g -std=c++20 --stdlib=libstdc++ -shared -O2 -Wall -Werror -x c++ -ltbb -ltbbmalloc -fPIC --rtlib=compiler-rt - -o '+fn+'.tmp').split(), input=prg.encode('utf-8'))
      os.rename(fn+'.tmp', fn)
    self.lib = ctypes.CDLL(fn)
    self.prg = self.lib[f"launch_and_wait_{name}"]

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    self.prg.argtypes = [ctypes.c_uint,                   # block x
                         ctypes.c_uint,                   # block y
                         ctypes.c_uint,                   # block z
                         ctypes.c_uint,                   # thread x
                         ctypes.c_uint,                   # thread y
                         ctypes.c_uint,                   # thread z
                         ctypes.c_uint,                   # shared mem
                         ctypes.c_void_p,                 # stream
                        *[ctypes.c_void_p for _ in args]]
    # Launch the kernel and wait the stream.
    self.prg(global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, ctypes.c_void_p(0), *[data._buf for data in args])

# Some hacks for HIP CPU.
#   1) hipLaunchKernelGGL will crash if it invokes kernels from another shared library, so we append a kernel launcher function after the kernel and compile them into one shared library.
#   2) hipLaunchKernelGGL is async so we have to wait on the stream by calling hipStreamSynchronize after hipLaunchKernelGGL.
def build_kernel_launcher(bufnames, buftypes):
  return ['extern "C" void launch_and_wait_KERNEL_NAME_PLACEHOLDER('] + [r"""
  std::uint32_t grid_dim_x,
  std::uint32_t grid_dim_y,
  std::uint32_t grid_dim_z,
  std::uint32_t block_dim_x,
  std::uint32_t block_dim_y,
  std::uint32_t block_dim_z,
  std::uint32_t shared_mem_bytes,
  hipStream_t stream,
"""] + [', '.join([f'{t} {bufnames[i]}' for i,t in buftypes])] + [r"""
) {
  hipLaunchKernelGGL(KERNEL_NAME_PLACEHOLDER, dim3(grid_dim_x, grid_dim_y, grid_dim_z), dim3(block_dim_x, block_dim_y, block_dim_z), shared_mem_bytes, stream,
"""] + [', '.join([f'{bufnames[i]}' for i,t in buftypes])] + [');\nhipStreamSynchronize(stream);\n}']

class HIPCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = r"""
#include <hip/hip_common.h>
#include <hip/hip_math_constants.h>
#define INFINITY (__builtin_inff())
#define NAN HIP_NAN_F
__device__ float4 max(float4 x, float4 y) {
  return float4(fmax(x.x, y.x), fmax(x.y, y.y), fmax(x.z, y.z), fmax(x.w, y.w));
}

__device__ float4 pow(float x, float4 y) {
  return float4(pow(x, y.x), pow(x, y.y), pow(x, y.z), pow(x, y.w));
}

__device__ float4 pow(float4 x, float4 y) {
  return float4(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w));
}
__device__ float4 log2(float4 x) {
  return float4(log2(x.x), log2(x.y), log2(x.z), log2(x.w));
}
extern "C" __global__""",
    smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
    half_prekernel = r"""
#include <hip/hip_fp16.h>
using half4 = HIP_vector_type<half, 4>;
__device__ float vload_half(size_t offset, const half *p) {
  return (float)*(p + offset);
}

__device__ float4 vload_half4(size_t offset, const half * p) {
  return make_float4((float)*(p + offset *4), (float)*(p + offset *4 + 1), (float)*(p + offset *4 + 2), (float)*(p + offset *4 +3));
}
__device__ void vstore_half(float data, size_t offset, half *p) {
  *(p + offset) = (half)data;
}
    """,
    uses_vload=True,
    gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

class HIPCodegenCPU(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = r"""
#include <math.h>
typedef unsigned char uchar;
using half_float::half;
MAKE_VECTOR_TYPE(half, half)

inline float max(float x, float y) {
  return fmax(x, y);
}

inline float4 max(float4 x, float4 y) {
  return float4(fmax(x.x, y.x), fmax(x.y, y.y), fmax(x.z, y.z), fmax(x.w, y.w));
}

inline float4 pow(float x, float4 y) {
  return float4(pow(x, y.x), pow(x, y.y), pow(x, y.z), pow(x, y.w));
}

inline float4 pow(float4 x, float4 y) {
  return float4(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w));
}

inline float4 log2(float4 x) {
  return float4(log2(x.x), log2(x.y), log2(x.z), log2(x.w));
}

inline float vload_half(size_t offset, const half *p) {
  return (float)*(p + offset);
}

inline float4 vload_half4(size_t offset, const half * p) {
  return make_float4((float)*(p + offset *4), (float)*(p + offset *4 + 1), (float)*(p + offset *4 + 2), (float)*(p + offset *4 +3));
}
inline void vstore_half(float data, size_t offset, half *p) {
  *(p + offset) = (half)data;
}
extern "C" __global__
""",
    smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4", uses_vload=True,
    half_prekernel = "",
    need_kernel_launcher = True,
    build_kernel_launcher = build_kernel_launcher,
    gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

HIPBuffer = Compiled(RawHIPBufferCPU if HIP_CPU else RawHIPBuffer, HIPCodegenCPU if HIP_CPU else HIPCodegen, HIPProgramCPU if HIP_CPU else HIPProgram, hip.hipDeviceSynchronize)
