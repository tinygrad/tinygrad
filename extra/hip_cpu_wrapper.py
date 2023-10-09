import ctypes
import fcntl
import hashlib
import os
import subprocess
import tempfile
from tinygrad.helpers import DType
from tinygrad.renderer.cstyle import CStyleLanguage
from typing import Tuple, List

COMMON_FLAGS = '-Wall -Werror -std=c++20 -fPIC --stdlib=libstdc++ -O2'

KERNEL_HEADERS = """
#include <hip/hip_vector_types.h>
#include <hip/hip_defines.h>
#include <hip/detail/types.hpp>
#include <math.h>
"""

KERNEL_PCH = '/tmp/hip_cpu.pch'

CXX = os.getenv('CXX', default='clang')

def _process(filename: str, cmd: str, **rest):
  if os.path.exists(filename):
    return
  with open(f'{filename}.lock', 'w') as file:
    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
    try:
      subprocess.check_output(args=f"{cmd} -o {filename}".split(), **rest)
    finally:
      fcntl.flock(file.fileno(), fcntl.LOCK_UN)

def _init_lib():
  _process(filename=KERNEL_PCH, cmd=f"{CXX} {COMMON_FLAGS} -I/usr/local/src/include -x c++-header -", input=KERNEL_HEADERS.encode('utf-8'))

  prog = """
#include <hip/hip_api.h>
#include <hip/hip_vector_types.h>
#include <hip/hip_device_launch_parameters.h>

hip::detail::Dim3 getBlockIdxCall() noexcept { return hip::detail::BIdx::call(); }
hip::detail::Dim3 getThreadIdxCall() noexcept { return hip::detail::TIdx::call(); }

extern "C" {
void my_syncthreads() { __syncthreads(); }
hipError_t my_hipMalloc(void** ptr, std::size_t size) { return hipMalloc(ptr, size); }
hipError_t my_hipFree(void* ptr) { return hipFree(ptr); }
hipError_t my_hipMemcpy(void* dst, const void* src, std::size_t size, hipMemcpyKind kind = hipMemcpyDefault) { return hipMemcpy(dst, src, size, kind); }
hipError_t my_hipMemcpyAsync(void* dst, const void* src, std::size_t size, hipMemcpyKind kind, hipStream_t stream) { return hipMemcpyAsync(dst, src, size, kind, stream); }
hipError_t my_hipDeviceSynchronize() { return hipDeviceSynchronize(); }
hipError_t my_hipEventCreate(hipEvent_t* event) { return hipEventCreate(event); }
hipError_t my_hipEventDestroy(hipEvent_t event) { return hipEventDestroy(event); }
hipError_t my_hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) { return hipEventElapsedTime(ms, start, stop); }
hipError_t my_hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr) { return hipEventRecord(event, stream); }
hipError_t my_hipEventSynchronize(hipEvent_t event) { return hipEventSynchronize(event); }
hipError_t my_hipStreamSynchronize(hipStream_t stream) { return hipStreamSynchronize(stream); }
hipError_t my_hipStreamCreate(hipStream_t* stream_p) { return hipStreamCreate(stream_p); }
hipError_t my_hipStreamDestroy(hipStream_t stream) { return hipStreamDestroy(stream); }

void launch_kernel_trampoline(
  std::uint32_t grid_dim_x,
  std::uint32_t grid_dim_y,
  std::uint32_t grid_dim_z,
  std::uint32_t block_dim_x,
  std::uint32_t block_dim_y,
  std::uint32_t block_dim_z,
  std::uint32_t shared_mem_bytes,
  hipStream_t stream,
  void (*kernel_f)(void*),
  void* args) {
  hipLaunchKernelGGL(kernel_f, dim3(grid_dim_x, grid_dim_y, grid_dim_z), dim3(block_dim_x, block_dim_y, block_dim_z), shared_mem_bytes, stream, args);
  hipStreamSynchronize(stream);
}
} // extern "C"
"""
  filename = f"{tempfile.gettempdir()}/libclang_{hashlib.md5(prog.encode('utf-8')).hexdigest()}.so"
  _process(filename=filename, cmd=f'{CXX} {COMMON_FLAGS} --rtlib=compiler-rt -include-pch {KERNEL_PCH} -shared -x c++ -ltbb -ltbbmalloc -', input=prog.encode('utf-8'))
  return ctypes.CDLL(filename)

_libhip = _init_lib()

hipMemcpyHostToHost = 0
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault = 4

def hipCheckStatus(status):
  if status != 0:
    raise RuntimeError('HIP error %s' % status)

def hipDeviceSynchronize():
  status = _libhip.my_hipDeviceSynchronize()
  hipCheckStatus(status)

_libhip.my_hipMalloc.restype = int
_libhip.my_hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
def hipMalloc(count):
  ptr = ctypes.c_void_p()
  status = _libhip.my_hipMalloc(ctypes.byref(ptr), count)
  hipCheckStatus(status)
  return ptr.value

_libhip.my_hipFree.restype = int
_libhip.my_hipFree.argtypes = [ctypes.c_void_p]
def hipFree(ptr):
  ptr = ctypes.cast(ptr, ctypes.c_void_p)
  status = _libhip.my_hipFree(ptr)
  hipCheckStatus(status)

_libhip.my_hipMemcpy.restype = int
_libhip.my_hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def hipMemcpy(dst, src, count, direction):
  dst = ctypes.cast(dst, ctypes.c_void_p)
  src = ctypes.cast(src, ctypes.c_void_p)
  status = _libhip.my_hipMemcpy(dst, src, ctypes.c_size_t(count), direction)
  hipCheckStatus(status)

_libhip.my_hipMemcpyAsync.restype = int
_libhip.my_hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
def hipMemcpyAsync(dst, src, count, direction, stream):
  dst = ctypes.cast(dst, ctypes.c_void_p)
  src = ctypes.cast(src, ctypes.c_void_p)
  status = _libhip.my_hipMemcpyAsync(dst, src, ctypes.c_size_t(count), direction, stream)
  hipCheckStatus(status)

def hipEventCreate():
  ptr = ctypes.c_void_p()
  status = _libhip.my_hipEventCreate(ctypes.byref(ptr))
  hipCheckStatus(status)
  return ptr

_libhip.my_hipEventRecord.restype = int
_libhip.my_hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def hipEventRecord(event, stream=None):
  event = ctypes.cast(event, ctypes.c_void_p)
  stream = ctypes.cast(stream, ctypes.c_void_p)
  status = _libhip.my_hipEventRecord(event, stream)
  hipCheckStatus(status)

_libhip.my_hipEventSynchronize.restype = int
_libhip.my_hipEventSynchronize.argtypes = [ctypes.c_void_p]
def hipEventSynchronize(event):
  event = ctypes.cast(event, ctypes.c_void_p)
  status = _libhip.my_hipEventSynchronize(event)
  hipCheckStatus(status)

_libhip.my_hipEventElapsedTime.restype = int
_libhip.my_hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
def hipEventElapsedTime(start, stop):
  t = ctypes.c_float()
  start = ctypes.cast(start, ctypes.c_void_p)
  stop = ctypes.cast(stop, ctypes.c_void_p)
  status = _libhip.my_hipEventElapsedTime(ctypes.byref(t), start, stop)
  hipCheckStatus(status)
  return t.value

_libhip.my_hipStreamSynchronize.restype = int
_libhip.my_hipStreamSynchronize.argtypes = [ctypes.c_void_p]
def hipStreamSynchronize(stream):
  status = _libhip.my_hipStreamSynchronize(stream)
  hipCheckStatus(status)

_libhip.my_hipStreamCreate.restype = int
_libhip.my_hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
def hipStreamCreate():
  ptr = ctypes.c_void_p()
  status = _libhip.my_hipStreamCreate(ctypes.byref(ptr))
  hipCheckStatus(status)
  return ptr

_libhip.my_hipStreamDestroy.restype = int
_libhip.my_hipStreamDestroy.argtypes = [ctypes.c_void_p]
def hipStreamDestroy(stream):
  status = _libhip.my_hipStreamDestroy(stream)
  hipCheckStatus(status)

def hipGetDeviceCount():
  return 1

def hiprtcCreateProgram(source, name, header_names, header_sources):
  filename = f"{tempfile.gettempdir()}/clang_{hashlib.md5(source.encode('utf-8')).hexdigest()}.so"
  libname = os.path.splitext(os.path.basename(_libhip._name))[0][3:]
  _process(
    filename=filename,
    cmd=f'{CXX} {COMMON_FLAGS} --rtlib=compiler-rt -include-pch {KERNEL_PCH} -shared -Wl,-rpath=/tmp -L/tmp -I/usr/local/src/include -x c++ -l{libname} -',
    input=source.encode('utf-8'))
  return ctypes.CDLL(filename)

def hipGetDeviceProperties(deviceId: int):
  class Properties:
    def __init__(self):
      self.gcnArchName = "cpu"
      self.totalGlobalMem = 32 * 1024 * 1024
  return Properties()

def hiprtcCompileProgram(prog, options): pass

def hiprtcGetCode(prog):
  return prog

def hipModuleLoadData(data):
  return data

def hipModuleGetFunction(module, func_name):
  return module[f'kernel_{func_name}']

def hipModuleLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct):
  _libhip.launch_kernel_trampoline(bx, by, bz, tx, ty, tz, shared, ctypes.cast(stream, ctypes.c_void_p), ctypes.cast(kernel, ctypes.c_void_p), ctypes.byref(struct))

def hipSetDevice(dev): pass
def hipStreamBeginCapture(stream, mode=0): pass
def hipStreamEndCapture(stream): pass
def hipStreamGetCaptureInfo_v2(stream): pass
def hipStreamUpdateCaptureDependencies(stream, deps, flags=0): pass
def hipGraphInstantiate(graph): pass
def hipGraphDestroy(graph): pass
def hipGraphExecDestroy(gexec): pass
def hipGraphLaunch(graph_exec, stream=0): pass
def hipGraphAddKernelNode(graph, deps, params): pass
def hipGraphExecKernelNodeSetParams(gexec, node, params): pass
hipStreamSetCaptureDependencies = 1
def updateKernelNodeParams(npwrapper, *args, grid=(1,1,1), block=(1,1,1), updated_args=None): pass
def buildKernelNodeParams(*args, func=None, grid=(1,1,1), block=(1,1,1), sharedMemBytes=0, argsStructType=None): pass
def hipModuleUnload(module): pass

class HIPLanguage(CStyleLanguage):
  def render_kernel(self, function_name:str, kernel: List[str], bufs: List[Tuple[str, DType]], local_size: List[int], prekernel: List[str]) -> str:
    rendered_kernel = super().render_kernel(function_name, kernel, bufs, local_size, prekernel)
    return f"""
{KERNEL_HEADERS}

using hipStream_t = void*;

template<hip::detail::Dim3 (*fn)() noexcept>
struct MyCoord {{
  struct X {{ operator std::uint32_t () const {{ return fn().x; }} }};
  struct Y {{ operator std::uint32_t () const {{ return fn().y; }} }};
  struct Z {{ operator std::uint32_t () const {{ return fn().z; }} }};
  static constexpr X x{{}};
  static constexpr Y y{{}};
  static constexpr Z z{{}};
}};

extern "C" void my_syncthreads();
#define __syncthreads my_syncthreads

extern hip::detail::Dim3 getBlockIdxCall() noexcept;
extern hip::detail::Dim3 getThreadIdxCall() noexcept;

[[maybe_unused]] MyCoord<&getBlockIdxCall> blockIdx;
[[maybe_unused]] MyCoord<&getThreadIdxCall> threadIdx;

#define HIP_vector_type hip::detail::Vector_type

inline float max(float x, float y) {{ return fmax(x, y); }}

{rendered_kernel.replace('(__shared__', '(')}

extern "C"{{
struct kernel_args {{
  {"; ".join([f"{buf[1].name}* {buf[0]}" for buf in bufs])};
}};

void kernel_{function_name}(void* params) {{
  auto* args = (kernel_args*)params;
  {function_name}({", ".join([f"args->{buf[0]}" for buf in bufs])});
}}
}} // extern "C"
"""

def language(kernel_prefix: str, half_prekernel: str, **kwargs):
  kernel_prefix = "\n".join(kernel_prefix.splitlines()[3:])
  half_prekernel_lines = half_prekernel.splitlines()
  half_prekernel_lines.insert(1, "using half_float::half;")
  half_prekernel = "\n".join(half_prekernel_lines)
  return HIPLanguage(
    kernel_prefix=kernel_prefix,
    half_prekernel=half_prekernel,
    **kwargs)
