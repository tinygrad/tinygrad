import ctypes
import subprocess
import tempfile
import hashlib
import os

prg = r"""
#include <hip/hip_runtime.h>
extern "C" {
hipError_t my_hipMalloc(void** ptr, std::size_t size) { return hipMalloc(ptr, size); }
hipError_t my_hipFree(void* ptr) { return hip::detail::deallocate(ptr); }
hipError_t my_hipMemcpy(void* dst, const void* src, std::size_t size, hipMemcpyKind kind = hipMemcpyDefault) { return hipMemcpy(dst, src, size, kind); }
hipError_t my_hipMemcpyAsync(void* dst, const void* src, std::size_t size, hipMemcpyKind kind, hipStream_t stream) { return hipMemcpyAsync(dst, src, size, kind, stream); }
hipError_t my_hipDeviceSynchronize() { return hipDeviceSynchronize(); }
hipError_t my_hipEventCreate(hipEvent_t* event) { return hipEventCreate(event); }
hipError_t my_hipEventDestroy(hipEvent_t event) { return hipEventDestroy(event); }
hipError_t my_hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) { return hipEventElapsedTime(ms, start, stop); }
hipError_t my_hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr) { return hipEventRecord(event, stream); }
hipError_t my_hipEventSynchronize(hipEvent_t event) { return hipEventSynchronize(event); }
hipError_t my_hipStreamSynchronize(hipStream_t 	stream) { return hipStreamSynchronize(stream); }
}
"""
fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.so"
if not os.path.exists(fn):
  subprocess.check_output(args=('clang -g -std=c++20 --stdlib=libstdc++ -shared -O2 -Wall -Werror -x c++ -ltbb -ltbbmalloc -fPIC --rtlib=compiler-rt - -o '+fn+'.tmp').split(), input=prg.encode('utf-8'))
  os.rename(fn+'.tmp', fn)
_libhip = ctypes.CDLL(fn)

hipMemcpyHostToHost = 0
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault = 4

def hipCheckStatus(status):
  if status != 0:
    raise RuntimeError('HIP error %s' % status)

def hipDeviceSynchronize():
  status = _libhip['my_hipDeviceSynchronize']()
  hipCheckStatus(status)

_libhip['my_hipMalloc'].restype = int
_libhip['my_hipMalloc'].argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
def hipMalloc(count):
  ptr = ctypes.c_void_p()
  c_count = ctypes.c_size_t(count)
  status = _libhip['my_hipMalloc'](ctypes.byref(ptr), c_count)
  hipCheckStatus(status)
  return ptr

_libhip['my_hipFree'].restype = int
_libhip['my_hipFree'].argtypes = [ctypes.c_void_p]
def hipFree(ptr):
  status = _libhip['my_hipFree'](ptr)
  hipCheckStatus(status)
  return ptr

_libhip['my_hipMemcpy'].restype = int
_libhip['my_hipMemcpy'].argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def hipMemcpy(dst, src, count, direction):
  status = _libhip['my_hipMemcpy'](dst, src, ctypes.c_size_t(count), direction)
  hipCheckStatus(status)

_libhip['my_hipMemcpyAsync'].restype = int
_libhip['my_hipMemcpyAsync'].argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
def hipMemcpyAsync_htod(dst, src, count, stream):
  status = _libhip['my_hipMemcpyAsync'](dst, src, ctypes.c_size_t(count), hipMemcpyHostToDevice, stream)
  hipCheckStatus(status)

def hipMemcpyAsync_dtoh(dst, src, count, stream):
  status = _libhip['my_hipMemcpyAsync'](dst, src, ctypes.c_size_t(count), hipMemcpyDeviceToHost, stream)
  hipCheckStatus(status)

def hipMemcpyAsync(dst, src, count, direction, stream):
  status = _libhip['my_hipMemcpyAsync'](dst, src, ctypes.c_size_t(count), direction, stream)
  hipCheckStatus(status)

def hipEventCreate():
  ptr = ctypes.c_void_p()
  status = _libhip['my_hipEventCreate'](ctypes.byref(ptr))
  hipCheckStatus(status)
  return ptr

_libhip['my_hipEventRecord'].restype = int
_libhip['my_hipEventRecord'].argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def hipEventRecord(event, stream=None):
  status = _libhip['my_hipEventRecord'](event, stream)
  hipCheckStatus(status)

_libhip['my_hipEventDestroy'].restype = int
_libhip['my_hipEventDestroy'].argtypes = [ctypes.c_void_p]
def hipEventDestroy(event):
  status = _libhip['my_hipEventDestroy'](event)
  hipCheckStatus(status)

_libhip['my_hipEventSynchronize'].restype = int
_libhip['my_hipEventSynchronize'].argtypes = [ctypes.c_void_p]
def hipEventSynchronize(event):
  status = _libhip['my_hipEventSynchronize'](event)
  hipCheckStatus(status)

_libhip['my_hipEventElapsedTime'].restype = int
_libhip['my_hipEventElapsedTime'].argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]

def hipEventElapsedTime(start, stop):
  t = ctypes.c_float()
  status = _libhip['my_hipEventElapsedTime'](ctypes.byref(t), start, stop)
  hipCheckStatus(status)
  return t.value

_libhip['my_hipStreamSynchronize'].restype = int
_libhip['my_hipStreamSynchronize'].argtypes = [ctypes.c_void_p]
def hipStreamSynchronize(stream):
  status = _libhip['my_hipStreamSynchronize'](stream)
  hipCheckStatus(status)

def hipGetDevice(): pass
def hiprtcCreateProgram(source, name, header_names, header_sources): pass
def hipGetDeviceProperties(deviceId: int): pass
def hiprtcCompileProgram(prog, options): pass
def hiprtcGetCode(prog): pass
def hipModuleLoadData(data): pass
def hipModuleGetFunction(module, func_name): pass
def hipModuleLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct): pass