from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, List, Tuple
import ctypes, time, numpy as np
from tinygrad.runtime.autogen import cuda as cuda_types
from tinygrad.helpers import mv_address

for attr in dir(cuda_types):
  if not attr.startswith('__'):
    globals()[attr] = getattr(cuda_types, attr)

try:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  # gpuocelot_lib = ctypes.CDLL("/home/nimlgen/gpuocelot/ocelot/build/libgpuocelot.so")
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501
except Exception: pass

# Global state
class CUDAState:
  def __init__(self):
    self.memory: Dict[int, bytearray] = {}
    self.events: Dict[int, float] = {}  # Event ID -> timestamp
    self.modules: Dict[int, Dict[str, Callable]] = {}  # Module ID -> {func_name: func}
    self.current_context: Optional[int] = None
    self.contexts: Dict[int, Dict] = {}  # Context ID -> context data
    self.devices: Dict[int, Dict] = {}   # Device ID -> device data
    self.next_ptr = 1000  # For memory allocation
    self.next_event_id = 1
    self.next_module_id = 1
    self.next_context_id = 1

cuda_state = CUDAState()

# Helper functions
def check_context():
  if cuda_state.current_context is None:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS

# CUDA API simulation
def cuInit(flags: int) -> int:
  return CUDA_SUCCESS

def cuDeviceGet(device, ordinal: int) -> int:
  if ordinal < 0:
    return CUDA_ERROR_INVALID_VALUE
  device._obj.value = ordinal
  cuda_state.devices[ordinal] = {"compute_capability": (3, 5)}
  return CUDA_SUCCESS

def cuCtxCreate_v2(pctx, flags: int, dev: int) -> int:
  ctx_id = cuda_state.next_context_id
  cuda_state.next_context_id += 1
  cuda_state.contexts[ctx_id] = {"device": dev, "flags": flags}
  pctx._obj.value = ctx_id
  return CUDA_SUCCESS

def cuCtxSetCurrent(context: CUcontext) -> int:
  if context.value not in cuda_state.contexts:
    return CUDA_ERROR_INVALID_VALUE
  cuda_state.current_context = context.value
  return CUDA_SUCCESS

def cuMemAlloc_v2(dptr, bytesize: int) -> int:
  x = memoryview(bytearray(bytesize))
  dptr._obj.value = mv_address(x)
  cuda_state.memory[dptr._obj.value] = x
  return CUDA_SUCCESS

def cuMemFree_v2(dptr: CUdeviceptr_v2) -> int:
  if dptr.value in cuda_state.memory:
    del cuda_state.memory[dptr.value]
    return CUDA_SUCCESS
  return CUDA_ERROR_INVALID_VALUE

def cuMemcpyHtoDAsync_v2(dst: CUdeviceptr_v2, src: ctypes.c_void_p, bytesize: int, stream: Any) -> int:
  if dst.value not in cuda_state.memory:
    return CUDA_ERROR_INVALID_VALUE
  ctypes.memmove(dst.value, src, bytesize)
  return CUDA_SUCCESS

def cuMemcpyDtoH_v2(dst: ctypes.c_void_p, src: CUdeviceptr_v2, bytesize: int) -> int:
  if src.value not in cuda_state.memory:
    return CUDA_ERROR_INVALID_VALUE
  ctypes.memmove(dst, src.value, bytesize)
  return CUDA_SUCCESS

def cuEventCreate(phEvent, flags: int) -> int:
  event_id = cuda_state.next_event_id
  cuda_state.next_event_id += 1
  cuda_state.events[event_id] = 0.0  # Initialize with current time
  phEvent._obj.value = event_id
  return CUDA_SUCCESS

def cuEventRecord(hEvent: CUevent, hStream: Any) -> int:
  if hEvent.value not in cuda_state.events:
    return CUDA_ERROR_INVALID_VALUE
  cuda_state.events[hEvent.value] = time.time()
  return CUDA_SUCCESS

def cuEventSynchronize(hEvent: CUevent) -> int:
  if hEvent.value not in cuda_state.events:
    return CUDA_ERROR_INVALID_VALUE
  time.sleep(0.001)  # Simulate some synchronization delay
  return CUDA_SUCCESS

def cuEventElapsedTime(pMilliseconds, hStart: CUevent, hEnd: CUevent) -> int:
  if hStart.value not in cuda_state.events or hEnd.value not in cuda_state.events:
    return CUDA_ERROR_INVALID_VALUE
  elapsed = (cuda_state.events[hEnd.value] - cuda_state.events[hStart.value]) * 1000
  pMilliseconds._obj.value = elapsed
  return CUDA_SUCCESS

def cuEventDestroy_v2(hEvent: CUevent) -> int:
  if hEvent.value in cuda_state.events:
    del cuda_state.events[hEvent.value]
  return CUDA_SUCCESS

def cuModuleLoadData(module, image: bytes) -> int:
  module_id = cuda_state.next_module_id
  cuda_state.next_module_id += 1
  cuda_state.modules[module_id] = memoryview(bytearray(image))
  module._obj.value = module_id
  return CUDA_SUCCESS

def cuModuleGetFunction(hfunc, hmod: CUmodule, name: bytes) -> int:
  if hmod.value not in cuda_state.modules:
    return CUDA_ERROR_INVALID_VALUE
  hfunc._obj.value = mv_address(cuda_state.modules[hmod.value])
  return CUDA_SUCCESS

def cuModuleUnload(hmod: CUmodule) -> int:
  if hmod.value in cuda_state.modules:
    del cuda_state.modules[hmod.value]
  return CUDA_SUCCESS

def cuLaunchKernel(f: CUfunction, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, sharedMemBytes: int,
                   hStream: Any, kernelParams: Any, extra: Any) -> int:
  cargs = [ctypes.cast(getattr(extra, field[0]), ctypes.c_void_p) for field in extra._fields_]
  gpuocelot_lib.ptx_run(ctypes.cast(f.value, ctypes.c_char_p), len(cargs), (ctypes.c_void_p*len(cargs))(*cargs), lx, ly, lz, gx, gy, gz, 0)
  return CUDA_SUCCESS

def cuDeviceComputeCapability(major, minor, dev: int) -> int:
  if dev not in cuda_state.devices:
    return CUDA_ERROR_INVALID_VALUE
  major._obj.value = cuda_state.devices[dev]["compute_capability"][0]
  minor._obj.value = cuda_state.devices[dev]["compute_capability"][1]
  return CUDA_SUCCESS

def cuDeviceCanAccessPeer(canAccessPeer, dev: int, peerDev: int) -> int:
  canAccessPeer._obj.value = 1  # Always allow peer access in simulation
  return CUDA_SUCCESS

def cuCtxEnablePeerAccess(peerContext: CUcontext, flags: int) -> int:
  return CUDA_SUCCESS

def cuMemHostAlloc(pp, bytesize: int, flags: int) -> int:
  memory = ctypes.create_string_buffer(bytesize)
  pp._obj.value = ctypes.cast(memory, ctypes.c_void_p).value
  return CUDA_SUCCESS

def cuMemFreeHost(p: ctypes.c_void_p) -> int: return CUDA_SUCCESS

def cuMemcpyDtoDAsync_v2(dst: CUdeviceptr_v2, src: CUdeviceptr_v2, bytesize: int, stream: Any) -> int:
  if src.value not in cuda_state.memory or dst.value not in cuda_state.memory:
    return CUDA_ERROR_INVALID_VALUE
  ctypes.memmove(dst.value, src.value, bytesize)
  return CUDA_SUCCESS

def cuFuncSetAttribute(hfunc: CUfunction, attrib: int, value: int) -> int:
  return CUDA_SUCCESS

def cuCtxSynchronize() -> int: return CUDA_SUCCESS

def cuGetErrorString(error: int, pStr) -> int:
  error_strings = {
    CUDA_SUCCESS: b"CUDA_SUCCESS",
    CUDA_ERROR_INVALID_VALUE: b"CUDA_ERROR_INVALID_VALUE",
    CUDA_ERROR_OUT_OF_MEMORY: b"CUDA_ERROR_OUT_OF_MEMORY",
  }
  error_str = error_strings.get(error, b"Unknown CUDA error")
  buf = ctypes.create_string_buffer(error_str)
  # Set the pointer to point to our error string buffer
  pStr._obj.value = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)).value
  return CUDA_SUCCESS
