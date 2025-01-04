from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, List, Tuple
import ctypes, time, numpy as np
from tinygrad.runtime.autogen import cuda

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
    return cuda.CUDA_ERROR_INVALID_VALUE
  return cuda.CUDA_SUCCESS

# CUDA API simulation
def cuInit(flags: int) -> int:
  return cuda.CUDA_SUCCESS

def cuDeviceGet(device: ctypes.POINTER(cuda.CUdevice), ordinal: int) -> int:
  if ordinal < 0:
    return cuda.CUDA_ERROR_INVALID_VALUE
  device.contents.value = ordinal
  cuda_state.devices[ordinal] = {"compute_capability": (7, 5)}  # Default to SM 75
  return cuda.CUDA_SUCCESS

def cuCtxCreate_v2(pctx: ctypes.POINTER(cuda.CUcontext), flags: int, dev: int) -> int:
  ctx_id = cuda_state.next_context_id
  cuda_state.next_context_id += 1
  cuda_state.contexts[ctx_id] = {"device": dev, "flags": flags}
  pctx.contents.value = ctx_id
  return cuda.CUDA_SUCCESS

def cuCtxSetCurrent(context: cuda.CUcontext) -> int:
  if context.value not in cuda_state.contexts:
    return cuda.CUDA_ERROR_INVALID_VALUE
  cuda_state.current_context = context.value
  return cuda.CUDA_SUCCESS

def cuMemAlloc_v2(dptr: ctypes.POINTER(cuda.CUdeviceptr_v2), bytesize: int) -> int:
  ptr = cuda_state.next_ptr
  cuda_state.next_ptr += bytesize
  cuda_state.memory[ptr] = bytearray(bytesize)
  dptr.contents.value = ptr
  return cuda.CUDA_SUCCESS

def cuMemFree_v2(dptr: cuda.CUdeviceptr_v2) -> int:
  if dptr.value in cuda_state.memory:
    del cuda_state.memory[dptr.value]
    return cuda.CUDA_SUCCESS
  return cuda.CUDA_ERROR_INVALID_VALUE

def cuMemcpyHtoDAsync_v2(dst: cuda.CUdeviceptr_v2, src: ctypes.c_void_p, bytesize: int, stream: Any) -> int:
  if dst.value not in cuda_state.memory:
    return cuda.CUDA_ERROR_INVALID_VALUE
  src_array = (ctypes.c_char * bytesize).from_address(src)
  cuda_state.memory[dst.value] = bytearray(src_array)
  return cuda.CUDA_SUCCESS

def cuMemcpyDtoH_v2(dst: ctypes.c_void_p, src: cuda.CUdeviceptr_v2, bytesize: int) -> int:
  if src.value not in cuda_state.memory:
    return cuda.CUDA_ERROR_INVALID_VALUE
  dst_array = (ctypes.c_char * bytesize).from_address(dst)
  for i in range(bytesize):
    dst_array[i] = cuda_state.memory[src.value][i]
  return cuda.CUDA_SUCCESS

def cuEventCreate(phEvent: ctypes.POINTER(cuda.CUevent), flags: int) -> int:
  event_id = cuda_state.next_event_id
  cuda_state.next_event_id += 1
  cuda_state.events[event_id] = 0.0  # Initialize with current time
  phEvent.contents.value = event_id
  return cuda.CUDA_SUCCESS

def cuEventRecord(hEvent: cuda.CUevent, hStream: Any) -> int:
  if hEvent.value not in cuda_state.events:
    return cuda.CUDA_ERROR_INVALID_VALUE
  cuda_state.events[hEvent.value] = time.time()
  return cuda.CUDA_SUCCESS

def cuEventSynchronize(hEvent: cuda.CUevent) -> int:
  if hEvent.value not in cuda_state.events:
    return cuda.CUDA_ERROR_INVALID_VALUE
  time.sleep(0.001)  # Simulate some synchronization delay
  return cuda.CUDA_SUCCESS

def cuEventElapsedTime(pMilliseconds: ctypes.POINTER(ctypes.c_float), hStart: cuda.CUevent, hEnd: cuda.CUevent) -> int:
  if hStart.value not in cuda_state.events or hEnd.value not in cuda_state.events:
    return cuda.CUDA_ERROR_INVALID_VALUE
  elapsed = (cuda_state.events[hEnd.value] - cuda_state.events[hStart.value]) * 1000
  pMilliseconds.contents.value = elapsed
  return cuda.CUDA_SUCCESS

def cuEventDestroy_v2(hEvent: cuda.CUevent) -> int:
  if hEvent.value in cuda_state.events:
    del cuda_state.events[hEvent.value]
  return cuda.CUDA_SUCCESS

def cuModuleLoadData(module: ctypes.POINTER(cuda.CUmodule), image: bytes) -> int:
  module_id = cuda_state.next_module_id
  cuda_state.next_module_id += 1
  cuda_state.modules[module_id] = {}  # Initialize empty module
  module.contents.value = module_id
  return cuda.CUDA_SUCCESS

def cuModuleGetFunction(hfunc: ctypes.POINTER(cuda.CUfunction), hmod: cuda.CUmodule, name: bytes) -> int:
  if hmod.value not in cuda_state.modules:
    return cuda.CUDA_ERROR_INVALID_VALUE
  # Simulate function handle creation
  hfunc.contents.value = hash(name)
  return cuda.CUDA_SUCCESS

def cuModuleUnload(hmod: cuda.CUmodule) -> int:
  if hmod.value in cuda_state.modules:
    del cuda_state.modules[hmod.value]
  return cuda.CUDA_SUCCESS

def cuLaunchKernel(f: cuda.CUfunction, 
           gridDimX: int, gridDimY: int, gridDimZ: int,
           blockDimX: int, blockDimY: int, blockDimZ: int,
           sharedMemBytes: int,
           hStream: Any,
           kernelParams: Any,
           extra: Any) -> int:
  # Simulate kernel execution with a small delay
  time.sleep(0.001)
  return cuda.CUDA_SUCCESS

def cuDeviceComputeCapability(major: ctypes.POINTER(ctypes.c_int), 
               minor: ctypes.POINTER(ctypes.c_int),
               dev: int) -> int:
  if dev not in cuda_state.devices:
    return cuda.CUDA_ERROR_INVALID_VALUE
  major.contents.value = cuda_state.devices[dev]["compute_capability"][0]
  minor.contents.value = cuda_state.devices[dev]["compute_capability"][1]
  return cuda.CUDA_SUCCESS

def cuDeviceCanAccessPeer(canAccessPeer: ctypes.POINTER(ctypes.c_int),
             dev: int,
             peerDev: int) -> int:
  canAccessPeer.contents.value = 1  # Always allow peer access in simulation
  return cuda.CUDA_SUCCESS

def cuCtxEnablePeerAccess(peerContext: cuda.CUcontext, flags: int) -> int:
  return cuda.CUDA_SUCCESS

def cuMemHostAlloc(pp: ctypes.POINTER(ctypes.c_void_p), bytesize: int, flags: int) -> int:
  memory = ctypes.create_string_buffer(bytesize)
  pp.contents.value = ctypes.cast(memory, ctypes.c_void_p).value
  return cuda.CUDA_SUCCESS

def cuMemFreeHost(p: ctypes.c_void_p) -> int: return cuda.CUDA_SUCCESS

def cuMemcpyDtoDAsync_v2(dst: cuda.CUdeviceptr_v2, src: cuda.CUdeviceptr_v2, bytesize: int, stream: Any) -> int:
  if src.value not in cuda_state.memory or dst.value not in cuda_state.memory:
    return cuda.CUDA_ERROR_INVALID_VALUE

  cuda_state.memory[dst.value] = cuda_state.memory[src.value][:bytesize]
  return cuda.CUDA_SUCCESS

def cuFuncSetAttribute(hfunc: cuda.CUfunction, attrib: int, value: int) -> int:
  return cuda.CUDA_SUCCESS

def cuGetErrorString(error: int, pStr: ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) -> int:
  error_str = cuda.cudaError_enum__enumvalues.get(error, b"Unknown CUDA error")
  buf = ctypes.create_string_buffer(error_str)
  pStr.contents = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
  return cuda.CUDA_SUCCESS
