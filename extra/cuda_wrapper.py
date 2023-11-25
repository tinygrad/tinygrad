import subprocess
from cuda import nvrtc, cuda, cudart

def cu_get_include_paths(compiler):
  try:
    # Run the compiler command to get the include paths
    result = subprocess.check_output([compiler, "-E", "-x", "c", "-", "-v"], input="", stderr=subprocess.STDOUT, universal_newlines=True)
    lines = result.splitlines()

    includes = []
    for line in lines:
      if line.startswith("#$ INCLUDES="):
        line = line.strip().rstrip()[len("#$ INCLUDES=\""):-1]
        includes = line.split()
    return includes

  except Exception as e:
    print(f"An error occurred, CUDA might be unavailable: {e}")
    return []

cuda_includes = cu_get_include_paths(compiler="nvcc")

import ctypes
from tinygrad.helpers import DEBUG
import sys
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

# try:
_libcuda = ctypes.cdll.LoadLibrary("libcuda.so.1")
# print(_libcuda)
# _libcudart = ctypes.cdll.LoadLibrary("libcudart.so")
_libcudartrtc = ctypes.cdll.LoadLibrary("libnvrtc.so")

_libcuda.cuGetErrorString.restype = ctypes.c_int
_libcuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.c_void_p]
def cuGetErrorString(status):
  ptr = ctypes.c_char_p()
  status = _libcuda.cuGetErrorString(status, ctypes.byref(ptr))
  if status != 0: raise RuntimeError("CUDA error: cuGetErrorString failed")
  return ptr.value.decode("utf-8")

def cuCheckStatus(status):
  if status != 0: raise RuntimeError("CUDA error %s: %s" % (status, cuGetErrorString(status)))

_libcuda.cuInit.restype = int
_libcuda.cuInit.argtypes = [ctypes.c_uint]
def cuInit(flags):
  status = _libcuda.cuInit(flags)
  cuCheckStatus(status)

_libcuda.cuCtxCreate.restype = int
_libcuda.cuCtxCreate.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
def cuCtxCreate(flags, device):
  ptr = ctypes.c_void_p()
  status = _libcuda.cuCtxCreate(ctypes.byref(ptr), flags, device)
  cuCheckStatus(status)
  return ptr

_libcuda.cuCtxSynchronize.restype = int
_libcuda.cuCtxSynchronize.argtypes = []
def cuCtxSynchronize():
  status = _libcuda.cuCtxSynchronize()
  cuCheckStatus(status)

_libcuda.cuStreamSynchronize.restype = int
_libcuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
def cuStreamSynchronize(stream):
  status = _libcuda.cuStreamSynchronize(stream)
  cuCheckStatus(status)

_libcuda.cuEventCreate.restype = int
_libcuda.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
def cuEventCreate():
  ptr = ctypes.c_void_p()
  status = _libcuda.cuEventCreate(ctypes.byref(ptr))
  cuCheckStatus(status)
  return ptr

_libcuda.cuEventRecord.restype = int
_libcuda.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cuEventRecord(event, stream=None):
  status = _libcuda.cuEventRecord(event, stream)
  cuCheckStatus(status)

_libcuda.cuEventDestroy.restype = int
_libcuda.cuEventDestroy.argtypes = [ctypes.c_void_p]
def cuEventDestroy(event):
  status = _libcuda.cuEventDestroy(event)
  cuCheckStatus(status)

_libcuda.cuEventSynchronize.restype = int
_libcuda.cuEventSynchronize.argtypes = [ctypes.c_void_p]
def cuEventSynchronize(event):
  status = _libcuda.cuEventSynchronize(event)
  cuCheckStatus(status)

_libcuda.cuEventElapsedTime.restype = int
_libcuda.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
def cuEventElapsedTime(start, stop):
  t = ctypes.c_float()
  status = _libcuda.cuEventElapsedTime(ctypes.byref(t), start, stop)
  cuCheckStatus(status)
  return t.value

## Stream Management

# Stream capture modes:
hipStreamCaptureModeGlobal = 0
hipStreamCaptureModeThreadLocal = 1
hipStreamCaptureModeRelaxed = 2

_libcuda.cuStreamCreate.restype = int
_libcuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
def cuStreamCreate():
  ptr = ctypes.c_void_p()
  status = _libcuda.cuStreamCreate(ctypes.byref(ptr))
  cuCheckStatus(status)
  return ptr

_libcuda.cuStreamDestroy.restype = int
_libcuda.cuStreamDestroy.argtypes = [ctypes.c_void_p]
def cuStreamDestroy(stream):
  status = _libcuda.cuStreamDestroy(stream)
  cuCheckStatus(status)

_libcuda.cuStreamBeginCapture.restype = int
_libcuda.cuStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
def cuStreamBeginCapture(stream, mode=hipStreamCaptureModeGlobal):
  t = ctypes.c_float()
  status = _libcuda.cuStreamBeginCapture(stream, mode)
  cuCheckStatus(status)

_libcuda.cuStreamEndCapture.restype = int
_libcuda.cuStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cuStreamEndCapture(stream):
  ptr = ctypes.c_void_p()
  status = _libcuda.cuStreamEndCapture(stream, ctypes.byref(ptr))
  cuCheckStatus(status)
  return ptr

_libcuda.cuStreamGetCaptureInfo_v2.restype = int
_libcuda.cuStreamGetCaptureInfo_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cuStreamGetCaptureInfo_v2(stream):
  status_out = ctypes.c_void_p()
  id_out = ctypes.c_ulonglong()
  graph_out = ctypes.c_void_p()
  deps_out = ctypes.POINTER(ctypes.c_void_p)()
  num_deps = ctypes.c_size_t()
  status = _libcuda.cuStreamGetCaptureInfo_v2(stream, ctypes.byref(status_out), ctypes.byref(id_out), ctypes.byref(graph_out), ctypes.byref(deps_out), ctypes.byref(num_deps))
  cuCheckStatus(status)
  deps = [ctypes.cast(deps_out[i], ctypes.c_void_p) for i in range(num_deps.value)]
  return status_out, id_out.value, graph_out, deps

hipStreamAddCaptureDependencies = 0
hipStreamSetCaptureDependencies = 1

_libcuda.cuStreamUpdateCaptureDependencies.restype = int
_libcuda.cuStreamUpdateCaptureDependencies.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
def cuStreamUpdateCaptureDependencies(stream, deps, flags=hipStreamAddCaptureDependencies):
  deps_in = (ctypes.c_void_p * len(deps))()
  deps_in[:] = deps
  num_deps = ctypes.c_size_t()
  num_deps.value = len(deps)
  flags_in = ctypes.c_uint()
  flags_in.value = flags
  status = _libcuda.cuStreamUpdateCaptureDependencies(stream, deps_in, num_deps, flags_in)
  cuCheckStatus(status)


## Graph Management

_libcuda.cuGraphCreate.restype = int
_libcuda.cuGraphCreate.argtypes = [ctypes.c_void_p, ctypes.c_uint]
def cuGraphCreate():
  ptr = ctypes.c_void_p()
  status = _libcuda.cuGraphCreate(ctypes.byref(ptr), 0)
  cuCheckStatus(status)
  return ptr

_libcuda.cuGraphInstantiate.restype = int
_libcuda.cuGraphInstantiate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cuGraphInstantiate(graph):
  ptr = ctypes.c_void_p()
  status = _libcuda.cuGraphInstantiate(ctypes.byref(ptr), graph, 0, 0, 0)
  cuCheckStatus(status)
  return ptr

_libcuda.cuGraphDestroy.restype = int
_libcuda.cuGraphDestroy.argtypes = [ctypes.c_void_p]
def cuGraphDestroy(graph):
  status = _libcuda.cuGraphDestroy(graph)
  cuCheckStatus(status)

_libcuda.cuGraphExecDestroy.restype = int
_libcuda.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
def cuGraphExecDestroy(gexec):
  status = _libcuda.cuGraphExecDestroy(gexec)
  cuCheckStatus(status)

_libcuda.cuGraphLaunch.restype = int
_libcuda.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cuGraphLaunch(graph_exec, stream=0):
  status = _libcuda.cuGraphLaunch(graph_exec, stream)
  cuCheckStatus(status)

class hipKernelNodeParams(ctypes.Structure):
  _fields_ = [("blockDimX", ctypes.c_uint32), ("blockDimY", ctypes.c_uint32), ("blockDimZ", ctypes.c_uint32),
              ("extra", ctypes.POINTER(ctypes.c_void_p)),
              ("func", ctypes.c_void_p),
              ("gridDimX", ctypes.c_uint32), ("gridDimY", ctypes.c_uint32), ("gridDimZ", ctypes.c_uint32),
              ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
              ("sharedMemBytes", ctypes.c_uint32)]

@dataclass
class kernelNodeParamsWrapper():
  c_struct: Any
  context: Any = None

def getCStructForType(argtypes):
  fields = []
  for j,typ in enumerate(argtypes):
    fields.append((f'field{j}', typ))

  class CStructure(ctypes.Structure):
    _pack_ = 1
    _fields_ = fields
  return CStructure

def setKernelNodeLaunchDims(npwrapper:kernelNodeParamsWrapper, grid, block):
  npwrapper.c_struct.blockDimX = block[0]
  npwrapper.c_struct.blockDimY = block[1]
  npwrapper.c_struct.blockDimZ = block[2]
  npwrapper.c_struct.gridDimX = grid[0]
  npwrapper.c_struct.gridDimY = grid[1]
  npwrapper.c_struct.gridDimZ = grid[2]

def setKernelNodeParams(npwrapper:kernelNodeParamsWrapper, args, ids):
  for j,i in enumerate(ids):
    setattr(npwrapper.context[1], f'field{i}', args[j])

def buildKernelNodeParams(args, argtypes, func, grid, block, sharedMemBytes=0):
  c_struct_t = getCStructForType(argtypes)
  struct = c_struct_t(*args)
  size = ctypes.c_size_t(ctypes.sizeof(struct))
  p_size = ctypes.c_void_p(ctypes.addressof(size))
  p_struct = ctypes.c_void_p(ctypes.addressof(struct))
  config = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), p_struct,
                                ctypes.c_void_p(2), p_size, ctypes.c_void_p(3))
  params = hipKernelNodeParams(block[0], block[1], block[2], config, func, grid[0], grid[1], grid[2], None, sharedMemBytes)
  return kernelNodeParamsWrapper(c_struct=params, context=(size, struct, config))

_libcuda.cuGraphAddKernelNode.restype = int
_libcuda.cuGraphAddKernelNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
def cuGraphAddKernelNode(graph, deps, params:kernelNodeParamsWrapper):
  graph_node = ctypes.c_void_p()
  deps_in = (ctypes.c_void_p * len(deps))()
  deps_in[:] = deps
  num_deps = ctypes.c_size_t(len(deps))
  status = _libcuda.cuGraphAddKernelNode(ctypes.byref(graph_node), graph, deps_in, num_deps, ctypes.byref(params.c_struct))
  cuCheckStatus(status)
  return graph_node

_libcuda.cuGraphExecKernelNodeSetParams.restype = int
_libcuda.cuGraphExecKernelNodeSetParams.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cuGraphExecKernelNodeSetParams(gexec, node, params:kernelNodeParamsWrapper):
  status = _libcuda.cuGraphExecKernelNodeSetParams(gexec, node, ctypes.byref(params.c_struct))
  cuCheckStatus(status)

_libcuda.cuMemAlloc.restype = int
_libcuda.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
def cuMemAlloc(count):
  ptr = ctypes.c_void_p()
  status = _libcuda.cuMemAlloc(ctypes.byref(ptr), count)
  cuCheckStatus(status)
  return ptr.value

_libcuda.cuMemFree.restype = int
_libcuda.cuMemFree.argtypes = [ctypes.c_void_p]
def cuMemFree(ptr):
  status = _libcuda.cuMemFree(ptr)
  cuCheckStatus(status)

# memory copy modes
hipMemcpyHostToHost = 0
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault = 4

_libcuda.cuMemcpy.restype = int
_libcuda.cuMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def cuMemcpy(dst, src, count, direction):
  status = _libcuda.cuMemcpy(dst, src, ctypes.c_size_t(count), direction)
  cuCheckStatus(status)

_libcuda.cuMemcpyAsync.restype = int
_libcuda.cuMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
def cuMemcpyAsync(dst, src, count, direction, stream):
  status = _libcuda.cuMemcpyAsync(dst, src, ctypes.c_size_t(count), direction, stream)
  cuCheckStatus(status)

_libcuda.cuMemcpyAsync.restype = int
_libcuda.cuMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
def cuMemcpyHtoDAsync(dst, src, count, stream):
  status = _libcuda.cuMemcpyHtoDAsync(dst, src, ctypes.c_size_t(count), stream)
  cuCheckStatus(status)

_libcuda.cuMemcpyAsync.restype = int
_libcuda.cuMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
def cuMemcpyDtoH(dst, src, count):
  status = _libcuda.cuMemcpyDtoH(dst, src, ctypes.c_size_t(count))
  cuCheckStatus(status)

_libcuda.cuMemGetInfo.restype = int
_libcuda.cuMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cuMemGetInfo():
  free = ctypes.c_size_t()
  total = ctypes.c_size_t()
  status = _libcuda.cuMemGetInfo(ctypes.byref(free), ctypes.byref(total))
  cuCheckStatus(status)
  return free.value, total.value

class hipIpcMemHandle_t(ctypes.Structure):
  _fields_ = [("reserved", ctypes.c_char * 64)]

_libcuda.cuIpcGetMemHandle.restype = int
_libcuda.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p]
def cuIpcGetMemHandle(ptr):
  handle = hipIpcMemHandle_t()
  status = _libcuda.cuIpcGetMemHandle(ctypes.byref(handle), ptr)
  cuCheckStatus(status)
  return handle

_libcuda.cuIpcOpenMemHandle.restype = int
_libcuda.cuIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint]
def cuIpcOpenMemHandle(handle, flags):
  ptr = ctypes.c_void_p()
  status = _libcuda.cuIpcOpenMemHandle(ctypes.byref(ptr), handle, flags)
  cuCheckStatus(status)
  return ptr.value

_libcuda.cuIpcCloseMemHandle.restype = int
_libcuda.cuIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
def cuIpcCloseMemHandle(ptr):
  status = _libcuda.cuIpcCloseMemHandle(ptr)
  cuCheckStatus(status)

_libcuda.cuDeviceGet.restype = int
_libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
def cuDeviceGet(id):
  dev = ctypes.c_int()
  status = _libcuda.cuDeviceGet(ctypes.byref(dev), id)
  cuCheckStatus(status)
  return dev.value

_libcuda.cuDeviceGetCount.restype = int
_libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
def cuDeviceGetCount():
  count = ctypes.c_int()
  status = _libcuda.cuDeviceGetCount(ctypes.byref(count))
  cuCheckStatus(status)
  return count.value

class hipDeviceArch(ctypes.Structure):
  _fields_ = [
    # *32-bit Atomics*
    # 32-bit integer atomics for global memory.
    ("hasGlobalInt32Atomics", ctypes.c_uint, 1),

    # 32-bit float atomic exch for global memory.
    ("hasGlobalFloatAtomicExch", ctypes.c_uint, 1),

    # 32-bit integer atomics for shared memory.
    ("hasSharedInt32Atomics", ctypes.c_uint, 1),

    # 32-bit float atomic exch for shared memory.
    ("hasSharedFloatAtomicExch", ctypes.c_uint, 1),

    # 32-bit float atomic add in global and shared memory.
    ("hasFloatAtomicAdd", ctypes.c_uint, 1),

    # *64-bit Atomics*
    # 64-bit integer atomics for global memory.
    ("hasGlobalInt64Atomics", ctypes.c_uint, 1),

    # 64-bit integer atomics for shared memory.
    ("hasSharedInt64Atomics", ctypes.c_uint, 1),

    # *Doubles*
    # Double-precision floating point.
    ("hasDoubles", ctypes.c_uint, 1),

    # *Warp cross-lane operations*
    # Warp vote instructions (__any, __all).
    ("hasWarpVote", ctypes.c_uint, 1),

    # Warp ballot instructions (__ballot).
    ("hasWarpBallot", ctypes.c_uint, 1),

    # Warp shuffle operations. (__shfl_*).
    ("hasWarpShuffle", ctypes.c_uint, 1),

    # Funnel two words into one with shift&mask caps.
    ("hasFunnelShift", ctypes.c_uint, 1),

    # *Sync*
    # __threadfence_system.
    ("hasThreadFenceSystem", ctypes.c_uint, 1),

    # __syncthreads_count, syncthreads_and, syncthreads_or.
    ("hasSyncThreadsExt", ctypes.c_uint, 1),

    # *Misc*
    # Surface functions.
    ("hasSurfaceFuncs", ctypes.c_uint, 1),

    # Grid and group dims are 3D (rather than 2D).
    ("has3dGrid", ctypes.c_uint, 1),

    # Dynamic parallelism.
    ("hasDynamicParallelism", ctypes.c_uint, 1),
  ]

class hipDeviceProperties(ctypes.Structure):
  _fields_ = [
    # Device name
    ("_name", ctypes.c_char * 256),

    # Size of global memory region (in bytes)
    ("totalGlobalMem", ctypes.c_size_t),

    # Size of shared memory region (in bytes).
    ("sharedMemPerBlock", ctypes.c_size_t),

    # Registers per block.
    ("regsPerBlock", ctypes.c_int),

    # Warp size.
    ("warpSize", ctypes.c_int),

    # Max work items per work group or workgroup max size.
    ("maxThreadsPerBlock", ctypes.c_int),

    # Max number of threads in each dimension (XYZ) of a block.
    ("maxThreadsDim", ctypes.c_int * 3),

    # Max grid dimensions (XYZ).
    ("maxGridSize", ctypes.c_int * 3),

    # Max clock frequency of the multiProcessors in khz.
    ("clockRate", ctypes.c_int),

    # Max global memory clock frequency in khz.
    ("memoryClockRate", ctypes.c_int),

    # Global memory bus width in bits.
    ("memoryBusWidth", ctypes.c_int),

    # Size of shared memory region (in bytes).
    ("totalConstMem", ctypes.c_size_t),

    # Major compute capability.  On HCC, this is an approximation and features may
    # differ from CUDA CC.  See the arch feature flags for portable ways to query
    # feature caps.
    ("major", ctypes.c_int),

    # Minor compute capability.  On HCC, this is an approximation and features may
    # differ from CUDA CC.  See the arch feature flags for portable ways to query
    # feature caps.
    ("minor", ctypes.c_int),

    # Number of multi-processors (compute units).
    ("multiProcessorCount", ctypes.c_int),

    # L2 cache size.
    ("l2CacheSize", ctypes.c_int),

    # Maximum resident threads per multi-processor.
    ("maxThreadsPerMultiProcessor", ctypes.c_int),

    # Compute mode.
    ("computeMode", ctypes.c_int),

    # Frequency in khz of the timer used by the device-side "clock*"
    # instructions.  New for HIP.
    ("clockInstructionRate", ctypes.c_int),

    # Architectural feature flags.  New for HIP.
    ("arch", hipDeviceArch),

    # Device can possibly execute multiple kernels concurrently.
    ("concurrentKernels", ctypes.c_int),

    # PCI Domain ID
    ("pciDomainID", ctypes.c_int),

    # PCI Bus ID.
    ("pciBusID", ctypes.c_int),

    # PCI Device ID.
    ("pciDeviceID", ctypes.c_int),

    # Maximum Shared Memory Per Multiprocessor.
    ("maxSharedMemoryPerMultiProcessor", ctypes.c_size_t),

    # 1 if device is on a multi-GPU board, 0 if not.
    ("isMultiGpuBoard", ctypes.c_int),

    # Check whether HIP can map host memory
    ("canMapHostMemory", ctypes.c_int),

    # DEPRECATED: use gcnArchName instead
    ("gcnArch", ctypes.c_int),

    # AMD GCN Arch Name.
    ("_gcnArchName", ctypes.c_char * 256),

    # APU vs dGPU
    ("integrated", ctypes.c_int),

    # HIP device supports cooperative launch
    ("cooperativeLaunch", ctypes.c_int),

    # HIP device supports cooperative launch on multiple devices
    ("cooperativeMultiDeviceLaunch", ctypes.c_int),

    # Maximum size for 1D textures bound to linear memory
    ("maxTexture1DLinear", ctypes.c_int),

    # Maximum number of elements in 1D images
    ("maxTexture1D", ctypes.c_int),

    # Maximum dimensions (width, height) of 2D images, in image elements
    ("maxTexture2D", ctypes.c_int * 2),

    # Maximum dimensions (width, height, depth) of 3D images, in image elements
    ("maxTexture3D", ctypes.c_int * 3),

    # Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    ("hdpMemFlushCntl", ctypes.POINTER(ctypes.c_uint)),

    # Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    ("hdpRegFlushCntl", ctypes.POINTER(ctypes.c_uint)),

    # Maximum pitch in bytes allowed by memory copies
    ("memPitch", ctypes.c_size_t),

    # Alignment requirement for textures
    ("textureAlignment", ctypes.c_size_t),

    # Pitch alignment requirement for texture references bound to pitched memory
    ("texturePitchAlignment", ctypes.c_size_t),

    # Run time limit for kernels executed on the device
    ("kernelExecTimeoutEnabled", ctypes.c_int),

    # Device has ECC support enabled
    ("ECCEnabled", ctypes.c_int),

    # 1:If device is Tesla device using TCC driver, else 0
    ("tccDriver", ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched functions
    ("cooperativeMultiDeviceUnmatchedFunc", ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched grid dimensions
    ("cooperativeMultiDeviceUnmatchedGridDim", ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched block dimensions
    ("cooperativeMultiDeviceUnmatchedBlockDim", ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched shared memories
    ("cooperativeMultiDeviceUnmatchedSharedMem", ctypes.c_int),

    # 1: if it is a large PCI bar device, else 0
    ("isLargeBar", ctypes.c_int),

    # Revision of the GPU in this device
    ("asicRevision", ctypes.c_int),

    # Device supports allocating managed memory on this system
    ("managedMemory", ctypes.c_int),

    # Host can directly access managed memory on the device without migration
    ("directManagedMemAccessFromHost", ctypes.c_int),

    # Device can coherently access managed memory concurrently with the CPU
    ("concurrentManagedAccess", ctypes.c_int),

    # Device supports coherently accessing pageable memory
    # without calling hipHostRegister on it
    ("pageableMemoryAccess", ctypes.c_int),

    # Device accesses pageable memory via the host"s page tables
    ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int),
  ]

  @property
  def name(self):
    return self._name.decode("utf-8")

  @property
  def gcnArchName(self):
    return self._gcnArchName.decode("utf-8")

# TODO: Fix this
_libcuda.cuDeviceGetProperties.restype = int
_libcuda.cuDeviceGetProperties.argtypes = [ctypes.POINTER(hipDeviceProperties), ctypes.c_int]
def cuDeviceGetProperties(deviceId: int):
  device_properties = hipDeviceProperties()
  status = _libcuda.cuDeviceGetProperties(ctypes.pointer(device_properties), deviceId)
  cuCheckStatus(status)
  return device_properties

_libcuda.cuModuleLoadData.restype = int
_libcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
def cuModuleLoadData(data):
  module = ctypes.c_void_p()
  data_ptr = np.char.array(data).ctypes.data
  status = _libcuda.cuModuleLoadData(ctypes.byref(module), data_ptr)
  cuCheckStatus(status)
  return module

_libcuda.cuModuleGetFunction.restype = int
_libcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
def cuModuleGetFunction(module, func_name):
  kernel = ctypes.c_void_p()
  status = _libcuda.cuModuleGetFunction(ctypes.byref(kernel), module, func_name.encode("utf-8"))
  cuCheckStatus(status)
  return kernel

_libcuda.cuModuleUnload.restype = int
_libcuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
def cuModuleUnload(module):
  status = _libcuda.cuModuleUnload(module)
  cuCheckStatus(status)

_libcuda.cuLaunchKernel.restype = int
_libcuda.cuLaunchKernel.argtypes = [ctypes.c_void_p,
                                          ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # block dim
                                          ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # thread dim
                                          ctypes.c_uint, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
def cuLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct):
  c_bx, c_by, c_bz = ctypes.c_uint(bx), ctypes.c_uint(by), ctypes.c_uint(bz)
  c_tx, c_ty, c_tz = ctypes.c_uint(tx), ctypes.c_uint(ty), ctypes.c_uint(tz)
  c_shared = ctypes.c_uint(shared)

  param_buffer_ptr, param_buffer_size, param_buffer_end = ctypes.c_void_p(1), ctypes.c_void_p(2), ctypes.c_void_p(0)
  size = ctypes.c_size_t(ctypes.sizeof(struct))
  p_size, p_struct = ctypes.c_void_p(ctypes.addressof(size)), ctypes.c_void_p(ctypes.addressof(struct))
  config = (ctypes.c_void_p * 5)(param_buffer_ptr, p_struct, param_buffer_size, p_size, param_buffer_end)

  status = _libcuda.cuLaunchKernel(kernel, c_bx, c_by, c_bz, c_tx, c_ty, c_tz, c_shared, stream, None, config)
  cuCheckStatus(status)

_libcudartrtc.nvrtcCreateProgram.restype = int
_libcudartrtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char),
                                            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
def curtcCreateProgram(source, name, header_names, header_sources):
  c_header_names, c_header_sources = (ctypes.c_char_p * len(header_names))(), (ctypes.c_char_p * len(header_sources))()
  c_header_names[:], c_header_sources[:] = [h.encode("utf-8") for h in header_names], [h.encode("utf-8") for h in header_sources]

  prog = ctypes.c_void_p()
  status = _libcudartrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode("utf-8"), name.encode("utf-8"), len(header_names), c_header_sources, c_header_names)
  cuCheckStatus(status)
  return prog

_libcudartrtc.nvrtcDestroyProgram.restype = int
_libcudartrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
def curtcDestroyProgram(prog):
  status = _libcudartrtc.nvrtcDestroyProgram(ctypes.byref(prog))
  cuCheckStatus(status)


_libcudartrtc.nvrtcGetProgramLogSize.restype = int
_libcudartrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
_libcudartrtc.nvrtcGetProgramLog.restype = int
_libcudartrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
def curtcGetProgramLog(prog):
  logsz = ctypes.c_size_t()
  status = _libcudartrtc.nvrtcGetProgramLogSize(prog, logsz)
  cuCheckStatus(status)
  logstr = ctypes.create_string_buffer(logsz.value)
  status = _libcudartrtc.nvrtcGetProgramLog(prog, logstr)
  cuCheckStatus(status)
  return logstr.value.decode()

_libcudartrtc.nvrtcCompileProgram.restype = int
_libcudartrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
def curtcCompileProgram(prog, options):
  c_options = (ctypes.c_char_p * len(options))()
  c_options[:] = [o.encode("utf-8") for o in options]

  status = _libcudartrtc.nvrtcCompileProgram(prog, len(options), c_options)
  if status == 6: print(hiprtcGetProgramLog(prog))
  cuCheckStatus(status)

_libcudartrtc.nvrtcGetPTXSize.restype = int
_libcudartrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
_libcudartrtc.nvrtcGetPTXSize.restype = int
_libcudartrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
def curtcGetCode(prog):
  code_size = ctypes.c_size_t()
  status = _libcudartrtc.nvrtcGetPTXSize(prog, ctypes.byref(code_size))
  cuCheckStatus(status)
  e_code = ("0" * code_size.value).encode("utf-8")
  status = _libcudartrtc.nvrtcGetPTXSize(prog, e_code)
  cuCheckStatus(status)
  return e_code
# except:
#   if DEBUG >= 1: print("WARNING: libcuda.so or libnvrtc.so not found. CUDA support will not work.")
