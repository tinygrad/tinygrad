import ctypes
from tinygrad.helpers import DEBUG
import sys
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

try:
  _libhip = ctypes.cdll.LoadLibrary("libamdhip64.so")
  _libhiprtc = ctypes.cdll.LoadLibrary("libhiprtc.so")

  _libhip.hipGetErrorString.restype = ctypes.c_char_p
  _libhip.hipGetErrorString.argtypes = [ctypes.c_int]
  def hipGetErrorString(status):
    return _libhip.hipGetErrorString(status).decode("utf-8")

  def hipCheckStatus(status):
    if status != 0: raise RuntimeError("HIP error %s: %s" % (status, hipGetErrorString(status)))

  _libhip.hipDeviceSynchronize.restype = int
  _libhip.hipDeviceSynchronize.argtypes = []
  def hipDeviceSynchronize():
    status = _libhip.hipDeviceSynchronize()
    hipCheckStatus(status)

  _libhip.hipStreamSynchronize.restype = int
  _libhip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
  def hipStreamSynchronize(stream):
    status = _libhip.hipStreamSynchronize(stream)
    hipCheckStatus(status)

  _libhip.hipEventCreate.restype = int
  _libhip.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  def hipEventCreate():
    ptr = ctypes.c_void_p()
    status = _libhip.hipEventCreate(ctypes.byref(ptr))
    hipCheckStatus(status)
    return ptr

  _libhip.hipEventRecord.restype = int
  _libhip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def hipEventRecord(event, stream=None):
    status = _libhip.hipEventRecord(event, stream)
    hipCheckStatus(status)

  _libhip.hipEventDestroy.restype = int
  _libhip.hipEventDestroy.argtypes = [ctypes.c_void_p]
  def hipEventDestroy(event):
    status = _libhip.hipEventDestroy(event)
    hipCheckStatus(status)

  _libhip.hipEventSynchronize.restype = int
  _libhip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
  def hipEventSynchronize(event):
    status = _libhip.hipEventSynchronize(event)
    hipCheckStatus(status)

  _libhip.hipEventElapsedTime.restype = int
  _libhip.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
  def hipEventElapsedTime(start, stop):
    t = ctypes.c_float()
    status = _libhip.hipEventElapsedTime(ctypes.byref(t), start, stop)
    hipCheckStatus(status)
    return t.value

  ## Stream Management

  # Stream capture modes:
  hipStreamCaptureModeGlobal = 0
  hipStreamCaptureModeThreadLocal = 1
  hipStreamCaptureModeRelaxed = 2

  _libhip.hipStreamCreate.restype = int
  _libhip.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  def hipStreamCreate():
    ptr = ctypes.c_void_p()
    status = _libhip.hipStreamCreate(ctypes.byref(ptr))
    hipCheckStatus(status)
    return ptr

  _libhip.hipStreamDestroy.restype = int
  _libhip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
  def hipStreamDestroy(stream):
    status = _libhip.hipStreamDestroy(stream)
    hipCheckStatus(status)

  _libhip.hipStreamBeginCapture.restype = int
  _libhip.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
  def hipStreamBeginCapture(stream, mode=hipStreamCaptureModeGlobal):
    t = ctypes.c_float()
    status = _libhip.hipStreamBeginCapture(stream, mode)
    hipCheckStatus(status)

  _libhip.hipStreamEndCapture.restype = int
  _libhip.hipStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def hipStreamEndCapture(stream):
    ptr = ctypes.c_void_p()
    status = _libhip.hipStreamEndCapture(stream, ctypes.byref(ptr))
    hipCheckStatus(status)
    return ptr

  _libhip.hipStreamGetCaptureInfo_v2.restype = int
  _libhip.hipStreamGetCaptureInfo_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  def hipStreamGetCaptureInfo_v2(stream):
    status_out = ctypes.c_void_p()
    id_out = ctypes.c_ulonglong()
    graph_out = ctypes.c_void_p()
    deps_out = ctypes.POINTER(ctypes.c_void_p)()
    num_deps = ctypes.c_size_t()
    status = _libhip.hipStreamGetCaptureInfo_v2(stream, ctypes.byref(status_out), ctypes.byref(id_out), ctypes.byref(graph_out), ctypes.byref(deps_out), ctypes.byref(num_deps))
    hipCheckStatus(status)
    deps = [ctypes.cast(deps_out[i], ctypes.c_void_p) for i in range(num_deps.value)]
    return status_out, id_out.value, graph_out, deps

  hipStreamAddCaptureDependencies = 0
  hipStreamSetCaptureDependencies = 1

  _libhip.hipStreamUpdateCaptureDependencies.restype = int
  _libhip.hipStreamUpdateCaptureDependencies.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
  def hipStreamUpdateCaptureDependencies(stream, deps, flags=hipStreamAddCaptureDependencies):
    deps_in = (ctypes.c_void_p * len(deps))()
    deps_in[:] = deps
    num_deps = ctypes.c_size_t()
    num_deps.value = len(deps)
    flags_in = ctypes.c_uint()
    flags_in.value = flags
    status = _libhip.hipStreamUpdateCaptureDependencies(stream, deps_in, num_deps, flags_in)
    hipCheckStatus(status)


  ## Graph Management

  _libhip.hipGraphCreate.restype = int
  _libhip.hipGraphCreate.argtypes = [ctypes.c_void_p, ctypes.c_uint]
  def hipGraphCreate():
    ptr = ctypes.c_void_p()
    status = _libhip.hipGraphCreate(ctypes.byref(ptr), 0)
    hipCheckStatus(status)
    return ptr

  _libhip.hipGraphInstantiate.restype = int
  _libhip.hipGraphInstantiate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  def hipGraphInstantiate(graph):
    ptr = ctypes.c_void_p()
    status = _libhip.hipGraphInstantiate(ctypes.byref(ptr), graph, 0, 0, 0)
    hipCheckStatus(status)
    return ptr

  _libhip.hipGraphDestroy.restype = int
  _libhip.hipGraphDestroy.argtypes = [ctypes.c_void_p]
  def hipGraphDestroy(graph):
    status = _libhip.hipGraphDestroy(graph)
    hipCheckStatus(status)

  _libhip.hipGraphExecDestroy.restype = int
  _libhip.hipGraphExecDestroy.argtypes = [ctypes.c_void_p]
  def hipGraphExecDestroy(gexec):
    status = _libhip.hipGraphExecDestroy(gexec)
    hipCheckStatus(status)

  _libhip.hipGraphLaunch.restype = int
  _libhip.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def hipGraphLaunch(graph_exec, stream=0):
    status = _libhip.hipGraphLaunch(graph_exec, stream)
    hipCheckStatus(status)

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

  # Better to cache struct_types since they reused often and take a lot of time to create.
  struct_type_cache: Dict[str, Any] = {}
  def __get_struct(name, field_list):
    global struct_type_cache
    if name in struct_type_cache:
      return struct_type_cache[name]
    class CStructure(ctypes.Structure):
      _fields_ = field_list
    struct_type_cache[name] = CStructure
    return struct_type_cache[name]

  def getStructTypeForArgs(*args):
    types = ""
    fields: List[Tuple[str, Any]] = []
    for idx in range(len(args)):
      if args[idx].__class__ is int:
        types += 'i'
        fields.append((f'field{idx}', ctypes.c_int))
      else:
        types += 'P'
        fields.append((f'field{idx}', ctypes.c_void_p))
    return __get_struct(types, fields)

  def updateKernelNodeParams(npwrapper:kernelNodeParamsWrapper, *args, grid=(1,1,1), block=(1,1,1), updated_args=None):
    _, struct, _ = npwrapper.context
    if updated_args is not None:
      for i in updated_args:
        setattr(struct, f'field{i}', (args[i] if args[i].__class__ is int else args[i]._buf))
    else:
      for i,d in enumerate(args):
        setattr(struct, f'field{i}', (d if d.__class__ is int else d._buf))
    npwrapper.c_struct.blockDimX = block[0]
    npwrapper.c_struct.blockDimY = block[1]
    npwrapper.c_struct.blockDimZ = block[2]
    npwrapper.c_struct.gridDimX = grid[0]
    npwrapper.c_struct.gridDimY = grid[1]
    npwrapper.c_struct.gridDimZ = grid[2]

  def buildKernelNodeParams(*args, func=None, grid=(1,1,1), block=(1,1,1), sharedMemBytes=0, argsStructType=None):
    data = [d if d.__class__ is int else d._buf for d in args]
    if argsStructType is None: argsStructType = getStructTypeForArgs(*args)
    struct = argsStructType(*data)
    size = ctypes.c_size_t(ctypes.sizeof(struct))
    p_size = ctypes.c_void_p(ctypes.addressof(size))
    p_struct = ctypes.c_void_p(ctypes.addressof(struct))
    config = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), p_struct,
                                  ctypes.c_void_p(2), p_size, ctypes.c_void_p(3))
    params = hipKernelNodeParams(block[0], block[1], block[2], config, func, grid[0], grid[1], grid[2], None, sharedMemBytes)
    return kernelNodeParamsWrapper(c_struct=params, context=(size, struct, config))

  _libhip.hipGraphAddKernelNode.restype = int
  _libhip.hipGraphAddKernelNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
  def hipGraphAddKernelNode(graph, deps, params:kernelNodeParamsWrapper):
    graph_node = ctypes.c_void_p()
    deps_in = (ctypes.c_void_p * len(deps))()
    deps_in[:] = deps
    num_deps = ctypes.c_size_t(len(deps))
    status = _libhip.hipGraphAddKernelNode(ctypes.byref(graph_node), graph, deps_in, num_deps, ctypes.byref(params.c_struct))
    hipCheckStatus(status)
    return graph_node

  _libhip.hipGraphExecKernelNodeSetParams.restype = int
  _libhip.hipGraphExecKernelNodeSetParams.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  def hipGraphExecKernelNodeSetParams(gexec, node, params:kernelNodeParamsWrapper):
    status = _libhip.hipGraphExecKernelNodeSetParams(gexec, node, ctypes.byref(params.c_struct))
    hipCheckStatus(status)

  _libhip.hipMalloc.restype = int
  _libhip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
  def hipMalloc(count):
    ptr = ctypes.c_void_p()
    status = _libhip.hipMalloc(ctypes.byref(ptr), count)
    hipCheckStatus(status)
    return ptr.value

  _libhip.hipFree.restype = int
  _libhip.hipFree.argtypes = [ctypes.c_void_p]
  def hipFree(ptr):
    status = _libhip.hipFree(ptr)
    hipCheckStatus(status)

  # memory copy modes
  hipMemcpyHostToHost = 0
  hipMemcpyHostToDevice = 1
  hipMemcpyDeviceToHost = 2
  hipMemcpyDeviceToDevice = 3
  hipMemcpyDefault = 4

  _libhip.hipMemcpy.restype = int
  _libhip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
  def hipMemcpy(dst, src, count, direction):
    status = _libhip.hipMemcpy(dst, src, ctypes.c_size_t(count), direction)
    hipCheckStatus(status)

  _libhip.hipMemcpyAsync.restype = int
  _libhip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
  def hipMemcpyAsync(dst, src, count, direction, stream):
    status = _libhip.hipMemcpyAsync(dst, src, ctypes.c_size_t(count), direction, stream)
    hipCheckStatus(status)

  _libhip.hipDeviceEnablePeerAccess.restype = int
  _libhip.hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
  def hipDeviceEnablePeerAccess(peerDevice, flags):
    status = _libhip.hipDeviceEnablePeerAccess(peerDevice, flags)
    hipCheckStatus(status)

  _libhip.hipMemGetInfo.restype = int
  _libhip.hipMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def hipMemGetInfo():
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libhip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    hipCheckStatus(status)
    return free.value, total.value

  class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]

  _libhip.hipIpcGetMemHandle.restype = int
  _libhip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p]
  def hipIpcGetMemHandle(ptr):
    handle = hipIpcMemHandle_t()
    status = _libhip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    hipCheckStatus(status)
    return handle

  _libhip.hipIpcOpenMemHandle.restype = int
  _libhip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint]
  def hipIpcOpenMemHandle(handle, flags):
    ptr = ctypes.c_void_p()
    status = _libhip.hipIpcOpenMemHandle(ctypes.byref(ptr), handle, flags)
    hipCheckStatus(status)
    return ptr.value

  _libhip.hipIpcCloseMemHandle.restype = int
  _libhip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
  def hipIpcCloseMemHandle(ptr):
    status = _libhip.hipIpcCloseMemHandle(ptr)
    hipCheckStatus(status)

  _libhip.hipSetDevice.restype = int
  _libhip.hipSetDevice.argtypes = [ctypes.c_int]
  def hipSetDevice(dev):
    status = _libhip.hipSetDevice(dev)
    hipCheckStatus(status)

  _libhip.hipGetDevice.restype = int
  _libhip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
  def hipGetDevice():
    dev = ctypes.c_int()
    status = _libhip.hipGetDevice(ctypes.byref(dev))
    hipCheckStatus(status)
    return dev.value

  _libhip.hipGetDeviceCount.restype = int
  _libhip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
  def hipGetDeviceCount():
    count = ctypes.c_int()
    status = _libhip.hipGetDeviceCount(ctypes.byref(count))
    hipCheckStatus(status)
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

  _libhip.hipGetDeviceProperties.restype = int
  _libhip.hipGetDeviceProperties.argtypes = [ctypes.POINTER(hipDeviceProperties), ctypes.c_int]
  def hipGetDeviceProperties(deviceId: int):
    device_properties = hipDeviceProperties()
    status = _libhip.hipGetDeviceProperties(ctypes.pointer(device_properties), deviceId)
    hipCheckStatus(status)
    return device_properties

  _libhip.hipModuleLoadData.restype = int
  _libhip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
  def hipModuleLoadData(data):
    module = ctypes.c_void_p()
    status = _libhip.hipModuleLoadData(ctypes.byref(module), data)
    hipCheckStatus(status)
    return module

  _libhip.hipModuleGetFunction.restype = int
  _libhip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
  def hipModuleGetFunction(module, func_name):
    kernel = ctypes.c_void_p()
    status = _libhip.hipModuleGetFunction(ctypes.byref(kernel), module, func_name.encode("utf-8"))
    hipCheckStatus(status)
    return kernel

  _libhip.hipModuleUnload.restype = int
  _libhip.hipModuleUnload.argtypes = [ctypes.c_void_p]
  def hipModuleUnload(module):
    status = _libhip.hipModuleUnload(module)
    hipCheckStatus(status)

  _libhip.hipModuleLaunchKernel.restype = int
  _libhip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p,
                                            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # block dim
                                            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # thread dim
                                            ctypes.c_uint, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
  def hipModuleLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct):
    c_bx, c_by, c_bz = ctypes.c_uint(bx), ctypes.c_uint(by), ctypes.c_uint(bz)
    c_tx, c_ty, c_tz = ctypes.c_uint(tx), ctypes.c_uint(ty), ctypes.c_uint(tz)
    c_shared = ctypes.c_uint(shared)

    param_buffer_ptr, param_buffer_size, param_buffer_end = ctypes.c_void_p(1), ctypes.c_void_p(2), ctypes.c_void_p(3)
    size = ctypes.c_size_t(ctypes.sizeof(struct))
    p_size, p_struct = ctypes.c_void_p(ctypes.addressof(size)), ctypes.c_void_p(ctypes.addressof(struct))
    config = (ctypes.c_void_p * 5)(param_buffer_ptr, p_struct, param_buffer_size, p_size, param_buffer_end)

    status = _libhip.hipModuleLaunchKernel(kernel, c_bx, c_by, c_bz, c_tx, c_ty, c_tz, c_shared, stream, None, config)
    hipCheckStatus(status)

  _libhiprtc.hiprtcCreateProgram.restype = int
  _libhiprtc.hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char),
                                             ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
  def hiprtcCreateProgram(source, name, header_names, header_sources):
    c_header_names, c_header_sources = (ctypes.c_char_p * len(header_names))(), (ctypes.c_char_p * len(header_sources))()
    c_header_names[:], c_header_sources[:] = [h.encode("utf-8") for h in header_names], [h.encode("utf-8") for h in header_sources]

    prog = ctypes.c_void_p()
    status = _libhiprtc.hiprtcCreateProgram(ctypes.byref(prog), source.encode("utf-8"), name.encode("utf-8"), len(header_names), c_header_sources, c_header_names)
    hipCheckStatus(status)
    return prog

  _libhiprtc.hiprtcDestroyProgram.restype = int
  _libhiprtc.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  def hiprtcDestroyProgram(prog):
    status = _libhiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
    hipCheckStatus(status)


  _libhiprtc.hiprtcGetProgramLogSize.restype = int
  _libhiprtc.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
  _libhiprtc.hiprtcGetProgramLog.restype = int
  _libhiprtc.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
  def hiprtcGetProgramLog(prog):
    logsz = ctypes.c_size_t()
    status = _libhiprtc.hiprtcGetProgramLogSize(prog, logsz)
    hipCheckStatus(status)
    logstr = ctypes.create_string_buffer(logsz.value)
    status = _libhiprtc.hiprtcGetProgramLog(prog, logstr)
    hipCheckStatus(status)
    return logstr.value.decode()

  _libhiprtc.hiprtcCompileProgram.restype = int
  _libhiprtc.hiprtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
  def hiprtcCompileProgram(prog, options):
    c_options = (ctypes.c_char_p * len(options))()
    c_options[:] = [o.encode("utf-8") for o in options]

    status = _libhiprtc.hiprtcCompileProgram(prog, len(options), c_options)
    if status == 6: print(hiprtcGetProgramLog(prog))
    hipCheckStatus(status)

  _libhiprtc.hiprtcGetCodeSize.restype = int
  _libhiprtc.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
  _libhiprtc.hiprtcGetCode.restype = int
  _libhiprtc.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
  def hiprtcGetCode(prog):
    code_size = ctypes.c_size_t()
    status = _libhiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
    hipCheckStatus(status)
    e_code = ("0" * code_size.value).encode("utf-8")
    status = _libhiprtc.hiprtcGetCode(prog, e_code)
    hipCheckStatus(status)
    return e_code
except:
  if DEBUG >= 1: print("WARNING: libamdhip64.so or libhiprtc.so not found. HIP support will not work.")
