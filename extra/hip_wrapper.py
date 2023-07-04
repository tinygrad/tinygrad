import ctypes
import sys

try:
  _libhip = ctypes.cdll.LoadLibrary('libamdhip64.so')
except:
  raise OSError('cant find libamdhip64.so.')

try:
  _libhiprtc = ctypes.cdll.LoadLibrary('libhiprtc.so')
except:
  raise OSError('cant find libhiprtc.so.')


def hipCheckStatus(status):
  if status != 0:
    raise RuntimeError('HIP error %s' % status)

_libhip.hipDeviceSynchronize.restype = int
_libhip.hipDeviceSynchronize.argtypes = []


def hipDeviceSynchronize():
  status = _libhip.hipDeviceSynchronize()
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
_libhip.hipEventElapsedTime.argtypes = [ctypes.POINTER(
    ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]


def hipEventElapsedTime(start, stop):
  t = ctypes.c_float()
  status = _libhip.hipEventElapsedTime(ctypes.byref(t), start, stop)
  hipCheckStatus(status)
  return t.value


_libhip.hipMalloc.restype = int
_libhip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                              ctypes.c_size_t]


def hipMalloc(count):
  ptr = ctypes.c_void_p()
  status = _libhip.hipMalloc(ctypes.byref(ptr), count)
  hipCheckStatus(status)
  return ptr


_libhip.hipFree.restype = int
_libhip.hipFree.argtypes = [ctypes.c_void_p]


def hipFree(ptr):
  status = _libhip.hipFree(ptr)
  hipCheckStatus(status)


# Memory copy modes:
hipMemcpyHostToHost = 0
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault = 4

_libhip.hipMemcpy.restype = int
_libhip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def hipMemcpy_htod(dst, src, count):
  status = _libhip.hipMemcpy(dst, src, ctypes.c_size_t(count), hipMemcpyHostToDevice)
  hipCheckStatus(status)


def hipMemcpy_dtoh(dst, src, count):
  status = _libhip.hipMemcpy(dst, src, ctypes.c_size_t(count), hipMemcpyDeviceToHost)
  hipCheckStatus(status)


_libhip.hipMemcpyAsync.restype = int
_libhip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]


def hipMemcpyAsync_htod(dst, src, count, stream):
  status = _libhip.hipMemcpyAsync(dst, src, ctypes.c_size_t(count), hipMemcpyHostToDevice, stream)
  hipCheckStatus(status)


def hipMemcpyAsync_dtoh(dst, src, count, stream):
  status = _libhip.hipMemcpyAsync(dst, src, ctypes.c_size_t(count), hipMemcpyDeviceToHost, stream)
  hipCheckStatus(status)


def hipMemcpyAsync(dst, src, count, direction, stream):
  status = _libhip.hipMemcpyAsync(dst, src, ctypes.c_size_t(count), direction, stream)
  hipCheckStatus(status)


_libhip.hipMemGetInfo.restype = int
_libhip.hipMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def hipMemGetInfo():
  free = ctypes.c_size_t()
  total = ctypes.c_size_t()
  status = _libhip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
  hipCheckStatus(status)
  return free.value, total.value


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


class hipDeviceArch(ctypes.Structure):
  _fields_ = [
    # *32-bit Atomics*
    # 32-bit integer atomics for global memory.
    ('hasGlobalInt32Atomics', ctypes.c_uint, 1),

    # 32-bit float atomic exch for global memory.
    ('hasGlobalFloatAtomicExch', ctypes.c_uint, 1),

    # 32-bit integer atomics for shared memory.
    ('hasSharedInt32Atomics', ctypes.c_uint, 1),

    # 32-bit float atomic exch for shared memory.
    ('hasSharedFloatAtomicExch', ctypes.c_uint, 1),

    # 32-bit float atomic add in global and shared memory.
    ('hasFloatAtomicAdd', ctypes.c_uint, 1),

    # *64-bit Atomics*
    # 64-bit integer atomics for global memory.
    ('hasGlobalInt64Atomics', ctypes.c_uint, 1),

    # 64-bit integer atomics for shared memory.
    ('hasSharedInt64Atomics', ctypes.c_uint, 1),

    # *Doubles*
    # Double-precision floating point.
    ('hasDoubles', ctypes.c_uint, 1),

    # *Warp cross-lane operations*
    # Warp vote instructions (__any, __all).
    ('hasWarpVote', ctypes.c_uint, 1),

    # Warp ballot instructions (__ballot).
    ('hasWarpBallot', ctypes.c_uint, 1),

    # Warp shuffle operations. (__shfl_*).
    ('hasWarpShuffle', ctypes.c_uint, 1),

    # Funnel two words into one with shift&mask caps.
    ('hasFunnelShift', ctypes.c_uint, 1),

    # *Sync*
    # __threadfence_system.
    ('hasThreadFenceSystem', ctypes.c_uint, 1),

    # __syncthreads_count, syncthreads_and, syncthreads_or.
    ('hasSyncThreadsExt', ctypes.c_uint, 1),

    # *Misc*
    # Surface functions.
    ('hasSurfaceFuncs', ctypes.c_uint, 1),

    # Grid and group dims are 3D (rather than 2D).
    ('has3dGrid', ctypes.c_uint, 1),

    # Dynamic parallelism.
    ('hasDynamicParallelism', ctypes.c_uint, 1),
  ]


class hipDeviceProperties(ctypes.Structure):
  _fields_ = [
    # Device name
    ('_name', ctypes.c_char * 256),

    # Size of global memory region (in bytes)
    ('totalGlobalMem', ctypes.c_size_t),

    # Size of shared memory region (in bytes).
    ('sharedMemPerBlock', ctypes.c_size_t),

    # Registers per block.
    ('regsPerBlock', ctypes.c_int),

    # Warp size.
    ('warpSize', ctypes.c_int),

    # Max work items per work group or workgroup max size.
    ('maxThreadsPerBlock', ctypes.c_int),

    # Max number of threads in each dimension (XYZ) of a block.
    ('maxThreadsDim', ctypes.c_int * 3),

    # Max grid dimensions (XYZ).
    ('maxGridSize', ctypes.c_int * 3),

    # Max clock frequency of the multiProcessors in khz.
    ('clockRate', ctypes.c_int),

    # Max global memory clock frequency in khz.
    ('memoryClockRate', ctypes.c_int),

    # Global memory bus width in bits.
    ('memoryBusWidth', ctypes.c_int),

    # Size of shared memory region (in bytes).
    ('totalConstMem', ctypes.c_size_t),

    # Major compute capability.  On HCC, this is an approximation and features may
    # differ from CUDA CC.  See the arch feature flags for portable ways to query
    # feature caps.
    ('major', ctypes.c_int),

    # Minor compute capability.  On HCC, this is an approximation and features may
    # differ from CUDA CC.  See the arch feature flags for portable ways to query
    # feature caps.
    ('minor', ctypes.c_int),

    # Number of multi-processors (compute units).
    ('multiProcessorCount', ctypes.c_int),

    # L2 cache size.
    ('l2CacheSize', ctypes.c_int),

    # Maximum resident threads per multi-processor.
    ('maxThreadsPerMultiProcessor', ctypes.c_int),

    # Compute mode.
    ('computeMode', ctypes.c_int),

    # Frequency in khz of the timer used by the device-side "clock*"
    # instructions.  New for HIP.
    ('clockInstructionRate', ctypes.c_int),

    # Architectural feature flags.  New for HIP.
    ('arch', hipDeviceArch),

    # Device can possibly execute multiple kernels concurrently.
    ('concurrentKernels', ctypes.c_int),

    # PCI Domain ID
    ('pciDomainID', ctypes.c_int),

    # PCI Bus ID.
    ('pciBusID', ctypes.c_int),

    # PCI Device ID.
    ('pciDeviceID', ctypes.c_int),

    # Maximum Shared Memory Per Multiprocessor.
    ('maxSharedMemoryPerMultiProcessor', ctypes.c_size_t),

    # 1 if device is on a multi-GPU board, 0 if not.
    ('isMultiGpuBoard', ctypes.c_int),

    # Check whether HIP can map host memory
    ('canMapHostMemory', ctypes.c_int),

    # DEPRECATED: use gcnArchName instead
    ('gcnArch', ctypes.c_int),

    # AMD GCN Arch Name.
    ('_gcnArchName', ctypes.c_char * 256),

    # APU vs dGPU
    ('integrated', ctypes.c_int),

    # HIP device supports cooperative launch
    ('cooperativeLaunch', ctypes.c_int),

    # HIP device supports cooperative launch on multiple devices
    ('cooperativeMultiDeviceLaunch', ctypes.c_int),

    # Maximum size for 1D textures bound to linear memory
    ('maxTexture1DLinear', ctypes.c_int),

    # Maximum number of elements in 1D images
    ('maxTexture1D', ctypes.c_int),

    # Maximum dimensions (width, height) of 2D images, in image elements
    ('maxTexture2D', ctypes.c_int * 2),

    # Maximum dimensions (width, height, depth) of 3D images, in image elements
    ('maxTexture3D', ctypes.c_int * 3),

    # Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    ('hdpMemFlushCntl', ctypes.POINTER(ctypes.c_uint)),

    # Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    ('hdpRegFlushCntl', ctypes.POINTER(ctypes.c_uint)),

    # Maximum pitch in bytes allowed by memory copies
    ('memPitch', ctypes.c_size_t),

    # Alignment requirement for textures
    ('textureAlignment', ctypes.c_size_t),

    # Pitch alignment requirement for texture references bound to pitched memory
    ('texturePitchAlignment', ctypes.c_size_t),

    # Run time limit for kernels executed on the device
    ('kernelExecTimeoutEnabled', ctypes.c_int),

    # Device has ECC support enabled
    ('ECCEnabled', ctypes.c_int),

    # 1:If device is Tesla device using TCC driver, else 0
    ('tccDriver', ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched functions
    ('cooperativeMultiDeviceUnmatchedFunc', ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched grid dimensions
    ('cooperativeMultiDeviceUnmatchedGridDim', ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched block dimensions
    ('cooperativeMultiDeviceUnmatchedBlockDim', ctypes.c_int),

    # HIP device supports cooperative launch on multiple
    # devices with unmatched shared memories
    ('cooperativeMultiDeviceUnmatchedSharedMem', ctypes.c_int),

    # 1: if it is a large PCI bar device, else 0
    ('isLargeBar', ctypes.c_int),

    # Revision of the GPU in this device
    ('asicRevision', ctypes.c_int),

    # Device supports allocating managed memory on this system
    ('managedMemory', ctypes.c_int),

    # Host can directly access managed memory on the device without migration
    ('directManagedMemAccessFromHost', ctypes.c_int),

    # Device can coherently access managed memory concurrently with the CPU
    ('concurrentManagedAccess', ctypes.c_int),

    # Device supports coherently accessing pageable memory
    # without calling hipHostRegister on it
    ('pageableMemoryAccess', ctypes.c_int),

    # Device accesses pageable memory via the host's page tables
    ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int),
  ]

  @property
  def name(self):
    return self._name.decode('utf-8')

  @property
  def gcnArchName(self):
    return self._gcnArchName.decode('utf-8')


_libhip.hipGetDeviceProperties.restype = int
_libhip.hipGetDeviceProperties.argtypes = [ctypes.POINTER(hipDeviceProperties), ctypes.c_int]


def hipGetDeviceProperties(deviceId: int):
  device_properties = hipDeviceProperties()
  status = _libhip.hipGetDeviceProperties(ctypes.pointer(device_properties), deviceId)
  hipCheckStatus(status)
  return device_properties


_libhip.hipModuleLoadData.restype = int
_libhip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),  # Module
                                      ctypes.c_void_p]                 # Image


def hipModuleLoadData(data):
  module = ctypes.c_void_p()
  status = _libhip.hipModuleLoadData(ctypes.byref(module), data)
  hipCheckStatus(status)
  return module


_libhip.hipModuleGetFunction.restype = int
_libhip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),  # Kernel
                                         ctypes.c_void_p,                    # Module
                                         ctypes.POINTER(ctypes.c_char)]      # kernel name


def hipModuleGetFunction(module, func_name):
  e_func_name = func_name.encode('utf-8')
  kernel = ctypes.c_void_p()
  status = _libhip.hipModuleGetFunction(ctypes.byref(kernel), module, e_func_name)
  hipCheckStatus(status)
  return kernel


_libhip.hipModuleUnload.restype = int
_libhip.hipModuleUnload.argtypes = [ctypes.c_void_p]


def hipModuleUnload(module):
  status = _libhip.hipModuleUnload(module)
  hipCheckStatus(status)


_libhip.hipModuleLaunchKernel.restype = int
_libhip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p,                 # kernel
                                          ctypes.c_uint,                   # block x
                                          ctypes.c_uint,                   # block y
                                          ctypes.c_uint,                   # block z
                                          ctypes.c_uint,                   # thread x
                                          ctypes.c_uint,                   # thread y
                                          ctypes.c_uint,                   # thread z
                                          ctypes.c_uint,                   # shared mem
                                          ctypes.c_void_p,                 # stream
                                          # kernel params
                                          ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_void_p)]  # extra


def hipModuleLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct):
  c_bx = ctypes.c_uint(bx)
  c_by = ctypes.c_uint(by)
  c_bz = ctypes.c_uint(bz)
  c_tx = ctypes.c_uint(tx)
  c_ty = ctypes.c_uint(ty)
  c_tz = ctypes.c_uint(tz)
  c_shared = ctypes.c_uint(shared)

  ctypes.sizeof(struct)
  hip_launch_param_buffer_ptr = ctypes.c_void_p(1)
  hip_launch_param_buffer_size = ctypes.c_void_p(2)
  hip_launch_param_buffer_end = ctypes.c_void_p(0)
  hip_launch_param_buffer_end = ctypes.c_void_p(3)
  size = ctypes.c_size_t(ctypes.sizeof(struct))
  p_size = ctypes.c_void_p(ctypes.addressof(size))
  p_struct = ctypes.c_void_p(ctypes.addressof(struct))
  config = (ctypes.c_void_p * 5)(hip_launch_param_buffer_ptr, p_struct,
                                 hip_launch_param_buffer_size, p_size, hip_launch_param_buffer_end)
  nullptr = ctypes.POINTER(ctypes.c_void_p)(ctypes.c_void_p(0))

  status = _libhip.hipModuleLaunchKernel(
      kernel, c_bx, c_by, c_bz, c_tx, c_ty, c_tz, c_shared, stream, None, config)
  hipCheckStatus(status)


_libhiprtc.hiprtcCreateProgram.restype = int
_libhiprtc.hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p),  # hiprtcProgram
                                           ctypes.POINTER(
                                               ctypes.c_char),   # Source
                                           ctypes.POINTER(
                                               ctypes.c_char),   # Name
                                           ctypes.c_int,                    # numberOfHeaders
                                           ctypes.POINTER(
                                               ctypes.c_char_p),  # header
                                           ctypes.POINTER(ctypes.c_char_p)]  # headerNames


def hiprtcCreateProgram(source, name, header_names, header_sources):
  e_source = source.encode('utf-8')
  e_name = name.encode('utf-8')
  e_header_names = list()
  e_header_sources = list()
  for header_name in header_names:
    e_header_name = header_name.encode('utf-8')
    e_header_names.append(e_header_name)
  for header_source in header_sources:
    e_header_source = header_source.encode('utf-8')
    e_header_sources.append(e_header_source)

  prog = ctypes.c_void_p()
  c_header_names = (ctypes.c_char_p * len(e_header_names))()
  c_header_names[:] = e_header_names
  c_header_sources = (ctypes.c_char_p * len(e_header_sources))()
  c_header_sources[:] = e_header_sources
  status = _libhiprtc.hiprtcCreateProgram(ctypes.byref(
    prog), e_source, e_name, len(e_header_names), c_header_sources, c_header_names)
  hipCheckStatus(status)
  return prog


_libhiprtc.hiprtcDestroyProgram.restype = int
_libhiprtc.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]  # hiprtcProgram


def hiprtcDestroyProgram(prog):
  status = _libhiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
  hipCheckStatus(status)


_libhiprtc.hiprtcCompileProgram.restype = int
_libhiprtc.hiprtcCompileProgram.argtypes = [ctypes.c_void_p,                 # hiprtcProgram
                                            ctypes.c_int,                    # num of options
                                            ctypes.POINTER(ctypes.c_char_p)]  # options


def hiprtcCompileProgram(prog, options):
  e_options = list()
  for option in options:
    e_options.append(option.encode('utf-8'))
  c_options = (ctypes.c_char_p * len(e_options))()
  c_options[:] = e_options
  status = _libhiprtc.hiprtcCompileProgram(prog, len(c_options), c_options)
  if status == 6: print(hiprtcGetProgramLog(prog))
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

_libhiprtc.hiprtcGetCodeSize.restype = int
_libhiprtc.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p,                 # hiprtcProgram
                                         ctypes.POINTER(ctypes.c_size_t)]  # Size of log
_libhiprtc.hiprtcGetCode.restype = int
_libhiprtc.hiprtcGetCode.argtypes = [ctypes.c_void_p,               # hiprtcProgram
                                     ctypes.POINTER(ctypes.c_char)]  # log


def hiprtcGetCode(prog):
  code_size = ctypes.c_size_t()
  status = _libhiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
  hipCheckStatus(status)
  code = "0" * code_size.value
  e_code = code.encode('utf-8')
  status = _libhiprtc.hiprtcGetCode(prog, e_code)
  hipCheckStatus(status)
  return e_code
