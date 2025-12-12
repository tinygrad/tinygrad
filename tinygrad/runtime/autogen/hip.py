# mypy: ignore-errors
import ctypes, os
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
def dll():
  try: return ctypes.CDLL(unwrap(os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so'))
  except: pass
  return None
dll = dll()

_anonenum0 = CEnum(ctypes.c_uint32)
HIP_SUCCESS = _anonenum0.define('HIP_SUCCESS', 0)
HIP_ERROR_INVALID_VALUE = _anonenum0.define('HIP_ERROR_INVALID_VALUE', 1)
HIP_ERROR_NOT_INITIALIZED = _anonenum0.define('HIP_ERROR_NOT_INITIALIZED', 2)
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = _anonenum0.define('HIP_ERROR_LAUNCH_OUT_OF_RESOURCES', 3)

class hipDeviceArch_t(Struct): pass
hipDeviceArch_t._fields_ = [
  ('hasGlobalInt32Atomics', ctypes.c_uint32,1),
  ('hasGlobalFloatAtomicExch', ctypes.c_uint32,1),
  ('hasSharedInt32Atomics', ctypes.c_uint32,1),
  ('hasSharedFloatAtomicExch', ctypes.c_uint32,1),
  ('hasFloatAtomicAdd', ctypes.c_uint32,1),
  ('hasGlobalInt64Atomics', ctypes.c_uint32,1),
  ('hasSharedInt64Atomics', ctypes.c_uint32,1),
  ('hasDoubles', ctypes.c_uint32,1),
  ('hasWarpVote', ctypes.c_uint32,1),
  ('hasWarpBallot', ctypes.c_uint32,1),
  ('hasWarpShuffle', ctypes.c_uint32,1),
  ('hasFunnelShift', ctypes.c_uint32,1),
  ('hasThreadFenceSystem', ctypes.c_uint32,1),
  ('hasSyncThreadsExt', ctypes.c_uint32,1),
  ('hasSurfaceFuncs', ctypes.c_uint32,1),
  ('has3dGrid', ctypes.c_uint32,1),
  ('hasDynamicParallelism', ctypes.c_uint32,1),
]
class hipUUID_t(Struct): pass
hipUUID_t._fields_ = [
  ('bytes', (ctypes.c_char * 16)),
]
hipUUID = hipUUID_t
class hipDeviceProp_tR0600(Struct): pass
size_t = ctypes.c_uint64
hipDeviceProp_tR0600._fields_ = [
  ('name', (ctypes.c_char * 256)),
  ('uuid', hipUUID),
  ('luid', (ctypes.c_char * 8)),
  ('luidDeviceNodeMask', ctypes.c_uint32),
  ('totalGlobalMem', size_t),
  ('sharedMemPerBlock', size_t),
  ('regsPerBlock', ctypes.c_int32),
  ('warpSize', ctypes.c_int32),
  ('memPitch', size_t),
  ('maxThreadsPerBlock', ctypes.c_int32),
  ('maxThreadsDim', (ctypes.c_int32 * 3)),
  ('maxGridSize', (ctypes.c_int32 * 3)),
  ('clockRate', ctypes.c_int32),
  ('totalConstMem', size_t),
  ('major', ctypes.c_int32),
  ('minor', ctypes.c_int32),
  ('textureAlignment', size_t),
  ('texturePitchAlignment', size_t),
  ('deviceOverlap', ctypes.c_int32),
  ('multiProcessorCount', ctypes.c_int32),
  ('kernelExecTimeoutEnabled', ctypes.c_int32),
  ('integrated', ctypes.c_int32),
  ('canMapHostMemory', ctypes.c_int32),
  ('computeMode', ctypes.c_int32),
  ('maxTexture1D', ctypes.c_int32),
  ('maxTexture1DMipmap', ctypes.c_int32),
  ('maxTexture1DLinear', ctypes.c_int32),
  ('maxTexture2D', (ctypes.c_int32 * 2)),
  ('maxTexture2DMipmap', (ctypes.c_int32 * 2)),
  ('maxTexture2DLinear', (ctypes.c_int32 * 3)),
  ('maxTexture2DGather', (ctypes.c_int32 * 2)),
  ('maxTexture3D', (ctypes.c_int32 * 3)),
  ('maxTexture3DAlt', (ctypes.c_int32 * 3)),
  ('maxTextureCubemap', ctypes.c_int32),
  ('maxTexture1DLayered', (ctypes.c_int32 * 2)),
  ('maxTexture2DLayered', (ctypes.c_int32 * 3)),
  ('maxTextureCubemapLayered', (ctypes.c_int32 * 2)),
  ('maxSurface1D', ctypes.c_int32),
  ('maxSurface2D', (ctypes.c_int32 * 2)),
  ('maxSurface3D', (ctypes.c_int32 * 3)),
  ('maxSurface1DLayered', (ctypes.c_int32 * 2)),
  ('maxSurface2DLayered', (ctypes.c_int32 * 3)),
  ('maxSurfaceCubemap', ctypes.c_int32),
  ('maxSurfaceCubemapLayered', (ctypes.c_int32 * 2)),
  ('surfaceAlignment', size_t),
  ('concurrentKernels', ctypes.c_int32),
  ('ECCEnabled', ctypes.c_int32),
  ('pciBusID', ctypes.c_int32),
  ('pciDeviceID', ctypes.c_int32),
  ('pciDomainID', ctypes.c_int32),
  ('tccDriver', ctypes.c_int32),
  ('asyncEngineCount', ctypes.c_int32),
  ('unifiedAddressing', ctypes.c_int32),
  ('memoryClockRate', ctypes.c_int32),
  ('memoryBusWidth', ctypes.c_int32),
  ('l2CacheSize', ctypes.c_int32),
  ('persistingL2CacheMaxSize', ctypes.c_int32),
  ('maxThreadsPerMultiProcessor', ctypes.c_int32),
  ('streamPrioritiesSupported', ctypes.c_int32),
  ('globalL1CacheSupported', ctypes.c_int32),
  ('localL1CacheSupported', ctypes.c_int32),
  ('sharedMemPerMultiprocessor', size_t),
  ('regsPerMultiprocessor', ctypes.c_int32),
  ('managedMemory', ctypes.c_int32),
  ('isMultiGpuBoard', ctypes.c_int32),
  ('multiGpuBoardGroupID', ctypes.c_int32),
  ('hostNativeAtomicSupported', ctypes.c_int32),
  ('singleToDoublePrecisionPerfRatio', ctypes.c_int32),
  ('pageableMemoryAccess', ctypes.c_int32),
  ('concurrentManagedAccess', ctypes.c_int32),
  ('computePreemptionSupported', ctypes.c_int32),
  ('canUseHostPointerForRegisteredMem', ctypes.c_int32),
  ('cooperativeLaunch', ctypes.c_int32),
  ('cooperativeMultiDeviceLaunch', ctypes.c_int32),
  ('sharedMemPerBlockOptin', size_t),
  ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int32),
  ('directManagedMemAccessFromHost', ctypes.c_int32),
  ('maxBlocksPerMultiProcessor', ctypes.c_int32),
  ('accessPolicyMaxWindowSize', ctypes.c_int32),
  ('reservedSharedMemPerBlock', size_t),
  ('hostRegisterSupported', ctypes.c_int32),
  ('sparseHipArraySupported', ctypes.c_int32),
  ('hostRegisterReadOnlySupported', ctypes.c_int32),
  ('timelineSemaphoreInteropSupported', ctypes.c_int32),
  ('memoryPoolsSupported', ctypes.c_int32),
  ('gpuDirectRDMASupported', ctypes.c_int32),
  ('gpuDirectRDMAFlushWritesOptions', ctypes.c_uint32),
  ('gpuDirectRDMAWritesOrdering', ctypes.c_int32),
  ('memoryPoolSupportedHandleTypes', ctypes.c_uint32),
  ('deferredMappingHipArraySupported', ctypes.c_int32),
  ('ipcEventSupported', ctypes.c_int32),
  ('clusterLaunch', ctypes.c_int32),
  ('unifiedFunctionPointers', ctypes.c_int32),
  ('reserved', (ctypes.c_int32 * 63)),
  ('hipReserved', (ctypes.c_int32 * 32)),
  ('gcnArchName', (ctypes.c_char * 256)),
  ('maxSharedMemoryPerMultiProcessor', size_t),
  ('clockInstructionRate', ctypes.c_int32),
  ('arch', hipDeviceArch_t),
  ('hdpMemFlushCntl', ctypes.POINTER(ctypes.c_uint32)),
  ('hdpRegFlushCntl', ctypes.POINTER(ctypes.c_uint32)),
  ('cooperativeMultiDeviceUnmatchedFunc', ctypes.c_int32),
  ('cooperativeMultiDeviceUnmatchedGridDim', ctypes.c_int32),
  ('cooperativeMultiDeviceUnmatchedBlockDim', ctypes.c_int32),
  ('cooperativeMultiDeviceUnmatchedSharedMem', ctypes.c_int32),
  ('isLargeBar', ctypes.c_int32),
  ('asicRevision', ctypes.c_int32),
]
hipMemoryType = CEnum(ctypes.c_uint32)
hipMemoryTypeUnregistered = hipMemoryType.define('hipMemoryTypeUnregistered', 0)
hipMemoryTypeHost = hipMemoryType.define('hipMemoryTypeHost', 1)
hipMemoryTypeDevice = hipMemoryType.define('hipMemoryTypeDevice', 2)
hipMemoryTypeManaged = hipMemoryType.define('hipMemoryTypeManaged', 3)
hipMemoryTypeArray = hipMemoryType.define('hipMemoryTypeArray', 10)
hipMemoryTypeUnified = hipMemoryType.define('hipMemoryTypeUnified', 11)

class hipPointerAttribute_t(Struct): pass
hipPointerAttribute_t._fields_ = [
  ('type', hipMemoryType),
  ('device', ctypes.c_int32),
  ('devicePointer', ctypes.c_void_p),
  ('hostPointer', ctypes.c_void_p),
  ('isManaged', ctypes.c_int32),
  ('allocationFlags', ctypes.c_uint32),
]
hipError_t = CEnum(ctypes.c_uint32)
hipSuccess = hipError_t.define('hipSuccess', 0)
hipErrorInvalidValue = hipError_t.define('hipErrorInvalidValue', 1)
hipErrorOutOfMemory = hipError_t.define('hipErrorOutOfMemory', 2)
hipErrorMemoryAllocation = hipError_t.define('hipErrorMemoryAllocation', 2)
hipErrorNotInitialized = hipError_t.define('hipErrorNotInitialized', 3)
hipErrorInitializationError = hipError_t.define('hipErrorInitializationError', 3)
hipErrorDeinitialized = hipError_t.define('hipErrorDeinitialized', 4)
hipErrorProfilerDisabled = hipError_t.define('hipErrorProfilerDisabled', 5)
hipErrorProfilerNotInitialized = hipError_t.define('hipErrorProfilerNotInitialized', 6)
hipErrorProfilerAlreadyStarted = hipError_t.define('hipErrorProfilerAlreadyStarted', 7)
hipErrorProfilerAlreadyStopped = hipError_t.define('hipErrorProfilerAlreadyStopped', 8)
hipErrorInvalidConfiguration = hipError_t.define('hipErrorInvalidConfiguration', 9)
hipErrorInvalidPitchValue = hipError_t.define('hipErrorInvalidPitchValue', 12)
hipErrorInvalidSymbol = hipError_t.define('hipErrorInvalidSymbol', 13)
hipErrorInvalidDevicePointer = hipError_t.define('hipErrorInvalidDevicePointer', 17)
hipErrorInvalidMemcpyDirection = hipError_t.define('hipErrorInvalidMemcpyDirection', 21)
hipErrorInsufficientDriver = hipError_t.define('hipErrorInsufficientDriver', 35)
hipErrorMissingConfiguration = hipError_t.define('hipErrorMissingConfiguration', 52)
hipErrorPriorLaunchFailure = hipError_t.define('hipErrorPriorLaunchFailure', 53)
hipErrorInvalidDeviceFunction = hipError_t.define('hipErrorInvalidDeviceFunction', 98)
hipErrorNoDevice = hipError_t.define('hipErrorNoDevice', 100)
hipErrorInvalidDevice = hipError_t.define('hipErrorInvalidDevice', 101)
hipErrorInvalidImage = hipError_t.define('hipErrorInvalidImage', 200)
hipErrorInvalidContext = hipError_t.define('hipErrorInvalidContext', 201)
hipErrorContextAlreadyCurrent = hipError_t.define('hipErrorContextAlreadyCurrent', 202)
hipErrorMapFailed = hipError_t.define('hipErrorMapFailed', 205)
hipErrorMapBufferObjectFailed = hipError_t.define('hipErrorMapBufferObjectFailed', 205)
hipErrorUnmapFailed = hipError_t.define('hipErrorUnmapFailed', 206)
hipErrorArrayIsMapped = hipError_t.define('hipErrorArrayIsMapped', 207)
hipErrorAlreadyMapped = hipError_t.define('hipErrorAlreadyMapped', 208)
hipErrorNoBinaryForGpu = hipError_t.define('hipErrorNoBinaryForGpu', 209)
hipErrorAlreadyAcquired = hipError_t.define('hipErrorAlreadyAcquired', 210)
hipErrorNotMapped = hipError_t.define('hipErrorNotMapped', 211)
hipErrorNotMappedAsArray = hipError_t.define('hipErrorNotMappedAsArray', 212)
hipErrorNotMappedAsPointer = hipError_t.define('hipErrorNotMappedAsPointer', 213)
hipErrorECCNotCorrectable = hipError_t.define('hipErrorECCNotCorrectable', 214)
hipErrorUnsupportedLimit = hipError_t.define('hipErrorUnsupportedLimit', 215)
hipErrorContextAlreadyInUse = hipError_t.define('hipErrorContextAlreadyInUse', 216)
hipErrorPeerAccessUnsupported = hipError_t.define('hipErrorPeerAccessUnsupported', 217)
hipErrorInvalidKernelFile = hipError_t.define('hipErrorInvalidKernelFile', 218)
hipErrorInvalidGraphicsContext = hipError_t.define('hipErrorInvalidGraphicsContext', 219)
hipErrorInvalidSource = hipError_t.define('hipErrorInvalidSource', 300)
hipErrorFileNotFound = hipError_t.define('hipErrorFileNotFound', 301)
hipErrorSharedObjectSymbolNotFound = hipError_t.define('hipErrorSharedObjectSymbolNotFound', 302)
hipErrorSharedObjectInitFailed = hipError_t.define('hipErrorSharedObjectInitFailed', 303)
hipErrorOperatingSystem = hipError_t.define('hipErrorOperatingSystem', 304)
hipErrorInvalidHandle = hipError_t.define('hipErrorInvalidHandle', 400)
hipErrorInvalidResourceHandle = hipError_t.define('hipErrorInvalidResourceHandle', 400)
hipErrorIllegalState = hipError_t.define('hipErrorIllegalState', 401)
hipErrorNotFound = hipError_t.define('hipErrorNotFound', 500)
hipErrorNotReady = hipError_t.define('hipErrorNotReady', 600)
hipErrorIllegalAddress = hipError_t.define('hipErrorIllegalAddress', 700)
hipErrorLaunchOutOfResources = hipError_t.define('hipErrorLaunchOutOfResources', 701)
hipErrorLaunchTimeOut = hipError_t.define('hipErrorLaunchTimeOut', 702)
hipErrorPeerAccessAlreadyEnabled = hipError_t.define('hipErrorPeerAccessAlreadyEnabled', 704)
hipErrorPeerAccessNotEnabled = hipError_t.define('hipErrorPeerAccessNotEnabled', 705)
hipErrorSetOnActiveProcess = hipError_t.define('hipErrorSetOnActiveProcess', 708)
hipErrorContextIsDestroyed = hipError_t.define('hipErrorContextIsDestroyed', 709)
hipErrorAssert = hipError_t.define('hipErrorAssert', 710)
hipErrorHostMemoryAlreadyRegistered = hipError_t.define('hipErrorHostMemoryAlreadyRegistered', 712)
hipErrorHostMemoryNotRegistered = hipError_t.define('hipErrorHostMemoryNotRegistered', 713)
hipErrorLaunchFailure = hipError_t.define('hipErrorLaunchFailure', 719)
hipErrorCooperativeLaunchTooLarge = hipError_t.define('hipErrorCooperativeLaunchTooLarge', 720)
hipErrorNotSupported = hipError_t.define('hipErrorNotSupported', 801)
hipErrorStreamCaptureUnsupported = hipError_t.define('hipErrorStreamCaptureUnsupported', 900)
hipErrorStreamCaptureInvalidated = hipError_t.define('hipErrorStreamCaptureInvalidated', 901)
hipErrorStreamCaptureMerge = hipError_t.define('hipErrorStreamCaptureMerge', 902)
hipErrorStreamCaptureUnmatched = hipError_t.define('hipErrorStreamCaptureUnmatched', 903)
hipErrorStreamCaptureUnjoined = hipError_t.define('hipErrorStreamCaptureUnjoined', 904)
hipErrorStreamCaptureIsolation = hipError_t.define('hipErrorStreamCaptureIsolation', 905)
hipErrorStreamCaptureImplicit = hipError_t.define('hipErrorStreamCaptureImplicit', 906)
hipErrorCapturedEvent = hipError_t.define('hipErrorCapturedEvent', 907)
hipErrorStreamCaptureWrongThread = hipError_t.define('hipErrorStreamCaptureWrongThread', 908)
hipErrorGraphExecUpdateFailure = hipError_t.define('hipErrorGraphExecUpdateFailure', 910)
hipErrorUnknown = hipError_t.define('hipErrorUnknown', 999)
hipErrorRuntimeMemory = hipError_t.define('hipErrorRuntimeMemory', 1052)
hipErrorRuntimeOther = hipError_t.define('hipErrorRuntimeOther', 1053)
hipErrorTbd = hipError_t.define('hipErrorTbd', 1054)

hipDeviceAttribute_t = CEnum(ctypes.c_uint32)
hipDeviceAttributeCudaCompatibleBegin = hipDeviceAttribute_t.define('hipDeviceAttributeCudaCompatibleBegin', 0)
hipDeviceAttributeEccEnabled = hipDeviceAttribute_t.define('hipDeviceAttributeEccEnabled', 0)
hipDeviceAttributeAccessPolicyMaxWindowSize = hipDeviceAttribute_t.define('hipDeviceAttributeAccessPolicyMaxWindowSize', 1)
hipDeviceAttributeAsyncEngineCount = hipDeviceAttribute_t.define('hipDeviceAttributeAsyncEngineCount', 2)
hipDeviceAttributeCanMapHostMemory = hipDeviceAttribute_t.define('hipDeviceAttributeCanMapHostMemory', 3)
hipDeviceAttributeCanUseHostPointerForRegisteredMem = hipDeviceAttribute_t.define('hipDeviceAttributeCanUseHostPointerForRegisteredMem', 4)
hipDeviceAttributeClockRate = hipDeviceAttribute_t.define('hipDeviceAttributeClockRate', 5)
hipDeviceAttributeComputeMode = hipDeviceAttribute_t.define('hipDeviceAttributeComputeMode', 6)
hipDeviceAttributeComputePreemptionSupported = hipDeviceAttribute_t.define('hipDeviceAttributeComputePreemptionSupported', 7)
hipDeviceAttributeConcurrentKernels = hipDeviceAttribute_t.define('hipDeviceAttributeConcurrentKernels', 8)
hipDeviceAttributeConcurrentManagedAccess = hipDeviceAttribute_t.define('hipDeviceAttributeConcurrentManagedAccess', 9)
hipDeviceAttributeCooperativeLaunch = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeLaunch', 10)
hipDeviceAttributeCooperativeMultiDeviceLaunch = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceLaunch', 11)
hipDeviceAttributeDeviceOverlap = hipDeviceAttribute_t.define('hipDeviceAttributeDeviceOverlap', 12)
hipDeviceAttributeDirectManagedMemAccessFromHost = hipDeviceAttribute_t.define('hipDeviceAttributeDirectManagedMemAccessFromHost', 13)
hipDeviceAttributeGlobalL1CacheSupported = hipDeviceAttribute_t.define('hipDeviceAttributeGlobalL1CacheSupported', 14)
hipDeviceAttributeHostNativeAtomicSupported = hipDeviceAttribute_t.define('hipDeviceAttributeHostNativeAtomicSupported', 15)
hipDeviceAttributeIntegrated = hipDeviceAttribute_t.define('hipDeviceAttributeIntegrated', 16)
hipDeviceAttributeIsMultiGpuBoard = hipDeviceAttribute_t.define('hipDeviceAttributeIsMultiGpuBoard', 17)
hipDeviceAttributeKernelExecTimeout = hipDeviceAttribute_t.define('hipDeviceAttributeKernelExecTimeout', 18)
hipDeviceAttributeL2CacheSize = hipDeviceAttribute_t.define('hipDeviceAttributeL2CacheSize', 19)
hipDeviceAttributeLocalL1CacheSupported = hipDeviceAttribute_t.define('hipDeviceAttributeLocalL1CacheSupported', 20)
hipDeviceAttributeLuid = hipDeviceAttribute_t.define('hipDeviceAttributeLuid', 21)
hipDeviceAttributeLuidDeviceNodeMask = hipDeviceAttribute_t.define('hipDeviceAttributeLuidDeviceNodeMask', 22)
hipDeviceAttributeComputeCapabilityMajor = hipDeviceAttribute_t.define('hipDeviceAttributeComputeCapabilityMajor', 23)
hipDeviceAttributeManagedMemory = hipDeviceAttribute_t.define('hipDeviceAttributeManagedMemory', 24)
hipDeviceAttributeMaxBlocksPerMultiProcessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlocksPerMultiProcessor', 25)
hipDeviceAttributeMaxBlockDimX = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlockDimX', 26)
hipDeviceAttributeMaxBlockDimY = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlockDimY', 27)
hipDeviceAttributeMaxBlockDimZ = hipDeviceAttribute_t.define('hipDeviceAttributeMaxBlockDimZ', 28)
hipDeviceAttributeMaxGridDimX = hipDeviceAttribute_t.define('hipDeviceAttributeMaxGridDimX', 29)
hipDeviceAttributeMaxGridDimY = hipDeviceAttribute_t.define('hipDeviceAttributeMaxGridDimY', 30)
hipDeviceAttributeMaxGridDimZ = hipDeviceAttribute_t.define('hipDeviceAttributeMaxGridDimZ', 31)
hipDeviceAttributeMaxSurface1D = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface1D', 32)
hipDeviceAttributeMaxSurface1DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface1DLayered', 33)
hipDeviceAttributeMaxSurface2D = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface2D', 34)
hipDeviceAttributeMaxSurface2DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface2DLayered', 35)
hipDeviceAttributeMaxSurface3D = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurface3D', 36)
hipDeviceAttributeMaxSurfaceCubemap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurfaceCubemap', 37)
hipDeviceAttributeMaxSurfaceCubemapLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSurfaceCubemapLayered', 38)
hipDeviceAttributeMaxTexture1DWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DWidth', 39)
hipDeviceAttributeMaxTexture1DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DLayered', 40)
hipDeviceAttributeMaxTexture1DLinear = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DLinear', 41)
hipDeviceAttributeMaxTexture1DMipmap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture1DMipmap', 42)
hipDeviceAttributeMaxTexture2DWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DWidth', 43)
hipDeviceAttributeMaxTexture2DHeight = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DHeight', 44)
hipDeviceAttributeMaxTexture2DGather = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DGather', 45)
hipDeviceAttributeMaxTexture2DLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DLayered', 46)
hipDeviceAttributeMaxTexture2DLinear = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DLinear', 47)
hipDeviceAttributeMaxTexture2DMipmap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture2DMipmap', 48)
hipDeviceAttributeMaxTexture3DWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DWidth', 49)
hipDeviceAttributeMaxTexture3DHeight = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DHeight', 50)
hipDeviceAttributeMaxTexture3DDepth = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DDepth', 51)
hipDeviceAttributeMaxTexture3DAlt = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTexture3DAlt', 52)
hipDeviceAttributeMaxTextureCubemap = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTextureCubemap', 53)
hipDeviceAttributeMaxTextureCubemapLayered = hipDeviceAttribute_t.define('hipDeviceAttributeMaxTextureCubemapLayered', 54)
hipDeviceAttributeMaxThreadsDim = hipDeviceAttribute_t.define('hipDeviceAttributeMaxThreadsDim', 55)
hipDeviceAttributeMaxThreadsPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeMaxThreadsPerBlock', 56)
hipDeviceAttributeMaxThreadsPerMultiProcessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxThreadsPerMultiProcessor', 57)
hipDeviceAttributeMaxPitch = hipDeviceAttribute_t.define('hipDeviceAttributeMaxPitch', 58)
hipDeviceAttributeMemoryBusWidth = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryBusWidth', 59)
hipDeviceAttributeMemoryClockRate = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryClockRate', 60)
hipDeviceAttributeComputeCapabilityMinor = hipDeviceAttribute_t.define('hipDeviceAttributeComputeCapabilityMinor', 61)
hipDeviceAttributeMultiGpuBoardGroupID = hipDeviceAttribute_t.define('hipDeviceAttributeMultiGpuBoardGroupID', 62)
hipDeviceAttributeMultiprocessorCount = hipDeviceAttribute_t.define('hipDeviceAttributeMultiprocessorCount', 63)
hipDeviceAttributeUnused1 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused1', 64)
hipDeviceAttributePageableMemoryAccess = hipDeviceAttribute_t.define('hipDeviceAttributePageableMemoryAccess', 65)
hipDeviceAttributePageableMemoryAccessUsesHostPageTables = hipDeviceAttribute_t.define('hipDeviceAttributePageableMemoryAccessUsesHostPageTables', 66)
hipDeviceAttributePciBusId = hipDeviceAttribute_t.define('hipDeviceAttributePciBusId', 67)
hipDeviceAttributePciDeviceId = hipDeviceAttribute_t.define('hipDeviceAttributePciDeviceId', 68)
hipDeviceAttributePciDomainID = hipDeviceAttribute_t.define('hipDeviceAttributePciDomainID', 69)
hipDeviceAttributePersistingL2CacheMaxSize = hipDeviceAttribute_t.define('hipDeviceAttributePersistingL2CacheMaxSize', 70)
hipDeviceAttributeMaxRegistersPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeMaxRegistersPerBlock', 71)
hipDeviceAttributeMaxRegistersPerMultiprocessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxRegistersPerMultiprocessor', 72)
hipDeviceAttributeReservedSharedMemPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeReservedSharedMemPerBlock', 73)
hipDeviceAttributeMaxSharedMemoryPerBlock = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSharedMemoryPerBlock', 74)
hipDeviceAttributeSharedMemPerBlockOptin = hipDeviceAttribute_t.define('hipDeviceAttributeSharedMemPerBlockOptin', 75)
hipDeviceAttributeSharedMemPerMultiprocessor = hipDeviceAttribute_t.define('hipDeviceAttributeSharedMemPerMultiprocessor', 76)
hipDeviceAttributeSingleToDoublePrecisionPerfRatio = hipDeviceAttribute_t.define('hipDeviceAttributeSingleToDoublePrecisionPerfRatio', 77)
hipDeviceAttributeStreamPrioritiesSupported = hipDeviceAttribute_t.define('hipDeviceAttributeStreamPrioritiesSupported', 78)
hipDeviceAttributeSurfaceAlignment = hipDeviceAttribute_t.define('hipDeviceAttributeSurfaceAlignment', 79)
hipDeviceAttributeTccDriver = hipDeviceAttribute_t.define('hipDeviceAttributeTccDriver', 80)
hipDeviceAttributeTextureAlignment = hipDeviceAttribute_t.define('hipDeviceAttributeTextureAlignment', 81)
hipDeviceAttributeTexturePitchAlignment = hipDeviceAttribute_t.define('hipDeviceAttributeTexturePitchAlignment', 82)
hipDeviceAttributeTotalConstantMemory = hipDeviceAttribute_t.define('hipDeviceAttributeTotalConstantMemory', 83)
hipDeviceAttributeTotalGlobalMem = hipDeviceAttribute_t.define('hipDeviceAttributeTotalGlobalMem', 84)
hipDeviceAttributeUnifiedAddressing = hipDeviceAttribute_t.define('hipDeviceAttributeUnifiedAddressing', 85)
hipDeviceAttributeUnused2 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused2', 86)
hipDeviceAttributeWarpSize = hipDeviceAttribute_t.define('hipDeviceAttributeWarpSize', 87)
hipDeviceAttributeMemoryPoolsSupported = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryPoolsSupported', 88)
hipDeviceAttributeVirtualMemoryManagementSupported = hipDeviceAttribute_t.define('hipDeviceAttributeVirtualMemoryManagementSupported', 89)
hipDeviceAttributeHostRegisterSupported = hipDeviceAttribute_t.define('hipDeviceAttributeHostRegisterSupported', 90)
hipDeviceAttributeMemoryPoolSupportedHandleTypes = hipDeviceAttribute_t.define('hipDeviceAttributeMemoryPoolSupportedHandleTypes', 91)
hipDeviceAttributeCudaCompatibleEnd = hipDeviceAttribute_t.define('hipDeviceAttributeCudaCompatibleEnd', 9999)
hipDeviceAttributeAmdSpecificBegin = hipDeviceAttribute_t.define('hipDeviceAttributeAmdSpecificBegin', 10000)
hipDeviceAttributeClockInstructionRate = hipDeviceAttribute_t.define('hipDeviceAttributeClockInstructionRate', 10000)
hipDeviceAttributeUnused3 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused3', 10001)
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = hipDeviceAttribute_t.define('hipDeviceAttributeMaxSharedMemoryPerMultiprocessor', 10002)
hipDeviceAttributeUnused4 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused4', 10003)
hipDeviceAttributeUnused5 = hipDeviceAttribute_t.define('hipDeviceAttributeUnused5', 10004)
hipDeviceAttributeHdpMemFlushCntl = hipDeviceAttribute_t.define('hipDeviceAttributeHdpMemFlushCntl', 10005)
hipDeviceAttributeHdpRegFlushCntl = hipDeviceAttribute_t.define('hipDeviceAttributeHdpRegFlushCntl', 10006)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc', 10007)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim', 10008)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim', 10009)
hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = hipDeviceAttribute_t.define('hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem', 10010)
hipDeviceAttributeIsLargeBar = hipDeviceAttribute_t.define('hipDeviceAttributeIsLargeBar', 10011)
hipDeviceAttributeAsicRevision = hipDeviceAttribute_t.define('hipDeviceAttributeAsicRevision', 10012)
hipDeviceAttributeCanUseStreamWaitValue = hipDeviceAttribute_t.define('hipDeviceAttributeCanUseStreamWaitValue', 10013)
hipDeviceAttributeImageSupport = hipDeviceAttribute_t.define('hipDeviceAttributeImageSupport', 10014)
hipDeviceAttributePhysicalMultiProcessorCount = hipDeviceAttribute_t.define('hipDeviceAttributePhysicalMultiProcessorCount', 10015)
hipDeviceAttributeFineGrainSupport = hipDeviceAttribute_t.define('hipDeviceAttributeFineGrainSupport', 10016)
hipDeviceAttributeWallClockRate = hipDeviceAttribute_t.define('hipDeviceAttributeWallClockRate', 10017)
hipDeviceAttributeAmdSpecificEnd = hipDeviceAttribute_t.define('hipDeviceAttributeAmdSpecificEnd', 19999)
hipDeviceAttributeVendorSpecificBegin = hipDeviceAttribute_t.define('hipDeviceAttributeVendorSpecificBegin', 20000)

hipDriverProcAddressQueryResult = CEnum(ctypes.c_uint32)
HIP_GET_PROC_ADDRESS_SUCCESS = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_SUCCESS', 0)
HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', 1)
HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT', 2)

hipComputeMode = CEnum(ctypes.c_uint32)
hipComputeModeDefault = hipComputeMode.define('hipComputeModeDefault', 0)
hipComputeModeExclusive = hipComputeMode.define('hipComputeModeExclusive', 1)
hipComputeModeProhibited = hipComputeMode.define('hipComputeModeProhibited', 2)
hipComputeModeExclusiveProcess = hipComputeMode.define('hipComputeModeExclusiveProcess', 3)

hipFlushGPUDirectRDMAWritesOptions = CEnum(ctypes.c_uint32)
hipFlushGPUDirectRDMAWritesOptionHost = hipFlushGPUDirectRDMAWritesOptions.define('hipFlushGPUDirectRDMAWritesOptionHost', 1)
hipFlushGPUDirectRDMAWritesOptionMemOps = hipFlushGPUDirectRDMAWritesOptions.define('hipFlushGPUDirectRDMAWritesOptionMemOps', 2)

hipGPUDirectRDMAWritesOrdering = CEnum(ctypes.c_uint32)
hipGPUDirectRDMAWritesOrderingNone = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingNone', 0)
hipGPUDirectRDMAWritesOrderingOwner = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingOwner', 100)
hipGPUDirectRDMAWritesOrderingAllDevices = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingAllDevices', 200)

try: (hip_init:=dll.hip_init).restype, hip_init.argtypes = hipError_t, []
except AttributeError: pass

hipDeviceptr_t = ctypes.c_void_p
hipChannelFormatKind = CEnum(ctypes.c_uint32)
hipChannelFormatKindSigned = hipChannelFormatKind.define('hipChannelFormatKindSigned', 0)
hipChannelFormatKindUnsigned = hipChannelFormatKind.define('hipChannelFormatKindUnsigned', 1)
hipChannelFormatKindFloat = hipChannelFormatKind.define('hipChannelFormatKindFloat', 2)
hipChannelFormatKindNone = hipChannelFormatKind.define('hipChannelFormatKindNone', 3)

class hipChannelFormatDesc(Struct): pass
hipChannelFormatDesc._fields_ = [
  ('x', ctypes.c_int32),
  ('y', ctypes.c_int32),
  ('z', ctypes.c_int32),
  ('w', ctypes.c_int32),
  ('f', hipChannelFormatKind),
]
class hipArray(Struct): pass
hipArray_t = ctypes.POINTER(hipArray)
hipArray_const_t = ctypes.POINTER(hipArray)
hipArray_Format = CEnum(ctypes.c_uint32)
HIP_AD_FORMAT_UNSIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT8', 1)
HIP_AD_FORMAT_UNSIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT16', 2)
HIP_AD_FORMAT_UNSIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT32', 3)
HIP_AD_FORMAT_SIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT8', 8)
HIP_AD_FORMAT_SIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT16', 9)
HIP_AD_FORMAT_SIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT32', 10)
HIP_AD_FORMAT_HALF = hipArray_Format.define('HIP_AD_FORMAT_HALF', 16)
HIP_AD_FORMAT_FLOAT = hipArray_Format.define('HIP_AD_FORMAT_FLOAT', 32)

class HIP_ARRAY_DESCRIPTOR(Struct): pass
class HIP_ARRAY3D_DESCRIPTOR(Struct): pass
class hip_Memcpy2D(Struct): pass
class hipMipmappedArray(Struct): pass
hipMipmappedArray._fields_ = [
  ('data', ctypes.c_void_p),
  ('desc', hipChannelFormatDesc),
  ('type', ctypes.c_uint32),
  ('width', ctypes.c_uint32),
  ('height', ctypes.c_uint32),
  ('depth', ctypes.c_uint32),
  ('min_mipmap_level', ctypes.c_uint32),
  ('max_mipmap_level', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('format', hipArray_Format),
  ('num_channels', ctypes.c_uint32),
]
hipMipmappedArray_t = ctypes.POINTER(hipMipmappedArray)
hipmipmappedArray = ctypes.POINTER(hipMipmappedArray)
hipMipmappedArray_const_t = ctypes.POINTER(hipMipmappedArray)
hipResourceType = CEnum(ctypes.c_uint32)
hipResourceTypeArray = hipResourceType.define('hipResourceTypeArray', 0)
hipResourceTypeMipmappedArray = hipResourceType.define('hipResourceTypeMipmappedArray', 1)
hipResourceTypeLinear = hipResourceType.define('hipResourceTypeLinear', 2)
hipResourceTypePitch2D = hipResourceType.define('hipResourceTypePitch2D', 3)

HIPresourcetype_enum = CEnum(ctypes.c_uint32)
HIP_RESOURCE_TYPE_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_ARRAY', 0)
HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
HIP_RESOURCE_TYPE_LINEAR = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_LINEAR', 2)
HIP_RESOURCE_TYPE_PITCH2D = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_PITCH2D', 3)

HIPresourcetype = HIPresourcetype_enum
hipResourcetype = HIPresourcetype_enum
HIPaddress_mode_enum = CEnum(ctypes.c_uint32)
HIP_TR_ADDRESS_MODE_WRAP = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_WRAP', 0)
HIP_TR_ADDRESS_MODE_CLAMP = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_CLAMP', 1)
HIP_TR_ADDRESS_MODE_MIRROR = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_MIRROR', 2)
HIP_TR_ADDRESS_MODE_BORDER = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_BORDER', 3)

HIPaddress_mode = HIPaddress_mode_enum
HIPfilter_mode_enum = CEnum(ctypes.c_uint32)
HIP_TR_FILTER_MODE_POINT = HIPfilter_mode_enum.define('HIP_TR_FILTER_MODE_POINT', 0)
HIP_TR_FILTER_MODE_LINEAR = HIPfilter_mode_enum.define('HIP_TR_FILTER_MODE_LINEAR', 1)

HIPfilter_mode = HIPfilter_mode_enum
class HIP_TEXTURE_DESC_st(Struct): pass
HIP_TEXTURE_DESC_st._fields_ = [
  ('addressMode', (HIPaddress_mode * 3)),
  ('filterMode', HIPfilter_mode),
  ('flags', ctypes.c_uint32),
  ('maxAnisotropy', ctypes.c_uint32),
  ('mipmapFilterMode', HIPfilter_mode),
  ('mipmapLevelBias', ctypes.c_float),
  ('minMipmapLevelClamp', ctypes.c_float),
  ('maxMipmapLevelClamp', ctypes.c_float),
  ('borderColor', (ctypes.c_float * 4)),
  ('reserved', (ctypes.c_int32 * 12)),
]
HIP_TEXTURE_DESC = HIP_TEXTURE_DESC_st
hipResourceViewFormat = CEnum(ctypes.c_uint32)
hipResViewFormatNone = hipResourceViewFormat.define('hipResViewFormatNone', 0)
hipResViewFormatUnsignedChar1 = hipResourceViewFormat.define('hipResViewFormatUnsignedChar1', 1)
hipResViewFormatUnsignedChar2 = hipResourceViewFormat.define('hipResViewFormatUnsignedChar2', 2)
hipResViewFormatUnsignedChar4 = hipResourceViewFormat.define('hipResViewFormatUnsignedChar4', 3)
hipResViewFormatSignedChar1 = hipResourceViewFormat.define('hipResViewFormatSignedChar1', 4)
hipResViewFormatSignedChar2 = hipResourceViewFormat.define('hipResViewFormatSignedChar2', 5)
hipResViewFormatSignedChar4 = hipResourceViewFormat.define('hipResViewFormatSignedChar4', 6)
hipResViewFormatUnsignedShort1 = hipResourceViewFormat.define('hipResViewFormatUnsignedShort1', 7)
hipResViewFormatUnsignedShort2 = hipResourceViewFormat.define('hipResViewFormatUnsignedShort2', 8)
hipResViewFormatUnsignedShort4 = hipResourceViewFormat.define('hipResViewFormatUnsignedShort4', 9)
hipResViewFormatSignedShort1 = hipResourceViewFormat.define('hipResViewFormatSignedShort1', 10)
hipResViewFormatSignedShort2 = hipResourceViewFormat.define('hipResViewFormatSignedShort2', 11)
hipResViewFormatSignedShort4 = hipResourceViewFormat.define('hipResViewFormatSignedShort4', 12)
hipResViewFormatUnsignedInt1 = hipResourceViewFormat.define('hipResViewFormatUnsignedInt1', 13)
hipResViewFormatUnsignedInt2 = hipResourceViewFormat.define('hipResViewFormatUnsignedInt2', 14)
hipResViewFormatUnsignedInt4 = hipResourceViewFormat.define('hipResViewFormatUnsignedInt4', 15)
hipResViewFormatSignedInt1 = hipResourceViewFormat.define('hipResViewFormatSignedInt1', 16)
hipResViewFormatSignedInt2 = hipResourceViewFormat.define('hipResViewFormatSignedInt2', 17)
hipResViewFormatSignedInt4 = hipResourceViewFormat.define('hipResViewFormatSignedInt4', 18)
hipResViewFormatHalf1 = hipResourceViewFormat.define('hipResViewFormatHalf1', 19)
hipResViewFormatHalf2 = hipResourceViewFormat.define('hipResViewFormatHalf2', 20)
hipResViewFormatHalf4 = hipResourceViewFormat.define('hipResViewFormatHalf4', 21)
hipResViewFormatFloat1 = hipResourceViewFormat.define('hipResViewFormatFloat1', 22)
hipResViewFormatFloat2 = hipResourceViewFormat.define('hipResViewFormatFloat2', 23)
hipResViewFormatFloat4 = hipResourceViewFormat.define('hipResViewFormatFloat4', 24)
hipResViewFormatUnsignedBlockCompressed1 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed1', 25)
hipResViewFormatUnsignedBlockCompressed2 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed2', 26)
hipResViewFormatUnsignedBlockCompressed3 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed3', 27)
hipResViewFormatUnsignedBlockCompressed4 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed4', 28)
hipResViewFormatSignedBlockCompressed4 = hipResourceViewFormat.define('hipResViewFormatSignedBlockCompressed4', 29)
hipResViewFormatUnsignedBlockCompressed5 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed5', 30)
hipResViewFormatSignedBlockCompressed5 = hipResourceViewFormat.define('hipResViewFormatSignedBlockCompressed5', 31)
hipResViewFormatUnsignedBlockCompressed6H = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed6H', 32)
hipResViewFormatSignedBlockCompressed6H = hipResourceViewFormat.define('hipResViewFormatSignedBlockCompressed6H', 33)
hipResViewFormatUnsignedBlockCompressed7 = hipResourceViewFormat.define('hipResViewFormatUnsignedBlockCompressed7', 34)

HIPresourceViewFormat_enum = CEnum(ctypes.c_uint32)
HIP_RES_VIEW_FORMAT_NONE = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_NONE', 0)
HIP_RES_VIEW_FORMAT_UINT_1X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_1X8', 1)
HIP_RES_VIEW_FORMAT_UINT_2X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_2X8', 2)
HIP_RES_VIEW_FORMAT_UINT_4X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_4X8', 3)
HIP_RES_VIEW_FORMAT_SINT_1X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_1X8', 4)
HIP_RES_VIEW_FORMAT_SINT_2X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_2X8', 5)
HIP_RES_VIEW_FORMAT_SINT_4X8 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_4X8', 6)
HIP_RES_VIEW_FORMAT_UINT_1X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_1X16', 7)
HIP_RES_VIEW_FORMAT_UINT_2X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_2X16', 8)
HIP_RES_VIEW_FORMAT_UINT_4X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_4X16', 9)
HIP_RES_VIEW_FORMAT_SINT_1X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_1X16', 10)
HIP_RES_VIEW_FORMAT_SINT_2X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_2X16', 11)
HIP_RES_VIEW_FORMAT_SINT_4X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_4X16', 12)
HIP_RES_VIEW_FORMAT_UINT_1X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_1X32', 13)
HIP_RES_VIEW_FORMAT_UINT_2X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_2X32', 14)
HIP_RES_VIEW_FORMAT_UINT_4X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UINT_4X32', 15)
HIP_RES_VIEW_FORMAT_SINT_1X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_1X32', 16)
HIP_RES_VIEW_FORMAT_SINT_2X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_2X32', 17)
HIP_RES_VIEW_FORMAT_SINT_4X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SINT_4X32', 18)
HIP_RES_VIEW_FORMAT_FLOAT_1X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_1X16', 19)
HIP_RES_VIEW_FORMAT_FLOAT_2X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_2X16', 20)
HIP_RES_VIEW_FORMAT_FLOAT_4X16 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_4X16', 21)
HIP_RES_VIEW_FORMAT_FLOAT_1X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_1X32', 22)
HIP_RES_VIEW_FORMAT_FLOAT_2X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_2X32', 23)
HIP_RES_VIEW_FORMAT_FLOAT_4X32 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_FLOAT_4X32', 24)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC1', 25)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC2', 26)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC3', 27)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC4', 28)
HIP_RES_VIEW_FORMAT_SIGNED_BC4 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SIGNED_BC4', 29)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC5', 30)
HIP_RES_VIEW_FORMAT_SIGNED_BC5 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SIGNED_BC5', 31)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H', 32)
HIP_RES_VIEW_FORMAT_SIGNED_BC6H = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_SIGNED_BC6H', 33)
HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = HIPresourceViewFormat_enum.define('HIP_RES_VIEW_FORMAT_UNSIGNED_BC7', 34)

HIPresourceViewFormat = HIPresourceViewFormat_enum
class hipResourceDesc(Struct): pass
class HIP_RESOURCE_DESC_st(Struct): pass
HIP_RESOURCE_DESC = HIP_RESOURCE_DESC_st
class hipResourceViewDesc(Struct): pass
class HIP_RESOURCE_VIEW_DESC_st(Struct): pass
HIP_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC_st
hipMemcpyKind = CEnum(ctypes.c_uint32)
hipMemcpyHostToHost = hipMemcpyKind.define('hipMemcpyHostToHost', 0)
hipMemcpyHostToDevice = hipMemcpyKind.define('hipMemcpyHostToDevice', 1)
hipMemcpyDeviceToHost = hipMemcpyKind.define('hipMemcpyDeviceToHost', 2)
hipMemcpyDeviceToDevice = hipMemcpyKind.define('hipMemcpyDeviceToDevice', 3)
hipMemcpyDefault = hipMemcpyKind.define('hipMemcpyDefault', 4)
hipMemcpyDeviceToDeviceNoCU = hipMemcpyKind.define('hipMemcpyDeviceToDeviceNoCU', 1024)

class hipPitchedPtr(Struct): pass
class hipExtent(Struct): pass
class hipPos(Struct): pass
class hipMemcpy3DParms(Struct): pass
class HIP_MEMCPY3D(Struct): pass
hipFunction_attribute = CEnum(ctypes.c_uint32)
HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK', 0)
HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', 1)
HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES', 2)
HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 3)
HIP_FUNC_ATTRIBUTE_NUM_REGS = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_NUM_REGS', 4)
HIP_FUNC_ATTRIBUTE_PTX_VERSION = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_PTX_VERSION', 5)
HIP_FUNC_ATTRIBUTE_BINARY_VERSION = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_BINARY_VERSION', 6)
HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA', 7)
HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES', 8)
HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT', 9)
HIP_FUNC_ATTRIBUTE_MAX = hipFunction_attribute.define('HIP_FUNC_ATTRIBUTE_MAX', 10)

hipPointer_attribute = CEnum(ctypes.c_uint32)
HIP_POINTER_ATTRIBUTE_CONTEXT = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_CONTEXT', 1)
HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_MEMORY_TYPE', 2)
HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_DEVICE_POINTER', 3)
HIP_POINTER_ATTRIBUTE_HOST_POINTER = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_HOST_POINTER', 4)
HIP_POINTER_ATTRIBUTE_P2P_TOKENS = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_P2P_TOKENS', 5)
HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS', 6)
HIP_POINTER_ATTRIBUTE_BUFFER_ID = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_BUFFER_ID', 7)
HIP_POINTER_ATTRIBUTE_IS_MANAGED = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_IS_MANAGED', 8)
HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL', 9)
HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE', 10)
HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR', 11)
HIP_POINTER_ATTRIBUTE_RANGE_SIZE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_RANGE_SIZE', 12)
HIP_POINTER_ATTRIBUTE_MAPPED = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_MAPPED', 13)
HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES', 14)
HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE', 15)
HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS', 16)
HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = hipPointer_attribute.define('HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE', 17)

hipDeviceProp_t = hipDeviceProp_tR0600
GENERIC_GRID_LAUNCH = 1
DEPRECATED = lambda msg: __attribute__ ((deprecated(msg)))
hipIpcMemLazyEnablePeerAccess = 0x01
HIP_IPC_HANDLE_SIZE = 64
hipStreamDefault = 0x00
hipStreamNonBlocking = 0x01
hipEventDefault = 0x0
hipEventBlockingSync = 0x1
hipEventDisableTiming = 0x2
hipEventInterprocess = 0x4
hipEventDisableSystemFence = 0x20000000
hipEventReleaseToDevice = 0x40000000
hipEventReleaseToSystem = 0x80000000
hipHostMallocDefault = 0x0
hipHostMallocPortable = 0x1
hipHostMallocMapped = 0x2
hipHostMallocWriteCombined = 0x4
hipHostMallocNumaUser = 0x20000000
hipHostMallocCoherent = 0x40000000
hipHostMallocNonCoherent = 0x80000000
hipMemAttachGlobal = 0x01
hipMemAttachHost = 0x02
hipMemAttachSingle = 0x04
hipDeviceMallocDefault = 0x0
hipDeviceMallocFinegrained = 0x1
hipMallocSignalMemory = 0x2
hipDeviceMallocUncached = 0x3
hipDeviceMallocContiguous = 0x4
hipHostRegisterDefault = 0x0
hipHostRegisterPortable = 0x1
hipHostRegisterMapped = 0x2
hipHostRegisterIoMemory = 0x4
hipHostRegisterReadOnly = 0x08
hipExtHostRegisterCoarseGrained = 0x8
hipDeviceScheduleAuto = 0x0
hipDeviceScheduleSpin = 0x1
hipDeviceScheduleYield = 0x2
hipDeviceScheduleBlockingSync = 0x4
hipDeviceScheduleMask = 0x7
hipDeviceMapHost = 0x8
hipDeviceLmemResizeToMax = 0x10
hipArrayDefault = 0x00
hipArrayLayered = 0x01
hipArraySurfaceLoadStore = 0x02
hipArrayCubemap = 0x04
hipArrayTextureGather = 0x08
hipOccupancyDefault = 0x00
hipOccupancyDisableCachingOverride = 0x01
hipCooperativeLaunchMultiDeviceNoPreSync = 0x01
hipCooperativeLaunchMultiDeviceNoPostSync = 0x02
hipExtAnyOrderLaunch = 0x01
hipStreamWaitValueGte = 0x0
hipStreamWaitValueEq = 0x1
hipStreamWaitValueAnd = 0x2
hipStreamWaitValueNor = 0x3
hipExternalMemoryDedicated = 0x1
hipGraphKernelNodePortDefault = 0
hipGraphKernelNodePortLaunchCompletion = 2
hipGraphKernelNodePortProgrammatic = 1
USE_PEER_NON_UNIFIED = 1
HIP_TRSA_OVERRIDE_FORMAT = 0x01
HIP_TRSF_READ_AS_INTEGER = 0x01
HIP_TRSF_NORMALIZED_COORDINATES = 0x02
HIP_TRSF_SRGB = 0x10