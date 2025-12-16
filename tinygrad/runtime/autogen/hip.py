# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
import os
dll = DLL('hip', os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
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

class ihipModuleSymbol_t(Struct): pass
hipFunction_t = Pointer(ihipModuleSymbol_t)
uint32_t = ctypes.c_uint32
size_t = ctypes.c_uint64
class ihipStream_t(Struct): pass
hipStream_t = Pointer(ihipStream_t)
class ihipEvent_t(Struct): pass
hipEvent_t = Pointer(ihipEvent_t)
try: (hipExtModuleLaunchKernel:=dll.hipExtModuleLaunchKernel).restype, hipExtModuleLaunchKernel.argtypes = hipError_t, [hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p), hipEvent_t, hipEvent_t, uint32_t]
except AttributeError: pass

try: (hipHccModuleLaunchKernel:=dll.hipHccModuleLaunchKernel).restype, hipHccModuleLaunchKernel.argtypes = hipError_t, [hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p), hipEvent_t, hipEvent_t]
except AttributeError: pass

class dim3(Struct): pass
dim3.SIZE = 12
dim3._fields_ = ['x', 'y', 'z']
setattr(dim3, 'x', field(0, uint32_t))
setattr(dim3, 'y', field(4, uint32_t))
setattr(dim3, 'z', field(8, uint32_t))
try: (hipExtLaunchKernel:=dll.hipExtLaunchKernel).restype, hipExtLaunchKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32]
except AttributeError: pass

hiprtcResult = CEnum(ctypes.c_uint32)
HIPRTC_SUCCESS = hiprtcResult.define('HIPRTC_SUCCESS', 0)
HIPRTC_ERROR_OUT_OF_MEMORY = hiprtcResult.define('HIPRTC_ERROR_OUT_OF_MEMORY', 1)
HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = hiprtcResult.define('HIPRTC_ERROR_PROGRAM_CREATION_FAILURE', 2)
HIPRTC_ERROR_INVALID_INPUT = hiprtcResult.define('HIPRTC_ERROR_INVALID_INPUT', 3)
HIPRTC_ERROR_INVALID_PROGRAM = hiprtcResult.define('HIPRTC_ERROR_INVALID_PROGRAM', 4)
HIPRTC_ERROR_INVALID_OPTION = hiprtcResult.define('HIPRTC_ERROR_INVALID_OPTION', 5)
HIPRTC_ERROR_COMPILATION = hiprtcResult.define('HIPRTC_ERROR_COMPILATION', 6)
HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = hiprtcResult.define('HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE', 7)
HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = hiprtcResult.define('HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION', 8)
HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = hiprtcResult.define('HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION', 9)
HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = hiprtcResult.define('HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID', 10)
HIPRTC_ERROR_INTERNAL_ERROR = hiprtcResult.define('HIPRTC_ERROR_INTERNAL_ERROR', 11)
HIPRTC_ERROR_LINKING = hiprtcResult.define('HIPRTC_ERROR_LINKING', 100)

hiprtcJIT_option = CEnum(ctypes.c_uint32)
HIPRTC_JIT_MAX_REGISTERS = hiprtcJIT_option.define('HIPRTC_JIT_MAX_REGISTERS', 0)
HIPRTC_JIT_THREADS_PER_BLOCK = hiprtcJIT_option.define('HIPRTC_JIT_THREADS_PER_BLOCK', 1)
HIPRTC_JIT_WALL_TIME = hiprtcJIT_option.define('HIPRTC_JIT_WALL_TIME', 2)
HIPRTC_JIT_INFO_LOG_BUFFER = hiprtcJIT_option.define('HIPRTC_JIT_INFO_LOG_BUFFER', 3)
HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = hiprtcJIT_option.define('HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES', 4)
HIPRTC_JIT_ERROR_LOG_BUFFER = hiprtcJIT_option.define('HIPRTC_JIT_ERROR_LOG_BUFFER', 5)
HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = hiprtcJIT_option.define('HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES', 6)
HIPRTC_JIT_OPTIMIZATION_LEVEL = hiprtcJIT_option.define('HIPRTC_JIT_OPTIMIZATION_LEVEL', 7)
HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = hiprtcJIT_option.define('HIPRTC_JIT_TARGET_FROM_HIPCONTEXT', 8)
HIPRTC_JIT_TARGET = hiprtcJIT_option.define('HIPRTC_JIT_TARGET', 9)
HIPRTC_JIT_FALLBACK_STRATEGY = hiprtcJIT_option.define('HIPRTC_JIT_FALLBACK_STRATEGY', 10)
HIPRTC_JIT_GENERATE_DEBUG_INFO = hiprtcJIT_option.define('HIPRTC_JIT_GENERATE_DEBUG_INFO', 11)
HIPRTC_JIT_LOG_VERBOSE = hiprtcJIT_option.define('HIPRTC_JIT_LOG_VERBOSE', 12)
HIPRTC_JIT_GENERATE_LINE_INFO = hiprtcJIT_option.define('HIPRTC_JIT_GENERATE_LINE_INFO', 13)
HIPRTC_JIT_CACHE_MODE = hiprtcJIT_option.define('HIPRTC_JIT_CACHE_MODE', 14)
HIPRTC_JIT_NEW_SM3X_OPT = hiprtcJIT_option.define('HIPRTC_JIT_NEW_SM3X_OPT', 15)
HIPRTC_JIT_FAST_COMPILE = hiprtcJIT_option.define('HIPRTC_JIT_FAST_COMPILE', 16)
HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = hiprtcJIT_option.define('HIPRTC_JIT_GLOBAL_SYMBOL_NAMES', 17)
HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = hiprtcJIT_option.define('HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS', 18)
HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = hiprtcJIT_option.define('HIPRTC_JIT_GLOBAL_SYMBOL_COUNT', 19)
HIPRTC_JIT_LTO = hiprtcJIT_option.define('HIPRTC_JIT_LTO', 20)
HIPRTC_JIT_FTZ = hiprtcJIT_option.define('HIPRTC_JIT_FTZ', 21)
HIPRTC_JIT_PREC_DIV = hiprtcJIT_option.define('HIPRTC_JIT_PREC_DIV', 22)
HIPRTC_JIT_PREC_SQRT = hiprtcJIT_option.define('HIPRTC_JIT_PREC_SQRT', 23)
HIPRTC_JIT_FMA = hiprtcJIT_option.define('HIPRTC_JIT_FMA', 24)
HIPRTC_JIT_NUM_OPTIONS = hiprtcJIT_option.define('HIPRTC_JIT_NUM_OPTIONS', 25)
HIPRTC_JIT_IR_TO_ISA_OPT_EXT = hiprtcJIT_option.define('HIPRTC_JIT_IR_TO_ISA_OPT_EXT', 10000)
HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT = hiprtcJIT_option.define('HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT', 10001)

hiprtcJITInputType = CEnum(ctypes.c_uint32)
HIPRTC_JIT_INPUT_CUBIN = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_CUBIN', 0)
HIPRTC_JIT_INPUT_PTX = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_PTX', 1)
HIPRTC_JIT_INPUT_FATBINARY = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_FATBINARY', 2)
HIPRTC_JIT_INPUT_OBJECT = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_OBJECT', 3)
HIPRTC_JIT_INPUT_LIBRARY = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LIBRARY', 4)
HIPRTC_JIT_INPUT_NVVM = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_NVVM', 5)
HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hiprtcJITInputType.define('HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES', 6)
HIPRTC_JIT_INPUT_LLVM_BITCODE = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LLVM_BITCODE', 100)
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE', 101)
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = hiprtcJITInputType.define('HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE', 102)
HIPRTC_JIT_NUM_INPUT_TYPES = hiprtcJITInputType.define('HIPRTC_JIT_NUM_INPUT_TYPES', 9)

class ihiprtcLinkState(Struct): pass
hiprtcLinkState = Pointer(ihiprtcLinkState)
try: (hiprtcGetErrorString:=dll.hiprtcGetErrorString).restype, hiprtcGetErrorString.argtypes = Pointer(ctypes.c_char), [hiprtcResult]
except AttributeError: pass

try: (hiprtcVersion:=dll.hiprtcVersion).restype, hiprtcVersion.argtypes = hiprtcResult, [Pointer(ctypes.c_int32), Pointer(ctypes.c_int32)]
except AttributeError: pass

class _hiprtcProgram(Struct): pass
hiprtcProgram = Pointer(_hiprtcProgram)
try: (hiprtcAddNameExpression:=dll.hiprtcAddNameExpression).restype, hiprtcAddNameExpression.argtypes = hiprtcResult, [hiprtcProgram, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hiprtcCompileProgram:=dll.hiprtcCompileProgram).restype, hiprtcCompileProgram.argtypes = hiprtcResult, [hiprtcProgram, ctypes.c_int32, Pointer(Pointer(ctypes.c_char))]
except AttributeError: pass

try: (hiprtcCreateProgram:=dll.hiprtcCreateProgram).restype, hiprtcCreateProgram.argtypes = hiprtcResult, [Pointer(hiprtcProgram), Pointer(ctypes.c_char), Pointer(ctypes.c_char), ctypes.c_int32, Pointer(Pointer(ctypes.c_char)), Pointer(Pointer(ctypes.c_char))]
except AttributeError: pass

try: (hiprtcDestroyProgram:=dll.hiprtcDestroyProgram).restype, hiprtcDestroyProgram.argtypes = hiprtcResult, [Pointer(hiprtcProgram)]
except AttributeError: pass

try: (hiprtcGetLoweredName:=dll.hiprtcGetLoweredName).restype, hiprtcGetLoweredName.argtypes = hiprtcResult, [hiprtcProgram, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char))]
except AttributeError: pass

try: (hiprtcGetProgramLog:=dll.hiprtcGetProgramLog).restype, hiprtcGetProgramLog.argtypes = hiprtcResult, [hiprtcProgram, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hiprtcGetProgramLogSize:=dll.hiprtcGetProgramLogSize).restype, hiprtcGetProgramLogSize.argtypes = hiprtcResult, [hiprtcProgram, Pointer(size_t)]
except AttributeError: pass

try: (hiprtcGetCode:=dll.hiprtcGetCode).restype, hiprtcGetCode.argtypes = hiprtcResult, [hiprtcProgram, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hiprtcGetCodeSize:=dll.hiprtcGetCodeSize).restype, hiprtcGetCodeSize.argtypes = hiprtcResult, [hiprtcProgram, Pointer(size_t)]
except AttributeError: pass

try: (hiprtcGetBitcode:=dll.hiprtcGetBitcode).restype, hiprtcGetBitcode.argtypes = hiprtcResult, [hiprtcProgram, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hiprtcGetBitcodeSize:=dll.hiprtcGetBitcodeSize).restype, hiprtcGetBitcodeSize.argtypes = hiprtcResult, [hiprtcProgram, Pointer(size_t)]
except AttributeError: pass

try: (hiprtcLinkCreate:=dll.hiprtcLinkCreate).restype, hiprtcLinkCreate.argtypes = hiprtcResult, [ctypes.c_uint32, Pointer(hiprtcJIT_option), Pointer(ctypes.c_void_p), Pointer(hiprtcLinkState)]
except AttributeError: pass

try: (hiprtcLinkAddFile:=dll.hiprtcLinkAddFile).restype, hiprtcLinkAddFile.argtypes = hiprtcResult, [hiprtcLinkState, hiprtcJITInputType, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(hiprtcJIT_option), Pointer(ctypes.c_void_p)]
except AttributeError: pass

try: (hiprtcLinkAddData:=dll.hiprtcLinkAddData).restype, hiprtcLinkAddData.argtypes = hiprtcResult, [hiprtcLinkState, hiprtcJITInputType, ctypes.c_void_p, size_t, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(hiprtcJIT_option), Pointer(ctypes.c_void_p)]
except AttributeError: pass

try: (hiprtcLinkComplete:=dll.hiprtcLinkComplete).restype, hiprtcLinkComplete.argtypes = hiprtcResult, [hiprtcLinkState, Pointer(ctypes.c_void_p), Pointer(size_t)]
except AttributeError: pass

try: (hiprtcLinkDestroy:=dll.hiprtcLinkDestroy).restype, hiprtcLinkDestroy.argtypes = hiprtcResult, [hiprtcLinkState]
except AttributeError: pass

_anonenum0 = CEnum(ctypes.c_uint32)
HIP_SUCCESS = _anonenum0.define('HIP_SUCCESS', 0)
HIP_ERROR_INVALID_VALUE = _anonenum0.define('HIP_ERROR_INVALID_VALUE', 1)
HIP_ERROR_NOT_INITIALIZED = _anonenum0.define('HIP_ERROR_NOT_INITIALIZED', 2)
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = _anonenum0.define('HIP_ERROR_LAUNCH_OUT_OF_RESOURCES', 3)

class hipDeviceArch_t(Struct): pass
hipDeviceArch_t.SIZE = 4
hipDeviceArch_t._fields_ = ['hasGlobalInt32Atomics', 'hasGlobalFloatAtomicExch', 'hasSharedInt32Atomics', 'hasSharedFloatAtomicExch', 'hasFloatAtomicAdd', 'hasGlobalInt64Atomics', 'hasSharedInt64Atomics', 'hasDoubles', 'hasWarpVote', 'hasWarpBallot', 'hasWarpShuffle', 'hasFunnelShift', 'hasThreadFenceSystem', 'hasSyncThreadsExt', 'hasSurfaceFuncs', 'has3dGrid', 'hasDynamicParallelism']
setattr(hipDeviceArch_t, 'hasGlobalInt32Atomics', field(0, ctypes.c_uint32, 1, 0))
setattr(hipDeviceArch_t, 'hasGlobalFloatAtomicExch', field(0, ctypes.c_uint32, 1, 1))
setattr(hipDeviceArch_t, 'hasSharedInt32Atomics', field(0, ctypes.c_uint32, 1, 2))
setattr(hipDeviceArch_t, 'hasSharedFloatAtomicExch', field(0, ctypes.c_uint32, 1, 3))
setattr(hipDeviceArch_t, 'hasFloatAtomicAdd', field(0, ctypes.c_uint32, 1, 4))
setattr(hipDeviceArch_t, 'hasGlobalInt64Atomics', field(0, ctypes.c_uint32, 1, 5))
setattr(hipDeviceArch_t, 'hasSharedInt64Atomics', field(0, ctypes.c_uint32, 1, 6))
setattr(hipDeviceArch_t, 'hasDoubles', field(0, ctypes.c_uint32, 1, 7))
setattr(hipDeviceArch_t, 'hasWarpVote', field(1, ctypes.c_uint32, 1, 0))
setattr(hipDeviceArch_t, 'hasWarpBallot', field(1, ctypes.c_uint32, 1, 1))
setattr(hipDeviceArch_t, 'hasWarpShuffle', field(1, ctypes.c_uint32, 1, 2))
setattr(hipDeviceArch_t, 'hasFunnelShift', field(1, ctypes.c_uint32, 1, 3))
setattr(hipDeviceArch_t, 'hasThreadFenceSystem', field(1, ctypes.c_uint32, 1, 4))
setattr(hipDeviceArch_t, 'hasSyncThreadsExt', field(1, ctypes.c_uint32, 1, 5))
setattr(hipDeviceArch_t, 'hasSurfaceFuncs', field(1, ctypes.c_uint32, 1, 6))
setattr(hipDeviceArch_t, 'has3dGrid', field(1, ctypes.c_uint32, 1, 7))
setattr(hipDeviceArch_t, 'hasDynamicParallelism', field(2, ctypes.c_uint32, 1, 0))
class hipUUID_t(Struct): pass
hipUUID_t.SIZE = 16
hipUUID_t._fields_ = ['bytes']
setattr(hipUUID_t, 'bytes', field(0, Array(ctypes.c_char, 16)))
hipUUID = hipUUID_t
class hipDeviceProp_tR0600(Struct): pass
hipDeviceProp_tR0600.SIZE = 1472
hipDeviceProp_tR0600._fields_ = ['name', 'uuid', 'luid', 'luidDeviceNodeMask', 'totalGlobalMem', 'sharedMemPerBlock', 'regsPerBlock', 'warpSize', 'memPitch', 'maxThreadsPerBlock', 'maxThreadsDim', 'maxGridSize', 'clockRate', 'totalConstMem', 'major', 'minor', 'textureAlignment', 'texturePitchAlignment', 'deviceOverlap', 'multiProcessorCount', 'kernelExecTimeoutEnabled', 'integrated', 'canMapHostMemory', 'computeMode', 'maxTexture1D', 'maxTexture1DMipmap', 'maxTexture1DLinear', 'maxTexture2D', 'maxTexture2DMipmap', 'maxTexture2DLinear', 'maxTexture2DGather', 'maxTexture3D', 'maxTexture3DAlt', 'maxTextureCubemap', 'maxTexture1DLayered', 'maxTexture2DLayered', 'maxTextureCubemapLayered', 'maxSurface1D', 'maxSurface2D', 'maxSurface3D', 'maxSurface1DLayered', 'maxSurface2DLayered', 'maxSurfaceCubemap', 'maxSurfaceCubemapLayered', 'surfaceAlignment', 'concurrentKernels', 'ECCEnabled', 'pciBusID', 'pciDeviceID', 'pciDomainID', 'tccDriver', 'asyncEngineCount', 'unifiedAddressing', 'memoryClockRate', 'memoryBusWidth', 'l2CacheSize', 'persistingL2CacheMaxSize', 'maxThreadsPerMultiProcessor', 'streamPrioritiesSupported', 'globalL1CacheSupported', 'localL1CacheSupported', 'sharedMemPerMultiprocessor', 'regsPerMultiprocessor', 'managedMemory', 'isMultiGpuBoard', 'multiGpuBoardGroupID', 'hostNativeAtomicSupported', 'singleToDoublePrecisionPerfRatio', 'pageableMemoryAccess', 'concurrentManagedAccess', 'computePreemptionSupported', 'canUseHostPointerForRegisteredMem', 'cooperativeLaunch', 'cooperativeMultiDeviceLaunch', 'sharedMemPerBlockOptin', 'pageableMemoryAccessUsesHostPageTables', 'directManagedMemAccessFromHost', 'maxBlocksPerMultiProcessor', 'accessPolicyMaxWindowSize', 'reservedSharedMemPerBlock', 'hostRegisterSupported', 'sparseHipArraySupported', 'hostRegisterReadOnlySupported', 'timelineSemaphoreInteropSupported', 'memoryPoolsSupported', 'gpuDirectRDMASupported', 'gpuDirectRDMAFlushWritesOptions', 'gpuDirectRDMAWritesOrdering', 'memoryPoolSupportedHandleTypes', 'deferredMappingHipArraySupported', 'ipcEventSupported', 'clusterLaunch', 'unifiedFunctionPointers', 'reserved', 'hipReserved', 'gcnArchName', 'maxSharedMemoryPerMultiProcessor', 'clockInstructionRate', 'arch', 'hdpMemFlushCntl', 'hdpRegFlushCntl', 'cooperativeMultiDeviceUnmatchedFunc', 'cooperativeMultiDeviceUnmatchedGridDim', 'cooperativeMultiDeviceUnmatchedBlockDim', 'cooperativeMultiDeviceUnmatchedSharedMem', 'isLargeBar', 'asicRevision']
setattr(hipDeviceProp_tR0600, 'name', field(0, Array(ctypes.c_char, 256)))
setattr(hipDeviceProp_tR0600, 'uuid', field(256, hipUUID))
setattr(hipDeviceProp_tR0600, 'luid', field(272, Array(ctypes.c_char, 8)))
setattr(hipDeviceProp_tR0600, 'luidDeviceNodeMask', field(280, ctypes.c_uint32))
setattr(hipDeviceProp_tR0600, 'totalGlobalMem', field(288, size_t))
setattr(hipDeviceProp_tR0600, 'sharedMemPerBlock', field(296, size_t))
setattr(hipDeviceProp_tR0600, 'regsPerBlock', field(304, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'warpSize', field(308, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'memPitch', field(312, size_t))
setattr(hipDeviceProp_tR0600, 'maxThreadsPerBlock', field(320, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxThreadsDim', field(324, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxGridSize', field(336, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'clockRate', field(348, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'totalConstMem', field(352, size_t))
setattr(hipDeviceProp_tR0600, 'major', field(360, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'minor', field(364, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'textureAlignment', field(368, size_t))
setattr(hipDeviceProp_tR0600, 'texturePitchAlignment', field(376, size_t))
setattr(hipDeviceProp_tR0600, 'deviceOverlap', field(384, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'multiProcessorCount', field(388, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'kernelExecTimeoutEnabled', field(392, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'integrated', field(396, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'canMapHostMemory', field(400, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'computeMode', field(404, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxTexture1D', field(408, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxTexture1DMipmap', field(412, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxTexture1DLinear', field(416, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxTexture2D', field(420, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxTexture2DMipmap', field(428, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxTexture2DLinear', field(436, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxTexture2DGather', field(448, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxTexture3D', field(456, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxTexture3DAlt', field(468, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxTextureCubemap', field(480, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxTexture1DLayered', field(484, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxTexture2DLayered', field(492, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxTextureCubemapLayered', field(504, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxSurface1D', field(512, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxSurface2D', field(516, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxSurface3D', field(524, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxSurface1DLayered', field(536, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'maxSurface2DLayered', field(544, Array(ctypes.c_int32, 3)))
setattr(hipDeviceProp_tR0600, 'maxSurfaceCubemap', field(556, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxSurfaceCubemapLayered', field(560, Array(ctypes.c_int32, 2)))
setattr(hipDeviceProp_tR0600, 'surfaceAlignment', field(568, size_t))
setattr(hipDeviceProp_tR0600, 'concurrentKernels', field(576, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'ECCEnabled', field(580, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'pciBusID', field(584, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'pciDeviceID', field(588, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'pciDomainID', field(592, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'tccDriver', field(596, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'asyncEngineCount', field(600, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'unifiedAddressing', field(604, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'memoryClockRate', field(608, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'memoryBusWidth', field(612, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'l2CacheSize', field(616, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'persistingL2CacheMaxSize', field(620, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxThreadsPerMultiProcessor', field(624, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'streamPrioritiesSupported', field(628, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'globalL1CacheSupported', field(632, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'localL1CacheSupported', field(636, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'sharedMemPerMultiprocessor', field(640, size_t))
setattr(hipDeviceProp_tR0600, 'regsPerMultiprocessor', field(648, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'managedMemory', field(652, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'isMultiGpuBoard', field(656, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'multiGpuBoardGroupID', field(660, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'hostNativeAtomicSupported', field(664, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'singleToDoublePrecisionPerfRatio', field(668, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'pageableMemoryAccess', field(672, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'concurrentManagedAccess', field(676, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'computePreemptionSupported', field(680, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'canUseHostPointerForRegisteredMem', field(684, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'cooperativeLaunch', field(688, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'cooperativeMultiDeviceLaunch', field(692, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'sharedMemPerBlockOptin', field(696, size_t))
setattr(hipDeviceProp_tR0600, 'pageableMemoryAccessUsesHostPageTables', field(704, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'directManagedMemAccessFromHost', field(708, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'maxBlocksPerMultiProcessor', field(712, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'accessPolicyMaxWindowSize', field(716, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'reservedSharedMemPerBlock', field(720, size_t))
setattr(hipDeviceProp_tR0600, 'hostRegisterSupported', field(728, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'sparseHipArraySupported', field(732, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'hostRegisterReadOnlySupported', field(736, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'timelineSemaphoreInteropSupported', field(740, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'memoryPoolsSupported', field(744, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'gpuDirectRDMASupported', field(748, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'gpuDirectRDMAFlushWritesOptions', field(752, ctypes.c_uint32))
setattr(hipDeviceProp_tR0600, 'gpuDirectRDMAWritesOrdering', field(756, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'memoryPoolSupportedHandleTypes', field(760, ctypes.c_uint32))
setattr(hipDeviceProp_tR0600, 'deferredMappingHipArraySupported', field(764, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'ipcEventSupported', field(768, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'clusterLaunch', field(772, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'unifiedFunctionPointers', field(776, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'reserved', field(780, Array(ctypes.c_int32, 63)))
setattr(hipDeviceProp_tR0600, 'hipReserved', field(1032, Array(ctypes.c_int32, 32)))
setattr(hipDeviceProp_tR0600, 'gcnArchName', field(1160, Array(ctypes.c_char, 256)))
setattr(hipDeviceProp_tR0600, 'maxSharedMemoryPerMultiProcessor', field(1416, size_t))
setattr(hipDeviceProp_tR0600, 'clockInstructionRate', field(1424, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'arch', field(1428, hipDeviceArch_t))
setattr(hipDeviceProp_tR0600, 'hdpMemFlushCntl', field(1432, Pointer(ctypes.c_uint32)))
setattr(hipDeviceProp_tR0600, 'hdpRegFlushCntl', field(1440, Pointer(ctypes.c_uint32)))
setattr(hipDeviceProp_tR0600, 'cooperativeMultiDeviceUnmatchedFunc', field(1448, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'cooperativeMultiDeviceUnmatchedGridDim', field(1452, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'cooperativeMultiDeviceUnmatchedBlockDim', field(1456, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'cooperativeMultiDeviceUnmatchedSharedMem', field(1460, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'isLargeBar', field(1464, ctypes.c_int32))
setattr(hipDeviceProp_tR0600, 'asicRevision', field(1468, ctypes.c_int32))
hipMemoryType = CEnum(ctypes.c_uint32)
hipMemoryTypeUnregistered = hipMemoryType.define('hipMemoryTypeUnregistered', 0)
hipMemoryTypeHost = hipMemoryType.define('hipMemoryTypeHost', 1)
hipMemoryTypeDevice = hipMemoryType.define('hipMemoryTypeDevice', 2)
hipMemoryTypeManaged = hipMemoryType.define('hipMemoryTypeManaged', 3)
hipMemoryTypeArray = hipMemoryType.define('hipMemoryTypeArray', 10)
hipMemoryTypeUnified = hipMemoryType.define('hipMemoryTypeUnified', 11)

class hipPointerAttribute_t(Struct): pass
hipPointerAttribute_t.SIZE = 32
hipPointerAttribute_t._fields_ = ['type', 'device', 'devicePointer', 'hostPointer', 'isManaged', 'allocationFlags']
setattr(hipPointerAttribute_t, 'type', field(0, hipMemoryType))
setattr(hipPointerAttribute_t, 'device', field(4, ctypes.c_int32))
setattr(hipPointerAttribute_t, 'devicePointer', field(8, ctypes.c_void_p))
setattr(hipPointerAttribute_t, 'hostPointer', field(16, ctypes.c_void_p))
setattr(hipPointerAttribute_t, 'isManaged', field(24, ctypes.c_int32))
setattr(hipPointerAttribute_t, 'allocationFlags', field(28, ctypes.c_uint32))
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

class ihipCtx_t(Struct): pass
hipCtx_t = Pointer(ihipCtx_t)
hipDevice_t = ctypes.c_int32
hipDeviceP2PAttr = CEnum(ctypes.c_uint32)
hipDevP2PAttrPerformanceRank = hipDeviceP2PAttr.define('hipDevP2PAttrPerformanceRank', 0)
hipDevP2PAttrAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrAccessSupported', 1)
hipDevP2PAttrNativeAtomicSupported = hipDeviceP2PAttr.define('hipDevP2PAttrNativeAtomicSupported', 2)
hipDevP2PAttrHipArrayAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrHipArrayAccessSupported', 3)

class hipIpcMemHandle_st(Struct): pass
hipIpcMemHandle_st.SIZE = 64
hipIpcMemHandle_st._fields_ = ['reserved']
setattr(hipIpcMemHandle_st, 'reserved', field(0, Array(ctypes.c_char, 64)))
hipIpcMemHandle_t = hipIpcMemHandle_st
class hipIpcEventHandle_st(Struct): pass
hipIpcEventHandle_st.SIZE = 64
hipIpcEventHandle_st._fields_ = ['reserved']
setattr(hipIpcEventHandle_st, 'reserved', field(0, Array(ctypes.c_char, 64)))
hipIpcEventHandle_t = hipIpcEventHandle_st
class ihipModule_t(Struct): pass
hipModule_t = Pointer(ihipModule_t)
class ihipMemPoolHandle_t(Struct): pass
hipMemPool_t = Pointer(ihipMemPoolHandle_t)
class hipFuncAttributes(Struct): pass
hipFuncAttributes.SIZE = 56
hipFuncAttributes._fields_ = ['binaryVersion', 'cacheModeCA', 'constSizeBytes', 'localSizeBytes', 'maxDynamicSharedSizeBytes', 'maxThreadsPerBlock', 'numRegs', 'preferredShmemCarveout', 'ptxVersion', 'sharedSizeBytes']
setattr(hipFuncAttributes, 'binaryVersion', field(0, ctypes.c_int32))
setattr(hipFuncAttributes, 'cacheModeCA', field(4, ctypes.c_int32))
setattr(hipFuncAttributes, 'constSizeBytes', field(8, size_t))
setattr(hipFuncAttributes, 'localSizeBytes', field(16, size_t))
setattr(hipFuncAttributes, 'maxDynamicSharedSizeBytes', field(24, ctypes.c_int32))
setattr(hipFuncAttributes, 'maxThreadsPerBlock', field(28, ctypes.c_int32))
setattr(hipFuncAttributes, 'numRegs', field(32, ctypes.c_int32))
setattr(hipFuncAttributes, 'preferredShmemCarveout', field(36, ctypes.c_int32))
setattr(hipFuncAttributes, 'ptxVersion', field(40, ctypes.c_int32))
setattr(hipFuncAttributes, 'sharedSizeBytes', field(48, size_t))
hipLimit_t = CEnum(ctypes.c_uint32)
hipLimitStackSize = hipLimit_t.define('hipLimitStackSize', 0)
hipLimitPrintfFifoSize = hipLimit_t.define('hipLimitPrintfFifoSize', 1)
hipLimitMallocHeapSize = hipLimit_t.define('hipLimitMallocHeapSize', 2)
hipLimitRange = hipLimit_t.define('hipLimitRange', 3)

hipMemoryAdvise = CEnum(ctypes.c_uint32)
hipMemAdviseSetReadMostly = hipMemoryAdvise.define('hipMemAdviseSetReadMostly', 1)
hipMemAdviseUnsetReadMostly = hipMemoryAdvise.define('hipMemAdviseUnsetReadMostly', 2)
hipMemAdviseSetPreferredLocation = hipMemoryAdvise.define('hipMemAdviseSetPreferredLocation', 3)
hipMemAdviseUnsetPreferredLocation = hipMemoryAdvise.define('hipMemAdviseUnsetPreferredLocation', 4)
hipMemAdviseSetAccessedBy = hipMemoryAdvise.define('hipMemAdviseSetAccessedBy', 5)
hipMemAdviseUnsetAccessedBy = hipMemoryAdvise.define('hipMemAdviseUnsetAccessedBy', 6)
hipMemAdviseSetCoarseGrain = hipMemoryAdvise.define('hipMemAdviseSetCoarseGrain', 100)
hipMemAdviseUnsetCoarseGrain = hipMemoryAdvise.define('hipMemAdviseUnsetCoarseGrain', 101)

hipMemRangeCoherencyMode = CEnum(ctypes.c_uint32)
hipMemRangeCoherencyModeFineGrain = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeFineGrain', 0)
hipMemRangeCoherencyModeCoarseGrain = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeCoarseGrain', 1)
hipMemRangeCoherencyModeIndeterminate = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeIndeterminate', 2)

hipMemRangeAttribute = CEnum(ctypes.c_uint32)
hipMemRangeAttributeReadMostly = hipMemRangeAttribute.define('hipMemRangeAttributeReadMostly', 1)
hipMemRangeAttributePreferredLocation = hipMemRangeAttribute.define('hipMemRangeAttributePreferredLocation', 2)
hipMemRangeAttributeAccessedBy = hipMemRangeAttribute.define('hipMemRangeAttributeAccessedBy', 3)
hipMemRangeAttributeLastPrefetchLocation = hipMemRangeAttribute.define('hipMemRangeAttributeLastPrefetchLocation', 4)
hipMemRangeAttributeCoherencyMode = hipMemRangeAttribute.define('hipMemRangeAttributeCoherencyMode', 100)

hipMemPoolAttr = CEnum(ctypes.c_uint32)
hipMemPoolReuseFollowEventDependencies = hipMemPoolAttr.define('hipMemPoolReuseFollowEventDependencies', 1)
hipMemPoolReuseAllowOpportunistic = hipMemPoolAttr.define('hipMemPoolReuseAllowOpportunistic', 2)
hipMemPoolReuseAllowInternalDependencies = hipMemPoolAttr.define('hipMemPoolReuseAllowInternalDependencies', 3)
hipMemPoolAttrReleaseThreshold = hipMemPoolAttr.define('hipMemPoolAttrReleaseThreshold', 4)
hipMemPoolAttrReservedMemCurrent = hipMemPoolAttr.define('hipMemPoolAttrReservedMemCurrent', 5)
hipMemPoolAttrReservedMemHigh = hipMemPoolAttr.define('hipMemPoolAttrReservedMemHigh', 6)
hipMemPoolAttrUsedMemCurrent = hipMemPoolAttr.define('hipMemPoolAttrUsedMemCurrent', 7)
hipMemPoolAttrUsedMemHigh = hipMemPoolAttr.define('hipMemPoolAttrUsedMemHigh', 8)

hipMemLocationType = CEnum(ctypes.c_uint32)
hipMemLocationTypeInvalid = hipMemLocationType.define('hipMemLocationTypeInvalid', 0)
hipMemLocationTypeDevice = hipMemLocationType.define('hipMemLocationTypeDevice', 1)

class hipMemLocation(Struct): pass
hipMemLocation.SIZE = 8
hipMemLocation._fields_ = ['type', 'id']
setattr(hipMemLocation, 'type', field(0, hipMemLocationType))
setattr(hipMemLocation, 'id', field(4, ctypes.c_int32))
hipMemAccessFlags = CEnum(ctypes.c_uint32)
hipMemAccessFlagsProtNone = hipMemAccessFlags.define('hipMemAccessFlagsProtNone', 0)
hipMemAccessFlagsProtRead = hipMemAccessFlags.define('hipMemAccessFlagsProtRead', 1)
hipMemAccessFlagsProtReadWrite = hipMemAccessFlags.define('hipMemAccessFlagsProtReadWrite', 3)

class hipMemAccessDesc(Struct): pass
hipMemAccessDesc.SIZE = 12
hipMemAccessDesc._fields_ = ['location', 'flags']
setattr(hipMemAccessDesc, 'location', field(0, hipMemLocation))
setattr(hipMemAccessDesc, 'flags', field(8, hipMemAccessFlags))
hipMemAllocationType = CEnum(ctypes.c_uint32)
hipMemAllocationTypeInvalid = hipMemAllocationType.define('hipMemAllocationTypeInvalid', 0)
hipMemAllocationTypePinned = hipMemAllocationType.define('hipMemAllocationTypePinned', 1)
hipMemAllocationTypeMax = hipMemAllocationType.define('hipMemAllocationTypeMax', 2147483647)

hipMemAllocationHandleType = CEnum(ctypes.c_uint32)
hipMemHandleTypeNone = hipMemAllocationHandleType.define('hipMemHandleTypeNone', 0)
hipMemHandleTypePosixFileDescriptor = hipMemAllocationHandleType.define('hipMemHandleTypePosixFileDescriptor', 1)
hipMemHandleTypeWin32 = hipMemAllocationHandleType.define('hipMemHandleTypeWin32', 2)
hipMemHandleTypeWin32Kmt = hipMemAllocationHandleType.define('hipMemHandleTypeWin32Kmt', 4)

class hipMemPoolProps(Struct): pass
hipMemPoolProps.SIZE = 88
hipMemPoolProps._fields_ = ['allocType', 'handleTypes', 'location', 'win32SecurityAttributes', 'maxSize', 'reserved']
setattr(hipMemPoolProps, 'allocType', field(0, hipMemAllocationType))
setattr(hipMemPoolProps, 'handleTypes', field(4, hipMemAllocationHandleType))
setattr(hipMemPoolProps, 'location', field(8, hipMemLocation))
setattr(hipMemPoolProps, 'win32SecurityAttributes', field(16, ctypes.c_void_p))
setattr(hipMemPoolProps, 'maxSize', field(24, size_t))
setattr(hipMemPoolProps, 'reserved', field(32, Array(ctypes.c_ubyte, 56)))
class hipMemPoolPtrExportData(Struct): pass
hipMemPoolPtrExportData.SIZE = 64
hipMemPoolPtrExportData._fields_ = ['reserved']
setattr(hipMemPoolPtrExportData, 'reserved', field(0, Array(ctypes.c_ubyte, 64)))
hipJitOption = CEnum(ctypes.c_uint32)
hipJitOptionMaxRegisters = hipJitOption.define('hipJitOptionMaxRegisters', 0)
hipJitOptionThreadsPerBlock = hipJitOption.define('hipJitOptionThreadsPerBlock', 1)
hipJitOptionWallTime = hipJitOption.define('hipJitOptionWallTime', 2)
hipJitOptionInfoLogBuffer = hipJitOption.define('hipJitOptionInfoLogBuffer', 3)
hipJitOptionInfoLogBufferSizeBytes = hipJitOption.define('hipJitOptionInfoLogBufferSizeBytes', 4)
hipJitOptionErrorLogBuffer = hipJitOption.define('hipJitOptionErrorLogBuffer', 5)
hipJitOptionErrorLogBufferSizeBytes = hipJitOption.define('hipJitOptionErrorLogBufferSizeBytes', 6)
hipJitOptionOptimizationLevel = hipJitOption.define('hipJitOptionOptimizationLevel', 7)
hipJitOptionTargetFromContext = hipJitOption.define('hipJitOptionTargetFromContext', 8)
hipJitOptionTarget = hipJitOption.define('hipJitOptionTarget', 9)
hipJitOptionFallbackStrategy = hipJitOption.define('hipJitOptionFallbackStrategy', 10)
hipJitOptionGenerateDebugInfo = hipJitOption.define('hipJitOptionGenerateDebugInfo', 11)
hipJitOptionLogVerbose = hipJitOption.define('hipJitOptionLogVerbose', 12)
hipJitOptionGenerateLineInfo = hipJitOption.define('hipJitOptionGenerateLineInfo', 13)
hipJitOptionCacheMode = hipJitOption.define('hipJitOptionCacheMode', 14)
hipJitOptionSm3xOpt = hipJitOption.define('hipJitOptionSm3xOpt', 15)
hipJitOptionFastCompile = hipJitOption.define('hipJitOptionFastCompile', 16)
hipJitOptionNumOptions = hipJitOption.define('hipJitOptionNumOptions', 17)

hipFuncAttribute = CEnum(ctypes.c_uint32)
hipFuncAttributeMaxDynamicSharedMemorySize = hipFuncAttribute.define('hipFuncAttributeMaxDynamicSharedMemorySize', 8)
hipFuncAttributePreferredSharedMemoryCarveout = hipFuncAttribute.define('hipFuncAttributePreferredSharedMemoryCarveout', 9)
hipFuncAttributeMax = hipFuncAttribute.define('hipFuncAttributeMax', 10)

hipFuncCache_t = CEnum(ctypes.c_uint32)
hipFuncCachePreferNone = hipFuncCache_t.define('hipFuncCachePreferNone', 0)
hipFuncCachePreferShared = hipFuncCache_t.define('hipFuncCachePreferShared', 1)
hipFuncCachePreferL1 = hipFuncCache_t.define('hipFuncCachePreferL1', 2)
hipFuncCachePreferEqual = hipFuncCache_t.define('hipFuncCachePreferEqual', 3)

hipSharedMemConfig = CEnum(ctypes.c_uint32)
hipSharedMemBankSizeDefault = hipSharedMemConfig.define('hipSharedMemBankSizeDefault', 0)
hipSharedMemBankSizeFourByte = hipSharedMemConfig.define('hipSharedMemBankSizeFourByte', 1)
hipSharedMemBankSizeEightByte = hipSharedMemConfig.define('hipSharedMemBankSizeEightByte', 2)

class hipLaunchParams_t(Struct): pass
hipLaunchParams_t.SIZE = 56
hipLaunchParams_t._fields_ = ['func', 'gridDim', 'blockDim', 'args', 'sharedMem', 'stream']
setattr(hipLaunchParams_t, 'func', field(0, ctypes.c_void_p))
setattr(hipLaunchParams_t, 'gridDim', field(8, dim3))
setattr(hipLaunchParams_t, 'blockDim', field(20, dim3))
setattr(hipLaunchParams_t, 'args', field(32, Pointer(ctypes.c_void_p)))
setattr(hipLaunchParams_t, 'sharedMem', field(40, size_t))
setattr(hipLaunchParams_t, 'stream', field(48, hipStream_t))
hipLaunchParams = hipLaunchParams_t
class hipFunctionLaunchParams_t(Struct): pass
hipFunctionLaunchParams_t.SIZE = 56
hipFunctionLaunchParams_t._fields_ = ['function', 'gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes', 'hStream', 'kernelParams']
setattr(hipFunctionLaunchParams_t, 'function', field(0, hipFunction_t))
setattr(hipFunctionLaunchParams_t, 'gridDimX', field(8, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'gridDimY', field(12, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'gridDimZ', field(16, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'blockDimX', field(20, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'blockDimY', field(24, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'blockDimZ', field(28, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'sharedMemBytes', field(32, ctypes.c_uint32))
setattr(hipFunctionLaunchParams_t, 'hStream', field(40, hipStream_t))
setattr(hipFunctionLaunchParams_t, 'kernelParams', field(48, Pointer(ctypes.c_void_p)))
hipFunctionLaunchParams = hipFunctionLaunchParams_t
hipExternalMemoryHandleType_enum = CEnum(ctypes.c_uint32)
hipExternalMemoryHandleTypeOpaqueFd = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueFd', 1)
hipExternalMemoryHandleTypeOpaqueWin32 = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueWin32', 2)
hipExternalMemoryHandleTypeOpaqueWin32Kmt = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueWin32Kmt', 3)
hipExternalMemoryHandleTypeD3D12Heap = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D12Heap', 4)
hipExternalMemoryHandleTypeD3D12Resource = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D12Resource', 5)
hipExternalMemoryHandleTypeD3D11Resource = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D11Resource', 6)
hipExternalMemoryHandleTypeD3D11ResourceKmt = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D11ResourceKmt', 7)
hipExternalMemoryHandleTypeNvSciBuf = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeNvSciBuf', 8)

hipExternalMemoryHandleType = hipExternalMemoryHandleType_enum
class hipExternalMemoryHandleDesc_st(Struct): pass
class _anonunion1(Union): pass
class _anonstruct2(Struct): pass
_anonstruct2.SIZE = 16
_anonstruct2._fields_ = ['handle', 'name']
setattr(_anonstruct2, 'handle', field(0, ctypes.c_void_p))
setattr(_anonstruct2, 'name', field(8, ctypes.c_void_p))
_anonunion1.SIZE = 16
_anonunion1._fields_ = ['fd', 'win32', 'nvSciBufObject']
setattr(_anonunion1, 'fd', field(0, ctypes.c_int32))
setattr(_anonunion1, 'win32', field(0, _anonstruct2))
setattr(_anonunion1, 'nvSciBufObject', field(0, ctypes.c_void_p))
hipExternalMemoryHandleDesc_st.SIZE = 104
hipExternalMemoryHandleDesc_st._fields_ = ['type', 'handle', 'size', 'flags', 'reserved']
setattr(hipExternalMemoryHandleDesc_st, 'type', field(0, hipExternalMemoryHandleType))
setattr(hipExternalMemoryHandleDesc_st, 'handle', field(8, _anonunion1))
setattr(hipExternalMemoryHandleDesc_st, 'size', field(24, ctypes.c_uint64))
setattr(hipExternalMemoryHandleDesc_st, 'flags', field(32, ctypes.c_uint32))
setattr(hipExternalMemoryHandleDesc_st, 'reserved', field(36, Array(ctypes.c_uint32, 16)))
hipExternalMemoryHandleDesc = hipExternalMemoryHandleDesc_st
class hipExternalMemoryBufferDesc_st(Struct): pass
hipExternalMemoryBufferDesc_st.SIZE = 88
hipExternalMemoryBufferDesc_st._fields_ = ['offset', 'size', 'flags', 'reserved']
setattr(hipExternalMemoryBufferDesc_st, 'offset', field(0, ctypes.c_uint64))
setattr(hipExternalMemoryBufferDesc_st, 'size', field(8, ctypes.c_uint64))
setattr(hipExternalMemoryBufferDesc_st, 'flags', field(16, ctypes.c_uint32))
setattr(hipExternalMemoryBufferDesc_st, 'reserved', field(20, Array(ctypes.c_uint32, 16)))
hipExternalMemoryBufferDesc = hipExternalMemoryBufferDesc_st
class hipExternalMemoryMipmappedArrayDesc_st(Struct): pass
class hipChannelFormatDesc(Struct): pass
hipChannelFormatKind = CEnum(ctypes.c_uint32)
hipChannelFormatKindSigned = hipChannelFormatKind.define('hipChannelFormatKindSigned', 0)
hipChannelFormatKindUnsigned = hipChannelFormatKind.define('hipChannelFormatKindUnsigned', 1)
hipChannelFormatKindFloat = hipChannelFormatKind.define('hipChannelFormatKindFloat', 2)
hipChannelFormatKindNone = hipChannelFormatKind.define('hipChannelFormatKindNone', 3)

hipChannelFormatDesc.SIZE = 20
hipChannelFormatDesc._fields_ = ['x', 'y', 'z', 'w', 'f']
setattr(hipChannelFormatDesc, 'x', field(0, ctypes.c_int32))
setattr(hipChannelFormatDesc, 'y', field(4, ctypes.c_int32))
setattr(hipChannelFormatDesc, 'z', field(8, ctypes.c_int32))
setattr(hipChannelFormatDesc, 'w', field(12, ctypes.c_int32))
setattr(hipChannelFormatDesc, 'f', field(16, hipChannelFormatKind))
class hipExtent(Struct): pass
hipExtent.SIZE = 24
hipExtent._fields_ = ['width', 'height', 'depth']
setattr(hipExtent, 'width', field(0, size_t))
setattr(hipExtent, 'height', field(8, size_t))
setattr(hipExtent, 'depth', field(16, size_t))
hipExternalMemoryMipmappedArrayDesc_st.SIZE = 64
hipExternalMemoryMipmappedArrayDesc_st._fields_ = ['offset', 'formatDesc', 'extent', 'flags', 'numLevels']
setattr(hipExternalMemoryMipmappedArrayDesc_st, 'offset', field(0, ctypes.c_uint64))
setattr(hipExternalMemoryMipmappedArrayDesc_st, 'formatDesc', field(8, hipChannelFormatDesc))
setattr(hipExternalMemoryMipmappedArrayDesc_st, 'extent', field(32, hipExtent))
setattr(hipExternalMemoryMipmappedArrayDesc_st, 'flags', field(56, ctypes.c_uint32))
setattr(hipExternalMemoryMipmappedArrayDesc_st, 'numLevels', field(60, ctypes.c_uint32))
hipExternalMemoryMipmappedArrayDesc = hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t = ctypes.c_void_p
hipExternalSemaphoreHandleType_enum = CEnum(ctypes.c_uint32)
hipExternalSemaphoreHandleTypeOpaqueFd = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeOpaqueFd', 1)
hipExternalSemaphoreHandleTypeOpaqueWin32 = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeOpaqueWin32', 2)
hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeOpaqueWin32Kmt', 3)
hipExternalSemaphoreHandleTypeD3D12Fence = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeD3D12Fence', 4)
hipExternalSemaphoreHandleTypeD3D11Fence = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeD3D11Fence', 5)
hipExternalSemaphoreHandleTypeNvSciSync = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeNvSciSync', 6)
hipExternalSemaphoreHandleTypeKeyedMutex = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeKeyedMutex', 7)
hipExternalSemaphoreHandleTypeKeyedMutexKmt = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeKeyedMutexKmt', 8)
hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeTimelineSemaphoreFd', 9)
hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = hipExternalSemaphoreHandleType_enum.define('hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32', 10)

hipExternalSemaphoreHandleType = hipExternalSemaphoreHandleType_enum
class hipExternalSemaphoreHandleDesc_st(Struct): pass
class _anonunion3(Union): pass
class _anonstruct4(Struct): pass
_anonstruct4.SIZE = 16
_anonstruct4._fields_ = ['handle', 'name']
setattr(_anonstruct4, 'handle', field(0, ctypes.c_void_p))
setattr(_anonstruct4, 'name', field(8, ctypes.c_void_p))
_anonunion3.SIZE = 16
_anonunion3._fields_ = ['fd', 'win32', 'NvSciSyncObj']
setattr(_anonunion3, 'fd', field(0, ctypes.c_int32))
setattr(_anonunion3, 'win32', field(0, _anonstruct4))
setattr(_anonunion3, 'NvSciSyncObj', field(0, ctypes.c_void_p))
hipExternalSemaphoreHandleDesc_st.SIZE = 96
hipExternalSemaphoreHandleDesc_st._fields_ = ['type', 'handle', 'flags', 'reserved']
setattr(hipExternalSemaphoreHandleDesc_st, 'type', field(0, hipExternalSemaphoreHandleType))
setattr(hipExternalSemaphoreHandleDesc_st, 'handle', field(8, _anonunion3))
setattr(hipExternalSemaphoreHandleDesc_st, 'flags', field(24, ctypes.c_uint32))
setattr(hipExternalSemaphoreHandleDesc_st, 'reserved', field(28, Array(ctypes.c_uint32, 16)))
hipExternalSemaphoreHandleDesc = hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t = ctypes.c_void_p
class hipExternalSemaphoreSignalParams_st(Struct): pass
class _anonstruct5(Struct): pass
class _anonstruct6(Struct): pass
_anonstruct6.SIZE = 8
_anonstruct6._fields_ = ['value']
setattr(_anonstruct6, 'value', field(0, ctypes.c_uint64))
class _anonunion7(Union): pass
_anonunion7.SIZE = 8
_anonunion7._fields_ = ['fence', 'reserved']
setattr(_anonunion7, 'fence', field(0, ctypes.c_void_p))
setattr(_anonunion7, 'reserved', field(0, ctypes.c_uint64))
class _anonstruct8(Struct): pass
_anonstruct8.SIZE = 8
_anonstruct8._fields_ = ['key']
setattr(_anonstruct8, 'key', field(0, ctypes.c_uint64))
_anonstruct5.SIZE = 72
_anonstruct5._fields_ = ['fence', 'nvSciSync', 'keyedMutex', 'reserved']
setattr(_anonstruct5, 'fence', field(0, _anonstruct6))
setattr(_anonstruct5, 'nvSciSync', field(8, _anonunion7))
setattr(_anonstruct5, 'keyedMutex', field(16, _anonstruct8))
setattr(_anonstruct5, 'reserved', field(24, Array(ctypes.c_uint32, 12)))
hipExternalSemaphoreSignalParams_st.SIZE = 144
hipExternalSemaphoreSignalParams_st._fields_ = ['params', 'flags', 'reserved']
setattr(hipExternalSemaphoreSignalParams_st, 'params', field(0, _anonstruct5))
setattr(hipExternalSemaphoreSignalParams_st, 'flags', field(72, ctypes.c_uint32))
setattr(hipExternalSemaphoreSignalParams_st, 'reserved', field(76, Array(ctypes.c_uint32, 16)))
hipExternalSemaphoreSignalParams = hipExternalSemaphoreSignalParams_st
class hipExternalSemaphoreWaitParams_st(Struct): pass
class _anonstruct9(Struct): pass
class _anonstruct10(Struct): pass
_anonstruct10.SIZE = 8
_anonstruct10._fields_ = ['value']
setattr(_anonstruct10, 'value', field(0, ctypes.c_uint64))
class _anonunion11(Union): pass
_anonunion11.SIZE = 8
_anonunion11._fields_ = ['fence', 'reserved']
setattr(_anonunion11, 'fence', field(0, ctypes.c_void_p))
setattr(_anonunion11, 'reserved', field(0, ctypes.c_uint64))
class _anonstruct12(Struct): pass
_anonstruct12.SIZE = 16
_anonstruct12._fields_ = ['key', 'timeoutMs']
setattr(_anonstruct12, 'key', field(0, ctypes.c_uint64))
setattr(_anonstruct12, 'timeoutMs', field(8, ctypes.c_uint32))
_anonstruct9.SIZE = 72
_anonstruct9._fields_ = ['fence', 'nvSciSync', 'keyedMutex', 'reserved']
setattr(_anonstruct9, 'fence', field(0, _anonstruct10))
setattr(_anonstruct9, 'nvSciSync', field(8, _anonunion11))
setattr(_anonstruct9, 'keyedMutex', field(16, _anonstruct12))
setattr(_anonstruct9, 'reserved', field(32, Array(ctypes.c_uint32, 10)))
hipExternalSemaphoreWaitParams_st.SIZE = 144
hipExternalSemaphoreWaitParams_st._fields_ = ['params', 'flags', 'reserved']
setattr(hipExternalSemaphoreWaitParams_st, 'params', field(0, _anonstruct9))
setattr(hipExternalSemaphoreWaitParams_st, 'flags', field(72, ctypes.c_uint32))
setattr(hipExternalSemaphoreWaitParams_st, 'reserved', field(76, Array(ctypes.c_uint32, 16)))
hipExternalSemaphoreWaitParams = hipExternalSemaphoreWaitParams_st
try: (__hipGetPCH:=dll.__hipGetPCH).restype, __hipGetPCH.argtypes = None, [Pointer(Pointer(ctypes.c_char)), Pointer(ctypes.c_uint32)]
except AttributeError: pass

hipGraphicsRegisterFlags = CEnum(ctypes.c_uint32)
hipGraphicsRegisterFlagsNone = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsNone', 0)
hipGraphicsRegisterFlagsReadOnly = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsReadOnly', 1)
hipGraphicsRegisterFlagsWriteDiscard = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsWriteDiscard', 2)
hipGraphicsRegisterFlagsSurfaceLoadStore = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsSurfaceLoadStore', 4)
hipGraphicsRegisterFlagsTextureGather = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsTextureGather', 8)

class _hipGraphicsResource(Struct): pass
hipGraphicsResource = _hipGraphicsResource
hipGraphicsResource_t = Pointer(_hipGraphicsResource)
class ihipGraph(Struct): pass
hipGraph_t = Pointer(ihipGraph)
class hipGraphNode(Struct): pass
hipGraphNode_t = Pointer(hipGraphNode)
class hipGraphExec(Struct): pass
hipGraphExec_t = Pointer(hipGraphExec)
class hipUserObject(Struct): pass
hipUserObject_t = Pointer(hipUserObject)
hipGraphNodeType = CEnum(ctypes.c_uint32)
hipGraphNodeTypeKernel = hipGraphNodeType.define('hipGraphNodeTypeKernel', 0)
hipGraphNodeTypeMemcpy = hipGraphNodeType.define('hipGraphNodeTypeMemcpy', 1)
hipGraphNodeTypeMemset = hipGraphNodeType.define('hipGraphNodeTypeMemset', 2)
hipGraphNodeTypeHost = hipGraphNodeType.define('hipGraphNodeTypeHost', 3)
hipGraphNodeTypeGraph = hipGraphNodeType.define('hipGraphNodeTypeGraph', 4)
hipGraphNodeTypeEmpty = hipGraphNodeType.define('hipGraphNodeTypeEmpty', 5)
hipGraphNodeTypeWaitEvent = hipGraphNodeType.define('hipGraphNodeTypeWaitEvent', 6)
hipGraphNodeTypeEventRecord = hipGraphNodeType.define('hipGraphNodeTypeEventRecord', 7)
hipGraphNodeTypeExtSemaphoreSignal = hipGraphNodeType.define('hipGraphNodeTypeExtSemaphoreSignal', 8)
hipGraphNodeTypeExtSemaphoreWait = hipGraphNodeType.define('hipGraphNodeTypeExtSemaphoreWait', 9)
hipGraphNodeTypeMemAlloc = hipGraphNodeType.define('hipGraphNodeTypeMemAlloc', 10)
hipGraphNodeTypeMemFree = hipGraphNodeType.define('hipGraphNodeTypeMemFree', 11)
hipGraphNodeTypeMemcpyFromSymbol = hipGraphNodeType.define('hipGraphNodeTypeMemcpyFromSymbol', 12)
hipGraphNodeTypeMemcpyToSymbol = hipGraphNodeType.define('hipGraphNodeTypeMemcpyToSymbol', 13)
hipGraphNodeTypeCount = hipGraphNodeType.define('hipGraphNodeTypeCount', 14)

hipHostFn_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
class hipHostNodeParams(Struct): pass
hipHostNodeParams.SIZE = 16
hipHostNodeParams._fields_ = ['fn', 'userData']
setattr(hipHostNodeParams, 'fn', field(0, hipHostFn_t))
setattr(hipHostNodeParams, 'userData', field(8, ctypes.c_void_p))
class hipKernelNodeParams(Struct): pass
hipKernelNodeParams.SIZE = 64
hipKernelNodeParams._fields_ = ['blockDim', 'extra', 'func', 'gridDim', 'kernelParams', 'sharedMemBytes']
setattr(hipKernelNodeParams, 'blockDim', field(0, dim3))
setattr(hipKernelNodeParams, 'extra', field(16, Pointer(ctypes.c_void_p)))
setattr(hipKernelNodeParams, 'func', field(24, ctypes.c_void_p))
setattr(hipKernelNodeParams, 'gridDim', field(32, dim3))
setattr(hipKernelNodeParams, 'kernelParams', field(48, Pointer(ctypes.c_void_p)))
setattr(hipKernelNodeParams, 'sharedMemBytes', field(56, ctypes.c_uint32))
class hipMemsetParams(Struct): pass
hipMemsetParams.SIZE = 48
hipMemsetParams._fields_ = ['dst', 'elementSize', 'height', 'pitch', 'value', 'width']
setattr(hipMemsetParams, 'dst', field(0, ctypes.c_void_p))
setattr(hipMemsetParams, 'elementSize', field(8, ctypes.c_uint32))
setattr(hipMemsetParams, 'height', field(16, size_t))
setattr(hipMemsetParams, 'pitch', field(24, size_t))
setattr(hipMemsetParams, 'value', field(32, ctypes.c_uint32))
setattr(hipMemsetParams, 'width', field(40, size_t))
class hipMemAllocNodeParams(Struct): pass
hipMemAllocNodeParams.SIZE = 120
hipMemAllocNodeParams._fields_ = ['poolProps', 'accessDescs', 'accessDescCount', 'bytesize', 'dptr']
setattr(hipMemAllocNodeParams, 'poolProps', field(0, hipMemPoolProps))
setattr(hipMemAllocNodeParams, 'accessDescs', field(88, Pointer(hipMemAccessDesc)))
setattr(hipMemAllocNodeParams, 'accessDescCount', field(96, size_t))
setattr(hipMemAllocNodeParams, 'bytesize', field(104, size_t))
setattr(hipMemAllocNodeParams, 'dptr', field(112, ctypes.c_void_p))
hipAccessProperty = CEnum(ctypes.c_uint32)
hipAccessPropertyNormal = hipAccessProperty.define('hipAccessPropertyNormal', 0)
hipAccessPropertyStreaming = hipAccessProperty.define('hipAccessPropertyStreaming', 1)
hipAccessPropertyPersisting = hipAccessProperty.define('hipAccessPropertyPersisting', 2)

class hipAccessPolicyWindow(Struct): pass
hipAccessPolicyWindow.SIZE = 32
hipAccessPolicyWindow._fields_ = ['base_ptr', 'hitProp', 'hitRatio', 'missProp', 'num_bytes']
setattr(hipAccessPolicyWindow, 'base_ptr', field(0, ctypes.c_void_p))
setattr(hipAccessPolicyWindow, 'hitProp', field(8, hipAccessProperty))
setattr(hipAccessPolicyWindow, 'hitRatio', field(12, ctypes.c_float))
setattr(hipAccessPolicyWindow, 'missProp', field(16, hipAccessProperty))
setattr(hipAccessPolicyWindow, 'num_bytes', field(24, size_t))
hipLaunchAttributeID = CEnum(ctypes.c_uint32)
hipLaunchAttributeAccessPolicyWindow = hipLaunchAttributeID.define('hipLaunchAttributeAccessPolicyWindow', 1)
hipLaunchAttributeCooperative = hipLaunchAttributeID.define('hipLaunchAttributeCooperative', 2)
hipLaunchAttributePriority = hipLaunchAttributeID.define('hipLaunchAttributePriority', 8)

class hipLaunchAttributeValue(Union): pass
hipLaunchAttributeValue.SIZE = 32
hipLaunchAttributeValue._fields_ = ['accessPolicyWindow', 'cooperative', 'priority']
setattr(hipLaunchAttributeValue, 'accessPolicyWindow', field(0, hipAccessPolicyWindow))
setattr(hipLaunchAttributeValue, 'cooperative', field(0, ctypes.c_int32))
setattr(hipLaunchAttributeValue, 'priority', field(0, ctypes.c_int32))
class HIP_MEMSET_NODE_PARAMS(Struct): pass
hipDeviceptr_t = ctypes.c_void_p
HIP_MEMSET_NODE_PARAMS.SIZE = 40
HIP_MEMSET_NODE_PARAMS._fields_ = ['dst', 'pitch', 'value', 'elementSize', 'width', 'height']
setattr(HIP_MEMSET_NODE_PARAMS, 'dst', field(0, hipDeviceptr_t))
setattr(HIP_MEMSET_NODE_PARAMS, 'pitch', field(8, size_t))
setattr(HIP_MEMSET_NODE_PARAMS, 'value', field(16, ctypes.c_uint32))
setattr(HIP_MEMSET_NODE_PARAMS, 'elementSize', field(20, ctypes.c_uint32))
setattr(HIP_MEMSET_NODE_PARAMS, 'width', field(24, size_t))
setattr(HIP_MEMSET_NODE_PARAMS, 'height', field(32, size_t))
hipGraphExecUpdateResult = CEnum(ctypes.c_uint32)
hipGraphExecUpdateSuccess = hipGraphExecUpdateResult.define('hipGraphExecUpdateSuccess', 0)
hipGraphExecUpdateError = hipGraphExecUpdateResult.define('hipGraphExecUpdateError', 1)
hipGraphExecUpdateErrorTopologyChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorTopologyChanged', 2)
hipGraphExecUpdateErrorNodeTypeChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorNodeTypeChanged', 3)
hipGraphExecUpdateErrorFunctionChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorFunctionChanged', 4)
hipGraphExecUpdateErrorParametersChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorParametersChanged', 5)
hipGraphExecUpdateErrorNotSupported = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorNotSupported', 6)
hipGraphExecUpdateErrorUnsupportedFunctionChange = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorUnsupportedFunctionChange', 7)

hipStreamCaptureMode = CEnum(ctypes.c_uint32)
hipStreamCaptureModeGlobal = hipStreamCaptureMode.define('hipStreamCaptureModeGlobal', 0)
hipStreamCaptureModeThreadLocal = hipStreamCaptureMode.define('hipStreamCaptureModeThreadLocal', 1)
hipStreamCaptureModeRelaxed = hipStreamCaptureMode.define('hipStreamCaptureModeRelaxed', 2)

hipStreamCaptureStatus = CEnum(ctypes.c_uint32)
hipStreamCaptureStatusNone = hipStreamCaptureStatus.define('hipStreamCaptureStatusNone', 0)
hipStreamCaptureStatusActive = hipStreamCaptureStatus.define('hipStreamCaptureStatusActive', 1)
hipStreamCaptureStatusInvalidated = hipStreamCaptureStatus.define('hipStreamCaptureStatusInvalidated', 2)

hipStreamUpdateCaptureDependenciesFlags = CEnum(ctypes.c_uint32)
hipStreamAddCaptureDependencies = hipStreamUpdateCaptureDependenciesFlags.define('hipStreamAddCaptureDependencies', 0)
hipStreamSetCaptureDependencies = hipStreamUpdateCaptureDependenciesFlags.define('hipStreamSetCaptureDependencies', 1)

hipGraphMemAttributeType = CEnum(ctypes.c_uint32)
hipGraphMemAttrUsedMemCurrent = hipGraphMemAttributeType.define('hipGraphMemAttrUsedMemCurrent', 0)
hipGraphMemAttrUsedMemHigh = hipGraphMemAttributeType.define('hipGraphMemAttrUsedMemHigh', 1)
hipGraphMemAttrReservedMemCurrent = hipGraphMemAttributeType.define('hipGraphMemAttrReservedMemCurrent', 2)
hipGraphMemAttrReservedMemHigh = hipGraphMemAttributeType.define('hipGraphMemAttrReservedMemHigh', 3)

hipUserObjectFlags = CEnum(ctypes.c_uint32)
hipUserObjectNoDestructorSync = hipUserObjectFlags.define('hipUserObjectNoDestructorSync', 1)

hipUserObjectRetainFlags = CEnum(ctypes.c_uint32)
hipGraphUserObjectMove = hipUserObjectRetainFlags.define('hipGraphUserObjectMove', 1)

hipGraphInstantiateFlags = CEnum(ctypes.c_uint32)
hipGraphInstantiateFlagAutoFreeOnLaunch = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagAutoFreeOnLaunch', 1)
hipGraphInstantiateFlagUpload = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagUpload', 2)
hipGraphInstantiateFlagDeviceLaunch = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagDeviceLaunch', 4)
hipGraphInstantiateFlagUseNodePriority = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagUseNodePriority', 8)

hipGraphDebugDotFlags = CEnum(ctypes.c_uint32)
hipGraphDebugDotFlagsVerbose = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsVerbose', 1)
hipGraphDebugDotFlagsKernelNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsKernelNodeParams', 4)
hipGraphDebugDotFlagsMemcpyNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsMemcpyNodeParams', 8)
hipGraphDebugDotFlagsMemsetNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsMemsetNodeParams', 16)
hipGraphDebugDotFlagsHostNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsHostNodeParams', 32)
hipGraphDebugDotFlagsEventNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsEventNodeParams', 64)
hipGraphDebugDotFlagsExtSemasSignalNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsExtSemasSignalNodeParams', 128)
hipGraphDebugDotFlagsExtSemasWaitNodeParams = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsExtSemasWaitNodeParams', 256)
hipGraphDebugDotFlagsKernelNodeAttributes = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsKernelNodeAttributes', 512)
hipGraphDebugDotFlagsHandles = hipGraphDebugDotFlags.define('hipGraphDebugDotFlagsHandles', 1024)

hipGraphInstantiateResult = CEnum(ctypes.c_uint32)
hipGraphInstantiateSuccess = hipGraphInstantiateResult.define('hipGraphInstantiateSuccess', 0)
hipGraphInstantiateError = hipGraphInstantiateResult.define('hipGraphInstantiateError', 1)
hipGraphInstantiateInvalidStructure = hipGraphInstantiateResult.define('hipGraphInstantiateInvalidStructure', 2)
hipGraphInstantiateNodeOperationNotSupported = hipGraphInstantiateResult.define('hipGraphInstantiateNodeOperationNotSupported', 3)
hipGraphInstantiateMultipleDevicesNotSupported = hipGraphInstantiateResult.define('hipGraphInstantiateMultipleDevicesNotSupported', 4)

class hipGraphInstantiateParams(Struct): pass
hipGraphInstantiateParams.SIZE = 32
hipGraphInstantiateParams._fields_ = ['errNode_out', 'flags', 'result_out', 'uploadStream']
setattr(hipGraphInstantiateParams, 'errNode_out', field(0, hipGraphNode_t))
setattr(hipGraphInstantiateParams, 'flags', field(8, ctypes.c_uint64))
setattr(hipGraphInstantiateParams, 'result_out', field(16, hipGraphInstantiateResult))
setattr(hipGraphInstantiateParams, 'uploadStream', field(24, hipStream_t))
class hipMemAllocationProp(Struct): pass
class _anonstruct13(Struct): pass
_anonstruct13.SIZE = 4
_anonstruct13._fields_ = ['compressionType', 'gpuDirectRDMACapable', 'usage']
setattr(_anonstruct13, 'compressionType', field(0, ctypes.c_ubyte))
setattr(_anonstruct13, 'gpuDirectRDMACapable', field(1, ctypes.c_ubyte))
setattr(_anonstruct13, 'usage', field(2, ctypes.c_uint16))
hipMemAllocationProp.SIZE = 32
hipMemAllocationProp._fields_ = ['type', 'requestedHandleType', 'location', 'win32HandleMetaData', 'allocFlags']
setattr(hipMemAllocationProp, 'type', field(0, hipMemAllocationType))
setattr(hipMemAllocationProp, 'requestedHandleType', field(4, hipMemAllocationHandleType))
setattr(hipMemAllocationProp, 'location', field(8, hipMemLocation))
setattr(hipMemAllocationProp, 'win32HandleMetaData', field(16, ctypes.c_void_p))
setattr(hipMemAllocationProp, 'allocFlags', field(24, _anonstruct13))
class hipExternalSemaphoreSignalNodeParams(Struct): pass
hipExternalSemaphoreSignalNodeParams.SIZE = 24
hipExternalSemaphoreSignalNodeParams._fields_ = ['extSemArray', 'paramsArray', 'numExtSems']
setattr(hipExternalSemaphoreSignalNodeParams, 'extSemArray', field(0, Pointer(hipExternalSemaphore_t)))
setattr(hipExternalSemaphoreSignalNodeParams, 'paramsArray', field(8, Pointer(hipExternalSemaphoreSignalParams)))
setattr(hipExternalSemaphoreSignalNodeParams, 'numExtSems', field(16, ctypes.c_uint32))
class hipExternalSemaphoreWaitNodeParams(Struct): pass
hipExternalSemaphoreWaitNodeParams.SIZE = 24
hipExternalSemaphoreWaitNodeParams._fields_ = ['extSemArray', 'paramsArray', 'numExtSems']
setattr(hipExternalSemaphoreWaitNodeParams, 'extSemArray', field(0, Pointer(hipExternalSemaphore_t)))
setattr(hipExternalSemaphoreWaitNodeParams, 'paramsArray', field(8, Pointer(hipExternalSemaphoreWaitParams)))
setattr(hipExternalSemaphoreWaitNodeParams, 'numExtSems', field(16, ctypes.c_uint32))
class ihipMemGenericAllocationHandle(Struct): pass
hipMemGenericAllocationHandle_t = Pointer(ihipMemGenericAllocationHandle)
hipMemAllocationGranularity_flags = CEnum(ctypes.c_uint32)
hipMemAllocationGranularityMinimum = hipMemAllocationGranularity_flags.define('hipMemAllocationGranularityMinimum', 0)
hipMemAllocationGranularityRecommended = hipMemAllocationGranularity_flags.define('hipMemAllocationGranularityRecommended', 1)

hipMemHandleType = CEnum(ctypes.c_uint32)
hipMemHandleTypeGeneric = hipMemHandleType.define('hipMemHandleTypeGeneric', 0)

hipMemOperationType = CEnum(ctypes.c_uint32)
hipMemOperationTypeMap = hipMemOperationType.define('hipMemOperationTypeMap', 1)
hipMemOperationTypeUnmap = hipMemOperationType.define('hipMemOperationTypeUnmap', 2)

hipArraySparseSubresourceType = CEnum(ctypes.c_uint32)
hipArraySparseSubresourceTypeSparseLevel = hipArraySparseSubresourceType.define('hipArraySparseSubresourceTypeSparseLevel', 0)
hipArraySparseSubresourceTypeMiptail = hipArraySparseSubresourceType.define('hipArraySparseSubresourceTypeMiptail', 1)

class hipArrayMapInfo(Struct): pass
hipResourceType = CEnum(ctypes.c_uint32)
hipResourceTypeArray = hipResourceType.define('hipResourceTypeArray', 0)
hipResourceTypeMipmappedArray = hipResourceType.define('hipResourceTypeMipmappedArray', 1)
hipResourceTypeLinear = hipResourceType.define('hipResourceTypeLinear', 2)
hipResourceTypePitch2D = hipResourceType.define('hipResourceTypePitch2D', 3)

class _anonunion14(Union): pass
class hipMipmappedArray(Struct): pass
hipArray_Format = CEnum(ctypes.c_uint32)
HIP_AD_FORMAT_UNSIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT8', 1)
HIP_AD_FORMAT_UNSIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT16', 2)
HIP_AD_FORMAT_UNSIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT32', 3)
HIP_AD_FORMAT_SIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT8', 8)
HIP_AD_FORMAT_SIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT16', 9)
HIP_AD_FORMAT_SIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT32', 10)
HIP_AD_FORMAT_HALF = hipArray_Format.define('HIP_AD_FORMAT_HALF', 16)
HIP_AD_FORMAT_FLOAT = hipArray_Format.define('HIP_AD_FORMAT_FLOAT', 32)

hipMipmappedArray.SIZE = 64
hipMipmappedArray._fields_ = ['data', 'desc', 'type', 'width', 'height', 'depth', 'min_mipmap_level', 'max_mipmap_level', 'flags', 'format', 'num_channels']
setattr(hipMipmappedArray, 'data', field(0, ctypes.c_void_p))
setattr(hipMipmappedArray, 'desc', field(8, hipChannelFormatDesc))
setattr(hipMipmappedArray, 'type', field(28, ctypes.c_uint32))
setattr(hipMipmappedArray, 'width', field(32, ctypes.c_uint32))
setattr(hipMipmappedArray, 'height', field(36, ctypes.c_uint32))
setattr(hipMipmappedArray, 'depth', field(40, ctypes.c_uint32))
setattr(hipMipmappedArray, 'min_mipmap_level', field(44, ctypes.c_uint32))
setattr(hipMipmappedArray, 'max_mipmap_level', field(48, ctypes.c_uint32))
setattr(hipMipmappedArray, 'flags', field(52, ctypes.c_uint32))
setattr(hipMipmappedArray, 'format', field(56, hipArray_Format))
setattr(hipMipmappedArray, 'num_channels', field(60, ctypes.c_uint32))
class hipArray(Struct): pass
hipArray_t = Pointer(hipArray)
_anonunion14.SIZE = 64
_anonunion14._fields_ = ['mipmap', 'array']
setattr(_anonunion14, 'mipmap', field(0, hipMipmappedArray))
setattr(_anonunion14, 'array', field(0, hipArray_t))
class _anonunion15(Union): pass
class _anonstruct16(Struct): pass
_anonstruct16.SIZE = 32
_anonstruct16._fields_ = ['level', 'layer', 'offsetX', 'offsetY', 'offsetZ', 'extentWidth', 'extentHeight', 'extentDepth']
setattr(_anonstruct16, 'level', field(0, ctypes.c_uint32))
setattr(_anonstruct16, 'layer', field(4, ctypes.c_uint32))
setattr(_anonstruct16, 'offsetX', field(8, ctypes.c_uint32))
setattr(_anonstruct16, 'offsetY', field(12, ctypes.c_uint32))
setattr(_anonstruct16, 'offsetZ', field(16, ctypes.c_uint32))
setattr(_anonstruct16, 'extentWidth', field(20, ctypes.c_uint32))
setattr(_anonstruct16, 'extentHeight', field(24, ctypes.c_uint32))
setattr(_anonstruct16, 'extentDepth', field(28, ctypes.c_uint32))
class _anonstruct17(Struct): pass
_anonstruct17.SIZE = 24
_anonstruct17._fields_ = ['layer', 'offset', 'size']
setattr(_anonstruct17, 'layer', field(0, ctypes.c_uint32))
setattr(_anonstruct17, 'offset', field(8, ctypes.c_uint64))
setattr(_anonstruct17, 'size', field(16, ctypes.c_uint64))
_anonunion15.SIZE = 32
_anonunion15._fields_ = ['sparseLevel', 'miptail']
setattr(_anonunion15, 'sparseLevel', field(0, _anonstruct16))
setattr(_anonunion15, 'miptail', field(0, _anonstruct17))
class _anonunion18(Union): pass
_anonunion18.SIZE = 8
_anonunion18._fields_ = ['memHandle']
setattr(_anonunion18, 'memHandle', field(0, hipMemGenericAllocationHandle_t))
hipArrayMapInfo.SIZE = 152
hipArrayMapInfo._fields_ = ['resourceType', 'resource', 'subresourceType', 'subresource', 'memOperationType', 'memHandleType', 'memHandle', 'offset', 'deviceBitMask', 'flags', 'reserved']
setattr(hipArrayMapInfo, 'resourceType', field(0, hipResourceType))
setattr(hipArrayMapInfo, 'resource', field(8, _anonunion14))
setattr(hipArrayMapInfo, 'subresourceType', field(72, hipArraySparseSubresourceType))
setattr(hipArrayMapInfo, 'subresource', field(80, _anonunion15))
setattr(hipArrayMapInfo, 'memOperationType', field(112, hipMemOperationType))
setattr(hipArrayMapInfo, 'memHandleType', field(116, hipMemHandleType))
setattr(hipArrayMapInfo, 'memHandle', field(120, _anonunion18))
setattr(hipArrayMapInfo, 'offset', field(128, ctypes.c_uint64))
setattr(hipArrayMapInfo, 'deviceBitMask', field(136, ctypes.c_uint32))
setattr(hipArrayMapInfo, 'flags', field(140, ctypes.c_uint32))
setattr(hipArrayMapInfo, 'reserved', field(144, Array(ctypes.c_uint32, 2)))
class hipMemcpyNodeParams(Struct): pass
class hipMemcpy3DParms(Struct): pass
class hipPos(Struct): pass
hipPos.SIZE = 24
hipPos._fields_ = ['x', 'y', 'z']
setattr(hipPos, 'x', field(0, size_t))
setattr(hipPos, 'y', field(8, size_t))
setattr(hipPos, 'z', field(16, size_t))
class hipPitchedPtr(Struct): pass
hipPitchedPtr.SIZE = 32
hipPitchedPtr._fields_ = ['ptr', 'pitch', 'xsize', 'ysize']
setattr(hipPitchedPtr, 'ptr', field(0, ctypes.c_void_p))
setattr(hipPitchedPtr, 'pitch', field(8, size_t))
setattr(hipPitchedPtr, 'xsize', field(16, size_t))
setattr(hipPitchedPtr, 'ysize', field(24, size_t))
hipMemcpyKind = CEnum(ctypes.c_uint32)
hipMemcpyHostToHost = hipMemcpyKind.define('hipMemcpyHostToHost', 0)
hipMemcpyHostToDevice = hipMemcpyKind.define('hipMemcpyHostToDevice', 1)
hipMemcpyDeviceToHost = hipMemcpyKind.define('hipMemcpyDeviceToHost', 2)
hipMemcpyDeviceToDevice = hipMemcpyKind.define('hipMemcpyDeviceToDevice', 3)
hipMemcpyDefault = hipMemcpyKind.define('hipMemcpyDefault', 4)
hipMemcpyDeviceToDeviceNoCU = hipMemcpyKind.define('hipMemcpyDeviceToDeviceNoCU', 1024)

hipMemcpy3DParms.SIZE = 160
hipMemcpy3DParms._fields_ = ['srcArray', 'srcPos', 'srcPtr', 'dstArray', 'dstPos', 'dstPtr', 'extent', 'kind']
setattr(hipMemcpy3DParms, 'srcArray', field(0, hipArray_t))
setattr(hipMemcpy3DParms, 'srcPos', field(8, hipPos))
setattr(hipMemcpy3DParms, 'srcPtr', field(32, hipPitchedPtr))
setattr(hipMemcpy3DParms, 'dstArray', field(64, hipArray_t))
setattr(hipMemcpy3DParms, 'dstPos', field(72, hipPos))
setattr(hipMemcpy3DParms, 'dstPtr', field(96, hipPitchedPtr))
setattr(hipMemcpy3DParms, 'extent', field(128, hipExtent))
setattr(hipMemcpy3DParms, 'kind', field(152, hipMemcpyKind))
hipMemcpyNodeParams.SIZE = 176
hipMemcpyNodeParams._fields_ = ['flags', 'reserved', 'copyParams']
setattr(hipMemcpyNodeParams, 'flags', field(0, ctypes.c_int32))
setattr(hipMemcpyNodeParams, 'reserved', field(4, Array(ctypes.c_int32, 3)))
setattr(hipMemcpyNodeParams, 'copyParams', field(16, hipMemcpy3DParms))
class hipChildGraphNodeParams(Struct): pass
hipChildGraphNodeParams.SIZE = 8
hipChildGraphNodeParams._fields_ = ['graph']
setattr(hipChildGraphNodeParams, 'graph', field(0, hipGraph_t))
class hipEventWaitNodeParams(Struct): pass
hipEventWaitNodeParams.SIZE = 8
hipEventWaitNodeParams._fields_ = ['event']
setattr(hipEventWaitNodeParams, 'event', field(0, hipEvent_t))
class hipEventRecordNodeParams(Struct): pass
hipEventRecordNodeParams.SIZE = 8
hipEventRecordNodeParams._fields_ = ['event']
setattr(hipEventRecordNodeParams, 'event', field(0, hipEvent_t))
class hipMemFreeNodeParams(Struct): pass
hipMemFreeNodeParams.SIZE = 8
hipMemFreeNodeParams._fields_ = ['dptr']
setattr(hipMemFreeNodeParams, 'dptr', field(0, ctypes.c_void_p))
class hipGraphNodeParams(Struct): pass
hipGraphNodeParams.SIZE = 256
hipGraphNodeParams._fields_ = ['type', 'reserved0', 'reserved1', 'kernel', 'memcpy', 'memset', 'host', 'graph', 'eventWait', 'eventRecord', 'extSemSignal', 'extSemWait', 'alloc', 'free', 'reserved2']
setattr(hipGraphNodeParams, 'type', field(0, hipGraphNodeType))
setattr(hipGraphNodeParams, 'reserved0', field(4, Array(ctypes.c_int32, 3)))
setattr(hipGraphNodeParams, 'reserved1', field(16, Array(ctypes.c_int64, 29)))
setattr(hipGraphNodeParams, 'kernel', field(16, hipKernelNodeParams))
setattr(hipGraphNodeParams, 'memcpy', field(16, hipMemcpyNodeParams))
setattr(hipGraphNodeParams, 'memset', field(16, hipMemsetParams))
setattr(hipGraphNodeParams, 'host', field(16, hipHostNodeParams))
setattr(hipGraphNodeParams, 'graph', field(16, hipChildGraphNodeParams))
setattr(hipGraphNodeParams, 'eventWait', field(16, hipEventWaitNodeParams))
setattr(hipGraphNodeParams, 'eventRecord', field(16, hipEventRecordNodeParams))
setattr(hipGraphNodeParams, 'extSemSignal', field(16, hipExternalSemaphoreSignalNodeParams))
setattr(hipGraphNodeParams, 'extSemWait', field(16, hipExternalSemaphoreWaitNodeParams))
setattr(hipGraphNodeParams, 'alloc', field(16, hipMemAllocNodeParams))
setattr(hipGraphNodeParams, 'free', field(16, hipMemFreeNodeParams))
setattr(hipGraphNodeParams, 'reserved2', field(248, ctypes.c_int64))
hipGraphDependencyType = CEnum(ctypes.c_uint32)
hipGraphDependencyTypeDefault = hipGraphDependencyType.define('hipGraphDependencyTypeDefault', 0)
hipGraphDependencyTypeProgrammatic = hipGraphDependencyType.define('hipGraphDependencyTypeProgrammatic', 1)

class hipGraphEdgeData(Struct): pass
hipGraphEdgeData.SIZE = 8
hipGraphEdgeData._fields_ = ['from_port', 'reserved', 'to_port', 'type']
setattr(hipGraphEdgeData, 'from_port', field(0, ctypes.c_ubyte))
setattr(hipGraphEdgeData, 'reserved', field(1, Array(ctypes.c_ubyte, 5)))
setattr(hipGraphEdgeData, 'to_port', field(6, ctypes.c_ubyte))
setattr(hipGraphEdgeData, 'type', field(7, ctypes.c_ubyte))
try: (hipInit:=dll.hipInit).restype, hipInit.argtypes = hipError_t, [ctypes.c_uint32]
except AttributeError: pass

try: (hipDriverGetVersion:=dll.hipDriverGetVersion).restype, hipDriverGetVersion.argtypes = hipError_t, [Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipRuntimeGetVersion:=dll.hipRuntimeGetVersion).restype, hipRuntimeGetVersion.argtypes = hipError_t, [Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipDeviceGet:=dll.hipDeviceGet).restype, hipDeviceGet.argtypes = hipError_t, [Pointer(hipDevice_t), ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceComputeCapability:=dll.hipDeviceComputeCapability).restype, hipDeviceComputeCapability.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), hipDevice_t]
except AttributeError: pass

try: (hipDeviceGetName:=dll.hipDeviceGetName).restype, hipDeviceGetName.argtypes = hipError_t, [Pointer(ctypes.c_char), ctypes.c_int32, hipDevice_t]
except AttributeError: pass

try: (hipDeviceGetUuid:=dll.hipDeviceGetUuid).restype, hipDeviceGetUuid.argtypes = hipError_t, [Pointer(hipUUID), hipDevice_t]
except AttributeError: pass

try: (hipDeviceGetP2PAttribute:=dll.hipDeviceGetP2PAttribute).restype, hipDeviceGetP2PAttribute.argtypes = hipError_t, [Pointer(ctypes.c_int32), hipDeviceP2PAttr, ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceGetPCIBusId:=dll.hipDeviceGetPCIBusId).restype, hipDeviceGetPCIBusId.argtypes = hipError_t, [Pointer(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceGetByPCIBusId:=dll.hipDeviceGetByPCIBusId).restype, hipDeviceGetByPCIBusId.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hipDeviceTotalMem:=dll.hipDeviceTotalMem).restype, hipDeviceTotalMem.argtypes = hipError_t, [Pointer(size_t), hipDevice_t]
except AttributeError: pass

try: (hipDeviceSynchronize:=dll.hipDeviceSynchronize).restype, hipDeviceSynchronize.argtypes = hipError_t, []
except AttributeError: pass

try: (hipDeviceReset:=dll.hipDeviceReset).restype, hipDeviceReset.argtypes = hipError_t, []
except AttributeError: pass

try: (hipSetDevice:=dll.hipSetDevice).restype, hipSetDevice.argtypes = hipError_t, [ctypes.c_int32]
except AttributeError: pass

try: (hipSetValidDevices:=dll.hipSetValidDevices).restype, hipSetValidDevices.argtypes = hipError_t, [Pointer(ctypes.c_int32), ctypes.c_int32]
except AttributeError: pass

try: (hipGetDevice:=dll.hipGetDevice).restype, hipGetDevice.argtypes = hipError_t, [Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipGetDeviceCount:=dll.hipGetDeviceCount).restype, hipGetDeviceCount.argtypes = hipError_t, [Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipDeviceGetAttribute:=dll.hipDeviceGetAttribute).restype, hipDeviceGetAttribute.argtypes = hipError_t, [Pointer(ctypes.c_int32), hipDeviceAttribute_t, ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceGetDefaultMemPool:=dll.hipDeviceGetDefaultMemPool).restype, hipDeviceGetDefaultMemPool.argtypes = hipError_t, [Pointer(hipMemPool_t), ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceSetMemPool:=dll.hipDeviceSetMemPool).restype, hipDeviceSetMemPool.argtypes = hipError_t, [ctypes.c_int32, hipMemPool_t]
except AttributeError: pass

try: (hipDeviceGetMemPool:=dll.hipDeviceGetMemPool).restype, hipDeviceGetMemPool.argtypes = hipError_t, [Pointer(hipMemPool_t), ctypes.c_int32]
except AttributeError: pass

try: (hipGetDevicePropertiesR0600:=dll.hipGetDevicePropertiesR0600).restype, hipGetDevicePropertiesR0600.argtypes = hipError_t, [Pointer(hipDeviceProp_tR0600), ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceSetCacheConfig:=dll.hipDeviceSetCacheConfig).restype, hipDeviceSetCacheConfig.argtypes = hipError_t, [hipFuncCache_t]
except AttributeError: pass

try: (hipDeviceGetCacheConfig:=dll.hipDeviceGetCacheConfig).restype, hipDeviceGetCacheConfig.argtypes = hipError_t, [Pointer(hipFuncCache_t)]
except AttributeError: pass

try: (hipDeviceGetLimit:=dll.hipDeviceGetLimit).restype, hipDeviceGetLimit.argtypes = hipError_t, [Pointer(size_t), hipLimit_t]
except AttributeError: pass

try: (hipDeviceSetLimit:=dll.hipDeviceSetLimit).restype, hipDeviceSetLimit.argtypes = hipError_t, [hipLimit_t, size_t]
except AttributeError: pass

try: (hipDeviceGetSharedMemConfig:=dll.hipDeviceGetSharedMemConfig).restype, hipDeviceGetSharedMemConfig.argtypes = hipError_t, [Pointer(hipSharedMemConfig)]
except AttributeError: pass

try: (hipGetDeviceFlags:=dll.hipGetDeviceFlags).restype, hipGetDeviceFlags.argtypes = hipError_t, [Pointer(ctypes.c_uint32)]
except AttributeError: pass

try: (hipDeviceSetSharedMemConfig:=dll.hipDeviceSetSharedMemConfig).restype, hipDeviceSetSharedMemConfig.argtypes = hipError_t, [hipSharedMemConfig]
except AttributeError: pass

try: (hipSetDeviceFlags:=dll.hipSetDeviceFlags).restype, hipSetDeviceFlags.argtypes = hipError_t, [ctypes.c_uint32]
except AttributeError: pass

try: (hipChooseDeviceR0600:=dll.hipChooseDeviceR0600).restype, hipChooseDeviceR0600.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(hipDeviceProp_tR0600)]
except AttributeError: pass

try: (hipExtGetLinkTypeAndHopCount:=dll.hipExtGetLinkTypeAndHopCount).restype, hipExtGetLinkTypeAndHopCount.argtypes = hipError_t, [ctypes.c_int32, ctypes.c_int32, Pointer(uint32_t), Pointer(uint32_t)]
except AttributeError: pass

try: (hipIpcGetMemHandle:=dll.hipIpcGetMemHandle).restype, hipIpcGetMemHandle.argtypes = hipError_t, [Pointer(hipIpcMemHandle_t), ctypes.c_void_p]
except AttributeError: pass

try: (hipIpcOpenMemHandle:=dll.hipIpcOpenMemHandle).restype, hipIpcOpenMemHandle.argtypes = hipError_t, [Pointer(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipIpcCloseMemHandle:=dll.hipIpcCloseMemHandle).restype, hipIpcCloseMemHandle.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hipIpcGetEventHandle:=dll.hipIpcGetEventHandle).restype, hipIpcGetEventHandle.argtypes = hipError_t, [Pointer(hipIpcEventHandle_t), hipEvent_t]
except AttributeError: pass

try: (hipIpcOpenEventHandle:=dll.hipIpcOpenEventHandle).restype, hipIpcOpenEventHandle.argtypes = hipError_t, [Pointer(hipEvent_t), hipIpcEventHandle_t]
except AttributeError: pass

try: (hipFuncSetAttribute:=dll.hipFuncSetAttribute).restype, hipFuncSetAttribute.argtypes = hipError_t, [ctypes.c_void_p, hipFuncAttribute, ctypes.c_int32]
except AttributeError: pass

try: (hipFuncSetCacheConfig:=dll.hipFuncSetCacheConfig).restype, hipFuncSetCacheConfig.argtypes = hipError_t, [ctypes.c_void_p, hipFuncCache_t]
except AttributeError: pass

try: (hipFuncSetSharedMemConfig:=dll.hipFuncSetSharedMemConfig).restype, hipFuncSetSharedMemConfig.argtypes = hipError_t, [ctypes.c_void_p, hipSharedMemConfig]
except AttributeError: pass

try: (hipGetLastError:=dll.hipGetLastError).restype, hipGetLastError.argtypes = hipError_t, []
except AttributeError: pass

try: (hipExtGetLastError:=dll.hipExtGetLastError).restype, hipExtGetLastError.argtypes = hipError_t, []
except AttributeError: pass

try: (hipPeekAtLastError:=dll.hipPeekAtLastError).restype, hipPeekAtLastError.argtypes = hipError_t, []
except AttributeError: pass

try: (hipGetErrorName:=dll.hipGetErrorName).restype, hipGetErrorName.argtypes = Pointer(ctypes.c_char), [hipError_t]
except AttributeError: pass

try: (hipGetErrorString:=dll.hipGetErrorString).restype, hipGetErrorString.argtypes = Pointer(ctypes.c_char), [hipError_t]
except AttributeError: pass

try: (hipDrvGetErrorName:=dll.hipDrvGetErrorName).restype, hipDrvGetErrorName.argtypes = hipError_t, [hipError_t, Pointer(Pointer(ctypes.c_char))]
except AttributeError: pass

try: (hipDrvGetErrorString:=dll.hipDrvGetErrorString).restype, hipDrvGetErrorString.argtypes = hipError_t, [hipError_t, Pointer(Pointer(ctypes.c_char))]
except AttributeError: pass

try: (hipStreamCreate:=dll.hipStreamCreate).restype, hipStreamCreate.argtypes = hipError_t, [Pointer(hipStream_t)]
except AttributeError: pass

try: (hipStreamCreateWithFlags:=dll.hipStreamCreateWithFlags).restype, hipStreamCreateWithFlags.argtypes = hipError_t, [Pointer(hipStream_t), ctypes.c_uint32]
except AttributeError: pass

try: (hipStreamCreateWithPriority:=dll.hipStreamCreateWithPriority).restype, hipStreamCreateWithPriority.argtypes = hipError_t, [Pointer(hipStream_t), ctypes.c_uint32, ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceGetStreamPriorityRange:=dll.hipDeviceGetStreamPriorityRange).restype, hipDeviceGetStreamPriorityRange.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipStreamDestroy:=dll.hipStreamDestroy).restype, hipStreamDestroy.argtypes = hipError_t, [hipStream_t]
except AttributeError: pass

try: (hipStreamQuery:=dll.hipStreamQuery).restype, hipStreamQuery.argtypes = hipError_t, [hipStream_t]
except AttributeError: pass

try: (hipStreamSynchronize:=dll.hipStreamSynchronize).restype, hipStreamSynchronize.argtypes = hipError_t, [hipStream_t]
except AttributeError: pass

try: (hipStreamWaitEvent:=dll.hipStreamWaitEvent).restype, hipStreamWaitEvent.argtypes = hipError_t, [hipStream_t, hipEvent_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipStreamGetFlags:=dll.hipStreamGetFlags).restype, hipStreamGetFlags.argtypes = hipError_t, [hipStream_t, Pointer(ctypes.c_uint32)]
except AttributeError: pass

try: (hipStreamGetPriority:=dll.hipStreamGetPriority).restype, hipStreamGetPriority.argtypes = hipError_t, [hipStream_t, Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipStreamGetDevice:=dll.hipStreamGetDevice).restype, hipStreamGetDevice.argtypes = hipError_t, [hipStream_t, Pointer(hipDevice_t)]
except AttributeError: pass

try: (hipExtStreamCreateWithCUMask:=dll.hipExtStreamCreateWithCUMask).restype, hipExtStreamCreateWithCUMask.argtypes = hipError_t, [Pointer(hipStream_t), uint32_t, Pointer(uint32_t)]
except AttributeError: pass

try: (hipExtStreamGetCUMask:=dll.hipExtStreamGetCUMask).restype, hipExtStreamGetCUMask.argtypes = hipError_t, [hipStream_t, uint32_t, Pointer(uint32_t)]
except AttributeError: pass

hipStreamCallback_t = ctypes.CFUNCTYPE(None, Pointer(ihipStream_t), hipError_t, ctypes.c_void_p)
try: (hipStreamAddCallback:=dll.hipStreamAddCallback).restype, hipStreamAddCallback.argtypes = hipError_t, [hipStream_t, hipStreamCallback_t, ctypes.c_void_p, ctypes.c_uint32]
except AttributeError: pass

try: (hipStreamWaitValue32:=dll.hipStreamWaitValue32).restype, hipStreamWaitValue32.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32, uint32_t]
except AttributeError: pass

uint64_t = ctypes.c_uint64
try: (hipStreamWaitValue64:=dll.hipStreamWaitValue64).restype, hipStreamWaitValue64.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32, uint64_t]
except AttributeError: pass

try: (hipStreamWriteValue32:=dll.hipStreamWriteValue32).restype, hipStreamWriteValue32.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipStreamWriteValue64:=dll.hipStreamWriteValue64).restype, hipStreamWriteValue64.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipEventCreateWithFlags:=dll.hipEventCreateWithFlags).restype, hipEventCreateWithFlags.argtypes = hipError_t, [Pointer(hipEvent_t), ctypes.c_uint32]
except AttributeError: pass

try: (hipEventCreate:=dll.hipEventCreate).restype, hipEventCreate.argtypes = hipError_t, [Pointer(hipEvent_t)]
except AttributeError: pass

try: (hipEventRecord:=dll.hipEventRecord).restype, hipEventRecord.argtypes = hipError_t, [hipEvent_t, hipStream_t]
except AttributeError: pass

try: (hipEventDestroy:=dll.hipEventDestroy).restype, hipEventDestroy.argtypes = hipError_t, [hipEvent_t]
except AttributeError: pass

try: (hipEventSynchronize:=dll.hipEventSynchronize).restype, hipEventSynchronize.argtypes = hipError_t, [hipEvent_t]
except AttributeError: pass

try: (hipEventElapsedTime:=dll.hipEventElapsedTime).restype, hipEventElapsedTime.argtypes = hipError_t, [Pointer(ctypes.c_float), hipEvent_t, hipEvent_t]
except AttributeError: pass

try: (hipEventQuery:=dll.hipEventQuery).restype, hipEventQuery.argtypes = hipError_t, [hipEvent_t]
except AttributeError: pass

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

try: (hipPointerSetAttribute:=dll.hipPointerSetAttribute).restype, hipPointerSetAttribute.argtypes = hipError_t, [ctypes.c_void_p, hipPointer_attribute, hipDeviceptr_t]
except AttributeError: pass

try: (hipPointerGetAttributes:=dll.hipPointerGetAttributes).restype, hipPointerGetAttributes.argtypes = hipError_t, [Pointer(hipPointerAttribute_t), ctypes.c_void_p]
except AttributeError: pass

try: (hipPointerGetAttribute:=dll.hipPointerGetAttribute).restype, hipPointerGetAttribute.argtypes = hipError_t, [ctypes.c_void_p, hipPointer_attribute, hipDeviceptr_t]
except AttributeError: pass

try: (hipDrvPointerGetAttributes:=dll.hipDrvPointerGetAttributes).restype, hipDrvPointerGetAttributes.argtypes = hipError_t, [ctypes.c_uint32, Pointer(hipPointer_attribute), Pointer(ctypes.c_void_p), hipDeviceptr_t]
except AttributeError: pass

try: (hipImportExternalSemaphore:=dll.hipImportExternalSemaphore).restype, hipImportExternalSemaphore.argtypes = hipError_t, [Pointer(hipExternalSemaphore_t), Pointer(hipExternalSemaphoreHandleDesc)]
except AttributeError: pass

try: (hipSignalExternalSemaphoresAsync:=dll.hipSignalExternalSemaphoresAsync).restype, hipSignalExternalSemaphoresAsync.argtypes = hipError_t, [Pointer(hipExternalSemaphore_t), Pointer(hipExternalSemaphoreSignalParams), ctypes.c_uint32, hipStream_t]
except AttributeError: pass

try: (hipWaitExternalSemaphoresAsync:=dll.hipWaitExternalSemaphoresAsync).restype, hipWaitExternalSemaphoresAsync.argtypes = hipError_t, [Pointer(hipExternalSemaphore_t), Pointer(hipExternalSemaphoreWaitParams), ctypes.c_uint32, hipStream_t]
except AttributeError: pass

try: (hipDestroyExternalSemaphore:=dll.hipDestroyExternalSemaphore).restype, hipDestroyExternalSemaphore.argtypes = hipError_t, [hipExternalSemaphore_t]
except AttributeError: pass

try: (hipImportExternalMemory:=dll.hipImportExternalMemory).restype, hipImportExternalMemory.argtypes = hipError_t, [Pointer(hipExternalMemory_t), Pointer(hipExternalMemoryHandleDesc)]
except AttributeError: pass

try: (hipExternalMemoryGetMappedBuffer:=dll.hipExternalMemoryGetMappedBuffer).restype, hipExternalMemoryGetMappedBuffer.argtypes = hipError_t, [Pointer(ctypes.c_void_p), hipExternalMemory_t, Pointer(hipExternalMemoryBufferDesc)]
except AttributeError: pass

try: (hipDestroyExternalMemory:=dll.hipDestroyExternalMemory).restype, hipDestroyExternalMemory.argtypes = hipError_t, [hipExternalMemory_t]
except AttributeError: pass

hipMipmappedArray_t = Pointer(hipMipmappedArray)
try: (hipExternalMemoryGetMappedMipmappedArray:=dll.hipExternalMemoryGetMappedMipmappedArray).restype, hipExternalMemoryGetMappedMipmappedArray.argtypes = hipError_t, [Pointer(hipMipmappedArray_t), hipExternalMemory_t, Pointer(hipExternalMemoryMipmappedArrayDesc)]
except AttributeError: pass

try: (hipMalloc:=dll.hipMalloc).restype, hipMalloc.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t]
except AttributeError: pass

try: (hipExtMallocWithFlags:=dll.hipExtMallocWithFlags).restype, hipExtMallocWithFlags.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipMallocHost:=dll.hipMallocHost).restype, hipMallocHost.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t]
except AttributeError: pass

try: (hipMemAllocHost:=dll.hipMemAllocHost).restype, hipMemAllocHost.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t]
except AttributeError: pass

try: (hipHostMalloc:=dll.hipHostMalloc).restype, hipHostMalloc.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipMallocManaged:=dll.hipMallocManaged).restype, hipMallocManaged.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipMemPrefetchAsync:=dll.hipMemPrefetchAsync).restype, hipMemPrefetchAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_int32, hipStream_t]
except AttributeError: pass

try: (hipMemAdvise:=dll.hipMemAdvise).restype, hipMemAdvise.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipMemoryAdvise, ctypes.c_int32]
except AttributeError: pass

try: (hipMemRangeGetAttribute:=dll.hipMemRangeGetAttribute).restype, hipMemRangeGetAttribute.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipMemRangeAttribute, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hipMemRangeGetAttributes:=dll.hipMemRangeGetAttributes).restype, hipMemRangeGetAttributes.argtypes = hipError_t, [Pointer(ctypes.c_void_p), Pointer(size_t), Pointer(hipMemRangeAttribute), size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hipStreamAttachMemAsync:=dll.hipStreamAttachMemAsync).restype, hipStreamAttachMemAsync.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipMallocAsync:=dll.hipMallocAsync).restype, hipMallocAsync.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, hipStream_t]
except AttributeError: pass

try: (hipFreeAsync:=dll.hipFreeAsync).restype, hipFreeAsync.argtypes = hipError_t, [ctypes.c_void_p, hipStream_t]
except AttributeError: pass

try: (hipMemPoolTrimTo:=dll.hipMemPoolTrimTo).restype, hipMemPoolTrimTo.argtypes = hipError_t, [hipMemPool_t, size_t]
except AttributeError: pass

try: (hipMemPoolSetAttribute:=dll.hipMemPoolSetAttribute).restype, hipMemPoolSetAttribute.argtypes = hipError_t, [hipMemPool_t, hipMemPoolAttr, ctypes.c_void_p]
except AttributeError: pass

try: (hipMemPoolGetAttribute:=dll.hipMemPoolGetAttribute).restype, hipMemPoolGetAttribute.argtypes = hipError_t, [hipMemPool_t, hipMemPoolAttr, ctypes.c_void_p]
except AttributeError: pass

try: (hipMemPoolSetAccess:=dll.hipMemPoolSetAccess).restype, hipMemPoolSetAccess.argtypes = hipError_t, [hipMemPool_t, Pointer(hipMemAccessDesc), size_t]
except AttributeError: pass

try: (hipMemPoolGetAccess:=dll.hipMemPoolGetAccess).restype, hipMemPoolGetAccess.argtypes = hipError_t, [Pointer(hipMemAccessFlags), hipMemPool_t, Pointer(hipMemLocation)]
except AttributeError: pass

try: (hipMemPoolCreate:=dll.hipMemPoolCreate).restype, hipMemPoolCreate.argtypes = hipError_t, [Pointer(hipMemPool_t), Pointer(hipMemPoolProps)]
except AttributeError: pass

try: (hipMemPoolDestroy:=dll.hipMemPoolDestroy).restype, hipMemPoolDestroy.argtypes = hipError_t, [hipMemPool_t]
except AttributeError: pass

try: (hipMallocFromPoolAsync:=dll.hipMallocFromPoolAsync).restype, hipMallocFromPoolAsync.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, hipMemPool_t, hipStream_t]
except AttributeError: pass

try: (hipMemPoolExportToShareableHandle:=dll.hipMemPoolExportToShareableHandle).restype, hipMemPoolExportToShareableHandle.argtypes = hipError_t, [ctypes.c_void_p, hipMemPool_t, hipMemAllocationHandleType, ctypes.c_uint32]
except AttributeError: pass

try: (hipMemPoolImportFromShareableHandle:=dll.hipMemPoolImportFromShareableHandle).restype, hipMemPoolImportFromShareableHandle.argtypes = hipError_t, [Pointer(hipMemPool_t), ctypes.c_void_p, hipMemAllocationHandleType, ctypes.c_uint32]
except AttributeError: pass

try: (hipMemPoolExportPointer:=dll.hipMemPoolExportPointer).restype, hipMemPoolExportPointer.argtypes = hipError_t, [Pointer(hipMemPoolPtrExportData), ctypes.c_void_p]
except AttributeError: pass

try: (hipMemPoolImportPointer:=dll.hipMemPoolImportPointer).restype, hipMemPoolImportPointer.argtypes = hipError_t, [Pointer(ctypes.c_void_p), hipMemPool_t, Pointer(hipMemPoolPtrExportData)]
except AttributeError: pass

try: (hipHostAlloc:=dll.hipHostAlloc).restype, hipHostAlloc.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipHostGetDevicePointer:=dll.hipHostGetDevicePointer).restype, hipHostGetDevicePointer.argtypes = hipError_t, [Pointer(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32]
except AttributeError: pass

try: (hipHostGetFlags:=dll.hipHostGetFlags).restype, hipHostGetFlags.argtypes = hipError_t, [Pointer(ctypes.c_uint32), ctypes.c_void_p]
except AttributeError: pass

try: (hipHostRegister:=dll.hipHostRegister).restype, hipHostRegister.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipHostUnregister:=dll.hipHostUnregister).restype, hipHostUnregister.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hipMallocPitch:=dll.hipMallocPitch).restype, hipMallocPitch.argtypes = hipError_t, [Pointer(ctypes.c_void_p), Pointer(size_t), size_t, size_t]
except AttributeError: pass

try: (hipMemAllocPitch:=dll.hipMemAllocPitch).restype, hipMemAllocPitch.argtypes = hipError_t, [Pointer(hipDeviceptr_t), Pointer(size_t), size_t, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipFree:=dll.hipFree).restype, hipFree.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hipFreeHost:=dll.hipFreeHost).restype, hipFreeHost.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hipHostFree:=dll.hipHostFree).restype, hipHostFree.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hipMemcpy:=dll.hipMemcpy).restype, hipMemcpy.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpyWithStream:=dll.hipMemcpyWithStream).restype, hipMemcpyWithStream.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

try: (hipMemcpyHtoD:=dll.hipMemcpyHtoD).restype, hipMemcpyHtoD.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hipMemcpyDtoH:=dll.hipMemcpyDtoH).restype, hipMemcpyDtoH.argtypes = hipError_t, [ctypes.c_void_p, hipDeviceptr_t, size_t]
except AttributeError: pass

try: (hipMemcpyDtoD:=dll.hipMemcpyDtoD).restype, hipMemcpyDtoD.argtypes = hipError_t, [hipDeviceptr_t, hipDeviceptr_t, size_t]
except AttributeError: pass

try: (hipMemcpyAtoD:=dll.hipMemcpyAtoD).restype, hipMemcpyAtoD.argtypes = hipError_t, [hipDeviceptr_t, hipArray_t, size_t, size_t]
except AttributeError: pass

try: (hipMemcpyDtoA:=dll.hipMemcpyDtoA).restype, hipMemcpyDtoA.argtypes = hipError_t, [hipArray_t, size_t, hipDeviceptr_t, size_t]
except AttributeError: pass

try: (hipMemcpyAtoA:=dll.hipMemcpyAtoA).restype, hipMemcpyAtoA.argtypes = hipError_t, [hipArray_t, size_t, hipArray_t, size_t, size_t]
except AttributeError: pass

try: (hipMemcpyHtoDAsync:=dll.hipMemcpyHtoDAsync).restype, hipMemcpyHtoDAsync.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_void_p, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemcpyDtoHAsync:=dll.hipMemcpyDtoHAsync).restype, hipMemcpyDtoHAsync.argtypes = hipError_t, [ctypes.c_void_p, hipDeviceptr_t, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemcpyDtoDAsync:=dll.hipMemcpyDtoDAsync).restype, hipMemcpyDtoDAsync.argtypes = hipError_t, [hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemcpyAtoHAsync:=dll.hipMemcpyAtoHAsync).restype, hipMemcpyAtoHAsync.argtypes = hipError_t, [ctypes.c_void_p, hipArray_t, size_t, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemcpyHtoAAsync:=dll.hipMemcpyHtoAAsync).restype, hipMemcpyHtoAAsync.argtypes = hipError_t, [hipArray_t, size_t, ctypes.c_void_p, size_t, hipStream_t]
except AttributeError: pass

try: (hipModuleGetGlobal:=dll.hipModuleGetGlobal).restype, hipModuleGetGlobal.argtypes = hipError_t, [Pointer(hipDeviceptr_t), Pointer(size_t), hipModule_t, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hipGetSymbolAddress:=dll.hipGetSymbolAddress).restype, hipGetSymbolAddress.argtypes = hipError_t, [Pointer(ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hipGetSymbolSize:=dll.hipGetSymbolSize).restype, hipGetSymbolSize.argtypes = hipError_t, [Pointer(size_t), ctypes.c_void_p]
except AttributeError: pass

try: (hipGetProcAddress:=dll.hipGetProcAddress).restype, hipGetProcAddress.argtypes = hipError_t, [Pointer(ctypes.c_char), Pointer(ctypes.c_void_p), ctypes.c_int32, uint64_t, Pointer(hipDriverProcAddressQueryResult)]
except AttributeError: pass

try: (hipMemcpyToSymbol:=dll.hipMemcpyToSymbol).restype, hipMemcpyToSymbol.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpyToSymbolAsync:=dll.hipMemcpyToSymbolAsync).restype, hipMemcpyToSymbolAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

try: (hipMemcpyFromSymbol:=dll.hipMemcpyFromSymbol).restype, hipMemcpyFromSymbol.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpyFromSymbolAsync:=dll.hipMemcpyFromSymbolAsync).restype, hipMemcpyFromSymbolAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

try: (hipMemcpyAsync:=dll.hipMemcpyAsync).restype, hipMemcpyAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

try: (hipMemset:=dll.hipMemset).restype, hipMemset.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int32, size_t]
except AttributeError: pass

try: (hipMemsetD8:=dll.hipMemsetD8).restype, hipMemsetD8.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_ubyte, size_t]
except AttributeError: pass

try: (hipMemsetD8Async:=dll.hipMemsetD8Async).restype, hipMemsetD8Async.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_ubyte, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemsetD16:=dll.hipMemsetD16).restype, hipMemsetD16.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_uint16, size_t]
except AttributeError: pass

try: (hipMemsetD16Async:=dll.hipMemsetD16Async).restype, hipMemsetD16Async.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_uint16, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemsetD32:=dll.hipMemsetD32).restype, hipMemsetD32.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_int32, size_t]
except AttributeError: pass

try: (hipMemsetAsync:=dll.hipMemsetAsync).restype, hipMemsetAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemsetD32Async:=dll.hipMemsetD32Async).restype, hipMemsetD32Async.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_int32, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemset2D:=dll.hipMemset2D).restype, hipMemset2D.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t]
except AttributeError: pass

try: (hipMemset2DAsync:=dll.hipMemset2DAsync).restype, hipMemset2DAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t, hipStream_t]
except AttributeError: pass

try: (hipMemset3D:=dll.hipMemset3D).restype, hipMemset3D.argtypes = hipError_t, [hipPitchedPtr, ctypes.c_int32, hipExtent]
except AttributeError: pass

try: (hipMemset3DAsync:=dll.hipMemset3DAsync).restype, hipMemset3DAsync.argtypes = hipError_t, [hipPitchedPtr, ctypes.c_int32, hipExtent, hipStream_t]
except AttributeError: pass

try: (hipMemGetInfo:=dll.hipMemGetInfo).restype, hipMemGetInfo.argtypes = hipError_t, [Pointer(size_t), Pointer(size_t)]
except AttributeError: pass

try: (hipMemPtrGetInfo:=dll.hipMemPtrGetInfo).restype, hipMemPtrGetInfo.argtypes = hipError_t, [ctypes.c_void_p, Pointer(size_t)]
except AttributeError: pass

try: (hipMallocArray:=dll.hipMallocArray).restype, hipMallocArray.argtypes = hipError_t, [Pointer(hipArray_t), Pointer(hipChannelFormatDesc), size_t, size_t, ctypes.c_uint32]
except AttributeError: pass

class HIP_ARRAY_DESCRIPTOR(Struct): pass
HIP_ARRAY_DESCRIPTOR.SIZE = 24
HIP_ARRAY_DESCRIPTOR._fields_ = ['Width', 'Height', 'Format', 'NumChannels']
setattr(HIP_ARRAY_DESCRIPTOR, 'Width', field(0, size_t))
setattr(HIP_ARRAY_DESCRIPTOR, 'Height', field(8, size_t))
setattr(HIP_ARRAY_DESCRIPTOR, 'Format', field(16, hipArray_Format))
setattr(HIP_ARRAY_DESCRIPTOR, 'NumChannels', field(20, ctypes.c_uint32))
try: (hipArrayCreate:=dll.hipArrayCreate).restype, hipArrayCreate.argtypes = hipError_t, [Pointer(hipArray_t), Pointer(HIP_ARRAY_DESCRIPTOR)]
except AttributeError: pass

try: (hipArrayDestroy:=dll.hipArrayDestroy).restype, hipArrayDestroy.argtypes = hipError_t, [hipArray_t]
except AttributeError: pass

class HIP_ARRAY3D_DESCRIPTOR(Struct): pass
HIP_ARRAY3D_DESCRIPTOR.SIZE = 40
HIP_ARRAY3D_DESCRIPTOR._fields_ = ['Width', 'Height', 'Depth', 'Format', 'NumChannels', 'Flags']
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Width', field(0, size_t))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Height', field(8, size_t))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Depth', field(16, size_t))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Format', field(24, hipArray_Format))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'NumChannels', field(28, ctypes.c_uint32))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Flags', field(32, ctypes.c_uint32))
try: (hipArray3DCreate:=dll.hipArray3DCreate).restype, hipArray3DCreate.argtypes = hipError_t, [Pointer(hipArray_t), Pointer(HIP_ARRAY3D_DESCRIPTOR)]
except AttributeError: pass

try: (hipMalloc3D:=dll.hipMalloc3D).restype, hipMalloc3D.argtypes = hipError_t, [Pointer(hipPitchedPtr), hipExtent]
except AttributeError: pass

try: (hipFreeArray:=dll.hipFreeArray).restype, hipFreeArray.argtypes = hipError_t, [hipArray_t]
except AttributeError: pass

try: (hipMalloc3DArray:=dll.hipMalloc3DArray).restype, hipMalloc3DArray.argtypes = hipError_t, [Pointer(hipArray_t), Pointer(hipChannelFormatDesc), hipExtent, ctypes.c_uint32]
except AttributeError: pass

try: (hipArrayGetInfo:=dll.hipArrayGetInfo).restype, hipArrayGetInfo.argtypes = hipError_t, [Pointer(hipChannelFormatDesc), Pointer(hipExtent), Pointer(ctypes.c_uint32), hipArray_t]
except AttributeError: pass

try: (hipArrayGetDescriptor:=dll.hipArrayGetDescriptor).restype, hipArrayGetDescriptor.argtypes = hipError_t, [Pointer(HIP_ARRAY_DESCRIPTOR), hipArray_t]
except AttributeError: pass

try: (hipArray3DGetDescriptor:=dll.hipArray3DGetDescriptor).restype, hipArray3DGetDescriptor.argtypes = hipError_t, [Pointer(HIP_ARRAY3D_DESCRIPTOR), hipArray_t]
except AttributeError: pass

try: (hipMemcpy2D:=dll.hipMemcpy2D).restype, hipMemcpy2D.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

class hip_Memcpy2D(Struct): pass
hip_Memcpy2D.SIZE = 128
hip_Memcpy2D._fields_ = ['srcXInBytes', 'srcY', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'srcPitch', 'dstXInBytes', 'dstY', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'dstPitch', 'WidthInBytes', 'Height']
setattr(hip_Memcpy2D, 'srcXInBytes', field(0, size_t))
setattr(hip_Memcpy2D, 'srcY', field(8, size_t))
setattr(hip_Memcpy2D, 'srcMemoryType', field(16, hipMemoryType))
setattr(hip_Memcpy2D, 'srcHost', field(24, ctypes.c_void_p))
setattr(hip_Memcpy2D, 'srcDevice', field(32, hipDeviceptr_t))
setattr(hip_Memcpy2D, 'srcArray', field(40, hipArray_t))
setattr(hip_Memcpy2D, 'srcPitch', field(48, size_t))
setattr(hip_Memcpy2D, 'dstXInBytes', field(56, size_t))
setattr(hip_Memcpy2D, 'dstY', field(64, size_t))
setattr(hip_Memcpy2D, 'dstMemoryType', field(72, hipMemoryType))
setattr(hip_Memcpy2D, 'dstHost', field(80, ctypes.c_void_p))
setattr(hip_Memcpy2D, 'dstDevice', field(88, hipDeviceptr_t))
setattr(hip_Memcpy2D, 'dstArray', field(96, hipArray_t))
setattr(hip_Memcpy2D, 'dstPitch', field(104, size_t))
setattr(hip_Memcpy2D, 'WidthInBytes', field(112, size_t))
setattr(hip_Memcpy2D, 'Height', field(120, size_t))
try: (hipMemcpyParam2D:=dll.hipMemcpyParam2D).restype, hipMemcpyParam2D.argtypes = hipError_t, [Pointer(hip_Memcpy2D)]
except AttributeError: pass

try: (hipMemcpyParam2DAsync:=dll.hipMemcpyParam2DAsync).restype, hipMemcpyParam2DAsync.argtypes = hipError_t, [Pointer(hip_Memcpy2D), hipStream_t]
except AttributeError: pass

try: (hipMemcpy2DAsync:=dll.hipMemcpy2DAsync).restype, hipMemcpy2DAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

try: (hipMemcpy2DToArray:=dll.hipMemcpy2DToArray).restype, hipMemcpy2DToArray.argtypes = hipError_t, [hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpy2DToArrayAsync:=dll.hipMemcpy2DToArrayAsync).restype, hipMemcpy2DToArrayAsync.argtypes = hipError_t, [hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

hipArray_const_t = Pointer(hipArray)
try: (hipMemcpy2DArrayToArray:=dll.hipMemcpy2DArrayToArray).restype, hipMemcpy2DArrayToArray.argtypes = hipError_t, [hipArray_t, size_t, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpyToArray:=dll.hipMemcpyToArray).restype, hipMemcpyToArray.argtypes = hipError_t, [hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpyFromArray:=dll.hipMemcpyFromArray).restype, hipMemcpyFromArray.argtypes = hipError_t, [ctypes.c_void_p, hipArray_const_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpy2DFromArray:=dll.hipMemcpy2DFromArray).restype, hipMemcpy2DFromArray.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipMemcpy2DFromArrayAsync:=dll.hipMemcpy2DFromArrayAsync).restype, hipMemcpy2DFromArrayAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

try: (hipMemcpyAtoH:=dll.hipMemcpyAtoH).restype, hipMemcpyAtoH.argtypes = hipError_t, [ctypes.c_void_p, hipArray_t, size_t, size_t]
except AttributeError: pass

try: (hipMemcpyHtoA:=dll.hipMemcpyHtoA).restype, hipMemcpyHtoA.argtypes = hipError_t, [hipArray_t, size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hipMemcpy3D:=dll.hipMemcpy3D).restype, hipMemcpy3D.argtypes = hipError_t, [Pointer(hipMemcpy3DParms)]
except AttributeError: pass

try: (hipMemcpy3DAsync:=dll.hipMemcpy3DAsync).restype, hipMemcpy3DAsync.argtypes = hipError_t, [Pointer(hipMemcpy3DParms), hipStream_t]
except AttributeError: pass

class HIP_MEMCPY3D(Struct): pass
HIP_MEMCPY3D.SIZE = 184
HIP_MEMCPY3D._fields_ = ['srcXInBytes', 'srcY', 'srcZ', 'srcLOD', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'srcPitch', 'srcHeight', 'dstXInBytes', 'dstY', 'dstZ', 'dstLOD', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'dstPitch', 'dstHeight', 'WidthInBytes', 'Height', 'Depth']
setattr(HIP_MEMCPY3D, 'srcXInBytes', field(0, size_t))
setattr(HIP_MEMCPY3D, 'srcY', field(8, size_t))
setattr(HIP_MEMCPY3D, 'srcZ', field(16, size_t))
setattr(HIP_MEMCPY3D, 'srcLOD', field(24, size_t))
setattr(HIP_MEMCPY3D, 'srcMemoryType', field(32, hipMemoryType))
setattr(HIP_MEMCPY3D, 'srcHost', field(40, ctypes.c_void_p))
setattr(HIP_MEMCPY3D, 'srcDevice', field(48, hipDeviceptr_t))
setattr(HIP_MEMCPY3D, 'srcArray', field(56, hipArray_t))
setattr(HIP_MEMCPY3D, 'srcPitch', field(64, size_t))
setattr(HIP_MEMCPY3D, 'srcHeight', field(72, size_t))
setattr(HIP_MEMCPY3D, 'dstXInBytes', field(80, size_t))
setattr(HIP_MEMCPY3D, 'dstY', field(88, size_t))
setattr(HIP_MEMCPY3D, 'dstZ', field(96, size_t))
setattr(HIP_MEMCPY3D, 'dstLOD', field(104, size_t))
setattr(HIP_MEMCPY3D, 'dstMemoryType', field(112, hipMemoryType))
setattr(HIP_MEMCPY3D, 'dstHost', field(120, ctypes.c_void_p))
setattr(HIP_MEMCPY3D, 'dstDevice', field(128, hipDeviceptr_t))
setattr(HIP_MEMCPY3D, 'dstArray', field(136, hipArray_t))
setattr(HIP_MEMCPY3D, 'dstPitch', field(144, size_t))
setattr(HIP_MEMCPY3D, 'dstHeight', field(152, size_t))
setattr(HIP_MEMCPY3D, 'WidthInBytes', field(160, size_t))
setattr(HIP_MEMCPY3D, 'Height', field(168, size_t))
setattr(HIP_MEMCPY3D, 'Depth', field(176, size_t))
try: (hipDrvMemcpy3D:=dll.hipDrvMemcpy3D).restype, hipDrvMemcpy3D.argtypes = hipError_t, [Pointer(HIP_MEMCPY3D)]
except AttributeError: pass

try: (hipDrvMemcpy3DAsync:=dll.hipDrvMemcpy3DAsync).restype, hipDrvMemcpy3DAsync.argtypes = hipError_t, [Pointer(HIP_MEMCPY3D), hipStream_t]
except AttributeError: pass

try: (hipDeviceCanAccessPeer:=dll.hipDeviceCanAccessPeer).restype, hipDeviceCanAccessPeer.argtypes = hipError_t, [Pointer(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: (hipDeviceEnablePeerAccess:=dll.hipDeviceEnablePeerAccess).restype, hipDeviceEnablePeerAccess.argtypes = hipError_t, [ctypes.c_int32, ctypes.c_uint32]
except AttributeError: pass

try: (hipDeviceDisablePeerAccess:=dll.hipDeviceDisablePeerAccess).restype, hipDeviceDisablePeerAccess.argtypes = hipError_t, [ctypes.c_int32]
except AttributeError: pass

try: (hipMemGetAddressRange:=dll.hipMemGetAddressRange).restype, hipMemGetAddressRange.argtypes = hipError_t, [Pointer(hipDeviceptr_t), Pointer(size_t), hipDeviceptr_t]
except AttributeError: pass

try: (hipMemcpyPeer:=dll.hipMemcpyPeer).restype, hipMemcpyPeer.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t]
except AttributeError: pass

try: (hipMemcpyPeerAsync:=dll.hipMemcpyPeerAsync).restype, hipMemcpyPeerAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t]
except AttributeError: pass

try: (hipCtxCreate:=dll.hipCtxCreate).restype, hipCtxCreate.argtypes = hipError_t, [Pointer(hipCtx_t), ctypes.c_uint32, hipDevice_t]
except AttributeError: pass

try: (hipCtxDestroy:=dll.hipCtxDestroy).restype, hipCtxDestroy.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

try: (hipCtxPopCurrent:=dll.hipCtxPopCurrent).restype, hipCtxPopCurrent.argtypes = hipError_t, [Pointer(hipCtx_t)]
except AttributeError: pass

try: (hipCtxPushCurrent:=dll.hipCtxPushCurrent).restype, hipCtxPushCurrent.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

try: (hipCtxSetCurrent:=dll.hipCtxSetCurrent).restype, hipCtxSetCurrent.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

try: (hipCtxGetCurrent:=dll.hipCtxGetCurrent).restype, hipCtxGetCurrent.argtypes = hipError_t, [Pointer(hipCtx_t)]
except AttributeError: pass

try: (hipCtxGetDevice:=dll.hipCtxGetDevice).restype, hipCtxGetDevice.argtypes = hipError_t, [Pointer(hipDevice_t)]
except AttributeError: pass

try: (hipCtxGetApiVersion:=dll.hipCtxGetApiVersion).restype, hipCtxGetApiVersion.argtypes = hipError_t, [hipCtx_t, Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipCtxGetCacheConfig:=dll.hipCtxGetCacheConfig).restype, hipCtxGetCacheConfig.argtypes = hipError_t, [Pointer(hipFuncCache_t)]
except AttributeError: pass

try: (hipCtxSetCacheConfig:=dll.hipCtxSetCacheConfig).restype, hipCtxSetCacheConfig.argtypes = hipError_t, [hipFuncCache_t]
except AttributeError: pass

try: (hipCtxSetSharedMemConfig:=dll.hipCtxSetSharedMemConfig).restype, hipCtxSetSharedMemConfig.argtypes = hipError_t, [hipSharedMemConfig]
except AttributeError: pass

try: (hipCtxGetSharedMemConfig:=dll.hipCtxGetSharedMemConfig).restype, hipCtxGetSharedMemConfig.argtypes = hipError_t, [Pointer(hipSharedMemConfig)]
except AttributeError: pass

try: (hipCtxSynchronize:=dll.hipCtxSynchronize).restype, hipCtxSynchronize.argtypes = hipError_t, []
except AttributeError: pass

try: (hipCtxGetFlags:=dll.hipCtxGetFlags).restype, hipCtxGetFlags.argtypes = hipError_t, [Pointer(ctypes.c_uint32)]
except AttributeError: pass

try: (hipCtxEnablePeerAccess:=dll.hipCtxEnablePeerAccess).restype, hipCtxEnablePeerAccess.argtypes = hipError_t, [hipCtx_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipCtxDisablePeerAccess:=dll.hipCtxDisablePeerAccess).restype, hipCtxDisablePeerAccess.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

try: (hipDevicePrimaryCtxGetState:=dll.hipDevicePrimaryCtxGetState).restype, hipDevicePrimaryCtxGetState.argtypes = hipError_t, [hipDevice_t, Pointer(ctypes.c_uint32), Pointer(ctypes.c_int32)]
except AttributeError: pass

try: (hipDevicePrimaryCtxRelease:=dll.hipDevicePrimaryCtxRelease).restype, hipDevicePrimaryCtxRelease.argtypes = hipError_t, [hipDevice_t]
except AttributeError: pass

try: (hipDevicePrimaryCtxRetain:=dll.hipDevicePrimaryCtxRetain).restype, hipDevicePrimaryCtxRetain.argtypes = hipError_t, [Pointer(hipCtx_t), hipDevice_t]
except AttributeError: pass

try: (hipDevicePrimaryCtxReset:=dll.hipDevicePrimaryCtxReset).restype, hipDevicePrimaryCtxReset.argtypes = hipError_t, [hipDevice_t]
except AttributeError: pass

try: (hipDevicePrimaryCtxSetFlags:=dll.hipDevicePrimaryCtxSetFlags).restype, hipDevicePrimaryCtxSetFlags.argtypes = hipError_t, [hipDevice_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipModuleLoad:=dll.hipModuleLoad).restype, hipModuleLoad.argtypes = hipError_t, [Pointer(hipModule_t), Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hipModuleUnload:=dll.hipModuleUnload).restype, hipModuleUnload.argtypes = hipError_t, [hipModule_t]
except AttributeError: pass

try: (hipModuleGetFunction:=dll.hipModuleGetFunction).restype, hipModuleGetFunction.argtypes = hipError_t, [Pointer(hipFunction_t), hipModule_t, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hipFuncGetAttributes:=dll.hipFuncGetAttributes).restype, hipFuncGetAttributes.argtypes = hipError_t, [Pointer(hipFuncAttributes), ctypes.c_void_p]
except AttributeError: pass

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

try: (hipFuncGetAttribute:=dll.hipFuncGetAttribute).restype, hipFuncGetAttribute.argtypes = hipError_t, [Pointer(ctypes.c_int32), hipFunction_attribute, hipFunction_t]
except AttributeError: pass

try: (hipGetFuncBySymbol:=dll.hipGetFuncBySymbol).restype, hipGetFuncBySymbol.argtypes = hipError_t, [Pointer(hipFunction_t), ctypes.c_void_p]
except AttributeError: pass

class textureReference(Struct): pass
hipTextureReadMode = CEnum(ctypes.c_uint32)
hipReadModeElementType = hipTextureReadMode.define('hipReadModeElementType', 0)
hipReadModeNormalizedFloat = hipTextureReadMode.define('hipReadModeNormalizedFloat', 1)

hipTextureFilterMode = CEnum(ctypes.c_uint32)
hipFilterModePoint = hipTextureFilterMode.define('hipFilterModePoint', 0)
hipFilterModeLinear = hipTextureFilterMode.define('hipFilterModeLinear', 1)

hipTextureAddressMode = CEnum(ctypes.c_uint32)
hipAddressModeWrap = hipTextureAddressMode.define('hipAddressModeWrap', 0)
hipAddressModeClamp = hipTextureAddressMode.define('hipAddressModeClamp', 1)
hipAddressModeMirror = hipTextureAddressMode.define('hipAddressModeMirror', 2)
hipAddressModeBorder = hipTextureAddressMode.define('hipAddressModeBorder', 3)

class __hip_texture(Struct): pass
hipTextureObject_t = Pointer(__hip_texture)
textureReference.SIZE = 88
textureReference._fields_ = ['normalized', 'readMode', 'filterMode', 'addressMode', 'channelDesc', 'sRGB', 'maxAnisotropy', 'mipmapFilterMode', 'mipmapLevelBias', 'minMipmapLevelClamp', 'maxMipmapLevelClamp', 'textureObject', 'numChannels', 'format']
setattr(textureReference, 'normalized', field(0, ctypes.c_int32))
setattr(textureReference, 'readMode', field(4, hipTextureReadMode))
setattr(textureReference, 'filterMode', field(8, hipTextureFilterMode))
setattr(textureReference, 'addressMode', field(12, Array(hipTextureAddressMode, 3)))
setattr(textureReference, 'channelDesc', field(24, hipChannelFormatDesc))
setattr(textureReference, 'sRGB', field(44, ctypes.c_int32))
setattr(textureReference, 'maxAnisotropy', field(48, ctypes.c_uint32))
setattr(textureReference, 'mipmapFilterMode', field(52, hipTextureFilterMode))
setattr(textureReference, 'mipmapLevelBias', field(56, ctypes.c_float))
setattr(textureReference, 'minMipmapLevelClamp', field(60, ctypes.c_float))
setattr(textureReference, 'maxMipmapLevelClamp', field(64, ctypes.c_float))
setattr(textureReference, 'textureObject', field(72, hipTextureObject_t))
setattr(textureReference, 'numChannels', field(80, ctypes.c_int32))
setattr(textureReference, 'format', field(84, hipArray_Format))
try: (hipModuleGetTexRef:=dll.hipModuleGetTexRef).restype, hipModuleGetTexRef.argtypes = hipError_t, [Pointer(Pointer(textureReference)), hipModule_t, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (hipModuleLoadData:=dll.hipModuleLoadData).restype, hipModuleLoadData.argtypes = hipError_t, [Pointer(hipModule_t), ctypes.c_void_p]
except AttributeError: pass

try: (hipModuleLoadDataEx:=dll.hipModuleLoadDataEx).restype, hipModuleLoadDataEx.argtypes = hipError_t, [Pointer(hipModule_t), ctypes.c_void_p, ctypes.c_uint32, Pointer(hipJitOption), Pointer(ctypes.c_void_p)]
except AttributeError: pass

try: (hipModuleLaunchKernel:=dll.hipModuleLaunchKernel).restype, hipModuleLaunchKernel.argtypes = hipError_t, [hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p)]
except AttributeError: pass

try: (hipModuleLaunchCooperativeKernel:=dll.hipModuleLaunchCooperativeKernel).restype, hipModuleLaunchCooperativeKernel.argtypes = hipError_t, [hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, Pointer(ctypes.c_void_p)]
except AttributeError: pass

try: (hipModuleLaunchCooperativeKernelMultiDevice:=dll.hipModuleLaunchCooperativeKernelMultiDevice).restype, hipModuleLaunchCooperativeKernelMultiDevice.argtypes = hipError_t, [Pointer(hipFunctionLaunchParams), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (hipLaunchCooperativeKernel:=dll.hipLaunchCooperativeKernel).restype, hipLaunchCooperativeKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), ctypes.c_uint32, hipStream_t]
except AttributeError: pass

try: (hipLaunchCooperativeKernelMultiDevice:=dll.hipLaunchCooperativeKernelMultiDevice).restype, hipLaunchCooperativeKernelMultiDevice.argtypes = hipError_t, [Pointer(hipLaunchParams), ctypes.c_int32, ctypes.c_uint32]
except AttributeError: pass

try: (hipExtLaunchMultiKernelMultiDevice:=dll.hipExtLaunchMultiKernelMultiDevice).restype, hipExtLaunchMultiKernelMultiDevice.argtypes = hipError_t, [Pointer(hipLaunchParams), ctypes.c_int32, ctypes.c_uint32]
except AttributeError: pass

try: (hipModuleOccupancyMaxPotentialBlockSize:=dll.hipModuleOccupancyMaxPotentialBlockSize).restype, hipModuleOccupancyMaxPotentialBlockSize.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32]
except AttributeError: pass

try: (hipModuleOccupancyMaxPotentialBlockSizeWithFlags:=dll.hipModuleOccupancyMaxPotentialBlockSizeWithFlags).restype, hipModuleOccupancyMaxPotentialBlockSizeWithFlags.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32, ctypes.c_uint32]
except AttributeError: pass

try: (hipModuleOccupancyMaxActiveBlocksPerMultiprocessor:=dll.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor).restype, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = hipError_t, [Pointer(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t]
except AttributeError: pass

try: (hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:=dll.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags).restype, hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = hipError_t, [Pointer(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipOccupancyMaxActiveBlocksPerMultiprocessor:=dll.hipOccupancyMaxActiveBlocksPerMultiprocessor).restype, hipOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = hipError_t, [Pointer(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int32, size_t]
except AttributeError: pass

try: (hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:=dll.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags).restype, hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = hipError_t, [Pointer(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int32, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipOccupancyMaxPotentialBlockSize:=dll.hipOccupancyMaxPotentialBlockSize).restype, hipOccupancyMaxPotentialBlockSize.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

try: (hipProfilerStart:=dll.hipProfilerStart).restype, hipProfilerStart.argtypes = hipError_t, []
except AttributeError: pass

try: (hipProfilerStop:=dll.hipProfilerStop).restype, hipProfilerStop.argtypes = hipError_t, []
except AttributeError: pass

try: (hipConfigureCall:=dll.hipConfigureCall).restype, hipConfigureCall.argtypes = hipError_t, [dim3, dim3, size_t, hipStream_t]
except AttributeError: pass

try: (hipSetupArgument:=dll.hipSetupArgument).restype, hipSetupArgument.argtypes = hipError_t, [ctypes.c_void_p, size_t, size_t]
except AttributeError: pass

try: (hipLaunchByPtr:=dll.hipLaunchByPtr).restype, hipLaunchByPtr.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

try: (__hipPushCallConfiguration:=dll.__hipPushCallConfiguration).restype, __hipPushCallConfiguration.argtypes = hipError_t, [dim3, dim3, size_t, hipStream_t]
except AttributeError: pass

try: (__hipPopCallConfiguration:=dll.__hipPopCallConfiguration).restype, __hipPopCallConfiguration.argtypes = hipError_t, [Pointer(dim3), Pointer(dim3), Pointer(size_t), Pointer(hipStream_t)]
except AttributeError: pass

try: (hipLaunchKernel:=dll.hipLaunchKernel).restype, hipLaunchKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), size_t, hipStream_t]
except AttributeError: pass

try: (hipLaunchHostFunc:=dll.hipLaunchHostFunc).restype, hipLaunchHostFunc.argtypes = hipError_t, [hipStream_t, hipHostFn_t, ctypes.c_void_p]
except AttributeError: pass

try: (hipDrvMemcpy2DUnaligned:=dll.hipDrvMemcpy2DUnaligned).restype, hipDrvMemcpy2DUnaligned.argtypes = hipError_t, [Pointer(hip_Memcpy2D)]
except AttributeError: pass

try: (hipExtLaunchKernel:=dll.hipExtLaunchKernel).restype, hipExtLaunchKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32]
except AttributeError: pass

class hipResourceDesc(Struct): pass
class _anonunion19(Union): pass
class _anonstruct20(Struct): pass
_anonstruct20.SIZE = 8
_anonstruct20._fields_ = ['array']
setattr(_anonstruct20, 'array', field(0, hipArray_t))
class _anonstruct21(Struct): pass
_anonstruct21.SIZE = 8
_anonstruct21._fields_ = ['mipmap']
setattr(_anonstruct21, 'mipmap', field(0, hipMipmappedArray_t))
class _anonstruct22(Struct): pass
_anonstruct22.SIZE = 40
_anonstruct22._fields_ = ['devPtr', 'desc', 'sizeInBytes']
setattr(_anonstruct22, 'devPtr', field(0, ctypes.c_void_p))
setattr(_anonstruct22, 'desc', field(8, hipChannelFormatDesc))
setattr(_anonstruct22, 'sizeInBytes', field(32, size_t))
class _anonstruct23(Struct): pass
_anonstruct23.SIZE = 56
_anonstruct23._fields_ = ['devPtr', 'desc', 'width', 'height', 'pitchInBytes']
setattr(_anonstruct23, 'devPtr', field(0, ctypes.c_void_p))
setattr(_anonstruct23, 'desc', field(8, hipChannelFormatDesc))
setattr(_anonstruct23, 'width', field(32, size_t))
setattr(_anonstruct23, 'height', field(40, size_t))
setattr(_anonstruct23, 'pitchInBytes', field(48, size_t))
_anonunion19.SIZE = 56
_anonunion19._fields_ = ['array', 'mipmap', 'linear', 'pitch2D']
setattr(_anonunion19, 'array', field(0, _anonstruct20))
setattr(_anonunion19, 'mipmap', field(0, _anonstruct21))
setattr(_anonunion19, 'linear', field(0, _anonstruct22))
setattr(_anonunion19, 'pitch2D', field(0, _anonstruct23))
hipResourceDesc.SIZE = 64
hipResourceDesc._fields_ = ['resType', 'res']
setattr(hipResourceDesc, 'resType', field(0, hipResourceType))
setattr(hipResourceDesc, 'res', field(8, _anonunion19))
class hipTextureDesc(Struct): pass
hipTextureDesc.SIZE = 64
hipTextureDesc._fields_ = ['addressMode', 'filterMode', 'readMode', 'sRGB', 'borderColor', 'normalizedCoords', 'maxAnisotropy', 'mipmapFilterMode', 'mipmapLevelBias', 'minMipmapLevelClamp', 'maxMipmapLevelClamp']
setattr(hipTextureDesc, 'addressMode', field(0, Array(hipTextureAddressMode, 3)))
setattr(hipTextureDesc, 'filterMode', field(12, hipTextureFilterMode))
setattr(hipTextureDesc, 'readMode', field(16, hipTextureReadMode))
setattr(hipTextureDesc, 'sRGB', field(20, ctypes.c_int32))
setattr(hipTextureDesc, 'borderColor', field(24, Array(ctypes.c_float, 4)))
setattr(hipTextureDesc, 'normalizedCoords', field(40, ctypes.c_int32))
setattr(hipTextureDesc, 'maxAnisotropy', field(44, ctypes.c_uint32))
setattr(hipTextureDesc, 'mipmapFilterMode', field(48, hipTextureFilterMode))
setattr(hipTextureDesc, 'mipmapLevelBias', field(52, ctypes.c_float))
setattr(hipTextureDesc, 'minMipmapLevelClamp', field(56, ctypes.c_float))
setattr(hipTextureDesc, 'maxMipmapLevelClamp', field(60, ctypes.c_float))
class hipResourceViewDesc(Struct): pass
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

hipResourceViewDesc.SIZE = 48
hipResourceViewDesc._fields_ = ['format', 'width', 'height', 'depth', 'firstMipmapLevel', 'lastMipmapLevel', 'firstLayer', 'lastLayer']
setattr(hipResourceViewDesc, 'format', field(0, hipResourceViewFormat))
setattr(hipResourceViewDesc, 'width', field(8, size_t))
setattr(hipResourceViewDesc, 'height', field(16, size_t))
setattr(hipResourceViewDesc, 'depth', field(24, size_t))
setattr(hipResourceViewDesc, 'firstMipmapLevel', field(32, ctypes.c_uint32))
setattr(hipResourceViewDesc, 'lastMipmapLevel', field(36, ctypes.c_uint32))
setattr(hipResourceViewDesc, 'firstLayer', field(40, ctypes.c_uint32))
setattr(hipResourceViewDesc, 'lastLayer', field(44, ctypes.c_uint32))
try: (hipCreateTextureObject:=dll.hipCreateTextureObject).restype, hipCreateTextureObject.argtypes = hipError_t, [Pointer(hipTextureObject_t), Pointer(hipResourceDesc), Pointer(hipTextureDesc), Pointer(hipResourceViewDesc)]
except AttributeError: pass

try: (hipDestroyTextureObject:=dll.hipDestroyTextureObject).restype, hipDestroyTextureObject.argtypes = hipError_t, [hipTextureObject_t]
except AttributeError: pass

try: (hipGetChannelDesc:=dll.hipGetChannelDesc).restype, hipGetChannelDesc.argtypes = hipError_t, [Pointer(hipChannelFormatDesc), hipArray_const_t]
except AttributeError: pass

try: (hipGetTextureObjectResourceDesc:=dll.hipGetTextureObjectResourceDesc).restype, hipGetTextureObjectResourceDesc.argtypes = hipError_t, [Pointer(hipResourceDesc), hipTextureObject_t]
except AttributeError: pass

try: (hipGetTextureObjectResourceViewDesc:=dll.hipGetTextureObjectResourceViewDesc).restype, hipGetTextureObjectResourceViewDesc.argtypes = hipError_t, [Pointer(hipResourceViewDesc), hipTextureObject_t]
except AttributeError: pass

try: (hipGetTextureObjectTextureDesc:=dll.hipGetTextureObjectTextureDesc).restype, hipGetTextureObjectTextureDesc.argtypes = hipError_t, [Pointer(hipTextureDesc), hipTextureObject_t]
except AttributeError: pass

class HIP_RESOURCE_DESC_st(Struct): pass
HIP_RESOURCE_DESC = HIP_RESOURCE_DESC_st
HIPresourcetype_enum = CEnum(ctypes.c_uint32)
HIP_RESOURCE_TYPE_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_ARRAY', 0)
HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
HIP_RESOURCE_TYPE_LINEAR = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_LINEAR', 2)
HIP_RESOURCE_TYPE_PITCH2D = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_PITCH2D', 3)

HIPresourcetype = HIPresourcetype_enum
class _anonunion24(Union): pass
class _anonstruct25(Struct): pass
_anonstruct25.SIZE = 8
_anonstruct25._fields_ = ['hArray']
setattr(_anonstruct25, 'hArray', field(0, hipArray_t))
class _anonstruct26(Struct): pass
_anonstruct26.SIZE = 8
_anonstruct26._fields_ = ['hMipmappedArray']
setattr(_anonstruct26, 'hMipmappedArray', field(0, hipMipmappedArray_t))
class _anonstruct27(Struct): pass
_anonstruct27.SIZE = 24
_anonstruct27._fields_ = ['devPtr', 'format', 'numChannels', 'sizeInBytes']
setattr(_anonstruct27, 'devPtr', field(0, hipDeviceptr_t))
setattr(_anonstruct27, 'format', field(8, hipArray_Format))
setattr(_anonstruct27, 'numChannels', field(12, ctypes.c_uint32))
setattr(_anonstruct27, 'sizeInBytes', field(16, size_t))
class _anonstruct28(Struct): pass
_anonstruct28.SIZE = 40
_anonstruct28._fields_ = ['devPtr', 'format', 'numChannels', 'width', 'height', 'pitchInBytes']
setattr(_anonstruct28, 'devPtr', field(0, hipDeviceptr_t))
setattr(_anonstruct28, 'format', field(8, hipArray_Format))
setattr(_anonstruct28, 'numChannels', field(12, ctypes.c_uint32))
setattr(_anonstruct28, 'width', field(16, size_t))
setattr(_anonstruct28, 'height', field(24, size_t))
setattr(_anonstruct28, 'pitchInBytes', field(32, size_t))
class _anonstruct29(Struct): pass
_anonstruct29.SIZE = 128
_anonstruct29._fields_ = ['reserved']
setattr(_anonstruct29, 'reserved', field(0, Array(ctypes.c_int32, 32)))
_anonunion24.SIZE = 128
_anonunion24._fields_ = ['array', 'mipmap', 'linear', 'pitch2D', 'reserved']
setattr(_anonunion24, 'array', field(0, _anonstruct25))
setattr(_anonunion24, 'mipmap', field(0, _anonstruct26))
setattr(_anonunion24, 'linear', field(0, _anonstruct27))
setattr(_anonunion24, 'pitch2D', field(0, _anonstruct28))
setattr(_anonunion24, 'reserved', field(0, _anonstruct29))
HIP_RESOURCE_DESC_st.SIZE = 144
HIP_RESOURCE_DESC_st._fields_ = ['resType', 'res', 'flags']
setattr(HIP_RESOURCE_DESC_st, 'resType', field(0, HIPresourcetype))
setattr(HIP_RESOURCE_DESC_st, 'res', field(8, _anonunion24))
setattr(HIP_RESOURCE_DESC_st, 'flags', field(136, ctypes.c_uint32))
class HIP_TEXTURE_DESC_st(Struct): pass
HIP_TEXTURE_DESC = HIP_TEXTURE_DESC_st
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
HIP_TEXTURE_DESC_st.SIZE = 104
HIP_TEXTURE_DESC_st._fields_ = ['addressMode', 'filterMode', 'flags', 'maxAnisotropy', 'mipmapFilterMode', 'mipmapLevelBias', 'minMipmapLevelClamp', 'maxMipmapLevelClamp', 'borderColor', 'reserved']
setattr(HIP_TEXTURE_DESC_st, 'addressMode', field(0, Array(HIPaddress_mode, 3)))
setattr(HIP_TEXTURE_DESC_st, 'filterMode', field(12, HIPfilter_mode))
setattr(HIP_TEXTURE_DESC_st, 'flags', field(16, ctypes.c_uint32))
setattr(HIP_TEXTURE_DESC_st, 'maxAnisotropy', field(20, ctypes.c_uint32))
setattr(HIP_TEXTURE_DESC_st, 'mipmapFilterMode', field(24, HIPfilter_mode))
setattr(HIP_TEXTURE_DESC_st, 'mipmapLevelBias', field(28, ctypes.c_float))
setattr(HIP_TEXTURE_DESC_st, 'minMipmapLevelClamp', field(32, ctypes.c_float))
setattr(HIP_TEXTURE_DESC_st, 'maxMipmapLevelClamp', field(36, ctypes.c_float))
setattr(HIP_TEXTURE_DESC_st, 'borderColor', field(40, Array(ctypes.c_float, 4)))
setattr(HIP_TEXTURE_DESC_st, 'reserved', field(56, Array(ctypes.c_int32, 12)))
class HIP_RESOURCE_VIEW_DESC_st(Struct): pass
HIP_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC_st
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
HIP_RESOURCE_VIEW_DESC_st.SIZE = 112
HIP_RESOURCE_VIEW_DESC_st._fields_ = ['format', 'width', 'height', 'depth', 'firstMipmapLevel', 'lastMipmapLevel', 'firstLayer', 'lastLayer', 'reserved']
setattr(HIP_RESOURCE_VIEW_DESC_st, 'format', field(0, HIPresourceViewFormat))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'width', field(8, size_t))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'height', field(16, size_t))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'depth', field(24, size_t))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'firstMipmapLevel', field(32, ctypes.c_uint32))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'lastMipmapLevel', field(36, ctypes.c_uint32))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'firstLayer', field(40, ctypes.c_uint32))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'lastLayer', field(44, ctypes.c_uint32))
setattr(HIP_RESOURCE_VIEW_DESC_st, 'reserved', field(48, Array(ctypes.c_uint32, 16)))
try: (hipTexObjectCreate:=dll.hipTexObjectCreate).restype, hipTexObjectCreate.argtypes = hipError_t, [Pointer(hipTextureObject_t), Pointer(HIP_RESOURCE_DESC), Pointer(HIP_TEXTURE_DESC), Pointer(HIP_RESOURCE_VIEW_DESC)]
except AttributeError: pass

try: (hipTexObjectDestroy:=dll.hipTexObjectDestroy).restype, hipTexObjectDestroy.argtypes = hipError_t, [hipTextureObject_t]
except AttributeError: pass

try: (hipTexObjectGetResourceDesc:=dll.hipTexObjectGetResourceDesc).restype, hipTexObjectGetResourceDesc.argtypes = hipError_t, [Pointer(HIP_RESOURCE_DESC), hipTextureObject_t]
except AttributeError: pass

try: (hipTexObjectGetResourceViewDesc:=dll.hipTexObjectGetResourceViewDesc).restype, hipTexObjectGetResourceViewDesc.argtypes = hipError_t, [Pointer(HIP_RESOURCE_VIEW_DESC), hipTextureObject_t]
except AttributeError: pass

try: (hipTexObjectGetTextureDesc:=dll.hipTexObjectGetTextureDesc).restype, hipTexObjectGetTextureDesc.argtypes = hipError_t, [Pointer(HIP_TEXTURE_DESC), hipTextureObject_t]
except AttributeError: pass

try: (hipMallocMipmappedArray:=dll.hipMallocMipmappedArray).restype, hipMallocMipmappedArray.argtypes = hipError_t, [Pointer(hipMipmappedArray_t), Pointer(hipChannelFormatDesc), hipExtent, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (hipFreeMipmappedArray:=dll.hipFreeMipmappedArray).restype, hipFreeMipmappedArray.argtypes = hipError_t, [hipMipmappedArray_t]
except AttributeError: pass

hipMipmappedArray_const_t = Pointer(hipMipmappedArray)
try: (hipGetMipmappedArrayLevel:=dll.hipGetMipmappedArrayLevel).restype, hipGetMipmappedArrayLevel.argtypes = hipError_t, [Pointer(hipArray_t), hipMipmappedArray_const_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipMipmappedArrayCreate:=dll.hipMipmappedArrayCreate).restype, hipMipmappedArrayCreate.argtypes = hipError_t, [Pointer(hipMipmappedArray_t), Pointer(HIP_ARRAY3D_DESCRIPTOR), ctypes.c_uint32]
except AttributeError: pass

try: (hipMipmappedArrayDestroy:=dll.hipMipmappedArrayDestroy).restype, hipMipmappedArrayDestroy.argtypes = hipError_t, [hipMipmappedArray_t]
except AttributeError: pass

try: (hipMipmappedArrayGetLevel:=dll.hipMipmappedArrayGetLevel).restype, hipMipmappedArrayGetLevel.argtypes = hipError_t, [Pointer(hipArray_t), hipMipmappedArray_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipBindTextureToMipmappedArray:=dll.hipBindTextureToMipmappedArray).restype, hipBindTextureToMipmappedArray.argtypes = hipError_t, [Pointer(textureReference), hipMipmappedArray_const_t, Pointer(hipChannelFormatDesc)]
except AttributeError: pass

try: (hipGetTextureReference:=dll.hipGetTextureReference).restype, hipGetTextureReference.argtypes = hipError_t, [Pointer(Pointer(textureReference)), ctypes.c_void_p]
except AttributeError: pass

try: (hipTexRefGetBorderColor:=dll.hipTexRefGetBorderColor).restype, hipTexRefGetBorderColor.argtypes = hipError_t, [Pointer(ctypes.c_float), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetArray:=dll.hipTexRefGetArray).restype, hipTexRefGetArray.argtypes = hipError_t, [Pointer(hipArray_t), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefSetAddressMode:=dll.hipTexRefSetAddressMode).restype, hipTexRefSetAddressMode.argtypes = hipError_t, [Pointer(textureReference), ctypes.c_int32, hipTextureAddressMode]
except AttributeError: pass

try: (hipTexRefSetArray:=dll.hipTexRefSetArray).restype, hipTexRefSetArray.argtypes = hipError_t, [Pointer(textureReference), hipArray_const_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipTexRefSetFilterMode:=dll.hipTexRefSetFilterMode).restype, hipTexRefSetFilterMode.argtypes = hipError_t, [Pointer(textureReference), hipTextureFilterMode]
except AttributeError: pass

try: (hipTexRefSetFlags:=dll.hipTexRefSetFlags).restype, hipTexRefSetFlags.argtypes = hipError_t, [Pointer(textureReference), ctypes.c_uint32]
except AttributeError: pass

try: (hipTexRefSetFormat:=dll.hipTexRefSetFormat).restype, hipTexRefSetFormat.argtypes = hipError_t, [Pointer(textureReference), hipArray_Format, ctypes.c_int32]
except AttributeError: pass

try: (hipBindTexture:=dll.hipBindTexture).restype, hipBindTexture.argtypes = hipError_t, [Pointer(size_t), Pointer(textureReference), ctypes.c_void_p, Pointer(hipChannelFormatDesc), size_t]
except AttributeError: pass

try: (hipBindTexture2D:=dll.hipBindTexture2D).restype, hipBindTexture2D.argtypes = hipError_t, [Pointer(size_t), Pointer(textureReference), ctypes.c_void_p, Pointer(hipChannelFormatDesc), size_t, size_t, size_t]
except AttributeError: pass

try: (hipBindTextureToArray:=dll.hipBindTextureToArray).restype, hipBindTextureToArray.argtypes = hipError_t, [Pointer(textureReference), hipArray_const_t, Pointer(hipChannelFormatDesc)]
except AttributeError: pass

try: (hipGetTextureAlignmentOffset:=dll.hipGetTextureAlignmentOffset).restype, hipGetTextureAlignmentOffset.argtypes = hipError_t, [Pointer(size_t), Pointer(textureReference)]
except AttributeError: pass

try: (hipUnbindTexture:=dll.hipUnbindTexture).restype, hipUnbindTexture.argtypes = hipError_t, [Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetAddress:=dll.hipTexRefGetAddress).restype, hipTexRefGetAddress.argtypes = hipError_t, [Pointer(hipDeviceptr_t), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetAddressMode:=dll.hipTexRefGetAddressMode).restype, hipTexRefGetAddressMode.argtypes = hipError_t, [Pointer(hipTextureAddressMode), Pointer(textureReference), ctypes.c_int32]
except AttributeError: pass

try: (hipTexRefGetFilterMode:=dll.hipTexRefGetFilterMode).restype, hipTexRefGetFilterMode.argtypes = hipError_t, [Pointer(hipTextureFilterMode), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetFlags:=dll.hipTexRefGetFlags).restype, hipTexRefGetFlags.argtypes = hipError_t, [Pointer(ctypes.c_uint32), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetFormat:=dll.hipTexRefGetFormat).restype, hipTexRefGetFormat.argtypes = hipError_t, [Pointer(hipArray_Format), Pointer(ctypes.c_int32), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetMaxAnisotropy:=dll.hipTexRefGetMaxAnisotropy).restype, hipTexRefGetMaxAnisotropy.argtypes = hipError_t, [Pointer(ctypes.c_int32), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetMipmapFilterMode:=dll.hipTexRefGetMipmapFilterMode).restype, hipTexRefGetMipmapFilterMode.argtypes = hipError_t, [Pointer(hipTextureFilterMode), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetMipmapLevelBias:=dll.hipTexRefGetMipmapLevelBias).restype, hipTexRefGetMipmapLevelBias.argtypes = hipError_t, [Pointer(ctypes.c_float), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetMipmapLevelClamp:=dll.hipTexRefGetMipmapLevelClamp).restype, hipTexRefGetMipmapLevelClamp.argtypes = hipError_t, [Pointer(ctypes.c_float), Pointer(ctypes.c_float), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefGetMipMappedArray:=dll.hipTexRefGetMipMappedArray).restype, hipTexRefGetMipMappedArray.argtypes = hipError_t, [Pointer(hipMipmappedArray_t), Pointer(textureReference)]
except AttributeError: pass

try: (hipTexRefSetAddress:=dll.hipTexRefSetAddress).restype, hipTexRefSetAddress.argtypes = hipError_t, [Pointer(size_t), Pointer(textureReference), hipDeviceptr_t, size_t]
except AttributeError: pass

try: (hipTexRefSetAddress2D:=dll.hipTexRefSetAddress2D).restype, hipTexRefSetAddress2D.argtypes = hipError_t, [Pointer(textureReference), Pointer(HIP_ARRAY_DESCRIPTOR), hipDeviceptr_t, size_t]
except AttributeError: pass

try: (hipTexRefSetMaxAnisotropy:=dll.hipTexRefSetMaxAnisotropy).restype, hipTexRefSetMaxAnisotropy.argtypes = hipError_t, [Pointer(textureReference), ctypes.c_uint32]
except AttributeError: pass

try: (hipTexRefSetBorderColor:=dll.hipTexRefSetBorderColor).restype, hipTexRefSetBorderColor.argtypes = hipError_t, [Pointer(textureReference), Pointer(ctypes.c_float)]
except AttributeError: pass

try: (hipTexRefSetMipmapFilterMode:=dll.hipTexRefSetMipmapFilterMode).restype, hipTexRefSetMipmapFilterMode.argtypes = hipError_t, [Pointer(textureReference), hipTextureFilterMode]
except AttributeError: pass

try: (hipTexRefSetMipmapLevelBias:=dll.hipTexRefSetMipmapLevelBias).restype, hipTexRefSetMipmapLevelBias.argtypes = hipError_t, [Pointer(textureReference), ctypes.c_float]
except AttributeError: pass

try: (hipTexRefSetMipmapLevelClamp:=dll.hipTexRefSetMipmapLevelClamp).restype, hipTexRefSetMipmapLevelClamp.argtypes = hipError_t, [Pointer(textureReference), ctypes.c_float, ctypes.c_float]
except AttributeError: pass

try: (hipTexRefSetMipmappedArray:=dll.hipTexRefSetMipmappedArray).restype, hipTexRefSetMipmappedArray.argtypes = hipError_t, [Pointer(textureReference), Pointer(hipMipmappedArray), ctypes.c_uint32]
except AttributeError: pass

try: (hipApiName:=dll.hipApiName).restype, hipApiName.argtypes = Pointer(ctypes.c_char), [uint32_t]
except AttributeError: pass

try: (hipKernelNameRef:=dll.hipKernelNameRef).restype, hipKernelNameRef.argtypes = Pointer(ctypes.c_char), [hipFunction_t]
except AttributeError: pass

try: (hipKernelNameRefByPtr:=dll.hipKernelNameRefByPtr).restype, hipKernelNameRefByPtr.argtypes = Pointer(ctypes.c_char), [ctypes.c_void_p, hipStream_t]
except AttributeError: pass

try: (hipGetStreamDeviceId:=dll.hipGetStreamDeviceId).restype, hipGetStreamDeviceId.argtypes = ctypes.c_int32, [hipStream_t]
except AttributeError: pass

try: (hipStreamBeginCapture:=dll.hipStreamBeginCapture).restype, hipStreamBeginCapture.argtypes = hipError_t, [hipStream_t, hipStreamCaptureMode]
except AttributeError: pass

try: (hipStreamBeginCaptureToGraph:=dll.hipStreamBeginCaptureToGraph).restype, hipStreamBeginCaptureToGraph.argtypes = hipError_t, [hipStream_t, hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphEdgeData), size_t, hipStreamCaptureMode]
except AttributeError: pass

try: (hipStreamEndCapture:=dll.hipStreamEndCapture).restype, hipStreamEndCapture.argtypes = hipError_t, [hipStream_t, Pointer(hipGraph_t)]
except AttributeError: pass

try: (hipStreamGetCaptureInfo:=dll.hipStreamGetCaptureInfo).restype, hipStreamGetCaptureInfo.argtypes = hipError_t, [hipStream_t, Pointer(hipStreamCaptureStatus), Pointer(ctypes.c_uint64)]
except AttributeError: pass

try: (hipStreamGetCaptureInfo_v2:=dll.hipStreamGetCaptureInfo_v2).restype, hipStreamGetCaptureInfo_v2.argtypes = hipError_t, [hipStream_t, Pointer(hipStreamCaptureStatus), Pointer(ctypes.c_uint64), Pointer(hipGraph_t), Pointer(Pointer(hipGraphNode_t)), Pointer(size_t)]
except AttributeError: pass

try: (hipStreamIsCapturing:=dll.hipStreamIsCapturing).restype, hipStreamIsCapturing.argtypes = hipError_t, [hipStream_t, Pointer(hipStreamCaptureStatus)]
except AttributeError: pass

try: (hipStreamUpdateCaptureDependencies:=dll.hipStreamUpdateCaptureDependencies).restype, hipStreamUpdateCaptureDependencies.argtypes = hipError_t, [hipStream_t, Pointer(hipGraphNode_t), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipThreadExchangeStreamCaptureMode:=dll.hipThreadExchangeStreamCaptureMode).restype, hipThreadExchangeStreamCaptureMode.argtypes = hipError_t, [Pointer(hipStreamCaptureMode)]
except AttributeError: pass

try: (hipGraphCreate:=dll.hipGraphCreate).restype, hipGraphCreate.argtypes = hipError_t, [Pointer(hipGraph_t), ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphDestroy:=dll.hipGraphDestroy).restype, hipGraphDestroy.argtypes = hipError_t, [hipGraph_t]
except AttributeError: pass

try: (hipGraphAddDependencies:=dll.hipGraphAddDependencies).restype, hipGraphAddDependencies.argtypes = hipError_t, [hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphNode_t), size_t]
except AttributeError: pass

try: (hipGraphRemoveDependencies:=dll.hipGraphRemoveDependencies).restype, hipGraphRemoveDependencies.argtypes = hipError_t, [hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphNode_t), size_t]
except AttributeError: pass

try: (hipGraphGetEdges:=dll.hipGraphGetEdges).restype, hipGraphGetEdges.argtypes = hipError_t, [hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphNode_t), Pointer(size_t)]
except AttributeError: pass

try: (hipGraphGetNodes:=dll.hipGraphGetNodes).restype, hipGraphGetNodes.argtypes = hipError_t, [hipGraph_t, Pointer(hipGraphNode_t), Pointer(size_t)]
except AttributeError: pass

try: (hipGraphGetRootNodes:=dll.hipGraphGetRootNodes).restype, hipGraphGetRootNodes.argtypes = hipError_t, [hipGraph_t, Pointer(hipGraphNode_t), Pointer(size_t)]
except AttributeError: pass

try: (hipGraphNodeGetDependencies:=dll.hipGraphNodeGetDependencies).restype, hipGraphNodeGetDependencies.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipGraphNode_t), Pointer(size_t)]
except AttributeError: pass

try: (hipGraphNodeGetDependentNodes:=dll.hipGraphNodeGetDependentNodes).restype, hipGraphNodeGetDependentNodes.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipGraphNode_t), Pointer(size_t)]
except AttributeError: pass

try: (hipGraphNodeGetType:=dll.hipGraphNodeGetType).restype, hipGraphNodeGetType.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipGraphNodeType)]
except AttributeError: pass

try: (hipGraphDestroyNode:=dll.hipGraphDestroyNode).restype, hipGraphDestroyNode.argtypes = hipError_t, [hipGraphNode_t]
except AttributeError: pass

try: (hipGraphClone:=dll.hipGraphClone).restype, hipGraphClone.argtypes = hipError_t, [Pointer(hipGraph_t), hipGraph_t]
except AttributeError: pass

try: (hipGraphNodeFindInClone:=dll.hipGraphNodeFindInClone).restype, hipGraphNodeFindInClone.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraphNode_t, hipGraph_t]
except AttributeError: pass

try: (hipGraphInstantiate:=dll.hipGraphInstantiate).restype, hipGraphInstantiate.argtypes = hipError_t, [Pointer(hipGraphExec_t), hipGraph_t, Pointer(hipGraphNode_t), Pointer(ctypes.c_char), size_t]
except AttributeError: pass

try: (hipGraphInstantiateWithFlags:=dll.hipGraphInstantiateWithFlags).restype, hipGraphInstantiateWithFlags.argtypes = hipError_t, [Pointer(hipGraphExec_t), hipGraph_t, ctypes.c_uint64]
except AttributeError: pass

try: (hipGraphInstantiateWithParams:=dll.hipGraphInstantiateWithParams).restype, hipGraphInstantiateWithParams.argtypes = hipError_t, [Pointer(hipGraphExec_t), hipGraph_t, Pointer(hipGraphInstantiateParams)]
except AttributeError: pass

try: (hipGraphLaunch:=dll.hipGraphLaunch).restype, hipGraphLaunch.argtypes = hipError_t, [hipGraphExec_t, hipStream_t]
except AttributeError: pass

try: (hipGraphUpload:=dll.hipGraphUpload).restype, hipGraphUpload.argtypes = hipError_t, [hipGraphExec_t, hipStream_t]
except AttributeError: pass

try: (hipGraphAddNode:=dll.hipGraphAddNode).restype, hipGraphAddNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipGraphNodeParams)]
except AttributeError: pass

try: (hipGraphExecDestroy:=dll.hipGraphExecDestroy).restype, hipGraphExecDestroy.argtypes = hipError_t, [hipGraphExec_t]
except AttributeError: pass

try: (hipGraphExecUpdate:=dll.hipGraphExecUpdate).restype, hipGraphExecUpdate.argtypes = hipError_t, [hipGraphExec_t, hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphExecUpdateResult)]
except AttributeError: pass

try: (hipGraphAddKernelNode:=dll.hipGraphAddKernelNode).restype, hipGraphAddKernelNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipKernelNodeParams)]
except AttributeError: pass

try: (hipGraphKernelNodeGetParams:=dll.hipGraphKernelNodeGetParams).restype, hipGraphKernelNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipKernelNodeParams)]
except AttributeError: pass

try: (hipGraphKernelNodeSetParams:=dll.hipGraphKernelNodeSetParams).restype, hipGraphKernelNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipKernelNodeParams)]
except AttributeError: pass

try: (hipGraphExecKernelNodeSetParams:=dll.hipGraphExecKernelNodeSetParams).restype, hipGraphExecKernelNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(hipKernelNodeParams)]
except AttributeError: pass

try: (hipDrvGraphAddMemcpyNode:=dll.hipDrvGraphAddMemcpyNode).restype, hipDrvGraphAddMemcpyNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(HIP_MEMCPY3D), hipCtx_t]
except AttributeError: pass

try: (hipGraphAddMemcpyNode:=dll.hipGraphAddMemcpyNode).restype, hipGraphAddMemcpyNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipMemcpy3DParms)]
except AttributeError: pass

try: (hipGraphMemcpyNodeGetParams:=dll.hipGraphMemcpyNodeGetParams).restype, hipGraphMemcpyNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipMemcpy3DParms)]
except AttributeError: pass

try: (hipGraphMemcpyNodeSetParams:=dll.hipGraphMemcpyNodeSetParams).restype, hipGraphMemcpyNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipMemcpy3DParms)]
except AttributeError: pass

try: (hipGraphKernelNodeSetAttribute:=dll.hipGraphKernelNodeSetAttribute).restype, hipGraphKernelNodeSetAttribute.argtypes = hipError_t, [hipGraphNode_t, hipLaunchAttributeID, Pointer(hipLaunchAttributeValue)]
except AttributeError: pass

try: (hipGraphKernelNodeGetAttribute:=dll.hipGraphKernelNodeGetAttribute).restype, hipGraphKernelNodeGetAttribute.argtypes = hipError_t, [hipGraphNode_t, hipLaunchAttributeID, Pointer(hipLaunchAttributeValue)]
except AttributeError: pass

try: (hipGraphExecMemcpyNodeSetParams:=dll.hipGraphExecMemcpyNodeSetParams).restype, hipGraphExecMemcpyNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(hipMemcpy3DParms)]
except AttributeError: pass

try: (hipGraphAddMemcpyNode1D:=dll.hipGraphAddMemcpyNode1D).restype, hipGraphAddMemcpyNode1D.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphMemcpyNodeSetParams1D:=dll.hipGraphMemcpyNodeSetParams1D).restype, hipGraphMemcpyNodeSetParams1D.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphExecMemcpyNodeSetParams1D:=dll.hipGraphExecMemcpyNodeSetParams1D).restype, hipGraphExecMemcpyNodeSetParams1D.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphAddMemcpyNodeFromSymbol:=dll.hipGraphAddMemcpyNodeFromSymbol).restype, hipGraphAddMemcpyNodeFromSymbol.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphMemcpyNodeSetParamsFromSymbol:=dll.hipGraphMemcpyNodeSetParamsFromSymbol).restype, hipGraphMemcpyNodeSetParamsFromSymbol.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphExecMemcpyNodeSetParamsFromSymbol:=dll.hipGraphExecMemcpyNodeSetParamsFromSymbol).restype, hipGraphExecMemcpyNodeSetParamsFromSymbol.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphAddMemcpyNodeToSymbol:=dll.hipGraphAddMemcpyNodeToSymbol).restype, hipGraphAddMemcpyNodeToSymbol.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphMemcpyNodeSetParamsToSymbol:=dll.hipGraphMemcpyNodeSetParamsToSymbol).restype, hipGraphMemcpyNodeSetParamsToSymbol.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphExecMemcpyNodeSetParamsToSymbol:=dll.hipGraphExecMemcpyNodeSetParamsToSymbol).restype, hipGraphExecMemcpyNodeSetParamsToSymbol.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

try: (hipGraphAddMemsetNode:=dll.hipGraphAddMemsetNode).restype, hipGraphAddMemsetNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipMemsetParams)]
except AttributeError: pass

try: (hipGraphMemsetNodeGetParams:=dll.hipGraphMemsetNodeGetParams).restype, hipGraphMemsetNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipMemsetParams)]
except AttributeError: pass

try: (hipGraphMemsetNodeSetParams:=dll.hipGraphMemsetNodeSetParams).restype, hipGraphMemsetNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipMemsetParams)]
except AttributeError: pass

try: (hipGraphExecMemsetNodeSetParams:=dll.hipGraphExecMemsetNodeSetParams).restype, hipGraphExecMemsetNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(hipMemsetParams)]
except AttributeError: pass

try: (hipGraphAddHostNode:=dll.hipGraphAddHostNode).restype, hipGraphAddHostNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipHostNodeParams)]
except AttributeError: pass

try: (hipGraphHostNodeGetParams:=dll.hipGraphHostNodeGetParams).restype, hipGraphHostNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipHostNodeParams)]
except AttributeError: pass

try: (hipGraphHostNodeSetParams:=dll.hipGraphHostNodeSetParams).restype, hipGraphHostNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipHostNodeParams)]
except AttributeError: pass

try: (hipGraphExecHostNodeSetParams:=dll.hipGraphExecHostNodeSetParams).restype, hipGraphExecHostNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(hipHostNodeParams)]
except AttributeError: pass

try: (hipGraphAddChildGraphNode:=dll.hipGraphAddChildGraphNode).restype, hipGraphAddChildGraphNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, hipGraph_t]
except AttributeError: pass

try: (hipGraphChildGraphNodeGetGraph:=dll.hipGraphChildGraphNodeGetGraph).restype, hipGraphChildGraphNodeGetGraph.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipGraph_t)]
except AttributeError: pass

try: (hipGraphExecChildGraphNodeSetParams:=dll.hipGraphExecChildGraphNodeSetParams).restype, hipGraphExecChildGraphNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, hipGraph_t]
except AttributeError: pass

try: (hipGraphAddEmptyNode:=dll.hipGraphAddEmptyNode).restype, hipGraphAddEmptyNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t]
except AttributeError: pass

try: (hipGraphAddEventRecordNode:=dll.hipGraphAddEventRecordNode).restype, hipGraphAddEventRecordNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, hipEvent_t]
except AttributeError: pass

try: (hipGraphEventRecordNodeGetEvent:=dll.hipGraphEventRecordNodeGetEvent).restype, hipGraphEventRecordNodeGetEvent.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipEvent_t)]
except AttributeError: pass

try: (hipGraphEventRecordNodeSetEvent:=dll.hipGraphEventRecordNodeSetEvent).restype, hipGraphEventRecordNodeSetEvent.argtypes = hipError_t, [hipGraphNode_t, hipEvent_t]
except AttributeError: pass

try: (hipGraphExecEventRecordNodeSetEvent:=dll.hipGraphExecEventRecordNodeSetEvent).restype, hipGraphExecEventRecordNodeSetEvent.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, hipEvent_t]
except AttributeError: pass

try: (hipGraphAddEventWaitNode:=dll.hipGraphAddEventWaitNode).restype, hipGraphAddEventWaitNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, hipEvent_t]
except AttributeError: pass

try: (hipGraphEventWaitNodeGetEvent:=dll.hipGraphEventWaitNodeGetEvent).restype, hipGraphEventWaitNodeGetEvent.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipEvent_t)]
except AttributeError: pass

try: (hipGraphEventWaitNodeSetEvent:=dll.hipGraphEventWaitNodeSetEvent).restype, hipGraphEventWaitNodeSetEvent.argtypes = hipError_t, [hipGraphNode_t, hipEvent_t]
except AttributeError: pass

try: (hipGraphExecEventWaitNodeSetEvent:=dll.hipGraphExecEventWaitNodeSetEvent).restype, hipGraphExecEventWaitNodeSetEvent.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, hipEvent_t]
except AttributeError: pass

try: (hipGraphAddMemAllocNode:=dll.hipGraphAddMemAllocNode).restype, hipGraphAddMemAllocNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipMemAllocNodeParams)]
except AttributeError: pass

try: (hipGraphMemAllocNodeGetParams:=dll.hipGraphMemAllocNodeGetParams).restype, hipGraphMemAllocNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipMemAllocNodeParams)]
except AttributeError: pass

try: (hipGraphAddMemFreeNode:=dll.hipGraphAddMemFreeNode).restype, hipGraphAddMemFreeNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p]
except AttributeError: pass

try: (hipGraphMemFreeNodeGetParams:=dll.hipGraphMemFreeNodeGetParams).restype, hipGraphMemFreeNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p]
except AttributeError: pass

try: (hipDeviceGetGraphMemAttribute:=dll.hipDeviceGetGraphMemAttribute).restype, hipDeviceGetGraphMemAttribute.argtypes = hipError_t, [ctypes.c_int32, hipGraphMemAttributeType, ctypes.c_void_p]
except AttributeError: pass

try: (hipDeviceSetGraphMemAttribute:=dll.hipDeviceSetGraphMemAttribute).restype, hipDeviceSetGraphMemAttribute.argtypes = hipError_t, [ctypes.c_int32, hipGraphMemAttributeType, ctypes.c_void_p]
except AttributeError: pass

try: (hipDeviceGraphMemTrim:=dll.hipDeviceGraphMemTrim).restype, hipDeviceGraphMemTrim.argtypes = hipError_t, [ctypes.c_int32]
except AttributeError: pass

try: (hipUserObjectCreate:=dll.hipUserObjectCreate).restype, hipUserObjectCreate.argtypes = hipError_t, [Pointer(hipUserObject_t), ctypes.c_void_p, hipHostFn_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (hipUserObjectRelease:=dll.hipUserObjectRelease).restype, hipUserObjectRelease.argtypes = hipError_t, [hipUserObject_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipUserObjectRetain:=dll.hipUserObjectRetain).restype, hipUserObjectRetain.argtypes = hipError_t, [hipUserObject_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphRetainUserObject:=dll.hipGraphRetainUserObject).restype, hipGraphRetainUserObject.argtypes = hipError_t, [hipGraph_t, hipUserObject_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphReleaseUserObject:=dll.hipGraphReleaseUserObject).restype, hipGraphReleaseUserObject.argtypes = hipError_t, [hipGraph_t, hipUserObject_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphDebugDotPrint:=dll.hipGraphDebugDotPrint).restype, hipGraphDebugDotPrint.argtypes = hipError_t, [hipGraph_t, Pointer(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphKernelNodeCopyAttributes:=dll.hipGraphKernelNodeCopyAttributes).restype, hipGraphKernelNodeCopyAttributes.argtypes = hipError_t, [hipGraphNode_t, hipGraphNode_t]
except AttributeError: pass

try: (hipGraphNodeSetEnabled:=dll.hipGraphNodeSetEnabled).restype, hipGraphNodeSetEnabled.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphNodeGetEnabled:=dll.hipGraphNodeGetEnabled).restype, hipGraphNodeGetEnabled.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(ctypes.c_uint32)]
except AttributeError: pass

try: (hipGraphAddExternalSemaphoresWaitNode:=dll.hipGraphAddExternalSemaphoresWaitNode).restype, hipGraphAddExternalSemaphoresWaitNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

try: (hipGraphAddExternalSemaphoresSignalNode:=dll.hipGraphAddExternalSemaphoresSignalNode).restype, hipGraphAddExternalSemaphoresSignalNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

try: (hipGraphExternalSemaphoresSignalNodeSetParams:=dll.hipGraphExternalSemaphoresSignalNodeSetParams).restype, hipGraphExternalSemaphoresSignalNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

try: (hipGraphExternalSemaphoresWaitNodeSetParams:=dll.hipGraphExternalSemaphoresWaitNodeSetParams).restype, hipGraphExternalSemaphoresWaitNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

try: (hipGraphExternalSemaphoresSignalNodeGetParams:=dll.hipGraphExternalSemaphoresSignalNodeGetParams).restype, hipGraphExternalSemaphoresSignalNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

try: (hipGraphExternalSemaphoresWaitNodeGetParams:=dll.hipGraphExternalSemaphoresWaitNodeGetParams).restype, hipGraphExternalSemaphoresWaitNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, Pointer(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

try: (hipGraphExecExternalSemaphoresSignalNodeSetParams:=dll.hipGraphExecExternalSemaphoresSignalNodeSetParams).restype, hipGraphExecExternalSemaphoresSignalNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

try: (hipGraphExecExternalSemaphoresWaitNodeSetParams:=dll.hipGraphExecExternalSemaphoresWaitNodeSetParams).restype, hipGraphExecExternalSemaphoresWaitNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, Pointer(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

try: (hipDrvGraphAddMemsetNode:=dll.hipDrvGraphAddMemsetNode).restype, hipDrvGraphAddMemsetNode.argtypes = hipError_t, [Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(HIP_MEMSET_NODE_PARAMS), hipCtx_t]
except AttributeError: pass

try: (hipMemAddressFree:=dll.hipMemAddressFree).restype, hipMemAddressFree.argtypes = hipError_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hipMemAddressReserve:=dll.hipMemAddressReserve).restype, hipMemAddressReserve.argtypes = hipError_t, [Pointer(ctypes.c_void_p), size_t, size_t, ctypes.c_void_p, ctypes.c_uint64]
except AttributeError: pass

try: (hipMemCreate:=dll.hipMemCreate).restype, hipMemCreate.argtypes = hipError_t, [Pointer(hipMemGenericAllocationHandle_t), size_t, Pointer(hipMemAllocationProp), ctypes.c_uint64]
except AttributeError: pass

try: (hipMemExportToShareableHandle:=dll.hipMemExportToShareableHandle).restype, hipMemExportToShareableHandle.argtypes = hipError_t, [ctypes.c_void_p, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, ctypes.c_uint64]
except AttributeError: pass

try: (hipMemGetAccess:=dll.hipMemGetAccess).restype, hipMemGetAccess.argtypes = hipError_t, [Pointer(ctypes.c_uint64), Pointer(hipMemLocation), ctypes.c_void_p]
except AttributeError: pass

try: (hipMemGetAllocationGranularity:=dll.hipMemGetAllocationGranularity).restype, hipMemGetAllocationGranularity.argtypes = hipError_t, [Pointer(size_t), Pointer(hipMemAllocationProp), hipMemAllocationGranularity_flags]
except AttributeError: pass

try: (hipMemGetAllocationPropertiesFromHandle:=dll.hipMemGetAllocationPropertiesFromHandle).restype, hipMemGetAllocationPropertiesFromHandle.argtypes = hipError_t, [Pointer(hipMemAllocationProp), hipMemGenericAllocationHandle_t]
except AttributeError: pass

try: (hipMemImportFromShareableHandle:=dll.hipMemImportFromShareableHandle).restype, hipMemImportFromShareableHandle.argtypes = hipError_t, [Pointer(hipMemGenericAllocationHandle_t), ctypes.c_void_p, hipMemAllocationHandleType]
except AttributeError: pass

try: (hipMemMap:=dll.hipMemMap).restype, hipMemMap.argtypes = hipError_t, [ctypes.c_void_p, size_t, size_t, hipMemGenericAllocationHandle_t, ctypes.c_uint64]
except AttributeError: pass

try: (hipMemMapArrayAsync:=dll.hipMemMapArrayAsync).restype, hipMemMapArrayAsync.argtypes = hipError_t, [Pointer(hipArrayMapInfo), ctypes.c_uint32, hipStream_t]
except AttributeError: pass

try: (hipMemRelease:=dll.hipMemRelease).restype, hipMemRelease.argtypes = hipError_t, [hipMemGenericAllocationHandle_t]
except AttributeError: pass

try: (hipMemRetainAllocationHandle:=dll.hipMemRetainAllocationHandle).restype, hipMemRetainAllocationHandle.argtypes = hipError_t, [Pointer(hipMemGenericAllocationHandle_t), ctypes.c_void_p]
except AttributeError: pass

try: (hipMemSetAccess:=dll.hipMemSetAccess).restype, hipMemSetAccess.argtypes = hipError_t, [ctypes.c_void_p, size_t, Pointer(hipMemAccessDesc), size_t]
except AttributeError: pass

try: (hipMemUnmap:=dll.hipMemUnmap).restype, hipMemUnmap.argtypes = hipError_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hipGraphicsMapResources:=dll.hipGraphicsMapResources).restype, hipGraphicsMapResources.argtypes = hipError_t, [ctypes.c_int32, Pointer(hipGraphicsResource_t), hipStream_t]
except AttributeError: pass

try: (hipGraphicsSubResourceGetMappedArray:=dll.hipGraphicsSubResourceGetMappedArray).restype, hipGraphicsSubResourceGetMappedArray.argtypes = hipError_t, [Pointer(hipArray_t), hipGraphicsResource_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (hipGraphicsResourceGetMappedPointer:=dll.hipGraphicsResourceGetMappedPointer).restype, hipGraphicsResourceGetMappedPointer.argtypes = hipError_t, [Pointer(ctypes.c_void_p), Pointer(size_t), hipGraphicsResource_t]
except AttributeError: pass

try: (hipGraphicsUnmapResources:=dll.hipGraphicsUnmapResources).restype, hipGraphicsUnmapResources.argtypes = hipError_t, [ctypes.c_int32, Pointer(hipGraphicsResource_t), hipStream_t]
except AttributeError: pass

try: (hipGraphicsUnregisterResource:=dll.hipGraphicsUnregisterResource).restype, hipGraphicsUnregisterResource.argtypes = hipError_t, [hipGraphicsResource_t]
except AttributeError: pass

class __hip_surface(Struct): pass
hipSurfaceObject_t = Pointer(__hip_surface)
try: (hipCreateSurfaceObject:=dll.hipCreateSurfaceObject).restype, hipCreateSurfaceObject.argtypes = hipError_t, [Pointer(hipSurfaceObject_t), Pointer(hipResourceDesc)]
except AttributeError: pass

try: (hipDestroySurfaceObject:=dll.hipDestroySurfaceObject).restype, hipDestroySurfaceObject.argtypes = hipError_t, [hipSurfaceObject_t]
except AttributeError: pass

hipmipmappedArray = Pointer(hipMipmappedArray)
hipResourcetype = HIPresourcetype_enum
hipGetDeviceProperties = hipGetDevicePropertiesR0600
hipDeviceProp_t = hipDeviceProp_tR0600
hipChooseDevice = hipChooseDeviceR0600
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
hipKernelNodeAttrID = hipLaunchAttributeID
hipKernelNodeAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow
hipKernelNodeAttributeCooperative = hipLaunchAttributeCooperative
hipKernelNodeAttributePriority = hipLaunchAttributePriority
hipKernelNodeAttrValue = hipLaunchAttributeValue
hipGraphKernelNodePortDefault = 0
hipGraphKernelNodePortLaunchCompletion = 2
hipGraphKernelNodePortProgrammatic = 1
USE_PEER_NON_UNIFIED = 1
HIP_TRSA_OVERRIDE_FORMAT = 0x01
HIP_TRSF_READ_AS_INTEGER = 0x01
HIP_TRSF_NORMALIZED_COORDINATES = 0x02
HIP_TRSF_SRGB = 0x10