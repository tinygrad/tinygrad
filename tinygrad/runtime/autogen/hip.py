# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
import os
dll = DLL('hip', os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
class ihipModuleSymbol_t(Struct): pass
hipFunction_t = Pointer(ihipModuleSymbol_t)
uint32_t = ctypes.c_uint32
size_t = ctypes.c_uint64
class ihipStream_t(Struct): pass
hipStream_t = Pointer(ihipStream_t)
class ihipEvent_t(Struct): pass
hipEvent_t = Pointer(ihipEvent_t)
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

@dll.bind((hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p), hipEvent_t, hipEvent_t, uint32_t), hipError_t)
def hipExtModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags): ...
@dll.bind((hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p), hipEvent_t, hipEvent_t), hipError_t)
def hipHccModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent): ...
class dim3(Struct): pass
dim3.SIZE = 12
dim3._fields_ = ['x', 'y', 'z']
setattr(dim3, 'x', field(0, uint32_t))
setattr(dim3, 'y', field(4, uint32_t))
setattr(dim3, 'z', field(8, uint32_t))
@dll.bind((ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32), hipError_t)
def hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags): ...
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
@dll.bind((hiprtcResult), Pointer(ctypes.c_char))
def hiprtcGetErrorString(result): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32)), hiprtcResult)
def hiprtcVersion(major, minor): ...
class _hiprtcProgram(Struct): pass
hiprtcProgram = Pointer(_hiprtcProgram)
@dll.bind((hiprtcProgram, Pointer(ctypes.c_char)), hiprtcResult)
def hiprtcAddNameExpression(prog, name_expression): ...
@dll.bind((hiprtcProgram, ctypes.c_int32, Pointer(Pointer(ctypes.c_char))), hiprtcResult)
def hiprtcCompileProgram(prog, numOptions, options): ...
@dll.bind((Pointer(hiprtcProgram), Pointer(ctypes.c_char), Pointer(ctypes.c_char), ctypes.c_int32, Pointer(Pointer(ctypes.c_char)), Pointer(Pointer(ctypes.c_char))), hiprtcResult)
def hiprtcCreateProgram(prog, src, name, numHeaders, headers, includeNames): ...
@dll.bind((Pointer(hiprtcProgram)), hiprtcResult)
def hiprtcDestroyProgram(prog): ...
@dll.bind((hiprtcProgram, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char))), hiprtcResult)
def hiprtcGetLoweredName(prog, name_expression, lowered_name): ...
@dll.bind((hiprtcProgram, Pointer(ctypes.c_char)), hiprtcResult)
def hiprtcGetProgramLog(prog, log): ...
@dll.bind((hiprtcProgram, Pointer(size_t)), hiprtcResult)
def hiprtcGetProgramLogSize(prog, logSizeRet): ...
@dll.bind((hiprtcProgram, Pointer(ctypes.c_char)), hiprtcResult)
def hiprtcGetCode(prog, code): ...
@dll.bind((hiprtcProgram, Pointer(size_t)), hiprtcResult)
def hiprtcGetCodeSize(prog, codeSizeRet): ...
@dll.bind((hiprtcProgram, Pointer(ctypes.c_char)), hiprtcResult)
def hiprtcGetBitcode(prog, bitcode): ...
@dll.bind((hiprtcProgram, Pointer(size_t)), hiprtcResult)
def hiprtcGetBitcodeSize(prog, bitcode_size): ...
@dll.bind((ctypes.c_uint32, Pointer(hiprtcJIT_option), Pointer(ctypes.c_void_p), Pointer(hiprtcLinkState)), hiprtcResult)
def hiprtcLinkCreate(num_options, option_ptr, option_vals_pptr, hip_link_state_ptr): ...
@dll.bind((hiprtcLinkState, hiprtcJITInputType, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(hiprtcJIT_option), Pointer(ctypes.c_void_p)), hiprtcResult)
def hiprtcLinkAddFile(hip_link_state, input_type, file_path, num_options, options_ptr, option_values): ...
@dll.bind((hiprtcLinkState, hiprtcJITInputType, ctypes.c_void_p, size_t, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(hiprtcJIT_option), Pointer(ctypes.c_void_p)), hiprtcResult)
def hiprtcLinkAddData(hip_link_state, input_type, image, image_size, name, num_options, options_ptr, option_values): ...
@dll.bind((hiprtcLinkState, Pointer(ctypes.c_void_p), Pointer(size_t)), hiprtcResult)
def hiprtcLinkComplete(hip_link_state, bin_out, size_out): ...
@dll.bind((hiprtcLinkState), hiprtcResult)
def hiprtcLinkDestroy(hip_link_state): ...
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

@dll.bind((), hipError_t)
def hip_init(): ...
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
@dll.bind((Pointer(Pointer(ctypes.c_char)), Pointer(ctypes.c_uint32)), None)
def __hipGetPCH(pch, size): ...
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
@dll.bind((ctypes.c_uint32), hipError_t)
def hipInit(flags): ...
@dll.bind((Pointer(ctypes.c_int32)), hipError_t)
def hipDriverGetVersion(driverVersion): ...
@dll.bind((Pointer(ctypes.c_int32)), hipError_t)
def hipRuntimeGetVersion(runtimeVersion): ...
@dll.bind((Pointer(hipDevice_t), ctypes.c_int32), hipError_t)
def hipDeviceGet(device, ordinal): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), hipDevice_t), hipError_t)
def hipDeviceComputeCapability(major, minor, device): ...
@dll.bind((Pointer(ctypes.c_char), ctypes.c_int32, hipDevice_t), hipError_t)
def hipDeviceGetName(name, len, device): ...
@dll.bind((Pointer(hipUUID), hipDevice_t), hipError_t)
def hipDeviceGetUuid(uuid, device): ...
@dll.bind((Pointer(ctypes.c_int32), hipDeviceP2PAttr, ctypes.c_int32, ctypes.c_int32), hipError_t)
def hipDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice): ...
@dll.bind((Pointer(ctypes.c_char), ctypes.c_int32, ctypes.c_int32), hipError_t)
def hipDeviceGetPCIBusId(pciBusId, len, device): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_char)), hipError_t)
def hipDeviceGetByPCIBusId(device, pciBusId): ...
@dll.bind((Pointer(size_t), hipDevice_t), hipError_t)
def hipDeviceTotalMem(bytes, device): ...
@dll.bind((), hipError_t)
def hipDeviceSynchronize(): ...
@dll.bind((), hipError_t)
def hipDeviceReset(): ...
@dll.bind((ctypes.c_int32), hipError_t)
def hipSetDevice(deviceId): ...
@dll.bind((Pointer(ctypes.c_int32), ctypes.c_int32), hipError_t)
def hipSetValidDevices(device_arr, len): ...
@dll.bind((Pointer(ctypes.c_int32)), hipError_t)
def hipGetDevice(deviceId): ...
@dll.bind((Pointer(ctypes.c_int32)), hipError_t)
def hipGetDeviceCount(count): ...
@dll.bind((Pointer(ctypes.c_int32), hipDeviceAttribute_t, ctypes.c_int32), hipError_t)
def hipDeviceGetAttribute(pi, attr, deviceId): ...
@dll.bind((Pointer(hipMemPool_t), ctypes.c_int32), hipError_t)
def hipDeviceGetDefaultMemPool(mem_pool, device): ...
@dll.bind((ctypes.c_int32, hipMemPool_t), hipError_t)
def hipDeviceSetMemPool(device, mem_pool): ...
@dll.bind((Pointer(hipMemPool_t), ctypes.c_int32), hipError_t)
def hipDeviceGetMemPool(mem_pool, device): ...
@dll.bind((Pointer(hipDeviceProp_tR0600), ctypes.c_int32), hipError_t)
def hipGetDevicePropertiesR0600(prop, deviceId): ...
@dll.bind((hipFuncCache_t), hipError_t)
def hipDeviceSetCacheConfig(cacheConfig): ...
@dll.bind((Pointer(hipFuncCache_t)), hipError_t)
def hipDeviceGetCacheConfig(cacheConfig): ...
@dll.bind((Pointer(size_t), hipLimit_t), hipError_t)
def hipDeviceGetLimit(pValue, limit): ...
@dll.bind((hipLimit_t, size_t), hipError_t)
def hipDeviceSetLimit(limit, value): ...
@dll.bind((Pointer(hipSharedMemConfig)), hipError_t)
def hipDeviceGetSharedMemConfig(pConfig): ...
@dll.bind((Pointer(ctypes.c_uint32)), hipError_t)
def hipGetDeviceFlags(flags): ...
@dll.bind((hipSharedMemConfig), hipError_t)
def hipDeviceSetSharedMemConfig(config): ...
@dll.bind((ctypes.c_uint32), hipError_t)
def hipSetDeviceFlags(flags): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(hipDeviceProp_tR0600)), hipError_t)
def hipChooseDeviceR0600(device, prop): ...
@dll.bind((ctypes.c_int32, ctypes.c_int32, Pointer(uint32_t), Pointer(uint32_t)), hipError_t)
def hipExtGetLinkTypeAndHopCount(device1, device2, linktype, hopcount): ...
@dll.bind((Pointer(hipIpcMemHandle_t), ctypes.c_void_p), hipError_t)
def hipIpcGetMemHandle(handle, devPtr): ...
@dll.bind((Pointer(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint32), hipError_t)
def hipIpcOpenMemHandle(devPtr, handle, flags): ...
@dll.bind((ctypes.c_void_p), hipError_t)
def hipIpcCloseMemHandle(devPtr): ...
@dll.bind((Pointer(hipIpcEventHandle_t), hipEvent_t), hipError_t)
def hipIpcGetEventHandle(handle, event): ...
@dll.bind((Pointer(hipEvent_t), hipIpcEventHandle_t), hipError_t)
def hipIpcOpenEventHandle(event, handle): ...
@dll.bind((ctypes.c_void_p, hipFuncAttribute, ctypes.c_int32), hipError_t)
def hipFuncSetAttribute(func, attr, value): ...
@dll.bind((ctypes.c_void_p, hipFuncCache_t), hipError_t)
def hipFuncSetCacheConfig(func, config): ...
@dll.bind((ctypes.c_void_p, hipSharedMemConfig), hipError_t)
def hipFuncSetSharedMemConfig(func, config): ...
@dll.bind((), hipError_t)
def hipGetLastError(): ...
@dll.bind((), hipError_t)
def hipExtGetLastError(): ...
@dll.bind((), hipError_t)
def hipPeekAtLastError(): ...
@dll.bind((hipError_t), Pointer(ctypes.c_char))
def hipGetErrorName(hip_error): ...
@dll.bind((hipError_t), Pointer(ctypes.c_char))
def hipGetErrorString(hipError): ...
@dll.bind((hipError_t, Pointer(Pointer(ctypes.c_char))), hipError_t)
def hipDrvGetErrorName(hipError, errorString): ...
@dll.bind((hipError_t, Pointer(Pointer(ctypes.c_char))), hipError_t)
def hipDrvGetErrorString(hipError, errorString): ...
@dll.bind((Pointer(hipStream_t)), hipError_t)
def hipStreamCreate(stream): ...
@dll.bind((Pointer(hipStream_t), ctypes.c_uint32), hipError_t)
def hipStreamCreateWithFlags(stream, flags): ...
@dll.bind((Pointer(hipStream_t), ctypes.c_uint32, ctypes.c_int32), hipError_t)
def hipStreamCreateWithPriority(stream, flags, priority): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32)), hipError_t)
def hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority): ...
@dll.bind((hipStream_t), hipError_t)
def hipStreamDestroy(stream): ...
@dll.bind((hipStream_t), hipError_t)
def hipStreamQuery(stream): ...
@dll.bind((hipStream_t), hipError_t)
def hipStreamSynchronize(stream): ...
@dll.bind((hipStream_t, hipEvent_t, ctypes.c_uint32), hipError_t)
def hipStreamWaitEvent(stream, event, flags): ...
@dll.bind((hipStream_t, Pointer(ctypes.c_uint32)), hipError_t)
def hipStreamGetFlags(stream, flags): ...
@dll.bind((hipStream_t, Pointer(ctypes.c_int32)), hipError_t)
def hipStreamGetPriority(stream, priority): ...
@dll.bind((hipStream_t, Pointer(hipDevice_t)), hipError_t)
def hipStreamGetDevice(stream, device): ...
@dll.bind((Pointer(hipStream_t), uint32_t, Pointer(uint32_t)), hipError_t)
def hipExtStreamCreateWithCUMask(stream, cuMaskSize, cuMask): ...
@dll.bind((hipStream_t, uint32_t, Pointer(uint32_t)), hipError_t)
def hipExtStreamGetCUMask(stream, cuMaskSize, cuMask): ...
hipStreamCallback_t = ctypes.CFUNCTYPE(None, Pointer(ihipStream_t), hipError_t, ctypes.c_void_p)
@dll.bind((hipStream_t, hipStreamCallback_t, ctypes.c_void_p, ctypes.c_uint32), hipError_t)
def hipStreamAddCallback(stream, callback, userData, flags): ...
@dll.bind((hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32, uint32_t), hipError_t)
def hipStreamWaitValue32(stream, ptr, value, flags, mask): ...
uint64_t = ctypes.c_uint64
@dll.bind((hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32, uint64_t), hipError_t)
def hipStreamWaitValue64(stream, ptr, value, flags, mask): ...
@dll.bind((hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32), hipError_t)
def hipStreamWriteValue32(stream, ptr, value, flags): ...
@dll.bind((hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32), hipError_t)
def hipStreamWriteValue64(stream, ptr, value, flags): ...
@dll.bind((Pointer(hipEvent_t), ctypes.c_uint32), hipError_t)
def hipEventCreateWithFlags(event, flags): ...
@dll.bind((Pointer(hipEvent_t)), hipError_t)
def hipEventCreate(event): ...
@dll.bind((hipEvent_t, hipStream_t), hipError_t)
def hipEventRecord(event, stream): ...
@dll.bind((hipEvent_t), hipError_t)
def hipEventDestroy(event): ...
@dll.bind((hipEvent_t), hipError_t)
def hipEventSynchronize(event): ...
@dll.bind((Pointer(ctypes.c_float), hipEvent_t, hipEvent_t), hipError_t)
def hipEventElapsedTime(ms, start, stop): ...
@dll.bind((hipEvent_t), hipError_t)
def hipEventQuery(event): ...
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

@dll.bind((ctypes.c_void_p, hipPointer_attribute, hipDeviceptr_t), hipError_t)
def hipPointerSetAttribute(value, attribute, ptr): ...
@dll.bind((Pointer(hipPointerAttribute_t), ctypes.c_void_p), hipError_t)
def hipPointerGetAttributes(attributes, ptr): ...
@dll.bind((ctypes.c_void_p, hipPointer_attribute, hipDeviceptr_t), hipError_t)
def hipPointerGetAttribute(data, attribute, ptr): ...
@dll.bind((ctypes.c_uint32, Pointer(hipPointer_attribute), Pointer(ctypes.c_void_p), hipDeviceptr_t), hipError_t)
def hipDrvPointerGetAttributes(numAttributes, attributes, data, ptr): ...
@dll.bind((Pointer(hipExternalSemaphore_t), Pointer(hipExternalSemaphoreHandleDesc)), hipError_t)
def hipImportExternalSemaphore(extSem_out, semHandleDesc): ...
@dll.bind((Pointer(hipExternalSemaphore_t), Pointer(hipExternalSemaphoreSignalParams), ctypes.c_uint32, hipStream_t), hipError_t)
def hipSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream): ...
@dll.bind((Pointer(hipExternalSemaphore_t), Pointer(hipExternalSemaphoreWaitParams), ctypes.c_uint32, hipStream_t), hipError_t)
def hipWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream): ...
@dll.bind((hipExternalSemaphore_t), hipError_t)
def hipDestroyExternalSemaphore(extSem): ...
@dll.bind((Pointer(hipExternalMemory_t), Pointer(hipExternalMemoryHandleDesc)), hipError_t)
def hipImportExternalMemory(extMem_out, memHandleDesc): ...
@dll.bind((Pointer(ctypes.c_void_p), hipExternalMemory_t, Pointer(hipExternalMemoryBufferDesc)), hipError_t)
def hipExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc): ...
@dll.bind((hipExternalMemory_t), hipError_t)
def hipDestroyExternalMemory(extMem): ...
hipMipmappedArray_t = Pointer(hipMipmappedArray)
@dll.bind((Pointer(hipMipmappedArray_t), hipExternalMemory_t, Pointer(hipExternalMemoryMipmappedArrayDesc)), hipError_t)
def hipExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t), hipError_t)
def hipMalloc(ptr, size): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32), hipError_t)
def hipExtMallocWithFlags(ptr, sizeBytes, flags): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t), hipError_t)
def hipMallocHost(ptr, size): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t), hipError_t)
def hipMemAllocHost(ptr, size): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32), hipError_t)
def hipHostMalloc(ptr, size, flags): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32), hipError_t)
def hipMallocManaged(dev_ptr, size, flags): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_int32, hipStream_t), hipError_t)
def hipMemPrefetchAsync(dev_ptr, count, device, stream): ...
@dll.bind((ctypes.c_void_p, size_t, hipMemoryAdvise, ctypes.c_int32), hipError_t)
def hipMemAdvise(dev_ptr, count, advice, device): ...
@dll.bind((ctypes.c_void_p, size_t, hipMemRangeAttribute, ctypes.c_void_p, size_t), hipError_t)
def hipMemRangeGetAttribute(data, data_size, attribute, dev_ptr, count): ...
@dll.bind((Pointer(ctypes.c_void_p), Pointer(size_t), Pointer(hipMemRangeAttribute), size_t, ctypes.c_void_p, size_t), hipError_t)
def hipMemRangeGetAttributes(data, data_sizes, attributes, num_attributes, dev_ptr, count): ...
@dll.bind((hipStream_t, ctypes.c_void_p, size_t, ctypes.c_uint32), hipError_t)
def hipStreamAttachMemAsync(stream, dev_ptr, length, flags): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, hipStream_t), hipError_t)
def hipMallocAsync(dev_ptr, size, stream): ...
@dll.bind((ctypes.c_void_p, hipStream_t), hipError_t)
def hipFreeAsync(dev_ptr, stream): ...
@dll.bind((hipMemPool_t, size_t), hipError_t)
def hipMemPoolTrimTo(mem_pool, min_bytes_to_hold): ...
@dll.bind((hipMemPool_t, hipMemPoolAttr, ctypes.c_void_p), hipError_t)
def hipMemPoolSetAttribute(mem_pool, attr, value): ...
@dll.bind((hipMemPool_t, hipMemPoolAttr, ctypes.c_void_p), hipError_t)
def hipMemPoolGetAttribute(mem_pool, attr, value): ...
@dll.bind((hipMemPool_t, Pointer(hipMemAccessDesc), size_t), hipError_t)
def hipMemPoolSetAccess(mem_pool, desc_list, count): ...
@dll.bind((Pointer(hipMemAccessFlags), hipMemPool_t, Pointer(hipMemLocation)), hipError_t)
def hipMemPoolGetAccess(flags, mem_pool, location): ...
@dll.bind((Pointer(hipMemPool_t), Pointer(hipMemPoolProps)), hipError_t)
def hipMemPoolCreate(mem_pool, pool_props): ...
@dll.bind((hipMemPool_t), hipError_t)
def hipMemPoolDestroy(mem_pool): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, hipMemPool_t, hipStream_t), hipError_t)
def hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream): ...
@dll.bind((ctypes.c_void_p, hipMemPool_t, hipMemAllocationHandleType, ctypes.c_uint32), hipError_t)
def hipMemPoolExportToShareableHandle(shared_handle, mem_pool, handle_type, flags): ...
@dll.bind((Pointer(hipMemPool_t), ctypes.c_void_p, hipMemAllocationHandleType, ctypes.c_uint32), hipError_t)
def hipMemPoolImportFromShareableHandle(mem_pool, shared_handle, handle_type, flags): ...
@dll.bind((Pointer(hipMemPoolPtrExportData), ctypes.c_void_p), hipError_t)
def hipMemPoolExportPointer(export_data, dev_ptr): ...
@dll.bind((Pointer(ctypes.c_void_p), hipMemPool_t, Pointer(hipMemPoolPtrExportData)), hipError_t)
def hipMemPoolImportPointer(dev_ptr, mem_pool, export_data): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32), hipError_t)
def hipHostAlloc(ptr, size, flags): ...
@dll.bind((Pointer(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32), hipError_t)
def hipHostGetDevicePointer(devPtr, hstPtr, flags): ...
@dll.bind((Pointer(ctypes.c_uint32), ctypes.c_void_p), hipError_t)
def hipHostGetFlags(flagsPtr, hostPtr): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_uint32), hipError_t)
def hipHostRegister(hostPtr, sizeBytes, flags): ...
@dll.bind((ctypes.c_void_p), hipError_t)
def hipHostUnregister(hostPtr): ...
@dll.bind((Pointer(ctypes.c_void_p), Pointer(size_t), size_t, size_t), hipError_t)
def hipMallocPitch(ptr, pitch, width, height): ...
@dll.bind((Pointer(hipDeviceptr_t), Pointer(size_t), size_t, size_t, ctypes.c_uint32), hipError_t)
def hipMemAllocPitch(dptr, pitch, widthInBytes, height, elementSizeBytes): ...
@dll.bind((ctypes.c_void_p), hipError_t)
def hipFree(ptr): ...
@dll.bind((ctypes.c_void_p), hipError_t)
def hipFreeHost(ptr): ...
@dll.bind((ctypes.c_void_p), hipError_t)
def hipHostFree(ptr): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind), hipError_t)
def hipMemcpy(dst, src, sizeBytes, kind): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpyWithStream(dst, src, sizeBytes, kind, stream): ...
@dll.bind((hipDeviceptr_t, ctypes.c_void_p, size_t), hipError_t)
def hipMemcpyHtoD(dst, src, sizeBytes): ...
@dll.bind((ctypes.c_void_p, hipDeviceptr_t, size_t), hipError_t)
def hipMemcpyDtoH(dst, src, sizeBytes): ...
@dll.bind((hipDeviceptr_t, hipDeviceptr_t, size_t), hipError_t)
def hipMemcpyDtoD(dst, src, sizeBytes): ...
@dll.bind((hipDeviceptr_t, hipArray_t, size_t, size_t), hipError_t)
def hipMemcpyAtoD(dstDevice, srcArray, srcOffset, ByteCount): ...
@dll.bind((hipArray_t, size_t, hipDeviceptr_t, size_t), hipError_t)
def hipMemcpyDtoA(dstArray, dstOffset, srcDevice, ByteCount): ...
@dll.bind((hipArray_t, size_t, hipArray_t, size_t, size_t), hipError_t)
def hipMemcpyAtoA(dstArray, dstOffset, srcArray, srcOffset, ByteCount): ...
@dll.bind((hipDeviceptr_t, ctypes.c_void_p, size_t, hipStream_t), hipError_t)
def hipMemcpyHtoDAsync(dst, src, sizeBytes, stream): ...
@dll.bind((ctypes.c_void_p, hipDeviceptr_t, size_t, hipStream_t), hipError_t)
def hipMemcpyDtoHAsync(dst, src, sizeBytes, stream): ...
@dll.bind((hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t), hipError_t)
def hipMemcpyDtoDAsync(dst, src, sizeBytes, stream): ...
@dll.bind((ctypes.c_void_p, hipArray_t, size_t, size_t, hipStream_t), hipError_t)
def hipMemcpyAtoHAsync(dstHost, srcArray, srcOffset, ByteCount, stream): ...
@dll.bind((hipArray_t, size_t, ctypes.c_void_p, size_t, hipStream_t), hipError_t)
def hipMemcpyHtoAAsync(dstArray, dstOffset, srcHost, ByteCount, stream): ...
@dll.bind((Pointer(hipDeviceptr_t), Pointer(size_t), hipModule_t, Pointer(ctypes.c_char)), hipError_t)
def hipModuleGetGlobal(dptr, bytes, hmod, name): ...
@dll.bind((Pointer(ctypes.c_void_p), ctypes.c_void_p), hipError_t)
def hipGetSymbolAddress(devPtr, symbol): ...
@dll.bind((Pointer(size_t), ctypes.c_void_p), hipError_t)
def hipGetSymbolSize(size, symbol): ...
@dll.bind((Pointer(ctypes.c_char), Pointer(ctypes.c_void_p), ctypes.c_int32, uint64_t, Pointer(hipDriverProcAddressQueryResult)), hipError_t)
def hipGetProcAddress(symbol, pfn, hipVersion, flags, symbolStatus): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpyToSymbol(symbol, src, sizeBytes, offset, kind): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, stream): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpyFromSymbol(dst, symbol, sizeBytes, offset, kind): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, stream): ...
@dll.bind((ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpyAsync(dst, src, sizeBytes, kind, stream): ...
@dll.bind((ctypes.c_void_p, ctypes.c_int32, size_t), hipError_t)
def hipMemset(dst, value, sizeBytes): ...
@dll.bind((hipDeviceptr_t, ctypes.c_ubyte, size_t), hipError_t)
def hipMemsetD8(dest, value, count): ...
@dll.bind((hipDeviceptr_t, ctypes.c_ubyte, size_t, hipStream_t), hipError_t)
def hipMemsetD8Async(dest, value, count, stream): ...
@dll.bind((hipDeviceptr_t, ctypes.c_uint16, size_t), hipError_t)
def hipMemsetD16(dest, value, count): ...
@dll.bind((hipDeviceptr_t, ctypes.c_uint16, size_t, hipStream_t), hipError_t)
def hipMemsetD16Async(dest, value, count, stream): ...
@dll.bind((hipDeviceptr_t, ctypes.c_int32, size_t), hipError_t)
def hipMemsetD32(dest, value, count): ...
@dll.bind((ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t), hipError_t)
def hipMemsetAsync(dst, value, sizeBytes, stream): ...
@dll.bind((hipDeviceptr_t, ctypes.c_int32, size_t, hipStream_t), hipError_t)
def hipMemsetD32Async(dst, value, count, stream): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t), hipError_t)
def hipMemset2D(dst, pitch, value, width, height): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t, hipStream_t), hipError_t)
def hipMemset2DAsync(dst, pitch, value, width, height, stream): ...
@dll.bind((hipPitchedPtr, ctypes.c_int32, hipExtent), hipError_t)
def hipMemset3D(pitchedDevPtr, value, extent): ...
@dll.bind((hipPitchedPtr, ctypes.c_int32, hipExtent, hipStream_t), hipError_t)
def hipMemset3DAsync(pitchedDevPtr, value, extent, stream): ...
@dll.bind((Pointer(size_t), Pointer(size_t)), hipError_t)
def hipMemGetInfo(free, total): ...
@dll.bind((ctypes.c_void_p, Pointer(size_t)), hipError_t)
def hipMemPtrGetInfo(ptr, size): ...
@dll.bind((Pointer(hipArray_t), Pointer(hipChannelFormatDesc), size_t, size_t, ctypes.c_uint32), hipError_t)
def hipMallocArray(array, desc, width, height, flags): ...
class HIP_ARRAY_DESCRIPTOR(Struct): pass
HIP_ARRAY_DESCRIPTOR.SIZE = 24
HIP_ARRAY_DESCRIPTOR._fields_ = ['Width', 'Height', 'Format', 'NumChannels']
setattr(HIP_ARRAY_DESCRIPTOR, 'Width', field(0, size_t))
setattr(HIP_ARRAY_DESCRIPTOR, 'Height', field(8, size_t))
setattr(HIP_ARRAY_DESCRIPTOR, 'Format', field(16, hipArray_Format))
setattr(HIP_ARRAY_DESCRIPTOR, 'NumChannels', field(20, ctypes.c_uint32))
@dll.bind((Pointer(hipArray_t), Pointer(HIP_ARRAY_DESCRIPTOR)), hipError_t)
def hipArrayCreate(pHandle, pAllocateArray): ...
@dll.bind((hipArray_t), hipError_t)
def hipArrayDestroy(array): ...
class HIP_ARRAY3D_DESCRIPTOR(Struct): pass
HIP_ARRAY3D_DESCRIPTOR.SIZE = 40
HIP_ARRAY3D_DESCRIPTOR._fields_ = ['Width', 'Height', 'Depth', 'Format', 'NumChannels', 'Flags']
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Width', field(0, size_t))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Height', field(8, size_t))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Depth', field(16, size_t))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Format', field(24, hipArray_Format))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'NumChannels', field(28, ctypes.c_uint32))
setattr(HIP_ARRAY3D_DESCRIPTOR, 'Flags', field(32, ctypes.c_uint32))
@dll.bind((Pointer(hipArray_t), Pointer(HIP_ARRAY3D_DESCRIPTOR)), hipError_t)
def hipArray3DCreate(array, pAllocateArray): ...
@dll.bind((Pointer(hipPitchedPtr), hipExtent), hipError_t)
def hipMalloc3D(pitchedDevPtr, extent): ...
@dll.bind((hipArray_t), hipError_t)
def hipFreeArray(array): ...
@dll.bind((Pointer(hipArray_t), Pointer(hipChannelFormatDesc), hipExtent, ctypes.c_uint32), hipError_t)
def hipMalloc3DArray(array, desc, extent, flags): ...
@dll.bind((Pointer(hipChannelFormatDesc), Pointer(hipExtent), Pointer(ctypes.c_uint32), hipArray_t), hipError_t)
def hipArrayGetInfo(desc, extent, flags, array): ...
@dll.bind((Pointer(HIP_ARRAY_DESCRIPTOR), hipArray_t), hipError_t)
def hipArrayGetDescriptor(pArrayDescriptor, array): ...
@dll.bind((Pointer(HIP_ARRAY3D_DESCRIPTOR), hipArray_t), hipError_t)
def hipArray3DGetDescriptor(pArrayDescriptor, array): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind): ...
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
@dll.bind((Pointer(hip_Memcpy2D)), hipError_t)
def hipMemcpyParam2D(pCopy): ...
@dll.bind((Pointer(hip_Memcpy2D), hipStream_t), hipError_t)
def hipMemcpyParam2DAsync(pCopy, stream): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream): ...
@dll.bind((hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind): ...
@dll.bind((hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream): ...
hipArray_const_t = Pointer(hipArray)
@dll.bind((hipArray_t, size_t, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind): ...
@dll.bind((hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, hipMemcpyKind), hipError_t)
def hipMemcpyToArray(dst, wOffset, hOffset, src, count, kind): ...
@dll.bind((ctypes.c_void_p, hipArray_const_t, size_t, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpyFromArray(dst, srcArray, wOffset, hOffset, count, kind): ...
@dll.bind((ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind), hipError_t)
def hipMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind): ...
@dll.bind((ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind, hipStream_t), hipError_t)
def hipMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream): ...
@dll.bind((ctypes.c_void_p, hipArray_t, size_t, size_t), hipError_t)
def hipMemcpyAtoH(dst, srcArray, srcOffset, count): ...
@dll.bind((hipArray_t, size_t, ctypes.c_void_p, size_t), hipError_t)
def hipMemcpyHtoA(dstArray, dstOffset, srcHost, count): ...
@dll.bind((Pointer(hipMemcpy3DParms)), hipError_t)
def hipMemcpy3D(p): ...
@dll.bind((Pointer(hipMemcpy3DParms), hipStream_t), hipError_t)
def hipMemcpy3DAsync(p, stream): ...
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
@dll.bind((Pointer(HIP_MEMCPY3D)), hipError_t)
def hipDrvMemcpy3D(pCopy): ...
@dll.bind((Pointer(HIP_MEMCPY3D), hipStream_t), hipError_t)
def hipDrvMemcpy3DAsync(pCopy, stream): ...
@dll.bind((Pointer(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32), hipError_t)
def hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId): ...
@dll.bind((ctypes.c_int32, ctypes.c_uint32), hipError_t)
def hipDeviceEnablePeerAccess(peerDeviceId, flags): ...
@dll.bind((ctypes.c_int32), hipError_t)
def hipDeviceDisablePeerAccess(peerDeviceId): ...
@dll.bind((Pointer(hipDeviceptr_t), Pointer(size_t), hipDeviceptr_t), hipError_t)
def hipMemGetAddressRange(pbase, psize, dptr): ...
@dll.bind((ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t), hipError_t)
def hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes): ...
@dll.bind((ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t), hipError_t)
def hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream): ...
@dll.bind((Pointer(hipCtx_t), ctypes.c_uint32, hipDevice_t), hipError_t)
def hipCtxCreate(ctx, flags, device): ...
@dll.bind((hipCtx_t), hipError_t)
def hipCtxDestroy(ctx): ...
@dll.bind((Pointer(hipCtx_t)), hipError_t)
def hipCtxPopCurrent(ctx): ...
@dll.bind((hipCtx_t), hipError_t)
def hipCtxPushCurrent(ctx): ...
@dll.bind((hipCtx_t), hipError_t)
def hipCtxSetCurrent(ctx): ...
@dll.bind((Pointer(hipCtx_t)), hipError_t)
def hipCtxGetCurrent(ctx): ...
@dll.bind((Pointer(hipDevice_t)), hipError_t)
def hipCtxGetDevice(device): ...
@dll.bind((hipCtx_t, Pointer(ctypes.c_int32)), hipError_t)
def hipCtxGetApiVersion(ctx, apiVersion): ...
@dll.bind((Pointer(hipFuncCache_t)), hipError_t)
def hipCtxGetCacheConfig(cacheConfig): ...
@dll.bind((hipFuncCache_t), hipError_t)
def hipCtxSetCacheConfig(cacheConfig): ...
@dll.bind((hipSharedMemConfig), hipError_t)
def hipCtxSetSharedMemConfig(config): ...
@dll.bind((Pointer(hipSharedMemConfig)), hipError_t)
def hipCtxGetSharedMemConfig(pConfig): ...
@dll.bind((), hipError_t)
def hipCtxSynchronize(): ...
@dll.bind((Pointer(ctypes.c_uint32)), hipError_t)
def hipCtxGetFlags(flags): ...
@dll.bind((hipCtx_t, ctypes.c_uint32), hipError_t)
def hipCtxEnablePeerAccess(peerCtx, flags): ...
@dll.bind((hipCtx_t), hipError_t)
def hipCtxDisablePeerAccess(peerCtx): ...
@dll.bind((hipDevice_t, Pointer(ctypes.c_uint32), Pointer(ctypes.c_int32)), hipError_t)
def hipDevicePrimaryCtxGetState(dev, flags, active): ...
@dll.bind((hipDevice_t), hipError_t)
def hipDevicePrimaryCtxRelease(dev): ...
@dll.bind((Pointer(hipCtx_t), hipDevice_t), hipError_t)
def hipDevicePrimaryCtxRetain(pctx, dev): ...
@dll.bind((hipDevice_t), hipError_t)
def hipDevicePrimaryCtxReset(dev): ...
@dll.bind((hipDevice_t, ctypes.c_uint32), hipError_t)
def hipDevicePrimaryCtxSetFlags(dev, flags): ...
@dll.bind((Pointer(hipModule_t), Pointer(ctypes.c_char)), hipError_t)
def hipModuleLoad(module, fname): ...
@dll.bind((hipModule_t), hipError_t)
def hipModuleUnload(module): ...
@dll.bind((Pointer(hipFunction_t), hipModule_t, Pointer(ctypes.c_char)), hipError_t)
def hipModuleGetFunction(function, module, kname): ...
@dll.bind((Pointer(hipFuncAttributes), ctypes.c_void_p), hipError_t)
def hipFuncGetAttributes(attr, func): ...
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

@dll.bind((Pointer(ctypes.c_int32), hipFunction_attribute, hipFunction_t), hipError_t)
def hipFuncGetAttribute(value, attrib, hfunc): ...
@dll.bind((Pointer(hipFunction_t), ctypes.c_void_p), hipError_t)
def hipGetFuncBySymbol(functionPtr, symbolPtr): ...
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
@dll.bind((Pointer(Pointer(textureReference)), hipModule_t, Pointer(ctypes.c_char)), hipError_t)
def hipModuleGetTexRef(texRef, hmod, name): ...
@dll.bind((Pointer(hipModule_t), ctypes.c_void_p), hipError_t)
def hipModuleLoadData(module, image): ...
@dll.bind((Pointer(hipModule_t), ctypes.c_void_p, ctypes.c_uint32, Pointer(hipJitOption), Pointer(ctypes.c_void_p)), hipError_t)
def hipModuleLoadDataEx(module, image, numOptions, options, optionValues): ...
@dll.bind((hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p)), hipError_t)
def hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra): ...
@dll.bind((hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, Pointer(ctypes.c_void_p)), hipError_t)
def hipModuleLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams): ...
@dll.bind((Pointer(hipFunctionLaunchParams), ctypes.c_uint32, ctypes.c_uint32), hipError_t)
def hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags): ...
@dll.bind((ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), ctypes.c_uint32, hipStream_t), hipError_t)
def hipLaunchCooperativeKernel(f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream): ...
@dll.bind((Pointer(hipLaunchParams), ctypes.c_int32, ctypes.c_uint32), hipError_t)
def hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags): ...
@dll.bind((Pointer(hipLaunchParams), ctypes.c_int32, ctypes.c_uint32), hipError_t)
def hipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32), hipError_t)
def hipModuleOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32, ctypes.c_uint32), hipError_t)
def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit, flags): ...
@dll.bind((Pointer(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t), hipError_t)
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk): ...
@dll.bind((Pointer(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t, ctypes.c_uint32), hipError_t)
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize, dynSharedMemPerBlk, flags): ...
@dll.bind((Pointer(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int32, size_t), hipError_t)
def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk): ...
@dll.bind((Pointer(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int32, size_t, ctypes.c_uint32), hipError_t)
def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize, dynSharedMemPerBlk, flags): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), ctypes.c_void_p, size_t, ctypes.c_int32), hipError_t)
def hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit): ...
@dll.bind((), hipError_t)
def hipProfilerStart(): ...
@dll.bind((), hipError_t)
def hipProfilerStop(): ...
@dll.bind((dim3, dim3, size_t, hipStream_t), hipError_t)
def hipConfigureCall(gridDim, blockDim, sharedMem, stream): ...
@dll.bind((ctypes.c_void_p, size_t, size_t), hipError_t)
def hipSetupArgument(arg, size, offset): ...
@dll.bind((ctypes.c_void_p), hipError_t)
def hipLaunchByPtr(func): ...
@dll.bind((dim3, dim3, size_t, hipStream_t), hipError_t)
def __hipPushCallConfiguration(gridDim, blockDim, sharedMem, stream): ...
@dll.bind((Pointer(dim3), Pointer(dim3), Pointer(size_t), Pointer(hipStream_t)), hipError_t)
def __hipPopCallConfiguration(gridDim, blockDim, sharedMem, stream): ...
@dll.bind((ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), size_t, hipStream_t), hipError_t)
def hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream): ...
@dll.bind((hipStream_t, hipHostFn_t, ctypes.c_void_p), hipError_t)
def hipLaunchHostFunc(stream, fn, userData): ...
@dll.bind((Pointer(hip_Memcpy2D)), hipError_t)
def hipDrvMemcpy2DUnaligned(pCopy): ...
@dll.bind((ctypes.c_void_p, dim3, dim3, Pointer(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32), hipError_t)
def hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags): ...
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
@dll.bind((Pointer(hipTextureObject_t), Pointer(hipResourceDesc), Pointer(hipTextureDesc), Pointer(hipResourceViewDesc)), hipError_t)
def hipCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc): ...
@dll.bind((hipTextureObject_t), hipError_t)
def hipDestroyTextureObject(textureObject): ...
@dll.bind((Pointer(hipChannelFormatDesc), hipArray_const_t), hipError_t)
def hipGetChannelDesc(desc, array): ...
@dll.bind((Pointer(hipResourceDesc), hipTextureObject_t), hipError_t)
def hipGetTextureObjectResourceDesc(pResDesc, textureObject): ...
@dll.bind((Pointer(hipResourceViewDesc), hipTextureObject_t), hipError_t)
def hipGetTextureObjectResourceViewDesc(pResViewDesc, textureObject): ...
@dll.bind((Pointer(hipTextureDesc), hipTextureObject_t), hipError_t)
def hipGetTextureObjectTextureDesc(pTexDesc, textureObject): ...
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
@dll.bind((Pointer(hipTextureObject_t), Pointer(HIP_RESOURCE_DESC), Pointer(HIP_TEXTURE_DESC), Pointer(HIP_RESOURCE_VIEW_DESC)), hipError_t)
def hipTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc): ...
@dll.bind((hipTextureObject_t), hipError_t)
def hipTexObjectDestroy(texObject): ...
@dll.bind((Pointer(HIP_RESOURCE_DESC), hipTextureObject_t), hipError_t)
def hipTexObjectGetResourceDesc(pResDesc, texObject): ...
@dll.bind((Pointer(HIP_RESOURCE_VIEW_DESC), hipTextureObject_t), hipError_t)
def hipTexObjectGetResourceViewDesc(pResViewDesc, texObject): ...
@dll.bind((Pointer(HIP_TEXTURE_DESC), hipTextureObject_t), hipError_t)
def hipTexObjectGetTextureDesc(pTexDesc, texObject): ...
@dll.bind((Pointer(hipMipmappedArray_t), Pointer(hipChannelFormatDesc), hipExtent, ctypes.c_uint32, ctypes.c_uint32), hipError_t)
def hipMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags): ...
@dll.bind((hipMipmappedArray_t), hipError_t)
def hipFreeMipmappedArray(mipmappedArray): ...
hipMipmappedArray_const_t = Pointer(hipMipmappedArray)
@dll.bind((Pointer(hipArray_t), hipMipmappedArray_const_t, ctypes.c_uint32), hipError_t)
def hipGetMipmappedArrayLevel(levelArray, mipmappedArray, level): ...
@dll.bind((Pointer(hipMipmappedArray_t), Pointer(HIP_ARRAY3D_DESCRIPTOR), ctypes.c_uint32), hipError_t)
def hipMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels): ...
@dll.bind((hipMipmappedArray_t), hipError_t)
def hipMipmappedArrayDestroy(hMipmappedArray): ...
@dll.bind((Pointer(hipArray_t), hipMipmappedArray_t, ctypes.c_uint32), hipError_t)
def hipMipmappedArrayGetLevel(pLevelArray, hMipMappedArray, level): ...
@dll.bind((Pointer(textureReference), hipMipmappedArray_const_t, Pointer(hipChannelFormatDesc)), hipError_t)
def hipBindTextureToMipmappedArray(tex, mipmappedArray, desc): ...
@dll.bind((Pointer(Pointer(textureReference)), ctypes.c_void_p), hipError_t)
def hipGetTextureReference(texref, symbol): ...
@dll.bind((Pointer(ctypes.c_float), Pointer(textureReference)), hipError_t)
def hipTexRefGetBorderColor(pBorderColor, texRef): ...
@dll.bind((Pointer(hipArray_t), Pointer(textureReference)), hipError_t)
def hipTexRefGetArray(pArray, texRef): ...
@dll.bind((Pointer(textureReference), ctypes.c_int32, hipTextureAddressMode), hipError_t)
def hipTexRefSetAddressMode(texRef, dim, am): ...
@dll.bind((Pointer(textureReference), hipArray_const_t, ctypes.c_uint32), hipError_t)
def hipTexRefSetArray(tex, array, flags): ...
@dll.bind((Pointer(textureReference), hipTextureFilterMode), hipError_t)
def hipTexRefSetFilterMode(texRef, fm): ...
@dll.bind((Pointer(textureReference), ctypes.c_uint32), hipError_t)
def hipTexRefSetFlags(texRef, Flags): ...
@dll.bind((Pointer(textureReference), hipArray_Format, ctypes.c_int32), hipError_t)
def hipTexRefSetFormat(texRef, fmt, NumPackedComponents): ...
@dll.bind((Pointer(size_t), Pointer(textureReference), ctypes.c_void_p, Pointer(hipChannelFormatDesc), size_t), hipError_t)
def hipBindTexture(offset, tex, devPtr, desc, size): ...
@dll.bind((Pointer(size_t), Pointer(textureReference), ctypes.c_void_p, Pointer(hipChannelFormatDesc), size_t, size_t, size_t), hipError_t)
def hipBindTexture2D(offset, tex, devPtr, desc, width, height, pitch): ...
@dll.bind((Pointer(textureReference), hipArray_const_t, Pointer(hipChannelFormatDesc)), hipError_t)
def hipBindTextureToArray(tex, array, desc): ...
@dll.bind((Pointer(size_t), Pointer(textureReference)), hipError_t)
def hipGetTextureAlignmentOffset(offset, texref): ...
@dll.bind((Pointer(textureReference)), hipError_t)
def hipUnbindTexture(tex): ...
@dll.bind((Pointer(hipDeviceptr_t), Pointer(textureReference)), hipError_t)
def hipTexRefGetAddress(dev_ptr, texRef): ...
@dll.bind((Pointer(hipTextureAddressMode), Pointer(textureReference), ctypes.c_int32), hipError_t)
def hipTexRefGetAddressMode(pam, texRef, dim): ...
@dll.bind((Pointer(hipTextureFilterMode), Pointer(textureReference)), hipError_t)
def hipTexRefGetFilterMode(pfm, texRef): ...
@dll.bind((Pointer(ctypes.c_uint32), Pointer(textureReference)), hipError_t)
def hipTexRefGetFlags(pFlags, texRef): ...
@dll.bind((Pointer(hipArray_Format), Pointer(ctypes.c_int32), Pointer(textureReference)), hipError_t)
def hipTexRefGetFormat(pFormat, pNumChannels, texRef): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(textureReference)), hipError_t)
def hipTexRefGetMaxAnisotropy(pmaxAnsio, texRef): ...
@dll.bind((Pointer(hipTextureFilterMode), Pointer(textureReference)), hipError_t)
def hipTexRefGetMipmapFilterMode(pfm, texRef): ...
@dll.bind((Pointer(ctypes.c_float), Pointer(textureReference)), hipError_t)
def hipTexRefGetMipmapLevelBias(pbias, texRef): ...
@dll.bind((Pointer(ctypes.c_float), Pointer(ctypes.c_float), Pointer(textureReference)), hipError_t)
def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef): ...
@dll.bind((Pointer(hipMipmappedArray_t), Pointer(textureReference)), hipError_t)
def hipTexRefGetMipMappedArray(pArray, texRef): ...
@dll.bind((Pointer(size_t), Pointer(textureReference), hipDeviceptr_t, size_t), hipError_t)
def hipTexRefSetAddress(ByteOffset, texRef, dptr, bytes): ...
@dll.bind((Pointer(textureReference), Pointer(HIP_ARRAY_DESCRIPTOR), hipDeviceptr_t, size_t), hipError_t)
def hipTexRefSetAddress2D(texRef, desc, dptr, Pitch): ...
@dll.bind((Pointer(textureReference), ctypes.c_uint32), hipError_t)
def hipTexRefSetMaxAnisotropy(texRef, maxAniso): ...
@dll.bind((Pointer(textureReference), Pointer(ctypes.c_float)), hipError_t)
def hipTexRefSetBorderColor(texRef, pBorderColor): ...
@dll.bind((Pointer(textureReference), hipTextureFilterMode), hipError_t)
def hipTexRefSetMipmapFilterMode(texRef, fm): ...
@dll.bind((Pointer(textureReference), ctypes.c_float), hipError_t)
def hipTexRefSetMipmapLevelBias(texRef, bias): ...
@dll.bind((Pointer(textureReference), ctypes.c_float, ctypes.c_float), hipError_t)
def hipTexRefSetMipmapLevelClamp(texRef, minMipMapLevelClamp, maxMipMapLevelClamp): ...
@dll.bind((Pointer(textureReference), Pointer(hipMipmappedArray), ctypes.c_uint32), hipError_t)
def hipTexRefSetMipmappedArray(texRef, mipmappedArray, Flags): ...
@dll.bind((uint32_t), Pointer(ctypes.c_char))
def hipApiName(id): ...
@dll.bind((hipFunction_t), Pointer(ctypes.c_char))
def hipKernelNameRef(f): ...
@dll.bind((ctypes.c_void_p, hipStream_t), Pointer(ctypes.c_char))
def hipKernelNameRefByPtr(hostFunction, stream): ...
@dll.bind((hipStream_t), ctypes.c_int32)
def hipGetStreamDeviceId(stream): ...
@dll.bind((hipStream_t, hipStreamCaptureMode), hipError_t)
def hipStreamBeginCapture(stream, mode): ...
@dll.bind((hipStream_t, hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphEdgeData), size_t, hipStreamCaptureMode), hipError_t)
def hipStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode): ...
@dll.bind((hipStream_t, Pointer(hipGraph_t)), hipError_t)
def hipStreamEndCapture(stream, pGraph): ...
@dll.bind((hipStream_t, Pointer(hipStreamCaptureStatus), Pointer(ctypes.c_uint64)), hipError_t)
def hipStreamGetCaptureInfo(stream, pCaptureStatus, pId): ...
@dll.bind((hipStream_t, Pointer(hipStreamCaptureStatus), Pointer(ctypes.c_uint64), Pointer(hipGraph_t), Pointer(Pointer(hipGraphNode_t)), Pointer(size_t)), hipError_t)
def hipStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out): ...
@dll.bind((hipStream_t, Pointer(hipStreamCaptureStatus)), hipError_t)
def hipStreamIsCapturing(stream, pCaptureStatus): ...
@dll.bind((hipStream_t, Pointer(hipGraphNode_t), size_t, ctypes.c_uint32), hipError_t)
def hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags): ...
@dll.bind((Pointer(hipStreamCaptureMode)), hipError_t)
def hipThreadExchangeStreamCaptureMode(mode): ...
@dll.bind((Pointer(hipGraph_t), ctypes.c_uint32), hipError_t)
def hipGraphCreate(pGraph, flags): ...
@dll.bind((hipGraph_t), hipError_t)
def hipGraphDestroy(graph): ...
@dll.bind((hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphNode_t), size_t), hipError_t)
def hipGraphAddDependencies(graph, _nfrom, to, numDependencies): ...
@dll.bind((hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphNode_t), size_t), hipError_t)
def hipGraphRemoveDependencies(graph, _nfrom, to, numDependencies): ...
@dll.bind((hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphNode_t), Pointer(size_t)), hipError_t)
def hipGraphGetEdges(graph, _nfrom, to, numEdges): ...
@dll.bind((hipGraph_t, Pointer(hipGraphNode_t), Pointer(size_t)), hipError_t)
def hipGraphGetNodes(graph, nodes, numNodes): ...
@dll.bind((hipGraph_t, Pointer(hipGraphNode_t), Pointer(size_t)), hipError_t)
def hipGraphGetRootNodes(graph, pRootNodes, pNumRootNodes): ...
@dll.bind((hipGraphNode_t, Pointer(hipGraphNode_t), Pointer(size_t)), hipError_t)
def hipGraphNodeGetDependencies(node, pDependencies, pNumDependencies): ...
@dll.bind((hipGraphNode_t, Pointer(hipGraphNode_t), Pointer(size_t)), hipError_t)
def hipGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes): ...
@dll.bind((hipGraphNode_t, Pointer(hipGraphNodeType)), hipError_t)
def hipGraphNodeGetType(node, pType): ...
@dll.bind((hipGraphNode_t), hipError_t)
def hipGraphDestroyNode(node): ...
@dll.bind((Pointer(hipGraph_t), hipGraph_t), hipError_t)
def hipGraphClone(pGraphClone, originalGraph): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraphNode_t, hipGraph_t), hipError_t)
def hipGraphNodeFindInClone(pNode, originalNode, clonedGraph): ...
@dll.bind((Pointer(hipGraphExec_t), hipGraph_t, Pointer(hipGraphNode_t), Pointer(ctypes.c_char), size_t), hipError_t)
def hipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize): ...
@dll.bind((Pointer(hipGraphExec_t), hipGraph_t, ctypes.c_uint64), hipError_t)
def hipGraphInstantiateWithFlags(pGraphExec, graph, flags): ...
@dll.bind((Pointer(hipGraphExec_t), hipGraph_t, Pointer(hipGraphInstantiateParams)), hipError_t)
def hipGraphInstantiateWithParams(pGraphExec, graph, instantiateParams): ...
@dll.bind((hipGraphExec_t, hipStream_t), hipError_t)
def hipGraphLaunch(graphExec, stream): ...
@dll.bind((hipGraphExec_t, hipStream_t), hipError_t)
def hipGraphUpload(graphExec, stream): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipGraphNodeParams)), hipError_t)
def hipGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams): ...
@dll.bind((hipGraphExec_t), hipError_t)
def hipGraphExecDestroy(graphExec): ...
@dll.bind((hipGraphExec_t, hipGraph_t, Pointer(hipGraphNode_t), Pointer(hipGraphExecUpdateResult)), hipError_t)
def hipGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipKernelNodeParams)), hipError_t)
def hipGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipKernelNodeParams)), hipError_t)
def hipGraphKernelNodeGetParams(node, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipKernelNodeParams)), hipError_t)
def hipGraphKernelNodeSetParams(node, pNodeParams): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(hipKernelNodeParams)), hipError_t)
def hipGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(HIP_MEMCPY3D), hipCtx_t), hipError_t)
def hipDrvGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipMemcpy3DParms)), hipError_t)
def hipGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipMemcpy3DParms)), hipError_t)
def hipGraphMemcpyNodeGetParams(node, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipMemcpy3DParms)), hipError_t)
def hipGraphMemcpyNodeSetParams(node, pNodeParams): ...
@dll.bind((hipGraphNode_t, hipLaunchAttributeID, Pointer(hipLaunchAttributeValue)), hipError_t)
def hipGraphKernelNodeSetAttribute(hNode, attr, value): ...
@dll.bind((hipGraphNode_t, hipLaunchAttributeID, Pointer(hipLaunchAttributeValue)), hipError_t)
def hipGraphKernelNodeGetAttribute(hNode, attr, value): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(hipMemcpy3DParms)), hipError_t)
def hipGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind), hipError_t)
def hipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind): ...
@dll.bind((hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind), hipError_t)
def hipGraphMemcpyNodeSetParams1D(node, dst, src, count, kind): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind), hipError_t)
def hipGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind): ...
@dll.bind((hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind): ...
@dll.bind((hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind), hipError_t)
def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipMemsetParams)), hipError_t)
def hipGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipMemsetParams)), hipError_t)
def hipGraphMemsetNodeGetParams(node, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipMemsetParams)), hipError_t)
def hipGraphMemsetNodeSetParams(node, pNodeParams): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(hipMemsetParams)), hipError_t)
def hipGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipHostNodeParams)), hipError_t)
def hipGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipHostNodeParams)), hipError_t)
def hipGraphHostNodeGetParams(node, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipHostNodeParams)), hipError_t)
def hipGraphHostNodeSetParams(node, pNodeParams): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(hipHostNodeParams)), hipError_t)
def hipGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, hipGraph_t), hipError_t)
def hipGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph): ...
@dll.bind((hipGraphNode_t, Pointer(hipGraph_t)), hipError_t)
def hipGraphChildGraphNodeGetGraph(node, pGraph): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, hipGraph_t), hipError_t)
def hipGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t), hipError_t)
def hipGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, hipEvent_t), hipError_t)
def hipGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event): ...
@dll.bind((hipGraphNode_t, Pointer(hipEvent_t)), hipError_t)
def hipGraphEventRecordNodeGetEvent(node, event_out): ...
@dll.bind((hipGraphNode_t, hipEvent_t), hipError_t)
def hipGraphEventRecordNodeSetEvent(node, event): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, hipEvent_t), hipError_t)
def hipGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, hipEvent_t), hipError_t)
def hipGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event): ...
@dll.bind((hipGraphNode_t, Pointer(hipEvent_t)), hipError_t)
def hipGraphEventWaitNodeGetEvent(node, event_out): ...
@dll.bind((hipGraphNode_t, hipEvent_t), hipError_t)
def hipGraphEventWaitNodeSetEvent(node, event): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, hipEvent_t), hipError_t)
def hipGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipMemAllocNodeParams)), hipError_t)
def hipGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipMemAllocNodeParams)), hipError_t)
def hipGraphMemAllocNodeGetParams(node, pNodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, ctypes.c_void_p), hipError_t)
def hipGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dev_ptr): ...
@dll.bind((hipGraphNode_t, ctypes.c_void_p), hipError_t)
def hipGraphMemFreeNodeGetParams(node, dev_ptr): ...
@dll.bind((ctypes.c_int32, hipGraphMemAttributeType, ctypes.c_void_p), hipError_t)
def hipDeviceGetGraphMemAttribute(device, attr, value): ...
@dll.bind((ctypes.c_int32, hipGraphMemAttributeType, ctypes.c_void_p), hipError_t)
def hipDeviceSetGraphMemAttribute(device, attr, value): ...
@dll.bind((ctypes.c_int32), hipError_t)
def hipDeviceGraphMemTrim(device): ...
@dll.bind((Pointer(hipUserObject_t), ctypes.c_void_p, hipHostFn_t, ctypes.c_uint32, ctypes.c_uint32), hipError_t)
def hipUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags): ...
@dll.bind((hipUserObject_t, ctypes.c_uint32), hipError_t)
def hipUserObjectRelease(object, count): ...
@dll.bind((hipUserObject_t, ctypes.c_uint32), hipError_t)
def hipUserObjectRetain(object, count): ...
@dll.bind((hipGraph_t, hipUserObject_t, ctypes.c_uint32, ctypes.c_uint32), hipError_t)
def hipGraphRetainUserObject(graph, object, count, flags): ...
@dll.bind((hipGraph_t, hipUserObject_t, ctypes.c_uint32), hipError_t)
def hipGraphReleaseUserObject(graph, object, count): ...
@dll.bind((hipGraph_t, Pointer(ctypes.c_char), ctypes.c_uint32), hipError_t)
def hipGraphDebugDotPrint(graph, path, flags): ...
@dll.bind((hipGraphNode_t, hipGraphNode_t), hipError_t)
def hipGraphKernelNodeCopyAttributes(hSrc, hDst): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, ctypes.c_uint32), hipError_t)
def hipGraphNodeSetEnabled(hGraphExec, hNode, isEnabled): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(ctypes.c_uint32)), hipError_t)
def hipGraphNodeGetEnabled(hGraphExec, hNode, isEnabled): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipExternalSemaphoreWaitNodeParams)), hipError_t)
def hipGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(hipExternalSemaphoreSignalNodeParams)), hipError_t)
def hipGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipExternalSemaphoreSignalNodeParams)), hipError_t)
def hipGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipExternalSemaphoreWaitNodeParams)), hipError_t)
def hipGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams): ...
@dll.bind((hipGraphNode_t, Pointer(hipExternalSemaphoreSignalNodeParams)), hipError_t)
def hipGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out): ...
@dll.bind((hipGraphNode_t, Pointer(hipExternalSemaphoreWaitNodeParams)), hipError_t)
def hipGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(hipExternalSemaphoreSignalNodeParams)), hipError_t)
def hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((hipGraphExec_t, hipGraphNode_t, Pointer(hipExternalSemaphoreWaitNodeParams)), hipError_t)
def hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((Pointer(hipGraphNode_t), hipGraph_t, Pointer(hipGraphNode_t), size_t, Pointer(HIP_MEMSET_NODE_PARAMS), hipCtx_t), hipError_t)
def hipDrvGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx): ...
@dll.bind((ctypes.c_void_p, size_t), hipError_t)
def hipMemAddressFree(devPtr, size): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, size_t, ctypes.c_void_p, ctypes.c_uint64), hipError_t)
def hipMemAddressReserve(ptr, size, alignment, addr, flags): ...
@dll.bind((Pointer(hipMemGenericAllocationHandle_t), size_t, Pointer(hipMemAllocationProp), ctypes.c_uint64), hipError_t)
def hipMemCreate(handle, size, prop, flags): ...
@dll.bind((ctypes.c_void_p, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, ctypes.c_uint64), hipError_t)
def hipMemExportToShareableHandle(shareableHandle, handle, handleType, flags): ...
@dll.bind((Pointer(ctypes.c_uint64), Pointer(hipMemLocation), ctypes.c_void_p), hipError_t)
def hipMemGetAccess(flags, location, ptr): ...
@dll.bind((Pointer(size_t), Pointer(hipMemAllocationProp), hipMemAllocationGranularity_flags), hipError_t)
def hipMemGetAllocationGranularity(granularity, prop, option): ...
@dll.bind((Pointer(hipMemAllocationProp), hipMemGenericAllocationHandle_t), hipError_t)
def hipMemGetAllocationPropertiesFromHandle(prop, handle): ...
@dll.bind((Pointer(hipMemGenericAllocationHandle_t), ctypes.c_void_p, hipMemAllocationHandleType), hipError_t)
def hipMemImportFromShareableHandle(handle, osHandle, shHandleType): ...
@dll.bind((ctypes.c_void_p, size_t, size_t, hipMemGenericAllocationHandle_t, ctypes.c_uint64), hipError_t)
def hipMemMap(ptr, size, offset, handle, flags): ...
@dll.bind((Pointer(hipArrayMapInfo), ctypes.c_uint32, hipStream_t), hipError_t)
def hipMemMapArrayAsync(mapInfoList, count, stream): ...
@dll.bind((hipMemGenericAllocationHandle_t), hipError_t)
def hipMemRelease(handle): ...
@dll.bind((Pointer(hipMemGenericAllocationHandle_t), ctypes.c_void_p), hipError_t)
def hipMemRetainAllocationHandle(handle, addr): ...
@dll.bind((ctypes.c_void_p, size_t, Pointer(hipMemAccessDesc), size_t), hipError_t)
def hipMemSetAccess(ptr, size, desc, count): ...
@dll.bind((ctypes.c_void_p, size_t), hipError_t)
def hipMemUnmap(ptr, size): ...
@dll.bind((ctypes.c_int32, Pointer(hipGraphicsResource_t), hipStream_t), hipError_t)
def hipGraphicsMapResources(count, resources, stream): ...
@dll.bind((Pointer(hipArray_t), hipGraphicsResource_t, ctypes.c_uint32, ctypes.c_uint32), hipError_t)
def hipGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel): ...
@dll.bind((Pointer(ctypes.c_void_p), Pointer(size_t), hipGraphicsResource_t), hipError_t)
def hipGraphicsResourceGetMappedPointer(devPtr, size, resource): ...
@dll.bind((ctypes.c_int32, Pointer(hipGraphicsResource_t), hipStream_t), hipError_t)
def hipGraphicsUnmapResources(count, resources, stream): ...
@dll.bind((hipGraphicsResource_t), hipError_t)
def hipGraphicsUnregisterResource(resource): ...
class __hip_surface(Struct): pass
hipSurfaceObject_t = Pointer(__hip_surface)
@dll.bind((Pointer(hipSurfaceObject_t), Pointer(hipResourceDesc)), hipError_t)
def hipCreateSurfaceObject(pSurfObject, pResDesc): ...
@dll.bind((hipSurfaceObject_t), hipError_t)
def hipDestroySurfaceObject(surfaceObject): ...
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