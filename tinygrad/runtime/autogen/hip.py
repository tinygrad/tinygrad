# mypy: ignore-errors
import ctypes, os
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
def dll():
  try: return ctypes.CDLL(os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
  except: pass
  return None
dll = dll()
hipError_t = CEnum(ctypes.c_uint)
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

class ihipModuleSymbol_t(ctypes.Structure): pass
ihipModuleSymbol_t._fields_ = []
hipFunction_t = ctypes.POINTER(ihipModuleSymbol_t)
uint32_t = ctypes.c_uint
size_t = ctypes.c_ulong
class ihipStream_t(ctypes.Structure): pass
ihipStream_t._fields_ = []
hipStream_t = ctypes.POINTER(ihipStream_t)
class ihipEvent_t(ctypes.Structure): pass
ihipEvent_t._fields_ = []
hipEvent_t = ctypes.POINTER(ihipEvent_t)
# __attribute__((visibility("default"))) hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void **kernelParams, void **extra, hipEvent_t startEvent = __null, hipEvent_t stopEvent = __null, uint32_t flags = 0)
try: (hipExtModuleLaunchKernel:=dll.hipExtModuleLaunchKernel).restype, hipExtModuleLaunchKernel.argtypes = hipError_t, [hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), hipEvent_t, hipEvent_t, uint32_t]
except AttributeError: pass

# __attribute__((deprecated("use hipExtModuleLaunchKernel instead"))) __attribute__((visibility("default"))) hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ, uint32_t localWorkSizeX, uint32_t localWorkSizeY, uint32_t localWorkSizeZ, size_t sharedMemBytes, hipStream_t hStream, void **kernelParams, void **extra, hipEvent_t startEvent = __null, hipEvent_t stopEvent = __null)
try: (hipHccModuleLaunchKernel:=dll.hipHccModuleLaunchKernel).restype, hipHccModuleLaunchKernel.argtypes = hipError_t, [hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), hipEvent_t, hipEvent_t]
except AttributeError: pass

class dim3(ctypes.Structure): pass
dim3._fields_ = [
  ('x', uint32_t),
  ('y', uint32_t),
  ('z', uint32_t),
]
# hipError_t hipExtLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags)
try: (hipExtLaunchKernel:=dll.hipExtLaunchKernel).restype, hipExtLaunchKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int]
except AttributeError: pass

hiprtcResult = CEnum(ctypes.c_uint)
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

hiprtcJIT_option = CEnum(ctypes.c_uint)
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

hiprtcJITInputType = CEnum(ctypes.c_uint)
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

class ihiprtcLinkState(ctypes.Structure): pass
ihiprtcLinkState._fields_ = []
hiprtcLinkState = ctypes.POINTER(ihiprtcLinkState)
# const char *hiprtcGetErrorString(hiprtcResult result)
try: (hiprtcGetErrorString:=dll.hiprtcGetErrorString).restype, hiprtcGetErrorString.argtypes = ctypes.POINTER(ctypes.c_char), [hiprtcResult]
except AttributeError: pass

# hiprtcResult hiprtcVersion(int *major, int *minor)
try: (hiprtcVersion:=dll.hiprtcVersion).restype, hiprtcVersion.argtypes = hiprtcResult, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

class _hiprtcProgram(ctypes.Structure): pass
_hiprtcProgram._fields_ = []
hiprtcProgram = ctypes.POINTER(_hiprtcProgram)
# hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char *name_expression)
try: (hiprtcAddNameExpression:=dll.hiprtcAddNameExpression).restype, hiprtcAddNameExpression.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char **options)
try: (hiprtcCompileProgram:=dll.hiprtcCompileProgram).restype, hiprtcCompileProgram.argtypes = hiprtcResult, [hiprtcProgram, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# hiprtcResult hiprtcCreateProgram(hiprtcProgram *prog, const char *src, const char *name, int numHeaders, const char **headers, const char **includeNames)
try: (hiprtcCreateProgram:=dll.hiprtcCreateProgram).restype, hiprtcCreateProgram.argtypes = hiprtcResult, [ctypes.POINTER(hiprtcProgram), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# hiprtcResult hiprtcDestroyProgram(hiprtcProgram *prog)
try: (hiprtcDestroyProgram:=dll.hiprtcDestroyProgram).restype, hiprtcDestroyProgram.argtypes = hiprtcResult, [ctypes.POINTER(hiprtcProgram)]
except AttributeError: pass

# hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog, const char *name_expression, const char **lowered_name)
try: (hiprtcGetLoweredName:=dll.hiprtcGetLoweredName).restype, hiprtcGetLoweredName.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char *log)
try: (hiprtcGetProgramLog:=dll.hiprtcGetProgramLog).restype, hiprtcGetProgramLog.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t *logSizeRet)
try: (hiprtcGetProgramLogSize:=dll.hiprtcGetProgramLogSize).restype, hiprtcGetProgramLogSize.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# hiprtcResult hiprtcGetCode(hiprtcProgram prog, char *code)
try: (hiprtcGetCode:=dll.hiprtcGetCode).restype, hiprtcGetCode.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t *codeSizeRet)
try: (hiprtcGetCodeSize:=dll.hiprtcGetCodeSize).restype, hiprtcGetCodeSize.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# hiprtcResult hiprtcGetBitcode(hiprtcProgram prog, char *bitcode)
try: (hiprtcGetBitcode:=dll.hiprtcGetBitcode).restype, hiprtcGetBitcode.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog, size_t *bitcode_size)
try: (hiprtcGetBitcodeSize:=dll.hiprtcGetBitcodeSize).restype, hiprtcGetBitcodeSize.argtypes = hiprtcResult, [hiprtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# hiprtcResult hiprtcLinkCreate(unsigned int num_options, hiprtcJIT_option *option_ptr, void **option_vals_pptr, hiprtcLinkState *hip_link_state_ptr)
try: (hiprtcLinkCreate:=dll.hiprtcLinkCreate).restype, hiprtcLinkCreate.argtypes = hiprtcResult, [ctypes.c_uint, ctypes.POINTER(hiprtcJIT_option), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(hiprtcLinkState)]
except AttributeError: pass

# hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state, hiprtcJITInputType input_type, const char *file_path, unsigned int num_options, hiprtcJIT_option *options_ptr, void **option_values)
try: (hiprtcLinkAddFile:=dll.hiprtcLinkAddFile).restype, hiprtcLinkAddFile.argtypes = hiprtcResult, [hiprtcLinkState, hiprtcJITInputType, ctypes.POINTER(ctypes.c_char), ctypes.c_uint, ctypes.POINTER(hiprtcJIT_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state, hiprtcJITInputType input_type, void *image, size_t image_size, const char *name, unsigned int num_options, hiprtcJIT_option *options_ptr, void **option_values)
try: (hiprtcLinkAddData:=dll.hiprtcLinkAddData).restype, hiprtcLinkAddData.argtypes = hiprtcResult, [hiprtcLinkState, hiprtcJITInputType, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint, ctypes.POINTER(hiprtcJIT_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state, void **bin_out, size_t *size_out)
try: (hiprtcLinkComplete:=dll.hiprtcLinkComplete).restype, hiprtcLinkComplete.argtypes = hiprtcResult, [hiprtcLinkState, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t)]
except AttributeError: pass

# hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state)
try: (hiprtcLinkDestroy:=dll.hiprtcLinkDestroy).restype, hiprtcLinkDestroy.argtypes = hiprtcResult, [hiprtcLinkState]
except AttributeError: pass

_anonenum0 = CEnum(ctypes.c_uint)
HIP_SUCCESS = _anonenum0.define('HIP_SUCCESS', 0)
HIP_ERROR_INVALID_VALUE = _anonenum0.define('HIP_ERROR_INVALID_VALUE', 1)
HIP_ERROR_NOT_INITIALIZED = _anonenum0.define('HIP_ERROR_NOT_INITIALIZED', 2)
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = _anonenum0.define('HIP_ERROR_LAUNCH_OUT_OF_RESOURCES', 3)

class hipDeviceArch_t(ctypes.Structure): pass
hipDeviceArch_t._fields_ = [
  ('hasGlobalInt32Atomics', ctypes.c_uint,1),
  ('hasGlobalFloatAtomicExch', ctypes.c_uint,1),
  ('hasSharedInt32Atomics', ctypes.c_uint,1),
  ('hasSharedFloatAtomicExch', ctypes.c_uint,1),
  ('hasFloatAtomicAdd', ctypes.c_uint,1),
  ('hasGlobalInt64Atomics', ctypes.c_uint,1),
  ('hasSharedInt64Atomics', ctypes.c_uint,1),
  ('hasDoubles', ctypes.c_uint,1),
  ('hasWarpVote', ctypes.c_uint,1),
  ('hasWarpBallot', ctypes.c_uint,1),
  ('hasWarpShuffle', ctypes.c_uint,1),
  ('hasFunnelShift', ctypes.c_uint,1),
  ('hasThreadFenceSystem', ctypes.c_uint,1),
  ('hasSyncThreadsExt', ctypes.c_uint,1),
  ('hasSurfaceFuncs', ctypes.c_uint,1),
  ('has3dGrid', ctypes.c_uint,1),
  ('hasDynamicParallelism', ctypes.c_uint,1),
]
class hipUUID_t(ctypes.Structure): pass
hipUUID_t._fields_ = [
  ('bytes', (ctypes.c_char * 16)),
]
hipUUID = hipUUID_t
class hipDeviceProp_tR0600(ctypes.Structure): pass
hipDeviceProp_tR0600._fields_ = [
  ('name', (ctypes.c_char * 256)),
  ('uuid', hipUUID),
  ('luid', (ctypes.c_char * 8)),
  ('luidDeviceNodeMask', ctypes.c_uint),
  ('totalGlobalMem', size_t),
  ('sharedMemPerBlock', size_t),
  ('regsPerBlock', ctypes.c_int),
  ('warpSize', ctypes.c_int),
  ('memPitch', size_t),
  ('maxThreadsPerBlock', ctypes.c_int),
  ('maxThreadsDim', (ctypes.c_int * 3)),
  ('maxGridSize', (ctypes.c_int * 3)),
  ('clockRate', ctypes.c_int),
  ('totalConstMem', size_t),
  ('major', ctypes.c_int),
  ('minor', ctypes.c_int),
  ('textureAlignment', size_t),
  ('texturePitchAlignment', size_t),
  ('deviceOverlap', ctypes.c_int),
  ('multiProcessorCount', ctypes.c_int),
  ('kernelExecTimeoutEnabled', ctypes.c_int),
  ('integrated', ctypes.c_int),
  ('canMapHostMemory', ctypes.c_int),
  ('computeMode', ctypes.c_int),
  ('maxTexture1D', ctypes.c_int),
  ('maxTexture1DMipmap', ctypes.c_int),
  ('maxTexture1DLinear', ctypes.c_int),
  ('maxTexture2D', (ctypes.c_int * 2)),
  ('maxTexture2DMipmap', (ctypes.c_int * 2)),
  ('maxTexture2DLinear', (ctypes.c_int * 3)),
  ('maxTexture2DGather', (ctypes.c_int * 2)),
  ('maxTexture3D', (ctypes.c_int * 3)),
  ('maxTexture3DAlt', (ctypes.c_int * 3)),
  ('maxTextureCubemap', ctypes.c_int),
  ('maxTexture1DLayered', (ctypes.c_int * 2)),
  ('maxTexture2DLayered', (ctypes.c_int * 3)),
  ('maxTextureCubemapLayered', (ctypes.c_int * 2)),
  ('maxSurface1D', ctypes.c_int),
  ('maxSurface2D', (ctypes.c_int * 2)),
  ('maxSurface3D', (ctypes.c_int * 3)),
  ('maxSurface1DLayered', (ctypes.c_int * 2)),
  ('maxSurface2DLayered', (ctypes.c_int * 3)),
  ('maxSurfaceCubemap', ctypes.c_int),
  ('maxSurfaceCubemapLayered', (ctypes.c_int * 2)),
  ('surfaceAlignment', size_t),
  ('concurrentKernels', ctypes.c_int),
  ('ECCEnabled', ctypes.c_int),
  ('pciBusID', ctypes.c_int),
  ('pciDeviceID', ctypes.c_int),
  ('pciDomainID', ctypes.c_int),
  ('tccDriver', ctypes.c_int),
  ('asyncEngineCount', ctypes.c_int),
  ('unifiedAddressing', ctypes.c_int),
  ('memoryClockRate', ctypes.c_int),
  ('memoryBusWidth', ctypes.c_int),
  ('l2CacheSize', ctypes.c_int),
  ('persistingL2CacheMaxSize', ctypes.c_int),
  ('maxThreadsPerMultiProcessor', ctypes.c_int),
  ('streamPrioritiesSupported', ctypes.c_int),
  ('globalL1CacheSupported', ctypes.c_int),
  ('localL1CacheSupported', ctypes.c_int),
  ('sharedMemPerMultiprocessor', size_t),
  ('regsPerMultiprocessor', ctypes.c_int),
  ('managedMemory', ctypes.c_int),
  ('isMultiGpuBoard', ctypes.c_int),
  ('multiGpuBoardGroupID', ctypes.c_int),
  ('hostNativeAtomicSupported', ctypes.c_int),
  ('singleToDoublePrecisionPerfRatio', ctypes.c_int),
  ('pageableMemoryAccess', ctypes.c_int),
  ('concurrentManagedAccess', ctypes.c_int),
  ('computePreemptionSupported', ctypes.c_int),
  ('canUseHostPointerForRegisteredMem', ctypes.c_int),
  ('cooperativeLaunch', ctypes.c_int),
  ('cooperativeMultiDeviceLaunch', ctypes.c_int),
  ('sharedMemPerBlockOptin', size_t),
  ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int),
  ('directManagedMemAccessFromHost', ctypes.c_int),
  ('maxBlocksPerMultiProcessor', ctypes.c_int),
  ('accessPolicyMaxWindowSize', ctypes.c_int),
  ('reservedSharedMemPerBlock', size_t),
  ('hostRegisterSupported', ctypes.c_int),
  ('sparseHipArraySupported', ctypes.c_int),
  ('hostRegisterReadOnlySupported', ctypes.c_int),
  ('timelineSemaphoreInteropSupported', ctypes.c_int),
  ('memoryPoolsSupported', ctypes.c_int),
  ('gpuDirectRDMASupported', ctypes.c_int),
  ('gpuDirectRDMAFlushWritesOptions', ctypes.c_uint),
  ('gpuDirectRDMAWritesOrdering', ctypes.c_int),
  ('memoryPoolSupportedHandleTypes', ctypes.c_uint),
  ('deferredMappingHipArraySupported', ctypes.c_int),
  ('ipcEventSupported', ctypes.c_int),
  ('clusterLaunch', ctypes.c_int),
  ('unifiedFunctionPointers', ctypes.c_int),
  ('reserved', (ctypes.c_int * 63)),
  ('hipReserved', (ctypes.c_int * 32)),
  ('gcnArchName', (ctypes.c_char * 256)),
  ('maxSharedMemoryPerMultiProcessor', size_t),
  ('clockInstructionRate', ctypes.c_int),
  ('arch', hipDeviceArch_t),
  ('hdpMemFlushCntl', ctypes.POINTER(ctypes.c_uint)),
  ('hdpRegFlushCntl', ctypes.POINTER(ctypes.c_uint)),
  ('cooperativeMultiDeviceUnmatchedFunc', ctypes.c_int),
  ('cooperativeMultiDeviceUnmatchedGridDim', ctypes.c_int),
  ('cooperativeMultiDeviceUnmatchedBlockDim', ctypes.c_int),
  ('cooperativeMultiDeviceUnmatchedSharedMem', ctypes.c_int),
  ('isLargeBar', ctypes.c_int),
  ('asicRevision', ctypes.c_int),
]
hipMemoryType = CEnum(ctypes.c_uint)
hipMemoryTypeUnregistered = hipMemoryType.define('hipMemoryTypeUnregistered', 0)
hipMemoryTypeHost = hipMemoryType.define('hipMemoryTypeHost', 1)
hipMemoryTypeDevice = hipMemoryType.define('hipMemoryTypeDevice', 2)
hipMemoryTypeManaged = hipMemoryType.define('hipMemoryTypeManaged', 3)
hipMemoryTypeArray = hipMemoryType.define('hipMemoryTypeArray', 10)
hipMemoryTypeUnified = hipMemoryType.define('hipMemoryTypeUnified', 11)

class hipPointerAttribute_t(ctypes.Structure): pass
hipPointerAttribute_t._fields_ = [
  ('type', hipMemoryType),
  ('device', ctypes.c_int),
  ('devicePointer', ctypes.c_void_p),
  ('hostPointer', ctypes.c_void_p),
  ('isManaged', ctypes.c_int),
  ('allocationFlags', ctypes.c_uint),
]
hipDeviceAttribute_t = CEnum(ctypes.c_uint)
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

hipDriverProcAddressQueryResult = CEnum(ctypes.c_uint)
HIP_GET_PROC_ADDRESS_SUCCESS = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_SUCCESS', 0)
HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', 1)
HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = hipDriverProcAddressQueryResult.define('HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT', 2)

hipComputeMode = CEnum(ctypes.c_uint)
hipComputeModeDefault = hipComputeMode.define('hipComputeModeDefault', 0)
hipComputeModeExclusive = hipComputeMode.define('hipComputeModeExclusive', 1)
hipComputeModeProhibited = hipComputeMode.define('hipComputeModeProhibited', 2)
hipComputeModeExclusiveProcess = hipComputeMode.define('hipComputeModeExclusiveProcess', 3)

hipFlushGPUDirectRDMAWritesOptions = CEnum(ctypes.c_uint)
hipFlushGPUDirectRDMAWritesOptionHost = hipFlushGPUDirectRDMAWritesOptions.define('hipFlushGPUDirectRDMAWritesOptionHost', 1)
hipFlushGPUDirectRDMAWritesOptionMemOps = hipFlushGPUDirectRDMAWritesOptions.define('hipFlushGPUDirectRDMAWritesOptionMemOps', 2)

hipGPUDirectRDMAWritesOrdering = CEnum(ctypes.c_uint)
hipGPUDirectRDMAWritesOrderingNone = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingNone', 0)
hipGPUDirectRDMAWritesOrderingOwner = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingOwner', 100)
hipGPUDirectRDMAWritesOrderingAllDevices = hipGPUDirectRDMAWritesOrdering.define('hipGPUDirectRDMAWritesOrderingAllDevices', 200)

# hipError_t hip_init()
try: (hip_init:=dll.hip_init).restype, hip_init.argtypes = hipError_t, []
except AttributeError: pass

class ihipCtx_t(ctypes.Structure): pass
ihipCtx_t._fields_ = []
hipCtx_t = ctypes.POINTER(ihipCtx_t)
hipDevice_t = ctypes.c_int
hipDeviceP2PAttr = CEnum(ctypes.c_uint)
hipDevP2PAttrPerformanceRank = hipDeviceP2PAttr.define('hipDevP2PAttrPerformanceRank', 0)
hipDevP2PAttrAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrAccessSupported', 1)
hipDevP2PAttrNativeAtomicSupported = hipDeviceP2PAttr.define('hipDevP2PAttrNativeAtomicSupported', 2)
hipDevP2PAttrHipArrayAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrHipArrayAccessSupported', 3)

class hipIpcMemHandle_st(ctypes.Structure): pass
hipIpcMemHandle_st._fields_ = [
  ('reserved', (ctypes.c_char * 64)),
]
hipIpcMemHandle_t = hipIpcMemHandle_st
class hipIpcEventHandle_st(ctypes.Structure): pass
hipIpcEventHandle_st._fields_ = [
  ('reserved', (ctypes.c_char * 64)),
]
hipIpcEventHandle_t = hipIpcEventHandle_st
class ihipModule_t(ctypes.Structure): pass
ihipModule_t._fields_ = []
hipModule_t = ctypes.POINTER(ihipModule_t)
class ihipMemPoolHandle_t(ctypes.Structure): pass
ihipMemPoolHandle_t._fields_ = []
hipMemPool_t = ctypes.POINTER(ihipMemPoolHandle_t)
class hipFuncAttributes(ctypes.Structure): pass
hipFuncAttributes._fields_ = [
  ('binaryVersion', ctypes.c_int),
  ('cacheModeCA', ctypes.c_int),
  ('constSizeBytes', size_t),
  ('localSizeBytes', size_t),
  ('maxDynamicSharedSizeBytes', ctypes.c_int),
  ('maxThreadsPerBlock', ctypes.c_int),
  ('numRegs', ctypes.c_int),
  ('preferredShmemCarveout', ctypes.c_int),
  ('ptxVersion', ctypes.c_int),
  ('sharedSizeBytes', size_t),
]
hipLimit_t = CEnum(ctypes.c_uint)
hipLimitStackSize = hipLimit_t.define('hipLimitStackSize', 0)
hipLimitPrintfFifoSize = hipLimit_t.define('hipLimitPrintfFifoSize', 1)
hipLimitMallocHeapSize = hipLimit_t.define('hipLimitMallocHeapSize', 2)
hipLimitRange = hipLimit_t.define('hipLimitRange', 3)

hipMemoryAdvise = CEnum(ctypes.c_uint)
hipMemAdviseSetReadMostly = hipMemoryAdvise.define('hipMemAdviseSetReadMostly', 1)
hipMemAdviseUnsetReadMostly = hipMemoryAdvise.define('hipMemAdviseUnsetReadMostly', 2)
hipMemAdviseSetPreferredLocation = hipMemoryAdvise.define('hipMemAdviseSetPreferredLocation', 3)
hipMemAdviseUnsetPreferredLocation = hipMemoryAdvise.define('hipMemAdviseUnsetPreferredLocation', 4)
hipMemAdviseSetAccessedBy = hipMemoryAdvise.define('hipMemAdviseSetAccessedBy', 5)
hipMemAdviseUnsetAccessedBy = hipMemoryAdvise.define('hipMemAdviseUnsetAccessedBy', 6)
hipMemAdviseSetCoarseGrain = hipMemoryAdvise.define('hipMemAdviseSetCoarseGrain', 100)
hipMemAdviseUnsetCoarseGrain = hipMemoryAdvise.define('hipMemAdviseUnsetCoarseGrain', 101)

hipMemRangeCoherencyMode = CEnum(ctypes.c_uint)
hipMemRangeCoherencyModeFineGrain = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeFineGrain', 0)
hipMemRangeCoherencyModeCoarseGrain = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeCoarseGrain', 1)
hipMemRangeCoherencyModeIndeterminate = hipMemRangeCoherencyMode.define('hipMemRangeCoherencyModeIndeterminate', 2)

hipMemRangeAttribute = CEnum(ctypes.c_uint)
hipMemRangeAttributeReadMostly = hipMemRangeAttribute.define('hipMemRangeAttributeReadMostly', 1)
hipMemRangeAttributePreferredLocation = hipMemRangeAttribute.define('hipMemRangeAttributePreferredLocation', 2)
hipMemRangeAttributeAccessedBy = hipMemRangeAttribute.define('hipMemRangeAttributeAccessedBy', 3)
hipMemRangeAttributeLastPrefetchLocation = hipMemRangeAttribute.define('hipMemRangeAttributeLastPrefetchLocation', 4)
hipMemRangeAttributeCoherencyMode = hipMemRangeAttribute.define('hipMemRangeAttributeCoherencyMode', 100)

hipMemPoolAttr = CEnum(ctypes.c_uint)
hipMemPoolReuseFollowEventDependencies = hipMemPoolAttr.define('hipMemPoolReuseFollowEventDependencies', 1)
hipMemPoolReuseAllowOpportunistic = hipMemPoolAttr.define('hipMemPoolReuseAllowOpportunistic', 2)
hipMemPoolReuseAllowInternalDependencies = hipMemPoolAttr.define('hipMemPoolReuseAllowInternalDependencies', 3)
hipMemPoolAttrReleaseThreshold = hipMemPoolAttr.define('hipMemPoolAttrReleaseThreshold', 4)
hipMemPoolAttrReservedMemCurrent = hipMemPoolAttr.define('hipMemPoolAttrReservedMemCurrent', 5)
hipMemPoolAttrReservedMemHigh = hipMemPoolAttr.define('hipMemPoolAttrReservedMemHigh', 6)
hipMemPoolAttrUsedMemCurrent = hipMemPoolAttr.define('hipMemPoolAttrUsedMemCurrent', 7)
hipMemPoolAttrUsedMemHigh = hipMemPoolAttr.define('hipMemPoolAttrUsedMemHigh', 8)

hipMemLocationType = CEnum(ctypes.c_uint)
hipMemLocationTypeInvalid = hipMemLocationType.define('hipMemLocationTypeInvalid', 0)
hipMemLocationTypeDevice = hipMemLocationType.define('hipMemLocationTypeDevice', 1)

class hipMemLocation(ctypes.Structure): pass
hipMemLocation._fields_ = [
  ('type', hipMemLocationType),
  ('id', ctypes.c_int),
]
hipMemAccessFlags = CEnum(ctypes.c_uint)
hipMemAccessFlagsProtNone = hipMemAccessFlags.define('hipMemAccessFlagsProtNone', 0)
hipMemAccessFlagsProtRead = hipMemAccessFlags.define('hipMemAccessFlagsProtRead', 1)
hipMemAccessFlagsProtReadWrite = hipMemAccessFlags.define('hipMemAccessFlagsProtReadWrite', 3)

class hipMemAccessDesc(ctypes.Structure): pass
hipMemAccessDesc._fields_ = [
  ('location', hipMemLocation),
  ('flags', hipMemAccessFlags),
]
hipMemAllocationType = CEnum(ctypes.c_uint)
hipMemAllocationTypeInvalid = hipMemAllocationType.define('hipMemAllocationTypeInvalid', 0)
hipMemAllocationTypePinned = hipMemAllocationType.define('hipMemAllocationTypePinned', 1)
hipMemAllocationTypeMax = hipMemAllocationType.define('hipMemAllocationTypeMax', 2147483647)

hipMemAllocationHandleType = CEnum(ctypes.c_uint)
hipMemHandleTypeNone = hipMemAllocationHandleType.define('hipMemHandleTypeNone', 0)
hipMemHandleTypePosixFileDescriptor = hipMemAllocationHandleType.define('hipMemHandleTypePosixFileDescriptor', 1)
hipMemHandleTypeWin32 = hipMemAllocationHandleType.define('hipMemHandleTypeWin32', 2)
hipMemHandleTypeWin32Kmt = hipMemAllocationHandleType.define('hipMemHandleTypeWin32Kmt', 4)

class hipMemPoolProps(ctypes.Structure): pass
hipMemPoolProps._fields_ = [
  ('allocType', hipMemAllocationType),
  ('handleTypes', hipMemAllocationHandleType),
  ('location', hipMemLocation),
  ('win32SecurityAttributes', ctypes.c_void_p),
  ('maxSize', size_t),
  ('reserved', (ctypes.c_ubyte * 56)),
]
class hipMemPoolPtrExportData(ctypes.Structure): pass
hipMemPoolPtrExportData._fields_ = [
  ('reserved', (ctypes.c_ubyte * 64)),
]
hipJitOption = CEnum(ctypes.c_uint)
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

hipFuncAttribute = CEnum(ctypes.c_uint)
hipFuncAttributeMaxDynamicSharedMemorySize = hipFuncAttribute.define('hipFuncAttributeMaxDynamicSharedMemorySize', 8)
hipFuncAttributePreferredSharedMemoryCarveout = hipFuncAttribute.define('hipFuncAttributePreferredSharedMemoryCarveout', 9)
hipFuncAttributeMax = hipFuncAttribute.define('hipFuncAttributeMax', 10)

hipFuncCache_t = CEnum(ctypes.c_uint)
hipFuncCachePreferNone = hipFuncCache_t.define('hipFuncCachePreferNone', 0)
hipFuncCachePreferShared = hipFuncCache_t.define('hipFuncCachePreferShared', 1)
hipFuncCachePreferL1 = hipFuncCache_t.define('hipFuncCachePreferL1', 2)
hipFuncCachePreferEqual = hipFuncCache_t.define('hipFuncCachePreferEqual', 3)

hipSharedMemConfig = CEnum(ctypes.c_uint)
hipSharedMemBankSizeDefault = hipSharedMemConfig.define('hipSharedMemBankSizeDefault', 0)
hipSharedMemBankSizeFourByte = hipSharedMemConfig.define('hipSharedMemBankSizeFourByte', 1)
hipSharedMemBankSizeEightByte = hipSharedMemConfig.define('hipSharedMemBankSizeEightByte', 2)

class hipLaunchParams_t(ctypes.Structure): pass
hipLaunchParams_t._fields_ = [
  ('func', ctypes.c_void_p),
  ('gridDim', dim3),
  ('blockDim', dim3),
  ('args', ctypes.POINTER(ctypes.c_void_p)),
  ('sharedMem', size_t),
  ('stream', hipStream_t),
]
hipLaunchParams = hipLaunchParams_t
class hipFunctionLaunchParams_t(ctypes.Structure): pass
hipFunctionLaunchParams_t._fields_ = [
  ('function', hipFunction_t),
  ('gridDimX', ctypes.c_uint),
  ('gridDimY', ctypes.c_uint),
  ('gridDimZ', ctypes.c_uint),
  ('blockDimX', ctypes.c_uint),
  ('blockDimY', ctypes.c_uint),
  ('blockDimZ', ctypes.c_uint),
  ('sharedMemBytes', ctypes.c_uint),
  ('hStream', hipStream_t),
  ('kernelParams', ctypes.POINTER(ctypes.c_void_p)),
]
hipFunctionLaunchParams = hipFunctionLaunchParams_t
hipExternalMemoryHandleType_enum = CEnum(ctypes.c_uint)
hipExternalMemoryHandleTypeOpaqueFd = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueFd', 1)
hipExternalMemoryHandleTypeOpaqueWin32 = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueWin32', 2)
hipExternalMemoryHandleTypeOpaqueWin32Kmt = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeOpaqueWin32Kmt', 3)
hipExternalMemoryHandleTypeD3D12Heap = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D12Heap', 4)
hipExternalMemoryHandleTypeD3D12Resource = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D12Resource', 5)
hipExternalMemoryHandleTypeD3D11Resource = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D11Resource', 6)
hipExternalMemoryHandleTypeD3D11ResourceKmt = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeD3D11ResourceKmt', 7)
hipExternalMemoryHandleTypeNvSciBuf = hipExternalMemoryHandleType_enum.define('hipExternalMemoryHandleTypeNvSciBuf', 8)

hipExternalMemoryHandleType = hipExternalMemoryHandleType_enum
class hipExternalMemoryHandleDesc_st(ctypes.Structure): pass
class hipExternalMemoryHandleDesc_st_handle(ctypes.Union): pass
class hipExternalMemoryHandleDesc_st_handle_win32(ctypes.Structure): pass
hipExternalMemoryHandleDesc_st_handle_win32._fields_ = [
  ('handle', ctypes.c_void_p),
  ('name', ctypes.c_void_p),
]
hipExternalMemoryHandleDesc_st_handle._fields_ = [
  ('fd', ctypes.c_int),
  ('win32', hipExternalMemoryHandleDesc_st_handle_win32),
  ('nvSciBufObject', ctypes.c_void_p),
]
hipExternalMemoryHandleDesc_st._fields_ = [
  ('type', hipExternalMemoryHandleType),
  ('handle', hipExternalMemoryHandleDesc_st_handle),
  ('size', ctypes.c_ulonglong),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
hipExternalMemoryHandleDesc = hipExternalMemoryHandleDesc_st
class hipExternalMemoryBufferDesc_st(ctypes.Structure): pass
hipExternalMemoryBufferDesc_st._fields_ = [
  ('offset', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
hipExternalMemoryBufferDesc = hipExternalMemoryBufferDesc_st
class hipExternalMemoryMipmappedArrayDesc_st(ctypes.Structure): pass
class hipChannelFormatDesc(ctypes.Structure): pass
hipChannelFormatKind = CEnum(ctypes.c_uint)
hipChannelFormatKindSigned = hipChannelFormatKind.define('hipChannelFormatKindSigned', 0)
hipChannelFormatKindUnsigned = hipChannelFormatKind.define('hipChannelFormatKindUnsigned', 1)
hipChannelFormatKindFloat = hipChannelFormatKind.define('hipChannelFormatKindFloat', 2)
hipChannelFormatKindNone = hipChannelFormatKind.define('hipChannelFormatKindNone', 3)

hipChannelFormatDesc._fields_ = [
  ('x', ctypes.c_int),
  ('y', ctypes.c_int),
  ('z', ctypes.c_int),
  ('w', ctypes.c_int),
  ('f', hipChannelFormatKind),
]
class hipExtent(ctypes.Structure): pass
hipExtent._fields_ = [
  ('width', size_t),
  ('height', size_t),
  ('depth', size_t),
]
hipExternalMemoryMipmappedArrayDesc_st._fields_ = [
  ('offset', ctypes.c_ulonglong),
  ('formatDesc', hipChannelFormatDesc),
  ('extent', hipExtent),
  ('flags', ctypes.c_uint),
  ('numLevels', ctypes.c_uint),
]
hipExternalMemoryMipmappedArrayDesc = hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t = ctypes.c_void_p
hipExternalSemaphoreHandleType_enum = CEnum(ctypes.c_uint)
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
class hipExternalSemaphoreHandleDesc_st(ctypes.Structure): pass
class hipExternalSemaphoreHandleDesc_st_handle(ctypes.Union): pass
class hipExternalSemaphoreHandleDesc_st_handle_win32(ctypes.Structure): pass
hipExternalSemaphoreHandleDesc_st_handle_win32._fields_ = [
  ('handle', ctypes.c_void_p),
  ('name', ctypes.c_void_p),
]
hipExternalSemaphoreHandleDesc_st_handle._fields_ = [
  ('fd', ctypes.c_int),
  ('win32', hipExternalSemaphoreHandleDesc_st_handle_win32),
  ('NvSciSyncObj', ctypes.c_void_p),
]
hipExternalSemaphoreHandleDesc_st._fields_ = [
  ('type', hipExternalSemaphoreHandleType),
  ('handle', hipExternalSemaphoreHandleDesc_st_handle),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
hipExternalSemaphoreHandleDesc = hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t = ctypes.c_void_p
class hipExternalSemaphoreSignalParams_st(ctypes.Structure): pass
class hipExternalSemaphoreSignalParams_st_params(ctypes.Structure): pass
class hipExternalSemaphoreSignalParams_st_params_fence(ctypes.Structure): pass
hipExternalSemaphoreSignalParams_st_params_fence._fields_ = [
  ('value', ctypes.c_ulonglong),
]
class hipExternalSemaphoreSignalParams_st_params_nvSciSync(ctypes.Union): pass
hipExternalSemaphoreSignalParams_st_params_nvSciSync._fields_ = [
  ('fence', ctypes.c_void_p),
  ('reserved', ctypes.c_ulonglong),
]
class hipExternalSemaphoreSignalParams_st_params_keyedMutex(ctypes.Structure): pass
hipExternalSemaphoreSignalParams_st_params_keyedMutex._fields_ = [
  ('key', ctypes.c_ulonglong),
]
hipExternalSemaphoreSignalParams_st_params._fields_ = [
  ('fence', hipExternalSemaphoreSignalParams_st_params_fence),
  ('nvSciSync', hipExternalSemaphoreSignalParams_st_params_nvSciSync),
  ('keyedMutex', hipExternalSemaphoreSignalParams_st_params_keyedMutex),
  ('reserved', (ctypes.c_uint * 12)),
]
hipExternalSemaphoreSignalParams_st._fields_ = [
  ('params', hipExternalSemaphoreSignalParams_st_params),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
hipExternalSemaphoreSignalParams = hipExternalSemaphoreSignalParams_st
class hipExternalSemaphoreWaitParams_st(ctypes.Structure): pass
class hipExternalSemaphoreWaitParams_st_params(ctypes.Structure): pass
class hipExternalSemaphoreWaitParams_st_params_fence(ctypes.Structure): pass
hipExternalSemaphoreWaitParams_st_params_fence._fields_ = [
  ('value', ctypes.c_ulonglong),
]
class hipExternalSemaphoreWaitParams_st_params_nvSciSync(ctypes.Union): pass
hipExternalSemaphoreWaitParams_st_params_nvSciSync._fields_ = [
  ('fence', ctypes.c_void_p),
  ('reserved', ctypes.c_ulonglong),
]
class hipExternalSemaphoreWaitParams_st_params_keyedMutex(ctypes.Structure): pass
hipExternalSemaphoreWaitParams_st_params_keyedMutex._fields_ = [
  ('key', ctypes.c_ulonglong),
  ('timeoutMs', ctypes.c_uint),
]
hipExternalSemaphoreWaitParams_st_params._fields_ = [
  ('fence', hipExternalSemaphoreWaitParams_st_params_fence),
  ('nvSciSync', hipExternalSemaphoreWaitParams_st_params_nvSciSync),
  ('keyedMutex', hipExternalSemaphoreWaitParams_st_params_keyedMutex),
  ('reserved', (ctypes.c_uint * 10)),
]
hipExternalSemaphoreWaitParams_st._fields_ = [
  ('params', hipExternalSemaphoreWaitParams_st_params),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
hipExternalSemaphoreWaitParams = hipExternalSemaphoreWaitParams_st
# void __hipGetPCH(const char **pch, unsigned int *size)
try: (__hipGetPCH:=dll.__hipGetPCH).restype, __hipGetPCH.argtypes = None, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

hipGraphicsRegisterFlags = CEnum(ctypes.c_uint)
hipGraphicsRegisterFlagsNone = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsNone', 0)
hipGraphicsRegisterFlagsReadOnly = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsReadOnly', 1)
hipGraphicsRegisterFlagsWriteDiscard = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsWriteDiscard', 2)
hipGraphicsRegisterFlagsSurfaceLoadStore = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsSurfaceLoadStore', 4)
hipGraphicsRegisterFlagsTextureGather = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsTextureGather', 8)

class _hipGraphicsResource(ctypes.Structure): pass
_hipGraphicsResource._fields_ = []
hipGraphicsResource = _hipGraphicsResource
hipGraphicsResource_t = ctypes.POINTER(_hipGraphicsResource)
class ihipGraph(ctypes.Structure): pass
ihipGraph._fields_ = []
hipGraph_t = ctypes.POINTER(ihipGraph)
class hipGraphNode(ctypes.Structure): pass
hipGraphNode._fields_ = []
hipGraphNode_t = ctypes.POINTER(hipGraphNode)
class hipGraphExec(ctypes.Structure): pass
hipGraphExec._fields_ = []
hipGraphExec_t = ctypes.POINTER(hipGraphExec)
class hipUserObject(ctypes.Structure): pass
hipUserObject._fields_ = []
hipUserObject_t = ctypes.POINTER(hipUserObject)
hipGraphNodeType = CEnum(ctypes.c_uint)
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
class hipHostNodeParams(ctypes.Structure): pass
hipHostNodeParams._fields_ = [
  ('fn', hipHostFn_t),
  ('userData', ctypes.c_void_p),
]
class hipKernelNodeParams(ctypes.Structure): pass
hipKernelNodeParams._fields_ = [
  ('blockDim', dim3),
  ('extra', ctypes.POINTER(ctypes.c_void_p)),
  ('func', ctypes.c_void_p),
  ('gridDim', dim3),
  ('kernelParams', ctypes.POINTER(ctypes.c_void_p)),
  ('sharedMemBytes', ctypes.c_uint),
]
class hipMemsetParams(ctypes.Structure): pass
hipMemsetParams._fields_ = [
  ('dst', ctypes.c_void_p),
  ('elementSize', ctypes.c_uint),
  ('height', size_t),
  ('pitch', size_t),
  ('value', ctypes.c_uint),
  ('width', size_t),
]
class hipMemAllocNodeParams(ctypes.Structure): pass
hipMemAllocNodeParams._fields_ = [
  ('poolProps', hipMemPoolProps),
  ('accessDescs', ctypes.POINTER(hipMemAccessDesc)),
  ('accessDescCount', size_t),
  ('bytesize', size_t),
  ('dptr', ctypes.c_void_p),
]
hipAccessProperty = CEnum(ctypes.c_uint)
hipAccessPropertyNormal = hipAccessProperty.define('hipAccessPropertyNormal', 0)
hipAccessPropertyStreaming = hipAccessProperty.define('hipAccessPropertyStreaming', 1)
hipAccessPropertyPersisting = hipAccessProperty.define('hipAccessPropertyPersisting', 2)

class hipAccessPolicyWindow(ctypes.Structure): pass
hipAccessPolicyWindow._fields_ = [
  ('base_ptr', ctypes.c_void_p),
  ('hitProp', hipAccessProperty),
  ('hitRatio', ctypes.c_float),
  ('missProp', hipAccessProperty),
  ('num_bytes', size_t),
]
hipLaunchAttributeID = CEnum(ctypes.c_uint)
hipLaunchAttributeAccessPolicyWindow = hipLaunchAttributeID.define('hipLaunchAttributeAccessPolicyWindow', 1)
hipLaunchAttributeCooperative = hipLaunchAttributeID.define('hipLaunchAttributeCooperative', 2)
hipLaunchAttributePriority = hipLaunchAttributeID.define('hipLaunchAttributePriority', 8)

class hipLaunchAttributeValue(ctypes.Union): pass
hipLaunchAttributeValue._fields_ = [
  ('accessPolicyWindow', hipAccessPolicyWindow),
  ('cooperative', ctypes.c_int),
  ('priority', ctypes.c_int),
]
class HIP_MEMSET_NODE_PARAMS(ctypes.Structure): pass
hipDeviceptr_t = ctypes.c_void_p
HIP_MEMSET_NODE_PARAMS._fields_ = [
  ('dst', hipDeviceptr_t),
  ('pitch', size_t),
  ('value', ctypes.c_uint),
  ('elementSize', ctypes.c_uint),
  ('width', size_t),
  ('height', size_t),
]
hipGraphExecUpdateResult = CEnum(ctypes.c_uint)
hipGraphExecUpdateSuccess = hipGraphExecUpdateResult.define('hipGraphExecUpdateSuccess', 0)
hipGraphExecUpdateError = hipGraphExecUpdateResult.define('hipGraphExecUpdateError', 1)
hipGraphExecUpdateErrorTopologyChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorTopologyChanged', 2)
hipGraphExecUpdateErrorNodeTypeChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorNodeTypeChanged', 3)
hipGraphExecUpdateErrorFunctionChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorFunctionChanged', 4)
hipGraphExecUpdateErrorParametersChanged = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorParametersChanged', 5)
hipGraphExecUpdateErrorNotSupported = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorNotSupported', 6)
hipGraphExecUpdateErrorUnsupportedFunctionChange = hipGraphExecUpdateResult.define('hipGraphExecUpdateErrorUnsupportedFunctionChange', 7)

hipStreamCaptureMode = CEnum(ctypes.c_uint)
hipStreamCaptureModeGlobal = hipStreamCaptureMode.define('hipStreamCaptureModeGlobal', 0)
hipStreamCaptureModeThreadLocal = hipStreamCaptureMode.define('hipStreamCaptureModeThreadLocal', 1)
hipStreamCaptureModeRelaxed = hipStreamCaptureMode.define('hipStreamCaptureModeRelaxed', 2)

hipStreamCaptureStatus = CEnum(ctypes.c_uint)
hipStreamCaptureStatusNone = hipStreamCaptureStatus.define('hipStreamCaptureStatusNone', 0)
hipStreamCaptureStatusActive = hipStreamCaptureStatus.define('hipStreamCaptureStatusActive', 1)
hipStreamCaptureStatusInvalidated = hipStreamCaptureStatus.define('hipStreamCaptureStatusInvalidated', 2)

hipStreamUpdateCaptureDependenciesFlags = CEnum(ctypes.c_uint)
hipStreamAddCaptureDependencies = hipStreamUpdateCaptureDependenciesFlags.define('hipStreamAddCaptureDependencies', 0)
hipStreamSetCaptureDependencies = hipStreamUpdateCaptureDependenciesFlags.define('hipStreamSetCaptureDependencies', 1)

hipGraphMemAttributeType = CEnum(ctypes.c_uint)
hipGraphMemAttrUsedMemCurrent = hipGraphMemAttributeType.define('hipGraphMemAttrUsedMemCurrent', 0)
hipGraphMemAttrUsedMemHigh = hipGraphMemAttributeType.define('hipGraphMemAttrUsedMemHigh', 1)
hipGraphMemAttrReservedMemCurrent = hipGraphMemAttributeType.define('hipGraphMemAttrReservedMemCurrent', 2)
hipGraphMemAttrReservedMemHigh = hipGraphMemAttributeType.define('hipGraphMemAttrReservedMemHigh', 3)

hipUserObjectFlags = CEnum(ctypes.c_uint)
hipUserObjectNoDestructorSync = hipUserObjectFlags.define('hipUserObjectNoDestructorSync', 1)

hipUserObjectRetainFlags = CEnum(ctypes.c_uint)
hipGraphUserObjectMove = hipUserObjectRetainFlags.define('hipGraphUserObjectMove', 1)

hipGraphInstantiateFlags = CEnum(ctypes.c_uint)
hipGraphInstantiateFlagAutoFreeOnLaunch = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagAutoFreeOnLaunch', 1)
hipGraphInstantiateFlagUpload = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagUpload', 2)
hipGraphInstantiateFlagDeviceLaunch = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagDeviceLaunch', 4)
hipGraphInstantiateFlagUseNodePriority = hipGraphInstantiateFlags.define('hipGraphInstantiateFlagUseNodePriority', 8)

hipGraphDebugDotFlags = CEnum(ctypes.c_uint)
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

hipGraphInstantiateResult = CEnum(ctypes.c_uint)
hipGraphInstantiateSuccess = hipGraphInstantiateResult.define('hipGraphInstantiateSuccess', 0)
hipGraphInstantiateError = hipGraphInstantiateResult.define('hipGraphInstantiateError', 1)
hipGraphInstantiateInvalidStructure = hipGraphInstantiateResult.define('hipGraphInstantiateInvalidStructure', 2)
hipGraphInstantiateNodeOperationNotSupported = hipGraphInstantiateResult.define('hipGraphInstantiateNodeOperationNotSupported', 3)
hipGraphInstantiateMultipleDevicesNotSupported = hipGraphInstantiateResult.define('hipGraphInstantiateMultipleDevicesNotSupported', 4)

class hipGraphInstantiateParams(ctypes.Structure): pass
hipGraphInstantiateParams._fields_ = [
  ('errNode_out', hipGraphNode_t),
  ('flags', ctypes.c_ulonglong),
  ('result_out', hipGraphInstantiateResult),
  ('uploadStream', hipStream_t),
]
class hipMemAllocationProp(ctypes.Structure): pass
class hipMemAllocationProp_allocFlags(ctypes.Structure): pass
hipMemAllocationProp_allocFlags._fields_ = [
  ('compressionType', ctypes.c_ubyte),
  ('gpuDirectRDMACapable', ctypes.c_ubyte),
  ('usage', ctypes.c_ushort),
]
hipMemAllocationProp._fields_ = [
  ('type', hipMemAllocationType),
  ('requestedHandleType', hipMemAllocationHandleType),
  ('location', hipMemLocation),
  ('win32HandleMetaData', ctypes.c_void_p),
  ('allocFlags', hipMemAllocationProp_allocFlags),
]
class hipExternalSemaphoreSignalNodeParams(ctypes.Structure): pass
hipExternalSemaphoreSignalNodeParams._fields_ = [
  ('extSemArray', ctypes.POINTER(hipExternalSemaphore_t)),
  ('paramsArray', ctypes.POINTER(hipExternalSemaphoreSignalParams)),
  ('numExtSems', ctypes.c_uint),
]
class hipExternalSemaphoreWaitNodeParams(ctypes.Structure): pass
hipExternalSemaphoreWaitNodeParams._fields_ = [
  ('extSemArray', ctypes.POINTER(hipExternalSemaphore_t)),
  ('paramsArray', ctypes.POINTER(hipExternalSemaphoreWaitParams)),
  ('numExtSems', ctypes.c_uint),
]
class ihipMemGenericAllocationHandle(ctypes.Structure): pass
ihipMemGenericAllocationHandle._fields_ = []
hipMemGenericAllocationHandle_t = ctypes.POINTER(ihipMemGenericAllocationHandle)
hipMemAllocationGranularity_flags = CEnum(ctypes.c_uint)
hipMemAllocationGranularityMinimum = hipMemAllocationGranularity_flags.define('hipMemAllocationGranularityMinimum', 0)
hipMemAllocationGranularityRecommended = hipMemAllocationGranularity_flags.define('hipMemAllocationGranularityRecommended', 1)

hipMemHandleType = CEnum(ctypes.c_uint)
hipMemHandleTypeGeneric = hipMemHandleType.define('hipMemHandleTypeGeneric', 0)

hipMemOperationType = CEnum(ctypes.c_uint)
hipMemOperationTypeMap = hipMemOperationType.define('hipMemOperationTypeMap', 1)
hipMemOperationTypeUnmap = hipMemOperationType.define('hipMemOperationTypeUnmap', 2)

hipArraySparseSubresourceType = CEnum(ctypes.c_uint)
hipArraySparseSubresourceTypeSparseLevel = hipArraySparseSubresourceType.define('hipArraySparseSubresourceTypeSparseLevel', 0)
hipArraySparseSubresourceTypeMiptail = hipArraySparseSubresourceType.define('hipArraySparseSubresourceTypeMiptail', 1)

class hipArrayMapInfo(ctypes.Structure): pass
hipResourceType = CEnum(ctypes.c_uint)
hipResourceTypeArray = hipResourceType.define('hipResourceTypeArray', 0)
hipResourceTypeMipmappedArray = hipResourceType.define('hipResourceTypeMipmappedArray', 1)
hipResourceTypeLinear = hipResourceType.define('hipResourceTypeLinear', 2)
hipResourceTypePitch2D = hipResourceType.define('hipResourceTypePitch2D', 3)

class hipArrayMapInfo_resource(ctypes.Union): pass
class hipMipmappedArray(ctypes.Structure): pass
hipArray_Format = CEnum(ctypes.c_uint)
HIP_AD_FORMAT_UNSIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT8', 1)
HIP_AD_FORMAT_UNSIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT16', 2)
HIP_AD_FORMAT_UNSIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT32', 3)
HIP_AD_FORMAT_SIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT8', 8)
HIP_AD_FORMAT_SIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT16', 9)
HIP_AD_FORMAT_SIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT32', 10)
HIP_AD_FORMAT_HALF = hipArray_Format.define('HIP_AD_FORMAT_HALF', 16)
HIP_AD_FORMAT_FLOAT = hipArray_Format.define('HIP_AD_FORMAT_FLOAT', 32)

hipMipmappedArray._fields_ = [
  ('data', ctypes.c_void_p),
  ('desc', hipChannelFormatDesc),
  ('type', ctypes.c_uint),
  ('width', ctypes.c_uint),
  ('height', ctypes.c_uint),
  ('depth', ctypes.c_uint),
  ('min_mipmap_level', ctypes.c_uint),
  ('max_mipmap_level', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('format', hipArray_Format),
  ('num_channels', ctypes.c_uint),
]
class hipArray(ctypes.Structure): pass
hipArray._fields_ = []
hipArray_t = ctypes.POINTER(hipArray)
hipArrayMapInfo_resource._fields_ = [
  ('mipmap', hipMipmappedArray),
  ('array', hipArray_t),
]
class hipArrayMapInfo_subresource(ctypes.Union): pass
class hipArrayMapInfo_subresource_sparseLevel(ctypes.Structure): pass
hipArrayMapInfo_subresource_sparseLevel._fields_ = [
  ('level', ctypes.c_uint),
  ('layer', ctypes.c_uint),
  ('offsetX', ctypes.c_uint),
  ('offsetY', ctypes.c_uint),
  ('offsetZ', ctypes.c_uint),
  ('extentWidth', ctypes.c_uint),
  ('extentHeight', ctypes.c_uint),
  ('extentDepth', ctypes.c_uint),
]
class hipArrayMapInfo_subresource_miptail(ctypes.Structure): pass
hipArrayMapInfo_subresource_miptail._fields_ = [
  ('layer', ctypes.c_uint),
  ('offset', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
]
hipArrayMapInfo_subresource._fields_ = [
  ('sparseLevel', hipArrayMapInfo_subresource_sparseLevel),
  ('miptail', hipArrayMapInfo_subresource_miptail),
]
class hipArrayMapInfo_memHandle(ctypes.Union): pass
hipArrayMapInfo_memHandle._fields_ = [
  ('memHandle', hipMemGenericAllocationHandle_t),
]
hipArrayMapInfo._fields_ = [
  ('resourceType', hipResourceType),
  ('resource', hipArrayMapInfo_resource),
  ('subresourceType', hipArraySparseSubresourceType),
  ('subresource', hipArrayMapInfo_subresource),
  ('memOperationType', hipMemOperationType),
  ('memHandleType', hipMemHandleType),
  ('memHandle', hipArrayMapInfo_memHandle),
  ('offset', ctypes.c_ulonglong),
  ('deviceBitMask', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 2)),
]
class hipMemcpyNodeParams(ctypes.Structure): pass
class hipMemcpy3DParms(ctypes.Structure): pass
class hipPos(ctypes.Structure): pass
hipPos._fields_ = [
  ('x', size_t),
  ('y', size_t),
  ('z', size_t),
]
class hipPitchedPtr(ctypes.Structure): pass
hipPitchedPtr._fields_ = [
  ('ptr', ctypes.c_void_p),
  ('pitch', size_t),
  ('xsize', size_t),
  ('ysize', size_t),
]
hipMemcpyKind = CEnum(ctypes.c_uint)
hipMemcpyHostToHost = hipMemcpyKind.define('hipMemcpyHostToHost', 0)
hipMemcpyHostToDevice = hipMemcpyKind.define('hipMemcpyHostToDevice', 1)
hipMemcpyDeviceToHost = hipMemcpyKind.define('hipMemcpyDeviceToHost', 2)
hipMemcpyDeviceToDevice = hipMemcpyKind.define('hipMemcpyDeviceToDevice', 3)
hipMemcpyDefault = hipMemcpyKind.define('hipMemcpyDefault', 4)
hipMemcpyDeviceToDeviceNoCU = hipMemcpyKind.define('hipMemcpyDeviceToDeviceNoCU', 1024)

hipMemcpy3DParms._fields_ = [
  ('srcArray', hipArray_t),
  ('srcPos', hipPos),
  ('srcPtr', hipPitchedPtr),
  ('dstArray', hipArray_t),
  ('dstPos', hipPos),
  ('dstPtr', hipPitchedPtr),
  ('extent', hipExtent),
  ('kind', hipMemcpyKind),
]
hipMemcpyNodeParams._fields_ = [
  ('flags', ctypes.c_int),
  ('reserved', (ctypes.c_int * 3)),
  ('copyParams', hipMemcpy3DParms),
]
class hipChildGraphNodeParams(ctypes.Structure): pass
hipChildGraphNodeParams._fields_ = [
  ('graph', hipGraph_t),
]
class hipEventWaitNodeParams(ctypes.Structure): pass
hipEventWaitNodeParams._fields_ = [
  ('event', hipEvent_t),
]
class hipEventRecordNodeParams(ctypes.Structure): pass
hipEventRecordNodeParams._fields_ = [
  ('event', hipEvent_t),
]
class hipMemFreeNodeParams(ctypes.Structure): pass
hipMemFreeNodeParams._fields_ = [
  ('dptr', ctypes.c_void_p),
]
class hipGraphNodeParams(ctypes.Structure): pass
class _anonunion1(ctypes.Union): pass
_anonunion1._fields_ = [
  ('reserved1', (ctypes.c_longlong * 29)),
  ('kernel', hipKernelNodeParams),
  ('memcpy', hipMemcpyNodeParams),
  ('memset', hipMemsetParams),
  ('host', hipHostNodeParams),
  ('graph', hipChildGraphNodeParams),
  ('eventWait', hipEventWaitNodeParams),
  ('eventRecord', hipEventRecordNodeParams),
  ('extSemSignal', hipExternalSemaphoreSignalNodeParams),
  ('extSemWait', hipExternalSemaphoreWaitNodeParams),
  ('alloc', hipMemAllocNodeParams),
  ('free', hipMemFreeNodeParams),
]
hipGraphNodeParams._anonymous_ = ['_0']
hipGraphNodeParams._fields_ = [
  ('type', hipGraphNodeType),
  ('reserved0', (ctypes.c_int * 3)),
  ('_0', _anonunion1),
  ('reserved2', ctypes.c_longlong),
]
hipGraphDependencyType = CEnum(ctypes.c_uint)
hipGraphDependencyTypeDefault = hipGraphDependencyType.define('hipGraphDependencyTypeDefault', 0)
hipGraphDependencyTypeProgrammatic = hipGraphDependencyType.define('hipGraphDependencyTypeProgrammatic', 1)

class hipGraphEdgeData(ctypes.Structure): pass
hipGraphEdgeData._fields_ = [
  ('from_port', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 5)),
  ('to_port', ctypes.c_ubyte),
  ('type', ctypes.c_ubyte),
]
# hipError_t hipInit(unsigned int flags)
try: (hipInit:=dll.hipInit).restype, hipInit.argtypes = hipError_t, [ctypes.c_uint]
except AttributeError: pass

# hipError_t hipDriverGetVersion(int *driverVersion)
try: (hipDriverGetVersion:=dll.hipDriverGetVersion).restype, hipDriverGetVersion.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# hipError_t hipRuntimeGetVersion(int *runtimeVersion)
try: (hipRuntimeGetVersion:=dll.hipRuntimeGetVersion).restype, hipRuntimeGetVersion.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# hipError_t hipDeviceGet(hipDevice_t *device, int ordinal)
try: (hipDeviceGet:=dll.hipDeviceGet).restype, hipDeviceGet.argtypes = hipError_t, [ctypes.POINTER(hipDevice_t), ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device)
try: (hipDeviceComputeCapability:=dll.hipDeviceComputeCapability).restype, hipDeviceComputeCapability.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), hipDevice_t]
except AttributeError: pass

# hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device)
try: (hipDeviceGetName:=dll.hipDeviceGetName).restype, hipDeviceGetName.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, hipDevice_t]
except AttributeError: pass

# hipError_t hipDeviceGetUuid(hipUUID *uuid, hipDevice_t device)
try: (hipDeviceGetUuid:=dll.hipDeviceGetUuid).restype, hipDeviceGetUuid.argtypes = hipError_t, [ctypes.POINTER(hipUUID), hipDevice_t]
except AttributeError: pass

# hipError_t hipDeviceGetP2PAttribute(int *value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice)
try: (hipDeviceGetP2PAttribute:=dll.hipDeviceGetP2PAttribute).restype, hipDeviceGetP2PAttribute.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), hipDeviceP2PAttr, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int device)
try: (hipDeviceGetPCIBusId:=dll.hipDeviceGetPCIBusId).restype, hipDeviceGetPCIBusId.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceGetByPCIBusId(int *device, const char *pciBusId)
try: (hipDeviceGetByPCIBusId:=dll.hipDeviceGetByPCIBusId).restype, hipDeviceGetByPCIBusId.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t device)
try: (hipDeviceTotalMem:=dll.hipDeviceTotalMem).restype, hipDeviceTotalMem.argtypes = hipError_t, [ctypes.POINTER(size_t), hipDevice_t]
except AttributeError: pass

# hipError_t hipDeviceSynchronize()
try: (hipDeviceSynchronize:=dll.hipDeviceSynchronize).restype, hipDeviceSynchronize.argtypes = hipError_t, []
except AttributeError: pass

# hipError_t hipDeviceReset()
try: (hipDeviceReset:=dll.hipDeviceReset).restype, hipDeviceReset.argtypes = hipError_t, []
except AttributeError: pass

# hipError_t hipSetDevice(int deviceId)
try: (hipSetDevice:=dll.hipSetDevice).restype, hipSetDevice.argtypes = hipError_t, [ctypes.c_int]
except AttributeError: pass

# hipError_t hipSetValidDevices(int *device_arr, int len)
try: (hipSetValidDevices:=dll.hipSetValidDevices).restype, hipSetValidDevices.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
except AttributeError: pass

# hipError_t hipGetDevice(int *deviceId)
try: (hipGetDevice:=dll.hipGetDevice).restype, hipGetDevice.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# hipError_t hipGetDeviceCount(int *count)
try: (hipGetDeviceCount:=dll.hipGetDeviceCount).restype, hipGetDeviceCount.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr, int deviceId)
try: (hipDeviceGetAttribute:=dll.hipDeviceGetAttribute).restype, hipDeviceGetAttribute.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), hipDeviceAttribute_t, ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t *mem_pool, int device)
try: (hipDeviceGetDefaultMemPool:=dll.hipDeviceGetDefaultMemPool).restype, hipDeviceGetDefaultMemPool.argtypes = hipError_t, [ctypes.POINTER(hipMemPool_t), ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool)
try: (hipDeviceSetMemPool:=dll.hipDeviceSetMemPool).restype, hipDeviceSetMemPool.argtypes = hipError_t, [ctypes.c_int, hipMemPool_t]
except AttributeError: pass

# hipError_t hipDeviceGetMemPool(hipMemPool_t *mem_pool, int device)
try: (hipDeviceGetMemPool:=dll.hipDeviceGetMemPool).restype, hipDeviceGetMemPool.argtypes = hipError_t, [ctypes.POINTER(hipMemPool_t), ctypes.c_int]
except AttributeError: pass

# hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600 *prop, int deviceId)
try: (hipGetDevicePropertiesR0600:=dll.hipGetDevicePropertiesR0600).restype, hipGetDevicePropertiesR0600.argtypes = hipError_t, [ctypes.POINTER(hipDeviceProp_tR0600), ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig)
try: (hipDeviceSetCacheConfig:=dll.hipDeviceSetCacheConfig).restype, hipDeviceSetCacheConfig.argtypes = hipError_t, [hipFuncCache_t]
except AttributeError: pass

# hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig)
try: (hipDeviceGetCacheConfig:=dll.hipDeviceGetCacheConfig).restype, hipDeviceGetCacheConfig.argtypes = hipError_t, [ctypes.POINTER(hipFuncCache_t)]
except AttributeError: pass

# hipError_t hipDeviceGetLimit(size_t *pValue, enum hipLimit_t limit)
try: (hipDeviceGetLimit:=dll.hipDeviceGetLimit).restype, hipDeviceGetLimit.argtypes = hipError_t, [ctypes.POINTER(size_t), hipLimit_t]
except AttributeError: pass

# hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value)
try: (hipDeviceSetLimit:=dll.hipDeviceSetLimit).restype, hipDeviceSetLimit.argtypes = hipError_t, [hipLimit_t, size_t]
except AttributeError: pass

# hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *pConfig)
try: (hipDeviceGetSharedMemConfig:=dll.hipDeviceGetSharedMemConfig).restype, hipDeviceGetSharedMemConfig.argtypes = hipError_t, [ctypes.POINTER(hipSharedMemConfig)]
except AttributeError: pass

# hipError_t hipGetDeviceFlags(unsigned int *flags)
try: (hipGetDeviceFlags:=dll.hipGetDeviceFlags).restype, hipGetDeviceFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config)
try: (hipDeviceSetSharedMemConfig:=dll.hipDeviceSetSharedMemConfig).restype, hipDeviceSetSharedMemConfig.argtypes = hipError_t, [hipSharedMemConfig]
except AttributeError: pass

# hipError_t hipSetDeviceFlags(unsigned int flags)
try: (hipSetDeviceFlags:=dll.hipSetDeviceFlags).restype, hipSetDeviceFlags.argtypes = hipError_t, [ctypes.c_uint]
except AttributeError: pass

# hipError_t hipChooseDeviceR0600(int *device, const hipDeviceProp_tR0600 *prop)
try: (hipChooseDeviceR0600:=dll.hipChooseDeviceR0600).restype, hipChooseDeviceR0600.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(hipDeviceProp_tR0600)]
except AttributeError: pass

# hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t *linktype, uint32_t *hopcount)
try: (hipExtGetLinkTypeAndHopCount:=dll.hipExtGetLinkTypeAndHopCount).restype, hipExtGetLinkTypeAndHopCount.argtypes = hipError_t, [ctypes.c_int, ctypes.c_int, ctypes.POINTER(uint32_t), ctypes.POINTER(uint32_t)]
except AttributeError: pass

# hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t *handle, void *devPtr)
try: (hipIpcGetMemHandle:=dll.hipIpcGetMemHandle).restype, hipIpcGetMemHandle.argtypes = hipError_t, [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipIpcOpenMemHandle(void **devPtr, hipIpcMemHandle_t handle, unsigned int flags)
try: (hipIpcOpenMemHandle:=dll.hipIpcOpenMemHandle).restype, hipIpcOpenMemHandle.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipIpcCloseMemHandle(void *devPtr)
try: (hipIpcCloseMemHandle:=dll.hipIpcCloseMemHandle).restype, hipIpcCloseMemHandle.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t *handle, hipEvent_t event)
try: (hipIpcGetEventHandle:=dll.hipIpcGetEventHandle).restype, hipIpcGetEventHandle.argtypes = hipError_t, [ctypes.POINTER(hipIpcEventHandle_t), hipEvent_t]
except AttributeError: pass

# hipError_t hipIpcOpenEventHandle(hipEvent_t *event, hipIpcEventHandle_t handle)
try: (hipIpcOpenEventHandle:=dll.hipIpcOpenEventHandle).restype, hipIpcOpenEventHandle.argtypes = hipError_t, [ctypes.POINTER(hipEvent_t), hipIpcEventHandle_t]
except AttributeError: pass

# hipError_t hipFuncSetAttribute(const void *func, hipFuncAttribute attr, int value)
try: (hipFuncSetAttribute:=dll.hipFuncSetAttribute).restype, hipFuncSetAttribute.argtypes = hipError_t, [ctypes.c_void_p, hipFuncAttribute, ctypes.c_int]
except AttributeError: pass

# hipError_t hipFuncSetCacheConfig(const void *func, hipFuncCache_t config)
try: (hipFuncSetCacheConfig:=dll.hipFuncSetCacheConfig).restype, hipFuncSetCacheConfig.argtypes = hipError_t, [ctypes.c_void_p, hipFuncCache_t]
except AttributeError: pass

# hipError_t hipFuncSetSharedMemConfig(const void *func, hipSharedMemConfig config)
try: (hipFuncSetSharedMemConfig:=dll.hipFuncSetSharedMemConfig).restype, hipFuncSetSharedMemConfig.argtypes = hipError_t, [ctypes.c_void_p, hipSharedMemConfig]
except AttributeError: pass

# hipError_t hipGetLastError()
try: (hipGetLastError:=dll.hipGetLastError).restype, hipGetLastError.argtypes = hipError_t, []
except AttributeError: pass

# hipError_t hipExtGetLastError()
try: (hipExtGetLastError:=dll.hipExtGetLastError).restype, hipExtGetLastError.argtypes = hipError_t, []
except AttributeError: pass

# hipError_t hipPeekAtLastError()
try: (hipPeekAtLastError:=dll.hipPeekAtLastError).restype, hipPeekAtLastError.argtypes = hipError_t, []
except AttributeError: pass

# const char *hipGetErrorName(hipError_t hip_error)
try: (hipGetErrorName:=dll.hipGetErrorName).restype, hipGetErrorName.argtypes = ctypes.POINTER(ctypes.c_char), [hipError_t]
except AttributeError: pass

# const char *hipGetErrorString(hipError_t hipError)
try: (hipGetErrorString:=dll.hipGetErrorString).restype, hipGetErrorString.argtypes = ctypes.POINTER(ctypes.c_char), [hipError_t]
except AttributeError: pass

# hipError_t hipDrvGetErrorName(hipError_t hipError, const char **errorString)
try: (hipDrvGetErrorName:=dll.hipDrvGetErrorName).restype, hipDrvGetErrorName.argtypes = hipError_t, [hipError_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# hipError_t hipDrvGetErrorString(hipError_t hipError, const char **errorString)
try: (hipDrvGetErrorString:=dll.hipDrvGetErrorString).restype, hipDrvGetErrorString.argtypes = hipError_t, [hipError_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# hipError_t hipStreamCreate(hipStream_t *stream)
try: (hipStreamCreate:=dll.hipStreamCreate).restype, hipStreamCreate.argtypes = hipError_t, [ctypes.POINTER(hipStream_t)]
except AttributeError: pass

# hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
try: (hipStreamCreateWithFlags:=dll.hipStreamCreateWithFlags).restype, hipStreamCreateWithFlags.argtypes = hipError_t, [ctypes.POINTER(hipStream_t), ctypes.c_uint]
except AttributeError: pass

# hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags, int priority)
try: (hipStreamCreateWithPriority:=dll.hipStreamCreateWithPriority).restype, hipStreamCreateWithPriority.argtypes = hipError_t, [ctypes.POINTER(hipStream_t), ctypes.c_uint, ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
try: (hipDeviceGetStreamPriorityRange:=dll.hipDeviceGetStreamPriorityRange).restype, hipDeviceGetStreamPriorityRange.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# hipError_t hipStreamDestroy(hipStream_t stream)
try: (hipStreamDestroy:=dll.hipStreamDestroy).restype, hipStreamDestroy.argtypes = hipError_t, [hipStream_t]
except AttributeError: pass

# hipError_t hipStreamQuery(hipStream_t stream)
try: (hipStreamQuery:=dll.hipStreamQuery).restype, hipStreamQuery.argtypes = hipError_t, [hipStream_t]
except AttributeError: pass

# hipError_t hipStreamSynchronize(hipStream_t stream)
try: (hipStreamSynchronize:=dll.hipStreamSynchronize).restype, hipStreamSynchronize.argtypes = hipError_t, [hipStream_t]
except AttributeError: pass

# hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags = 0)
try: (hipStreamWaitEvent:=dll.hipStreamWaitEvent).restype, hipStreamWaitEvent.argtypes = hipError_t, [hipStream_t, hipEvent_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
try: (hipStreamGetFlags:=dll.hipStreamGetFlags).restype, hipStreamGetFlags.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# hipError_t hipStreamGetPriority(hipStream_t stream, int *priority)
try: (hipStreamGetPriority:=dll.hipStreamGetPriority).restype, hipStreamGetPriority.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t *device)
try: (hipStreamGetDevice:=dll.hipStreamGetDevice).restype, hipStreamGetDevice.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(hipDevice_t)]
except AttributeError: pass

# hipError_t hipExtStreamCreateWithCUMask(hipStream_t *stream, uint32_t cuMaskSize, const uint32_t *cuMask)
try: (hipExtStreamCreateWithCUMask:=dll.hipExtStreamCreateWithCUMask).restype, hipExtStreamCreateWithCUMask.argtypes = hipError_t, [ctypes.POINTER(hipStream_t), uint32_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

# hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t *cuMask)
try: (hipExtStreamGetCUMask:=dll.hipExtStreamGetCUMask).restype, hipExtStreamGetCUMask.argtypes = hipError_t, [hipStream_t, uint32_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

hipStreamCallback_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(ihipStream_t), hipError_t, ctypes.c_void_p)
# hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void *userData, unsigned int flags)
try: (hipStreamAddCallback:=dll.hipStreamAddCallback).restype, hipStreamAddCallback.argtypes = hipError_t, [hipStream_t, hipStreamCallback_t, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipStreamWaitValue32(hipStream_t stream, void *ptr, uint32_t value, unsigned int flags, uint32_t mask = 4294967295U)
try: (hipStreamWaitValue32:=dll.hipStreamWaitValue32).restype, hipStreamWaitValue32.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint, uint32_t]
except AttributeError: pass

uint64_t = ctypes.c_ulong
# hipError_t hipStreamWaitValue64(hipStream_t stream, void *ptr, uint64_t value, unsigned int flags, uint64_t mask = 18446744073709551615UL)
try: (hipStreamWaitValue64:=dll.hipStreamWaitValue64).restype, hipStreamWaitValue64.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint, uint64_t]
except AttributeError: pass

# hipError_t hipStreamWriteValue32(hipStream_t stream, void *ptr, uint32_t value, unsigned int flags)
try: (hipStreamWriteValue32:=dll.hipStreamWriteValue32).restype, hipStreamWriteValue32.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipStreamWriteValue64(hipStream_t stream, void *ptr, uint64_t value, unsigned int flags)
try: (hipStreamWriteValue64:=dll.hipStreamWriteValue64).restype, hipStreamWriteValue64.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned int flags)
try: (hipEventCreateWithFlags:=dll.hipEventCreateWithFlags).restype, hipEventCreateWithFlags.argtypes = hipError_t, [ctypes.POINTER(hipEvent_t), ctypes.c_uint]
except AttributeError: pass

# hipError_t hipEventCreate(hipEvent_t *event)
try: (hipEventCreate:=dll.hipEventCreate).restype, hipEventCreate.argtypes = hipError_t, [ctypes.POINTER(hipEvent_t)]
except AttributeError: pass

# hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = __null)
try: (hipEventRecord:=dll.hipEventRecord).restype, hipEventRecord.argtypes = hipError_t, [hipEvent_t, hipStream_t]
except AttributeError: pass

# hipError_t hipEventDestroy(hipEvent_t event)
try: (hipEventDestroy:=dll.hipEventDestroy).restype, hipEventDestroy.argtypes = hipError_t, [hipEvent_t]
except AttributeError: pass

# hipError_t hipEventSynchronize(hipEvent_t event)
try: (hipEventSynchronize:=dll.hipEventSynchronize).restype, hipEventSynchronize.argtypes = hipError_t, [hipEvent_t]
except AttributeError: pass

# hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
try: (hipEventElapsedTime:=dll.hipEventElapsedTime).restype, hipEventElapsedTime.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_float), hipEvent_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipEventQuery(hipEvent_t event)
try: (hipEventQuery:=dll.hipEventQuery).restype, hipEventQuery.argtypes = hipError_t, [hipEvent_t]
except AttributeError: pass

hipPointer_attribute = CEnum(ctypes.c_uint)
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

# hipError_t hipPointerSetAttribute(const void *value, hipPointer_attribute attribute, hipDeviceptr_t ptr)
try: (hipPointerSetAttribute:=dll.hipPointerSetAttribute).restype, hipPointerSetAttribute.argtypes = hipError_t, [ctypes.c_void_p, hipPointer_attribute, hipDeviceptr_t]
except AttributeError: pass

# hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *ptr)
try: (hipPointerGetAttributes:=dll.hipPointerGetAttributes).restype, hipPointerGetAttributes.argtypes = hipError_t, [ctypes.POINTER(hipPointerAttribute_t), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipPointerGetAttribute(void *data, hipPointer_attribute attribute, hipDeviceptr_t ptr)
try: (hipPointerGetAttribute:=dll.hipPointerGetAttribute).restype, hipPointerGetAttribute.argtypes = hipError_t, [ctypes.c_void_p, hipPointer_attribute, hipDeviceptr_t]
except AttributeError: pass

# hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute *attributes, void **data, hipDeviceptr_t ptr)
try: (hipDrvPointerGetAttributes:=dll.hipDrvPointerGetAttributes).restype, hipDrvPointerGetAttributes.argtypes = hipError_t, [ctypes.c_uint, ctypes.POINTER(hipPointer_attribute), ctypes.POINTER(ctypes.c_void_p), hipDeviceptr_t]
except AttributeError: pass

# hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t *extSem_out, const hipExternalSemaphoreHandleDesc *semHandleDesc)
try: (hipImportExternalSemaphore:=dll.hipImportExternalSemaphore).restype, hipImportExternalSemaphore.argtypes = hipError_t, [ctypes.POINTER(hipExternalSemaphore_t), ctypes.POINTER(hipExternalSemaphoreHandleDesc)]
except AttributeError: pass

# hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t *extSemArray, const hipExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, hipStream_t stream)
try: (hipSignalExternalSemaphoresAsync:=dll.hipSignalExternalSemaphoresAsync).restype, hipSignalExternalSemaphoresAsync.argtypes = hipError_t, [ctypes.POINTER(hipExternalSemaphore_t), ctypes.POINTER(hipExternalSemaphoreSignalParams), ctypes.c_uint, hipStream_t]
except AttributeError: pass

# hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t *extSemArray, const hipExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, hipStream_t stream)
try: (hipWaitExternalSemaphoresAsync:=dll.hipWaitExternalSemaphoresAsync).restype, hipWaitExternalSemaphoresAsync.argtypes = hipError_t, [ctypes.POINTER(hipExternalSemaphore_t), ctypes.POINTER(hipExternalSemaphoreWaitParams), ctypes.c_uint, hipStream_t]
except AttributeError: pass

# hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem)
try: (hipDestroyExternalSemaphore:=dll.hipDestroyExternalSemaphore).restype, hipDestroyExternalSemaphore.argtypes = hipError_t, [hipExternalSemaphore_t]
except AttributeError: pass

# hipError_t hipImportExternalMemory(hipExternalMemory_t *extMem_out, const hipExternalMemoryHandleDesc *memHandleDesc)
try: (hipImportExternalMemory:=dll.hipImportExternalMemory).restype, hipImportExternalMemory.argtypes = hipError_t, [ctypes.POINTER(hipExternalMemory_t), ctypes.POINTER(hipExternalMemoryHandleDesc)]
except AttributeError: pass

# hipError_t hipExternalMemoryGetMappedBuffer(void **devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc *bufferDesc)
try: (hipExternalMemoryGetMappedBuffer:=dll.hipExternalMemoryGetMappedBuffer).restype, hipExternalMemoryGetMappedBuffer.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), hipExternalMemory_t, ctypes.POINTER(hipExternalMemoryBufferDesc)]
except AttributeError: pass

# hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem)
try: (hipDestroyExternalMemory:=dll.hipDestroyExternalMemory).restype, hipDestroyExternalMemory.argtypes = hipError_t, [hipExternalMemory_t]
except AttributeError: pass

hipMipmappedArray_t = ctypes.POINTER(hipMipmappedArray)
# hipError_t hipExternalMemoryGetMappedMipmappedArray(hipMipmappedArray_t *mipmap, hipExternalMemory_t extMem, const hipExternalMemoryMipmappedArrayDesc *mipmapDesc)
try: (hipExternalMemoryGetMappedMipmappedArray:=dll.hipExternalMemoryGetMappedMipmappedArray).restype, hipExternalMemoryGetMappedMipmappedArray.argtypes = hipError_t, [ctypes.POINTER(hipMipmappedArray_t), hipExternalMemory_t, ctypes.POINTER(hipExternalMemoryMipmappedArrayDesc)]
except AttributeError: pass

# hipError_t hipMalloc(void **ptr, size_t size)
try: (hipMalloc:=dll.hipMalloc).restype, hipMalloc.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t]
except AttributeError: pass

# hipError_t hipExtMallocWithFlags(void **ptr, size_t sizeBytes, unsigned int flags)
try: (hipExtMallocWithFlags:=dll.hipExtMallocWithFlags).restype, hipExtMallocWithFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("use hipHostMalloc instead"))) hipError_t hipMallocHost(void **ptr, size_t size)
try: (hipMallocHost:=dll.hipMallocHost).restype, hipMallocHost.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t]
except AttributeError: pass

# __attribute__((deprecated("use hipHostMalloc instead"))) hipError_t hipMemAllocHost(void **ptr, size_t size)
try: (hipMemAllocHost:=dll.hipMemAllocHost).restype, hipMemAllocHost.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t]
except AttributeError: pass

# hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags)
try: (hipHostMalloc:=dll.hipHostMalloc).restype, hipHostMalloc.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMallocManaged(void **dev_ptr, size_t size, unsigned int flags = 1)
try: (hipMallocManaged:=dll.hipMallocManaged).restype, hipMallocManaged.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMemPrefetchAsync(const void *dev_ptr, size_t count, int device, hipStream_t stream = 0)
try: (hipMemPrefetchAsync:=dll.hipMemPrefetchAsync).restype, hipMemPrefetchAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_int, hipStream_t]
except AttributeError: pass

# hipError_t hipMemAdvise(const void *dev_ptr, size_t count, hipMemoryAdvise advice, int device)
try: (hipMemAdvise:=dll.hipMemAdvise).restype, hipMemAdvise.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipMemoryAdvise, ctypes.c_int]
except AttributeError: pass

# hipError_t hipMemRangeGetAttribute(void *data, size_t data_size, hipMemRangeAttribute attribute, const void *dev_ptr, size_t count)
try: (hipMemRangeGetAttribute:=dll.hipMemRangeGetAttribute).restype, hipMemRangeGetAttribute.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipMemRangeAttribute, ctypes.c_void_p, size_t]
except AttributeError: pass

# hipError_t hipMemRangeGetAttributes(void **data, size_t *data_sizes, hipMemRangeAttribute *attributes, size_t num_attributes, const void *dev_ptr, size_t count)
try: (hipMemRangeGetAttributes:=dll.hipMemRangeGetAttributes).restype, hipMemRangeGetAttributes.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), ctypes.POINTER(hipMemRangeAttribute), size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# hipError_t hipStreamAttachMemAsync(hipStream_t stream, void *dev_ptr, size_t length = 0, unsigned int flags = 4)
try: (hipStreamAttachMemAsync:=dll.hipStreamAttachMemAsync).restype, hipStreamAttachMemAsync.argtypes = hipError_t, [hipStream_t, ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream)
try: (hipMallocAsync:=dll.hipMallocAsync).restype, hipMallocAsync.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipFreeAsync(void *dev_ptr, hipStream_t stream)
try: (hipFreeAsync:=dll.hipFreeAsync).restype, hipFreeAsync.argtypes = hipError_t, [ctypes.c_void_p, hipStream_t]
except AttributeError: pass

# hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold)
try: (hipMemPoolTrimTo:=dll.hipMemPoolTrimTo).restype, hipMemPoolTrimTo.argtypes = hipError_t, [hipMemPool_t, size_t]
except AttributeError: pass

# hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void *value)
try: (hipMemPoolSetAttribute:=dll.hipMemPoolSetAttribute).restype, hipMemPoolSetAttribute.argtypes = hipError_t, [hipMemPool_t, hipMemPoolAttr, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void *value)
try: (hipMemPoolGetAttribute:=dll.hipMemPoolGetAttribute).restype, hipMemPoolGetAttribute.argtypes = hipError_t, [hipMemPool_t, hipMemPoolAttr, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc *desc_list, size_t count)
try: (hipMemPoolSetAccess:=dll.hipMemPoolSetAccess).restype, hipMemPoolSetAccess.argtypes = hipError_t, [hipMemPool_t, ctypes.POINTER(hipMemAccessDesc), size_t]
except AttributeError: pass

# hipError_t hipMemPoolGetAccess(hipMemAccessFlags *flags, hipMemPool_t mem_pool, hipMemLocation *location)
try: (hipMemPoolGetAccess:=dll.hipMemPoolGetAccess).restype, hipMemPoolGetAccess.argtypes = hipError_t, [ctypes.POINTER(hipMemAccessFlags), hipMemPool_t, ctypes.POINTER(hipMemLocation)]
except AttributeError: pass

# hipError_t hipMemPoolCreate(hipMemPool_t *mem_pool, const hipMemPoolProps *pool_props)
try: (hipMemPoolCreate:=dll.hipMemPoolCreate).restype, hipMemPoolCreate.argtypes = hipError_t, [ctypes.POINTER(hipMemPool_t), ctypes.POINTER(hipMemPoolProps)]
except AttributeError: pass

# hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool)
try: (hipMemPoolDestroy:=dll.hipMemPoolDestroy).restype, hipMemPoolDestroy.argtypes = hipError_t, [hipMemPool_t]
except AttributeError: pass

# hipError_t hipMallocFromPoolAsync(void **dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream)
try: (hipMallocFromPoolAsync:=dll.hipMallocFromPoolAsync).restype, hipMallocFromPoolAsync.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, hipMemPool_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemPoolExportToShareableHandle(void *shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags)
try: (hipMemPoolExportToShareableHandle:=dll.hipMemPoolExportToShareableHandle).restype, hipMemPoolExportToShareableHandle.argtypes = hipError_t, [ctypes.c_void_p, hipMemPool_t, hipMemAllocationHandleType, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t *mem_pool, void *shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags)
try: (hipMemPoolImportFromShareableHandle:=dll.hipMemPoolImportFromShareableHandle).restype, hipMemPoolImportFromShareableHandle.argtypes = hipError_t, [ctypes.POINTER(hipMemPool_t), ctypes.c_void_p, hipMemAllocationHandleType, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData *export_data, void *dev_ptr)
try: (hipMemPoolExportPointer:=dll.hipMemPoolExportPointer).restype, hipMemPoolExportPointer.argtypes = hipError_t, [ctypes.POINTER(hipMemPoolPtrExportData), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMemPoolImportPointer(void **dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData *export_data)
try: (hipMemPoolImportPointer:=dll.hipMemPoolImportPointer).restype, hipMemPoolImportPointer.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), hipMemPool_t, ctypes.POINTER(hipMemPoolPtrExportData)]
except AttributeError: pass

# __attribute__((deprecated("use hipHostMalloc instead"))) hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags)
try: (hipHostAlloc:=dll.hipHostAlloc).restype, hipHostAlloc.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr, unsigned int flags)
try: (hipHostGetDevicePointer:=dll.hipHostGetDevicePointer).restype, hipHostGetDevicePointer.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipHostGetFlags(unsigned int *flagsPtr, void *hostPtr)
try: (hipHostGetFlags:=dll.hipHostGetFlags).restype, hipHostGetFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags)
try: (hipHostRegister:=dll.hipHostRegister).restype, hipHostRegister.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipHostUnregister(void *hostPtr)
try: (hipHostUnregister:=dll.hipHostUnregister).restype, hipHostUnregister.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height)
try: (hipMallocPitch:=dll.hipMallocPitch).restype, hipMallocPitch.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), size_t, size_t]
except AttributeError: pass

# hipError_t hipMemAllocPitch(hipDeviceptr_t *dptr, size_t *pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes)
try: (hipMemAllocPitch:=dll.hipMemAllocPitch).restype, hipMemAllocPitch.argtypes = hipError_t, [ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(size_t), size_t, size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipFree(void *ptr)
try: (hipFree:=dll.hipFree).restype, hipFree.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

# __attribute__((deprecated("use hipHostFree instead"))) hipError_t hipFreeHost(void *ptr)
try: (hipFreeHost:=dll.hipFreeHost).restype, hipFreeHost.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipHostFree(void *ptr)
try: (hipHostFree:=dll.hipHostFree).restype, hipHostFree.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
try: (hipMemcpy:=dll.hipMemcpy).restype, hipMemcpy.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipMemcpyWithStream(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
try: (hipMemcpyWithStream:=dll.hipMemcpyWithStream).restype, hipMemcpyWithStream.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes)
try: (hipMemcpyHtoD:=dll.hipMemcpyHtoD).restype, hipMemcpyHtoD.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes)
try: (hipMemcpyDtoH:=dll.hipMemcpyDtoH).restype, hipMemcpyDtoH.argtypes = hipError_t, [ctypes.c_void_p, hipDeviceptr_t, size_t]
except AttributeError: pass

# hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)
try: (hipMemcpyDtoD:=dll.hipMemcpyDtoD).restype, hipMemcpyDtoD.argtypes = hipError_t, [hipDeviceptr_t, hipDeviceptr_t, size_t]
except AttributeError: pass

# hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset, size_t ByteCount)
try: (hipMemcpyAtoD:=dll.hipMemcpyAtoD).restype, hipMemcpyAtoD.argtypes = hipError_t, [hipDeviceptr_t, hipArray_t, size_t, size_t]
except AttributeError: pass

# hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice, size_t ByteCount)
try: (hipMemcpyDtoA:=dll.hipMemcpyDtoA).restype, hipMemcpyDtoA.argtypes = hipError_t, [hipArray_t, size_t, hipDeviceptr_t, size_t]
except AttributeError: pass

# hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray, size_t srcOffset, size_t ByteCount)
try: (hipMemcpyAtoA:=dll.hipMemcpyAtoA).restype, hipMemcpyAtoA.argtypes = hipError_t, [hipArray_t, size_t, hipArray_t, size_t, size_t]
except AttributeError: pass

# hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes, hipStream_t stream)
try: (hipMemcpyHtoDAsync:=dll.hipMemcpyHtoDAsync).restype, hipMemcpyHtoDAsync.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_void_p, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
try: (hipMemcpyDtoHAsync:=dll.hipMemcpyDtoHAsync).restype, hipMemcpyDtoHAsync.argtypes = hipError_t, [ctypes.c_void_p, hipDeviceptr_t, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
try: (hipMemcpyDtoDAsync:=dll.hipMemcpyDtoDAsync).restype, hipMemcpyDtoDAsync.argtypes = hipError_t, [hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyAtoHAsync(void *dstHost, hipArray_t srcArray, size_t srcOffset, size_t ByteCount, hipStream_t stream)
try: (hipMemcpyAtoHAsync:=dll.hipMemcpyAtoHAsync).restype, hipMemcpyAtoHAsync.argtypes = hipError_t, [ctypes.c_void_p, hipArray_t, size_t, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, hipStream_t stream)
try: (hipMemcpyHtoAAsync:=dll.hipMemcpyHtoAAsync).restype, hipMemcpyHtoAAsync.argtypes = hipError_t, [hipArray_t, size_t, ctypes.c_void_p, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes, hipModule_t hmod, const char *name)
try: (hipModuleGetGlobal:=dll.hipModuleGetGlobal).restype, hipModuleGetGlobal.argtypes = hipError_t, [ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(size_t), hipModule_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hipError_t hipGetSymbolAddress(void **devPtr, const void *symbol)
try: (hipGetSymbolAddress:=dll.hipGetSymbolAddress).restype, hipGetSymbolAddress.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipGetSymbolSize(size_t *size, const void *symbol)
try: (hipGetSymbolSize:=dll.hipGetSymbolSize).restype, hipGetSymbolSize.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipGetProcAddress(const char *symbol, void **pfn, int hipVersion, uint64_t flags, hipDriverProcAddressQueryResult *symbolStatus)
try: (hipGetProcAddress:=dll.hipGetProcAddress).restype, hipGetProcAddress.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, uint64_t, ctypes.POINTER(hipDriverProcAddressQueryResult)]
except AttributeError: pass

# hipError_t hipMemcpyToSymbol(const void *symbol, const void *src, size_t sizeBytes, size_t offset = 0, hipMemcpyKind kind = hipMemcpyHostToDevice)
try: (hipMemcpyToSymbol:=dll.hipMemcpyToSymbol).restype, hipMemcpyToSymbol.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipMemcpyToSymbolAsync(const void *symbol, const void *src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream = 0)
try: (hipMemcpyToSymbolAsync:=dll.hipMemcpyToSymbolAsync).restype, hipMemcpyToSymbolAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes, size_t offset = 0, hipMemcpyKind kind = hipMemcpyDeviceToHost)
try: (hipMemcpyFromSymbol:=dll.hipMemcpyFromSymbol).restype, hipMemcpyFromSymbol.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream = 0)
try: (hipMemcpyFromSymbolAsync:=dll.hipMemcpyFromSymbolAsync).restype, hipMemcpyFromSymbolAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream = 0)
try: (hipMemcpyAsync:=dll.hipMemcpyAsync).restype, hipMemcpyAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

# hipError_t hipMemset(void *dst, int value, size_t sizeBytes)
try: (hipMemset:=dll.hipMemset).restype, hipMemset.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int, size_t]
except AttributeError: pass

# hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count)
try: (hipMemsetD8:=dll.hipMemsetD8).restype, hipMemsetD8.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_ubyte, size_t]
except AttributeError: pass

# hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream = 0)
try: (hipMemsetD8Async:=dll.hipMemsetD8Async).restype, hipMemsetD8Async.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_ubyte, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count)
try: (hipMemsetD16:=dll.hipMemsetD16).restype, hipMemsetD16.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_ushort, size_t]
except AttributeError: pass

# hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream = 0)
try: (hipMemsetD16Async:=dll.hipMemsetD16Async).restype, hipMemsetD16Async.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_ushort, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count)
try: (hipMemsetD32:=dll.hipMemsetD32).restype, hipMemsetD32.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_int, size_t]
except AttributeError: pass

# hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream = 0)
try: (hipMemsetAsync:=dll.hipMemsetAsync).restype, hipMemsetAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream = 0)
try: (hipMemsetD32Async:=dll.hipMemsetD32Async).restype, hipMemsetD32Async.argtypes = hipError_t, [hipDeviceptr_t, ctypes.c_int, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemset2D(void *dst, size_t pitch, int value, size_t width, size_t height)
try: (hipMemset2D:=dll.hipMemset2D).restype, hipMemset2D.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_int, size_t, size_t]
except AttributeError: pass

# hipError_t hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream = 0)
try: (hipMemset2DAsync:=dll.hipMemset2DAsync).restype, hipMemset2DAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_int, size_t, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent)
try: (hipMemset3D:=dll.hipMemset3D).restype, hipMemset3D.argtypes = hipError_t, [hipPitchedPtr, ctypes.c_int, hipExtent]
except AttributeError: pass

# hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream = 0)
try: (hipMemset3DAsync:=dll.hipMemset3DAsync).restype, hipMemset3DAsync.argtypes = hipError_t, [hipPitchedPtr, ctypes.c_int, hipExtent, hipStream_t]
except AttributeError: pass

# hipError_t hipMemGetInfo(size_t *free, size_t *total)
try: (hipMemGetInfo:=dll.hipMemGetInfo).restype, hipMemGetInfo.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipMemPtrGetInfo(void *ptr, size_t *size)
try: (hipMemPtrGetInfo:=dll.hipMemPtrGetInfo).restype, hipMemPtrGetInfo.argtypes = hipError_t, [ctypes.c_void_p, ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipMallocArray(hipArray_t *array, const hipChannelFormatDesc *desc, size_t width, size_t height = 0, unsigned int flags = 0)
try: (hipMallocArray:=dll.hipMallocArray).restype, hipMallocArray.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), ctypes.POINTER(hipChannelFormatDesc), size_t, size_t, ctypes.c_uint]
except AttributeError: pass

class HIP_ARRAY_DESCRIPTOR(ctypes.Structure): pass
HIP_ARRAY_DESCRIPTOR._fields_ = [
  ('Width', size_t),
  ('Height', size_t),
  ('Format', hipArray_Format),
  ('NumChannels', ctypes.c_uint),
]
# hipError_t hipArrayCreate(hipArray_t *pHandle, const HIP_ARRAY_DESCRIPTOR *pAllocateArray)
try: (hipArrayCreate:=dll.hipArrayCreate).restype, hipArrayCreate.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), ctypes.POINTER(HIP_ARRAY_DESCRIPTOR)]
except AttributeError: pass

# hipError_t hipArrayDestroy(hipArray_t array)
try: (hipArrayDestroy:=dll.hipArrayDestroy).restype, hipArrayDestroy.argtypes = hipError_t, [hipArray_t]
except AttributeError: pass

class HIP_ARRAY3D_DESCRIPTOR(ctypes.Structure): pass
HIP_ARRAY3D_DESCRIPTOR._fields_ = [
  ('Width', size_t),
  ('Height', size_t),
  ('Depth', size_t),
  ('Format', hipArray_Format),
  ('NumChannels', ctypes.c_uint),
  ('Flags', ctypes.c_uint),
]
# hipError_t hipArray3DCreate(hipArray_t *array, const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray)
try: (hipArray3DCreate:=dll.hipArray3DCreate).restype, hipArray3DCreate.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR)]
except AttributeError: pass

# hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent)
try: (hipMalloc3D:=dll.hipMalloc3D).restype, hipMalloc3D.argtypes = hipError_t, [ctypes.POINTER(hipPitchedPtr), hipExtent]
except AttributeError: pass

# hipError_t hipFreeArray(hipArray_t array)
try: (hipFreeArray:=dll.hipFreeArray).restype, hipFreeArray.argtypes = hipError_t, [hipArray_t]
except AttributeError: pass

# hipError_t hipMalloc3DArray(hipArray_t *array, const struct hipChannelFormatDesc *desc, struct hipExtent extent, unsigned int flags)
try: (hipMalloc3DArray:=dll.hipMalloc3DArray).restype, hipMalloc3DArray.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), ctypes.POINTER(hipChannelFormatDesc), hipExtent, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipArrayGetInfo(hipChannelFormatDesc *desc, hipExtent *extent, unsigned int *flags, hipArray_t array)
try: (hipArrayGetInfo:=dll.hipArrayGetInfo).restype, hipArrayGetInfo.argtypes = hipError_t, [ctypes.POINTER(hipChannelFormatDesc), ctypes.POINTER(hipExtent), ctypes.POINTER(ctypes.c_uint), hipArray_t]
except AttributeError: pass

# hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR *pArrayDescriptor, hipArray_t array)
try: (hipArrayGetDescriptor:=dll.hipArrayGetDescriptor).restype, hipArrayGetDescriptor.argtypes = hipError_t, [ctypes.POINTER(HIP_ARRAY_DESCRIPTOR), hipArray_t]
except AttributeError: pass

# hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR *pArrayDescriptor, hipArray_t array)
try: (hipArray3DGetDescriptor:=dll.hipArray3DGetDescriptor).restype, hipArray3DGetDescriptor.argtypes = hipError_t, [ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR), hipArray_t]
except AttributeError: pass

# hipError_t hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)
try: (hipMemcpy2D:=dll.hipMemcpy2D).restype, hipMemcpy2D.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

class hip_Memcpy2D(ctypes.Structure): pass
hip_Memcpy2D._fields_ = [
  ('srcXInBytes', size_t),
  ('srcY', size_t),
  ('srcMemoryType', hipMemoryType),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', hipDeviceptr_t),
  ('srcArray', hipArray_t),
  ('srcPitch', size_t),
  ('dstXInBytes', size_t),
  ('dstY', size_t),
  ('dstMemoryType', hipMemoryType),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', hipDeviceptr_t),
  ('dstArray', hipArray_t),
  ('dstPitch', size_t),
  ('WidthInBytes', size_t),
  ('Height', size_t),
]
# hipError_t hipMemcpyParam2D(const hip_Memcpy2D *pCopy)
try: (hipMemcpyParam2D:=dll.hipMemcpyParam2D).restype, hipMemcpyParam2D.argtypes = hipError_t, [ctypes.POINTER(hip_Memcpy2D)]
except AttributeError: pass

# hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D *pCopy, hipStream_t stream = 0)
try: (hipMemcpyParam2DAsync:=dll.hipMemcpyParam2DAsync).restype, hipMemcpyParam2DAsync.argtypes = hipError_t, [ctypes.POINTER(hip_Memcpy2D), hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream = 0)
try: (hipMemcpy2DAsync:=dll.hipMemcpy2DAsync).restype, hipMemcpy2DAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind)
try: (hipMemcpy2DToArray:=dll.hipMemcpy2DToArray).restype, hipMemcpy2DToArray.argtypes = hipError_t, [hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream = 0)
try: (hipMemcpy2DToArrayAsync:=dll.hipMemcpy2DToArrayAsync).restype, hipMemcpy2DToArrayAsync.argtypes = hipError_t, [hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

class const_hipArray(ctypes.Structure): pass
const_hipArray._fields_ = []
hipArray_const_t = ctypes.POINTER(const_hipArray)
# hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind)
try: (hipMemcpy2DArrayToArray:=dll.hipMemcpy2DArrayToArray).restype, hipMemcpy2DArrayToArray.argtypes = hipError_t, [hipArray_t, size_t, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, hipMemcpyKind kind)
try: (hipMemcpyToArray:=dll.hipMemcpyToArray).restype, hipMemcpyToArray.argtypes = hipError_t, [hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipMemcpyFromArray(void *dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind)
try: (hipMemcpyFromArray:=dll.hipMemcpyFromArray).restype, hipMemcpyFromArray.argtypes = hipError_t, [ctypes.c_void_p, hipArray_const_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipMemcpy2DFromArray(void *dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind)
try: (hipMemcpy2DFromArray:=dll.hipMemcpy2DFromArray).restype, hipMemcpy2DFromArray.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipMemcpy2DFromArrayAsync(void *dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream = 0)
try: (hipMemcpy2DFromArrayAsync:=dll.hipMemcpy2DFromArrayAsync).restype, hipMemcpy2DFromArrayAsync.argtypes = hipError_t, [ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind, hipStream_t]
except AttributeError: pass

# hipError_t hipMemcpyAtoH(void *dst, hipArray_t srcArray, size_t srcOffset, size_t count)
try: (hipMemcpyAtoH:=dll.hipMemcpyAtoH).restype, hipMemcpyAtoH.argtypes = hipError_t, [ctypes.c_void_p, hipArray_t, size_t, size_t]
except AttributeError: pass

# hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void *srcHost, size_t count)
try: (hipMemcpyHtoA:=dll.hipMemcpyHtoA).restype, hipMemcpyHtoA.argtypes = hipError_t, [hipArray_t, size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p)
try: (hipMemcpy3D:=dll.hipMemcpy3D).restype, hipMemcpy3D.argtypes = hipError_t, [ctypes.POINTER(hipMemcpy3DParms)]
except AttributeError: pass

# hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms *p, hipStream_t stream = 0)
try: (hipMemcpy3DAsync:=dll.hipMemcpy3DAsync).restype, hipMemcpy3DAsync.argtypes = hipError_t, [ctypes.POINTER(hipMemcpy3DParms), hipStream_t]
except AttributeError: pass

class HIP_MEMCPY3D(ctypes.Structure): pass
HIP_MEMCPY3D._fields_ = [
  ('srcXInBytes', size_t),
  ('srcY', size_t),
  ('srcZ', size_t),
  ('srcLOD', size_t),
  ('srcMemoryType', hipMemoryType),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', hipDeviceptr_t),
  ('srcArray', hipArray_t),
  ('srcPitch', size_t),
  ('srcHeight', size_t),
  ('dstXInBytes', size_t),
  ('dstY', size_t),
  ('dstZ', size_t),
  ('dstLOD', size_t),
  ('dstMemoryType', hipMemoryType),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', hipDeviceptr_t),
  ('dstArray', hipArray_t),
  ('dstPitch', size_t),
  ('dstHeight', size_t),
  ('WidthInBytes', size_t),
  ('Height', size_t),
  ('Depth', size_t),
]
# hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D *pCopy)
try: (hipDrvMemcpy3D:=dll.hipDrvMemcpy3D).restype, hipDrvMemcpy3D.argtypes = hipError_t, [ctypes.POINTER(HIP_MEMCPY3D)]
except AttributeError: pass

# hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D *pCopy, hipStream_t stream)
try: (hipDrvMemcpy3DAsync:=dll.hipDrvMemcpy3DAsync).restype, hipDrvMemcpy3DAsync.argtypes = hipError_t, [ctypes.POINTER(HIP_MEMCPY3D), hipStream_t]
except AttributeError: pass

# hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId, int peerDeviceId)
try: (hipDeviceCanAccessPeer:=dll.hipDeviceCanAccessPeer).restype, hipDeviceCanAccessPeer.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags)
try: (hipDeviceEnablePeerAccess:=dll.hipDeviceEnablePeerAccess).restype, hipDeviceEnablePeerAccess.argtypes = hipError_t, [ctypes.c_int, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipDeviceDisablePeerAccess(int peerDeviceId)
try: (hipDeviceDisablePeerAccess:=dll.hipDeviceDisablePeerAccess).restype, hipDeviceDisablePeerAccess.argtypes = hipError_t, [ctypes.c_int]
except AttributeError: pass

# hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize, hipDeviceptr_t dptr)
try: (hipMemGetAddressRange:=dll.hipMemGetAddressRange).restype, hipMemGetAddressRange.argtypes = hipError_t, [ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(size_t), hipDeviceptr_t]
except AttributeError: pass

# hipError_t hipMemcpyPeer(void *dst, int dstDeviceId, const void *src, int srcDeviceId, size_t sizeBytes)
try: (hipMemcpyPeer:=dll.hipMemcpyPeer).restype, hipMemcpyPeer.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, size_t]
except AttributeError: pass

# hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src, int srcDevice, size_t sizeBytes, hipStream_t stream = 0)
try: (hipMemcpyPeerAsync:=dll.hipMemcpyPeerAsync).restype, hipMemcpyPeerAsync.argtypes = hipError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, size_t, hipStream_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device)
try: (hipCtxCreate:=dll.hipCtxCreate).restype, hipCtxCreate.argtypes = hipError_t, [ctypes.POINTER(hipCtx_t), ctypes.c_uint, hipDevice_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxDestroy(hipCtx_t ctx)
try: (hipCtxDestroy:=dll.hipCtxDestroy).restype, hipCtxDestroy.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxPopCurrent(hipCtx_t *ctx)
try: (hipCtxPopCurrent:=dll.hipCtxPopCurrent).restype, hipCtxPopCurrent.argtypes = hipError_t, [ctypes.POINTER(hipCtx_t)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxPushCurrent(hipCtx_t ctx)
try: (hipCtxPushCurrent:=dll.hipCtxPushCurrent).restype, hipCtxPushCurrent.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxSetCurrent(hipCtx_t ctx)
try: (hipCtxSetCurrent:=dll.hipCtxSetCurrent).restype, hipCtxSetCurrent.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxGetCurrent(hipCtx_t *ctx)
try: (hipCtxGetCurrent:=dll.hipCtxGetCurrent).restype, hipCtxGetCurrent.argtypes = hipError_t, [ctypes.POINTER(hipCtx_t)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxGetDevice(hipDevice_t *device)
try: (hipCtxGetDevice:=dll.hipCtxGetDevice).restype, hipCtxGetDevice.argtypes = hipError_t, [ctypes.POINTER(hipDevice_t)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int *apiVersion)
try: (hipCtxGetApiVersion:=dll.hipCtxGetApiVersion).restype, hipCtxGetApiVersion.argtypes = hipError_t, [hipCtx_t, ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxGetCacheConfig(hipFuncCache_t *cacheConfig)
try: (hipCtxGetCacheConfig:=dll.hipCtxGetCacheConfig).restype, hipCtxGetCacheConfig.argtypes = hipError_t, [ctypes.POINTER(hipFuncCache_t)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig)
try: (hipCtxSetCacheConfig:=dll.hipCtxSetCacheConfig).restype, hipCtxSetCacheConfig.argtypes = hipError_t, [hipFuncCache_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config)
try: (hipCtxSetSharedMemConfig:=dll.hipCtxSetSharedMemConfig).restype, hipCtxSetSharedMemConfig.argtypes = hipError_t, [hipSharedMemConfig]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig *pConfig)
try: (hipCtxGetSharedMemConfig:=dll.hipCtxGetSharedMemConfig).restype, hipCtxGetSharedMemConfig.argtypes = hipError_t, [ctypes.POINTER(hipSharedMemConfig)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxSynchronize()
try: (hipCtxSynchronize:=dll.hipCtxSynchronize).restype, hipCtxSynchronize.argtypes = hipError_t, []
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxGetFlags(unsigned int *flags)
try: (hipCtxGetFlags:=dll.hipCtxGetFlags).restype, hipCtxGetFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags)
try: (hipCtxEnablePeerAccess:=dll.hipCtxEnablePeerAccess).restype, hipCtxEnablePeerAccess.argtypes = hipError_t, [hipCtx_t, ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx)
try: (hipCtxDisablePeerAccess:=dll.hipCtxDisablePeerAccess).restype, hipCtxDisablePeerAccess.argtypes = hipError_t, [hipCtx_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int *flags, int *active)
try: (hipDevicePrimaryCtxGetState:=dll.hipDevicePrimaryCtxGetState).restype, hipDevicePrimaryCtxGetState.argtypes = hipError_t, [hipDevice_t, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev)
try: (hipDevicePrimaryCtxRelease:=dll.hipDevicePrimaryCtxRelease).restype, hipDevicePrimaryCtxRelease.argtypes = hipError_t, [hipDevice_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t dev)
try: (hipDevicePrimaryCtxRetain:=dll.hipDevicePrimaryCtxRetain).restype, hipDevicePrimaryCtxRetain.argtypes = hipError_t, [ctypes.POINTER(hipCtx_t), hipDevice_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev)
try: (hipDevicePrimaryCtxReset:=dll.hipDevicePrimaryCtxReset).restype, hipDevicePrimaryCtxReset.argtypes = hipError_t, [hipDevice_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags)
try: (hipDevicePrimaryCtxSetFlags:=dll.hipDevicePrimaryCtxSetFlags).restype, hipDevicePrimaryCtxSetFlags.argtypes = hipError_t, [hipDevice_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipModuleLoad(hipModule_t *module, const char *fname)
try: (hipModuleLoad:=dll.hipModuleLoad).restype, hipModuleLoad.argtypes = hipError_t, [ctypes.POINTER(hipModule_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hipError_t hipModuleUnload(hipModule_t module)
try: (hipModuleUnload:=dll.hipModuleUnload).restype, hipModuleUnload.argtypes = hipError_t, [hipModule_t]
except AttributeError: pass

# hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname)
try: (hipModuleGetFunction:=dll.hipModuleGetFunction).restype, hipModuleGetFunction.argtypes = hipError_t, [ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hipError_t hipFuncGetAttributes(struct hipFuncAttributes *attr, const void *func)
try: (hipFuncGetAttributes:=dll.hipFuncGetAttributes).restype, hipFuncGetAttributes.argtypes = hipError_t, [ctypes.POINTER(hipFuncAttributes), ctypes.c_void_p]
except AttributeError: pass

hipFunction_attribute = CEnum(ctypes.c_uint)
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

# hipError_t hipFuncGetAttribute(int *value, hipFunction_attribute attrib, hipFunction_t hfunc)
try: (hipFuncGetAttribute:=dll.hipFuncGetAttribute).restype, hipFuncGetAttribute.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), hipFunction_attribute, hipFunction_t]
except AttributeError: pass

# hipError_t hipGetFuncBySymbol(hipFunction_t *functionPtr, const void *symbolPtr)
try: (hipGetFuncBySymbol:=dll.hipGetFuncBySymbol).restype, hipGetFuncBySymbol.argtypes = hipError_t, [ctypes.POINTER(hipFunction_t), ctypes.c_void_p]
except AttributeError: pass

class textureReference(ctypes.Structure): pass
hipTextureReadMode = CEnum(ctypes.c_uint)
hipReadModeElementType = hipTextureReadMode.define('hipReadModeElementType', 0)
hipReadModeNormalizedFloat = hipTextureReadMode.define('hipReadModeNormalizedFloat', 1)

hipTextureFilterMode = CEnum(ctypes.c_uint)
hipFilterModePoint = hipTextureFilterMode.define('hipFilterModePoint', 0)
hipFilterModeLinear = hipTextureFilterMode.define('hipFilterModeLinear', 1)

hipTextureAddressMode = CEnum(ctypes.c_uint)
hipAddressModeWrap = hipTextureAddressMode.define('hipAddressModeWrap', 0)
hipAddressModeClamp = hipTextureAddressMode.define('hipAddressModeClamp', 1)
hipAddressModeMirror = hipTextureAddressMode.define('hipAddressModeMirror', 2)
hipAddressModeBorder = hipTextureAddressMode.define('hipAddressModeBorder', 3)

class __hip_texture(ctypes.Structure): pass
__hip_texture._fields_ = []
hipTextureObject_t = ctypes.POINTER(__hip_texture)
textureReference._fields_ = [
  ('normalized', ctypes.c_int),
  ('readMode', hipTextureReadMode),
  ('filterMode', hipTextureFilterMode),
  ('addressMode', (hipTextureAddressMode * 3)),
  ('channelDesc', hipChannelFormatDesc),
  ('sRGB', ctypes.c_int),
  ('maxAnisotropy', ctypes.c_uint),
  ('mipmapFilterMode', hipTextureFilterMode),
  ('mipmapLevelBias', ctypes.c_float),
  ('minMipmapLevelClamp', ctypes.c_float),
  ('maxMipmapLevelClamp', ctypes.c_float),
  ('textureObject', hipTextureObject_t),
  ('numChannels', ctypes.c_int),
  ('format', hipArray_Format),
]
# hipError_t hipModuleGetTexRef(textureReference **texRef, hipModule_t hmod, const char *name)
try: (hipModuleGetTexRef:=dll.hipModuleGetTexRef).restype, hipModuleGetTexRef.argtypes = hipError_t, [ctypes.POINTER(ctypes.POINTER(textureReference)), hipModule_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# hipError_t hipModuleLoadData(hipModule_t *module, const void *image)
try: (hipModuleLoadData:=dll.hipModuleLoadData).restype, hipModuleLoadData.argtypes = hipError_t, [ctypes.POINTER(hipModule_t), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image, unsigned int numOptions, hipJitOption *options, void **optionValues)
try: (hipModuleLoadDataEx:=dll.hipModuleLoadDataEx).restype, hipModuleLoadDataEx.argtypes = hipError_t, [ctypes.POINTER(hipModule_t), ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(hipJitOption), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void **kernelParams, void **extra)
try: (hipModuleLaunchKernel:=dll.hipModuleLaunchKernel).restype, hipModuleLaunchKernel.argtypes = hipError_t, [hipFunction_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, hipStream_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void **kernelParams)
try: (hipModuleLaunchCooperativeKernel:=dll.hipModuleLaunchCooperativeKernel).restype, hipModuleLaunchCooperativeKernel.argtypes = hipError_t, [hipFunction_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, hipStream_t, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags)
try: (hipModuleLaunchCooperativeKernelMultiDevice:=dll.hipModuleLaunchCooperativeKernelMultiDevice).restype, hipModuleLaunchCooperativeKernelMultiDevice.argtypes = hipError_t, [ctypes.POINTER(hipFunctionLaunchParams), ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipLaunchCooperativeKernel(const void *f, dim3 gridDim, dim3 blockDimX, void **kernelParams, unsigned int sharedMemBytes, hipStream_t stream)
try: (hipLaunchCooperativeKernel:=dll.hipLaunchCooperativeKernel).restype, hipLaunchCooperativeKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, hipStream_t]
except AttributeError: pass

# hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams *launchParamsList, int numDevices, unsigned int flags)
try: (hipLaunchCooperativeKernelMultiDevice:=dll.hipLaunchCooperativeKernelMultiDevice).restype, hipLaunchCooperativeKernelMultiDevice.argtypes = hipError_t, [ctypes.POINTER(hipLaunchParams), ctypes.c_int, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams *launchParamsList, int numDevices, unsigned int flags)
try: (hipExtLaunchMultiKernelMultiDevice:=dll.hipExtLaunchMultiKernelMultiDevice).restype, hipExtLaunchMultiKernelMultiDevice.argtypes = hipError_t, [ctypes.POINTER(hipLaunchParams), ctypes.c_int, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipModuleOccupancyMaxPotentialBlockSize(int *gridSize, int *blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit)
try: (hipModuleOccupancyMaxPotentialBlockSize:=dll.hipModuleOccupancyMaxPotentialBlockSize).restype, hipModuleOccupancyMaxPotentialBlockSize.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), hipFunction_t, size_t, ctypes.c_int]
except AttributeError: pass

# hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int *gridSize, int *blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags)
try: (hipModuleOccupancyMaxPotentialBlockSizeWithFlags:=dll.hipModuleOccupancyMaxPotentialBlockSizeWithFlags).restype, hipModuleOccupancyMaxPotentialBlockSizeWithFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), hipFunction_t, size_t, ctypes.c_int, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk)
try: (hipModuleOccupancyMaxActiveBlocksPerMultiprocessor:=dll.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor).restype, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), hipFunction_t, ctypes.c_int, size_t]
except AttributeError: pass

# hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags)
try: (hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:=dll.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags).restype, hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), hipFunction_t, ctypes.c_int, size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *f, int blockSize, size_t dynSharedMemPerBlk)
try: (hipOccupancyMaxActiveBlocksPerMultiprocessor:=dll.hipOccupancyMaxActiveBlocksPerMultiprocessor).restype, hipOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, size_t]
except AttributeError: pass

# hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags = 0)
try: (hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:=dll.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags).restype, hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipOccupancyMaxPotentialBlockSize(int *gridSize, int *blockSize, const void *f, size_t dynSharedMemPerBlk, int blockSizeLimit)
try: (hipOccupancyMaxPotentialBlockSize:=dll.hipOccupancyMaxPotentialBlockSize).restype, hipOccupancyMaxPotentialBlockSize.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, size_t, ctypes.c_int]
except AttributeError: pass

# __attribute__((deprecated("use roctracer/rocTX instead"))) hipError_t hipProfilerStart()
try: (hipProfilerStart:=dll.hipProfilerStart).restype, hipProfilerStart.argtypes = hipError_t, []
except AttributeError: pass

# __attribute__((deprecated("use roctracer/rocTX instead"))) hipError_t hipProfilerStop()
try: (hipProfilerStop:=dll.hipProfilerStop).restype, hipProfilerStop.argtypes = hipError_t, []
except AttributeError: pass

# hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, hipStream_t stream = 0)
try: (hipConfigureCall:=dll.hipConfigureCall).restype, hipConfigureCall.argtypes = hipError_t, [dim3, dim3, size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipSetupArgument(const void *arg, size_t size, size_t offset)
try: (hipSetupArgument:=dll.hipSetupArgument).restype, hipSetupArgument.argtypes = hipError_t, [ctypes.c_void_p, size_t, size_t]
except AttributeError: pass

# hipError_t hipLaunchByPtr(const void *func)
try: (hipLaunchByPtr:=dll.hipLaunchByPtr).restype, hipLaunchByPtr.argtypes = hipError_t, [ctypes.c_void_p]
except AttributeError: pass

# hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, hipStream_t stream = 0)
try: (__hipPushCallConfiguration:=dll.__hipPushCallConfiguration).restype, __hipPushCallConfiguration.argtypes = hipError_t, [dim3, dim3, size_t, hipStream_t]
except AttributeError: pass

# hipError_t __hipPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, hipStream_t *stream)
try: (__hipPopCallConfiguration:=dll.__hipPopCallConfiguration).restype, __hipPopCallConfiguration.argtypes = hipError_t, [ctypes.POINTER(dim3), ctypes.POINTER(dim3), ctypes.POINTER(size_t), ctypes.POINTER(hipStream_t)]
except AttributeError: pass

# hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes = 0, hipStream_t stream = 0)
try: (hipLaunchKernel:=dll.hipLaunchKernel).restype, hipLaunchKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t]
except AttributeError: pass

# hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void *userData)
try: (hipLaunchHostFunc:=dll.hipLaunchHostFunc).restype, hipLaunchHostFunc.argtypes = hipError_t, [hipStream_t, hipHostFn_t, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D *pCopy)
try: (hipDrvMemcpy2DUnaligned:=dll.hipDrvMemcpy2DUnaligned).restype, hipDrvMemcpy2DUnaligned.argtypes = hipError_t, [ctypes.POINTER(hip_Memcpy2D)]
except AttributeError: pass

# hipError_t hipExtLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes, hipStream_t stream, hipEvent_t startEvent, hipEvent_t stopEvent, int flags)
try: (hipExtLaunchKernel:=dll.hipExtLaunchKernel).restype, hipExtLaunchKernel.argtypes = hipError_t, [ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int]
except AttributeError: pass

class hipResourceDesc(ctypes.Structure): pass
class hipResourceDesc_res(ctypes.Union): pass
class hipResourceDesc_res_array(ctypes.Structure): pass
hipResourceDesc_res_array._fields_ = [
  ('array', hipArray_t),
]
class hipResourceDesc_res_mipmap(ctypes.Structure): pass
hipResourceDesc_res_mipmap._fields_ = [
  ('mipmap', hipMipmappedArray_t),
]
class hipResourceDesc_res_linear(ctypes.Structure): pass
hipResourceDesc_res_linear._fields_ = [
  ('devPtr', ctypes.c_void_p),
  ('desc', hipChannelFormatDesc),
  ('sizeInBytes', size_t),
]
class hipResourceDesc_res_pitch2D(ctypes.Structure): pass
hipResourceDesc_res_pitch2D._fields_ = [
  ('devPtr', ctypes.c_void_p),
  ('desc', hipChannelFormatDesc),
  ('width', size_t),
  ('height', size_t),
  ('pitchInBytes', size_t),
]
hipResourceDesc_res._fields_ = [
  ('array', hipResourceDesc_res_array),
  ('mipmap', hipResourceDesc_res_mipmap),
  ('linear', hipResourceDesc_res_linear),
  ('pitch2D', hipResourceDesc_res_pitch2D),
]
hipResourceDesc._fields_ = [
  ('resType', hipResourceType),
  ('res', hipResourceDesc_res),
]
class hipTextureDesc(ctypes.Structure): pass
hipTextureDesc._fields_ = [
  ('addressMode', (hipTextureAddressMode * 3)),
  ('filterMode', hipTextureFilterMode),
  ('readMode', hipTextureReadMode),
  ('sRGB', ctypes.c_int),
  ('borderColor', (ctypes.c_float * 4)),
  ('normalizedCoords', ctypes.c_int),
  ('maxAnisotropy', ctypes.c_uint),
  ('mipmapFilterMode', hipTextureFilterMode),
  ('mipmapLevelBias', ctypes.c_float),
  ('minMipmapLevelClamp', ctypes.c_float),
  ('maxMipmapLevelClamp', ctypes.c_float),
]
class hipResourceViewDesc(ctypes.Structure): pass
hipResourceViewFormat = CEnum(ctypes.c_uint)
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

hipResourceViewDesc._fields_ = [
  ('format', hipResourceViewFormat),
  ('width', size_t),
  ('height', size_t),
  ('depth', size_t),
  ('firstMipmapLevel', ctypes.c_uint),
  ('lastMipmapLevel', ctypes.c_uint),
  ('firstLayer', ctypes.c_uint),
  ('lastLayer', ctypes.c_uint),
]
# hipError_t hipCreateTextureObject(hipTextureObject_t *pTexObject, const hipResourceDesc *pResDesc, const hipTextureDesc *pTexDesc, const struct hipResourceViewDesc *pResViewDesc)
try: (hipCreateTextureObject:=dll.hipCreateTextureObject).restype, hipCreateTextureObject.argtypes = hipError_t, [ctypes.POINTER(hipTextureObject_t), ctypes.POINTER(hipResourceDesc), ctypes.POINTER(hipTextureDesc), ctypes.POINTER(hipResourceViewDesc)]
except AttributeError: pass

# hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject)
try: (hipDestroyTextureObject:=dll.hipDestroyTextureObject).restype, hipDestroyTextureObject.argtypes = hipError_t, [hipTextureObject_t]
except AttributeError: pass

# hipError_t hipGetChannelDesc(hipChannelFormatDesc *desc, hipArray_const_t array)
try: (hipGetChannelDesc:=dll.hipGetChannelDesc).restype, hipGetChannelDesc.argtypes = hipError_t, [ctypes.POINTER(hipChannelFormatDesc), hipArray_const_t]
except AttributeError: pass

# hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc *pResDesc, hipTextureObject_t textureObject)
try: (hipGetTextureObjectResourceDesc:=dll.hipGetTextureObjectResourceDesc).restype, hipGetTextureObjectResourceDesc.argtypes = hipError_t, [ctypes.POINTER(hipResourceDesc), hipTextureObject_t]
except AttributeError: pass

# hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc *pResViewDesc, hipTextureObject_t textureObject)
try: (hipGetTextureObjectResourceViewDesc:=dll.hipGetTextureObjectResourceViewDesc).restype, hipGetTextureObjectResourceViewDesc.argtypes = hipError_t, [ctypes.POINTER(hipResourceViewDesc), hipTextureObject_t]
except AttributeError: pass

# hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc *pTexDesc, hipTextureObject_t textureObject)
try: (hipGetTextureObjectTextureDesc:=dll.hipGetTextureObjectTextureDesc).restype, hipGetTextureObjectTextureDesc.argtypes = hipError_t, [ctypes.POINTER(hipTextureDesc), hipTextureObject_t]
except AttributeError: pass

class HIP_RESOURCE_DESC_st(ctypes.Structure): pass
HIPresourcetype_enum = CEnum(ctypes.c_uint)
HIP_RESOURCE_TYPE_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_ARRAY', 0)
HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
HIP_RESOURCE_TYPE_LINEAR = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_LINEAR', 2)
HIP_RESOURCE_TYPE_PITCH2D = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_PITCH2D', 3)

HIPresourcetype = HIPresourcetype_enum
class HIP_RESOURCE_DESC_st_res(ctypes.Union): pass
class HIP_RESOURCE_DESC_st_res_array(ctypes.Structure): pass
HIP_RESOURCE_DESC_st_res_array._fields_ = [
  ('hArray', hipArray_t),
]
class HIP_RESOURCE_DESC_st_res_mipmap(ctypes.Structure): pass
HIP_RESOURCE_DESC_st_res_mipmap._fields_ = [
  ('hMipmappedArray', hipMipmappedArray_t),
]
class HIP_RESOURCE_DESC_st_res_linear(ctypes.Structure): pass
HIP_RESOURCE_DESC_st_res_linear._fields_ = [
  ('devPtr', hipDeviceptr_t),
  ('format', hipArray_Format),
  ('numChannels', ctypes.c_uint),
  ('sizeInBytes', size_t),
]
class HIP_RESOURCE_DESC_st_res_pitch2D(ctypes.Structure): pass
HIP_RESOURCE_DESC_st_res_pitch2D._fields_ = [
  ('devPtr', hipDeviceptr_t),
  ('format', hipArray_Format),
  ('numChannels', ctypes.c_uint),
  ('width', size_t),
  ('height', size_t),
  ('pitchInBytes', size_t),
]
class HIP_RESOURCE_DESC_st_res_reserved(ctypes.Structure): pass
HIP_RESOURCE_DESC_st_res_reserved._fields_ = [
  ('reserved', (ctypes.c_int * 32)),
]
HIP_RESOURCE_DESC_st_res._fields_ = [
  ('array', HIP_RESOURCE_DESC_st_res_array),
  ('mipmap', HIP_RESOURCE_DESC_st_res_mipmap),
  ('linear', HIP_RESOURCE_DESC_st_res_linear),
  ('pitch2D', HIP_RESOURCE_DESC_st_res_pitch2D),
  ('reserved', HIP_RESOURCE_DESC_st_res_reserved),
]
HIP_RESOURCE_DESC_st._fields_ = [
  ('resType', HIPresourcetype),
  ('res', HIP_RESOURCE_DESC_st_res),
  ('flags', ctypes.c_uint),
]
HIP_RESOURCE_DESC = HIP_RESOURCE_DESC_st
class HIP_TEXTURE_DESC_st(ctypes.Structure): pass
HIPaddress_mode_enum = CEnum(ctypes.c_uint)
HIP_TR_ADDRESS_MODE_WRAP = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_WRAP', 0)
HIP_TR_ADDRESS_MODE_CLAMP = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_CLAMP', 1)
HIP_TR_ADDRESS_MODE_MIRROR = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_MIRROR', 2)
HIP_TR_ADDRESS_MODE_BORDER = HIPaddress_mode_enum.define('HIP_TR_ADDRESS_MODE_BORDER', 3)

HIPaddress_mode = HIPaddress_mode_enum
HIPfilter_mode_enum = CEnum(ctypes.c_uint)
HIP_TR_FILTER_MODE_POINT = HIPfilter_mode_enum.define('HIP_TR_FILTER_MODE_POINT', 0)
HIP_TR_FILTER_MODE_LINEAR = HIPfilter_mode_enum.define('HIP_TR_FILTER_MODE_LINEAR', 1)

HIPfilter_mode = HIPfilter_mode_enum
HIP_TEXTURE_DESC_st._fields_ = [
  ('addressMode', (HIPaddress_mode * 3)),
  ('filterMode', HIPfilter_mode),
  ('flags', ctypes.c_uint),
  ('maxAnisotropy', ctypes.c_uint),
  ('mipmapFilterMode', HIPfilter_mode),
  ('mipmapLevelBias', ctypes.c_float),
  ('minMipmapLevelClamp', ctypes.c_float),
  ('maxMipmapLevelClamp', ctypes.c_float),
  ('borderColor', (ctypes.c_float * 4)),
  ('reserved', (ctypes.c_int * 12)),
]
HIP_TEXTURE_DESC = HIP_TEXTURE_DESC_st
class HIP_RESOURCE_VIEW_DESC_st(ctypes.Structure): pass
HIPresourceViewFormat_enum = CEnum(ctypes.c_uint)
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
HIP_RESOURCE_VIEW_DESC_st._fields_ = [
  ('format', HIPresourceViewFormat),
  ('width', size_t),
  ('height', size_t),
  ('depth', size_t),
  ('firstMipmapLevel', ctypes.c_uint),
  ('lastMipmapLevel', ctypes.c_uint),
  ('firstLayer', ctypes.c_uint),
  ('lastLayer', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
HIP_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC_st
# hipError_t hipTexObjectCreate(hipTextureObject_t *pTexObject, const HIP_RESOURCE_DESC *pResDesc, const HIP_TEXTURE_DESC *pTexDesc, const HIP_RESOURCE_VIEW_DESC *pResViewDesc)
try: (hipTexObjectCreate:=dll.hipTexObjectCreate).restype, hipTexObjectCreate.argtypes = hipError_t, [ctypes.POINTER(hipTextureObject_t), ctypes.POINTER(HIP_RESOURCE_DESC), ctypes.POINTER(HIP_TEXTURE_DESC), ctypes.POINTER(HIP_RESOURCE_VIEW_DESC)]
except AttributeError: pass

# hipError_t hipTexObjectDestroy(hipTextureObject_t texObject)
try: (hipTexObjectDestroy:=dll.hipTexObjectDestroy).restype, hipTexObjectDestroy.argtypes = hipError_t, [hipTextureObject_t]
except AttributeError: pass

# hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC *pResDesc, hipTextureObject_t texObject)
try: (hipTexObjectGetResourceDesc:=dll.hipTexObjectGetResourceDesc).restype, hipTexObjectGetResourceDesc.argtypes = hipError_t, [ctypes.POINTER(HIP_RESOURCE_DESC), hipTextureObject_t]
except AttributeError: pass

# hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC *pResViewDesc, hipTextureObject_t texObject)
try: (hipTexObjectGetResourceViewDesc:=dll.hipTexObjectGetResourceViewDesc).restype, hipTexObjectGetResourceViewDesc.argtypes = hipError_t, [ctypes.POINTER(HIP_RESOURCE_VIEW_DESC), hipTextureObject_t]
except AttributeError: pass

# hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC *pTexDesc, hipTextureObject_t texObject)
try: (hipTexObjectGetTextureDesc:=dll.hipTexObjectGetTextureDesc).restype, hipTexObjectGetTextureDesc.argtypes = hipError_t, [ctypes.POINTER(HIP_TEXTURE_DESC), hipTextureObject_t]
except AttributeError: pass

# hipError_t hipMallocMipmappedArray(hipMipmappedArray_t *mipmappedArray, const struct hipChannelFormatDesc *desc, struct hipExtent extent, unsigned int numLevels, unsigned int flags = 0)
try: (hipMallocMipmappedArray:=dll.hipMallocMipmappedArray).restype, hipMallocMipmappedArray.argtypes = hipError_t, [ctypes.POINTER(hipMipmappedArray_t), ctypes.POINTER(hipChannelFormatDesc), hipExtent, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray)
try: (hipFreeMipmappedArray:=dll.hipFreeMipmappedArray).restype, hipFreeMipmappedArray.argtypes = hipError_t, [hipMipmappedArray_t]
except AttributeError: pass

class const_hipMipmappedArray(ctypes.Structure): pass
const_hipMipmappedArray._fields_ = [
  ('data', ctypes.c_void_p),
  ('desc', hipChannelFormatDesc),
  ('type', ctypes.c_uint),
  ('width', ctypes.c_uint),
  ('height', ctypes.c_uint),
  ('depth', ctypes.c_uint),
  ('min_mipmap_level', ctypes.c_uint),
  ('max_mipmap_level', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('format', hipArray_Format),
  ('num_channels', ctypes.c_uint),
]
hipMipmappedArray_const_t = ctypes.POINTER(const_hipMipmappedArray)
# hipError_t hipGetMipmappedArrayLevel(hipArray_t *levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level)
try: (hipGetMipmappedArrayLevel:=dll.hipGetMipmappedArrayLevel).restype, hipGetMipmappedArrayLevel.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), hipMipmappedArray_const_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t *pHandle, HIP_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels)
try: (hipMipmappedArrayCreate:=dll.hipMipmappedArrayCreate).restype, hipMipmappedArrayCreate.argtypes = hipError_t, [ctypes.POINTER(hipMipmappedArray_t), ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR), ctypes.c_uint]
except AttributeError: pass

# hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray)
try: (hipMipmappedArrayDestroy:=dll.hipMipmappedArrayDestroy).restype, hipMipmappedArrayDestroy.argtypes = hipError_t, [hipMipmappedArray_t]
except AttributeError: pass

# hipError_t hipMipmappedArrayGetLevel(hipArray_t *pLevelArray, hipMipmappedArray_t hMipMappedArray, unsigned int level)
try: (hipMipmappedArrayGetLevel:=dll.hipMipmappedArrayGetLevel).restype, hipMipmappedArrayGetLevel.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), hipMipmappedArray_t, ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipBindTextureToMipmappedArray(const textureReference *tex, hipMipmappedArray_const_t mipmappedArray, const hipChannelFormatDesc *desc)
try: (hipBindTextureToMipmappedArray:=dll.hipBindTextureToMipmappedArray).restype, hipBindTextureToMipmappedArray.argtypes = hipError_t, [ctypes.POINTER(textureReference), hipMipmappedArray_const_t, ctypes.POINTER(hipChannelFormatDesc)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipGetTextureReference(const textureReference **texref, const void *symbol)
try: (hipGetTextureReference:=dll.hipGetTextureReference).restype, hipGetTextureReference.argtypes = hipError_t, [ctypes.POINTER(ctypes.POINTER(textureReference)), ctypes.c_void_p]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetBorderColor(float *pBorderColor, const textureReference *texRef)
try: (hipTexRefGetBorderColor:=dll.hipTexRefGetBorderColor).restype, hipTexRefGetBorderColor.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetArray(hipArray_t *pArray, const textureReference *texRef)
try: (hipTexRefGetArray:=dll.hipTexRefGetArray).restype, hipTexRefGetArray.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetAddressMode(textureReference *texRef, int dim, enum hipTextureAddressMode am)
try: (hipTexRefSetAddressMode:=dll.hipTexRefSetAddressMode).restype, hipTexRefSetAddressMode.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.c_int, hipTextureAddressMode]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetArray(textureReference *tex, hipArray_const_t array, unsigned int flags)
try: (hipTexRefSetArray:=dll.hipTexRefSetArray).restype, hipTexRefSetArray.argtypes = hipError_t, [ctypes.POINTER(textureReference), hipArray_const_t, ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetFilterMode(textureReference *texRef, enum hipTextureFilterMode fm)
try: (hipTexRefSetFilterMode:=dll.hipTexRefSetFilterMode).restype, hipTexRefSetFilterMode.argtypes = hipError_t, [ctypes.POINTER(textureReference), hipTextureFilterMode]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetFlags(textureReference *texRef, unsigned int Flags)
try: (hipTexRefSetFlags:=dll.hipTexRefSetFlags).restype, hipTexRefSetFlags.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetFormat(textureReference *texRef, hipArray_Format fmt, int NumPackedComponents)
try: (hipTexRefSetFormat:=dll.hipTexRefSetFormat).restype, hipTexRefSetFormat.argtypes = hipError_t, [ctypes.POINTER(textureReference), hipArray_Format, ctypes.c_int]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipBindTexture(size_t *offset, const textureReference *tex, const void *devPtr, const hipChannelFormatDesc *desc, size_t size = (2147483647 * 2U + 1U))
try: (hipBindTexture:=dll.hipBindTexture).restype, hipBindTexture.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.POINTER(textureReference), ctypes.c_void_p, ctypes.POINTER(hipChannelFormatDesc), size_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipBindTexture2D(size_t *offset, const textureReference *tex, const void *devPtr, const hipChannelFormatDesc *desc, size_t width, size_t height, size_t pitch)
try: (hipBindTexture2D:=dll.hipBindTexture2D).restype, hipBindTexture2D.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.POINTER(textureReference), ctypes.c_void_p, ctypes.POINTER(hipChannelFormatDesc), size_t, size_t, size_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipBindTextureToArray(const textureReference *tex, hipArray_const_t array, const hipChannelFormatDesc *desc)
try: (hipBindTextureToArray:=dll.hipBindTextureToArray).restype, hipBindTextureToArray.argtypes = hipError_t, [ctypes.POINTER(textureReference), hipArray_const_t, ctypes.POINTER(hipChannelFormatDesc)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipGetTextureAlignmentOffset(size_t *offset, const textureReference *texref)
try: (hipGetTextureAlignmentOffset:=dll.hipGetTextureAlignmentOffset).restype, hipGetTextureAlignmentOffset.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipUnbindTexture(const textureReference *tex)
try: (hipUnbindTexture:=dll.hipUnbindTexture).restype, hipUnbindTexture.argtypes = hipError_t, [ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetAddress(hipDeviceptr_t *dev_ptr, const textureReference *texRef)
try: (hipTexRefGetAddress:=dll.hipTexRefGetAddress).restype, hipTexRefGetAddress.argtypes = hipError_t, [ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode *pam, const textureReference *texRef, int dim)
try: (hipTexRefGetAddressMode:=dll.hipTexRefGetAddressMode).restype, hipTexRefGetAddressMode.argtypes = hipError_t, [ctypes.POINTER(hipTextureAddressMode), ctypes.POINTER(textureReference), ctypes.c_int]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode *pfm, const textureReference *texRef)
try: (hipTexRefGetFilterMode:=dll.hipTexRefGetFilterMode).restype, hipTexRefGetFilterMode.argtypes = hipError_t, [ctypes.POINTER(hipTextureFilterMode), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetFlags(unsigned int *pFlags, const textureReference *texRef)
try: (hipTexRefGetFlags:=dll.hipTexRefGetFlags).restype, hipTexRefGetFlags.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetFormat(hipArray_Format *pFormat, int *pNumChannels, const textureReference *texRef)
try: (hipTexRefGetFormat:=dll.hipTexRefGetFormat).restype, hipTexRefGetFormat.argtypes = hipError_t, [ctypes.POINTER(hipArray_Format), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetMaxAnisotropy(int *pmaxAnsio, const textureReference *texRef)
try: (hipTexRefGetMaxAnisotropy:=dll.hipTexRefGetMaxAnisotropy).restype, hipTexRefGetMaxAnisotropy.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode *pfm, const textureReference *texRef)
try: (hipTexRefGetMipmapFilterMode:=dll.hipTexRefGetMipmapFilterMode).restype, hipTexRefGetMipmapFilterMode.argtypes = hipError_t, [ctypes.POINTER(hipTextureFilterMode), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetMipmapLevelBias(float *pbias, const textureReference *texRef)
try: (hipTexRefGetMipmapLevelBias:=dll.hipTexRefGetMipmapLevelBias).restype, hipTexRefGetMipmapLevelBias.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, const textureReference *texRef)
try: (hipTexRefGetMipmapLevelClamp:=dll.hipTexRefGetMipmapLevelClamp).restype, hipTexRefGetMipmapLevelClamp.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t *pArray, const textureReference *texRef)
try: (hipTexRefGetMipMappedArray:=dll.hipTexRefGetMipMappedArray).restype, hipTexRefGetMipMappedArray.argtypes = hipError_t, [ctypes.POINTER(hipMipmappedArray_t), ctypes.POINTER(textureReference)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetAddress(size_t *ByteOffset, textureReference *texRef, hipDeviceptr_t dptr, size_t bytes)
try: (hipTexRefSetAddress:=dll.hipTexRefSetAddress).restype, hipTexRefSetAddress.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.POINTER(textureReference), hipDeviceptr_t, size_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetAddress2D(textureReference *texRef, const HIP_ARRAY_DESCRIPTOR *desc, hipDeviceptr_t dptr, size_t Pitch)
try: (hipTexRefSetAddress2D:=dll.hipTexRefSetAddress2D).restype, hipTexRefSetAddress2D.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.POINTER(HIP_ARRAY_DESCRIPTOR), hipDeviceptr_t, size_t]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetMaxAnisotropy(textureReference *texRef, unsigned int maxAniso)
try: (hipTexRefSetMaxAnisotropy:=dll.hipTexRefSetMaxAnisotropy).restype, hipTexRefSetMaxAnisotropy.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetBorderColor(textureReference *texRef, float *pBorderColor)
try: (hipTexRefSetBorderColor:=dll.hipTexRefSetBorderColor).restype, hipTexRefSetBorderColor.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.POINTER(ctypes.c_float)]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetMipmapFilterMode(textureReference *texRef, enum hipTextureFilterMode fm)
try: (hipTexRefSetMipmapFilterMode:=dll.hipTexRefSetMipmapFilterMode).restype, hipTexRefSetMipmapFilterMode.argtypes = hipError_t, [ctypes.POINTER(textureReference), hipTextureFilterMode]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetMipmapLevelBias(textureReference *texRef, float bias)
try: (hipTexRefSetMipmapLevelBias:=dll.hipTexRefSetMipmapLevelBias).restype, hipTexRefSetMipmapLevelBias.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.c_float]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetMipmapLevelClamp(textureReference *texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp)
try: (hipTexRefSetMipmapLevelClamp:=dll.hipTexRefSetMipmapLevelClamp).restype, hipTexRefSetMipmapLevelClamp.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.c_float, ctypes.c_float]
except AttributeError: pass

# __attribute__((deprecated("This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"))) hipError_t hipTexRefSetMipmappedArray(textureReference *texRef, struct hipMipmappedArray *mipmappedArray, unsigned int Flags)
try: (hipTexRefSetMipmappedArray:=dll.hipTexRefSetMipmappedArray).restype, hipTexRefSetMipmappedArray.argtypes = hipError_t, [ctypes.POINTER(textureReference), ctypes.POINTER(hipMipmappedArray), ctypes.c_uint]
except AttributeError: pass

# const char *hipApiName(uint32_t id)
try: (hipApiName:=dll.hipApiName).restype, hipApiName.argtypes = ctypes.POINTER(ctypes.c_char), [uint32_t]
except AttributeError: pass

# const char *hipKernelNameRef(const hipFunction_t f)
try: (hipKernelNameRef:=dll.hipKernelNameRef).restype, hipKernelNameRef.argtypes = ctypes.POINTER(ctypes.c_char), [hipFunction_t]
except AttributeError: pass

# const char *hipKernelNameRefByPtr(const void *hostFunction, hipStream_t stream)
try: (hipKernelNameRefByPtr:=dll.hipKernelNameRefByPtr).restype, hipKernelNameRefByPtr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, hipStream_t]
except AttributeError: pass

# int hipGetStreamDeviceId(hipStream_t stream)
try: (hipGetStreamDeviceId:=dll.hipGetStreamDeviceId).restype, hipGetStreamDeviceId.argtypes = ctypes.c_int, [hipStream_t]
except AttributeError: pass

# hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode)
try: (hipStreamBeginCapture:=dll.hipStreamBeginCapture).restype, hipStreamBeginCapture.argtypes = hipError_t, [hipStream_t, hipStreamCaptureMode]
except AttributeError: pass

# hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph, const hipGraphNode_t *dependencies, const hipGraphEdgeData *dependencyData, size_t numDependencies, hipStreamCaptureMode mode)
try: (hipStreamBeginCaptureToGraph:=dll.hipStreamBeginCaptureToGraph).restype, hipStreamBeginCaptureToGraph.argtypes = hipError_t, [hipStream_t, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphEdgeData), size_t, hipStreamCaptureMode]
except AttributeError: pass

# hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t *pGraph)
try: (hipStreamEndCapture:=dll.hipStreamEndCapture).restype, hipStreamEndCapture.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(hipGraph_t)]
except AttributeError: pass

# hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus *pCaptureStatus, unsigned long long *pId)
try: (hipStreamGetCaptureInfo:=dll.hipStreamGetCaptureInfo).restype, hipStreamGetCaptureInfo.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(hipStreamCaptureStatus), ctypes.POINTER(ctypes.c_ulonglong)]
except AttributeError: pass

# hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus *captureStatus_out, unsigned long long *id_out = 0, hipGraph_t *graph_out = 0, const hipGraphNode_t **dependencies_out = 0, size_t *numDependencies_out = 0)
try: (hipStreamGetCaptureInfo_v2:=dll.hipStreamGetCaptureInfo_v2).restype, hipStreamGetCaptureInfo_v2.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(hipStreamCaptureStatus), ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(hipGraph_t), ctypes.POINTER(ctypes.POINTER(hipGraphNode_t)), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus *pCaptureStatus)
try: (hipStreamIsCapturing:=dll.hipStreamIsCapturing).restype, hipStreamIsCapturing.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(hipStreamCaptureStatus)]
except AttributeError: pass

# hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t *dependencies, size_t numDependencies, unsigned int flags = 0)
try: (hipStreamUpdateCaptureDependencies:=dll.hipStreamUpdateCaptureDependencies).restype, hipStreamUpdateCaptureDependencies.argtypes = hipError_t, [hipStream_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode *mode)
try: (hipThreadExchangeStreamCaptureMode:=dll.hipThreadExchangeStreamCaptureMode).restype, hipThreadExchangeStreamCaptureMode.argtypes = hipError_t, [ctypes.POINTER(hipStreamCaptureMode)]
except AttributeError: pass

# hipError_t hipGraphCreate(hipGraph_t *pGraph, unsigned int flags)
try: (hipGraphCreate:=dll.hipGraphCreate).restype, hipGraphCreate.argtypes = hipError_t, [ctypes.POINTER(hipGraph_t), ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphDestroy(hipGraph_t graph)
try: (hipGraphDestroy:=dll.hipGraphDestroy).restype, hipGraphDestroy.argtypes = hipError_t, [hipGraph_t]
except AttributeError: pass

# hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t *from, const hipGraphNode_t *to, size_t numDependencies)
try: (hipGraphAddDependencies:=dll.hipGraphAddDependencies).restype, hipGraphAddDependencies.argtypes = hipError_t, [hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphNode_t), size_t]
except AttributeError: pass

# hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t *from, const hipGraphNode_t *to, size_t numDependencies)
try: (hipGraphRemoveDependencies:=dll.hipGraphRemoveDependencies).restype, hipGraphRemoveDependencies.argtypes = hipError_t, [hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphNode_t), size_t]
except AttributeError: pass

# hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t *from, hipGraphNode_t *to, size_t *numEdges)
try: (hipGraphGetEdges:=dll.hipGraphGetEdges).restype, hipGraphGetEdges.argtypes = hipError_t, [hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t *nodes, size_t *numNodes)
try: (hipGraphGetNodes:=dll.hipGraphGetNodes).restype, hipGraphGetNodes.argtypes = hipError_t, [hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t *pRootNodes, size_t *pNumRootNodes)
try: (hipGraphGetRootNodes:=dll.hipGraphGetRootNodes).restype, hipGraphGetRootNodes.argtypes = hipError_t, [hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t *pDependencies, size_t *pNumDependencies)
try: (hipGraphNodeGetDependencies:=dll.hipGraphNodeGetDependencies).restype, hipGraphNodeGetDependencies.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t *pDependentNodes, size_t *pNumDependentNodes)
try: (hipGraphNodeGetDependentNodes:=dll.hipGraphNodeGetDependentNodes).restype, hipGraphNodeGetDependentNodes.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType *pType)
try: (hipGraphNodeGetType:=dll.hipGraphNodeGetType).restype, hipGraphNodeGetType.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipGraphNodeType)]
except AttributeError: pass

# hipError_t hipGraphDestroyNode(hipGraphNode_t node)
try: (hipGraphDestroyNode:=dll.hipGraphDestroyNode).restype, hipGraphDestroyNode.argtypes = hipError_t, [hipGraphNode_t]
except AttributeError: pass

# hipError_t hipGraphClone(hipGraph_t *pGraphClone, hipGraph_t originalGraph)
try: (hipGraphClone:=dll.hipGraphClone).restype, hipGraphClone.argtypes = hipError_t, [ctypes.POINTER(hipGraph_t), hipGraph_t]
except AttributeError: pass

# hipError_t hipGraphNodeFindInClone(hipGraphNode_t *pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph)
try: (hipGraphNodeFindInClone:=dll.hipGraphNodeFindInClone).restype, hipGraphNodeFindInClone.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraphNode_t, hipGraph_t]
except AttributeError: pass

# hipError_t hipGraphInstantiate(hipGraphExec_t *pGraphExec, hipGraph_t graph, hipGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize)
try: (hipGraphInstantiate:=dll.hipGraphInstantiate).restype, hipGraphInstantiate.argtypes = hipError_t, [ctypes.POINTER(hipGraphExec_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t *pGraphExec, hipGraph_t graph, unsigned long long flags)
try: (hipGraphInstantiateWithFlags:=dll.hipGraphInstantiateWithFlags).restype, hipGraphInstantiateWithFlags.argtypes = hipError_t, [ctypes.POINTER(hipGraphExec_t), hipGraph_t, ctypes.c_ulonglong]
except AttributeError: pass

# hipError_t hipGraphInstantiateWithParams(hipGraphExec_t *pGraphExec, hipGraph_t graph, hipGraphInstantiateParams *instantiateParams)
try: (hipGraphInstantiateWithParams:=dll.hipGraphInstantiateWithParams).restype, hipGraphInstantiateWithParams.argtypes = hipError_t, [ctypes.POINTER(hipGraphExec_t), hipGraph_t, ctypes.POINTER(hipGraphInstantiateParams)]
except AttributeError: pass

# hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)
try: (hipGraphLaunch:=dll.hipGraphLaunch).restype, hipGraphLaunch.argtypes = hipError_t, [hipGraphExec_t, hipStream_t]
except AttributeError: pass

# hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream)
try: (hipGraphUpload:=dll.hipGraphUpload).restype, hipGraphUpload.argtypes = hipError_t, [hipGraphExec_t, hipStream_t]
except AttributeError: pass

# hipError_t hipGraphAddNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, hipGraphNodeParams *nodeParams)
try: (hipGraphAddNode:=dll.hipGraphAddNode).restype, hipGraphAddNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipGraphNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec)
try: (hipGraphExecDestroy:=dll.hipGraphExecDestroy).restype, hipGraphExecDestroy.argtypes = hipError_t, [hipGraphExec_t]
except AttributeError: pass

# hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t *hErrorNode_out, hipGraphExecUpdateResult *updateResult_out)
try: (hipGraphExecUpdate:=dll.hipGraphExecUpdate).restype, hipGraphExecUpdate.argtypes = hipError_t, [hipGraphExec_t, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphExecUpdateResult)]
except AttributeError: pass

# hipError_t hipGraphAddKernelNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const hipKernelNodeParams *pNodeParams)
try: (hipGraphAddKernelNode:=dll.hipGraphAddKernelNode).restype, hipGraphAddKernelNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipKernelNodeParams)]
except AttributeError: pass

# hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams *pNodeParams)
try: (hipGraphKernelNodeGetParams:=dll.hipGraphKernelNodeGetParams).restype, hipGraphKernelNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipKernelNodeParams)]
except AttributeError: pass

# hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams *pNodeParams)
try: (hipGraphKernelNodeSetParams:=dll.hipGraphKernelNodeSetParams).restype, hipGraphKernelNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipKernelNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams *pNodeParams)
try: (hipGraphExecKernelNodeSetParams:=dll.hipGraphExecKernelNodeSetParams).restype, hipGraphExecKernelNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipKernelNodeParams)]
except AttributeError: pass

# hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t *phGraphNode, hipGraph_t hGraph, const hipGraphNode_t *dependencies, size_t numDependencies, const HIP_MEMCPY3D *copyParams, hipCtx_t ctx)
try: (hipDrvGraphAddMemcpyNode:=dll.hipDrvGraphAddMemcpyNode).restype, hipDrvGraphAddMemcpyNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(HIP_MEMCPY3D), hipCtx_t]
except AttributeError: pass

# hipError_t hipGraphAddMemcpyNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const hipMemcpy3DParms *pCopyParams)
try: (hipGraphAddMemcpyNode:=dll.hipGraphAddMemcpyNode).restype, hipGraphAddMemcpyNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemcpy3DParms)]
except AttributeError: pass

# hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms *pNodeParams)
try: (hipGraphMemcpyNodeGetParams:=dll.hipGraphMemcpyNodeGetParams).restype, hipGraphMemcpyNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipMemcpy3DParms)]
except AttributeError: pass

# hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms *pNodeParams)
try: (hipGraphMemcpyNodeSetParams:=dll.hipGraphMemcpyNodeSetParams).restype, hipGraphMemcpyNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipMemcpy3DParms)]
except AttributeError: pass

# hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipLaunchAttributeID attr, const hipLaunchAttributeValue *value)
try: (hipGraphKernelNodeSetAttribute:=dll.hipGraphKernelNodeSetAttribute).restype, hipGraphKernelNodeSetAttribute.argtypes = hipError_t, [hipGraphNode_t, hipLaunchAttributeID, ctypes.POINTER(hipLaunchAttributeValue)]
except AttributeError: pass

# hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipLaunchAttributeID attr, hipLaunchAttributeValue *value)
try: (hipGraphKernelNodeGetAttribute:=dll.hipGraphKernelNodeGetAttribute).restype, hipGraphKernelNodeGetAttribute.argtypes = hipError_t, [hipGraphNode_t, hipLaunchAttributeID, ctypes.POINTER(hipLaunchAttributeValue)]
except AttributeError: pass

# hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipMemcpy3DParms *pNodeParams)
try: (hipGraphExecMemcpyNodeSetParams:=dll.hipGraphExecMemcpyNodeSetParams).restype, hipGraphExecMemcpyNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipMemcpy3DParms)]
except AttributeError: pass

# hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *src, size_t count, hipMemcpyKind kind)
try: (hipGraphAddMemcpyNode1D:=dll.hipGraphAddMemcpyNode1D).restype, hipGraphAddMemcpyNode1D.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void *dst, const void *src, size_t count, hipMemcpyKind kind)
try: (hipGraphMemcpyNodeSetParams1D:=dll.hipGraphMemcpyNodeSetParams1D).restype, hipGraphMemcpyNodeSetParams1D.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void *dst, const void *src, size_t count, hipMemcpyKind kind)
try: (hipGraphExecMemcpyNodeSetParams1D:=dll.hipGraphExecMemcpyNodeSetParams1D).restype, hipGraphExecMemcpyNodeSetParams1D.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *symbol, size_t count, size_t offset, hipMemcpyKind kind)
try: (hipGraphAddMemcpyNodeFromSymbol:=dll.hipGraphAddMemcpyNodeFromSymbol).restype, hipGraphAddMemcpyNodeFromSymbol.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, hipMemcpyKind kind)
try: (hipGraphMemcpyNodeSetParamsFromSymbol:=dll.hipGraphMemcpyNodeSetParamsFromSymbol).restype, hipGraphMemcpyNodeSetParamsFromSymbol.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, hipMemcpyKind kind)
try: (hipGraphExecMemcpyNodeSetParamsFromSymbol:=dll.hipGraphExecMemcpyNodeSetParamsFromSymbol).restype, hipGraphExecMemcpyNodeSetParamsFromSymbol.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const void *symbol, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
try: (hipGraphAddMemcpyNodeToSymbol:=dll.hipGraphAddMemcpyNodeToSymbol).restype, hipGraphAddMemcpyNodeToSymbol.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
try: (hipGraphMemcpyNodeSetParamsToSymbol:=dll.hipGraphMemcpyNodeSetParamsToSymbol).restype, hipGraphMemcpyNodeSetParamsToSymbol.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
try: (hipGraphExecMemcpyNodeSetParamsToSymbol:=dll.hipGraphExecMemcpyNodeSetParamsToSymbol).restype, hipGraphExecMemcpyNodeSetParamsToSymbol.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, hipMemcpyKind]
except AttributeError: pass

# hipError_t hipGraphAddMemsetNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const hipMemsetParams *pMemsetParams)
try: (hipGraphAddMemsetNode:=dll.hipGraphAddMemsetNode).restype, hipGraphAddMemsetNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemsetParams)]
except AttributeError: pass

# hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams *pNodeParams)
try: (hipGraphMemsetNodeGetParams:=dll.hipGraphMemsetNodeGetParams).restype, hipGraphMemsetNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipMemsetParams)]
except AttributeError: pass

# hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams *pNodeParams)
try: (hipGraphMemsetNodeSetParams:=dll.hipGraphMemsetNodeSetParams).restype, hipGraphMemsetNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipMemsetParams)]
except AttributeError: pass

# hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipMemsetParams *pNodeParams)
try: (hipGraphExecMemsetNodeSetParams:=dll.hipGraphExecMemsetNodeSetParams).restype, hipGraphExecMemsetNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipMemsetParams)]
except AttributeError: pass

# hipError_t hipGraphAddHostNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const hipHostNodeParams *pNodeParams)
try: (hipGraphAddHostNode:=dll.hipGraphAddHostNode).restype, hipGraphAddHostNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipHostNodeParams)]
except AttributeError: pass

# hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams *pNodeParams)
try: (hipGraphHostNodeGetParams:=dll.hipGraphHostNodeGetParams).restype, hipGraphHostNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipHostNodeParams)]
except AttributeError: pass

# hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams *pNodeParams)
try: (hipGraphHostNodeSetParams:=dll.hipGraphHostNodeSetParams).restype, hipGraphHostNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipHostNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams *pNodeParams)
try: (hipGraphExecHostNodeSetParams:=dll.hipGraphExecHostNodeSetParams).restype, hipGraphExecHostNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipHostNodeParams)]
except AttributeError: pass

# hipError_t hipGraphAddChildGraphNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, hipGraph_t childGraph)
try: (hipGraphAddChildGraphNode:=dll.hipGraphAddChildGraphNode).restype, hipGraphAddChildGraphNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipGraph_t]
except AttributeError: pass

# hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t *pGraph)
try: (hipGraphChildGraphNodeGetGraph:=dll.hipGraphChildGraphNodeGetGraph).restype, hipGraphChildGraphNodeGetGraph.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipGraph_t)]
except AttributeError: pass

# hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph)
try: (hipGraphExecChildGraphNodeSetParams:=dll.hipGraphExecChildGraphNodeSetParams).restype, hipGraphExecChildGraphNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, hipGraph_t]
except AttributeError: pass

# hipError_t hipGraphAddEmptyNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies)
try: (hipGraphAddEmptyNode:=dll.hipGraphAddEmptyNode).restype, hipGraphAddEmptyNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t]
except AttributeError: pass

# hipError_t hipGraphAddEventRecordNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, hipEvent_t event)
try: (hipGraphAddEventRecordNode:=dll.hipGraphAddEventRecordNode).restype, hipGraphAddEventRecordNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t *event_out)
try: (hipGraphEventRecordNodeGetEvent:=dll.hipGraphEventRecordNodeGetEvent).restype, hipGraphEventRecordNodeGetEvent.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipEvent_t)]
except AttributeError: pass

# hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event)
try: (hipGraphEventRecordNodeSetEvent:=dll.hipGraphEventRecordNodeSetEvent).restype, hipGraphEventRecordNodeSetEvent.argtypes = hipError_t, [hipGraphNode_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event)
try: (hipGraphExecEventRecordNodeSetEvent:=dll.hipGraphExecEventRecordNodeSetEvent).restype, hipGraphExecEventRecordNodeSetEvent.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipGraphAddEventWaitNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, hipEvent_t event)
try: (hipGraphAddEventWaitNode:=dll.hipGraphAddEventWaitNode).restype, hipGraphAddEventWaitNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t *event_out)
try: (hipGraphEventWaitNodeGetEvent:=dll.hipGraphEventWaitNodeGetEvent).restype, hipGraphEventWaitNodeGetEvent.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipEvent_t)]
except AttributeError: pass

# hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event)
try: (hipGraphEventWaitNodeSetEvent:=dll.hipGraphEventWaitNodeSetEvent).restype, hipGraphEventWaitNodeSetEvent.argtypes = hipError_t, [hipGraphNode_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event)
try: (hipGraphExecEventWaitNodeSetEvent:=dll.hipGraphExecEventWaitNodeSetEvent).restype, hipGraphExecEventWaitNodeSetEvent.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, hipEvent_t]
except AttributeError: pass

# hipError_t hipGraphAddMemAllocNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, hipMemAllocNodeParams *pNodeParams)
try: (hipGraphAddMemAllocNode:=dll.hipGraphAddMemAllocNode).restype, hipGraphAddMemAllocNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemAllocNodeParams)]
except AttributeError: pass

# hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams *pNodeParams)
try: (hipGraphMemAllocNodeGetParams:=dll.hipGraphMemAllocNodeGetParams).restype, hipGraphMemAllocNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipMemAllocNodeParams)]
except AttributeError: pass

# hipError_t hipGraphAddMemFreeNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, void *dev_ptr)
try: (hipGraphAddMemFreeNode:=dll.hipGraphAddMemFreeNode).restype, hipGraphAddMemFreeNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void *dev_ptr)
try: (hipGraphMemFreeNodeGetParams:=dll.hipGraphMemFreeNodeGetParams).restype, hipGraphMemFreeNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void *value)
try: (hipDeviceGetGraphMemAttribute:=dll.hipDeviceGetGraphMemAttribute).restype, hipDeviceGetGraphMemAttribute.argtypes = hipError_t, [ctypes.c_int, hipGraphMemAttributeType, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void *value)
try: (hipDeviceSetGraphMemAttribute:=dll.hipDeviceSetGraphMemAttribute).restype, hipDeviceSetGraphMemAttribute.argtypes = hipError_t, [ctypes.c_int, hipGraphMemAttributeType, ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipDeviceGraphMemTrim(int device)
try: (hipDeviceGraphMemTrim:=dll.hipDeviceGraphMemTrim).restype, hipDeviceGraphMemTrim.argtypes = hipError_t, [ctypes.c_int]
except AttributeError: pass

# hipError_t hipUserObjectCreate(hipUserObject_t *object_out, void *ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags)
try: (hipUserObjectCreate:=dll.hipUserObjectCreate).restype, hipUserObjectCreate.argtypes = hipError_t, [ctypes.POINTER(hipUserObject_t), ctypes.c_void_p, hipHostFn_t, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count = 1)
try: (hipUserObjectRelease:=dll.hipUserObjectRelease).restype, hipUserObjectRelease.argtypes = hipError_t, [hipUserObject_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count = 1)
try: (hipUserObjectRetain:=dll.hipUserObjectRetain).restype, hipUserObjectRetain.argtypes = hipError_t, [hipUserObject_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count = 1, unsigned int flags = 0)
try: (hipGraphRetainUserObject:=dll.hipGraphRetainUserObject).restype, hipGraphRetainUserObject.argtypes = hipError_t, [hipGraph_t, hipUserObject_t, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count = 1)
try: (hipGraphReleaseUserObject:=dll.hipGraphReleaseUserObject).restype, hipGraphReleaseUserObject.argtypes = hipError_t, [hipGraph_t, hipUserObject_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char *path, unsigned int flags)
try: (hipGraphDebugDotPrint:=dll.hipGraphDebugDotPrint).restype, hipGraphDebugDotPrint.argtypes = hipError_t, [hipGraph_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst)
try: (hipGraphKernelNodeCopyAttributes:=dll.hipGraphKernelNodeCopyAttributes).restype, hipGraphKernelNodeCopyAttributes.argtypes = hipError_t, [hipGraphNode_t, hipGraphNode_t]
except AttributeError: pass

# hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int isEnabled)
try: (hipGraphNodeSetEnabled:=dll.hipGraphNodeSetEnabled).restype, hipGraphNodeSetEnabled.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int *isEnabled)
try: (hipGraphNodeGetEnabled:=dll.hipGraphNodeGetEnabled).restype, hipGraphNodeGetEnabled.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# hipError_t hipGraphAddExternalSemaphoresWaitNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const hipExternalSemaphoreWaitNodeParams *nodeParams)
try: (hipGraphAddExternalSemaphoresWaitNode:=dll.hipGraphAddExternalSemaphoresWaitNode).restype, hipGraphAddExternalSemaphoresWaitNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

# hipError_t hipGraphAddExternalSemaphoresSignalNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t *pDependencies, size_t numDependencies, const hipExternalSemaphoreSignalNodeParams *nodeParams)
try: (hipGraphAddExternalSemaphoresSignalNode:=dll.hipGraphAddExternalSemaphoresSignalNode).restype, hipGraphAddExternalSemaphoresSignalNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams *nodeParams)
try: (hipGraphExternalSemaphoresSignalNodeSetParams:=dll.hipGraphExternalSemaphoresSignalNodeSetParams).restype, hipGraphExternalSemaphoresSignalNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams *nodeParams)
try: (hipGraphExternalSemaphoresWaitNodeSetParams:=dll.hipGraphExternalSemaphoresWaitNodeSetParams).restype, hipGraphExternalSemaphoresWaitNodeSetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams *params_out)
try: (hipGraphExternalSemaphoresSignalNodeGetParams:=dll.hipGraphExternalSemaphoresSignalNodeGetParams).restype, hipGraphExternalSemaphoresSignalNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams *params_out)
try: (hipGraphExternalSemaphoresWaitNodeGetParams:=dll.hipGraphExternalSemaphoresWaitNodeGetParams).restype, hipGraphExternalSemaphoresWaitNodeGetParams.argtypes = hipError_t, [hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams *nodeParams)
try: (hipGraphExecExternalSemaphoresSignalNodeSetParams:=dll.hipGraphExecExternalSemaphoresSignalNodeSetParams).restype, hipGraphExecExternalSemaphoresSignalNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)]
except AttributeError: pass

# hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams *nodeParams)
try: (hipGraphExecExternalSemaphoresWaitNodeSetParams:=dll.hipGraphExecExternalSemaphoresWaitNodeSetParams).restype, hipGraphExecExternalSemaphoresWaitNodeSetParams.argtypes = hipError_t, [hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)]
except AttributeError: pass

# hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t *phGraphNode, hipGraph_t hGraph, const hipGraphNode_t *dependencies, size_t numDependencies, const HIP_MEMSET_NODE_PARAMS *memsetParams, hipCtx_t ctx)
try: (hipDrvGraphAddMemsetNode:=dll.hipDrvGraphAddMemsetNode).restype, hipDrvGraphAddMemsetNode.argtypes = hipError_t, [ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(HIP_MEMSET_NODE_PARAMS), hipCtx_t]
except AttributeError: pass

# hipError_t hipMemAddressFree(void *devPtr, size_t size)
try: (hipMemAddressFree:=dll.hipMemAddressFree).restype, hipMemAddressFree.argtypes = hipError_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

# hipError_t hipMemAddressReserve(void **ptr, size_t size, size_t alignment, void *addr, unsigned long long flags)
try: (hipMemAddressReserve:=dll.hipMemAddressReserve).restype, hipMemAddressReserve.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), size_t, size_t, ctypes.c_void_p, ctypes.c_ulonglong]
except AttributeError: pass

# hipError_t hipMemCreate(hipMemGenericAllocationHandle_t *handle, size_t size, const hipMemAllocationProp *prop, unsigned long long flags)
try: (hipMemCreate:=dll.hipMemCreate).restype, hipMemCreate.argtypes = hipError_t, [ctypes.POINTER(hipMemGenericAllocationHandle_t), size_t, ctypes.POINTER(hipMemAllocationProp), ctypes.c_ulonglong]
except AttributeError: pass

# hipError_t hipMemExportToShareableHandle(void *shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags)
try: (hipMemExportToShareableHandle:=dll.hipMemExportToShareableHandle).restype, hipMemExportToShareableHandle.argtypes = hipError_t, [ctypes.c_void_p, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, ctypes.c_ulonglong]
except AttributeError: pass

# hipError_t hipMemGetAccess(unsigned long long *flags, const hipMemLocation *location, void *ptr)
try: (hipMemGetAccess:=dll.hipMemGetAccess).restype, hipMemGetAccess.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(hipMemLocation), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMemGetAllocationGranularity(size_t *granularity, const hipMemAllocationProp *prop, hipMemAllocationGranularity_flags option)
try: (hipMemGetAllocationGranularity:=dll.hipMemGetAllocationGranularity).restype, hipMemGetAllocationGranularity.argtypes = hipError_t, [ctypes.POINTER(size_t), ctypes.POINTER(hipMemAllocationProp), hipMemAllocationGranularity_flags]
except AttributeError: pass

# hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp *prop, hipMemGenericAllocationHandle_t handle)
try: (hipMemGetAllocationPropertiesFromHandle:=dll.hipMemGetAllocationPropertiesFromHandle).restype, hipMemGetAllocationPropertiesFromHandle.argtypes = hipError_t, [ctypes.POINTER(hipMemAllocationProp), hipMemGenericAllocationHandle_t]
except AttributeError: pass

# hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t *handle, void *osHandle, hipMemAllocationHandleType shHandleType)
try: (hipMemImportFromShareableHandle:=dll.hipMemImportFromShareableHandle).restype, hipMemImportFromShareableHandle.argtypes = hipError_t, [ctypes.POINTER(hipMemGenericAllocationHandle_t), ctypes.c_void_p, hipMemAllocationHandleType]
except AttributeError: pass

# hipError_t hipMemMap(void *ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags)
try: (hipMemMap:=dll.hipMemMap).restype, hipMemMap.argtypes = hipError_t, [ctypes.c_void_p, size_t, size_t, hipMemGenericAllocationHandle_t, ctypes.c_ulonglong]
except AttributeError: pass

# hipError_t hipMemMapArrayAsync(hipArrayMapInfo *mapInfoList, unsigned int count, hipStream_t stream)
try: (hipMemMapArrayAsync:=dll.hipMemMapArrayAsync).restype, hipMemMapArrayAsync.argtypes = hipError_t, [ctypes.POINTER(hipArrayMapInfo), ctypes.c_uint, hipStream_t]
except AttributeError: pass

# hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle)
try: (hipMemRelease:=dll.hipMemRelease).restype, hipMemRelease.argtypes = hipError_t, [hipMemGenericAllocationHandle_t]
except AttributeError: pass

# hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t *handle, void *addr)
try: (hipMemRetainAllocationHandle:=dll.hipMemRetainAllocationHandle).restype, hipMemRetainAllocationHandle.argtypes = hipError_t, [ctypes.POINTER(hipMemGenericAllocationHandle_t), ctypes.c_void_p]
except AttributeError: pass

# hipError_t hipMemSetAccess(void *ptr, size_t size, const hipMemAccessDesc *desc, size_t count)
try: (hipMemSetAccess:=dll.hipMemSetAccess).restype, hipMemSetAccess.argtypes = hipError_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hipMemAccessDesc), size_t]
except AttributeError: pass

# hipError_t hipMemUnmap(void *ptr, size_t size)
try: (hipMemUnmap:=dll.hipMemUnmap).restype, hipMemUnmap.argtypes = hipError_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

# hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t *resources, hipStream_t stream = 0)
try: (hipGraphicsMapResources:=dll.hipGraphicsMapResources).restype, hipGraphicsMapResources.argtypes = hipError_t, [ctypes.c_int, ctypes.POINTER(hipGraphicsResource_t), hipStream_t]
except AttributeError: pass

# hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t *array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
try: (hipGraphicsSubResourceGetMappedArray:=dll.hipGraphicsSubResourceGetMappedArray).restype, hipGraphicsSubResourceGetMappedArray.argtypes = hipError_t, [ctypes.POINTER(hipArray_t), hipGraphicsResource_t, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# hipError_t hipGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, hipGraphicsResource_t resource)
try: (hipGraphicsResourceGetMappedPointer:=dll.hipGraphicsResourceGetMappedPointer).restype, hipGraphicsResourceGetMappedPointer.argtypes = hipError_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), hipGraphicsResource_t]
except AttributeError: pass

# hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t *resources, hipStream_t stream = 0)
try: (hipGraphicsUnmapResources:=dll.hipGraphicsUnmapResources).restype, hipGraphicsUnmapResources.argtypes = hipError_t, [ctypes.c_int, ctypes.POINTER(hipGraphicsResource_t), hipStream_t]
except AttributeError: pass

# hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource)
try: (hipGraphicsUnregisterResource:=dll.hipGraphicsUnregisterResource).restype, hipGraphicsUnregisterResource.argtypes = hipError_t, [hipGraphicsResource_t]
except AttributeError: pass

class __hip_surface(ctypes.Structure): pass
__hip_surface._fields_ = []
hipSurfaceObject_t = ctypes.POINTER(__hip_surface)
# hipError_t hipCreateSurfaceObject(hipSurfaceObject_t *pSurfObject, const hipResourceDesc *pResDesc)
try: (hipCreateSurfaceObject:=dll.hipCreateSurfaceObject).restype, hipCreateSurfaceObject.argtypes = hipError_t, [ctypes.POINTER(hipSurfaceObject_t), ctypes.POINTER(hipResourceDesc)]
except AttributeError: pass

# hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject)
try: (hipDestroySurfaceObject:=dll.hipDestroySurfaceObject).restype, hipDestroySurfaceObject.argtypes = hipError_t, [hipSurfaceObject_t]
except AttributeError: pass

hipmipmappedArray = ctypes.POINTER(hipMipmappedArray)
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