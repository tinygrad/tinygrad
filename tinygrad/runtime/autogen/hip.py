# mypy: ignore-errors
from __future__ import annotations
import ctypes
from typing import Annotated
from tinygrad.runtime.support.c import DLL, record, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
import os
dll = DLL('hip', os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
class ihipModuleSymbol_t(ctypes.Structure): pass
hipFunction_t = ctypes.POINTER(ihipModuleSymbol_t)
uint32_t = ctypes.c_uint32
size_t = ctypes.c_uint64
class ihipStream_t(ctypes.Structure): pass
hipStream_t = ctypes.POINTER(ihipStream_t)
class ihipEvent_t(ctypes.Structure): pass
hipEvent_t = ctypes.POINTER(ihipEvent_t)
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

@dll.bind
def hipExtModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:ctypes.POINTER(ctypes.POINTER(None)), extra:ctypes.POINTER(ctypes.POINTER(None)), startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:uint32_t) -> hipError_t: ...
@dll.bind
def hipHccModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:ctypes.POINTER(ctypes.POINTER(None)), extra:ctypes.POINTER(ctypes.POINTER(None)), startEvent:hipEvent_t, stopEvent:hipEvent_t) -> hipError_t: ...
@record
class dim3:
  SIZE = 12
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
  z: Annotated[uint32_t, 8]
@dll.bind
def hipExtLaunchKernel(function_address:ctypes.POINTER(None), numBlocks:dim3, dimBlocks:dim3, args:ctypes.POINTER(ctypes.POINTER(None)), sharedMemBytes:size_t, stream:hipStream_t, startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:ctypes.c_int32) -> hipError_t: ...
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

class ihiprtcLinkState(ctypes.Structure): pass
hiprtcLinkState = ctypes.POINTER(ihiprtcLinkState)
@dll.bind
def hiprtcGetErrorString(result:hiprtcResult) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def hiprtcVersion(major:ctypes.POINTER(ctypes.c_int32), minor:ctypes.POINTER(ctypes.c_int32)) -> hiprtcResult: ...
class _hiprtcProgram(ctypes.Structure): pass
hiprtcProgram = ctypes.POINTER(_hiprtcProgram)
@dll.bind
def hiprtcAddNameExpression(prog:hiprtcProgram, name_expression:ctypes.POINTER(ctypes.c_char)) -> hiprtcResult: ...
@dll.bind
def hiprtcCompileProgram(prog:hiprtcProgram, numOptions:ctypes.c_int32, options:ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) -> hiprtcResult: ...
@dll.bind
def hiprtcCreateProgram(prog:ctypes.POINTER(hiprtcProgram), src:ctypes.POINTER(ctypes.c_char), name:ctypes.POINTER(ctypes.c_char), numHeaders:ctypes.c_int32, headers:ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), includeNames:ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) -> hiprtcResult: ...
@dll.bind
def hiprtcDestroyProgram(prog:ctypes.POINTER(hiprtcProgram)) -> hiprtcResult: ...
@dll.bind
def hiprtcGetLoweredName(prog:hiprtcProgram, name_expression:ctypes.POINTER(ctypes.c_char), lowered_name:ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) -> hiprtcResult: ...
@dll.bind
def hiprtcGetProgramLog(prog:hiprtcProgram, log:ctypes.POINTER(ctypes.c_char)) -> hiprtcResult: ...
@dll.bind
def hiprtcGetProgramLogSize(prog:hiprtcProgram, logSizeRet:ctypes.POINTER(size_t)) -> hiprtcResult: ...
@dll.bind
def hiprtcGetCode(prog:hiprtcProgram, code:ctypes.POINTER(ctypes.c_char)) -> hiprtcResult: ...
@dll.bind
def hiprtcGetCodeSize(prog:hiprtcProgram, codeSizeRet:ctypes.POINTER(size_t)) -> hiprtcResult: ...
@dll.bind
def hiprtcGetBitcode(prog:hiprtcProgram, bitcode:ctypes.POINTER(ctypes.c_char)) -> hiprtcResult: ...
@dll.bind
def hiprtcGetBitcodeSize(prog:hiprtcProgram, bitcode_size:ctypes.POINTER(size_t)) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkCreate(num_options:ctypes.c_uint32, option_ptr:ctypes.POINTER(hiprtcJIT_option), option_vals_pptr:ctypes.POINTER(ctypes.POINTER(None)), hip_link_state_ptr:ctypes.POINTER(hiprtcLinkState)) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkAddFile(hip_link_state:hiprtcLinkState, input_type:hiprtcJITInputType, file_path:ctypes.POINTER(ctypes.c_char), num_options:ctypes.c_uint32, options_ptr:ctypes.POINTER(hiprtcJIT_option), option_values:ctypes.POINTER(ctypes.POINTER(None))) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkAddData(hip_link_state:hiprtcLinkState, input_type:hiprtcJITInputType, image:ctypes.POINTER(None), image_size:size_t, name:ctypes.POINTER(ctypes.c_char), num_options:ctypes.c_uint32, options_ptr:ctypes.POINTER(hiprtcJIT_option), option_values:ctypes.POINTER(ctypes.POINTER(None))) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkComplete(hip_link_state:hiprtcLinkState, bin_out:ctypes.POINTER(ctypes.POINTER(None)), size_out:ctypes.POINTER(size_t)) -> hiprtcResult: ...
@dll.bind
def hiprtcLinkDestroy(hip_link_state:hiprtcLinkState) -> hiprtcResult: ...
_anonenum0 = CEnum(ctypes.c_uint32)
HIP_SUCCESS = _anonenum0.define('HIP_SUCCESS', 0)
HIP_ERROR_INVALID_VALUE = _anonenum0.define('HIP_ERROR_INVALID_VALUE', 1)
HIP_ERROR_NOT_INITIALIZED = _anonenum0.define('HIP_ERROR_NOT_INITIALIZED', 2)
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = _anonenum0.define('HIP_ERROR_LAUNCH_OUT_OF_RESOURCES', 3)

@record
class hipDeviceArch_t:
  SIZE = 4
  hasGlobalInt32Atomics: Annotated[ctypes.c_uint32, 0, 1, 0]
  hasGlobalFloatAtomicExch: Annotated[ctypes.c_uint32, 0, 1, 1]
  hasSharedInt32Atomics: Annotated[ctypes.c_uint32, 0, 1, 2]
  hasSharedFloatAtomicExch: Annotated[ctypes.c_uint32, 0, 1, 3]
  hasFloatAtomicAdd: Annotated[ctypes.c_uint32, 0, 1, 4]
  hasGlobalInt64Atomics: Annotated[ctypes.c_uint32, 0, 1, 5]
  hasSharedInt64Atomics: Annotated[ctypes.c_uint32, 0, 1, 6]
  hasDoubles: Annotated[ctypes.c_uint32, 0, 1, 7]
  hasWarpVote: Annotated[ctypes.c_uint32, 1, 1, 0]
  hasWarpBallot: Annotated[ctypes.c_uint32, 1, 1, 1]
  hasWarpShuffle: Annotated[ctypes.c_uint32, 1, 1, 2]
  hasFunnelShift: Annotated[ctypes.c_uint32, 1, 1, 3]
  hasThreadFenceSystem: Annotated[ctypes.c_uint32, 1, 1, 4]
  hasSyncThreadsExt: Annotated[ctypes.c_uint32, 1, 1, 5]
  hasSurfaceFuncs: Annotated[ctypes.c_uint32, 1, 1, 6]
  has3dGrid: Annotated[ctypes.c_uint32, 1, 1, 7]
  hasDynamicParallelism: Annotated[ctypes.c_uint32, 2, 1, 0]
@record
class hipUUID_t:
  SIZE = 16
  bytes: Annotated[(ctypes.c_char* 16), 0]
hipUUID = hipUUID_t
@record
class hipDeviceProp_tR0600:
  SIZE = 1472
  name: Annotated[(ctypes.c_char* 256), 0]
  uuid: Annotated[hipUUID, 256]
  luid: Annotated[(ctypes.c_char* 8), 272]
  luidDeviceNodeMask: Annotated[ctypes.c_uint32, 280]
  totalGlobalMem: Annotated[size_t, 288]
  sharedMemPerBlock: Annotated[size_t, 296]
  regsPerBlock: Annotated[ctypes.c_int32, 304]
  warpSize: Annotated[ctypes.c_int32, 308]
  memPitch: Annotated[size_t, 312]
  maxThreadsPerBlock: Annotated[ctypes.c_int32, 320]
  maxThreadsDim: Annotated[(ctypes.c_int32* 3), 324]
  maxGridSize: Annotated[(ctypes.c_int32* 3), 336]
  clockRate: Annotated[ctypes.c_int32, 348]
  totalConstMem: Annotated[size_t, 352]
  major: Annotated[ctypes.c_int32, 360]
  minor: Annotated[ctypes.c_int32, 364]
  textureAlignment: Annotated[size_t, 368]
  texturePitchAlignment: Annotated[size_t, 376]
  deviceOverlap: Annotated[ctypes.c_int32, 384]
  multiProcessorCount: Annotated[ctypes.c_int32, 388]
  kernelExecTimeoutEnabled: Annotated[ctypes.c_int32, 392]
  integrated: Annotated[ctypes.c_int32, 396]
  canMapHostMemory: Annotated[ctypes.c_int32, 400]
  computeMode: Annotated[ctypes.c_int32, 404]
  maxTexture1D: Annotated[ctypes.c_int32, 408]
  maxTexture1DMipmap: Annotated[ctypes.c_int32, 412]
  maxTexture1DLinear: Annotated[ctypes.c_int32, 416]
  maxTexture2D: Annotated[(ctypes.c_int32* 2), 420]
  maxTexture2DMipmap: Annotated[(ctypes.c_int32* 2), 428]
  maxTexture2DLinear: Annotated[(ctypes.c_int32* 3), 436]
  maxTexture2DGather: Annotated[(ctypes.c_int32* 2), 448]
  maxTexture3D: Annotated[(ctypes.c_int32* 3), 456]
  maxTexture3DAlt: Annotated[(ctypes.c_int32* 3), 468]
  maxTextureCubemap: Annotated[ctypes.c_int32, 480]
  maxTexture1DLayered: Annotated[(ctypes.c_int32* 2), 484]
  maxTexture2DLayered: Annotated[(ctypes.c_int32* 3), 492]
  maxTextureCubemapLayered: Annotated[(ctypes.c_int32* 2), 504]
  maxSurface1D: Annotated[ctypes.c_int32, 512]
  maxSurface2D: Annotated[(ctypes.c_int32* 2), 516]
  maxSurface3D: Annotated[(ctypes.c_int32* 3), 524]
  maxSurface1DLayered: Annotated[(ctypes.c_int32* 2), 536]
  maxSurface2DLayered: Annotated[(ctypes.c_int32* 3), 544]
  maxSurfaceCubemap: Annotated[ctypes.c_int32, 556]
  maxSurfaceCubemapLayered: Annotated[(ctypes.c_int32* 2), 560]
  surfaceAlignment: Annotated[size_t, 568]
  concurrentKernels: Annotated[ctypes.c_int32, 576]
  ECCEnabled: Annotated[ctypes.c_int32, 580]
  pciBusID: Annotated[ctypes.c_int32, 584]
  pciDeviceID: Annotated[ctypes.c_int32, 588]
  pciDomainID: Annotated[ctypes.c_int32, 592]
  tccDriver: Annotated[ctypes.c_int32, 596]
  asyncEngineCount: Annotated[ctypes.c_int32, 600]
  unifiedAddressing: Annotated[ctypes.c_int32, 604]
  memoryClockRate: Annotated[ctypes.c_int32, 608]
  memoryBusWidth: Annotated[ctypes.c_int32, 612]
  l2CacheSize: Annotated[ctypes.c_int32, 616]
  persistingL2CacheMaxSize: Annotated[ctypes.c_int32, 620]
  maxThreadsPerMultiProcessor: Annotated[ctypes.c_int32, 624]
  streamPrioritiesSupported: Annotated[ctypes.c_int32, 628]
  globalL1CacheSupported: Annotated[ctypes.c_int32, 632]
  localL1CacheSupported: Annotated[ctypes.c_int32, 636]
  sharedMemPerMultiprocessor: Annotated[size_t, 640]
  regsPerMultiprocessor: Annotated[ctypes.c_int32, 648]
  managedMemory: Annotated[ctypes.c_int32, 652]
  isMultiGpuBoard: Annotated[ctypes.c_int32, 656]
  multiGpuBoardGroupID: Annotated[ctypes.c_int32, 660]
  hostNativeAtomicSupported: Annotated[ctypes.c_int32, 664]
  singleToDoublePrecisionPerfRatio: Annotated[ctypes.c_int32, 668]
  pageableMemoryAccess: Annotated[ctypes.c_int32, 672]
  concurrentManagedAccess: Annotated[ctypes.c_int32, 676]
  computePreemptionSupported: Annotated[ctypes.c_int32, 680]
  canUseHostPointerForRegisteredMem: Annotated[ctypes.c_int32, 684]
  cooperativeLaunch: Annotated[ctypes.c_int32, 688]
  cooperativeMultiDeviceLaunch: Annotated[ctypes.c_int32, 692]
  sharedMemPerBlockOptin: Annotated[size_t, 696]
  pageableMemoryAccessUsesHostPageTables: Annotated[ctypes.c_int32, 704]
  directManagedMemAccessFromHost: Annotated[ctypes.c_int32, 708]
  maxBlocksPerMultiProcessor: Annotated[ctypes.c_int32, 712]
  accessPolicyMaxWindowSize: Annotated[ctypes.c_int32, 716]
  reservedSharedMemPerBlock: Annotated[size_t, 720]
  hostRegisterSupported: Annotated[ctypes.c_int32, 728]
  sparseHipArraySupported: Annotated[ctypes.c_int32, 732]
  hostRegisterReadOnlySupported: Annotated[ctypes.c_int32, 736]
  timelineSemaphoreInteropSupported: Annotated[ctypes.c_int32, 740]
  memoryPoolsSupported: Annotated[ctypes.c_int32, 744]
  gpuDirectRDMASupported: Annotated[ctypes.c_int32, 748]
  gpuDirectRDMAFlushWritesOptions: Annotated[ctypes.c_uint32, 752]
  gpuDirectRDMAWritesOrdering: Annotated[ctypes.c_int32, 756]
  memoryPoolSupportedHandleTypes: Annotated[ctypes.c_uint32, 760]
  deferredMappingHipArraySupported: Annotated[ctypes.c_int32, 764]
  ipcEventSupported: Annotated[ctypes.c_int32, 768]
  clusterLaunch: Annotated[ctypes.c_int32, 772]
  unifiedFunctionPointers: Annotated[ctypes.c_int32, 776]
  reserved: Annotated[(ctypes.c_int32* 63), 780]
  hipReserved: Annotated[(ctypes.c_int32* 32), 1032]
  gcnArchName: Annotated[(ctypes.c_char* 256), 1160]
  maxSharedMemoryPerMultiProcessor: Annotated[size_t, 1416]
  clockInstructionRate: Annotated[ctypes.c_int32, 1424]
  arch: Annotated[hipDeviceArch_t, 1428]
  hdpMemFlushCntl: Annotated[ctypes.POINTER(ctypes.c_uint32), 1432]
  hdpRegFlushCntl: Annotated[ctypes.POINTER(ctypes.c_uint32), 1440]
  cooperativeMultiDeviceUnmatchedFunc: Annotated[ctypes.c_int32, 1448]
  cooperativeMultiDeviceUnmatchedGridDim: Annotated[ctypes.c_int32, 1452]
  cooperativeMultiDeviceUnmatchedBlockDim: Annotated[ctypes.c_int32, 1456]
  cooperativeMultiDeviceUnmatchedSharedMem: Annotated[ctypes.c_int32, 1460]
  isLargeBar: Annotated[ctypes.c_int32, 1464]
  asicRevision: Annotated[ctypes.c_int32, 1468]
hipMemoryType = CEnum(ctypes.c_uint32)
hipMemoryTypeUnregistered = hipMemoryType.define('hipMemoryTypeUnregistered', 0)
hipMemoryTypeHost = hipMemoryType.define('hipMemoryTypeHost', 1)
hipMemoryTypeDevice = hipMemoryType.define('hipMemoryTypeDevice', 2)
hipMemoryTypeManaged = hipMemoryType.define('hipMemoryTypeManaged', 3)
hipMemoryTypeArray = hipMemoryType.define('hipMemoryTypeArray', 10)
hipMemoryTypeUnified = hipMemoryType.define('hipMemoryTypeUnified', 11)

@record
class hipPointerAttribute_t:
  SIZE = 32
  type: Annotated[hipMemoryType, 0]
  device: Annotated[ctypes.c_int32, 4]
  devicePointer: Annotated[ctypes.POINTER(None), 8]
  hostPointer: Annotated[ctypes.POINTER(None), 16]
  isManaged: Annotated[ctypes.c_int32, 24]
  allocationFlags: Annotated[ctypes.c_uint32, 28]
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

@dll.bind
def hip_init() -> hipError_t: ...
class ihipCtx_t(ctypes.Structure): pass
hipCtx_t = ctypes.POINTER(ihipCtx_t)
hipDevice_t = ctypes.c_int32
hipDeviceP2PAttr = CEnum(ctypes.c_uint32)
hipDevP2PAttrPerformanceRank = hipDeviceP2PAttr.define('hipDevP2PAttrPerformanceRank', 0)
hipDevP2PAttrAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrAccessSupported', 1)
hipDevP2PAttrNativeAtomicSupported = hipDeviceP2PAttr.define('hipDevP2PAttrNativeAtomicSupported', 2)
hipDevP2PAttrHipArrayAccessSupported = hipDeviceP2PAttr.define('hipDevP2PAttrHipArrayAccessSupported', 3)

@record
class hipIpcMemHandle_st:
  SIZE = 64
  reserved: Annotated[(ctypes.c_char* 64), 0]
hipIpcMemHandle_t = hipIpcMemHandle_st
@record
class hipIpcEventHandle_st:
  SIZE = 64
  reserved: Annotated[(ctypes.c_char* 64), 0]
hipIpcEventHandle_t = hipIpcEventHandle_st
class ihipModule_t(ctypes.Structure): pass
hipModule_t = ctypes.POINTER(ihipModule_t)
class ihipMemPoolHandle_t(ctypes.Structure): pass
hipMemPool_t = ctypes.POINTER(ihipMemPoolHandle_t)
@record
class hipFuncAttributes:
  SIZE = 56
  binaryVersion: Annotated[ctypes.c_int32, 0]
  cacheModeCA: Annotated[ctypes.c_int32, 4]
  constSizeBytes: Annotated[size_t, 8]
  localSizeBytes: Annotated[size_t, 16]
  maxDynamicSharedSizeBytes: Annotated[ctypes.c_int32, 24]
  maxThreadsPerBlock: Annotated[ctypes.c_int32, 28]
  numRegs: Annotated[ctypes.c_int32, 32]
  preferredShmemCarveout: Annotated[ctypes.c_int32, 36]
  ptxVersion: Annotated[ctypes.c_int32, 40]
  sharedSizeBytes: Annotated[size_t, 48]
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

@record
class hipMemLocation:
  SIZE = 8
  type: Annotated[hipMemLocationType, 0]
  id: Annotated[ctypes.c_int32, 4]
hipMemAccessFlags = CEnum(ctypes.c_uint32)
hipMemAccessFlagsProtNone = hipMemAccessFlags.define('hipMemAccessFlagsProtNone', 0)
hipMemAccessFlagsProtRead = hipMemAccessFlags.define('hipMemAccessFlagsProtRead', 1)
hipMemAccessFlagsProtReadWrite = hipMemAccessFlags.define('hipMemAccessFlagsProtReadWrite', 3)

@record
class hipMemAccessDesc:
  SIZE = 12
  location: Annotated[hipMemLocation, 0]
  flags: Annotated[hipMemAccessFlags, 8]
hipMemAllocationType = CEnum(ctypes.c_uint32)
hipMemAllocationTypeInvalid = hipMemAllocationType.define('hipMemAllocationTypeInvalid', 0)
hipMemAllocationTypePinned = hipMemAllocationType.define('hipMemAllocationTypePinned', 1)
hipMemAllocationTypeMax = hipMemAllocationType.define('hipMemAllocationTypeMax', 2147483647)

hipMemAllocationHandleType = CEnum(ctypes.c_uint32)
hipMemHandleTypeNone = hipMemAllocationHandleType.define('hipMemHandleTypeNone', 0)
hipMemHandleTypePosixFileDescriptor = hipMemAllocationHandleType.define('hipMemHandleTypePosixFileDescriptor', 1)
hipMemHandleTypeWin32 = hipMemAllocationHandleType.define('hipMemHandleTypeWin32', 2)
hipMemHandleTypeWin32Kmt = hipMemAllocationHandleType.define('hipMemHandleTypeWin32Kmt', 4)

@record
class hipMemPoolProps:
  SIZE = 88
  allocType: Annotated[hipMemAllocationType, 0]
  handleTypes: Annotated[hipMemAllocationHandleType, 4]
  location: Annotated[hipMemLocation, 8]
  win32SecurityAttributes: Annotated[ctypes.POINTER(None), 16]
  maxSize: Annotated[size_t, 24]
  reserved: Annotated[(ctypes.c_ubyte* 56), 32]
@record
class hipMemPoolPtrExportData:
  SIZE = 64
  reserved: Annotated[(ctypes.c_ubyte* 64), 0]
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

@record
class hipLaunchParams_t:
  SIZE = 56
  func: Annotated[ctypes.POINTER(None), 0]
  gridDim: Annotated[dim3, 8]
  blockDim: Annotated[dim3, 20]
  args: Annotated[ctypes.POINTER(ctypes.POINTER(None)), 32]
  sharedMem: Annotated[size_t, 40]
  stream: Annotated[hipStream_t, 48]
hipLaunchParams = hipLaunchParams_t
@record
class hipFunctionLaunchParams_t:
  SIZE = 56
  function: Annotated[hipFunction_t, 0]
  gridDimX: Annotated[ctypes.c_uint32, 8]
  gridDimY: Annotated[ctypes.c_uint32, 12]
  gridDimZ: Annotated[ctypes.c_uint32, 16]
  blockDimX: Annotated[ctypes.c_uint32, 20]
  blockDimY: Annotated[ctypes.c_uint32, 24]
  blockDimZ: Annotated[ctypes.c_uint32, 28]
  sharedMemBytes: Annotated[ctypes.c_uint32, 32]
  hStream: Annotated[hipStream_t, 40]
  kernelParams: Annotated[ctypes.POINTER(ctypes.POINTER(None)), 48]
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
@record
class hipExternalMemoryHandleDesc_st:
  SIZE = 104
  type: Annotated[hipExternalMemoryHandleType, 0]
  handle: Annotated[_anonunion1, 8]
  size: Annotated[ctypes.c_uint64, 24]
  flags: Annotated[ctypes.c_uint32, 32]
  reserved: Annotated[(ctypes.c_uint32* 16), 36]
@record
class _anonunion1:
  SIZE = 16
  fd: Annotated[ctypes.c_int32, 0]
  win32: Annotated[_anonstruct2, 0]
  nvSciBufObject: Annotated[ctypes.POINTER(None), 0]
@record
class _anonstruct2:
  SIZE = 16
  handle: Annotated[ctypes.POINTER(None), 0]
  name: Annotated[ctypes.POINTER(None), 8]
hipExternalMemoryHandleDesc = hipExternalMemoryHandleDesc_st
@record
class hipExternalMemoryBufferDesc_st:
  SIZE = 88
  offset: Annotated[ctypes.c_uint64, 0]
  size: Annotated[ctypes.c_uint64, 8]
  flags: Annotated[ctypes.c_uint32, 16]
  reserved: Annotated[(ctypes.c_uint32* 16), 20]
hipExternalMemoryBufferDesc = hipExternalMemoryBufferDesc_st
@record
class hipExternalMemoryMipmappedArrayDesc_st:
  SIZE = 64
  offset: Annotated[ctypes.c_uint64, 0]
  formatDesc: Annotated[hipChannelFormatDesc, 8]
  extent: Annotated[hipExtent, 32]
  flags: Annotated[ctypes.c_uint32, 56]
  numLevels: Annotated[ctypes.c_uint32, 60]
@record
class hipChannelFormatDesc:
  SIZE = 20
  x: Annotated[ctypes.c_int32, 0]
  y: Annotated[ctypes.c_int32, 4]
  z: Annotated[ctypes.c_int32, 8]
  w: Annotated[ctypes.c_int32, 12]
  f: Annotated[hipChannelFormatKind, 16]
hipChannelFormatKind = CEnum(ctypes.c_uint32)
hipChannelFormatKindSigned = hipChannelFormatKind.define('hipChannelFormatKindSigned', 0)
hipChannelFormatKindUnsigned = hipChannelFormatKind.define('hipChannelFormatKindUnsigned', 1)
hipChannelFormatKindFloat = hipChannelFormatKind.define('hipChannelFormatKindFloat', 2)
hipChannelFormatKindNone = hipChannelFormatKind.define('hipChannelFormatKindNone', 3)

@record
class hipExtent:
  SIZE = 24
  width: Annotated[size_t, 0]
  height: Annotated[size_t, 8]
  depth: Annotated[size_t, 16]
hipExternalMemoryMipmappedArrayDesc = hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t = ctypes.POINTER(None)
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
@record
class hipExternalSemaphoreHandleDesc_st:
  SIZE = 96
  type: Annotated[hipExternalSemaphoreHandleType, 0]
  handle: Annotated[_anonunion3, 8]
  flags: Annotated[ctypes.c_uint32, 24]
  reserved: Annotated[(ctypes.c_uint32* 16), 28]
@record
class _anonunion3:
  SIZE = 16
  fd: Annotated[ctypes.c_int32, 0]
  win32: Annotated[_anonstruct4, 0]
  NvSciSyncObj: Annotated[ctypes.POINTER(None), 0]
@record
class _anonstruct4:
  SIZE = 16
  handle: Annotated[ctypes.POINTER(None), 0]
  name: Annotated[ctypes.POINTER(None), 8]
hipExternalSemaphoreHandleDesc = hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t = ctypes.POINTER(None)
@record
class hipExternalSemaphoreSignalParams_st:
  SIZE = 144
  params: Annotated[_anonstruct5, 0]
  flags: Annotated[ctypes.c_uint32, 72]
  reserved: Annotated[(ctypes.c_uint32* 16), 76]
@record
class _anonstruct5:
  SIZE = 72
  fence: Annotated[_anonstruct6, 0]
  nvSciSync: Annotated[_anonunion7, 8]
  keyedMutex: Annotated[_anonstruct8, 16]
  reserved: Annotated[(ctypes.c_uint32* 12), 24]
@record
class _anonstruct6:
  SIZE = 8
  value: Annotated[ctypes.c_uint64, 0]
@record
class _anonunion7:
  SIZE = 8
  fence: Annotated[ctypes.POINTER(None), 0]
  reserved: Annotated[ctypes.c_uint64, 0]
@record
class _anonstruct8:
  SIZE = 8
  key: Annotated[ctypes.c_uint64, 0]
hipExternalSemaphoreSignalParams = hipExternalSemaphoreSignalParams_st
@record
class hipExternalSemaphoreWaitParams_st:
  SIZE = 144
  params: Annotated[_anonstruct9, 0]
  flags: Annotated[ctypes.c_uint32, 72]
  reserved: Annotated[(ctypes.c_uint32* 16), 76]
@record
class _anonstruct9:
  SIZE = 72
  fence: Annotated[_anonstruct10, 0]
  nvSciSync: Annotated[_anonunion11, 8]
  keyedMutex: Annotated[_anonstruct12, 16]
  reserved: Annotated[(ctypes.c_uint32* 10), 32]
@record
class _anonstruct10:
  SIZE = 8
  value: Annotated[ctypes.c_uint64, 0]
@record
class _anonunion11:
  SIZE = 8
  fence: Annotated[ctypes.POINTER(None), 0]
  reserved: Annotated[ctypes.c_uint64, 0]
@record
class _anonstruct12:
  SIZE = 16
  key: Annotated[ctypes.c_uint64, 0]
  timeoutMs: Annotated[ctypes.c_uint32, 8]
hipExternalSemaphoreWaitParams = hipExternalSemaphoreWaitParams_st
@dll.bind
def __hipGetPCH(pch:ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), size:ctypes.POINTER(ctypes.c_uint32)) -> None: ...
hipGraphicsRegisterFlags = CEnum(ctypes.c_uint32)
hipGraphicsRegisterFlagsNone = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsNone', 0)
hipGraphicsRegisterFlagsReadOnly = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsReadOnly', 1)
hipGraphicsRegisterFlagsWriteDiscard = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsWriteDiscard', 2)
hipGraphicsRegisterFlagsSurfaceLoadStore = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsSurfaceLoadStore', 4)
hipGraphicsRegisterFlagsTextureGather = hipGraphicsRegisterFlags.define('hipGraphicsRegisterFlagsTextureGather', 8)

class _hipGraphicsResource(ctypes.Structure): pass
hipGraphicsResource = _hipGraphicsResource
hipGraphicsResource_t = ctypes.POINTER(_hipGraphicsResource)
class ihipGraph(ctypes.Structure): pass
hipGraph_t = ctypes.POINTER(ihipGraph)
class hipGraphNode(ctypes.Structure): pass
hipGraphNode_t = ctypes.POINTER(hipGraphNode)
class hipGraphExec(ctypes.Structure): pass
hipGraphExec_t = ctypes.POINTER(hipGraphExec)
class hipUserObject(ctypes.Structure): pass
hipUserObject_t = ctypes.POINTER(hipUserObject)
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

class hipHostNodeParams(ctypes.Structure): pass
@record
class hipKernelNodeParams:
  SIZE = 64
  blockDim: Annotated[dim3, 0]
  extra: Annotated[ctypes.POINTER(ctypes.POINTER(None)), 16]
  func: Annotated[ctypes.POINTER(None), 24]
  gridDim: Annotated[dim3, 32]
  kernelParams: Annotated[ctypes.POINTER(ctypes.POINTER(None)), 48]
  sharedMemBytes: Annotated[ctypes.c_uint32, 56]
@record
class hipMemsetParams:
  SIZE = 48
  dst: Annotated[ctypes.POINTER(None), 0]
  elementSize: Annotated[ctypes.c_uint32, 8]
  height: Annotated[size_t, 16]
  pitch: Annotated[size_t, 24]
  value: Annotated[ctypes.c_uint32, 32]
  width: Annotated[size_t, 40]
@record
class hipMemAllocNodeParams:
  SIZE = 120
  poolProps: Annotated[hipMemPoolProps, 0]
  accessDescs: Annotated[ctypes.POINTER(hipMemAccessDesc), 88]
  accessDescCount: Annotated[size_t, 96]
  bytesize: Annotated[size_t, 104]
  dptr: Annotated[ctypes.POINTER(None), 112]
hipAccessProperty = CEnum(ctypes.c_uint32)
hipAccessPropertyNormal = hipAccessProperty.define('hipAccessPropertyNormal', 0)
hipAccessPropertyStreaming = hipAccessProperty.define('hipAccessPropertyStreaming', 1)
hipAccessPropertyPersisting = hipAccessProperty.define('hipAccessPropertyPersisting', 2)

@record
class hipAccessPolicyWindow:
  SIZE = 32
  base_ptr: Annotated[ctypes.POINTER(None), 0]
  hitProp: Annotated[hipAccessProperty, 8]
  hitRatio: Annotated[ctypes.c_float, 12]
  missProp: Annotated[hipAccessProperty, 16]
  num_bytes: Annotated[size_t, 24]
hipLaunchAttributeID = CEnum(ctypes.c_uint32)
hipLaunchAttributeAccessPolicyWindow = hipLaunchAttributeID.define('hipLaunchAttributeAccessPolicyWindow', 1)
hipLaunchAttributeCooperative = hipLaunchAttributeID.define('hipLaunchAttributeCooperative', 2)
hipLaunchAttributePriority = hipLaunchAttributeID.define('hipLaunchAttributePriority', 8)

@record
class hipLaunchAttributeValue:
  SIZE = 32
  accessPolicyWindow: Annotated[hipAccessPolicyWindow, 0]
  cooperative: Annotated[ctypes.c_int32, 0]
  priority: Annotated[ctypes.c_int32, 0]
@record
class HIP_MEMSET_NODE_PARAMS:
  SIZE = 40
  dst: Annotated[hipDeviceptr_t, 0]
  pitch: Annotated[size_t, 8]
  value: Annotated[ctypes.c_uint32, 16]
  elementSize: Annotated[ctypes.c_uint32, 20]
  width: Annotated[size_t, 24]
  height: Annotated[size_t, 32]
hipDeviceptr_t = ctypes.POINTER(None)
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

@record
class hipGraphInstantiateParams:
  SIZE = 32
  errNode_out: Annotated[hipGraphNode_t, 0]
  flags: Annotated[ctypes.c_uint64, 8]
  result_out: Annotated[hipGraphInstantiateResult, 16]
  uploadStream: Annotated[hipStream_t, 24]
@record
class hipMemAllocationProp:
  SIZE = 32
  type: Annotated[hipMemAllocationType, 0]
  requestedHandleType: Annotated[hipMemAllocationHandleType, 4]
  location: Annotated[hipMemLocation, 8]
  win32HandleMetaData: Annotated[ctypes.POINTER(None), 16]
  allocFlags: Annotated[_anonstruct13, 24]
@record
class _anonstruct13:
  SIZE = 4
  compressionType: Annotated[ctypes.c_ubyte, 0]
  gpuDirectRDMACapable: Annotated[ctypes.c_ubyte, 1]
  usage: Annotated[ctypes.c_uint16, 2]
@record
class hipExternalSemaphoreSignalNodeParams:
  SIZE = 24
  extSemArray: Annotated[ctypes.POINTER(hipExternalSemaphore_t), 0]
  paramsArray: Annotated[ctypes.POINTER(hipExternalSemaphoreSignalParams), 8]
  numExtSems: Annotated[ctypes.c_uint32, 16]
@record
class hipExternalSemaphoreWaitNodeParams:
  SIZE = 24
  extSemArray: Annotated[ctypes.POINTER(hipExternalSemaphore_t), 0]
  paramsArray: Annotated[ctypes.POINTER(hipExternalSemaphoreWaitParams), 8]
  numExtSems: Annotated[ctypes.c_uint32, 16]
class ihipMemGenericAllocationHandle(ctypes.Structure): pass
hipMemGenericAllocationHandle_t = ctypes.POINTER(ihipMemGenericAllocationHandle)
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

@record
class hipArrayMapInfo:
  SIZE = 152
  resourceType: Annotated[hipResourceType, 0]
  resource: Annotated[_anonunion14, 8]
  subresourceType: Annotated[hipArraySparseSubresourceType, 72]
  subresource: Annotated[_anonunion15, 80]
  memOperationType: Annotated[hipMemOperationType, 112]
  memHandleType: Annotated[hipMemHandleType, 116]
  memHandle: Annotated[_anonunion18, 120]
  offset: Annotated[ctypes.c_uint64, 128]
  deviceBitMask: Annotated[ctypes.c_uint32, 136]
  flags: Annotated[ctypes.c_uint32, 140]
  reserved: Annotated[(ctypes.c_uint32* 2), 144]
hipResourceType = CEnum(ctypes.c_uint32)
hipResourceTypeArray = hipResourceType.define('hipResourceTypeArray', 0)
hipResourceTypeMipmappedArray = hipResourceType.define('hipResourceTypeMipmappedArray', 1)
hipResourceTypeLinear = hipResourceType.define('hipResourceTypeLinear', 2)
hipResourceTypePitch2D = hipResourceType.define('hipResourceTypePitch2D', 3)

@record
class _anonunion14:
  SIZE = 64
  mipmap: Annotated[hipMipmappedArray, 0]
  array: Annotated[hipArray_t, 0]
@record
class hipMipmappedArray:
  SIZE = 64
  data: Annotated[ctypes.POINTER(None), 0]
  desc: Annotated[hipChannelFormatDesc, 8]
  type: Annotated[ctypes.c_uint32, 28]
  width: Annotated[ctypes.c_uint32, 32]
  height: Annotated[ctypes.c_uint32, 36]
  depth: Annotated[ctypes.c_uint32, 40]
  min_mipmap_level: Annotated[ctypes.c_uint32, 44]
  max_mipmap_level: Annotated[ctypes.c_uint32, 48]
  flags: Annotated[ctypes.c_uint32, 52]
  format: Annotated[hipArray_Format, 56]
  num_channels: Annotated[ctypes.c_uint32, 60]
hipArray_Format = CEnum(ctypes.c_uint32)
HIP_AD_FORMAT_UNSIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT8', 1)
HIP_AD_FORMAT_UNSIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT16', 2)
HIP_AD_FORMAT_UNSIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_UNSIGNED_INT32', 3)
HIP_AD_FORMAT_SIGNED_INT8 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT8', 8)
HIP_AD_FORMAT_SIGNED_INT16 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT16', 9)
HIP_AD_FORMAT_SIGNED_INT32 = hipArray_Format.define('HIP_AD_FORMAT_SIGNED_INT32', 10)
HIP_AD_FORMAT_HALF = hipArray_Format.define('HIP_AD_FORMAT_HALF', 16)
HIP_AD_FORMAT_FLOAT = hipArray_Format.define('HIP_AD_FORMAT_FLOAT', 32)

class hipArray(ctypes.Structure): pass
hipArray_t = ctypes.POINTER(hipArray)
@record
class _anonunion15:
  SIZE = 32
  sparseLevel: Annotated[_anonstruct16, 0]
  miptail: Annotated[_anonstruct17, 0]
@record
class _anonstruct16:
  SIZE = 32
  level: Annotated[ctypes.c_uint32, 0]
  layer: Annotated[ctypes.c_uint32, 4]
  offsetX: Annotated[ctypes.c_uint32, 8]
  offsetY: Annotated[ctypes.c_uint32, 12]
  offsetZ: Annotated[ctypes.c_uint32, 16]
  extentWidth: Annotated[ctypes.c_uint32, 20]
  extentHeight: Annotated[ctypes.c_uint32, 24]
  extentDepth: Annotated[ctypes.c_uint32, 28]
@record
class _anonstruct17:
  SIZE = 24
  layer: Annotated[ctypes.c_uint32, 0]
  offset: Annotated[ctypes.c_uint64, 8]
  size: Annotated[ctypes.c_uint64, 16]
@record
class _anonunion18:
  SIZE = 8
  memHandle: Annotated[hipMemGenericAllocationHandle_t, 0]
@record
class hipMemcpyNodeParams:
  SIZE = 176
  flags: Annotated[ctypes.c_int32, 0]
  reserved: Annotated[(ctypes.c_int32* 3), 4]
  copyParams: Annotated[hipMemcpy3DParms, 16]
@record
class hipMemcpy3DParms:
  SIZE = 160
  srcArray: Annotated[hipArray_t, 0]
  srcPos: Annotated[hipPos, 8]
  srcPtr: Annotated[hipPitchedPtr, 32]
  dstArray: Annotated[hipArray_t, 64]
  dstPos: Annotated[hipPos, 72]
  dstPtr: Annotated[hipPitchedPtr, 96]
  extent: Annotated[hipExtent, 128]
  kind: Annotated[hipMemcpyKind, 152]
@record
class hipPos:
  SIZE = 24
  x: Annotated[size_t, 0]
  y: Annotated[size_t, 8]
  z: Annotated[size_t, 16]
@record
class hipPitchedPtr:
  SIZE = 32
  ptr: Annotated[ctypes.POINTER(None), 0]
  pitch: Annotated[size_t, 8]
  xsize: Annotated[size_t, 16]
  ysize: Annotated[size_t, 24]
hipMemcpyKind = CEnum(ctypes.c_uint32)
hipMemcpyHostToHost = hipMemcpyKind.define('hipMemcpyHostToHost', 0)
hipMemcpyHostToDevice = hipMemcpyKind.define('hipMemcpyHostToDevice', 1)
hipMemcpyDeviceToHost = hipMemcpyKind.define('hipMemcpyDeviceToHost', 2)
hipMemcpyDeviceToDevice = hipMemcpyKind.define('hipMemcpyDeviceToDevice', 3)
hipMemcpyDefault = hipMemcpyKind.define('hipMemcpyDefault', 4)
hipMemcpyDeviceToDeviceNoCU = hipMemcpyKind.define('hipMemcpyDeviceToDeviceNoCU', 1024)

@record
class hipChildGraphNodeParams:
  SIZE = 8
  graph: Annotated[hipGraph_t, 0]
@record
class hipEventWaitNodeParams:
  SIZE = 8
  event: Annotated[hipEvent_t, 0]
@record
class hipEventRecordNodeParams:
  SIZE = 8
  event: Annotated[hipEvent_t, 0]
@record
class hipMemFreeNodeParams:
  SIZE = 8
  dptr: Annotated[ctypes.POINTER(None), 0]
@record
class hipGraphNodeParams:
  SIZE = 256
  type: Annotated[hipGraphNodeType, 0]
  reserved0: Annotated[(ctypes.c_int32* 3), 4]
  reserved1: Annotated[(ctypes.c_int64* 29), 16]
  kernel: Annotated[hipKernelNodeParams, 16]
  memcpy: Annotated[hipMemcpyNodeParams, 16]
  memset: Annotated[hipMemsetParams, 16]
  host: Annotated[hipHostNodeParams, 16]
  graph: Annotated[hipChildGraphNodeParams, 16]
  eventWait: Annotated[hipEventWaitNodeParams, 16]
  eventRecord: Annotated[hipEventRecordNodeParams, 16]
  extSemSignal: Annotated[hipExternalSemaphoreSignalNodeParams, 16]
  extSemWait: Annotated[hipExternalSemaphoreWaitNodeParams, 16]
  alloc: Annotated[hipMemAllocNodeParams, 16]
  free: Annotated[hipMemFreeNodeParams, 16]
  reserved2: Annotated[ctypes.c_int64, 248]
hipGraphDependencyType = CEnum(ctypes.c_uint32)
hipGraphDependencyTypeDefault = hipGraphDependencyType.define('hipGraphDependencyTypeDefault', 0)
hipGraphDependencyTypeProgrammatic = hipGraphDependencyType.define('hipGraphDependencyTypeProgrammatic', 1)

@record
class hipGraphEdgeData:
  SIZE = 8
  from_port: Annotated[ctypes.c_ubyte, 0]
  reserved: Annotated[(ctypes.c_ubyte* 5), 1]
  to_port: Annotated[ctypes.c_ubyte, 6]
  type: Annotated[ctypes.c_ubyte, 7]
@dll.bind
def hipInit(flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipDriverGetVersion(driverVersion:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipRuntimeGetVersion(runtimeVersion:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipDeviceGet(device:ctypes.POINTER(hipDevice_t), ordinal:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceComputeCapability(major:ctypes.POINTER(ctypes.c_int32), minor:ctypes.POINTER(ctypes.c_int32), device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetName(name:ctypes.POINTER(ctypes.c_char), len:ctypes.c_int32, device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetUuid(uuid:ctypes.POINTER(hipUUID), device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetP2PAttribute(value:ctypes.POINTER(ctypes.c_int32), attr:hipDeviceP2PAttr, srcDevice:ctypes.c_int32, dstDevice:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceGetPCIBusId(pciBusId:ctypes.POINTER(ctypes.c_char), len:ctypes.c_int32, device:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceGetByPCIBusId(device:ctypes.POINTER(ctypes.c_int32), pciBusId:ctypes.POINTER(ctypes.c_char)) -> hipError_t: ...
@dll.bind
def hipDeviceTotalMem(bytes:ctypes.POINTER(size_t), device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDeviceSynchronize() -> hipError_t: ...
@dll.bind
def hipDeviceReset() -> hipError_t: ...
@dll.bind
def hipSetDevice(deviceId:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipSetValidDevices(device_arr:ctypes.POINTER(ctypes.c_int32), len:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipGetDevice(deviceId:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipGetDeviceCount(count:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipDeviceGetAttribute(pi:ctypes.POINTER(ctypes.c_int32), attr:hipDeviceAttribute_t, deviceId:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceGetDefaultMemPool(mem_pool:ctypes.POINTER(hipMemPool_t), device:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceSetMemPool(device:ctypes.c_int32, mem_pool:hipMemPool_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetMemPool(mem_pool:ctypes.POINTER(hipMemPool_t), device:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipGetDevicePropertiesR0600(prop:ctypes.POINTER(hipDeviceProp_tR0600), deviceId:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceSetCacheConfig(cacheConfig:hipFuncCache_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetCacheConfig(cacheConfig:ctypes.POINTER(hipFuncCache_t)) -> hipError_t: ...
@dll.bind
def hipDeviceGetLimit(pValue:ctypes.POINTER(size_t), limit:hipLimit_t) -> hipError_t: ...
@dll.bind
def hipDeviceSetLimit(limit:hipLimit_t, value:size_t) -> hipError_t: ...
@dll.bind
def hipDeviceGetSharedMemConfig(pConfig:ctypes.POINTER(hipSharedMemConfig)) -> hipError_t: ...
@dll.bind
def hipGetDeviceFlags(flags:ctypes.POINTER(ctypes.c_uint32)) -> hipError_t: ...
@dll.bind
def hipDeviceSetSharedMemConfig(config:hipSharedMemConfig) -> hipError_t: ...
@dll.bind
def hipSetDeviceFlags(flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipChooseDeviceR0600(device:ctypes.POINTER(ctypes.c_int32), prop:ctypes.POINTER(hipDeviceProp_tR0600)) -> hipError_t: ...
@dll.bind
def hipExtGetLinkTypeAndHopCount(device1:ctypes.c_int32, device2:ctypes.c_int32, linktype:ctypes.POINTER(uint32_t), hopcount:ctypes.POINTER(uint32_t)) -> hipError_t: ...
@dll.bind
def hipIpcGetMemHandle(handle:ctypes.POINTER(hipIpcMemHandle_t), devPtr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipIpcOpenMemHandle(devPtr:ctypes.POINTER(ctypes.POINTER(None)), handle:hipIpcMemHandle_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipIpcCloseMemHandle(devPtr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipIpcGetEventHandle(handle:ctypes.POINTER(hipIpcEventHandle_t), event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipIpcOpenEventHandle(event:ctypes.POINTER(hipEvent_t), handle:hipIpcEventHandle_t) -> hipError_t: ...
@dll.bind
def hipFuncSetAttribute(func:ctypes.POINTER(None), attr:hipFuncAttribute, value:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipFuncSetCacheConfig(func:ctypes.POINTER(None), config:hipFuncCache_t) -> hipError_t: ...
@dll.bind
def hipFuncSetSharedMemConfig(func:ctypes.POINTER(None), config:hipSharedMemConfig) -> hipError_t: ...
@dll.bind
def hipGetLastError() -> hipError_t: ...
@dll.bind
def hipExtGetLastError() -> hipError_t: ...
@dll.bind
def hipPeekAtLastError() -> hipError_t: ...
@dll.bind
def hipGetErrorName(hip_error:hipError_t) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def hipGetErrorString(hipError:hipError_t) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def hipDrvGetErrorName(hipError:hipError_t, errorString:ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) -> hipError_t: ...
@dll.bind
def hipDrvGetErrorString(hipError:hipError_t, errorString:ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) -> hipError_t: ...
@dll.bind
def hipStreamCreate(stream:ctypes.POINTER(hipStream_t)) -> hipError_t: ...
@dll.bind
def hipStreamCreateWithFlags(stream:ctypes.POINTER(hipStream_t), flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipStreamCreateWithPriority(stream:ctypes.POINTER(hipStream_t), flags:ctypes.c_uint32, priority:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceGetStreamPriorityRange(leastPriority:ctypes.POINTER(ctypes.c_int32), greatestPriority:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipStreamDestroy(stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipStreamQuery(stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipStreamSynchronize(stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipStreamWaitEvent(stream:hipStream_t, event:hipEvent_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipStreamGetFlags(stream:hipStream_t, flags:ctypes.POINTER(ctypes.c_uint32)) -> hipError_t: ...
@dll.bind
def hipStreamGetPriority(stream:hipStream_t, priority:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipStreamGetDevice(stream:hipStream_t, device:ctypes.POINTER(hipDevice_t)) -> hipError_t: ...
@dll.bind
def hipExtStreamCreateWithCUMask(stream:ctypes.POINTER(hipStream_t), cuMaskSize:uint32_t, cuMask:ctypes.POINTER(uint32_t)) -> hipError_t: ...
@dll.bind
def hipExtStreamGetCUMask(stream:hipStream_t, cuMaskSize:uint32_t, cuMask:ctypes.POINTER(uint32_t)) -> hipError_t: ...
@dll.bind
def hipStreamWaitValue32(stream:hipStream_t, ptr:ctypes.POINTER(None), value:uint32_t, flags:ctypes.c_uint32, mask:uint32_t) -> hipError_t: ...
uint64_t = ctypes.c_uint64
@dll.bind
def hipStreamWaitValue64(stream:hipStream_t, ptr:ctypes.POINTER(None), value:uint64_t, flags:ctypes.c_uint32, mask:uint64_t) -> hipError_t: ...
@dll.bind
def hipStreamWriteValue32(stream:hipStream_t, ptr:ctypes.POINTER(None), value:uint32_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipStreamWriteValue64(stream:hipStream_t, ptr:ctypes.POINTER(None), value:uint64_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipEventCreateWithFlags(event:ctypes.POINTER(hipEvent_t), flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipEventCreate(event:ctypes.POINTER(hipEvent_t)) -> hipError_t: ...
@dll.bind
def hipEventRecord(event:hipEvent_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipEventDestroy(event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipEventSynchronize(event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipEventElapsedTime(ms:ctypes.POINTER(ctypes.c_float), start:hipEvent_t, stop:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipEventQuery(event:hipEvent_t) -> hipError_t: ...
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

@dll.bind
def hipPointerSetAttribute(value:ctypes.POINTER(None), attribute:hipPointer_attribute, ptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipPointerGetAttributes(attributes:ctypes.POINTER(hipPointerAttribute_t), ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipPointerGetAttribute(data:ctypes.POINTER(None), attribute:hipPointer_attribute, ptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipDrvPointerGetAttributes(numAttributes:ctypes.c_uint32, attributes:ctypes.POINTER(hipPointer_attribute), data:ctypes.POINTER(ctypes.POINTER(None)), ptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipImportExternalSemaphore(extSem_out:ctypes.POINTER(hipExternalSemaphore_t), semHandleDesc:ctypes.POINTER(hipExternalSemaphoreHandleDesc)) -> hipError_t: ...
@dll.bind
def hipSignalExternalSemaphoresAsync(extSemArray:ctypes.POINTER(hipExternalSemaphore_t), paramsArray:ctypes.POINTER(hipExternalSemaphoreSignalParams), numExtSems:ctypes.c_uint32, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipWaitExternalSemaphoresAsync(extSemArray:ctypes.POINTER(hipExternalSemaphore_t), paramsArray:ctypes.POINTER(hipExternalSemaphoreWaitParams), numExtSems:ctypes.c_uint32, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipDestroyExternalSemaphore(extSem:hipExternalSemaphore_t) -> hipError_t: ...
@dll.bind
def hipImportExternalMemory(extMem_out:ctypes.POINTER(hipExternalMemory_t), memHandleDesc:ctypes.POINTER(hipExternalMemoryHandleDesc)) -> hipError_t: ...
@dll.bind
def hipExternalMemoryGetMappedBuffer(devPtr:ctypes.POINTER(ctypes.POINTER(None)), extMem:hipExternalMemory_t, bufferDesc:ctypes.POINTER(hipExternalMemoryBufferDesc)) -> hipError_t: ...
@dll.bind
def hipDestroyExternalMemory(extMem:hipExternalMemory_t) -> hipError_t: ...
hipMipmappedArray_t = ctypes.POINTER(hipMipmappedArray)
@dll.bind
def hipExternalMemoryGetMappedMipmappedArray(mipmap:ctypes.POINTER(hipMipmappedArray_t), extMem:hipExternalMemory_t, mipmapDesc:ctypes.POINTER(hipExternalMemoryMipmappedArrayDesc)) -> hipError_t: ...
@dll.bind
def hipMalloc(ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t) -> hipError_t: ...
@dll.bind
def hipExtMallocWithFlags(ptr:ctypes.POINTER(ctypes.POINTER(None)), sizeBytes:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMallocHost(ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t) -> hipError_t: ...
@dll.bind
def hipMemAllocHost(ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t) -> hipError_t: ...
@dll.bind
def hipHostMalloc(ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMallocManaged(dev_ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMemPrefetchAsync(dev_ptr:ctypes.POINTER(None), count:size_t, device:ctypes.c_int32, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemAdvise(dev_ptr:ctypes.POINTER(None), count:size_t, advice:hipMemoryAdvise, device:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipMemRangeGetAttribute(data:ctypes.POINTER(None), data_size:size_t, attribute:hipMemRangeAttribute, dev_ptr:ctypes.POINTER(None), count:size_t) -> hipError_t: ...
@dll.bind
def hipMemRangeGetAttributes(data:ctypes.POINTER(ctypes.POINTER(None)), data_sizes:ctypes.POINTER(size_t), attributes:ctypes.POINTER(hipMemRangeAttribute), num_attributes:size_t, dev_ptr:ctypes.POINTER(None), count:size_t) -> hipError_t: ...
@dll.bind
def hipStreamAttachMemAsync(stream:hipStream_t, dev_ptr:ctypes.POINTER(None), length:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMallocAsync(dev_ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipFreeAsync(dev_ptr:ctypes.POINTER(None), stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemPoolTrimTo(mem_pool:hipMemPool_t, min_bytes_to_hold:size_t) -> hipError_t: ...
@dll.bind
def hipMemPoolSetAttribute(mem_pool:hipMemPool_t, attr:hipMemPoolAttr, value:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMemPoolGetAttribute(mem_pool:hipMemPool_t, attr:hipMemPoolAttr, value:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMemPoolSetAccess(mem_pool:hipMemPool_t, desc_list:ctypes.POINTER(hipMemAccessDesc), count:size_t) -> hipError_t: ...
@dll.bind
def hipMemPoolGetAccess(flags:ctypes.POINTER(hipMemAccessFlags), mem_pool:hipMemPool_t, location:ctypes.POINTER(hipMemLocation)) -> hipError_t: ...
@dll.bind
def hipMemPoolCreate(mem_pool:ctypes.POINTER(hipMemPool_t), pool_props:ctypes.POINTER(hipMemPoolProps)) -> hipError_t: ...
@dll.bind
def hipMemPoolDestroy(mem_pool:hipMemPool_t) -> hipError_t: ...
@dll.bind
def hipMallocFromPoolAsync(dev_ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t, mem_pool:hipMemPool_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemPoolExportToShareableHandle(shared_handle:ctypes.POINTER(None), mem_pool:hipMemPool_t, handle_type:hipMemAllocationHandleType, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMemPoolImportFromShareableHandle(mem_pool:ctypes.POINTER(hipMemPool_t), shared_handle:ctypes.POINTER(None), handle_type:hipMemAllocationHandleType, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMemPoolExportPointer(export_data:ctypes.POINTER(hipMemPoolPtrExportData), dev_ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMemPoolImportPointer(dev_ptr:ctypes.POINTER(ctypes.POINTER(None)), mem_pool:hipMemPool_t, export_data:ctypes.POINTER(hipMemPoolPtrExportData)) -> hipError_t: ...
@dll.bind
def hipHostAlloc(ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipHostGetDevicePointer(devPtr:ctypes.POINTER(ctypes.POINTER(None)), hstPtr:ctypes.POINTER(None), flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipHostGetFlags(flagsPtr:ctypes.POINTER(ctypes.c_uint32), hostPtr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipHostRegister(hostPtr:ctypes.POINTER(None), sizeBytes:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipHostUnregister(hostPtr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMallocPitch(ptr:ctypes.POINTER(ctypes.POINTER(None)), pitch:ctypes.POINTER(size_t), width:size_t, height:size_t) -> hipError_t: ...
@dll.bind
def hipMemAllocPitch(dptr:ctypes.POINTER(hipDeviceptr_t), pitch:ctypes.POINTER(size_t), widthInBytes:size_t, height:size_t, elementSizeBytes:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipFree(ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipFreeHost(ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipHostFree(ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMemcpy(dst:ctypes.POINTER(None), src:ctypes.POINTER(None), sizeBytes:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyWithStream(dst:ctypes.POINTER(None), src:ctypes.POINTER(None), sizeBytes:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoD(dst:hipDeviceptr_t, src:ctypes.POINTER(None), sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoH(dst:ctypes.POINTER(None), src:hipDeviceptr_t, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoD(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoD(dstDevice:hipDeviceptr_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoA(dstArray:hipArray_t, dstOffset:size_t, srcDevice:hipDeviceptr_t, ByteCount:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoA(dstArray:hipArray_t, dstOffset:size_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoDAsync(dst:hipDeviceptr_t, src:ctypes.POINTER(None), sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoHAsync(dst:ctypes.POINTER(None), src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyDtoDAsync(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoHAsync(dstHost:ctypes.POINTER(None), srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoAAsync(dstArray:hipArray_t, dstOffset:size_t, srcHost:ctypes.POINTER(None), ByteCount:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipModuleGetGlobal(dptr:ctypes.POINTER(hipDeviceptr_t), bytes:ctypes.POINTER(size_t), hmod:hipModule_t, name:ctypes.POINTER(ctypes.c_char)) -> hipError_t: ...
@dll.bind
def hipGetSymbolAddress(devPtr:ctypes.POINTER(ctypes.POINTER(None)), symbol:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipGetSymbolSize(size:ctypes.POINTER(size_t), symbol:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipGetProcAddress(symbol:ctypes.POINTER(ctypes.c_char), pfn:ctypes.POINTER(ctypes.POINTER(None)), hipVersion:ctypes.c_int32, flags:uint64_t, symbolStatus:ctypes.POINTER(hipDriverProcAddressQueryResult)) -> hipError_t: ...
@dll.bind
def hipMemcpyToSymbol(symbol:ctypes.POINTER(None), src:ctypes.POINTER(None), sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyToSymbolAsync(symbol:ctypes.POINTER(None), src:ctypes.POINTER(None), sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyFromSymbol(dst:ctypes.POINTER(None), symbol:ctypes.POINTER(None), sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyFromSymbolAsync(dst:ctypes.POINTER(None), symbol:ctypes.POINTER(None), sizeBytes:size_t, offset:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAsync(dst:ctypes.POINTER(None), src:ctypes.POINTER(None), sizeBytes:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemset(dst:ctypes.POINTER(None), value:ctypes.c_int32, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetD8(dest:hipDeviceptr_t, value:ctypes.c_ubyte, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetD8Async(dest:hipDeviceptr_t, value:ctypes.c_ubyte, count:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemsetD16(dest:hipDeviceptr_t, value:ctypes.c_uint16, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetD16Async(dest:hipDeviceptr_t, value:ctypes.c_uint16, count:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemsetD32(dest:hipDeviceptr_t, value:ctypes.c_int32, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemsetAsync(dst:ctypes.POINTER(None), value:ctypes.c_int32, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemsetD32Async(dst:hipDeviceptr_t, value:ctypes.c_int32, count:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemset2D(dst:ctypes.POINTER(None), pitch:size_t, value:ctypes.c_int32, width:size_t, height:size_t) -> hipError_t: ...
@dll.bind
def hipMemset2DAsync(dst:ctypes.POINTER(None), pitch:size_t, value:ctypes.c_int32, width:size_t, height:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemset3D(pitchedDevPtr:hipPitchedPtr, value:ctypes.c_int32, extent:hipExtent) -> hipError_t: ...
@dll.bind
def hipMemset3DAsync(pitchedDevPtr:hipPitchedPtr, value:ctypes.c_int32, extent:hipExtent, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemGetInfo(free:ctypes.POINTER(size_t), total:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipMemPtrGetInfo(ptr:ctypes.POINTER(None), size:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipMallocArray(array:ctypes.POINTER(hipArray_t), desc:ctypes.POINTER(hipChannelFormatDesc), width:size_t, height:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@record
class HIP_ARRAY_DESCRIPTOR:
  SIZE = 24
  Width: Annotated[size_t, 0]
  Height: Annotated[size_t, 8]
  Format: Annotated[hipArray_Format, 16]
  NumChannels: Annotated[ctypes.c_uint32, 20]
@dll.bind
def hipArrayCreate(pHandle:ctypes.POINTER(hipArray_t), pAllocateArray:ctypes.POINTER(HIP_ARRAY_DESCRIPTOR)) -> hipError_t: ...
@dll.bind
def hipArrayDestroy(array:hipArray_t) -> hipError_t: ...
@record
class HIP_ARRAY3D_DESCRIPTOR:
  SIZE = 40
  Width: Annotated[size_t, 0]
  Height: Annotated[size_t, 8]
  Depth: Annotated[size_t, 16]
  Format: Annotated[hipArray_Format, 24]
  NumChannels: Annotated[ctypes.c_uint32, 28]
  Flags: Annotated[ctypes.c_uint32, 32]
@dll.bind
def hipArray3DCreate(array:ctypes.POINTER(hipArray_t), pAllocateArray:ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR)) -> hipError_t: ...
@dll.bind
def hipMalloc3D(pitchedDevPtr:ctypes.POINTER(hipPitchedPtr), extent:hipExtent) -> hipError_t: ...
@dll.bind
def hipFreeArray(array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipMalloc3DArray(array:ctypes.POINTER(hipArray_t), desc:ctypes.POINTER(hipChannelFormatDesc), extent:hipExtent, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipArrayGetInfo(desc:ctypes.POINTER(hipChannelFormatDesc), extent:ctypes.POINTER(hipExtent), flags:ctypes.POINTER(ctypes.c_uint32), array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipArrayGetDescriptor(pArrayDescriptor:ctypes.POINTER(HIP_ARRAY_DESCRIPTOR), array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipArray3DGetDescriptor(pArrayDescriptor:ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR), array:hipArray_t) -> hipError_t: ...
@dll.bind
def hipMemcpy2D(dst:ctypes.POINTER(None), dpitch:size_t, src:ctypes.POINTER(None), spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@record
class hip_Memcpy2D:
  SIZE = 128
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcMemoryType: Annotated[hipMemoryType, 16]
  srcHost: Annotated[ctypes.POINTER(None), 24]
  srcDevice: Annotated[hipDeviceptr_t, 32]
  srcArray: Annotated[hipArray_t, 40]
  srcPitch: Annotated[size_t, 48]
  dstXInBytes: Annotated[size_t, 56]
  dstY: Annotated[size_t, 64]
  dstMemoryType: Annotated[hipMemoryType, 72]
  dstHost: Annotated[ctypes.POINTER(None), 80]
  dstDevice: Annotated[hipDeviceptr_t, 88]
  dstArray: Annotated[hipArray_t, 96]
  dstPitch: Annotated[size_t, 104]
  WidthInBytes: Annotated[size_t, 112]
  Height: Annotated[size_t, 120]
@dll.bind
def hipMemcpyParam2D(pCopy:ctypes.POINTER(hip_Memcpy2D)) -> hipError_t: ...
@dll.bind
def hipMemcpyParam2DAsync(pCopy:ctypes.POINTER(hip_Memcpy2D), stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpy2DAsync(dst:ctypes.POINTER(None), dpitch:size_t, src:ctypes.POINTER(None), spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpy2DToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.POINTER(None), spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpy2DToArrayAsync(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.POINTER(None), spitch:size_t, width:size_t, height:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
hipArray_const_t = ctypes.POINTER(hipArray)
@dll.bind
def hipMemcpy2DArrayToArray(dst:hipArray_t, wOffsetDst:size_t, hOffsetDst:size_t, src:hipArray_const_t, wOffsetSrc:size_t, hOffsetSrc:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:ctypes.POINTER(None), count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpyFromArray(dst:ctypes.POINTER(None), srcArray:hipArray_const_t, wOffset:size_t, hOffset:size_t, count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpy2DFromArray(dst:ctypes.POINTER(None), dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipMemcpy2DFromArrayAsync(dst:ctypes.POINTER(None), dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:hipMemcpyKind, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemcpyAtoH(dst:ctypes.POINTER(None), srcArray:hipArray_t, srcOffset:size_t, count:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyHtoA(dstArray:hipArray_t, dstOffset:size_t, srcHost:ctypes.POINTER(None), count:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpy3D(p:ctypes.POINTER(hipMemcpy3DParms)) -> hipError_t: ...
@dll.bind
def hipMemcpy3DAsync(p:ctypes.POINTER(hipMemcpy3DParms), stream:hipStream_t) -> hipError_t: ...
@record
class HIP_MEMCPY3D:
  SIZE = 184
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcZ: Annotated[size_t, 16]
  srcLOD: Annotated[size_t, 24]
  srcMemoryType: Annotated[hipMemoryType, 32]
  srcHost: Annotated[ctypes.POINTER(None), 40]
  srcDevice: Annotated[hipDeviceptr_t, 48]
  srcArray: Annotated[hipArray_t, 56]
  srcPitch: Annotated[size_t, 64]
  srcHeight: Annotated[size_t, 72]
  dstXInBytes: Annotated[size_t, 80]
  dstY: Annotated[size_t, 88]
  dstZ: Annotated[size_t, 96]
  dstLOD: Annotated[size_t, 104]
  dstMemoryType: Annotated[hipMemoryType, 112]
  dstHost: Annotated[ctypes.POINTER(None), 120]
  dstDevice: Annotated[hipDeviceptr_t, 128]
  dstArray: Annotated[hipArray_t, 136]
  dstPitch: Annotated[size_t, 144]
  dstHeight: Annotated[size_t, 152]
  WidthInBytes: Annotated[size_t, 160]
  Height: Annotated[size_t, 168]
  Depth: Annotated[size_t, 176]
@dll.bind
def hipDrvMemcpy3D(pCopy:ctypes.POINTER(HIP_MEMCPY3D)) -> hipError_t: ...
@dll.bind
def hipDrvMemcpy3DAsync(pCopy:ctypes.POINTER(HIP_MEMCPY3D), stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipDeviceCanAccessPeer(canAccessPeer:ctypes.POINTER(ctypes.c_int32), deviceId:ctypes.c_int32, peerDeviceId:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipDeviceEnablePeerAccess(peerDeviceId:ctypes.c_int32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipDeviceDisablePeerAccess(peerDeviceId:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipMemGetAddressRange(pbase:ctypes.POINTER(hipDeviceptr_t), psize:ctypes.POINTER(size_t), dptr:hipDeviceptr_t) -> hipError_t: ...
@dll.bind
def hipMemcpyPeer(dst:ctypes.POINTER(None), dstDeviceId:ctypes.c_int32, src:ctypes.POINTER(None), srcDeviceId:ctypes.c_int32, sizeBytes:size_t) -> hipError_t: ...
@dll.bind
def hipMemcpyPeerAsync(dst:ctypes.POINTER(None), dstDeviceId:ctypes.c_int32, src:ctypes.POINTER(None), srcDevice:ctypes.c_int32, sizeBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipCtxCreate(ctx:ctypes.POINTER(hipCtx_t), flags:ctypes.c_uint32, device:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipCtxDestroy(ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipCtxPopCurrent(ctx:ctypes.POINTER(hipCtx_t)) -> hipError_t: ...
@dll.bind
def hipCtxPushCurrent(ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipCtxSetCurrent(ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipCtxGetCurrent(ctx:ctypes.POINTER(hipCtx_t)) -> hipError_t: ...
@dll.bind
def hipCtxGetDevice(device:ctypes.POINTER(hipDevice_t)) -> hipError_t: ...
@dll.bind
def hipCtxGetApiVersion(ctx:hipCtx_t, apiVersion:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipCtxGetCacheConfig(cacheConfig:ctypes.POINTER(hipFuncCache_t)) -> hipError_t: ...
@dll.bind
def hipCtxSetCacheConfig(cacheConfig:hipFuncCache_t) -> hipError_t: ...
@dll.bind
def hipCtxSetSharedMemConfig(config:hipSharedMemConfig) -> hipError_t: ...
@dll.bind
def hipCtxGetSharedMemConfig(pConfig:ctypes.POINTER(hipSharedMemConfig)) -> hipError_t: ...
@dll.bind
def hipCtxSynchronize() -> hipError_t: ...
@dll.bind
def hipCtxGetFlags(flags:ctypes.POINTER(ctypes.c_uint32)) -> hipError_t: ...
@dll.bind
def hipCtxEnablePeerAccess(peerCtx:hipCtx_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipCtxDisablePeerAccess(peerCtx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxGetState(dev:hipDevice_t, flags:ctypes.POINTER(ctypes.c_uint32), active:ctypes.POINTER(ctypes.c_int32)) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxRelease(dev:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxRetain(pctx:ctypes.POINTER(hipCtx_t), dev:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxReset(dev:hipDevice_t) -> hipError_t: ...
@dll.bind
def hipDevicePrimaryCtxSetFlags(dev:hipDevice_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipModuleLoad(module:ctypes.POINTER(hipModule_t), fname:ctypes.POINTER(ctypes.c_char)) -> hipError_t: ...
@dll.bind
def hipModuleUnload(module:hipModule_t) -> hipError_t: ...
@dll.bind
def hipModuleGetFunction(function:ctypes.POINTER(hipFunction_t), module:hipModule_t, kname:ctypes.POINTER(ctypes.c_char)) -> hipError_t: ...
@dll.bind
def hipFuncGetAttributes(attr:ctypes.POINTER(hipFuncAttributes), func:ctypes.POINTER(None)) -> hipError_t: ...
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

@dll.bind
def hipFuncGetAttribute(value:ctypes.POINTER(ctypes.c_int32), attrib:hipFunction_attribute, hfunc:hipFunction_t) -> hipError_t: ...
@dll.bind
def hipGetFuncBySymbol(functionPtr:ctypes.POINTER(hipFunction_t), symbolPtr:ctypes.POINTER(None)) -> hipError_t: ...
@record
class textureReference:
  SIZE = 88
  normalized: Annotated[ctypes.c_int32, 0]
  readMode: Annotated[hipTextureReadMode, 4]
  filterMode: Annotated[hipTextureFilterMode, 8]
  addressMode: Annotated[(hipTextureAddressMode* 3), 12]
  channelDesc: Annotated[hipChannelFormatDesc, 24]
  sRGB: Annotated[ctypes.c_int32, 44]
  maxAnisotropy: Annotated[ctypes.c_uint32, 48]
  mipmapFilterMode: Annotated[hipTextureFilterMode, 52]
  mipmapLevelBias: Annotated[ctypes.c_float, 56]
  minMipmapLevelClamp: Annotated[ctypes.c_float, 60]
  maxMipmapLevelClamp: Annotated[ctypes.c_float, 64]
  textureObject: Annotated[hipTextureObject_t, 72]
  numChannels: Annotated[ctypes.c_int32, 80]
  format: Annotated[hipArray_Format, 84]
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

class __hip_texture(ctypes.Structure): pass
hipTextureObject_t = ctypes.POINTER(__hip_texture)
@dll.bind
def hipModuleGetTexRef(texRef:ctypes.POINTER(ctypes.POINTER(textureReference)), hmod:hipModule_t, name:ctypes.POINTER(ctypes.c_char)) -> hipError_t: ...
@dll.bind
def hipModuleLoadData(module:ctypes.POINTER(hipModule_t), image:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipModuleLoadDataEx(module:ctypes.POINTER(hipModule_t), image:ctypes.POINTER(None), numOptions:ctypes.c_uint32, options:ctypes.POINTER(hipJitOption), optionValues:ctypes.POINTER(ctypes.POINTER(None))) -> hipError_t: ...
@dll.bind
def hipModuleLaunchKernel(f:hipFunction_t, gridDimX:ctypes.c_uint32, gridDimY:ctypes.c_uint32, gridDimZ:ctypes.c_uint32, blockDimX:ctypes.c_uint32, blockDimY:ctypes.c_uint32, blockDimZ:ctypes.c_uint32, sharedMemBytes:ctypes.c_uint32, stream:hipStream_t, kernelParams:ctypes.POINTER(ctypes.POINTER(None)), extra:ctypes.POINTER(ctypes.POINTER(None))) -> hipError_t: ...
@dll.bind
def hipModuleLaunchCooperativeKernel(f:hipFunction_t, gridDimX:ctypes.c_uint32, gridDimY:ctypes.c_uint32, gridDimZ:ctypes.c_uint32, blockDimX:ctypes.c_uint32, blockDimY:ctypes.c_uint32, blockDimZ:ctypes.c_uint32, sharedMemBytes:ctypes.c_uint32, stream:hipStream_t, kernelParams:ctypes.POINTER(ctypes.POINTER(None))) -> hipError_t: ...
@dll.bind
def hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList:ctypes.POINTER(hipFunctionLaunchParams), numDevices:ctypes.c_uint32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipLaunchCooperativeKernel(f:ctypes.POINTER(None), gridDim:dim3, blockDimX:dim3, kernelParams:ctypes.POINTER(ctypes.POINTER(None)), sharedMemBytes:ctypes.c_uint32, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipLaunchCooperativeKernelMultiDevice(launchParamsList:ctypes.POINTER(hipLaunchParams), numDevices:ctypes.c_int32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipExtLaunchMultiKernelMultiDevice(launchParamsList:ctypes.POINTER(hipLaunchParams), numDevices:ctypes.c_int32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxPotentialBlockSize(gridSize:ctypes.POINTER(ctypes.c_int32), blockSize:ctypes.POINTER(ctypes.c_int32), f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize:ctypes.POINTER(ctypes.c_int32), blockSize:ctypes.POINTER(ctypes.c_int32), f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:ctypes.c_int32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:ctypes.POINTER(ctypes.c_int32), f:hipFunction_t, blockSize:ctypes.c_int32, dynSharedMemPerBlk:size_t) -> hipError_t: ...
@dll.bind
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:ctypes.POINTER(ctypes.c_int32), f:hipFunction_t, blockSize:ctypes.c_int32, dynSharedMemPerBlk:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:ctypes.POINTER(ctypes.c_int32), f:ctypes.POINTER(None), blockSize:ctypes.c_int32, dynSharedMemPerBlk:size_t) -> hipError_t: ...
@dll.bind
def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:ctypes.POINTER(ctypes.c_int32), f:ctypes.POINTER(None), blockSize:ctypes.c_int32, dynSharedMemPerBlk:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipOccupancyMaxPotentialBlockSize(gridSize:ctypes.POINTER(ctypes.c_int32), blockSize:ctypes.POINTER(ctypes.c_int32), f:ctypes.POINTER(None), dynSharedMemPerBlk:size_t, blockSizeLimit:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipProfilerStart() -> hipError_t: ...
@dll.bind
def hipProfilerStop() -> hipError_t: ...
@dll.bind
def hipConfigureCall(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipSetupArgument(arg:ctypes.POINTER(None), size:size_t, offset:size_t) -> hipError_t: ...
@dll.bind
def hipLaunchByPtr(func:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def __hipPushCallConfiguration(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def __hipPopCallConfiguration(gridDim:ctypes.POINTER(dim3), blockDim:ctypes.POINTER(dim3), sharedMem:ctypes.POINTER(size_t), stream:ctypes.POINTER(hipStream_t)) -> hipError_t: ...
@dll.bind
def hipLaunchKernel(function_address:ctypes.POINTER(None), numBlocks:dim3, dimBlocks:dim3, args:ctypes.POINTER(ctypes.POINTER(None)), sharedMemBytes:size_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipDrvMemcpy2DUnaligned(pCopy:ctypes.POINTER(hip_Memcpy2D)) -> hipError_t: ...
@record
class hipResourceDesc:
  SIZE = 64
  resType: Annotated[hipResourceType, 0]
  res: Annotated[_anonunion19, 8]
@record
class _anonunion19:
  SIZE = 56
  array: Annotated[_anonstruct20, 0]
  mipmap: Annotated[_anonstruct21, 0]
  linear: Annotated[_anonstruct22, 0]
  pitch2D: Annotated[_anonstruct23, 0]
@record
class _anonstruct20:
  SIZE = 8
  array: Annotated[hipArray_t, 0]
@record
class _anonstruct21:
  SIZE = 8
  mipmap: Annotated[hipMipmappedArray_t, 0]
@record
class _anonstruct22:
  SIZE = 40
  devPtr: Annotated[ctypes.POINTER(None), 0]
  desc: Annotated[hipChannelFormatDesc, 8]
  sizeInBytes: Annotated[size_t, 32]
@record
class _anonstruct23:
  SIZE = 56
  devPtr: Annotated[ctypes.POINTER(None), 0]
  desc: Annotated[hipChannelFormatDesc, 8]
  width: Annotated[size_t, 32]
  height: Annotated[size_t, 40]
  pitchInBytes: Annotated[size_t, 48]
@record
class hipTextureDesc:
  SIZE = 64
  addressMode: Annotated[(hipTextureAddressMode* 3), 0]
  filterMode: Annotated[hipTextureFilterMode, 12]
  readMode: Annotated[hipTextureReadMode, 16]
  sRGB: Annotated[ctypes.c_int32, 20]
  borderColor: Annotated[(ctypes.c_float* 4), 24]
  normalizedCoords: Annotated[ctypes.c_int32, 40]
  maxAnisotropy: Annotated[ctypes.c_uint32, 44]
  mipmapFilterMode: Annotated[hipTextureFilterMode, 48]
  mipmapLevelBias: Annotated[ctypes.c_float, 52]
  minMipmapLevelClamp: Annotated[ctypes.c_float, 56]
  maxMipmapLevelClamp: Annotated[ctypes.c_float, 60]
@record
class hipResourceViewDesc:
  SIZE = 48
  format: Annotated[hipResourceViewFormat, 0]
  width: Annotated[size_t, 8]
  height: Annotated[size_t, 16]
  depth: Annotated[size_t, 24]
  firstMipmapLevel: Annotated[ctypes.c_uint32, 32]
  lastMipmapLevel: Annotated[ctypes.c_uint32, 36]
  firstLayer: Annotated[ctypes.c_uint32, 40]
  lastLayer: Annotated[ctypes.c_uint32, 44]
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

@dll.bind
def hipCreateTextureObject(pTexObject:ctypes.POINTER(hipTextureObject_t), pResDesc:ctypes.POINTER(hipResourceDesc), pTexDesc:ctypes.POINTER(hipTextureDesc), pResViewDesc:ctypes.POINTER(hipResourceViewDesc)) -> hipError_t: ...
@dll.bind
def hipDestroyTextureObject(textureObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipGetChannelDesc(desc:ctypes.POINTER(hipChannelFormatDesc), array:hipArray_const_t) -> hipError_t: ...
@dll.bind
def hipGetTextureObjectResourceDesc(pResDesc:ctypes.POINTER(hipResourceDesc), textureObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipGetTextureObjectResourceViewDesc(pResViewDesc:ctypes.POINTER(hipResourceViewDesc), textureObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipGetTextureObjectTextureDesc(pTexDesc:ctypes.POINTER(hipTextureDesc), textureObject:hipTextureObject_t) -> hipError_t: ...
@record
class HIP_RESOURCE_DESC_st:
  SIZE = 144
  resType: Annotated[HIPresourcetype, 0]
  res: Annotated[_anonunion24, 8]
  flags: Annotated[ctypes.c_uint32, 136]
HIP_RESOURCE_DESC = HIP_RESOURCE_DESC_st
HIPresourcetype_enum = CEnum(ctypes.c_uint32)
HIP_RESOURCE_TYPE_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_ARRAY', 0)
HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
HIP_RESOURCE_TYPE_LINEAR = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_LINEAR', 2)
HIP_RESOURCE_TYPE_PITCH2D = HIPresourcetype_enum.define('HIP_RESOURCE_TYPE_PITCH2D', 3)

HIPresourcetype = HIPresourcetype_enum
@record
class _anonunion24:
  SIZE = 128
  array: Annotated[_anonstruct25, 0]
  mipmap: Annotated[_anonstruct26, 0]
  linear: Annotated[_anonstruct27, 0]
  pitch2D: Annotated[_anonstruct28, 0]
  reserved: Annotated[_anonstruct29, 0]
@record
class _anonstruct25:
  SIZE = 8
  hArray: Annotated[hipArray_t, 0]
@record
class _anonstruct26:
  SIZE = 8
  hMipmappedArray: Annotated[hipMipmappedArray_t, 0]
@record
class _anonstruct27:
  SIZE = 24
  devPtr: Annotated[hipDeviceptr_t, 0]
  format: Annotated[hipArray_Format, 8]
  numChannels: Annotated[ctypes.c_uint32, 12]
  sizeInBytes: Annotated[size_t, 16]
@record
class _anonstruct28:
  SIZE = 40
  devPtr: Annotated[hipDeviceptr_t, 0]
  format: Annotated[hipArray_Format, 8]
  numChannels: Annotated[ctypes.c_uint32, 12]
  width: Annotated[size_t, 16]
  height: Annotated[size_t, 24]
  pitchInBytes: Annotated[size_t, 32]
@record
class _anonstruct29:
  SIZE = 128
  reserved: Annotated[(ctypes.c_int32* 32), 0]
@record
class HIP_TEXTURE_DESC_st:
  SIZE = 104
  addressMode: Annotated[(HIPaddress_mode* 3), 0]
  filterMode: Annotated[HIPfilter_mode, 12]
  flags: Annotated[ctypes.c_uint32, 16]
  maxAnisotropy: Annotated[ctypes.c_uint32, 20]
  mipmapFilterMode: Annotated[HIPfilter_mode, 24]
  mipmapLevelBias: Annotated[ctypes.c_float, 28]
  minMipmapLevelClamp: Annotated[ctypes.c_float, 32]
  maxMipmapLevelClamp: Annotated[ctypes.c_float, 36]
  borderColor: Annotated[(ctypes.c_float* 4), 40]
  reserved: Annotated[(ctypes.c_int32* 12), 56]
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
@record
class HIP_RESOURCE_VIEW_DESC_st:
  SIZE = 112
  format: Annotated[HIPresourceViewFormat, 0]
  width: Annotated[size_t, 8]
  height: Annotated[size_t, 16]
  depth: Annotated[size_t, 24]
  firstMipmapLevel: Annotated[ctypes.c_uint32, 32]
  lastMipmapLevel: Annotated[ctypes.c_uint32, 36]
  firstLayer: Annotated[ctypes.c_uint32, 40]
  lastLayer: Annotated[ctypes.c_uint32, 44]
  reserved: Annotated[(ctypes.c_uint32* 16), 48]
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
@dll.bind
def hipTexObjectCreate(pTexObject:ctypes.POINTER(hipTextureObject_t), pResDesc:ctypes.POINTER(HIP_RESOURCE_DESC), pTexDesc:ctypes.POINTER(HIP_TEXTURE_DESC), pResViewDesc:ctypes.POINTER(HIP_RESOURCE_VIEW_DESC)) -> hipError_t: ...
@dll.bind
def hipTexObjectDestroy(texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipTexObjectGetResourceDesc(pResDesc:ctypes.POINTER(HIP_RESOURCE_DESC), texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipTexObjectGetResourceViewDesc(pResViewDesc:ctypes.POINTER(HIP_RESOURCE_VIEW_DESC), texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipTexObjectGetTextureDesc(pTexDesc:ctypes.POINTER(HIP_TEXTURE_DESC), texObject:hipTextureObject_t) -> hipError_t: ...
@dll.bind
def hipMallocMipmappedArray(mipmappedArray:ctypes.POINTER(hipMipmappedArray_t), desc:ctypes.POINTER(hipChannelFormatDesc), extent:hipExtent, numLevels:ctypes.c_uint32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipFreeMipmappedArray(mipmappedArray:hipMipmappedArray_t) -> hipError_t: ...
hipMipmappedArray_const_t = ctypes.POINTER(hipMipmappedArray)
@dll.bind
def hipGetMipmappedArrayLevel(levelArray:ctypes.POINTER(hipArray_t), mipmappedArray:hipMipmappedArray_const_t, level:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMipmappedArrayCreate(pHandle:ctypes.POINTER(hipMipmappedArray_t), pMipmappedArrayDesc:ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR), numMipmapLevels:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipMipmappedArrayDestroy(hMipmappedArray:hipMipmappedArray_t) -> hipError_t: ...
@dll.bind
def hipMipmappedArrayGetLevel(pLevelArray:ctypes.POINTER(hipArray_t), hMipMappedArray:hipMipmappedArray_t, level:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipBindTextureToMipmappedArray(tex:ctypes.POINTER(textureReference), mipmappedArray:hipMipmappedArray_const_t, desc:ctypes.POINTER(hipChannelFormatDesc)) -> hipError_t: ...
@dll.bind
def hipGetTextureReference(texref:ctypes.POINTER(ctypes.POINTER(textureReference)), symbol:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipTexRefGetBorderColor(pBorderColor:ctypes.POINTER(ctypes.c_float), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetArray(pArray:ctypes.POINTER(hipArray_t), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefSetAddressMode(texRef:ctypes.POINTER(textureReference), dim:ctypes.c_int32, am:hipTextureAddressMode) -> hipError_t: ...
@dll.bind
def hipTexRefSetArray(tex:ctypes.POINTER(textureReference), array:hipArray_const_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipTexRefSetFilterMode(texRef:ctypes.POINTER(textureReference), fm:hipTextureFilterMode) -> hipError_t: ...
@dll.bind
def hipTexRefSetFlags(texRef:ctypes.POINTER(textureReference), Flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipTexRefSetFormat(texRef:ctypes.POINTER(textureReference), fmt:hipArray_Format, NumPackedComponents:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipBindTexture(offset:ctypes.POINTER(size_t), tex:ctypes.POINTER(textureReference), devPtr:ctypes.POINTER(None), desc:ctypes.POINTER(hipChannelFormatDesc), size:size_t) -> hipError_t: ...
@dll.bind
def hipBindTexture2D(offset:ctypes.POINTER(size_t), tex:ctypes.POINTER(textureReference), devPtr:ctypes.POINTER(None), desc:ctypes.POINTER(hipChannelFormatDesc), width:size_t, height:size_t, pitch:size_t) -> hipError_t: ...
@dll.bind
def hipBindTextureToArray(tex:ctypes.POINTER(textureReference), array:hipArray_const_t, desc:ctypes.POINTER(hipChannelFormatDesc)) -> hipError_t: ...
@dll.bind
def hipGetTextureAlignmentOffset(offset:ctypes.POINTER(size_t), texref:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipUnbindTexture(tex:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetAddress(dev_ptr:ctypes.POINTER(hipDeviceptr_t), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetAddressMode(pam:ctypes.POINTER(hipTextureAddressMode), texRef:ctypes.POINTER(textureReference), dim:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipTexRefGetFilterMode(pfm:ctypes.POINTER(hipTextureFilterMode), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetFlags(pFlags:ctypes.POINTER(ctypes.c_uint32), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetFormat(pFormat:ctypes.POINTER(hipArray_Format), pNumChannels:ctypes.POINTER(ctypes.c_int32), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetMaxAnisotropy(pmaxAnsio:ctypes.POINTER(ctypes.c_int32), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipmapFilterMode(pfm:ctypes.POINTER(hipTextureFilterMode), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipmapLevelBias(pbias:ctypes.POINTER(ctypes.c_float), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp:ctypes.POINTER(ctypes.c_float), pmaxMipmapLevelClamp:ctypes.POINTER(ctypes.c_float), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefGetMipMappedArray(pArray:ctypes.POINTER(hipMipmappedArray_t), texRef:ctypes.POINTER(textureReference)) -> hipError_t: ...
@dll.bind
def hipTexRefSetAddress(ByteOffset:ctypes.POINTER(size_t), texRef:ctypes.POINTER(textureReference), dptr:hipDeviceptr_t, bytes:size_t) -> hipError_t: ...
@dll.bind
def hipTexRefSetAddress2D(texRef:ctypes.POINTER(textureReference), desc:ctypes.POINTER(HIP_ARRAY_DESCRIPTOR), dptr:hipDeviceptr_t, Pitch:size_t) -> hipError_t: ...
@dll.bind
def hipTexRefSetMaxAnisotropy(texRef:ctypes.POINTER(textureReference), maxAniso:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipTexRefSetBorderColor(texRef:ctypes.POINTER(textureReference), pBorderColor:ctypes.POINTER(ctypes.c_float)) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmapFilterMode(texRef:ctypes.POINTER(textureReference), fm:hipTextureFilterMode) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmapLevelBias(texRef:ctypes.POINTER(textureReference), bias:ctypes.c_float) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmapLevelClamp(texRef:ctypes.POINTER(textureReference), minMipMapLevelClamp:ctypes.c_float, maxMipMapLevelClamp:ctypes.c_float) -> hipError_t: ...
@dll.bind
def hipTexRefSetMipmappedArray(texRef:ctypes.POINTER(textureReference), mipmappedArray:ctypes.POINTER(hipMipmappedArray), Flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipApiName(id:uint32_t) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def hipKernelNameRef(f:hipFunction_t) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def hipKernelNameRefByPtr(hostFunction:ctypes.POINTER(None), stream:hipStream_t) -> ctypes.POINTER(ctypes.c_char): ...
@dll.bind
def hipGetStreamDeviceId(stream:hipStream_t) -> ctypes.c_int32: ...
@dll.bind
def hipStreamBeginCapture(stream:hipStream_t, mode:hipStreamCaptureMode) -> hipError_t: ...
@dll.bind
def hipStreamBeginCaptureToGraph(stream:hipStream_t, graph:hipGraph_t, dependencies:ctypes.POINTER(hipGraphNode_t), dependencyData:ctypes.POINTER(hipGraphEdgeData), numDependencies:size_t, mode:hipStreamCaptureMode) -> hipError_t: ...
@dll.bind
def hipStreamEndCapture(stream:hipStream_t, pGraph:ctypes.POINTER(hipGraph_t)) -> hipError_t: ...
@dll.bind
def hipStreamGetCaptureInfo(stream:hipStream_t, pCaptureStatus:ctypes.POINTER(hipStreamCaptureStatus), pId:ctypes.POINTER(ctypes.c_uint64)) -> hipError_t: ...
@dll.bind
def hipStreamGetCaptureInfo_v2(stream:hipStream_t, captureStatus_out:ctypes.POINTER(hipStreamCaptureStatus), id_out:ctypes.POINTER(ctypes.c_uint64), graph_out:ctypes.POINTER(hipGraph_t), dependencies_out:ctypes.POINTER(ctypes.POINTER(hipGraphNode_t)), numDependencies_out:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipStreamIsCapturing(stream:hipStream_t, pCaptureStatus:ctypes.POINTER(hipStreamCaptureStatus)) -> hipError_t: ...
@dll.bind
def hipStreamUpdateCaptureDependencies(stream:hipStream_t, dependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipThreadExchangeStreamCaptureMode(mode:ctypes.POINTER(hipStreamCaptureMode)) -> hipError_t: ...
@dll.bind
def hipGraphCreate(pGraph:ctypes.POINTER(hipGraph_t), flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphDestroy(graph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphAddDependencies(graph:hipGraph_t, _from:ctypes.POINTER(hipGraphNode_t), to:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t) -> hipError_t: ...
@dll.bind
def hipGraphRemoveDependencies(graph:hipGraph_t, _from:ctypes.POINTER(hipGraphNode_t), to:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t) -> hipError_t: ...
@dll.bind
def hipGraphGetEdges(graph:hipGraph_t, _from:ctypes.POINTER(hipGraphNode_t), to:ctypes.POINTER(hipGraphNode_t), numEdges:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipGraphGetNodes(graph:hipGraph_t, nodes:ctypes.POINTER(hipGraphNode_t), numNodes:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipGraphGetRootNodes(graph:hipGraph_t, pRootNodes:ctypes.POINTER(hipGraphNode_t), pNumRootNodes:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetDependencies(node:hipGraphNode_t, pDependencies:ctypes.POINTER(hipGraphNode_t), pNumDependencies:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetDependentNodes(node:hipGraphNode_t, pDependentNodes:ctypes.POINTER(hipGraphNode_t), pNumDependentNodes:ctypes.POINTER(size_t)) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetType(node:hipGraphNode_t, pType:ctypes.POINTER(hipGraphNodeType)) -> hipError_t: ...
@dll.bind
def hipGraphDestroyNode(node:hipGraphNode_t) -> hipError_t: ...
@dll.bind
def hipGraphClone(pGraphClone:ctypes.POINTER(hipGraph_t), originalGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphNodeFindInClone(pNode:ctypes.POINTER(hipGraphNode_t), originalNode:hipGraphNode_t, clonedGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphInstantiate(pGraphExec:ctypes.POINTER(hipGraphExec_t), graph:hipGraph_t, pErrorNode:ctypes.POINTER(hipGraphNode_t), pLogBuffer:ctypes.POINTER(ctypes.c_char), bufferSize:size_t) -> hipError_t: ...
@dll.bind
def hipGraphInstantiateWithFlags(pGraphExec:ctypes.POINTER(hipGraphExec_t), graph:hipGraph_t, flags:ctypes.c_uint64) -> hipError_t: ...
@dll.bind
def hipGraphInstantiateWithParams(pGraphExec:ctypes.POINTER(hipGraphExec_t), graph:hipGraph_t, instantiateParams:ctypes.POINTER(hipGraphInstantiateParams)) -> hipError_t: ...
@dll.bind
def hipGraphLaunch(graphExec:hipGraphExec_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphUpload(graphExec:hipGraphExec_t, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphAddNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, nodeParams:ctypes.POINTER(hipGraphNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExecDestroy(graphExec:hipGraphExec_t) -> hipError_t: ...
@dll.bind
def hipGraphExecUpdate(hGraphExec:hipGraphExec_t, hGraph:hipGraph_t, hErrorNode_out:ctypes.POINTER(hipGraphNode_t), updateResult_out:ctypes.POINTER(hipGraphExecUpdateResult)) -> hipError_t: ...
@dll.bind
def hipGraphAddKernelNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, pNodeParams:ctypes.POINTER(hipKernelNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipKernelNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipKernelNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExecKernelNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipKernelNodeParams)) -> hipError_t: ...
@dll.bind
def hipDrvGraphAddMemcpyNode(phGraphNode:ctypes.POINTER(hipGraphNode_t), hGraph:hipGraph_t, dependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, copyParams:ctypes.POINTER(HIP_MEMCPY3D), ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, pCopyParams:ctypes.POINTER(hipMemcpy3DParms)) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemcpy3DParms)) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemcpy3DParms)) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeSetAttribute(hNode:hipGraphNode_t, attr:hipLaunchAttributeID, value:ctypes.POINTER(hipLaunchAttributeValue)) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeGetAttribute(hNode:hipGraphNode_t, attr:hipLaunchAttributeID, value:ctypes.POINTER(hipLaunchAttributeValue)) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemcpy3DParms)) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNode1D(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, dst:ctypes.POINTER(None), src:ctypes.POINTER(None), count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParams1D(node:hipGraphNode_t, dst:ctypes.POINTER(None), src:ctypes.POINTER(None), count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParams1D(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:ctypes.POINTER(None), src:ctypes.POINTER(None), count:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNodeFromSymbol(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, dst:ctypes.POINTER(None), symbol:ctypes.POINTER(None), count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParamsFromSymbol(node:hipGraphNode_t, dst:ctypes.POINTER(None), symbol:ctypes.POINTER(None), count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:ctypes.POINTER(None), symbol:ctypes.POINTER(None), count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphAddMemcpyNodeToSymbol(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, symbol:ctypes.POINTER(None), src:ctypes.POINTER(None), count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphMemcpyNodeSetParamsToSymbol(node:hipGraphNode_t, symbol:ctypes.POINTER(None), src:ctypes.POINTER(None), count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, symbol:ctypes.POINTER(None), src:ctypes.POINTER(None), count:size_t, offset:size_t, kind:hipMemcpyKind) -> hipError_t: ...
@dll.bind
def hipGraphAddMemsetNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, pMemsetParams:ctypes.POINTER(hipMemsetParams)) -> hipError_t: ...
@dll.bind
def hipGraphMemsetNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemsetParams)) -> hipError_t: ...
@dll.bind
def hipGraphMemsetNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemsetParams)) -> hipError_t: ...
@dll.bind
def hipGraphExecMemsetNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemsetParams)) -> hipError_t: ...
@dll.bind
def hipGraphAddHostNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, pNodeParams:ctypes.POINTER(hipHostNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphHostNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipHostNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphHostNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipHostNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExecHostNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipHostNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphAddChildGraphNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, childGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphChildGraphNodeGetGraph(node:hipGraphNode_t, pGraph:ctypes.POINTER(hipGraph_t)) -> hipError_t: ...
@dll.bind
def hipGraphExecChildGraphNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, childGraph:hipGraph_t) -> hipError_t: ...
@dll.bind
def hipGraphAddEmptyNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t) -> hipError_t: ...
@dll.bind
def hipGraphAddEventRecordNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphEventRecordNodeGetEvent(node:hipGraphNode_t, event_out:ctypes.POINTER(hipEvent_t)) -> hipError_t: ...
@dll.bind
def hipGraphEventRecordNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphExecEventRecordNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphAddEventWaitNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphEventWaitNodeGetEvent(node:hipGraphNode_t, event_out:ctypes.POINTER(hipEvent_t)) -> hipError_t: ...
@dll.bind
def hipGraphEventWaitNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphExecEventWaitNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> hipError_t: ...
@dll.bind
def hipGraphAddMemAllocNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, pNodeParams:ctypes.POINTER(hipMemAllocNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphMemAllocNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes.POINTER(hipMemAllocNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphAddMemFreeNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, dev_ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipGraphMemFreeNodeGetParams(node:hipGraphNode_t, dev_ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipDeviceGetGraphMemAttribute(device:ctypes.c_int32, attr:hipGraphMemAttributeType, value:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipDeviceSetGraphMemAttribute(device:ctypes.c_int32, attr:hipGraphMemAttributeType, value:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipDeviceGraphMemTrim(device:ctypes.c_int32) -> hipError_t: ...
@dll.bind
def hipUserObjectRelease(object:hipUserObject_t, count:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipUserObjectRetain(object:hipUserObject_t, count:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphRetainUserObject(graph:hipGraph_t, object:hipUserObject_t, count:ctypes.c_uint32, flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphReleaseUserObject(graph:hipGraph_t, object:hipUserObject_t, count:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphDebugDotPrint(graph:hipGraph_t, path:ctypes.POINTER(ctypes.c_char), flags:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphKernelNodeCopyAttributes(hSrc:hipGraphNode_t, hDst:hipGraphNode_t) -> hipError_t: ...
@dll.bind
def hipGraphNodeSetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphNodeGetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:ctypes.POINTER(ctypes.c_uint32)) -> hipError_t: ...
@dll.bind
def hipGraphAddExternalSemaphoresWaitNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, nodeParams:ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphAddExternalSemaphoresSignalNode(pGraphNode:ctypes.POINTER(hipGraphNode_t), graph:hipGraph_t, pDependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, nodeParams:ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresSignalNodeSetParams(hNode:hipGraphNode_t, nodeParams:ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresWaitNodeSetParams(hNode:hipGraphNode_t, nodeParams:ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresSignalNodeGetParams(hNode:hipGraphNode_t, params_out:ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExternalSemaphoresWaitNodeGetParams(hNode:hipGraphNode_t, params_out:ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:ctypes.POINTER(hipExternalSemaphoreSignalNodeParams)) -> hipError_t: ...
@dll.bind
def hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:ctypes.POINTER(hipExternalSemaphoreWaitNodeParams)) -> hipError_t: ...
@dll.bind
def hipDrvGraphAddMemsetNode(phGraphNode:ctypes.POINTER(hipGraphNode_t), hGraph:hipGraph_t, dependencies:ctypes.POINTER(hipGraphNode_t), numDependencies:size_t, memsetParams:ctypes.POINTER(HIP_MEMSET_NODE_PARAMS), ctx:hipCtx_t) -> hipError_t: ...
@dll.bind
def hipMemAddressFree(devPtr:ctypes.POINTER(None), size:size_t) -> hipError_t: ...
@dll.bind
def hipMemAddressReserve(ptr:ctypes.POINTER(ctypes.POINTER(None)), size:size_t, alignment:size_t, addr:ctypes.POINTER(None), flags:ctypes.c_uint64) -> hipError_t: ...
@dll.bind
def hipMemCreate(handle:ctypes.POINTER(hipMemGenericAllocationHandle_t), size:size_t, prop:ctypes.POINTER(hipMemAllocationProp), flags:ctypes.c_uint64) -> hipError_t: ...
@dll.bind
def hipMemExportToShareableHandle(shareableHandle:ctypes.POINTER(None), handle:hipMemGenericAllocationHandle_t, handleType:hipMemAllocationHandleType, flags:ctypes.c_uint64) -> hipError_t: ...
@dll.bind
def hipMemGetAccess(flags:ctypes.POINTER(ctypes.c_uint64), location:ctypes.POINTER(hipMemLocation), ptr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMemGetAllocationGranularity(granularity:ctypes.POINTER(size_t), prop:ctypes.POINTER(hipMemAllocationProp), option:hipMemAllocationGranularity_flags) -> hipError_t: ...
@dll.bind
def hipMemGetAllocationPropertiesFromHandle(prop:ctypes.POINTER(hipMemAllocationProp), handle:hipMemGenericAllocationHandle_t) -> hipError_t: ...
@dll.bind
def hipMemImportFromShareableHandle(handle:ctypes.POINTER(hipMemGenericAllocationHandle_t), osHandle:ctypes.POINTER(None), shHandleType:hipMemAllocationHandleType) -> hipError_t: ...
@dll.bind
def hipMemMap(ptr:ctypes.POINTER(None), size:size_t, offset:size_t, handle:hipMemGenericAllocationHandle_t, flags:ctypes.c_uint64) -> hipError_t: ...
@dll.bind
def hipMemMapArrayAsync(mapInfoList:ctypes.POINTER(hipArrayMapInfo), count:ctypes.c_uint32, stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipMemRelease(handle:hipMemGenericAllocationHandle_t) -> hipError_t: ...
@dll.bind
def hipMemRetainAllocationHandle(handle:ctypes.POINTER(hipMemGenericAllocationHandle_t), addr:ctypes.POINTER(None)) -> hipError_t: ...
@dll.bind
def hipMemSetAccess(ptr:ctypes.POINTER(None), size:size_t, desc:ctypes.POINTER(hipMemAccessDesc), count:size_t) -> hipError_t: ...
@dll.bind
def hipMemUnmap(ptr:ctypes.POINTER(None), size:size_t) -> hipError_t: ...
@dll.bind
def hipGraphicsMapResources(count:ctypes.c_int32, resources:ctypes.POINTER(hipGraphicsResource_t), stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphicsSubResourceGetMappedArray(array:ctypes.POINTER(hipArray_t), resource:hipGraphicsResource_t, arrayIndex:ctypes.c_uint32, mipLevel:ctypes.c_uint32) -> hipError_t: ...
@dll.bind
def hipGraphicsResourceGetMappedPointer(devPtr:ctypes.POINTER(ctypes.POINTER(None)), size:ctypes.POINTER(size_t), resource:hipGraphicsResource_t) -> hipError_t: ...
@dll.bind
def hipGraphicsUnmapResources(count:ctypes.c_int32, resources:ctypes.POINTER(hipGraphicsResource_t), stream:hipStream_t) -> hipError_t: ...
@dll.bind
def hipGraphicsUnregisterResource(resource:hipGraphicsResource_t) -> hipError_t: ...
class __hip_surface(ctypes.Structure): pass
hipSurfaceObject_t = ctypes.POINTER(__hip_surface)
@dll.bind
def hipCreateSurfaceObject(pSurfObject:ctypes.POINTER(hipSurfaceObject_t), pResDesc:ctypes.POINTER(hipResourceDesc)) -> hipError_t: ...
@dll.bind
def hipDestroySurfaceObject(surfaceObject:hipSurfaceObject_t) -> hipError_t: ...
hipmipmappedArray = ctypes.POINTER(hipMipmappedArray)
hipResourcetype = HIPresourcetype_enum
init_records()
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