# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import os
dll = c.DLL('hip', os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so')
hipError_t: dict[int, str] = {(hipSuccess:=0): 'hipSuccess', (hipErrorInvalidValue:=1): 'hipErrorInvalidValue', (hipErrorOutOfMemory:=2): 'hipErrorOutOfMemory', (hipErrorMemoryAllocation:=2): 'hipErrorMemoryAllocation', (hipErrorNotInitialized:=3): 'hipErrorNotInitialized', (hipErrorInitializationError:=3): 'hipErrorInitializationError', (hipErrorDeinitialized:=4): 'hipErrorDeinitialized', (hipErrorProfilerDisabled:=5): 'hipErrorProfilerDisabled', (hipErrorProfilerNotInitialized:=6): 'hipErrorProfilerNotInitialized', (hipErrorProfilerAlreadyStarted:=7): 'hipErrorProfilerAlreadyStarted', (hipErrorProfilerAlreadyStopped:=8): 'hipErrorProfilerAlreadyStopped', (hipErrorInvalidConfiguration:=9): 'hipErrorInvalidConfiguration', (hipErrorInvalidPitchValue:=12): 'hipErrorInvalidPitchValue', (hipErrorInvalidSymbol:=13): 'hipErrorInvalidSymbol', (hipErrorInvalidDevicePointer:=17): 'hipErrorInvalidDevicePointer', (hipErrorInvalidMemcpyDirection:=21): 'hipErrorInvalidMemcpyDirection', (hipErrorInsufficientDriver:=35): 'hipErrorInsufficientDriver', (hipErrorMissingConfiguration:=52): 'hipErrorMissingConfiguration', (hipErrorPriorLaunchFailure:=53): 'hipErrorPriorLaunchFailure', (hipErrorInvalidDeviceFunction:=98): 'hipErrorInvalidDeviceFunction', (hipErrorNoDevice:=100): 'hipErrorNoDevice', (hipErrorInvalidDevice:=101): 'hipErrorInvalidDevice', (hipErrorInvalidImage:=200): 'hipErrorInvalidImage', (hipErrorInvalidContext:=201): 'hipErrorInvalidContext', (hipErrorContextAlreadyCurrent:=202): 'hipErrorContextAlreadyCurrent', (hipErrorMapFailed:=205): 'hipErrorMapFailed', (hipErrorMapBufferObjectFailed:=205): 'hipErrorMapBufferObjectFailed', (hipErrorUnmapFailed:=206): 'hipErrorUnmapFailed', (hipErrorArrayIsMapped:=207): 'hipErrorArrayIsMapped', (hipErrorAlreadyMapped:=208): 'hipErrorAlreadyMapped', (hipErrorNoBinaryForGpu:=209): 'hipErrorNoBinaryForGpu', (hipErrorAlreadyAcquired:=210): 'hipErrorAlreadyAcquired', (hipErrorNotMapped:=211): 'hipErrorNotMapped', (hipErrorNotMappedAsArray:=212): 'hipErrorNotMappedAsArray', (hipErrorNotMappedAsPointer:=213): 'hipErrorNotMappedAsPointer', (hipErrorECCNotCorrectable:=214): 'hipErrorECCNotCorrectable', (hipErrorUnsupportedLimit:=215): 'hipErrorUnsupportedLimit', (hipErrorContextAlreadyInUse:=216): 'hipErrorContextAlreadyInUse', (hipErrorPeerAccessUnsupported:=217): 'hipErrorPeerAccessUnsupported', (hipErrorInvalidKernelFile:=218): 'hipErrorInvalidKernelFile', (hipErrorInvalidGraphicsContext:=219): 'hipErrorInvalidGraphicsContext', (hipErrorInvalidSource:=300): 'hipErrorInvalidSource', (hipErrorFileNotFound:=301): 'hipErrorFileNotFound', (hipErrorSharedObjectSymbolNotFound:=302): 'hipErrorSharedObjectSymbolNotFound', (hipErrorSharedObjectInitFailed:=303): 'hipErrorSharedObjectInitFailed', (hipErrorOperatingSystem:=304): 'hipErrorOperatingSystem', (hipErrorInvalidHandle:=400): 'hipErrorInvalidHandle', (hipErrorInvalidResourceHandle:=400): 'hipErrorInvalidResourceHandle', (hipErrorIllegalState:=401): 'hipErrorIllegalState', (hipErrorNotFound:=500): 'hipErrorNotFound', (hipErrorNotReady:=600): 'hipErrorNotReady', (hipErrorIllegalAddress:=700): 'hipErrorIllegalAddress', (hipErrorLaunchOutOfResources:=701): 'hipErrorLaunchOutOfResources', (hipErrorLaunchTimeOut:=702): 'hipErrorLaunchTimeOut', (hipErrorPeerAccessAlreadyEnabled:=704): 'hipErrorPeerAccessAlreadyEnabled', (hipErrorPeerAccessNotEnabled:=705): 'hipErrorPeerAccessNotEnabled', (hipErrorSetOnActiveProcess:=708): 'hipErrorSetOnActiveProcess', (hipErrorContextIsDestroyed:=709): 'hipErrorContextIsDestroyed', (hipErrorAssert:=710): 'hipErrorAssert', (hipErrorHostMemoryAlreadyRegistered:=712): 'hipErrorHostMemoryAlreadyRegistered', (hipErrorHostMemoryNotRegistered:=713): 'hipErrorHostMemoryNotRegistered', (hipErrorLaunchFailure:=719): 'hipErrorLaunchFailure', (hipErrorCooperativeLaunchTooLarge:=720): 'hipErrorCooperativeLaunchTooLarge', (hipErrorNotSupported:=801): 'hipErrorNotSupported', (hipErrorStreamCaptureUnsupported:=900): 'hipErrorStreamCaptureUnsupported', (hipErrorStreamCaptureInvalidated:=901): 'hipErrorStreamCaptureInvalidated', (hipErrorStreamCaptureMerge:=902): 'hipErrorStreamCaptureMerge', (hipErrorStreamCaptureUnmatched:=903): 'hipErrorStreamCaptureUnmatched', (hipErrorStreamCaptureUnjoined:=904): 'hipErrorStreamCaptureUnjoined', (hipErrorStreamCaptureIsolation:=905): 'hipErrorStreamCaptureIsolation', (hipErrorStreamCaptureImplicit:=906): 'hipErrorStreamCaptureImplicit', (hipErrorCapturedEvent:=907): 'hipErrorCapturedEvent', (hipErrorStreamCaptureWrongThread:=908): 'hipErrorStreamCaptureWrongThread', (hipErrorGraphExecUpdateFailure:=910): 'hipErrorGraphExecUpdateFailure', (hipErrorInvalidChannelDescriptor:=911): 'hipErrorInvalidChannelDescriptor', (hipErrorInvalidTexture:=912): 'hipErrorInvalidTexture', (hipErrorUnknown:=999): 'hipErrorUnknown', (hipErrorRuntimeMemory:=1052): 'hipErrorRuntimeMemory', (hipErrorRuntimeOther:=1053): 'hipErrorRuntimeOther', (hipErrorTbd:=1054): 'hipErrorTbd'}
class ihipModuleSymbol_t(c.Struct): pass
hipFunction_t: TypeAlias = ctypes.POINTER(ihipModuleSymbol_t)
uint32_t: TypeAlias = ctypes.c_uint32
size_t: TypeAlias = ctypes.c_uint64
class ihipStream_t(c.Struct): pass
hipStream_t: TypeAlias = ctypes.POINTER(ihipStream_t)
class ihipEvent_t(c.Struct): pass
hipEvent_t: TypeAlias = ctypes.POINTER(ihipEvent_t)
@dll.bind(ctypes.c_uint32, hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), hipEvent_t, hipEvent_t, uint32_t)
def hipExtModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:ctypes._Pointer[ctypes.c_void_p], extra:ctypes._Pointer[ctypes.c_void_p], startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:uint32_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), hipEvent_t, hipEvent_t)
def hipHccModuleLaunchKernel(f:hipFunction_t, globalWorkSizeX:uint32_t, globalWorkSizeY:uint32_t, globalWorkSizeZ:uint32_t, localWorkSizeX:uint32_t, localWorkSizeY:uint32_t, localWorkSizeZ:uint32_t, sharedMemBytes:size_t, hStream:hipStream_t, kernelParams:ctypes._Pointer[ctypes.c_void_p], extra:ctypes._Pointer[ctypes.c_void_p], startEvent:hipEvent_t, stopEvent:hipEvent_t) -> ctypes.c_uint32: ...
@c.record
class dim3(c.Struct):
  SIZE = 12
  x: int
  y: int
  z: int
dim3.register_fields([('x', uint32_t, 0), ('y', uint32_t, 4), ('z', uint32_t, 8)])
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t, hipEvent_t, hipEvent_t, ctypes.c_int32)
def hipExtLaunchKernel(function_address:int|None, numBlocks:dim3, dimBlocks:dim3, args:ctypes._Pointer[ctypes.c_void_p], sharedMemBytes:size_t, stream:hipStream_t, startEvent:hipEvent_t, stopEvent:hipEvent_t, flags:int) -> ctypes.c_uint32: ...
hiprtcResult: dict[int, str] = {(HIPRTC_SUCCESS:=0): 'HIPRTC_SUCCESS', (HIPRTC_ERROR_OUT_OF_MEMORY:=1): 'HIPRTC_ERROR_OUT_OF_MEMORY', (HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:=2): 'HIPRTC_ERROR_PROGRAM_CREATION_FAILURE', (HIPRTC_ERROR_INVALID_INPUT:=3): 'HIPRTC_ERROR_INVALID_INPUT', (HIPRTC_ERROR_INVALID_PROGRAM:=4): 'HIPRTC_ERROR_INVALID_PROGRAM', (HIPRTC_ERROR_INVALID_OPTION:=5): 'HIPRTC_ERROR_INVALID_OPTION', (HIPRTC_ERROR_COMPILATION:=6): 'HIPRTC_ERROR_COMPILATION', (HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:=7): 'HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE', (HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:=8): 'HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION', (HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:=9): 'HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION', (HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:=10): 'HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID', (HIPRTC_ERROR_INTERNAL_ERROR:=11): 'HIPRTC_ERROR_INTERNAL_ERROR', (HIPRTC_ERROR_LINKING:=100): 'HIPRTC_ERROR_LINKING'}
class ihiprtcLinkState(c.Struct): pass
hiprtcLinkState: TypeAlias = ctypes.POINTER(ihiprtcLinkState)
@dll.bind(ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)
def hiprtcGetErrorString(result:ctypes.c_uint32) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32))
def hiprtcVersion(major:ctypes._Pointer[ctypes.c_int32], minor:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
class _hiprtcProgram(c.Struct): pass
hiprtcProgram: TypeAlias = ctypes.POINTER(_hiprtcProgram)
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(ctypes.c_char))
def hiprtcAddNameExpression(prog:hiprtcProgram, name_expression:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
def hiprtcCompileProgram(prog:hiprtcProgram, numOptions:int, options:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hiprtcProgram), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
def hiprtcCreateProgram(prog:ctypes._Pointer[hiprtcProgram], src:ctypes._Pointer[ctypes.c_char], name:ctypes._Pointer[ctypes.c_char], numHeaders:int, headers:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)], includeNames:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hiprtcProgram))
def hiprtcDestroyProgram(prog:ctypes._Pointer[hiprtcProgram]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
def hiprtcGetLoweredName(prog:hiprtcProgram, name_expression:ctypes._Pointer[ctypes.c_char], lowered_name:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(ctypes.c_char))
def hiprtcGetProgramLog(prog:hiprtcProgram, log:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(size_t))
def hiprtcGetProgramLogSize(prog:hiprtcProgram, logSizeRet:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(ctypes.c_char))
def hiprtcGetCode(prog:hiprtcProgram, code:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(size_t))
def hiprtcGetCodeSize(prog:hiprtcProgram, codeSizeRet:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(ctypes.c_char))
def hiprtcGetBitcode(prog:hiprtcProgram, bitcode:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcProgram, ctypes.POINTER(size_t))
def hiprtcGetBitcodeSize(prog:hiprtcProgram, bitcode_size:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
hipJitOption: dict[int, str] = {(hipJitOptionMaxRegisters:=0): 'hipJitOptionMaxRegisters', (hipJitOptionThreadsPerBlock:=1): 'hipJitOptionThreadsPerBlock', (hipJitOptionWallTime:=2): 'hipJitOptionWallTime', (hipJitOptionInfoLogBuffer:=3): 'hipJitOptionInfoLogBuffer', (hipJitOptionInfoLogBufferSizeBytes:=4): 'hipJitOptionInfoLogBufferSizeBytes', (hipJitOptionErrorLogBuffer:=5): 'hipJitOptionErrorLogBuffer', (hipJitOptionErrorLogBufferSizeBytes:=6): 'hipJitOptionErrorLogBufferSizeBytes', (hipJitOptionOptimizationLevel:=7): 'hipJitOptionOptimizationLevel', (hipJitOptionTargetFromContext:=8): 'hipJitOptionTargetFromContext', (hipJitOptionTarget:=9): 'hipJitOptionTarget', (hipJitOptionFallbackStrategy:=10): 'hipJitOptionFallbackStrategy', (hipJitOptionGenerateDebugInfo:=11): 'hipJitOptionGenerateDebugInfo', (hipJitOptionLogVerbose:=12): 'hipJitOptionLogVerbose', (hipJitOptionGenerateLineInfo:=13): 'hipJitOptionGenerateLineInfo', (hipJitOptionCacheMode:=14): 'hipJitOptionCacheMode', (hipJitOptionSm3xOpt:=15): 'hipJitOptionSm3xOpt', (hipJitOptionFastCompile:=16): 'hipJitOptionFastCompile', (hipJitOptionGlobalSymbolNames:=17): 'hipJitOptionGlobalSymbolNames', (hipJitOptionGlobalSymbolAddresses:=18): 'hipJitOptionGlobalSymbolAddresses', (hipJitOptionGlobalSymbolCount:=19): 'hipJitOptionGlobalSymbolCount', (hipJitOptionLto:=20): 'hipJitOptionLto', (hipJitOptionFtz:=21): 'hipJitOptionFtz', (hipJitOptionPrecDiv:=22): 'hipJitOptionPrecDiv', (hipJitOptionPrecSqrt:=23): 'hipJitOptionPrecSqrt', (hipJitOptionFma:=24): 'hipJitOptionFma', (hipJitOptionPositionIndependentCode:=25): 'hipJitOptionPositionIndependentCode', (hipJitOptionMinCTAPerSM:=26): 'hipJitOptionMinCTAPerSM', (hipJitOptionMaxThreadsPerBlock:=27): 'hipJitOptionMaxThreadsPerBlock', (hipJitOptionOverrideDirectiveValues:=28): 'hipJitOptionOverrideDirectiveValues', (hipJitOptionNumOptions:=29): 'hipJitOptionNumOptions', (hipJitOptionIRtoISAOptExt:=10000): 'hipJitOptionIRtoISAOptExt', (hipJitOptionIRtoISAOptCountExt:=10001): 'hipJitOptionIRtoISAOptCountExt'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(hiprtcLinkState))
def hiprtcLinkCreate(num_options:int, option_ptr:ctypes._Pointer[ctypes.c_uint32], option_vals_pptr:ctypes._Pointer[ctypes.c_void_p], hip_link_state_ptr:ctypes._Pointer[hiprtcLinkState]) -> ctypes.c_uint32: ...
hipJitInputType: dict[int, str] = {(hipJitInputCubin:=0): 'hipJitInputCubin', (hipJitInputPtx:=1): 'hipJitInputPtx', (hipJitInputFatBinary:=2): 'hipJitInputFatBinary', (hipJitInputObject:=3): 'hipJitInputObject', (hipJitInputLibrary:=4): 'hipJitInputLibrary', (hipJitInputNvvm:=5): 'hipJitInputNvvm', (hipJitNumLegacyInputTypes:=6): 'hipJitNumLegacyInputTypes', (hipJitInputLLVMBitcode:=100): 'hipJitInputLLVMBitcode', (hipJitInputLLVMBundledBitcode:=101): 'hipJitInputLLVMBundledBitcode', (hipJitInputLLVMArchivesOfBundledBitcode:=102): 'hipJitInputLLVMArchivesOfBundledBitcode', (hipJitInputSpirv:=103): 'hipJitInputSpirv', (hipJitNumInputTypes:=10): 'hipJitNumInputTypes'}
@dll.bind(ctypes.c_uint32, hiprtcLinkState, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p))
def hiprtcLinkAddFile(hip_link_state:hiprtcLinkState, input_type:ctypes.c_uint32, file_path:ctypes._Pointer[ctypes.c_char], num_options:int, options_ptr:ctypes._Pointer[ctypes.c_uint32], option_values:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcLinkState, ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p))
def hiprtcLinkAddData(hip_link_state:hiprtcLinkState, input_type:ctypes.c_uint32, image:int|None, image_size:size_t, name:ctypes._Pointer[ctypes.c_char], num_options:int, options_ptr:ctypes._Pointer[ctypes.c_uint32], option_values:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcLinkState, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t))
def hiprtcLinkComplete(hip_link_state:hiprtcLinkState, bin_out:ctypes._Pointer[ctypes.c_void_p], size_out:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hiprtcLinkState)
def hiprtcLinkDestroy(hip_link_state:hiprtcLinkState) -> ctypes.c_uint32: ...
_anonenum0: dict[int, str] = {(HIP_SUCCESS:=0): 'HIP_SUCCESS', (HIP_ERROR_INVALID_VALUE:=1): 'HIP_ERROR_INVALID_VALUE', (HIP_ERROR_NOT_INITIALIZED:=2): 'HIP_ERROR_NOT_INITIALIZED', (HIP_ERROR_LAUNCH_OUT_OF_RESOURCES:=3): 'HIP_ERROR_LAUNCH_OUT_OF_RESOURCES'}
@c.record
class hipDeviceArch_t(c.Struct):
  SIZE = 4
  hasGlobalInt32Atomics: int
  hasGlobalFloatAtomicExch: int
  hasSharedInt32Atomics: int
  hasSharedFloatAtomicExch: int
  hasFloatAtomicAdd: int
  hasGlobalInt64Atomics: int
  hasSharedInt64Atomics: int
  hasDoubles: int
  hasWarpVote: int
  hasWarpBallot: int
  hasWarpShuffle: int
  hasFunnelShift: int
  hasThreadFenceSystem: int
  hasSyncThreadsExt: int
  hasSurfaceFuncs: int
  has3dGrid: int
  hasDynamicParallelism: int
hipDeviceArch_t.register_fields([('hasGlobalInt32Atomics', ctypes.c_uint32, 0, 1, 0), ('hasGlobalFloatAtomicExch', ctypes.c_uint32, 0, 1, 1), ('hasSharedInt32Atomics', ctypes.c_uint32, 0, 1, 2), ('hasSharedFloatAtomicExch', ctypes.c_uint32, 0, 1, 3), ('hasFloatAtomicAdd', ctypes.c_uint32, 0, 1, 4), ('hasGlobalInt64Atomics', ctypes.c_uint32, 0, 1, 5), ('hasSharedInt64Atomics', ctypes.c_uint32, 0, 1, 6), ('hasDoubles', ctypes.c_uint32, 0, 1, 7), ('hasWarpVote', ctypes.c_uint32, 1, 1, 0), ('hasWarpBallot', ctypes.c_uint32, 1, 1, 1), ('hasWarpShuffle', ctypes.c_uint32, 1, 1, 2), ('hasFunnelShift', ctypes.c_uint32, 1, 1, 3), ('hasThreadFenceSystem', ctypes.c_uint32, 1, 1, 4), ('hasSyncThreadsExt', ctypes.c_uint32, 1, 1, 5), ('hasSurfaceFuncs', ctypes.c_uint32, 1, 1, 6), ('has3dGrid', ctypes.c_uint32, 1, 1, 7), ('hasDynamicParallelism', ctypes.c_uint32, 2, 1, 0)])
@c.record
class hipUUID_t(c.Struct):
  SIZE = 16
  bytes: ctypes.Array[ctypes.c_char]
hipUUID_t.register_fields([('bytes', (ctypes.c_char * 16), 0)])
hipUUID: TypeAlias = hipUUID_t
@c.record
class hipDeviceProp_tR0600(c.Struct):
  SIZE = 1472
  name: ctypes.Array[ctypes.c_char]
  uuid: hipUUID_t
  luid: ctypes.Array[ctypes.c_char]
  luidDeviceNodeMask: int
  totalGlobalMem: int
  sharedMemPerBlock: int
  regsPerBlock: int
  warpSize: int
  memPitch: int
  maxThreadsPerBlock: int
  maxThreadsDim: ctypes.Array[ctypes.c_int32]
  maxGridSize: ctypes.Array[ctypes.c_int32]
  clockRate: int
  totalConstMem: int
  major: int
  minor: int
  textureAlignment: int
  texturePitchAlignment: int
  deviceOverlap: int
  multiProcessorCount: int
  kernelExecTimeoutEnabled: int
  integrated: int
  canMapHostMemory: int
  computeMode: int
  maxTexture1D: int
  maxTexture1DMipmap: int
  maxTexture1DLinear: int
  maxTexture2D: ctypes.Array[ctypes.c_int32]
  maxTexture2DMipmap: ctypes.Array[ctypes.c_int32]
  maxTexture2DLinear: ctypes.Array[ctypes.c_int32]
  maxTexture2DGather: ctypes.Array[ctypes.c_int32]
  maxTexture3D: ctypes.Array[ctypes.c_int32]
  maxTexture3DAlt: ctypes.Array[ctypes.c_int32]
  maxTextureCubemap: int
  maxTexture1DLayered: ctypes.Array[ctypes.c_int32]
  maxTexture2DLayered: ctypes.Array[ctypes.c_int32]
  maxTextureCubemapLayered: ctypes.Array[ctypes.c_int32]
  maxSurface1D: int
  maxSurface2D: ctypes.Array[ctypes.c_int32]
  maxSurface3D: ctypes.Array[ctypes.c_int32]
  maxSurface1DLayered: ctypes.Array[ctypes.c_int32]
  maxSurface2DLayered: ctypes.Array[ctypes.c_int32]
  maxSurfaceCubemap: int
  maxSurfaceCubemapLayered: ctypes.Array[ctypes.c_int32]
  surfaceAlignment: int
  concurrentKernels: int
  ECCEnabled: int
  pciBusID: int
  pciDeviceID: int
  pciDomainID: int
  tccDriver: int
  asyncEngineCount: int
  unifiedAddressing: int
  memoryClockRate: int
  memoryBusWidth: int
  l2CacheSize: int
  persistingL2CacheMaxSize: int
  maxThreadsPerMultiProcessor: int
  streamPrioritiesSupported: int
  globalL1CacheSupported: int
  localL1CacheSupported: int
  sharedMemPerMultiprocessor: int
  regsPerMultiprocessor: int
  managedMemory: int
  isMultiGpuBoard: int
  multiGpuBoardGroupID: int
  hostNativeAtomicSupported: int
  singleToDoublePrecisionPerfRatio: int
  pageableMemoryAccess: int
  concurrentManagedAccess: int
  computePreemptionSupported: int
  canUseHostPointerForRegisteredMem: int
  cooperativeLaunch: int
  cooperativeMultiDeviceLaunch: int
  sharedMemPerBlockOptin: int
  pageableMemoryAccessUsesHostPageTables: int
  directManagedMemAccessFromHost: int
  maxBlocksPerMultiProcessor: int
  accessPolicyMaxWindowSize: int
  reservedSharedMemPerBlock: int
  hostRegisterSupported: int
  sparseHipArraySupported: int
  hostRegisterReadOnlySupported: int
  timelineSemaphoreInteropSupported: int
  memoryPoolsSupported: int
  gpuDirectRDMASupported: int
  gpuDirectRDMAFlushWritesOptions: int
  gpuDirectRDMAWritesOrdering: int
  memoryPoolSupportedHandleTypes: int
  deferredMappingHipArraySupported: int
  ipcEventSupported: int
  clusterLaunch: int
  unifiedFunctionPointers: int
  reserved: ctypes.Array[ctypes.c_int32]
  hipReserved: ctypes.Array[ctypes.c_int32]
  gcnArchName: ctypes.Array[ctypes.c_char]
  maxSharedMemoryPerMultiProcessor: int
  clockInstructionRate: int
  arch: hipDeviceArch_t
  hdpMemFlushCntl: ctypes._Pointer[ctypes.c_uint32]
  hdpRegFlushCntl: ctypes._Pointer[ctypes.c_uint32]
  cooperativeMultiDeviceUnmatchedFunc: int
  cooperativeMultiDeviceUnmatchedGridDim: int
  cooperativeMultiDeviceUnmatchedBlockDim: int
  cooperativeMultiDeviceUnmatchedSharedMem: int
  isLargeBar: int
  asicRevision: int
hipDeviceProp_tR0600.register_fields([('name', (ctypes.c_char * 256), 0), ('uuid', hipUUID, 256), ('luid', (ctypes.c_char * 8), 272), ('luidDeviceNodeMask', ctypes.c_uint32, 280), ('totalGlobalMem', size_t, 288), ('sharedMemPerBlock', size_t, 296), ('regsPerBlock', ctypes.c_int32, 304), ('warpSize', ctypes.c_int32, 308), ('memPitch', size_t, 312), ('maxThreadsPerBlock', ctypes.c_int32, 320), ('maxThreadsDim', (ctypes.c_int32 * 3), 324), ('maxGridSize', (ctypes.c_int32 * 3), 336), ('clockRate', ctypes.c_int32, 348), ('totalConstMem', size_t, 352), ('major', ctypes.c_int32, 360), ('minor', ctypes.c_int32, 364), ('textureAlignment', size_t, 368), ('texturePitchAlignment', size_t, 376), ('deviceOverlap', ctypes.c_int32, 384), ('multiProcessorCount', ctypes.c_int32, 388), ('kernelExecTimeoutEnabled', ctypes.c_int32, 392), ('integrated', ctypes.c_int32, 396), ('canMapHostMemory', ctypes.c_int32, 400), ('computeMode', ctypes.c_int32, 404), ('maxTexture1D', ctypes.c_int32, 408), ('maxTexture1DMipmap', ctypes.c_int32, 412), ('maxTexture1DLinear', ctypes.c_int32, 416), ('maxTexture2D', (ctypes.c_int32 * 2), 420), ('maxTexture2DMipmap', (ctypes.c_int32 * 2), 428), ('maxTexture2DLinear', (ctypes.c_int32 * 3), 436), ('maxTexture2DGather', (ctypes.c_int32 * 2), 448), ('maxTexture3D', (ctypes.c_int32 * 3), 456), ('maxTexture3DAlt', (ctypes.c_int32 * 3), 468), ('maxTextureCubemap', ctypes.c_int32, 480), ('maxTexture1DLayered', (ctypes.c_int32 * 2), 484), ('maxTexture2DLayered', (ctypes.c_int32 * 3), 492), ('maxTextureCubemapLayered', (ctypes.c_int32 * 2), 504), ('maxSurface1D', ctypes.c_int32, 512), ('maxSurface2D', (ctypes.c_int32 * 2), 516), ('maxSurface3D', (ctypes.c_int32 * 3), 524), ('maxSurface1DLayered', (ctypes.c_int32 * 2), 536), ('maxSurface2DLayered', (ctypes.c_int32 * 3), 544), ('maxSurfaceCubemap', ctypes.c_int32, 556), ('maxSurfaceCubemapLayered', (ctypes.c_int32 * 2), 560), ('surfaceAlignment', size_t, 568), ('concurrentKernels', ctypes.c_int32, 576), ('ECCEnabled', ctypes.c_int32, 580), ('pciBusID', ctypes.c_int32, 584), ('pciDeviceID', ctypes.c_int32, 588), ('pciDomainID', ctypes.c_int32, 592), ('tccDriver', ctypes.c_int32, 596), ('asyncEngineCount', ctypes.c_int32, 600), ('unifiedAddressing', ctypes.c_int32, 604), ('memoryClockRate', ctypes.c_int32, 608), ('memoryBusWidth', ctypes.c_int32, 612), ('l2CacheSize', ctypes.c_int32, 616), ('persistingL2CacheMaxSize', ctypes.c_int32, 620), ('maxThreadsPerMultiProcessor', ctypes.c_int32, 624), ('streamPrioritiesSupported', ctypes.c_int32, 628), ('globalL1CacheSupported', ctypes.c_int32, 632), ('localL1CacheSupported', ctypes.c_int32, 636), ('sharedMemPerMultiprocessor', size_t, 640), ('regsPerMultiprocessor', ctypes.c_int32, 648), ('managedMemory', ctypes.c_int32, 652), ('isMultiGpuBoard', ctypes.c_int32, 656), ('multiGpuBoardGroupID', ctypes.c_int32, 660), ('hostNativeAtomicSupported', ctypes.c_int32, 664), ('singleToDoublePrecisionPerfRatio', ctypes.c_int32, 668), ('pageableMemoryAccess', ctypes.c_int32, 672), ('concurrentManagedAccess', ctypes.c_int32, 676), ('computePreemptionSupported', ctypes.c_int32, 680), ('canUseHostPointerForRegisteredMem', ctypes.c_int32, 684), ('cooperativeLaunch', ctypes.c_int32, 688), ('cooperativeMultiDeviceLaunch', ctypes.c_int32, 692), ('sharedMemPerBlockOptin', size_t, 696), ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int32, 704), ('directManagedMemAccessFromHost', ctypes.c_int32, 708), ('maxBlocksPerMultiProcessor', ctypes.c_int32, 712), ('accessPolicyMaxWindowSize', ctypes.c_int32, 716), ('reservedSharedMemPerBlock', size_t, 720), ('hostRegisterSupported', ctypes.c_int32, 728), ('sparseHipArraySupported', ctypes.c_int32, 732), ('hostRegisterReadOnlySupported', ctypes.c_int32, 736), ('timelineSemaphoreInteropSupported', ctypes.c_int32, 740), ('memoryPoolsSupported', ctypes.c_int32, 744), ('gpuDirectRDMASupported', ctypes.c_int32, 748), ('gpuDirectRDMAFlushWritesOptions', ctypes.c_uint32, 752), ('gpuDirectRDMAWritesOrdering', ctypes.c_int32, 756), ('memoryPoolSupportedHandleTypes', ctypes.c_uint32, 760), ('deferredMappingHipArraySupported', ctypes.c_int32, 764), ('ipcEventSupported', ctypes.c_int32, 768), ('clusterLaunch', ctypes.c_int32, 772), ('unifiedFunctionPointers', ctypes.c_int32, 776), ('reserved', (ctypes.c_int32 * 63), 780), ('hipReserved', (ctypes.c_int32 * 32), 1032), ('gcnArchName', (ctypes.c_char * 256), 1160), ('maxSharedMemoryPerMultiProcessor', size_t, 1416), ('clockInstructionRate', ctypes.c_int32, 1424), ('arch', hipDeviceArch_t, 1428), ('hdpMemFlushCntl', ctypes.POINTER(ctypes.c_uint32), 1432), ('hdpRegFlushCntl', ctypes.POINTER(ctypes.c_uint32), 1440), ('cooperativeMultiDeviceUnmatchedFunc', ctypes.c_int32, 1448), ('cooperativeMultiDeviceUnmatchedGridDim', ctypes.c_int32, 1452), ('cooperativeMultiDeviceUnmatchedBlockDim', ctypes.c_int32, 1456), ('cooperativeMultiDeviceUnmatchedSharedMem', ctypes.c_int32, 1460), ('isLargeBar', ctypes.c_int32, 1464), ('asicRevision', ctypes.c_int32, 1468)])
hipMemoryType: dict[int, str] = {(hipMemoryTypeUnregistered:=0): 'hipMemoryTypeUnregistered', (hipMemoryTypeHost:=1): 'hipMemoryTypeHost', (hipMemoryTypeDevice:=2): 'hipMemoryTypeDevice', (hipMemoryTypeManaged:=3): 'hipMemoryTypeManaged', (hipMemoryTypeArray:=10): 'hipMemoryTypeArray', (hipMemoryTypeUnified:=11): 'hipMemoryTypeUnified'}
@c.record
class hipPointerAttribute_t(c.Struct):
  SIZE = 32
  type: int
  device: int
  devicePointer: int|None
  hostPointer: int|None
  isManaged: int
  allocationFlags: int
hipPointerAttribute_t.register_fields([('type', ctypes.c_uint32, 0), ('device', ctypes.c_int32, 4), ('devicePointer', ctypes.c_void_p, 8), ('hostPointer', ctypes.c_void_p, 16), ('isManaged', ctypes.c_int32, 24), ('allocationFlags', ctypes.c_uint32, 28)])
hipDeviceAttribute_t: dict[int, str] = {(hipDeviceAttributeCudaCompatibleBegin:=0): 'hipDeviceAttributeCudaCompatibleBegin', (hipDeviceAttributeEccEnabled:=0): 'hipDeviceAttributeEccEnabled', (hipDeviceAttributeAccessPolicyMaxWindowSize:=1): 'hipDeviceAttributeAccessPolicyMaxWindowSize', (hipDeviceAttributeAsyncEngineCount:=2): 'hipDeviceAttributeAsyncEngineCount', (hipDeviceAttributeCanMapHostMemory:=3): 'hipDeviceAttributeCanMapHostMemory', (hipDeviceAttributeCanUseHostPointerForRegisteredMem:=4): 'hipDeviceAttributeCanUseHostPointerForRegisteredMem', (hipDeviceAttributeClockRate:=5): 'hipDeviceAttributeClockRate', (hipDeviceAttributeComputeMode:=6): 'hipDeviceAttributeComputeMode', (hipDeviceAttributeComputePreemptionSupported:=7): 'hipDeviceAttributeComputePreemptionSupported', (hipDeviceAttributeConcurrentKernels:=8): 'hipDeviceAttributeConcurrentKernels', (hipDeviceAttributeConcurrentManagedAccess:=9): 'hipDeviceAttributeConcurrentManagedAccess', (hipDeviceAttributeCooperativeLaunch:=10): 'hipDeviceAttributeCooperativeLaunch', (hipDeviceAttributeCooperativeMultiDeviceLaunch:=11): 'hipDeviceAttributeCooperativeMultiDeviceLaunch', (hipDeviceAttributeDeviceOverlap:=12): 'hipDeviceAttributeDeviceOverlap', (hipDeviceAttributeDirectManagedMemAccessFromHost:=13): 'hipDeviceAttributeDirectManagedMemAccessFromHost', (hipDeviceAttributeGlobalL1CacheSupported:=14): 'hipDeviceAttributeGlobalL1CacheSupported', (hipDeviceAttributeHostNativeAtomicSupported:=15): 'hipDeviceAttributeHostNativeAtomicSupported', (hipDeviceAttributeIntegrated:=16): 'hipDeviceAttributeIntegrated', (hipDeviceAttributeIsMultiGpuBoard:=17): 'hipDeviceAttributeIsMultiGpuBoard', (hipDeviceAttributeKernelExecTimeout:=18): 'hipDeviceAttributeKernelExecTimeout', (hipDeviceAttributeL2CacheSize:=19): 'hipDeviceAttributeL2CacheSize', (hipDeviceAttributeLocalL1CacheSupported:=20): 'hipDeviceAttributeLocalL1CacheSupported', (hipDeviceAttributeLuid:=21): 'hipDeviceAttributeLuid', (hipDeviceAttributeLuidDeviceNodeMask:=22): 'hipDeviceAttributeLuidDeviceNodeMask', (hipDeviceAttributeComputeCapabilityMajor:=23): 'hipDeviceAttributeComputeCapabilityMajor', (hipDeviceAttributeManagedMemory:=24): 'hipDeviceAttributeManagedMemory', (hipDeviceAttributeMaxBlocksPerMultiProcessor:=25): 'hipDeviceAttributeMaxBlocksPerMultiProcessor', (hipDeviceAttributeMaxBlockDimX:=26): 'hipDeviceAttributeMaxBlockDimX', (hipDeviceAttributeMaxBlockDimY:=27): 'hipDeviceAttributeMaxBlockDimY', (hipDeviceAttributeMaxBlockDimZ:=28): 'hipDeviceAttributeMaxBlockDimZ', (hipDeviceAttributeMaxGridDimX:=29): 'hipDeviceAttributeMaxGridDimX', (hipDeviceAttributeMaxGridDimY:=30): 'hipDeviceAttributeMaxGridDimY', (hipDeviceAttributeMaxGridDimZ:=31): 'hipDeviceAttributeMaxGridDimZ', (hipDeviceAttributeMaxSurface1D:=32): 'hipDeviceAttributeMaxSurface1D', (hipDeviceAttributeMaxSurface1DLayered:=33): 'hipDeviceAttributeMaxSurface1DLayered', (hipDeviceAttributeMaxSurface2D:=34): 'hipDeviceAttributeMaxSurface2D', (hipDeviceAttributeMaxSurface2DLayered:=35): 'hipDeviceAttributeMaxSurface2DLayered', (hipDeviceAttributeMaxSurface3D:=36): 'hipDeviceAttributeMaxSurface3D', (hipDeviceAttributeMaxSurfaceCubemap:=37): 'hipDeviceAttributeMaxSurfaceCubemap', (hipDeviceAttributeMaxSurfaceCubemapLayered:=38): 'hipDeviceAttributeMaxSurfaceCubemapLayered', (hipDeviceAttributeMaxTexture1DWidth:=39): 'hipDeviceAttributeMaxTexture1DWidth', (hipDeviceAttributeMaxTexture1DLayered:=40): 'hipDeviceAttributeMaxTexture1DLayered', (hipDeviceAttributeMaxTexture1DLinear:=41): 'hipDeviceAttributeMaxTexture1DLinear', (hipDeviceAttributeMaxTexture1DMipmap:=42): 'hipDeviceAttributeMaxTexture1DMipmap', (hipDeviceAttributeMaxTexture2DWidth:=43): 'hipDeviceAttributeMaxTexture2DWidth', (hipDeviceAttributeMaxTexture2DHeight:=44): 'hipDeviceAttributeMaxTexture2DHeight', (hipDeviceAttributeMaxTexture2DGather:=45): 'hipDeviceAttributeMaxTexture2DGather', (hipDeviceAttributeMaxTexture2DLayered:=46): 'hipDeviceAttributeMaxTexture2DLayered', (hipDeviceAttributeMaxTexture2DLinear:=47): 'hipDeviceAttributeMaxTexture2DLinear', (hipDeviceAttributeMaxTexture2DMipmap:=48): 'hipDeviceAttributeMaxTexture2DMipmap', (hipDeviceAttributeMaxTexture3DWidth:=49): 'hipDeviceAttributeMaxTexture3DWidth', (hipDeviceAttributeMaxTexture3DHeight:=50): 'hipDeviceAttributeMaxTexture3DHeight', (hipDeviceAttributeMaxTexture3DDepth:=51): 'hipDeviceAttributeMaxTexture3DDepth', (hipDeviceAttributeMaxTexture3DAlt:=52): 'hipDeviceAttributeMaxTexture3DAlt', (hipDeviceAttributeMaxTextureCubemap:=53): 'hipDeviceAttributeMaxTextureCubemap', (hipDeviceAttributeMaxTextureCubemapLayered:=54): 'hipDeviceAttributeMaxTextureCubemapLayered', (hipDeviceAttributeMaxThreadsDim:=55): 'hipDeviceAttributeMaxThreadsDim', (hipDeviceAttributeMaxThreadsPerBlock:=56): 'hipDeviceAttributeMaxThreadsPerBlock', (hipDeviceAttributeMaxThreadsPerMultiProcessor:=57): 'hipDeviceAttributeMaxThreadsPerMultiProcessor', (hipDeviceAttributeMaxPitch:=58): 'hipDeviceAttributeMaxPitch', (hipDeviceAttributeMemoryBusWidth:=59): 'hipDeviceAttributeMemoryBusWidth', (hipDeviceAttributeMemoryClockRate:=60): 'hipDeviceAttributeMemoryClockRate', (hipDeviceAttributeComputeCapabilityMinor:=61): 'hipDeviceAttributeComputeCapabilityMinor', (hipDeviceAttributeMultiGpuBoardGroupID:=62): 'hipDeviceAttributeMultiGpuBoardGroupID', (hipDeviceAttributeMultiprocessorCount:=63): 'hipDeviceAttributeMultiprocessorCount', (hipDeviceAttributeUnused1:=64): 'hipDeviceAttributeUnused1', (hipDeviceAttributePageableMemoryAccess:=65): 'hipDeviceAttributePageableMemoryAccess', (hipDeviceAttributePageableMemoryAccessUsesHostPageTables:=66): 'hipDeviceAttributePageableMemoryAccessUsesHostPageTables', (hipDeviceAttributePciBusId:=67): 'hipDeviceAttributePciBusId', (hipDeviceAttributePciDeviceId:=68): 'hipDeviceAttributePciDeviceId', (hipDeviceAttributePciDomainId:=69): 'hipDeviceAttributePciDomainId', (hipDeviceAttributePciDomainID:=69): 'hipDeviceAttributePciDomainID', (hipDeviceAttributePersistingL2CacheMaxSize:=70): 'hipDeviceAttributePersistingL2CacheMaxSize', (hipDeviceAttributeMaxRegistersPerBlock:=71): 'hipDeviceAttributeMaxRegistersPerBlock', (hipDeviceAttributeMaxRegistersPerMultiprocessor:=72): 'hipDeviceAttributeMaxRegistersPerMultiprocessor', (hipDeviceAttributeReservedSharedMemPerBlock:=73): 'hipDeviceAttributeReservedSharedMemPerBlock', (hipDeviceAttributeMaxSharedMemoryPerBlock:=74): 'hipDeviceAttributeMaxSharedMemoryPerBlock', (hipDeviceAttributeSharedMemPerBlockOptin:=75): 'hipDeviceAttributeSharedMemPerBlockOptin', (hipDeviceAttributeSharedMemPerMultiprocessor:=76): 'hipDeviceAttributeSharedMemPerMultiprocessor', (hipDeviceAttributeSingleToDoublePrecisionPerfRatio:=77): 'hipDeviceAttributeSingleToDoublePrecisionPerfRatio', (hipDeviceAttributeStreamPrioritiesSupported:=78): 'hipDeviceAttributeStreamPrioritiesSupported', (hipDeviceAttributeSurfaceAlignment:=79): 'hipDeviceAttributeSurfaceAlignment', (hipDeviceAttributeTccDriver:=80): 'hipDeviceAttributeTccDriver', (hipDeviceAttributeTextureAlignment:=81): 'hipDeviceAttributeTextureAlignment', (hipDeviceAttributeTexturePitchAlignment:=82): 'hipDeviceAttributeTexturePitchAlignment', (hipDeviceAttributeTotalConstantMemory:=83): 'hipDeviceAttributeTotalConstantMemory', (hipDeviceAttributeTotalGlobalMem:=84): 'hipDeviceAttributeTotalGlobalMem', (hipDeviceAttributeUnifiedAddressing:=85): 'hipDeviceAttributeUnifiedAddressing', (hipDeviceAttributeUnused2:=86): 'hipDeviceAttributeUnused2', (hipDeviceAttributeWarpSize:=87): 'hipDeviceAttributeWarpSize', (hipDeviceAttributeMemoryPoolsSupported:=88): 'hipDeviceAttributeMemoryPoolsSupported', (hipDeviceAttributeVirtualMemoryManagementSupported:=89): 'hipDeviceAttributeVirtualMemoryManagementSupported', (hipDeviceAttributeHostRegisterSupported:=90): 'hipDeviceAttributeHostRegisterSupported', (hipDeviceAttributeMemoryPoolSupportedHandleTypes:=91): 'hipDeviceAttributeMemoryPoolSupportedHandleTypes', (hipDeviceAttributeCudaCompatibleEnd:=9999): 'hipDeviceAttributeCudaCompatibleEnd', (hipDeviceAttributeAmdSpecificBegin:=10000): 'hipDeviceAttributeAmdSpecificBegin', (hipDeviceAttributeClockInstructionRate:=10000): 'hipDeviceAttributeClockInstructionRate', (hipDeviceAttributeUnused3:=10001): 'hipDeviceAttributeUnused3', (hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:=10002): 'hipDeviceAttributeMaxSharedMemoryPerMultiprocessor', (hipDeviceAttributeUnused4:=10003): 'hipDeviceAttributeUnused4', (hipDeviceAttributeUnused5:=10004): 'hipDeviceAttributeUnused5', (hipDeviceAttributeHdpMemFlushCntl:=10005): 'hipDeviceAttributeHdpMemFlushCntl', (hipDeviceAttributeHdpRegFlushCntl:=10006): 'hipDeviceAttributeHdpRegFlushCntl', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:=10007): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:=10008): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:=10009): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim', (hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:=10010): 'hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem', (hipDeviceAttributeIsLargeBar:=10011): 'hipDeviceAttributeIsLargeBar', (hipDeviceAttributeAsicRevision:=10012): 'hipDeviceAttributeAsicRevision', (hipDeviceAttributeCanUseStreamWaitValue:=10013): 'hipDeviceAttributeCanUseStreamWaitValue', (hipDeviceAttributeImageSupport:=10014): 'hipDeviceAttributeImageSupport', (hipDeviceAttributePhysicalMultiProcessorCount:=10015): 'hipDeviceAttributePhysicalMultiProcessorCount', (hipDeviceAttributeFineGrainSupport:=10016): 'hipDeviceAttributeFineGrainSupport', (hipDeviceAttributeWallClockRate:=10017): 'hipDeviceAttributeWallClockRate', (hipDeviceAttributeNumberOfXccs:=10018): 'hipDeviceAttributeNumberOfXccs', (hipDeviceAttributeMaxAvailableVgprsPerThread:=10019): 'hipDeviceAttributeMaxAvailableVgprsPerThread', (hipDeviceAttributePciChipId:=10020): 'hipDeviceAttributePciChipId', (hipDeviceAttributeAmdSpecificEnd:=19999): 'hipDeviceAttributeAmdSpecificEnd', (hipDeviceAttributeVendorSpecificBegin:=20000): 'hipDeviceAttributeVendorSpecificBegin'}
hipDriverProcAddressQueryResult: dict[int, str] = {(HIP_GET_PROC_ADDRESS_SUCCESS:=0): 'HIP_GET_PROC_ADDRESS_SUCCESS', (HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND:=1): 'HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', (HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT:=2): 'HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT'}
hipComputeMode: dict[int, str] = {(hipComputeModeDefault:=0): 'hipComputeModeDefault', (hipComputeModeExclusive:=1): 'hipComputeModeExclusive', (hipComputeModeProhibited:=2): 'hipComputeModeProhibited', (hipComputeModeExclusiveProcess:=3): 'hipComputeModeExclusiveProcess'}
hipFlushGPUDirectRDMAWritesOptions: dict[int, str] = {(hipFlushGPUDirectRDMAWritesOptionHost:=1): 'hipFlushGPUDirectRDMAWritesOptionHost', (hipFlushGPUDirectRDMAWritesOptionMemOps:=2): 'hipFlushGPUDirectRDMAWritesOptionMemOps'}
hipGPUDirectRDMAWritesOrdering: dict[int, str] = {(hipGPUDirectRDMAWritesOrderingNone:=0): 'hipGPUDirectRDMAWritesOrderingNone', (hipGPUDirectRDMAWritesOrderingOwner:=100): 'hipGPUDirectRDMAWritesOrderingOwner', (hipGPUDirectRDMAWritesOrderingAllDevices:=200): 'hipGPUDirectRDMAWritesOrderingAllDevices'}
@dll.bind(ctypes.c_uint32)
def hip_init() -> ctypes.c_uint32: ...
class ihipCtx_t(c.Struct): pass
hipCtx_t: TypeAlias = ctypes.POINTER(ihipCtx_t)
hipDevice_t: TypeAlias = ctypes.c_int32
hipDeviceP2PAttr: dict[int, str] = {(hipDevP2PAttrPerformanceRank:=0): 'hipDevP2PAttrPerformanceRank', (hipDevP2PAttrAccessSupported:=1): 'hipDevP2PAttrAccessSupported', (hipDevP2PAttrNativeAtomicSupported:=2): 'hipDevP2PAttrNativeAtomicSupported', (hipDevP2PAttrHipArrayAccessSupported:=3): 'hipDevP2PAttrHipArrayAccessSupported'}
hipDriverEntryPointQueryResult: dict[int, str] = {(hipDriverEntryPointSuccess:=0): 'hipDriverEntryPointSuccess', (hipDriverEntryPointSymbolNotFound:=1): 'hipDriverEntryPointSymbolNotFound', (hipDriverEntryPointVersionNotSufficent:=2): 'hipDriverEntryPointVersionNotSufficent'}
@c.record
class hipIpcMemHandle_st(c.Struct):
  SIZE = 64
  reserved: ctypes.Array[ctypes.c_char]
hipIpcMemHandle_st.register_fields([('reserved', (ctypes.c_char * 64), 0)])
hipIpcMemHandle_t: TypeAlias = hipIpcMemHandle_st
@c.record
class hipIpcEventHandle_st(c.Struct):
  SIZE = 64
  reserved: ctypes.Array[ctypes.c_char]
hipIpcEventHandle_st.register_fields([('reserved', (ctypes.c_char * 64), 0)])
hipIpcEventHandle_t: TypeAlias = hipIpcEventHandle_st
class ihipModule_t(c.Struct): pass
hipModule_t: TypeAlias = ctypes.POINTER(ihipModule_t)
class ihipLinkState_t(c.Struct): pass
hipLinkState_t: TypeAlias = ctypes.POINTER(ihipLinkState_t)
class ihipLibrary_t(c.Struct): pass
hipLibrary_t: TypeAlias = ctypes.POINTER(ihipLibrary_t)
class ihipKernel_t(c.Struct): pass
hipKernel_t: TypeAlias = ctypes.POINTER(ihipKernel_t)
class ihipMemPoolHandle_t(c.Struct): pass
hipMemPool_t: TypeAlias = ctypes.POINTER(ihipMemPoolHandle_t)
@c.record
class hipFuncAttributes(c.Struct):
  SIZE = 56
  binaryVersion: int
  cacheModeCA: int
  constSizeBytes: int
  localSizeBytes: int
  maxDynamicSharedSizeBytes: int
  maxThreadsPerBlock: int
  numRegs: int
  preferredShmemCarveout: int
  ptxVersion: int
  sharedSizeBytes: int
hipFuncAttributes.register_fields([('binaryVersion', ctypes.c_int32, 0), ('cacheModeCA', ctypes.c_int32, 4), ('constSizeBytes', size_t, 8), ('localSizeBytes', size_t, 16), ('maxDynamicSharedSizeBytes', ctypes.c_int32, 24), ('maxThreadsPerBlock', ctypes.c_int32, 28), ('numRegs', ctypes.c_int32, 32), ('preferredShmemCarveout', ctypes.c_int32, 36), ('ptxVersion', ctypes.c_int32, 40), ('sharedSizeBytes', size_t, 48)])
hipLimit_t: dict[int, str] = {(hipLimitStackSize:=0): 'hipLimitStackSize', (hipLimitPrintfFifoSize:=1): 'hipLimitPrintfFifoSize', (hipLimitMallocHeapSize:=2): 'hipLimitMallocHeapSize', (hipExtLimitScratchMin:=4096): 'hipExtLimitScratchMin', (hipExtLimitScratchMax:=4097): 'hipExtLimitScratchMax', (hipExtLimitScratchCurrent:=4098): 'hipExtLimitScratchCurrent', (hipLimitRange:=4099): 'hipLimitRange'}
hipStreamBatchMemOpType: dict[int, str] = {(hipStreamMemOpWaitValue32:=1): 'hipStreamMemOpWaitValue32', (hipStreamMemOpWriteValue32:=2): 'hipStreamMemOpWriteValue32', (hipStreamMemOpWaitValue64:=4): 'hipStreamMemOpWaitValue64', (hipStreamMemOpWriteValue64:=5): 'hipStreamMemOpWriteValue64', (hipStreamMemOpBarrier:=6): 'hipStreamMemOpBarrier', (hipStreamMemOpFlushRemoteWrites:=3): 'hipStreamMemOpFlushRemoteWrites'}
@c.record
class hipStreamBatchMemOpParams_union(c.Struct):
  SIZE = 48
  operation: int
  waitValue: hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t
  writeValue: hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t
  flushRemoteWrites: hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t
  memoryBarrier: hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t
  pad: ctypes.Array[ctypes.c_uint64]
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t(c.Struct):
  SIZE = 40
  operation: int
  address: int|None
  value: int
  value64: int
  flags: int
  alias: int|None
hipDeviceptr_t: TypeAlias = ctypes.c_void_p
uint64_t: TypeAlias = ctypes.c_uint64
hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('address', hipDeviceptr_t, 8), ('value', uint32_t, 16), ('value64', uint64_t, 16), ('flags', ctypes.c_uint32, 24), ('alias', hipDeviceptr_t, 32)])
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t(c.Struct):
  SIZE = 40
  operation: int
  address: int|None
  value: int
  value64: int
  flags: int
  alias: int|None
hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('address', hipDeviceptr_t, 8), ('value', uint32_t, 16), ('value64', uint64_t, 16), ('flags', ctypes.c_uint32, 24), ('alias', hipDeviceptr_t, 32)])
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t(c.Struct):
  SIZE = 8
  operation: int
  flags: int
hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
@c.record
class hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t(c.Struct):
  SIZE = 8
  operation: int
  flags: int
hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t.register_fields([('operation', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
hipStreamBatchMemOpParams_union.register_fields([('operation', ctypes.c_uint32, 0), ('waitValue', hipStreamBatchMemOpParams_union_hipStreamMemOpWaitValueParams_t, 0), ('writeValue', hipStreamBatchMemOpParams_union_hipStreamMemOpWriteValueParams_t, 0), ('flushRemoteWrites', hipStreamBatchMemOpParams_union_hipStreamMemOpFlushRemoteWritesParams_t, 0), ('memoryBarrier', hipStreamBatchMemOpParams_union_hipStreamMemOpMemoryBarrierParams_t, 0), ('pad', (uint64_t * 6), 0)])
hipStreamBatchMemOpParams: TypeAlias = hipStreamBatchMemOpParams_union
@c.record
class hipBatchMemOpNodeParams(c.Struct):
  SIZE = 32
  ctx: ctypes._Pointer[ihipCtx_t]
  count: int
  paramArray: ctypes._Pointer[hipStreamBatchMemOpParams_union]
  flags: int
hipBatchMemOpNodeParams.register_fields([('ctx', hipCtx_t, 0), ('count', ctypes.c_uint32, 8), ('paramArray', ctypes.POINTER(hipStreamBatchMemOpParams), 16), ('flags', ctypes.c_uint32, 24)])
hipMemoryAdvise: dict[int, str] = {(hipMemAdviseSetReadMostly:=1): 'hipMemAdviseSetReadMostly', (hipMemAdviseUnsetReadMostly:=2): 'hipMemAdviseUnsetReadMostly', (hipMemAdviseSetPreferredLocation:=3): 'hipMemAdviseSetPreferredLocation', (hipMemAdviseUnsetPreferredLocation:=4): 'hipMemAdviseUnsetPreferredLocation', (hipMemAdviseSetAccessedBy:=5): 'hipMemAdviseSetAccessedBy', (hipMemAdviseUnsetAccessedBy:=6): 'hipMemAdviseUnsetAccessedBy', (hipMemAdviseSetCoarseGrain:=100): 'hipMemAdviseSetCoarseGrain', (hipMemAdviseUnsetCoarseGrain:=101): 'hipMemAdviseUnsetCoarseGrain'}
hipMemRangeCoherencyMode: dict[int, str] = {(hipMemRangeCoherencyModeFineGrain:=0): 'hipMemRangeCoherencyModeFineGrain', (hipMemRangeCoherencyModeCoarseGrain:=1): 'hipMemRangeCoherencyModeCoarseGrain', (hipMemRangeCoherencyModeIndeterminate:=2): 'hipMemRangeCoherencyModeIndeterminate'}
hipMemRangeAttribute: dict[int, str] = {(hipMemRangeAttributeReadMostly:=1): 'hipMemRangeAttributeReadMostly', (hipMemRangeAttributePreferredLocation:=2): 'hipMemRangeAttributePreferredLocation', (hipMemRangeAttributeAccessedBy:=3): 'hipMemRangeAttributeAccessedBy', (hipMemRangeAttributeLastPrefetchLocation:=4): 'hipMemRangeAttributeLastPrefetchLocation', (hipMemRangeAttributeCoherencyMode:=100): 'hipMemRangeAttributeCoherencyMode'}
hipMemPoolAttr: dict[int, str] = {(hipMemPoolReuseFollowEventDependencies:=1): 'hipMemPoolReuseFollowEventDependencies', (hipMemPoolReuseAllowOpportunistic:=2): 'hipMemPoolReuseAllowOpportunistic', (hipMemPoolReuseAllowInternalDependencies:=3): 'hipMemPoolReuseAllowInternalDependencies', (hipMemPoolAttrReleaseThreshold:=4): 'hipMemPoolAttrReleaseThreshold', (hipMemPoolAttrReservedMemCurrent:=5): 'hipMemPoolAttrReservedMemCurrent', (hipMemPoolAttrReservedMemHigh:=6): 'hipMemPoolAttrReservedMemHigh', (hipMemPoolAttrUsedMemCurrent:=7): 'hipMemPoolAttrUsedMemCurrent', (hipMemPoolAttrUsedMemHigh:=8): 'hipMemPoolAttrUsedMemHigh'}
hipMemAccessFlags: dict[int, str] = {(hipMemAccessFlagsProtNone:=0): 'hipMemAccessFlagsProtNone', (hipMemAccessFlagsProtRead:=1): 'hipMemAccessFlagsProtRead', (hipMemAccessFlagsProtReadWrite:=3): 'hipMemAccessFlagsProtReadWrite'}
@c.record
class hipMemAccessDesc(c.Struct):
  SIZE = 12
  location: hipMemLocation
  flags: int
@c.record
class hipMemLocation(c.Struct):
  SIZE = 8
  type: int
  id: int
hipMemLocationType: dict[int, str] = {(hipMemLocationTypeInvalid:=0): 'hipMemLocationTypeInvalid', (hipMemLocationTypeNone:=0): 'hipMemLocationTypeNone', (hipMemLocationTypeDevice:=1): 'hipMemLocationTypeDevice', (hipMemLocationTypeHost:=2): 'hipMemLocationTypeHost', (hipMemLocationTypeHostNuma:=3): 'hipMemLocationTypeHostNuma', (hipMemLocationTypeHostNumaCurrent:=4): 'hipMemLocationTypeHostNumaCurrent'}
hipMemLocation.register_fields([('type', ctypes.c_uint32, 0), ('id', ctypes.c_int32, 4)])
hipMemAccessDesc.register_fields([('location', hipMemLocation, 0), ('flags', ctypes.c_uint32, 8)])
hipMemAllocationType: dict[int, str] = {(hipMemAllocationTypeInvalid:=0): 'hipMemAllocationTypeInvalid', (hipMemAllocationTypePinned:=1): 'hipMemAllocationTypePinned', (hipMemAllocationTypeUncached:=1073741824): 'hipMemAllocationTypeUncached', (hipMemAllocationTypeMax:=2147483647): 'hipMemAllocationTypeMax'}
hipMemAllocationHandleType: dict[int, str] = {(hipMemHandleTypeNone:=0): 'hipMemHandleTypeNone', (hipMemHandleTypePosixFileDescriptor:=1): 'hipMemHandleTypePosixFileDescriptor', (hipMemHandleTypeWin32:=2): 'hipMemHandleTypeWin32', (hipMemHandleTypeWin32Kmt:=4): 'hipMemHandleTypeWin32Kmt'}
@c.record
class hipMemPoolProps(c.Struct):
  SIZE = 88
  allocType: int
  handleTypes: int
  location: hipMemLocation
  win32SecurityAttributes: int|None
  maxSize: int
  reserved: ctypes.Array[ctypes.c_ubyte]
hipMemPoolProps.register_fields([('allocType', ctypes.c_uint32, 0), ('handleTypes', ctypes.c_uint32, 4), ('location', hipMemLocation, 8), ('win32SecurityAttributes', ctypes.c_void_p, 16), ('maxSize', size_t, 24), ('reserved', (ctypes.c_ubyte * 56), 32)])
@c.record
class hipMemPoolPtrExportData(c.Struct):
  SIZE = 64
  reserved: ctypes.Array[ctypes.c_ubyte]
hipMemPoolPtrExportData.register_fields([('reserved', (ctypes.c_ubyte * 64), 0)])
hipFuncAttribute: dict[int, str] = {(hipFuncAttributeMaxDynamicSharedMemorySize:=8): 'hipFuncAttributeMaxDynamicSharedMemorySize', (hipFuncAttributePreferredSharedMemoryCarveout:=9): 'hipFuncAttributePreferredSharedMemoryCarveout', (hipFuncAttributeMax:=10): 'hipFuncAttributeMax'}
hipFuncCache_t: dict[int, str] = {(hipFuncCachePreferNone:=0): 'hipFuncCachePreferNone', (hipFuncCachePreferShared:=1): 'hipFuncCachePreferShared', (hipFuncCachePreferL1:=2): 'hipFuncCachePreferL1', (hipFuncCachePreferEqual:=3): 'hipFuncCachePreferEqual'}
hipSharedMemConfig: dict[int, str] = {(hipSharedMemBankSizeDefault:=0): 'hipSharedMemBankSizeDefault', (hipSharedMemBankSizeFourByte:=1): 'hipSharedMemBankSizeFourByte', (hipSharedMemBankSizeEightByte:=2): 'hipSharedMemBankSizeEightByte'}
@c.record
class hipLaunchParams_t(c.Struct):
  SIZE = 56
  func: int|None
  gridDim: dim3
  blockDim: dim3
  args: ctypes._Pointer[ctypes.c_void_p]
  sharedMem: int
  stream: ctypes._Pointer[ihipStream_t]
hipLaunchParams_t.register_fields([('func', ctypes.c_void_p, 0), ('gridDim', dim3, 8), ('blockDim', dim3, 20), ('args', ctypes.POINTER(ctypes.c_void_p), 32), ('sharedMem', size_t, 40), ('stream', hipStream_t, 48)])
hipLaunchParams: TypeAlias = hipLaunchParams_t
@c.record
class hipFunctionLaunchParams_t(c.Struct):
  SIZE = 56
  function: ctypes._Pointer[ihipModuleSymbol_t]
  gridDimX: int
  gridDimY: int
  gridDimZ: int
  blockDimX: int
  blockDimY: int
  blockDimZ: int
  sharedMemBytes: int
  hStream: ctypes._Pointer[ihipStream_t]
  kernelParams: ctypes._Pointer[ctypes.c_void_p]
hipFunctionLaunchParams_t.register_fields([('function', hipFunction_t, 0), ('gridDimX', ctypes.c_uint32, 8), ('gridDimY', ctypes.c_uint32, 12), ('gridDimZ', ctypes.c_uint32, 16), ('blockDimX', ctypes.c_uint32, 20), ('blockDimY', ctypes.c_uint32, 24), ('blockDimZ', ctypes.c_uint32, 28), ('sharedMemBytes', ctypes.c_uint32, 32), ('hStream', hipStream_t, 40), ('kernelParams', ctypes.POINTER(ctypes.c_void_p), 48)])
hipFunctionLaunchParams: TypeAlias = hipFunctionLaunchParams_t
hipExternalMemoryHandleType_enum: dict[int, str] = {(hipExternalMemoryHandleTypeOpaqueFd:=1): 'hipExternalMemoryHandleTypeOpaqueFd', (hipExternalMemoryHandleTypeOpaqueWin32:=2): 'hipExternalMemoryHandleTypeOpaqueWin32', (hipExternalMemoryHandleTypeOpaqueWin32Kmt:=3): 'hipExternalMemoryHandleTypeOpaqueWin32Kmt', (hipExternalMemoryHandleTypeD3D12Heap:=4): 'hipExternalMemoryHandleTypeD3D12Heap', (hipExternalMemoryHandleTypeD3D12Resource:=5): 'hipExternalMemoryHandleTypeD3D12Resource', (hipExternalMemoryHandleTypeD3D11Resource:=6): 'hipExternalMemoryHandleTypeD3D11Resource', (hipExternalMemoryHandleTypeD3D11ResourceKmt:=7): 'hipExternalMemoryHandleTypeD3D11ResourceKmt', (hipExternalMemoryHandleTypeNvSciBuf:=8): 'hipExternalMemoryHandleTypeNvSciBuf'}
hipExternalMemoryHandleType: TypeAlias = ctypes.c_uint32
@c.record
class hipExternalMemoryHandleDesc_st(c.Struct):
  SIZE = 104
  type: int
  handle: hipExternalMemoryHandleDesc_st_handle
  size: int
  flags: int
  reserved: ctypes.Array[ctypes.c_uint32]
@c.record
class hipExternalMemoryHandleDesc_st_handle(c.Struct):
  SIZE = 16
  fd: int
  win32: hipExternalMemoryHandleDesc_st_handle_win32
  nvSciBufObject: int|None
@c.record
class hipExternalMemoryHandleDesc_st_handle_win32(c.Struct):
  SIZE = 16
  handle: int|None
  name: int|None
hipExternalMemoryHandleDesc_st_handle_win32.register_fields([('handle', ctypes.c_void_p, 0), ('name', ctypes.c_void_p, 8)])
hipExternalMemoryHandleDesc_st_handle.register_fields([('fd', ctypes.c_int32, 0), ('win32', hipExternalMemoryHandleDesc_st_handle_win32, 0), ('nvSciBufObject', ctypes.c_void_p, 0)])
hipExternalMemoryHandleDesc_st.register_fields([('type', hipExternalMemoryHandleType, 0), ('handle', hipExternalMemoryHandleDesc_st_handle, 8), ('size', ctypes.c_uint64, 24), ('flags', ctypes.c_uint32, 32), ('reserved', (ctypes.c_uint32 * 16), 36)])
hipExternalMemoryHandleDesc: TypeAlias = hipExternalMemoryHandleDesc_st
@c.record
class hipExternalMemoryBufferDesc_st(c.Struct):
  SIZE = 88
  offset: int
  size: int
  flags: int
  reserved: ctypes.Array[ctypes.c_uint32]
hipExternalMemoryBufferDesc_st.register_fields([('offset', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('reserved', (ctypes.c_uint32 * 16), 20)])
hipExternalMemoryBufferDesc: TypeAlias = hipExternalMemoryBufferDesc_st
@c.record
class hipExternalMemoryMipmappedArrayDesc_st(c.Struct):
  SIZE = 64
  offset: int
  formatDesc: hipChannelFormatDesc
  extent: hipExtent
  flags: int
  numLevels: int
@c.record
class hipChannelFormatDesc(c.Struct):
  SIZE = 20
  x: int
  y: int
  z: int
  w: int
  f: int
hipChannelFormatKind: dict[int, str] = {(hipChannelFormatKindSigned:=0): 'hipChannelFormatKindSigned', (hipChannelFormatKindUnsigned:=1): 'hipChannelFormatKindUnsigned', (hipChannelFormatKindFloat:=2): 'hipChannelFormatKindFloat', (hipChannelFormatKindNone:=3): 'hipChannelFormatKindNone'}
hipChannelFormatDesc.register_fields([('x', ctypes.c_int32, 0), ('y', ctypes.c_int32, 4), ('z', ctypes.c_int32, 8), ('w', ctypes.c_int32, 12), ('f', ctypes.c_uint32, 16)])
@c.record
class hipExtent(c.Struct):
  SIZE = 24
  width: int
  height: int
  depth: int
hipExtent.register_fields([('width', size_t, 0), ('height', size_t, 8), ('depth', size_t, 16)])
hipExternalMemoryMipmappedArrayDesc_st.register_fields([('offset', ctypes.c_uint64, 0), ('formatDesc', hipChannelFormatDesc, 8), ('extent', hipExtent, 32), ('flags', ctypes.c_uint32, 56), ('numLevels', ctypes.c_uint32, 60)])
hipExternalMemoryMipmappedArrayDesc: TypeAlias = hipExternalMemoryMipmappedArrayDesc_st
hipExternalMemory_t: TypeAlias = ctypes.c_void_p
hipExternalSemaphoreHandleType_enum: dict[int, str] = {(hipExternalSemaphoreHandleTypeOpaqueFd:=1): 'hipExternalSemaphoreHandleTypeOpaqueFd', (hipExternalSemaphoreHandleTypeOpaqueWin32:=2): 'hipExternalSemaphoreHandleTypeOpaqueWin32', (hipExternalSemaphoreHandleTypeOpaqueWin32Kmt:=3): 'hipExternalSemaphoreHandleTypeOpaqueWin32Kmt', (hipExternalSemaphoreHandleTypeD3D12Fence:=4): 'hipExternalSemaphoreHandleTypeD3D12Fence', (hipExternalSemaphoreHandleTypeD3D11Fence:=5): 'hipExternalSemaphoreHandleTypeD3D11Fence', (hipExternalSemaphoreHandleTypeNvSciSync:=6): 'hipExternalSemaphoreHandleTypeNvSciSync', (hipExternalSemaphoreHandleTypeKeyedMutex:=7): 'hipExternalSemaphoreHandleTypeKeyedMutex', (hipExternalSemaphoreHandleTypeKeyedMutexKmt:=8): 'hipExternalSemaphoreHandleTypeKeyedMutexKmt', (hipExternalSemaphoreHandleTypeTimelineSemaphoreFd:=9): 'hipExternalSemaphoreHandleTypeTimelineSemaphoreFd', (hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32:=10): 'hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32'}
hipExternalSemaphoreHandleType: TypeAlias = ctypes.c_uint32
@c.record
class hipExternalSemaphoreHandleDesc_st(c.Struct):
  SIZE = 96
  type: int
  handle: hipExternalSemaphoreHandleDesc_st_handle
  flags: int
  reserved: ctypes.Array[ctypes.c_uint32]
@c.record
class hipExternalSemaphoreHandleDesc_st_handle(c.Struct):
  SIZE = 16
  fd: int
  win32: hipExternalSemaphoreHandleDesc_st_handle_win32
  NvSciSyncObj: int|None
@c.record
class hipExternalSemaphoreHandleDesc_st_handle_win32(c.Struct):
  SIZE = 16
  handle: int|None
  name: int|None
hipExternalSemaphoreHandleDesc_st_handle_win32.register_fields([('handle', ctypes.c_void_p, 0), ('name', ctypes.c_void_p, 8)])
hipExternalSemaphoreHandleDesc_st_handle.register_fields([('fd', ctypes.c_int32, 0), ('win32', hipExternalSemaphoreHandleDesc_st_handle_win32, 0), ('NvSciSyncObj', ctypes.c_void_p, 0)])
hipExternalSemaphoreHandleDesc_st.register_fields([('type', hipExternalSemaphoreHandleType, 0), ('handle', hipExternalSemaphoreHandleDesc_st_handle, 8), ('flags', ctypes.c_uint32, 24), ('reserved', (ctypes.c_uint32 * 16), 28)])
hipExternalSemaphoreHandleDesc: TypeAlias = hipExternalSemaphoreHandleDesc_st
hipExternalSemaphore_t: TypeAlias = ctypes.c_void_p
@c.record
class hipExternalSemaphoreSignalParams_st(c.Struct):
  SIZE = 144
  params: hipExternalSemaphoreSignalParams_st_params
  flags: int
  reserved: ctypes.Array[ctypes.c_uint32]
@c.record
class hipExternalSemaphoreSignalParams_st_params(c.Struct):
  SIZE = 72
  fence: hipExternalSemaphoreSignalParams_st_params_fence
  nvSciSync: hipExternalSemaphoreSignalParams_st_params_nvSciSync
  keyedMutex: hipExternalSemaphoreSignalParams_st_params_keyedMutex
  reserved: ctypes.Array[ctypes.c_uint32]
@c.record
class hipExternalSemaphoreSignalParams_st_params_fence(c.Struct):
  SIZE = 8
  value: int
hipExternalSemaphoreSignalParams_st_params_fence.register_fields([('value', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreSignalParams_st_params_nvSciSync(c.Struct):
  SIZE = 8
  fence: int|None
  reserved: int
hipExternalSemaphoreSignalParams_st_params_nvSciSync.register_fields([('fence', ctypes.c_void_p, 0), ('reserved', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreSignalParams_st_params_keyedMutex(c.Struct):
  SIZE = 8
  key: int
hipExternalSemaphoreSignalParams_st_params_keyedMutex.register_fields([('key', ctypes.c_uint64, 0)])
hipExternalSemaphoreSignalParams_st_params.register_fields([('fence', hipExternalSemaphoreSignalParams_st_params_fence, 0), ('nvSciSync', hipExternalSemaphoreSignalParams_st_params_nvSciSync, 8), ('keyedMutex', hipExternalSemaphoreSignalParams_st_params_keyedMutex, 16), ('reserved', (ctypes.c_uint32 * 12), 24)])
hipExternalSemaphoreSignalParams_st.register_fields([('params', hipExternalSemaphoreSignalParams_st_params, 0), ('flags', ctypes.c_uint32, 72), ('reserved', (ctypes.c_uint32 * 16), 76)])
hipExternalSemaphoreSignalParams: TypeAlias = hipExternalSemaphoreSignalParams_st
@c.record
class hipExternalSemaphoreWaitParams_st(c.Struct):
  SIZE = 144
  params: hipExternalSemaphoreWaitParams_st_params
  flags: int
  reserved: ctypes.Array[ctypes.c_uint32]
@c.record
class hipExternalSemaphoreWaitParams_st_params(c.Struct):
  SIZE = 72
  fence: hipExternalSemaphoreWaitParams_st_params_fence
  nvSciSync: hipExternalSemaphoreWaitParams_st_params_nvSciSync
  keyedMutex: hipExternalSemaphoreWaitParams_st_params_keyedMutex
  reserved: ctypes.Array[ctypes.c_uint32]
@c.record
class hipExternalSemaphoreWaitParams_st_params_fence(c.Struct):
  SIZE = 8
  value: int
hipExternalSemaphoreWaitParams_st_params_fence.register_fields([('value', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreWaitParams_st_params_nvSciSync(c.Struct):
  SIZE = 8
  fence: int|None
  reserved: int
hipExternalSemaphoreWaitParams_st_params_nvSciSync.register_fields([('fence', ctypes.c_void_p, 0), ('reserved', ctypes.c_uint64, 0)])
@c.record
class hipExternalSemaphoreWaitParams_st_params_keyedMutex(c.Struct):
  SIZE = 16
  key: int
  timeoutMs: int
hipExternalSemaphoreWaitParams_st_params_keyedMutex.register_fields([('key', ctypes.c_uint64, 0), ('timeoutMs', ctypes.c_uint32, 8)])
hipExternalSemaphoreWaitParams_st_params.register_fields([('fence', hipExternalSemaphoreWaitParams_st_params_fence, 0), ('nvSciSync', hipExternalSemaphoreWaitParams_st_params_nvSciSync, 8), ('keyedMutex', hipExternalSemaphoreWaitParams_st_params_keyedMutex, 16), ('reserved', (ctypes.c_uint32 * 10), 32)])
hipExternalSemaphoreWaitParams_st.register_fields([('params', hipExternalSemaphoreWaitParams_st_params, 0), ('flags', ctypes.c_uint32, 72), ('reserved', (ctypes.c_uint32 * 16), 76)])
hipExternalSemaphoreWaitParams: TypeAlias = hipExternalSemaphoreWaitParams_st
@dll.bind(None, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint32))
def __hipGetPCH(pch:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)], size:ctypes._Pointer[ctypes.c_uint32]) -> None: ...
hipGraphicsRegisterFlags: dict[int, str] = {(hipGraphicsRegisterFlagsNone:=0): 'hipGraphicsRegisterFlagsNone', (hipGraphicsRegisterFlagsReadOnly:=1): 'hipGraphicsRegisterFlagsReadOnly', (hipGraphicsRegisterFlagsWriteDiscard:=2): 'hipGraphicsRegisterFlagsWriteDiscard', (hipGraphicsRegisterFlagsSurfaceLoadStore:=4): 'hipGraphicsRegisterFlagsSurfaceLoadStore', (hipGraphicsRegisterFlagsTextureGather:=8): 'hipGraphicsRegisterFlagsTextureGather'}
class _hipGraphicsResource(c.Struct): pass
hipGraphicsResource: TypeAlias = _hipGraphicsResource
hipGraphicsResource_t: TypeAlias = ctypes.POINTER(_hipGraphicsResource)
class ihipGraph(c.Struct): pass
hipGraph_t: TypeAlias = ctypes.POINTER(ihipGraph)
class hipGraphNode(c.Struct): pass
hipGraphNode_t: TypeAlias = ctypes.POINTER(hipGraphNode)
class hipGraphExec(c.Struct): pass
hipGraphExec_t: TypeAlias = ctypes.POINTER(hipGraphExec)
class hipUserObject(c.Struct): pass
hipUserObject_t: TypeAlias = ctypes.POINTER(hipUserObject)
hipGraphNodeType: dict[int, str] = {(hipGraphNodeTypeKernel:=0): 'hipGraphNodeTypeKernel', (hipGraphNodeTypeMemcpy:=1): 'hipGraphNodeTypeMemcpy', (hipGraphNodeTypeMemset:=2): 'hipGraphNodeTypeMemset', (hipGraphNodeTypeHost:=3): 'hipGraphNodeTypeHost', (hipGraphNodeTypeGraph:=4): 'hipGraphNodeTypeGraph', (hipGraphNodeTypeEmpty:=5): 'hipGraphNodeTypeEmpty', (hipGraphNodeTypeWaitEvent:=6): 'hipGraphNodeTypeWaitEvent', (hipGraphNodeTypeEventRecord:=7): 'hipGraphNodeTypeEventRecord', (hipGraphNodeTypeExtSemaphoreSignal:=8): 'hipGraphNodeTypeExtSemaphoreSignal', (hipGraphNodeTypeExtSemaphoreWait:=9): 'hipGraphNodeTypeExtSemaphoreWait', (hipGraphNodeTypeMemAlloc:=10): 'hipGraphNodeTypeMemAlloc', (hipGraphNodeTypeMemFree:=11): 'hipGraphNodeTypeMemFree', (hipGraphNodeTypeMemcpyFromSymbol:=12): 'hipGraphNodeTypeMemcpyFromSymbol', (hipGraphNodeTypeMemcpyToSymbol:=13): 'hipGraphNodeTypeMemcpyToSymbol', (hipGraphNodeTypeBatchMemOp:=14): 'hipGraphNodeTypeBatchMemOp', (hipGraphNodeTypeCount:=15): 'hipGraphNodeTypeCount'}
hipHostFn_t: TypeAlias = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
@c.record
class hipHostNodeParams(c.Struct):
  SIZE = 16
  fn: ctypes._CFunctionType
  userData: int|None
hipHostNodeParams.register_fields([('fn', hipHostFn_t, 0), ('userData', ctypes.c_void_p, 8)])
@c.record
class hipKernelNodeParams(c.Struct):
  SIZE = 64
  blockDim: dim3
  extra: ctypes._Pointer[ctypes.c_void_p]
  func: int|None
  gridDim: dim3
  kernelParams: ctypes._Pointer[ctypes.c_void_p]
  sharedMemBytes: int
hipKernelNodeParams.register_fields([('blockDim', dim3, 0), ('extra', ctypes.POINTER(ctypes.c_void_p), 16), ('func', ctypes.c_void_p, 24), ('gridDim', dim3, 32), ('kernelParams', ctypes.POINTER(ctypes.c_void_p), 48), ('sharedMemBytes', ctypes.c_uint32, 56)])
@c.record
class hipMemsetParams(c.Struct):
  SIZE = 48
  dst: int|None
  elementSize: int
  height: int
  pitch: int
  value: int
  width: int
hipMemsetParams.register_fields([('dst', ctypes.c_void_p, 0), ('elementSize', ctypes.c_uint32, 8), ('height', size_t, 16), ('pitch', size_t, 24), ('value', ctypes.c_uint32, 32), ('width', size_t, 40)])
@c.record
class hipMemAllocNodeParams(c.Struct):
  SIZE = 120
  poolProps: hipMemPoolProps
  accessDescs: ctypes._Pointer[hipMemAccessDesc]
  accessDescCount: int
  bytesize: int
  dptr: int|None
hipMemAllocNodeParams.register_fields([('poolProps', hipMemPoolProps, 0), ('accessDescs', ctypes.POINTER(hipMemAccessDesc), 88), ('accessDescCount', size_t, 96), ('bytesize', size_t, 104), ('dptr', ctypes.c_void_p, 112)])
hipAccessProperty: dict[int, str] = {(hipAccessPropertyNormal:=0): 'hipAccessPropertyNormal', (hipAccessPropertyStreaming:=1): 'hipAccessPropertyStreaming', (hipAccessPropertyPersisting:=2): 'hipAccessPropertyPersisting'}
@c.record
class hipAccessPolicyWindow(c.Struct):
  SIZE = 32
  base_ptr: int|None
  hitProp: int
  hitRatio: float
  missProp: int
  num_bytes: int
hipAccessPolicyWindow.register_fields([('base_ptr', ctypes.c_void_p, 0), ('hitProp', ctypes.c_uint32, 8), ('hitRatio', ctypes.c_float, 12), ('missProp', ctypes.c_uint32, 16), ('num_bytes', size_t, 24)])
@c.record
class hipLaunchMemSyncDomainMap(c.Struct):
  SIZE = 2
  default_: int
  remote: int
hipLaunchMemSyncDomainMap.register_fields([('default_', ctypes.c_ubyte, 0), ('remote', ctypes.c_ubyte, 1)])
hipLaunchMemSyncDomain: dict[int, str] = {(hipLaunchMemSyncDomainDefault:=0): 'hipLaunchMemSyncDomainDefault', (hipLaunchMemSyncDomainRemote:=1): 'hipLaunchMemSyncDomainRemote'}
hipSynchronizationPolicy: dict[int, str] = {(hipSyncPolicyAuto:=1): 'hipSyncPolicyAuto', (hipSyncPolicySpin:=2): 'hipSyncPolicySpin', (hipSyncPolicyYield:=3): 'hipSyncPolicyYield', (hipSyncPolicyBlockingSync:=4): 'hipSyncPolicyBlockingSync'}
hipLaunchAttributeID: dict[int, str] = {(hipLaunchAttributeAccessPolicyWindow:=1): 'hipLaunchAttributeAccessPolicyWindow', (hipLaunchAttributeCooperative:=2): 'hipLaunchAttributeCooperative', (hipLaunchAttributeSynchronizationPolicy:=3): 'hipLaunchAttributeSynchronizationPolicy', (hipLaunchAttributePriority:=8): 'hipLaunchAttributePriority', (hipLaunchAttributeMemSyncDomainMap:=9): 'hipLaunchAttributeMemSyncDomainMap', (hipLaunchAttributeMemSyncDomain:=10): 'hipLaunchAttributeMemSyncDomain', (hipLaunchAttributeMax:=11): 'hipLaunchAttributeMax'}
@c.record
class hipLaunchAttributeValue(c.Struct):
  SIZE = 64
  pad: ctypes.Array[ctypes.c_char]
  accessPolicyWindow: hipAccessPolicyWindow
  cooperative: int
  priority: int
  syncPolicy: int
  memSyncDomainMap: hipLaunchMemSyncDomainMap
  memSyncDomain: int
hipLaunchAttributeValue.register_fields([('pad', (ctypes.c_char * 64), 0), ('accessPolicyWindow', hipAccessPolicyWindow, 0), ('cooperative', ctypes.c_int32, 0), ('priority', ctypes.c_int32, 0), ('syncPolicy', ctypes.c_uint32, 0), ('memSyncDomainMap', hipLaunchMemSyncDomainMap, 0), ('memSyncDomain', ctypes.c_uint32, 0)])
hipGraphExecUpdateResult: dict[int, str] = {(hipGraphExecUpdateSuccess:=0): 'hipGraphExecUpdateSuccess', (hipGraphExecUpdateError:=1): 'hipGraphExecUpdateError', (hipGraphExecUpdateErrorTopologyChanged:=2): 'hipGraphExecUpdateErrorTopologyChanged', (hipGraphExecUpdateErrorNodeTypeChanged:=3): 'hipGraphExecUpdateErrorNodeTypeChanged', (hipGraphExecUpdateErrorFunctionChanged:=4): 'hipGraphExecUpdateErrorFunctionChanged', (hipGraphExecUpdateErrorParametersChanged:=5): 'hipGraphExecUpdateErrorParametersChanged', (hipGraphExecUpdateErrorNotSupported:=6): 'hipGraphExecUpdateErrorNotSupported', (hipGraphExecUpdateErrorUnsupportedFunctionChange:=7): 'hipGraphExecUpdateErrorUnsupportedFunctionChange'}
hipStreamCaptureMode: dict[int, str] = {(hipStreamCaptureModeGlobal:=0): 'hipStreamCaptureModeGlobal', (hipStreamCaptureModeThreadLocal:=1): 'hipStreamCaptureModeThreadLocal', (hipStreamCaptureModeRelaxed:=2): 'hipStreamCaptureModeRelaxed'}
hipStreamCaptureStatus: dict[int, str] = {(hipStreamCaptureStatusNone:=0): 'hipStreamCaptureStatusNone', (hipStreamCaptureStatusActive:=1): 'hipStreamCaptureStatusActive', (hipStreamCaptureStatusInvalidated:=2): 'hipStreamCaptureStatusInvalidated'}
hipStreamUpdateCaptureDependenciesFlags: dict[int, str] = {(hipStreamAddCaptureDependencies:=0): 'hipStreamAddCaptureDependencies', (hipStreamSetCaptureDependencies:=1): 'hipStreamSetCaptureDependencies'}
hipGraphMemAttributeType: dict[int, str] = {(hipGraphMemAttrUsedMemCurrent:=0): 'hipGraphMemAttrUsedMemCurrent', (hipGraphMemAttrUsedMemHigh:=1): 'hipGraphMemAttrUsedMemHigh', (hipGraphMemAttrReservedMemCurrent:=2): 'hipGraphMemAttrReservedMemCurrent', (hipGraphMemAttrReservedMemHigh:=3): 'hipGraphMemAttrReservedMemHigh'}
hipUserObjectFlags: dict[int, str] = {(hipUserObjectNoDestructorSync:=1): 'hipUserObjectNoDestructorSync'}
hipUserObjectRetainFlags: dict[int, str] = {(hipGraphUserObjectMove:=1): 'hipGraphUserObjectMove'}
hipGraphInstantiateFlags: dict[int, str] = {(hipGraphInstantiateFlagAutoFreeOnLaunch:=1): 'hipGraphInstantiateFlagAutoFreeOnLaunch', (hipGraphInstantiateFlagUpload:=2): 'hipGraphInstantiateFlagUpload', (hipGraphInstantiateFlagDeviceLaunch:=4): 'hipGraphInstantiateFlagDeviceLaunch', (hipGraphInstantiateFlagUseNodePriority:=8): 'hipGraphInstantiateFlagUseNodePriority'}
hipGraphDebugDotFlags: dict[int, str] = {(hipGraphDebugDotFlagsVerbose:=1): 'hipGraphDebugDotFlagsVerbose', (hipGraphDebugDotFlagsKernelNodeParams:=4): 'hipGraphDebugDotFlagsKernelNodeParams', (hipGraphDebugDotFlagsMemcpyNodeParams:=8): 'hipGraphDebugDotFlagsMemcpyNodeParams', (hipGraphDebugDotFlagsMemsetNodeParams:=16): 'hipGraphDebugDotFlagsMemsetNodeParams', (hipGraphDebugDotFlagsHostNodeParams:=32): 'hipGraphDebugDotFlagsHostNodeParams', (hipGraphDebugDotFlagsEventNodeParams:=64): 'hipGraphDebugDotFlagsEventNodeParams', (hipGraphDebugDotFlagsExtSemasSignalNodeParams:=128): 'hipGraphDebugDotFlagsExtSemasSignalNodeParams', (hipGraphDebugDotFlagsExtSemasWaitNodeParams:=256): 'hipGraphDebugDotFlagsExtSemasWaitNodeParams', (hipGraphDebugDotFlagsKernelNodeAttributes:=512): 'hipGraphDebugDotFlagsKernelNodeAttributes', (hipGraphDebugDotFlagsHandles:=1024): 'hipGraphDebugDotFlagsHandles'}
hipGraphInstantiateResult: dict[int, str] = {(hipGraphInstantiateSuccess:=0): 'hipGraphInstantiateSuccess', (hipGraphInstantiateError:=1): 'hipGraphInstantiateError', (hipGraphInstantiateInvalidStructure:=2): 'hipGraphInstantiateInvalidStructure', (hipGraphInstantiateNodeOperationNotSupported:=3): 'hipGraphInstantiateNodeOperationNotSupported', (hipGraphInstantiateMultipleDevicesNotSupported:=4): 'hipGraphInstantiateMultipleDevicesNotSupported'}
@c.record
class hipGraphInstantiateParams(c.Struct):
  SIZE = 32
  errNode_out: ctypes._Pointer[hipGraphNode]
  flags: int
  result_out: int
  uploadStream: ctypes._Pointer[ihipStream_t]
hipGraphInstantiateParams.register_fields([('errNode_out', hipGraphNode_t, 0), ('flags', ctypes.c_uint64, 8), ('result_out', ctypes.c_uint32, 16), ('uploadStream', hipStream_t, 24)])
@c.record
class hipMemAllocationProp(c.Struct):
  SIZE = 32
  type: int
  requestedHandleType: int
  requestedHandleTypes: int
  location: hipMemLocation
  win32HandleMetaData: int|None
  allocFlags: hipMemAllocationProp_allocFlags
@c.record
class hipMemAllocationProp_allocFlags(c.Struct):
  SIZE = 4
  compressionType: int
  gpuDirectRDMACapable: int
  usage: int
hipMemAllocationProp_allocFlags.register_fields([('compressionType', ctypes.c_ubyte, 0), ('gpuDirectRDMACapable', ctypes.c_ubyte, 1), ('usage', ctypes.c_uint16, 2)])
hipMemAllocationProp.register_fields([('type', ctypes.c_uint32, 0), ('requestedHandleType', ctypes.c_uint32, 4), ('requestedHandleTypes', ctypes.c_uint32, 4), ('location', hipMemLocation, 8), ('win32HandleMetaData', ctypes.c_void_p, 16), ('allocFlags', hipMemAllocationProp_allocFlags, 24)])
@c.record
class hipExternalSemaphoreSignalNodeParams(c.Struct):
  SIZE = 24
  extSemArray: ctypes._Pointer[ctypes.c_void_p]
  paramsArray: ctypes._Pointer[hipExternalSemaphoreSignalParams_st]
  numExtSems: int
hipExternalSemaphoreSignalNodeParams.register_fields([('extSemArray', ctypes.POINTER(hipExternalSemaphore_t), 0), ('paramsArray', ctypes.POINTER(hipExternalSemaphoreSignalParams), 8), ('numExtSems', ctypes.c_uint32, 16)])
@c.record
class hipExternalSemaphoreWaitNodeParams(c.Struct):
  SIZE = 24
  extSemArray: ctypes._Pointer[ctypes.c_void_p]
  paramsArray: ctypes._Pointer[hipExternalSemaphoreWaitParams_st]
  numExtSems: int
hipExternalSemaphoreWaitNodeParams.register_fields([('extSemArray', ctypes.POINTER(hipExternalSemaphore_t), 0), ('paramsArray', ctypes.POINTER(hipExternalSemaphoreWaitParams), 8), ('numExtSems', ctypes.c_uint32, 16)])
class ihipMemGenericAllocationHandle(c.Struct): pass
hipMemGenericAllocationHandle_t: TypeAlias = ctypes.POINTER(ihipMemGenericAllocationHandle)
hipMemAllocationGranularity_flags: dict[int, str] = {(hipMemAllocationGranularityMinimum:=0): 'hipMemAllocationGranularityMinimum', (hipMemAllocationGranularityRecommended:=1): 'hipMemAllocationGranularityRecommended'}
hipMemHandleType: dict[int, str] = {(hipMemHandleTypeGeneric:=0): 'hipMemHandleTypeGeneric'}
hipMemOperationType: dict[int, str] = {(hipMemOperationTypeMap:=1): 'hipMemOperationTypeMap', (hipMemOperationTypeUnmap:=2): 'hipMemOperationTypeUnmap'}
hipArraySparseSubresourceType: dict[int, str] = {(hipArraySparseSubresourceTypeSparseLevel:=0): 'hipArraySparseSubresourceTypeSparseLevel', (hipArraySparseSubresourceTypeMiptail:=1): 'hipArraySparseSubresourceTypeMiptail'}
@c.record
class hipArrayMapInfo(c.Struct):
  SIZE = 152
  resourceType: int
  resource: hipArrayMapInfo_resource
  subresourceType: int
  subresource: hipArrayMapInfo_subresource
  memOperationType: int
  memHandleType: int
  memHandle: hipArrayMapInfo_memHandle
  offset: int
  deviceBitMask: int
  flags: int
  reserved: ctypes.Array[ctypes.c_uint32]
hipResourceType: dict[int, str] = {(hipResourceTypeArray:=0): 'hipResourceTypeArray', (hipResourceTypeMipmappedArray:=1): 'hipResourceTypeMipmappedArray', (hipResourceTypeLinear:=2): 'hipResourceTypeLinear', (hipResourceTypePitch2D:=3): 'hipResourceTypePitch2D'}
@c.record
class hipArrayMapInfo_resource(c.Struct):
  SIZE = 64
  mipmap: hipMipmappedArray
  array: ctypes._Pointer[hipArray]
@c.record
class hipMipmappedArray(c.Struct):
  SIZE = 64
  data: int|None
  desc: hipChannelFormatDesc
  type: int
  width: int
  height: int
  depth: int
  min_mipmap_level: int
  max_mipmap_level: int
  flags: int
  format: int
  num_channels: int
hipArray_Format: dict[int, str] = {(HIP_AD_FORMAT_UNSIGNED_INT8:=1): 'HIP_AD_FORMAT_UNSIGNED_INT8', (HIP_AD_FORMAT_UNSIGNED_INT16:=2): 'HIP_AD_FORMAT_UNSIGNED_INT16', (HIP_AD_FORMAT_UNSIGNED_INT32:=3): 'HIP_AD_FORMAT_UNSIGNED_INT32', (HIP_AD_FORMAT_SIGNED_INT8:=8): 'HIP_AD_FORMAT_SIGNED_INT8', (HIP_AD_FORMAT_SIGNED_INT16:=9): 'HIP_AD_FORMAT_SIGNED_INT16', (HIP_AD_FORMAT_SIGNED_INT32:=10): 'HIP_AD_FORMAT_SIGNED_INT32', (HIP_AD_FORMAT_HALF:=16): 'HIP_AD_FORMAT_HALF', (HIP_AD_FORMAT_FLOAT:=32): 'HIP_AD_FORMAT_FLOAT'}
hipMipmappedArray.register_fields([('data', ctypes.c_void_p, 0), ('desc', hipChannelFormatDesc, 8), ('type', ctypes.c_uint32, 28), ('width', ctypes.c_uint32, 32), ('height', ctypes.c_uint32, 36), ('depth', ctypes.c_uint32, 40), ('min_mipmap_level', ctypes.c_uint32, 44), ('max_mipmap_level', ctypes.c_uint32, 48), ('flags', ctypes.c_uint32, 52), ('format', ctypes.c_uint32, 56), ('num_channels', ctypes.c_uint32, 60)])
class hipArray(c.Struct): pass
hipArray_t: TypeAlias = ctypes.POINTER(hipArray)
hipArrayMapInfo_resource.register_fields([('mipmap', hipMipmappedArray, 0), ('array', hipArray_t, 0)])
@c.record
class hipArrayMapInfo_subresource(c.Struct):
  SIZE = 32
  sparseLevel: hipArrayMapInfo_subresource_sparseLevel
  miptail: hipArrayMapInfo_subresource_miptail
@c.record
class hipArrayMapInfo_subresource_sparseLevel(c.Struct):
  SIZE = 32
  level: int
  layer: int
  offsetX: int
  offsetY: int
  offsetZ: int
  extentWidth: int
  extentHeight: int
  extentDepth: int
hipArrayMapInfo_subresource_sparseLevel.register_fields([('level', ctypes.c_uint32, 0), ('layer', ctypes.c_uint32, 4), ('offsetX', ctypes.c_uint32, 8), ('offsetY', ctypes.c_uint32, 12), ('offsetZ', ctypes.c_uint32, 16), ('extentWidth', ctypes.c_uint32, 20), ('extentHeight', ctypes.c_uint32, 24), ('extentDepth', ctypes.c_uint32, 28)])
@c.record
class hipArrayMapInfo_subresource_miptail(c.Struct):
  SIZE = 24
  layer: int
  offset: int
  size: int
hipArrayMapInfo_subresource_miptail.register_fields([('layer', ctypes.c_uint32, 0), ('offset', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16)])
hipArrayMapInfo_subresource.register_fields([('sparseLevel', hipArrayMapInfo_subresource_sparseLevel, 0), ('miptail', hipArrayMapInfo_subresource_miptail, 0)])
@c.record
class hipArrayMapInfo_memHandle(c.Struct):
  SIZE = 8
  memHandle: ctypes._Pointer[ihipMemGenericAllocationHandle]
hipArrayMapInfo_memHandle.register_fields([('memHandle', hipMemGenericAllocationHandle_t, 0)])
hipArrayMapInfo.register_fields([('resourceType', ctypes.c_uint32, 0), ('resource', hipArrayMapInfo_resource, 8), ('subresourceType', ctypes.c_uint32, 72), ('subresource', hipArrayMapInfo_subresource, 80), ('memOperationType', ctypes.c_uint32, 112), ('memHandleType', ctypes.c_uint32, 116), ('memHandle', hipArrayMapInfo_memHandle, 120), ('offset', ctypes.c_uint64, 128), ('deviceBitMask', ctypes.c_uint32, 136), ('flags', ctypes.c_uint32, 140), ('reserved', (ctypes.c_uint32 * 2), 144)])
@c.record
class hipMemcpyNodeParams(c.Struct):
  SIZE = 176
  flags: int
  reserved: ctypes.Array[ctypes.c_int32]
  copyParams: hipMemcpy3DParms
@c.record
class hipMemcpy3DParms(c.Struct):
  SIZE = 160
  srcArray: ctypes._Pointer[hipArray]
  srcPos: hipPos
  srcPtr: hipPitchedPtr
  dstArray: ctypes._Pointer[hipArray]
  dstPos: hipPos
  dstPtr: hipPitchedPtr
  extent: hipExtent
  kind: int
@c.record
class hipPos(c.Struct):
  SIZE = 24
  x: int
  y: int
  z: int
hipPos.register_fields([('x', size_t, 0), ('y', size_t, 8), ('z', size_t, 16)])
@c.record
class hipPitchedPtr(c.Struct):
  SIZE = 32
  ptr: int|None
  pitch: int
  xsize: int
  ysize: int
hipPitchedPtr.register_fields([('ptr', ctypes.c_void_p, 0), ('pitch', size_t, 8), ('xsize', size_t, 16), ('ysize', size_t, 24)])
hipMemcpyKind: dict[int, str] = {(hipMemcpyHostToHost:=0): 'hipMemcpyHostToHost', (hipMemcpyHostToDevice:=1): 'hipMemcpyHostToDevice', (hipMemcpyDeviceToHost:=2): 'hipMemcpyDeviceToHost', (hipMemcpyDeviceToDevice:=3): 'hipMemcpyDeviceToDevice', (hipMemcpyDefault:=4): 'hipMemcpyDefault', (hipMemcpyDeviceToDeviceNoCU:=1024): 'hipMemcpyDeviceToDeviceNoCU'}
hipMemcpy3DParms.register_fields([('srcArray', hipArray_t, 0), ('srcPos', hipPos, 8), ('srcPtr', hipPitchedPtr, 32), ('dstArray', hipArray_t, 64), ('dstPos', hipPos, 72), ('dstPtr', hipPitchedPtr, 96), ('extent', hipExtent, 128), ('kind', ctypes.c_uint32, 152)])
hipMemcpyNodeParams.register_fields([('flags', ctypes.c_int32, 0), ('reserved', (ctypes.c_int32 * 3), 4), ('copyParams', hipMemcpy3DParms, 16)])
@c.record
class hipChildGraphNodeParams(c.Struct):
  SIZE = 8
  graph: ctypes._Pointer[ihipGraph]
hipChildGraphNodeParams.register_fields([('graph', hipGraph_t, 0)])
@c.record
class hipEventWaitNodeParams(c.Struct):
  SIZE = 8
  event: ctypes._Pointer[ihipEvent_t]
hipEventWaitNodeParams.register_fields([('event', hipEvent_t, 0)])
@c.record
class hipEventRecordNodeParams(c.Struct):
  SIZE = 8
  event: ctypes._Pointer[ihipEvent_t]
hipEventRecordNodeParams.register_fields([('event', hipEvent_t, 0)])
@c.record
class hipMemFreeNodeParams(c.Struct):
  SIZE = 8
  dptr: int|None
hipMemFreeNodeParams.register_fields([('dptr', ctypes.c_void_p, 0)])
@c.record
class hipGraphNodeParams(c.Struct):
  SIZE = 256
  type: int
  reserved0: ctypes.Array[ctypes.c_int32]
  reserved1: ctypes.Array[ctypes.c_int64]
  kernel: hipKernelNodeParams
  memcpy: hipMemcpyNodeParams
  memset: hipMemsetParams
  host: hipHostNodeParams
  graph: hipChildGraphNodeParams
  eventWait: hipEventWaitNodeParams
  eventRecord: hipEventRecordNodeParams
  extSemSignal: hipExternalSemaphoreSignalNodeParams
  extSemWait: hipExternalSemaphoreWaitNodeParams
  alloc: hipMemAllocNodeParams
  free: hipMemFreeNodeParams
  reserved2: int
hipGraphNodeParams.register_fields([('type', ctypes.c_uint32, 0), ('reserved0', (ctypes.c_int32 * 3), 4), ('reserved1', (ctypes.c_int64 * 29), 16), ('kernel', hipKernelNodeParams, 16), ('memcpy', hipMemcpyNodeParams, 16), ('memset', hipMemsetParams, 16), ('host', hipHostNodeParams, 16), ('graph', hipChildGraphNodeParams, 16), ('eventWait', hipEventWaitNodeParams, 16), ('eventRecord', hipEventRecordNodeParams, 16), ('extSemSignal', hipExternalSemaphoreSignalNodeParams, 16), ('extSemWait', hipExternalSemaphoreWaitNodeParams, 16), ('alloc', hipMemAllocNodeParams, 16), ('free', hipMemFreeNodeParams, 16), ('reserved2', ctypes.c_int64, 248)])
hipGraphDependencyType: dict[int, str] = {(hipGraphDependencyTypeDefault:=0): 'hipGraphDependencyTypeDefault', (hipGraphDependencyTypeProgrammatic:=1): 'hipGraphDependencyTypeProgrammatic'}
@c.record
class hipGraphEdgeData(c.Struct):
  SIZE = 8
  from_port: int
  reserved: ctypes.Array[ctypes.c_ubyte]
  to_port: int
  type: int
hipGraphEdgeData.register_fields([('from_port', ctypes.c_ubyte, 0), ('reserved', (ctypes.c_ubyte * 5), 1), ('to_port', ctypes.c_ubyte, 6), ('type', ctypes.c_ubyte, 7)])
@c.record
class hipLaunchAttribute_st(c.Struct):
  SIZE = 72
  id: int
  pad: ctypes.Array[ctypes.c_char]
  val: hipLaunchAttributeValue
  value: hipLaunchAttributeValue
hipLaunchAttribute_st.register_fields([('id', ctypes.c_uint32, 0), ('pad', (ctypes.c_char * 4), 4), ('val', hipLaunchAttributeValue, 8), ('value', hipLaunchAttributeValue, 8)])
hipLaunchAttribute: TypeAlias = hipLaunchAttribute_st
@c.record
class hipLaunchConfig_st(c.Struct):
  SIZE = 56
  gridDim: dim3
  blockDim: dim3
  dynamicSmemBytes: int
  stream: ctypes._Pointer[ihipStream_t]
  attrs: ctypes._Pointer[hipLaunchAttribute_st]
  numAttrs: int
hipLaunchConfig_st.register_fields([('gridDim', dim3, 0), ('blockDim', dim3, 12), ('dynamicSmemBytes', size_t, 24), ('stream', hipStream_t, 32), ('attrs', ctypes.POINTER(hipLaunchAttribute), 40), ('numAttrs', ctypes.c_uint32, 48)])
hipLaunchConfig_t: TypeAlias = hipLaunchConfig_st
@c.record
class HIP_LAUNCH_CONFIG_st(c.Struct):
  SIZE = 56
  gridDimX: int
  gridDimY: int
  gridDimZ: int
  blockDimX: int
  blockDimY: int
  blockDimZ: int
  sharedMemBytes: int
  hStream: ctypes._Pointer[ihipStream_t]
  attrs: ctypes._Pointer[hipLaunchAttribute_st]
  numAttrs: int
HIP_LAUNCH_CONFIG_st.register_fields([('gridDimX', ctypes.c_uint32, 0), ('gridDimY', ctypes.c_uint32, 4), ('gridDimZ', ctypes.c_uint32, 8), ('blockDimX', ctypes.c_uint32, 12), ('blockDimY', ctypes.c_uint32, 16), ('blockDimZ', ctypes.c_uint32, 20), ('sharedMemBytes', ctypes.c_uint32, 24), ('hStream', hipStream_t, 32), ('attrs', ctypes.POINTER(hipLaunchAttribute), 40), ('numAttrs', ctypes.c_uint32, 48)])
HIP_LAUNCH_CONFIG: TypeAlias = HIP_LAUNCH_CONFIG_st
hipMemRangeHandleType: dict[int, str] = {(hipMemRangeHandleTypeDmaBufFd:=1): 'hipMemRangeHandleTypeDmaBufFd', (hipMemRangeHandleTypeMax:=2147483647): 'hipMemRangeHandleTypeMax'}
hipMemRangeFlags: dict[int, str] = {(hipMemRangeFlagDmaBufMappingTypePcie:=1): 'hipMemRangeFlagDmaBufMappingTypePcie', (hipMemRangeFlagsMax:=2147483647): 'hipMemRangeFlagsMax'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipInit(flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32))
def hipDriverGetVersion(driverVersion:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32))
def hipRuntimeGetVersion(runtimeVersion:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDevice_t), ctypes.c_int32)
def hipDeviceGet(device:ctypes._Pointer[hipDevice_t], ordinal:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), hipDevice_t)
def hipDeviceComputeCapability(major:ctypes._Pointer[ctypes.c_int32], minor:ctypes._Pointer[ctypes.c_int32], device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, hipDevice_t)
def hipDeviceGetName(name:ctypes._Pointer[ctypes.c_char], len:int, device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipUUID), hipDevice_t)
def hipDeviceGetUuid(uuid:ctypes._Pointer[hipUUID], device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32)
def hipDeviceGetP2PAttribute(value:ctypes._Pointer[ctypes.c_int32], attr:ctypes.c_uint32, srcDevice:int, dstDevice:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32)
def hipDeviceGetPCIBusId(pciBusId:ctypes._Pointer[ctypes.c_char], len:int, device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_char))
def hipDeviceGetByPCIBusId(device:ctypes._Pointer[ctypes.c_int32], pciBusId:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), hipDevice_t)
def hipDeviceTotalMem(bytes:ctypes._Pointer[size_t], device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipDeviceSynchronize() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipDeviceReset() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hipSetDevice(deviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
def hipSetValidDevices(device_arr:ctypes._Pointer[ctypes.c_int32], len:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32))
def hipGetDevice(deviceId:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32))
def hipGetDeviceCount(count:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_int32)
def hipDeviceGetAttribute(pi:ctypes._Pointer[ctypes.c_int32], attr:ctypes.c_uint32, deviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemPool_t), ctypes.c_int32)
def hipDeviceGetDefaultMemPool(mem_pool:ctypes._Pointer[hipMemPool_t], device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, hipMemPool_t)
def hipDeviceSetMemPool(device:int, mem_pool:hipMemPool_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemPool_t), ctypes.c_int32)
def hipDeviceGetMemPool(mem_pool:ctypes._Pointer[hipMemPool_t], device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDeviceProp_tR0600), ctypes.c_int32)
def hipGetDevicePropertiesR0600(prop:ctypes._Pointer[hipDeviceProp_tR0600], deviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(hipChannelFormatDesc), ctypes.c_int32)
def hipDeviceGetTexture1DLinearMaxWidth(max_width:ctypes._Pointer[size_t], desc:ctypes._Pointer[hipChannelFormatDesc], device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipDeviceSetCacheConfig(cacheConfig:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipDeviceGetCacheConfig(cacheConfig:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.c_uint32)
def hipDeviceGetLimit(pValue:ctypes._Pointer[size_t], limit:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, size_t)
def hipDeviceSetLimit(limit:ctypes.c_uint32, value:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipDeviceGetSharedMemConfig(pConfig:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipGetDeviceFlags(flags:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipDeviceSetSharedMemConfig(config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipSetDeviceFlags(flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(hipDeviceProp_tR0600))
def hipChooseDeviceR0600(device:ctypes._Pointer[ctypes.c_int32], prop:ctypes._Pointer[hipDeviceProp_tR0600]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(uint32_t), ctypes.POINTER(uint32_t))
def hipExtGetLinkTypeAndHopCount(device1:int, device2:int, linktype:ctypes._Pointer[uint32_t], hopcount:ctypes._Pointer[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p)
def hipIpcGetMemHandle(handle:ctypes._Pointer[hipIpcMemHandle_t], devPtr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint32)
def hipIpcOpenMemHandle(devPtr:ctypes._Pointer[ctypes.c_void_p], handle:hipIpcMemHandle_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipIpcCloseMemHandle(devPtr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipIpcEventHandle_t), hipEvent_t)
def hipIpcGetEventHandle(handle:ctypes._Pointer[hipIpcEventHandle_t], event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipEvent_t), hipIpcEventHandle_t)
def hipIpcOpenEventHandle(event:ctypes._Pointer[hipEvent_t], handle:hipIpcEventHandle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32)
def hipFuncSetAttribute(func:int|None, attr:ctypes.c_uint32, value:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
def hipFuncSetCacheConfig(func:int|None, config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
def hipFuncSetSharedMemConfig(func:int|None, config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipGetLastError() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipExtGetLastError() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipPeekAtLastError() -> ctypes.c_uint32: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)
def hipGetErrorName(hip_error:ctypes.c_uint32) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)
def hipGetErrorString(hipError:ctypes.c_uint32) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
def hipDrvGetErrorName(hipError:ctypes.c_uint32, errorString:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
def hipDrvGetErrorString(hipError:ctypes.c_uint32, errorString:ctypes._Pointer[ctypes.POINTER(ctypes.c_char)]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipStream_t))
def hipStreamCreate(stream:ctypes._Pointer[hipStream_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipStream_t), ctypes.c_uint32)
def hipStreamCreateWithFlags(stream:ctypes._Pointer[hipStream_t], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipStream_t), ctypes.c_uint32, ctypes.c_int32)
def hipStreamCreateWithPriority(stream:ctypes._Pointer[hipStream_t], flags:int, priority:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32))
def hipDeviceGetStreamPriorityRange(leastPriority:ctypes._Pointer[ctypes.c_int32], greatestPriority:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t)
def hipStreamDestroy(stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t)
def hipStreamQuery(stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t)
def hipStreamSynchronize(stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, hipEvent_t, ctypes.c_uint32)
def hipStreamWaitEvent(stream:hipStream_t, event:hipEvent_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_uint32))
def hipStreamGetFlags(stream:hipStream_t, flags:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_uint64))
def hipStreamGetId(stream:hipStream_t, streamId:ctypes._Pointer[ctypes.c_uint64]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_int32))
def hipStreamGetPriority(stream:hipStream_t, priority:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(hipDevice_t))
def hipStreamGetDevice(stream:hipStream_t, device:ctypes._Pointer[hipDevice_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipStream_t), uint32_t, ctypes.POINTER(uint32_t))
def hipExtStreamCreateWithCUMask(stream:ctypes._Pointer[hipStream_t], cuMaskSize:uint32_t, cuMask:ctypes._Pointer[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, uint32_t, ctypes.POINTER(uint32_t))
def hipExtStreamGetCUMask(stream:hipStream_t, cuMaskSize:uint32_t, cuMask:ctypes._Pointer[uint32_t]) -> ctypes.c_uint32: ...
hipStreamCallback_t: TypeAlias = ctypes.CFUNCTYPE(None, ctypes.POINTER(ihipStream_t), ctypes.c_uint32, ctypes.c_void_p)
@dll.bind(ctypes.c_uint32, hipStream_t, hipStreamCallback_t, ctypes.c_void_p, ctypes.c_uint32)
def hipStreamAddCallback(stream:hipStream_t, callback:hipStreamCallback_t, userData:int|None, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32, ctypes.POINTER(hipLaunchAttributeValue))
def hipStreamSetAttribute(stream:hipStream_t, attr:ctypes.c_uint32, value:ctypes._Pointer[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32, ctypes.POINTER(hipLaunchAttributeValue))
def hipStreamGetAttribute(stream:hipStream_t, attr:ctypes.c_uint32, value_out:ctypes._Pointer[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32, uint32_t)
def hipStreamWaitValue32(stream:hipStream_t, ptr:int|None, value:uint32_t, flags:int, mask:uint32_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32, uint64_t)
def hipStreamWaitValue64(stream:hipStream_t, ptr:int|None, value:uint64_t, flags:int, mask:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint32_t, ctypes.c_uint32)
def hipStreamWriteValue32(stream:hipStream_t, ptr:int|None, value:uint32_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, uint64_t, ctypes.c_uint32)
def hipStreamWriteValue64(stream:hipStream_t, ptr:int|None, value:uint64_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32, ctypes.POINTER(hipStreamBatchMemOpParams), ctypes.c_uint32)
def hipStreamBatchMemOp(stream:hipStream_t, count:int, paramArray:ctypes._Pointer[hipStreamBatchMemOpParams], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipBatchMemOpNodeParams))
def hipGraphAddBatchMemOpNode(phGraphNode:ctypes._Pointer[hipGraphNode_t], hGraph:hipGraph_t, dependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, nodeParams:ctypes._Pointer[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipBatchMemOpNodeParams))
def hipGraphBatchMemOpNodeGetParams(hNode:hipGraphNode_t, nodeParams_out:ctypes._Pointer[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipBatchMemOpNodeParams))
def hipGraphBatchMemOpNodeSetParams(hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipBatchMemOpNodeParams))
def hipGraphExecBatchMemOpNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[hipBatchMemOpNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipEvent_t), ctypes.c_uint32)
def hipEventCreateWithFlags(event:ctypes._Pointer[hipEvent_t], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipEvent_t))
def hipEventCreate(event:ctypes._Pointer[hipEvent_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t, hipStream_t, ctypes.c_uint32)
def hipEventRecordWithFlags(event:hipEvent_t, stream:hipStream_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t, hipStream_t)
def hipEventRecord(event:hipEvent_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t)
def hipEventDestroy(event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t)
def hipEventSynchronize(event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), hipEvent_t, hipEvent_t)
def hipEventElapsedTime(ms:ctypes._Pointer[ctypes.c_float], start:hipEvent_t, stop:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipEvent_t)
def hipEventQuery(event:hipEvent_t) -> ctypes.c_uint32: ...
hipPointer_attribute: dict[int, str] = {(HIP_POINTER_ATTRIBUTE_CONTEXT:=1): 'HIP_POINTER_ATTRIBUTE_CONTEXT', (HIP_POINTER_ATTRIBUTE_MEMORY_TYPE:=2): 'HIP_POINTER_ATTRIBUTE_MEMORY_TYPE', (HIP_POINTER_ATTRIBUTE_DEVICE_POINTER:=3): 'HIP_POINTER_ATTRIBUTE_DEVICE_POINTER', (HIP_POINTER_ATTRIBUTE_HOST_POINTER:=4): 'HIP_POINTER_ATTRIBUTE_HOST_POINTER', (HIP_POINTER_ATTRIBUTE_P2P_TOKENS:=5): 'HIP_POINTER_ATTRIBUTE_P2P_TOKENS', (HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS:=6): 'HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS', (HIP_POINTER_ATTRIBUTE_BUFFER_ID:=7): 'HIP_POINTER_ATTRIBUTE_BUFFER_ID', (HIP_POINTER_ATTRIBUTE_IS_MANAGED:=8): 'HIP_POINTER_ATTRIBUTE_IS_MANAGED', (HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL:=9): 'HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL', (HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE:=10): 'HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE', (HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR:=11): 'HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR', (HIP_POINTER_ATTRIBUTE_RANGE_SIZE:=12): 'HIP_POINTER_ATTRIBUTE_RANGE_SIZE', (HIP_POINTER_ATTRIBUTE_MAPPED:=13): 'HIP_POINTER_ATTRIBUTE_MAPPED', (HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:=14): 'HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES', (HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:=15): 'HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE', (HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS:=16): 'HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS', (HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:=17): 'HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE'}
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, hipDeviceptr_t)
def hipPointerSetAttribute(value:int|None, attribute:ctypes.c_uint32, ptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipPointerAttribute_t), ctypes.c_void_p)
def hipPointerGetAttributes(attributes:ctypes._Pointer[hipPointerAttribute_t], ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, hipDeviceptr_t)
def hipPointerGetAttribute(data:int|None, attribute:ctypes.c_uint32, ptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p), hipDeviceptr_t)
def hipDrvPointerGetAttributes(numAttributes:int, attributes:ctypes._Pointer[ctypes.c_uint32], data:ctypes._Pointer[ctypes.c_void_p], ptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipExternalSemaphore_t), ctypes.POINTER(hipExternalSemaphoreHandleDesc))
def hipImportExternalSemaphore(extSem_out:ctypes._Pointer[hipExternalSemaphore_t], semHandleDesc:ctypes._Pointer[hipExternalSemaphoreHandleDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipExternalSemaphore_t), ctypes.POINTER(hipExternalSemaphoreSignalParams), ctypes.c_uint32, hipStream_t)
def hipSignalExternalSemaphoresAsync(extSemArray:ctypes._Pointer[hipExternalSemaphore_t], paramsArray:ctypes._Pointer[hipExternalSemaphoreSignalParams], numExtSems:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipExternalSemaphore_t), ctypes.POINTER(hipExternalSemaphoreWaitParams), ctypes.c_uint32, hipStream_t)
def hipWaitExternalSemaphoresAsync(extSemArray:ctypes._Pointer[hipExternalSemaphore_t], paramsArray:ctypes._Pointer[hipExternalSemaphoreWaitParams], numExtSems:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipExternalSemaphore_t)
def hipDestroyExternalSemaphore(extSem:hipExternalSemaphore_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipExternalMemory_t), ctypes.POINTER(hipExternalMemoryHandleDesc))
def hipImportExternalMemory(extMem_out:ctypes._Pointer[hipExternalMemory_t], memHandleDesc:ctypes._Pointer[hipExternalMemoryHandleDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), hipExternalMemory_t, ctypes.POINTER(hipExternalMemoryBufferDesc))
def hipExternalMemoryGetMappedBuffer(devPtr:ctypes._Pointer[ctypes.c_void_p], extMem:hipExternalMemory_t, bufferDesc:ctypes._Pointer[hipExternalMemoryBufferDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipExternalMemory_t)
def hipDestroyExternalMemory(extMem:hipExternalMemory_t) -> ctypes.c_uint32: ...
hipMipmappedArray_t: TypeAlias = ctypes.POINTER(hipMipmappedArray)
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMipmappedArray_t), hipExternalMemory_t, ctypes.POINTER(hipExternalMemoryMipmappedArrayDesc))
def hipExternalMemoryGetMappedMipmappedArray(mipmap:ctypes._Pointer[hipMipmappedArray_t], extMem:hipExternalMemory_t, mipmapDesc:ctypes._Pointer[hipExternalMemoryMipmappedArrayDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t)
def hipMalloc(ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint32)
def hipExtMallocWithFlags(ptr:ctypes._Pointer[ctypes.c_void_p], sizeBytes:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t)
def hipMallocHost(ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t)
def hipMemAllocHost(ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint32)
def hipHostMalloc(ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint32)
def hipMallocManaged(dev_ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_int32, hipStream_t)
def hipMemPrefetchAsync(dev_ptr:int|None, count:size_t, device:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hipMemLocation, ctypes.c_uint32, hipStream_t)
def hipMemPrefetchAsync_v2(dev_ptr:int|None, count:size_t, location:hipMemLocation, flags:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32, ctypes.c_int32)
def hipMemAdvise(dev_ptr:int|None, count:size_t, advice:ctypes.c_uint32, device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32, hipMemLocation)
def hipMemAdvise_v2(dev_ptr:int|None, count:size_t, advice:ctypes.c_uint32, location:hipMemLocation) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32, ctypes.c_void_p, size_t)
def hipMemRangeGetAttribute(data:int|None, data_size:size_t, attribute:ctypes.c_uint32, dev_ptr:int|None, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_uint32), size_t, ctypes.c_void_p, size_t)
def hipMemRangeGetAttributes(data:ctypes._Pointer[ctypes.c_void_p], data_sizes:ctypes._Pointer[size_t], attributes:ctypes._Pointer[ctypes.c_uint32], num_attributes:size_t, dev_ptr:int|None, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipStreamAttachMemAsync(stream:hipStream_t, dev_ptr:int|None, length:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t)
def hipMallocAsync(dev_ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipStream_t)
def hipFreeAsync(dev_ptr:int|None, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, size_t)
def hipMemPoolTrimTo(mem_pool:hipMemPool_t, min_bytes_to_hold:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, ctypes.c_uint32, ctypes.c_void_p)
def hipMemPoolSetAttribute(mem_pool:hipMemPool_t, attr:ctypes.c_uint32, value:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, ctypes.c_uint32, ctypes.c_void_p)
def hipMemPoolGetAttribute(mem_pool:hipMemPool_t, attr:ctypes.c_uint32, value:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t, ctypes.POINTER(hipMemAccessDesc), size_t)
def hipMemPoolSetAccess(mem_pool:hipMemPool_t, desc_list:ctypes._Pointer[hipMemAccessDesc], count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), hipMemPool_t, ctypes.POINTER(hipMemLocation))
def hipMemPoolGetAccess(flags:ctypes._Pointer[ctypes.c_uint32], mem_pool:hipMemPool_t, location:ctypes._Pointer[hipMemLocation]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemPool_t), ctypes.POINTER(hipMemPoolProps))
def hipMemPoolCreate(mem_pool:ctypes._Pointer[hipMemPool_t], pool_props:ctypes._Pointer[hipMemPoolProps]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemPool_t)
def hipMemPoolDestroy(mem_pool:hipMemPool_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, hipMemPool_t, hipStream_t)
def hipMallocFromPoolAsync(dev_ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t, mem_pool:hipMemPool_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipMemPool_t, ctypes.c_uint32, ctypes.c_uint32)
def hipMemPoolExportToShareableHandle(shared_handle:int|None, mem_pool:hipMemPool_t, handle_type:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemPool_t), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
def hipMemPoolImportFromShareableHandle(mem_pool:ctypes._Pointer[hipMemPool_t], shared_handle:int|None, handle_type:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemPoolPtrExportData), ctypes.c_void_p)
def hipMemPoolExportPointer(export_data:ctypes._Pointer[hipMemPoolPtrExportData], dev_ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), hipMemPool_t, ctypes.POINTER(hipMemPoolPtrExportData))
def hipMemPoolImportPointer(dev_ptr:ctypes._Pointer[ctypes.c_void_p], mem_pool:hipMemPool_t, export_data:ctypes._Pointer[hipMemPoolPtrExportData]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint32)
def hipHostAlloc(ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32)
def hipHostGetDevicePointer(devPtr:ctypes._Pointer[ctypes.c_void_p], hstPtr:int|None, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p)
def hipHostGetFlags(flagsPtr:ctypes._Pointer[ctypes.c_uint32], hostPtr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipHostRegister(hostPtr:int|None, sizeBytes:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipHostUnregister(hostPtr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), size_t, size_t)
def hipMallocPitch(ptr:ctypes._Pointer[ctypes.c_void_p], pitch:ctypes._Pointer[size_t], width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(size_t), size_t, size_t, ctypes.c_uint32)
def hipMemAllocPitch(dptr:ctypes._Pointer[hipDeviceptr_t], pitch:ctypes._Pointer[size_t], widthInBytes:size_t, height:size_t, elementSizeBytes:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipFree(ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipFreeHost(ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipHostFree(ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipMemcpy(dst:int|None, src:int|None, sizeBytes:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyWithStream(dst:int|None, src:int|None, sizeBytes:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_void_p, size_t)
def hipMemcpyHtoD(dst:hipDeviceptr_t, src:int|None, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipDeviceptr_t, size_t)
def hipMemcpyDtoH(dst:int|None, src:hipDeviceptr_t, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, hipDeviceptr_t, size_t)
def hipMemcpyDtoD(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, hipArray_t, size_t, size_t)
def hipMemcpyAtoD(dstDevice:hipDeviceptr_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, hipDeviceptr_t, size_t)
def hipMemcpyDtoA(dstArray:hipArray_t, dstOffset:size_t, srcDevice:hipDeviceptr_t, ByteCount:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, hipArray_t, size_t, size_t)
def hipMemcpyAtoA(dstArray:hipArray_t, dstOffset:size_t, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_void_p, size_t, hipStream_t)
def hipMemcpyHtoDAsync(dst:hipDeviceptr_t, src:int|None, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipDeviceptr_t, size_t, hipStream_t)
def hipMemcpyDtoHAsync(dst:int|None, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t)
def hipMemcpyDtoDAsync(dst:hipDeviceptr_t, src:hipDeviceptr_t, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipArray_t, size_t, size_t, hipStream_t)
def hipMemcpyAtoHAsync(dstHost:int|None, srcArray:hipArray_t, srcOffset:size_t, ByteCount:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, ctypes.c_void_p, size_t, hipStream_t)
def hipMemcpyHtoAAsync(dstArray:hipArray_t, dstOffset:size_t, srcHost:int|None, ByteCount:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(size_t), hipModule_t, ctypes.POINTER(ctypes.c_char))
def hipModuleGetGlobal(dptr:ctypes._Pointer[hipDeviceptr_t], bytes:ctypes._Pointer[size_t], hmod:hipModule_t, name:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
def hipGetSymbolAddress(devPtr:ctypes._Pointer[ctypes.c_void_p], symbol:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.c_void_p)
def hipGetSymbolSize(size:ctypes._Pointer[size_t], symbol:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32, uint64_t, ctypes.POINTER(ctypes.c_uint32))
def hipGetProcAddress(symbol:ctypes._Pointer[ctypes.c_char], pfn:ctypes._Pointer[ctypes.c_void_p], hipVersion:int, flags:uint64_t, symbolStatus:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipMemcpyToSymbol(symbol:int|None, src:int|None, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyToSymbolAsync(symbol:int|None, src:int|None, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipMemcpyFromSymbol(dst:int|None, symbol:int|None, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyFromSymbolAsync(dst:int|None, symbol:int|None, sizeBytes:size_t, offset:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpyAsync(dst:int|None, src:int|None, sizeBytes:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, size_t)
def hipMemset(dst:int|None, value:int, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_ubyte, size_t)
def hipMemsetD8(dest:hipDeviceptr_t, value:int, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_ubyte, size_t, hipStream_t)
def hipMemsetD8Async(dest:hipDeviceptr_t, value:int, count:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_uint16, size_t)
def hipMemsetD16(dest:hipDeviceptr_t, value:int, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_uint16, size_t, hipStream_t)
def hipMemsetD16Async(dest:hipDeviceptr_t, value:int, count:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_int32, size_t)
def hipMemsetD32(dest:hipDeviceptr_t, value:int, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t)
def hipMemsetAsync(dst:int|None, value:int, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, ctypes.c_int32, size_t, hipStream_t)
def hipMemsetD32Async(dst:hipDeviceptr_t, value:int, count:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t)
def hipMemset2D(dst:int|None, pitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_int32, size_t, size_t, hipStream_t)
def hipMemset2DAsync(dst:int|None, pitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipPitchedPtr, ctypes.c_int32, hipExtent)
def hipMemset3D(pitchedDevPtr:hipPitchedPtr, value:int, extent:hipExtent) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipPitchedPtr, ctypes.c_int32, hipExtent, hipStream_t)
def hipMemset3DAsync(pitchedDevPtr:hipPitchedPtr, value:int, extent:hipExtent, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_ubyte, size_t, size_t)
def hipMemsetD2D8(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_ubyte, size_t, size_t, hipStream_t)
def hipMemsetD2D8Async(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint16, size_t, size_t)
def hipMemsetD2D16(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint16, size_t, size_t, hipStream_t)
def hipMemsetD2D16Async(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint32, size_t, size_t)
def hipMemsetD2D32(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDeviceptr_t, size_t, ctypes.c_uint32, size_t, size_t, hipStream_t)
def hipMemsetD2D32Async(dst:hipDeviceptr_t, dstPitch:size_t, value:int, width:size_t, height:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(size_t))
def hipMemGetInfo(free:ctypes._Pointer[size_t], total:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(size_t))
def hipMemPtrGetInfo(ptr:int|None, size:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), ctypes.POINTER(hipChannelFormatDesc), size_t, size_t, ctypes.c_uint32)
def hipMallocArray(array:ctypes._Pointer[hipArray_t], desc:ctypes._Pointer[hipChannelFormatDesc], width:size_t, height:size_t, flags:int) -> ctypes.c_uint32: ...
@c.record
class HIP_ARRAY_DESCRIPTOR(c.Struct):
  SIZE = 24
  Width: int
  Height: int
  Format: int
  NumChannels: int
HIP_ARRAY_DESCRIPTOR.register_fields([('Width', size_t, 0), ('Height', size_t, 8), ('Format', ctypes.c_uint32, 16), ('NumChannels', ctypes.c_uint32, 20)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), ctypes.POINTER(HIP_ARRAY_DESCRIPTOR))
def hipArrayCreate(pHandle:ctypes._Pointer[hipArray_t], pAllocateArray:ctypes._Pointer[HIP_ARRAY_DESCRIPTOR]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t)
def hipArrayDestroy(array:hipArray_t) -> ctypes.c_uint32: ...
@c.record
class HIP_ARRAY3D_DESCRIPTOR(c.Struct):
  SIZE = 40
  Width: int
  Height: int
  Depth: int
  Format: int
  NumChannels: int
  Flags: int
HIP_ARRAY3D_DESCRIPTOR.register_fields([('Width', size_t, 0), ('Height', size_t, 8), ('Depth', size_t, 16), ('Format', ctypes.c_uint32, 24), ('NumChannels', ctypes.c_uint32, 28), ('Flags', ctypes.c_uint32, 32)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR))
def hipArray3DCreate(array:ctypes._Pointer[hipArray_t], pAllocateArray:ctypes._Pointer[HIP_ARRAY3D_DESCRIPTOR]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipPitchedPtr), hipExtent)
def hipMalloc3D(pitchedDevPtr:ctypes._Pointer[hipPitchedPtr], extent:hipExtent) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t)
def hipFreeArray(array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), ctypes.POINTER(hipChannelFormatDesc), hipExtent, ctypes.c_uint32)
def hipMalloc3DArray(array:ctypes._Pointer[hipArray_t], desc:ctypes._Pointer[hipChannelFormatDesc], extent:hipExtent, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipChannelFormatDesc), ctypes.POINTER(hipExtent), ctypes.POINTER(ctypes.c_uint32), hipArray_t)
def hipArrayGetInfo(desc:ctypes._Pointer[hipChannelFormatDesc], extent:ctypes._Pointer[hipExtent], flags:ctypes._Pointer[ctypes.c_uint32], array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_ARRAY_DESCRIPTOR), hipArray_t)
def hipArrayGetDescriptor(pArrayDescriptor:ctypes._Pointer[HIP_ARRAY_DESCRIPTOR], array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR), hipArray_t)
def hipArray3DGetDescriptor(pArrayDescriptor:ctypes._Pointer[HIP_ARRAY3D_DESCRIPTOR], array:hipArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2D(dst:int|None, dpitch:size_t, src:int|None, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@c.record
class hip_Memcpy2D(c.Struct):
  SIZE = 128
  srcXInBytes: int
  srcY: int
  srcMemoryType: int
  srcHost: int|None
  srcDevice: int|None
  srcArray: ctypes._Pointer[hipArray]
  srcPitch: int
  dstXInBytes: int
  dstY: int
  dstMemoryType: int
  dstHost: int|None
  dstDevice: int|None
  dstArray: ctypes._Pointer[hipArray]
  dstPitch: int
  WidthInBytes: int
  Height: int
hip_Memcpy2D.register_fields([('srcXInBytes', size_t, 0), ('srcY', size_t, 8), ('srcMemoryType', ctypes.c_uint32, 16), ('srcHost', ctypes.c_void_p, 24), ('srcDevice', hipDeviceptr_t, 32), ('srcArray', hipArray_t, 40), ('srcPitch', size_t, 48), ('dstXInBytes', size_t, 56), ('dstY', size_t, 64), ('dstMemoryType', ctypes.c_uint32, 72), ('dstHost', ctypes.c_void_p, 80), ('dstDevice', hipDeviceptr_t, 88), ('dstArray', hipArray_t, 96), ('dstPitch', size_t, 104), ('WidthInBytes', size_t, 112), ('Height', size_t, 120)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hip_Memcpy2D))
def hipMemcpyParam2D(pCopy:ctypes._Pointer[hip_Memcpy2D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hip_Memcpy2D), hipStream_t)
def hipMemcpyParam2DAsync(pCopy:ctypes._Pointer[hip_Memcpy2D], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpy2DAsync(dst:int|None, dpitch:size_t, src:int|None, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2DToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:int|None, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpy2DToArrayAsync(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:int|None, spitch:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
hipArray_const_t: TypeAlias = ctypes.POINTER(hipArray)
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2DArrayToArray(dst:hipArray_t, wOffsetDst:size_t, hOffsetDst:size_t, src:hipArray_const_t, wOffsetSrc:size_t, hOffsetSrc:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, size_t, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipMemcpyToArray(dst:hipArray_t, wOffset:size_t, hOffset:size_t, src:int|None, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipArray_const_t, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpyFromArray(dst:int|None, srcArray:hipArray_const_t, wOffset:size_t, hOffset:size_t, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, ctypes.c_uint32)
def hipMemcpy2DFromArray(dst:int|None, dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, ctypes.c_uint32, hipStream_t)
def hipMemcpy2DFromArrayAsync(dst:int|None, dpitch:size_t, src:hipArray_const_t, wOffset:size_t, hOffset:size_t, width:size_t, height:size_t, kind:ctypes.c_uint32, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipArray_t, size_t, size_t)
def hipMemcpyAtoH(dst:int|None, srcArray:hipArray_t, srcOffset:size_t, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipArray_t, size_t, ctypes.c_void_p, size_t)
def hipMemcpyHtoA(dstArray:hipArray_t, dstOffset:size_t, srcHost:int|None, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemcpy3DParms))
def hipMemcpy3D(p:ctypes._Pointer[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemcpy3DParms), hipStream_t)
def hipMemcpy3DAsync(p:ctypes._Pointer[hipMemcpy3DParms], stream:hipStream_t) -> ctypes.c_uint32: ...
@c.record
class HIP_MEMCPY3D(c.Struct):
  SIZE = 184
  srcXInBytes: int
  srcY: int
  srcZ: int
  srcLOD: int
  srcMemoryType: int
  srcHost: int|None
  srcDevice: int|None
  srcArray: ctypes._Pointer[hipArray]
  srcPitch: int
  srcHeight: int
  dstXInBytes: int
  dstY: int
  dstZ: int
  dstLOD: int
  dstMemoryType: int
  dstHost: int|None
  dstDevice: int|None
  dstArray: ctypes._Pointer[hipArray]
  dstPitch: int
  dstHeight: int
  WidthInBytes: int
  Height: int
  Depth: int
HIP_MEMCPY3D.register_fields([('srcXInBytes', size_t, 0), ('srcY', size_t, 8), ('srcZ', size_t, 16), ('srcLOD', size_t, 24), ('srcMemoryType', ctypes.c_uint32, 32), ('srcHost', ctypes.c_void_p, 40), ('srcDevice', hipDeviceptr_t, 48), ('srcArray', hipArray_t, 56), ('srcPitch', size_t, 64), ('srcHeight', size_t, 72), ('dstXInBytes', size_t, 80), ('dstY', size_t, 88), ('dstZ', size_t, 96), ('dstLOD', size_t, 104), ('dstMemoryType', ctypes.c_uint32, 112), ('dstHost', ctypes.c_void_p, 120), ('dstDevice', hipDeviceptr_t, 128), ('dstArray', hipArray_t, 136), ('dstPitch', size_t, 144), ('dstHeight', size_t, 152), ('WidthInBytes', size_t, 160), ('Height', size_t, 168), ('Depth', size_t, 176)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_MEMCPY3D))
def hipDrvMemcpy3D(pCopy:ctypes._Pointer[HIP_MEMCPY3D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_MEMCPY3D), hipStream_t)
def hipDrvMemcpy3DAsync(pCopy:ctypes._Pointer[HIP_MEMCPY3D], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(size_t), hipDeviceptr_t)
def hipMemGetAddressRange(pbase:ctypes._Pointer[hipDeviceptr_t], psize:ctypes._Pointer[size_t], dptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@c.record
class hipMemcpyAttributes(c.Struct):
  SIZE = 24
  srcAccessOrder: int
  srcLocHint: hipMemLocation
  dstLocHint: hipMemLocation
  flags: int
hipMemcpySrcAccessOrder: dict[int, str] = {(hipMemcpySrcAccessOrderInvalid:=0): 'hipMemcpySrcAccessOrderInvalid', (hipMemcpySrcAccessOrderStream:=1): 'hipMemcpySrcAccessOrderStream', (hipMemcpySrcAccessOrderDuringApiCall:=2): 'hipMemcpySrcAccessOrderDuringApiCall', (hipMemcpySrcAccessOrderAny:=3): 'hipMemcpySrcAccessOrderAny', (hipMemcpySrcAccessOrderMax:=2147483647): 'hipMemcpySrcAccessOrderMax'}
hipMemcpyAttributes.register_fields([('srcAccessOrder', ctypes.c_uint32, 0), ('srcLocHint', hipMemLocation, 4), ('dstLocHint', hipMemLocation, 12), ('flags', ctypes.c_uint32, 20)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), size_t, ctypes.POINTER(hipMemcpyAttributes), ctypes.POINTER(size_t), size_t, ctypes.POINTER(size_t), hipStream_t)
def hipMemcpyBatchAsync(dsts:ctypes._Pointer[ctypes.c_void_p], srcs:ctypes._Pointer[ctypes.c_void_p], sizes:ctypes._Pointer[size_t], count:size_t, attrs:ctypes._Pointer[hipMemcpyAttributes], attrsIdxs:ctypes._Pointer[size_t], numAttrs:size_t, failIdx:ctypes._Pointer[size_t], stream:hipStream_t) -> ctypes.c_uint32: ...
@c.record
class hipMemcpy3DBatchOp(c.Struct):
  SIZE = 112
  src: hipMemcpy3DOperand
  dst: hipMemcpy3DOperand
  extent: hipExtent
  srcAccessOrder: int
  flags: int
@c.record
class hipMemcpy3DOperand(c.Struct):
  SIZE = 40
  type: int
  op: hipMemcpy3DOperand_op
hipMemcpy3DOperandType: dict[int, str] = {(hipMemcpyOperandTypePointer:=1): 'hipMemcpyOperandTypePointer', (hipMemcpyOperandTypeArray:=2): 'hipMemcpyOperandTypeArray', (hipMemcpyOperandTypeMax:=2147483647): 'hipMemcpyOperandTypeMax'}
@c.record
class hipMemcpy3DOperand_op(c.Struct):
  SIZE = 32
  ptr: hipMemcpy3DOperand_op_ptr
  array: hipMemcpy3DOperand_op_array
@c.record
class hipMemcpy3DOperand_op_ptr(c.Struct):
  SIZE = 32
  ptr: int|None
  rowLength: int
  layerHeight: int
  locHint: hipMemLocation
hipMemcpy3DOperand_op_ptr.register_fields([('ptr', ctypes.c_void_p, 0), ('rowLength', size_t, 8), ('layerHeight', size_t, 16), ('locHint', hipMemLocation, 24)])
@c.record
class hipMemcpy3DOperand_op_array(c.Struct):
  SIZE = 32
  array: ctypes._Pointer[hipArray]
  offset: hipOffset3D
@c.record
class hipOffset3D(c.Struct):
  SIZE = 24
  x: int
  y: int
  z: int
hipOffset3D.register_fields([('x', size_t, 0), ('y', size_t, 8), ('z', size_t, 16)])
hipMemcpy3DOperand_op_array.register_fields([('array', hipArray_t, 0), ('offset', hipOffset3D, 8)])
hipMemcpy3DOperand_op.register_fields([('ptr', hipMemcpy3DOperand_op_ptr, 0), ('array', hipMemcpy3DOperand_op_array, 0)])
hipMemcpy3DOperand.register_fields([('type', ctypes.c_uint32, 0), ('op', hipMemcpy3DOperand_op, 8)])
hipMemcpy3DBatchOp.register_fields([('src', hipMemcpy3DOperand, 0), ('dst', hipMemcpy3DOperand, 40), ('extent', hipExtent, 80), ('srcAccessOrder', ctypes.c_uint32, 104), ('flags', ctypes.c_uint32, 108)])
@dll.bind(ctypes.c_uint32, size_t, ctypes.POINTER(hipMemcpy3DBatchOp), ctypes.POINTER(size_t), ctypes.c_uint64, hipStream_t)
def hipMemcpy3DBatchAsync(numOps:size_t, opList:ctypes._Pointer[hipMemcpy3DBatchOp], failIdx:ctypes._Pointer[size_t], flags:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@c.record
class hipMemcpy3DPeerParms(c.Struct):
  SIZE = 168
  srcArray: ctypes._Pointer[hipArray]
  srcPos: hipPos
  srcPtr: hipPitchedPtr
  srcDevice: int
  dstArray: ctypes._Pointer[hipArray]
  dstPos: hipPos
  dstPtr: hipPitchedPtr
  dstDevice: int
  extent: hipExtent
hipMemcpy3DPeerParms.register_fields([('srcArray', hipArray_t, 0), ('srcPos', hipPos, 8), ('srcPtr', hipPitchedPtr, 32), ('srcDevice', ctypes.c_int32, 64), ('dstArray', hipArray_t, 72), ('dstPos', hipPos, 80), ('dstPtr', hipPitchedPtr, 104), ('dstDevice', ctypes.c_int32, 136), ('extent', hipExtent, 144)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemcpy3DPeerParms))
def hipMemcpy3DPeer(p:ctypes._Pointer[hipMemcpy3DPeerParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemcpy3DPeerParms), hipStream_t)
def hipMemcpy3DPeerAsync(p:ctypes._Pointer[hipMemcpy3DPeerParms], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32)
def hipDeviceCanAccessPeer(canAccessPeer:ctypes._Pointer[ctypes.c_int32], deviceId:int, peerDeviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32)
def hipDeviceEnablePeerAccess(peerDeviceId:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hipDeviceDisablePeerAccess(peerDeviceId:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t)
def hipMemcpyPeer(dst:int|None, dstDeviceId:int, src:int|None, srcDeviceId:int, sizeBytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, size_t, hipStream_t)
def hipMemcpyPeerAsync(dst:int|None, dstDeviceId:int, src:int|None, srcDevice:int, sizeBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipCtx_t), ctypes.c_uint32, hipDevice_t)
def hipCtxCreate(ctx:ctypes._Pointer[hipCtx_t], flags:int, device:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxDestroy(ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipCtx_t))
def hipCtxPopCurrent(ctx:ctypes._Pointer[hipCtx_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxPushCurrent(ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxSetCurrent(ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipCtx_t))
def hipCtxGetCurrent(ctx:ctypes._Pointer[hipCtx_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDevice_t))
def hipCtxGetDevice(device:ctypes._Pointer[hipDevice_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t, ctypes.POINTER(ctypes.c_uint32))
def hipCtxGetApiVersion(ctx:hipCtx_t, apiVersion:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipCtxGetCacheConfig(cacheConfig:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipCtxSetCacheConfig(cacheConfig:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32)
def hipCtxSetSharedMemConfig(config:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipCtxGetSharedMemConfig(pConfig:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipCtxSynchronize() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipCtxGetFlags(flags:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t, ctypes.c_uint32)
def hipCtxEnablePeerAccess(peerCtx:hipCtx_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipCtx_t)
def hipCtxDisablePeerAccess(peerCtx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32))
def hipDevicePrimaryCtxGetState(dev:hipDevice_t, flags:ctypes._Pointer[ctypes.c_uint32], active:ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t)
def hipDevicePrimaryCtxRelease(dev:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipCtx_t), hipDevice_t)
def hipDevicePrimaryCtxRetain(pctx:ctypes._Pointer[hipCtx_t], dev:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t)
def hipDevicePrimaryCtxReset(dev:hipDevice_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipDevice_t, ctypes.c_uint32)
def hipDevicePrimaryCtxSetFlags(dev:hipDevice_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipModule_t), ctypes.c_void_p)
def hipModuleLoadFatBinary(module:ctypes._Pointer[hipModule_t], fatbin:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipModule_t), ctypes.POINTER(ctypes.c_char))
def hipModuleLoad(module:ctypes._Pointer[hipModule_t], fname:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipModule_t)
def hipModuleUnload(module:hipModule_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.POINTER(ctypes.c_char))
def hipModuleGetFunction(function:ctypes._Pointer[hipFunction_t], module:hipModule_t, kname:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), hipModule_t)
def hipModuleGetFunctionCount(count:ctypes._Pointer[ctypes.c_uint32], mod:hipModule_t) -> ctypes.c_uint32: ...
hipLibraryOption_e: dict[int, str] = {(hipLibraryHostUniversalFunctionAndDataTable:=0): 'hipLibraryHostUniversalFunctionAndDataTable', (hipLibraryBinaryIsPreserved:=1): 'hipLibraryBinaryIsPreserved'}
hipLibraryOption: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipLibrary_t), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(hipLibraryOption)), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32)
def hipLibraryLoadData(library:ctypes._Pointer[hipLibrary_t], code:int|None, jitOptions:ctypes._Pointer[ctypes.POINTER(ctypes.c_uint32)], jitOptionsValues:ctypes._Pointer[ctypes.c_void_p], numJitOptions:int, libraryOptions:ctypes._Pointer[ctypes.POINTER(hipLibraryOption)], libraryOptionValues:ctypes._Pointer[ctypes.c_void_p], numLibraryOptions:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipLibrary_t), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(hipLibraryOption)), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32)
def hipLibraryLoadFromFile(library:ctypes._Pointer[hipLibrary_t], fileName:ctypes._Pointer[ctypes.c_char], jitOptions:ctypes._Pointer[ctypes.POINTER(ctypes.c_uint32)], jitOptionsValues:ctypes._Pointer[ctypes.c_void_p], numJitOptions:int, libraryOptions:ctypes._Pointer[ctypes.POINTER(hipLibraryOption)], libraryOptionValues:ctypes._Pointer[ctypes.c_void_p], numLibraryOptions:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLibrary_t)
def hipLibraryUnload(library:hipLibrary_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipKernel_t), hipLibrary_t, ctypes.POINTER(ctypes.c_char))
def hipLibraryGetKernel(pKernel:ctypes._Pointer[hipKernel_t], library:hipLibrary_t, name:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), hipLibrary_t)
def hipLibraryGetKernelCount(count:ctypes._Pointer[ctypes.c_uint32], library:hipLibrary_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipFuncAttributes), ctypes.c_void_p)
def hipFuncGetAttributes(attr:ctypes._Pointer[hipFuncAttributes], func:int|None) -> ctypes.c_uint32: ...
hipFunction_attribute: dict[int, str] = {(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:=0): 'HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK', (HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:=1): 'HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:=2): 'HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:=3): 'HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_NUM_REGS:=4): 'HIP_FUNC_ATTRIBUTE_NUM_REGS', (HIP_FUNC_ATTRIBUTE_PTX_VERSION:=5): 'HIP_FUNC_ATTRIBUTE_PTX_VERSION', (HIP_FUNC_ATTRIBUTE_BINARY_VERSION:=6): 'HIP_FUNC_ATTRIBUTE_BINARY_VERSION', (HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:=7): 'HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA', (HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:=8): 'HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES', (HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:=9): 'HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT', (HIP_FUNC_ATTRIBUTE_MAX:=10): 'HIP_FUNC_ATTRIBUTE_MAX'}
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, hipFunction_t)
def hipFuncGetAttribute(value:ctypes._Pointer[ctypes.c_int32], attrib:ctypes.c_uint32, hfunc:hipFunction_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipFunction_t), ctypes.c_void_p)
def hipGetFuncBySymbol(functionPtr:ctypes._Pointer[hipFunction_t], symbolPtr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32))
def hipGetDriverEntryPoint(symbol:ctypes._Pointer[ctypes.c_char], funcPtr:ctypes._Pointer[ctypes.c_void_p], flags:int, driverStatus:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@c.record
class textureReference(c.Struct):
  SIZE = 88
  normalized: int
  readMode: int
  filterMode: int
  addressMode: ctypes.Array[ctypes.c_uint32]
  channelDesc: hipChannelFormatDesc
  sRGB: int
  maxAnisotropy: int
  mipmapFilterMode: int
  mipmapLevelBias: float
  minMipmapLevelClamp: float
  maxMipmapLevelClamp: float
  textureObject: ctypes._Pointer[__hip_texture]
  numChannels: int
  format: int
hipTextureReadMode: dict[int, str] = {(hipReadModeElementType:=0): 'hipReadModeElementType', (hipReadModeNormalizedFloat:=1): 'hipReadModeNormalizedFloat'}
hipTextureFilterMode: dict[int, str] = {(hipFilterModePoint:=0): 'hipFilterModePoint', (hipFilterModeLinear:=1): 'hipFilterModeLinear'}
hipTextureAddressMode: dict[int, str] = {(hipAddressModeWrap:=0): 'hipAddressModeWrap', (hipAddressModeClamp:=1): 'hipAddressModeClamp', (hipAddressModeMirror:=2): 'hipAddressModeMirror', (hipAddressModeBorder:=3): 'hipAddressModeBorder'}
class __hip_texture(c.Struct): pass
hipTextureObject_t: TypeAlias = ctypes.POINTER(__hip_texture)
textureReference.register_fields([('normalized', ctypes.c_int32, 0), ('readMode', ctypes.c_uint32, 4), ('filterMode', ctypes.c_uint32, 8), ('addressMode', (ctypes.c_uint32 * 3), 12), ('channelDesc', hipChannelFormatDesc, 24), ('sRGB', ctypes.c_int32, 44), ('maxAnisotropy', ctypes.c_uint32, 48), ('mipmapFilterMode', ctypes.c_uint32, 52), ('mipmapLevelBias', ctypes.c_float, 56), ('minMipmapLevelClamp', ctypes.c_float, 60), ('maxMipmapLevelClamp', ctypes.c_float, 64), ('textureObject', hipTextureObject_t, 72), ('numChannels', ctypes.c_int32, 80), ('format', ctypes.c_uint32, 84)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(textureReference)), hipModule_t, ctypes.POINTER(ctypes.c_char))
def hipModuleGetTexRef(texRef:ctypes._Pointer[ctypes.POINTER(textureReference)], hmod:hipModule_t, name:ctypes._Pointer[ctypes.c_char]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipModule_t), ctypes.c_void_p)
def hipModuleLoadData(module:ctypes._Pointer[hipModule_t], image:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipModule_t), ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p))
def hipModuleLoadDataEx(module:ctypes._Pointer[hipModule_t], image:int|None, numOptions:int, options:ctypes._Pointer[ctypes.c_uint32], optionValues:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t, ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p))
def hipLinkAddData(state:hipLinkState_t, type:ctypes.c_uint32, data:int|None, size:size_t, name:ctypes._Pointer[ctypes.c_char], numOptions:int, options:ctypes._Pointer[ctypes.c_uint32], optionValues:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p))
def hipLinkAddFile(state:hipLinkState_t, type:ctypes.c_uint32, path:ctypes._Pointer[ctypes.c_char], numOptions:int, options:ctypes._Pointer[ctypes.c_uint32], optionValues:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t))
def hipLinkComplete(state:hipLinkState_t, hipBinOut:ctypes._Pointer[ctypes.c_void_p], sizeOut:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(hipLinkState_t))
def hipLinkCreate(numOptions:int, options:ctypes._Pointer[ctypes.c_uint32], optionValues:ctypes._Pointer[ctypes.c_void_p], stateOut:ctypes._Pointer[hipLinkState_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipLinkState_t)
def hipLinkDestroy(state:hipLinkState_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))
def hipModuleLaunchKernel(f:hipFunction_t, gridDimX:int, gridDimY:int, gridDimZ:int, blockDimX:int, blockDimY:int, blockDimZ:int, sharedMemBytes:int, stream:hipStream_t, kernelParams:ctypes._Pointer[ctypes.c_void_p], extra:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipFunction_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_void_p))
def hipModuleLaunchCooperativeKernel(f:hipFunction_t, gridDimX:int, gridDimY:int, gridDimZ:int, blockDimX:int, blockDimY:int, blockDimZ:int, sharedMemBytes:int, stream:hipStream_t, kernelParams:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipFunctionLaunchParams), ctypes.c_uint32, ctypes.c_uint32)
def hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList:ctypes._Pointer[hipFunctionLaunchParams], numDevices:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, hipStream_t)
def hipLaunchCooperativeKernel(f:int|None, gridDim:dim3, blockDimX:dim3, kernelParams:ctypes._Pointer[ctypes.c_void_p], sharedMemBytes:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipLaunchParams), ctypes.c_int32, ctypes.c_uint32)
def hipLaunchCooperativeKernelMultiDevice(launchParamsList:ctypes._Pointer[hipLaunchParams], numDevices:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipLaunchParams), ctypes.c_int32, ctypes.c_uint32)
def hipExtLaunchMultiKernelMultiDevice(launchParamsList:ctypes._Pointer[hipLaunchParams], numDevices:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipLaunchConfig_t), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
def hipLaunchKernelExC(config:ctypes._Pointer[hipLaunchConfig_t], fPtr:int|None, args:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_LAUNCH_CONFIG), hipFunction_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))
def hipDrvLaunchKernelEx(config:ctypes._Pointer[HIP_LAUNCH_CONFIG], f:hipFunction_t, params:ctypes._Pointer[ctypes.c_void_p], extra:ctypes._Pointer[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipDeviceptr_t, size_t, ctypes.c_uint32, ctypes.c_uint64)
def hipMemGetHandleForAddressRange(handle:int|None, dptr:hipDeviceptr_t, size:size_t, handleType:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32)
def hipModuleOccupancyMaxPotentialBlockSize(gridSize:ctypes._Pointer[ctypes.c_int32], blockSize:ctypes._Pointer[ctypes.c_int32], f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), hipFunction_t, size_t, ctypes.c_int32, ctypes.c_uint32)
def hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize:ctypes._Pointer[ctypes.c_int32], blockSize:ctypes._Pointer[ctypes.c_int32], f:hipFunction_t, dynSharedMemPerBlk:size_t, blockSizeLimit:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t)
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:ctypes._Pointer[ctypes.c_int32], f:hipFunction_t, blockSize:int, dynSharedMemPerBlk:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), hipFunction_t, ctypes.c_int32, size_t, ctypes.c_uint32)
def hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:ctypes._Pointer[ctypes.c_int32], f:hipFunction_t, blockSize:int, dynSharedMemPerBlk:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int32, size_t)
def hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:ctypes._Pointer[ctypes.c_int32], f:int|None, blockSize:int, dynSharedMemPerBlk:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p, ctypes.c_int32, size_t, ctypes.c_uint32)
def hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:ctypes._Pointer[ctypes.c_int32], f:int|None, blockSize:int, dynSharedMemPerBlk:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p, size_t, ctypes.c_int32)
def hipOccupancyMaxPotentialBlockSize(gridSize:ctypes._Pointer[ctypes.c_int32], blockSize:ctypes._Pointer[ctypes.c_int32], f:int|None, dynSharedMemPerBlk:size_t, blockSizeLimit:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipProfilerStart() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hipProfilerStop() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, dim3, dim3, size_t, hipStream_t)
def hipConfigureCall(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, size_t)
def hipSetupArgument(arg:int|None, size:size_t, offset:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hipLaunchByPtr(func:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, dim3, dim3, size_t, hipStream_t)
def __hipPushCallConfiguration(gridDim:dim3, blockDim:dim3, sharedMem:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(dim3), ctypes.POINTER(dim3), ctypes.POINTER(size_t), ctypes.POINTER(hipStream_t))
def __hipPopCallConfiguration(gridDim:ctypes._Pointer[dim3], blockDim:ctypes._Pointer[dim3], sharedMem:ctypes._Pointer[size_t], stream:ctypes._Pointer[hipStream_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, dim3, dim3, ctypes.POINTER(ctypes.c_void_p), size_t, hipStream_t)
def hipLaunchKernel(function_address:int|None, numBlocks:dim3, dimBlocks:dim3, args:ctypes._Pointer[ctypes.c_void_p], sharedMemBytes:size_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, hipHostFn_t, ctypes.c_void_p)
def hipLaunchHostFunc(stream:hipStream_t, fn:hipHostFn_t, userData:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hip_Memcpy2D))
def hipDrvMemcpy2DUnaligned(pCopy:ctypes._Pointer[hip_Memcpy2D]) -> ctypes.c_uint32: ...
@c.record
class hipResourceDesc(c.Struct):
  SIZE = 64
  resType: int
  res: hipResourceDesc_res
@c.record
class hipResourceDesc_res(c.Struct):
  SIZE = 56
  array: hipResourceDesc_res_array
  mipmap: hipResourceDesc_res_mipmap
  linear: hipResourceDesc_res_linear
  pitch2D: hipResourceDesc_res_pitch2D
@c.record
class hipResourceDesc_res_array(c.Struct):
  SIZE = 8
  array: ctypes._Pointer[hipArray]
hipResourceDesc_res_array.register_fields([('array', hipArray_t, 0)])
@c.record
class hipResourceDesc_res_mipmap(c.Struct):
  SIZE = 8
  mipmap: ctypes._Pointer[hipMipmappedArray]
hipResourceDesc_res_mipmap.register_fields([('mipmap', hipMipmappedArray_t, 0)])
@c.record
class hipResourceDesc_res_linear(c.Struct):
  SIZE = 40
  devPtr: int|None
  desc: hipChannelFormatDesc
  sizeInBytes: int
hipResourceDesc_res_linear.register_fields([('devPtr', ctypes.c_void_p, 0), ('desc', hipChannelFormatDesc, 8), ('sizeInBytes', size_t, 32)])
@c.record
class hipResourceDesc_res_pitch2D(c.Struct):
  SIZE = 56
  devPtr: int|None
  desc: hipChannelFormatDesc
  width: int
  height: int
  pitchInBytes: int
hipResourceDesc_res_pitch2D.register_fields([('devPtr', ctypes.c_void_p, 0), ('desc', hipChannelFormatDesc, 8), ('width', size_t, 32), ('height', size_t, 40), ('pitchInBytes', size_t, 48)])
hipResourceDesc_res.register_fields([('array', hipResourceDesc_res_array, 0), ('mipmap', hipResourceDesc_res_mipmap, 0), ('linear', hipResourceDesc_res_linear, 0), ('pitch2D', hipResourceDesc_res_pitch2D, 0)])
hipResourceDesc.register_fields([('resType', ctypes.c_uint32, 0), ('res', hipResourceDesc_res, 8)])
@c.record
class hipTextureDesc(c.Struct):
  SIZE = 64
  addressMode: ctypes.Array[ctypes.c_uint32]
  filterMode: int
  readMode: int
  sRGB: int
  borderColor: ctypes.Array[ctypes.c_float]
  normalizedCoords: int
  maxAnisotropy: int
  mipmapFilterMode: int
  mipmapLevelBias: float
  minMipmapLevelClamp: float
  maxMipmapLevelClamp: float
hipTextureDesc.register_fields([('addressMode', (ctypes.c_uint32 * 3), 0), ('filterMode', ctypes.c_uint32, 12), ('readMode', ctypes.c_uint32, 16), ('sRGB', ctypes.c_int32, 20), ('borderColor', (ctypes.c_float * 4), 24), ('normalizedCoords', ctypes.c_int32, 40), ('maxAnisotropy', ctypes.c_uint32, 44), ('mipmapFilterMode', ctypes.c_uint32, 48), ('mipmapLevelBias', ctypes.c_float, 52), ('minMipmapLevelClamp', ctypes.c_float, 56), ('maxMipmapLevelClamp', ctypes.c_float, 60)])
@c.record
class hipResourceViewDesc(c.Struct):
  SIZE = 48
  format: int
  width: int
  height: int
  depth: int
  firstMipmapLevel: int
  lastMipmapLevel: int
  firstLayer: int
  lastLayer: int
hipResourceViewFormat: dict[int, str] = {(hipResViewFormatNone:=0): 'hipResViewFormatNone', (hipResViewFormatUnsignedChar1:=1): 'hipResViewFormatUnsignedChar1', (hipResViewFormatUnsignedChar2:=2): 'hipResViewFormatUnsignedChar2', (hipResViewFormatUnsignedChar4:=3): 'hipResViewFormatUnsignedChar4', (hipResViewFormatSignedChar1:=4): 'hipResViewFormatSignedChar1', (hipResViewFormatSignedChar2:=5): 'hipResViewFormatSignedChar2', (hipResViewFormatSignedChar4:=6): 'hipResViewFormatSignedChar4', (hipResViewFormatUnsignedShort1:=7): 'hipResViewFormatUnsignedShort1', (hipResViewFormatUnsignedShort2:=8): 'hipResViewFormatUnsignedShort2', (hipResViewFormatUnsignedShort4:=9): 'hipResViewFormatUnsignedShort4', (hipResViewFormatSignedShort1:=10): 'hipResViewFormatSignedShort1', (hipResViewFormatSignedShort2:=11): 'hipResViewFormatSignedShort2', (hipResViewFormatSignedShort4:=12): 'hipResViewFormatSignedShort4', (hipResViewFormatUnsignedInt1:=13): 'hipResViewFormatUnsignedInt1', (hipResViewFormatUnsignedInt2:=14): 'hipResViewFormatUnsignedInt2', (hipResViewFormatUnsignedInt4:=15): 'hipResViewFormatUnsignedInt4', (hipResViewFormatSignedInt1:=16): 'hipResViewFormatSignedInt1', (hipResViewFormatSignedInt2:=17): 'hipResViewFormatSignedInt2', (hipResViewFormatSignedInt4:=18): 'hipResViewFormatSignedInt4', (hipResViewFormatHalf1:=19): 'hipResViewFormatHalf1', (hipResViewFormatHalf2:=20): 'hipResViewFormatHalf2', (hipResViewFormatHalf4:=21): 'hipResViewFormatHalf4', (hipResViewFormatFloat1:=22): 'hipResViewFormatFloat1', (hipResViewFormatFloat2:=23): 'hipResViewFormatFloat2', (hipResViewFormatFloat4:=24): 'hipResViewFormatFloat4', (hipResViewFormatUnsignedBlockCompressed1:=25): 'hipResViewFormatUnsignedBlockCompressed1', (hipResViewFormatUnsignedBlockCompressed2:=26): 'hipResViewFormatUnsignedBlockCompressed2', (hipResViewFormatUnsignedBlockCompressed3:=27): 'hipResViewFormatUnsignedBlockCompressed3', (hipResViewFormatUnsignedBlockCompressed4:=28): 'hipResViewFormatUnsignedBlockCompressed4', (hipResViewFormatSignedBlockCompressed4:=29): 'hipResViewFormatSignedBlockCompressed4', (hipResViewFormatUnsignedBlockCompressed5:=30): 'hipResViewFormatUnsignedBlockCompressed5', (hipResViewFormatSignedBlockCompressed5:=31): 'hipResViewFormatSignedBlockCompressed5', (hipResViewFormatUnsignedBlockCompressed6H:=32): 'hipResViewFormatUnsignedBlockCompressed6H', (hipResViewFormatSignedBlockCompressed6H:=33): 'hipResViewFormatSignedBlockCompressed6H', (hipResViewFormatUnsignedBlockCompressed7:=34): 'hipResViewFormatUnsignedBlockCompressed7'}
hipResourceViewDesc.register_fields([('format', ctypes.c_uint32, 0), ('width', size_t, 8), ('height', size_t, 16), ('depth', size_t, 24), ('firstMipmapLevel', ctypes.c_uint32, 32), ('lastMipmapLevel', ctypes.c_uint32, 36), ('firstLayer', ctypes.c_uint32, 40), ('lastLayer', ctypes.c_uint32, 44)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipTextureObject_t), ctypes.POINTER(hipResourceDesc), ctypes.POINTER(hipTextureDesc), ctypes.POINTER(hipResourceViewDesc))
def hipCreateTextureObject(pTexObject:ctypes._Pointer[hipTextureObject_t], pResDesc:ctypes._Pointer[hipResourceDesc], pTexDesc:ctypes._Pointer[hipTextureDesc], pResViewDesc:ctypes._Pointer[hipResourceViewDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipTextureObject_t)
def hipDestroyTextureObject(textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipChannelFormatDesc), hipArray_const_t)
def hipGetChannelDesc(desc:ctypes._Pointer[hipChannelFormatDesc], array:hipArray_const_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipResourceDesc), hipTextureObject_t)
def hipGetTextureObjectResourceDesc(pResDesc:ctypes._Pointer[hipResourceDesc], textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipResourceViewDesc), hipTextureObject_t)
def hipGetTextureObjectResourceViewDesc(pResViewDesc:ctypes._Pointer[hipResourceViewDesc], textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipTextureDesc), hipTextureObject_t)
def hipGetTextureObjectTextureDesc(pTexDesc:ctypes._Pointer[hipTextureDesc], textureObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@c.record
class HIP_RESOURCE_DESC_st(c.Struct):
  SIZE = 144
  resType: int
  res: HIP_RESOURCE_DESC_st_res
  flags: int
HIP_RESOURCE_DESC: TypeAlias = HIP_RESOURCE_DESC_st
HIPresourcetype_enum: dict[int, str] = {(HIP_RESOURCE_TYPE_ARRAY:=0): 'HIP_RESOURCE_TYPE_ARRAY', (HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY:=1): 'HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY', (HIP_RESOURCE_TYPE_LINEAR:=2): 'HIP_RESOURCE_TYPE_LINEAR', (HIP_RESOURCE_TYPE_PITCH2D:=3): 'HIP_RESOURCE_TYPE_PITCH2D'}
HIPresourcetype: TypeAlias = ctypes.c_uint32
@c.record
class HIP_RESOURCE_DESC_st_res(c.Struct):
  SIZE = 128
  array: HIP_RESOURCE_DESC_st_res_array
  mipmap: HIP_RESOURCE_DESC_st_res_mipmap
  linear: HIP_RESOURCE_DESC_st_res_linear
  pitch2D: HIP_RESOURCE_DESC_st_res_pitch2D
  reserved: HIP_RESOURCE_DESC_st_res_reserved
@c.record
class HIP_RESOURCE_DESC_st_res_array(c.Struct):
  SIZE = 8
  hArray: ctypes._Pointer[hipArray]
HIP_RESOURCE_DESC_st_res_array.register_fields([('hArray', hipArray_t, 0)])
@c.record
class HIP_RESOURCE_DESC_st_res_mipmap(c.Struct):
  SIZE = 8
  hMipmappedArray: ctypes._Pointer[hipMipmappedArray]
HIP_RESOURCE_DESC_st_res_mipmap.register_fields([('hMipmappedArray', hipMipmappedArray_t, 0)])
@c.record
class HIP_RESOURCE_DESC_st_res_linear(c.Struct):
  SIZE = 24
  devPtr: int|None
  format: int
  numChannels: int
  sizeInBytes: int
HIP_RESOURCE_DESC_st_res_linear.register_fields([('devPtr', hipDeviceptr_t, 0), ('format', ctypes.c_uint32, 8), ('numChannels', ctypes.c_uint32, 12), ('sizeInBytes', size_t, 16)])
@c.record
class HIP_RESOURCE_DESC_st_res_pitch2D(c.Struct):
  SIZE = 40
  devPtr: int|None
  format: int
  numChannels: int
  width: int
  height: int
  pitchInBytes: int
HIP_RESOURCE_DESC_st_res_pitch2D.register_fields([('devPtr', hipDeviceptr_t, 0), ('format', ctypes.c_uint32, 8), ('numChannels', ctypes.c_uint32, 12), ('width', size_t, 16), ('height', size_t, 24), ('pitchInBytes', size_t, 32)])
@c.record
class HIP_RESOURCE_DESC_st_res_reserved(c.Struct):
  SIZE = 128
  reserved: ctypes.Array[ctypes.c_int32]
HIP_RESOURCE_DESC_st_res_reserved.register_fields([('reserved', (ctypes.c_int32 * 32), 0)])
HIP_RESOURCE_DESC_st_res.register_fields([('array', HIP_RESOURCE_DESC_st_res_array, 0), ('mipmap', HIP_RESOURCE_DESC_st_res_mipmap, 0), ('linear', HIP_RESOURCE_DESC_st_res_linear, 0), ('pitch2D', HIP_RESOURCE_DESC_st_res_pitch2D, 0), ('reserved', HIP_RESOURCE_DESC_st_res_reserved, 0)])
HIP_RESOURCE_DESC_st.register_fields([('resType', HIPresourcetype, 0), ('res', HIP_RESOURCE_DESC_st_res, 8), ('flags', ctypes.c_uint32, 136)])
@c.record
class HIP_TEXTURE_DESC_st(c.Struct):
  SIZE = 104
  addressMode: ctypes.Array[ctypes.c_uint32]
  filterMode: int
  flags: int
  maxAnisotropy: int
  mipmapFilterMode: int
  mipmapLevelBias: float
  minMipmapLevelClamp: float
  maxMipmapLevelClamp: float
  borderColor: ctypes.Array[ctypes.c_float]
  reserved: ctypes.Array[ctypes.c_int32]
HIP_TEXTURE_DESC: TypeAlias = HIP_TEXTURE_DESC_st
HIPaddress_mode_enum: dict[int, str] = {(HIP_TR_ADDRESS_MODE_WRAP:=0): 'HIP_TR_ADDRESS_MODE_WRAP', (HIP_TR_ADDRESS_MODE_CLAMP:=1): 'HIP_TR_ADDRESS_MODE_CLAMP', (HIP_TR_ADDRESS_MODE_MIRROR:=2): 'HIP_TR_ADDRESS_MODE_MIRROR', (HIP_TR_ADDRESS_MODE_BORDER:=3): 'HIP_TR_ADDRESS_MODE_BORDER'}
HIPaddress_mode: TypeAlias = ctypes.c_uint32
HIPfilter_mode_enum: dict[int, str] = {(HIP_TR_FILTER_MODE_POINT:=0): 'HIP_TR_FILTER_MODE_POINT', (HIP_TR_FILTER_MODE_LINEAR:=1): 'HIP_TR_FILTER_MODE_LINEAR'}
HIPfilter_mode: TypeAlias = ctypes.c_uint32
HIP_TEXTURE_DESC_st.register_fields([('addressMode', (HIPaddress_mode * 3), 0), ('filterMode', HIPfilter_mode, 12), ('flags', ctypes.c_uint32, 16), ('maxAnisotropy', ctypes.c_uint32, 20), ('mipmapFilterMode', HIPfilter_mode, 24), ('mipmapLevelBias', ctypes.c_float, 28), ('minMipmapLevelClamp', ctypes.c_float, 32), ('maxMipmapLevelClamp', ctypes.c_float, 36), ('borderColor', (ctypes.c_float * 4), 40), ('reserved', (ctypes.c_int32 * 12), 56)])
@c.record
class HIP_RESOURCE_VIEW_DESC_st(c.Struct):
  SIZE = 112
  format: int
  width: int
  height: int
  depth: int
  firstMipmapLevel: int
  lastMipmapLevel: int
  firstLayer: int
  lastLayer: int
  reserved: ctypes.Array[ctypes.c_uint32]
HIP_RESOURCE_VIEW_DESC: TypeAlias = HIP_RESOURCE_VIEW_DESC_st
HIPresourceViewFormat_enum: dict[int, str] = {(HIP_RES_VIEW_FORMAT_NONE:=0): 'HIP_RES_VIEW_FORMAT_NONE', (HIP_RES_VIEW_FORMAT_UINT_1X8:=1): 'HIP_RES_VIEW_FORMAT_UINT_1X8', (HIP_RES_VIEW_FORMAT_UINT_2X8:=2): 'HIP_RES_VIEW_FORMAT_UINT_2X8', (HIP_RES_VIEW_FORMAT_UINT_4X8:=3): 'HIP_RES_VIEW_FORMAT_UINT_4X8', (HIP_RES_VIEW_FORMAT_SINT_1X8:=4): 'HIP_RES_VIEW_FORMAT_SINT_1X8', (HIP_RES_VIEW_FORMAT_SINT_2X8:=5): 'HIP_RES_VIEW_FORMAT_SINT_2X8', (HIP_RES_VIEW_FORMAT_SINT_4X8:=6): 'HIP_RES_VIEW_FORMAT_SINT_4X8', (HIP_RES_VIEW_FORMAT_UINT_1X16:=7): 'HIP_RES_VIEW_FORMAT_UINT_1X16', (HIP_RES_VIEW_FORMAT_UINT_2X16:=8): 'HIP_RES_VIEW_FORMAT_UINT_2X16', (HIP_RES_VIEW_FORMAT_UINT_4X16:=9): 'HIP_RES_VIEW_FORMAT_UINT_4X16', (HIP_RES_VIEW_FORMAT_SINT_1X16:=10): 'HIP_RES_VIEW_FORMAT_SINT_1X16', (HIP_RES_VIEW_FORMAT_SINT_2X16:=11): 'HIP_RES_VIEW_FORMAT_SINT_2X16', (HIP_RES_VIEW_FORMAT_SINT_4X16:=12): 'HIP_RES_VIEW_FORMAT_SINT_4X16', (HIP_RES_VIEW_FORMAT_UINT_1X32:=13): 'HIP_RES_VIEW_FORMAT_UINT_1X32', (HIP_RES_VIEW_FORMAT_UINT_2X32:=14): 'HIP_RES_VIEW_FORMAT_UINT_2X32', (HIP_RES_VIEW_FORMAT_UINT_4X32:=15): 'HIP_RES_VIEW_FORMAT_UINT_4X32', (HIP_RES_VIEW_FORMAT_SINT_1X32:=16): 'HIP_RES_VIEW_FORMAT_SINT_1X32', (HIP_RES_VIEW_FORMAT_SINT_2X32:=17): 'HIP_RES_VIEW_FORMAT_SINT_2X32', (HIP_RES_VIEW_FORMAT_SINT_4X32:=18): 'HIP_RES_VIEW_FORMAT_SINT_4X32', (HIP_RES_VIEW_FORMAT_FLOAT_1X16:=19): 'HIP_RES_VIEW_FORMAT_FLOAT_1X16', (HIP_RES_VIEW_FORMAT_FLOAT_2X16:=20): 'HIP_RES_VIEW_FORMAT_FLOAT_2X16', (HIP_RES_VIEW_FORMAT_FLOAT_4X16:=21): 'HIP_RES_VIEW_FORMAT_FLOAT_4X16', (HIP_RES_VIEW_FORMAT_FLOAT_1X32:=22): 'HIP_RES_VIEW_FORMAT_FLOAT_1X32', (HIP_RES_VIEW_FORMAT_FLOAT_2X32:=23): 'HIP_RES_VIEW_FORMAT_FLOAT_2X32', (HIP_RES_VIEW_FORMAT_FLOAT_4X32:=24): 'HIP_RES_VIEW_FORMAT_FLOAT_4X32', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC1:=25): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC1', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC2:=26): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC2', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC3:=27): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC3', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC4:=28): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC4', (HIP_RES_VIEW_FORMAT_SIGNED_BC4:=29): 'HIP_RES_VIEW_FORMAT_SIGNED_BC4', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC5:=30): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC5', (HIP_RES_VIEW_FORMAT_SIGNED_BC5:=31): 'HIP_RES_VIEW_FORMAT_SIGNED_BC5', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H:=32): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H', (HIP_RES_VIEW_FORMAT_SIGNED_BC6H:=33): 'HIP_RES_VIEW_FORMAT_SIGNED_BC6H', (HIP_RES_VIEW_FORMAT_UNSIGNED_BC7:=34): 'HIP_RES_VIEW_FORMAT_UNSIGNED_BC7'}
HIPresourceViewFormat: TypeAlias = ctypes.c_uint32
HIP_RESOURCE_VIEW_DESC_st.register_fields([('format', HIPresourceViewFormat, 0), ('width', size_t, 8), ('height', size_t, 16), ('depth', size_t, 24), ('firstMipmapLevel', ctypes.c_uint32, 32), ('lastMipmapLevel', ctypes.c_uint32, 36), ('firstLayer', ctypes.c_uint32, 40), ('lastLayer', ctypes.c_uint32, 44), ('reserved', (ctypes.c_uint32 * 16), 48)])
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipTextureObject_t), ctypes.POINTER(HIP_RESOURCE_DESC), ctypes.POINTER(HIP_TEXTURE_DESC), ctypes.POINTER(HIP_RESOURCE_VIEW_DESC))
def hipTexObjectCreate(pTexObject:ctypes._Pointer[hipTextureObject_t], pResDesc:ctypes._Pointer[HIP_RESOURCE_DESC], pTexDesc:ctypes._Pointer[HIP_TEXTURE_DESC], pResViewDesc:ctypes._Pointer[HIP_RESOURCE_VIEW_DESC]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipTextureObject_t)
def hipTexObjectDestroy(texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_RESOURCE_DESC), hipTextureObject_t)
def hipTexObjectGetResourceDesc(pResDesc:ctypes._Pointer[HIP_RESOURCE_DESC], texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_RESOURCE_VIEW_DESC), hipTextureObject_t)
def hipTexObjectGetResourceViewDesc(pResViewDesc:ctypes._Pointer[HIP_RESOURCE_VIEW_DESC], texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(HIP_TEXTURE_DESC), hipTextureObject_t)
def hipTexObjectGetTextureDesc(pTexDesc:ctypes._Pointer[HIP_TEXTURE_DESC], texObject:hipTextureObject_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMipmappedArray_t), ctypes.POINTER(hipChannelFormatDesc), hipExtent, ctypes.c_uint32, ctypes.c_uint32)
def hipMallocMipmappedArray(mipmappedArray:ctypes._Pointer[hipMipmappedArray_t], desc:ctypes._Pointer[hipChannelFormatDesc], extent:hipExtent, numLevels:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMipmappedArray_t)
def hipFreeMipmappedArray(mipmappedArray:hipMipmappedArray_t) -> ctypes.c_uint32: ...
hipMipmappedArray_const_t: TypeAlias = ctypes.POINTER(hipMipmappedArray)
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), hipMipmappedArray_const_t, ctypes.c_uint32)
def hipGetMipmappedArrayLevel(levelArray:ctypes._Pointer[hipArray_t], mipmappedArray:hipMipmappedArray_const_t, level:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMipmappedArray_t), ctypes.POINTER(HIP_ARRAY3D_DESCRIPTOR), ctypes.c_uint32)
def hipMipmappedArrayCreate(pHandle:ctypes._Pointer[hipMipmappedArray_t], pMipmappedArrayDesc:ctypes._Pointer[HIP_ARRAY3D_DESCRIPTOR], numMipmapLevels:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMipmappedArray_t)
def hipMipmappedArrayDestroy(hMipmappedArray:hipMipmappedArray_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), hipMipmappedArray_t, ctypes.c_uint32)
def hipMipmappedArrayGetLevel(pLevelArray:ctypes._Pointer[hipArray_t], hMipMappedArray:hipMipmappedArray_t, level:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), hipMipmappedArray_const_t, ctypes.POINTER(hipChannelFormatDesc))
def hipBindTextureToMipmappedArray(tex:ctypes._Pointer[textureReference], mipmappedArray:hipMipmappedArray_const_t, desc:ctypes._Pointer[hipChannelFormatDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(textureReference)), ctypes.c_void_p)
def hipGetTextureReference(texref:ctypes._Pointer[ctypes.POINTER(textureReference)], symbol:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(textureReference))
def hipTexRefGetBorderColor(pBorderColor:ctypes._Pointer[ctypes.c_float], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), ctypes.POINTER(textureReference))
def hipTexRefGetArray(pArray:ctypes._Pointer[hipArray_t], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_int32, ctypes.c_uint32)
def hipTexRefSetAddressMode(texRef:ctypes._Pointer[textureReference], dim:int, am:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), hipArray_const_t, ctypes.c_uint32)
def hipTexRefSetArray(tex:ctypes._Pointer[textureReference], array:hipArray_const_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_uint32)
def hipTexRefSetFilterMode(texRef:ctypes._Pointer[textureReference], fm:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_uint32)
def hipTexRefSetFlags(texRef:ctypes._Pointer[textureReference], Flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_uint32, ctypes.c_int32)
def hipTexRefSetFormat(texRef:ctypes._Pointer[textureReference], fmt:ctypes.c_uint32, NumPackedComponents:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(textureReference), ctypes.c_void_p, ctypes.POINTER(hipChannelFormatDesc), size_t)
def hipBindTexture(offset:ctypes._Pointer[size_t], tex:ctypes._Pointer[textureReference], devPtr:int|None, desc:ctypes._Pointer[hipChannelFormatDesc], size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(textureReference), ctypes.c_void_p, ctypes.POINTER(hipChannelFormatDesc), size_t, size_t, size_t)
def hipBindTexture2D(offset:ctypes._Pointer[size_t], tex:ctypes._Pointer[textureReference], devPtr:int|None, desc:ctypes._Pointer[hipChannelFormatDesc], width:size_t, height:size_t, pitch:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), hipArray_const_t, ctypes.POINTER(hipChannelFormatDesc))
def hipBindTextureToArray(tex:ctypes._Pointer[textureReference], array:hipArray_const_t, desc:ctypes._Pointer[hipChannelFormatDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(textureReference))
def hipGetTextureAlignmentOffset(offset:ctypes._Pointer[size_t], texref:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference))
def hipUnbindTexture(tex:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipDeviceptr_t), ctypes.POINTER(textureReference))
def hipTexRefGetAddress(dev_ptr:ctypes._Pointer[hipDeviceptr_t], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(textureReference), ctypes.c_int32)
def hipTexRefGetAddressMode(pam:ctypes._Pointer[ctypes.c_uint32], texRef:ctypes._Pointer[textureReference], dim:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(textureReference))
def hipTexRefGetFilterMode(pfm:ctypes._Pointer[ctypes.c_uint32], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(textureReference))
def hipTexRefGetFlags(pFlags:ctypes._Pointer[ctypes.c_uint32], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(textureReference))
def hipTexRefGetFormat(pFormat:ctypes._Pointer[ctypes.c_uint32], pNumChannels:ctypes._Pointer[ctypes.c_int32], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(textureReference))
def hipTexRefGetMaxAnisotropy(pmaxAnsio:ctypes._Pointer[ctypes.c_int32], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(textureReference))
def hipTexRefGetMipmapFilterMode(pfm:ctypes._Pointer[ctypes.c_uint32], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(textureReference))
def hipTexRefGetMipmapLevelBias(pbias:ctypes._Pointer[ctypes.c_float], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(textureReference))
def hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp:ctypes._Pointer[ctypes.c_float], pmaxMipmapLevelClamp:ctypes._Pointer[ctypes.c_float], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMipmappedArray_t), ctypes.POINTER(textureReference))
def hipTexRefGetMipMappedArray(pArray:ctypes._Pointer[hipMipmappedArray_t], texRef:ctypes._Pointer[textureReference]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(textureReference), hipDeviceptr_t, size_t)
def hipTexRefSetAddress(ByteOffset:ctypes._Pointer[size_t], texRef:ctypes._Pointer[textureReference], dptr:hipDeviceptr_t, bytes:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.POINTER(HIP_ARRAY_DESCRIPTOR), hipDeviceptr_t, size_t)
def hipTexRefSetAddress2D(texRef:ctypes._Pointer[textureReference], desc:ctypes._Pointer[HIP_ARRAY_DESCRIPTOR], dptr:hipDeviceptr_t, Pitch:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_uint32)
def hipTexRefSetMaxAnisotropy(texRef:ctypes._Pointer[textureReference], maxAniso:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.POINTER(ctypes.c_float))
def hipTexRefSetBorderColor(texRef:ctypes._Pointer[textureReference], pBorderColor:ctypes._Pointer[ctypes.c_float]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_uint32)
def hipTexRefSetMipmapFilterMode(texRef:ctypes._Pointer[textureReference], fm:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_float)
def hipTexRefSetMipmapLevelBias(texRef:ctypes._Pointer[textureReference], bias:float) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.c_float, ctypes.c_float)
def hipTexRefSetMipmapLevelClamp(texRef:ctypes._Pointer[textureReference], minMipMapLevelClamp:float, maxMipMapLevelClamp:float) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(textureReference), ctypes.POINTER(hipMipmappedArray), ctypes.c_uint32)
def hipTexRefSetMipmappedArray(texRef:ctypes._Pointer[textureReference], mipmappedArray:ctypes._Pointer[hipMipmappedArray], Flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), uint32_t)
def hipApiName(id:uint32_t) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), hipFunction_t)
def hipKernelNameRef(f:hipFunction_t) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, hipStream_t)
def hipKernelNameRefByPtr(hostFunction:int|None, stream:hipStream_t) -> ctypes._Pointer[ctypes.c_char]: ...
@dll.bind(ctypes.c_int32, hipStream_t)
def hipGetStreamDeviceId(stream:hipStream_t) -> int: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.c_uint32)
def hipStreamBeginCapture(stream:hipStream_t, mode:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphEdgeData), size_t, ctypes.c_uint32)
def hipStreamBeginCaptureToGraph(stream:hipStream_t, graph:hipGraph_t, dependencies:ctypes._Pointer[hipGraphNode_t], dependencyData:ctypes._Pointer[hipGraphEdgeData], numDependencies:size_t, mode:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(hipGraph_t))
def hipStreamEndCapture(stream:hipStream_t, pGraph:ctypes._Pointer[hipGraph_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64))
def hipStreamGetCaptureInfo(stream:hipStream_t, pCaptureStatus:ctypes._Pointer[ctypes.c_uint32], pId:ctypes._Pointer[ctypes.c_uint64]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(hipGraph_t), ctypes.POINTER(ctypes.POINTER(hipGraphNode_t)), ctypes.POINTER(size_t))
def hipStreamGetCaptureInfo_v2(stream:hipStream_t, captureStatus_out:ctypes._Pointer[ctypes.c_uint32], id_out:ctypes._Pointer[ctypes.c_uint64], graph_out:ctypes._Pointer[hipGraph_t], dependencies_out:ctypes._Pointer[ctypes.POINTER(hipGraphNode_t)], numDependencies_out:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(ctypes.c_uint32))
def hipStreamIsCapturing(stream:hipStream_t, pCaptureStatus:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipStream_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_uint32)
def hipStreamUpdateCaptureDependencies(stream:hipStream_t, dependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
def hipThreadExchangeStreamCaptureMode(mode:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraph_t), ctypes.c_uint32)
def hipGraphCreate(pGraph:ctypes._Pointer[hipGraph_t], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t)
def hipGraphDestroy(graph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphNode_t), size_t)
def hipGraphAddDependencies(graph:hipGraph_t, _from:ctypes._Pointer[hipGraphNode_t], to:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphNode_t), size_t)
def hipGraphRemoveDependencies(graph:hipGraph_t, _from:ctypes._Pointer[hipGraphNode_t], to:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t))
def hipGraphGetEdges(graph:hipGraph_t, _from:ctypes._Pointer[hipGraphNode_t], to:ctypes._Pointer[hipGraphNode_t], numEdges:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t))
def hipGraphGetNodes(graph:hipGraph_t, nodes:ctypes._Pointer[hipGraphNode_t], numNodes:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t))
def hipGraphGetRootNodes(graph:hipGraph_t, pRootNodes:ctypes._Pointer[hipGraphNode_t], pNumRootNodes:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t))
def hipGraphNodeGetDependencies(node:hipGraphNode_t, pDependencies:ctypes._Pointer[hipGraphNode_t], pNumDependencies:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(size_t))
def hipGraphNodeGetDependentNodes(node:hipGraphNode_t, pDependentNodes:ctypes._Pointer[hipGraphNode_t], pNumDependentNodes:ctypes._Pointer[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(ctypes.c_uint32))
def hipGraphNodeGetType(node:hipGraphNode_t, pType:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t)
def hipGraphDestroyNode(node:hipGraphNode_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraph_t), hipGraph_t)
def hipGraphClone(pGraphClone:ctypes._Pointer[hipGraph_t], originalGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraphNode_t, hipGraph_t)
def hipGraphNodeFindInClone(pNode:ctypes._Pointer[hipGraphNode_t], originalNode:hipGraphNode_t, clonedGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphExec_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(ctypes.c_char), size_t)
def hipGraphInstantiate(pGraphExec:ctypes._Pointer[hipGraphExec_t], graph:hipGraph_t, pErrorNode:ctypes._Pointer[hipGraphNode_t], pLogBuffer:ctypes._Pointer[ctypes.c_char], bufferSize:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphExec_t), hipGraph_t, ctypes.c_uint64)
def hipGraphInstantiateWithFlags(pGraphExec:ctypes._Pointer[hipGraphExec_t], graph:hipGraph_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphExec_t), hipGraph_t, ctypes.POINTER(hipGraphInstantiateParams))
def hipGraphInstantiateWithParams(pGraphExec:ctypes._Pointer[hipGraphExec_t], graph:hipGraph_t, instantiateParams:ctypes._Pointer[hipGraphInstantiateParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipStream_t)
def hipGraphLaunch(graphExec:hipGraphExec_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipStream_t)
def hipGraphUpload(graphExec:hipGraphExec_t, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipGraphNodeParams))
def hipGraphAddNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, nodeParams:ctypes._Pointer[hipGraphNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, ctypes.POINTER(ctypes.c_uint64))
def hipGraphExecGetFlags(graphExec:hipGraphExec_t, flags:ctypes._Pointer[ctypes.c_uint64]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipGraphNodeParams))
def hipGraphNodeSetParams(node:hipGraphNode_t, nodeParams:ctypes._Pointer[hipGraphNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipGraphNodeParams))
def hipGraphExecNodeSetParams(graphExec:hipGraphExec_t, node:hipGraphNode_t, nodeParams:ctypes._Pointer[hipGraphNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t)
def hipGraphExecDestroy(graphExec:hipGraphExec_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraph_t, ctypes.POINTER(hipGraphNode_t), ctypes.POINTER(ctypes.c_uint32))
def hipGraphExecUpdate(hGraphExec:hipGraphExec_t, hGraph:hipGraph_t, hErrorNode_out:ctypes._Pointer[hipGraphNode_t], updateResult_out:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipKernelNodeParams))
def hipGraphAddKernelNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, pNodeParams:ctypes._Pointer[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipKernelNodeParams))
def hipGraphKernelNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipKernelNodeParams))
def hipGraphKernelNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipKernelNodeParams))
def hipGraphExecKernelNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipKernelNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(HIP_MEMCPY3D), hipCtx_t)
def hipDrvGraphAddMemcpyNode(phGraphNode:ctypes._Pointer[hipGraphNode_t], hGraph:hipGraph_t, dependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, copyParams:ctypes._Pointer[HIP_MEMCPY3D], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemcpy3DParms))
def hipGraphAddMemcpyNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, pCopyParams:ctypes._Pointer[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipMemcpy3DParms))
def hipGraphMemcpyNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipMemcpy3DParms))
def hipGraphMemcpyNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_uint32, ctypes.POINTER(hipLaunchAttributeValue))
def hipGraphKernelNodeSetAttribute(hNode:hipGraphNode_t, attr:ctypes.c_uint32, value:ctypes._Pointer[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_uint32, ctypes.POINTER(hipLaunchAttributeValue))
def hipGraphKernelNodeGetAttribute(hNode:hipGraphNode_t, attr:ctypes.c_uint32, value:ctypes._Pointer[hipLaunchAttributeValue]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipMemcpy3DParms))
def hipGraphExecMemcpyNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemcpy3DParms]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipGraphAddMemcpyNode1D(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, dst:int|None, src:int|None, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipGraphMemcpyNodeSetParams1D(node:hipGraphNode_t, dst:int|None, src:int|None, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32)
def hipGraphExecMemcpyNodeSetParams1D(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:int|None, src:int|None, count:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphAddMemcpyNodeFromSymbol(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, dst:int|None, symbol:int|None, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphMemcpyNodeSetParamsFromSymbol(node:hipGraphNode_t, dst:int|None, symbol:int|None, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, dst:int|None, symbol:int|None, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphAddMemcpyNodeToSymbol(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, symbol:int|None, src:int|None, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphMemcpyNodeSetParamsToSymbol(node:hipGraphNode_t, symbol:int|None, src:int|None, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.c_uint32)
def hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, symbol:int|None, src:int|None, count:size_t, offset:size_t, kind:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemsetParams))
def hipGraphAddMemsetNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, pMemsetParams:ctypes._Pointer[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipMemsetParams))
def hipGraphMemsetNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipMemsetParams))
def hipGraphMemsetNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipMemsetParams))
def hipGraphExecMemsetNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemsetParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipHostNodeParams))
def hipGraphAddHostNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, pNodeParams:ctypes._Pointer[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipHostNodeParams))
def hipGraphHostNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipHostNodeParams))
def hipGraphHostNodeSetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipHostNodeParams))
def hipGraphExecHostNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipHostNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipGraph_t)
def hipGraphAddChildGraphNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, childGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipGraph_t))
def hipGraphChildGraphNodeGetGraph(node:hipGraphNode_t, pGraph:ctypes._Pointer[hipGraph_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, hipGraph_t)
def hipGraphExecChildGraphNodeSetParams(hGraphExec:hipGraphExec_t, node:hipGraphNode_t, childGraph:hipGraph_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t)
def hipGraphAddEmptyNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipEvent_t)
def hipGraphAddEventRecordNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipEvent_t))
def hipGraphEventRecordNodeGetEvent(node:hipGraphNode_t, event_out:ctypes._Pointer[hipEvent_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, hipEvent_t)
def hipGraphEventRecordNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, hipEvent_t)
def hipGraphExecEventRecordNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipEvent_t)
def hipGraphAddEventWaitNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipEvent_t))
def hipGraphEventWaitNodeGetEvent(node:hipGraphNode_t, event_out:ctypes._Pointer[hipEvent_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, hipEvent_t)
def hipGraphEventWaitNodeSetEvent(node:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, hipEvent_t)
def hipGraphExecEventWaitNodeSetEvent(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, event:hipEvent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemAllocNodeParams))
def hipGraphAddMemAllocNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, pNodeParams:ctypes._Pointer[hipMemAllocNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipMemAllocNodeParams))
def hipGraphMemAllocNodeGetParams(node:hipGraphNode_t, pNodeParams:ctypes._Pointer[hipMemAllocNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.c_void_p)
def hipGraphAddMemFreeNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, dev_ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.c_void_p)
def hipGraphMemFreeNodeGetParams(node:hipGraphNode_t, dev_ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p)
def hipDeviceGetGraphMemAttribute(device:int, attr:ctypes.c_uint32, value:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p)
def hipDeviceSetGraphMemAttribute(device:int, attr:ctypes.c_uint32, value:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hipDeviceGraphMemTrim(device:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipUserObject_t), ctypes.c_void_p, hipHostFn_t, ctypes.c_uint32, ctypes.c_uint32)
def hipUserObjectCreate(object_out:ctypes._Pointer[hipUserObject_t], ptr:int|None, destroy:hipHostFn_t, initialRefcount:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipUserObject_t, ctypes.c_uint32)
def hipUserObjectRelease(object:hipUserObject_t, count:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipUserObject_t, ctypes.c_uint32)
def hipUserObjectRetain(object:hipUserObject_t, count:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, hipUserObject_t, ctypes.c_uint32, ctypes.c_uint32)
def hipGraphRetainUserObject(graph:hipGraph_t, object:hipUserObject_t, count:int, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, hipUserObject_t, ctypes.c_uint32)
def hipGraphReleaseUserObject(graph:hipGraph_t, object:hipUserObject_t, count:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraph_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)
def hipGraphDebugDotPrint(graph:hipGraph_t, path:ctypes._Pointer[ctypes.c_char], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, hipGraphNode_t)
def hipGraphKernelNodeCopyAttributes(hSrc:hipGraphNode_t, hDst:hipGraphNode_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.c_uint32)
def hipGraphNodeSetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(ctypes.c_uint32))
def hipGraphNodeGetEnabled(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, isEnabled:ctypes._Pointer[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams))
def hipGraphAddExternalSemaphoresWaitNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, nodeParams:ctypes._Pointer[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams))
def hipGraphAddExternalSemaphoresSignalNode(pGraphNode:ctypes._Pointer[hipGraphNode_t], graph:hipGraph_t, pDependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, nodeParams:ctypes._Pointer[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams))
def hipGraphExternalSemaphoresSignalNodeSetParams(hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams))
def hipGraphExternalSemaphoresWaitNodeSetParams(hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams))
def hipGraphExternalSemaphoresSignalNodeGetParams(hNode:hipGraphNode_t, params_out:ctypes._Pointer[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams))
def hipGraphExternalSemaphoresWaitNodeGetParams(hNode:hipGraphNode_t, params_out:ctypes._Pointer[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreSignalNodeParams))
def hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[hipExternalSemaphoreSignalNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipExternalSemaphoreWaitNodeParams))
def hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[hipExternalSemaphoreWaitNodeParams]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(HIP_MEMCPY3D))
def hipDrvGraphMemcpyNodeGetParams(hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[HIP_MEMCPY3D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphNode_t, ctypes.POINTER(HIP_MEMCPY3D))
def hipDrvGraphMemcpyNodeSetParams(hNode:hipGraphNode_t, nodeParams:ctypes._Pointer[HIP_MEMCPY3D]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, ctypes.POINTER(hipMemsetParams), hipCtx_t)
def hipDrvGraphAddMemsetNode(phGraphNode:ctypes._Pointer[hipGraphNode_t], hGraph:hipGraph_t, dependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, memsetParams:ctypes._Pointer[hipMemsetParams], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipGraphNode_t), hipGraph_t, ctypes.POINTER(hipGraphNode_t), size_t, hipDeviceptr_t)
def hipDrvGraphAddMemFreeNode(phGraphNode:ctypes._Pointer[hipGraphNode_t], hGraph:hipGraph_t, dependencies:ctypes._Pointer[hipGraphNode_t], numDependencies:size_t, dptr:hipDeviceptr_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(HIP_MEMCPY3D), hipCtx_t)
def hipDrvGraphExecMemcpyNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, copyParams:ctypes._Pointer[HIP_MEMCPY3D], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphExec_t, hipGraphNode_t, ctypes.POINTER(hipMemsetParams), hipCtx_t)
def hipDrvGraphExecMemsetNodeSetParams(hGraphExec:hipGraphExec_t, hNode:hipGraphNode_t, memsetParams:ctypes._Pointer[hipMemsetParams], ctx:hipCtx_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hipMemAddressFree(devPtr:int|None, size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), size_t, size_t, ctypes.c_void_p, ctypes.c_uint64)
def hipMemAddressReserve(ptr:ctypes._Pointer[ctypes.c_void_p], size:size_t, alignment:size_t, addr:int|None, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemGenericAllocationHandle_t), size_t, ctypes.POINTER(hipMemAllocationProp), ctypes.c_uint64)
def hipMemCreate(handle:ctypes._Pointer[hipMemGenericAllocationHandle_t], size:size_t, prop:ctypes._Pointer[hipMemAllocationProp], flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hipMemGenericAllocationHandle_t, ctypes.c_uint32, ctypes.c_uint64)
def hipMemExportToShareableHandle(shareableHandle:int|None, handle:hipMemGenericAllocationHandle_t, handleType:ctypes.c_uint32, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(hipMemLocation), ctypes.c_void_p)
def hipMemGetAccess(flags:ctypes._Pointer[ctypes.c_uint64], location:ctypes._Pointer[hipMemLocation], ptr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(size_t), ctypes.POINTER(hipMemAllocationProp), ctypes.c_uint32)
def hipMemGetAllocationGranularity(granularity:ctypes._Pointer[size_t], prop:ctypes._Pointer[hipMemAllocationProp], option:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemAllocationProp), hipMemGenericAllocationHandle_t)
def hipMemGetAllocationPropertiesFromHandle(prop:ctypes._Pointer[hipMemAllocationProp], handle:hipMemGenericAllocationHandle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemGenericAllocationHandle_t), ctypes.c_void_p, ctypes.c_uint32)
def hipMemImportFromShareableHandle(handle:ctypes._Pointer[hipMemGenericAllocationHandle_t], osHandle:int|None, shHandleType:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, size_t, hipMemGenericAllocationHandle_t, ctypes.c_uint64)
def hipMemMap(ptr:int|None, size:size_t, offset:size_t, handle:hipMemGenericAllocationHandle_t, flags:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArrayMapInfo), ctypes.c_uint32, hipStream_t)
def hipMemMapArrayAsync(mapInfoList:ctypes._Pointer[hipArrayMapInfo], count:int, stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipMemGenericAllocationHandle_t)
def hipMemRelease(handle:hipMemGenericAllocationHandle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipMemGenericAllocationHandle_t), ctypes.c_void_p)
def hipMemRetainAllocationHandle(handle:ctypes._Pointer[hipMemGenericAllocationHandle_t], addr:int|None) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, ctypes.POINTER(hipMemAccessDesc), size_t)
def hipMemSetAccess(ptr:int|None, size:size_t, desc:ctypes._Pointer[hipMemAccessDesc], count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hipMemUnmap(ptr:int|None, size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(hipGraphicsResource_t), hipStream_t)
def hipGraphicsMapResources(count:int, resources:ctypes._Pointer[hipGraphicsResource_t], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipArray_t), hipGraphicsResource_t, ctypes.c_uint32, ctypes.c_uint32)
def hipGraphicsSubResourceGetMappedArray(array:ctypes._Pointer[hipArray_t], resource:hipGraphicsResource_t, arrayIndex:int, mipLevel:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), hipGraphicsResource_t)
def hipGraphicsResourceGetMappedPointer(devPtr:ctypes._Pointer[ctypes.c_void_p], size:ctypes._Pointer[size_t], resource:hipGraphicsResource_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(hipGraphicsResource_t), hipStream_t)
def hipGraphicsUnmapResources(count:int, resources:ctypes._Pointer[hipGraphicsResource_t], stream:hipStream_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipGraphicsResource_t)
def hipGraphicsUnregisterResource(resource:hipGraphicsResource_t) -> ctypes.c_uint32: ...
class __hip_surface(c.Struct): pass
hipSurfaceObject_t: TypeAlias = ctypes.POINTER(__hip_surface)
@dll.bind(ctypes.c_uint32, ctypes.POINTER(hipSurfaceObject_t), ctypes.POINTER(hipResourceDesc))
def hipCreateSurfaceObject(pSurfObject:ctypes._Pointer[hipSurfaceObject_t], pResDesc:ctypes._Pointer[hipResourceDesc]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hipSurfaceObject_t)
def hipDestroySurfaceObject(surfaceObject:hipSurfaceObject_t) -> ctypes.c_uint32: ...
hipmipmappedArray: TypeAlias = ctypes.POINTER(hipMipmappedArray)
hipResourcetype: TypeAlias = ctypes.c_uint32
hipMemcpyFlags: dict[int, str] = {(hipMemcpyFlagDefault:=0): 'hipMemcpyFlagDefault', (hipMemcpyFlagPreferOverlapWithCompute:=1): 'hipMemcpyFlagPreferOverlapWithCompute'}
hiprtcJIT_option = hipJitOption # type: ignore
HIPRTC_JIT_MAX_REGISTERS = hipJitOptionMaxRegisters # type: ignore
HIPRTC_JIT_THREADS_PER_BLOCK = hipJitOptionThreadsPerBlock # type: ignore
HIPRTC_JIT_WALL_TIME = hipJitOptionWallTime # type: ignore
HIPRTC_JIT_INFO_LOG_BUFFER = hipJitOptionInfoLogBuffer # type: ignore
HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = hipJitOptionInfoLogBufferSizeBytes # type: ignore
HIPRTC_JIT_ERROR_LOG_BUFFER = hipJitOptionErrorLogBuffer # type: ignore
HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = hipJitOptionErrorLogBufferSizeBytes # type: ignore
HIPRTC_JIT_OPTIMIZATION_LEVEL = hipJitOptionOptimizationLevel # type: ignore
HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = hipJitOptionTargetFromContext # type: ignore
HIPRTC_JIT_TARGET = hipJitOptionTarget # type: ignore
HIPRTC_JIT_FALLBACK_STRATEGY = hipJitOptionFallbackStrategy # type: ignore
HIPRTC_JIT_GENERATE_DEBUG_INFO = hipJitOptionGenerateDebugInfo # type: ignore
HIPRTC_JIT_LOG_VERBOSE = hipJitOptionLogVerbose # type: ignore
HIPRTC_JIT_GENERATE_LINE_INFO = hipJitOptionGenerateLineInfo # type: ignore
HIPRTC_JIT_CACHE_MODE = hipJitOptionCacheMode # type: ignore
HIPRTC_JIT_NEW_SM3X_OPT = hipJitOptionSm3xOpt # type: ignore
HIPRTC_JIT_FAST_COMPILE = hipJitOptionFastCompile # type: ignore
HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = hipJitOptionGlobalSymbolNames # type: ignore
HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = hipJitOptionGlobalSymbolAddresses # type: ignore
HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = hipJitOptionGlobalSymbolCount # type: ignore
HIPRTC_JIT_LTO = hipJitOptionLto # type: ignore
HIPRTC_JIT_FTZ = hipJitOptionFtz # type: ignore
HIPRTC_JIT_PREC_DIV = hipJitOptionPrecDiv # type: ignore
HIPRTC_JIT_PREC_SQRT = hipJitOptionPrecSqrt # type: ignore
HIPRTC_JIT_FMA = hipJitOptionFma # type: ignore
HIPRTC_JIT_POSITION_INDEPENDENT_CODE = hipJitOptionPositionIndependentCode # type: ignore
HIPRTC_JIT_MIN_CTA_PER_SM = hipJitOptionMinCTAPerSM # type: ignore
HIPRTC_JIT_MAX_THREADS_PER_BLOCK = hipJitOptionMaxThreadsPerBlock # type: ignore
HIPRTC_JIT_OVERRIDE_DIRECT_VALUES = hipJitOptionOverrideDirectiveValues # type: ignore
HIPRTC_JIT_NUM_OPTIONS = hipJitOptionNumOptions # type: ignore
HIPRTC_JIT_IR_TO_ISA_OPT_EXT = hipJitOptionIRtoISAOptExt # type: ignore
HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT = hipJitOptionIRtoISAOptCountExt # type: ignore
hiprtcJITInputType = hipJitInputType # type: ignore
HIPRTC_JIT_INPUT_CUBIN = hipJitInputCubin # type: ignore
HIPRTC_JIT_INPUT_PTX = hipJitInputPtx # type: ignore
HIPRTC_JIT_INPUT_FATBINARY = hipJitInputFatBinary # type: ignore
HIPRTC_JIT_INPUT_OBJECT = hipJitInputObject # type: ignore
HIPRTC_JIT_INPUT_LIBRARY = hipJitInputLibrary # type: ignore
HIPRTC_JIT_INPUT_NVVM = hipJitInputNvvm # type: ignore
HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hipJitNumLegacyInputTypes # type: ignore
HIPRTC_JIT_INPUT_LLVM_BITCODE = hipJitInputLLVMBitcode # type: ignore
HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = hipJitInputLLVMBundledBitcode # type: ignore
HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = hipJitInputLLVMArchivesOfBundledBitcode # type: ignore
HIPRTC_JIT_INPUT_SPIRV = hipJitInputSpirv # type: ignore
HIPRTC_JIT_NUM_INPUT_TYPES = hipJitNumInputTypes # type: ignore
hipGetDeviceProperties = hipGetDevicePropertiesR0600 # type: ignore
hipDeviceProp_t = hipDeviceProp_tR0600 # type: ignore
hipChooseDevice = hipChooseDeviceR0600 # type: ignore
GENERIC_GRID_LAUNCH = 1 # type: ignore
HIP_DEPRECATED = lambda msg: __attribute__((deprecated(msg))) # type: ignore
hipIpcMemLazyEnablePeerAccess = 0x01 # type: ignore
HIP_IPC_HANDLE_SIZE = 64 # type: ignore
hipStreamDefault = 0x00 # type: ignore
hipStreamNonBlocking = 0x01 # type: ignore
hipEventDefault = 0x0 # type: ignore
hipEventBlockingSync = 0x1 # type: ignore
hipEventDisableTiming = 0x2 # type: ignore
hipEventInterprocess = 0x4 # type: ignore
hipEventRecordDefault = 0x00 # type: ignore
hipEventRecordExternal = 0x01 # type: ignore
hipEventWaitDefault = 0x00 # type: ignore
hipEventWaitExternal = 0x01 # type: ignore
hipEventDisableSystemFence = 0x20000000 # type: ignore
hipEventReleaseToDevice = 0x40000000 # type: ignore
hipEventReleaseToSystem = 0x80000000 # type: ignore
hipEnableDefault = 0x0 # type: ignore
hipEnableLegacyStream = 0x1 # type: ignore
hipEnablePerThreadDefaultStream = 0x2 # type: ignore
hipHostAllocDefault = 0x0 # type: ignore
hipHostMallocDefault = 0x0 # type: ignore
hipHostAllocPortable = 0x1 # type: ignore
hipHostMallocPortable = 0x1 # type: ignore
hipHostAllocMapped = 0x2 # type: ignore
hipHostMallocMapped = 0x2 # type: ignore
hipHostAllocWriteCombined = 0x4 # type: ignore
hipHostMallocWriteCombined = 0x4 # type: ignore
hipHostMallocUncached = 0x10000000 # type: ignore
hipHostAllocUncached = hipHostMallocUncached # type: ignore
hipHostMallocNumaUser = 0x20000000 # type: ignore
hipHostMallocCoherent = 0x40000000 # type: ignore
hipHostMallocNonCoherent = 0x80000000 # type: ignore
hipMemAttachGlobal = 0x01 # type: ignore
hipMemAttachHost = 0x02 # type: ignore
hipMemAttachSingle = 0x04 # type: ignore
hipDeviceMallocDefault = 0x0 # type: ignore
hipDeviceMallocFinegrained = 0x1 # type: ignore
hipMallocSignalMemory = 0x2 # type: ignore
hipDeviceMallocUncached = 0x3 # type: ignore
hipDeviceMallocContiguous = 0x4 # type: ignore
hipHostRegisterDefault = 0x0 # type: ignore
hipHostRegisterPortable = 0x1 # type: ignore
hipHostRegisterMapped = 0x2 # type: ignore
hipHostRegisterIoMemory = 0x4 # type: ignore
hipHostRegisterReadOnly = 0x08 # type: ignore
hipExtHostRegisterCoarseGrained = 0x8 # type: ignore
hipExtHostRegisterUncached = 0x80000000 # type: ignore
hipDeviceScheduleAuto = 0x0 # type: ignore
hipDeviceScheduleSpin = 0x1 # type: ignore
hipDeviceScheduleYield = 0x2 # type: ignore
hipDeviceScheduleBlockingSync = 0x4 # type: ignore
hipDeviceScheduleMask = 0x7 # type: ignore
hipDeviceMapHost = 0x8 # type: ignore
hipDeviceLmemResizeToMax = 0x10 # type: ignore
hipArrayDefault = 0x00 # type: ignore
hipArrayLayered = 0x01 # type: ignore
hipArraySurfaceLoadStore = 0x02 # type: ignore
hipArrayCubemap = 0x04 # type: ignore
hipArrayTextureGather = 0x08 # type: ignore
hipOccupancyDefault = 0x00 # type: ignore
hipOccupancyDisableCachingOverride = 0x01 # type: ignore
hipCooperativeLaunchMultiDeviceNoPreSync = 0x01 # type: ignore
hipCooperativeLaunchMultiDeviceNoPostSync = 0x02 # type: ignore
hipExtAnyOrderLaunch = 0x01 # type: ignore
hipStreamWaitValueGte = 0x0 # type: ignore
hipStreamWaitValueEq = 0x1 # type: ignore
hipStreamWaitValueAnd = 0x2 # type: ignore
hipStreamWaitValueNor = 0x3 # type: ignore
hipExternalMemoryDedicated = 0x1 # type: ignore
hipStreamAttrID = hipLaunchAttributeID # type: ignore
hipStreamAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow # type: ignore
hipStreamAttributeSynchronizationPolicy = hipLaunchAttributeSynchronizationPolicy # type: ignore
hipStreamAttributeMemSyncDomainMap = hipLaunchAttributeMemSyncDomainMap # type: ignore
hipStreamAttributeMemSyncDomain = hipLaunchAttributeMemSyncDomain # type: ignore
hipStreamAttributePriority = hipLaunchAttributePriority # type: ignore
hipStreamAttrValue = hipLaunchAttributeValue # type: ignore
hipKernelNodeAttrID = hipLaunchAttributeID # type: ignore
hipKernelNodeAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow # type: ignore
hipKernelNodeAttributeCooperative = hipLaunchAttributeCooperative # type: ignore
hipKernelNodeAttributePriority = hipLaunchAttributePriority # type: ignore
hipKernelNodeAttrValue = hipLaunchAttributeValue # type: ignore
hipDrvLaunchAttributeCooperative = hipLaunchAttributeCooperative # type: ignore
hipDrvLaunchAttributeID = hipLaunchAttributeID # type: ignore
hipDrvLaunchAttributeValue = hipLaunchAttributeValue # type: ignore
hipDrvLaunchAttribute = hipLaunchAttribute # type: ignore
hipGraphKernelNodePortDefault = 0 # type: ignore
hipGraphKernelNodePortLaunchCompletion = 2 # type: ignore
hipGraphKernelNodePortProgrammatic = 1 # type: ignore
HIP_TRSA_OVERRIDE_FORMAT = 0x01 # type: ignore
HIP_TRSF_READ_AS_INTEGER = 0x01 # type: ignore
HIP_TRSF_NORMALIZED_COORDINATES = 0x02 # type: ignore
HIP_TRSF_SRGB = 0x10 # type: ignore