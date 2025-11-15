# mypy: ignore-errors
import ctypes
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
from ctypes.util import find_library
def dll():
  try: return ctypes.CDLL(unwrap(find_library('nvrtc')))
  except: pass
  return None
dll = dll()

nvrtcResult = CEnum(ctypes.c_uint32)
NVRTC_SUCCESS = nvrtcResult.define('NVRTC_SUCCESS', 0)
NVRTC_ERROR_OUT_OF_MEMORY = nvrtcResult.define('NVRTC_ERROR_OUT_OF_MEMORY', 1)
NVRTC_ERROR_PROGRAM_CREATION_FAILURE = nvrtcResult.define('NVRTC_ERROR_PROGRAM_CREATION_FAILURE', 2)
NVRTC_ERROR_INVALID_INPUT = nvrtcResult.define('NVRTC_ERROR_INVALID_INPUT', 3)
NVRTC_ERROR_INVALID_PROGRAM = nvrtcResult.define('NVRTC_ERROR_INVALID_PROGRAM', 4)
NVRTC_ERROR_INVALID_OPTION = nvrtcResult.define('NVRTC_ERROR_INVALID_OPTION', 5)
NVRTC_ERROR_COMPILATION = nvrtcResult.define('NVRTC_ERROR_COMPILATION', 6)
NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = nvrtcResult.define('NVRTC_ERROR_BUILTIN_OPERATION_FAILURE', 7)
NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = nvrtcResult.define('NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION', 8)
NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = nvrtcResult.define('NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION', 9)
NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = nvrtcResult.define('NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID', 10)
NVRTC_ERROR_INTERNAL_ERROR = nvrtcResult.define('NVRTC_ERROR_INTERNAL_ERROR', 11)

# const char *nvrtcGetErrorString(nvrtcResult result)
try: (nvrtcGetErrorString:=dll.nvrtcGetErrorString).restype, nvrtcGetErrorString.argtypes = ctypes.POINTER(ctypes.c_char), [nvrtcResult]
except AttributeError: pass

# nvrtcResult nvrtcVersion(int *major, int *minor)
try: (nvrtcVersion:=dll.nvrtcVersion).restype, nvrtcVersion.argtypes = nvrtcResult, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

# nvrtcResult nvrtcGetNumSupportedArchs(int *numArchs)
try: (nvrtcGetNumSupportedArchs:=dll.nvrtcGetNumSupportedArchs).restype, nvrtcGetNumSupportedArchs.argtypes = nvrtcResult, [ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

# nvrtcResult nvrtcGetSupportedArchs(int *supportedArchs)
try: (nvrtcGetSupportedArchs:=dll.nvrtcGetSupportedArchs).restype, nvrtcGetSupportedArchs.argtypes = nvrtcResult, [ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

class struct__nvrtcProgram(Struct): pass
nvrtcProgram = ctypes.POINTER(struct__nvrtcProgram)
# nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src, const char *name, int numHeaders, const char *const *headers, const char *const *includeNames)
try: (nvrtcCreateProgram:=dll.nvrtcCreateProgram).restype, nvrtcCreateProgram.argtypes = nvrtcResult, [ctypes.POINTER(nvrtcProgram), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog)
try: (nvrtcDestroyProgram:=dll.nvrtcDestroyProgram).restype, nvrtcDestroyProgram.argtypes = nvrtcResult, [ctypes.POINTER(nvrtcProgram)]
except AttributeError: pass

# nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char *const *options)
try: (nvrtcCompileProgram:=dll.nvrtcCompileProgram).restype, nvrtcCompileProgram.argtypes = nvrtcResult, [nvrtcProgram, ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

size_t = ctypes.c_uint64
# nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet)
try: (nvrtcGetPTXSize:=dll.nvrtcGetPTXSize).restype, nvrtcGetPTXSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx)
try: (nvrtcGetPTX:=dll.nvrtcGetPTX).restype, nvrtcGetPTX.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet)
try: (nvrtcGetCUBINSize:=dll.nvrtcGetCUBINSize).restype, nvrtcGetCUBINSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin)
try: (nvrtcGetCUBIN:=dll.nvrtcGetCUBIN).restype, nvrtcGetCUBIN.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((deprecated("This function will be removed in a future release. Please use nvrtcGetLTOIRSize instead"))) nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t *nvvmSizeRet)
try: (nvrtcGetNVVMSize:=dll.nvrtcGetNVVMSize).restype, nvrtcGetNVVMSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# __attribute__((deprecated("This function will be removed in a future release. Please use nvrtcGetLTOIR instead"))) nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm)
try: (nvrtcGetNVVM:=dll.nvrtcGetNVVM).restype, nvrtcGetNVVM.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t *LTOIRSizeRet)
try: (nvrtcGetLTOIRSize:=dll.nvrtcGetLTOIRSize).restype, nvrtcGetLTOIRSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *LTOIR)
try: (nvrtcGetLTOIR:=dll.nvrtcGetLTOIR).restype, nvrtcGetLTOIR.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t *optixirSizeRet)
try: (nvrtcGetOptiXIRSize:=dll.nvrtcGetOptiXIRSize).restype, nvrtcGetOptiXIRSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char *optixir)
try: (nvrtcGetOptiXIR:=dll.nvrtcGetOptiXIR).restype, nvrtcGetOptiXIR.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet)
try: (nvrtcGetProgramLogSize:=dll.nvrtcGetProgramLogSize).restype, nvrtcGetProgramLogSize.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log)
try: (nvrtcGetProgramLog:=dll.nvrtcGetProgramLog).restype, nvrtcGetProgramLog.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char *const name_expression)
try: (nvrtcAddNameExpression:=dll.nvrtcAddNameExpression).restype, nvrtcAddNameExpression.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char *const name_expression, const char **lowered_name)
try: (nvrtcGetLoweredName:=dll.nvrtcGetLoweredName).restype, nvrtcGetLoweredName.argtypes = nvrtcResult, [nvrtcProgram, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

__DEPRECATED__ = lambda msg: __attribute__((deprecated(msg)))