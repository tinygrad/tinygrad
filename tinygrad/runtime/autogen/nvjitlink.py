# mypy: ignore-errors
import ctypes
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
from ctypes.util import find_library
def dll():
  try: return ctypes.CDLL(unwrap(find_library('nvJitLink')))
  except: pass
  return None
dll = dll()

nvJitLinkResult = CEnum(ctypes.c_uint32)
NVJITLINK_SUCCESS = nvJitLinkResult.define('NVJITLINK_SUCCESS', 0)
NVJITLINK_ERROR_UNRECOGNIZED_OPTION = nvJitLinkResult.define('NVJITLINK_ERROR_UNRECOGNIZED_OPTION', 1)
NVJITLINK_ERROR_MISSING_ARCH = nvJitLinkResult.define('NVJITLINK_ERROR_MISSING_ARCH', 2)
NVJITLINK_ERROR_INVALID_INPUT = nvJitLinkResult.define('NVJITLINK_ERROR_INVALID_INPUT', 3)
NVJITLINK_ERROR_PTX_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_PTX_COMPILE', 4)
NVJITLINK_ERROR_NVVM_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_NVVM_COMPILE', 5)
NVJITLINK_ERROR_INTERNAL = nvJitLinkResult.define('NVJITLINK_ERROR_INTERNAL', 6)

nvJitLinkInputType = CEnum(ctypes.c_uint32)
NVJITLINK_INPUT_NONE = nvJitLinkInputType.define('NVJITLINK_INPUT_NONE', 0)
NVJITLINK_INPUT_CUBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_CUBIN', 1)
NVJITLINK_INPUT_PTX = nvJitLinkInputType.define('NVJITLINK_INPUT_PTX', 2)
NVJITLINK_INPUT_LTOIR = nvJitLinkInputType.define('NVJITLINK_INPUT_LTOIR', 3)
NVJITLINK_INPUT_FATBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_FATBIN', 4)
NVJITLINK_INPUT_OBJECT = nvJitLinkInputType.define('NVJITLINK_INPUT_OBJECT', 5)
NVJITLINK_INPUT_LIBRARY = nvJitLinkInputType.define('NVJITLINK_INPUT_LIBRARY', 6)

class struct_nvJitLink(Struct): pass
nvJitLinkHandle = ctypes.POINTER(struct_nvJitLink)
uint32_t = ctypes.c_uint32
# nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions, const char **options)
try: (nvJitLinkCreate:=dll.nvJitLinkCreate).restype, nvJitLinkCreate.argtypes = nvJitLinkResult, [ctypes.POINTER(nvJitLinkHandle), uint32_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle)
try: (nvJitLinkDestroy:=dll.nvJitLinkDestroy).restype, nvJitLinkDestroy.argtypes = nvJitLinkResult, [ctypes.POINTER(nvJitLinkHandle)]
except AttributeError: pass

size_t = ctypes.c_uint64
# nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name)
try: (nvJitLinkAddData:=dll.nvJitLinkAddData).restype, nvJitLinkAddData.argtypes = nvJitLinkResult, [nvJitLinkHandle, nvJitLinkInputType, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char *fileName)
try: (nvJitLinkAddFile:=dll.nvJitLinkAddFile).restype, nvJitLinkAddFile.argtypes = nvJitLinkResult, [nvJitLinkHandle, nvJitLinkInputType, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle)
try: (nvJitLinkComplete:=dll.nvJitLinkComplete).restype, nvJitLinkComplete.argtypes = nvJitLinkResult, [nvJitLinkHandle]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t *size)
try: (nvJitLinkGetLinkedCubinSize:=dll.nvJitLinkGetLinkedCubinSize).restype, nvJitLinkGetLinkedCubinSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin)
try: (nvJitLinkGetLinkedCubin:=dll.nvJitLinkGetLinkedCubin).restype, nvJitLinkGetLinkedCubin.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.c_void_p]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t *size)
try: (nvJitLinkGetLinkedPtxSize:=dll.nvJitLinkGetLinkedPtxSize).restype, nvJitLinkGetLinkedPtxSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char *ptx)
try: (nvJitLinkGetLinkedPtx:=dll.nvJitLinkGetLinkedPtx).restype, nvJitLinkGetLinkedPtx.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t *size)
try: (nvJitLinkGetErrorLogSize:=dll.nvJitLinkGetErrorLogSize).restype, nvJitLinkGetErrorLogSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char *log)
try: (nvJitLinkGetErrorLog:=dll.nvJitLinkGetErrorLog).restype, nvJitLinkGetErrorLog.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t *size)
try: (nvJitLinkGetInfoLogSize:=dll.nvJitLinkGetInfoLogSize).restype, nvJitLinkGetInfoLogSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char *log)
try: (nvJitLinkGetInfoLog:=dll.nvJitLinkGetInfoLog).restype, nvJitLinkGetInfoLog.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# nvJitLinkResult nvJitLinkVersion(unsigned int *major, unsigned int *minor)
try: (nvJitLinkVersion:=dll.nvJitLinkVersion).restype, nvJitLinkVersion.argtypes = nvJitLinkResult, [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

