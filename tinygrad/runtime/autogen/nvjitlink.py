# mypy: ignore-errors
import ctypes, ctypes.util
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
dll = ctypes.CDLL(ctypes.util.find_library('nvJitLink'))

nvJitLinkResult = CEnum(ctypes.c_uint)
NVJITLINK_SUCCESS = nvJitLinkResult.define('NVJITLINK_SUCCESS', 0)
NVJITLINK_ERROR_UNRECOGNIZED_OPTION = nvJitLinkResult.define('NVJITLINK_ERROR_UNRECOGNIZED_OPTION', 1)
NVJITLINK_ERROR_MISSING_ARCH = nvJitLinkResult.define('NVJITLINK_ERROR_MISSING_ARCH', 2)
NVJITLINK_ERROR_INVALID_INPUT = nvJitLinkResult.define('NVJITLINK_ERROR_INVALID_INPUT', 3)
NVJITLINK_ERROR_PTX_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_PTX_COMPILE', 4)
NVJITLINK_ERROR_NVVM_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_NVVM_COMPILE', 5)
NVJITLINK_ERROR_INTERNAL = nvJitLinkResult.define('NVJITLINK_ERROR_INTERNAL', 6)
NVJITLINK_ERROR_THREADPOOL = nvJitLinkResult.define('NVJITLINK_ERROR_THREADPOOL', 7)
NVJITLINK_ERROR_UNRECOGNIZED_INPUT = nvJitLinkResult.define('NVJITLINK_ERROR_UNRECOGNIZED_INPUT', 8)
NVJITLINK_ERROR_FINALIZE = nvJitLinkResult.define('NVJITLINK_ERROR_FINALIZE', 9)

nvJitLinkInputType = CEnum(ctypes.c_uint)
NVJITLINK_INPUT_NONE = nvJitLinkInputType.define('NVJITLINK_INPUT_NONE', 0)
NVJITLINK_INPUT_CUBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_CUBIN', 1)
NVJITLINK_INPUT_PTX = nvJitLinkInputType.define('NVJITLINK_INPUT_PTX', 2)
NVJITLINK_INPUT_LTOIR = nvJitLinkInputType.define('NVJITLINK_INPUT_LTOIR', 3)
NVJITLINK_INPUT_FATBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_FATBIN', 4)
NVJITLINK_INPUT_OBJECT = nvJitLinkInputType.define('NVJITLINK_INPUT_OBJECT', 5)
NVJITLINK_INPUT_LIBRARY = nvJitLinkInputType.define('NVJITLINK_INPUT_LIBRARY', 6)
NVJITLINK_INPUT_INDEX = nvJitLinkInputType.define('NVJITLINK_INPUT_INDEX', 7)
NVJITLINK_INPUT_ANY = nvJitLinkInputType.define('NVJITLINK_INPUT_ANY', 10)

class struct_nvJitLink(ctypes.Structure): pass
struct_nvJitLink._fields_ = []
nvJitLinkHandle = ctypes.POINTER(struct_nvJitLink)
uint32_t = ctypes.c_uint
# extern nvJitLinkResult __nvJitLinkCreate_12_6(nvJitLinkHandle *handle, uint32_t numOptions, const char **options)
try: (__nvJitLinkCreate_12_6:=dll.__nvJitLinkCreate_12_6).restype,__nvJitLinkCreate_12_6.argtypes = nvJitLinkResult,[ctypes.POINTER(nvJitLinkHandle), uint32_t, ctypes.POINTER(ctypes.c_char_p)]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkDestroy_12_6(nvJitLinkHandle *handle)
try: (__nvJitLinkDestroy_12_6:=dll.__nvJitLinkDestroy_12_6).restype,__nvJitLinkDestroy_12_6.argtypes = nvJitLinkResult,[ctypes.POINTER(nvJitLinkHandle)]
except AttributeError: pass

size_t = ctypes.c_ulong
# extern nvJitLinkResult __nvJitLinkAddData_12_6(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name)
try: (__nvJitLinkAddData_12_6:=dll.__nvJitLinkAddData_12_6).restype,__nvJitLinkAddData_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, nvJitLinkInputType, ctypes.c_void_p, size_t, ctypes.c_char_p]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkAddFile_12_6(nvJitLinkHandle handle, nvJitLinkInputType inputType, const char *fileName)
try: (__nvJitLinkAddFile_12_6:=dll.__nvJitLinkAddFile_12_6).restype,__nvJitLinkAddFile_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, nvJitLinkInputType, ctypes.c_char_p]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkComplete_12_6(nvJitLinkHandle handle)
try: (__nvJitLinkComplete_12_6:=dll.__nvJitLinkComplete_12_6).restype,__nvJitLinkComplete_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetLinkedCubinSize_12_6(nvJitLinkHandle handle, size_t *size)
try: (__nvJitLinkGetLinkedCubinSize_12_6:=dll.__nvJitLinkGetLinkedCubinSize_12_6).restype,__nvJitLinkGetLinkedCubinSize_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetLinkedCubin_12_6(nvJitLinkHandle handle, void *cubin)
try: (__nvJitLinkGetLinkedCubin_12_6:=dll.__nvJitLinkGetLinkedCubin_12_6).restype,__nvJitLinkGetLinkedCubin_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.c_void_p]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetLinkedPtxSize_12_6(nvJitLinkHandle handle, size_t *size)
try: (__nvJitLinkGetLinkedPtxSize_12_6:=dll.__nvJitLinkGetLinkedPtxSize_12_6).restype,__nvJitLinkGetLinkedPtxSize_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetLinkedPtx_12_6(nvJitLinkHandle handle, char *ptx)
try: (__nvJitLinkGetLinkedPtx_12_6:=dll.__nvJitLinkGetLinkedPtx_12_6).restype,__nvJitLinkGetLinkedPtx_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.c_char_p]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetErrorLogSize_12_6(nvJitLinkHandle handle, size_t *size)
try: (__nvJitLinkGetErrorLogSize_12_6:=dll.__nvJitLinkGetErrorLogSize_12_6).restype,__nvJitLinkGetErrorLogSize_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetErrorLog_12_6(nvJitLinkHandle handle, char *log)
try: (__nvJitLinkGetErrorLog_12_6:=dll.__nvJitLinkGetErrorLog_12_6).restype,__nvJitLinkGetErrorLog_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.c_char_p]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetInfoLogSize_12_6(nvJitLinkHandle handle, size_t *size)
try: (__nvJitLinkGetInfoLogSize_12_6:=dll.__nvJitLinkGetInfoLogSize_12_6).restype,__nvJitLinkGetInfoLogSize_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern nvJitLinkResult __nvJitLinkGetInfoLog_12_6(nvJitLinkHandle handle, char *log)
try: (__nvJitLinkGetInfoLog_12_6:=dll.__nvJitLinkGetInfoLog_12_6).restype,__nvJitLinkGetInfoLog_12_6.argtypes = nvJitLinkResult,[nvJitLinkHandle, ctypes.c_char_p]
except AttributeError: pass

# extern nvJitLinkResult nvJitLinkVersion(unsigned int *major, unsigned int *minor)
try: (nvJitLinkVersion:=dll.nvJitLinkVersion).restype,nvJitLinkVersion.argtypes = nvJitLinkResult,[ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

NVJITLINK_ERROR_NULL_INPUT = NVJITLINK_ERROR_INVALID_INPUT
NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS = NVJITLINK_ERROR_INVALID_INPUT
NVJITLINK_ERROR_INCORRECT_INPUT_TYPE = NVJITLINK_ERROR_INVALID_INPUT
NVJITLINK_ERROR_ARCH_MISMATCH = NVJITLINK_ERROR_INTERNAL
NVJITLINK_ERROR_OUTDATED_LIBRARY = NVJITLINK_ERROR_INTERNAL
NVJITLINK_ERROR_MISSING_FATBIN = NVJITLINK_ERROR_INVALID_INPUT