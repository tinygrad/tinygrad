# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
import sysconfig
dll = DLL('nvjitlink', 'nvJitLink', f'/usr/local/cuda/targets/{sysconfig.get_config_vars().get("MULTIARCH", "").rsplit("-", 1)[0]}/lib')
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
nvJitLinkHandle = Pointer(struct_nvJitLink)
uint32_t = ctypes.c_uint32
try: (nvJitLinkCreate:=dll.nvJitLinkCreate).restype, nvJitLinkCreate.argtypes = nvJitLinkResult, [Pointer(nvJitLinkHandle), uint32_t, Pointer(Pointer(ctypes.c_char))]
except AttributeError: pass

try: (nvJitLinkDestroy:=dll.nvJitLinkDestroy).restype, nvJitLinkDestroy.argtypes = nvJitLinkResult, [Pointer(nvJitLinkHandle)]
except AttributeError: pass

size_t = ctypes.c_uint64
try: (nvJitLinkAddData:=dll.nvJitLinkAddData).restype, nvJitLinkAddData.argtypes = nvJitLinkResult, [nvJitLinkHandle, nvJitLinkInputType, ctypes.c_void_p, size_t, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (nvJitLinkAddFile:=dll.nvJitLinkAddFile).restype, nvJitLinkAddFile.argtypes = nvJitLinkResult, [nvJitLinkHandle, nvJitLinkInputType, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (nvJitLinkComplete:=dll.nvJitLinkComplete).restype, nvJitLinkComplete.argtypes = nvJitLinkResult, [nvJitLinkHandle]
except AttributeError: pass

try: (nvJitLinkGetLinkedCubinSize:=dll.nvJitLinkGetLinkedCubinSize).restype, nvJitLinkGetLinkedCubinSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(size_t)]
except AttributeError: pass

try: (nvJitLinkGetLinkedCubin:=dll.nvJitLinkGetLinkedCubin).restype, nvJitLinkGetLinkedCubin.argtypes = nvJitLinkResult, [nvJitLinkHandle, ctypes.c_void_p]
except AttributeError: pass

try: (nvJitLinkGetLinkedPtxSize:=dll.nvJitLinkGetLinkedPtxSize).restype, nvJitLinkGetLinkedPtxSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(size_t)]
except AttributeError: pass

try: (nvJitLinkGetLinkedPtx:=dll.nvJitLinkGetLinkedPtx).restype, nvJitLinkGetLinkedPtx.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (nvJitLinkGetErrorLogSize:=dll.nvJitLinkGetErrorLogSize).restype, nvJitLinkGetErrorLogSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(size_t)]
except AttributeError: pass

try: (nvJitLinkGetErrorLog:=dll.nvJitLinkGetErrorLog).restype, nvJitLinkGetErrorLog.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (nvJitLinkGetInfoLogSize:=dll.nvJitLinkGetInfoLogSize).restype, nvJitLinkGetInfoLogSize.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(size_t)]
except AttributeError: pass

try: (nvJitLinkGetInfoLog:=dll.nvJitLinkGetInfoLog).restype, nvJitLinkGetInfoLog.argtypes = nvJitLinkResult, [nvJitLinkHandle, Pointer(ctypes.c_char)]
except AttributeError: pass

try: (nvJitLinkVersion:=dll.nvJitLinkVersion).restype, nvJitLinkVersion.argtypes = nvJitLinkResult, [Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)]
except AttributeError: pass

