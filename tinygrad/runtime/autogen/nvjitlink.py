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
@dll.bind((Pointer(nvJitLinkHandle), uint32_t, Pointer(Pointer(ctypes.c_char)),), nvJitLinkResult)
def nvJitLinkCreate(handle, numOptions, options): ...
@dll.bind((Pointer(nvJitLinkHandle),), nvJitLinkResult)
def nvJitLinkDestroy(handle): ...
size_t = ctypes.c_uint64
@dll.bind((nvJitLinkHandle, nvJitLinkInputType, ctypes.c_void_p, size_t, Pointer(ctypes.c_char),), nvJitLinkResult)
def nvJitLinkAddData(handle, inputType, data, size, name): ...
@dll.bind((nvJitLinkHandle, nvJitLinkInputType, Pointer(ctypes.c_char),), nvJitLinkResult)
def nvJitLinkAddFile(handle, inputType, fileName): ...
@dll.bind((nvJitLinkHandle,), nvJitLinkResult)
def nvJitLinkComplete(handle): ...
@dll.bind((nvJitLinkHandle, Pointer(size_t),), nvJitLinkResult)
def nvJitLinkGetLinkedCubinSize(handle, size): ...
@dll.bind((nvJitLinkHandle, ctypes.c_void_p,), nvJitLinkResult)
def nvJitLinkGetLinkedCubin(handle, cubin): ...
@dll.bind((nvJitLinkHandle, Pointer(size_t),), nvJitLinkResult)
def nvJitLinkGetLinkedPtxSize(handle, size): ...
@dll.bind((nvJitLinkHandle, Pointer(ctypes.c_char),), nvJitLinkResult)
def nvJitLinkGetLinkedPtx(handle, ptx): ...
@dll.bind((nvJitLinkHandle, Pointer(size_t),), nvJitLinkResult)
def nvJitLinkGetErrorLogSize(handle, size): ...
@dll.bind((nvJitLinkHandle, Pointer(ctypes.c_char),), nvJitLinkResult)
def nvJitLinkGetErrorLog(handle, log): ...
@dll.bind((nvJitLinkHandle, Pointer(size_t),), nvJitLinkResult)
def nvJitLinkGetInfoLogSize(handle, size): ...
@dll.bind((nvJitLinkHandle, Pointer(ctypes.c_char),), nvJitLinkResult)
def nvJitLinkGetInfoLog(handle, log): ...
@dll.bind((Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32),), nvJitLinkResult)
def nvJitLinkVersion(major, minor): ...
