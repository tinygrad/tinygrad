from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import DLL, record, Array, POINTER, CFUNCTYPE, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
import sysconfig
dll = DLL('nvjitlink', 'nvJitLink', f'/usr/local/cuda/targets/{sysconfig.get_config_vars().get("MULTIARCH", "").rsplit("-", 1)[0]}/lib')
nvJitLinkResult = CEnum(Annotated[int, ctypes.c_uint32])
NVJITLINK_SUCCESS = nvJitLinkResult.define('NVJITLINK_SUCCESS', 0)
NVJITLINK_ERROR_UNRECOGNIZED_OPTION = nvJitLinkResult.define('NVJITLINK_ERROR_UNRECOGNIZED_OPTION', 1)
NVJITLINK_ERROR_MISSING_ARCH = nvJitLinkResult.define('NVJITLINK_ERROR_MISSING_ARCH', 2)
NVJITLINK_ERROR_INVALID_INPUT = nvJitLinkResult.define('NVJITLINK_ERROR_INVALID_INPUT', 3)
NVJITLINK_ERROR_PTX_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_PTX_COMPILE', 4)
NVJITLINK_ERROR_NVVM_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_NVVM_COMPILE', 5)
NVJITLINK_ERROR_INTERNAL = nvJitLinkResult.define('NVJITLINK_ERROR_INTERNAL', 6)

nvJitLinkInputType = CEnum(Annotated[int, ctypes.c_uint32])
NVJITLINK_INPUT_NONE = nvJitLinkInputType.define('NVJITLINK_INPUT_NONE', 0)
NVJITLINK_INPUT_CUBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_CUBIN', 1)
NVJITLINK_INPUT_PTX = nvJitLinkInputType.define('NVJITLINK_INPUT_PTX', 2)
NVJITLINK_INPUT_LTOIR = nvJitLinkInputType.define('NVJITLINK_INPUT_LTOIR', 3)
NVJITLINK_INPUT_FATBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_FATBIN', 4)
NVJITLINK_INPUT_OBJECT = nvJitLinkInputType.define('NVJITLINK_INPUT_OBJECT', 5)
NVJITLINK_INPUT_LIBRARY = nvJitLinkInputType.define('NVJITLINK_INPUT_LIBRARY', 6)

class struct_nvJitLink(ctypes.Structure): pass
nvJitLinkHandle: TypeAlias = POINTER(struct_nvJitLink)
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@dll.bind
def nvJitLinkCreate(handle:POINTER(nvJitLinkHandle), numOptions:uint32_t, options:POINTER(POINTER(Annotated[bytes, ctypes.c_char]))) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkDestroy(handle:POINTER(nvJitLinkHandle)) -> nvJitLinkResult: ...
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def nvJitLinkAddData(handle:nvJitLinkHandle, inputType:nvJitLinkInputType, data:POINTER(None), size:size_t, name:POINTER(Annotated[bytes, ctypes.c_char])) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkAddFile(handle:nvJitLinkHandle, inputType:nvJitLinkInputType, fileName:POINTER(Annotated[bytes, ctypes.c_char])) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkComplete(handle:nvJitLinkHandle) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedCubinSize(handle:nvJitLinkHandle, size:POINTER(size_t)) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedCubin(handle:nvJitLinkHandle, cubin:POINTER(None)) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedPtxSize(handle:nvJitLinkHandle, size:POINTER(size_t)) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedPtx(handle:nvJitLinkHandle, ptx:POINTER(Annotated[bytes, ctypes.c_char])) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetErrorLogSize(handle:nvJitLinkHandle, size:POINTER(size_t)) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetErrorLog(handle:nvJitLinkHandle, log:POINTER(Annotated[bytes, ctypes.c_char])) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetInfoLogSize(handle:nvJitLinkHandle, size:POINTER(size_t)) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetInfoLog(handle:nvJitLinkHandle, log:POINTER(Annotated[bytes, ctypes.c_char])) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkVersion(major:POINTER(Annotated[int, ctypes.c_uint32]), minor:POINTER(Annotated[int, ctypes.c_uint32])) -> nvJitLinkResult: ...
init_records()
