# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('llvm_qcom', 'llvm-qcom')
cl_llvm_instance: TypeAlias = ctypes.c_void_p
@dll.bind
def cl_compiler_create_llvm_instance() -> cl_llvm_instance: ...
@dll.bind
def cl_compiler_destroy_llvm_instance(inst:cl_llvm_instance) -> None: ...
class enum_cl_handle_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
CL_HANDLE_COMPILED = enum_cl_handle_type.define('CL_HANDLE_COMPILED', 1)
CL_HANDLE_LIBRARY = enum_cl_handle_type.define('CL_HANDLE_LIBRARY', 2)
CL_HANDLE_LINKED = enum_cl_handle_type.define('CL_HANDLE_LINKED', 3)

class struct_cl_compiled_data(ctypes.Structure): pass
class struct_cl_executable_data(ctypes.Structure): pass
@c.record
class cl_handle(c.Struct):
  SIZE = 16
  type: Annotated[enum_cl_handle_type, 0]
  compiled: Annotated[c.POINTER[struct_cl_compiled_data], 8]
  executable: Annotated[c.POINTER[struct_cl_executable_data], 8]
@dll.bind
def cl_compiler_compile_source(inst:cl_llvm_instance, chip_id:Annotated[int, ctypes.c_int32], mode:Annotated[int, ctypes.c_int32], options:c.POINTER[Annotated[bytes, ctypes.c_char]], p5:Annotated[int, ctypes.c_int32], p6:Annotated[int, ctypes.c_int32], p7:Annotated[int, ctypes.c_int32], source:c.POINTER[Annotated[bytes, ctypes.c_char]], source_len:Annotated[int, ctypes.c_int32], source_type:Annotated[int, ctypes.c_int32], p11:ctypes.c_void_p) -> c.POINTER[cl_handle]: ...
@dll.bind
def cl_compiler_link_program(inst:cl_llvm_instance, chip_id:Annotated[int, ctypes.c_int32], mode:Annotated[int, ctypes.c_int32], options:c.POINTER[Annotated[bytes, ctypes.c_char]], num_handles:Annotated[int, ctypes.c_int32], input_handles:c.POINTER[c.POINTER[cl_handle]]) -> c.POINTER[cl_handle]: ...
@dll.bind
def cl_compiler_handle_create_binary(handle:c.POINTER[cl_handle], out_ptr:c.POINTER[ctypes.c_void_p], out_size:c.POINTER[Annotated[int, ctypes.c_int32]]) -> None: ...
class _anonstruct0(ctypes.Structure): pass
cl_lib_section: TypeAlias = _anonstruct0
class _anonstruct1(ctypes.Structure): pass
cl_lib_header: TypeAlias = _anonstruct1
class _anonstruct2(ctypes.Structure): pass
cl_lib_prog: TypeAlias = _anonstruct2
class _anonstruct3(ctypes.Structure): pass
cl_lib_img_desc: TypeAlias = _anonstruct3
@dll.bind
def cl_compiler_free_handle(handle:c.POINTER[cl_handle]) -> None: ...
@dll.bind
def cl_compiler_free_assembly(ptr:ctypes.c_void_p) -> None: ...
c.init_records()
CL_MODE_32BIT = 0 # type: ignore
CL_MODE_64BIT = 1 # type: ignore
CL_SRC_STR = 0 # type: ignore
CL_SRC_BLOB = 1 # type: ignore
CL_LIB_PROGRAM = 0 # type: ignore
CL_LIB_CONSTS = 6 # type: ignore
CL_LIB_IMAGE = 7 # type: ignore
CL_LIB_CODE = 10 # type: ignore
CL_LIB_IMAGE_DESC = 11 # type: ignore