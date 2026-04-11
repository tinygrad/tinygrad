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

@c.record
class struct_cl_compiled_data(c.Struct):
  SIZE = 48
  chip_id: Annotated[uint64_t, 0]
  mode: Annotated[uint32_t, 8]
  llvm_bitcode: Annotated[ctypes.c_void_p, 16]
  llvm_bitcode_size: Annotated[uint64_t, 24]
  build_log: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 32]
  build_log_len: Annotated[uint32_t, 40]
  error_code: Annotated[uint32_t, 44]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_cl_executable_data(c.Struct):
  SIZE = 80
  num_kernels: Annotated[int32_t, 0]
  kernel_props: Annotated[ctypes.c_void_p, 8]
  error_code: Annotated[uint32_t, 16]
  build_log: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 24]
  _unk0: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[32]], 32]
  chip_id: Annotated[uint64_t, 64]
  mode: Annotated[uint32_t, 72]
int32_t: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class cl_handle(c.Struct):
  SIZE = 16
  type: Annotated[enum_cl_handle_type, 0]
  compiled: Annotated[c.POINTER[struct_cl_compiled_data], 8]
  executable: Annotated[c.POINTER[struct_cl_executable_data], 8]
@dll.bind
def cl_compiler_compile_source(inst:cl_llvm_instance, chip_id:uint64_t, mode:Annotated[int, ctypes.c_int32], options:c.POINTER[Annotated[bytes, ctypes.c_char]], p5:Annotated[int, ctypes.c_int32], p6:uint64_t, p7:uint64_t, source:c.POINTER[Annotated[bytes, ctypes.c_char]], source_len:uint64_t, source_type:uint64_t, p11:ctypes.c_void_p) -> c.POINTER[cl_handle]: ...
@dll.bind
def cl_compiler_link_program(inst:cl_llvm_instance, chip_id:uint64_t, mode:Annotated[int, ctypes.c_int32], options:c.POINTER[Annotated[bytes, ctypes.c_char]], num_handles:Annotated[int, ctypes.c_int32], input_handles:c.POINTER[c.POINTER[cl_handle]]) -> c.POINTER[cl_handle]: ...
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def cl_compiler_handle_create_binary(handle:c.POINTER[cl_handle], out_ptr:c.POINTER[ctypes.c_void_p], out_size:c.POINTER[size_t]) -> None: ...
@c.record
class cl_lib_section(c.Struct):
  SIZE = 20
  id: Annotated[uint32_t, 0]
  offset: Annotated[uint32_t, 4]
  size: Annotated[uint32_t, 8]
  count: Annotated[uint32_t, 12]
  entry_size: Annotated[uint32_t, 16]
@c.record
class cl_lib_header(c.Struct):
  SIZE = 48
  _unk0: Annotated[c.Array[uint32_t, Literal[6]], 0]
  num_sections: Annotated[uint32_t, 24]
  _unk1: Annotated[c.Array[uint32_t, Literal[5]], 28]
  sections: Annotated[c.Array[cl_lib_section, Literal[0]], 48]
@c.record
class cl_lib_prog(c.Struct):
  SIZE = 28
  name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[8]], 0]
  _unk0: Annotated[c.Array[uint32_t, Literal[3]], 8]
  fregs: Annotated[uint32_t, 20]
  hregs: Annotated[uint32_t, 24]
@c.record
class cl_lib_img_desc(c.Struct):
  SIZE = 344
  _unk0: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[196]], 0]
  prg_offset: Annotated[uint32_t, 196]
  pvtmem: Annotated[uint32_t, 200]
  _unk1: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[12]], 204]
  shmem: Annotated[uint32_t, 216]
  samp_cnt: Annotated[uint32_t, 220]
  _unk2: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[40]], 224]
  brnchstck: Annotated[uint32_t, 264]
  _unk4: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[76]], 268]
  kernel_name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[0]], 344]
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