# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('llvm_qcom', 'llvm-qcom')
cl_llvm_instance: TypeAlias = ctypes.c_void_p
@dll.bind(cl_llvm_instance)
def cl_compiler_create_llvm_instance() -> cl_llvm_instance: ...
@dll.bind(None, cl_llvm_instance)
def cl_compiler_destroy_llvm_instance(inst:cl_llvm_instance) -> None: ...
enum_cl_handle_type: dict[int, str] = {(CL_HANDLE_COMPILED:=1): 'CL_HANDLE_COMPILED', (CL_HANDLE_LIBRARY:=2): 'CL_HANDLE_LIBRARY', (CL_HANDLE_LINKED:=3): 'CL_HANDLE_LINKED'}
@c.record
class struct_cl_compiled_data(c.Struct):
  SIZE = 48
  chip_id: int
  mode: int
  llvm_bitcode: int|None
  llvm_bitcode_size: int
  build_log: ctypes._Pointer[ctypes.c_char]
  build_log_len: int
  error_code: int
uint64_t: TypeAlias = ctypes.c_uint64
uint32_t: TypeAlias = ctypes.c_uint32
struct_cl_compiled_data.register_fields([('chip_id', uint64_t, 0), ('mode', uint32_t, 8), ('llvm_bitcode', ctypes.c_void_p, 16), ('llvm_bitcode_size', uint64_t, 24), ('build_log', ctypes.POINTER(ctypes.c_char), 32), ('build_log_len', uint32_t, 40), ('error_code', uint32_t, 44)])
@c.record
class struct_cl_executable_data(c.Struct):
  SIZE = 80
  num_kernels: int
  kernel_props: int|None
  error_code: int
  build_log: ctypes._Pointer[ctypes.c_char]
  _unk0: ctypes.Array[ctypes.c_char]
  chip_id: int
  mode: int
int32_t: TypeAlias = ctypes.c_int32
struct_cl_executable_data.register_fields([('num_kernels', int32_t, 0), ('kernel_props', ctypes.c_void_p, 8), ('error_code', uint32_t, 16), ('build_log', ctypes.POINTER(ctypes.c_char), 24), ('_unk0', (ctypes.c_char * 32), 32), ('chip_id', uint64_t, 64), ('mode', uint32_t, 72)])
@c.record
class cl_handle(c.Struct):
  SIZE = 16
  type: int
  compiled: ctypes._Pointer[struct_cl_compiled_data]
  executable: ctypes._Pointer[struct_cl_executable_data]
cl_handle.register_fields([('type', ctypes.c_uint32, 0), ('compiled', ctypes.POINTER(struct_cl_compiled_data), 8), ('executable', ctypes.POINTER(struct_cl_executable_data), 8)])
@dll.bind(ctypes.POINTER(cl_handle), cl_llvm_instance, uint64_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, uint64_t, uint64_t, ctypes.POINTER(ctypes.c_char), uint64_t, uint64_t, ctypes.c_void_p)
def cl_compiler_compile_source(inst:cl_llvm_instance, chip_id:uint64_t, mode:int, options:ctypes._Pointer[ctypes.c_char], p5:int, p6:uint64_t, p7:uint64_t, source:ctypes._Pointer[ctypes.c_char], source_len:uint64_t, source_type:uint64_t, p11:int|None) -> ctypes._Pointer[cl_handle]: ...
@dll.bind(ctypes.POINTER(cl_handle), cl_llvm_instance, uint64_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(cl_handle)))
def cl_compiler_link_program(inst:cl_llvm_instance, chip_id:uint64_t, mode:int, options:ctypes._Pointer[ctypes.c_char], num_handles:int, input_handles:ctypes._Pointer[ctypes.POINTER(cl_handle)]) -> ctypes._Pointer[cl_handle]: ...
size_t: TypeAlias = ctypes.c_uint64
@dll.bind(None, ctypes.POINTER(cl_handle), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t))
def cl_compiler_handle_create_binary(handle:ctypes._Pointer[cl_handle], out_ptr:ctypes._Pointer[ctypes.c_void_p], out_size:ctypes._Pointer[size_t]) -> None: ...
@c.record
class cl_lib_section(c.Struct):
  SIZE = 20
  id: int
  offset: int
  size: int
  count: int
  entry_size: int
cl_lib_section.register_fields([('id', uint32_t, 0), ('offset', uint32_t, 4), ('size', uint32_t, 8), ('count', uint32_t, 12), ('entry_size', uint32_t, 16)])
@c.record
class cl_lib_header(c.Struct):
  SIZE = 48
  _unk0: ctypes.Array[ctypes.c_uint32]
  num_sections: int
  _unk1: ctypes.Array[ctypes.c_uint32]
  sections: ctypes.Array[cl_lib_section]
cl_lib_header.register_fields([('_unk0', (uint32_t * 6), 0), ('num_sections', uint32_t, 24), ('_unk1', (uint32_t * 5), 28), ('sections', (cl_lib_section * 0), 48)])
@c.record
class cl_lib_prog(c.Struct):
  SIZE = 28
  name: ctypes.Array[ctypes.c_char]
  _unk0: ctypes.Array[ctypes.c_uint32]
  fregs: int
  hregs: int
cl_lib_prog.register_fields([('name', (ctypes.c_char * 8), 0), ('_unk0', (uint32_t * 3), 8), ('fregs', uint32_t, 20), ('hregs', uint32_t, 24)])
@c.record
class cl_lib_img_desc(c.Struct):
  SIZE = 344
  _unk0: ctypes.Array[ctypes.c_char]
  prg_offset: int
  pvtmem: int
  _unk1: ctypes.Array[ctypes.c_char]
  shmem: int
  samp_cnt: int
  _unk2: ctypes.Array[ctypes.c_char]
  brnchstck: int
  _unk4: ctypes.Array[ctypes.c_char]
  kernel_name: ctypes.Array[ctypes.c_char]
cl_lib_img_desc.register_fields([('_unk0', (ctypes.c_char * 196), 0), ('prg_offset', uint32_t, 196), ('pvtmem', uint32_t, 200), ('_unk1', (ctypes.c_char * 12), 204), ('shmem', uint32_t, 216), ('samp_cnt', uint32_t, 220), ('_unk2', (ctypes.c_char * 40), 224), ('brnchstck', uint32_t, 264), ('_unk4', (ctypes.c_char * 76), 268), ('kernel_name', (ctypes.c_char * 0), 344)])
@dll.bind(None, ctypes.POINTER(cl_handle))
def cl_compiler_free_handle(handle:ctypes._Pointer[cl_handle]) -> None: ...
@dll.bind(None, ctypes.c_void_p)
def cl_compiler_free_assembly(ptr:int|None) -> None: ...
CL_MODE_32BIT = 0 # type: ignore
CL_MODE_64BIT = 1 # type: ignore
CL_SRC_STR = 0 # type: ignore
CL_SRC_BLOB = 1 # type: ignore
CL_LIB_PROGRAM = 0 # type: ignore
CL_LIB_CONSTS = 6 # type: ignore
CL_LIB_IMAGE = 7 # type: ignore
CL_LIB_CODE = 10 # type: ignore
CL_LIB_IMAGE_DESC = 11 # type: ignore