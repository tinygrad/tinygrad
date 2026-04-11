# mypy: disable-error-code="empty-body"
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('llvm_qcom', 'llvm-qcom')
cl_llvm_instance: TypeAlias = ctypes.c_void_p
@dll.bind
def cl_compiler_create_llvm_instance() -> cl_llvm_instance: ...
@dll.bind
def cl_compiler_destroy_llvm_instance(inst:cl_llvm_instance) -> None: ...
class enum_cl_handle_type(ctypes.c_uint32, c.Enum): pass
CL_HANDLE_COMPILED = enum_cl_handle_type.define('CL_HANDLE_COMPILED', 1)
CL_HANDLE_LIBRARY = enum_cl_handle_type.define('CL_HANDLE_LIBRARY', 2)
CL_HANDLE_LINKED = enum_cl_handle_type.define('CL_HANDLE_LINKED', 3)

@c.record
class struct_cl_compiled_data(c.Struct):
  SIZE = 48
  chip_id: 'uint64_t'
  mode: 'uint32_t'
  llvm_bitcode: 'ctypes.c_void_p'
  llvm_bitcode_size: 'uint64_t'
  build_log: 'c.POINTER[ctypes.c_char]'
  build_log_len: 'uint32_t'
  error_code: 'uint32_t'
uint64_t: TypeAlias = ctypes.c_uint64
uint32_t: TypeAlias = ctypes.c_uint32
struct_cl_compiled_data.register_fields([('chip_id', uint64_t, 0), ('mode', uint32_t, 8), ('llvm_bitcode', ctypes.c_void_p, 16), ('llvm_bitcode_size', uint64_t, 24), ('build_log', c.POINTER[ctypes.c_char], 32), ('build_log_len', uint32_t, 40), ('error_code', uint32_t, 44)])
@c.record
class struct_cl_executable_data(c.Struct):
  SIZE = 80
  num_kernels: 'int32_t'
  kernel_props: 'ctypes.c_void_p'
  error_code: 'uint32_t'
  build_log: 'c.POINTER[ctypes.c_char]'
  _unk0: 'c.Array[ctypes.c_char, Literal[32]]'
  chip_id: 'uint64_t'
  mode: 'uint32_t'
int32_t: TypeAlias = ctypes.c_int32
struct_cl_executable_data.register_fields([('num_kernels', int32_t, 0), ('kernel_props', ctypes.c_void_p, 8), ('error_code', uint32_t, 16), ('build_log', c.POINTER[ctypes.c_char], 24), ('_unk0', c.Array[ctypes.c_char, Literal[32]], 32), ('chip_id', uint64_t, 64), ('mode', uint32_t, 72)])
@c.record
class cl_handle(c.Struct):
  SIZE = 16
  type: 'enum_cl_handle_type'
  compiled: 'c.POINTER[struct_cl_compiled_data]'
  executable: 'c.POINTER[struct_cl_executable_data]'
cl_handle.register_fields([('type', enum_cl_handle_type, 0), ('compiled', c.POINTER[struct_cl_compiled_data], 8), ('executable', c.POINTER[struct_cl_executable_data], 8)])
@dll.bind
def cl_compiler_compile_source(inst:cl_llvm_instance, chip_id:uint64_t, mode:ctypes.c_int32, options:c.POINTER[ctypes.c_char], p5:ctypes.c_int32, p6:uint64_t, p7:uint64_t, source:c.POINTER[ctypes.c_char], source_len:uint64_t, source_type:uint64_t, p11:ctypes.c_void_p) -> c.POINTER[cl_handle]: ...
@dll.bind
def cl_compiler_link_program(inst:cl_llvm_instance, chip_id:uint64_t, mode:ctypes.c_int32, options:c.POINTER[ctypes.c_char], num_handles:ctypes.c_int32, input_handles:c.POINTER[c.POINTER[cl_handle]]) -> c.POINTER[cl_handle]: ...
size_t: TypeAlias = ctypes.c_uint64
@dll.bind
def cl_compiler_handle_create_binary(handle:c.POINTER[cl_handle], out_ptr:c.POINTER[ctypes.c_void_p], out_size:c.POINTER[size_t]) -> None: ...
@c.record
class cl_lib_section(c.Struct):
  SIZE = 20
  id: 'uint32_t'
  offset: 'uint32_t'
  size: 'uint32_t'
  count: 'uint32_t'
  entry_size: 'uint32_t'
cl_lib_section.register_fields([('id', uint32_t, 0), ('offset', uint32_t, 4), ('size', uint32_t, 8), ('count', uint32_t, 12), ('entry_size', uint32_t, 16)])
@c.record
class cl_lib_header(c.Struct):
  SIZE = 48
  _unk0: 'c.Array[uint32_t, Literal[6]]'
  num_sections: 'uint32_t'
  _unk1: 'c.Array[uint32_t, Literal[5]]'
  sections: 'c.Array[cl_lib_section, Literal[0]]'
cl_lib_header.register_fields([('_unk0', c.Array[uint32_t, Literal[6]], 0), ('num_sections', uint32_t, 24), ('_unk1', c.Array[uint32_t, Literal[5]], 28), ('sections', c.Array[cl_lib_section, Literal[0]], 48)])
@c.record
class cl_lib_prog(c.Struct):
  SIZE = 28
  name: 'c.Array[ctypes.c_char, Literal[8]]'
  _unk0: 'c.Array[uint32_t, Literal[3]]'
  fregs: 'uint32_t'
  hregs: 'uint32_t'
cl_lib_prog.register_fields([('name', c.Array[ctypes.c_char, Literal[8]], 0), ('_unk0', c.Array[uint32_t, Literal[3]], 8), ('fregs', uint32_t, 20), ('hregs', uint32_t, 24)])
@c.record
class cl_lib_img_desc(c.Struct):
  SIZE = 344
  _unk0: 'c.Array[ctypes.c_char, Literal[196]]'
  prg_offset: 'uint32_t'
  pvtmem: 'uint32_t'
  _unk1: 'c.Array[ctypes.c_char, Literal[12]]'
  shmem: 'uint32_t'
  samp_cnt: 'uint32_t'
  _unk2: 'c.Array[ctypes.c_char, Literal[40]]'
  brnchstck: 'uint32_t'
  _unk4: 'c.Array[ctypes.c_char, Literal[76]]'
  kernel_name: 'c.Array[ctypes.c_char, Literal[0]]'
cl_lib_img_desc.register_fields([('_unk0', c.Array[ctypes.c_char, Literal[196]], 0), ('prg_offset', uint32_t, 196), ('pvtmem', uint32_t, 200), ('_unk1', c.Array[ctypes.c_char, Literal[12]], 204), ('shmem', uint32_t, 216), ('samp_cnt', uint32_t, 220), ('_unk2', c.Array[ctypes.c_char, Literal[40]], 224), ('brnchstck', uint32_t, 264), ('_unk4', c.Array[ctypes.c_char, Literal[76]], 268), ('kernel_name', c.Array[ctypes.c_char, Literal[0]], 344)])
@dll.bind
def cl_compiler_free_handle(handle:c.POINTER[cl_handle]) -> None: ...
@dll.bind
def cl_compiler_free_assembly(ptr:ctypes.c_void_p) -> None: ...
CL_MODE_32BIT = 0 # type: ignore
CL_MODE_64BIT = 1 # type: ignore
CL_SRC_STR = 0 # type: ignore
CL_SRC_BLOB = 1 # type: ignore
CL_LIB_PROGRAM = 0 # type: ignore
CL_LIB_CONSTS = 6 # type: ignore
CL_LIB_IMAGE = 7 # type: ignore
CL_LIB_CODE = 10 # type: ignore
CL_LIB_IMAGE_DESC = 11 # type: ignore