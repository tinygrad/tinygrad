import ctypes, struct
from tinygrad.device import Compiler
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.compiler_mesa import disas_adreno

# see https://github.com/sirhcm/tinydreno
dll = DLL("llvm-qcom", ["llvm-qcom"])

(create_llvm_instance:=dll.cl_compiler_create_llvm_instance).restype, create_llvm_instance.argtypes = ctypes.c_void_p, []

(compile_source:=dll.cl_compiler_compile_source).restype = ctypes.c_void_p
compile_source.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                           ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]

(link_program:=dll.cl_compiler_link_program).restype = ctypes.c_void_p
link_program.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]

(get_error_code:=dll.cl_compiler_get_error_code).restype, get_error_code.argtypes = ctypes.c_int, [ctypes.c_void_p]
(get_build_log:=dll.cl_compiler_get_build_log).restype, get_build_log.argtypes = ctypes.c_char_p, [ctypes.c_void_p]

(handle_create_binary:=dll.cl_compiler_handle_create_binary).restype = None
handle_create_binary.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]

(free_handle:=dll.cl_compiler_free_handle).restype, free_handle.argtypes = None, [ctypes.c_void_p]
(free_assembly:=dll.cl_compiler_free_assembly).restype, free_assembly.argtypes = None, [ctypes.c_void_p]
(destroy_llvm_instance:=dll.cl_compiler_destroy_llvm_instance).restype, destroy_llvm_instance.argtypes = None, [ctypes.c_void_p]

MODE_32BIT, MODE_64BIT, SRC_STR, SRC_BLOB = 0, 1, 0, 1

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]
def checked(handle):
  assert handle is not None and get_error_code(handle) == 0, "QCOM Compilation Error" + ("" if handle is None else f": {get_build_log(handle)}")
  return handle

class QCOMCompiler(Compiler):
  def __init__(self, chip_id):
    print(hex(chip_id))
    self.chip_id, self.llvm_inst = chip_id, create_llvm_instance()
    super().__init__(f"compile_qcomcl_{chip_id}")

  def __del__(self): destroy_llvm_instance(self.llvm_inst)

  def __reduce__(self): return QCOMCompiler, (self.chip_id,)

  def compile(self, src) -> bytes:
    ch = checked(compile_source(self.llvm_inst, self.chip_id, MODE_64BIT, b"", 0, 0, 0, src.encode(), 0, SRC_STR, None))
    lh = checked(link_program(self.llvm_inst, self.chip_id, MODE_64BIT, None, 1, ctypes.pointer(ctypes.c_void_p(ch))))
    handle_create_binary(lh, ctypes.byref(ptr:=ctypes.c_void_p()), ctypes.byref(sz:=ctypes.c_size_t()))
    for h in [ch, lh]: free_handle(h)
    ret = ctypes.string_at(ptr, sz.value)
    free_assembly(ptr)
    return ret

  def disassemble(self, lib: bytes): disas_adreno(lib[(ofs:=_read_lib(lib, 0xc0)):ofs+_read_lib(lib, 0x100)], self.chip_id)
