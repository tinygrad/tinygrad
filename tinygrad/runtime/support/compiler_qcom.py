import ctypes, struct, sys, platform, pathlib
from tinygrad.device import Compiler
from tinygrad.helpers import DEBUG, system, fetch
from tinygrad.runtime.support.compiler_mesa import disas_adreno

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class QCOMCompiler(Compiler):
  def __init__(self, arch:str):
    assert arch.split(',')[0] == "a630", "only a630 supported"
    self.arch, self.chip_id = arch, 0x6030001
    super().__init__(f"compile_qcomcl_{arch}")

  def __reduce__(self): return QCOMCompiler, (self.arch,)
  def compile(self, src) -> bytes:
    if platform.machine() == "aarch64": return _compile(src, self.chip_id)
    tg, lib = pathlib.Path(__file__).parent.parent.parent, fetch("https://github.com/sirhcm/tinydreno/raw/refs/heads/master/libllvm-qcom.so")
    return system(f"docker run --rm -i --platform linux/aarch64 -v {tg}:/tinygrad -v {lib}:/lib/libllvm-qcom.so -e PYTHONPATH=/ python:3.12-slim python3 /tinygrad/runtime/support/compiler_qcom.py {self.chip_id}", input=src.encode(), decode=False)
  def disassemble(self, lib: bytes): disas_adreno(lib[(ofs:=_read_lib(lib, 0xc0)):ofs+_read_lib(lib, 0x100)], self.chip_id)

def _compile(src:str, chip_id:int) -> bytes:
  from tinygrad.runtime.autogen import llvm_qcom as qcom

  def checked(handle):
    if not handle or (data:=(hc.executable if (hc:=handle.contents).type == qcom.CL_HANDLE_LINKED else hc.compiled).contents).error_code != 0:
      raise RuntimeError("QCOM Compilation Error" + ("" if not handle else f": {ctypes.string_at(data.build_log).decode()}"))
    return handle

  llvm = qcom.cl_compiler_create_llvm_instance()
  ch = checked(qcom.cl_compiler_compile_source(llvm, chip_id, qcom.CL_MODE_64BIT, b"", 0, 0, 0, src.encode(), 0, qcom.CL_SRC_STR, None))
  if DEBUG >= 8: print(system("llvm-dis", input=ctypes.string_at((comp:=ch.contents.compiled.contents).llvm_bitcode, comp.llvm_bitcode_size)))
  lh = checked(qcom.cl_compiler_link_program(llvm, chip_id, qcom.CL_MODE_64BIT, None, 1, ch))
  qcom.cl_compiler_handle_create_binary(lh, ctypes.byref(ptr:=ctypes.c_void_p()), ctypes.byref(sz:=ctypes.c_size_t()))
  for h in [ch, lh]: qcom.cl_compiler_free_handle(h)
  ret = ctypes.string_at(ptr, sz.value)
  qcom.cl_compiler_free_assembly(ptr)
  return ret

if __name__ == "__main__": sys.stdout.buffer.write(_compile(sys.stdin.read(), int(sys.argv[1])))
