import ctypes, struct, platform, pathlib, subprocess, sys
from tinygrad.device import Compiler
from tinygrad.helpers import DEBUG, system, fetch, unwrap
from tinygrad.runtime.support.compiler_mesa import disas_adreno
# see https://github.com/sirhcm/tinydreno
from tinygrad.runtime.autogen import llvm_qcom

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class QCOMCompiler(Compiler):
  def __init__(self, arch:str):
    assert arch.split(',')[0] == "a630", "only a630 supported"
    self.arch, self.chip_id = arch, 0x6030001
    if platform.machine() == "aarch64": self.llvm_inst = llvm_qcom.cl_compiler_create_llvm_instance()
    else:
      tg, lib = pathlib.Path(__file__).parent.parent.parent, fetch("https://github.com/sirhcm/tinydreno/raw/refs/heads/master/libllvm-qcom.so")
      self.compiler_process = subprocess.Popen((f"docker run --rm -i --platform linux/aarch64 -v {tg}:/tinygrad -v {lib}:/lib/libllvm-qcom.so "
                                                f"-e PYTHONPATH=/ python:3.12-slim python /tinygrad/runtime/support/compiler_qcom.py {arch}").split(),
                                               stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
    super().__init__(f"compile_qcomcl_{arch}")

  def __del__(self):
    if platform.machine() == "aarch64": llvm_qcom.cl_compiler_destroy_llvm_instance(self.llvm_inst)
    else: self.compiler_process.kill()

  def __reduce__(self): return QCOMCompiler, (self.arch,)

  def checked(self, handle):
    if not handle or (data:=(hc.executable if (hc:=handle.contents).type == llvm_qcom.CL_HANDLE_LINKED else hc.compiled).contents).error_code != 0:
      llvm_qcom.cl_compiler_destroy_llvm_instance(self.llvm_inst)
      self.llvm_inst = llvm_qcom.cl_compiler_create_llvm_instance()
      raise RuntimeError("QCOM Compilation Error" + ("" if not handle else f": {ctypes.string_at(data.build_log).decode()}"))
    return handle

  def compile(self, src) -> bytes:
    if platform.machine() != "aarch64":
      unwrap(self.compiler_process.stdin).write(struct.pack("I", len(src.encode())) + src.encode())
      if (lib:=unwrap(self.compiler_process.stdout).read(struct.unpack("I", unwrap(self.compiler_process.stdout).read(4))[0])): return lib
      raise RuntimeError("QCOM Compilation Error")
    ch = self.checked(llvm_qcom.cl_compiler_compile_source(self.llvm_inst, self.chip_id, llvm_qcom.CL_MODE_64BIT, b"", 0, 0, 0, src.encode(), 0,
                                                           llvm_qcom.CL_SRC_STR, None))
    if DEBUG >= 8: print(system("llvm-dis", input=ctypes.string_at((comp:=ch.contents.compiled.contents).llvm_bitcode, comp.llvm_bitcode_size)))
    lh = self.checked(llvm_qcom.cl_compiler_link_program(self.llvm_inst, self.chip_id, llvm_qcom.CL_MODE_64BIT, None, 1, ch))
    llvm_qcom.cl_compiler_handle_create_binary(lh, ctypes.byref(ptr:=ctypes.c_void_p()), ctypes.byref(sz:=ctypes.c_size_t()))
    for h in [ch, lh]: llvm_qcom.cl_compiler_free_handle(h)
    ret = ctypes.string_at(ptr, sz.value)
    llvm_qcom.cl_compiler_free_assembly(ptr)
    return ret

  def disassemble(self, lib: bytes): disas_adreno(lib[(ofs:=_read_lib(lib, 0xc0)):ofs+_read_lib(lib, 0x100)], self.chip_id)

if __name__ == "__main__":
  assert platform.machine() == "aarch64"
  compiler = QCOMCompiler(sys.argv[1])
  while (amt:=sys.stdin.buffer.read(4)):
    try: lib = compiler.compile(sys.stdin.read(struct.unpack("I", amt)[0]))
    except Exception as e:
      lib = b""
      print(e, file=sys.stderr, flush=True)
    sys.stdout.buffer.write(struct.pack("I", len(lib)) + lib)
    sys.stdout.buffer.flush()
