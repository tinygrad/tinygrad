import ctypes, os, struct, platform, pathlib, shutil, tarfile, tempfile
from tinygrad.device import Compiler
from tinygrad.helpers import DEBUG, system, fetch
from tinygrad.runtime.support.compileonce import compile_once
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
      self.fs, root = tempfile.TemporaryDirectory(), pathlib.Path(__file__).parents[3]
      with tarfile.open(fetch('https://git.tinygrad.win/sirhcm/images/releases/download/v2/qcomcl.tar.gz')) as t: t.extractall(self.fs.name)
      once = f"{pathlib.Path(__file__).parent}/compileonce.py {QCOMCompiler.__module__}:QCOMCompiler {arch}"
      self.compiler_env = None
      # the compiler is aarch64 only, run it emulated. for qemu user mode PYTHONPATH must point at the tinygrad on the host
      if (qemu:=shutil.which("qemu-aarch64-static")):
        self.compiler_cmd = f"{qemu} -cpu max,pauth=off -L {self.fs.name} {self.fs.name}/usr/bin/python3 {once}".split()
        self.compiler_env = {**os.environ, "PYTHONPATH": str(root)}
      else:
        self.compiler_cmd = (f"docker run --rm -i --platform linux/aarch64 -v {self.fs.name}/usr:/usr -v {root}:{root} "
                             f"-e PYTHONPATH={root} -e QEMU_CPU=max,pauth=off gcr.io/distroless/static python3 {once}").split()
    super().__init__(f"compile_qcomcl_{arch}")

  def __del__(self):
    if platform.machine() == "aarch64": llvm_qcom.cl_compiler_destroy_llvm_instance(self.llvm_inst)

  def __reduce__(self): return QCOMCompiler, (self.arch,)

  def checked(self, handle):
    if not handle or (data:=(hc.executable if (hc:=handle.contents).type == llvm_qcom.CL_HANDLE_LINKED else hc.compiled).contents).error_code != 0:
      llvm_qcom.cl_compiler_destroy_llvm_instance(self.llvm_inst)
      self.llvm_inst = llvm_qcom.cl_compiler_create_llvm_instance()
      raise RuntimeError("QCOM Compilation Error" + ("" if not handle else f": {ctypes.string_at(data.build_log).decode()}"))
    return handle

  def compile(self, src) -> bytes:
    if platform.machine() != "aarch64": return compile_once(self.compiler_cmd, src, self.compiler_env)
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
