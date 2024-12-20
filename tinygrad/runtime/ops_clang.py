from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile, os, tinygrad.runtime.autogen.libc as libc
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import getenv, cpu_time_execution, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer

# dlmopen is a GNU extension and so it won't exist on OSX or non-glibc linux
ISOLATE_DLOPEN = getenv("ISOLATE_DLOPEN", int(hasattr(libc, "dlmopen")))

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None, objdump_tool='objdump'):
    self.args = ['-march=native'] if args is None else args
    self.objdump_tool = objdump_tool
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-fno-math-errno', '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, self.objdump_tool)

class ClangProgram:
  # We can't use libc.LM_ID_NEWLM for every dlmopen because (at least in glibc) the maximum number of namespaces is limited and too small for us.
  # However having one namespace for all ClangPrograms shouldn't be a problem as long as we use RTLD_LOCAL for every dlmopen.
  NAMESPACE_ID = libc.Lmid_t(libc.LM_ID_NEWLM if ISOLATE_DLOPEN else libc.LM_ID_BASE)
  def __init__(self, name:str, lib:bytes):
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      if ISOLATE_DLOPEN:
        handle = libc.dlmopen(ClangProgram.NAMESPACE_ID, str(cached_file_path.name).encode(), os.RTLD_NOW | os.RTLD_LOCAL)
      else:
        handle = libc.dlopen(str(cached_file_path.name).encode(), os.RTLD_NOW | os.RTLD_LOCAL)
    if handle is None: raise RuntimeError("failed to import dynamic library: try ISOLATE_DLOPEN=0")
    if ClangProgram.NAMESPACE_ID.value == libc.LM_ID_NEWLM:
      assert libc.dlinfo(handle, libc.RTLD_DI_LMID, ctypes.byref(ClangProgram.NAMESPACE_ID)) == 0
    self.fxn = ctypes.CDLL(None, handle=handle)[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram)
