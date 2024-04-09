import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, MallocAllocator, Compiler, CompilerOptions
from tinygrad.helpers import cpu_time_execution
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

CLANG_PROGRAM_HEADER = '#include <stdbool.h>\n#include <tgmath.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n'

class ClangCompiler(Compiler):
  compiler_opts = CompilerOptions("CLANG", supports_float4=False, has_local=False)
  def render(self, name:str, uops) -> str: return CLANG_PROGRAM_HEADER + uops_to_cstyle(CStyleLanguage(buffer_suffix=" restrict"), name, uops)
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(args=('clang -shared -march=native -O2 -Wall -Werror -x c -fPIC - -o '+ str(output_file.name)).split(),
                              input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangCompiler("compile_clang"), ClangProgram)
