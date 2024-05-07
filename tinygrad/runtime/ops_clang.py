import ctypes, subprocess, pathlib, tempfile, os, math
from tinygrad.device import Compiled, MallocAllocator, Compiler, CompilerOptions
from tinygrad.helpers import cpu_time_execution, getenv
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

OMP_HEADER, OMP_FLAGS = ("#include <omp.h>\n", "-Xpreprocessor -fopenmp -lomp") if (OMP_SET:=getenv("OMP", 0)) else ("", "")
CLANG_PROGRAM_HEADER = OMP_HEADER +'#include <stdbool.h>\n#include <tgmath.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n'

class ClangCompiler(Compiler):
  compiler_opts = CompilerOptions("CLANG", supports_float4=False, has_local=OMP_SET, global_max=[1,1,64], local_max=[2,1,1])
  def render(self, name:str, uops) -> str: return CLANG_PROGRAM_HEADER + uops_to_cstyle(CStyleLanguage(buffer_suffix=" restrict"), name, uops)
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(args=('clang -shared -march=native '+ OMP_FLAGS +' -O2 -Wall -Werror -x c -fPIC - -o '+ str(output_file.name)).split(),
                              input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False):
    # NOTE: local needs exact number of threads, but global could be dynamic and blocked. maybe global should even be serial
    if global_size is not None: os.environ["OMP_NUM_THREADS"] = ",".join(map(str, [math.prod(global_size)] + ([math.prod(local_size)] if local_size is not None else [])))
    if global_size is not None and local_size is not None: os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"
    return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangCompiler("compile_clang"), ClangProgram)
