import ctypes, subprocess, pathlib, tempfile, math, mmap as mmap_flags
from tinygrad.device import Compiled, MallocAllocator, Compiler, CompilerOptions
from tinygrad.helpers import cpu_time_execution
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

CLANG_PROGRAM_HEADER = '#include <stdbool.h>\n#include <tgmath.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n'

# implement jit as per here
# https://gist.github.com/jumbojets/1d7008901826eb9de1f2aa608f368847
# NOTE: mmap doesnt have mprotect, so have to load libc
# https://github.com/python/cpython/issues/114233

libc = ctypes.cdll.LoadLibrary(None)
mmap = libc.mmap
mmap.restype = ctypes.c_void_p
mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t)

mprotect = libc.mprotect
mprotect.restype = ctypes.c_int
mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int)

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
    mmap_size = 2**math.ceil(math.log2(len(lib))+1)
    code_addr = mmap(None, mmap_size, mmap_flags.PROT_READ | mmap_flags.PROT_WRITE, mmap_flags.MAP_ANON | mmap_flags.MAP_PRIVATE, -1, 0)
    if code_addr == -1: raise OSError('mmap failed to allocate memory')
    ctypes.memmove(code_addr, lib, len(lib))
    if mprotect(code_addr, mmap_size, mmap_flags.PROT_READ | mmap_flags.PROT_EXEC, 0) < 0: raise OSError('mprotect failed to make memory executable')
    self.fxn = ctypes.CFUNCTYPE(None)(code_addr + 0x0000000000003e44) # TODO: just get offset from symbol table?

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangCompiler("compile_clang"), ClangProgram)
