from tinygrad.device import Compiled, MallocAllocator, Compiler, CompilerOptions, CDLLStyleProgram
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

CLANG_PROG_HEADER = '#include <stdbool.h>\n#include <tgmath.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n'
CLANG_COMPILE_CMD = 'clang -shared -march=native -O2 -Wall -Werror -x c -fPIC - -o '

class ClangCompiler(Compiler):
  compiler_opts = CompilerOptions("CLANG", supports_float4=False, has_local=False)
  def render(self, name:str, uops, bufsz=[]) -> str: return CLANG_PROG_HEADER + uops_to_cstyle(CStyleLanguage(buffer_suffix=" restrict"), name, uops)
  def compile(self, src:str) -> bytes: return self.compile_file(CLANG_COMPILE_CMD, src)

ClangProgram = CDLLStyleProgram

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangCompiler("compile_clang"), ClangProgram)
