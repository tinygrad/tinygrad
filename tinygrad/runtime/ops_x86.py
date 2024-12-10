from tinygrad.device import Compiled, MallocAllocator
from tinygrad.renderer.x86asm import X86Renderer
from tinygrad.runtime.ops_clang import ClangCompiler, ClangProgram

class X86Device(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, X86Renderer(), ClangCompiler(cachekey="compile_x86", lang=['assembler', '-masm=intel']), ClangProgram)
