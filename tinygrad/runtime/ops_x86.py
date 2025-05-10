from tinygrad.device import Compiled, MallocAllocator
from tinygrad.renderer.x86 import X86Renderer
from tinygrad.runtime.ops_cpu import ClangJITCompiler, CPUProgram

class X86Device(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, X86Renderer(),
                      ClangJITCompiler(cachekey="compile_x86", lang_args=['assembler', '-masm=intel']), CPUProgram)
