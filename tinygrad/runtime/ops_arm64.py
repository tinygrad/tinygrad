from tinygrad.device import Compiled, MallocAllocator
from tinygrad.renderer.asm import Arm64Renderer
from tinygrad.runtime.ops_cpu import ClangJITCompiler, CPUProgram

class ARM64Device(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, Arm64Renderer(),
                      ClangJITCompiler(cachekey="compile_arm64", lang_args=['assembler'], opt_args=[]), CPUProgram)
