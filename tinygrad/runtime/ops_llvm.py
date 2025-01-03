from tinygrad.device import Compiled, MallocAllocator
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.runtime.ops_clang import ClangJITCompiler, CPUProgram

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, LLVMRenderer(), ClangJITCompiler('compile_llvm_jit', ['-x', 'ir']), CPUProgram)
