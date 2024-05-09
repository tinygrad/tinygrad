import  subprocess, platform, os
from tinygrad.device import Compiled, MallocAllocator, Compiler, CompilerOptions
from tinygrad.runtime.driver.elf_loader import load_elf
from tinygrad.helpers import cpu_time_execution
from tinygrad.renderer.cstyle import ClangRenderer

class ClangCompiler(Compiler):
  compiler_opts = CompilerOptions("CLANG", supports_float4=False, has_local=False)
  def render(self, name:str, uops) -> str: return ClangRenderer(name, uops)
  def compile(self, src:str) -> bytes:
    return subprocess.check_output(('clang', '-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2',
                                    '-ffixed-x18', '-fno-builtin','-ffreestanding', '-Wall', '-Werror',
                                    '-include', f'{os.path.dirname(__file__)}/autogen/tinymath.h', '-', '-o', '-'),
                                    input=src.encode('utf-8'))

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib, self.fxn = name, lib, load_elf(lib)[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangCompiler("compile_clang_object"), ClangProgram, ClangGraph)
