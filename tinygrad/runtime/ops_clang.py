import os, platform, subprocess
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.driver.elf_loader import ElfLoader, addrof
from tinygrad.helpers import cpu_time_execution
from tinygrad.renderer.cstyle import ClangRenderer

loader = ElfLoader()

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    platspec = ('-ffixed-x18',) if platform.machine() == "arm64" else ('-Xclang=-fnative-half-type', '-Xclang=-fnative-half-arguments-and-returns')
    return subprocess.check_output(('clang', '-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2',
                                    '-fno-builtin' ,'-ffreestanding', '-Wall', '-Werror', '-include',
                                    f'{os.path.dirname(__file__)}/autogen/tinymath.h', '-', '-o', '-')+platspec,
                                    input=src.encode('utf-8'))

with open(f'{os.path.dirname(__file__)}/autogen/tinymath.c') as fd:
  syms = loader.load_elf(ClangCompiler().compile(fd.read()))
  loader.link({k:addrof(v) for k,v in syms.items()})

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib, self.fxn = name, lib, loader.load_elf(lib)[name]
  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler("compile_clang_object"), ClangProgram, ClangGraph)
