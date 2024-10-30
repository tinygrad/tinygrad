from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution, DEBUG, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen.libc import mmap, memcpy
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = args or []
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      subprocess.check_output(['objcopy', output_file.name, '-j', '.text', '-O', 'binary'])
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    flags = MAP_ANON | MAP_PRIVATE | 0x100 # no MAP_FIXED in mmap for some reason
    self.fxn = ctypes.CFUNCTYPE(None)(memcpy(mmap(0, len(lib), PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0), lib, len(lib)))

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
