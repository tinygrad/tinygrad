from typing import Optional, List
import ctypes, subprocess, mmap, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution, DEBUG, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import libc

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=True) as file:
      elf_bytes = subprocess.check_output(['clang', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-c', '-fPIC', '-nostdlib', '-march=native', '-', '-o', pathlib.Path(file.name)], input=src.encode('utf-8'))
      raw_function_bytes = subprocess.check_output(['objcopy', '-O', 'binary', '--only-section=.text', pathlib.Path(file.name), '/dev/stdout'], input=elf_bytes)
      return raw_function_bytes

class ClangProgram:
  def __init__(self, name:str, obj:bytes):
    if DEBUG >= 6: cpu_objdump(obj)
    self.name, self.obj = name, obj
    self.buf, self.len = libc.mmap(0, len(obj), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0), len(obj)
    ctypes.memmove(self.buf, (ctypes.c_char * len(obj)).from_buffer_copy(obj), len(obj))
    self.fxn = lambda *args: ctypes.CFUNCTYPE(None)(self.buf)(*args)
  def __del__(self):
    libc.munmap(self.buf, self.len)
  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
