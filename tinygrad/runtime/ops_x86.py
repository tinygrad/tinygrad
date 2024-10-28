from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution
from tinygrad.renderer.x86asm import X86Renderer

class X86Compiler(Compiler):
  def __init__(self, cachekey="compile_x86", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-x', 'assembler', '-masm=intel', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class X86Program:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class X86Device(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, X86Renderer(), X86Compiler(), X86Program)
