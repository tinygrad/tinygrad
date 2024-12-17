from __future__ import annotations
from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile, functools
from tinygrad.device import Compiled, Compiler, MallocAllocator, ProfileDeviceEvent, cpu_profile
from tinygrad.helpers import cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None, objdump_tool='objdump'):
    self.args = ['-march=native'] if args is None else args
    self.objdump_tool = objdump_tool
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, self.objdump_tool)

class ClangProgram:
  def __init__(self, dev:ClangDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): 
    with cpu_profile(self.dev.device, self.name, is_copy=False) as cpu_time_execution: self.fxn(*bufs, *vals)
    return cpu_time_execution.en - cpu_time_execution.st

class ClangDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), functools.partial(ClangProgram, self))
    self.profile_events += [ProfileDeviceEvent(device)]
