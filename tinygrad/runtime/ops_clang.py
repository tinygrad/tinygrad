import platform, tempfile, pathlib, subprocess
from tinygrad.helpers import cpu_objdump, capstone_flatdump
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.renderer.cstyle import ClangRenderer

# Used by ops_dsp.py
class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:list[str]|None=None, objdump_tool='objdump'):
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

class ClangJITCompiler(Compiler):
  def __init__(self, cachekey="compile_clang_jit"): super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    args = ['-march=native', f'--target={platform.machine()}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib']
    arch_args = ['-ffixed-x18'] if platform.machine() == 'arm64' else []
    obj = subprocess.check_output(['clang', '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    return jit_loader(obj)

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangRenderer(), ClangJITCompiler(), CPUProgram)
