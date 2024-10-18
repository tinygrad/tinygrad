from typing import Optional, List
import ctypes, subprocess, mmap, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import OSX, cpu_objdump, cpu_time_execution, DEBUG
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import libc

linker_script = '''SECTIONS {
  .text : {
    . = 0x0;
    *(.tinygrad);
    *(.text*);
    *(.rodata*);
    *(.tbss*);
  }
}'''
class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=True) as file, tempfile.NamedTemporaryFile(delete=True) as linker_file:
      pathlib.Path(linker_file.name).write_text(linker_script)
      subprocess.check_output([
        'clang',*self.args,'-O2','-Wall','-Werror','-Wno-gcc-compat','-x','c','-','-fPIE','-fPIC','-static','-lc','-lm',
        '-o',pathlib.Path(file.name),'-T',pathlib.Path(linker_file.name),'-nostdlib','-ffreestanding','-Wl,--build-id=none'
      ], input=src.encode('utf-8'))
      return subprocess.check_output(['objcopy', '-O', 'binary', '--only-section=.text', pathlib.Path(file.name), '/dev/stdout'])

class ClangProgram:
  def __init__(self, name:str, code:bytes):
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as f:
        pathlib.Path(f.name).write_bytes(code)
        print(subprocess.check_output(['objdump','-b','binary','-m','i','--disassembler-color=on','-D',f.name]).decode('utf-8'))
    self.name, self.obj = name, code
    self.buf, self.len = libc.mmap(0, len(code), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                                   mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0), len(code)
    ctypes.memmove(self.buf, (ctypes.c_char * len(code)).from_buffer_copy(code), len(code))
  def __del__(self):
    if hasattr(self, "buf") and hasattr(self, 'len'): libc.munmap(self.buf, self.len)
  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: ctypes.CFUNCTYPE(None)(self.buf)(*bufs, *vals), enable=wait)

class ClangCompilerOSX(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-shared', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgramOSX:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph, ClangGraphOSX
    if OSX: super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompilerOSX(), ClangProgramOSX, ClangGraphOSX)
    else:   super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
