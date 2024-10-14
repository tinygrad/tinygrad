from typing import Optional, List
import ctypes, subprocess, mmap, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution, DEBUG
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import libc

linker_script = '''
SECTIONS {
    .text : {
        . = 0x0;
        *(.text);
        *(.rodata*);
    }
    /DISCARD/ : {
      *(.*);
    } : phdr
}
'''

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=True) as file, tempfile.NamedTemporaryFile(delete=True) as linker_file:
      pathlib.Path(linker_file.name).write_text(linker_script)
      subprocess.check_output([
        'clang', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-', '-fmerge-all-constants',
        '-ffreestanding', '-fPIE', '-fPIC', '-nostdlib', '-', '-o', pathlib.Path(file.name),
        '-T', pathlib.Path(linker_file.name)
      ], input=src.encode('utf-8'))
      raw_function_bytes = subprocess.check_output([
        'objcopy', '-O', 'binary', '--only-section=.text', pathlib.Path(file.name), '/dev/stdout'])
      return raw_function_bytes

class ClangProgram:
  def __init__(self, name:str, code:bytes):
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as f:
        pathlib.Path(f.name).write_bytes(code)
        print(subprocess.check_output(['objdump', '-b', 'binary', '-m', 'i', '--disassembler-color=on', '-D', f.name]).decode('utf-8'))
    self.name, self.obj = name, code
    self.buf, self.len = libc.mmap(0, len(code), mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                                   mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0), len(code)
    ctypes.memmove(self.buf, (ctypes.c_char * len(code)).from_buffer_copy(code), len(code))
    self.fxn = lambda *args: ctypes.CFUNCTYPE(None)(self.buf)(*args)
  def __del__(self):
    libc.munmap(self.buf, self.len)
  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
