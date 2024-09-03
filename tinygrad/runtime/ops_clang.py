import ctypes, subprocess, pathlib, tempfile, mmap, platform
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import OSX, cpu_time_execution, DEBUG, cpu_objdump, mv_address
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    triple = f"{platform.machine()}-none-unknown-elf"
    asm = subprocess.check_output(['clang', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-S', '-ffreestanding', '-nostdlib', '-target',
                                   triple, '-fno-jump-tables', '-fno-math-errno', '-fPIC', '-', '-o', '-'], input=src.encode('utf-8'))
    lines, data, text, state = asm.decode('utf-8').split('\n'), [], [], ""
    for l in lines:
      if '.text' in l: state = 'text'
      elif '.rodata' in l: state = 'rodata'
      elif '.note.GNU-stack' in l: break
      elif state == 'text': text.append(l.replace('adrp', 'adr') if platform.machine() == 'aarch64' else l)
      elif state == 'rodata': data.append(l)
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', '-target', triple, '-x', 'assembler', '-c', '-', '-o', str(output_file.name)],
                              input=('\n'.join(text+data)+'\n').encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    image = elf_loader(lib)[0]
    addr = libc.mmap(0, len(image), mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_ANON | mmap.MAP_PRIVATE, -1, 0)
    assert addr != 0xffffffffffffffff
    ctypes.memmove(addr, mv_address(image), len(image))
    if OSX: libc.sys_icache_invalidate(addr, len(image))
    assert libc.mprotect(addr, len(image), mmap.PROT_READ | mmap.PROT_EXEC) != 0xffffffffffffffff
    self.fxn = ctypes.CFUNCTYPE(None)(addr)

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler("compile_clang"), ClangProgram, ClangGraph)
