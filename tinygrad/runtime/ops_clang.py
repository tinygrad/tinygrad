import os, platform, subprocess, ctypes, pickle
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.support.elf_linker import fixup_relocations
from tinygrad.helpers import cpu_time_execution, cpu_objdump, DEBUG, OSX
from tinygrad.renderer.cstyle import ClangRenderer

libc = ctypes.CDLL(None)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t)

MAP_JIT = 0x0800 if OSX else 0x0

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    platspec = ('-ffixed-x18',) if platform.machine() == "arm64" else ('-Xclang=-fnative-half-type', '-Xclang=-fnative-half-arguments-and-returns')
    return fixup_relocations(subprocess.check_output(('clang', '-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native',
                                                      '-fPIC', '-O2', '-fno-builtin', '-ffreestanding', '-nostdlib', '-fno-math-errno',
                                                      '-Wall', '-Wno-unused-function', '-Wunused-command-line-argument', '-Werror',
                                                      '-include', f'{os.path.dirname(__file__)}/support/tinymath.h', '-', '-o', '-')+platspec,
                                                      input=src.encode('utf-8')))

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    exports, lib = pickle.loads(lib)
    if DEBUG >= 6: cpu_objdump(lib)
    self.name, self.lib = name, lib
    self.map = libc.mmap(None, len(lib), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0)
    if OSX: libc.pthread_jit_write_protect_np(False)
    ctypes.memmove(self.map, lib, len(lib))
    if OSX:
      libc.pthread_jit_write_protect_np(True)
      libc.sys_icache_invalidate(ctypes.c_void_p(self.map), ctypes.c_size_t(len(lib)))
    else:
      # TODO: clear instruction cache
      pass
    self.fxn = ctypes.cast(self.map+exports[name], ctypes.CFUNCTYPE(None))
  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler("compile_clang_object"), ClangProgram, ClangGraph)
