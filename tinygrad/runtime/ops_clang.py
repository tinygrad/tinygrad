import os, platform, subprocess, ctypes, pickle, tinygrad.runtime.autogen.libc as libc
from mmap import PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.runtime.support.elf import fixup_relocations
from tinygrad.helpers import cpu_time_execution, OSX
from tinygrad.renderer.cstyle import ClangRenderer

if OSX: mac_libc = ctypes.CDLL(ctypes.util.find_library('c')) # needed for jit write protect and sys icache invalidate
MAP_JIT = 0x0800 if OSX else 0

class ClangCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    args = ('clang', '-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-Wall',
            '-Wno-unused-function', '-Wno-unused-command-line-argument', '-Werror', '-include', f'{os.path.dirname(__file__)}/support/tinymath.h',
            '-ffreestanding', '-nostdlib', '-ffixed-x18' if platform.machine() == "arm64" else '-Xclang=-fnative-half-type', '-', '-o', '-')
    return fixup_relocations(subprocess.check_output(args, input=src.encode('utf-8')))

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    exports, lib = pickle.loads(lib)
    self.map, self.mlen = libc.mmap(None, len(lib), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0), len(lib)
    if OSX: mac_libc.pthread_jit_write_protect_np(False)
    ctypes.memmove(self.map, lib, len(lib))
    if OSX:
      mac_libc.pthread_jit_write_protect_np(True)
      mac_libc.sys_icache_invalidate(ctypes.c_void_p(self.map), ctypes.c_size_t(len(lib)))
    # TODO: clear instruction cache on linux
    self.fxn = ctypes.cast(self.map+exports[name], ctypes.CFUNCTYPE(None))
  def __del__(self): libc.munmap(self.map, self.mlen)
  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    # default arm abi requires stack slots to be 8 byte aligned, macos abi doesn't
    if platform.machine() == 'arm64' and OSX and len(args) > 8: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler("compile_clang_object"), ClangProgram, ClangGraph)
