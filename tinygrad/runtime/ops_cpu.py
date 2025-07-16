from __future__ import annotations
import platform, subprocess, sys, ctypes, functools
from typing import ClassVar
from tinygrad.helpers import capstone_flatdump, getenv, from_mv, to_mv, OSX, mv_address
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram, BufferSpec
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface, HCQAllocatorBase
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import sint

class CPUSignal(HCQSignal):
  def __init__(self, base_buf:HCQBuffer|None=None, **kwargs):
    super().__init__(base_buf, **kwargs, timestamp_divider=1e3, dev_t=CPUDevice)

class ClangJITCompiler(Compiler):
  def __init__(self, cachekey="compile_clang_jit"): super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    args = ['-march=native', f'--target={target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib', '-fno-ident']
    arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    obj = subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    return jit_loader(obj)

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class CPUComputeQueue(HWQueue):
  def _exec(self, prg, gx, gy, gz, lx, ly, lz, *args):
    prg.fxn(*[ctypes.c_int64(a) if isinstance(a, int) else ctypes.c_int64(a.va_addr) for a in args])
  def _signal(self, signal_addr, value): to_mv(signal_addr, 4).cast('I')[0] = value
  def _wait(self, signal_addr, value):
    while to_mv(signal_addr, 4).cast('I')[0] != value: pass
  def _timestamp(self, timestamp_addr): to_mv(timestamp_addr, 8).cast('Q')[0] = time.perf_counter_ns()

  def cmd(self, cmd, *args):
    self.q(cmd, len(args), *args)
    return self

  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    return self.cmd(self._exec, prg, *global_size, *local_size, *[x.va_addr for x in args_state.bufs], *[x for x in args_state.vals])
  def wait(self, signal, value=0): return self.cmd(self._wait, signal.value_addr, value)
  def timestamp(self, signal): return self.cmd(self._timestamp, signal.timestamp_addr)
  def signal(self, signal, value:sint=0): return self.cmd(self._signal, signal.value_addr, value)

  def _submit(self, dev):
    off = 0
    while off < len(self._q):
      self._q[off](*self._q[off + 2:off + 2 + self._q[off + 1]])
      off += self._q[off + 1] + 2
    return self

class HCQCPUProgram(HCQProgram):
  def __init__(self, dev:CPUDevice, name:str, lib:bytes):
    self.dev, self.name = dev, name

    if sys.platform == "win32":
      PAGE_EXECUTE_READWRITE = 0x40
      MEM_COMMIT =  0x1000
      MEM_RESERVE = 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.mem = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.mem, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.mem), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.mem)
    else:
      from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap(-1, len(lib), MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), PROT_READ | PROT_WRITE | PROT_EXEC)

      if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(False)
      self.mem.write(lib)
      if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))

      self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))
    
    super().__init__(HCQArgsState, self.dev, self.name, kernargs_alloc_size=0)

class HCQCPUAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    buf = MallocAllocator._alloc(size, options)
    return HCQBuffer(va:=ctypes.addressof(buf), sz:=ctypes.sizeof(buf), meta=buf, view=MMIOInterface(va, sz, fmt='B'))
  def _as_buffer(self, src) -> memoryview: return to_mv(src.va_addr, src.size)
  def _as_dmaref(self, buf): return DMACPURef(ctypes.addressof(buf), ctypes.sizeof(buf))
  def _copyin(self, dest, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src.va_addr, len(dest))
  def _offset(self, buf, size:int, offset:int): return from_mv(self._as_buffer(buf)[offset:offset+size])

class CPUDevice(HCQCompiled):
  devices: ClassVar[list[HCQCompiled]] = []
  signal_pages: ClassVar[list[HCQBuffer]] = []
  signal_pool: ClassVar[list[HCQBuffer]] = []

  def __init__(self, device:str):
    super().__init__(device, HCQCPUAllocator(self), ClangRenderer(), ClangJITCompiler(), functools.partial(HCQCPUProgram, self), CPUSignal, CPUComputeQueue, None)
