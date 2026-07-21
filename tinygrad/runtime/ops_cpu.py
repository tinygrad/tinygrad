from __future__ import annotations
import platform, sys, os, ctypes, functools, mmap, threading, array
from tinygrad.helpers import to_mv, OSX, WIN, mv_address, suppress_finalizing, unwrap, data64_le
from tinygrad.device import BufferSpec
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
from tinygrad.runtime.support.hcq import CLikeArgsState
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from tinygrad.codegen import to_program
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import sint, Ops, AxisType, KernelInfo

MAX_ARGS, CMD_SIZE, RING_SLOTS = 16, 17, 8192

def cpu_program(f):
  @functools.cache
  def wrapped(dev):
    prg = to_program(f().sink(arg=KernelInfo(f.__name__), tag=1), ClangRenderer(dev.renderer.target))
    return CPUProgram(dev, prg.arg.function_name, next(x.arg for x in prg.src if x.op is Ops.BINARY), native=True)
  return wrapped

@cpu_program
def signal_prog():
  value = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return UOp.param(0, dtypes.uint32, (1,)).index(UOp.const(dtypes.int, 0)).store(value.cast(dtypes.uint32))

@cpu_program
def wait_prog():
  value = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return (UOp.param(0, dtypes.uint32, (1,)).index(UOp.const(dtypes.int, 0)).load(arg="volatile") >= value.cast(dtypes.uint32)).wait()

@cpu_program
def timestamp_prog():
  out, fn = UOp.param(0, dtypes.uint64, (1,)), UOp.param(1, dtypes.uint64, (1,))
  if WIN: return out.index(UOp.const(dtypes.int, 0)).store(UOp.const(dtypes.uint64, 0))
  ts = UOp.placeholder((2,), dtypes.uint64, slot=0, addrspace=AddrSpace.REG)
  call = UOp(Ops.CALL, src=(fn.index(UOp.const(dtypes.int, 0)).load(), UOp.const(dtypes.int, 1), ts.index(UOp.const(dtypes.int, 0))))
  val = ts.after(call).index(UOp.const(dtypes.int, 0)).load() * UOp.const(dtypes.uint64, 1_000_000_000) + \
        ts.after(call).index(UOp.const(dtypes.int, 1)).load()
  return out.index(UOp.const(dtypes.int, 0)).store(val)

@cpu_program
def quit_prog():
  fn = UOp.param(0, dtypes.uint64, (1 if WIN else 3,))
  if WIN: return UOp(Ops.CALL, src=(fn.index(UOp.const(dtypes.int, 0)).load(), UOp.const(dtypes.uint64, 0)))
  sem = UOp.param(1, dtypes.uint64, (1,))
  close = UOp(Ops.CALL, src=(fn.index(UOp.const(dtypes.int, 2)).load(), sem.index(UOp.const(dtypes.int, 0))))
  return UOp(Ops.CALL, src=(fn.after(close).index(UOp.const(dtypes.int, 0)).load(), UOp.const(dtypes.uint64, 0)))

@cpu_program
def worker_prog():
  ring, wait, sem = UOp.param(0, dtypes.uint64, (RING_SLOTS * CMD_SIZE,)), UOp.param(1, dtypes.uint64, (1,)), UOp.param(2, dtypes.uint64, (1,))
  loop = UOp.range(2**31-1, 1, AxisType.LOOP, dtype=dtypes.int)
  ready = ((wait.after(loop).index(UOp.const(dtypes.int, 0)).load(arg="volatile") > loop.cast(dtypes.uint64)).wait() if WIN else
           UOp(Ops.CALL, src=(wait.after(loop).index(UOp.const(dtypes.int, 0)).load(arg="volatile"),
                              sem.after(loop).index(UOp.const(dtypes.int, 0))))).barrier()
  base = (loop.cast(dtypes.uint64) % UOp.const(dtypes.uint64, RING_SLOTS)) * UOp.const(dtypes.uint64, CMD_SIZE)
  entry = [ring.after(ready).index(base + UOp.const(dtypes.uint64, i)).load(arg="volatile") for i in range(CMD_SIZE)]
  return UOp(Ops.CALL, src=tuple(entry)).end(loop)

class CPUComputeQueue(HWQueue):
  def __init__(self, dev):
    super().__init__()
    self.dev = dev
  def _cmd(self, prog, args=(), vals=()): return self.exec(prg:=prog(self.dev), prg.fill_kernargs(args, vals), None, None)
  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    if (lvp:=isinstance(args_state, LVPArgsState)): self.bind_args_state(args_state)

    args:list[sint|None] = [args_state.buf.va_addr] if lvp else [*[x.va_addr for x in args_state.bufs], *args_state.vals]
    assert len(args) <= MAX_ARGS, f"CPU programs support at most {MAX_ARGS} arguments"

    for tid in range(1 if lvp else (global_size or (1,))[0]):
      if not lvp and 'core_id' in prg.runtimevars: args[len(args_state.bufs)+prg.runtimevars['core_id']] = tid
      self.q(prg, *[unwrap(x) for x in args], *([0] * (MAX_ARGS - len(args))))
    return self
  def wait(self, signal, value=0): return self._cmd(wait_prog, (signal.base_buf,), (value,))
  def timestamp(self, signal): return self._cmd(timestamp_prog, (signal.base_buf.offset(8, 8), self.dev.func_table.offset(0, 8)))
  def signal(self, signal, value:sint=0): return self._cmd(signal_prog, (signal.base_buf,), (value,))
  def _submit(self, dev):
    for off in range(0, len(self._q), CMD_SIZE):
      entry = [self._q[off].addr, *self._q[off+1:off+CMD_SIZE]]
      dev.ring_view[(base:=(dev.ring_pos % RING_SLOTS) * CMD_SIZE):base+CMD_SIZE] = array.array('Q', (int(x) & ((1<<64)-1) for x in entry))
      dev.ring_pos += 1
      if WIN: dev.sys_view[0] = dev.ring_pos
      else: assert libc.sem_post(dev.sem) == 0

class LVPArgsState(CLikeArgsState):
  def __init__(self, buf, prg, bufs, vals=()): super().__init__(buf, prg, bufs, vals, [*data64_le(buf.va_addr + 12), (len(bufs) + len(vals)) * 2])

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(HCQProgram):
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev, name:str, lib:bytes, runtimevars:dict[str, int]|None=None, native=False, **kwargs):
    self.runtimevars = runtimevars or {}

    LVP = isinstance(dev.renderer, LVPRenderer) and not native
    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.mem = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.mem, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.mem), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.mem)
      self.addr = self.mem
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      if LVP: lib = jit_loader(lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m'])
      self.mem.write(lib)
      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if CPUProgram.rt_lib is not None:
        CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))
      else:
        # msync should be a universal POSIX way to do this
        from tinygrad.runtime.autogen import libc
        libc.msync(ctypes.c_void_p(mv_address(self.mem)), len(lib), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))
      self.addr = mv_address(self.mem)

    super().__init__(LVPArgsState if LVP else HCQArgsState, dev, name, kernargs_alloc_size=12+256 if LVP else 0)

  @suppress_finalizing
  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.mem), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class CPUAllocator(HCQAllocator):
  def __init__(self, dev:CPUDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: addr, buf = options.external_ptr, None
    elif WIN: addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _as_buffer(self, src) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.va_addr, src.size)
  def _do_map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")
    return HCQBuffer(buf.view.addr, buf.size, view=buf.view, owner=buf.owner)
  def _unmap(self, mb): pass  # CPU _do_map returns a view wrapper, nothing to release

class CPUDevice(HCQCompiled):
  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], functools.partial(CPUProgram, self),
      HCQSignal, functools.partial(CPUComputeQueue, self), arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.ring = self.allocator.alloc(RING_SLOTS * CMD_SIZE * 8, BufferSpec())
    self.ring_view, self.ring_pos = self.ring.cpu_view().view(fmt='Q'), 0

    if WIN:
      self.sys = self.allocator.alloc(8, BufferSpec())
      self.sys_view, sem_addr = self.sys.cpu_view().view(fmt='Q'), 0
      self.sys_view[0] = 0
    else:
      self.sem = libc.sem_open(sem_name:=f"/tinygrad-{os.getpid()}-{id(self):x}".encode(), os.O_CREAT|os.O_EXCL, 0o600, 0) # type: ignore[call-arg]
      if (sem_addr:=unwrap(ctypes.cast(self.sem, ctypes.c_void_p).value)) == ctypes.c_void_p(-1).value or libc.sem_unlink(sem_name):
        raise OSError(ctypes.get_errno(), "semaphore")
      self.sem_buf = HCQBuffer(sem_addr, 1, owner=self)

    # TODO: move to hcq2 infra
    self.func_table = self.allocator.alloc(32, BufferSpec())
    self.func_table.cpu_view().view(fmt='Q')[:] = array.array('Q', (0 if WIN else unwrap(ctypes.cast(libc.dll.clock_gettime, ctypes.c_void_p).value),
      unwrap(ctypes.cast(ctypes.windll.kernel32.ExitThread if WIN else libc.dll.pthread_exit, ctypes.c_void_p).value),  # type: ignore[attr-defined]
      0 if WIN else unwrap(ctypes.cast(libc.dll.sem_wait, ctypes.c_void_p).value),
      0 if WIN else unwrap(ctypes.cast(libc.dll.sem_close, ctypes.c_void_p).value)))

    self.worker = threading.Thread(target=worker_prog(self).fxn, args=(ctypes.c_uint64(self.ring.va_addr),
      ctypes.c_uint64(self.sys.va_addr if WIN else self.func_table.va_addr+16), ctypes.c_uint64(sem_addr)), daemon=True)
    self.worker.start()

  def finalize(self):
    CPUComputeQueue(self)._cmd(quit_prog, (self.func_table.offset(8, 8),) if WIN else (self.func_table.offset(8, 24), self.sem_buf)).submit(self)
