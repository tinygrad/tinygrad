from __future__ import annotations
import platform, sys, ctypes, functools, mmap, threading, array
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
  fn = UOp.param(0, dtypes.uint64, (1,))
  return UOp(Ops.CALL, src=(fn.index(UOp.const(dtypes.int, 0)).load(), UOp.const(dtypes.uint64, 0)))

@cpu_program
def worker_prog():
  state, ring = UOp.param(0, dtypes.uint64, (1,)), UOp.param(1, dtypes.uint64, (RING_SLOTS * CMD_SIZE,))
  head = UOp.placeholder((1,), dtypes.uint64, slot=0, addrspace=AddrSpace.REG)[0].set(0)
  loop = UOp.range(2**31-1, 1, AxisType.LOOP, dtype=dtypes.int)
  head_i, state_i = head.after(loop), state.after(loop)
  head_val = head_i[0].load(arg="volatile")
  ready = (state_i.index(UOp.const(dtypes.int, 0)).load(arg="volatile") > head_val).wait().barrier()
  base = (head_val % UOp.const(dtypes.uint64, RING_SLOTS)) * UOp.const(dtypes.uint64, CMD_SIZE)
  entry = [ring.after(ready).index(base + UOp.const(dtypes.uint64, i)).load(arg="volatile") for i in range(CMD_SIZE)]
  call = UOp(Ops.CALL, src=tuple(entry))
  return head.after(call)[0].store(head_val + UOp.const(dtypes.uint64, 1)).end(loop)

class CPUComputeQueue(HWQueue):
  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    if (lvp:=isinstance(args_state, LVPArgsState)): self.bind_args_state(args_state)

    args = [args_state.buf.va_addr] if lvp else [*[x.va_addr for x in args_state.bufs], *args_state.vals]
    assert len(args) <= MAX_ARGS, f"CPU programs support at most {MAX_ARGS} arguments"

    for tid in range(1 if lvp else (global_size or (1,))[0]):
      if not lvp and 'core_id' in prg.runtimevars: args[len(args_state.bufs)+prg.runtimevars['core_id']] = tid
      self.q(prg, *[unwrap(x) for x in args], *([0] * (MAX_ARGS - len(args))))
    return self
  def wait(self, signal, value=0):
    return self.exec(p:=wait_prog(unwrap(signal.owner)), p.fill_kernargs((signal.base_buf,), (value,)), None, None)
  def timestamp(self, signal):
    return self.exec(p:=timestamp_prog(unwrap(signal.owner)),
                     p.fill_kernargs((signal.base_buf.offset(8, 8), unwrap(signal.owner).fns.offset(0, 8))), None, None)
  def signal(self, signal, value:sint=0):
    return self.exec(p:=signal_prog(unwrap(signal.owner)), p.fill_kernargs((signal.base_buf,), (value,)), None, None)
  def _submit(self, dev):
    for off in range(0, len(self._q), CMD_SIZE):
      prg = self._q[off]
      entry = [prg.addr, *self._q[off+1:off+CMD_SIZE]]
      base = (dev.sys_view[0] % RING_SLOTS) * CMD_SIZE
      dev.ring_view[base:base+CMD_SIZE] = array.array('Q', (int(x) & ((1<<64)-1) for x in entry))
      dev.sys_view[0] += 1

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
                     HCQSignal, CPUComputeQueue, arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")
    worker = worker_prog(self)
    for p in (signal_prog, wait_prog, timestamp_prog, quit_prog): p(self)
    self.sys = self.allocator.alloc(8, BufferSpec())
    self.ring = self.allocator.alloc(RING_SLOTS * CMD_SIZE * 8, BufferSpec())
    self.fns = self.allocator.alloc(16, BufferSpec())
    self.sys_view, self.ring_view = self.sys.cpu_view().view(fmt='Q'), self.ring.cpu_view().view(fmt='Q')
    self.sys_view[0] = 0
    exit_fn = ctypes.windll.kernel32.ExitThread if sys.platform == "win32" else libc.dll.pthread_exit  # type: ignore[attr-defined]
    self.fns.cpu_view().view(fmt='Q')[:] = array.array('Q', (0 if WIN else unwrap(ctypes.cast(libc.dll.clock_gettime, ctypes.c_void_p).value),
                                                            unwrap(ctypes.cast(exit_fn, ctypes.c_void_p).value)))
    self.worker = threading.Thread(target=worker.fxn, args=(ctypes.c_uint64(self.sys.va_addr), ctypes.c_uint64(self.ring.va_addr)), daemon=True)
    self.worker.start()

  def finalize(self): CPUComputeQueue().exec(p:=quit_prog(self), p.fill_kernargs((self.fns.offset(8, 8),)), None, None).submit(self)
