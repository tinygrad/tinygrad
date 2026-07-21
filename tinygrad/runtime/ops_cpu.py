from __future__ import annotations
import platform, sys, ctypes, enum, functools, mmap, threading, array
from tinygrad.helpers import to_mv, OSX, WIN, mv_address, suppress_finalizing, unwrap, data64_le, Context
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
from tinygrad.runtime.support.hcq import CLikeArgsState
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from extra.hcq2.hcq2 import make_ext_call, make_placeholder, sym_addr, hcq_compile_program, hcq_link, unwrap_after
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import sint, Ops, AxisType, KernelInfo, PatternMatcher, UPat

class CpuOp(enum.IntEnum): SIGNAL, WAIT, TIMESTAMP, EXEC, QUIT = range(1, 6)

def read_next(ring, head): return ring.index(head % ring.max_numel()).load(), head + UOp.const(dtypes.uint64, 1)

def make_null_buf(dtype): return make_placeholder("CPU", (1<<64) // dtype.itemsize, dtype, name="null_buf", unique=False)

def op_signal(gate, a):
  idx = (a[0] >> UOp.const(dtypes.uint64, 2)).valid(gate)
  return make_null_buf(dtypes.uint32).index(idx).store(a[1].cast(dtypes.uint32)).barrier()

def op_wait(gate, a):
  idx = (a[0] >> UOp.const(dtypes.uint64, 2)).valid(gate)
  sig = make_null_buf(dtypes.uint32).index(idx).load(arg="volatile")
  return ((~gate) | (sig >= a[1].cast(dtypes.uint32))).wait()

def op_timestamp(gate, a):
  ts = UOp.placeholder((2,), dtypes.uint64, slot=4, addrspace=AddrSpace.REG)  # struct timespec { long s; long ns; }
  call = make_ext_call(libc.dll.clock_gettime, UOp.const(dtypes.int32, 1), ts.index(UOp.const(dtypes.int, 0)), gate=gate)
  val = (ts.after(call).index(UOp.const(dtypes.int, 0).valid(gate)).load() * UOp.const(dtypes.uint64, 1_000_000_000) + \
         ts.after(call).index(UOp.const(dtypes.int, 1).valid(gate)).load()) if not WIN else UOp.const(dtypes.uint64, 0)
  return make_null_buf(dtypes.uint64).index((a[0] >> UOp.const(dtypes.uint64, 3)).valid(gate)).store(val)

def op_exec(gate, a): return UOp(Ops.CALL, src=(*a, gate))

def op_quit(gate, a):
  fn = getattr(getattr(ctypes, "windll").kernel32, "ExitThread") if WIN else getattr(libc.dll, "pthread_exit")
  return make_ext_call(fn, UOp.const(dtypes.uint32 if WIN else dtypes.uint64, 0), gate=gate)

handlers = {CpuOp.SIGNAL: op_signal, CpuOp.WAIT: op_wait, CpuOp.TIMESTAMP: op_timestamp, CpuOp.EXEC: op_exec, CpuOp.QUIT: op_quit}

def cpu_worker(ring_size=32768, args_cnt=3):
  tail = make_placeholder("CPU", 1, dtypes.uint64, name="sys", unique=False)
  ring = make_placeholder("CPU", ring_size, dtypes.uint64, name="ring", unique=False)
  head = UOp.placeholder((1,), dtypes.uint64, slot=3, addrspace=AddrSpace.REG)[0].set(0)

  loop = UOp.range(2**31-1, 1, AxisType.LOOP, dtype=dtypes.int)
  head_i, tail_i = head.after(loop), tail.after(loop)

  head_val = head_i[0].load()
  barrier = UOp(Ops.WAIT, src=(tail_i.index(UOp.const(dtypes.int, 0)).load(arg="volatile") > head_val,)).barrier()

  ring_b = ring.after(barrier)
  op, head_val = read_next(ring_b, head_val)
  args = [None] * args_cnt
  for i in range(args_cnt): args[i], head_val = read_next(ring_b, head_val)

  arms = [fn(op.eq(cmd), args) for cmd, fn in handlers.items()]

  advance = head.after(*(x.barrier() for x in arms))[0].store(head_val)
  return advance.barrier().end(loop).sink(arg=KernelInfo(name="cpu_worker"), tag=1)

def build_worker(ring_size:int, renderer) -> tuple[UOp, UOp]:
  worker_renderer = renderer if isinstance(renderer, (ClangRenderer, CPULLVMRenderer)) else ClangRenderer(renderer.target)
  return hcq_compile_program(cpu_worker(ring_size), worker_renderer, "cpu_worker")

def cpu_ext_buf(addr:int) -> Buffer: return Buffer("CPU", 1, dtypes.uint8, options=BufferSpec(external_ptr=addr), preallocate=True)
def cpu_sym_buf(b:UOp) -> Buffer|None:
  if isinstance(b.tag, tuple) and b.tag[0] == "sym": return cpu_ext_buf(sym_addr(b.tag[1]))  # dlsym'd C symbol address
  return None if b.tag is None else Buffer("CPU", b.max_numel(), b.dtype, preallocate=True)

class CPUComputeQueue(HWQueue):
  def _cmd(self, *entry):
    self.q(*(entry + (0, 0, 0))[:4])
    return self
  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    self.bind_args_state(args_state)
    fn_ptr = ctypes.cast(prg.fxn, ctypes.c_void_p).value
    # kernels split by core_id: one entry per tid so the worker calls fn N times with tid injected
    for tid in range((global_size or (1,))[0]): self._cmd(CpuOp.EXEC, fn_ptr, args_state.buf.va_addr, tid)
    return self
  def wait(self, signal, value=0): return self._cmd(CpuOp.WAIT, signal.value_addr, value)
  def timestamp(self, signal): return self._cmd(CpuOp.TIMESTAMP, signal.timestamp_addr)
  def signal(self, signal, value:sint=0): return self._cmd(CpuOp.SIGNAL, signal.value_addr, value)
  def _submit(self, dev):
    dev._ensureworkers(1)
    for off in range(0, len(self._q), 4):  # append each fixed 4-u64 entry, then release-publish the new tail via sys[0]
      base = dev.sys_view[0] % dev.RING_SZ
      dev.ring_view[base:base+4] = array.array('Q', self._q[off:off+4])
      dev.sys_view[0] += 4

class LVPArgsState(CLikeArgsState):
  def __init__(self, buf, prg, bufs, vals=()): super().__init__(buf, prg, bufs, vals, [*data64_le(buf.va_addr + 12), (len(bufs) + len(vals)) * 2])

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(HCQProgram):
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev, name:str, lib:bytes, runtimevars:tuple[UOp, ...]|None=None, **kwargs):
    self.runtimevars:tuple[UOp, ...] = runtimevars or ()

    LVP = isinstance(dev.renderer, LVPRenderer)
    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.mem = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.mem, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.mem), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.mem)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      if LVP and lib.startswith(libc.ELFMAG.encode()):
        lib = jit_loader(lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m'])
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
        libc.msync(ctypes.c_void_p(mv_address(self.mem)), len(lib), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))

    super().__init__(LVPArgsState if LVP else CLikeArgsState, dev, name, kernargs_alloc_size=12+256 if LVP else 256)

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
  RING_SZ = 32768

  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], functools.partial(CPUProgram, self),
                     HCQSignal, CPUComputeQueue, arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="null_buf"), lambda: cpu_ext_buf(0)),  # base 0: byte-indexes absolute memory
      (UPat(Ops.PARAM, name="b"), cpu_sym_buf),
    ])
    self.worker_rt:CPUProgram|None = None
    self.workers:list[threading.Thread] = []

  # no worker spawned yet ⇒ nothing running ⇒ nothing to wait on. guards the re-entrant sync a buffer __del__ fires mid-build (the SIGNAL
  # that would satisfy it is queued behind this very build in _submit, so waiting here would deadlock).
  def synchronize(self, timeout:int|None=None): return super().synchronize(timeout) if self.workers else None

  def _ensureworkers(self, cnt:int):
    if self.worker_rt is None:
      with Context(TRACK_MATCH_STATS=0):
        call, prg = build_worker(self.RING_SZ, self.renderer)
        self._worker_bufs = [s.buffer for s in hcq_link(UOp(Ops.LINEAR, src=(call,))).src[0].src[1:]]
      bufs = dict(zip((unwrap_after(s).tag for s in call.src[1:]), self._worker_bufs))  # tag -> resolved buffer
      self.sys_view, self.ring_view = bufs["sys"]._buf.cpu_view().view(fmt='Q'), bufs["ring"]._buf.cpu_view().view(fmt='Q')
      self.sys_view[0] = 0  # tail starts empty (worker's head also starts at 0)

      self.worker_args = self.allocator.alloc(len(self._worker_bufs) * 8, BufferSpec())  # each arg's address, in PARAM slot order
      self.worker_args.cpu_view().view(fmt='Q')[:] = array.array('Q', [b._buf.va_addr for b in self._worker_bufs])
      self.worker_rt = self.runtime(prg.arg.function_name, next(s.arg for s in prg.src if s.op is Ops.BINARY))

    assert self.worker_rt is not None
    while len(self.workers) < cnt:
      self.workers.append(t:=threading.Thread(target=self.worker_rt.fxn, args=(ctypes.c_uint64(self.worker_args.va_addr),), daemon=True))
      t.start()

  def finalize(self):
    for _ in self.workers: CPUComputeQueue()._cmd(CpuOp.QUIT).submit(self)
    for t in self.workers: t.join(timeout=2.0)
