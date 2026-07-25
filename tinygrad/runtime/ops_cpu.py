from __future__ import annotations
import platform, sys, os, ctypes, functools, mmap, threading, array, struct, time
from typing import Callable, cast
from tinygrad.helpers import from_mv, OSX, WIN, Context, mv_address, suppress_finalizing, to_tuple, unwrap
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.runtime.support.hcq2 import HCQ2Compiled, HCQAllocator, make_cmdbuf
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from tinygrad.codegen import do_to_program
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UPat, PatternMatcher, KernelInfo, graph_rewrite

CMD_SIZE, RING_SLOTS = 32, (16 << 10)
CPU_BUFS = ("ring", "put_value", "func_table") # buffers a cmd or a submit refers to. the tag is the attribute name on the device
FUNCS = ("clock_gettime", "pthread_exit", "sem_wait", "sem_close") # libc entrypoints the progs call, by index into func_table

# *****************
# progs: the queue's instruction set, each compiled into a buffer that the cmds point at

def signal_prog():
  val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return UOp.param(0, dtypes.uint32, (1,))[0].store(val.cast(dtypes.uint32))

def wait_prog():
  val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return (v:=UOp.param(0, dtypes.uint32, (1,), volatile=True).after(l:=UOp.loop(0))[0].load()).end(l, v < val.cast(dtypes.uint32))

def timestamp_prog():
  if WIN: val = UOp.const(dtypes.uint64, 0)
  else:
    fn, ts = UOp.param(1, dtypes.uint64, (1,)), UOp.placeholder((2,), dtypes.uint64, slot=0, addrspace=AddrSpace.REG)
    call = fn[0].load().call(UOp.const(dtypes.int, 6 if OSX else 1), ts[0], ret_dtype=dtypes.void) # clock_gettime(CLOCK_MONOTONIC, &ts)
    val = ts.after(call)[0].load() * 1_000_000_000 + ts.after(call)[1].load()
  return UOp.param(0, dtypes.uint64, (1,))[0].store(val)

def quit_prog():
  fn = UOp.param(0, dtypes.uint64, (1 if WIN else 3,))
  if WIN: return fn[0].load().call(UOp.const(dtypes.uint64, 0), ret_dtype=dtypes.void) # ExitThread(0)
  sem = UOp.param(1, dtypes.uint64, (1,))

  close = fn[2].load().call(sem[0], ret_dtype=dtypes.void) # sem_close(sem)
  return fn.after(close)[0].load().call(UOp.const(dtypes.uint64, 0), ret_dtype=dtypes.void) # pthread_exit(0)

def worker_prog():
  ring = UOp.param(0, dtypes.uint64, (RING_SLOTS * CMD_SIZE,), volatile=True)
  wait, sem = UOp.param(1, dtypes.uint64, (1,), volatile=True), UOp.param(2, dtypes.uint64, (1,))
  cur = UOp.range(2**64-1, 0, dtype=dtypes.uint64)

  # spin on windows, sem_wait to sleep on posix
  if WIN: ready = (v:=wait.after(lw:=UOp.loop(1), cur)[0].load()).end(lw, v <= cur)
  else: ready = wait.after(cur)[0].load().call(sem.after(cur)[0], ret_dtype=dtypes.void)

  entry = [ring.after(ready).index((cur % RING_SLOTS) * CMD_SIZE + i).load() for i in range(CMD_SIZE)]
  return entry[0].call(*entry[1:], ret_dtype=dtypes.void).end(cur)

# *****************

# one ring entry: the address of the code to run, its args, then zero padding
def cpu_cmd(name:str, code:UOp, *args:UOp) -> UOp:
  assert len(args) < CMD_SIZE, f"a cmd fits {CMD_SIZE-1} args, {name} wants {len(args)}"
  words = (code,) + tuple(a.cast(dtypes.uint64) for a in args)
  return UOp(Ops.INS, arg=name, src=words + (UOp.const(dtypes.uint64, 0),) * (CMD_SIZE - len(words)))

# a prebuilt prog is a buffer, so its cmd refers to it by tag and the address is resolved at link
def prog_cmd(devs:tuple[str, ...], prog:Callable, *args:UOp) -> UOp:
  return cpu_cmd(prog.__name__, UOp.placeholder((1,), dtypes.uint64, 0, device=devs).rtag(prog.__name__).getaddr(devs), *args)

def cpu_wait(ctx, dst, val): return prog_cmd(ctx, wait_prog, dst.getaddr(ctx), val)
def cpu_store(ctx, dst, val): return prog_cmd(ctx, signal_prog, dst.getaddr(ctx), val)
def cpu_timestamp(ctx, dst):
  return prog_cmd(ctx, timestamp_prog, dst.getaddr(ctx), UOp.placeholder((len(FUNCS),), dtypes.uint64, 0, device=ctx).rtag("func_table").getaddr(ctx))

_kernels:dict[bytes, CPUProgram] = {}
def cpu_program(ctx, call, prg):
  if (fxn:=_kernels.get(prg.key)) is None:
    fxn = _kernels[prg.key] = CPUProgram(cast(CPUDevice, Device[ctx[0]]), prg.arg.function_name, prg.src[3].arg,
                                        runtimevars=prg.arg.runtimevars)
  args = [get_call_arg_uops(call)[gi].getaddr(ctx) for gi in prg.arg.globals] + list(prg.arg.vars)
  return cpu_cmd(prg.arg.function_name, UOp.const(dtypes.uint64, fxn.addr), *args)

pm_cpu_opsel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), cpu_program),
  (UPat(Ops.INS, arg="barrier"), lambda: UOp(Ops.NOOP, dtypes.void, ())),
  (UPat(Ops.INS, arg="wait", src=(UPat(name="dst"), UPat(name="val"))), cpu_wait),
  (UPat(Ops.INS, arg="store", src=(UPat(name="dst"), UPat(name="val"))), cpu_store),
  (UPat(Ops.INS, arg="timestamp", src=(UPat(name="dst"),)), cpu_timestamp),
])

def encode_queue(q:UOp) -> UOp:
  cmds = graph_rewrite(q, pm_cpu_opsel, ctx=(devs:=to_tuple(q.arg[0])), walk=True, name=f"{q.arg[1]} opsel")
  cmdbuf = make_cmdbuf(cmds, devs, dtypes.uint64)
  zero, entries = UOp.const(dtypes.int, 0), cmdbuf.nbytes() // (CMD_SIZE * dtypes.uint64.itemsize)
  ring, put = (UOp.placeholder((sz,), dtypes.uint64, 0, device=devs).rtag(name) for name, sz in (("ring", RING_SLOTS * CMD_SIZE), ("put_value", 1)))

  i = UOp.range(entries * CMD_SIZE, 0, dtype=dtypes.int, src=(cmdbuf,))
  ring_idx = (((put_b:=put.index(zero)) * CMD_SIZE + i.cast(dtypes.uint64)) % (RING_SLOTS * CMD_SIZE)).cast(dtypes.int)
  copy = ring.index(ring_idx).store(cmdbuf.index(i).load()).end(i)
  return put.after(copy).index(zero, dtype=put.dtype).store(put_b + entries)

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram:
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev:CPUDevice, name:str, lib:bytes, runtimevars:dict[str, int]|None=None, native=False, **kwargs):
    self.dev, self.name, self.runtimevars = dev, name, runtimevars or {}

    self.lvp = isinstance(dev.renderer, LVPRenderer) and not native
    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.addr = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.addr, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.addr), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.addr)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
      self.addr = mv_address(self.mem)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      if self.lvp: lib = jit_loader(lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m'])
      self.mem.write(lib)
      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if CPUProgram.rt_lib is not None: CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(self.addr), ctypes.c_void_p(self.addr + len(lib)))
      else:
        # msync should be a universal POSIX way to do this
        libc.msync(ctypes.c_void_p(self.addr), len(lib), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None)(self.addr)

  @suppress_finalizing
  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

  # lvp takes one pointer to a blob of (blob address, arg count, buf addrs, vals)
  def lvp_kernargs(self, bufs:tuple[HCQBuffer, ...], vals:tuple[int|None, ...]) -> int:
    kb = self.dev.rt_buffer.ensure_allocated()._buf.offset(self.dev.rt_allocator.alloc(sz:=12 + len(bufs) * 8 + len(vals) * 4, 8), sz)
    blob = struct.pack(f'<QI{len(bufs)}Q{len(vals)}I', kb.va_addr + 12, (len(bufs) + len(vals)) * 2, *[b.va_addr for b in bufs], *vals)
    kb.cpu_view().view(size=len(blob), fmt='B')[:] = blob
    return kb.va_addr

  def __call__(self, *bufs:HCQBuffer, global_size:tuple[int, ...]|None=None, local_size:tuple[int, ...]|None=None,
               vals:tuple[int|None, ...]=(), wait:bool=False, timeout:int|None=None) -> float|None:
    args:list = [self.lvp_kernargs(bufs, vals)] if self.lvp else [*[b.va_addr for b in bufs], *vals]
    st = time.perf_counter()
    for tid in range(1 if self.lvp else (global_size or (1,))[0]):
      if not self.lvp and 'core_id' in self.runtimevars: args[len(bufs)+self.runtimevars['core_id']] = tid
      self.fxn(*[ctypes.c_uint64(unwrap(x)) for x in args])
    if not wait: return None
    self.dev.synchronize(timeout)
    return time.perf_counter() - st

class CPUAllocator(HCQAllocator['CPUDevice']):
  def __init__(self, dev:CPUDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: addr, buf = options.external_ptr, None
    elif WIN: addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _free(self, buf:HCQBuffer, options:BufferSpec|None=None):
    if options is None or options.external_ptr is None: [d.synchronize() for d in buf.mapped_devs] # a device it's mapped into may still be reading
  def _memmove(self, dest, src, sz:int):
    self.dev.synchronize() # the worker may still be running programs on either side
    ctypes.memmove(dest, src, sz)
  def _copyin(self, dest:HCQBuffer, src:memoryview): self._memmove(dest.va_addr, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src:HCQBuffer): self._memmove(from_mv(dest), src.va_addr, len(dest))
  def _do_map(self, buf:HCQBuffer) -> HCQBuffer: # just wraps the view, so nothing is registered and there is nothing to release
    if type(buf.view) is not MMIOInterface: raise RuntimeError("Cannot map buffer without view to cpu")
    return HCQBuffer(buf.view.addr, buf.size, view=buf.view, owner=buf.owner)
  def _unmap(self, mb): pass

class CPUDevice(HCQ2Compiled):
  pm_lower = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),)), encode_queue)])

  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], functools.partial(CPUProgram, self),
      arch={'amd64':'x86_64','aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, tag=t), lambda ctx, t=t: getattr(ctx[0], t)) for t in CPU_BUFS] + [
      (UPat(Ops.PARAM, tag="COMPUTE:0_timeline_signal"), lambda ctx: ctx[0].timeline_signal()),
      (UPat(Ops.PARAM, tag="COMPUTE:0_timeline_value"), lambda ctx: ctx[0].timeline_value()),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx[0].prog_buf(b.tag) if b.tag in ctx[0].prgs else None),
    ]) + self.pm_bufferize

    # posix uses sem to put cpus into sleep
    self.sem_addr = self.posted = 0
    if not WIN:
      self.sem = libc.sem_open(sem_name:=f"/tinygrad-{os.getpid()}-{id(self):x}".encode(), os.O_CREAT|os.O_EXCL, 0o600, 0) # type: ignore[call-arg]
      self.sem_addr = unwrap(ctypes.cast(self.sem, ctypes.c_void_p).value)
      if self.sem_addr == ctypes.c_void_p(-1).value or libc.sem_unlink(sem_name): raise OSError(ctypes.get_errno(), "semaphore")

    with Context(EMULATED_DTYPES="", TRACK_MATCH_STATS=0):
      prgs = {f.__name__: f().sink(arg=KernelInfo(f.__name__), tag=1) for f in (signal_prog, wait_prog, timestamp_prog, quit_prog, worker_prog)}
      self.prgs = {n: self.runtime(n, do_to_program(v, ClangRenderer(self.renderer.target)).src[3].arg, native=True) for n,v in prgs.items()}

  @functools.cached_property
  def ring(self) -> Buffer: return Buffer(self.device, RING_SLOTS * CMD_SIZE, dtypes.uint64, preallocate=True)

  @functools.cached_property
  def put_value(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint64, preallocate=True)

  @functools.cached_property
  def doorbell(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint64, preallocate=True) # windows has no sem, the worker spins on this counter

  @functools.cache
  def prog_buf(self, tag:str) -> Buffer: # a compiled prog is just a buffer, the cmd carries its address
    return Buffer(self.device, 1, dtypes.uint8, options=BufferSpec(external_ptr=self.prgs[tag].addr), preallocate=True)

  @functools.cached_property
  def func_table(self) -> Buffer:
    exitthread = ctypes.windll.kernel32.ExitThread if WIN else 0 # type: ignore[attr-defined]
    fns = [exitthread if f == "pthread_exit" else 0 for f in FUNCS] if WIN else [getattr(libc.dll, f) for f in FUNCS]
    addrs = array.array('Q', [unwrap(ctypes.cast(f, ctypes.c_void_p).value) if f else 0 for f in fns])
    (ft:=Buffer(self.device, len(FUNCS), dtypes.uint64, preallocate=True)).as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[:] = addrs
    return ft

  def func_addr(self, name:str) -> int: return self.func_table._buf.va_addr + 8 * FUNCS.index(name)

  @functools.cache
  def ensure_worker(self):
    args = [self.ring._buf.va_addr, self.doorbell._buf.va_addr if WIN else self.func_addr("sem_wait"), self.sem_addr]
    threading.Thread(target=self.prgs["worker_prog"].fxn, daemon=True, args=[ctypes.c_uint64(x) for x in args]).start()

  # tell the worker how many cmds are waiting: one sem_post each on posix, the running count on windows
  def ring_doorbell(self):
    self.ensure_worker() # there are cmds in the ring now, so it needs a consumer
    put = int(self.put_value.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0])
    if WIN: self.doorbell.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = put
    else:
      for _ in range(put - self.posted): assert libc.sem_post(self.sem) == 0
    self.posted = put

  # write one cmd into the ring by hand, which is what a submit does in C. only the quit cmd needs it
  def push_cmd(self, *entry:int):
    ring, put = (b.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q') for b in (self.ring, self.put_value))
    ring[(base:=(int(put[0]) % RING_SLOTS) * CMD_SIZE):base+CMD_SIZE] = array.array('Q', [*entry] + [0] * (CMD_SIZE - len(entry)))
    put[0] += 1
    self.ring_doorbell()

  def finalize(self):
    super().finalize()
    if not self.posted: return # nothing was ever queued, so there is no worker to stop
    self.push_cmd(self.prgs["quit_prog"].addr, self.func_addr("pthread_exit"), *(() if WIN else (self.sem_addr,)))
