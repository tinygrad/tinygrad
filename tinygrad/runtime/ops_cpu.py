from __future__ import annotations
import platform, sys, os, ctypes, functools, mmap, threading, array, itertools, pathlib
from dataclasses import replace
from typing import cast
from tinygrad.helpers import to_mv, OSX, WIN, Context, mv_address, suppress_finalizing, unwrap, data64_le, partition, getenv
from tinygrad.device import Buffer, BufferSpec, TinyELF
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
from tinygrad.runtime.support.hcq import CLikeArgsState
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from tinygrad.codegen import do_to_program
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import sint, KernelInfo, Ops, UPat, PatternMatcher, graph_rewrite

MAX_ARGS, CMD_SIZE, RING_SLOTS = 32, 33, (16 << 10)
CPU_CORES = getenv("CPU_COUNT", max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)))
PARALLEL_WORKERS, PARALLEL_PARTICIPANTS = min(31, max(1, CPU_CORES-1)), min(32, max(2, CPU_CORES))

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

def post_many_prog():
  sem, post_fn = UOp.param(0, dtypes.uint64, (1,)), UOp.param(1, dtypes.uint64, (1,))
  count = UOp.param(2, dtypes.int, (), vmin_vmax=(1, RING_SLOTS), name="count", addrspace=AddrSpace.ALU)
  post = UOp.range(count, 0)
  return post_fn.after(post)[0].load().call(sem.after(post)[0], ret_dtype=dtypes.void).end(post)

def worker_prog():
  ring = UOp.param(0, dtypes.uint64, (RING_SLOTS * CMD_SIZE,), volatile=True)
  wait, sem = UOp.param(1, dtypes.uint64, (1,), volatile=True), UOp.param(2, dtypes.uint64, (1,))
  progress = UOp.param(3, dtypes.uint64, (1,), volatile=True)
  cur = UOp.range(2**64-1, 0, dtype=dtypes.uint64)

  # spin on windows, sem_wait to sleep on posix
  if WIN: ready = (v:=wait.after(lw:=UOp.loop(1), cur)[0].load()).end(lw, v <= cur)
  else: ready = wait.after(cur)[0].load().call(sem.after(cur)[0], ret_dtype=dtypes.void)

  entry = [ring.after(ready).index((cur % RING_SLOTS) * CMD_SIZE + i).load() for i in range(CMD_SIZE)]
  return progress.after(entry[0].call(*entry[1:], ret_dtype=dtypes.void), cur)[0].store(cur + 1).end(cur)

def parallel_wait_prog():
  generation = UOp.param(0, dtypes.uint64, (1,), volatile=True)
  completed = UOp.param(1, dtypes.uint64, (1,), volatile=True)
  ready = UOp.param(2, dtypes.uint64, (1,), volatile=True)
  wait_fn, sem = UOp.param(3, dtypes.uint64, (1,)), UOp.param(4, dtypes.uint64, (1,))
  relax = "__builtin_ia32_pause();" if platform.machine().lower() in ("x86_64", "amd64") else \
          '__asm__ __volatile__("yield");' if platform.machine().lower() in ("aarch64", "arm64") else ""
  value = UOp(Ops.CUSTOMI, dtypes.uint64, (generation.index(0), completed.index(0), wait_fn[0].load(), sem.index(0)), arg=
    "({{ unsigned long _seen = *((volatile unsigned long *){1}), _v; "
    "do {{ int _i = 0; do {{ _v = *((volatile unsigned long *){0}); if (_v > _seen) break; __RELAX__ }} while (++_i < __SPIN__); "
    "if (_v <= _seen) while (((int (*)(unsigned long)){2})((unsigned long){3}) != 0) {{}}; }} while (_v <= _seen); _v; }})"
    .replace("__RELAX__", relax).replace("__SPIN__", str(getenv("CPU_UOP_SPIN", 10000000))))
  return ready[0].store(value)

def parallel_worker_prog():
  generation = UOp.param(0, dtypes.uint64, (1,), volatile=True)
  group_count = UOp.param(1, dtypes.uint64, (1,), volatile=True)
  ring_addr = UOp.param(2, dtypes.uint64, (1,), volatile=True)
  completed = UOp.param(3, dtypes.uint64, (PARALLEL_WORKERS,), volatile=True)
  worker_id = UOp.param(4, dtypes.int, (), vmin_vmax=(1, PARALLEL_WORKERS), name="worker_id", addrspace=AddrSpace.ALU)
  wait_fn, sem = UOp.param(5, dtypes.uint64, (1,)), UOp.param(6, dtypes.uint64, (1,))
  ready_values, helper_fn = UOp.param(7, dtypes.uint64, (PARALLEL_WORKERS,), volatile=True), UOp.param(8, dtypes.uint64, (1,))
  cur = UOp.loop(0)
  worker_idx = worker_id-1
  ready = helper_fn.after(cur)[0].load().call(generation.after(cur).index(0), completed.after(cur).index(worker_idx),
                                               ready_values.after(cur).index(worker_idx), wait_fn.after(cur).index(0),
                                               sem.after(cur).index(0), ret_dtype=dtypes.void)
  seen = ready_values.after(ready)[worker_idx].load()
  worker = worker_id.cast(dtypes.uint64)
  count = group_count.after(seen)[0].load()
  work_count = (worker < count).where((count - worker + PARALLEL_PARTICIPANTS - 1) // PARALLEL_PARTICIPANTS, 0)
  work = UOp.range(work_count, 2, dtype=dtypes.uint64)
  command = worker + work * UOp.const(dtypes.uint64, PARALLEL_PARTICIPANTS)
  address = ring_addr.after(seen)[0].load()
  entry = [UOp(Ops.CUSTOM, dtypes.uint64, (address, command * CMD_SIZE + i),
               arg="*((volatile unsigned long *){0} + {1})") for i in range(CMD_SIZE)]
  finished = entry[0].call(*entry[1:], ret_dtype=dtypes.void).end(work)
  return completed.after(finished, seen)[worker_id-1].store(seen).end(cur, count.ne(0))

def parallel_dispatch_prog():
  commands = UOp.param(0, dtypes.uint64, (RING_SLOTS * CMD_SIZE,), volatile=True)
  generation = UOp.param(1, dtypes.uint64, (1,), volatile=True)
  group_count = UOp.param(2, dtypes.uint64, (1,), volatile=True)
  ring_addr = UOp.param(3, dtypes.uint64, (1,), volatile=True)
  completed = UOp.param(4, dtypes.uint64, (PARALLEL_WORKERS,), volatile=True)
  count = UOp.param(5, dtypes.int, (), vmin_vmax=(1, PARALLEL_PARTICIPANTS), name="count", addrspace=AddrSpace.ALU)
  post_fn, sems = UOp.param(6, dtypes.uint64, (1,)), UOp.param(7, dtypes.uint64, (PARALLEL_WORKERS,))
  address = UOp(Ops.CUSTOM, dtypes.uint64, (commands.index(0),), arg="(unsigned long){0}")
  publish = UOp.group(ring_addr[0].store(address), group_count[0].store(count.cast(dtypes.uint64)))
  current = generation.after(publish)[0].load()
  next_generation = current + 1
  signal = generation[0].store(next_generation)
  if WIN: wake = signal
  else:
    wake_worker = UOp.range(count - 1, 3)
    wake = post_fn.after(signal)[0].load().call(sems[wake_worker].load(), ret_dtype=dtypes.void).end(wake_worker)
  work = UOp.range((count + PARALLEL_PARTICIPANTS - 1) // PARALLEL_PARTICIPANTS, 2)
  command = work * PARALLEL_PARTICIPANTS
  entry = [commands.after(wake).index(command * CMD_SIZE + i).load() for i in range(CMD_SIZE)]
  own_done = entry[0].call(*entry[1:], ret_dtype=dtypes.void).end(work)
  worker = UOp.range(count - 1, 0)
  wait = UOp.loop(1)
  done = completed.after(own_done, wait).index(worker).load()
  return done.end(wait, done < next_generation).end(worker).sink(arg=KernelInfo("parallel_dispatch_prog"), tag=1)

def host_wait(ctx, dst:UOp, val:UOp) -> UOp:
  return (cur:=dst.after(loop:=UOp.loop(next(ctx))).index(UOp.const(dtypes.int, 0)).load()).end(loop, cur < val)

pm_host_opsel = PatternMatcher([(UPat(Ops.INS, arg="wait", src=(UPat(name="dst"), UPat(name="val"))), host_wait)])

def encode_host_queue(q:UOp) -> UOp:
  # TODO: subset of hcq2 for now
  spins, (store,) = partition(graph_rewrite(q, pm_host_opsel, ctx=itertools.count(), walk=True, name="host opsel").src, lambda u: u.op is Ops.END)
  assert store.op is Ops.INS and store.arg == "store", f"host queue cannot encode {store.op} {store.arg}"
  return store.src[0].after(*spins).index(UOp.const(dtypes.int, 0)).store(store.src[1])

class CPUComputeQueue(HWQueue):
  def __init__(self, dev):
    super().__init__()
    self.dev = dev
    self._encoded:array.array|None = None
    self._exec_groups:list[tuple[int, int, bool]] = []
  def _cmd(self, prog, args=(), vals=()): return self.exec(prg:=self.dev.prgs[prog], prg.fill_kernargs(args, vals), None, None)
  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    if (lvp:=isinstance(args_state, LVPArgsState)): self.bind_args_state(args_state)
    args:list[sint|None] = [args_state.buf.va_addr] if lvp else [*[x.va_addr for x in args_state.bufs], *args_state.vals]
    assert len(args) <= MAX_ARGS, f"CPU program {prg.name!r} supports at most {MAX_ARGS} arguments, got {len(args)}"
    start = len(self._q) // CMD_SIZE
    for tid in range(1 if lvp else (global_size or (1,))[0]):
      if not lvp and 'core_id' in prg.runtimevars: args[prg.runtimevars['core_id']] = tid
      self.q(prg, *[unwrap(x) for x in args], *([0] * (MAX_ARGS - len(args))))
    self._exec_groups.append((start, len(self._q) // CMD_SIZE, prg.cpu_parallel))
    return self
  def wait(self, signal, value=0): return self._cmd(wait_prog, (signal.base_buf,), (value,))
  def timestamp(self, signal): return self._cmd(timestamp_prog, (signal.base_buf.offset(8, 8), self.dev.func_table._buf.offset(0, 8)))
  def signal(self, signal, value:sint=0): return self._cmd(signal_prog, (signal.base_buf,), (value,))
  def _submit(self, dev):
    dev.ensure_worker()
    if self._encoded is None:
      self._encoded = array.array('Q', ((x.addr if i % CMD_SIZE == 0 else int(x)) & ((1<<64)-1) for i,x in enumerate(self._q)))
    else:
      for off, _ in self.q_sints: self._encoded[off] = int(self._q[off]) & ((1<<64)-1)
    encoded = self._encoded
    parallel_buf = None
    if getattr(dev, "parallel_uops", False) and not getattr(dev, "parallel_shutdown", False):
      completed = dev.progress_view[0]
      still_inflight = []
      for end_pos,buf in dev.parallel_command_inflight:
        if end_pos <= completed: dev.parallel_command_pool.append(buf)
        else: still_inflight.append((end_pos, buf))
      dev.parallel_command_inflight = still_inflight
      parallel_words = sum((end-start) * CMD_SIZE for start,end,parallel in self._exec_groups if parallel and end-start > 1)
      if parallel_words:
        required_bytes = parallel_words * 8
        parallel_buf = next((buf for buf in dev.parallel_command_pool if buf.nbytes >= required_bytes), None)
        if parallel_buf is not None: dev.parallel_command_pool.remove(parallel_buf)
        else: parallel_buf = Buffer(dev.device, required_bytes, dtypes.uint8, preallocate=True)
        parallel_view = parallel_buf.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
        transformed, parallel_offset = array.array('Q'), 0
        dispatch = dev.prgs[parallel_dispatch_prog]
        for start,end,parallel in self._exec_groups:
          if not parallel or end-start == 1:
            transformed.extend(encoded[start*CMD_SIZE:end*CMD_SIZE])
            continue
          words = (end-start) * CMD_SIZE
          parallel_view[parallel_offset:parallel_offset+words] = encoded[start*CMD_SIZE:end*CMD_SIZE]
          args = [parallel_buf._buf.va_addr + parallel_offset * 8, dev.parallel_state._buf.va_addr,
                  dev.parallel_state._buf.va_addr + 8, dev.parallel_state._buf.va_addr + 16,
                  dev.parallel_state._buf.va_addr + 24, end-start, dev.func_table._buf.va_addr + 32,
                  dev.parallel_sems._buf.va_addr]
          transformed.extend(array.array('Q', [dispatch.addr, *args, *([0] * (MAX_ARGS-len(args)))]))
          parallel_offset += words
        encoded = transformed
    ring_view = dev.ring.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
    cmds, submitted = len(encoded) // CMD_SIZE, 0
    while submitted < cmds:
      consumed = dev.progress_view[0]
      if (available:=RING_SLOTS - (dev.ring_pos - consumed)) == 0: continue
      start = dev.ring_pos % RING_SLOTS
      count = min(cmds - submitted, available, RING_SLOTS - start)
      src = submitted * CMD_SIZE
      ring_view[start*CMD_SIZE:(start+count)*CMD_SIZE] = encoded[src:src+count*CMD_SIZE]
      dev.ring_pos += count
      if WIN: dev.sys.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = dev.ring_pos
      else: dev.post_many(dev.sem_addr, dev.func_table._buf.va_addr + 32, count)
      submitted += count
    if parallel_buf is not None: dev.parallel_command_inflight.append((dev.ring_pos, parallel_buf))

class LVPArgsState(CLikeArgsState):
  def __init__(self, buf, prg, bufs, vals=()): super().__init__(buf, prg, bufs, vals, [*data64_le(buf.va_addr + 12), (len(bufs) + len(vals)) * 2])

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(HCQProgram['CPUDevice']):
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev:CPUDevice, obj:TinyELF):
    self.signature, self.runtimevars = obj.signature, {name:slot for name,slot,*_ in obj.signature if name == 'core_id'}
    self.cpu_parallel = obj.cpu_parallel

    LVP = obj.target.renderer == "LVP"
    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.addr = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(obj.lib)), MEM_COMMIT | MEM_RESERVE,
                                                      PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.addr, obj.lib, len(obj.lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.addr), ctypes.c_size_t(len(obj.lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.addr)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(obj.lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
      self.addr = mv_address(self.mem)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      lib = jit_loader(obj.lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m']) if LVP else obj.lib
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

    super().__init__(LVPArgsState if LVP else HCQArgsState, dev, obj, kernargs_alloc_size=12+256 if LVP else 0)

  @suppress_finalizing
  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class CPUAllocator(HCQAllocator):
  def __init__(self, dev:CPUDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: addr, buf = options.external_ptr, None
    elif WIN: addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _as_buffer(self, src) -> memoryview: return to_mv(src.va_addr, src.size)
  def _do_map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")
    return HCQBuffer(buf.view.addr, buf.size, view=buf.view, owner=buf.owner)
  def _unmap(self, mb): pass  # CPU _do_map returns a view wrapper, nothing to release

class CPUDevice(HCQCompiled):
  pm_lower = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),)), encode_host_queue)])

  pm_bufferize = PatternMatcher([
    (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].timeline("sentinel", (1 << 64) - 1)),
    (UPat(Ops.PARAM, tag="COMPUTE:0_timeline_signal"), lambda ctx: ctx[0].timeline("signal", 0)),
    (UPat(Ops.PARAM, tag="COMPUTE:0_timeline_value"), lambda ctx: ctx[0].timeline("value", 1)),
  ])

  @functools.cache
  def timeline(self, tag:str, init_value:int) -> Buffer:
    (buf:=Buffer(self.device, 1, dtypes.uint64, preallocate=True)).as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = init_value
    return buf

  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], CPUProgram, HCQSignal,
      functools.partial(CPUComputeQueue, self), arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.ring_pos = 0

    # posix uses sem to put cpus into sleep
    self.sem_addr = 0
    if not WIN:
      self.sem = libc.sem_open(sem_name:=f"/tinygrad-{os.getpid()}-{id(self):x}".encode(), os.O_CREAT|os.O_EXCL, 0o600, 0) # type: ignore[call-arg]
      self.sem_addr = unwrap(ctypes.cast(self.sem, ctypes.c_void_p).value)
      if self.sem_addr == ctypes.c_void_p(-1).value or libc.sem_unlink(sem_name): raise OSError(ctypes.get_errno(), "semaphore")

    # TODO: move to hcq2
    with Context(EMULATED_DTYPES="", TRACK_MATCH_STATS=0):
      helpers = (signal_prog, wait_prog, timestamp_prog, quit_prog, worker_prog, post_many_prog,
                 parallel_wait_prog, parallel_worker_prog, parallel_dispatch_prog)
      prgs = {f: f().sink(arg=KernelInfo(f.__name__), tag=1) for f in helpers}
      renderer = ClangRenderer(replace(self.renderer.target, renderer="CLANG"))
      self.prgs = {f: self.runtime(do_to_program(v, renderer).to_elf()) for f,v in prgs.items()}
    if not WIN:
      self.post_many = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int)(cast(CPUProgram, self.prgs[post_many_prog]).addr)

    self.worker:threading.Thread|None = None
    self.parallel_uops = bool(getenv("CPU_PARALLEL_UOPS", 1)) and not WIN
    self.parallel_shutdown = False
    self._physical_affinity = False

  @functools.cached_property
  def ring(self) -> Buffer: return Buffer(self.device, RING_SLOTS * CMD_SIZE, dtypes.uint64, preallocate=True)
  @functools.cached_property
  def sys(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint64, preallocate=True)
  @functools.cached_property
  def progress(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint64, preallocate=True)
  @functools.cached_property
  def sem_buf(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint8, options=BufferSpec(external_ptr=self.sem_addr), preallocate=True)

  # TODO: move to hcq2 infra
  @functools.cached_property
  def func_table(self) -> Buffer:
    fns = ([0, ctypes.windll.kernel32.ExitThread, 0, 0, 0] if WIN else  # type: ignore[attr-defined]
           [libc.dll.clock_gettime, libc.dll.pthread_exit, libc.dll.sem_wait, libc.dll.sem_close, libc.dll.sem_post])
    addrs = array.array('Q', [unwrap(ctypes.cast(f, ctypes.c_void_p).value) if f else 0 for f in fns])
    (ft:=Buffer(self.device, len(fns), dtypes.uint64, preallocate=True)).as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[:] = addrs
    return ft

  @functools.cache
  def ensure_worker(self):
    self.progress_view = self.progress.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
    self.worker = threading.Thread(target=cast(CPUProgram, self.prgs[worker_prog]).fxn, daemon=True, args=[ctypes.c_uint64(x) for x in
      [self.ring._buf.va_addr, self.sys._buf.va_addr if WIN else self.func_table._buf.va_addr+16, self.sem_addr,
       self.progress._buf.va_addr]])
    self.worker.start()

    if self.parallel_uops:
      self.parallel_state = Buffer(self.device, 3 + 2 * PARALLEL_WORKERS, dtypes.uint64, preallocate=True)
      self.parallel_command_pool:list[Buffer] = []
      self.parallel_command_inflight:list[tuple[int, Buffer]] = []
      self.parallel_sems = Buffer(self.device, PARALLEL_WORKERS, dtypes.uint64, preallocate=True)
      parallel_sem_addrs, self.parallel_sem_handles = [], []
      for i in range(PARALLEL_WORKERS):
        sem = libc.sem_open(sem_name:=f"/tinygrad-{os.getpid()}-{id(self):x}-p{i}".encode(),
                            os.O_CREAT|os.O_EXCL, 0o600, 0)  # type: ignore[call-arg]
        if (sem_addr:=unwrap(ctypes.cast(sem, ctypes.c_void_p).value)) == ctypes.c_void_p(-1).value or libc.sem_unlink(sem_name):
          raise OSError(ctypes.get_errno(), "parallel semaphore")
        self.parallel_sem_handles.append(sem)
        parallel_sem_addrs.append(sem_addr)
      self.parallel_sems.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[:] = array.array('Q', parallel_sem_addrs)
      state_view = self.parallel_state.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
      self.parallel_generation_view, self.parallel_count_view = state_view[0:1], state_view[1:2]
      self.parallel_ring_addr_view = state_view[2:3]
      self.parallel_completed_view = state_view[3:3+PARALLEL_WORKERS]
      self.parallel_helper_fn = Buffer(self.device, 1, dtypes.uint64, preallocate=True)
      self.parallel_helper_fn.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = cast(CPUProgram, self.prgs[parallel_wait_prog]).addr
      parallel_ready_addr = self.parallel_state._buf.va_addr + (3 + PARALLEL_WORKERS) * 8
      self.parallel_workers = [threading.Thread(target=cast(CPUProgram, self.prgs[parallel_worker_prog]).fxn,
        args=(ctypes.c_uint64(self.parallel_state._buf.va_addr), ctypes.c_uint64(self.parallel_state._buf.va_addr + 8),
              ctypes.c_uint64(self.parallel_state._buf.va_addr + 16), ctypes.c_uint64(self.parallel_state._buf.va_addr + 24),
              ctypes.c_uint64(i+1), ctypes.c_uint64(self.func_table._buf.va_addr + 16),
              ctypes.c_uint64(parallel_sem_addrs[i]), ctypes.c_uint64(parallel_ready_addr),
              ctypes.c_uint64(self.parallel_helper_fn._buf.va_addr)),
        daemon=True) for i in range(PARALLEL_WORKERS)]
      for worker in self.parallel_workers: worker.start()

  def use_physical_cores(self):
    self.ensure_worker()
    if self._physical_affinity or not sys.platform.startswith("linux") or self.worker is None or self.worker.native_id is None: return
    allowed = os.sched_getaffinity(self.worker.native_id)
    physical:dict[tuple[str, str], int] = {}
    try:
      for cpu in sorted(allowed):
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text().strip(), (topology / "core_id").read_text().strip())
        physical.setdefault(key, cpu)
    except (OSError, ValueError): return
    if physical:
      cpus = tuple(physical.values())
      os.sched_setaffinity(self.worker.native_id, set(cpus))
      for i, worker in enumerate(getattr(self, "parallel_workers", ())):
        if worker.native_id is not None: os.sched_setaffinity(worker.native_id, {cpus[(i+1) % len(cpus)]})
    self._physical_affinity = True

  def finalize(self):
    if self.worker is None: return
    self.synchronize()
    if self.parallel_uops:
      self.parallel_shutdown = True
      self.parallel_count_view[0] = 0
      self.parallel_generation_view[0] += 1
      for sem in self.parallel_sem_handles: assert libc.sem_post(sem) == 0
      for worker in self.parallel_workers: worker.join()
      for sem in self.parallel_sem_handles: assert libc.sem_close(sem) == 0
    ft = self.func_table._buf
    CPUComputeQueue(self)._cmd(quit_prog, (ft.offset(8, 8),) if WIN else (ft.offset(8, 24), self.sem_buf._buf)).submit(self)
    self.worker = None
