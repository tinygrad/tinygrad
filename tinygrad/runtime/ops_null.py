import inspect, functools, math
from tinygrad.device import BufferStorage, BufferSpec, Buffer, Compiled, HostAllocator, MMIOInterface, Program, TinyELF
from tinygrad.device import ProfileGraphEntry, ProfileGraphEvent
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, dedup, mv_address, prod, to_tuple, cpu_events, perf_counter_us, NULL_ALLOW_COPYOUT, PROFILE
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.runtime.support.hcq2 import HWQueue, encode_submit, layout_args, pack_args
from tinygrad.runtime.support.memory import BumpAllocator

class NullRenderer(CStyleLanguage):
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}
  def asm(self, prg: UOp, lin: UOp) -> bytes:
    assert self.target.arch.startswith("gfx"), "only amd supports assembly"
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

BARRIER, EXEC, COPY, WAIT, STORE, TIMESTAMP = range(6)

_null_program_cache:dict[tuple[bytes, tuple[str, ...]], UOp] = {}
def null_program(lib:bytes, devs:tuple[str, ...]) -> UOp:
  if (buf:=_null_program_cache.get(key:=(lib, devs))) is None:
    b = UOp.placeholder((len(lib),), dtypes.uint8, next(UOp.unique_num), device=devs).rtag("program")
    buf = _null_program_cache[key] = b.after(b.store(UOp(Ops.BINARY, arg=lib).bitcast(b.dtype)))
  return buf

null_events:dict[tuple[str, str, bytes|None], int] = {}

class NullQueue(HWQueue):
  def cmd(self, op:int, *args:UOp|int): self.q(*[a if isinstance(a, UOp) else UOp.const(a, dtypes.uint64) for a in (op, *args, 0, 0, 0)][:4])
  def event(self, device:str, name:str, key:bytes|None=None) -> int: return null_events.setdefault((device, name, key), len(null_events))

  def exec(self, call:UOp, prg:UOp):
    args = [get_call_arg_uops(call)[gi].getaddr(self.devs) for gi in prg.arg.globals] + \
           [b.ccast(v.dtype) for v, b in zip(prg.arg.vars, get_call_var_uops(call, prg))]
    rows = layout_args(args)
    kernargs = UOp(Ops.LINEAR, src=tuple(pack_args(rows, max([8] + [o + w.dtype.itemsize for o, w in rows]))), arg="kernargs")
    self.cmd(EXEC, null_program(prg.src[3].arg, self.devs).getaddr(self.devs), kernargs.getaddr(self.devs),
             self.event(self.devs[0], prg.src[0].arg.function_name, prg.key))

  def copy(self, dst:UOp, src:UOp, sz:int):
    src_dev, dst_dev = to_tuple(src.device)[0], to_tuple(dst.device)[0]
    self.cmd(COPY, dst.getaddr(self.devs), src.getaddr(self.devs), self.event(f"{src_dev}:SDMA:0", f"{src_dev} -> {dst_dev}"))
  def memory_barrier(self): self.cmd(BARRIER)
  def wait(self, signal:UOp, value:UOp, eq:bool=False): self.cmd(WAIT, signal.getaddr(self.devs), value, int(eq))
  def signal(self, signal:UOp, value:UOp): self.cmd(STORE, signal.getaddr(self.devs), value)
  def timestamp(self, signal:UOp): self.cmd(TIMESTAMP, signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64))
  def submit(self, cmdbuf:UOp) -> UOp:
    doorbell = UOp.placeholder((1,), dtypes.uint64, device=self.devs, volatile=True, tag="doorbell")
    return doorbell.index(0).store(cmdbuf.bitcast(dtypes.uint64).index(0).load())

class NullProgram(Program['NullDevice']):
  def __init__(self, dev:'NullDevice', obj:TinyELF):
    self.streams = [(i, prod(shape)) for i, (name, _, _, shape) in enumerate(obj.signature) if (name or "").startswith("cmdbuf")]
  def __call__(self, *bufs, **kwargs) -> float|None:
    if not PROFILE: return None
    st, events = perf_counter_us(), list(null_events)
    words = [MMIOInterface(bufs[i], nbytes, fmt='Q')[:] for i, nbytes in self.streams]
    descs = [events[event] for w in words for op, event in zip(w[0::4], w[3::4]) if op in (EXEC, COPY)]
    counts = [sum(d[0] == device for d in descs[:i]) for i, (device, _, _) in enumerate(descs)]
    dur = max(1, math.ceil((perf_counter_us() - st) / (max(counts) + 1)))
    ents = [ProfileGraphEntry(device, name, 2*i, 2*i+1, key) for i, (device, name, key) in enumerate(descs)]
    cpu_events.append(ProfileGraphEvent(ents, [], [st + (c + k) * dur for c in counts for k in (0, 1)]))
    return None

class NullAllocator(HostAllocator):
  def __init__(self, dev:'NullDevice'):
    super().__init__(dev)
    self.va = BumpAllocator(1 << 40, base=1 << 40)

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: return BufferStorage(self.va.alloc(size, 0x1000))
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src):
    if not NULL_ALLOW_COPYOUT: raise RuntimeError("no copyout on NULL")
  def _map(self, buf:Buffer) -> BufferStorage: return BufferStorage(buf._buf if buf.device.startswith("NULL") else buf.host.addr)

class NullDevice(Compiled):
  pm_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg=f"submit_null_{q}", name="submit"),
                               lambda ctx, submit: encode_submit(NullQueue(ctx, submit))) for q in ("compute", "copy")])

  def __init__(self, device:str):
    assert (emu:=getenv("EMULATE", "")) == "", \
      "EMULATE is deprecated, use DEV=NULL:HIP:"+{"AMD":"gfx1100", "AMD_RDNA4":"gfx1201", "AMD_CDNA4":"gfx950"}.get(emu, "<arch>")
    renderers = [NullRenderer] + [r for m in [cstyle, nir, ptx, llvmir, wgsl] for r in m.__dict__.values()
                                  if inspect.isclass(r) and issubclass(r, Renderer)]
    super().__init__(device, NullAllocator(self), dedup(renderers), NullProgram)
    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, tag="timeline"), lambda ctx: ctx.timeline),
                                        (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.link_buffer(b.max_numel(), b.dtype))])

  @property
  def host(self) -> str: return self.device

  def link_buffer(self, size:int, dtype) -> Buffer:
    mv = memoryview(bytearray(max(size * dtype.itemsize, 1)))
    storage = BufferStorage(addr:=mv_address(mv), mv, MMIOInterface(addr, mv.nbytes, fmt='B'))
    return Buffer(self.device, size, dtype, opaque=storage, options=BufferSpec(external_ptr=addr, cpu_access=True))

  @functools.cached_property
  def timeline(self) -> Buffer: return self.link_buffer(2, dtypes.uint64)

  def synchronize(self, timeout:int|None=None): self.prof_ents.clear()
