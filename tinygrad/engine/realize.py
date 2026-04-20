from typing import cast, Callable, Iterator
import time, pprint, random, itertools, math, contextlib
from dataclasses import dataclass, replace, field
from tinygrad.helpers import all_same, colored, DEBUG, GlobalCounters, ansilen, NOOPT, all_int, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import BEAM, DEVECTORIZE, size_to_str, time_to_str, VALIDATE_WITH_CPU, cpu_profile, PROFILE, ProfilePointEvent, cpu_events
from tinygrad.helpers import prod, unwrap, EMULATED_DTYPES
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer, buffers, graph_rewrite
from tinygrad.device import Device, Buffer, MultiBuffer
from tinygrad.renderer import ProgramSpec, Estimates
from tinygrad.codegen import get_program

# **************** Stat ****************

def update_stats(display_name:str, device:str, estimates:Estimates, var_vals:dict[str, int], et:float|None, buf_count:int,
                 jit=False, metadata:tuple[Metadata, ...]=(), first_run=False):
  GlobalCounters.kernel_count += 1
  GlobalCounters.global_ops += (op_est:=sym_infer(estimates.ops, var_vals))
  GlobalCounters.global_mem += (mem_est:=sym_infer(estimates.mem, var_vals))
  if et is not None: GlobalCounters.time_sum_s += et
  if DEBUG >= 2:
    lds_est = sym_infer(estimates.lds, var_vals)
    header_color = 'magenta' if jit else ('green' if first_run else None)
    ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
    flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
    flops_str = f"{flops*1e-9:7.0f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:7.0f} TFLOPS", 'green')
    mem_str = f"{membw*1e-9:4.0f}|{ldsbw*1e-9:<6.0f} GB/s" if membw < 1e13 and ldsbw < 1e15 else \
      colored(f"{membw*1e-12:4.0f}|{ldsbw*1e-12:<6.0f} TB/s", 'green')
    print(f"{colored(f'*** {device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
      f" {display_name+' '*(46-ansilen(display_name))} arg {buf_count:2d} mem {GlobalCounters.mem_used/1e9:6.2f} GB"+
      ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})")+
      f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in metadata] if metadata else ''}")

# **************** Runners ****************

class Runner:
  def __init__(self, display_name:str, device:str, estimates=Estimates()):
    self.first_run, self.display_name, self.device, self.estimates = True, display_name, device, estimates
  @property
  def dev(self): return Device[self.device]
  def exec(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None) -> float|None:
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    raise NotImplementedError("override this")

def optimize_local_size(_prg:Callable, global_size:list[int], rawbufs:list[Buffer]) -> list[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try:
      return _prg(*[x._buf for x in test_rawbuffers],global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)],
                  local_size=local_size, wait=True)
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

class CompiledRunner(Runner):
  def __init__(self, p:ProgramSpec, prg=None):
    if DEBUG >= 3 and p.applied_opts: print(p.applied_opts)
    if DEBUG >= 4: print(p.src)
    if p.lib is None:
      with cpu_profile(TracingKey(f"compile {p.name}", (p.function_name,)), "TINY"):
        p = replace(p, lib=Device[p.device].compiler.compile_cached(p.src))
    self.p:ProgramSpec = p
    assert self.p.lib is not None
    if DEBUG >= 7: Device[p.device].compiler.disassemble(self.p.lib)
    self._prg = Device[p.device].runtime(p.function_name, self.p.lib, *p.aux, runtimevars=p.runtimevars) if prg is None else prg
    super().__init__(p.name, p.device, p.estimates)

  def __reduce__(self): return self.__class__, (self.p,)

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None, wait=False, timeout:int|None=None) -> float|None:
    if var_vals is None: var_vals = {}
    global_size, local_size = self.p.launch_dims(var_vals)
    if Device[self.p.device].renderer.has_local and local_size is None and all_int(self.p.global_size):
      local_size = optimize_local_size(self._prg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)
    return self._prg(*[x._buf for x in rawbufs], global_size=tuple(global_size), local_size=tuple(local_size) if local_size else None,
                     vals=tuple(var_vals[k.expr] if k.expr not in self.p.runtimevars else None for k in self.p.vars), wait=wait, timeout=timeout)

class ViewOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    sz = f"{total_sz/1e6:7.2f}M" if total_sz >= 1e6 else f"{total_sz:8d}"
    name = f"{type(self).__name__[6:].lower()} {sz}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and getattr(src.allocator.dev, 'fd', None) is not None
    if disk_supports_fast_copyout and hasattr(dest.allocator, 'copy_from_disk') and src.nbytes >= 4096 and dest.allocator.supports_copy_from_disk:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif isinstance(src.device, str) and src.device.startswith(("DISK", "TINYFS")) and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_memoryview(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  def copy(self, dest, src): dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)

class EncDec(Runner):
  def __init__(self, cf:UOp, total_sz:int, device:str):
    self.shape, self.pos_var = tuple(s.arg for s in cf.src if s.op is Ops.CONST), cf.variables()[0].expr
    name = f"enc/dec {total_sz/1e6:7.2f}M, HEVC" if total_sz >= 1e6 else f"enc/dec {total_sz:8d}, HEVC"
    super().__init__(colored(name, "yellow"), device, Estimates(lds=total_sz, mem=total_sz))
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    st = time.perf_counter()
    rawbufs[0].allocator._encode_decode(rawbufs[0]._buf, rawbufs[1]._buf, rawbufs[2]._buf,
                                        [x._buf for x in rawbufs[3:]], self.shape, var_vals[self.pos_var])
    if wait:
      Device[rawbufs[0].device].synchronize()
      return time.perf_counter() - st

# **************** method cache ****************

method_cache: dict[tuple[str, type, bytes, tuple, bool], CompiledRunner] = {}
def get_runner(device:str, ast:UOp) -> CompiledRunner:
  # TODO: this should be all context relevant to rendering
  context = (NOOPT.value, DEVECTORIZE.value, EMULATED_DTYPES.value)
  ckey = (device, type(Device[device].compiler), ast.key, context, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (device.split(":")[0], type(Device[device].compiler), ast.key, context, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device))
  else:
    prg: ProgramSpec = get_program(ast, Device[device].renderer)
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

# **************** lowering functions ****************

# NOTE: ctx is the buffers
si_lowerer = PatternMatcher([
  (UPat((Ops.SINK, Ops.PROGRAM, Ops.BEAM), name="sink"), lambda ctx,sink: get_runner(ctx[0].device, sink)),
  (UPat(Ops.BUFFER_VIEW), lambda ctx: ViewOp(ctx[0])),
  (UPat(Ops.COPY), lambda ctx: (BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(alc:=Device[ctx[0].device].allocator, '_transfer') and alc.supports_transfer and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device))),
  (UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="cf"), lambda ctx,cf: EncDec(cf, ctx[0].nbytes, ctx[0].device)),
  (UPat(Ops.CUSTOM_FUNCTION, arg="graph", name="cf"), lambda ctx,cf: Device[cf.device if isinstance(cf.device,str) else cf.device[0]].graph(cf, ctx))
])

@dataclass
class ExecItem:
  ast: UOp
  bufs: list[Buffer|None] = field(default_factory=list)
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  prg: Runner|None = None

  def lower(self):
    """Populate self.prg by lowering the AST."""
    if self.prg is not None: return self
    try: self.prg = cast(Runner, si_lowerer.rewrite(self.ast, self.bufs))
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {self.ast.op}")
        print("tensor operations:")
        pprint.pprint(self.metadata, indent=2)
      raise e
    return self

  def run(self, _var_vals:dict[str, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
    if self.prg is None: self.lower()
    assert self.prg is not None
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    # reorder bufs to match program globals if needed
    _bufs = [self.bufs[i] for i in self.prg.p.globals] if isinstance(self.prg, CompiledRunner) else self.bufs
    bufs = [unwrap(x) for x in _bufs] if jit else [unwrap(x).ensure_allocated() for x in _bufs]
    if PROFILE:
      payload = {"metadata":self.metadata, "var_vals":var_vals, "bufs":[b.trace_num for b in bufs], "name":self.prg.display_name}
      payload["outputs"], payload["inputs"] = (self.prg.p.outs, self.prg.p.ins) if isinstance(self.prg, CompiledRunner) else ([0], [1])
      cpu_events.append(ProfilePointEvent(self.prg.device, "exec", len(cpu_events), payload))
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      update_stats(self.prg.display_name, self.prg.device, self.prg.estimates, var_vals, et, len(bufs), jit, self.metadata, self.prg.first_run)
      self.prg.first_run = False
    return et

# **************** main run function ****************

capturing: list = []  # put classes with an add_linear method in here

def run_schedule(schedule:list[ExecItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  while len(schedule):
    ei = schedule.pop(0).lower()
    sink = ei.ast.src[0] if ei.ast.op is Ops.BEAM else ei.ast
    if VALIDATE_WITH_CPU and sink.op is Ops.SINK:
      # copy in allocated buffers from the GPU
      bufs = [b for b in ei.bufs if b is not None]
      nb: list[Buffer|None] = [Buffer("CPU", b.size, b.dtype) for b in bufs]
      for cpu_b, gpu_b in zip(nb, bufs):
        if cpu_b is not None and gpu_b.is_allocated(): cpu_b.ensure_allocated().copyin(gpu_b.as_memoryview())

      # run on GPU
      ei.run(var_vals, do_update_stats=do_update_stats)

      # validate the output buffers match (NOTE: this is assuming the output is buffer 0)
      ExecItem(sink, nb, ei.metadata, ei.fixedvars).run(var_vals, do_update_stats=do_update_stats)
      import numpy as np
      assert nb[0] is not None
      np.testing.assert_allclose(bufs[0].numpy(), nb[0].numpy(), rtol=1e-3, atol=1e-3)
    else:
      ei.run(var_vals, do_update_stats=do_update_stats)

# **************** run linear ****************

@dataclass
class ExecContext:
  var_vals: dict[str, int] = field(default_factory=dict)
  do_update_stats: bool = True

@contextlib.contextmanager
def track_stats(ctx:ExecContext, call:UOp, device:str, display_name:str, estimates:Estimates, bufs:list[Buffer], var_vals:dict[str, int], *,
                outputs=(0,), inputs=(1,), first_run=False):
  if PROFILE: cpu_events.append(ProfilePointEvent(device, "exec", len(cpu_events), {"metadata": call.arg.metadata, "var_vals": var_vals,
                                                  "bufs": [b.trace_num for b in bufs], "name": display_name, "outputs": outputs, "inputs": inputs}))
  timing: list[float|None] = [None]
  st = time.perf_counter()
  yield timing
  if not ctx.do_update_stats: return
  if timing[0] is None and DEBUG >= 2:
    Device[device].synchronize()
    timing[0] = time.perf_counter() - st
  update_stats(display_name, device, estimates, var_vals, timing[0], len(bufs), jit=False, metadata=call.arg.metadata, first_run=first_run)

def unwrap_multi(call:UOp) -> Iterator[tuple[list[Buffer], dict[str, int]]]:
  bufs = [b.buffer for b in call.src[1:] if b.op is not Ops.BIND]
  if not any(isinstance(b, MultiBuffer) for b in bufs): yield cast(list[Buffer], bufs), {}
  else:
    dnum = next((x.expr for x in call.src[0].variables() if x.expr == '_device_num'), None)
    for j, per_dev in enumerate(zip(*[cast(MultiBuffer, b).bufs for b in bufs])): yield list(per_dev), {dnum: j} if dnum else {}

def exec_view(ctx:ExecContext, call, ast):
  bufs = [b.buffer for b in call.src[1:] if b.op is not Ops.BIND]
  bv = bufs[1].view(call.src[1].arg, ast.dtype, ast.arg[1]*bufs[1].dtype.itemsize)
  with track_stats(ctx, call, bv.device, colored(f"view {bv.nbytes:8d} @ {bv.offset:<10d}", "yellow"), Estimates(), [bv, bufs[1]], ctx.var_vals):
    buffers[call.src[1]] = bv

def exec_copy(ctx:ExecContext, call, ast):
  for bufs, device_vars in unwrap_multi(call):
    dest, src = bufs[0].ensure_allocated(), bufs[1].ensure_allocated()
    xfer = hasattr(alc:=Device[dest.device].allocator,'_transfer') and alc.supports_transfer and dest.device.split(":")[0]==src.device.split(":")[0]
    prg = (BufferXfer if xfer else BufferCopy)(dest.nbytes, dest.device, src.device)
    name = colored(f"{'xfer' if xfer else 'copy'} {size_to_str(dest.nbytes):>8s}, {dest.device[:7]:>7s} <- {src.device[:7]:7s}", "yellow")
    with track_stats(ctx, call, dest.device, name, Estimates(lds=dest.nbytes, mem=dest.nbytes), [dest, src], {**ctx.var_vals, **device_vars}):
      prg.copy(dest, src)

def exec_kernel(ctx:ExecContext, call, ast):
  sink = ast.src[0] if ast.op is Ops.BEAM else ast

  for bufs, device_vars in unwrap_multi(call):
    var_vals = {**ctx.var_vals, **device_vars}
    prg = get_runner(bufs[0].device, ast)
    prg_bufs = [bufs[i].ensure_allocated() for i in prg.p.globals]

    if VALIDATE_WITH_CPU and sink.op is Ops.SINK:
      cpu_bufs = [Buffer("CPU", b.size, b.dtype).ensure_allocated().copyin(b.ensure_allocated().as_memoryview()) for b in bufs]

    with track_stats(ctx, call, prg.device, prg.display_name, prg.estimates, prg_bufs, var_vals,
                     outputs=tuple(prg.p.outs), inputs=tuple(prg.p.ins), first_run=prg.first_run) as timing:
      timing[0] = prg(prg_bufs, var_vals, wait=DEBUG >= 2)
      prg.first_run = False

    if VALIDATE_WITH_CPU and sink.op is Ops.SINK:
      import numpy as np
      cpu_prg = get_runner("CPU", sink)
      cpu_prg([cpu_bufs[i] for i in cpu_prg.p.globals], var_vals, wait=False)
      for i in prg.p.outs: np.testing.assert_allclose(prg_bufs[i].numpy(), cpu_bufs[i].numpy(), rtol=1e-3, atol=1e-3)

def exec_encdec(ctx:ExecContext, call, ast):
  bufs = [b.buffer.ensure_allocated() for b in call.src[1:] if b.op is not Ops.BIND]
  shape, pos_var = tuple(s.arg for s in ast.src if s.op is Ops.CONST), ast.variables()[0].expr
  with track_stats(ctx, call, bufs[0].device, colored(f"enc/dec {size_to_str(bufs[0].nbytes)}", "yellow"),
                   Estimates(lds=bufs[0].nbytes, mem=bufs[0].nbytes), bufs, ctx.var_vals):
    bufs[0].allocator._encode_decode(bufs[0]._buf, bufs[1]._buf, bufs[2]._buf, [x._buf for x in bufs[3:]], shape, ctx.var_vals[pos_var])

pm_beam = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True),
   lambda ctx,call,sink: call.replace(src=(UOp(Ops.BEAM, src=(sink,), arg=ctx), *call.src[1:]))),
])

pm_exec = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.BUFFER_VIEW, name="ast"),), name="call", allow_any_len=True), exec_view),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True), exec_copy),
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.PROGRAM, Ops.BEAM), name="ast"),), name="call", allow_any_len=True), exec_kernel),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="ast"),), name="call", allow_any_len=True), exec_encdec),
])

def run_linear(linear:UOp, var_vals:dict[str, int]|None=None, do_update_stats=True):
  if BEAM >= 1: linear = graph_rewrite(linear, pm_beam, ctx=BEAM.value, name="add beam")
  ctx = ExecContext(var_vals or {}, do_update_stats)
  for call in linear.src: pm_exec.rewrite(call, ctx)
