from typing import cast, Callable, Iterator
import time, random, itertools, math, contextlib, weakref
from dataclasses import dataclass, replace, field
from tinygrad.helpers import colored, DEBUG, GlobalCounters, ansilen, NOOPT, all_int, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import BEAM, DEVECTORIZE, size_to_str, time_to_str, VALIDATE_WITH_CPU, cpu_profile, PROFILE, ProfilePointEvent, cpu_events
from tinygrad.helpers import prod, EMULATED_DTYPES, flatten
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer, buffers, graph_rewrite
from tinygrad.device import Device, Buffer, MultiBuffer
from tinygrad.renderer import ProgramSpec, Estimates
from tinygrad.codegen import get_program, to_program

# **************** Stat ****************

def estimate_uop(call:UOp) -> Estimates:
  if call.src[0].op is Ops.SINK: call = pm_compile.rewrite(call)

  ast = call.src[0]
  if ast.op is Ops.PROGRAM: return ast.src[0].arg.estimates or Estimates()
  if ast.op is Ops.COPY or (ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "encdec"):
    nbytes = prod(call.src[1].shape) * call.src[1].dtype.itemsize
    return Estimates(lds=nbytes, mem=nbytes)
  return Estimates()

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

# **************** run linear ****************

capturing: list = []  # put classes with an add_linear method in here

@dataclass
class ExecContext:
  var_vals: dict[str, int] = field(default_factory=dict)
  input_uops: tuple[UOp, ...] = ()
  do_update_stats: bool = True
  jit: bool = False

def _resolve(b:UOp, inputs:tuple[UOp, ...]) -> UOp:
  if b.op in (Ops.BUFFER_VIEW, Ops.MSELECT) and b.src[0].op is Ops.PARAM: return b.replace(src=(inputs[b.src[0].arg], *b.src[1:]))
  return inputs[b.arg] if b.op is Ops.PARAM else b
def resolve_params(call:UOp, inputs:tuple[UOp, ...]) -> list[UOp]: return [_resolve(b, inputs) for b in call.src[1:] if b.op is not Ops.BIND]

@contextlib.contextmanager
def track_stats(ctx:ExecContext, call:UOp, device:str, display_name:str, bufs:list[Buffer], var_vals:dict[str, int],
                outputs=(0,), inputs=(1,), first_run=False):
  if PROFILE: cpu_events.append(ProfilePointEvent(device, "exec", len(cpu_events), {"metadata": call.arg.metadata, "var_vals": var_vals,
                                                  "bufs": [b.trace_num for b in bufs], "name": display_name, "outputs": outputs, "inputs": inputs}))
  timing: list[float|None] = [None]
  if DEBUG >= 2: st = time.perf_counter()
  yield timing
  if not ctx.do_update_stats: return
  if DEBUG >= 2 and timing[0] is None:
    Device[device].synchronize()
    timing[0] = time.perf_counter() - st
  update_stats(display_name, device, estimate_uop(call), var_vals, timing[0], len(bufs), jit=ctx.jit, metadata=call.arg.metadata, first_run=first_run)

def unwrap_multi(call:UOp, resolved:list[UOp]) -> Iterator[tuple[list[Buffer], dict[str, int]]]:
  bufs = [b.buffer for b in resolved]
  if not any(isinstance(b, MultiBuffer) for b in bufs): yield cast(list[Buffer], bufs), {}
  else:
    dnum = next((x.expr for x in call.src[0].variables() if x.expr == '_device_num'), None)
    for j, per_dev in enumerate(zip(*[cast(MultiBuffer, b).bufs for b in bufs])): yield list(per_dev), {dnum: j} if dnum else {}

def exec_view(ctx:ExecContext, call, ast):
  resolved = resolve_params(call, ctx.input_uops)
  bufs = [cast(Buffer, b.buffer) for b in resolved]
  bv = bufs[1].view(resolved[0].arg, ast.dtype, ast.arg[1]*bufs[1].dtype.itemsize)
  with track_stats(ctx, call, bv.device, colored(f"view {bv.nbytes:8d} @ {bv.offset:<10d}", "yellow"), [bv, bufs[1]], ctx.var_vals):
    buffers[resolved[0]] = bv

def exec_copy(ctx:ExecContext, call, ast):
  for bufs, device_vars in unwrap_multi(call, resolve_params(call, ctx.input_uops)):
    dest, src = bufs[0].ensure_allocated(), bufs[1].ensure_allocated()
    xfer = hasattr(dest.allocator,'_transfer') and dest.allocator.supports_transfer and dest.device.split(":")[0] == src.device.split(":")[0]
    name = colored(f"{'xfer' if xfer else 'copy'} {size_to_str(bufs[0].nbytes):>10}, {dest.device[:7]:>7s} <- {src.device[:7]:7s}", "yellow")
    with track_stats(ctx, call, dest.device, name, [dest, src], ctx.var_vals):
      if xfer:
        dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev) # type:ignore[attr-defined]
      elif src.device.startswith("DISK") and getattr(src.allocator.dev, 'fd', None) is not None \
           and hasattr(dest.allocator, 'copy_from_disk') and src.nbytes >= 4096 and dest.allocator.supports_copy_from_disk:
        dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
      elif src.device.startswith(("DISK", "TINYFS")) and hasattr(dest.allocator, '_as_buffer'):
        src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
      else: dest.copyin(src.as_memoryview(allow_zero_copy=True))

def exec_kernel(ctx:ExecContext, call, ast):
  for bufs, device_vars in unwrap_multi(call, resolve_params(call, ctx.input_uops)):
    var_vals = {**ctx.var_vals, **device_vars}
    prg = get_runner(bufs[0].device, ast)
    prg_bufs = [bufs[i].ensure_allocated() for i in prg.p.globals]

    if VALIDATE_WITH_CPU and ast.op is Ops.SINK:
      cpu_bufs = [Buffer("CPU", b.size, b.dtype).ensure_allocated().copyin(b.ensure_allocated().as_memoryview()) for b in bufs]

    with track_stats(ctx, call, prg.device, prg.display_name, prg_bufs, var_vals,
                     outputs=tuple(prg.p.outs), inputs=tuple(prg.p.ins), first_run=prg.first_run) as timing:
      timing[0] = prg(prg_bufs, var_vals, wait=DEBUG >= 2)
      prg.first_run = False

    if VALIDATE_WITH_CPU and ast.op is Ops.SINK:
      import numpy as np
      cpu_prg = get_runner("CPU", ast)
      cpu_prg([cpu_bufs[i] for i in cpu_prg.p.globals], var_vals, wait=False)
      for i in prg.p.outs: np.testing.assert_allclose(prg_bufs[i].numpy(), cpu_bufs[i].numpy(), rtol=1e-3, atol=1e-3)

def exec_encdec(ctx:ExecContext, call, ast):
  bufs = [cast(Buffer, b.buffer).ensure_allocated() for b in resolve_params(call, ctx.input_uops)]
  shape, pos_var = tuple(s.arg for s in ast.src if s.op is Ops.CONST), ast.variables()[0].expr
  with track_stats(ctx, call, bufs[0].device, colored(f"enc/dec {size_to_str(bufs[0].nbytes)}", "yellow"), bufs, ctx.var_vals):
    bufs[0].allocator._encode_decode(bufs[0]._buf, bufs[1]._buf, bufs[2]._buf, [x._buf for x in bufs[3:]], shape, ctx.var_vals[pos_var])

graph_cache:weakref.WeakKeyDictionary[UOp, Runner] = weakref.WeakKeyDictionary()
def exec_graph(ctx:ExecContext, call, cf):
  bufs = flatten([b.bufs if isinstance(b, MultiBuffer) else [b] for b in (u.buffer for u in resolve_params(call, ctx.input_uops))])
  if (runner:=graph_cache.get(cf)) is None:
    graph_cache[cf] = runner = Device[cf.device if isinstance(cf.device, str) else cf.device[0]].graph(cf, input_uops=ctx.input_uops)
  with track_stats(ctx, call, runner.device, runner.display_name, bufs, ctx.var_vals) as t:
    t[0] = runner(bufs, ctx.var_vals, wait=DEBUG >= 2, input_uops=ctx.input_uops) # type: ignore[call-arg]

# ctx is beam value
pm_beam = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True),
   lambda ctx,call,sink: call.replace(src=(sink.replace(arg=replace(sink.arg, beam=ctx)), *call.src[1:])) if sink.arg.beam == 0 else None),
])

pm_compile = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.PROGRAM), name="ast"),), name="call", allow_any_len=True), lambda call,ast:
    call.replace(src=(to_program(ast, Device[call.device if isinstance(call.device, str) else call.device[0]].renderer), *call.src[1:]))),
])

pm_exec = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.BUFFER_VIEW, name="ast"),), name="call", allow_any_len=True), exec_view),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True), exec_copy),
  (UPat(Ops.CALL, src=(UPat((Ops.PROGRAM, Ops.SINK), name="ast"),), name="call", allow_any_len=True), exec_kernel),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="ast"),), name="call", allow_any_len=True), exec_encdec),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="graph", name="cf"),), name="call", allow_any_len=True), exec_graph),
])

def compile_linear(linear:UOp, beam=0) -> UOp:
  if (beam_val:=(beam or BEAM.value)) >= 1: linear = graph_rewrite(linear, pm_beam, ctx=beam_val, walk=True)
  return graph_rewrite(linear, pm_compile, name="precompile kernels", walk=True) if not VALIDATE_WITH_CPU else linear

def run_linear(linear:UOp, var_vals:dict[str, int]|None=None, input_uops:tuple[UOp, ...]=(), do_update_stats=True, jit=False):
  if not jit: linear = compile_linear(linear)
  ctx = ExecContext(var_vals or {}, input_uops, do_update_stats, jit)
  for call in linear.src: pm_exec.rewrite(call, ctx)
