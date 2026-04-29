from typing import cast, Iterator, Any
import time, random, itertools, math, contextlib, weakref
from dataclasses import dataclass, replace, field
from tinygrad.helpers import colored, DEBUG, GlobalCounters, ansilen, all_int, Metadata, TRACEMETA, prod, flatten
from tinygrad.helpers import BEAM, size_to_str, time_to_str, VALIDATE_WITH_CPU, PROFILE, ProfilePointEvent, cpu_events
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer, buffers, graph_rewrite, ProgramInfo
from tinygrad.device import Device, Buffer, MultiBuffer
from tinygrad.renderer import Estimates
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.postrange import bufs_from_ast

# **************** Stat ****************

def estimate_uop(call:UOp) -> Estimates:
  if call.src[0].op is Ops.SINK: call = pm_compile.rewrite(call)

  ast = call.src[0]
  if ast.op is Ops.PROGRAM: return ast.src[0].arg.estimates or Estimates()
  if ast.op is Ops.COPY or (ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "encdec"):
    nbytes = prod(call.src[1].shape) * call.src[1].dtype.itemsize
    return Estimates(lds=nbytes, mem=nbytes)
  if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph":
    return runner.estimates if (runner:=graph_cache.get(ast)) is not None else Estimates()
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

first_run_cache:set[bytes] = set()
@contextlib.contextmanager
def track_stats(ctx:"ExecContext", call:UOp, device:str, display_name:str, bufs:list[Buffer], var_vals:dict[str, int], outputs=(0,), inputs=(1,)):
  if PROFILE: cpu_events.append(ProfilePointEvent(device, "exec", len(cpu_events), {"metadata": call.arg.metadata, "var_vals": var_vals,
                                                  "bufs": [b.trace_num for b in bufs], "name": display_name, "outputs": outputs, "inputs": inputs}))
  timing: list[float|None] = [None]
  if DEBUG >= 2: st = time.perf_counter()
  yield timing
  if not ctx.do_update_stats: return
  if DEBUG >= 2 and timing[0] is None:
    Device[device].synchronize()
    timing[0] = time.perf_counter() - st
  update_stats(display_name, device, estimate_uop(call), var_vals, timing[0], len(bufs), jit=ctx.jit, metadata=call.arg.metadata,
               first_run=call.src[0].key not in first_run_cache)
  first_run_cache.add(call.src[0].key)

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

local_size_cache: dict[bytes, tuple[int, ...]] = {}
def optimize_local_size(call:UOp, prg:UOp) -> UOp|None:
  device = prg.src[1].arg
  if prg.arg.local_size is not None or not Device[device].renderer.has_local or not all_int(prg.arg.global_size): return None

  if (local_size:=local_size_cache.get(prg.key)) is None:
    bufs = [b._buf for b in (b.allocate() for b in bufs_from_ast(prg.src[0], device))]
    rt = Device[device].runtime(prg.arg.function_name, prg.src[4].arg, *prg.arg.aux, runtimevars=prg.arg.runtimevars)
    def try_exec(local_size):
      try: return rt(*bufs, global_size=[g//l if g%l == 0 else g/l for g,l in zip(prg.arg.global_size, local_size)], local_size=local_size, wait=True)
      except Exception: return float('inf')

    MAX_WORKGROUP = 1024
    local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in prg.arg.global_size]
    local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
    best_time, best = min([(try_exec(ls), ls) for ls in random.sample(local_sizes, len(local_sizes))])
    assert not math.isinf(best_time), "all optimize_local_size exec failed"
    local_size = local_size_cache[prg.key] = tuple(best)

  new_global = tuple(g//l if g%l == 0 else g/l for g,l in zip(prg.arg.global_size, local_size))
  return call.replace(src=(prg.replace(arg=replace(prg.arg, global_size=new_global, local_size=local_size)), *call.src[1:]))

# **************** method cache ****************

runtime_cache: dict[tuple[bytes, str], Any] = {}
def get_runtime(device:str, ast:UOp):
  assert ast.op is Ops.PROGRAM and isinstance(ast.arg, ProgramInfo), "get_runtime should only be called with a PROGRAM ast"
  if (runtime:=runtime_cache.get(key:=(ast.key, device))) is None:
    if DEBUG >= 3 and ast.src[0].arg.applied_opts: print(ast.src[0].arg.applied_opts)
    if DEBUG >= 4: print(ast.src[3].arg)
    if DEBUG >= 7: Device[device].compiler.disassemble(ast.src[4].arg)
    runtime = runtime_cache[key] = Device[device].runtime(ast.arg.function_name, ast.src[4].arg, *ast.arg.aux, runtimevars=ast.arg.runtimevars)
  return runtime

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
    prg_bufs = [bufs[i].ensure_allocated() for i in ast.arg.globals]
    rt = get_runtime(device:=bufs[0].device, ast)
    global_size, local_size = ast.arg.launch_dims(var_vals)
    with track_stats(ctx, call, device, ast.arg.name, prg_bufs, var_vals, outputs=ast.arg.outs, inputs=ast.arg.ins) as tm:
      tm[0] = rt(*[b._buf for b in prg_bufs], global_size=global_size, local_size=local_size, vals=ast.arg.vals(var_vals), wait=DEBUG>=2)

def exec_validate(ctx:ExecContext, call, ast):
  import numpy as np
  for bufs, device_vars in unwrap_multi(call, resolve_params(call, ctx.input_uops)):
    bufs, dev_bufs = bufs[:len(bufs)//2], bufs[len(bufs)//2:]
    var_vals = {**ctx.var_vals, **device_vars}
    cpu_rt = get_runtime("CPU", prg:=to_program(ast.src[0], Device["CPU"].renderer))
    global_size, local_size = prg.arg.launch_dims(var_vals)
    cpu_rt(*[bufs[i].ensure_allocated()._buf for i in prg.arg.globals], global_size=global_size, local_size=local_size, vals=prg.arg.vals(var_vals))
    for i in prg.arg.outs: np.testing.assert_allclose(dev_bufs[i].ensure_allocated().numpy(), bufs[i].numpy(), rtol=1e-3, atol=1e-3)

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

# flatten LINEAR-in-LINEAR: any nested LINEAR child gets inlined into its parent's src
pm_flatten_linear = PatternMatcher([
  (UPat(Ops.LINEAR, custom_early_reject={Ops.LINEAR}, name="lin"),
   lambda lin: lin.replace(src=tuple(flatten(c.src if c.op is Ops.LINEAR else (c,) for c in lin.src)))),
])

def _validate(call:UOp, sink:UOp) -> UOp:
  params = tuple(p for p in call.src[1:] if p.op is not Ops.BIND)
  shadows = tuple(UOp.new_buffer(("CPU",)*len(p.device) if isinstance(p.device, tuple) else "CPU", prod(p.max_shape), p.dtype.base) for p in params)
  copies = tuple(p.copy_to_device(s.device).call(s, p) for s, p in zip(shadows, params))
  return UOp(Ops.LINEAR, src=copies + (call, UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(sink,), arg="validate").call(*shadows, *params)))
pm_validate = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True), _validate)]) + pm_flatten_linear

# ctx is beam value
pm_beam = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.SINK, name="sink"),), name="call", allow_any_len=True),
   lambda ctx,call,sink: call.replace(src=(sink.replace(arg=replace(sink.arg, beam=ctx)), *call.src[1:])) if sink.arg.beam == 0 else None),
])

pm_compile = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.PROGRAM), name="ast"),), name="call", allow_any_len=True), lambda call,ast:
    call.replace(src=(to_program(ast, Device[call.device if isinstance(call.device, str) else call.device[0]].renderer), *call.src[1:]))),
])

pm_optimize_local_size = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), optimize_local_size),
])

pm_exec = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.BUFFER_VIEW, name="ast"),), name="call", allow_any_len=True), exec_view),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True), exec_copy),
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True), exec_kernel),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="ast"),), name="call", allow_any_len=True), exec_encdec),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="graph", name="cf"),), name="call", allow_any_len=True), exec_graph),
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="validate", name="ast"),), name="call", allow_any_len=True), exec_validate),
])

def compile_linear(linear:UOp, beam=0, validate=False) -> UOp:
  if validate: linear = graph_rewrite(linear, pm_validate, name="validate", walk=True)
  if (beam_val:=(beam or BEAM.value)) >= 1: linear = graph_rewrite(linear, pm_beam, ctx=beam_val, walk=True)
  linear = graph_rewrite(linear, pm_compile, name="precompile kernels", walk=True)
  return graph_rewrite(linear, pm_optimize_local_size, name="optimize local size", walk=True)

def run_linear(linear:UOp, var_vals:dict[str, int]|None=None, input_uops:tuple[UOp, ...]=(), do_update_stats=True, jit=False):
  if not jit: linear = compile_linear(linear, validate=VALIDATE_WITH_CPU)
  ctx = ExecContext(var_vals or {}, input_uops, do_update_stats, jit)
  for call in linear.src: pm_exec.rewrite(call, ctx)
