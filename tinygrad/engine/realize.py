from typing import cast, Generator, Callable
import time, pprint, random, itertools, math
from dataclasses import dataclass, replace, field
from tinygrad.helpers import all_same, colored, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import DEVECTORIZE, VALIDATE_WITH_CPU, getenv, cpu_profile, PROFILE, ProfilePointEvent, cpu_events, prod, Context, unwrap
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer, graph_rewrite, print_uops, track_rewrites, KernelInfo, pyrender
from tinygrad.device import Device, Buffer
from tinygrad.renderer import Renderer, ProgramSpec, Estimates
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.codegen import full_rewrite
from tinygrad.codegen.opt import Opt

# **************** Program Creation ****************

@track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, (ret.function_name, ret.ast), ret=ret), replay=True)
def get_program(ast:UOp, renderer:Renderer|None=None, opts:list[Opt]|None=None) -> ProgramSpec:
  """
  Transform an AST into a ProgramSpec. May trigger BEAM search.

  Args:
    ast: The Ops.SINK rooted AST
    renderer: The renderer used to generate the code

  Returns:
    The ProgramSpec of the program.
  """

  if getenv("VIZ"): graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
  if DEBUG >= 5: print(pyrender(ast))

  # linearize
  if renderer is None: renderer = Device.default.renderer
  if opts is not None:
    assert ast.arg is None, "can't apply opts if sink has an arg"
    ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))
  try:
    uops = full_rewrite(ast, renderer)
  except RuntimeError as e:
    print("***** LINEARIZE FAILURE *****")
    print(e)
    print(pyrender(ast))
    raise
  assert uops[-1].op is Ops.SINK, "last uop must be sink"

  # print and render
  if DEBUG >= 6: print_uops(uops)
  src = renderer.render(uops)

  return ProgramSpec(uops[-1].arg.name if uops[-1].arg is not None else "test", src, renderer.device, ast, uops,
                     global_size=[1,1,1] if renderer.has_local or renderer.has_threads else None,
                     local_size=[1,1,1] if renderer.has_local else None)

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
  def __init__(self, p:ProgramSpec, precompiled:bytes|None=None, prg=None):
    if DEBUG >= 3: print(p.applied_opts)
    if DEBUG >= 4: print(p.src)
    self.p:ProgramSpec = p
    if precompiled is not None: self.lib = precompiled
    else:
      with cpu_profile(TracingKey(f"compile {p.name}", (p.function_name,)), "TINY"):
        self.lib = Device[p.device].compiler.compile_cached(p.src)
    if DEBUG >= 7: Device[p.device].compiler.disassemble(self.lib)
    self._prg = Device[p.device].runtime(p.function_name, self.lib) if prg is None else prg
    super().__init__(p.name, p.device, p.estimates)

  def __reduce__(self): return self.__class__, (self.p, self.lib)

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None, wait=False) -> float|None:
    if var_vals is None: var_vals = {}
    has_local = Device[self.p.device].renderer.has_local
    global_size, local_size = self.p.launch_dims(var_vals)
    if has_local and global_size is not None and local_size is None and all_int(self.p.global_size): # type: ignore[arg-type]
      local_size = optimize_local_size(self._prg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)
    lra = {}
    if global_size:
      lra['global_size'] = tuple(global_size)
      assert len(global_size) == 3, "global size must have len 3"
    if local_size:
      lra['local_size'] = tuple(local_size)
      assert len(local_size) == 3, "local size must have len 3"
    return self._prg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k.expr] for k in self.p.vars), wait=wait)

class ViewOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.dev, 'io_uring') and \
      getattr(src.allocator.dev, 'fd', None) is not None and dest.allocator.supports_copy_from_disk
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_disk') and disk_supports_fast_copyout and src.nbytes >= 4096:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif (src.device.startswith("DISK") or src.device.startswith("TINYFS")) and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
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

# **************** method cache ****************

method_cache: dict[tuple[str, type, bytes, tuple[int, ...], bool], CompiledRunner] = {}
def get_runner(device:str, ast:UOp) -> CompiledRunner:
  # TODO: this should be all context relevant to rendering
  context = (BEAM.value, NOOPT.value, DEVECTORIZE.value)
  ckey = (device, type(Device[device].compiler), ast.key, context, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (device.split(":")[0], type(Device[device].compiler), ast.key, context, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device), bret.lib)
  else:
    prg: ProgramSpec = get_program(ast, Device[device].renderer)
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

# **************** lowering functions ****************

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  bufs: list[Buffer|None]
  metadata: tuple[Metadata, ...]|None = None
  fixedvars: dict[str, int] = field(default_factory=dict)
  def run(self, _var_vals:dict[str, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    bufs = [unwrap(x) for x in self.bufs] if jit else [unwrap(x).ensure_allocated() for x in self.bufs]
    if PROFILE:
      payload = {"metadata":self.metadata, "var_vals":var_vals, "bufs":[b.trace_num for b in bufs], "name":self.prg.display_name}
      payload["outputs"], payload["inputs"] = (self.prg.p.outs, self.prg.p.ins) if isinstance(self.prg, CompiledRunner) else ([0], [1])
      cpu_events.append(ProfilePointEvent(self.prg.device, "exec", len(cpu_events), payload))
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.estimates.ops, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.estimates.mem, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        def units_to_str(x:float, units:dict, colors:list[str]=['GREEN', 'green', 'yellow', 'yellow', 'RED']) -> str:
          def align(x, sym, color, sym_width=max(map(len, units.keys()))): return colored(f"{x:3}{sym:{sym_width}}", color)
          return next((align(int(x//val), sym, colors[i]) for i, (sym, val) in enumerate(units.items()) if x//val>0),
                      align(0, list(units.keys())[-1], colors[-1]))
        lds_est = sym_infer(self.prg.estimates.lds, var_vals)
        mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        header_color = 'magenta' if jit else ('green' if self.prg.first_run else None)
        mem_str = units_to_str(mem_est, {"TB":1e12, "GB":1e9, "MB":1e6, "KB":1e3, "B":1})
        ops_str = units_to_str(op_est, {"TFLOPs":1e12, "GFLOPs":1e9, "MFLOPs":1e6, "KFLOPs":1e3, "FLOPs":1})
        time_str = "" if et is None else units_to_str(et,  {"s":1, "ms":1e-3, "us":1e-6, "ns":1e-9})
        flops_str = units_to_str(op_est/(et or 1e-20), {"TFLOPS":1e12, "GFLOPS":1e9, "MFLOPS":1e6, "KFLOPS":1e3, "FLOPS":1})
        membw_str = units_to_str(mem_est/(et or 1e-20), {"TB/s":1e12, "GB/s":1e9, "MB/s":1e6, "KB/s":1e3, "B/s":1})
        print(f"{colored(f'*** {self.prg.device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
              f" {self.prg.display_name+' '*(46-ansilen(self.prg.display_name))} args={len(bufs):2d}"+
              f" mem={mem_str} ops={ops_str} tm={time_str} FLOPS={flops_str} membw={membw_str}"+
              f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}")
      self.prg.first_run = False
    return et

# NOTE: ctx is the buffers
si_lowerer = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
  (UPat(Ops.BUFFER_VIEW), lambda ctx: (ViewOp(ctx[0]), list(ctx))),
  (UPat(Ops.COPY, name="copy"), lambda ctx,copy: ((BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(Device[ctx[0].device].allocator, '_transfer') and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device)), list(ctx))),
])
def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata, si.fixedvars)

def lower_schedule(schedule:list[ScheduleItem]) -> Generator[tuple[ScheduleItem, ExecItem], None, None]:
  while len(schedule):
    si = schedule.pop(0)
    try: yield (si, lower_schedule_item(si))
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {si.ast.op}")
        print("tensor operations:")
        pprint.pprint(si.metadata, indent=2)
      raise e

# **************** main run function ****************

capturing: list = []  # put classes with an add method in here

def run_schedule(schedule:list[ScheduleItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  for si, ei in lower_schedule(schedule):
    if len(capturing) and CAPTURING: capturing[0].add(ei)
    if VALIDATE_WITH_CPU and si.ast.op is Ops.SINK:
      # copy in allocated buffers from the GPU
      nb: tuple[Buffer, ...] = tuple(Buffer("CPU", b.size, b.dtype) for b in si.bufs)
      for cpu_b, gpu_b in zip(nb, si.bufs):
        if gpu_b.is_allocated(): cpu_b.ensure_allocated().copyin(gpu_b.as_buffer())

      # run on GPU
      ei.run(var_vals, do_update_stats=do_update_stats)

      # validate the output buffers match (NOTE: this is assuming the output is buffer 0)
      with Context(BEAM=0): lower_schedule_item(ScheduleItem(si.ast, nb, si.metadata, si.fixedvars)).run(var_vals, do_update_stats=do_update_stats)
      import numpy as np
      np.testing.assert_allclose(si.bufs[0].numpy(), nb[0].numpy(), rtol=1e-3, atol=1e-3)
    else:
      ei.run(var_vals, do_update_stats=do_update_stats)
