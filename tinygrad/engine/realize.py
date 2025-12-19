from typing import Callable
import time, random, itertools, math
from dataclasses import replace
from tinygrad.helpers import all_same, colored, DEBUG, BEAM, NOOPT, all_int, CAPTURING, TracingKey
from tinygrad.helpers import DEVECTORIZE, VALIDATE_WITH_CPU, getenv, cpu_profile, prod, Context
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, graph_rewrite, print_uops, track_rewrites, KernelInfo, pyrender
from tinygrad.device import Device, Buffer
from tinygrad.renderer import Renderer, ProgramSpec, Estimates
from tinygrad.engine.schedule import ScheduleItem
from tinygrad.codegen import full_rewrite
from tinygrad.codegen.opt import Opt

# **************** Program Creation ****************

@track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, (ret.function_name, ret.ast), ret=ret), replay=True)
def get_program(ast:UOp, renderer:Renderer, opts:list[Opt]|None=None) -> ProgramSpec:
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

class EncDec(Runner):
  def __init__(self, encdec:UOp, total_sz:int, device:str):
    self.shape, self.pos_var = encdec.arg[0], encdec.variables()[0].expr
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

# **************** lowering ****************

# NOTE: ctx is the buffers
si_lowerer = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
  (UPat(Ops.BUFFER_VIEW), lambda ctx: (ViewOp(ctx[0]), list(ctx))),
  (UPat(Ops.COPY, name="copy"), lambda ctx,copy: ((BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(Device[ctx[0].device].allocator, '_transfer') and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device)), list(ctx))),
  (UPat(Ops.ENCDEC, name="encdec"), lambda ctx,encdec: ((EncDec(encdec, ctx[0].nbytes, ctx[1].device)), list(ctx))),
])

# **************** main run function ****************

capturing: list = []  # put classes with an add method in here

def run_schedule(schedule:list[ScheduleItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  for si in schedule:
    si._lower()  # ensure prg is populated before capturing
    if len(capturing) and CAPTURING: capturing[0].add(si)
    if VALIDATE_WITH_CPU and si.ast.op is Ops.SINK:
      # copy in allocated buffers from the GPU
      bufs = [b for b in si.bufs if b is not None]
      nb: list[Buffer|None] = [Buffer("CPU", b.size, b.dtype) for b in bufs]
      for cpu_b, gpu_b in zip(nb, bufs):
        if cpu_b is not None and gpu_b.is_allocated(): cpu_b.ensure_allocated().copyin(gpu_b.as_buffer())

      # run on GPU
      si.run(var_vals, do_update_stats=do_update_stats)

      # validate the output buffers match (NOTE: this is assuming the output is buffer 0)
      with Context(BEAM=0): ScheduleItem(si.ast, nb, si.metadata, si.fixedvars).run(var_vals, do_update_stats=do_update_stats)
      import numpy as np
      assert nb[0] is not None
      np.testing.assert_allclose(bufs[0].numpy(), nb[0].numpy(), rtol=1e-3, atol=1e-3)
    else:
      si.run(var_vals, do_update_stats=do_update_stats)
