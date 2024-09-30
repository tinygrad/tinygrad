from typing import List, Dict, Optional, cast, Generator, Tuple, Union
import time, pprint
from collections import defaultdict
from dataclasses import dataclass, replace
from tinygrad.helpers import colored, getenv, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, Context, TRACEMETA, dedup
from tinygrad.helpers import NO_MEMORY_PLANNER
from tinygrad.ops import MetaOps, UOps, UOp
from tinygrad.dtype import dtypes
from tinygrad.device import Device, Buffer
from tinygrad.shape.symbolic import Variable, sym_infer, sint
from tinygrad.renderer import Renderer, Program
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.schedule import ScheduleItem

# **************** Program Creation ****************

logkerns, logkerns_level = open(getenv("LOGKERNS", ""), "a") if getenv("LOGKERNS", "") else None, getenv("LOGKERNS_LEVEL", 1)
def get_kernel(renderer:Renderer, ast:UOp) -> Kernel:
  if DEBUG >= 5:
    print(ast)
  k = Kernel(ast, opts=renderer).required_optimizations()
  if not NOOPT:
    if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
    if BEAM >= 1:
      from tinygrad.engine.search import beam_search, time_linearizer, bufs_from_lin
      kb, k_opt = Kernel(ast, opts=renderer).required_optimizations(), k
      rawbufs = bufs_from_lin(kb, allocate=False)
      if BEAM.value >= 100:
        from extra.mcts_search import mcts_search
        k = mcts_search(kb, rawbufs, BEAM.value)
      else:
        k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
      if beam_compare:=getenv("BEAM_COMPARE", 1):
        # TODO: move the HC/TC/BEAM compare to beam_search so it can be optionally cached which choice is better
        lins: List[Tuple[str, Kernel]] = [(f"beam{BEAM.value}", k), (("tc" if used_tensor_cores else "hc"), k_opt)]
        if used_tensor_cores: lins.append(("hc", Kernel(ast, opts=renderer).hand_coded_optimizations()))
        timed = sorted([(nm, tk, time_linearizer(tk, rawbufs, allow_test_size=False, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
        if DEBUG >= 3: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
        k = timed[0][1]
        if logkerns is not None and logkerns_level > 1: logkerns.writelines([f"{(lin.ast, lin.applied_opts)}\n" for (_,lin,_) in timed[1:]])
        if beam_compare == 2:
          from tinygrad import Tensor
          all_outs: List[List[Tensor]] = []
          with Context(DEBUG=0, BEAM=0, CAPTURING=0):
            rand_bufs = [Tensor.normal(buf.size, std=0.1, dtype=buf.dtype).data() if dtypes.is_float(buf.dtype) else \
                        (Tensor.randint(buf.size, low=0, high=2).cast(buf.dtype).data() if buf.dtype == dtypes.bool else \
                         Tensor.randint(buf.size, low=dtypes.min(buf.dtype), high=dtypes.max(buf.dtype), dtype=buf.dtype).data()) \
                         for buf in rawbufs]
          for _, tk in lins[::-1]:
            for buf,data in zip(rawbufs, rand_bufs): buf.ensure_allocated().copyin(data)
            time_linearizer(tk, rawbufs, allow_test_size=False, clear_l2=True, disable_cache=True)
            all_outs.append([Tensor(bytes(buf.as_buffer()), dtype=buf.dtype) for buf in rawbufs[:len(ast.src)]])
          with Context(DEBUG=0, BEAM=0, CAPTURING=0):
            for bufs in zip(*all_outs):
              for b in bufs[1:]:
                if dtypes.is_float(bufs[0].dtype):
                  # we check both atol and rtol here
                  diff_count = (((b-bufs[0]).abs() > 1e-3) * (((b-bufs[0])/bufs[0]).abs() > 1e-3)).sum().item()
                else:
                  diff_count = (b != bufs[0]).sum().item()
                if diff_count != 0:
                  raise RuntimeError(f"mismatch of {diff_count}/{b.numel()} items with type {b.dtype}, max {(b-bufs[0]).abs().max().item()}")
  if logkerns is not None: logkerns.writelines([f"{(k.ast, k.applied_opts)}\n"])
  if DEBUG >= 5: print((k.ast, k.applied_opts)) # print here to show final applied_opts for all kernels instead of just in beam_search
  return k

# **************** Runners ****************

class Runner:
  def __init__(self, display_name:str, dname:str, op_estimate:sint=0, mem_estimate:sint=0, lds_estimate:Optional[sint]=None):
    self.first_run, self.display_name, self.dname, self.op_estimate, self.mem_estimate, self.lds_estimate = \
      True, display_name, dname, op_estimate, mem_estimate, mem_estimate if lds_estimate is None else lds_estimate
  @property
  def device(self): return Device[self.dname]
  def exec(self, rawbufs:List[Buffer], var_vals:Optional[Dict[Variable, int]]=None) -> Optional[float]:
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False) -> Optional[float]:
    raise NotImplementedError("override this")

class CompiledRunner(Runner):
  def __init__(self, p:Program, precompiled:Optional[bytes]=None):
    if DEBUG >= 4: print(p.src)
    self.p:Program = p
    self.lib:bytes = precompiled if precompiled is not None else Device[p.dname].compiler.compile_cached(p.src)
    self.clprg = Device[p.dname].runtime(p.function_name, self.lib)
    super().__init__(p.name, p.dname, p.op_estimate, p.mem_estimate, p.lds_estimate)

  def __reduce__(self): return self.__class__, (self.p, self.lib)

  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False) -> Optional[float]:
    global_size, local_size = self.p.launch_dims(var_vals)
    if global_size is not None and local_size is None and all_int(self.p.global_size): # type: ignore[arg-type]
      # TODO: this is copied from get_program
      from tinygrad.engine.search import optimize_local_size
      local_size = optimize_local_size(self.clprg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)
    lra = {}
    if global_size:
      lra['global_size'] = tuple(global_size)
      assert len(global_size) == 3, "global size must have len 3"
    if local_size:
      lra['local_size'] = tuple(local_size)
      assert len(local_size) == 3, "local size must have len 3"
    return self.clprg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.p.vars), wait=wait)

class CustomOp(Runner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__(self.fxn.__name__, "CUSTOM", 0, 0)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False): self.fxn(*rawbufs)

class EmptyOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"empty {buf.size:10d} {buf.dtype}", "yellow"), buf.device)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False): pass

class ViewOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, 0, total_sz)
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.device, 'io_uring') and hasattr(src.allocator.device, 'fd')
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_disk') and disk_supports_fast_copyout and src.nbytes >= 4096:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif src.device.startswith("DISK") and hasattr(dest.allocator, 'as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator.copyout(dest.allocator.as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  def copy(self, dest, src): dest.allocator.transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.device, dest_dev=dest.allocator.device)

# **************** method cache ****************

method_cache: Dict[Tuple[str, bytes, int, int, bool], CompiledRunner] = {}
def get_runner(dname:str, ast:UOp) -> CompiledRunner:
  ckey = (dname, ast.key, BEAM.value, NOOPT.value, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (dname.split(":")[0], ast.key, BEAM.value, NOOPT.value, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, dname=dname), bret.lib)
  else:
    prg: Program = get_kernel(Device[dname].renderer, ast).to_program()
    if getenv("FUZZ_UOPS"):
      from test.external.fuzz_uops import UOpsFuzzerRunner
      return UOpsFuzzerRunner(replace(prg, dname=dname))
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, dname=dname))
  return ret

# **************** lowering functions ****************

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  bufs: List[Optional[Buffer]]
  metadata: Optional[Tuple[Metadata, ...]] = None
  def run(self, var_vals:Optional[Dict[Variable, int]]=None, wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]
    et = self.prg(bufs, var_vals if var_vals is not None else {}, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.op_estimate, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.mem_estimate, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        lds_est = sym_infer(self.prg.lds_estimate, var_vals)
        mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        ptm = (colored(f"{et*1e3:9.2f}ms", "yellow") if et > 0.01 else f"{et*1e6:9.2f}us") if et is not None else ""
        print(f"{colored(f'*** {self.prg.dname[:7]:7s} {GlobalCounters.kernel_count:4d}', 'magenta' if jit else ('green' if self.prg.first_run else None))} {self.prg.display_name+' '*(41-ansilen(self.prg.display_name))} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
              (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_est/((et or 1e-20)*1e9):9.2f} GFLOPS {mem_est/((et or 1e-20)*1e9):6.1f}|{lds_est/((et or 1e-20)*1e9):<7.1f} GB/s)" +  # noqa: E501
               f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}"))
      self.prg.first_run = False
    return et

def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  assert len(set(x.device for x in si.bufs)) == 1 or (si.ast.op is UOps.EXT and si.ast.arg[0] is MetaOps.COPY)
  if si.ast.op is UOps.SINK:
    runner = get_runner(si.outputs[0].device, si.ast)
    return ExecItem(runner, [si.bufs[x] for x in runner.p.globals], si.metadata)
  out, (op, arg) = si.outputs[0], si.ast.arg
  if op is MetaOps.COPY:
    kernel_type = BufferCopy
    if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]:
      kernel_type = BufferXfer
    return ExecItem(kernel_type(arg, out.device, si.inputs[0].device), list(si.bufs))
  if op is MetaOps.CUSTOM: return ExecItem(CustomOp(arg), list(si.bufs))
  if op is MetaOps.EMPTY: return ExecItem(EmptyOp(out), list(si.bufs))
  if op is MetaOps.VIEW: return ExecItem(ViewOp(out), list(si.bufs))
  raise RuntimeError(f"don't know how to lower {si.ast}")

def lower_schedule(schedule:List[ScheduleItem]) -> Generator[ExecItem, None, None]:
  while len(schedule):
    si = schedule.pop(0)
    try: yield lower_schedule_item(si)
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {si.ast.op}")
        print("tensor operations:")
        pprint.pprint(si.metadata, indent=2)
      raise e

# **************** main run function ****************

capturing: List = []  # put classes with an add method in here

def run_schedule(schedule:List[ScheduleItem], var_vals:Optional[Dict[Variable, int]]=None, do_update_stats=True):
  for ei in lower_schedule(schedule):
    if len(capturing) and CAPTURING: capturing[0].add(ei)
    ei.run(var_vals, do_update_stats=do_update_stats)

# **************** memory planning ****************

def _internal_memory_planner(buffers:List[Union[List[Buffer], Tuple[Buffer, ...]]], noopt_buffers=None, debug_prefix="") -> Dict[Buffer, Buffer]:
  if NO_MEMORY_PLANNER: return {}
  first_appearance, last_appearance = {}, {}
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i

  # Sort buffers by size in descending order, prioritizing largest buffers for allocation first.
  # Track free segments, each containing (start, stop, and buffer that could be reused on this segment).
  free_segs: Dict[Tuple, List[Tuple[int, int, Buffer]]] = defaultdict(list) # Dict[buffer key, Tuple[start, end, buffer to reuse on the seg]]
  def find_replace_buffer(buf, st, en):
    key = (buf.device, buf.dtype, buf.options) + ((buf.nbytes,) if not hasattr(Device[buf.device].allocator, "offset") else tuple())

    default_buf = (0, len(buffers) - 1, buf) # will return the buffer itself if the replace one is not found.
    seg_st, seg_en, seg_buf = next((free_segs[key].pop(i) for i,(sst,sen,_) in enumerate(free_segs[key]) if sst <= st and en <= sen), default_buf)

    free_segs[key] += [(seg_st, st - 1, seg_buf)] if st - 1 >= seg_st else []
    free_segs[key] += [(en + 1, seg_en, seg_buf)] if seg_en >= en + 1 else []

    return seg_buf if seg_buf.nbytes == buf.nbytes else Buffer(buf.device, buf.size, buf.dtype, base=seg_buf)

  buffer_requests = sorted([(first_appearance[buf], last_appearance[buf], buf) for buf in first_appearance.keys()], key=lambda x: -x[2].nbytes)
  assigned = {buf:find_replace_buffer(buf, st, en) for st, en, buf in buffer_requests}

  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.lb_refcount > 0 or (noopt_buffers is not None and buf.base in noopt_buffers): continue
      if buf._base is not None: assigned[buf] = Buffer(buf.device, buf.size, buf.dtype, base=assigned.get(buf.base, buf.base).base, offset=buf.offset)
      else: assigned[buf] = assigned.get(buf, buf)

  if DEBUG >= 1 and len(ak:=dedup(x for x in assigned.keys() if x._base is None)) != len(av:=dedup(x for x in assigned.values() if x._base is None)):
    print(debug_prefix+f"memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB -> {sum([x.nbytes for x in av])/1e6:.2f} MB,",
          f"{len(ak)} -> {len(av)} bufs")
  return assigned

def memory_planner(schedule:List[ScheduleItem]) -> List[ScheduleItem]:
  # Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  assigned = _internal_memory_planner([si.bufs for si in schedule],
                                      noopt_buffers={b for si in schedule if si.ast.op is not UOps.SINK for b in si.bufs})
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), si.metadata) for si in schedule]
