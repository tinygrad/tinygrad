from typing import List, Dict, Optional, cast, Generator, Tuple
import time
from dataclasses import dataclass
from tinygrad.helpers import colored, getenv, DEBUG, GlobalCounters, ansilen
from tinygrad.ops import ScheduleItem, BufferOps, LoadOps, copy_ast, LazyOp
from tinygrad.device import Runner, Device
from tinygrad.device import Buffer
from tinygrad.shape.symbolic import Variable, sym_infer

# **************** Runners ****************

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
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_fd') and src.nbytes >= 4096 and hasattr(src.allocator.device, 'fd'):
      dest.allocator.copy_from_fd(dest._buf, src.allocator.device.fd, src._buf.offset, src.nbytes)
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
  def copy(self, dest, src):
    if hasattr(dest.allocator.device, "track_cross_buffer") and hasattr(src.allocator, "track_cross_device"):
      dest.allocator.device.track_cross_buffer.append(src)
      src.allocator.track_cross_device.add(dest.allocator.device)
    dest.allocator.transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.device, dest_dev=dest.allocator.device)

# **************** lowering functions ****************

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  bufs: List[Optional[Buffer]]
  def run(self, var_vals:Optional[Dict[Variable, int]]=None, wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]
    et = self.prg(bufs, var_vals if var_vals is not None else {}, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_estimate:=sym_infer(self.prg.op_estimate, var_vals))
      GlobalCounters.global_mem += (mem_estimate:=sym_infer(self.prg.mem_estimate, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        ptm = (colored(f"{et*1e3:9.2f}ms", "yellow") if et > 0.01 else f"{et*1e6:9.2f}us") if et is not None else ""
        print(f"{colored(f'*** {self.prg.dname[:7]:7s} {GlobalCounters.kernel_count:4d}', 'magenta' if jit else ('green' if self.prg.first_run else None))} {self.prg.display_name+' '*(38-ansilen(self.prg.display_name))} arg {len(self.bufs):3d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
              (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))  # noqa: E501
      self.prg.first_run = False
    return et

def lower_runner(dname:str, ast:Tuple[LazyOp, ...], bufs) -> ExecItem:
  runner = Device[dname].get_runner(*ast)
  # TODO: globals isn't on the stupid diskrunner, remove the need for it
  return ExecItem(runner, [bufs[x[0]] for x in runner.p.globals] if hasattr(runner, 'p') else bufs)

def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  assert len(set(x.device for x in si.bufs)) == 1 or si.ast[0].op is LoadOps.COPY
  if si.ast[0].op is BufferOps.STORE: return lower_runner(si.outputs[0].device, si.ast, si.bufs)
  assert len(si.ast) == 1 and len(si.outputs) == 1, "only ASTRunner supports multioutput"
  out, ast = si.outputs[0], si.ast[0]
  if ast.op is LoadOps.COPY:
    kernel_type = BufferCopy
    if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]:
      if getenv("USE_COPY_KERNEL"): return lower_runner(out.device, (copy_ast(ast.arg),), si.bufs)
      kernel_type = BufferXfer
    return ExecItem(kernel_type(ast.arg, out.device, si.inputs[0].device), list(si.bufs))
  if ast.op is LoadOps.CUSTOM: return ExecItem(CustomOp(ast.arg), list(si.bufs))
  if ast.op is LoadOps.EMPTY: return ExecItem(EmptyOp(out), list(si.bufs))
  if ast.op is LoadOps.VIEW: return ExecItem(ViewOp(out), list(si.bufs))
  raise RuntimeError(f"don't know how to lower {ast}")

def lower_schedule(schedule:List[ScheduleItem]) -> Generator[ExecItem, None, None]:
  while len(schedule): yield lower_schedule_item(schedule.pop(0))

# **************** main run function ****************

capturing: List = []  # put classes with an add method in here

def run_schedule(schedule:List[ScheduleItem], var_vals:Optional[Dict[Variable, int]]=None):
  for ei in lower_schedule(schedule):
    if len(capturing): capturing[0].add(ei)
    ei.run(var_vals)
