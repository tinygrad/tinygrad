from typing import List, Dict, Optional, cast, Generator, DefaultDict, Tuple, Iterable
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.dtype import DType
from tinygrad.helpers import colored, getenv, dedup, DEBUG, GlobalCounters, ansilen
from tinygrad.ops import ScheduleItem, BufferOps, LoadOps, copy_ast
from tinygrad.device import Runner, Device, BufferCopy, BufferXfer
from tinygrad.buffer import Buffer
from tinygrad.shape.symbolic import Variable, sym_infer

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  rawbufs: List[Optional[Buffer]]
  def run(self, var_vals:Optional[Dict[Variable, int]]=None, wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    et = self.prg([cast(Buffer, x).ensure_allocated() for x in self.rawbufs], var_vals if var_vals is not None else {}, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_estimate:=sym_infer(self.prg.op_estimate, var_vals))
      GlobalCounters.global_mem += (mem_estimate:=sym_infer(self.prg.mem_estimate, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        ptm = (colored(f"{et*1e3:9.2f}ms", "yellow") if et > 0.01 else f"{et*1e6:9.2f}us") if et is not None else ""
        print(f"{colored(f'*** {self.prg.dname[:7]:7s} {GlobalCounters.kernel_count:4d}', 'magenta' if jit else ('green' if self.prg.first_run else None))} {self.prg.display_name+' '*(38-ansilen(self.prg.display_name))} arg {len(self.rawbufs):3d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
              (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))  # noqa: E501
      self.prg.first_run = False
    return et

class CustomOp(Runner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__(self.fxn.__name__, "CUSTOM", 0, 0)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False): self.fxn(*rawbufs)

class EmptyOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"empty {buf.size:10d} {buf.dtype}", "yellow"), buf.device)
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False): pass

def lower_schedule_item(si:ScheduleItem) -> Runner:
  assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or si.ast[0].op is LoadOps.COPY
  if si.ast[0].op is BufferOps.STORE: return Device[si.outputs[0].device].get_runner(*si.ast)
  assert len(si.ast) == 1 and len(si.outputs) == 1, "only ASTRunner supports multioutput"
  out, ast = si.outputs[0], si.ast[0]
  if ast.op is LoadOps.COPY:
    kernel_type = BufferCopy
    if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]:
      if getenv("USE_COPY_KERNEL"): return Device[out.device].get_runner(copy_ast(ast.arg))
      kernel_type = BufferXfer
    return kernel_type(ast.arg, out.device, si.inputs[0].device)
  if ast.op is LoadOps.CUSTOM: return CustomOp(ast.arg)
  if ast.op is LoadOps.EMPTY: return EmptyOp(out)
  raise RuntimeError(f"don't know how to lower {ast}")

def lower_schedule(schedule:List[ScheduleItem]) -> Generator[ExecItem, None, None]:
  while len(schedule): yield ExecItem(lower_schedule_item(si:=schedule.pop(0)), list(si.outputs+si.inputs))

capturing: List = []  # put classes with an add method in here

def _internal_memory_planner(buffers:List[Iterable[Buffer]], debug_prefix="") -> Dict[Buffer, Buffer]:
  last_appearance = {}
  for i,u in enumerate(buffers):
    for buf in u: last_appearance[buf] = i

  # LRU algorithm
  assigned: Dict[Buffer, Buffer] = {}
  local_cache: DefaultDict[Tuple[str, int, DType], List[Buffer]] = defaultdict(list)
  for i,u in enumerate(buffers):
    for buf in u:
      # all unallocated unparented buffers are fair game to replace
      if buf.is_allocated() or buf.lb_refcount > 0: continue
      key = (buf.device, buf.size, buf.dtype)
      if buf not in assigned:
        if len(ll:=local_cache[key]): assigned[buf] = ll.pop()
        else: assigned[buf] = Buffer(*key)
      if i == last_appearance[buf]:
        local_cache[key].append(assigned[buf])

  if DEBUG >= 1 and len(ak:=dedup(assigned.keys())) != len(av:=dedup(assigned.values())):
    print(debug_prefix+f"memory reduced from {sum([x.nbytes for x in ak])/1e6:.2f} MB -> {sum([x.nbytes for x in av])/1e6:.2f} MB,",
          f"{len(ak)} -> {len(av)} bufs")
  return assigned

def memory_planner(schedule:List[ScheduleItem]) -> List[ScheduleItem]:
  assigned = _internal_memory_planner([si.outputs+si.inputs for si in schedule])
  return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.outputs),
                               tuple(assigned.get(x, x) for x in si.inputs)) for si in schedule]

def run_schedule(schedule:List[ScheduleItem], var_vals:Optional[Dict[Variable, int]]=None):
  for ei in lower_schedule(schedule):
    if len(capturing): capturing[0].add(ei)
    ei.run(var_vals)
