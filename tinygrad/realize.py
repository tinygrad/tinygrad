from typing import List, Dict, Tuple, Generator
from dataclasses import dataclass
from tinygrad.ops import LoadOps
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.helpers import prod
from tinygrad.shape.symbolic import Variable
from tinygrad.lazy import ScheduleItem, LazyBuffer

@dataclass(frozen=True)
class LoweredItem:
  prg: JITRunner
  out: LazyBuffer
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule(schedule:List[ScheduleItem], disable_logging=False) -> Generator[LoweredItem, None, None]:
  while len(schedule):
    si = schedule.pop(0)
    assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.FROM, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
    if not disable_logging: log_schedule_item(si)
    if si.ast.op is LoadOps.FROM: prg = BufferCopy
    elif si.ast.op is LoadOps.CUSTOM: prg = CustomOp(si.ast.arg)
    elif si.ast.op is LoadOps.EMPTY: prg = None
    else: prg = Device[si.out.device].get_runner(si.ast)
    del si.out.op
    for v in si.out.views: del v.op
    yield LoweredItem(prg, si.out, si.inputs, si.var_vals)

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  for li in lower_schedule(schedule, disable_logging):
    assert all(x.realized for x in li.inputs), "can't run schedule, some inputs aren't realized"

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    li.out.realized = li.out.output_buffer if li.out.output_buffer is not None else \
      Buffer(li.out.device, prod((s if isinstance(s, int) else s.max for s in li.out.shape)), li.out.dtype)

    # get all the buffers
    rawbufs = [li.out.realized] + [x.realized for x in li.inputs]

    # run the function and put in JIT
    if li.prg: li.prg.exec(rawbufs, li.var_vals)
    assert li.out.realized.device == li.out.device, f"realized device is incorrect, {li.out.realized.device=} != {li.out.device=}"
    assert li.out.realized.dtype == li.out.dtype, f"realized dtype is incorrect, {li.out.realized.dtype=} != {li.out.dtype=}"
