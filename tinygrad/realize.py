from typing import List, Dict, Optional
from tinygrad.ops import LoadOps, ScheduleItem
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.helpers import prod
from tinygrad.shape.symbolic import Variable

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.FROM, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op is LoadOps.FROM: return BufferCopy
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  return Device[si.out.device].get_runner(si.ast)

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"

    # get the program
    prg = lower_schedule_item(si)

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)
    del si.out.op
    for v in si.out.views: del v.op

    # run the function (put it in JIT)
    if prg: prg.exec([si.out.realized] + [x.realized for x in si.inputs], si.var_vals)
