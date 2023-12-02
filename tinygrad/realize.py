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

def lower_scheduleitem(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.FROM, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
  if si.ast.op is LoadOps.EMPTY: prg: Optional[JITRunner] = None
  elif si.ast.op is LoadOps.FROM: prg = BufferCopy
  elif si.ast.op is LoadOps.CUSTOM: prg = CustomOp(si.ast.arg)
  else: prg = Device[si.out.device].get_runner(si.ast)
  del si.out.op
  for v in si.out.views: del v.op
  return prg

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)

    # get all the buffers
    rawbufs = [si.out.realized] + [x.realized for x in si.inputs]

    # run the function and put in JIT
    if prg := lower_scheduleitem(si): prg.exec(rawbufs, si.var_vals)

