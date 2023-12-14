from typing import List, Dict, Optional
from tinygrad.ops import LoadOps, ScheduleItem
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import print_tree, realized_lazybuffer
from tinygrad.helpers import prod, GlobalCounters
from tinygrad.shape.symbolic import Variable

# *** schedule running ***

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.COPY, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"  # noqa: E501
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op is LoadOps.COPY: return BufferCopy
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  return Device[si.out.device].get_runner(si.ast)

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  while len(schedule):
    si = schedule.pop(0)
    assert all(x.realized for x in si.inputs), f"can't run schedule, some inputs aren't realized {[x for x in si.inputs if x.realized is None]}"

    # get the program
    prg = lower_schedule_item(si)

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    assert si.out._base is None, "no setting of non-base LazyBuffers"
    si.out._realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)

    # run the function (put it in JIT)
    if prg: prg.exec([si.out.realized] + [x.realized for x in si.inputs], si.var_vals)
    realized_lazybuffer(si.out, GlobalCounters.kernel_count)
