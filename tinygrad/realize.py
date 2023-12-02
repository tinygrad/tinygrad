from typing import List, Dict, Tuple, Generator, Optional
from tinygrad.ops import ScheduleItem, LoadOps, BufferOps
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.helpers import prod
from tinygrad.shape.symbolic import Variable

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule(schedule:List[ScheduleItem]) -> Generator[Tuple[ScheduleItem, Optional[JITRunner]], None, None]:
  while len(schedule):
    si = schedule.pop(0)
    assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.FROM, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
    if si.ast.op is LoadOps.EMPTY: yield si, None
    elif si.ast.op is LoadOps.FROM: yield si, BufferCopy
    elif si.ast.op is LoadOps.CUSTOM: yield si, CustomOp(si.ast.arg)
    else: yield si, Device[si.out.device].get_runner(si.ast)
    del si.out.op
    for v in si.out.views: del v.op

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  for si, fxn in lower_schedule(schedule):
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"

    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # TODO: this is pretty wrong actually, who knows where else this buffer is used?
    # TODO: what if an assign is required? this silently is wrong
    # TODO: this logic doesn't belong here, it should be checked in assign or at least schedule
    if si.out.output_buffer is not None:
      for i,a in enumerate(si.inputs):
        # TODO: if this is contiguous it's fine
        if a.realized == si.out.output_buffer:
          if any(not x.arg.st.contiguous for x in si.ast.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
            si.out.output_buffer = None
            break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)

    # get all the buffers
    rawbufs = [si.out.realized] + [x.realized for x in si.inputs]

    # run the function and put in JIT
    if fxn: fxn.exec(rawbufs, si.var_vals)
    assert si.out.realized.device == si.out.device, f"realized device is incorrect, {si.out.realized.device=} != {si.out.device=}"
    assert si.out.realized.dtype == si.out.dtype, f"realized dtype is incorrect, {si.out.realized.dtype=} != {si.out.dtype=}"
