from typing import List, Dict, Optional
from tinygrad.helpers import colored
from tinygrad.ops import ScheduleItem, BufferOps, LoadOps
from tinygrad.device import JITRunner, Device, BufferCopy, BufferXfer, update_stats
from tinygrad.buffer import Buffer
from tinygrad.shape.symbolic import Variable

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

class EmptyOp(JITRunner):
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    update_stats(colored(f"empty {rawbufs[0].size:10d} {rawbufs[0].dtype}", "yellow"), 0, 0, {}, jit, 1, device=rawbufs[0].device)

def lower_schedule_item(si:ScheduleItem) -> JITRunner:
  assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or si.ast[0].op is LoadOps.COPY
  if si.ast[0].op is BufferOps.STORE: return Device[si.outputs[0].device].get_runner(*si.ast)
  assert len(si.ast) == 1 and len(si.outputs) == 1, "only ASTRunner supports multioutput"
  out, ast = si.outputs[0], si.ast[0]
  if ast.op is LoadOps.COPY:
    if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]: return BufferXfer()
    return BufferCopy()
  if ast.op is LoadOps.CUSTOM: return CustomOp(ast.arg)
  if ast.op is LoadOps.EMPTY: return EmptyOp()
  raise RuntimeError(f"don't know how to lower {ast}")

def run_schedule(schedule:List[ScheduleItem], var_vals:Optional[Dict[Variable, int]] = None):
  while len(schedule):
    si = schedule.pop(0)

    # get the program
    prg = lower_schedule_item(si)

    # allocate output buffers
    for out in si.outputs: out.ensure_allocated()

    # run the function (put it in JIT)
    prg.exec(list(si.outputs+si.inputs), var_vals if var_vals is not None else {})