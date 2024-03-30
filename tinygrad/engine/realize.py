from typing import List, Dict, Optional
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters
from tinygrad.device import Device, Buffer, BufferCopy, BufferXfer, JITRunner, update_stats
from tinygrad.features.graph import realized_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH, cpu_time_execution, DEBUG
from tinygrad.shape.symbolic import Variable

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

class SyncOp(JITRunner):
  def __init__(self, device):
    self.device, self.dname = Device[device], device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    et = cpu_time_execution(self.device.synchronize, enable=wait or DEBUG >= 1)
    update_stats(colored("synchronize", "RED"), 0, 0, {}, et, 1, device=self.dname)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or si.ast[0].op in {LoadOps.COPY, LoadOps.WAIT}
  if si.ast[0].op is BufferOps.STORE: return Device[si.outputs[0].device].get_runner(*si.ast)
  assert len(si.ast) == 1 and len(si.outputs) == 1, "only ASTRunner supports multioutput"
  out, ast = si.outputs[0], si.ast[0]
  if ast.op is LoadOps.COPY:
    if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]: return BufferXfer()
    return BufferCopy()
  if ast.op is LoadOps.CUSTOM: return CustomOp(ast.arg)
  if ast.op is LoadOps.SYNC: return SyncOp(out.device)
  return None

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def run_schedule(schedule:List[ScheduleItem]):
  while len(schedule):
    si = schedule.pop(0)
    if logops and si.ast[0].op not in LoadOps and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")

    # get the program
    prg = lower_schedule_item(si)
    dont_allocate = getattr(prg, "skip_allocation", False)

    for out in si.outputs:
      # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
      if out.size > 0 and not dont_allocate and out.op is not LoadOps.ASSIGN: out.buffer.allocate()

    # run the function (put it in JIT)
    real_buffers = [x.buffer for x in si.outputs+si.inputs if x.size != 0]
    assert dont_allocate or all(hasattr(x, "_buf") for x in real_buffers), f"can't run, some inputs aren't realized {real_buffers}"
    if prg: prg.exec(real_buffers, si.var_vals)
    elif (out:=si.outputs[0]).size > 0: update_stats(colored(f"empty {out.st.size:10d} {out.dtype}", "yellow"), 0, 0, {}, None, 1, device=out.device)
    if GRAPH:
      for out in si.outputs: realized_lazybuffer(out, GlobalCounters.kernel_count)
