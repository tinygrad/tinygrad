from typing import List, Dict, Optional, cast
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters
from tinygrad.device import Device, Buffer, BufferCopy, BufferXfer, BufferRead, JITRunner, update_stats, InterpretedASTRunner, Compiled
from tinygrad.graph import print_tree, realized_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH, cpu_time_execution, DEBUG
from tinygrad.shape.symbolic import Variable

# *** schedule running ***

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

class SyncEvent(JITRunner):
  def __init__(self, lb):
    self.lb, self.device, self.dname = lb, Device[lb.device], lb.device
    assert hasattr(self.device, "event_create")
    setattr(self.lb, "event", self.device.event_create())
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    assert hasattr(self.device, "event_record")
    self.device.event_record(self.lb.event)
    update_stats(colored("sync", "red"), 0, 0, {}, None, 1, device=self.dname)

class WaitEvent(JITRunner):
  def __init__(self, device, lb_sync):
    self.lb_sync, self.device, self.dname = lb_sync, Device[device], device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    assert hasattr(self.device, "event_wait")
    self.device.event_wait(self.lb_sync.event)
    update_stats(colored("wait", "RED"), 0, 0, {}, None, 1, device=self.dname)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op in {LoadOps.COPY, LoadOps.WAIT}, \
    f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op is LoadOps.COPY:
    if hasattr(Device[si.out.device].allocator, 'transfer') and type(Device[si.out.device]) is type(Device[si.inputs[0].device]): return BufferXfer()
    if si.inputs[0].device.startswith("DISK"): return BufferRead()
    return BufferCopy()
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  # TODO: this doesn't have to be only HIP, check if it has the event functions
  if si.ast.op in {LoadOps.SYNC, LoadOps.WAIT} and si.out.device.startswith("HIP") and si.inputs[0].device.startswith("HIP"):
    if si.ast.op is LoadOps.SYNC: return SyncEvent(si.out)
    if si.ast.op is LoadOps.WAIT: return WaitEvent(si.out.device, si.inputs[0])
  else:
    if si.ast.op is LoadOps.SYNC: return SyncOp(si.out.device) if isinstance(Device[si.out.device], Compiled) else None
    if si.ast.op is LoadOps.WAIT: return None
  return Device[si.out.device].get_runner(si.ast)

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def run_schedule(schedule:List[ScheduleItem]):
  while len(schedule):
    si = schedule.pop(0)
    if logops and si.ast.op not in LoadOps: logops.write(str(si.ast)+"\n")

    # get the program
    prg = lower_schedule_item(si)

    # invalidate the output buffer if there's a non contig usage of it in inputs
    if si.out.output_buffer is not None:
      for i,a in enumerate(si.inputs):
        if a.realized == si.out.output_buffer:
          if any(not x.arg.st.contiguous for x in si.ast.lazyops if x.op is BufferOps.LOAD and x.arg.idx == i+1):
            si.out.output_buffer = None
            break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    if si.out.size > 0:
      si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
        Buffer(si.out.device, si.out.size, si.out.dtype, "PLACEHOLDER" if isinstance(prg, InterpretedASTRunner) else None)
      del si.out.srcs

    # run the function (put it in JIT)
    real_buffers = [x.realized for x in (si.out,)+si.inputs if x.size != 0]
    assert all(x is not None for x in real_buffers), f"can't run, some inputs aren't realized {real_buffers}"
    if prg: prg.exec(cast(List[Buffer], real_buffers), si.var_vals)
    elif si.out.size > 0: update_stats(colored(f"empty {si.out.st.size:10d} {si.out.dtype}", "yellow"), 0, 0, {}, None, 1, device=si.out.device)
    if GRAPH: realized_lazybuffer(si.out, GlobalCounters.kernel_count)
