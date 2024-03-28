# NOTE: this will replace jit.py, realize.py, and a lot of the boilerplate in each graph executor
from __future__ import annotations
from typing import List, Dict, Optional, Union, DefaultDict, cast, Set
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.helpers import colored, getenv, GRAPH, cpu_time_execution, DEBUG
from tinygrad.features.graph import realized_lazybuffer
from tinygrad.ops import ScheduleItem, LoadOps, BufferOps, GlobalCounters
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import Variable
from tinygrad.device import Buffer, JITRunner, Device, BufferXfer, BufferRead, BufferCopy, update_stats

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or si.ast[0].op is LoadOps.COPY
  if si.ast[0].op is BufferOps.STORE: return Device[si.outputs[0].device].get_runner(*si.ast)
  assert len(si.ast) == 1 and len(si.outputs) == 1, "only ASTRunner supports multioutput"
  out, ast = si.outputs[0], si.ast[0]
  if ast.op is LoadOps.COPY:
    if hasattr(Device[out.device].allocator, 'transfer') and out.device.split(":")[0] == si.inputs[0].device.split(":")[0]: return BufferXfer()
    if si.inputs[0].device.startswith("DISK"): return BufferRead()
    return BufferCopy()
  if ast.op is LoadOps.CUSTOM: return CustomOp(ast.arg)
  return None

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def exec_si(si):
  prg = lower_schedule_item(si)
  if logops and si.ast[0].op not in LoadOps and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")

  for out in si.outputs:
    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    if out.size > 0:
      if out.op is LoadOps.ASSIGN and out.srcs[1].base.realized is not None:
        # if the buffer isn't realized, it might be a const or something. this is fine
        out.realized = out.srcs[1].base.realized
      else:
        out.realized = Buffer(out.device, out.size, out.dtype, "PLACEHOLDER" if getattr(prg, "skip_allocation", False) else None)
      del out.srcs

  # run the function (put it in JIT)
  real_buffers = [x.realized for x in si.outputs+si.inputs if x.size != 0]
  assert all(x is not None for x in real_buffers), f"can't run, some inputs aren't realized {real_buffers}"
  if prg: prg.exec(cast(List[Buffer], real_buffers), si.var_vals)
  elif (out:=si.outputs[0]).size > 0: update_stats(colored(f"empty {out.st.size:10d} {out.dtype}", "yellow"), 0, 0, {}, None, 1, device=out.device)
  if GRAPH:
    for out in si.outputs: realized_lazybuffer(out, GlobalCounters.kernel_count)

@dataclass(frozen=True)
class SyncItem:
  sid: int

@dataclass(frozen=True)
class WaitItem:
  sid: int

# this will interface with HWCommandQueue to replace Graph
class CommandQueue:
  def __init__(self, schedule:List[ScheduleItem], outs:Set[LazyBuffer]):
    sync_sid = 0

    # loop through the schedule, find (real) inputs, add assign outputs, and split into different devices
    self.q: DefaultDict[str, List[Union[ScheduleItem, SyncItem, WaitItem]]] = defaultdict(list)
    while len(schedule):
      si = schedule.pop(0)
      assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or (si.ast[0].op is LoadOps.COPY and len(si.outputs) == 1)
      device = si.outputs[0].device
      if si.ast[0].op is LoadOps.COPY and si.inputs[0].device not in {"EXT", "DISK"}:
        # add sync between the devices
        if not len(self.q[si.inputs[0].device]) or not isinstance(sync_item:=self.q[si.inputs[0].device][-1], SyncItem):
          sync_item = SyncItem(sync_sid)
          self.q[si.inputs[0].device].append(sync_item)
          sync_sid += 1
        self.q[device].append(WaitItem(sync_item.sid))
      self.q[device].append(si)

  def __call__(self):
    # this should be callable if we discover a full lazy graph has the same hash
    active_queues = list(self.q.keys())
    waiting_queues: DefaultDict[int, List[str]] = defaultdict(list)
    seen_sids = set()
    while len(active_queues):
      device = active_queues.pop(0)
      if not len(self.q[device]): continue
      si = self.q[device].pop(0)
      if isinstance(si, SyncItem):
        et = cpu_time_execution(Device[device].synchronize, enable=DEBUG>=2)
        update_stats(colored("synchronize", "RED"), 0, 0, {}, et, 1, device=device)
        if si.sid in waiting_queues:
          active_queues += waiting_queues[si.sid]
          waiting_queues[si.sid].clear()
        seen_sids.add(si.sid)
      elif isinstance(si, WaitItem):
        if si.sid not in seen_sids:
          waiting_queues[si.sid].append(device)
          continue
      else:
        exec_si(si)
      active_queues.append(device)
