# NOTE: this will replace jit.py, realize.py, and a lot of the boilerplate in each graph executor
from __future__ import annotations
from typing import List, Dict, Optional, Union, DefaultDict, cast, Tuple
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.dtype import DType
from tinygrad.helpers import colored, getenv, GRAPH, cpu_time_execution, DEBUG, GlobalCounters
from tinygrad.features.graph import realized_lazybuffer
from tinygrad.ops import ScheduleItem, LoadOps, BufferOps, LazyOp
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import Variable
from tinygrad.device import Buffer, JITRunner, Device, BufferXfer, BufferCopy, update_stats, BufferOptions

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

# NOTE: two syncitems aren't the same if they are in different places in the queue
@dataclass(eq=False)
class SyncItem:
  device: str
  waiters: int = 0
  def __repr__(self): return f"SyncItem({self.device}, waiters={self.waiters}, {id(self)})"

@dataclass(frozen=True)
class WaitItem:
  sync: SyncItem

@dataclass(frozen=True)
class CopyItem:
  output: Buffer
  input: Buffer

# this will interface with HWCommandQueue to replace Graph
class CommandQueue:
  def __init__(self, schedule:List[ScheduleItem], outs:List[LazyBuffer]):
    # loop through the schedule, find (real) inputs, add assign outputs, and split into different devices
    self.inputs: List[LazyBuffer] = []
    self.outputs: List[LazyBuffer] = outs[:]
    self.q: DefaultDict[str, List[Union[ScheduleItem, CopyItem, SyncItem, WaitItem]]] = defaultdict(list)

    def add_sync_item(device:str):
      if not len(self.q[device]) or not isinstance(sync_item:=self.q[device][-1], SyncItem):
        sync_item = SyncItem(device)
        self.q[device].append(sync_item)
      return sync_item

    def add_wait_item(device:str, syncitem:SyncItem):
      # if you are adding this right after a first sync, delete this one
      if len(self.q[device]) and isinstance(wi:=self.q[device][-1], WaitItem) and wi.sync.device == syncitem.device:
        self.q[device] = self.q[device][:-1]
        wi.sync.waiters -= 1
        if wi.sync.waiters == 0: self.q[wi.sync.device].remove(wi.sync)
      if (wi:=WaitItem(syncitem)) not in self.q[device]:
        syncitem.waiters += 1
        self.q[device].append(wi)

    while len(schedule):
      si = schedule.pop(0)
      assert len(set(x.device for x in si.outputs+si.inputs)) == 1 or (si.ast[0].op is LoadOps.COPY and len(si.outputs) == 1)
      queue = self.q[si.outputs[0].device]

      if si.ast[0].op is LoadOps.COPY:
        # TODO: add back copy device
        copy_device = si.outputs[0].device #+"-copy"
        add_wait_item(copy_device, add_sync_item(si.inputs[0].device))
        self.q[copy_device].append(CopyItem(si.outputs[0], si.inputs[0]))
        #add_wait_item(si.outputs[0].device, add_sync_item(copy_device))
        continue

      assert si.ast[0].op not in LoadOps
      queue.append(si)

  def __call__(self):
    #print("OUTS:", self.outputs)
    #for k,v in self.q.items():
    #  print("****", k)
    #  for si in v:
    #    print("  ", str(si)[:150])

    # this should be callable if we discover a full lazy graph has the same hash
    active_queues = list(self.q.keys())
    waiting_queues: DefaultDict[SyncItem, List[str]] = defaultdict(list)
    seen_sids = set()
    while len(active_queues):
      device = active_queues.pop(0)
      if not len(self.q[device]): continue
      si = self.q[device].pop(0)
      #print(device, si, active_queues, seen_sids)
      if isinstance(si, SyncItem):
        et = cpu_time_execution(Device[device].synchronize, enable=DEBUG>=2)
        update_stats(colored("synchronize", "RED"), 0, 0, {}, et, 1, device=device)
        if si in waiting_queues:
          active_queues += waiting_queues[si]
          waiting_queues[si].clear()
        seen_sids.add(si)
      elif isinstance(si, WaitItem):
        if si.sync not in seen_sids:
          waiting_queues[si.sync].append(device)
          continue
      elif isinstance(si, CopyItem):
        si.output.allocate()
        fxn = BufferXfer() if hasattr(Device[si.output.device].allocator, 'transfer') and \
          si.output.device.split(":")[0] == si.input.device.split(":")[0] else BufferCopy()
        fxn.exec([si.output, si.input])
      elif isinstance(si, ScheduleItem):
        for out in si.outputs: out.allocate()
        runner = Device[si.outputs[0].device].get_runner(*si.ast)
        runner.exec(si.outputs+si.inputs, si.var_vals)
      else: raise RuntimeError(f"unknown type {si}")
      active_queues.append(device)
