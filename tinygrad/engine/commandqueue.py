# NOTE: this will replace jit.py, realize.py, and a lot of the boilerplate in each graph executor
from __future__ import annotations
from typing import List, Dict, Union, DefaultDict
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.helpers import colored, cpu_time_execution, DEBUG, dedup, flatten
from tinygrad.ops import ScheduleItem, LoadOps, BufferOps
from tinygrad.shape.symbolic import Variable
from tinygrad.device import Buffer, JITRunner, Device, BufferXfer, BufferCopy, update_stats

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

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
  def __init__(self, schedule:List[ScheduleItem], outputs:List[Buffer]):
    # allocate all outputs
    for out in outputs:
      if not hasattr(out, "_buf"): out.allocate()

    # all unallocated buffers are fair game to replace
    unallocated = [[x for x in (si.outputs+si.inputs) if not hasattr(x, '_buf')] for si in schedule]
    all_unallocated = sorted(dedup(flatten(unallocated)), key=lambda x: x.nbytes, reverse=True)
    def to_range(lst): return list(range(min(lst), max(lst)+1))
    liveness = {buf:to_range([i for i,b in enumerate(unallocated) if buf in b]) for buf in all_unallocated}

    # greedy algorithm to assign buffers to others that don't liveness overlap
    assigned = {}
    while len(all_unallocated):
      buf = all_unallocated.pop(0)
      if buf.device.startswith("DISK"): continue  # don't mess with DISK
      assert not hasattr(buf, '_buf'), "allocated?"
      if buf in assigned: continue
      assigned[buf] = nbuf = Buffer(buf.device, buf.size, buf.dtype)
      used_range = liveness[buf]
      # see which other buffers we can assign it to
      for x in all_unallocated:
        if x.device == buf.device and x.dtype == buf.dtype and x.size <= buf.size:
          if not any(i in used_range for i in liveness[x]):
            used_range += liveness[x]
            assigned[x] = nbuf

    # do the buffer replacements in the schedule
    schedule = [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.outputs),
                             tuple(assigned.get(x, x) for x in si.inputs), si.var_vals) for si in schedule]

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

      # NOTE: LoadOps.EMPTY and LoadOps.CUSTOM are making it here
      queue.append(si)

  def __call__(self):
    active_queues = list(self.q.keys())
    waiting_queues: DefaultDict[SyncItem, List[str]] = defaultdict(list)
    seen_sids = set()
    while len(active_queues):
      device = active_queues.pop(0)
      if not len(self.q[device]): continue
      si = self.q[device].pop(0)
      #print(device, si, active_queues, seen_sids)
      if isinstance(si, SyncItem):
        # don't sync if there's other options
        if all(isinstance(self.q[x][0], SyncItem) for x in active_queues if len(self.q[x])):
          et = cpu_time_execution(Device[device].synchronize, enable=DEBUG>=2)
          update_stats(colored("synchronize", "RED"), 0, 0, {}, et, 1, device=device)
          if si in waiting_queues:
            active_queues += waiting_queues[si]
            waiting_queues[si].clear()
          seen_sids.add(si)
        else:
          # put it back
          self.q[device] = [si] + self.q[device]
      elif isinstance(si, WaitItem):
        if si.sync not in seen_sids:
          waiting_queues[si.sync].append(device)
          continue
      elif isinstance(si, CopyItem):
        if not hasattr(si.output, "_buf"): si.output.allocate()
        fxn = BufferXfer() if hasattr(Device[si.output.device].allocator, 'transfer') and \
          si.output.device.split(":")[0] == si.input.device.split(":")[0] else BufferCopy()
        fxn.exec([si.output, si.input])
      elif isinstance(si, ScheduleItem):
        for out in si.outputs:
          if not hasattr(out, "_buf") and not (out.device.startswith("DISK") and si.ast[0].op is BufferOps.STORE): out.allocate()
        if si.ast[0].op is not LoadOps.EMPTY:
          if si.ast[0].op is LoadOps.CUSTOM:
            runner:JITRunner = CustomOp(si.ast[0].arg)
          elif si.ast[0].op is BufferOps.STORE:
            runner = Device[si.outputs[0].device].get_runner(*si.ast)
          else: raise RuntimeError(f"unknown type {si}")
          runner.exec(list(si.outputs+si.inputs), si.var_vals)
        else:
          update_stats(colored(f"empty {si.outputs[0].size:10d} {si.outputs[0].dtype}", "yellow"), 0, 0, {}, None, 1, device=si.outputs[0].device)
      else: raise RuntimeError(f"unknown type {si}")
      active_queues.append(device)
