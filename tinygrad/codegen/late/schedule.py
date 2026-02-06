from tinygrad.uop.ops import UOp, OpType
from tinygrad.codegen.late.regalloc import Register
from dataclasses import dataclass
from typing import Callable
import math

# this is an execution unit
@dataclass
class Unit: pass

# this is a group of execution units, an op can execute in any of the units
@dataclass
class Resource:
  units: tuple[Unit, ...]
  # size of the reservation station, micro-ops go here if their operands aren't ready or there isn't space in the resource
  # -1 is for unified reservation station
  #  0 is for in-order core
  #  1 is for in-order units in out-of-order core
  buffer_size: int = -1

# op scheduling info
@dataclass
class OpInfo:
  latency: int # minimum delay added to the dependency chain
  # resources used, includes the cycle when the unit is reserved and the cycle when the unit is released. one unit is reserved per resource
  resources: tuple[tuple[Resource, int, int], ...]
  micro_ops: int = 1 # number of micro-ops issued

# info about the whole processor
@dataclass
class MachineInfo:
  issue_width: int # number of micro-ops that can be issued per cycle
  mop_buffer_size: int # number of micro-ops that can be buffered (this is the minimum between the size of the reorder buffer,
                       # entries in register file and size of the unified reservation station), for an in-order core this number is 0
  op_info: dict[OpType, Callable] # op scheduling info

class MachineScheduler:
  def __init__(self, sink:UOp, mach_info: MachineInfo):
    self.consumers = sink.get_consumer_map()
    self.mach_info = mach_info
    self.info: dict[UOp, OpInfo] = {x: mach_info.op_info[x.op](x) for x in self.consumers if x.op in mach_info.op_info}
    # path from all dependencies of x to x (exclusive) with longest latency
    self.depth: dict[UOp, int] = {}
    for x in self.consumers: self.depth[x] = max([self.depth[s] + self.info[s].latency for s in x.src], default=0)
    # path from all dependents of x to x (exclusive) with longest latency
    self.height: dict[UOp, int] = {}
    for x,y in reversed(self.consumers.items()): self.height[x] = max([self.height[c] + self.info[c].latency for c in y], default=0)
    # map from resource to total count
    self.res_count = {res:0 for info in self.info.values() for res,_,_ in info.resources}
    # map from unit to next cycle when it's free, used for hazard check
    self.unit_ready = {unit:0 for res in self.res_count for unit in res.units}

    self.latency_factor = math.lcm(mach_info.issue_width, *[len(res.units) for res in self.res_count])

    self.mop_factor = self.latency_factor // mach_info.issue_width
    # map from scheduled uop to cycle it was scheduled at, init with uops that aren't instructions
    self.sched = {x:0 for x in self.consumers if not x.src}
    # map from uop whose dependencies have all been scheduled to cycle in which all its operands are ready, used for hazard check
    self.pending = {x:0 for x in self.sched if set(x.src).issubset(self.sched)}
    # map from register set to amount of live regs in that set
    self.reg_set: dict[tuple[Register, ...], int] = {}
    # the current cycle in the timeline
    self.cycle: int = 0
    # micro-ops issued in the current cycle
    self.cycle_mops: int = 0
    # total micro-ops issued
    self.total_mops: int = 0
    # total amount of latency scheduled, longest path so far
    self.expected_latency: int = 0
    # the critical resource, oversubscribed
    self.crit_res: Resource|None = None

  # total scheduled latency, stalls can cause cycle > expected, out-of-order can cause cycle < expected
  @property
  def sched_latency(self): return max(self.expected_latency, self.cycle)
  @property
  def crit_count(self): return self.total_mops * self.mop_factor if self.crit_res is None else self.res_count[self.crit_res]
  # avoid x if it increases register pressure above limit, favor x if it reduces pressure above limit
  def check_reg_pressure(self, x:UOp) -> int:
    new_reg_set = self.reg_set.copy()
    # if s was defined in the same block as x and x is its last use then s register is free
    for s in x.src:
      if isinstance(s.arg, Register) and set(self.consumers[s]) - set(self.sched) == {x} and s.ranges == x.ranges: new_reg_set[s.arg.cons] -= 1
    if isinstance(x.arg, Register): new_reg_set[x.arg.cons] += 1
    # difference in pressure above limit, any reduction or increase below limit is ignored
    return sum(max(new_reg_set[r], len(r)) - max(self.reg_set[r], len(r)) for r in new_reg_set)
  # avoid x if it uses an oversubscribed resource TODO: why does llvm accumulate this?
  def check_res_pressure(self, x:UOp) -> int: return next((end for res,_,end in self.info[x].resources if res is self.crit_res), 0)
  # avoid x if it's in the critical path and a predecessor was issued recently, only relevant for out-of-order as otherwise x isn't ready
  def check_lower_bound_latency(self, x:UOp) -> int: return max(self.depth[x] - self.sched_latency, 0)
  # favor x according to its remaining latency chain
  def check_height(self, x:UOp) -> int: return -self.height[x]

  def pick(self) -> UOp|None:
    # check whether x can be issued this cycle
    def _is_ready(x:UOp) -> bool:
      # check issue width can fit new micro ops unless nothing has been issued this cycle
      # in that case an expensive op with micro ops > issue width can be issued, but in multiple cycles
      if self.cycle_mops > 0 and self.cycle_mops + self.info[x].micro_ops > self.mach_info.issue_width: return False
      # these checks are skipped for out-of-order cores as then x can still be dispatched this cycle regardless of hazards
      if self.mach_info.mop_buffer_size == 0:
        # data hazard (operands not ready) check
        if self.pending[x] < self.cycle: return False
        # structural hazard (resources not available) check
        if any(self.cycle < min(self.unit_ready[u] for u in res.units) for res,_,_ in self.info[x].resources): return False
      return True
    # pick the best according to heuristics
    return min([x for x in self.pending if _is_ready(x)], key=lambda k: (self.check_reg_pressure(k), self.check_res_pressure(k),
                                                          self.check_lower_bound_latency(k), self.check_height(k)), default=None)

  def bump_cycle(self, next_cycle:int):
    dec_mops = self.mach_info.issue_width * (next_cycle - self.cycle)
    self.cycle_mops = 0 if self.cycle_mops <= dec_mops else self.cycle_mops - dec_mops
    self.cycle = next_cycle

  def update(self, x:UOp|None):
    next_cycle = self.cycle
    if x is not None:
      # add x and the current cycle to the schedule
      # TODO: this prob shouldnt be a max
      self.sched[x] = max(self.pending.pop(x), self.cycle)
      # add consumers whose dependencies have all been scheduled to pending, and the first cycle when all its operands are ready
      for v in self.consumers[x]:
        if set(v.src).issubset(self.sched): self.pending[v] = max(self.sched[s] + self.info[s].latency for s in v.src)

      if self.mach_info.mop_buffer_size == 0: assert self.pending[x] <= next_cycle
      # when is mop_buffer_size == 1?
      elif self.mach_info.mop_buffer_size == 1: next_cycle = max(next_cycle, self.pending[x])
      # if this is an in-order resource in out-of-order core account for likely stall cycles
      elif any(res.buffer_size == 1 for res,_,_ in self.info[x].resources): next_cycle = max(next_cycle, self.pending[x])

      self.total_mops += self.info[x].micro_ops
      # if this threshold is hit the resource is less critical than mop issue
      if self.crit_res is not None and self.total_mops * self.mop_factor - self.res_count[self.crit_res] >= self.latency_factor: self.crit_res = None
      # update resources
      for res,start,end in self.info[x].resources:
        self.res_count[res] += self.latency_factor // len(res.units) * (end - start)
        if self.res_count[res] > self.crit_count: self.crit_res = res

      # update the cycle when unit in resource is released by x, only relevant for in-order
      if self.mach_info.mop_buffer_size == 0:
        #next_cycle = max(next_cycle, min(self.unit_ready[u] for res,_,_ in self.info[x].resources for u in res.units))
        for res,_,end in self.info[x].resources:
          unit = min([u for u in res.units], key=lambda k: self.unit_ready[k])
          # TODO: when is unit_ready ever greater for in-order?
          self.unit_ready[unit] = max(self.unit_ready[unit], next_cycle + end)

      self.expected_latency = max(self.expected_latency, self.depth[x])
      # if a stall occured, bump until stall clears
      if next_cycle > self.cycle: self.bump_cycle(next_cycle)

    self.cycle_mops += self.info[x].micro_ops
    while self.cycle_mops >= self.mach_info.issue_width:
      next_cycle += 1
      self.bump_cycle(next_cycle)

    # if this threshold is hit the resource isn't deemed critical anymore
    if self.crit_res is not None and not (self.crit_count - (self.latency_factor * self.sched_latency) >= self.latency_factor): self.crit_res = None

  def schedule(self) -> list[UOp]:
    # TODO: check acyclic latency for ooo
    while self.pending: self.update(self.pick())
    return list(self.sched)
