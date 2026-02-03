import itertools, heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop import X86Ops, X86GroupOp
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, UPat, Ops
from tinygrad.codegen import line_rewrite
from tinygrad.codegen.late.schedule import MachineScheduler, MachineInfo
from tinygrad.codegen.late.regalloc import RegallocContext, pm_regalloc, pm_insert_spills, Register
from tinygrad.uop.spec import type_verify
from tinygrad.helpers import SPEC, DEBUG, getenv, prod

def print_uop_asm(uops:list[UOp]):
  for i,u in enumerate(uops):
    formatted_srcs = [f"{x.arg}" for x in u.src if x.arg is not None]
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):40s} " f"{str(u.arg):32s} {str(formatted_srcs)}")

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = sink.get_consumer_map()
    self.reg_n = itertools.count()
    self.stack_size = 0
    arg_order = {Ops.PARAM: 0, Ops.DEFINE_VAR: 1, Ops.SPECIAL: 2}
    self.func_args = sorted([u for u in self.uses if u.op in arg_order], key=lambda k: (arg_order[k.op], k.arg))

  def inc_stack(self, amt:int):
    ret = self.stack_size
    self.stack_size += amt
    return ret

  def vreg(self, cons:tuple[Register, ...]|Register|None=None):
    return Register(f"v{next(self.reg_n)}", 0, cons=cons if isinstance(cons, tuple) else (cons,) if cons is not None else ())

isel_fixup = PatternMatcher([
  # NOOP / AFTER have the same register as first src
  (UPat((Ops.NOOP, Ops.AFTER), name="x"), lambda x: x.replace(arg=x.src[0].arg) if x.src and x.arg is None else None),
])

# TODO: this will eventually be a proper scheduler
def isa_linearize(sink:UOp) -> list[UOp]:
  from tinygrad.renderer.x86 import RSP
  # this is a toposort with priority
  lst = list(sink.toposort())
  out_degree:defaultdict[UOp, int] = defaultdict(int)
  priorities:dict[UOp, tuple[int, int, Any]] = {}

  # get consumers and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    for s in u.src: out_degree[s] += 1

    # we place UOps with higher run_counts later
    run_count = prod([int(r.vmax)+1 for r in u.ranges])

    # simple priority override. this is all bottom up now, smaller numbers will be closer to the top
    match u.op:
      case Ops.RANGE: priority = 5    # placing RANGE is good
      case Ops.END: priority = -5     # placing END is bad
      # stack pointer needs to be scheduled at the top of the kernel
      case X86Ops.DEFINE_REG: priority = -21 if u.arg == RSP else -20
      case X86Ops.IMM: priority = -10
      case _: priority = 0            # everything else has priority 0
    priorities[u] = (run_count, priority)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]))}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(-nkey[sink], sink)]
  newlst = []
  lock: UOp|None = None
  stupid: int = 0
  clobbers: set[UOp] = set()
  while heap or clobbers:
    # if heap is empty we have a cycle and the flag producer must be rematerialized
    # we schedule the flag producer and free the clobbers
    if not heap:
      assert lock is not None and clobbers
      newlst.append(lock)
      for c in clobbers: heapq.heappush(heap, (-nkey[c],c))
      clobbers.clear()
      lock, stupid = None, 0

    u = heapq.heappop(heap)[1]

    # flags introduce state that must be dealt with, can't overwrite the flag until all its users and producer are scheduled
    if lock is not None:
      # if this is the flag producer we free the flag clobbers and release the lock
      if lock is u:
        for c in clobbers: heapq.heappush(heap, (-nkey[c],c))
        clobbers.clear()
        lock, stupid = None, 0
      # if this is the user of or is another flag producer it can't be scheduled
      # if this is a loop boundry or has a lower run count than the flag user that introduced the lock we also don't schedule
      # loop boundries do clobber but we also don't want to insert stuff from outside the loop into the loop
      # if there's no loop we also don't want to add IMM and DEFINE_REG in the middle of the kernel
      elif u.op in X86GroupOp.ReadFlags and lock is not u.src[-1] or u.op in X86GroupOp.WriteFlags or \
        u.op in {Ops.RANGE, Ops.END, X86Ops.IMM, X86Ops.DEFINE_REG} or priorities[u][0] < stupid:
        clobbers.add(u)
        continue
    # if there's no lock and this is a flag user its flag producer becomes the lock
    elif u.op in X86GroupOp.ReadFlags: lock, stupid = u.src[-1], priorities[u][0]

    newlst.append(u)

    for v in u.src:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(heap, (-nkey[v],v))
  return newlst[::-1]

class ISARenderer(Renderer):
  isa_spec: PatternMatcher
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  mach_info: MachineInfo

  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  # TODO: these should go with the other rewrites after we know what to do with ProgramSpec and Estimates
  def lower(self, sink:UOp):
    sink = graph_rewrite(sink, self.pre_isel_matcher, name="pre instruction selection", bottom_up=True)
    isel_ctx = IselContext(sink)
    sink = graph_rewrite(sink, self.isel_matcher, ctx=isel_ctx, name="instruction selection", bottom_up=True)
    # TODO: remove, annoying needed for noops
    sink = graph_rewrite(sink, isel_fixup, name="instruction selection fixup")
    if getenv("MACHINE_SCHEDULER"): lst = MachineScheduler(sink, self.mach_info).schedule()
    else: lst = isa_linearize(sink)
    if DEBUG >= 8: print_uop_asm(lst)
    regalloc_ctx = RegallocContext(lst, self.isel_matcher, self.stack_pointer(), isel_ctx.stack_size)
    lst = line_rewrite(lst, pm_regalloc, regalloc_ctx)
    lst = line_rewrite(lst, pm_insert_spills, regalloc_ctx)
    lst = line_rewrite(lst, self.post_regalloc_matcher, regalloc_ctx)
    if DEBUG >= 7: print_uop_asm(lst)
    if SPEC: type_verify(lst, self.isa_spec)
    return lst

# TODO: shared matchers can go here