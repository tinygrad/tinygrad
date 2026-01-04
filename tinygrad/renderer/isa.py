from __future__ import annotations
from tinygrad.renderer import Renderer
from dataclasses import dataclass, field
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, UPat, Ops
from tinygrad.codegen import line_rewrite
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.uop.spec import type_verify
from tinygrad.helpers import SPEC, DEBUG
import itertools

def print_uop_asm(uops:list[UOp]):
  for i,u in enumerate(uops):
    formatted_srcs = [f"{x.arg}" for x in u.src if x.arg is not None]
    print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):40s} " f"{str(u.arg):32s} {str(formatted_srcs)}")

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  cons: tuple[Register, ...] = field(default_factory=tuple)

  def __str__(self): return self.name
  def __lt__(self, other): return self.index < other.index if other is not None else False

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = sink.get_consumer_map()
    self.reg_n = itertools.count()
    self.stack_size = 0

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

class ISARenderer(Renderer):
  isa_spec: PatternMatcher
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher

  def two_address(self, x:UOp) -> int|None: raise NotImplementedError("arch specific")
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  # TODO: these should go with the other rewrites after we know what to do with ProgramSpec and Estimates
  def lower(self, sink:UOp):
    from tinygrad.codegen.late.regalloc import RegallocContext, pm_regalloc, pm_insert_spills
    sink = graph_rewrite(sink, self.pre_isel_matcher, name="pre instruction selection", bottom_up=True)
    isel_ctx = IselContext(sink)
    sink = graph_rewrite(sink, self.isel_matcher, ctx=isel_ctx, name="instruction selection", bottom_up=True)
    # TODO: remove, annoying needed for noops
    sink = graph_rewrite(sink, isel_fixup, name="instruction selection fixup")
    lst = linearize(sink)
    if DEBUG >= 8: print_uop_asm(lst)
    regalloc_ctx = RegallocContext(lst, self, isel_ctx.stack_size)
    lst = line_rewrite(lst, pm_regalloc, regalloc_ctx)
    lst = line_rewrite(lst, pm_insert_spills, regalloc_ctx)
    lst = line_rewrite(lst, self.post_regalloc_matcher, regalloc_ctx)
    if DEBUG >= 7: print_uop_asm(lst)
    if SPEC: type_verify(lst, self.isa_spec)
    return lst

# TODO: shared matchers can go here