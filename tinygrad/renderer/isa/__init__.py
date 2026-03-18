from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, Ops
from tinygrad.codegen import line_rewrite
from tinygrad.uop.spec import type_verify
from tinygrad.helpers import SPEC, DEBUG

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  @property
  def cons(self): return self._cons or (self,)
  def __repr__(self): return self.name

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

  def vreg(self, cons:tuple[Register, ...]|Register):
    return Register(f"v{next(self.reg_n)}", 0, _cons=cons if isinstance(cons, tuple) else (cons,))

@dataclass
class PreRegAllocContext:
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)

class ISARenderer(Renderer):
  isa_spec: PatternMatcher
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher

  def is_two_address(self, x:UOp) -> bool: return False
  def should_rematerialize(self, x:UOp) -> bool: return False
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
  # TODO: these should go with the other rewrites in codegen
  def lower(self, sink:UOp):
    from tinygrad.codegen.late.linearizer import linearize
    from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite
    function_name = sink.arg.function_name
    sink = graph_rewrite(sink, self.pre_isel_matcher, name="pre instruction selection", bottom_up=True)
    isel_ctx = IselContext(sink)
    sink = graph_rewrite(sink, self.isel_matcher, ctx=isel_ctx, name="instruction selection", bottom_up=True)
    lst = linearize(sink)
    if self.pre_regalloc_matcher is not None: lst = line_rewrite(lst, self.pre_regalloc_matcher, PreRegAllocContext())
    regalloc_ctx = LinearScanRegallocContext(lst, self, isel_ctx.stack_size)
    lst = line_rewrite(lst, pm_regalloc_rewrite, regalloc_ctx)
    lst = line_rewrite(lst, self.post_regalloc_matcher, regalloc_ctx)
    if DEBUG >= 4: print(self.asm(lst, function_name))
    if SPEC: type_verify(lst, self.isa_spec)
    return lst