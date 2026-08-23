from __future__ import annotations
import itertools, functools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  size: int = 8
  def __repr__(self): return self.name
  def __hash__(self): return hash(self.name) * 256 + self.index

@dataclass(frozen=True)
class VRegister:
  name: str
  cons: tuple[Register, ...] = field(default_factory=tuple)
  width: int = 1
  alignment: int = 1
  parent: VRegister|None = None
  pos: int|None = None
  phi: tuple[VRegister,...]|None = None
  def __repr__(self): return self.name
  def is_sub(self) -> bool: return self.parent is not None
  def sub(self, i:int) -> VRegister:
    assert i < self.width, f"sub-register index out of width range ({i} >= {self.width})"
    return VRegister(f"{self.name}.{i}", self.cons, 1, self.alignment, self, i)
  @functools.cached_property
  def _hash(self): return hash((self.name, len(self.cons), self.width, self.alignment, self.pos, (self.parent.name if self.parent else None)))
  def __hash__(self): return self._hash
  @functools.cached_property
  def _candidates(self) -> list[tuple[Register,...]]:
    return [self.cons[i:i+self.width] for i in range(len(self.cons) - self.width + 1) if self.cons[i].index % self.alignment == 0]
  def candidates(self) -> list[tuple[Register,...]]: return self._candidates

def rdefs(u:UOp) -> tuple[VRegister|Register,...]:
  if u.op in {Ops.AFTER, Ops.NOOP} and len(u.src): return rdefs(u.src[0])
  return tuple(v for v in (u.tag if isinstance(u.tag, tuple) else (u.tag,)))
def rdef(u:UOp) -> None|tuple[VRegister|Register,...]: return rdefs(u)[0] if len(rdefs(u)) >= 1 else None

class PreRegallocContext:
  def __init__(self, sink:UOp, ren:ISARenderer):
    self.ren = ren
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n = itertools.count()
    self.lock: UOp|None = None
    self.clobbered: set[UOp] = set()
    def arg_key(u:UOp):
      if u.op is Ops.SPECIAL: return (2, u.arg)
      return (0, u.arg.slot) if u.arg.addrspace is not None else (1, u.expr)
    self.func_args = sorted([u for u in self.uses if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher
  post_regalloc_ctx: any|None = None
  spill_size: int = 0
  # NOTE: would be nice for this to be cached automatically like in UOp.ins() or something?
  # instead of needing to manually register important instructions in each renderer impl
  semantic_op: dict[any, UOp] = {} # preserve IR metadata post-isel
  reg_n = itertools.count()

  def vreg(self, cons:tuple[Register, ...], **kwargs) -> VRegister:
    return VRegister(f"vr{next(self.reg_n)}", cons if isinstance(cons, tuple) else (cons,), **kwargs)

  def is_two_address(self, x:UOp) -> bool: return False
  def stack_alloc(self, uops:list[UOp]) -> list[UOp]: return uops
  def spill_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, regs:tuple[Register,...]) -> list[UOp]: raise NotImplementedError("arch specific")
  def spill(self, spill_offset:int, x:UOp) -> list[UOp]: raise NotImplementedError("arch specific")
  def fill(self, spill_offset:int, x:UOp, regs:tuple[Register,...]) -> tuple[UOp, list[UOp]]: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
