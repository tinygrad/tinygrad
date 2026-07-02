from __future__ import annotations
import itertools, functools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  def __repr__(self): return self.name

# TODO: iter over subregisters
@dataclass(frozen=True)
class VRegister:
  name: str
  width: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  def __repr__(self): return self.name
  def sub(self, i:int) -> VSubRegister:
    assert self.width > 1, "sub registers only supported for wide vregs"
    return VSubRegister(self, i)

@dataclass(frozen=True)
class VSubRegister: # should this inherit?
  parent: VRegister
  pos: int
  def __repr__(self): return f"{self.parent.name}.{self.pos}"

# model view register dependencies through rewrites in here
AbstractReg = Register|VRegister|VSubRegister
def rdefs(u:UOp) -> tuple[AbstractReg,...]:
  if u.op in {Ops.GEP, Ops.INDEX}: # narrow
    idx = u.arg[0] if u.op is Ops.GEP else u.src[1].arg
    return (rdefs(u.src[0])[idx],) 
  # if u.op is Ops.GROUP: return tuple(r for s in u.src for r in rdefs(s)) # widen
  return tuple(v for v in (u.tag if isinstance(u.tag, tuple) else (u.tag,)) if isinstance(v, AbstractReg))
def vrdefs(u:UOp) -> tuple[VRegister,...]: return tuple(r for r in rdefs(u) if isinstance(r, VRegister))

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n, self.group_n = itertools.count(), itertools.count()
    self.lds_size = 0
    def arg_key(u:UOp):
      if u.op is Ops.SPECIAL: return (2, u.arg)
      return (0, u.arg.slot) if u.arg.addrspace is not None else (1, u.expr)
    self.func_args = sorted([u for u in self.uses if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)

  def vreg(self, cons:tuple[Register, ...], width:int=1) -> VRegister:
    return VRegister(f"vr{next(self.reg_n)}", width, cons if isinstance(cons, tuple) else (cons,))

@dataclass
class PreRegAllocContext:
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher
  post_regalloc_ctx: any|None = None

  def is_two_address(self, x:UOp) -> bool: return False
  def spill_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
