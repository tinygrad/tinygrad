from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  _gid: int|None = None
  _count: int|None = None
  _pos: int|None = None
  @property
  def cons(self): return self._cons or (self,)
  def __repr__(self): return self.name
  @staticmethod
  def contiguous(ctx, cons:tuple[Register,...], n) -> tuple[Register,...]:
    gid = next(ctx.group_n)
    stripes = tuple(tuple(cons[i*n+j] for i in range(len(cons) // n)) for j in range(n))
    return tuple(Register(f"vr{next(ctx.reg_n)}", 0, _cons=stripes[j], _gid=gid, _count=n, _pos=j) for j in range(n))

def regs(u:UOp) -> tuple[Register,...]:
  # model view register dependencies through rewrites in here
  if u.op in (Ops.NOOP, Ops.AFTER) and self.src: return regs(self.src[0])
  if u.op is Ops.GEP: return (regs(u.src[0])[u.arg[0]],) # narrow
  if u.op is Ops.GROUP: return tuple(r for s in u.src for r in regs(s)) # widen
  return u.tag if isinstance(u.tag, tuple) else (u.tag,)
def reg(u:UOp) -> Register: return regs(u)[0]

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n, self.group_n = itertools.count(), itertools.count()
    arg_order = {Ops.PARAM: 0, Ops.DEFINE_VAR: 1, Ops.SPECIAL: 2}
    self.func_args = sorted([u for u in self.uses if u.op in arg_order], key=lambda k: (arg_order[k.op], k.arg))

  def vreg(self, cons:tuple[tuple[Register,...],n]|tuple[Register, ...]|Register) -> tuple[Register,...]|Register:
    if isinstance(cons, tuple) and isinstance(cons[0], tuple): return Register.contiguous(self, *cons)
    return Register(f"vr{next(self.reg_n)}", 0, _cons=cons if isinstance(cons, tuple) else (cons,))

@dataclass
class PreRegAllocContext:
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher

  def is_two_address(self, x:UOp) -> bool: return False
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
