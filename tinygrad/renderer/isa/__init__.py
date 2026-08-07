from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.dtype import DType
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  size: int = 8
  @property
  def cons(self): return self._cons or (self,)
  def __hash__(self): return hash((self.name, self.index, self.size, tuple(r.index for r in self._cons)))
  def __repr__(self): return self.name

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n = itertools.count()
    self.scratch: dict = {}
    def arg_key(u:UOp):
      if u.op is Ops.SPECIAL: return (2, u.arg)
      return (0, u.arg.slot) if u.arg.addrspace is not None else (1, u.expr)
    self.func_args = sorted([u for u in self.uses if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)

  def vreg(self, cons:tuple[Register, ...]|Register):
    return Register(f"v{next(self.reg_n)}", 0, _cons=cons if isinstance(cons, tuple) else (cons,))

def greg(u:UOp):
  if u.op in {Ops.NOOP, Ops.AFTER} and u.src: return greg(u.src[0])
  if isinstance(u.tag, tuple): return u.tag[0]
  return u.tag

@dataclass
class PreRegAllocContext:
  uops: list[UOp]|None = None
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)
  scratch: dict = field(default_factory=dict)

class ISARenderer(Renderer):
  float4_dtypes: tuple[DType, ...]|None = None
  preferred_reduce_group: int|None = None
  local_prod_max: int|None = None
  wide_regalloc: bool = False
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher

  def prepare_pre_regalloc(self, lst:list[UOp]) -> tuple[list[UOp], dict]: return lst, {}
  def is_two_address(self, x:UOp) -> bool: return False
  def prefer_phys(self, x:UOp, src_phys:list) -> Register|None: return None
  def rematerialize(self, x:UOp) -> bool: return False
  # If True, a rematerialized value stays in its phys reg across uses (no re-remat each time).
  def keep_remat(self, x:UOp) -> bool: return False
  def register_slots(self, x:UOp, v:Register|None=None) -> int: return 1
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
