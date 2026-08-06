from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  size: int = 8
  def __repr__(self): return self.name

# TODO: iter over subregisters
@dataclass(frozen=True)
class VRegister:
  name: str
  cons: tuple[Register, ...] = field(default_factory=tuple)
  width: int = 1
  alignment: int = 1
  parent: VRegister|None = None
  pos: int|None = None
  def __repr__(self): return self.name
  def is_sub(self) -> bool: return self.parent is not None
  def sub(self, i:int) -> VRegister:
    assert i < self.width, f"sub-register index out of width range ({i} >= {self.width})"
    return VRegister(f"{self.name}.{i}", self.cons, self.width, self.alignment, self, i)
  def candidates(self) -> list[tuple[Register,...]]:
    return [self.cons[i:i+self.width] for i in range(len(self.cons) - self.width + 1) if self.cons[i].index % self.alignment == 0]

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
    self.regbufs: dict[tuple[UOp, int], VRegister] = {}

  def vreg(self, cons:tuple[Register, ...], **kwargs) -> VRegister:
    return VRegister(f"vr{next(self.reg_n)}", cons if isinstance(cons, tuple) else (cons,), **kwargs)
  def regptr(self, idx:UOp, cons:tuple[Register, ...], **kwargs) -> tuple[VRegister,...]:
    buf = idx.src[0]
    while buf.op is Ops.AFTER: buf = buf.src[0]
    idxs = (idx.src[1].val,) if idx.op is Ops.INDEX else range(idx.src[-1].val)
    return tuple(self.regbufs.setdefault((buf,i), self.vreg(cons, **kwargs)) for i in idxs)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher
  post_regalloc_ctx: any|None = None
  spill_size: int = 0

  def is_two_address(self, x:UOp) -> bool: return False
  def spill_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def stack_alloc(self, uops:list[UOp]) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, spill_offset:int, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_offset:int, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
