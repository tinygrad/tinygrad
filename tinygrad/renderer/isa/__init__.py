from __future__ import annotations
import itertools, functools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort, ProgramInfo

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
  def __repr__(self): return f"{self.name} <= phi[{','.join(str(e) for e in self.phi)}]" if self.phi is not None else f"{self.name}({self.width})"
  def is_sub(self) -> bool: return self.parent is not None
  def or_parent(self) -> VRegister: return self.parent if self.is_sub() else self
  def sub(self, i:int, length:int=1) -> VRegister:
    assert i+length <= self.width, f"sub-register index out of width range ({i} >= {self.width})"
    if self.is_sub(): return self.parent.sub(self.pos + i, length)
    return VRegister(f"{self.name}.{i}", self.cons, length, self.alignment, self, i)
  def __getitem__(self, idx):
    return self.sub(idx.start, idx.stop - idx.start + 1) if isinstance(idx, slice) else self.sub(idx)
  @functools.cached_property
  def _hash(self): return hash((self.name, len(self.cons), self.width, self.alignment, self.pos, (self.parent.name if self.parent else None)))
  def __hash__(self): return self._hash
  @functools.cached_property
  def _candidates(self) -> list[tuple[Register,...]]:
    return [self.cons[i:i+self.width] for i in range(len(self.cons) - self.width + 1) if self.cons[i].index % self.alignment == 0]
  def candidates(self) -> list[tuple[Register,...]]: return self._candidates

def rdefs(u:UOp) -> tuple[VRegister|Register,...]:
  if u.op in {Ops.AFTER, Ops.NOOP, Ops.BITCAST} and len(u.src): return rdefs(u.src[0])
  return tuple(v for v in (u.tag if isinstance(u.tag, tuple) else (u.tag,)) if isinstance(v, (VRegister, Register)))
def rdef(u:UOp) -> VRegister|Register|None: return rdefs(u)[0] if len(rdefs(u)) >= 1 else None

# all per-kernel state of the ISA pipeline (isel -> linearize -> regalloc -> post regalloc) lives here, not on the renderer
class PreLinearKernelCtx:
  def __init__(self, sink:UOp, ren:ISARenderer, info:ProgramInfo):
    self.ren = ren
    self.spill_size = 0
    self.loop_label: dict[UOp, str] = {}
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n = itertools.count()
    def arg_key(u:UOp):
      if u.op is Ops.SPECIAL: return (2, u.arg)
      return (0, u.arg.slot) if u.arg.addrspace is not None else (1, u.expr)
    self.func_args = sorted([u for u in self.uses if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)

  def vreg(self, cons:tuple[Register, ...], **kwargs) -> VRegister:
    return VRegister(f"vr{next(self.reg_n)}", cons if isinstance(cons, tuple) else (cons,), **kwargs)

  # returns arch specific placement information for the spilled virtual register, and grows the stack frame if it spills to memory
  def assign_spill_slot(self, v:VRegister, vdef:UOp) -> any: raise NotImplementedError("arch specific")
  # runs after regalloc, when the size of the stack frame is known
  def stack_alloc(self, uops:list[UOp]) -> list[UOp]: return uops

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher = PatternMatcher([])
  post_regalloc_matcher: PatternMatcher
  kernel_ctx_type: type = PreLinearKernelCtx
  # NOTE: would be nice for this to be cached automatically like in UOp.ins() or something?
  # instead of needing to manually register important instructions in each renderer impl
  # NOTE: this is a memo of the isa opcode -> Ops mapping, it's the same for every kernel so it can live on the renderer
  semantic_op: dict[any, Ops] = {} # preserve IR metadata post-isel

  def is_two_address(self, x:UOp) -> bool: return False
  def spill_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def vcopy(self, u:UOp, vr:VRegister) -> tuple[UOp, list[UOp]]: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, regs:tuple[Register,...]) -> list[UOp]: raise NotImplementedError("arch specific")
  def spill(self, spill_offset:int, x:UOp, sub_idx:int|None=None) -> list[UOp]: raise NotImplementedError("arch specific")
  def fill(self, spill_offset:int, sub_idx:int|None, x:UOp, regs:tuple[Register,...]) -> tuple[UOp, list[UOp]]: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
