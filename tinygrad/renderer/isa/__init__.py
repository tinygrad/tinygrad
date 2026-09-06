from __future__ import annotations
from typing import Any
import itertools, functools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, AddrSpace, ProgramInfo
from tinygrad.dtype import DType

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
  def __repr__(self): return f"{self.name}({self.width})"
  def is_sub(self) -> bool: return self.parent is not None and self.pos is not None
  def or_parent(self) -> VRegister: return self.parent if self.parent is not None else self
  def sub(self, i:int, length:int=1) -> VRegister:
    assert i+length <= self.width, f"sub-register index out of width range ({i} >= {self.width})"
    if self.parent is not None and self.pos is not None: return self.parent.sub(self.pos + i, length)
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

# a copy destination is either a virtual register block or a fixed set of pinned registers.
def copy_dst(dst:VRegister|Register|tuple[Register,...]) -> tuple[tuple[VRegister|Register,...], tuple]:
  if isinstance(dst, VRegister): return (tuple(dst.sub(i) for i in range(dst.width)) if dst.width > 1 else (dst,)), (dst,)
  return (regs := dst if isinstance(dst, tuple) else (dst,)), regs

def rdefs(u:UOp) -> tuple[VRegister|Register,...]:
  if u.op in {Ops.AFTER, Ops.NOOP, Ops.BITCAST} and len(u.src): return rdefs(u.src[0])
  return tuple(v for v in (u.tag if isinstance(u.tag, tuple) else (u.tag,)) if isinstance(v, (VRegister, Register)))
def rdef(u:UOp) -> VRegister|Register|None: return rdefs(u)[0] if len(rdefs(u)) >= 1 else None

# all per-kernel state of the ISA pipeline lives here to avoid shared device state overlap in renderer
class PreLinearKernelCtx:
  def __init__(self, sink:UOp, ren:ISARenderer, info:ProgramInfo):
    self.ren, self.spill_size = ren, 0
    self.loop_label: dict[UOp, str] = {}
    self.reg_n, self.buf_slot = itertools.count(), itertools.count(-1, -1)
    def arg_key(u:UOp): return (1, u.arg) if u.op is Ops.SPECIAL else (0, u.arg.slot)
    self.func_args = sorted([u for u in sink.toposort() if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)
    self.ins_schedule: dict[Any, Ops] = {}
    self.reserved_regs: set[Register] = set()

  def reserved(self, regs: Register|tuple[Register,...], dt:DType) -> UOp:
    self.reserved_regs.update((regs := (regs,) if isinstance(regs, Register) else regs))
    return UOp.placeholder((1,), dt, next(self.buf_slot), AddrSpace.REG).replace(tag=regs)

  def vreg(self, cons:Register|tuple[Register, ...], **kwargs) -> VRegister:
    return VRegister(f"vr{next(self.reg_n)}", cons if isinstance(cons, tuple) else (cons,), **kwargs)

  def assign_spill_slot(self, v:VRegister, vdef:UOp) -> Any: raise NotImplementedError("arch specific")

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  kernel_ctx_type: type = PreLinearKernelCtx

  def is_two_address(self, x:UOp) -> bool: return False
  def spill_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, u:UOp, dst:VRegister|Register|tuple[Register,...]) -> tuple[UOp, list[UOp]]: raise NotImplementedError("arch specific")
  def spill(self, spill_offset:Any, x:UOp) -> list[UOp]: raise NotImplementedError("arch specific")
  def fill(self, spill_offset:Any, x:UOp, regs:tuple[Register,...]) -> tuple[UOp, list[UOp]]:
    raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
