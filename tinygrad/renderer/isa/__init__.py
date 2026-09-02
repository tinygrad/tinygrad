from __future__ import annotations
from typing import Callable
import itertools, functools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort, ProgramInfo, ParamArg, AddrSpace

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
  def __init__(self, sink:UOp, ren:ISARenderer, info:ProgramInfo, max_reserved_regs:int):
    self.ren, self.spill_size = ren, 0
    self.loop_label: dict[UOp, str] = {}
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n, self.named_n, self.buf_slot = itertools.count(), 0, itertools.count(-1, -1)
    def arg_key(u:UOp):
      if u.op is Ops.SPECIAL: return (2, u.arg)
      return (0, u.arg.slot) if u.arg.addrspace is not None else (1, u.expr)
    self.func_args = sorted([u for u in self.uses if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)

    # maps reg BUFFERs to either contiguous reserved register block or pre-allocated spill slots
    self.bufblocks: dict[ParamArg, tuple[Register,...]|tuple[any,...]] = {}
    self.bufregs: dict[tuple[ParamArg, int], UOp|any] = {}
    # opcode -> equivalent IR Op, filled in by do_linearize after isel
    self.ins_schedule: dict[any, Ops] = {}

    rbufs: dict[int, UOp] = {(u.arg.size * (u.dtype.itemsize//4)):u for u in sink.toposort() if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG}
    sizes = list(sorted(rbufs.keys(), reverse=True))
    n_spill = next((i for i in range(len(sizes)) if sum(sizes[i:]) < max_reserved_regs), len(sizes))
    self.overflows = set(rbufs[sz].arg for sz in sizes[:n_spill])

  def bufreg(self, idx:UOp, allocator:Callable[[UOp], tuple[Register,...]]) -> UOp:
    n = idx.src[-1].src[0].val if idx.op is Ops.SHRINK else 1
    while idx.op is not Ops.INDEX: idx = idx.src[0]
    buf, off = idx.src
    while buf.op is not Ops.BUFFER: buf = buf.src[0]
    defs = [self.bufreg_elem(buf, off.src[0].val+i, allocator) for i in range(n)]
    return defs[0] if n == 1 else UOp.group(*defs, tag=tuple(r for d in defs for r in rdefs(d)))

  # a reg BUFFER reserves one contiguous block up front
  def bufreg_elem(self, buf:UOp, i:int, allocator:Callable[[UOp], tuple[Register,...]]) -> UOp:
    stride = max(buf.dtype.itemsize//4, 1)
    if (block := self.bufblocks.get(buf.arg)) is None:
      n, regs = buf.arg.size * stride, allocator(buf)
      assert self.named_n+n < len(regs), "no remaining pinnable registers for reg BUFFER"
      block = self.bufblocks[buf.arg] = regs[self.named_n:self.named_n+n]
      self.named_n += n
    if (d := self.bufregs.get((buf.arg, i))) is None:
      d = self.bufregs[(buf.arg, i)] = self.reserved(block[i*stride] if stride == 1 else block[i*stride:(i+1)*stride], buf.dtype)
    return d

  def reserved(self, regs: Register|tuple[Register,...], dt:DType) -> UOp:
    return UOp.placeholder((1,), dt, next(self.buf_slot), AddrSpace.REG).replace(tag=regs if isinstance(regs, tuple) else (regs,))

  def vreg(self, cons:tuple[Register, ...], **kwargs) -> VRegister:
    return VRegister(f"vr{next(self.reg_n)}", cons if isinstance(cons, tuple) else (cons,), **kwargs)

  # returns arch specific placement information for the spilled virtual register and grows stack frame
  def assign_spill_slot(self, v:VRegister, vdef:UOp) -> any: raise NotImplementedError("arch specific")
  # runs after regalloc, when the size of the stack frame is known
  def stack_alloc(self, uops:list[UOp]) -> list[UOp]: return uops

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher = PatternMatcher([])
  post_regalloc_matcher: PatternMatcher
  kernel_ctx_type: type = PreLinearKernelCtx

  def is_two_address(self, x:UOp) -> bool: return False
  def spill_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  # copy u into dst, returns the node defining dst and the instructions to emit for it (line rewrites need both, isel only the node)
  def copy(self, u:UOp, dst:VRegister|Register|tuple[Register,...]) -> tuple[UOp, list[UOp]]: raise NotImplementedError("arch specific")
  def spill(self, spill_offset:int, x:UOp, sub_idx:int|None=None) -> list[UOp]: raise NotImplementedError("arch specific")
  def fill(self, spill_offset:int, sub_idx:int|None, x:UOp, regs:tuple[Register,...]) -> tuple[UOp, list[UOp]]: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
