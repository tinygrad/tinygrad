from __future__ import annotations
from typing import Callable, cast
from dataclasses import dataclass, replace
from tinygrad.helpers import prod, Target, EMULATED_DTYPES, dedup
from tinygrad.uop.ops import Ops, UOp, sint, ssimplify, smin, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.renderer.tc import TensorCore
from tinygrad.device import Compiler
import functools

# an access takes its dtype from the buffer it indexes, so accessing at another dtype restates the storage on the buffer that owns it
def with_storage(x:UOp, dt:DType) -> UOp:
  if x.op in GroupOp.Defines: return x.replace(arg=replace(x.arg, dtype=dt))
  return x.replace(src=(with_storage(x.src[0], dt),)+x.src[1:])

@dataclass(frozen=True)
class Estimates:
  # number of FLOPS used in the Kernel
  ops:sint = 0
  # bytes accessed in loads and stores
  lds:sint = 0
  # total bytes accessed, counting only once for bytes that are accessed multiple times
  mem:sint = 0
  def __add__(self, o:Estimates): return Estimates(self.ops + o.ops, self.lds + o.lds, self.mem + o.mem)
  def simplify(self): return Estimates(ssimplify(self.ops), ssimplify(self.lds), ssimplify(self.mem))
  @staticmethod
  def from_uops(sink:UOp, ignore_indexing=False) -> Estimates:
    flops: sint = 0
    lds: sint = 0
    mem: dict[tuple[UOp, Ops], sint] = {}
    uops = list(sink.toposort(enter_calls=False))
    specials = tuple(u for u in uops if u.op is Ops.SPECIAL)

    def idx_alu(uops:list[UOp]):
      excluded: set[UOp] = set()
      for u in uops:
        if u.op in {Ops.INDEX, Ops.SHRINK}:
          excluded = excluded.union(set(UOp.sink(*u.src[1:]).toposort(lambda x: x.op not in {Ops.END, Ops.BACKEDGE})))
      return excluded

    @functools.cache
    def mults(ranges:tuple[UOp,...]) -> sint:
      mult = prod([cast(sint, r.src[0].without_after.ssimplify()) for r in ranges if r.dtype is not dtypes.void])
      return mult.substitute({x:x.const_like(0) for x in mult.toposort() if x.op is Ops.SPECIAL}) if isinstance(mult, UOp) else mult

    # computes estimates within a given caller scope, inline calls get resolved recursively
    def process_body(u:UOp, caller_mult:sint, bindings:dict[UOp,UOp]={}):
      nonlocal flops, lds, mem
      uops = u.toposort(enter_calls=False)
      excluded = idx_alu(uops) if ignore_indexing else set()
      for u in uops:
        mult = mults(tuple(bindings.get(r.src[0],r.src[0]) for r in u.ranges)) * caller_mult
        if u.op in {Ops.LOAD, Ops.STORE}:
          buf = bindings.get(u.buf_uop, u.buf_uop).without_after
          if buf.op is Ops.PARAM: # u.src[0] is INDEX, cap at buffer size for re-reads (e.g. matmul)
            accessed = mem.get((buf, u.op), 0) + u.src[0].max_numel() * u.src[0].dtype.itemsize * mult
            mem[(buf, u.op)] = smin(accessed, buf.max_numel() * buf.dtype.itemsize)
        if u.op is Ops.CALL:
          bindings = {p:u.src_without_body[p.arg.slot].substitute(bindings,walk=True)
            for p in u.body.toposort() if p.op is Ops.PARAM and p.arg.slot >= 0}
          process_body(u.body, mult, bindings)
        elif u.op is Ops.LOAD and u.src[0].addrspace != AddrSpace.REG:
          lds += u.max_numel() * u.dtype.itemsize * mult
        elif u.op is Ops.STORE and u.src[0].addrspace != AddrSpace.REG:
          lds += u.max_numel() * u.src[1].dtype.itemsize * mult
        elif u.op in GroupOp.ALU and u not in excluded:
          flops += (mult * (2 if u.op is Ops.MULACC else 1)) * u.max_numel()
        elif u.op is Ops.WMMA and u not in excluded:
          flops += 2 * prod(u.arg[0]) // u.arg[2] * mult

    special_mult = prod([cast(sint, s.src[0].ssimplify()) for s in uops if s.op is Ops.SPECIAL])
    process_body(sink, special_mult)
    return Estimates(ssimplify(flops), lds, sum(mem.values()))

class Renderer:
  target: Target
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_shared: bool = True
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  global_prod_max: tuple[int, ...]|None = None
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[Ops, Callable] = {}

  compiler: Compiler = Compiler()

  def __init__(self, target:Target): self.target = target
  def __reduce__(self): return self.__class__, (self.target,)
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
  def asm(self, prg:UOp, lin:UOp) -> bytes: raise NotImplementedError("needs an assembler")
  def supported_dtypes(self) -> set[DType]:
    # double can't be bitcast to anything without long support
    return set(dtypes.all) - ({dtypes.double} if dtypes.long in EMULATED_DTYPES.tolist(dtypes) else set())
