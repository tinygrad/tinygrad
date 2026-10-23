from __future__ import annotations
from typing import Callable, cast
from dataclasses import dataclass, replace
from tinygrad.helpers import prod, Target, EMULATED_DTYPES
from tinygrad.uop.ops import Ops, UOp, sint, ssimplify, smin, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.renderer.tc import TensorCore
from tinygrad.device import Compiler

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
  def from_uops(sink:UOp, ignore_indexing=False, enter_calls=True) -> Estimates:
    flops: sint = 0
    lds: sint = 0
    mem: dict[tuple[UOp, Ops], sint] = {}
    pending = [(sink, None, {})]

    # traverse in reverse order to greedily ignore indexing ALU if necessary
    while pending:
      sink, caller_mult, bindings = pending.pop()
      uops, multcache, excluded = list(sink.toposort()), {}, set()
      if caller_mult is None: caller_mult = prod([cast(sint, s.src[0].ssimplify()) for s in uops if s.op is Ops.SPECIAL])

      for u in reversed(uops):
        # compute compile time known multiplicity of op over RANGEs and CALL context
        if (rngs:=tuple(u.ranges)) not in multcache:
          resolved = tuple(r.substitute(bindings) for r in rngs)
          bnds = [cast(sint, r.src[0].without_after.ssimplify()) for r in resolved if r.dtype != dtypes.void and
            not any(u.addrspace is not AddrSpace.ALU for u in r.src[0].without_after.backward_slice_with_self)]
          local_mult = prod(bnds)
          if isinstance(local_mult, UOp):
            local_mult = local_mult.substitute({x:x.const_like(0) for x in local_mult.toposort() if x.op is Ops.SPECIAL})
          multcache[rngs] = local_mult * caller_mult
        mult = multcache[rngs]

        # indexing alu is optional for flop estimates
        if ignore_indexing and u.op in {Ops.INDEX, Ops.SHRINK}:
          excluded.update(set(UOp.sink(*u.src[1:]).toposort(lambda x: x.op not in {Ops.END, Ops.BACKEDGE})))
        elif u.op is Ops.CALL and enter_calls:
          # pass bindings instead of resolving whole body because the bound ops shouldnt get double counted
          child_bindings = {p:u.src_without_body[p.arg.slot].substitute(bindings,walk=True)
            for p in u.body.backward_slice if p.op is Ops.PARAM and p.arg.slot >= 0}
          pending.append((u.body, mult, child_bindings))
        elif u.op in {Ops.LOAD, Ops.STORE}:
          buf = bindings.get(u.buf_uop, u.buf_uop).without_after
          # u.src[0] is INDEX, cap at buffer size for re-reads (e.g. matmul)
          if buf.op is Ops.PARAM:
            accessed = mem.get((buf, u.op), 0) + u.src[0].max_numel() * u.src[0].dtype.itemsize * mult
            mem[(buf, u.op)] = smin(accessed, buf.max_numel() * buf.dtype.itemsize)
          if u.src[0].addrspace != AddrSpace.REG:
            lds += mult * u.max_numel() * (u if u.op is Ops.LOAD else u.src[1]).dtype.itemsize
        elif u.op in GroupOp.ALU and u not in excluded:
          flops += (mult * (2 if u.op is Ops.MULACC else 1)) * u.max_numel()
        elif u.op is Ops.WMMA and u not in excluded:
          flops += mult * (2 * prod(u.arg[0]) // u.arg[2])

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
