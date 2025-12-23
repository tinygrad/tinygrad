from __future__ import annotations
from typing import Callable, cast
from dataclasses import dataclass
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops, UOp, sint, ssimplify, GroupOp, PatternMatcher
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.codegen.opt.tc import TensorCore

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
  def from_uops(uops:list[UOp], ignore_indexing=False) -> Estimates:
    flops: sint = 0
    lds: sint = 0
    mem: dict[tuple[UOp, Ops], sint] = {}
    mults: sint = 1
    mult_stack: list[sint] = []
    dont_count: set[UOp] = set()
    if ignore_indexing:
      def range_gate(x): return x.op is not Ops.RANGE
      for u in uops:
        if u.op in {Ops.LOAD, Ops.STORE}:
          # if u.src[0] is INDEX, we have to include the buffer since it might be an AFTER
          dont_count = dont_count.union((UOp.sink(*u.src[0].src[1:]) if u.src[0].op is Ops.INDEX else u.src[0]).toposort(range_gate))
          # TODO: is this correct? this all needs to be cleaned up
          if len(u.src) > 2: dont_count = dont_count.union(u.src[2].toposort())
        elif u.op is Ops.IF:
          dont_count = dont_count.union(u.src[0].toposort())
    for u in uops:
      if u.op in {Ops.LOAD, Ops.STORE}:
        buf = u
        while len(buf.src): buf = buf.src[0]
        if buf.op is Ops.DEFINE_GLOBAL: # assume all DEFINE_GLOBAL memory is accessed
          mem[(buf, u.op)] = buf.ptrdtype.size * buf.dtype.itemsize
      if u.op is Ops.RANGE:
        mult_stack.append(mults)
        mults *= cast(sint, u.src[0].ssimplify())
        # SPECIAL are already counted in mults
        mults = mults.substitute({x:x.const_like(0) for x in mults.toposort() if x.op is Ops.SPECIAL}) if isinstance(mults, UOp) else mults
      elif u.op is Ops.END: mults = mult_stack.pop(-1)
      elif u.op is Ops.SPECIAL: mults *= cast(sint, u.src[0].ssimplify()) # NOTE: we don't push to the mult_stack here, you can't end these
      elif u.op is Ops.LOAD and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
        lds += u.dtype.itemsize * mults
      elif u.op is Ops.STORE and (not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG):
        lds += u.src[1].dtype.itemsize * mults
      elif u.op in GroupOp.ALU and u not in dont_count: flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.dtype.count
      elif u.op is Ops.WMMA and u not in dont_count: flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
    return Estimates(flops, lds, sum(mem.values()))

class Renderer:
  device: str = ""
  suffix: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  has_local: bool = True
  has_threads: bool = False
  has_shared: bool = True
  # NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  local_max: tuple[int, ...]|None = (0x8FFFFFFF,) * (3) # TODO: Ops.SPECIAL int32 indexes right now
  shared_max: int = 32768
  tensor_cores: list[TensorCore] = []
  pre_matcher: PatternMatcher|None = None
  extra_matcher: PatternMatcher|None = None
  code_for_op: dict[Ops, Callable] = {}

  def __reduce__(self): return self.__class__, ()
  def render(self, uops:list[UOp]) -> str: raise NotImplementedError("needs a renderer")
