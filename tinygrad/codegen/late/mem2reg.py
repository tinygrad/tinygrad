from tinygrad.renderer import Renderer
from tinygrad.renderer.isa import VRegister, rdef, rdefs, ISARenderer
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, Ops, ParamArg, AddrSpace
import itertools

def bptr(x:UOp) -> tuple[UOp, int]:
  while x.op is not Ops.INDEX: x=x.src[0]
  buf,idx = x.src
  while buf.op is Ops.AFTER: buf=buf.src[0]
  return (buf,idx.src[0].val)

# promotes REG space BUFFER memory loads/stores to SSA registers through control flow analysis/PHI resolution
# https://llvm.org/docs/Passes.html#mem2reg-promote-memory-to-register
class Mem2regContext:
  # in tinygrad phis are only necessary for loop carried dependencies ex.
  # stores that occur between load and one or more backedges
  def __init__(self, lst:list[UOp], ren:Renderer):
    assert isinstance(ren, ISARenderer), "mem2reg only supported for assembly backends"
    self.ren = ren
    self.current: dict[UOp, UOp] = {}
    self.nl: dict[tuple[UOp, int], int] = {}
    self.phi_copies: dict[VRegister, list[VRegister]] = {}

    lane_ctr = itertools.count()
    current: dict[tuple[UOp, int], UOp] = {}
    self.phis: dict[tuple[tuple[UOp, int], int], UOp] = {}
    flat: dict[tuple[UOp, int], dict[UOp, tuple[VRegister, int]]] = {}
    rng_ctx: dict[tuple[UOp, int], list[UOp]] = {}
    rngs = 0

    for u in lst:
      if u.op in {Ops.STORE, Ops.LOAD}:
        ptr = bptr(u.src[0])
        if ptr[0].addrspace is not AddrSpace.REG: continue
        if rngs: rng_ctx.setdefault(ptr, []).append(u)
        if u.op is Ops.STORE: current[ptr] = u
        if u.op is Ops.LOAD:
          if ptr not in flat: flat[ptr] = {}
          assert ptr in current, f"LOAD before STORE for buffer element: ({ptr[0].arg}, {ptr[1]})"
          flat[ptr][u] = (rdef(current[ptr]), len(flat[ptr])+1)

      if u.op is Ops.RANGE: rngs += 1
      if u.op is Ops.END:
        rngs -= 1
        if rngs == 0:
          for ptr,us in rng_ctx.items():
            i,ld = next(((i,u) for i,u in enumerate(us) if u.op is Ops.LOAD), (None, None))
            if ld is None: continue
            carry = next((u for u in reversed(us[i+1:]) if u.op is Ops.STORE), None)
            if carry is None: continue
            header,n = flat[ptr][ld]
            vr = ren.vreg(header.cons, width=header.width, alignment=header.alignment, phi=(header,rdef(carry)))
            phi = UOp.placeholder((1,), ptr[0].dtype, next(lane_ctr), AddrSpace.REG).replace(tag=(vr,))
            self.phis[(ptr, n)] = phi
            self.phi_copies.setdefault(header, []).append(vr)
            self.phi_copies.setdefault(rdef(carry), []).append(vr)
          rng_ctx.clear()

  def try_phi(self, idx:UOp, x:UOp) -> UOp|None:
    ptr = bptr(idx)
    self.nl[ptr] = self.nl.get(ptr, 0) + 1
    phi = self.phis.get((ptr, self.nl[ptr]), None)
    return (phi, [phi]) if phi is not None else None

  def try_merge_edge(self, x:UOp, idx:UOp, val:UOp) -> tuple[UOp, list[UOp]]|None:
    if (phis := self.phi_copies.get(rdef(x), None)):
      carry, copies = None, []
      for i, p in enumerate(phis):
        out = self.ren.vcopy(x.src[1], p)
        if i == 0: carry = out[0]
        copies.extend(out[1])
      self.current[bptr(idx)] = carry
      return x, [x] + copies
    else:
      out = self.ren.vcopy(val, rdef(x))
      self.current[bptr(idx)] = out[0]
      return out

pm_promote_regbufs = PatternMatcher([
  (UPat.var("idx").load(name="x"), lambda ctx,idx,x: ctx.try_phi(idx, x) or (ctx.current[bptr(idx)], [])),
  (UPat.var("idx").store(UPat.var("val"), name="x"), lambda ctx,x,idx,val: ctx.try_merge_edge(x, idx, val)),
])
