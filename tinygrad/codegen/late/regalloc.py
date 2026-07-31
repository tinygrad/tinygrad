import itertools
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AddrSpace
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef
from tinygrad.dtype import dtypes

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.STACK, Ops.INDEX}

class LinearScanRegallocContext:
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = uops, ren, itertools.count()
    self.prgpts: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
    self.uops = [u for u in uops if u.op not in PSEUDO_OPS|{Ops.BUFFER}]
    self.live_intervals: dict[VRegister, list[int]] = {}

    lis = self.live_intervals
    range_vars: list[VRegister] = []
    def _live_units(u:UOp) -> tuple[VRegister,...]: # account for subregister lifetimes in parent live intervals/ranges
      if u.op is Ops.INDEX and not len(rdefs(u)): return _live_units(u.src[0]) # hack
      return tuple(r.parent if r.is_sub() else r for r in rdefs(u) if isinstance(r, VRegister))
    for u in reversed(self.uops):
      pt, defs, uses = self.prgpts[u], _live_units(u), []
      for s in dedup(u.src): uses.extend(_live_units(s))
      for v in defs + tuple(uses): lis.setdefault(v, []).insert(0, pt)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lis[rv][-1] for rv in range_vars if lis[rv][0] <= lis[v][-1] < lis[rv][-1]), default=None)): lis[v].append(n)
      if u.op is Ops.RANGE: range_vars.extend(defs)

    # sort by width, constraint pressure and program order
    vregs = set()
    for u in uops: vregs.update(_live_units(u))
    vregs = sorted(vregs, key=lambda v: (-v.width, len(v._cons), lis[v][0], lis[v][-1]))

    self.pmap: dict[VRegister, tuple[Register,...]] = {}
    vmap: dict[Register, list[VRegister]] = {}
    physical_slots: dict[Register, list[tuple[int, int], ...]] = {}
    spill_offset = 0

    # greedy allocate, pick first block of width w in constraints that is free for whole live range
    def _inside(a:VRegister, b:VRegister): return lis[a][0] <= lis[b][-1] and lis[a][-1] >= lis[b][0]
    def _isfree(v:VRegister, block:list[Register,...]) -> bool: return all(not _inside(v,bv) for r in block if r in vmap for bv in vmap[r])
    for v in vregs:
      candidates: list[tuple[Register,...]] = [v._cons[i:i+v.width] for i in range(len(v._cons) - v.width + 1) if v._cons[i].index % v.alignment == 0]
      if (block := next((b for b in candidates if _isfree(v, b)), None)):
        self.pmap[v] = block
        for r in block: vmap.setdefault(r, []).append(v)
      else:
        raise NotImplementedError(f"spilling not implemented: {v}")

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  if x.op in PSEUDO_OPS: return None
  nsrc, ndefs, before, after = [], [], [], []
  i = next(ctx.idx)

  for s in x.src:
    if s.op is Ops.INDEX: nsrc.append(s.replace(tag=(rdefs(s.src[0])[s.src[1].arg],)))
    else: nsrc.append(s)

  for v in rdefs(x):
    if not isinstance(v, VRegister): ndefs.append(v)
    elif v.is_sub(): ndefs.append(ctx.pmap[v.parent][v.pos])
    else: ndefs.extend(ctx.pmap[v])

  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))
  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.LOAD, Ops.INS, Ops.GROUP, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL} | PSEUDO_OPS, name="x"), regalloc_rewrite),
])

def gbuf(idx:UOp):
  buf = idx.src[0]
  while buf.op is Ops.AFTER: buf = buf.src[0]
  return buf

@dataclass(frozen=True)
class RegSlot:
  buf: UOp
  index: int

  def __repr__(self): return f"RegSlot({self.buf.arg}, {self.index})"
  @staticmethod
  def get(idx:UOp):
    buf = gbuf(idx)
    if idx.op is Ops.INDEX: return (RegSlot(buf, idx.src[1].arg),)
    else: return tuple([RegSlot(buf, i) for i in range(idx.src[-1].arg)])

# this should happen pre-linearize
class Mem2RegContext:
  def __init__(self, ren:ISARenderer):
    self.ren, self.ridx = ren, itertools.count()
    self.home: dict[RegSlot, VRegister] = {}

  def vrs(self, u:UOp) -> tuple[VRegister]:
    return tuple(self.home.setdefault(slot, self.ren.mem2reg_alloc(f"mvr{next(self.ridx)}", u)) for slot in RegSlot.get(u.src[0]))

def promos(ctx, x:UOp, idx:UOp, val:UOp):
  while val.op is Ops.AFTER: val = val.src[0]
  if idx.op is Ops.SHRINK:
    lanes = [val.index(i) for i in range(idx.src[-1].arg)]
    copies = [ctx.ren.copy(l, vr) for l,vr in zip(lanes, ctx.vrs(x))]
    return UOp.group(*copies), lanes + copies
  else:
    copy = ctx.ren.copy(val, *ctx.vrs(x))
    lanes = [val.index(i) for i in range(len(copy.src))] if copy.op is Ops.GROUP else []
    lines = list(copy.src) if copy.op is Ops.GROUP else [copy]
    return copy, lanes + lines

pm_mem2reg_rewrite = PatternMatcher([
  # each index into shrink load gets its own transparent copy
  # maintains single vreg and index passthrough semantics clean for regalloc
  (UPat(Ops.INDEX, name="idx"), lambda ctx,idx: 
    ((nx := ld.replace(tag=(ld.tag[idx.src[1].arg],))), [nx])
    if (ld := gbuf(idx)).op is Ops.LOAD else None),
  # regspace LOAD is just an empty register carrier
  (UPat.var("idx").load(name="x"), lambda ctx,idx,x: \
    ((nx := x.replace(src=(), tag=ctx.vrs(x))), [nx] if idx.op is Ops.INDEX else []) if x.tag is None else None),
  # reg store is copy, should handle copying directly from memory?
  # ex. store((global buffer).index(0), (reg buffer).index(1))
  # TODO!!!: should perform a single load directly into reg buffer
  (UPat.var("idx").store(UPat.var("val"), name="x"), lambda ctx,idx,val,x: \
    promos(ctx, x,idx,val) if idx.addrspace is AddrSpace.REG else None),
])
