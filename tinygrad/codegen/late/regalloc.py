import itertools
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AddrSpace
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef
from tinygrad.dtype import dtypes

REGALLOC_OPS = {Ops.LOAD, Ops.INS, Ops.GROUP, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL}

class LinearScanRegallocContext:
  def vdef(self, v:VRegister) -> UOp: return self.uops[self.live_intervals[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = uops, ren, itertools.count()
    self.uops = [u for u in uops if u.op in REGALLOC_OPS]
    self.live_intervals: dict[VRegister, list[int]] = {}

    lis = self.live_intervals
    range_vars: list[VRegister] = []
    def _live_units(u:UOp) -> tuple[VRegister,...]: # account for subregister lifetimes in parent live intervals/ranges
      if u.op is Ops.INDEX and not (u.tag is not None and any(isinstance(v,VRegister) for v in u.tag)): return _live_units(u.src[0]) # hack
      return tuple(r.parent if r.is_sub() else r for r in rdefs(u) if isinstance(r, VRegister))
    for i, u in enumerate(reversed(self.uops)):
      defs, uses = _live_units(u), []
      for s in dedup(u.src): uses.extend(_live_units(s))
      for v in defs + tuple(uses):
        lis.setdefault(v, []).insert(0, len(self.uops) - i - 1)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lis[rv][-1] for rv in range_vars if lis[rv][0] <= lis[v][-1] < lis[rv][-1]), default=None)): lis[v].append(n)
      if u.op is Ops.RANGE: range_vars.extend(defs)

    # sort by width, constraint pressure and program order
    vregs = set()
    for u in uops: vregs.update(_live_units(u))
    vregs = sorted(vregs, key=lambda v: (-v.width, len(v._cons), lis[v][0], lis[v][-1]))

    self.pmap: dict[VRegister, tuple[Register,...]] = {}
    vmap: dict[Register, list[VRegister]] = {}

    spill_size = 0
    self.spills: dict[int, list[tuple[int, VRegister]]] = {}
    self.fills: dict[int, list[tuple[int, VRegister]]] = {}
    self.fmap: dict[int, list[tuple[VRegister, VRegister]]] = {}
    fidx = itertools.count()
    # greedy allocate, pick first block of width w in constraints that is free for whole live range
    def overlaps(a:VRegister, b:VRegister): return lis[a][0] <= lis[b][-1] and lis[a][-1] >= lis[b][0]
    def itfrs(v:VRegister, block:tuple[Register,...]): return set(vr for r in block if r in vmap for vr in vmap[r] if overlaps(v, vr))
    def alloc(v:VRegister):
      nonlocal spill_size
      candidates = v.candidates()
      if (block := next((b for b in candidates if not any(overlaps(v, bv) for r in b if r in vmap for bv in vmap[r])), None)):
        self.pmap[v] = block
        for r in block: vmap.setdefault(r, []).append(v)
      else:
        evicted = max(candidates, key=lambda b: min(next(i for i in lis[vr] if i >= lis[v][0]) for vr in itfrs(v,b)))
        news = []
        for ev in itfrs(v, evicted):
          j = next(j for j,p in enumerate(lis[ev]) if p >= lis[v][0])
          fr = VRegister(f"fr{next(fidx)}", ev._cons, ev.width, ev.alignment)
          lis[fr] = lis[ev][j:]
          lis[ev] = lis[ev][:j]
          # TODO: remove the buffer condition for x86
          sz = ev._cons[0].size if self.vdef(ev).op is not Ops.BUFFER else 8
          offset = spill_size + (sz - spill_size % sz) % sz
          self.spills.setdefault(lis[ev][0], []).append((offset,ev))
          self.fills.setdefault(lis[fr][0], []).append((offset,fr))
          spill_size += sz
          for i in lis[fr]: self.fmap.setdefault(i, []).append((ev,fr))
          news.append(fr)
        self.pmap[v] = evicted
        for r in evicted: vmap[r].append(v)
        return news

    for v in vregs:
      if (fills := alloc(v)) is not None: vregs.extend(fills)
    ren.spill_size = spill_size

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i, nsrc, ndefs, = next(ctx.idx), [], []
  alias = {ctx.pmap[vr][0]:ctx.pmap[fr][0] for (vr,fr) in ctx.fmap[i]} if i in ctx.fmap else {}

  # ew
  def _view(u:UOp): return rdefs(u) if u.op is not Ops.INDEX else (rdefs(u.src[0])[u.src[1].arg],)
  for s in x.src:
    s = s.replace(tag=tuple(alias.get(r,r) for r in _view(s)))
    nsrc.append(s)

  for v in rdefs(x):
    if not isinstance(v, VRegister): ndefs.append(v)
    elif v.is_sub(): ndefs.append(ctx.pmap[v.parent][v.pos])
    else: ndefs.extend(ctx.pmap[v])

  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))

  after = [ctx.ren.spill(slot,nx,*ctx.pmap[vr]) for slot,vr in ctx.spills[i]] if i in ctx.spills else []
  before = [ctx.ren.fill(slot,nx,*ctx.pmap[vr]) for slot,vr in ctx.fills[i]] if i in ctx.fills else []

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat(REGALLOC_OPS, name="x"), regalloc_rewrite),
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
