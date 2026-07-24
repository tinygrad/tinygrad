import itertools
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs
from tinygrad.dtype import dtypes

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.STACK, Ops.INDEX}

class LinearScanRegallocContext:
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = uops, ren, itertools.count()
    self.prgpts: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
    self.uops = [u for u in uops if u.op not in PSEUDO_OPS|{Ops.BUFFER}]
    self.live_intervals: dict[VRegister, list[int]] = {}

    # TODO: handle alignment
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
  i = next(ctx.idx)# ctx.prgpts[x]

  for s in x.src: # handle spills?
    if s.op is Ops.INDEX: nsrc.append(s.replace(tag=(rdefs(s.src[0])[s.src[1].arg],)))
    else: nsrc.append(s)

  for v in rdefs(x):
    if not isinstance(v, VRegister): continue
    if v.is_sub(): ndefs.append(ctx.pmap[v.parent][v.pos])
    else: ndefs.extend(ctx.pmap[v])

  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))
  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.GROUP, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL} | PSEUDO_OPS, name="x"), regalloc_rewrite),
])
