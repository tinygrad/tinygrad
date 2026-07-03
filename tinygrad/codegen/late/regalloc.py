import itertools
from os import wait
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, VSubRegister, rdefs
from tinygrad.dtype import dtypes, PtrDType

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GEP, Ops.STACK, Ops.INDEX}

class LinearScanRegallocContext:
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = uops, ren, itertools.count()
    prgpts: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}

    self.uops = [u for u in uops if u.op not in PSEUDO_OPS]
    self.live_intervals: dict[VRegister, list[int]] = {}
    self.pmap: dict[VRegister, tuple[Register|int,...]] = {}

    # TODO: handle alignment
    lis = self.live_intervals
    range_vars: list[VRegister] = []
    def _live_units(u:UOp) -> tuple[VRegister,...]: # account for subregister lifetimes in parent live intervals/ranges
      if u.op in {Ops.INDEX, Ops.GEP}: return _live_units(u.src[0])
      return tuple(r.parent if isinstance(r, VSubRegister) else r for r in rdefs(u) if isinstance(r, (VRegister, VSubRegister)))
    for u in reversed(self.uops):
      pt, defs, uses = prgpts[u], _live_units(u), []
      for s in dedup(u.src): uses.extend(_live_units(s))
      for v in defs + tuple(uses): lis.setdefault(v, []).insert(0, pt)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lis[rv][-1] for rv in range_vars if lis[rv][0] <= lis[v][-1] < lis[rv][-1]), default=None)): lis[v].append(n)
      if u.op is Ops.RANGE: range_vars.extend(defs)

    # sort by width, constraint pressure and program order
    vregs = set()
    for u in uops: vregs.update(_live_units(u))
    vregs = sorted(vregs, key=lambda v: (-v.width, len(v._cons)))

    live_ranges: dict[Vregister, tuple[int,int]] = { v: (iv[0], iv[-1]) for v,iv in lis.items() }
    physical_slots: dict[Register, list[tuple[int, int], ...]] = {} 
    for v in vregs:
      v_start, v_end = live_ranges[v]
      def _isfree(pregs:list[Register]):
        return all(not (a < v_end and v_start < b) for r in pregs if r in physical_slots for a,b in physical_slots[r])
      # greedy allocate, pick first block of width w in constraints that is free for whole live range
      if (block := next((v._cons[i:i+v.width] for i in range(len(v._cons)) if _isfree(v._cons[i:i+v.width])), None)):
        self.pmap[v] = block
        for r in block: physical_slots.setdefault(r, []).append(live_ranges[v])
      else: # spill
        # spilling requires a few things:
        # - pass offset into spill space of spilled register to load/store in rewrite
        # - identify...
        raise NotImplementedError("spilling not implemented")

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None
  nsrc, ndefs, before, after = [], [], [], []

  for s in x.src: # handle spills?
    if s.op in {Ops.INDEX, Ops.GEP}:
      idx = s.src[1].arg if s.op is Ops.INDEX else s.arg[0]
      nsrc.append(s.replace(tag=(rdefs(s.src[0])[idx],)))
    else: nsrc.append(s)

  for v in rdefs(x):
    if isinstance(v, VRegister): ndefs.extend(ctx.pmap[v])
    if isinstance(v, VSubRegister): ndefs.append(ctx.pmap[v.parent][v.pos])

  # nx = x.replace(tag=tuple(ndefs))
  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))
  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.GROUP, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL} | PSEUDO_OPS, name="x"), regalloc_rewrite),
])
