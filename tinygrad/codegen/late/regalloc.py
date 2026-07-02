import itertools
from os import wait
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, VSubRegister, vrdefs, rdefs
from tinygrad.dtype import dtypes, PtrDType

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.GEP, Ops.STACK, Ops.INDEX}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:Register) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = uops, ren, itertools.count()
    prgpts: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}

    self.uops = [u for u in uops if u.op not in PSEUDO_OPS]
    self.live_intervals: dict[VRegister, list[int]] = {}
    self.pmap: dict[VRegister, tuple[Register|int,...]] = {}

    # TODO: handle alignment
    lis = self.live_intervals
    range_vars: list[VRegister] = []
    for u in reversed(self.uops):
      pt, defs, uses = prgpts[u], vrdefs(u), []
      for s in dedup(u.src): uses.extend(vrdefs(s))
      for v in defs + tuple(uses):
        lis.setdefault(v, []).insert(0, pt)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lis[rv][-1] for rv in range_vars if lis[rv][0] <= lis[v][-1] < lis[rv][-1]), default=None)): lis[v].append(n)
      if u.op is Ops.RANGE: range_vars.extend(defs)

    vregs = []
    # sort by width, constraint pressure and program order
    for u in uops: vregs.extend(vrdefs(u))
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
        raise NotImplementedError("spilling not implemented")

    # debug shit
    for k,v in lis.items():
      print(k, v)

    for v in vregs:
      print(v, v.width, len(v._cons), self.pmap[v])

    for r,ivs in physical_slots.items():
      print(r, ivs)


def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None

  nsrc = []
  for j,s in enumerate(x.src): # handle spills?
    pass

  ndefs = []
  for v in rdefs(x):
    if isinstance(v, VRegister): ndefs.append(ctx.pmap[v][0])
    if isinstance(v, VSubRegister): ndefs.append(ctx.pmap[v.parent][v.pos])

  nx = x.replace(tag=tuple(ndefs))
  print(x.op, x.arg, "new defs:", ndefs)
  # nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))

  before, after = [], []
  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL} | PSEUDO_OPS, name="x"), regalloc_rewrite),
])
