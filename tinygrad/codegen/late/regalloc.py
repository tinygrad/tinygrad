import itertools
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AddrSpace
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef
from tinygrad.dtype import dtypes
from bisect import bisect_left

REG_OPS = {Ops.STORE, Ops.INS, Ops.STACK, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL, Ops.INDEX}

class LinearScanRegallocContext:
  def vdef(self, v:VRegister) -> UOp: return self.uops[self.live_intervals[v.parent if v.is_sub() else v][0]]

  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = [u for u in uops if u.op in REG_OPS], ren, itertools.count()
    self.live_intervals: dict[VRegister, list[int]] = {}

    ren.spill_size = 0

    lr = self.live_intervals
    range_vars: list[VRegister] = []
    def live_edge(u:UOp) -> tuple[VRegister,...]: return tuple(r.parent if r.is_sub() else r for r in rdefs(u) if isinstance(r, VRegister))
    for i, u in enumerate(reversed(self.uops)):
      defs, uses = live_edge(u), []
      for s in dedup(u.src): uses.extend(live_edge(s))
      for v in defs + tuple(uses):
        lr.setdefault(v, []).insert(0, len(self.uops) - i - 1)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lr[rv][-1] for rv in range_vars if lr[rv][0] <= lr[v][-1] < lr[rv][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE:
        # NOTE: cant derive range lifetime like this because of boundless LOOP
        range_vars.append(rdef(u))

    self.spills: dict[Register, any] = {} # mapping from virtual to generic stack placement information (arch specific)
    self.reals: dict[int, dict[VRegister, tuple[Register,...]]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, tuple[Register,...]]]] = {} # fills to be inserted at each program point
    live: dict[VRegister, tuple[Register,...]] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[VRegister, tuple[Register,...]]] = [] # mapping from virtual to real at loop entry

    # allocate the best register. Registers not in live or not used again are free and have priority,
    # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
    def alloc(v:VRegister, cons:list[tuple[Register, ...]]|None, i:int) -> tuple[Register,...]:
      cons = cons or v.candidates()
      live_inv = {r:k for k,v in live.items() for r in v}

      block = max(cons, key=lambda b: min(next((j-i for j in lr[live_inv[r]] if j >= i), len(self.uops)) \
        if r in live_inv else len(self.uops) for r in b))

      for r in block:
        if r in live_inv and (v := live_inv.get(r)) in live:
          live.pop(v)
          # phi evictions must be handled carefully to ensure loop carry gets reloaded and not silently clobbered
          if v.phi is not None and v not in self.spills and i <= lr[v][-1]:
            fill(v, self.live_intervals[v][1], (r,))
      return block

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:VRegister, i:int, cons:tuple[Register, ...]|None=None) -> tuple[Register,...]:
      if v not in self.spills:
        self.spills[v], self.ren.spill_size = self.ren.assign_spill_slot(v, self.vdef(v))
      rs = alloc(v, [cons] if cons is not None else None, i)
      if v.phi is None: # NOTE: phis insert their own fills at rewrite time
        self.insert_before.setdefault(i, []).append((v, rs))
      return rs

    for i,u in enumerate(self.uops):
      # allocate uses
      for s in u.src:
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance(v:=rdef(s), VRegister): continue
        if (vv := v.or_parent()) not in live: live[vv] = fill(vv,i)
        self.reals.setdefault(i, {})[v] = live[vv][v.pos:v.pos+v.width] if v.is_sub() else live[vv]

      # allocate defs
      for j,v in enumerate(rdefs(u)):
        if not isinstance(v, VRegister): continue
        cons = None
        if ren.is_two_address(u) and j == 0:
          uses = []
          for s in u.src:
            if rdef(s) in live: uses.extend(live.get(rdef(s)))
          cons = ([(uses[0],)] if uses[0] in v.cons else []) + [r for r in v.candidates() if r[0] not in uses]
        # parents can be defined by premature subregister op ex. collect then store
        if (vv := v.or_parent()) not in live:
          live[vv] = alloc(vv, cons, i+1 if u.op is not Ops.RANGE else i)
        self.reals.setdefault(i, {})[v] = live[vv][v.pos:v.pos+v.width] if v.is_sub() else live[vv]

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if v.phi is None and any(i <= l < lr[rdef(u)][-1] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.name, k.cons[0].index))
        live_in: dict[VRegister, tuple[Register,...]] = {}
        for v in sorted_uses:
          # if all the possible registers are already in live_in there's no space for this var
          if set(v.cons).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          live_in[v] = live[v]
        live_ins.append(live_in)

      # loop epilogue, reload registers that were live at loop entry
      if u.op is Ops.END:
        # TODO: if a uop is in a different reg in live out vs live in move between registers instead of loading
        # TODO: don't reload if first use in loop is a load
        for v,rs in live_ins.pop().items():
          if v not in live or live[v] != rs: live[v] = fill(v, i, rs)

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i, nsrc, before = next(ctx.idx), [], []
  for j,s in enumerate(x.src):
    if i in ctx.reals and isinstance((v := rdef(ctx.uops[i].src[j])), VRegister) and (vv := v.or_parent()) in ctx.spills:
      if vv.phi is not None:
        # spilled PHIs must be filled at every use.
        filled, fills = ctx.ren.fill(ctx.spills[vv], v.pos if v.is_sub() else None, ctx.vdef(vv), ctx.reals[i][v])
        nsrc.append(filled)
        before.extend(fills)
      else: nsrc.append(s.replace(tag=ctx.reals[i][v]))
    else:
      nsrc.append(s)

  ndefs, after = [], []
  for v in rdefs(x):
    if isinstance(v, VRegister): ndefs.extend(ctx.reals[i][v])
    else: ndefs.append(v)
  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))

  for v in rdefs(x):
    if not isinstance(v, VRegister): continue
    # spills are keyed by the parent, a subregister def still has to write back into the parent's slot at its own offset
    if (vv := v.or_parent()) in ctx.spills and not (x.op is Ops.BUFFER and vv.phi is not None):
      after.extend(ctx.ren.spill(ctx.spills[vv], nx, v.pos if v.is_sub() else None))
  for v,rs in ctx.insert_before.get(i, []):
    before.extend(ctx.ren.fill(ctx.spills[v], None, ctx.vdef(v), rs)[1])

  return nx, before + [nx] + after

def regspace(buf:UOp, c:UOp, x:UOp):
  if (vr := rdef(buf)) is None or c.val >= vr.width: return None
  svr = vr[(c.val)*2:(c.val*2)+1] if x.dtype.itemsize > 4 else vr[c.val]
  return (nx := x.replace(tag=(svr,))), [nx]

# INDEX -> subregister(s), conversion to regspace coordinates
pm_index_subregisters = PatternMatcher([
  (UPat.var("buf").index(UPat.cvar("c").cast(), name="x", tag=None), regspace)
])

def propogate_subs(ctx, x:UOp):
  # NOTE: take the per src width from the register block, not x.dtype. replacing the srcs below changes the
  # STACK's own inferred dtype (its src[0] becomes a 32 bit mov), so this rewrite has to stay idempotent
  vr, nsrc = rdef(x), []
  n = vr.width//len(x.src) if isinstance(vr, VRegister) and len(x.src) and vr.width%len(x.src) == 0 \
      else max(x.dtype.itemsize//4, 1)
  for i,s in enumerate(x.src):
    def _strip(x:UOp):
      while x.op in {Ops.BITCAST, Ops.AFTER}: x = x.src[0]
      return x
    # INDEX/reg LOAD srcs have to become copies, can be redundant but must enforce contiguity restraint.
    # Optimization would have to identify equivalent STACKs and tie register blocks
    if _strip(s).op in {Ops.INDEX, Ops.LOAD}: nsrc.append(ctx.vcopy(s, vr[i*n:(i+1)*n-1])[0])
    else: nsrc.append(s.replace(tag=(vr[i*n:(i+1)*n - 1],)))
  return x.replace(src=tuple(nsrc))

pm_prepare_regalloc = PatternMatcher([
  (UPat(Ops.STACK, name="x"), propogate_subs),
  (UPat((Ops.AFTER, Ops.BITCAST), name="x"), lambda x:
    x.replace(src=(x.src[0].replace(tag=x.tag), *x.src[1:]))
    if x.tag is not None else None),
])

pm_regalloc_rewrite = PatternMatcher([
  (UPat(REG_OPS, name="x"), regalloc_rewrite),
])
