import itertools
from typing import Any
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import Register, VRegister, rdefs, rdef, PreLinearKernelCtx

REG_OPS = {Ops.INS, Ops.STACK, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL, Ops.INDEX}

class LinearScanRegallocContext:
  def vdef(self, v:VRegister) -> UOp: return self.uops[self.live_intervals[v.or_parent()][0]]

  def __init__(self, uops:list[UOp], ctx:PreLinearKernelCtx):
    self.uops, self.ren, self.idx = [u for u in uops if u.op in REG_OPS], ctx.ren, itertools.count()
    self.live_intervals: dict[VRegister, list[int]] = {}

    lr = self.live_intervals
    range_vars: list[VRegister] = []
    def live_edge(u:UOp) -> tuple[VRegister,...]: return tuple(r.or_parent() for r in rdefs(u) if isinstance(r, VRegister))
    for i, u in enumerate(reversed(self.uops)):
      uses: list[VRegister] = []
      defs = live_edge(u)
      for s in dedup(u.src): uses.extend(live_edge(s))
      for v in defs + tuple(uses):
        lr.setdefault(v, []).insert(0, len(self.uops) - i - 1)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lr[rv][-1] for rv in range_vars if lr[rv][0] <= lr[v][-1] < lr[rv][-1]), default=None)) is not None:
          lr[v].append(n)
      if u.op is Ops.RANGE:
        # NOTE: cant derive range lifetime like this because of boundless LOOP
        range_vars.append(defs[0])

    self.spills: dict[VRegister, Any] = {} # mapping from virtual to generic stack placement information (arch specific)
    self.reals: dict[int, dict[VRegister, tuple[Register,...]]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[VRegister, tuple[Register,...]]]] = {} # fills to be inserted at each program point
    live: dict[VRegister, tuple[Register,...]] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[VRegister, tuple[Register,...]]] = [] # mapping from virtual to real at loop entry

    # allocate the best register. Registers not in live or not used again are free and have priority,
    # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
    def alloc(v:VRegister, cons:list[tuple[Register, ...]]|None, i:int) -> tuple[Register,...]:
      cons = cons or v.candidates()
      cons = [block for block in cons if all(r not in ctx.reserved_regs for r in block)]
      assert len(cons), f"no candidate register blocks provided for {v}"
      live_inv = {r:k for k,v in live.items() for r in v}

      block = max(cons, key=lambda b: min(next((j-i for j in lr[live_inv[r]] if j >= i), len(self.uops)) \
        if r in live_inv else len(self.uops) for r in b))

      for r in block:
        if r in live_inv and (ev := live_inv.get(r)) in live:
          live.pop(ev)
      return block

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:VRegister, i:int, cons:tuple[Register, ...]|None=None) -> tuple[Register,...]:
      if v not in self.spills:
        self.spills[v] = ctx.assign_spill_slot(v, self.vdef(v))
      rs = alloc(v, [cons] if cons is not None else None, i)
      self.insert_before.setdefault(i, []).append((v, rs))
      return rs

    def lslot(v:VRegister, rs:tuple[Register,...]) -> tuple[Register,...]:
      return rs[v.pos:v.pos+v.width] if v.is_sub() and v.pos is not None else rs
    for i,u in enumerate(self.uops):
      # allocate uses
      for s in u.src:
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance((sv:=rdef(s)), VRegister): continue
        if (vv := sv.or_parent()) not in live: live[vv] = fill(vv,i)
        self.reals.setdefault(i, {})[sv] = lslot(sv, live[vv])

      # allocate defs
      vdefs = [v for v in rdefs(u) if isinstance(v, VRegister)]
      for j,v in enumerate(vdefs):
        cons: list[tuple[Register,...]]|None = None
        if self.ren.is_two_address(u) and j == 0:
          use_regs = [rs for s in u.src if isinstance((vr := rdef(s)), VRegister) and (rs := live.get(vr, None)) is not None]
          if use_regs:
            pin, others = use_regs[0], {r for rs in use_regs[1:] for r in rs}
            cands = [b for b in v.candidates() if b != pin]
            free = [b for b in cands if others.isdisjoint(b)]
            cons = ([pin] if pin[0] in v.cons else []) + (free or cands)
        # parents can be defined by premature subregister op ex. collect then store
        if (vv := v.or_parent()) not in live:
          live[vv] = alloc(vv, cons, i+1 if u.op is not Ops.RANGE else i)
        self.reals.setdefault(i, {})[v] = lslot(v, live[vv])

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        rvr = rdef(u)
        assert isinstance(rvr, VRegister)
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < lr[rvr][-1] for l in lr[v])]
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

# push the register down the chain, inverse rdef() semantics
def retag(s:UOp, tag:tuple) -> UOp:
  if s.op in {Ops.AFTER, Ops.NOOP, Ops.BITCAST} and len(s.src): return s.replace(src=(retag(s.src[0], tag), *s.src[1:]))
  return s.replace(tag=tag)

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i, nsrc, before, after = next(ctx.idx), [], [], []
  for j,s in enumerate(x.src):
    if i in ctx.reals and isinstance((v := rdef(ctx.uops[i].src[j])), VRegister) and v.or_parent() in ctx.spills:
      nsrc.append(retag(s, ctx.reals[i][v]))
    else:
      nsrc.append(s)

  ndefs: list[Any] = []
  for v in rdefs(x):
    if isinstance(v, VRegister): ndefs.extend(ctx.reals[i][v])
    else: ndefs.append(v)
  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))

  for v in rdefs(x):
    if not isinstance(v, VRegister): continue
    if v in ctx.spills:
      after.extend(ctx.ren.spill(ctx.spills[v], nx))
  for v,rs in ctx.insert_before.get(i, []):
    before.extend(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), rs)[1])

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat(REG_OPS, name="x"), regalloc_rewrite),
])

def regspace(buf:UOp, c:UOp, x:UOp):
  if not len(defs := rdefs(buf)): return None
  if isinstance(vr := defs[0], Register):
    if c.val >= len(defs): return None
    nx = x.replace(tag=(defs[c.val],))
  else:
    if (buf.dtype.itemsize//4)*c.val >= vr.width: return None
    nx = x.replace(tag=(vr[c.val*2:c.val*2+1] if x.dtype.itemsize > 4 else vr[c.val],))
  return nx, [nx]

def propagate_subs(ctx, x:UOp):
  # a STACK over pinned defs (reg BUFFER loads) needs no virtual register, it just collects the pinned srcs
  if len(x.src) and all(isinstance(rdef(s), Register) for s in x.src):
    defs = tuple(r for s in x.src for r in rdefs(s) if isinstance(r, Register))
    if all(b.index == a.index+1 for a,b in zip(defs, defs[1:])): return x.replace(tag=defs)

  def _strip(x:UOp): return x.src[0] if x.op in {Ops.BITCAST, Ops.AFTER} else x
  vr, nsrc = rdef(x), []
  assert isinstance(vr, VRegister)
  n = vr.width//len(x.src) if len(x.src) and vr.width%len(x.src) == 0 else max(x.dtype.itemsize//4, 1)
  for i,s in enumerate(x.src):
    sub = vr[i*n:(i+1)*n-1]
    # pinned defs and INDEX srcs have to become copies, must enforce contiguity restraint.
    if isinstance(rdef(s), Register) or _strip(s).op is Ops.INDEX: nsrc.append(ctx.ren.copy(s, sub)[0])
    else: nsrc.append(s.replace(tag=(sub,)))
  return x.replace(src=tuple(nsrc))

pm_prepare_regalloc = PatternMatcher([
  (UPat(Ops.STACK, name="x"), propagate_subs),
  (UPat((Ops.AFTER, Ops.BITCAST), name="x"), lambda x:
    x.replace(src=(x.src[0].replace(tag=x.tag), *x.src[1:])) if x.tag is not None else None),
])

pm_index_subregisters = PatternMatcher([
  (UPat.var("buf").index(UPat.cvar("c").cast(), name="x", tag=None), regspace)
])
