import itertools
from tinygrad.device import CompileError
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register, greg
from tinygrad.dtype import dtypes

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.STACK}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:Register) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops = uops
    self.ren = ren
    self.wide = ren.wide_regalloc
    self.idx = itertools.count()
    self.regalloc_i = 0
    # the label associated with each loop NOTE: this is only used post regalloc and should be removed
    self.loop_label: dict[UOp, str] = {}

    # compute live ranges
    self.live_range: dict[Register, list[int]] = {}
    lr = self.live_range
    ranges: list[Register] = []
    for i,u in enumerate(reversed(uops)):
      if u.op in PSEUDO_OPS: continue
      defs = u.tag if isinstance(u.tag, tuple) else ()
      for v in defs + tuple(greg(s) for s in dedup(u.src)):
        if isinstance(v, Register): lr.setdefault(v, []).insert(0, len(uops) - 1 - i)
      for v in defs:
        if v in lr and (n:=max((lr[rng][-1] for rng in ranges if lr[rng][0] <= lr[v][-1] < lr[rng][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: ranges.append(greg(u))

    # allocate registers
    self.stack_size: int = 0
    self.locals: dict[UOp, UOp] = {}
    self.spills: dict[Register, UOp] = {} # mapping from virtual to stack slot
    self.remat: set[Register] = set()
    self.reals: dict[int, dict[Register, Register]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, Register]]] = {} # fills to be inserted at each program point
    if self.wide:
      real_idxs = [i for i,u in enumerate(uops) if u.op not in PSEUDO_OPS and u.op is not Ops.SINK]
      self.first_real_idx, self.last_real_idx = (real_idxs[0], real_idxs[-1]) if real_idxs else (-1, -1)
    live: dict[Register, Register] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[Register, Register]] = [] # mapping from virtual to real at loop entry

    def slots(v:Register) -> int: return ren.register_slots(self.vdef(v), v)

    pinned: set[int] = set()  # live source phys regs; defs must not steal (except two-address)

    def alloc(cons:tuple[Register, ...], i:int, v:Register|None=None, *, pin:bool=True) -> Register:
      if self.wide:
        from tinygrad.renderer.isa.rdna3 import wide_alloc
        assert v is not None
        return wide_alloc(cons, i, slots(v), v.cons, live, lr, len(uops), slots, pinned if pin else frozenset())
      live_inv = {rv:k for k,rv in live.items()}
      reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                    key=lambda rv: next((j-i for j in ([] if rv[1] is None else lr[rv[1]]) if j >= i), len(uops)))
      return live.pop(vreg) if vreg is not None else reg

    def fill(v:Register, i:int, cons:tuple[Register, ...]|None=None, *, pin:bool=True) -> Register:
      vd = self.vdef(v)
      if ren.rematerialize(vd):
        self.remat.add(v)
        for s in vd.src:
          if s.op is Ops.CONST: continue
          if isinstance(sv:=greg(s), Register):
            if sv not in live: live[sv] = fill(sv, i, pin=pin)
            self.reals.setdefault(i, {})[sv] = live[sv]
            if pin: pinned.update(range(live[sv].index, live[sv].index + slots(sv)))
      elif v not in self.spills:
        sz = 16 if v.cons[0].size == 16 else (8 if vd.op is Ops.BUFFER else vd.dtype.itemsize)
        offset = self.stack_size + (sz - self.stack_size % sz) % sz
        self.spills[v] = UOp.const(offset, dtypes.int32)
        self.stack_size = offset + sz
      r = alloc(cons if cons is not None else v.cons, i, v, pin=pin)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      if u.op in PSEUDO_OPS: continue
      pinned = set()
      for s in u.src:
        if u.op is Ops.END: continue
        if not isinstance(v:=greg(s), Register): continue
        # Remat usually rebuilds at every use; keep_remat ops reuse the phys reg.
        if v in self.remat and not ren.keep_remat(self.vdef(v)): live.pop(v, None)
        if v not in live: live[v] = fill(v, i)
        self.reals.setdefault(i, {})[v] = live[v]
        pinned.update(range(live[v].index, live[v].index + slots(v)))

      if isinstance(u.tag, tuple):
        for j,v in enumerate(u.tag):
          assert isinstance(v, Register) and lr[v][0] == i
          cons = v.cons
          if ren.is_two_address(u) and j == 0:
            uses = tuple(live.get(greg(s)) for s in u.src)
            if self.wide and uses[0] is not None and uses[0] in cons:
              live[v] = uses[0]
              self.reals.setdefault(i, {})[v] = uses[0]
              continue
            cons = ((uses[0],) if uses[0] is not None and uses[0] in cons else ()) + tuple(r for r in cons if r not in uses)
          elif j == 0 and (pref:=ren.prefer_phys(u, [live.get(greg(s)) for s in u.src])) is not None and pref in cons:
            # Alias onto a src sub-register (e.g. EXTRACT → WMMA pack+lane) — skip pinned check.
            live[v] = pref
            self.reals.setdefault(i, {})[v] = pref
            continue
          if pinned:
            filtered = tuple(r for r in cons if not (set(range(r.index, r.index + slots(v))) & pinned))
            if filtered: cons = filtered
            elif len(cons) > 1:
              raise CompileError(f"no unpinned regs for {v}")
            # len==1: dest constrained to one phys (may alias a pinned src) — allow
          live[v] = alloc(cons, i+1 if u.op is not Ops.RANGE else i, v)
          self.reals.setdefault(i, {})[v] = live[v]

      for rv in [rv for rv in live if rv in self.remat and not ren.keep_remat(self.vdef(rv))]: live.pop(rv, None)

      if u.op is Ops.BUFFER:
        self.locals[u] = UOp.const(self.stack_size, dtypes.int32)
        self.stack_size += u.max_numel() * u.dtype.itemsize

      if u.op is Ops.RANGE:
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < lr[greg(u)][-1] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.name, k.index))
        live_in: dict[Register, Register] = {}
        for v in sorted_uses:
          if set(v.cons).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          live_in[v] = live[v]
        live_ins.append(live_in)

      if u.op is Ops.END:
        # loop-carried restores need exact phys regs
        for v,r in live_ins.pop().items():
          if v not in live or live[v] != r: live[v] = fill(v, i, (r,), pin=False)

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  if ctx.wide:
    from tinygrad.renderer.isa.rdna3 import wide_regalloc_rewrite
    return wide_regalloc_rewrite(ctx, x)
  if x.op in (Ops.LOAD, Ops.STORE, Ops.SHRINK): return None
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None

  nsrc = []
  for j,s in enumerate(x.src):
    if i in ctx.reals and (v:=greg(ctx.uops[i].src[j])) in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), ctx.reals[i][v]))
    else: nsrc.append(s)
  ndefs = tuple(ctx.reals[i][v] for v in x.tag) if isinstance(x.tag, tuple) else x.tag
  if x.op is Ops.BUFFER: nx = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(ctx.locals[x], tag=ndefs))
  else: nx = x.replace(src=tuple(nsrc), tag=ndefs)

  before = [ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r) for v,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[v], nx) for v in x.tag if v in ctx.spills] if isinstance(x.tag, tuple) else []

  if ctx.stack_size > 0:
    sp = ctx.ren.stack_pointer()
    offset = UOp.const(ctx.stack_size, sp.dtype)
    if i == 0: before = [ctx.ren.isel_matcher.rewrite(UOp(Ops.SUB, src=(sp, offset), tag=sp.tag))] + before
    elif i == len(ctx.uops) - 2: before += [ctx.ren.isel_matcher.rewrite(UOp(Ops.ADD, src=(sp, offset), tag=sp.tag))]

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL, Ops.SHRINK, Ops.LOAD, Ops.STORE} | PSEUDO_OPS, name="x"),
        regalloc_rewrite),
])
