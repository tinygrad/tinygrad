from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.dtype import dtypes

PSEUDO_OPS = {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.STACK}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:Register) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops = uops
    self.ren = ren
    # the label associated with each loop NOTE: this is only used post regalloc and should be removed
    self.loop_label: dict[UOp, str] = {}

    # compute live ranges
    self.live_range: dict[Register, list[int]] = {}
    lr = self.live_range
    ranges: list[Register] = []
    for i,u in enumerate(reversed(uops)):
      if u.op in PSEUDO_OPS: continue
      defs = u.tag if isinstance(u.tag, tuple) else ()
      for v in defs + tuple(s.reg for s in dedup(u.src)):
        if isinstance(v, Register): lr.setdefault(v, []).insert(0, len(uops) - 1 - i)
      for v in defs:
        if v in lr and (n:=max((lr[rng][-1] for rng in ranges if lr[rng][0] <= lr[v][-1] < lr[rng][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: ranges.append(u.reg)

    # allocate registers
    self.stack_size: int = 0
    self.locals: dict[UOp, UOp] = {}
    self.spills: dict[Register, UOp] = {} # mapping from virtual to stack slot
    self.reals: dict[int, dict[Register, Register]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, Register]]] = {} # fills to be inserted at each program point
    self.regalloc_i = 0
    # prologue/epilogue must attach to real emitted instructions: skip leading/trailing PSEUDO_OPS and the SINK
    # terminator (regalloc_rewrite never processes SINK), otherwise stack alloc/dealloc would be dropped.
    real_idxs = [i for i,u in enumerate(uops) if u.op not in PSEUDO_OPS and u.op is not Ops.SINK]
    self.first_real_idx, self.last_real_idx = (real_idxs[0], real_idxs[-1]) if real_idxs else (-1, -1)
    live: dict[Register, Register] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[Register, Register]] = [] # mapping from virtual to real at loop entry

    def slots(v:Register) -> int: return ren.register_slots(self.vdef(v), v)

    def alloc(cons:tuple[Register, ...], i:int, nslots:int=1, allowed:tuple[Register, ...]|None=None) -> Register:
      allowed_idxs = {r.index for r in (allowed if allowed is not None else cons)}
      def blockers(reg:Register) -> tuple[Register, ...]:
        occupied = set(range(reg.index, reg.index+nslots))
        return tuple(v for v,r in live.items() if occupied & set(range(r.index, r.index+slots(v))))
      def next_use(v:Register) -> int:
        return next((j-i for j in lr[v] if j >= i), len(uops))
      candidates = [(r, blockers(r)) for r in cons if all(x in allowed_idxs for x in range(r.index, r.index+nslots))]
      # allocate the best register. Registers not in live or not used again are free and have priority,
      # otherwise pick the one whose earliest evicted value has the furthest next use. Regs that appear
      # first in cons have priority in case of a tie.
      reg,vregs = max(candidates, key=lambda rv: min((next_use(v) for v in rv[1]), default=len(uops)))
      for v in vregs: live.pop(v, None)
      return reg

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:Register, i:int, cons:tuple[Register, ...]|None=None) -> Register:
      if v not in self.spills:
        # the value of a BUFFER is its 64bit address
        dt = self.vdef(v).dtype
        sz = 8 if self.vdef(v).op is Ops.BUFFER else dt.itemsize
        offset = self.stack_size + (sz - self.stack_size % sz) % sz
        self.spills[v] = UOp.const(dtypes.int32, offset)
        self.stack_size = offset + sz
      r = alloc(cons if cons is not None else v.cons, i, slots(v), v.cons)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      if u.op in PSEUDO_OPS: continue
      # allocate uses
      for s in u.src:
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance(v:=s.reg, Register): continue
        if v not in live: live[v] = fill(v, i)
        self.reals.setdefault(i, {})[v] = live[v]

      # allocate defs
      if isinstance(u.tag, tuple):
        for j,v in enumerate(u.tag):
          # register should only be defined once
          assert isinstance(v, Register) and lr[v][0] == i
          cons = v.cons
          # two address instructions (src is reused by def) can only coalesce reused src. reused src goes first to get priority in case of a tiebreak
          if ren.is_two_address(u) and j == 0:
            uses = tuple(live.get(s.reg) for s in u.src)
            cons = ((uses[0],) if uses[0] in cons else ()) + tuple(r for r in cons if r not in uses)
          # HACK: cause the range is missing the comparison
          live[v] = alloc(cons, i+1 if u.op is not Ops.RANGE else i, slots(v), v.cons)
          self.reals.setdefault(i, {})[v] = live[v]

      # allocate stack array
      if u.op is Ops.BUFFER:
        self.locals[u] = UOp.const(dtypes.int32, self.stack_size)
        self.stack_size += u.max_numel() * u.dtype.itemsize

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < lr[u.reg][-1] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.name, k.index))
        live_in: dict[Register, Register] = {}
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
        for v,r in live_ins.pop().items():
          if v not in live or live[v] != r: live[v] = fill(v, i, (r,))

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = ctx.regalloc_i
  if x.op in PSEUDO_OPS: return None
  if x.op in (Ops.LOAD, Ops.STORE) and not ctx.insert_before.get(i):
    spilled = any(i in ctx.reals and isinstance(v:=ctx.uops[i].src[j].reg, Register) and v in ctx.spills
                  for j in range(len(x.src)))
    if not spilled and i not in (ctx.first_real_idx, ctx.last_real_idx): return None
  nsrc = []
  for j,s in enumerate(x.src):
    # v here is the virtual defined by the original s as s is the rewritten version
    if i in ctx.reals and (v:=ctx.uops[i].src[j].reg) in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), ctx.reals[i][v]))
    else: nsrc.append(s)
  ndefs = tuple(ctx.reals[i][v] for v in x.tag) if isinstance(x.tag, tuple) else x.tag
  if x.op is Ops.BUFFER: nx = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(ctx.locals[x], tag=ndefs))
  else: nx = x.replace(src=tuple(nsrc), tag=ndefs)

  before = [ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r) for v,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[v], nx) for v in x.tag if v in ctx.spills] if isinstance(x.tag, tuple) else []

  # alloc/dealloc stack
  if ctx.stack_size > 0:
    sp = ctx.ren.stack_pointer()
    offset = UOp(Ops.CONST, sp.dtype, arg=ctx.stack_size)
    if i == ctx.first_real_idx: before = [ctx.ren.isel_matcher.rewrite(UOp(Ops.SUB, sp.dtype, (sp, offset), tag=sp.tag))] + before
    elif i == ctx.last_real_idx: before += [ctx.ren.isel_matcher.rewrite(UOp(Ops.ADD, sp.dtype, (sp, offset), tag=sp.tag))]

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL, Ops.SHRINK, Ops.LOAD, Ops.STORE} | PSEUDO_OPS, name="x"),
        regalloc_rewrite),
])
