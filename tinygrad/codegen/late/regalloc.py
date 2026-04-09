import itertools
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.dtype import dtypes, PtrDType

PSEUDO_OPS = {Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP}
def _uop_key(u:UOp): return (u.op, u.dtype, u.arg)

# loosely based on: https://bernsteinbear.com/assets/img/register-spilling-range-splitting-ssa.pdf
class LinearScanRegallocContext:
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    live_range: dict[Register, list[int]] = {}
    live: dict[Register, Register] = {}
    live_ins: list[dict[Register, Register]] = []
    self.defs: dict[Register, UOp] = {} # mapping from virtual to uop that defines it
    self.real_defs: dict[Register, Register] = {} # mapping from virtual to real at definition
    self.spills: dict[Register, UOp] = {} # mapping from virtual to stack slot
    self.fills: dict[int, dict[int, tuple[Register, Register]]] = {} # mapping from program point to mapping from idx to virtual and real to fill to
    self.insert_before: dict[int, list[tuple[Register, Register]]] = {} # mapping from program point to fills to be inserted
    self.idx = itertools.count()
    self.ren = ren
    self.stack_size = 0
    # the label associated with each loop NOTE: this is only used post regalloc and should be removed
    self.loop_label: dict[UOp, str] = {}
    arg_order = {Ops.PARAM: 0, Ops.DEFINE_VAR: 1, Ops.SPECIAL: 2}
    self.func_arg_idxs = {_uop_key(u): i for i,u in enumerate(sorted({u for u in uops if u.op in arg_order}, key=lambda k: (arg_order[k.op], k.arg)))}
    self.local_offsets: dict[tuple, int] = {}
    for u in uops:
      if u.op not in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG): continue
      self.local_offsets.setdefault(_uop_key(u), self.stack_size)
      self.stack_size += u.dtype.nbytes()
    # compute live ranges
    lr = live_range
    ranges: list[Register] = []
    for i,u in enumerate(reversed(uops)):
      if u.op in PSEUDO_OPS: continue
      defs = u.tag if isinstance(u.tag, tuple) else ()
      for v in defs + tuple(s.reg for s in set(u.src)):
        if isinstance(v, Register): lr.setdefault(v, []).insert(0, len(uops) - 1 - i)
      for v in defs:
        if isinstance(v, Register): self.defs[v] = u
        if v in lr and (n:=max((lr[rng][-1] for rng in ranges if lr[rng][0] <= lr[v][-1] < lr[rng][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: ranges.append(u.reg)

    def alloc(cons:tuple[Register, ...], i:int) -> Register:
      live_inv = {v:k for k,v in live.items()}
      # allocate the best register. Registers not in live or not used again are free and have priority,
      # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
      reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                    key=lambda rv: next((j-i for j in ([] if rv[1] is None else live_range[rv[1]]) if j >= i), len(uops)))
      return live.pop(vreg) if vreg is not None else reg

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:Register, i:int, cons:tuple[Register, ...]|None=None) -> Register:
      if v not in self.spills:
        dt = self.defs[v].dtype
        sz = dt.scalar().itemsize * dt.count if not isinstance(dt, PtrDType) else 8
        assert sz > 0
        offset = self.stack_size + (sz - self.stack_size % sz) % sz
        self.spills[v] = UOp.const(dtypes.int32, offset)
        self.stack_size = offset + sz
      r = alloc(cons if cons is not None else v.cons, i)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      if u.op in PSEUDO_OPS: continue
      # allocate uses
      for j,s in enumerate(u.src):
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance(v:=s.reg, Register): continue
        if v not in live: live[v] = fill(v, i)
        if v in self.spills: self.fills.setdefault(i, {})[j] = (v, live[v])

      # allocate defs
      if isinstance(u.tag, tuple):
        for j,v in enumerate(u.tag):
          assert isinstance(v, Register) and v not in live
          cons = v.cons
          # two address instructions (src is reused by def) can only coalesce reused src. reused src goes first to get priority in case of a tiebreak
          if ren.is_two_address(u) and j == 0:
            ins = tuple(live.get(s.reg) for s in u.src)
            cons = ((ins[0],) if ins[0] in cons else ()) + tuple(r for r in cons if r not in ins)
            assert cons
          # HACK: cause the range is missing the comparison
          self.real_defs[v] = live[v] = alloc(cons, i+1 if u.op is not Ops.RANGE else i)

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < live_range[u.reg][-1] for l in live_range[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: next(l-i for l in live_range[k] if l >= i))
        live_in: dict[Register, Register] = {}
        for v in sorted_uses:
          # if all the possible registers are already in live_in there's no space for this var
          if set(v.cons).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          assert live[v] not in live_in.values()
          live_in[v] = live[v]
        live_ins.append(live_in)

      # loop epilogue, reload registers that were live at loop entry
      if u.op is Ops.END:
        # TODO: if a uop is in a different reg in live out vs live in move between registers instead of loading
        # TODO: don't reload if first use in loop is a load
        for v,r in live_ins.pop().items():
          if v not in live or live[v] != r: live[v] = fill(v, i, (r,))

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None
  nsrc = []
  for j,s in enumerate(x.src):
    if i in ctx.fills and j in ctx.fills[i]:
      v,r = ctx.fills[i][j]
      nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.defs[v], r))
    else: nsrc.append(s)
  ndefs = tuple(ctx.real_defs[v] for v in x.tag) if isinstance(x.tag, tuple) else x.tag
  nx = x.replace(src=tuple(nsrc), tag=ndefs)
  fills = [ctx.ren.fill(ctx.spills[v], ctx.defs[v], r) for v,r in ctx.insert_before.get(i, [])]
  spills = [ctx.ren.spill(ctx.spills[v], nx) for v in x.tag if v in ctx.spills] if isinstance(x.tag, tuple) else []
  return nx, fills + [nx] + spills

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.CONST, Ops.RANGE, Ops.END, Ops.NOOP, Ops.GROUP, Ops.AFTER, Ops.BARRIER,
         Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}, name="x"), regalloc_rewrite),
])
