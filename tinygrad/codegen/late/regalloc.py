import itertools
from dataclasses import dataclass
from tinygrad.helpers import dedup
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, AddrSpace
from tinygrad.renderer.isa import ISARenderer, Register, VRegister, rdefs, rdef
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.dtype import dtypes

REG_OPS = {Ops.LOAD, Ops.INS, Ops.GROUP, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL}

class LinearScanRegallocContext:
  # NOTE: wrong for fill regs
  def vdef(self, v:VRegister) -> UOp: return self.uops[self.live_intervals[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops, self.ren, self.idx = [u for u in uops if u.op in REG_OPS], ren, itertools.count()
    self.live_intervals: dict[VRegister, list[int]] = {}

    lr = self.live_intervals
    range_vars: list[VRegister] = []
    def live(u:UOp) -> tuple[VRegister,...]: # account for subregister lifetimes in parent live intervals/ranges
      if u.op is Ops.INDEX and not (u.tag is not None and any(isinstance(v,VRegister) for v in u.tag)): return live(u.src[0]) # hack
      return tuple(r.parent if r.is_sub() else r for r in rdefs(u) if isinstance(r, VRegister))
    for i, u in enumerate(reversed(self.uops)):
      defs, uses = live(u), []
      for s in dedup(u.src): uses.extend(live(s))
      for v in defs + tuple(uses):
        lr.setdefault(v, []).insert(0, len(self.uops) - i - 1)
      for v in defs: # if lifetime of v ends during range, pick latest range and add to lr
        if (n := max((lr[rv][-1] for rv in range_vars if lr[rv][0] <= lr[v][-1] < lr[rv][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: range_vars.append(rdef(u))

    # allocate registers
    self.stack_size: int = 0
    self.locals: dict[UOp, UOp] = {}
    self.spills: dict[Register, int] = {} # mapping from virtual to stack slot
    self.reals: dict[int, dict[VRegister, tuple[Register,...]]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, tuple[Register,...]]]] = {} # fills to be inserted at each program point
    live: dict[VRegister, tuple[Register,...]] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[VRegister, tuple[Register,...]]] = [] # mapping from virtual to real at loop entry

    # allocate the best register. Registers not in live or not used again are free and have priority,
    # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
    def alloc(v:VRegister, cons:list[tuple[Register, ...]], i:int) -> tuple[Register,...]:
      cons = cons or v.candidates()
      live_inv = {r:k for k,v in live.items() for r in v}
      block = max(cons, key=lambda b: min(next((j-i for j in lr[live_inv.get(r)] if j >= i), len(uops)) if r in live_inv else len(uops) for r in b))
      for r in block:
        if r in live_inv and (v := live_inv.get(r)) in live: live.pop(v)
      return block

    # assign register to spilled virtual and record load to be emitted before current uop, also assign it a stack slot
    def fill(v:VRegister, i:int, cons:tuple[Register, ...]|None=None) -> tuple[Register,...]:
      if v not in self.spills:
        # the value of a BUFFER is its 64bit address, XMM registers need 16 bytes
        sz = 16 if v.cons[0].size == 16 else (8 if self.vdef(v).op is Ops.BUFFER else self.vdef(v).dtype.itemsize)
        sz *= v.width
        offset = self.stack_size + (sz - self.stack_size % sz) % sz
        self.spills[v] = offset
        self.stack_size = offset + sz
      rs = alloc(v, [cons] if cons is not None else None, i)
      self.insert_before.setdefault(i, []).append((v, rs))
      return rs

    for i,u in enumerate(self.uops):
      # allocate uses
      for s in u.src:
        # HACK: cause of later hacks to lower range
        if u.op is Ops.END: continue
        if not isinstance(v:=rdef(s), VRegister): continue
        vv = v.parent if v.is_sub() else v
        if vv not in live: live[vv] = fill(vv,i)
        self.reals.setdefault(i, {})[v] = (live[v.parent][v.pos],) if v.is_sub() else live[v]

      # allocate defs
      for j,v in enumerate(rdefs(u)):
        # NOTE: X86 hack to imitate physical register lifetime constraints as vregs
        # - need to fix this
        if not isinstance(v, VRegister): continue
        if v.is_sub() and (vp := v.parent) in live:
          self.reals.setdefault(i, {})[v] = (live[vp][v.pos],)
          continue
        cons = None
        if ren.is_two_address(u) and j == 0:
          uses = []
          for s in u.src:
            if rdef(s) in live: uses.extend(live.get(rdef(s)))
          cons = ([(uses[0],)] if uses[0] in v.cons else []) + [r for r in v.candidates() if r[0] not in uses]
        vv = v.parent if v.is_sub() else v
        # parents can be defined by premature subregister op ex. collect then store
        if vv not in live:
          live[vv] = alloc(vv, cons, i+1 if u.op is not Ops.RANGE else i)
        self.reals.setdefault(i, {})[v] = (live[vv][v.pos],) if v.is_sub() else live[v]

      # loop prologue, avoid loading inside the loop
      if u.op is Ops.RANGE:
        # we move to registers vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < lr[rdef(u)][-1] for l in lr[v])]
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
    self.ren.spill_size = self.stack_size

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  i, nsrc, = next(ctx.idx), []
  for j,s in enumerate(x.src):
    if i in ctx.reals and (v := rdef(ctx.uops[i].src[j])) in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), *ctx.reals[i][v]))
    else: nsrc.append(s)

  ndefs = []
  for v in rdefs(x):
    if isinstance(v, VRegister): ndefs.extend(ctx.reals[i][v])
    else: ndefs.append(v)
  nx = x.replace(src=tuple(nsrc), tag=tuple(ndefs))

  after = [ctx.ren.spill(ctx.spills[v],nx) for v in rdefs(x) if v in ctx.spills]
  before = [ctx.ren.fill(ctx.spills[v],ctx.vdef(v),*rs) for v,rs in ctx.insert_before.get(i, [])]
  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat(REG_OPS, name="x"), regalloc_rewrite),
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
