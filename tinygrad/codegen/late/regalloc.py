import itertools
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.uop import X86GroupOp
from tinygrad.renderer.x86 import ISARenderer, Register
from tinygrad.dtype import dtypes, DType, PtrDType

# loosely based on: https://bernsteinbear.com/assets/img/register-spilling-range-splitting-ssa.pdf
class RegallocContext:
  def __init__(self, uops:list[UOp], ren:ISARenderer, stack_size:int=0):
    self.live_range: dict[Register, list[int]] = {}
    self.live: dict[Register, Register] = {}
    self.spills: dict[Register, UOp] = {}
    self.rewrite_to_vreg: dict[UOp, Register] = {}
    self.vreg_to_rewrite: dict[Register, UOp] = {}
    self.live_ins: list[dict[Register, Register]] = []
    self.idx = itertools.count()
    self.stack_size: int = stack_size
    self.ren = ren
    # live ranges, first pass builds ranges
    for i,u in enumerate(uops):
      if u.op in (Ops.NOOP, Ops.AFTER): continue
      if isinstance(u.arg, Register): self.live_range[u.arg] = [i]
      for v in set([s.arg for s in u.src if s.arg in self.live_range]): self.live_range[v].append(i)
    # second pass updates end of range, a var defined before a range and used inside it is needed for the whole range
    ranges: list[Register] = []
    for i,u in enumerate(reversed(uops)):
      for v in [s.arg for s in u.src if s.arg in self.live_range]:
        end = next((self.live_range[rng][-1] for rng in ranges if self.live_range[v][0] < self.live_range[rng][0]), 0)
        if end > self.live_range[v][-1]: self.live_range[v].append(end)
      if u.op is Ops.END: ranges.append(u.src[1].arg)
      if u.op is Ops.RANGE: ranges.pop()

# TODO: rm pointers
# nasty hacks to deal with pointers
def assign(ctx:RegallocContext, x:UOp, reg:Register):
  dt = dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype
  ret = ctx.ren.isel_matcher.rewrite(UOp(Ops.ASSIGN, dt, (x,), reg))
  assert ret is not None
  return ret.replace(dtype=x.dtype)
def load(ctx:RegallocContext, dt:DType, disp:UOp, reg:Register):
  ndt = dtypes.uint64 if isinstance(dt, PtrDType) else dt
  ret = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(disp).load(dtype=ndt, arg=reg))
  assert ret is not None
  return ret.replace(dtype=dt)
def store(ctx:RegallocContext, disp:UOp, x:UOp):
  nx = x.replace(dtype=dtypes.uint64 if isinstance(x.dtype, PtrDType) else x.dtype)
  ret = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(disp).store(nx))
  assert ret is not None
  return ret.replace(src=(s if s is not nx else x for s in ret.src))

def alloc(ctx:RegallocContext, cons:tuple[Register, ...], i:int) -> Register:
  live_inv = {v:k for k,v in ctx.live.items()}
  # allocate the best register. Registers not in live or not used again are free and have priority,
  # otherwise pick the one with the furthest next use. Regs that appear first in cons have priority in case of a tie
  reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                key=lambda rv: next((j-i for j in ([] if rv[1] is None else ctx.live_range[rv[1]]) if j >= i), float('inf')))
  if vreg is not None and vreg not in ctx.spills and ctx.live_range[vreg][-1] >= i:
    sz = ctx.vreg_to_rewrite[vreg].dtype.itemsize if not isinstance(ctx.vreg_to_rewrite[vreg].dtype, PtrDType) else 8
    assert sz > 0
    offset = ctx.stack_size + (sz - ctx.stack_size % sz) % sz
    ctx.spills[vreg] = UOp.const(dtypes.int32, offset)
    ctx.stack_size = offset + sz
  return ctx.live.pop(vreg, reg)

def regalloc(ctx:RegallocContext, x:UOp, i:int) -> tuple[UOp, list[UOp]]:
  nsrc, loads = [], []
  for s in x.src:
    # allocate srcs, if src was spilled it's replaced by a load, if it's live the load was already emited otherwise alloc and emit one
    if isinstance(s.arg, Register) and (v:=ctx.rewrite_to_vreg[s]) in ctx.spills:
      # TODO: the constraints only apply to the definition, you need to insert moves in the graph to "cleanse" the constraint
      # then those moves are removed after regalloc if they move to the same register. I think this is the llvm approach
      # alternatively you could beef up the register class to include constraints on the srcs, then you check those here
      if v not in ctx.live:
        ctx.live[v] = alloc(ctx, v.cons or (v,), i)
        s = load(ctx, s.dtype, ctx.spills[v], ctx.live[v])
        loads.append(s)
      else: s = load(ctx, s.dtype, ctx.spills[v], ctx.live[v])
    nsrc.append(s)
  # allocate destination
  if isinstance(v:=x.arg, Register) and v not in ctx.live:
    # if no cons it's a real register, so it can only be assigned to itself
    cons = v.cons or (v,)
    # two address instructions (src is used in dest) can only coalesce reused src. reused src goes first to get priority in case of a tiebreak
    if (j:=ctx.ren.two_address(x)) is not None:
      cons = (ctx.live[ctx.rewrite_to_vreg[x.src[j]]],) + \
        tuple(r for r in cons if r not in tuple(ctx.live.get(ctx.rewrite_to_vreg[s]) for s in x.src))
    ctx.live[v] = alloc(ctx, cons, i+1)

  nx = x.replace(src=tuple(nsrc), arg=ctx.live.get(v, v))
  ctx.rewrite_to_vreg[nx] = v
  if v not in ctx.vreg_to_rewrite: ctx.vreg_to_rewrite[v] = nx
  return nx, loads + [nx]

# move uops to registers before the loop to avoid loading inside the loop
def loop_prologue(ctx:RegallocContext, x:UOp, i:int):
  assert isinstance(x.arg, Register)
  nx, lst = regalloc(ctx, x, i)
  # we move to register vars used in the loop sorted by next use, vars not used in the loop will not be reloaded in the epilogue
  used_in_loop = [v for v in ctx.live.keys() | ctx.spills.keys() if any(i <= l < ctx.live_range[x.arg][-1] for l in ctx.live_range[v])]
  sorted_uses = sorted(used_in_loop, key=lambda k: next(l-i for l in ctx.live_range[k] if l >= i))
  live_in: dict[Register, Register] = {}
  loads = []
  for v in sorted_uses:
    # if all the possible registers are already in live_in there's no space for this var
    if set(v.cons or (v,)).issubset(live_in.values()): continue
    if v not in ctx.live:
      ctx.live[v] = alloc(ctx, v.cons or (v,), i)
      s = ctx.vreg_to_rewrite[v]
      loads.append(load(ctx, s.dtype, ctx.spills[v], ctx.live[v]))
    assert ctx.live[v] not in live_in.values()
    live_in |= {v: ctx.live[v]}
  ctx.live_ins.append(live_in)
  return nx, loads + lst

# reload registers that were live at loop entry
def loop_epilogue(ctx:RegallocContext, x:UOp, i:int):
  # TODO: if a uop is in a different reg in live out vs live in move between registers instead of loading
  # TODO: don't reload if first use in loop is a load
  loads = []
  for k,v in ctx.live_ins.pop().items():
    if k not in ctx.live or ctx.live[k] != v:
      ctx.live[k] = alloc(ctx, (v,), i)
      s = ctx.vreg_to_rewrite[k]
      loads.append(load(ctx, s.dtype, ctx.spills[k], ctx.live[k]))
  return x, loads + [x]

pm_regalloc = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: loop_prologue(ctx, x, next(ctx.idx))),
  (UPat(Ops.END, name="x"), lambda ctx,x: loop_epilogue(ctx, x, next(ctx.idx))),
  (UPat(X86GroupOp.All | {Ops.NOOP, Ops.GROUP, Ops.AFTER, Ops.CONST, Ops.BARRIER}, name="x"), lambda ctx,x: regalloc(ctx, x, next(ctx.idx))),
])

# annoying that this is another pm
pm_insert_spills = PatternMatcher([
  # insert spill after definition
  (UPat(X86GroupOp.All | {Ops.RANGE}, name="x"), lambda ctx,x:
   (x, [x, store(ctx, y, x)]) if (y:=ctx.spills.get(ctx.rewrite_to_vreg.get(x))) is not None else None),
])
