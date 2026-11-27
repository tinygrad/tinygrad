import itertools, sys
from dataclasses import dataclass, field, replace
from tinygrad.uop.ops import PatternMatcher, UOp, UPat, GroupOp, Ops, graph_rewrite, KernelInfo, BottomUpGate, resolve, sint, AddrSpace
from tinygrad.uop.ops import AxisType, rewrite_group, SENTINEL, StageOpts
from tinygrad.uop.symbolic import symbolic, invalid_pat
from tinygrad.uop.spec import type_verify, spec_kernel_graph
from tinygrad.codegen.simplify import pm_reduce_simplify
from tinygrad.helpers import dedup, VIZ, SPEC, MAX_KERNEL_BUFFERS, to_tuple
from tinygrad.codegen.prepare import pm_gated_index
from tinygrad.codegen.indexing import apply_movement_op

sys.setrecursionlimit(10000)

def new_ranges(shape:tuple[sint, ...], rid=itertools.count(1)) -> tuple[UOp, ...]:
  return tuple(UOp.range(sz, -next(rid), AxisType.WEAK) for sz in shape)

def broadcast_ranges(x:UOp, s:UOp, rngs:tuple[UOp, ...]) -> tuple[UOp, ...]:
  if x.op not in GroupOp.Broadcastable: return rngs
  return tuple(UOp.const(0) if resolve(sz == 1, default=False) else r for sz,r in zip(s.shape, rngs[len(x.shape)-len(s.shape):]))

pm_add_ranges = PatternMatcher([
  (UPat.var("dst").store(UPat.var("src")), lambda dst, src: dst.index(*(r:=new_ranges(dst.shape))).store(src.index(*r)).end(*r)),
  (UPat.var("x").reduce(name="r", allow_any_len=True), lambda r, x: x.reduce(*new_ranges(x.shape[:r.arg[1]]), *r.src[1:], arg=r.arg)),
  (UPat.var("x").stage(name="st", allow_any_len=True),
  lambda st, x: x.index(*(r:=new_ranges(x.shape))).stage(*st.src[1:], *r, arg=st.arg).view_as(st.shape)),
])

pm_push_ranges = PatternMatcher([
  (UPat.var("x").index(name="j", allow_any_len=True).index(name="i", allow_any_len=True), lambda i, j, x: x.index(*j.src[1:], *i.src[1:])),
  (UPat(GroupOp.Elementwise, name="x").index(name="ind", allow_any_len=True),
  lambda ind, x: x.replace(src=tuple(s if s.op is Ops.CONST else s.index(*broadcast_ranges(x, s, ind.src[1:])) for s in x.src))),
  (UPat.var("x").reduce(name="r", allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, r, x: x.index(*(rr:=r.src[1:])[:r.arg[1]], *ind.src[1:]).reduce(*rr, arg=(r.arg[0], 0))),
  (UPat(GroupOp.Movement, name="m", src=(UPat.var("x"),), allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, m, x: x.index(*apply_movement_op(m.op, x.shape, m.marg, ind.src[1:]))),
])

def _stack_select(rcmp:UOp, srcs:list[UOp]) -> UOp:
  # MSB-first bit tree.
  if len(srcs) == 1: return srcs[0]
  half = 1 << ((len(srcs)-1).bit_length()-1)
  return (rcmp % (2*half) < half).where(_stack_select(rcmp, srcs[:half]), _stack_select(rcmp, srcs[half:]))

def convert_stack_to_where(ind:UOp, x:UOp) -> UOp: return _stack_select(ind.src[1], [s.index(*ind.src[2:]) for s in x.src])

def convert_index_to_where(ind:UOp, x:UOp) -> UOp:
  ret = x.src[0].index(*(axes:=apply_movement_op(x.op, x.src[0].shape, x.marg, ind.src[1:]))) if x.op is Ops.PAD else ind
  gates = [r.get_valid() for r in (axes if x.op is Ops.PAD else ind.src[1:])]
  return UOp.const(True).uprod(*gates).where(ret, UOp.const(x.dtype.const(0))) if any(g is not UOp.const(True) for g in gates) else ret

def fold_invalid_after(x:UOp, ind:UOp, e:UOp, i:UOp) -> UOp|None:
  covered = all(r in UOp.sink(*ind.src[1:]).ranges for r in e.src[1:])
  return i if covered and resolve(UOp.const(1).uprod(*[r.src[0] for r in e.src[1:]]).eq(x.numel()), default=False) else None

pm_convert_index_to_where = PatternMatcher([(UPat.var("x").index(name="ind", allow_any_len=True), convert_index_to_where)])

pm_convert_ranges = PatternMatcher([
  (UPat(Ops.STACK, name="x").index(name="ind", allow_any_len=True), convert_stack_to_where),
  (UPat(Ops.PAD, name="x").index(name="ind", allow_any_len=True), convert_index_to_where),
  (UPat.var("x").after(UPat.var("x").index(name="ind", allow_any_len=True).store(invalid_pat).end(allow_any_len=True, name="e"))
  .index(allow_any_len=True), fold_invalid_after)
])

def kernel_boundary(x:UOp) -> bool:
  if x.op is Ops.STAGE: return x.arg is not None and x.arg.addrspace is AddrSpace.GLOBAL and not x.arg.removable
  return x.has_buffer_identity(after_ok=True) or x.op is Ops.CALL

def call_arg(x:UOp) -> bool:
  if x.op is Ops.RANGE: return x.axis_type is AxisType.DEVICE
  return kernel_boundary(x) or ((p:=x.without_after).op in {Ops.PARAM,Ops.EXTERN} and p.arg.addrspace is AddrSpace.ALU)

def count_uses(tsink:UOp) -> dict[UOp, UOp]:
  uses: dict[UOp, UOp] = {tsink:tsink}
  stages: dict[UOp, UOp] = {}
  for x in reversed(tsink.toposort()):
    if (use:=uses[x]) is SENTINEL and x.op in GroupOp.Elementwise|{Ops.REDUCE,Ops.SHRINK} and not x.is_virtual:
      stages.setdefault(x, x.stage(arg=StageOpts(device=x.device or tsink.device, removable=True)))
    # A kernel, a movement op, or a REDUCE are all distinct uses. If a uop receives >1 distinct use, it gets staged.
    if x in stages or kernel_boundary(x) or x.op in GroupOp.Movement|{Ops.REDUCE} and use is not SENTINEL: use = x
    for s in x.src:
      src_use = SENTINEL if x.op in GroupOp.Broadcastable|{Ops.EXPAND} and x.max_numel() >= 2*s.max_numel() else use
      if uses.setdefault(s, src_use) is not src_use: uses[s] = SENTINEL
    # MSTACK, MSELECT, and custom kernel or program srcs are always materialized. MSTACK and MSELECT srcs drop their views.
    if x.op is Ops.MSTACK or x.op is Ops.MSELECT or x.op is Ops.CALL and (x.body.op is Ops.PROGRAM or isinstance(x.body.arg, KernelInfo)):
      for s in x.src_without_body:
        if not call_arg(a:=s if x.op is Ops.CALL else s.base): stages[a] = a.stage(arg=StageOpts(device=a.device or tsink.device, removable=False))
  return {k:v for k,v in stages.items() if not v.arg.removable or k.ndim > 0}

pm_insert_stages = PatternMatcher([
  (UPat(GroupOp.All-{Ops.STAGE}, name="x"), lambda ctx, x: x.replace(src=tuple(ctx.get(s, s) for s in x.src)))
])

def split_graph(src:UOp, device=None) -> UOp:
  src_ranges = [r for r in src.ranges if r.axis_type is not AxisType.DEVICE]
  src_shape = tuple(s if (r.op is Ops.RANGE and (s:=r.src[0]).op is not Ops.CONST) else int(r.vmax+1) for r in src_ranges)
  st = src.substitute(dict(zip(src_ranges, rngs:=new_ranges(src_shape))), extra_pm=pm_gate_ranges).stage(*rngs, arg=StageOpts(src.device or device))
  return st.index(*src_ranges)

@dataclass
class BufferCtx:
  info: dict[UOp, tuple[list[UOp], bool]] = field(default_factory=dict)
  count_vars: bool = False
  has_limit: bool = False

def buffer_limit(d:str|tuple[str, ...]|None) -> int: return MAX_KERNEL_BUFFERS.value or {'WEBGPU':8}.get(str(to_tuple(d)[0]).split(":")[0], 0)

def count_srcs(ctx, op, src):
  buffers, reduce = [], False
  for s in src:
    if kernel_boundary(s) or (ctx.count_vars and s.is_variable): buffers.append(s)
    else: buffers, reduce = buffers + ctx.info[s][0], reduce or ctx.info[s][1]
  return dedup(buffers), reduce or (op is Ops.REDUCE and buffers)

def count_buffers(ctx, x:UOp):
  buffers, reduce = count_srcs(ctx, x.op, (prev_src:=x.src_without_body))
  if x.op is Ops.STAGE and (x.arg is not None and (not x.arg.removable or sum(not b.is_variable for b in buffers) > 3 or reduce)):
    x = x.replace(arg=replace(x.arg, removable=False))
  if ctx.has_limit and x.op in GroupOp.Elementwise|{Ops.STACK} and len(buffers) >= buffer_limit(x.device) > 0:
    x = x.replace(src=tuple(split_graph(s) if s.op in GroupOp.Elementwise|{Ops.STACK} and not s.is_virtual else s for s in x.src))
    buffers, reduce = count_srcs(ctx, x.op, tuple(ss if ss is sp else ss.buf_uop for ss,sp in zip(x.src, prev_src)))
  ctx.info[x] = (buffers, reduce)
  return x

pm_count_buffers = PatternMatcher([(UPat(GroupOp.All, name="x"), count_buffers)])

def gate_ranges(ctx, x:UOp) -> None:
  if not any(r in ctx for r in x.ranges): raise BottomUpGate()

pm_gate_ranges = PatternMatcher([(UPat(GroupOp.All, name="x"), gate_ranges)])

def stage_to_store(st:UOp) -> UOp|None:
  if st.arg is None or st.arg.addrspace is not AddrSpace.GLOBAL: return None
  if st.arg.device is None: raise RuntimeError("global storage needs a device, give the STAGE one")
  alloc = UOp.alloc(st.shape, st.dtype, device=st.arg.device)
  return alloc.base.after(alloc.index(*(r:=st.src[1:])).store(st.src[0]).end(*r)).view_as(st.shape) if not st.src[0].is_invalid else alloc

def fix_store_hazard(ctx:int, ind:UOp, dst:UOp, src:UOp) -> UOp|None:
  # Fresh ALLOCs inserted via rangeify are never store hazards.  Global STAGES are always inlined or lowered, so we can skip them initially.
  if (alloc:=dst.base).op is Ops.ALLOC and alloc.arg.slot >= ctx: return None
  reads = []
  for r in src.toposort(gate=lambda x: not ((x.op is Ops.STAGE and x.arg is not None and x.arg.addrspace is AddrSpace.GLOBAL) or kernel_boundary(x))):
    if r.op is Ops.INDEX and r.buf_uop is dst.buf_uop: reads.append(r)
  return None if all(r.src[1].get_idx() is ind.src[1].get_idx() for r in reads) else ind.store(split_graph(src, dst.device))

pm_stage_to_store = PatternMatcher([
  (UPat(Ops.STAGE, name="st"), stage_to_store),
  (UPat.var("dst").index(name="ind", allow_any_len=True).store(UPat.var("src")), fix_store_hazard),
  (UPat.var("x").stage(name="st", allow_any_len=True).index(name="ind", allow_any_len=True),
  lambda ind, st, x: x.substitute(dict(zip(st.src[1:], ind.src[1:])), extra_pm=pm_gate_ranges) if st.arg.removable else ind),
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstk"),), name="msel"), lambda msel, mstk: mstk.src[msel.arg]),
  (UPat((Ops.MSELECT,Ops.MSTACK), name="m"), lambda m: m.replace(src=tuple(s.base for s in m.src)).view_as(m.shape))
])

@dataclass
class SplitCtx:
  rids: dict[tuple[AxisType, int], int]
  args: list[UOp] = field(default_factory=list)

def add_arg(ctx, x:UOp) -> UOp|None:
  if not call_arg(x): return None
  ctx.args.append(a:=UOp.variable("_device_num", 0, x.vmax, x.dtype) if x.op is Ops.RANGE else x)
  return a.param_like(slot=len(ctx.args)-1)

def add_range(ctx, r:UOp) -> UOp:
  bound = r.src[0] if r.src[0].op is Ops.CONST else graph_rewrite(r.src[0], pm_kernel, ctx=ctx, walk=True)
  return r.replace(src=(bound,), arg=(r.arg[0], ctx.rids[r.arg]))

pm_kernel = PatternMatcher([(UPat(GroupOp.All, name="x"), add_arg), (UPat(Ops.RANGE, name="r"), add_range)])

def split_kernel(s:UOp) -> UOp|None:
  # Codegen prefers that ranges are ordered and renumbered according to creation order.  Initial ID < 0 => a RANGE inserted by rangeify.
  rargs = [x.arg for x in s.toposort(gate=lambda x: not kernel_boundary(x)) if x.op is Ops.RANGE]
  rids = {arg:i for i,arg in enumerate(sorted(rargs, key=lambda arg: (arg[0], arg[1] < 0, abs(arg[1]))))}
  s = graph_rewrite(s, pm_kernel, ctx=(ctx:=SplitCtx(rids)), bottom_up=True, walk=True, name="kernel")
  return s.end(*sorted([r for r in s.ranges if r.axis_type is not AxisType.DEVICE], key=lambda r: r.arg)).sink(arg=KernelInfo()).call(*ctx.args)

pm_split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="s"), lambda s: split_kernel(s.src[0] if s.op is Ops.END else s)),
  (UPat(GroupOp.Movement|{Ops.UNSHARD}, src=(UPat.var("x"),), allow_any_len=True), lambda x: x),
])

def remove_invariant_axes(ind, st, x):
  if not (st.arg is None or st.arg.removable): return None
  varying = [r if r in x.ranges else UOp.const(0) for r in st.src[1:]]
  return x.stage(*varying, arg=st.arg).index(*[sr if sr.op is Ops.CONST else ir for ir,sr in zip(ind.src[1:], varying)])

pm_simplify_graph = PatternMatcher([
  (UPat.var("x").stage(name="st", allow_any_len=True).index(name="ind", allow_any_len=True), remove_invariant_axes),
  (UPat(Ops.RESHAPE, src=(UPat(GroupOp.Defines, name="x"), UPat(Ops.STACK, src=()))), lambda x: x),
  (UPat.var("x").index(), lambda x: x.index(UOp.const(0)) if kernel_boundary(x) else x),
  (invalid_pat.stage(allow_any_len=True).index(allow_any_len=True), lambda i: i),
  (UPat.var("x").store(UPat.var("x")), lambda x: UOp(Ops.NOOP).after(*a.src[1:]) if (a:=x.src[0]).op is Ops.AFTER else UOp(Ops.NOOP)),
  (UPat(Ops.REDUCE, name="r"), lambda r: r.replace(src=(r.src[0], *[s for s in r.src[1:] if s.op is not Ops.CONST]))),
])

@rewrite_group(new_ctx=False)
def get_kernel_graph(tsink:UOp) -> UOp:
  tsink = graph_rewrite(tsink, pm_insert_stages, ctx=count_uses(tsink), bottom_up=True, name="insert stages")
  tsink = graph_rewrite(tsink, pm_add_ranges+pm_convert_index_to_where, walk=True, name="add ranges")
  tsink = graph_rewrite(tsink, pm_convert_ranges+pm_push_ranges, bottom_up=True, name="push ranges")
  tsink = graph_rewrite(tsink, pm_simplify_graph+pm_gated_index+symbolic+pm_reduce_simplify, name="simplify graph")

  ctx = BufferCtx(count_vars=tsink.device == 'WEBGPU', has_limit=bool(buffer_limit(tsink.device)))
  tsink = graph_rewrite(tsink, pm_count_buffers, ctx=ctx, walk=True, name="count buffers")
  tsink = graph_rewrite(tsink, pm_stage_to_store+pm_push_ranges, ctx=next(UOp.unique_num), bottom_up=True, name="stage to store")
  tsink = graph_rewrite(tsink, pm_split_kernels, bottom_up=True, name="split kernels")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC: type_verify(tsink, spec_kernel_graph)
  return tsink
