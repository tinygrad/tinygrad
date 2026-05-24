from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import flatten
from tinygrad.uop.coupled_reduce import CoupledReduceDescriptor, CoupledReduceField, CoupledReducePlan, check_coupled_reduce_descriptor, \
  CoupledReduceRejectReason as CoupledReduceRejectReason, CoupledReduceRejection, validate_coupled_reduce_plan, COUPLED_REDUCE_TARGET_TAG, \
  cooperative_tile_invariant_loads, _reduce_range_kinds
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite
from tinygrad.schedule.indexing import BufferizeOpts

@dataclass(frozen=True)
class CoupledReduceLowered:
  plan: CoupledReducePlan
  accumulators: tuple[UOp, ...]
  accumulator_slots: int
  input_ranges: tuple[UOp, ...]
  old_fields: tuple[UOp, ...]
  final_fields: tuple[UOp, ...]
  init_group: UOp
  update_group: UOp
  end: UOp
  final: UOp

def _expand_reduce_range(rng:UOp, rewrite:Callable[[UOp], UOp]) -> tuple[UOp, ...]:
  # an opt may split a reduce range (e.g. GROUPTOP turns r into new_rng*old_sz + replaced_rng); the descriptor's
  # reduce_ranges must remain a tuple of RANGE nodes, so extract them from the rewritten expression.
  # A TC opt may also split the descriptor's matmul-N axis into REDUCE outer + LOCAL/WARP/UPCAST WMMA tile
  # parts; only the reduce-domain parts belong in the descriptor (the WMMA-tile parts are TC-managed,
  # invisible to the per-thread serial reduce). Filter to reduce-domain ranges, fall back to all if none.
  rr = rewrite(rng)
  if rr.op is Ops.RANGE: return (rr,)
  all_ranges = tuple(n for n in rr.toposort() if n.op is Ops.RANGE)
  reduce_domain = tuple(n for n in all_ranges if n.arg[-1] in _reduce_range_kinds)
  return reduce_domain or all_ranges

def rewrite_coupled_reduce_descriptors(descriptors:tuple[CoupledReduceDescriptor, ...],
                                       rewrite:Callable[[UOp], UOp]) -> tuple[CoupledReduceDescriptor, ...]:
  if not isinstance(descriptors, tuple): raise TypeError("coupled_reduce descriptors must be a tuple")
  ret: list[CoupledReduceDescriptor] = []
  for d in descriptors:
    check_coupled_reduce_descriptor(d)
    new_rr = tuple(rr for r in d.plan.reduce_ranges for rr in _expand_reduce_range(r, rewrite))
    # if an opt split a reduce range, target.src[1:] now holds a non-RANGE expression; rebuild from new_rr
    new_target = rewrite(d.target)
    # TC's vectorization (do_expand) may wrap the original REDUCE target in UNROLL/CONTRACT layers. The
    # inner REDUCE keeps the original COUPLED_REDUCE_TARGET_TAG; unwrap so the descriptor target is a
    # REDUCE again. Only follow single-src wrappers so we don't pick up a sibling REDUCE in the body.
    while new_target.op is not Ops.REDUCE and len(new_target.src) == 1 and new_target.src[0].op is Ops.REDUCE:
      new_target = new_target.src[0]
    if new_target.op is Ops.REDUCE and new_target.src[1:] != new_rr:
      new_target = new_target.replace(src=(new_target.src[0],) + new_rr)
    ret.append(CoupledReduceDescriptor(new_target, replace(d.plan,
      fields=tuple(replace(f, init=rewrite(f.init), update=rewrite(f.update),
                           merge=rewrite(f.merge) if f.merge is not None else None) for f in d.plan.fields),
      reduce_ranges=new_rr, final=rewrite(d.plan.final))))
  return tuple(ret)

def _range_rebind_key(uop:UOp) -> tuple[tuple[UOp, ...], tuple]|None:
  return (uop.src, uop.arg[:-1]) if uop.op is Ops.RANGE else None

def _is_target_tag(tag:object) -> bool:
  return isinstance(tag, tuple) and len(tag) == 2 and tag[0] == COUPLED_REDUCE_TARGET_TAG

def _input_ranges_for(topo, reduce_ranges:tuple[UOp, ...]) -> tuple[UOp, ...]:
  # a range that lives inside a nested REDUCE body (its own reduce ranges) or as a defining range of a
  # STAGE (cooperative bufferize) is local to that scope, not an accumulator input — including it pulls
  # init.after() into that inner scope and re-initializes per inner iteration.
  ended = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
  inner_scoped = flatten([[r for r in x.src[1:] if r.op is Ops.RANGE] for x in topo if x.op in (Ops.REDUCE, Ops.STAGE)])
  return tuple(x for x in topo if x.op is Ops.RANGE and x not in reduce_ranges and x not in ended and x not in inner_scoped)

def _required_input_ranges(update_roots:tuple[UOp, ...], reduce_ranges:tuple[UOp, ...]) -> tuple[UOp, ...]:
  return _input_ranges_for({u:None for root in update_roots for u in root.toposort()}, reduce_ranges)

@dataclass(frozen=True)
class _Phase:
  accs: tuple[UOp, ...]
  init_group: UOp
  update_group: UOp
  end: UOp
  old_fields: tuple[UOp, ...]
  final_fields: tuple[UOp, ...]
  input_ranges: tuple[UOp, ...]

def _is_degenerate_range(r:UOp) -> bool:
  return r.op is Ops.RANGE and bool(r.src) and r.src[0].op is Ops.CONST and r.src[0].arg == 1

def _lower_phase(plan:CoupledReducePlan, slot:int, reduce_ranges:tuple[UOp, ...],
                 step_expr:Callable[[CoupledReduceField], UOp], extra_subs:dict[UOp, UOp]|None=None) -> _Phase:
  step_exprs = tuple(step_expr(f) for f in plan.fields)
  # cooperative input tile (env-gated): if a LOCAL input range and a per-thread REDUCE range share extent,
  # hoist invariant K-side loads to a LOCAL bufferize at per-block scope. apply to the sinked step_exprs
  # so all fields share ONE bufferize (otherwise 3 fields each create their own redundant LOCAL store).
  if getenv("CR_COOP_KV", 0):
    pre_input_ranges = _required_input_ranges(step_exprs, reduce_ranges)
    local_ranges = [r for r in pre_input_ranges if r.arg[-1] is AxisType.LOCAL]
    coop_pairs = [(L, R) for L in local_ranges for R in reduce_ranges
                  if R.arg[-1] is AxisType.REDUCE and R is not L and L.src[0].key == R.src[0].key]
    if coop_pairs:
      coop, tile = max(coop_pairs, key=lambda p: p[1].arg[0])
      sinked = UOp.sink(*step_exprs)
      wrapped = cooperative_tile_invariant_loads(sinked, coop, tile)
      step_exprs = tuple(wrapped.src)
  input_ranges = _required_input_ranges(step_exprs, reduce_ranges)
  dtype_fields: dict[DType, list[CoupledReduceField]] = defaultdict(list)
  for field in plan.fields: dtype_fields[field.dtype].append(field)
  accs: dict[DType, UOp] = {dtype: UOp.placeholder((len(fields),), dtype, slot+i, AddrSpace.REG)
                            for i,(dtype, fields) in enumerate(dtype_fields.items())}
  offsets = {field:i for fields in dtype_fields.values() for i,field in enumerate(fields)}
  def field_acc(field:CoupledReduceField, acc:UOp|None=None) -> UOp:
    return (accs[field.dtype] if acc is None else acc).index(UOp.const(dtypes.weakint, offsets[field]))
  # degenerate (all reduce ranges have size 1) → emit single-step lowering with no loop. init runs once,
  # update reads init values directly (not via a loop-scoped acc load), end has no ranges to close. This
  # is the form produced when TC consumes the descriptor's reduce ranges into WMMA-tile parallelism.
  degenerate = bool(reduce_ranges) and all(_is_degenerate_range(r) for r in reduce_ranges)
  init_group = UOp.group(*tuple(field_acc(f, accs[f.dtype].after(*input_ranges)).store(f.init) for f in plan.fields))
  if degenerate:
    subs: dict[UOp, UOp] = {f.state: f.init for f in plan.fields}
    old_fields = tuple(field_acc(f, accs[f.dtype].after(init_group)) for f in plan.fields)
    end_ranges: tuple[UOp, ...] = ()
  else:
    old_fields = tuple(field_acc(f, accs[f.dtype].after(init_group, *reduce_ranges)) for f in plan.fields)
    subs = {f.state: old for f, old in zip(plan.fields, old_fields)}
    end_ranges = reduce_ranges
  if extra_subs: subs.update(extra_subs)
  updates = tuple(expr.substitute(subs, walk=True) for expr in step_exprs)
  update_group = UOp.group(*tuple(field_acc(f).store(update) for f, update in zip(plan.fields, updates)))
  end = update_group.end(*end_ranges).rtag("mergeable")
  final_fields = tuple(field_acc(f, accs[f.dtype].after(end)) for f in plan.fields)
  return _Phase(tuple(accs.values()), init_group, update_group, end, old_fields, final_fields, input_ranges)

def lower_coupled_reduce_plan(plan:CoupledReducePlan, slot:int=0, target:UOp|None=None) -> CoupledReduceLowered|CoupledReduceRejection:
  if (rejection:=validate_coupled_reduce_plan(plan, target)) is not None: return rejection
  # split the reduce ranges into the per-thread serial range(s) and any GROUP_REDUCE ranges that need a cross-thread merge
  gfr = tuple(r for r in plan.reduce_ranges if r.arg[-1] is AxisType.GROUP_REDUCE)
  serial = tuple(r for r in plan.reduce_ranges if r not in gfr)
  can_group = bool(gfr) and all(f.merge is not None and f.in_state is not None for f in plan.fields)
  inner = _lower_phase(plan, slot, serial if can_group else plan.reduce_ranges, lambda f: f.update)
  if can_group:
    # each thread's per-field partial state goes through a LOCAL buffer, then a cross-thread reduce over
    # relabeled GROUP_REDUCE ranges merges them with field.merge as the per-step combinator.
    upstream_locals = tuple(r for r in inner.final_fields[0].toposort() if r.op is Ops.RANGE and r.arg[-1] is AxisType.LOCAL)
    field_bufs = tuple(ff.bufferize(*upstream_locals, *gfr, arg=BufferizeOpts(gfr[0].arg[0]+200, AddrSpace.LOCAL))
                       for ff in inner.final_fields)
    outer_reduce = tuple(r.replace(arg=(r.arg[0]+100, AxisType.REDUCE)) for r in gfr)
    in_state_subs: dict[UOp, UOp] = {}
    for i, f in enumerate(plan.fields):
      assert f.in_state is not None
      in_state_subs[f.in_state] = field_bufs[i].index(*upstream_locals, *outer_reduce)
    def merge_expr(f:CoupledReduceField) -> UOp:
      assert f.merge is not None
      return f.merge
    outer = _lower_phase(plan, slot + len(inner.accs), outer_reduce, merge_expr, in_state_subs)
    accs, last = inner.accs + outer.accs, outer
  else:
    accs, last = inner.accs, inner
  final = plan.final.substitute({f.state:ff for f, ff in zip(plan.fields, last.final_fields)}, walk=True)
  return CoupledReduceLowered(plan, accs, len(accs), last.input_ranges, last.old_fields,
                              last.final_fields, last.init_group, last.update_group, last.end, final)

def bind_coupled_reduce_descriptors(sink:UOp, descriptors:tuple[CoupledReduceDescriptor, ...]) -> dict[UOp, CoupledReducePlan]:
  if not isinstance(descriptors, tuple): raise TypeError("coupled_reduce descriptors must be a tuple")
  if not descriptors: return {}
  topo = tuple(sink.toposort())
  current = set(topo)
  by_key: defaultdict[bytes, tuple[UOp, ...]] = defaultdict(tuple)
  by_range: defaultdict[tuple[tuple[UOp, ...], tuple], tuple[UOp, ...]] = defaultdict(tuple)
  by_tag: defaultdict[object, tuple[UOp, ...]] = defaultdict(tuple)
  for u in topo:
    by_key[u.key] += (u,)
    if (rk:=_range_rebind_key(u)) is not None: by_range[rk] += (u,)
    if _is_target_tag(u.tag): by_tag[u.tag] += (u,)
  cache: dict[UOp, UOp] = {}

  def resolve(uop, *, strict=False) -> UOp|None:
    # exact match first, then by structural key, then by range-rebind-key (same src + arg-prefix)
    if uop in current: return uop
    if len(m := by_key.get(uop.key, ())) == 1: return m[0]
    if (rk := _range_rebind_key(uop)) is not None and len(m := by_range.get(rk, ())) == 1: return m[0]
    if strict: raise AssertionError(f"{'missing' if not m else 'ambiguous'} coupled reduce descriptor target for {uop}")
    return None

  def rebind(uop, preserve=frozenset()) -> UOp:
    if uop in preserve: return uop
    if uop in cache: return cache[uop]
    if (r := resolve(uop)) is not None:
      cache[uop] = r
      return r
    new_src = tuple(rebind(s, preserve) for s in uop.src)
    out = uop.replace(src=new_src) if new_src != uop.src else uop
    if (r := resolve(out)) is not None: out = r
    cache[uop] = out
    return out

  def resolve_target(target:UOp) -> UOp:
    if _is_target_tag(target.tag):
      tagged = by_tag.get(target.tag, ())
      if len(tagged) == 1: return tagged[0]
      raise AssertionError(f"{'missing' if not tagged else 'ambiguous'} coupled reduce descriptor target for tag {target.tag}")
    if (r := resolve(target)) is not None: return r
    rebound = rebind(target)
    if (r := resolve(rebound)) is not None: return r
    from tinygrad.uop.symbolic import gep_pushing
    r = resolve(graph_rewrite(rebound, gep_pushing, name="coupled target gep_pushing"), strict=True)
    return r if r is not None else rebound

  ret: dict[UOp, CoupledReducePlan] = {}
  for descriptor in descriptors:
    descriptor = check_coupled_reduce_descriptor(descriptor)
    states = frozenset(f.state for f in descriptor.plan.fields)
    if states & current: raise AssertionError("coupled reduce state placeholders must not appear in the target sink")
    target = resolve_target(descriptor.target)
    assert target.op is Ops.REDUCE, f"coupled reduce descriptor target resolved to non-REDUCE {target.op}"
    # use the post-bind target's actual reduce ranges (filtered to reduce-domain) as the authoritative
    # source — they reflect any range splits introduced by opts (TC/UNROLL/GROUP) that may not have been
    # fully tracked through separate descriptor rewrites (e.g. _descriptors_through running the expander
    # in isolation). The descriptor's per-thread serial reduce only cares about reduce-domain ranges; the
    # TC-introduced LOCAL/WARP/UPCAST splits are managed by the codegen rather than the descriptor.
    target_ranges = tuple(r for r in target.src[1:] if r.op is Ops.RANGE and r.arg[-1] in _reduce_range_kinds)
    plan = replace(descriptor.plan,
      fields=tuple(replace(f, init=rebind(f.init, states), update=rebind(f.update, states)) for f in descriptor.plan.fields),
      reduce_ranges=target_ranges or tuple(rebind(r) for r in descriptor.plan.reduce_ranges),
      final=rebind(descriptor.plan.final, states))
    if (rejection := validate_coupled_reduce_plan(plan, target)) is not None:
      raise AssertionError(f"invalid coupled reduce descriptor: {rejection.reason.name}: {rejection.detail}")
    if target in ret: raise AssertionError(f"duplicate coupled reduce descriptor target for {target}")
    ret[target] = plan
  return ret
