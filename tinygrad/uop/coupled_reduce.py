from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, auto
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, Ops, UOp, UPat, PatternMatcher, graph_rewrite

class CoupledReduceRejectReason(Enum):
  def __repr__(self): return str(self)
  INVALID_PLAN = auto(); MISSING_REDUCE_RANGE = auto(); UNSUPPORTED_RANGE_KIND = auto() # noqa: E702
  REDUCE_RANGE_IN_INIT = auto(); REDUCE_RANGE_IN_FINAL = auto(); RANGE_MISMATCH = auto(); DTYPE_MISMATCH = auto() # noqa: E702

@dataclass(frozen=True)
class CoupledReduceField:
  name: str|int
  dtype: DType
  init: UOp
  state: UOp     # the accumulator placeholder this field's update writes back to
  update: UOp    # expression over `state` and other fields' `state`s for the per-element step
  # optional associative-merge form: required for GROUP_REDUCE parallelization of the coupled reduce.
  # `in_state` is the placeholder for the incoming partial; `merge` combines own `state` with own `in_state`
  # (and may reference other fields' state/in_state for cross-field couplings like online softmax's m→l,o).
  in_state: UOp|None = None
  merge: UOp|None = None

@dataclass(frozen=True)
class CoupledReducePlan:
  fields: tuple[CoupledReduceField, ...]
  reduce_ranges: tuple[UOp, ...]
  final: UOp

@dataclass(frozen=True)
class CoupledReduceDescriptor:
  target: UOp
  plan: CoupledReducePlan

@dataclass(frozen=True)
class CoupledReduceRejection:
  plan: CoupledReducePlan
  reason: CoupledReduceRejectReason
  detail: str
  evidence: tuple[UOp, ...] = tuple()

_reduce_range_kinds = {AxisType.REDUCE, AxisType.GROUP_REDUCE, AxisType.UNROLL}
_reduce_ops = {Ops.ADD, Ops.MUL, Ops.MAX}
COUPLED_REDUCE_TARGET_TAG = "coupled_reduce_target"

def check_coupled_reduce_plan(plan:object) -> CoupledReducePlan:
  if not isinstance(plan, CoupledReducePlan): raise TypeError("coupled_reduce descriptor plan must be CoupledReducePlan")
  if not isinstance(plan.fields, tuple) or any(not isinstance(f, CoupledReduceField) for f in plan.fields):
    raise TypeError("coupled_reduce plan fields must be a tuple of CoupledReduceField")
  if not isinstance(plan.reduce_ranges, tuple) or any(not isinstance(r, UOp) for r in plan.reduce_ranges):
    raise TypeError("coupled_reduce plan reduce_ranges must be a tuple of UOps")
  if not isinstance(plan.final, UOp): raise TypeError("coupled_reduce plan final must be UOp")
  for f in plan.fields:
    if not all(isinstance(x, UOp) for x in (f.init, f.state, f.update)):
      raise TypeError("coupled_reduce field init/state/update must be UOps")
    if (f.in_state is None) != (f.merge is None):
      raise TypeError("coupled_reduce field in_state and merge must be provided together")
    if f.in_state is not None and not (isinstance(f.in_state, UOp) and isinstance(f.merge, UOp)):
      raise TypeError("coupled_reduce field in_state and merge must be UOps")
  return plan

def check_coupled_reduce_descriptor(descriptor:object) -> CoupledReduceDescriptor:
  if not isinstance(descriptor, CoupledReduceDescriptor): raise TypeError("coupled_reduce entries must be CoupledReduceDescriptor")
  if not isinstance(descriptor.target, UOp): raise TypeError("coupled_reduce descriptor target must be UOp")
  check_coupled_reduce_plan(descriptor.plan)
  return descriptor

def _contains_all(root:UOp, needles:tuple[UOp, ...]) -> bool: return all(x in root.ranges for x in needles)
def _has_op(root:UOp, op:Ops) -> bool: return any(u.op is op for u in root.toposort())
def _define_vars(root:UOp) -> set[UOp]: return {x for x in root.toposort() if x.op is Ops.DEFINE_VAR}

def _foreign(root:UOp, allowed:tuple[UOp, ...], reduce_only:bool) -> tuple[UOp, ...]:
  # ranges in `root` not in `allowed` and not closed by an inner REDUCE; reduce_only filters to reduce-domain ranges.
  # Also treat ranges that only flow through INDEX (address calculation, not iteration) as closed — rangeify at
  # small N may stage an inner Q@K reduce to a buffer, leaving the inner reduce range only in the load's index
  # expression. The codegen evaluates the index per-load; the range isn't an iteration the descriptor must drive.
  topo = root.toposort()
  closed = {r for u in topo if u.op is Ops.REDUCE for r in u.src[1:] if r.op is Ops.RANGE}
  closed = closed | _ranges_only_under_index(root)
  return tuple(r for r in topo if r.op is Ops.RANGE and r not in allowed and r not in closed
               and (not reduce_only or r.arg[-1] in _reduce_range_kinds))

def _ranges_only_under_index(root:UOp) -> set[UOp]:
  # find RANGE nodes such that every use of them in `root` flows through an INDEX op before reaching the root value.
  # These are address-only iterators (codegen evaluates the address per access) that don't need a REDUCE to close.
  topo = root.toposort()
  uses: dict[UOp, list[UOp]] = {u: [] for u in topo}
  for u in topo:
    for s in u.src:
      if s in uses: uses[s].append(u)
  ranges = {r for r in topo if r.op is Ops.RANGE}
  def reachable_outside_index(node:UOp, seen:set[UOp]) -> bool:
    if node in seen: return False
    seen.add(node)
    if node is root: return True
    for parent in uses.get(node, ()):
      if parent.op is Ops.INDEX and node in parent.src[1:]: continue  # this use is index-only
      if reachable_outside_index(parent, seen): return True
    return False
  return {r for r in ranges if not reachable_outside_index(r, set())}

def validate_coupled_reduce_plan(plan:CoupledReducePlan, target:UOp|None=None) -> CoupledReduceRejection|None:
  plan = check_coupled_reduce_plan(plan)
  R = CoupledReduceRejectReason
  def rej(reason, detail, evidence=()): return CoupledReduceRejection(plan, reason, detail, evidence)
  states = tuple(f.state for f in plan.fields)
  in_states = tuple(f.in_state for f in plan.fields if f.in_state is not None)
  names = tuple(f.name for f in plan.fields)
  target_ranges: tuple[UOp, ...] = tuple(target.ranges) if target is not None else ()
  allowed_ranges = target_ranges + plan.reduce_ranges
  allowed_dvars = {x for r in allowed_ranges for x in _define_vars(r)}
  def undecl(root, declared): return tuple(x for x in _define_vars(root) if x not in declared and x not in allowed_dvars)

  if len(plan.fields) < 2: return rej(R.INVALID_PLAN, "coupled reduce requires at least two fields")
  if any(not isinstance(f.name, (str, int)) for f in plan.fields): return rej(R.INVALID_PLAN, "field names must be str or int")
  if len(set(names)) != len(names): return rej(R.INVALID_PLAN, "field names must be unique")
  if len(set(states)) != len(states): return rej(R.INVALID_PLAN, "field states must be unique", states)
  for f in plan.fields:
    if f.state.op is not Ops.DEFINE_VAR or f.state.src:
      return rej(R.INVALID_PLAN, "field states must be dedicated DEFINE_VAR placeholders", (f.state,))
  if not plan.reduce_ranges: return rej(R.MISSING_REDUCE_RANGE, "coupled reduce requires explicit reduce ranges")
  if any(r.op is not Ops.RANGE for r in plan.reduce_ranges):
    return rej(R.MISSING_REDUCE_RANGE, "reduce ranges must be concrete RANGE nodes", plan.reduce_ranges)
  if any(r.arg[-1] not in _reduce_range_kinds for r in plan.reduce_ranges):
    return rej(R.UNSUPPORTED_RANGE_KIND, "reduce ranges must be reduce-domain ranges", plan.reduce_ranges)
  if target is not None:
    if target.op is not Ops.REDUCE: return rej(R.INVALID_PLAN, "target must be REDUCE", (target,))
    if not (isinstance(target.arg, tuple) and len(target.arg) == 2 and target.arg[0] in _reduce_ops and isinstance(target.arg[1], tuple)):
      return rej(R.INVALID_PLAN, "target must have a valid REDUCE arg", (target,))
    # the descriptor's reduce_ranges describe the per-thread serial reduce; target.src[1:] may also carry
    # non-reduce-domain ranges (LOCAL/WARP/UPCAST) added by TC opts to encode WMMA-tile parallelism. Filter
    # to reduce-domain when comparing — the codegen handles the non-reduce ones independently of the plan.
    # After TC+expander, the target can also become a horizontal REDUCE (src[1:] empty, reduce ranges
    # encoded inside the value's UPCAST/UNROLL structure); accept that form when the plan's reduce ranges
    # are reachable in the target value's toposort or are degenerate (size-1, fully consumed by TC tiling).
    target_reduce_ranges = tuple(r for r in target.src[1:] if r.op is Ops.RANGE and r.arg[-1] in _reduce_range_kinds)
    if target.src[1:]:
      if plan.reduce_ranges != target_reduce_ranges:
        return rej(R.RANGE_MISMATCH, "descriptor ranges do not match target reduce", plan.reduce_ranges + target_reduce_ranges)
    else:
      value_topo = set(target.src[0].toposort())
      missing = tuple(r for r in plan.reduce_ranges if r not in value_topo
                      and not (r.src and r.src[0].op is Ops.CONST and r.src[0].arg == 1))
      if missing: return rej(R.RANGE_MISMATCH, "descriptor reduce ranges not present in horizontal target value", missing)
    if plan.final.dtype != target.dtype:
      return rej(R.DTYPE_MISMATCH, "final dtype must match target dtype", (plan.final, target))
    # degenerate (size-1) reduce ranges may not be reachable in the target value after TC tiling fully
    # consumes them — the descriptor's "iteration" collapsed to a single step. Skip the containment check
    # for those; the single-step lowering in lower_coupled_reduce_plan handles them without an outer loop.
    non_degen = tuple(r for r in plan.reduce_ranges if not (r.src and r.src[0].op is Ops.CONST and r.src[0].arg == 1))
    if non_degen and not _contains_all(target.src[0], non_degen):
      return rej(R.INVALID_PLAN, "target value must depend on all reduce ranges", (target.src[0],))
  if any(r in plan.final.toposort() for r in plan.reduce_ranges):
    return rej(R.REDUCE_RANGE_IN_FINAL, "final projection must not depend on reduce ranges", (plan.final,))
  if foreign := _foreign(plan.final, plan.reduce_ranges, reduce_only=True):
    return rej(R.RANGE_MISMATCH, "final projection uses foreign reduce ranges", foreign)
  if target is not None and (foreign := _foreign(plan.final, target_ranges, reduce_only=False)):
    return rej(R.RANGE_MISMATCH, "final projection uses foreign ranges", foreign)
  if undeclared := undecl(plan.final, states):
    return rej(R.INVALID_PLAN, "final references undeclared state placeholders", undeclared)
  if any(f.in_state is not None and (f.in_state.op is not Ops.DEFINE_VAR or f.in_state.src) for f in plan.fields):
    return rej(R.INVALID_PLAN, "field in_states must be dedicated DEFINE_VAR placeholders", in_states)
  if len(set(in_states)) != len(in_states) or set(in_states) & set(states):
    return rej(R.INVALID_PLAN, "field in_states must be unique and distinct from states", in_states)
  def check_foreign(expr, role, allowed_outer):
    # foreign reduce ranges + (with target) foreign general ranges in `expr`
    if foreign := _foreign(expr, plan.reduce_ranges, reduce_only=True):
      return rej(R.RANGE_MISMATCH, f"field {role} uses foreign reduce ranges", foreign)
    if target is not None and (foreign := _foreign(expr, allowed_outer, reduce_only=False)):
      return rej(R.RANGE_MISMATCH, f"field {role} uses foreign ranges", foreign)
    return None
  for f in plan.fields:
    if f.init.dtype != f.dtype or f.state.dtype != f.dtype or f.update.dtype != f.dtype:
      return rej(R.DTYPE_MISMATCH, "field dtype must match init, state, and update UOps", (f.init, f.state, f.update))
    if f.merge is not None:
      assert f.in_state is not None
      if f.in_state.dtype != f.dtype or f.merge.dtype != f.dtype:
        return rej(R.DTYPE_MISMATCH, "field dtype must match in_state and merge UOps", (f.in_state, f.merge))
      if undeclared := undecl(f.merge, states + in_states):
        return rej(R.INVALID_PLAN, "field merge references undeclared state/in_state placeholders", undeclared)
    if any(r in f.init.toposort() for r in plan.reduce_ranges):
      return rej(R.REDUCE_RANGE_IN_INIT, "field init must not depend on reduce ranges", (f.init,))
    if state_refs := tuple(x for x in _define_vars(f.init) if x in states):
      return rej(R.INVALID_PLAN, "field init must not reference coupled reduce state placeholders", state_refs)
    if (r := check_foreign(f.init, "init", target_ranges)) is not None: return r
    update_topo = f.update.toposort()
    if control_only := tuple(r for r in plan.reduce_ranges if r in update_topo and r not in f.update.ranges):
      return rej(R.INVALID_PLAN, "field update uses reduce ranges only through control dependencies", control_only)
    if (r := check_foreign(f.update, "update", allowed_ranges)) is not None: return r
    if undeclared := undecl(f.update, states):
      return rej(R.INVALID_PLAN, "field update references undeclared state placeholders", undeclared)
  return None

def _is_add_reduce(uop:UOp) -> bool:
  return uop.op is Ops.REDUCE and uop.dtype is dtypes.float32 and isinstance(uop.arg, tuple) and len(uop.arg) == 2 and \
    uop.arg[0] is Ops.ADD and isinstance(uop.arg[1], tuple)

def _range_extents_match(src:tuple[UOp, ...], dst:tuple[UOp, ...]) -> bool:
  return len(src) == len(dst) and all(s.op is Ops.RANGE and d.op is Ops.RANGE and s.src[0].key == d.src[0].key for s,d in zip(src, dst))

def _match_weighted_average(expr:UOp) -> tuple[UOp, UOp, UOp]|None:
  """Match `sum_j(w_j * v_j) * reciprocal(sum_j(w_j))` and return (numerator_reduce, value, denom_update).
  numerator_reduce: REDUCE_ADD over target reduce range carrying `w*v`.
  value: per-element value factor `v`. denom_update: per-element weight `w` rebased into numerator's range frame."""
  if expr.op is not Ops.MUL or expr.dtype is not dtypes.float32: return None
  for numerator, reciprocal in (expr.src, expr.src[::-1]):
    if not (_is_add_reduce(numerator) and reciprocal.op is Ops.RECIPROCAL
            and len(reciprocal.src) == 1 and _is_add_reduce(reciprocal.src[0])): continue
    denom = reciprocal.src[0]
    nv, dv, tr, dr = numerator.src[0], denom.src[0], numerator.src[1:], denom.src[1:]
    if nv.dtype is not dtypes.float32 or dv.dtype is not dtypes.float32: continue
    if _has_op(nv, Ops.CAST) or _has_op(dv, Ops.CAST) or nv.op is not Ops.MUL: continue
    if not _range_extents_match(dr, tr) or not _contains_all(dv, dr): continue
    denom_update = dv.substitute(dict(zip(dr, tr)), walk=True)
    if not _contains_all(denom_update, tr): continue
    if sum(1 for f in nv.split_uop(Ops.MUL) if f.key == denom_update.key) != 1: continue
    value = next((b for a, b in (nv.src, nv.src[::-1]) if a.key == denom_update.key), None)
    if value is not None: return numerator, value, denom_update
  return None

def _normalized_weighted_add_reduce_descriptor(expr:UOp) -> tuple[CoupledReduceDescriptor, dict[UOp, UOp]]|None:
  if (match := _match_weighted_average(expr)) is None: return None
  numerator, _value, denom_update = match
  tag = expr.key.hex()[:16]
  ws = UOp.variable(f"normalized_weighted_add_weighted_sum_{tag}", -float("inf"), float("inf"), dtype=dtypes.float32)
  Ws = UOp.variable(f"normalized_weighted_add_weight_sum_{tag}", -float("inf"), float("inf"), dtype=dtypes.float32)
  zero = UOp.const(dtypes.float32, 0.0)
  plan = CoupledReducePlan((CoupledReduceField("weighted_sum", dtypes.float32, zero, ws, ws + numerator.src[0]),
                            CoupledReduceField("weight_sum", dtypes.float32, zero, Ws, Ws + denom_update)),
                           numerator.src[1:], ws * Ws.reciprocal())
  descriptor = CoupledReduceDescriptor(numerator, plan)
  return (descriptor, {}) if validate_coupled_reduce_plan(descriptor.plan, descriptor.target) is None else None

_SOFTMAX_LOG2_E = math.log2(math.e)

def _is_log2e(u): return u.op is Ops.CONST and isinstance(u.arg, float) and math.isclose(u.arg, _SOFTMAX_LOG2_E)
def _is_neg_one(u): return u.op is Ops.CONST and isinstance(u.arg, float) and u.arg == -1.0
def _is_max_reduce(u):
  return u.op is Ops.REDUCE and u.dtype is dtypes.float32 and isinstance(u.arg, tuple) and len(u.arg) == 2 and u.arg[0] is Ops.MAX

def _stable_softmax_logits(weight:UOp, target_ranges:tuple[UOp, ...]) -> UOp|None:
  # detect weight == EXP2(MUL(ADD(s, MUL(REDUCE_MAX(s, ...), -1)), log2(e))) and return s
  if weight.op is not Ops.EXP2 or weight.dtype is not dtypes.float32: return None
  inner = weight.src[0]
  if inner.op is not Ops.MUL or len(inner.src) != 2: return None
  diff = next((a for a, b in (inner.src, inner.src[::-1]) if _is_log2e(b)), None)
  if diff is None or diff.op is not Ops.ADD or len(diff.src) != 2: return None
  for a, b in (diff.src, diff.src[::-1]):
    if b.op is not Ops.MUL or len(b.src) != 2: continue
    for x, y in (b.src, b.src[::-1]):
      if not _is_neg_one(y) or not _is_max_reduce(x) or not _range_extents_match(x.src[1:], target_ranges): continue
      if x.src[0].substitute(dict(zip(x.src[1:], target_ranges)), walk=True).key == a.key: return a
  return None

def _split_inner_reduce_add(root:UOp, amt:int) -> UOp:
  # find the first inner REDUCE_ADD inside `root` over a const-extent RANGE that is divisible by `amt`,
  # and split that range as `outer REDUCE * amt + inner UNROLL`. fix_reduce_unroll then horizontally CONTRACTs
  # the vec back to scalar at the REDUCE boundary, so the surrounding expression still sees a scalar reduce.
  if amt <= 1: return root
  for u in root.toposort():
    if u.op is not Ops.REDUCE or not isinstance(u.arg, tuple) or u.arg[0] is not Ops.ADD: continue
    for r in u.src[1:]:
      if r.op is not Ops.RANGE or r.src[0].op is not Ops.CONST: continue
      full = int(r.src[0].arg)
      if full <= amt or full % amt != 0: continue
      outer = UOp.range(full // amt, r.arg[0] + 50000, AxisType.REDUCE)
      inner = UOp.range(amt, r.arg[0] + 50001, AxisType.UNROLL)
      new_value = u.src[0].substitute({r: outer * amt + inner}, walk=True)
      other = tuple(rr for rr in u.src[1:] if rr is not r)
      return root.substitute({u: u.replace(src=(new_value, outer, inner) + other)}, walk=True)
  return root

def _online_softmax_three_acc_descriptor(expr:UOp) -> tuple[CoupledReduceDescriptor, dict[UOp, UOp]]|None:
  if (match := _match_weighted_average(expr)) is None: return None
  numerator, value, denom_update = match
  target_ranges = numerator.src[1:]
  # the descriptor target must be a clean reduce (no nested reduce) so it survives codegen rebinding
  if _has_op(value, Ops.REDUCE): return None
  if (logits := _stable_softmax_logits(denom_update, target_ranges)) is None: return None
  # vectorize the inner dot product inside logits (e.g. the D-dim of QK in attention): split the inner REDUCE_ADD
  # range as outer*amt + inner UNROLL. the descriptor still sees logits as a scalar (the REDUCE horizontally
  # contracts the UNROLL), but the codegen emits a vec-wide FMA loop instead of `amt` scalar FMAs.
  logits = _split_inner_reduce_add(logits, getenv("CR_UNROLL_QK", 0))
  # vectorize the weighted dimension (V's head dim) into the o accumulator: the unique LOOP range that `value`
  # carries but `logits`/`target_ranges` don't. CR_TILE_D splits it as outer LOOP * inner UPCAST to shrink the
  # o vec width and free threadgroup memory for larger L*G under GROUP_REDUCE; otherwise full-UPCAST it.
  logits_rs = {r for r in logits.toposort() if r.op is Ops.RANGE}
  free = [r for r in value.toposort() if r.op is Ops.RANGE and r not in logits_rs and r not in target_ranges]
  # CR_TC opts out of the value STACK so TC opt can apply UPCAST splits on the D-axis without GEP-out-of-range
  # against a fixed-width descriptor STACK (the WMMA path needs un-vectorized scalar value)
  vec_axis = (free[0] if len(free) == 1 and free[0].arg[-1] is AxisType.LOOP
              and free[0].src[0].op is Ops.CONST and not getenv("CR_TC", 0) else None)
  vec_subs: dict[UOp, UOp] = {}
  if vec_axis is None:
    width, vdt, value_v = 1, dtypes.float32, value
  else:
    full = int(vec_axis.src[0].arg)
    tile = getenv("CR_TILE_D", 0)
    width = tile if 0 < tile < full and full % tile == 0 else full
    vdt = dtypes.float32.vec(width)
    if width == full:
      base_offset = UOp.const(vec_axis.dtype, 0)
      vec_subs[vec_axis] = vec_axis.replace(arg=(vec_axis.arg[0], AxisType.UPCAST))
    else:
      outer = UOp.range(full // width, vec_axis.arg[0] + 9000, AxisType.LOOP)
      inner = UOp.range(width, vec_axis.arg[0] + 9001, AxisType.UPCAST)
      base_offset = outer * width
      vec_subs[vec_axis] = base_offset + inner
    value_v = UOp(Ops.STACK, vdt, tuple(value.substitute({vec_axis: base_offset + UOp.const(vec_axis.dtype, i)}, walk=True)
                                        for i in range(width)))
  def bcast(u): return u if width == 1 else u.broadcast(width)
  # CR_J_UPCAST=B_j: process B_j parallel j's per per-thread step. on the M3 profile the online-softmax inner
  # loop is compute-bound by the serial exp2 chain (m_new[j] → α → contrib → o-rescale → m_state[j+1]); split
  # the target reduce as `outer REDUCE × inner UPCAST(B_j)`, materialize logits/value as STACKs over B_j
  # substitutions, and update the plan so each step does 1 max-reduce + 1 α-exp2 + B_j parallel contrib-exp2s +
  # vec FMAs. the contrib exps become independent ops the compiler can issue concurrently, attacking the 15× GFLOPS
  # gap vs the 4-kernel baseline. only applies when target has exactly one constant-extent reduce range.
  bj = getenv("CR_J_UPCAST", 0)
  if bj > 1 and len(target_ranges) == 1 and target_ranges[0].src[0].op is Ops.CONST and \
     (j_full := int(target_ranges[0].src[0].arg)) > bj and j_full % bj == 0:
    j = target_ranges[0]
    j_outer = UOp.range(j_full // bj, j.arg[0] + 8000, AxisType.REDUCE)
    j_inner = UOp.range(bj, j.arg[0] + 8001, AxisType.UPCAST)
    vec_subs[j] = j_outer * bj + j_inner
    def j_sub(u, i): return u.substitute({j: j_outer * bj + UOp.const(j.dtype, i)}, walk=True)
    logits_b = [j_sub(logits, i) for i in range(bj)]
    # build value as a single contiguous STACK of (B_j * B_d) scalar loads, d-major within each j (j-block
    # rows are non-contiguous; d-runs within a j-row ARE contiguous in V[h, j, d:d+B_d]). Then GEP-slice
    # per-j to recover B_j vec(B_d) values. load_store_folding sees the d-runs and folds them to vec loads
    # instead of the j-interleaved scalar grid the renderer would otherwise produce.
    if width > 1 and vec_axis is not None:
      flat_dt = dtypes.float32.vec(bj * width)
      flat = UOp(Ops.STACK, flat_dt, tuple(
        value.substitute({vec_axis: base_offset + UOp.const(vec_axis.dtype, i_d),
                          j: j_outer * bj + UOp.const(j.dtype, i_j)}, walk=True)
        for i_j in range(bj) for i_d in range(width)))
      value_b = [flat.gep(tuple(range(i_j * width, (i_j + 1) * width))) for i_j in range(bj)]
    else:
      value_b = [j_sub(value_v, i) for i in range(bj)]
    target_ranges_post: tuple[UOp, ...] = (j_outer,)
  else:
    logits_b, value_b, target_ranges_post = [logits], [value_v], target_ranges
  tag = expr.key.hex()[:16]
  log2_e = UOp.const(dtypes.float32, _SOFTMAX_LOG2_E)
  def var(name, dt): return UOp.variable(f"online_softmax_{name}_{tag}", -float("inf"), float("inf"), dtype=dt)
  m_state, l_state, o_state = var("max", dtypes.float32), var("denom", dtypes.float32), var("weighted", vdt)
  m_in, l_in, o_in = var("max_in", dtypes.float32), var("denom_in", dtypes.float32), var("weighted_in", vdt)
  def exp2_diff(a, b): return ((a - b).alu(Ops.MUL, log2_e)).alu(Ops.EXP2)
  # per-step update: combine running state with B_j parallel (logits, value) elements. for B_j=1 this is the
  # standard online softmax step; for B_j>1 the contrib exps and o-rescale FMAs across b are independent.
  from functools import reduce as _reduce
  row_max = _reduce(lambda a, b: a.alu(Ops.MAX, b), logits_b)
  m_new = m_state.alu(Ops.MAX, row_max)
  alpha = exp2_diff(m_state, m_new)
  contribs = [exp2_diff(lb, m_new) for lb in logits_b]
  l_update_step = l_state * alpha + _reduce(lambda a, b: a + b, contribs)
  o_update_step = o_state * bcast(alpha) + _reduce(lambda a, b: a + b, (vb * bcast(cb) for vb, cb in zip(value_b, contribs)))
  # associative merge: combine two partial (m,l,o) states — needed for GROUP_REDUCE parallelization of the key axis
  m_merged = m_state.alu(Ops.MAX, m_in)
  a_own, a_in = exp2_diff(m_state, m_merged), exp2_diff(m_in, m_merged)
  plan = CoupledReducePlan((
    CoupledReduceField("softmax_max", dtypes.float32, UOp.const(dtypes.float32, -float("inf")), m_state, m_new,
                       in_state=m_in, merge=m_merged),
    CoupledReduceField("softmax_denom", dtypes.float32, UOp.const(dtypes.float32, 0.0), l_state, l_update_step,
                       in_state=l_in, merge=l_state * a_own + l_in * a_in),
    CoupledReduceField("softmax_weighted", vdt, bcast(UOp.const(dtypes.float32, 0.0)), o_state, o_update_step,
                       in_state=o_in, merge=o_state * bcast(a_own) + o_in * bcast(a_in)),
  ), target_ranges_post, o_state * bcast(l_state.reciprocal()))
  # the target body is a discarded placeholder (the plan supplies the real computation): it must reference every
  # range the plan updates touch (logits carries the query-side ranges) and resist symbolic constant-hoisting,
  # so it is combined with MAX rather than MUL.
  target_body = _reduce(lambda a, b: a.alu(Ops.MAX, b), (bcast(lb).alu(Ops.MAX, vb) for lb, vb in zip(logits_b, value_b)))
  target = UOp(Ops.REDUCE, vdt, (target_body,) + target_ranges_post, numerator.arg)
  descriptor = CoupledReduceDescriptor(target, plan)
  if validate_coupled_reduce_plan(descriptor.plan, descriptor.target) is not None: return None
  return descriptor, vec_subs

def _unwrap_stage_index(u:UOp) -> UOp|None:
  # INDEX(STAGE(value, *def_ranges), *use_indices) inlines `value` with def_ranges substituted by use_indices,
  # as long as the use indices aren't themselves the def ranges (which would be a no-op).
  if u.src[0].op is not Ops.STAGE or len(u.src[0].src) != len(u.src): return None
  if not all(d.op is Ops.RANGE for d in u.src[0].src[1:]): return None
  stage_value, def_ranges, use_indices = u.src[0].src[0], u.src[0].src[1:], u.src[1:]
  if any(d.key == ui.key for d, ui in zip(def_ranges, use_indices)): return None
  return stage_value.substitute(dict(zip(def_ranges, use_indices)), walk=True)

def _factor_range_invariant_factors(reduce:UOp) -> UOp|None:
  if not (isinstance(reduce.arg, tuple) and len(reduce.arg) == 2 and reduce.arg[0] is Ops.ADD): return None
  ranges = reduce.src[1:]
  value = reduce.src[0]
  if not ranges or value.op is not Ops.MUL: return None
  range_set = set(ranges)
  invariant: list[UOp] = []
  variant: list[UOp] = []
  for f in value.split_uop(Ops.MUL):
    (variant if any(x in range_set for x in f.toposort()) else invariant).append(f)
  if not invariant or not variant: return None
  from functools import reduce as _r
  return _r(lambda a,b: a*b, invariant) * reduce.replace(src=(_r(lambda a,b: a*b, variant),) + ranges)

# rangeify primitives applied to the pre-detector normalization passes (instead of hand-rolled toposort + cache)
_pm_unwrap_stage_index = PatternMatcher([(UPat(Ops.INDEX, name="u"), _unwrap_stage_index)])
_pm_factor_range_invariant = PatternMatcher([(UPat(Ops.REDUCE, name="reduce"), _factor_range_invariant_factors)])

def rewrite_normalized_weighted_add_reduces(root:UOp) -> tuple[UOp, tuple[CoupledReduceDescriptor, ...]]:
  if _has_op(root, Ops.CAST): return root, ()
  # inline staged intermediates and hoist range-invariant factors so the algebraic weighted-average shape is visible
  transformed = graph_rewrite(graph_rewrite(root, _pm_unwrap_stage_index), _pm_factor_range_invariant)
  candidates: list[tuple[UOp, CoupledReduceDescriptor, dict[UOp, UOp]]] = []
  for u in transformed.toposort():
    for detect in (_online_softmax_three_acc_descriptor, _normalized_weighted_add_reduce_descriptor):
      if (r := detect(u)) is not None:
        candidates.append((u, *r))
        break
  if len(candidates) != 1: return root, ()
  normalized_expr, descriptor, extra_subs = candidates[0]
  # the 2-acc descriptor reuses the numerator reduce as its target — only safe if it has a single consumer.
  # the 3-acc descriptor uses a fresh target (absent from the graph), so the guard does not apply to it.
  refs = sum(1 for u in transformed.toposort() for s in u.src if s is descriptor.target)
  if refs > 1: return root, ()
  # tag the target so binding rebinds it exactly after later codegen and opt rewrites (which preserve tags)
  tagged = descriptor.target.replace(tag=(COUPLED_REDUCE_TARGET_TAG, normalized_expr.key.hex()[:16]))
  descriptor = CoupledReduceDescriptor(tagged, descriptor.plan)
  # extra substitutions from the detector (e.g. d-axis split into outer LOOP * inner UPCAST for o-tile vectorization)
  return transformed.substitute({normalized_expr: tagged, **extra_subs}, walk=True), (descriptor,)

def cooperative_tile_invariant_loads(root:UOp, cooperate_axis:UOp, tile_axis:UOp, loop_id_base:int=999000) -> UOp:
  """Wrap PARAM-indexed expressions inside `root` whose index depends on `tile_axis` but is invariant in
  `cooperate_axis` with a LOCAL bufferize hoisted to per-block scope. Each thread (cooperate_axis slot)
  stores its K[j_block,:] row once per block; reads pull via tile_axis (same extent as cooperate_axis).
  Inner-REDUCE ranges are substituted with fresh LOOP-typed copies in the bufferize value so its defining
  ranges (cooperate_axis + LOOP copies) live at per-block scope; .index uses the original REDUCE ranges."""
  from tinygrad.schedule.indexing import BufferizeOpts
  from tinygrad.dtype import AddrSpace
  if any(a.op is not Ops.RANGE for a in (cooperate_axis, tile_axis)) or cooperate_axis is tile_axis: return root
  if cooperate_axis.src[0].key != tile_axis.src[0].key: return root
  # find PARAM-INDEX nodes inside an inner REDUCE whose body is invariant in cooperate_axis and depends on tile_axis
  candidates: list[tuple[UOp, tuple[UOp, ...]]] = []
  for u in root.toposort():
    if u.op is not Ops.REDUCE or not u.src or tile_axis in u.src[1:]: continue
    inner = tuple(r for r in u.src[1:] if r.op is Ops.RANGE)
    for idx in u.src[0].toposort():
      if idx.op is not Ops.INDEX: continue
      base = idx.src[0]
      while base.op is Ops.INDEX and base.src: base = base.src[0]
      idx_rs = {r for r in idx.toposort() if r.op is Ops.RANGE}
      if base.op is Ops.PARAM and cooperate_axis not in idx_rs and tile_axis in idx_rs:
        candidates.append((idx, inner))
  subs, lid = {}, loop_id_base
  for idx, inner in candidates:
    loops = tuple(r.replace(arg=(lid + i, AxisType.LOOP)) for i, r in enumerate(inner))
    lid += len(inner)
    upstream = tuple(r for r in idx.toposort() if r.op is Ops.RANGE
                     and r.arg[-1] in (AxisType.LOCAL, AxisType.GROUP_REDUCE) and r not in (cooperate_axis, tile_axis))
    sv = idx.substitute({tile_axis: cooperate_axis, **dict(zip(inner, loops))}, walk=True)
    subs[idx] = sv.bufferize(*upstream, cooperate_axis, *loops,
                             arg=BufferizeOpts(device=None, addrspace=AddrSpace.LOCAL)).index(*upstream, tile_axis, *inner)
  return root.substitute(subs, walk=True) if subs else root


