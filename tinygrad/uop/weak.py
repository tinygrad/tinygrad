from dataclasses import replace
from tinygrad.dtype import dtypes, DType, AddrSpace, least_upper_dtype, strong_dtype, weak_dtype
from tinygrad.helpers import unwrap
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, GroupOp, dtype_from_uop, promo_dtype

def default_dtype(u:UOp):
  if u.dtype is dtypes.weakfloat: return dtypes.default_float
  return dtypes.long if u.overflows(dtypes.int32) else dtypes.int

def commit_weak(s:UOp, dt:DType) -> UOp:
  # a CONST re-mints, never takes a cast: at bool/weakint/weakfloat a CAST would be a second spelling of one literal
  return UOp.const(s.val, dt) if s.op is Ops.CONST else s.cast(dt)

# s may stay bare only where u derives a width from src anyway. both checks bind: a comparison's dtype is bool, a shift's is its lhs's
def bare_ok(u:UOp, s:UOp, src:tuple[UOp, ...]) -> bool:
  return s.op is Ops.CONST and u.op in GroupOp.Broadcastable and promo_dtype(src) not in dtypes.weaks \
         and dtype_from_uop(u.op, src, u.arg) not in dtypes.weaks

# absorb the weak CAST off each src (the consumer states that width instead), THEN give the still-bare weak CONSTs their own
# default. bare_ok is asked at the ABSORBED srcs -- the widths this node ends up meeting at, not the ones it came in with
def take_widths(u:UOp) -> tuple[UOp, ...]:
  src = tuple(s.src[0] if s.op is Ops.CAST and s.dtype in dtypes.weaks else s for s in u.src)
  return tuple(commit_weak(s, default_dtype(s)) if s.op is Ops.CONST and s.dtype in dtypes.weaks and not bare_ok(u, s, src) else s
               for s in src)

def commit_srcs_at(u:UOp, dt:DType) -> UOp|None:
  # the root re-derives: a shift's dtype is its lhs's, so committing the lhs commits the node too
  ret = u.replace(dtype=None, src=tuple(commit_weak(s, dt) if s.dtype in dtypes.weaks and not bare_ok(u, s, u.src) else s for s in u.src))
  return None if ret is u else ret

def commit_weak_srcs(u:UOp) -> UOp|None:
  if not any(s.dtype in dtypes.weaks for s in u.src) or (dt:=least_upper_dtype(*(s.dtype for s in u.src))) in dtypes.weaks: return None
  return commit_srcs_at(u, dt)

# a concrete CAST over a weak node states the width the value will live at. that width is a floor, never a narrowing
def cast_weak_srcs(c:UOp, u:UOp) -> UOp|None:
  # only within the kind: an int cast of a weakfloat node is a value conversion, not a statement about the node's width
  if c.dtype in dtypes.weaks or weak_dtype(c.dtype) is not u.dtype: return None
  return None if (ret:=commit_srcs_at(u, least_upper_dtype(c.dtype, default_dtype(u)))) is None else ret.cast(c.dtype)

# rides every round that can mint a weak const, and must reach fixpoint before pm_lower_weak below hands one its own default
pm_commit_weak = PatternMatcher([
  (UPat(GroupOp.Broadcastable, name="u"), commit_weak_srcs),
  (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dtypes.weaks)), allow_any_len=True, name="u"),
   lambda u: u.replace(src=(u.src[0], commit_weak(u.src[1], u.src[0].dtype), *u.src[2:]))),
  # NOTE: no CONST arm. a concrete CAST over a weak CONST is already the committed pair, minted that way by UOp.const
  (UPat(Ops.CAST, name="c", src=(UPat(GroupOp.ALU, dtype=dtypes.weaks, name="u"),)), cast_weak_srcs),
])

# every consumer edge. a CONST gets no rule of its own: CONST(v) -> CAST(dt, CONST(v)) contains its own input, which deadlocks
def absorb_weak_srcs(u:UOp) -> UOp|None:
  if u.op is Ops.CAST and u.src[0].op is Ops.CONST: return None  # a literal, not a consumer: its CONST is the value half
  return None if (src:=take_widths(u)) == u.src else u.replace(dtype=None, src=src)

# lowering is absorbing plus one step: a node whose own dtype is a width resolves it too, keeping the old weak dtype as a CAST
# NOTE: the round running this must not compose symbolic -- its cast collapse eats that CAST and the round cycles
def lower_weak_node(u:UOp) -> UOp|None:
  start = 1 if u.op is Ops.WHERE else 0  # WHERE's cond is bool, never part of the width unification
  src = take_widths(u)
  # a weak CONST take_widths kept bare is settled -- this node derives its width; any OTHER weak src is not ours to lower yet
  if src == u.src or any(s.dtype in dtypes.weaks and s.op is not Ops.CONST for s in src[start:]): return absorb_weak_srcs(u)
  # a Binary widens from its own bounds as well as the lowered srcs, every other op derives from the lowered srcs alone
  dt = strong_dtype(least_upper_dtype(default_dtype(u), *(s.dtype for s in src)) if u.op in GroupOp.Binary
                    else unwrap(dtype_from_uop(u.op, src, u.arg)))
  return u.replace(dtype=None, src=src[:start]+tuple(s if s.base.is_invalid or s.dtype in dtypes.weaks else commit_weak(s, dt)
                                                     for s in src[start:])).cast(u.dtype)

pm_lower_weak = PatternMatcher([
  # two stacked weak casts are two kind conversions: each resolves at its own kind's default
  (UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat.var("x"),)),), name="u"),
   lambda u,x: x.cast(default_dtype(u.src[0])).cast(default_dtype(u)).cast(u.dtype) if x.dtype not in dtypes.weaks else None),
  # a weakfloat Unary (sin/exp2/...) must resolve here, before the transcendental decomposition
  (UPat(GroupOp.Binary|GroupOp.Unary|{Ops.WHERE, Ops.RANGE, Ops.STACK, Ops.SPECIAL}, name="u"), lower_weak_node),
  (UPat((Ops.PARAM, Ops.BUFFER), dtype=dtypes.weakint, name="u"),
    lambda u: u.replace(dtype=None, arg=replace(u.arg, dtype=default_dtype(u))).cast(dtypes.weakint) if u.addrspace == AddrSpace.ALU else None),
  (UPat(GroupOp.All, name="u"), absorb_weak_srcs),
])

# drop a CAST over a weak const where the consumer restores the width anyway, so bare-CONST rules keep matching. two
# statements must survive the drop: the width the operands MEET at, and the node's own DERIVED dtype
def uncast_const(u:UOp) -> UOp|None:
  src = tuple(s.src[0] if s.op is Ops.CAST and s.dtype not in dtypes.weaks and s.src[0].op is Ops.CONST
              and s.src[0].dtype in dtypes.weaks else s for s in u.src)
  if src == u.src or promo_dtype(src) != promo_dtype(u.src) or dtype_from_uop(u.op, src, u.arg) is not u.dtype: return None
  return u.replace(src=src)

# the inverse of pm_cast_const: drop a width statement the consumer re-derives, so bare-keyed rules keep matching.
# composes into symbolic and never symbolic_simple: the widths it drops are first stated by the round above it
pm_uncast_const = PatternMatcher([(UPat(GroupOp.Broadcastable, name="u"), uncast_const)])

def cast_const(u:UOp, s:UOp) -> UOp:
  if s.op is not Ops.CONST or s.is_invalid: return s
  # a bool const already has a strong dtype, so its CAST is built raw: .cast(bool) would fold at construction
  if s.dtype not in dtypes.weaks: return UOp(Ops.CAST, src=(s,), arg=s.dtype)
  # the width its consumer derives; where nothing does, commit_weak is the identity and spec_program rejects it LOUDLY
  return commit_weak(s, promo_dtype(u.src)) if u.op in GroupOp.Broadcastable else s

# THE DOOR: wrap every remaining bare const in its CAST. keyed on the CONSUMER -- "bare" is a property of the edge
def cast_consts(u:UOp) -> UOp|None:
  if u.op is Ops.CAST and u.src[0].op is Ops.CONST: return None
  return None if (src:=tuple(cast_const(u, s) for s in u.src)) == u.src else u.replace(src=src)

pm_cast_const = PatternMatcher([(UPat(GroupOp.All, name="u"), cast_consts)])
