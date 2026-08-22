from dataclasses import replace
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid, least_upper_dtype, strong_dtype, weak_dtype
from tinygrad.helpers import unwrap
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, GroupOp, dtype_from_uop, promo_dtype

def default_dtype(u:UOp):
  if u.dtype is dtypes.weakfloat: return dtypes.default_float
  return dtypes.long if u.overflows(dtypes.int32) else dtypes.int

def commit_weak(s:UOp, dt:DType) -> UOp:
  # a CONST re-mints, never takes a cast: at bool/weakint/weakfloat a CAST would be a second spelling of one literal
  if s.op is Ops.STACK: return s.replace(dtype=None, src=tuple(commit_weak(x, dt) for x in s.src))
  return UOp.const(s.val, dt) if s.op is Ops.CONST else s.cast(dt)

# the decomps and non-native-float emulation ask this with the width a sibling src states: a bare literal commits there
def commit_weak_sibling(u:UOp, dt:DType|None) -> UOp|None:
  return None if dt is None else u.replace(src=tuple(commit_weak(s, dt) if s.op is Ops.CONST and s.dtype in dtypes.weaks else s for s in u.src))

# the concrete widths src state: where the operands meet and what u derives (different for comparisons and shifts)
def derived_widths(u:UOp, src:tuple[UOp, ...]) -> tuple[DType, DType]|None:
  if u.op not in GroupOp.Broadcastable or (meet:=promo_dtype(src)) in dtypes.weaks \
     or (result:=unwrap(dtype_from_uop(u.op, src, u.arg))) in dtypes.weaks: return None
  return meet, result

def commit_srcs_at(u:UOp, dt:DType) -> UOp|None:
  # the root re-derives: a shift's dtype is its lhs's, so committing the lhs commits the node too
  widths = derived_widths(u, u.src)
  ret = u.replace(dtype=None, src=tuple(commit_weak(s, dt) if s.dtype in dtypes.weaks and
                                        not (s.op is Ops.CONST and widths is not None) else s for s in u.src))
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
  (UPat(Ops.LOAD, src=(UPat(), UPat(dtype=dtypes.weaks)), allow_any_len=True, name="u"),
   lambda u: u.replace(src=(u.src[0], commit_weak(u.src[1], u.dtype), *u.src[2:]))),
  # NOTE: no CONST arm. a concrete CAST over a weak CONST is already the committed pair, minted that way by UOp.const
  (UPat(Ops.CAST, name="c", src=(UPat(GroupOp.ALU, dtype=dtypes.weaks, name="u"),)), cast_weak_srcs),
])

# Every consumer edge absorbs the weak CAST off its srcs (the consumer states that width) and defaults literals whose
# consumer does not derive a width; width-producing ops also settle their compute floor, leaving a trailing weak CAST.
# A weakfloat Unary (sin/exp2/...) must resolve here, before the transcendental decomposition.
# This round cannot compose symbolic: its cast collapse eats that CAST and cycles.
_lower_weak_ops = GroupOp.Binary|GroupOp.Unary|{Ops.WHERE, Ops.RANGE, Ops.STACK, Ops.SPECIAL}
def lower_weak_node(u:UOp) -> UOp|None:
  if u.op is Ops.CAST and u.src[0].op is Ops.CONST: return None  # a literal, not a consumer: its CONST is the value half
  src = tuple(s.src[0] if s.op is Ops.CAST and s.dtype in dtypes.weaks else s for s in u.src)
  widths = derived_widths(u, src)
  src = tuple(commit_weak(s, default_dtype(s)) if s.op is Ops.CONST and s.dtype in dtypes.weaks and widths is None else s
              for s in src)
  start = 1 if u.op is Ops.WHERE else 0  # WHERE's cond is bool, never part of the width unification
  # resolve whole once every weak expression lowered (derivable literals inherit this node's width and may wait):
  # a Binary widens from its own bounds as well as the lowered srcs, every other op derives from the lowered srcs alone
  if u.op in _lower_weak_ops and src != u.src and not any(s.dtype in dtypes.weaks and s.op is not Ops.CONST for s in src[start:]):
    dt = strong_dtype(least_upper_dtype(default_dtype(u), *(s.dtype for s in src)) if u.op in GroupOp.Binary
                      else unwrap(dtype_from_uop(u.op, src, u.arg)))
    return u.replace(dtype=None, src=src[:start]+tuple(s if s.base.is_invalid or s.dtype in dtypes.weaks else commit_weak(s, dt)
                                                       for s in src[start:])).cast(u.dtype)
  return None if src == u.src else u.replace(dtype=None, src=src)

pm_lower_weak = PatternMatcher([
  # a gated long index into a small buffer narrows; its out-of-gate value is discarded
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat.var("buf"), UPat.var("gate").where(UPat.var("idx", dtypes.long), UPat(Ops.CONST, arg=Invalid))),
        allow_any_len=True, name="u"),
   lambda u,buf,gate,idx: u.replace(src=(buf, idx.cast(dtypes.int).valid(gate))+u.src[2:]) if buf.max_numel()-1 <= dtypes.int32.max else None),
  # two stacked weak casts are two kind conversions: each resolves at its own kind's default
  (UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat.var("x"),)),), name="u"),
   lambda u,x: x.cast(default_dtype(u.src[0])).cast(default_dtype(u)).cast(u.dtype) if x.dtype not in dtypes.weaks else None),
  (UPat((Ops.PARAM, Ops.BUFFER), dtype=dtypes.weakint, name="u"),
    lambda u: u.replace(dtype=None, arg=replace(u.arg, dtype=default_dtype(u))).cast(dtypes.weakint) if u.addrspace == AddrSpace.ALU else None),
  (UPat(GroupOp.All, name="u"), lower_weak_node),
])

def cast_const(u:UOp, s:UOp) -> UOp:
  if s.op is not Ops.CONST or s.is_invalid: return s  # Invalid carries no width: the door never wraps it
  # bool is the one strong bare dtype, and its CAST is built raw: .cast(bool) would fold at construction
  if s.dtype is dtypes.bool: return UOp(Ops.CAST, src=(s,), arg=s.dtype)
  # the width its consumer derives; where nothing does, commit_weak is the identity and spec_program rejects it LOUDLY
  return commit_weak(s, widths[0]) if (widths:=derived_widths(u, u.src)) is not None else s

# THE DOOR: wrap every remaining bare const in its CAST. keyed on the CONSUMER -- "bare" is a property of the edge
def cast_consts(u:UOp) -> UOp|None:
  if u.op is Ops.CAST and u.src[0].op is Ops.CONST: return None  # a pair's CONST is the value half, not an edge
  return None if (src:=tuple(cast_const(u, s) for s in u.src)) == u.src else u.replace(src=src)

pm_cast_const = PatternMatcher([(UPat(GroupOp.All, name="u"), cast_consts)])
