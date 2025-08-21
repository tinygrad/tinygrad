from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.helpers import partition, flatten, prod

def unroll_range(r:UOp):
  # all ranges sub 5 can be UNROLLS
  if r.src[0].op is Ops.CONST and r.vmax < 5:
    i = r.arg
    s = r.vmax+1
    return UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s), tuple(range(s))),), ((i,s),), tag=1)

def fix_reduce(r:UOp):
  reduce_range, reduce_expand = partition(r.src[1:], lambda y: y.op is Ops.RANGE)
  if len(reduce_expand) == 0: return None
  assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand}"
  ret = r.src[0]
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, r.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis), tag=1)
  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return UOp(Ops.REDUCE, r.dtype, (ret,)+tuple(reduce_range), r.arg)

pm_postrange_opt = PatternMatcher([
  (UPat(Ops.RANGE, name="r"), unroll_range),
  (UPat(Ops.REDUCE, name="r"), fix_reduce),
])
