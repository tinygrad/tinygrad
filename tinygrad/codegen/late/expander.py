from tinygrad.dtype import AddrSpace
from tinygrad.helpers import partition
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, AxisType
from tinygrad.schedule.rangeify import BufferizeOpts

def fix_group_for_reduce(x:UOp):
  reduce_gfr, reduce_r = partition(x.src[1:], lambda u: u.op is Ops.RANGE and u.arg[1] == AxisType.GROUP_REDUCE)
  if len(reduce_gfr) == 0: return None

  # NOTE: if there's other locals here, we need them in the buffer too
  upstream_locals = [u for u in x.toposort() if u.op is Ops.RANGE and u.arg[1] == AxisType.LOCAL]

  # do only the non grouped reduces early
  ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
  reduce_loop = [x.replace(arg=(x.arg[0]+100, AxisType.REDUCE)) for x in reduce_gfr]
  buf = ret.bufferize(*upstream_locals, *reduce_gfr, arg=BufferizeOpts(reduce_gfr[0].arg[0], AddrSpace.LOCAL)).index(*upstream_locals, *reduce_loop)

  # do the final reduce (if/barrier are added in gpudims step)
  # NOTE: we remove all horizontal reduces here, they remain in the first reduce
  return buf.reduce(*reduce_loop, arg=(x.arg[0], ()))

pm_group_for_reduce = PatternMatcher([
  # fix group for reduce
  (UPat(Ops.REDUCE, name="x"), fix_group_for_reduce),
])
