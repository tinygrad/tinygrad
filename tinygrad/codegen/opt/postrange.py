from typing import cast
import math
from dataclasses import replace
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, KernelInfo
from tinygrad.helpers import colored, USE_TC, DEBUG
from tinygrad.codegen.opt.kernel import axis_colors, AxisType
from tinygrad.renderer import Renderer
from tinygrad.codegen.opt.tc import TensorCore

def apply_tensor_cores(ctx:tuple[dict, Renderer], in0:UOp, in1:UOp, r_range:UOp, reduceop:UOp):
  if not USE_TC: return None
  # tensor cores have three ranges. X, Y, and REDUCE
  in0_ranges = sorted([u for u in in0.ranges if u not in in1.ranges], key=lambda x: x.arg[0])
  in1_ranges = sorted([u for u in in1.ranges if u not in in0.ranges], key=lambda x: x.arg[0])
  if not len(in0_ranges) or not len(in1_ranges): return None
  in0_range, in1_range = in0_ranges[0], in1_ranges[0]
  if DEBUG >= 2: print('TC', in0_range.arg, in1_range.arg, r_range.arg)

  # confirm the dtype and size is good
  tc_opts: list[TensorCore] = []
  for tc in ctx[1].tensor_cores:
    if reduceop.dtype == tc.dtype_out and in0.dtype == tc.dtype_in and in1.dtype == tc.dtype_in:
      if all(i <= j for i,j in zip(tc.dims, [in0_range.vmax+1, in1_range.vmax+1, r_range.vmax+1])):
        tc_opts.append(tc)
  if len(tc_opts) == 0: return None
  tc = tc_opts[0]

  # create the new ranges as speced by the tensor core
  old_range = [in0_range, in1_range, r_range]
  new_range = [r.replace(src=(r.src[0]//tc.dims[i],), arg=r.arg[0:-1]+(0, r.arg[-1])) for i,r in enumerate(old_range)]
  new_range_args = [list(x.arg[0:-1]) for x in new_range]
  new_reduce_range = new_range[2]
  red_ranges = []

  locals_range = 90

  ne: list[UOp] = []
  for o in tc.opts:
    axis = 1-int(o[1])
    if o[0] == "u":
      new_range_args[axis][-1] += 1
      lrange = UOp.range(dtypes.int, 2, *new_range_args[axis], AxisType.UPCAST)
    else:
      lrange = UOp.range(dtypes.int, 2, locals_range, AxisType.LOCAL)
      locals_range += 1
    ne.append(lrange)
    new_range[axis] = (2 * new_range[axis]) + lrange
  for _, amt in tc.get_reduce_axes():
    new_range_args[2][-1] += 1
    lrange = UOp.range(dtypes.int, amt, *new_range_args[2], AxisType.UNROLL)
    ne.append(lrange)
    red_ranges.append(lrange)
    new_range[2] = (amt * new_range[2]) + lrange
  tne = [x.replace(tag=1) for x in ne]

  # replace ranges in other parts of the graph
  for x,y in zip(old_range, new_range): ctx[0][x] = y

  # apply the swizzled ranges to the srcs
  srcs = [s.substitute(dict(zip(old_range, new_range))).substitute(dict(zip(ne, tne))) for s in (in0, in1)]
  srcs = [x.substitute(dict(zip(tne, [ne[i] for i in p]))) for x,p in zip(srcs, tc.permutes_for_shape_str(tc.base_shape_str()))]

  ned = dict(zip(tc.base_shape_str(), ne))
  tc_reduce_axes = tuple([ned[f"r{i}"].arg[0] for i in range(len(tc.get_reduce_axes()))])
  base_upcast_axes = tuple([(ned[s].arg[0], 2) for s in tc.base_upcast_axes()])
  tc_upcast_axes = tuple([base_upcast_axes[:int(math.log2(tc.elements_per_thread[i]))] for i in range(3)])

  # construct the op
  # TODO: remove tc_upcast_axes from the arg
  wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, ctx[1].device, tc.threads, tc_upcast_axes, tc_reduce_axes)
  wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
    UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0]),
    UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1]),
    UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg)
  tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2])
  ret = tc_uop.reduce(new_reduce_range, arg=Ops.ADD)
  # confirm the UNROLLs aren't actually used, these need to be broadcast MUL
  assert all(u not in red_ranges for u in ret.toposort()), "UNROLLs in TC"
  return ret

pm_postrange_opt_early = PatternMatcher([
  ((UPat.var("in0")*UPat.var("in1")).reduce(UPat(Ops.RANGE, name="r_range"), name="reduceop", arg=Ops.ADD), apply_tensor_cores),
  (UPat(Ops.SINK, name="s"), lambda ctx,s: s.substitute(ctx[0])),
])

# *** late ***

def global_stores_are_global(s:UOp):
  if cast(PtrDType, s.src[0].dtype).addrspace != AddrSpace.GLOBAL: return None
  return s.substitute({u:u.replace(arg=u.arg[0:-1]+(AxisType.GLOBAL,)) for u in s.src[2:] if u.op is Ops.RANGE and u.arg[-1] is AxisType.LOOP})

def rename_sink(s:UOp):
  if s.arg is not None and s.arg.name != "test": return None

  # get all ranges (sorted)
  rngs = sorted([u for u in s.parents if u.op is Ops.RANGE], key=lambda x: x.arg[0:-1])

  # add name to kernel
  name = "k" + colored('_', 'BLACK').join(['']+[colored(x.src[0].render(), axis_colors[x.arg[-1]]) for x in rngs])
  return s.replace(arg=KernelInfo(name=name) if s.arg is None else replace(s.arg, name=name))

def split_range(r:UOp):
  if len(r.arg) > 2: return None
  N = 4
  rd = r.src[0].divides(N)
  if rd is None: return None
  sr = r.replace(src=(rd,), arg=r.arg[0:-1]+(0,r.arg[-1]))
  er = UOp(Ops.RANGE, dtypes.int, src=(UOp.const(dtypes.int, N),),
           arg=r.arg[0:-1]+(1,AxisType.UNROLL if r.arg[-1] is AxisType.REDUCE else AxisType.UPCAST))
  return sr*N+er

def flatten_range_in_terminators(r:UOp):
  off = 2 if r.op is Ops.STORE else 1
  rngs = r.src[off:]
  if not len(rngs): return None
  new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
  return r.replace(src=r.src[:off]+tuple(new_rngs))

pm_postrange_opt = PatternMatcher([
  # flatten ranges
  (UPat((Ops.REDUCE, Ops.STORE), name="r"), flatten_range_in_terminators),
  (UPat(Ops.STORE, name="s"), global_stores_are_global),
  # this is optional
  (UPat(Ops.RANGE, name="r"), split_range),
  # run this last
  (UPat(Ops.SINK, name="s"), rename_sink),
])
