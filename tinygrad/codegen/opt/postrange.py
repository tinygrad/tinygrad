from typing import cast
from dataclasses import replace
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, KernelInfo
from tinygrad.helpers import colored
from tinygrad.codegen.opt.kernel import axis_colors, AxisType

def rename_sink(s:UOp):
  if s.arg is not None and s.arg.name != "test": return None

  # get all ranges (sorted)
  rngs = sorted([u for u in s.parents if u.op is Ops.RANGE], key=lambda x: x.arg[0:-1])

  # add name to kernel
  name = "k" + colored('_', 'BLACK').join(['']+[colored(x.src[0].render(), axis_colors[x.arg[-1]]) for x in rngs])
  return s.replace(arg=KernelInfo(name=name) if s.arg is None else replace(s.arg, name=name))

def global_stores_are_global(s:UOp):
  if cast(PtrDType, s.src[0].dtype).addrspace != AddrSpace.GLOBAL: return None
  return s.substitute({u:u.replace(arg=u.arg[0:-1]+(AxisType.GLOBAL,)) for u in s.src[2:] if u.op is Ops.RANGE and u.arg[-1] is AxisType.LOOP})

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

  (UPat(Ops.RANGE, name="r"), split_range),
  (UPat(Ops.STORE, name="s"), global_stores_are_global),
  (UPat(Ops.SINK, name="s"), rename_sink),
])
