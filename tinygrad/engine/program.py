import itertools
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, track_rewrites
from tinygrad.codegen.lowerer2 import pm_lowerer, LowererContext
from tinygrad.uop.ops import graph_rewrite, KernelInfo
from tinygrad.uop.symbolic import sym, symbolic_simple, constant_folding
from tinygrad.renderer import Renderer, ProgramSpec, Estimates
from tinygrad.codegen import full_rewrite
from tinygrad.helpers import flatten, colored, partition, prod
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.dtype import dtypes

# Opt, OptOps should move here

# TODO: only used for hcopt
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.heuristic import hand_coded_optimizations

# **************** Program Creation ****************

range_sym = constant_folding+PatternMatcher([
  # remove range 1 (move to sym?)
  (UPat(Ops.RANGE, src=(UPat(Ops.CONST, arg=1),), name="r"), lambda r: r.const_like(0)),
  # common factor mul
  (UPat.var("x")*UPat.cvar("c1") + UPat.var("y")*UPat.cvar("c2"),
   lambda x,y,c1,c2: (x*(c1//c2)+y)*c2 if c1.arg >= c2.arg and c1.arg%c2.arg == 0 else None)
])

pm_simplify_merge_adjacent = PatternMatcher([
  # merge ranges
  ((UPat(Ops.RANGE, name="r1")*UPat.cvar("c")).named("m")+UPat(Ops.RANGE, name="r2"),
   lambda ctx,r1,c,m,r2: UOp(Ops.RANGE, dtypes.int, src=(r2.src[0]+c*(r1.src[0]-1),), arg=min(r1.arg, r2.arg)) \
     if all([x not in ctx or len(ctx[x]) == 1 for x in [r1,m,r2]]) else None),
])

def apply_opt(ctx:tuple[LowererContext, list[Opt]], r:UOp):
  lc,opts = ctx
  for opt in opts:
    assert opt.op is OptOps.UPCAST
    if opt.axis == r.arg:
      arg = opt.arg if opt.arg != 0 else (r.vmax+1)
      assert (r.vmax+1) % arg == 0, f"can't upcast {r.vmax+1} with {arg}"
      unroll = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(arg), tuple(range(arg))),), ((lc.range_number, arg),))
      lc.range_number += 1
      opts.remove(opt)
      return r.replace(src=(r.src[0]//arg,))*arg + unroll

pm_apply_opts = PatternMatcher([
  (UPat(Ops.RANGE, name="r"), apply_opt)
])

def fix_reduce(r:UOp):
  if len(r.src) == 1: return None
  new_ranges = []
  new_unrolls = []
  for x in UOp.sink(*r.src[1:]).toposort():
    if x.op is Ops.RANGE: new_ranges.append(x)
    if x.op is Ops.UNROLL: new_unrolls.append(x)
    # ignore things like CONST
  unroll_arg = [y for x in new_unrolls for y in x.arg]
  ret = UOp(Ops.CONTRACT, r.dtype.vec(prod([x[1] for x in unroll_arg])), (r.src[0],), tuple(unroll_arg)) if len(unroll_arg) else r.src[0]
  ret_r = r.replace(src=(ret,)+tuple(new_ranges))
  # TODO: can this be a generic rule?
  return ret_r if ret_r is not r else None

pm_fix_reduces = PatternMatcher([(UPat(Ops.REDUCE, name="r"), fix_reduce)])

@track_rewrites(name=lambda _renderer,_ast,ret: ret.name)
def get_program(renderer:Renderer, ast:UOp) -> ProgramSpec:
  assert ast.op is Ops.SINK, "last uop must be sink"

  # get applied_opts on the old path
  k = Kernel(ast, opts=renderer)
  applied_opts: list[Opt] = hand_coded_optimizations(k)
  print(applied_opts)

  # lowerer+sym
  ast: UOp = graph_rewrite(ast, pm_lowerer+range_sym, name="lowerer", ctx=(lc:=LowererContext()), bottom_up=True)

  # simplify merge adjacent
  # NOTE: this should remove range(1)
  children = ast.get_children_map()
  ast = graph_rewrite(ast, pm_simplify_merge_adjacent+symbolic_simple, name="simplify_merge_adjacent", ctx=children)

  # HACK: rewrite opts for the current ranges
  uops = ast.toposort()
  ranges = sorted([u for u in uops if u.op is Ops.RANGE], key=lambda x: x.arg)
  reduce_ranges = set(flatten([u.src[1:] for u in uops if u.op is Ops.REDUCE]))
  upcast_ranges, unroll_ranges = partition(ranges, lambda x: x not in reduce_ranges)
  new_applied_opts = []
  for opt in applied_opts:
    if opt.op is OptOps.UPCAST: new_applied_opts.append(Opt(OptOps.UPCAST, upcast_ranges[opt.axis].arg, opt.arg))
    elif opt.op is OptOps.UNROLL: new_applied_opts.append(Opt(OptOps.UPCAST, unroll_ranges[opt.axis].arg, opt.arg))
    else:
      raise RuntimeError(f"{opt.op} is not supported")

  # do opt apply
  ast = graph_rewrite(ast, pm_apply_opts+range_sym, name="apply_opts", ctx=(lc,new_applied_opts))
  assert len(new_applied_opts) == 0

  # fix reduces
  ast = graph_rewrite(ast, pm_fix_reduces, name="fix_reduces")


  """
  # handle ranges
  uops = ast.toposort()
  ranges = sorted([u for u in uops if u.op is Ops.RANGE], key=lambda x: x.arg)
  reduce_ranges = set(flatten([u.src[1:] for u in uops if u.op is Ops.REDUCE]))
  upcast_ranges, unroll_ranges = partition(ranges, lambda x: x not in reduce_ranges)
  all_ranges = upcast_ranges+unroll_ranges
  new_ranges = all_ranges[:]
  mul_axis = [1]*len(new_ranges)
  extra_axis = [[] for _ in range(len(new_ranges))]

  # process UPCAST/UNROLL requests
  for opt in applied_opts:
    if opt.op in {OptOps.UPCAST, OptOps.UNROLL}:
      axis = opt.axis if opt.op is OptOps.UPCAST else (opt.axis+len(upcast_ranges))
      arg = opt.arg if opt.arg != 0 else new_ranges[axis].vmax
      assert new_ranges[axis].vmax % arg == 0
      new_ranges[axis] = new_ranges[axis].replace(src=(UOp.const(dtypes.int, new_ranges[axis].vmax//arg),))
      mul_axis[axis] *= arg
      extra_axis[axis].append()
      unroll = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(arg), tuple(range(arg))),), ((axis,arg),))

  # put in the optimized ranges
  ast = ast.substitute(dict(zip(all_ranges, new_ranges)), name="replace ranges")
  """

  # codegen
  uops = full_rewrite(ast, renderer)
  assert uops[-1].op is Ops.SINK, "last uop must be sink"

  # determine name
  ranges = sorted([u for u in uops if u.op is Ops.RANGE], key=lambda x: x.arg)
  reduce_ranges = set(flatten([u.src[1:] for u in uops if u.op is Ops.DEFINE_ACC]))
  dims = [colored(x.src[0].arg, 'red' if x in reduce_ranges else 'blue') for x in ranges]
  uops[-1] = uops[-1].replace(arg=KernelInfo(name=('r' if len(reduce_ranges) else 'E')+'_'+colored('_', 'BLACK').join(dims)))

  # render
  src = renderer.render(uops)

  # group non-local bufs by the op type (LOAD or STORE) and the buffer arg. take the max access of that buffer in bytes
  # TODO: these max and min don't work on symbolic, and results are very wrong.
  mem_bytes = sum(max(x.src[0].dtype.nbytes() for x in group)
    for _, group in itertools.groupby([x for x in ast.toposort() if x.op in {Ops.LOAD, Ops.STORE} and x.src[0].base.op is Ops.DEFINE_GLOBAL],
                      key=lambda x: (x.op, x.src[0].base.arg)))
  return ProgramSpec(uops[-1].arg.name, src, renderer.device, ast, uops, applied_opts, mem_bytes,
                     global_size=[1,1,1] if renderer.has_local else None, local_size=[1,1,1] if renderer.has_local else None)

"""
logkerns, logkerns_level = open(getenv("LOGKERNS", ""), "a") if getenv("LOGKERNS", "") else None, getenv("LOGKERNS_LEVEL", 1)
def get_program(renderer:Renderer, ast:UOp) -> ProgramSpec:
  k = Kernel(ast, opts=renderer)
  if not NOOPT:
    if not k.apply_tensor_cores(getenv("TC", 1)): k.apply_opts(hand_coded_optimizations(k))
    if BEAM >= 1:
      from tinygrad.engine.search import beam_search, bufs_from_lin
      kb = Kernel(ast, opts=renderer)
      rawbufs = bufs_from_lin(kb, allocate=False)
      k = beam_search(kb, rawbufs, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))
  if logkerns is not None: logkerns.writelines([f"{(k.ast, k.applied_opts)}\n"])
  return k.to_program()
"""