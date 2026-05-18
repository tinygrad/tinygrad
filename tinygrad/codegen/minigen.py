from dataclasses import dataclass
from tinygrad.uop.ops import UOp, graph_rewrite, PatternMatcher, GroupOp, UPat, Ops
from tinygrad.uop.spec import type_verify, spec_program
from tinygrad.dtype import dtypes, Invalid, AddrSpace
from tinygrad.helpers import SPEC
from tinygrad.renderer import Renderer
from tinygrad.codegen.late.devectorizer import pm_add_loads, reduce_to_acc
from tinygrad.codegen.late.linearizer import CFGContext, pm_split_ends, pm_add_control_flow
from tinygrad.uop.decompositions import get_late_rewrite_patterns, get_transcendental_patterns, pm_dtype_decomps
from tinygrad.uop.symbolic import sym

pm_lower_weakint = PatternMatcher([
  (UPat(GroupOp.All, dtypes.weakint, name="x"), lambda x: x.replace(dtype=dtypes.int))
])

@dataclass
class ReduceContext:
  acc_num: int = 0
  local_num: int = 0

def stage_to_local(ctx:ReduceContext, x:UOp):
  # TODO: addrspace shouldn't be on dtype
  ret = UOp(Ops.DEFINE_LOCAL, x.dtype.ptr(size=x.numel(), addrspace=AddrSpace.LOCAL), (), arg=ctx.local_num)
  ctx.local_num += 1
  return ret.after(ret.reshape(*[u.vmax+1 for u in x.src[1:]]).index(*x.src[1:]).store(x.src[0]).end(*x.src[1:]))

pm_minimal_reduce = PatternMatcher([
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
  (UPat(Ops.STAGE, name="x"), stage_to_local),
])

pm_minimal_move_gates_from_index = PatternMatcher([
  # here we create the alt value for load to be 0s and remove the where Invalid
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").load(name="l"),
   lambda buf,gate,idx,cast,l: buf.index(idx, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate,idx,cast,data: buf.index(idx, ptr=True).cast(cast.dtype).store(data, gate)),
])


def minigen_to_sink(ast:UOp, ren:Renderer, optimize:bool) -> UOp:
  sink = ast

  # do single symbolic (this rewrites POW)
  sink = graph_rewrite(sink, sym, name="symbolic")

  # REDUCE is not allowed in programs, we need to do that REDUCE somewhere
  # this creates a register where the reduce happens
  # STAGE is also not allowed in programs, this is similar to REDUCE
  sink = graph_rewrite(sink, pm_minimal_reduce, ctx=ReduceContext(), name="remove reduce/stage")

  # we need to add loads
  # this is really a store to DEFINE_REG, but load is simpler
  # LOAD(DATA) is anonymous store -> AFTER(anon_buf, STORE(anon_buf, DATA))
  sink = graph_rewrite(sink, pm_add_loads, name="add loads")

  # we need to lower weakint for the program
  # this will be simpler when we have implicit dtype
  sink = graph_rewrite(sink, pm_lower_weakint, name="remove weakint")

  # move gates from INDEX to LOAD/STORE (Invalid isn't renderable)
  sink = graph_rewrite(sink, pm_minimal_move_gates_from_index, name="move gates")

  # split ENDs, renderable ENDs can only end one RANGE
  sink = graph_rewrite(sink, pm_split_ends, name="split ends")

  # **** enter decanonicalize *****

  # decompose ops like SIN/THREEFRY into renderable versions
  supported_ops = tuple(ren.code_for_op.keys())
  pm_decomp = get_late_rewrite_patterns(supported_ops, disable_fast_idiv=True) + \
              get_transcendental_patterns(supported_ops, force_transcendental=False)
  sink = graph_rewrite(sink, pm_decomp, ctx=ren.target, name="decompose ops to renderable")

  # decompose dtypes we don't support into renderable versions
  sink = graph_rewrite(sink, pm_dtype_decomps, ctx=(set(), ren.target), name="decomp dtypes")

  # this was the linearizer, add control flow edges where they are needed on RANGEs
  sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

  if SPEC: type_verify(sink, spec_program)
  return sink