from typing import Any
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, RewriteNotReady
from tinygrad.helpers import argsort, prod, all_same, pluralize, getenv

from tinygrad.uop.ops import track_rewrites, graph_rewrite_map, graph_rewrite

# 1. add contiguous where we have to

add_contiguous = PatternMatcher([(UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"),
                                  lambda ctx,x: x.replace(tag=1).contiguous() if x in ctx and x.tag is None else None)])
remove_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if x.tag is not None else None)])

# 2. mark all children

@dataclass
class ChildrenContext: children: dict[UOp, list[UOp]]|None = None
def extract_children(ctx:ChildrenContext, x:UOp):
  if ctx.children is not None: return
  # REDUCE_AXIS is fine here, should go to contig only (gate)
  ctx.children = {k:list(v.keys()) for k,v in x.get_children_map().items() if len(v) > 1 and any(x.op is Ops.REDUCE_AXIS for x in k.toposort())}
def mark_children(ctx:ChildrenContext, x:UOp):
  new_srcs = [(UOp(Ops.CHILD, s.dtype, src=(s,), arg=(ctx.children[s].index(x), len(ctx.children[s]))) if s in ctx.children else s) for s in x.src]
  return x.replace(src=tuple(new_srcs))
pm_children = PatternMatcher([
  (UPat(Ops.SINK, name="x"), extract_children),
  (UPat(GroupOp.All-{Ops.CHILD}, name="x"), mark_children),
])

# 3. rangeify

@dataclass
class RangeifyContext:
  idx: int = 0
  regs: int = 0
  seen_children: dict[UOp, dict[int, UOp]] = field(default_factory=dict)
  seen_child: dict[UOp, Any] = field(default_factory=dict)


@track_rewrites(name=lambda sink,ret: f"Schedule {pluralize('Kernel',len([u for u in ret[sink].toposort() if u.op is Ops.KERNEL]))}", replay=True)
def get_kernelize_map(sink:UOp) -> dict[UOp, UOp]:
  tensor_map = {sink:sink}
  realize_map = {x.base:None for x in sink.src}
  tensor_map = graph_rewrite_map(tensor_map[sink], add_contiguous, ctx=realize_map, bottom_up=True, input_map=tensor_map, name="add_contiguous")
  tensor_map = graph_rewrite_map(tensor_map[sink], remove_tags, input_map=tensor_map, name="finalize_contiguous")
  tensor_map = graph_rewrite_map(tensor_map[sink], pm_children, ctx=ChildrenContext(), bottom_up=True, input_map=tensor_map, name="children")
  tensor_map = graph_rewrite_map(tensor_map[sink], pm_rangeify, ctx=RangeifyContext(), bottom_up=True, input_map=tensor_map, name="rangeify")
  if getenv("VIZ"): graph_rewrite(tensor_map[sink], PatternMatcher([]), name="View Rangeify Graph")
  return tensor_map
