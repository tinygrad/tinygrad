from tinygrad.uop.ops import UOp, Ops, track_rewrites, graph_rewrite_map, PatternMatcher, UPat, GroupOp
from tinygrad.helpers import pluralize

merge_views = PatternMatcher([
  # merge adjacent views
  (UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="v1"),), name="v2"), lambda v1,v2: v1.replace(arg=v1.arg+v2.arg)),
  # replace MovementOps with VIEW
  (UPat(GroupOp.Movement, src=(UPat.var("x"),), name="mop"), lambda mop,x: x.base.view(mop.st)),
])

add_gbarrier = PatternMatcher([
  # add global barriers after every copy and contiguous
  (UPat((Ops.COPY, Ops.CONTIGUOUS), name="x"), lambda x: x.replace(tag=1).gbarrier() if x.tag is None else None)
])
remove_tags = PatternMatcher([(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if x.tag is not None else None)])

# TODO: write this all with RewriteStep
@track_rewrites(name_fxn=lambda big_sink,ret: f"Schedule {pluralize('Kernel',len([u for u in ret[big_sink].toposort() if u.op is Ops.KERNEL]))}")
def get_kernelize_map(big_sink:UOp) -> dict[UOp, UOp]:
  # optimize the graph at the tensor level
  tensor_map = graph_rewrite_map(big_sink, merge_views, name="Merge Views")

  # place GBARRIERS
  tensor_map = graph_rewrite_map(tensor_map[big_sink], add_gbarrier, input_map=tensor_map, name="Add GBARRIERs")
  tensor_map = graph_rewrite_map(tensor_map[big_sink], remove_tags, input_map=tensor_map, name="remove_tags")

  return tensor_map
