import unittest
from dataclasses import dataclass, field
from tinygrad import dtypes
from tinygrad.uop.ops import PatternMatcher, UOp, graph_rewrite, track_rewrites, Ops, UPat, GroupOp, RewriteNotReady

# we could insert CHILDREN node

@dataclass
class ChildrenContext:
  children: dict[UOp, list[UOp]]|None = None
  seen_children: dict[UOp, set[int]] = field(default_factory=dict)

def extract_children(ctx:ChildrenContext, x:UOp):
  if ctx.children is not None: return
  ctx.children = {}
  for k,v in x.get_children_map().items():
    if len(v) > 1: ctx.children[k] = list(v.keys())

def mark_children(ctx:ChildrenContext, x:UOp):
  found = False
  new_srcs = []
  for s in x.src:
    if s in ctx.children:
      ret = UOp(Ops.CHILDREN, s.dtype, (s.replace(tag=1),), arg=len(ctx.children[s]))
      ret = UOp(Ops.CHILD, s.dtype, src=(ret,), arg=ctx.children[s].index(x))
      new_srcs.append(ret)
      found = True
    else:
      new_srcs.append(s)
  return x.replace(src=tuple(new_srcs)) if found else None

pm_children = PatternMatcher([
  (UPat(Ops.SINK, name="x"), extract_children),
  (UPat(GroupOp.All, name="x"), mark_children),
])

def visit_child(ctx:ChildrenContext, x:UOp):
  print(f"visit CHILD {x.arg} bottom up")
  if x.src[0] not in ctx.seen_children: ctx.seen_children[x.src[0]] = set()
  ctx.seen_children[x.src[0]].add(x.arg)

def visit_children(ctx:ChildrenContext, x:UOp):
  if x.tag == 1: return None
  if len(ctx.seen_children[x]) != x.arg:
    print("visit CHILDREN bottom up -- not ready")
    raise RewriteNotReady
  print("visit CHILDREN bottom up -- READY")
  return x.replace(tag=1)

pm_child_visitor = PatternMatcher([
  (UPat(Ops.CHILD, name="x"), visit_child),
  (UPat(Ops.CHILDREN, name="x"), visit_children),
])

class TestChildrenRewrite(unittest.TestCase):
  @track_rewrites("test_children_rewrite")
  def test_children_rewrite(self):
    a = UOp.variable("a", 0, 10).exp2()
    b = a+2
    c = a+3
    d = b+c
    sink = d.sink()
    sink = graph_rewrite(sink, pm_children, ctx=ChildrenContext(), bottom_up=True)
    sink = graph_rewrite(sink, pm_child_visitor, ctx=ChildrenContext(), bottom_up=True)

if __name__ == '__main__':
  unittest.main()
