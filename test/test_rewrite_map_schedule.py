import unittest
from dataclasses import dataclass, field

from tinygrad import Tensor
from tinygrad.engine.realize import run_schedule
from tinygrad.helpers import unwrap
from tinygrad.ops import GroupOp, Ops, PatternMatcher, UOp, UPat, graph_rewrite, graph_rewrite_map, symbolic_simple, track_rewrites
from tinygrad.engine.schedule import ScheduleItem, remove_movement_ops
from tinygrad.shape.shapetracker import ShapeTracker

@dataclass(frozen=True)
class ScheduleContext:
  realizes:dict[UOp, UOp] = field(default_factory=dict)

def realize_metaop(ctx:ScheduleContext, root:UOp):
  buf_uop, op = root.src
  ctx.realizes[buf_uop] = UOp.store(buf_uop, ShapeTracker.from_shape(root.shape).to_uop(), op)
  return UOp.load(buf_uop, unwrap(root.st).to_uop(), dtype=root.dtype)

add_realizes = PatternMatcher([
  (UPat(Ops.VIEW, name="root", src=(UPat(), UPat(Ops.COPY))), realize_metaop),
])

def _add_buf(ctx:list[UOp], x:UOp):
  ctx.append(x)
  return UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), len(ctx)-1)
add_buffers = PatternMatcher([
  (UPat(Ops.BUFFER, name="x"), _add_buf),
  (UPat(Ops.SINK, src=(UPat.store(UPat.var("b"), UPat(), UPat(GroupOp.Meta, name="x")),)), lambda b,x: x.replace(src=(b, *x.src))),
])

class TestSchedule(unittest.TestCase):
  @track_rewrites(named=True)
  def test_tiny_copy(self):
    a = Tensor([1])
    #b.realize()
    outs:list[UOp] = [a.lazydata]
    # first, we do the const folding
    node_map = graph_rewrite_map(UOp.sink(*outs), remove_movement_ops+symbolic_simple)
    # then, we sink the node_map and realize
    node_map = graph_rewrite_map(UOp.sink(*[node_map[x] for x in outs]), add_realizes, ctx:=ScheduleContext())
    schedule: list[ScheduleItem] = []
    for store in ctx.realizes.values():
      ast = graph_rewrite(UOp.sink(store), add_buffers, bufs:=[])
      schedule.append(ScheduleItem(ast, tuple(x.buffer for x in bufs), (), ()))
    for k,v in node_map.items():
      if v.op is Ops.LOAD and v.buf_uop in ctx.realizes: k.become(v.buf_uop.view(k.st))
    run_schedule(schedule)
    self.assertIsNotNone(a.lazydata.realized)
    self.assertEqual(a.tolist(), [1])

if __name__ == "__main__":
  unittest.main()
