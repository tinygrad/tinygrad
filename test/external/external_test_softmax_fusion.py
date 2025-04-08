import unittest
from tinygrad import Tensor, GlobalCounters, Context, Device
from tinygrad.ops import Ops, UOp, graph_rewrite, PatternMatcher, track_rewrites, UPat
from tinygrad.codegen.kernel import Kernel

from tinygrad.dtype import dtypes  # noqa: F401  # pylint: disable=unused-import
from tinygrad.shape.shapetracker import ShapeTracker, View  # noqa: F401  # pylint: disable=unused-import

# softmax kernel
softmax_ast = eval("""UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.MUL, dtypes.float, arg=None, src=(
    x1:=UOp(Ops.EXP2, dtypes.float, arg=None, src=(
      UOp(Ops.MUL, dtypes.float, arg=None, src=(
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          x4:=UOp(Ops.VIEW, dtypes.float,
                   arg=ShapeTracker(views=(View(shape=(32, 10), strides=(10, 1), offset=0, mask=None, contiguous=True),)), src=(
            UOp(Ops.BUFFER, dtypes.float, arg=320, src=(
              x6:=UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),
              UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),)),)),
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.VIEW, dtypes.float,
                   arg=ShapeTracker(views=(View(shape=(32, 10), strides=(1, 0), offset=0, mask=None, contiguous=False),)), src=(
              UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (1,)), src=(
                 x4,)),)),
            UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
              x12:=UOp(Ops.VIEW, dtypes.void,
                   arg=ShapeTracker(views=(View(shape=(32, 10), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=(
                 x6,)),)),)),)),
        UOp(Ops.CONST, dtypes.float, arg=1.4426950408889634, src=(
           x12,)),)),)),
    UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(32, 10), strides=(1, 0), offset=0, mask=None, contiguous=False),)), src=(
      UOp(Ops.RECIP, dtypes.float, arg=None, src=(
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
           x1,)),)),)),)),))""")

pm_expand_view = PatternMatcher([
  (UPat(Ops.VIEW, name="view"),
   lambda view: UOp(Ops.EXPAND_AXIS, view.dtype, view.src,
                    tuple(i for i,x in enumerate(view.arg.views[-1].strides) if x == 0)) if view.arg.views[-1].strides == (1, 0) else None),
])

@track_rewrites()
def rewrite_softmax(ast):
  from tinygrad.engine.grouper import merge_views, add_buffer_ops, fix_kernel_ops
  sink = graph_rewrite(ast, pm_expand_view)
  buffers = (UOp(Ops.BUFFER, dtypes.float, arg=320, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),
    UOp(Ops.UNIQUE, dtypes.void, arg=1, src=()),)), UOp(Ops.BUFFER, dtypes.float, arg=320, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),
    UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),)))
  sink = graph_rewrite(sink, merge_views+add_buffer_ops+fix_kernel_ops, ctx=({}, buffers), bottom_up=True)
  return sink

class TestSoftmaxFusion(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with Context(TRACK_MATCH_STATS=0): cls.test = Tensor.ones(32, 10).contiguous().realize()

  def setUp(self):
    GlobalCounters.reset()

  def test_softmax(self):
    # this is the softmax from scaled_dot_product_attention
    # it becomes 3 kernels
    print("*** softmax ***")
    with Context(NOOPT=1, DEBUG=2):
      out = self.test.softmax(-1)
      out.realize()

  @unittest.skip("no EXPAND_AXIS")
  def test_softmax_fuse(self):
    sink = rewrite_softmax(softmax_ast)
    k = Kernel(sink, Device.default.renderer)
    prg = k.to_program()
    print(prg.src)

  def test_norm(self):
    print("*** norm ***")
    with Context(NOOPT=1, DEBUG=2):
      # NOTE: you don't actually need the expand, it's broadcasted
      out = self.test / self.test.mean(-1, keepdim=True).expand(32, 10)
      out.realize()

  def test_single_kernel_norm(self):
    with Context(NOOPT=1, DEBUG=2):
      inp = self.test.reshape(32, 10, 1)
      div = self.test.reshape(32, 1, 10).expand(32, 10, 10).mean(axis=-1, keepdim=True)
      out = inp / div
      out.realize()

  def test_single_kernel_softmax(self):
    with Context(NOOPT=1, DEBUG=2):
      inp = self.test.reshape(32, 10, 1)
      imx = self.test.reshape(32, 1, 10).expand(32, 10, 10).max(axis=-1, keepdim=True)
      m = inp - imx.detach()
      e = m.exp()
      ss = e.reshape(32,1,10).expand(32, 10, 10).sum(axis=-1, keepdim=True)
      out = e.div(ss)
      out.realize()

      """
      inp = self.test.reshape(32, 10, 1, 1)
      imx = self.test.reshape(32, 1, 10, 1).expand(32, 10, 10, 1).max(axis=-2, keepdim=True)
      m = inp - imx.detach()
      e = m.exp()
      ss = e.reshape(32,1,1,10).expand(32, 10, 1, 10).sum(axis=-1, keepdim=True)
      out = e.div(ss)
      out.realize()
      """

if __name__ == '__main__':
  unittest.main()
