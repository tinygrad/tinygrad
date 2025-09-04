import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import Context, prod
from tinygrad.uop.ops import AxisType
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations

# TODO: remove this
from tinygrad.codegen.opt.kernel import Kernel
from test.test_linearizer import push_views, helper_linearizer_opt

class TestHandCodedOpts(unittest.TestCase):
  def test_masked_upcast(self):
    layer_1 = Tensor.cat(*[Tensor.empty(5) for _ in range(4)])
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.empty(6, 20))

    s = layer_2.schedule()[-1]
    k = Kernel(push_views(s.ast))
    k.apply_opts(hand_coded_optimizations(k))
    assert len(k.bufs) == 6  # make sure all ops are done in one kernel
    # masked upcast should upcast masked axis of size 7
    # masked upcast should not upcast large (20) last axis
    # float4/other hcopt shouldn't upcast last axis, since we already have 7 upcast, and the last axis is not very contiguous
    assert k.upcasted == 1 and k.full_shape[-1] == 7

  @unittest.skipIf(Device.DEFAULT in {"METAL", "WEBGPU"}, "METAL/WEBGPU split this kernel since it has 37 buffers")
  def test_masked_upcast_wino(self):
    monster = Tensor.stack(*[Tensor.stack(*[Tensor.empty(16) for _ in range(6)]) for _ in range(6)])

    s = monster.schedule()[-1]
    k = Kernel(push_views(s.ast))
    k.apply_opts(hand_coded_optimizations(k))
    assert len(k.bufs) == 37  # make sure all ops are done in one kernel
    # should upcast the two Tensor.stacks
    assert k.upcasted >= 2 and k.full_shape[k.shape_len-k.upcasted:k.shape_len].count(6) == 2

  def test_masked_upcast_wino_full(self):
    with Context(WINO=1):
      x,w = Tensor.rand(1,4,8,8, requires_grad=True).realize(), Tensor.rand(4,4,3,3, requires_grad=True).realize()
      out = Tensor.conv2d(x,w, padding=1)
      out.mean().backward()

      upcasts = []
      wino_schedule = out.schedule()
      # collect upcasts of tile transform kernels
      for i, si in enumerate(wino_schedule):
        k = Kernel(push_views(si.ast))
        k.apply_opts(hand_coded_optimizations(k))
        if k.reduceop is not None: continue  # not a tile transform kernel (there is a gemm reduce kernel)
        if len(k.bufs) < 22: continue  # not a tile transform kernel (there's a permute kernel at the end)
        upcasts.append(tuple(k.full_shape[k.shape_len - k.upcasted:k.shape_len]))
      assert len(upcasts) == 3  # 3 transformation matrices
      assert len(wino_schedule) <= 4  # 4 kernels
      # this test case's inputs are too small, so one of the 4-stacks became a local, which is fine i guess
      assert upcasts.count((6, 6)) == 2 #and upcasts.count((4, 4)) == 1

      backward_schedule = Tensor.schedule(x.grad, w.grad)
      for si in backward_schedule:
        k = Kernel(push_views(si.ast))
        k.apply_opts(hand_coded_optimizations(k))
        if len(k.bufs) < 20: continue  # not a tile transform kernel
        # heuristic number to make sure that at least some upcasts but not too many upcasts are being done
        assert 6 <= prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 216
      assert len(backward_schedule) <= 13  # just the current number, but it could be better

  def test_masked_upcast_many(self):
    layer_1 = Tensor.cat(Tensor.rand(3, 4), Tensor.rand(4, 4))
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 7, 4))
    layer_3 = Tensor.cat(layer_2.unsqueeze(0), Tensor.rand(6, 7, 7, 4))

    k = helper_linearizer_opt(layer_3)[-1]
    assert len(k.bufs) == 5  # make sure all ops are done in one kernel
    # check that we don't do too many upcasts
    assert prod(k.full_shape[k.shape_len-k.upcasted:k.shape_len]) <= 49

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_matvec(self):
    N = 128
    a = Tensor.rand(1, N).realize()
    b = Tensor.rand(N, N).realize()
    c = a @ b

    k = helper_linearizer_opt(c)[-1]

    assert k.group_for_reduces == 1
    assert k.axis_types.count(AxisType.LOCAL) == 1
    assert k.upcasted == 1

if __name__ == '__main__':
  unittest.main()
