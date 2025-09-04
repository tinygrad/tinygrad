import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import prod
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
