import unittest
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.engine.realize import get_program
from tinygrad.helpers import AMX

@unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(uops: list[UOp], n=4):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype == dtypes.float.vec(n)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype == dtypes.float.vec(n)]))
  @staticmethod
  def count_half4(uops: list[UOp]):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype == dtypes.half.vec(4)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype == dtypes.half.vec(4)]))

  def test_float4_basic(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule()[0]
    realized_ast = s.ast
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer, opts=opts_to_apply)

    assert TestFloat4.count_float4(program.uops) == (2, 1)

  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "CPU with AMX upcasts float up to size 16")
  def test_float4_multidim(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=2)]).uops
    assert TestFloat4.count_float4(uops) == (4, 2)

  @unittest.skipUnless(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "Only CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_amx(self):
    def kernel_for_shape(size, shift):
      a = Tensor.empty(2, size).realize()
      b = Tensor.empty(2, size).realize()
      c = a + b

      s = c.schedule()[0]
      return get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=shift)]).uops

    sizes = [12, 8, 16]
    shifts = [3, 2, 4]
    expected_upcast_size = [4, 8, 16]
    expected_output = [(6,3), (2,1), (2,1)]

    for i in range(len(sizes)):
      assert TestFloat4.count_float4(kernel_for_shape(sizes[i], shifts[i]), expected_upcast_size[i]) == expected_output[i]

  def test_float4_unaligned_load(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    s = c.schedule()[0]
    realized_ast = s.ast
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer, opts=opts_to_apply)

    assert TestFloat4.count_float4(program.uops) == (0, 1)

  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_unaligned_load(self):
    a = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2)]).uops

    assert TestFloat4.count_float4(uops) == (0, 2)

  @unittest.skipUnless(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "Only CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_unaligned_load_amx(self):
    def kernel_for_shape(size, shift):
      a = Tensor.empty(2, size).realize().shrink(((0, 2), (1, size),))
      b = Tensor.empty(2, size).realize().shrink(((0, 2), (1, size),))
      c = a + b

      s = c.schedule()[0]
      return get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=shift)]).uops

    sizes = [13, 9, 17]
    shifts = [3, 2, 4]
    expected_upcast_size = [4, 8, 16]
    expected_output = [(0,3), (0,1), (0,1)]

    for i in range(len(sizes)):
      assert TestFloat4.count_float4(kernel_for_shape(sizes[i], shifts[i]), expected_upcast_size[i]) == expected_output[i]

  def test_float4_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 8).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # only the first and last conv dot products are aligned in a, and b is never aligned, so no
    # float4 should be emitted (the reduce axis of size 4 is the float4 axis here)

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UNROLL, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (0, 0)

  def test_float4_multidim_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 7).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # the first conv dot product is aligned in a. If we upcast the output and reduce
    # dimension, then we could do float4 for only that one set of loads, but we currently
    # don't.
    # UPDATE: now we do this fusion

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=0)]).uops

    assert TestFloat4.count_float4(uops) in {(0,1), (1,1)}

  def test_float4_expand(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(2).realize().reshape((2, 1)).expand((2,4)).reshape((8,))
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (0, 1)

  def test_float4_heterogeneous(self):
    a = Tensor.empty(8).realize()
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    # should float4 b but not a

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (1, 1)

  def test_half4_load_unrolled(self):
    # from llama 7B shard 4 gpus
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(96000), arg=ShapeTracker(views=(View(shape=(1, 3, 32000, 1), strides=(0, 32000, 1, 0), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(96000), arg=0, src=()),)),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
          UOp(Ops.CAST, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.half, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.VIEW, dtypes.half.ptr(9216), arg=ShapeTracker(views=(View(shape=(1, 3, 32000, 1024), strides=(0, 4096, 0, 1), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(9216), arg=1, src=()),)),)),
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.VIEW, dtypes.half.ptr(32768000), arg=ShapeTracker(views=(View(shape=(1, 3, 32000, 1024), strides=(0, 0, 1024, 1), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(32768000), arg=2, src=()),)),)),)),)),)),)),))

    # TODO: fix this, expected might change but should be positive
    for expected, opts in [
      ((7, 0), [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=3), Opt(op=OptOps.UNROLL, axis=0, arg=4)]),
      ((5, 0), [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4)]),
      ((2, 0), [Opt(op=OptOps.UNROLL, axis=0, arg=4)]),
    ]:
      program = get_program(ast, Device[Device.DEFAULT].renderer, opts=opts)

      count = TestFloat4.count_half4(program.uops)
      assert count == expected, f"{count=}, {expected=}"

  @unittest.skip("this doesn't happen anymore")
  def test_float4_acc(self):
    # from float32 stable diffusion red tinybox
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(33554432), arg=ShapeTracker(views=(View(shape=(1, 1, 128, 512, 512, 1, 1, 1), strides=(0, 0, 262144, 512, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(33554432), arg=0, src=()),)),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float.ptr(67108864), arg=ShapeTracker(views=(View(shape=(1, 1, 1, 256, 4, 514, 4, 514), strides=(0, 0, 0, 262144, 0, 512, 0, 1), offset=-513, mask=((0, 1), (0, 1), (0, 1), (0, 256), (0, 4), (1, 513), (0, 4), (1, 513)), contiguous=False), View(shape=(1, 1, 128, 512, 512, 256, 3, 3), strides=(0, 0, 0, 2056, 1, 4227136, 1058840, 515), offset=0, mask=None, contiguous=False))), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(67108864), arg=1, src=()),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float.ptr(294912), arg=ShapeTracker(views=(View(shape=(1, 1, 128, 512, 512, 256, 3, 3), strides=(0, 0, 2304, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(294912), arg=2, src=()),)),)),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.VIEW, dtypes.float.ptr(128), arg=ShapeTracker(views=(View(shape=(1, 1, 128, 512, 512, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(128), arg=3, src=()),)),)),)),)),))

    for expected, opts in [
      (1, [Opt(op=OptOps.UPCAST, axis=2, arg=4)]),
      (4, [Opt(op=OptOps.UPCAST, axis=2, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)]),
    ]:
      program = get_program(ast, Device[Device.DEFAULT].renderer, opts=opts)
      count = len([uop for uop in program.uops if uop.op is Ops.DEFINE_REG and uop.dtype == dtypes.float.vec(4)])
      assert count == expected, f"{count=}, {expected=}"

  @unittest.skip("this doesn't happen anymore")
  def test_float2_acc(self):
    # from resnet
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.half.ptr(212926464), arg=ShapeTracker(views=(View(shape=(1, 256, 1, 64, 1, 114, 1, 114), strides=(0, 831744, 0, 12996, 0, 114, 0, 1), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(212926464), arg=0, src=()),)),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (4, 6)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.VIEW, dtypes.half.ptr(462422016), arg=ShapeTracker(views=(View(shape=(256, 64, 3, 56, 2, 3, 56, 2), strides=(1806336, 28224, 3, 504, 0, 1, 9, 0), offset=0, mask=((0, 256), (0, 64), (0, 3), (0, 56), (0, 1), (0, 3), (0, 56), (0, 1)), contiguous=False), View(shape=(256, 64, 3, 115, 3, 115), strides=(7225344, 112896, 37632, 336, 112, 1), offset=0, mask=((0, 256), (0, 64), (0, 3), (0, 112), (0, 3), (0, 112)), contiguous=False), View(shape=(256, 64, 456, 456), strides=(7617600, 119025, 345, 1), offset=0, mask=((0, 256), (0, 64), (0, 345), (0, 345)), contiguous=False), View(shape=(1, 256, 1, 64, 4, 114, 4, 114), strides=(0, 13307904, 0, 207936, 51984, 456, 114, 1), offset=0, mask=None, contiguous=True))), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(462422016), arg=1, src=()),)),)),)),)),)),)),))
    for expected, opts in [
      (16, [Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=2), Opt(op=OptOps.LOCAL, axis=2, arg=3), Opt(op=OptOps.UPCAST, axis=3, arg=4)]),  # noqa: E501
      (4, [Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=2)]),
    ]:
      program = get_program(ast, Device[Device.DEFAULT].renderer, opts=opts)
      count = len([uop for uop in program.uops if uop.op is Ops.DEFINE_REG and uop.dtype == dtypes.float.vec(2)])
      assert count == expected, f"{count=}, {expected=}"

if __name__ == '__main__':
  unittest.main()
