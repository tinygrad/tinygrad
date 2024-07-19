import unittest

from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.ops import LazyOp, BufferOps, MemBuffer, BinaryOps, MetaOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.realize import get_kernels

class TestKernel(unittest.TestCase):
  def test_two_kernel_ast(self):
    ld_1 = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.int32, ShapeTracker.from_shape((1,))))
    ld_2 = LazyOp(BufferOps.LOAD, (), MemBuffer(2, dtypes.int32, ShapeTracker.from_shape((1,))))
    alu1 = LazyOp(BinaryOps.ADD, (ld_1, ld_2))
    st_0 = LazyOp(BufferOps.STORE, (alu1,), MemBuffer(0, dtypes.int32, ShapeTracker.from_shape((1,))))
    sink1 = LazyOp(MetaOps.KERNEL, (st_0,))
    c_1 = LazyOp(BufferOps.LOAD, (sink1,), MemBuffer(0, dtypes.int32, ShapeTracker.from_shape((1,))))
    st_1 = LazyOp(BufferOps.STORE, (c_1,), MemBuffer(0, dtypes.int32, ShapeTracker.from_shape((1,))))
    sink2 = LazyOp(MetaOps.KERNEL, (st_1,))
    kernels = tuple(get_kernels(Device[Device.DEFAULT].renderer, sink2))
    programs = [k.to_program() for k in kernels]

    assert(len(programs) == 2)
    assert(len(kernels) == 2)
    assert(kernels[0].ast == sink2)
    assert(kernels[1].ast == sink1)
