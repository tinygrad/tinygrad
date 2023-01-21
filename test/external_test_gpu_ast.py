#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_gpu import GPUBuffer, CLASTKernel

class TestAST(unittest.TestCase):
  def test_conv_zeroview_ast(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 3, 4), views=[View((1, 1, 3, 4), (2, 2, 2, 1), -3), ZeroView((1, 1, 1, 2), ((0, 1), (0, 1), (-1, 2), (-1, 3))), View((1, 1, 3, 4), (0, 0, 4, 1), 0)]))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 3, 4), views=[View((1, 1, 3, 4), (0, 0, 0, 0), 0)]))
    op1 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    ast = LazyOp(UnaryOps.RELU, (op1,), None)
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)

  def test_first_op_conv(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 3, 3, 3, 4), views=[View((1, 130, 258, 1, 12), (393216, 3072, 12, 12, 1), -3084), ZeroView((1, 128, 256, 1, 12), ((0, 1), (-1, 129), (-1, 257), (0, 1), (0, 12))), View((1, 64, 128, 8, 4, 3, 3, 3, 4), (0, 6192, 24, 0, 0, 3096, 12, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(128, 768, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 3, 3, 3, 4), views=[View((1, 64, 128, 8, 4, 3, 3, 3, 4), (0, 0, 0, 432, 4, 144, 16, 48, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 108, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 8, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(32,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    op3 = LazyOp(UnaryOps.RELU, (op2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    buf4 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    op4 = LazyOp(UnaryOps.EXP, (op2,), None)
    op5 = LazyOp(BinaryOps.SUB, (buf4,op4,), None)
    op6 = LazyOp(UnaryOps.RELU, (op5,), None)
    op7 = LazyOp(BinaryOps.MUL, (buf3,op6,), None)
    op8 = LazyOp(BinaryOps.SUB, (op3,op7,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op8,), (64, 1024, 4))
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)

  def test_second_op_conv(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 3, 3), views=[View((1, 66, 130, 32, 1), (262144, 4096, 32, 1, 1), -4128), ZeroView((1, 64, 128, 32, 1), ((0, 1), (-1, 65), (-1, 129), (0, 32), (0, 1))), View((1, 64, 128, 8, 4, 1, 1, 3, 3), (266240, 4160, 32, 4, 1, 12480, 12480, 4160, 32), 0)]), hostbuf=GPUBuffer(shape=(64, 1024, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 3, 3), views=[View((1, 64, 128, 8, 4, 1, 1, 3, 3), (0, 0, 0, 36, 1, 0, 0, 12, 4), 0)]), hostbuf=GPUBuffer(shape=(8, 9, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 8, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(32,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    op3 = LazyOp(UnaryOps.RELU, (op2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    buf4 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    op4 = LazyOp(UnaryOps.EXP, (op2,), None)
    op5 = LazyOp(BinaryOps.SUB, (buf4,op4,), None)
    op6 = LazyOp(UnaryOps.RELU, (op5,), None)
    op7 = LazyOp(BinaryOps.MUL, (buf3,op6,), None)
    op8 = LazyOp(BinaryOps.SUB, (op3,op7,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op8,), (64, 1024, 4))
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)

  # VALIDHACKS=1 IMAGE=2 DEBUG=4 PYTHONPATH="." GPU=1 OPT=2 python3 test/external_test_gpu_ast.py TestAST.test_reduce_op
  # 164 time 27.75 ms running re_S128_4            with [128]           None            count  4 runtime 1016.06 us      2.07 GFLOPS () -> (128, 1)
  # 169 time 22.51 ms running matmul               with [4, 16, 128]    [4, 16, 16]     count  5 runtime  110.08 us     19.06 GFLOPS ('-DMATMUL',) -> (128, 1)
  def test_reduce_op(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 128, 4, 1, 1, 512, 4), views=[View((1, 1, 1, 128, 4, 1, 1, 512, 4), (0, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 512, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 128, 4, 1, 1, 512, 4), views=[View((1, 1, 1, 128, 4, 1, 1, 512, 4), (0, 0, 0, 8192, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(128, 2048, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, 1, 128, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 128, 4, 1, 1, 1, 1), views=[View((1, 1, 1, 128, 4, 1, 1, 1, 1), (512, 512, 512, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(512,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    op3 = LazyOp(UnaryOps.RELU, (op2,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op3,), (1, 128, 4))
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)

  def test_full_reduce_op(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 512), views=[View((1, 512), (512, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 1, 1, 128, 4, 1, 1, 1, 1), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 512), views=[View((1, 512), (0, 1), 0)]), hostbuf=GPUBuffer(shape=(512,), force_create=True))
    op0 = LazyOp(BinaryOps.ADD, (buf0,buf1,), None)
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 512), views=[View((1, 512), (0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([2.], dtype=np.float32)))
    op1 = LazyOp(BinaryOps.POW, (op0,buf2,), None)
    op2 = LazyOp(ReduceOps.SUM, (op1,), (1, 1))
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 1), views=[View((1, 1), (0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([0.5], dtype=np.float32)))
    op3 = LazyOp(BinaryOps.POW, (op2,buf3,), None)
    buf4 = GPUBuffer(shape=ShapeTracker(shape=(1, 1), views=[View((1, 1), (0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.e-12], dtype=np.float32)))
    op4 = LazyOp(BinaryOps.SUB, (op3,buf4,), None)
    op5 = LazyOp(UnaryOps.RELU, (op4,), None)
    buf5 = GPUBuffer(shape=ShapeTracker(shape=(1, 1), views=[View((1, 1), (0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.e-12], dtype=np.float32)))
    op6 = LazyOp(BinaryOps.ADD, (op5,buf5,), None)
    buf6 = GPUBuffer(shape=ShapeTracker(shape=(1, 1), views=[View((1, 1), (0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([3.4e+38], dtype=np.float32)))
    op7 = LazyOp(BinaryOps.SUB, (op3,buf6,), None)
    op8 = LazyOp(UnaryOps.RELU, (op7,), None)
    op9 = LazyOp(BinaryOps.SUB, (op6,op8,), None)
    op10 = LazyOp(UnaryOps.RECIPROCAL, (op9,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op10,), (1, 1))
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)
  
  def test_1239_reduce(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1239, 4, 1, 1, 64, 4), views=[View((1, 1, 1, 1239, 4, 1, 1, 64, 4), (0, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 64, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1239, 4, 1, 1, 64, 4), views=[View((1, 1, 1, 1239, 4, 1, 1, 64, 4), (0, 0, 0, 1024, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(1239, 256,
    4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, 1, 1239, 4, 1, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (1, 1, 1, 1, 4956))
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)



if __name__ == '__main__':
  unittest.main()
