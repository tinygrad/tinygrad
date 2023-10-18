#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape.shapetracker import ShapeTracker, View, ZeroView
from tinygrad.runtime.ops_gpu import GPUBuffer, CLProgram, CLCodegen
#from tinygrad.runtime.ops_metal import MetalBuffer as GPUBuffer, MetalProgram as CLProgram, MetalCodegen as CLCodegen
from tinygrad.helpers import getenv
from extra.lib_test_ast import test_ast

import platform
OSX = platform.system() == "Darwin"

def compile_and_test_ast(ast, local_size=None):
  k = CLCodegen(ast)
  prg = k.codegen().build(CLProgram)
  if local_size is not None: prg.local_size = local_size
  for i in range(5): prg(prg.lower(k.bufs))
  if getenv("TEST", 0): test_ast(k)

class TestAST(unittest.TestCase):
  def test_conv_zeroview_ast(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 3, 4), views=[View((1, 1, 3, 4), (2, 2, 2, 1), -3), ZeroView((1, 1, 1, 2), ((0, 1), (0, 1), (-1, 2), (-1, 3))), View((1, 1, 3, 4), (0, 0, 4, 1), 0)]))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 3, 4), views=[View((1, 1, 3, 4), (0, 0, 0, 0), 0)]))
    op1 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    ast = LazyOp(UnaryOps.RELU, (op1,), None)
    compile_and_test_ast(ast)

  def test_cifar_conv(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(512, 1, 128, 32, 32, 64, 3, 3), views=[View((512, 64, 34, 34), (65536, 1024, 32, 1), -33), ZeroView((512, 64, 32, 32), ((0, 512), (0, 64), (-1, 33), (-1, 33))), View((512, 1, 128, 32, 32, 64, 3, 3), (73984, 73984, 0, 34, 1, 1156, 34, 1), 0)]), hostbuf=GPUBuffer(shape=(512, 64, 32, 32), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(512, 1, 128, 32, 32, 64, 3, 3), views=[View((512, 1, 128, 32, 32, 64, 3, 3), (0, 0, 576, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(128, 64, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    ast = LazyOp(ReduceOps.SUM, (op0,), (512, 1, 128, 32, 32, 1, 1, 1))
    compile_and_test_ast(ast)

  def test_cifar_conv_backward(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(256, 1, 512, 3, 3, 512, 8, 8), views=[View((256, 512, 10, 10), (64, 16384, 8, 1), -9), ZeroView((256, 512, 8, 8), ((0, 256), (0, 512), (-1, 9), (-1, 9))), View((256, 1, 512, 3, 3, 512, 8, 8), (51200, 51200, 0, 10, 1, 100, 10, 1), 0)]), hostbuf=GPUBuffer(shape=(512, 256, 8, 8), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(256, 1, 512, 3, 3, 512, 8, 8), views=[View((256, 1, 512, 3, 3, 512, 8, 8), (0, 0, 64, 0, 0, 32768, 8, 1), 0)]), hostbuf=GPUBuffer(shape=(512, 512, 8, 8), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (256, 1, 512, 3, 3, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (256, 512, 3, 3))
    compile_and_test_ast(ast)

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
    compile_and_test_ast(ast)

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
    compile_and_test_ast(ast)

  def test_third_op_conv(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 8, 4), views=[View((1, 64, 128, 4, 4, 1, 1, 8, 4), (0, 4096, 32, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(64, 1024, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 8, 4), views=[View((1, 64, 128, 4, 4, 1, 1, 8, 4), (0, 0, 0, 128, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(4, 32, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 4, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 4, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(16,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op2,), (64, 512, 4))
    compile_and_test_ast(ast)

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
    compile_and_test_ast(ast)

  def test_alt_reduce_op(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1, 1, 128, 4, 1, 1, 512, 4), views=[View((1, 1, 1, 1, 1, 128, 4, 1, 1, 512, 4), (0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 512, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1, 1, 128, 4, 1, 1, 512, 4), views=[View((1, 1, 1, 1, 1, 128, 4, 1, 1, 512, 4), (0, 0, 0, 0, 0, 8192, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(128, 2048, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, 1, 1, 1, 128, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1, 1, 128, 4, 1, 1, 1, 1), views=[View((1, 1, 1, 1, 1, 128, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(512,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1, 1, 128, 4, 1, 1, 1, 1), views=[View((1, 1, 1, 1, 1, 128, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), force_create=True))
    op3 = LazyOp(BinaryOps.MAX, (op2,buf3,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op3,), (1, 128, 4))
    compile_and_test_ast(ast)

  # re_S32_16_36_6 is fast
  def test_1x1_36_6(self):  # 36 <- 6
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 6, 4), views=[View((1, 32, 64, 1, 1, 36, 4, 1, 1, 6, 4), (0, 1536, 24, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 384, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 6, 4), views=[View((1, 32, 64, 1, 1, 36, 4, 1, 1, 6, 4), (0, 0, 0, 0, 0, 96, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(36, 24, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(144,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), force_create=True))
    op3 = LazyOp(BinaryOps.MAX, (op2,buf3,), None)
    buf4 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    op4 = LazyOp(UnaryOps.EXP, (op2,), None)
    op5 = LazyOp(BinaryOps.SUB, (buf4,op4,), None)
    buf5 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 36, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), force_create=True))
    op6 = LazyOp(BinaryOps.MAX, (op5,buf5,), None)
    op7 = LazyOp(BinaryOps.SUB, (op3,op6,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op7,), (32, 2304, 4))
    compile_and_test_ast(ast, None if OSX else (16, 16, 4))

  # re_S32_16_6_36 is slow
  def test_1x1_6_36(self):  # 6 <- 36
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 36, 4), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 36, 4), (0, 9216, 144, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 2304, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 36, 4), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 36, 4), (0, 0, 0, 0, 0, 576, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(6, 144, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(24,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), (0, 1536, 24, 0, 0, 4, 1, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(32, 384, 4), force_create=True))
    op3 = LazyOp(BinaryOps.ADD, (op2,buf3,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op3,), (32, 384, 4))
    compile_and_test_ast(ast, (6, 16, 4))

  # re_S32_16_6_24
  def test_1x1_6_24(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 24, 4), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 24, 4), (0, 6144, 96, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 1536, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 24, 4), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 24, 4), (0, 0, 0, 0, 0, 384, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(6, 96, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1))
    #buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), views=[View((1, 32, 64, 1, 1, 6, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(24,), force_create=True))
    #op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (32, 384, 4))
    compile_and_test_ast(ast, (6, 4, 8))

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
    compile_and_test_ast(ast)

  def test_1239_reduce(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1239, 4, 1, 1, 64, 4), views=[View((1, 1, 1, 1239, 4, 1, 1, 64, 4), (0, 0, 0, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 64, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 1, 1239, 4, 1, 1, 64, 4), views=[View((1, 1, 1, 1239, 4, 1, 1, 64, 4), (0, 0, 0, 1024, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(1239, 256,
    4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, 1, 1239, 4, 1, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (1, 1, 1, 1, 4956))
    compile_and_test_ast(ast)

  def test_enet_first_conv_bs32(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 3, 225, 225), (150528, 50176, 224, 1), 0), ZeroView((8, 3, 224, 224), ((0, 8), (0, 3), (0, 225), (0, 225))), View((8, 1, 32, 112, 112, 3, 3, 3), (151875, 151875, 0, 450, 2, 50625, 225, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 3, 224, 224), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 1, 32, 112, 112, 3, 3, 3), (0, 0, 27, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (8, 1, 32, 112, 112, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (8, 32, 112, 112))
    compile_and_test_ast(ast)

  def test_enet_reduce_bs32(self):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 32, 225, 225), (50176, 150528, 224, 1), 0), ZeroView((3, 32, 224, 224), ((0, 3), (0, 32), (0, 225), (0, 225))), View((3, 1, 32, 3, 3, 32, 112, 112), (1620000, 1620000, 0, 225, 1, 50625, 450, 2), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 224, 224), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 1, 32, 3, 3, 32, 112, 112), (0, 12845056, 401408, 0, 0, 12544, 112, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 1, 32, 1, 1, 32, 112, 112), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (3, 1, 32, 3, 3, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (3, 32, 3, 3))
    compile_and_test_ast(ast)


if __name__ == '__main__':
  unittest.main()
