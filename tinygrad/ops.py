# TODO: move Device to here and proxy buffer call
from enum import Enum
UnaryOps = Enum("UnaryOps", ["RELU", "EXP", "LOG", "NEG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "A", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE"])
ProcessingOps = Enum("ProcessingOps", ["CONV", "CONVT", "CONVDW"])

import os
DEBUG = int(os.getenv("PRINT_LLOPS", "0"))
class Ops:
  def unary_op(ctx, op, x, ret):
    if DEBUG: print(f"{op} : {x.shape} -> {ret.shape}")
    return ctx.op.unary_op(op, x, ret)

  def reduce_op(ctx, op, inp, ret):
    if DEBUG: print(f"{op} : {inp.shape} -> {ret.shape}")
    return ctx.op.reduce_op(op, inp, ret)

  def binary_op(ctx, op, x, y, ret):
    if DEBUG: print(f"{op} : {x.shape} + {y.shape} -> {ret.shape}")
    return ctx.op.binary_op(op, x, y, ret)

  def movement_op(ctx, op, inp, ret, arg=None):
    if DEBUG: print(f"{op} : {inp.shape} -> {ret.shape}")
    return ctx.op.movement_op(op, inp, ret, arg)

  def processing_op(ctx, op, x, y, ret, stride, groups):
    if DEBUG: print(f"{op} : {x.shape} + {y.shape} -> {ret.shape}")
    return ctx.op.processing_op(op, x, y, ret, stride, groups)