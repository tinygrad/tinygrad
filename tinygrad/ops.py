import os
from enum import Enum
from typing import Union, Type, NamedTuple, Tuple, Any, List
import functools, operator
from tinygrad.shapetracker import ShapeTracker

DEBUG = int(os.getenv("DEBUG", "0"))

# these are the llops your accelerator must implement, along with toCpu
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN", "RECIPROCAL"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "FLIP", "STRIDED", "PAD", "SHRINK"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU", "CONTIGUOUS"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

# Any == Union[LazyBuffer, DeviceBuffer]
def get_buffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_buffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])

# extend this if you don't have an exec_ast function
# used in CPUBuffer and TorchBuffer
class GenericExecAST:
  @classmethod
  def exec_ast(cls, ast:LazyOp):
    srcs = [cls.exec_ast(x) if isinstance(x, LazyOp) else x for x in ast.src]
    if isinstance(ast.op, UnaryOps): ret = srcs[0].unary_op(ast.op)
    elif isinstance(ast.op, BinaryOps):
      assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
      ret = srcs[0].binary_op(ast.op, srcs[1])
    elif isinstance(ast.op, ReduceOps): ret = srcs[0].reduce_op(ast.op, ast.arg)
    elif isinstance(ast.op, MovementOps): ret = srcs[0].movement_op(ast.op, ast.arg)
    elif isinstance(ast.op, ProcessingOps): ret = srcs[0].processing_op(ast.op, srcs[1], ast.arg)
    else: raise Exception("unknown op")
    return ret

def get_lazyop_shape(ast:LazyOp) -> Tuple[int, ...]:
  srcs = [get_lazyop_shape(x) if isinstance(x, LazyOp) else x.shape for x in ast.src]
  if isinstance(ast.op, UnaryOps): return srcs[0]
  elif isinstance(ast.op, BinaryOps):
    assert srcs[0] == srcs[1], f"BinaryOp must have matching shape {srcs[0]}, {srcs[1]}"
    return srcs[0]
  elif isinstance(ast.op, ReduceOps):
    assert all(r == n or n == 1 for r,n in zip(srcs[0], ast.arg))
    return ast.arg
  elif isinstance(ast.op, MovementOps):
    return ShapeTracker(srcs[0]).movement_op(ast.op, ast.arg).shape
  elif isinstance(ast.op, ProcessingOps):
    return (ast.arg.bs, ast.arg.groups * ast.arg.rcout, ast.arg.oy, ast.arg.ox)

# assumes you are using ShapeTracker
# used in GPUBuffer, OpenCLBuffer, and LLVMBuffer
class ExplicitExecAST:
  # universal
  def unary_op(x, op:UnaryOps): return type(x)(x.shape).exec_ast(LazyOp(op=op, src=(x,)))
  def binary_op(x, op:BinaryOps, y): return type(x)(x.shape).exec_ast(LazyOp(op=op, src=(x, y)))
  def reduce_op(x, op:ReduceOps, new_shape:Tuple[int, ...]): return type(x)(new_shape).exec_ast(LazyOp(op=op, src=(x,), arg=new_shape))

  # universal for shape tracked
  def movement_op(x, op:MovementOps, arg): return type(x)(ShapeTracker(x.st).movement_op(op, arg), x)
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)
