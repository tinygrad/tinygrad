from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Union, Type, NamedTuple, Tuple, Any, List, ClassVar
import functools, operator
from tinygrad.helpers import prod, shape_to_axis
from tinygrad.shape import ShapeTracker
from tinygrad.helpers import getenv

DEBUG = getenv("DEBUG", 0)

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
class UnaryOps(Enum): NOOP = auto(); NEG = auto(); RELU = auto(); EXP = auto(); LOG = auto(); GT0 = auto(); RECIPROCAL = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); FLIP = auto(); STRIDED = auto(); PAD = auto(); SHRINK = auto() # noqa: E702
class ProcessingOps(Enum): CONV = auto() # noqa: E702
class LoadOps(Enum): FROMCPU = auto(); CONTIGUOUS = auto() # noqa: E702

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
def map_buffers(real_srcs, x:LazyOp) -> LazyOp:
  if x in real_srcs: return map_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple((map_buffers(real_srcs, y) if isinstance(y, LazyOp) else real_srcs[y]) for y in x.src), x.arg)

# a placeholder class to extend by the exec classes
class DeviceBuffer:
  shape: Any   # should be Tuple[int, ...] but ndarray and torch.tensor have incompatible types
  @staticmethod
  def fromCPU(x:np.ndarray) -> DeviceBuffer: raise NotImplementedError("must be implemented")
  def toCPU(self:DeviceBuffer) -> np.ndarray: raise NotImplementedError("must be implemented")
  @classmethod
  def exec_ast(cls, ast:LazyOp): raise NotImplementedError("must be implemented")

# extend this if you don't have an exec_ast function
class GenericExecAST(DeviceBuffer):  # pylint: disable=abstract-method
  @classmethod
  def exec_ast(cls, ast:LazyOp, preprocess=lambda x: x):
    srcs = [cls.exec_ast(x, preprocess) if isinstance(x, LazyOp) else preprocess(x) for x in ast.src]
    if ast.op in UnaryOps:
      ret = srcs[0].unary_op(ast.op)
    elif ast.op in BinaryOps:
      assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
      ret = srcs[0].binary_op(ast.op, srcs[1])
    elif ast.op in ReduceOps:
      assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
      ret = srcs[0].reduce_op(ast.op, ast.arg)
    elif ast.op in MovementOps:
      ret = srcs[0].movement_op(ast.op, ast.arg)
    elif ast.op in ProcessingOps:
      ret = srcs[0].processing_op(ast.op, srcs[1], ast.arg)
    else:
      raise TypeError("unknown op")
    return ret

# used in CPUBuffer and TorchBuffer
class GenericBufExecAST(GenericExecAST):  # pylint: disable=abstract-method
  def contiguous(self): return self.unary_op(UnaryOps.NOOP)
  def unary_op(self, op): return type(self)(self.fxn_for_op[op](self.buf))
  def binary_op(self, op, y): return type(self)(self.fxn_for_op[op](self.buf, y.buf))
  def reduce_op(self, op, new_shape): return type(self)(self.fxn_for_op[op](self.buf, new_shape))
  def movement_op(self, op, arg=None): return type(self)(self.fxn_for_op[op](self.buf, arg)) if op in self.fxn_for_op else type(self)(getattr(self.buf, op.name.lower())(arg))

base_fxn_for_op = {
  UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.GT0: lambda x: operator.gt(x, 0.0), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
  BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul, BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  ReduceOps.MAX: lambda x, new_shape: (x.amax if hasattr(x, 'amax') else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
}

class GlobalCounters:
  global_ops : ClassVar[int] = 0
  global_mem : ClassVar[int] = 0
  time_sum : ClassVar[int] = 0
  # TODO: reset method

class GenericShape(GenericExecAST):  # pylint: disable=abstract-method
  def __init__(self, shape, flops=0): self.shape, self.flops = shape, flops
  def unary_op(self, op:UnaryOps): return GenericShape(self.shape, self.flops + prod(self.shape))
  def binary_op(self, op:BinaryOps, y): return GenericShape(self.shape, self.flops + y.flops + prod(self.shape))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return GenericShape(new_shape, self.flops + prod(self.shape))
  def movement_op(self, op:MovementOps, arg): return GenericShape(ShapeTracker(self.shape).movement_op(op, arg).shape, self.flops)
  # https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
  def processing_op(self, op:ProcessingOps, w, C): return GenericShape(C.out_shape, 2 * (C.bs * C.cout * C.oy * C.ox) * (C.cin * C.H * C.W))
def get_lazyop_info(ast:LazyOp): return GenericShape.exec_ast(ast, lambda x: GenericShape(x.shape))

# assumes you are using ShapeTracker
# used in GPUBuffer and LLVMBuffer
class ExplicitExecAST(DeviceBuffer):  # pylint: disable=abstract-method
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape

  # universal
  def unary_op(self, op:UnaryOps): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self,)))
  def binary_op(self, op:BinaryOps, y): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self, y)))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return type(self)(new_shape).exec_ast(LazyOp(op=op, src=(self,), arg=new_shape))

  # universal for shape tracked
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), self)

  # TODO: creating a new object is making a copy, breaking the thneed compiler
  def contiguous(self): return self if self.st.contiguous else self.unary_op(UnaryOps.NOOP)
  #def contiguous(self): return type(self)(self.shape, hostbuf=self) if self.st.contiguous else self.unary_op(UnaryOps.NOOP)
