import os
from enum import Enum
from typing import Union, Type, NamedTuple, Tuple, Any, List
import functools, operator
from tinygrad.helpers import prod, dedup, all_same
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

# a placeholder class to extend by the exec classes
class DeviceBuffer:
  shape: Any   # should be Tuple[int, ...] but ndarray and torch.tensor have incompatible types

# extend this if you don't have an exec_ast function
# used in CPUBuffer and TorchBuffer
class GenericExecAST(DeviceBuffer):
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
      raise Exception("unknown op")
    return ret

class GlobalCounters:
  global_ops, global_mem = 0, 0

class GenericShape(GenericExecAST):
  def __init__(self, shape, flops=0): self.shape, self.flops = shape, flops
  def unary_op(self, op:UnaryOps): return GenericShape(self.shape, self.flops + prod(self.shape))
  def binary_op(self, op:BinaryOps, y): return GenericShape(self.shape, self.flops + y.flops + prod(self.shape))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return GenericShape(new_shape, self.flops + prod(self.shape))
  def movement_op(self, op:MovementOps, arg): return GenericShape(ShapeTracker(self.shape).movement_op(op, arg).shape, self.flops)
  def processing_op(self, op:ProcessingOps, w, C): return GenericShape(C.out_shape, float("nan"))  # TODO: add flops for this
def get_lazyop_info(ast:LazyOp): return GenericShape.exec_ast(ast, lambda x: GenericShape(x.shape))

# assumes you are using ShapeTracker
# used in GPUBuffer, OpenCLBuffer, and LLVMBuffer
class ExplicitExecAST(DeviceBuffer):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape

  @classmethod
  def exec_ast(cls, ast:LazyOp): raise NotImplementedError("must be implemented")

  # universal
  def unary_op(self, op:UnaryOps): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self,)))
  def binary_op(self, op:BinaryOps, y): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self, y)))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return type(self)(new_shape).exec_ast(LazyOp(op=op, src=(self,), arg=new_shape))

  # universal for shape tracked
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), self)
  def contiguous(self): return self if self.st.contiguous else self.unary_op(UnaryOps.NOOP)

# ast kernel can contain one ReduceOp with arbitrary Binary/Unary ops
class ASTKernel:
  def __init__(self, ast:LazyOp):
    self.info = get_lazyop_info(ast)
    self.bufs = dedup(get_buffers(ast))
    reduceops = [x for x in get_lazyops(ast) if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None
    self.earlybufs = dedup(get_buffers(self.reduceop)) if self.reduceop else []

    # create the buffer we are returning (as the same type as the input buffers) and add it as the first buffer
    self.ret = type(self.bufs[0])(self.info.shape)
    self.bufs = [self.ret] + self.bufs

    # check valid AST kernel
    assert all_same([x.shape for x in self.earlybufs]), "all earlybufs must have the same shape"
    assert all_same([x.shape for x in self.bufs if x not in self.earlybufs]), "all latebufs must have the same shape"
    assert all_same([len(x.shape) for x in self.bufs]), "all bufs must have the same shape size"

  def process(self):
    # get shape, strides, and offset
    # if it's a multiview buffer we take the final view
    shapes = [x.shape for x in self.bufs]
    strides = [x.st.views[-1].strides for x in self.bufs]

    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    all_ones = [all(s[i]==1 for s in shapes) for i in range(len(shapes[0]))]
    # keep at least 1 one
    if all(all_ones):
      all_ones[-1] = False
    shapes = [[s[i] for i in range(len(s)) if not all_ones[i]] for s in shapes]
    strides = [[s[i] for i in range(len(s)) if not all_ones[i]] for s in strides]

    # find first mismatch, don't reduce this
    first_reduce = -1
    for i in range(len(shapes[0])):
      if not all_same([x[i] for x in shapes]):
        first_reduce = i
        break

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*strides[j][i]) or (strides[j][i] == 0 and rets[j][-1][1] == 0))
      # more can merge than this
      can_merge = all(can_merge) and i != first_reduce
      for j in range(len(shapes)):
        if can_merge:
          rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else:
          rets[j].append((shapes[j][i], strides[j][i]))
    self.shapes, self.strides = [[y[0] for y in x] for x in rets], [[y[1] for y in x] for x in rets]

    # include the offsets (as is)
    self.offsets = [x.st.views[-1].offset for x in self.bufs]

  # this should be aware of the three parts to the shape
  #  * the input/output dimensions
  #  * the reduce dimensions
  #  * the size outputted by each kernel
  def reshape_and_permute(self, new_shape_fxn, axis):
    new_shapes, new_strides = [], []
    for shape, stride in zip(self.shapes, self.strides):
      st = ShapeTracker(tuple(shape))
      st.strided(*zip(shape, stride))
      # TODO: handle reduced shape here
      st.reshape(*new_shape_fxn(shape))
      st.permute(*axis)
      assert len(st.views) == 1
      new_shapes.append(st.shape)
      new_strides.append(st.strides)
    self.shapes, self.strides = new_shapes, new_strides

