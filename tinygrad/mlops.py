from typing import Tuple, Optional
from tinygrad.helpers import argsort, ShapeType
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer
import math

class Contiguous(Function):
  def forward(self, x): return x.contiguous()
  def backward(self, grad_output): return grad_output

class Cast(Function):
  def forward(self, x, dtype):
    self.input_dtype = x.dtype
    return x.cast(dtype)
  def backward(self, grad_output):
    return grad_output.cast(self.input_dtype)

# ************* unary ops *************

class Sin(Function):
  def forward(self, x: LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.unary_op(UnaryOps.SIN)
  def backward(self, grad: LazyBuffer) -> LazyBuffer:
    return self.x.const_like(math.pi / 2).binary_op(BinaryOps.SUB, self.x).unary_op(UnaryOps.SIN).binary_op(BinaryOps.MUL, grad)
# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.binary_op(BinaryOps.MAX, x.const_like(0))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    mask = self.ret.const_like(1).binary_op(BinaryOps.SUB, self.ret.binary_op(BinaryOps.CMPEQ, self.ret.const_like(0)))
    return mask.binary_op(BinaryOps.MUL, grad_output)

class Log(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.unary_op(UnaryOps.LOG)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.binary_op(BinaryOps.DIV, self.x)

class Exp(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.unary_op(UnaryOps.EXP)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.binary_op(BinaryOps.MUL, grad_output)

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x:LazyBuffer, new_shape:ShapeType) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reduce_op(ReduceOps.SUM, new_shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.EXPAND, self.input_shape)

class Max(Function):
  def forward(self, x:LazyBuffer, new_shape:ShapeType) -> LazyBuffer:
    self.x, self.ret = x, x.reduce_op(ReduceOps.MAX, new_shape)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.binary_op(BinaryOps.CMPEQ, self.ret.movement_op(MovementOps.EXPAND, self.x.shape))

    # sum of locations, averaged
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape).movement_op(MovementOps.EXPAND, self.x.shape)
    max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

    grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, self.x.shape)
    return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded)

# ************* binary ops *************

class Equal(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return x.binary_op(BinaryOps.CMPEQ, y)

class Maximum(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    self.ret = x.binary_op(BinaryOps.MAX, y)
    return self.ret

  def backward(self, grad_output):
    mask = self.y.binary_op(BinaryOps.CMPEQ, self.ret)
    eq = self.x.binary_op(BinaryOps.CMPEQ, self.y)
    splitter = eq.const_like(2).binary_op(BinaryOps.SUB, eq).binary_op(BinaryOps.DIV, eq.const_like(2))

    return grad_output.binary_op(BinaryOps.MUL, mask.const_like(1).binary_op(BinaryOps.SUB, mask).binary_op(BinaryOps.ADD, eq)).binary_op(BinaryOps.MUL, splitter) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, mask).binary_op(BinaryOps.MUL, splitter) if self.needs_input_grad[1] else None

class Add(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return x.binary_op(BinaryOps.ADD, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer):
    return x.binary_op(BinaryOps.SUB, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output.const_like(0).binary_op(BinaryOps.SUB, grad_output) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer):
    self.x, self.y = x, y
    return x.binary_op(BinaryOps.MUL, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return self.y.binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.x.binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Pow(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer):
    self.x, self.y, self.ret = x, y, x.binary_op(BinaryOps.POW, y)
    return self.ret

  def backward(self, grad_output:LazyBuffer):
    return grad_output.binary_op(BinaryOps.MUL, self.y.binary_op(BinaryOps.MUL, self.ret.binary_op(BinaryOps.DIV, self.x))) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, self.x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL, self.ret)) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.binary_op(BinaryOps.DIV, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output.binary_op(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None, \
           grad_output.const_like(0).binary_op(BinaryOps.SUB, grad_output).binary_op(BinaryOps.MUL, self.x).binary_op(BinaryOps.DIV, self.y.binary_op(BinaryOps.MUL, self.y)) if self.needs_input_grad[1] else None

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:LazyBuffer, shape:ShapeType) -> LazyBuffer:
    self.input_shape = x.shape
    return x.movement_op(MovementOps.EXPAND, shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.reduce_op(ReduceOps.SUM, self.input_shape)

class Reshape(Function):
  def forward(self, x:LazyBuffer, shape:ShapeType) -> LazyBuffer:
    self.input_shape = x.shape
    return x.movement_op(MovementOps.RESHAPE, shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.RESHAPE, self.input_shape)

class Permute(Function):
  def forward(self, x:LazyBuffer, order:Tuple[int, ...]) -> LazyBuffer:
    self.input_order = order
    return x.movement_op(MovementOps.PERMUTE, order)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.movement_op(MovementOps.PERMUTE, argsort(self.input_order))

class Pad(Function):
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    self.narg = tuple((p[0], s+p[0]) for s,p in zip(x.shape, arg))
    return x.movement_op(MovementOps.PAD, arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.movement_op(MovementOps.SHRINK, self.narg)

class Shrink(Function):
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    self.narg = tuple((p[0], s-p[1]) for s,p in zip(x.shape, arg))
    return x.movement_op(MovementOps.SHRINK, arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.movement_op(MovementOps.PAD, self.narg)

class Flip(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]):
    self.arg = tuple(-1 if i in axis else 1 for i in range(len(x.shape)))
    return x.movement_op(MovementOps.STRIDE, self.arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.movement_op(MovementOps.STRIDE, self.arg)
