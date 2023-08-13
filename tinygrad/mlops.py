from typing import Tuple, Optional
from tinygrad.helpers import argsort, ShapeType, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer
import math

class Contiguous(Function):
  def forward(self, x): return x.contiguous()
  def backward(self, grad_output): return grad_output

class Cast(Function):
  __slots__ = "input_dtype", "bitcast"
  def forward(self, x:LazyBuffer, dtype:DType, bitcast=False):
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.cast((dtype, bitcast))
  def backward(self, grad_output:LazyBuffer):
    return grad_output.cast((self.input_dtype, self.bitcast))

# ************* unary ops *************

class Sin(Function):
  __slots__ = "x"
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.unary_op(UnaryOps.SIN)
  def backward(self, grad:LazyBuffer) -> LazyBuffer:
    return ((math.pi / 2) - self.x).unary_op(UnaryOps.SIN) * grad

# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):
  __slots__ = "ret"
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.binary_op(BinaryOps.MAX, 0)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return (0 < self.ret) * grad_output

class Log(Function):
  __slots__ = "x"
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.unary_op(UnaryOps.LOG2) * math.log(2)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output / self.x

class Exp(Function):
  __slots__ = "ret"
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = (x * (1/math.log(2))).unary_op(UnaryOps.EXP2)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret * grad_output

class Sqrt(Function):
  __slots__ = "ret"
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.unary_op(UnaryOps.SQRT)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output / (self.ret * 2)

# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
  __slots__ = "ret"
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = 1 / (1 + (x * (-1/math.log(2))).unary_op(UnaryOps.EXP2))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return (self.ret * (1 - self.ret)) * grad_output

# ************* reduce ops *************

class Sum(Function):
  __slots__ = "input_shape"
  def forward(self, x:LazyBuffer, new_shape:ShapeType) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reduce_op(ReduceOps.SUM, new_shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.expand(self.input_shape)

class Max(Function):
  __slots__ = "x", "ret"
  def forward(self, x:LazyBuffer, new_shape:ShapeType) -> LazyBuffer:
    self.x, self.ret = x, x.reduce_op(ReduceOps.MAX, new_shape)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = 1.0 - (self.x < self.ret.expand(self.x.shape))
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape).expand(self.x.shape)
    return (max_is_1s / div) * grad_output.expand(self.x.shape)

# ************* binary ops *************

class Less(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return x < y

class Add(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return x + y

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return x - y

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           -grad_output if self.needs_input_grad[1] else None

class Mul(Function):
  __slots__ = 'x', 'y'
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x * y

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return self.y * grad_output if self.needs_input_grad[0] else None, \
           self.x * grad_output if self.needs_input_grad[1] else None

class Div(Function):
  __slots__ = 'x', 'y'
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x / y

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output / self.y if self.needs_input_grad[0] else None, \
           (-grad_output * self.x) / (self.y * self.y) if self.needs_input_grad[1] else None

# ************* ternary ops *************

class Where(Function):
  __slots__ = "x"
  def forward(self, x:LazyBuffer, y:LazyBuffer, z:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.ternary_op(TernaryOps.WHERE, y, z)

  def backward(self, grad_output:LazyBuffer):
    return None, \
           self.x.ternary_op(TernaryOps.WHERE, grad_output, 0) if self.needs_input_grad[1] else None, \
           self.x.ternary_op(TernaryOps.WHERE, 0, grad_output) if self.needs_input_grad[2] else None

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  __slots__ = 'input_shape'
  def forward(self, x:LazyBuffer, shape:ShapeType) -> LazyBuffer:
    self.input_shape = x.shape
    return x.expand(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.reduce_op(ReduceOps.SUM, self.input_shape)

class Reshape(Function):
  __slots__ = 'input_shape'
  def forward(self, x:LazyBuffer, shape:ShapeType) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_output:LazyBuffer):
    return grad_output.reshape(self.input_shape)

class Permute(Function):
  __slots__ = 'input_order'
  def forward(self, x:LazyBuffer, order:Tuple[int, ...]) -> LazyBuffer:
    self.input_order = order
    return x.permute(order)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.permute(argsort(self.input_order))

class Pad(Function):
  __slots__ = 'narg'
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    self.narg = tuple([(p[0], s+p[0]) for s,p in zip(x.shape, arg)])
    return x.pad(arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.shrink(self.narg)

class Shrink(Function):
  __slots__ = 'narg'
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    self.narg = tuple([(p[0], s-p[1]) for s,p in zip(x.shape, arg)])
    return x.shrink(arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.pad(self.narg)

class Flip(Function):
  __slots__ = 'arg'
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]):
    self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])
    return x.stride(self.arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.stride(self.arg)
