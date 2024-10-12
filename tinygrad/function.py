"""This is where the forwards and backwards passes live."""
import math
from typing import Tuple, Optional
from tinygrad.helpers import argsort
from tinygrad.dtype import dtypes, DType, sum_acc_dtype
from tinygrad.ops import ReduceOps, resolve
from tinygrad.tensor import Function
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.shape.symbolic import sint

class Contiguous(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.contiguous()
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output

class ContiguousBackward(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.contiguous()

class Cast(Function):
  def forward(self, x:LazyBuffer, dtype:DType, bitcast:bool=False) -> LazyBuffer:
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.bitcast(dtype) if self.bitcast else x.cast(dtype)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.bitcast(self.input_dtype) if self.bitcast else grad_output.cast(self.input_dtype)

# ************* unary ops *************

class Reciprocal(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.recip()
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return -grad_output * self.ret * self.ret

class Sin(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.sin()

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return (math.pi/2 - self.x).sin() * grad_output

# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.max(0)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return self.ret.gt(0).cast(grad_output.dtype) * grad_output

class Log(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.log2() * math.log(2)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output / self.x

class Exp(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = (x * (1/math.log(2))).exp2()
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return self.ret * grad_output

class Sqrt(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.sqrt()
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output / (self.ret*2)

# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = (1 + (x * (-1/math.log(2))).exp2()).recip()
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return (self.ret * (1 - self.ret)) * grad_output

class Sign(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.ne(0).where(x.lt(0).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  # backward always return 0 to match torch
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.const_like(0)

# ************* binary ops *************

class Less(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.lt(y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Neq(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.ne(y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Xor(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x^y

class BitwiseAnd(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x&y

class BitwiseOr(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x|y

class Threefry(Function):
  def forward(self, x:LazyBuffer, seed:LazyBuffer) -> LazyBuffer: return x.threefry(seed)

class Add(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x+y

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x * y

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return (self.y * grad_output) if self.needs_input_grad[0] else None, \
           (self.x * grad_output) if self.needs_input_grad[1] else None

class IDiv(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x // y

# ************* ternary ops *************

class Where(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer, z:LazyBuffer) -> LazyBuffer:
    self.x = x
    return self.x.where(y, z)

  def backward(self, grad_output:LazyBuffer) -> Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]:
    return None, \
      self.x.where(grad_output, grad_output.const_like(0)) if self.needs_input_grad[1] else None, \
      self.x.where(grad_output.const_like(0), grad_output) if self.needs_input_grad[2] else None

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, axis)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.expand(self.input_shape)

class Prod(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.x, self.ret = x, x.r(ReduceOps.PROD, axis)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return (grad_output * self.ret).expand(self.x.shape) / self.x

class Max(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.x, self.ret, self.axis = x, x.r(ReduceOps.MAX, axis), axis
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.ne(self.ret.expand(self.x.shape)).ne(self.x.const_like(1).cast(dtypes.bool)).cast(grad_output.dtype)
    div = max_is_1s.r(ReduceOps.SUM, self.axis).expand(self.x.shape)
    return (max_is_1s/div) * grad_output.expand(self.x.shape)

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.expanded_axis = tuple(i for i, (si, so) in enumerate(zip(x.shape, shape)) if resolve(si != so))
    return x.expand(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.cast(sum_acc_dtype(grad_output.dtype)).r(ReduceOps.SUM, self.expanded_axis).cast(grad_output.dtype)

class Reshape(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.reshape(self.input_shape)

class Permute(Function):
  def forward(self, x:LazyBuffer, order:Tuple[int, ...]) -> LazyBuffer:
    self.input_order = order
    return x.permute(order)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.permute(argsort(self.input_order))

class Pad(Function):
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    self.narg = tuple([(p[0], s+p[0]) for s,p in zip(x.shape, arg)])
    return x.pad(arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.shrink(self.narg)

class Shrink(Function):
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer:
    self.narg = tuple([(p[0], s-p[1]) for s,p in zip(x.shape, arg)])
    return x.shrink(arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.pad(self.narg)

class Flip(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.arg = tuple([-1 if i in axis else 1 for i in range(len(x.shape))])
    return x.stride(self.arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.stride(self.arg)
