"""This is where the forwards and backwards passes live."""
import math
from typing import Tuple, Optional
from tinygrad.helpers import argsort
from tinygrad.dtype import dtypes, DType, sum_acc_dtype
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import sint
import numpy as np
from tinygrad.device import Device

class Contiguous(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.contiguous()
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output

class ContiguousBackward(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.contiguous()

class Cast(Function):
  def forward(self, x:LazyBuffer, dtype:DType, bitcast:bool=False) -> LazyBuffer:
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.cast(dtype, bitcast)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.cast(self.input_dtype, self.bitcast)

# ************* unary ops *************

class Neg(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.e(UnaryOps.NEG)
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.e(UnaryOps.NEG)

class Reciprocal(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.const(1).e(BinaryOps.DIV, x)
    return self.ret
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.ret).e(BinaryOps.MUL, self.ret)

class Sin(Function):

    def _sin_grand(self, x: LazyBuffer) -> LazyBuffer:
        self.beginning_dtype = x.dtype
        if Device.DEFAULT != "METAL":
            x = x.cast(dtypes.float64)
        else:
            x = x.cast(dtypes.float32)
        self.float_precision = x.dtype

        return self._sin(x).cast(self.beginning_dtype)
        # res = self._averaging_sin(x)
        # return res.cast(self.beginning_dtype)

        # Compute normal sin if below 4e13, else use averaging
        res = self._abs(x).e(BinaryOps.CMPLT, x.const(4e13)).e(TernaryOps.WHERE, self._sin(x), self._averaging_sin(x))
        return res.cast(self.beginning_dtype)


    def _averaging_sin(self, x: LazyBuffer) -> LazyBuffer:
        # Compute 5 sines and average
        offsets = [-3,-2, -1, 0, 1, 2, 3]
        sines = [self._sin(x.e(BinaryOps.ADD, x.const(offset*2*math.pi + offset*1e-19))) for offset in offsets]
        sum = x.const(0)
        for s in sines:
            print("s: ")
            print(__import__('tinygrad').Tensor(s).numpy())
            sum = sum.e(BinaryOps.ADD, s)
        res = sum.e(BinaryOps.DIV, x.const(len(sines)))
        return res


    def _sin(self, x: LazyBuffer) -> LazyBuffer:
        # self.beginning_dtype = x.dtype
        # if Device.DEFAULT != "METAL":
        #     x = x.cast(dtypes.float64)
        # else:
        #     x = x.cast(dtypes.float32)
        # self.float_precision = x.dtype
        x = self.reduce_angle(x)
        # return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 30, x.const(1)).cast(
        # return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 50, x.const(1)).cast(
        #     self.beginning_dtype
        # )
        return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 50, x.const(1))

    def horner_taylor_sin(
        self, x: LazyBuffer, xsq: LazyBuffer, n: int, s: LazyBuffer
    ) -> LazyBuffer:
        for i in range(n, 1, -1):
            xsqdivided = xsq.e(BinaryOps.DIV, x.const((2 * i - 1) * (2 * i - 2)))
            stxsqdivided = xsqdivided.e(BinaryOps.MUL, s)
            s = s.const(1).e(BinaryOps.SUB, stxsqdivided)
        return s.e(BinaryOps.MUL, x)

    def _abs(self, x: LazyBuffer) -> LazyBuffer:
        lt0 = x.e(BinaryOps.CMPLT, x.const(0))
        return lt0.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)

    def _is_even(self, x: LazyBuffer) -> LazyBuffer:
        x = x.cast(dtypes.uint64).cast(self.float_precision)
        q = x.e(BinaryOps.DIV, x.const(2))
        q_floor = q.cast(dtypes.uint64).cast(self.float_precision)
        diff = q.e(BinaryOps.SUB, q_floor)
        # is_even = diff.e(BinaryOps.CMPLT, diff.const(1e-14))
        is_even = diff.e(BinaryOps.CMPEQ, diff.const(0))
        return is_even

    def _mod(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        def v1(x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
            return x.e(BinaryOps.SUB, x.e(BinaryOps.DIV, y).cast(dtypes.int64).cast(self.float_precision).e(BinaryOps.MUL, y),)

        def v2(x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
            q = x.e(BinaryOps.DIV, y)
            q_floor = q.cast(dtypes.uint64).cast(self.float_precision)
            diff = q.e(BinaryOps.SUB, q_floor)
            x = diff.e(BinaryOps.MUL, y)
            return x
        
        # Return v1 if x < 1e14, else return v2
        return x.e(BinaryOps.CMPLT, x.const(1e13)).e(TernaryOps.WHERE, v1(x, y), v2(x, y))
        # return x.e(BinaryOps.CMPLT, x.const(1e5)).e(TernaryOps.WHERE, v1(x, y), v2(x, y))




    def reduce_angle(self, x: LazyBuffer) -> LazyBuffer:
        lt0 = x.e(BinaryOps.CMPLT, x.const(0))
        x = self._abs(x)
        x = lt0.e(TernaryOps.WHERE, x.e(BinaryOps.ADD, x.const(math.pi)), x)

        x = self._mod(x, x.const(2 * math.pi))
        return x
    
        # # Return mod 2pi if greater than a certain big value
        # fallback = self._mod(x, x.const(2*math.pi))
        # # fallback = self._mod(x, x.const(4 * math.pi))
        orig_x = x

        # Reduce to [-pi/2, pi/2]
        beginning_dtype = x.dtype
        old_dtype = x.dtype

        lt0 = x.e(BinaryOps.CMPLT, x.const(0))
        x = self._abs(x)

        halfpi = x.const(1.5707963267948966)
        d = halfpi
        divres = x.e(BinaryOps.DIV, d)

        # Check if divres is even. If yes, subtract final value from halfpi
        is_even = self._is_even(divres)

        divres_pi = x.e(BinaryOps.DIV, x.const(math.pi))
        is_even_pi = self._is_even(divres_pi)
        # sign = is_even_pi.e(TernaryOps.WHERE, x.const(-1), x.const(1))
        sign = is_even_pi.e(TernaryOps.WHERE, x.const(1), x.const(-1))

        # If negative, add pi
        x = lt0.e(TernaryOps.WHERE, x.e(BinaryOps.ADD, x.const(math.pi)), x)
        # temp = divres.cast(dtypes.uint64).cast(old_dtype).e(BinaryOps.MUL, d)
        # x = x.e(BinaryOps.SUB, temp)
        x = divres.e(BinaryOps.SUB, divres.cast(dtypes.uint64).cast(old_dtype)).e(BinaryOps.MUL, d)

        x = is_even.e(TernaryOps.WHERE, x, halfpi.e(BinaryOps.SUB, x))

        # If sign is -1, negate
        x = sign.e(BinaryOps.CMPEQ, x.const(1)).e(TernaryOps.WHERE, x, x.e(UnaryOps.NEG))

        # ltthresh = orig_x.e(BinaryOps.CMPLT, orig_x.const(1e14))
        # res = ltthresh.e(
        #     TernaryOps.WHERE, x.cast(beginning_dtype), fallback.cast(beginning_dtype)
        # )
        res = x.cast(beginning_dtype)

        # Return nan if value is inf or -inf
        res = orig_x.e(BinaryOps.CMPEQ, orig_x.const(float('inf'))).e(TernaryOps.WHERE, x.const(math.nan), res)
        res = orig_x.e(BinaryOps.CMPEQ, orig_x.const(float('-inf'))).e(TernaryOps.WHERE, x.const(math.nan), res)
        # print("reduced angle: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        return res


    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return self._sin_grand(x)
        # return x.e(UnaryOps.SIN)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # return self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad_output)
        return self._sin_grand(self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x)).e(BinaryOps.MUL, grad_output)

# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(BinaryOps.MAX, x.const(0))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.const(0).e(BinaryOps.CMPLT, self.ret).cast(grad_output.dtype).e(BinaryOps.MUL, grad_output)

class Log(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.e(BinaryOps.DIV, self.x)

class Exp(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(BinaryOps.MUL, x.const(1/math.log(2))).e(UnaryOps.EXP2)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return self.ret.e(BinaryOps.MUL, grad_output)

class Sqrt(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(UnaryOps.SQRT)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(BinaryOps.DIV, self.ret.e(BinaryOps.MUL, self.ret.const(2)))

# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.const(1).e(BinaryOps.DIV, x.const(1).e(BinaryOps.ADD, x.e(BinaryOps.MUL, x.const(-1/math.log(2))).e(UnaryOps.EXP2)))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.e(BinaryOps.MUL, self.ret.const(1).e(BinaryOps.SUB, self.ret)).e(BinaryOps.MUL, grad_output)

# ************* binary ops *************

class Less(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.CMPLT, y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Eq(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.CMPEQ, y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Xor(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.XOR, y)

class Add(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.ADD, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.SUB, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.MUL, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.DIV, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output.e(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None, \
           grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.x).e(BinaryOps.DIV, self.y.e(BinaryOps.MUL, self.y)) if self.needs_input_grad[1] else None  # noqa: E501

# ************* ternary ops *************

class Where(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer, z:LazyBuffer) -> LazyBuffer:
    self.x = x
    return self.x.e(TernaryOps.WHERE, y, z)

  def backward(self, grad_output:LazyBuffer) -> Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]:
    return None, \
      self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
      self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, axis)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.expand(self.input_shape)

class Max(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.x, self.ret, self.axis = x, x.r(ReduceOps.MAX, axis), axis
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.e(BinaryOps.CMPEQ, self.ret.expand(self.x.shape)).cast(dtypes.float)
    div = max_is_1s.r(ReduceOps.SUM, self.axis).expand(self.x.shape)
    return max_is_1s.e(BinaryOps.DIV, div).cast(grad_output.dtype).e(BinaryOps.MUL, grad_output.expand(self.x.shape))

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.expanded_axis = tuple(i for i, (si, so) in enumerate(zip(x.shape, shape)) if si != so)
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
    self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])
    return x.stride(self.arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.stride(self.arg)
