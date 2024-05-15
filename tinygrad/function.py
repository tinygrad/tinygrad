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

    def _sin(self, x: LazyBuffer) -> LazyBuffer:
        self.beginning_dtype = x.dtype
        if Device.DEFAULT != "METAL":
            x = x.cast(dtypes.float64)
        else:
            x = x.cast(dtypes.float32)
        self.float_precision = x.dtype
        x = self.reduce_angle(x)
        return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 30, x.const(1)).cast(
            self.beginning_dtype
        )

    def horner_taylor_sin(
        self, x: LazyBuffer, xsq: LazyBuffer, n: int, s: LazyBuffer
    ) -> LazyBuffer:
        for i in range(n, 1, -1):
            # s = s.const(1).e(BinaryOps.SUB, s.e(BinaryOps.MUL, xsq.e(BinaryOps.DIV, x.const((2*n-1)*(2*n-2)))))
            # s = s.const(1).e(BinaryOps.SUB, xsq.e(BinaryOps.DIV, x.const((2*n-1)*(2*n-2))).e(BinaryOps.MUL, s))
            # print("xsq: ")
            # print(__import__('tinygrad').Tensor(xsq).numpy())
            # print("(2*i-1) * (2*i - 2): ", (2*i-1)*(2*i-2))
            xsqdivided = xsq.e(BinaryOps.DIV, x.const((2 * i - 1) * (2 * i - 2)))
            # print("xsqdivided: ")
            # print(__import__('tinygrad').Tensor(xsqdivided).numpy())
            stxsqdivided = xsqdivided.e(BinaryOps.MUL, s)
            # print("stxsqdivided: ")
            # print(__import__('tinygrad').Tensor(stxsqdivided).numpy())
            s = s.const(1).e(BinaryOps.SUB, stxsqdivided)
            # print("s: ")
            # print(__import__('tinygrad').Tensor(s).numpy())
        return s.e(BinaryOps.MUL, x)

    def _abs(self, x: LazyBuffer) -> LazyBuffer:
        lt0 = x.e(BinaryOps.CMPLT, x.const(0))
        return lt0.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)

    def _is_even(self, x: LazyBuffer) -> LazyBuffer:
        # x = self._abs(x)
        # ev = x.cast(dtypes.uint64)
        # ev = ev.e(BinaryOps.MOD, ev.const(2))
        # return ev.e(BinaryOps.CMPEQ, ev.const(1))
        x = x.cast(dtypes.uint64).cast(self.float_precision)
        q = x.e(BinaryOps.DIV, x.const(2))
        # print("q: ")
        # print(__import__('tinygrad').Tensor(q).numpy())
        q_floor = q.cast(dtypes.uint64).cast(self.float_precision)
        # print("q_floor: ")
        # print(__import__('tinygrad').Tensor(q_floor).numpy())
        diff = q.e(BinaryOps.SUB, q_floor)
        # print("diff: ")
        # print(__import__('tinygrad').Tensor(diff).numpy())
        is_even = diff.e(BinaryOps.CMPLT, diff.const(1e-14))
        # print("is_even: ")
        # print(__import__('tinygrad').Tensor(is_even).numpy())
        return is_even

    def _mod(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(
            BinaryOps.SUB,
            x.e(BinaryOps.DIV, y)
            .cast(dtypes.int64)
            .cast(self.float_precision)
            .e(BinaryOps.MUL, y),
        )

    def reduce_angle(self, x: LazyBuffer) -> LazyBuffer:

        # Return mod 2pi if greater than a certain big value
        # fallback = self._mod(x, x.const(2*math.pi))
        fallback = self._mod(x, x.const(4 * math.pi))
        orig_x = x

        # Reduce to [-pi/2, pi/2]
        beginning_dtype = x.dtype
        if Device.DEFAULT != "METAL":
            x = x.cast(dtypes.float64)
        else:
            x = x.cast(dtypes.float32)
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
        temp = divres.cast(dtypes.uint64).cast(old_dtype).e(BinaryOps.MUL, d)
        x = x.e(BinaryOps.SUB, temp)

        # x = is_even.e(TernaryOps.WHERE, halfpi.e(BinaryOps.SUB, x), x)
        x = is_even.e(TernaryOps.WHERE, x, halfpi.e(BinaryOps.SUB, x))
        x = x.e(BinaryOps.MUL, sign)

        # return x.cast(beginning_dtype)
        # 1486116864
        # 0000000000
        # 69800000000000
        # 100000000000000.0
        # ltthresh = orig_x.e(BinaryOps.CMPLT, orig_x.const(69305000000000.0))
        ltthresh = orig_x.e(BinaryOps.CMPLT, orig_x.const(1e14))
        res = ltthresh.e(
            TernaryOps.WHERE, x.cast(beginning_dtype), fallback.cast(beginning_dtype)
        )

        # Return nan if value is infinity
        is_inf = orig_x.e(BinaryOps.CMPEQ, orig_x.const(math.inf))
        res = is_inf.e(TernaryOps.WHERE, x.const(math.nan), res)
        return res


    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return self._sin(x)
        # return x.e(UnaryOps.SIN)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # return self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad_output)
        return self._sin(self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x)).e(
            BinaryOps.MUL, grad_output
        )

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
