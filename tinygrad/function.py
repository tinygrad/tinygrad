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



  def taylor_sin(self, x:LazyBuffer) -> LazyBuffer:
    # Reduce to [0, 2pi]
    beginning_dtype = x.dtype
    if Device.DEFAULT != "METAL": x = x.cast(dtypes.float64)
    old_dtype = x.dtype
    # x = x.e(BinaryOps.SUB, x.e(BinaryOps.DIV, x.const(4*math.pi)).cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, x.const(4*math.pi)))
    d = x.const(2 * math.pi)
    lt_10p14 = x.e(BinaryOps.CMPLT, x.const(10**13))
    d = lt_10p14.e(TernaryOps.WHERE,d, x.const(4*math.pi))
    lt_10p16 = x.e(BinaryOps.CMPLT, x.const(10**16))
    d = lt_10p16.e(TernaryOps.WHERE, d, x.const(2**16*math.pi))
    # print("d: ")
    # print(__import__('tinygrad').Tensor(d).numpy())
    divres = x.e(BinaryOps.DIV, d)
    # print("divres: ")
    # print(__import__('tinygrad').Tensor(divres).numpy())
    # temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, x.const(2 * math.pi))
    temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, d)
    # print("temp: ")
    # print(__import__('tinygrad').Tensor(temp).numpy())
    # temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, divres.const(16*math.pi))
    x = x.e(BinaryOps.SUB, temp)
    # print("x: ")
    # print(__import__('tinygrad').Tensor(x).numpy())

    # no_terms = 30
    # no_terms = 16
    facts = [1.0, 0.16666666666666666, 0.008333333333333333, 0.0001984126984126984, 2.7557319223985893e-06, 2.505210838544172e-08, 1.6059043836821613e-10, 7.647163731819816e-13, 2.8114572543455206e-15, 8.22063524662433e-18, 1.9572941063391263e-20 , 3.868170170630684e-23, 6.446950284384474e-26, 9.183689863795546e-29, 1.1309962886447716e-31, 1.216125041553518e-34, 1.151633562077195e-37, 9.67759295863189e-41, 7.265460179153071e-44, 4.902469756513544e-47]
    # no_terms = 17
    no_terms = 13
    res = x.const(0)
    term = x
    xpow = x
    for i in range(no_terms):
      if i % 2 == 0:
        res = res.e(BinaryOps.ADD, term)
      else:
        res = res.e(BinaryOps.SUB, term)
      # term = term.e(BinaryOps.MUL, x).e(BinaryOps.DIV, x.const(2 * i + 2)).e(BinaryOps.MUL, x).e(BinaryOps.DIV, x.const(2 * i + 3))
      if i != no_terms - 1:
        # term = term.e(BinaryOps.MUL, x).e(BinaryOps.MUL, x).e(BinaryOps.DIV, x.const((2 * i + 2)*(2 * i + 3)))
        term = term.e(BinaryOps.MUL, x).e(BinaryOps.MUL, x).e(BinaryOps.DIV, x.const(2 * i * (2 * i + 1)))
        # xpow = xpow.e(BinaryOps.MUL, x.e(BinaryOps.MUL))
        # term = xpow.e(BinaryOps.MUL, x.const(facts[i+1]))
        # term = term.e(BinaryOps.MUL, x).e(BinaryOps.DIV, x.const((2 * i + 2)*(2 * i + 3))).e(BinaryOps.MUL, x)
    return res.cast(beginning_dtype)

  def _sin(self, x:LazyBuffer) -> LazyBuffer:
    return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 30, x.const(1)).cast(self.beginning_dtype)

  def horner_taylor_sin(self, x:LazyBuffer, xsq:LazyBuffer, n: int, s:LazyBuffer) -> LazyBuffer:
    # if n == 1:
    #   return s.e(BinaryOps.MUL, x)
    # s = s.const(1).e(BinaryOps.SUB, s.e(BinaryOps.MUL, xsq.e(BinaryOps.DIV, x.const((2*n-1)*(2*n-2)))))
    # return self.horner_taylor_sin(x, xsq, n-1, s)
    for i in range(n, 1, -1):
      # s = s.const(1).e(BinaryOps.SUB, s.e(BinaryOps.MUL, xsq.e(BinaryOps.DIV, x.const((2*n-1)*(2*n-2)))))
      # print("xsq: ")
      # print(__import__('tinygrad').Tensor(xsq).numpy())
      # print("(2*i-1) * (2*i - 2): ", (2*i-1)*(2*i-2))
      xsqdivided = xsq.e(BinaryOps.DIV, x.const((2*i-1)*(2*i-2)))
      # print("xsqdivided: ")
      # print(__import__('tinygrad').Tensor(xsqdivided).numpy())
      stxsqdivided = xsqdivided.e(BinaryOps.MUL, s)
      # print("stxsqdivided: ")
      # print(__import__('tinygrad').Tensor(stxsqdivided).numpy())
      s = s.const(1).e(BinaryOps.SUB, stxsqdivided)
      # print("s: ")
      # print(__import__('tinygrad').Tensor(s).numpy())
    return s.e(BinaryOps.MUL, x)

  def _abs(self, x:LazyBuffer) -> LazyBuffer:
    lt0 = x.e(BinaryOps.CMPLT, x.const(0))
    return lt0.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)

  def _is_even(self, x:LazyBuffer) -> LazyBuffer:
    x = self._abs(x)
    ev = x.cast(dtypes.uint64)
    print("ev: ")
    print(__import__('tinygrad').Tensor(ev).numpy())
    ev = ev.e(BinaryOps.MOD, ev.const(2))
    print("ev mod 2: ")
    print(__import__('tinygrad').Tensor(ev).numpy())
    return ev.e(BinaryOps.CMPEQ, ev.const(1))

  def _is_4k(self, x:LazyBuffer) -> LazyBuffer:
    ev = x.cast(dtypes.uint64)
    print("ev: ")
    print(__import__('tinygrad').Tensor(ev).numpy())
    ev = ev.e(BinaryOps.MOD, ev.const(4))
    print("ev mod 4: ")
    print(__import__('tinygrad').Tensor(ev).numpy())
    return ev.e(BinaryOps.CMPEQ, ev.const(0))




  def reduce_angle(self, x:LazyBuffer) -> LazyBuffer:
    # Reduce to [-pi/2, pi/2]
    beginning_dtype = x.dtype
    if Device.DEFAULT != "METAL": x = x.cast(dtypes.float64)
    else: x = x.cast(dtypes.float32)
    old_dtype = x.dtype

    lt0 = x.e(BinaryOps.CMPLT, x.const(0))
    # sign = lt0.e(TernaryOps.WHERE, x.const(-1), x.const(1))
    # print("sign: ")
    # print(__import__('tinygrad').Tensor(sign).numpy())

    # x = x.e(UnaryOps.ABS)


    x = self._abs(x)
    print("abs x: ")
    print(__import__('tinygrad').Tensor(x).numpy())


    halfpi = x.const(1.5707963267948966)
    # d = x.const(2 * math.pi)
    d = halfpi
    divres = x.e(BinaryOps.DIV, d)
    print("divres: ")
    print(__import__('tinygrad').Tensor(divres).numpy())

    # Check if divres is even. If yes, subtract final value from halfpi
    is_even = self._is_even(divres)
    # is_4k = self._is_4k(divres)
    print("is_even: ")
    print(__import__('tinygrad').Tensor(is_even).numpy())
    # x = is_even.e(TernaryOps.WHERE, halfpi.e(BinaryOps.SUB, x), x)
    # x = x.e(BinaryOps.MUL, sign)
    # x = is_even.e(TernaryOps.WHERE,x.e(UnaryOps.NEG), x)

    divres_pi = x.e(BinaryOps.DIV, x.const(math.pi))
    is_even_pi = self._is_even(divres_pi)
    sign = is_even_pi.e(TernaryOps.WHERE, x.const(-1), x.const(1))

    # If negative, add pi
    x = lt0.e(TernaryOps.WHERE, x.e(BinaryOps.ADD, x.const(math.pi)), x)
    # sign = is_4k.e(TernaryOps.WHERE, x.const(1), x.const(-1))
    # sign = is_even.e(TernaryOps.WHERE, x.const(1), x.const(-1))
    # temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, x.const(2 * math.pi))
    temp = divres.cast(dtypes.uint64).cast(old_dtype).e(BinaryOps.MUL, d)
    print("temp: ")
    print(__import__('tinygrad').Tensor(temp).numpy())
    x = x.e(BinaryOps.SUB, temp)


    x = is_even.e(TernaryOps.WHERE, halfpi.e(BinaryOps.SUB, x), x)
    # x = is_4k.e(TernaryOps.WHERE, x.e(BinaryOps.ADD, x.const(math.pi)), x)
    # print("reduced x abs: ")
    # print(__import__('tinygrad').Tensor(x).numpy())
    x = x.e(BinaryOps.MUL, sign)
    print("reduced x: ")
    print(__import__('tinygrad').Tensor(x).numpy())



    return x






  def forward(self, x:LazyBuffer) -> LazyBuffer:
    # x = x.e(UnaryOps.ANG_RED)
    # beginning_dtype = x.dtype
    self.beginning_dtype = x.dtype
    if Device.DEFAULT != "METAL": x = x.cast(dtypes.float64)
    else: x = x.cast(dtypes.float32)
    # old_dtype = x.dtype
    #
    # d = x.const(2 * math.pi)
    # lt_10p14 = x.e(BinaryOps.CMPLT, x.const(10**13))
    # d = lt_10p14.e(TernaryOps.WHERE,d, x.const(4*math.pi))
    # lt_10p16 = x.e(BinaryOps.CMPLT, x.const(10**16))
    # d = lt_10p16.e(TernaryOps.WHERE, d, x.const(2**16*math.pi))
    # # print("d: ")
    # # print(__import__('tinygrad').Tensor(d).numpy())
    # divres = x.e(BinaryOps.DIV, d)
    # # print("divres: ")
    # # print(__import__('tinygrad').Tensor(divres).numpy())
    # # temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, x.const(2 * math.pi))
    # temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, d)
    # print("temp: ")
    # print(__import__('tinygrad').Tensor(temp).numpy())
    # temp = divres.cast(dtypes.int64).cast(old_dtype).e(BinaryOps.MUL, divres.const(16*math.pi))
    # x = x.e(BinaryOps.SUB, temp)
    # print("reduced x: ")
    # print(__import__('tinygrad').Tensor(x).numpy())

    x = self.reduce_angle(x)
    self.x = x
    # return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 30, x.const(1)).cast(beginning_dtype)
    return self._sin(x)
    # return x.e(UnaryOps.SIN)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # return self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad_output)
    return self._sin(self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x)).e(BinaryOps.MUL, grad_output)

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
