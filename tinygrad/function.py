"""This is where the forwards and backwards passes live."""
import math
from typing import Tuple, Optional
from tinygrad.helpers import argsort
from tinygrad.dtype import dtypes, DType, sum_acc_dtype
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer
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
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.e(UnaryOps.SIN)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad_output)

class SinApprox(Function):
  def split_high_low(self, x: LazyBuffer, factor: float) -> Tuple[LazyBuffer, LazyBuffer]:
    c = x.const(factor)
    high = x.e(BinaryOps.MUL, c).e(BinaryOps.SUB, (x.e(BinaryOps.MUL, c).e(BinaryOps.SUB, x)))
    low = x.e(BinaryOps.SUB, high)
    return high, low
  def forward(self, x: LazyBuffer) -> LazyBuffer:
    self.x = x
    xsign = x.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, x.const(-1), x.const(1))
    x = x.e(BinaryOps.MUL, xsign)

    high_factor = float(2**13 + 1)
    half_pi = x.const(math.pi / 2)
    pi = x.const(math.pi)
    two_pi = x.const(math.pi * 2)
    two_pi_high = x.const(6.2831854820251465)
    two_pi_low = x.const(-1.7484555314695172e-07)
    r_two_pi_high = x.const(0.15915493667125702)
    r_two_pi_low = x.const(6.4206382432985265e-09)
    COEFFICIENTS = [1, -0.1666666666666626, 0.0083333333332913, -0.0001984126982654, 0.0000027557316778,
                      -0.0000000250518910, 0.0000000001604841, -0.0000000000007377]
    x_high, x_low = self.split_high_low(x, high_factor)
    k_high = x_high.e(BinaryOps.MUL, r_two_pi_high)
    k_low = x_high.e(BinaryOps.MUL, r_two_pi_low).e(BinaryOps.ADD, x_low.e(BinaryOps.MUL, r_two_pi_high))
    k_low = k_low.e(BinaryOps.ADD, x_low.e(BinaryOps.MUL, r_two_pi_low))
    k = k_high.e(BinaryOps.ADD, k_low).cast(dtypes.long).cast(x.dtype)
    k_high, k_low = self.split_high_low(k, high_factor)
    mul_high = k_high.e(BinaryOps.MUL, two_pi_high)
    mul_low = k_low.e(BinaryOps.MUL, two_pi_high).e(BinaryOps.ADD, k_high.e(BinaryOps.MUL, two_pi_low))
    mul_low = mul_low.e(BinaryOps.ADD, k_low.e(BinaryOps.MUL, two_pi_low))
    rem_high = x_high.e(BinaryOps.SUB, mul_high)
    rem_low = x_low.e(BinaryOps.SUB, mul_low)
    rem = rem_high.e(BinaryOps.ADD, rem_low)
    rem = rem.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, rem.e(BinaryOps.ADD, two_pi), rem)

    rem = pi.e(BinaryOps.CMPLT, rem).e(TernaryOps.WHERE, rem.e(BinaryOps.SUB, two_pi), rem)
    rsign = rem.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, x.const(-1), x.const(1))
    absrem = rem.e(BinaryOps.MUL, rsign)
    mpi2_pi2_x = half_pi.e(BinaryOps.CMPLT, absrem).e(TernaryOps.WHERE, pi.e(BinaryOps.SUB, absrem).e(BinaryOps.MUL, rsign), rem)

    result = x.const(COEFFICIENTS[-1])
    for coeff in COEFFICIENTS[-2::-1]:
      result = result.e(BinaryOps.MUL, mpi2_pi2_x).e(BinaryOps.MUL, mpi2_pi2_x).e(BinaryOps.ADD, x.const(coeff))
    return result.e(BinaryOps.MUL, mpi2_pi2_x).e(BinaryOps.MUL, xsign)
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.forward(self.x.e(BinaryOps.ADD, self.x.const(math.pi / 2))).e(BinaryOps.MUL, grad_output)


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


class Log2Approx(Function):
  def forward(self, x: LazyBuffer) -> LazyBuffer:
    self.x = x
    band, rb, range_rel_eps = 2, 64, 1e-5
    exponents = x.const(0)
    pos_powers = range(0, rb + 1)
    neg_powers = range(-rb, 0)
    ranges = [
      (float(band ** (p-1)), float(band ** p) * (1 + range_rel_eps))
      for p in pos_powers
    ] + [
      (float(band ** (p - 1)), float(band ** p) * (1 + range_rel_eps))
      for p in neg_powers
    ]
    COEFFICIENTS = [1.4426950408889634, -0.7213475202241454, 0.4808983379134929, -0.3606744438252881, 0.2885380556670273,
                    -0.2400045744858276, 0.2164902844497397, -0.0620352892023374, 0.9800031524195777, 3.5822088209556608,
                    11.5471082527186173, 23.3693016463376679, 31.4508031904777532, 24.6578446521392927, 9.1010085120176569]
    mantissa = x
    for p in range(-rb, rb + 1):
      gridpow = float(band ** p)
      lt_power = x.e(BinaryOps.CMPLT, x.const(float(ranges[p][1])))
      gt_power = x.const(float(ranges[p][0])).e(BinaryOps.CMPLT, x)
      exponents = lt_power.e(BinaryOps.MUL, gt_power).e(TernaryOps.WHERE, x.const(p), exponents)
      mantissa = lt_power.e(BinaryOps.MUL, gt_power).e(TernaryOps.WHERE, mantissa.e(BinaryOps.DIV, x.const(gridpow)), mantissa)

    lt_power = x.e(BinaryOps.CMPLT, x.const(float(ranges[-64][0])))
    gt_zero = x.const(0).e(BinaryOps.CMPLT, x)
    exponents = lt_power.e(BinaryOps.MUL, gt_zero).e(TernaryOps.WHERE, x.const(-rb), exponents)
    mantissa = lt_power.e(TernaryOps.WHERE, x.const(1.0), mantissa)

    eq_zero = x.e(BinaryOps.CMPEQ, x.const(0))
    exponents = eq_zero.e(TernaryOps.WHERE, x.const(float('-inf')), exponents)

    lt_zero = x.e(BinaryOps.CMPLT, x.const(0))
    exponents = lt_zero.e(TernaryOps.WHERE, x.const(float('nan')), exponents)

    gt_power = x.const(float(ranges[rb][1])).e(BinaryOps.CMPLT, x)
    lt_inf = x.e(BinaryOps.CMPLT, x.const(float('inf')))
    exponents = gt_power.e(BinaryOps.MUL, lt_inf).e(TernaryOps.WHERE, x.const(rb), exponents)
    mantissa = gt_power.e(TernaryOps.WHERE, x.const(1.0), mantissa)
    eq_inf = x.e(BinaryOps.CMPEQ, x.const(float('inf')))
    exponents = eq_inf.e(TernaryOps.WHERE, x.const(float('inf')), exponents)
    eq_nan = x.e(BinaryOps.CMPEQ, x.const(float('nan')))
    exponents = eq_nan.e(TernaryOps.WHERE, x.const(float('nan')), exponents)
    mantissa = eq_nan.e(TernaryOps.WHERE, x.const(1.0), mantissa)

    x = mantissa.e(BinaryOps.SUB, x.const(1))
    result = x.const(COEFFICIENTS[-1])
    for coeff in COEFFICIENTS[-2::-1]:
      result = result.e(BinaryOps.MUL, x).e(BinaryOps.ADD, x.const(coeff))
    return result.e(BinaryOps.MUL, x).e(BinaryOps.ADD, exponents)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(BinaryOps.DIV, self.x).e(BinaryOps.DIV, grad_output.const(math.log(2)))


class Exp(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(BinaryOps.MUL, x.const(1/math.log(2))).e(UnaryOps.EXP2)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return self.ret.e(BinaryOps.MUL, grad_output)


class Exp2Approx(Function):
  def _floor(self, x:LazyBuffer) -> LazyBuffer:
    x_dtype = x.dtype
    floor = x.cast(dtypes.long).cast(x_dtype)
    adjustment = x.e(BinaryOps.CMPLT, floor).cast(x_dtype)
    return x.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, floor.e(BinaryOps.SUB, adjustment), floor)

  def forward(self, x: LazyBuffer) -> LazyBuffer:
    self.x = x
    rb = 64
    power = self._floor(x)#.cast(dtypes.long)
    power = x.e(BinaryOps.CMPEQ, x.const(float('nan'))).e(TernaryOps.WHERE, x.const(float('nan')), power)
    power = x.e(BinaryOps.CMPEQ, x.const(float('-inf'))).e(TernaryOps.WHERE, x.const(float('-inf')), power)
    power = x.e(BinaryOps.CMPEQ, x.const(float('inf'))).e(TernaryOps.WHERE, x.const(float('inf')), power)
    dx = x.e(BinaryOps.SUB, power.cast(x.dtype))
    multiplier = x.const(1)
    COEFFICIENTS = [1, 0.6931471805599453, 0.2402265069591007, 0.0555041086650001, 0.0096181291071613, 0.0013333554702035,
                    0.0001540369568210, 0.0000154434811888]
    for p in [1/2, 1/4, 1/8, 1/16, 1/32]:
      condition = x.const(p).e(BinaryOps.CMPLT, dx)
      dx = condition.e(TernaryOps.WHERE, dx.e(BinaryOps.SUB, dx.const(p)), dx)
      multiplier = condition.e(TernaryOps.WHERE, multiplier.e(BinaryOps.MUL, dx.const(2 ** p)), multiplier)

    result = dx.const(COEFFICIENTS[-1])
    for coeff in COEFFICIENTS[-2::-1]:
      result = result.e(BinaryOps.MUL, dx).e(BinaryOps.ADD, dx.const(coeff))
    result = result.e(BinaryOps.MUL, multiplier)

    for p in range(-rb, rb+1):
      condition = power.e(BinaryOps.CMPEQ, power.const(p))
      result = condition.e(TernaryOps.WHERE, result.e(BinaryOps.MUL, result.const(float(2 ** p))), result)

    lt_power = power.e(BinaryOps.CMPLT, power.const(-rb))
    gt_minf = power.const(float('-inf')).e(BinaryOps.CMPLT, power)
    result = lt_power.e(BinaryOps.MUL, gt_minf).e(TernaryOps.WHERE, result.e(BinaryOps.MUL, result.const(float(2 ** (-rb)))), result)

    eq_minf = power.e(BinaryOps.CMPEQ, power.const(float('-inf')))
    result = eq_minf.e(TernaryOps.WHERE, power.const(0), result)

    gt_power = power.const(rb).e(BinaryOps.CMPLT, power)
    lt_inf = power.e(BinaryOps.CMPLT, result.const(float('inf')))
    result = gt_power.e(BinaryOps.MUL, lt_inf).e(TernaryOps.WHERE, result.const(float(2 ** rb)), result)

    eq_inf = power.e(BinaryOps.CMPEQ, x.const(float('inf')))
    result = eq_inf.e(TernaryOps.WHERE, x.const(float('inf')), result)

    eq_nan = power.e(BinaryOps.CMPEQ, x.const(float('nan')))
    result = eq_nan.e(TernaryOps.WHERE, x.const(float('nan')), result)

    self.ret = result
    return result

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.e(BinaryOps.MUL, grad_output).e(BinaryOps.MUL, self.ret.const(math.log(2)))


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
