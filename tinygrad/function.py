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
    return Log._forward(x)

  @staticmethod
  def _forward(x:LazyBuffer) -> LazyBuffer:
    return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.e(BinaryOps.DIV, self.x)

def pow2if(q:LazyBuffer) -> LazyBuffer: # returns float32
  assert q.dtype == dtypes.int32
  return q.e(BinaryOps.ADD, q.const(127)).e(BinaryOps.SHL, q.const(23)).cast(dtypes.float32, True)

def ldexp2kf(d:LazyBuffer, e:LazyBuffer) -> LazyBuffer: # returns float32
  assert d.dtype == dtypes.float32
  assert e.dtype == dtypes.int32
  return d.e(BinaryOps.MUL, pow2if(e.e(BinaryOps.SHR, e.const(1)))).e(BinaryOps.MUL, pow2if(e.e(BinaryOps.SUB, e.e(BinaryOps.SHR, e.const(1)))))

def rintfk(d:LazyBuffer) -> LazyBuffer: # returns int32
  assert d.dtype == dtypes.float32
  return d.e(BinaryOps.ADD, d.e(BinaryOps.CMPLT, d.const(0.0)).e(TernaryOps.WHERE, d.const(-0.5), d.const(0.5))).cast(dtypes.int32)

class Exp2(Function): # 3.5 ULP maximum error
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    q = rintfk(x)
    s = x.e(BinaryOps.SUB, q.cast(x.dtype))

    u = x.const(+0.1535920892e-3)
    # TODO: collapse in a loop
    u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x.const(+0.1339262701e-2))
    u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x.const(+0.9618384764e-2))
    u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x.const(+0.5550392344e-1))
    u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x.const(+0.2402265069e+0))
    u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x.const(+0.6931471825e+0))
    u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x.const(+0.1000000000e+1))

    u = ldexp2kf(u, q)

    u = x.e(BinaryOps.CMPLT, x.const(128.0)).e(TernaryOps.WHERE, u, x.const(math.inf))
    u = x.e(BinaryOps.CMPLT, x.const(-150.0)).e(TernaryOps.WHERE, x.const(0.0), u)

    self.ret = u
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.e(BinaryOps.MUL, self.ret.const(math.log(2))).e(BinaryOps.MUL, grad_output)

def ilogb2kf(d:LazyBuffer) -> LazyBuffer: # returns int32
  assert d.dtype == dtypes.float32
  dint = d.cast(dtypes.int32, True)
  return dint.e(BinaryOps.SHR, dint.const(23)).e(BinaryOps.AND, dint.const(255)).e(BinaryOps.SUB, dint.const(127))

def logk3f(d:LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32
  o = d.e(BinaryOps.CMPLT, d.const(1.17549435e-38))
  d = d.e(BinaryOps.MUL, o.e(TernaryOps.WHERE, d.const(1.8446744073709552e+19), d.const(1.0)))
  e = ilogb2kf(d.e(BinaryOps.MUL, d.const(1.0/0.75)))
  m = ldexp2kf(d, e.e(UnaryOps.NEG))
  e = o.e(TernaryOps.WHERE, e.e(BinaryOps.SUB, e.const(64)), e)
  x = m.e(BinaryOps.SUB, m.const(1.0)).e(BinaryOps.DIV, m.e(BinaryOps.ADD, m.const(1.0)))
  x2 = x.e(BinaryOps.MUL, x)

  t = d.const(0.2392828464508056640625)
  t = t.e(BinaryOps.MUL, x2).e(BinaryOps.ADD, d.const(0.28518211841583251953125))
  t = t.e(BinaryOps.MUL, x2).e(BinaryOps.ADD, d.const(0.400005877017974853515625))
  t = t.e(BinaryOps.MUL, x2).e(BinaryOps.ADD, d.const(0.666666686534881591796875))
  t = t.e(BinaryOps.MUL, x2).e(BinaryOps.ADD, d.const(2.0))

  x = x.e(BinaryOps.MUL, t).e(BinaryOps.ADD, x.const(0.693147180559945286226764).e(BinaryOps.MUL, e.cast(dtypes.float32)))
  return x

def expk3f(d:LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32
  q = rintfk(d.e(BinaryOps.MUL, d.const(1.4426950408889634)))
  qf = q.cast(d.dtype)
  s = qf.e(BinaryOps.MUL, d.const(-0.693145751953125)).e(BinaryOps.ADD, d)
  s = qf.e(BinaryOps.MUL, d.const(-1.428606765330187045e-06)).e(BinaryOps.ADD, s)

  u = d.const(0.000198527617612853646278381)
  u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, d.const(0.00139304355252534151077271))
  u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, d.const(0.00833336077630519866943359))
  u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, d.const(0.0416664853692054748535156))
  u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, d.const(0.166666671633720397949219))
  u = u.e(BinaryOps.MUL, s).e(BinaryOps.ADD, d.const(0.5))
  u = u.e(BinaryOps.MUL, s.e(BinaryOps.MUL, s)).e(BinaryOps.ADD, s.e(BinaryOps.ADD, d.const(1.0)))
  u = ldexp2kf(u, q)
  u = d.e(BinaryOps.CMPLT, d.const(-104.0)).e(TernaryOps.WHERE, u.const(0.0), u)
  return u

def fabsfk(d:LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32
  dint = d.cast(dtypes.int32, True)
  return dint.e(BinaryOps.AND, dint.const(0x7FFFFFFF)).cast(dtypes.float32, True)
  # return d.e(BinaryOps.CMPLT, d.const(0.0)).e(TernaryOps.WHERE, d.e(UnaryOps.NEG), d)

def xisnanf(d:LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32
  return d.e(BinaryOps.CMPNE, d)

def xisinff(x:LazyBuffer) -> LazyBuffer:
  assert x.dtype == dtypes.float32
  return x.e(BinaryOps.CMPNE, x.const(math.inf)).e(UnaryOps.NEG).e(BinaryOps.OR, x.e(BinaryOps.CMPNE, x.const(-math.inf)).e(UnaryOps.NEG))

def xsignbitf(d: LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32
  return d.e(BinaryOps.CMPLT, d.const(0.0))

def mulsignf(x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
  assert x.dtype == dtypes.float32
  assert y.dtype == dtypes.float32
  xint = x.cast(dtypes.int32, True)
  return xint.e(BinaryOps.XOR, y.cast(dtypes.int32, True).e(BinaryOps.AND, xint.const(1 << 31))).cast(dtypes.float32, True)

class Pow(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    self.ret = Pow._forward(x, y)
    return self.ret

  @staticmethod
  def _forward(x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    x_eq_zero = x.e(BinaryOps.CMPNE, x.const(0.0)).e(UnaryOps.NEG)
    x_eq_one = x.e(BinaryOps.CMPNE, x.const(1.0)).e(UnaryOps.NEG)
    y_eq_zero = y.e(BinaryOps.CMPNE, y.const(0.0)).e(UnaryOps.NEG)
    result = expk3f(logk3f(fabsfk(x)).e(BinaryOps.MUL, y))
    yint = y.cast(dtypes.int32)
    yiswhole = y.e(BinaryOps.CMPNE, yint.cast(y.dtype)).e(UnaryOps.NEG)
    yisodd = yint.e(BinaryOps.AND, yint.const(1)).e(BinaryOps.CMPNE, yint.const(0)).e(BinaryOps.AND, yiswhole)

    result = xisnanf(result).e(TernaryOps.WHERE, result.const(math.inf), result)
    yisoddexpr = yiswhole.e(TernaryOps.WHERE, yisodd.e(TernaryOps.WHERE, x.const(-1.0), x.const(1.0)), x.const(math.nan))
    result = result.e(BinaryOps.MUL, x.e(BinaryOps.CMPLT, x.const(0.0)).e(TernaryOps.WHERE, yisoddexpr, x.const(1.0)))
    efx = mulsignf(fabsfk(x).e(BinaryOps.SUB, x.const(1.0)), y)
    efxexpr = efx.e(BinaryOps.CMPNE, efx.const(0.0)).e(TernaryOps.WHERE, x.const(math.inf), x.const(1.0))
    result = xisinff(y).e(TernaryOps.WHERE, efx.e(BinaryOps.CMPLT, efx.const(0.0)).e(TernaryOps.WHERE, x.const(0.0), efxexpr), result)
    mulsign0 = xsignbitf(y).e(BinaryOps.XOR, x_eq_zero).e(TernaryOps.WHERE, x.const(0.0), x.const(math.inf))
    mulsign1 = yisodd.e(TernaryOps.WHERE, x, x.const(1.0))
    result = xisinff(x).e(BinaryOps.OR, x_eq_zero).e(TernaryOps.WHERE, mulsignf(mulsign0, mulsign1), result)
    result = xisnanf(x).e(BinaryOps.OR, xisnanf(y)).e(TernaryOps.WHERE, x.const(math.nan), result)
    result = y_eq_zero.e(BinaryOps.OR, x_eq_one).e(TernaryOps.WHERE, x.const(1.0), result)
    return result

  def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    # Gradient with respect to the base x
    grad_x = (grad_output.e(BinaryOps.MUL, self.y).e(BinaryOps.MUL, Pow._forward(self.x, self.y.e(BinaryOps.SUB, self.y.const(1))))) \
              if self.needs_input_grad[0] else None

    # Gradient with respect to the exponent y
    grad_y = (grad_output.e(BinaryOps.MUL, Pow._forward(self.x, self.y)).e(BinaryOps.MUL, Log._forward(self.x))) \
              if self.needs_input_grad[1] else None

    return grad_x, grad_y

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

class Sign(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.CMPNE, x.const(0)).e(
      TernaryOps.WHERE, x.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, x.const(-1), x.const(1)), x.const(0))
  # backward always return 0 to match torch
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.const(0)

# ************* binary ops *************

class Less(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.CMPLT, y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Neq(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.CMPNE, y)
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
    max_is_1s = self.x.const(1.0).cast(dtypes.float).e(BinaryOps.SUB, self.x.e(BinaryOps.CMPNE, self.ret.expand(self.x.shape)).cast(dtypes.float))
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
