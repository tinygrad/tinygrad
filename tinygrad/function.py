"""This is where the forwards and backwards passes live."""
import math
import numpy as np
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
    self.ret = x.e(UnaryOps.RECIP)
    return self.ret
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.ret).e(BinaryOps.MUL, self.ret)

# ****** helper functions for fast approxs ***********
def _fabsfk(d:LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32
  dint = d.cast(dtypes.int32, True)
  return dint.e(BinaryOps.AND, dint.const(0x7FFFFFFF)).cast(dtypes.float32, True)

def _fabsk(d:LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float64
  dint = d.cast(dtypes.int64, True)
  return dint.e(BinaryOps.AND, dint.const(0x7fffffffffffffff)).cast(dtypes.float64, True)

def _rintk(d:LazyBuffer) -> LazyBuffer: # returns int32
  assert d.dtype in (dtypes.float32, dtypes.float64)
  return_t = dtypes.int32 if d.dtype == dtypes.float32 else dtypes.int64
  return d.e(BinaryOps.ADD, d.e(BinaryOps.CMPLT, d.const(0.0)).e(TernaryOps.WHERE, d.const(-0.5), d.const(0.5))).cast(return_t)

def _mla(x:LazyBuffer, y:LazyBuffer, z:LazyBuffer) -> LazyBuffer:
  return x.e(BinaryOps.MUL, y).e(BinaryOps.ADD, z)

def _xisnan(d:LazyBuffer) -> LazyBuffer:
  return d.e(BinaryOps.CMPNE, d)

def _xisinf(x:LazyBuffer) -> LazyBuffer:
  return x.e(BinaryOps.CMPNE, x.const(math.inf)).e(UnaryOps.NEG).e(BinaryOps.OR, x.e(BinaryOps.CMPNE, x.const(-math.inf)).e(UnaryOps.NEG))

def _xsignbit(d: LazyBuffer) -> LazyBuffer:
  return d.e(BinaryOps.CMPLT, d.const(0.0))

def _ilogb2kf(d:LazyBuffer) -> LazyBuffer: # returns int32
  assert d.dtype == dtypes.float32
  dint = d.cast(dtypes.int32, True)
  return dint.e(BinaryOps.SHR, dint.const(23)).e(BinaryOps.AND, dint.const(255)).e(BinaryOps.SUB, dint.const(127))

def _pow2if(q:LazyBuffer) -> LazyBuffer: # returns float32
  assert q.dtype in (dtypes.int32, dtypes.int64)
  if q.dtype == dtypes.int32: return q.e(BinaryOps.ADD, q.const(127)).e(BinaryOps.SHL, q.const(23)).cast(dtypes.float32, True)
  if q.dtype == dtypes.int64: return q.e(BinaryOps.ADD, q.const(1023)).e(BinaryOps.SHL, q.const(52)).cast(dtypes.float64, True)
  
def _ldexp2kf(d:LazyBuffer, e:LazyBuffer) -> LazyBuffer: # returns float32
  assert d.dtype in (dtypes.float32, dtypes.float64)
  assert e.dtype in (dtypes.int32, dtypes.int64)
  return d.e(BinaryOps.MUL, _pow2if(e.e(BinaryOps.SHR, e.const(1)))).e(BinaryOps.MUL, _pow2if(e.e(BinaryOps.SUB, e.e(BinaryOps.SHR, e.const(1)))))

def _ldexp3kf(d:LazyBuffer, e:LazyBuffer) -> LazyBuffer:
  assert d.dtype in (dtypes.float32, dtypes.float64)
  assert e.dtype in (dtypes.int32, dtypes.int64)
  m1 = d.cast(dtypes.int32, True)
  m2 = e.e(BinaryOps.SHL, e.const(23))
  return m1.e(BinaryOps.ADD, m1).cast(d.dtype, True)

def _upperf(x:LazyBuffer) -> LazyBuffer:
  assert x.dtype == dtypes.float32
  m = m.cast(dtypes.int32)
  return m.e(BinaryOps.AND, x.const(0xfffff000)).cast(dtypes.float32)

def _mulsignf(x:LazyBuffer, y:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  assert x.dtype == dtypes.float32
  assert y.dtype == dtypes.float32
  xint = x.cast(dtypes.int32, True)
  return xint.e(BinaryOps.XOR, y.cast(dtypes.int32, True).e(BinaryOps.AND, xint.const(1 << 31))).cast(dtypes.float32, True)

def _remisubf(x:LazyBuffer) -> LazyBuffer:
  fr = x.e(BinaryOps.SUB, x.e(BinaryOps.MUL, x.const(1.0 / (1 << 10)).cast(dtypes.int32)).e(BinaryOps.MUL, x.const(1 << 10)).cast(x.dtypes))
  ret_i = x.const(7).cast(dtypes.int32).e(BinaryOps.AND, x.e(BinaryOps.CMPNE, x.const(0)).e(TernaryOps.WHERE, x.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, x.const(4), x.const(3)), x.const(0)).e(BinaryOps.ADD, fr.e(BinaryOps.MUL, fr.const(8))))
  ret_i = ret_i.e(BinaryOps.SUB, ret_i.const(3))
  ret_i = ret_i.e(BinaryOps.SHR, ret_i.const(1))
  #fr = fr - 0.25f * (int32_t)(fr * 4 + mulsignf(0.5f, x));
  fr = fr.e(BinaryOps.SUB, fr.const(0.25).e(BinaryOps.MUL, fr.e(BinaryOps.MUL, fr.const(4)).e(BinaryOps.ADD, _mulsignf(fr.const(0.5), x))))
  fr = _fabsfk(fr).e(BinaryOps.CMPNE, fr.const(0.125)).e(TernaryOps.WHERE, fr, _fabsfk(fr).e(BinaryOps.CMPLT, fr.const(0.125)).e(TernaryOps.WHERE, fr, fr.e(BinaryOps.SUB, _mulsignf(0.5, x))))

  # fr = fabsfk(fr) > 1e+10f ? 0 : fr;
  fr = fr.e(BinaryOps.CMPNE, fr.const(0.12499999254941940308)).e(TernaryOps.WHERE, x, fr)
  ret_i = fr.e(BinaryOps.CMPNE, fr.const(0.12499999254941940308)).e(TernaryOps.WHERE, ret_i.const(0), ret_i)
  return fr, ret_i

def _dfnormalize_f2_f2(t_x:LazyBuffer, t_y:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  s_x = t_x.e(BinaryOps.ADD, t_y)
  s_y = t_x.e(BinaryOps.SUB, s_x).e(BinaryOps.ADD, t_y)
  return s_x, s_y

def _dfmul_f2_f_f(x:LazyBuffer, y:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  xh = _upperf(x)
  yh = _upperf(y)
  xl = x.e(BinaryOps.SUB, xh)
  yl = y.e(BinaryOps.SUB, yh)
  rx = x.e(BinaryOps.MUL, y)
  ry = xh.e(BinaryOps.MUL, yh).e(
    BinaryOps.SUB,
    rx
  ).e(
    BinaryOps.ADD,
    xl.e(BinaryOps.MUL, yh)
  ).e(
    BinaryOps.ADD,
    xh.e(BinaryOps.MUL, yl)
  ).e(
    BinaryOps.ADD,
    xl.e(BinaryOps.MUL, yl)
  )

  return rx, ry

def _dfmul_f2_f2_f(x_x:LazyBuffer, x_y:LazyBuffer, y:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  xh = _upperf(x_x)
  yh = _upperf(y)
  xl = x_x.e(BinaryOps.SUB, xh)
  yl = y.e(BinaryOps.SUB, yh)
  rx = x_x.e(BinaryOps.MUL, y)
  ry = xh.e(BinaryOps.MUL, yh).e(
    BinaryOps.SUB,
    rx
  ).e(
    BinaryOps.ADD,
    xl.e(BinaryOps.MUL, yh)
  ).e(
    BinaryOps.ADD,
    xh.e(BinaryOps.MUL, yl)
  ).e(
    BinaryOps.ADD,
    xl.e(BinaryOps.MUL, yl)
  ).e(
    BinaryOps.ADD,
    x_y.e(BinaryOps.MUL, y)
  )

  return rx, ry

def _dfmul_f2_f2_f2(x_x:LazyBuffer, x_y:LazyBuffer, y_x:LazyBuffer, y_y:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  xh = _upperf(x_x)
  yh = _upperf(y_x)
  xl = x_x.e(BinaryOps.SUB, xh)
  yl = y_x.e(BinaryOps.SUB, yh)
  rx = x_x.e(BinaryOps.MUL, y_x)
  ry = xh.e(BinaryOps.MUL, yh).e(
    BinaryOps.SUB,
    rx
  ).e(
    BinaryOps.ADD,
    xl.e(BinaryOps.MUL, yh)
  ).e(
    BinaryOps.ADD,
    xh.e(BinaryOps.MUL, yl)
  ).e(
    BinaryOps.ADD,
    xl.e(BinaryOps.MUL, yl)
  ).e(
    BinaryOps.ADD,
    x_x.e(BinaryOps.MUL, y_y)
  ).e(
    BinaryOps.ADD,
    x_y.e(BinaryOps.MUL, y_x)
  )  

  return rx, ry

def _dfadd2_f2_f2_f2(x_x:LazyBuffer, x_y:LazyBuffer, y_x:LazyBuffer, y_y:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  r_x = x_x.e(BinaryOps.ADD, y_x)
  v = r_x.e(BinaryOps.SUB, x_x)
  r_y = x_x.e(BinaryOps.SUB, r_x.e(BinaryOps.SUB, v)).e(BinaryOps.ADD, y_x.e(BinaryOps.SUB, v))
  r_y = r_y.e(BinaryOps.ADD, x_y.e(BinaryOps.ADD, y_y))
  return r_x, r_y

def _xsin(d: LazyBuffer) -> LazyBuffer:
  TRIGRANGEMAXf = d.const(39000)
  TRIGRANGEMAX2f = d.const(125.0)
  
  PI_A2f = d.const(3.1414794921875)
  PI_B2f = d.const(0.00011315941810607910156)
  PI_C2f = d.const(1.9841872589410058936e-09)
  PI_D2f = d.const(1.2154201256553420762e-10)
  
  minus_PI_A2f = d.const(-3.1414794921875)
  minus_PI_B2f = d.const(-0.00011315941810607910156)
  minus_PI_C2f = d.const(-1.9841872589410058936e-09)
  minus_PI_D2f = d.const(-1.2154201256553420762e-10)
  M_1_PI = d.const(0.318309886183790671537767526745028724)

  u = d
  di = _rintk(d).cast(d.dtype)
  
  def __lv1q(x:LazyBuffer) -> LazyBuffer: return _rintk(x.e(BinaryOps.MUL, M_1_PI)).cast(d.dtype)
  def __lv2q(x:LazyBuffer) -> LazyBuffer: return _rintk(x.e(BinaryOps.MUL, M_1_PI)).cast(d.dtype)
        
  q = di.e(BinaryOps.CMPLT, TRIGRANGEMAX2f).e(
    TernaryOps.WHERE,
    __lv1q(d),
    __lv2q(d)
  )
  
  def __lv1(x:LazyBuffer) -> LazyBuffer:
    d = _mla(q, minus_PI_A2f, x)
    d = _mla(q, minus_PI_B2f, d)
    d = _mla(q, minus_PI_C2f, d)
    return d
      
  def __lv2(x:LazyBuffer) -> LazyBuffer:
    d = _mla(q, minus_PI_A2f, x)
    d = _mla(q, minus_PI_B2f, d)
    d = _mla(q, minus_PI_C2f, d)
    d = _mla(q, minus_PI_D2f, d)
    return d
      
  d = di.e(BinaryOps.CMPLT, TRIGRANGEMAX2f).e(
    TernaryOps.WHERE,
    __lv1(d),
    __lv2(d)
  )

  s = d.e(BinaryOps.MUL, d)
  d = d.e(BinaryOps.MUL, q.cast(dtypes.int32).e(BinaryOps.AND, d.const(1).cast(dtypes.int32)).e(BinaryOps.CMPNE, d.const(0).cast(dtypes.int32)).e(TernaryOps.WHERE, d.const(1), d.const(-1)))

  u = d.const(2.6083159809786593541503e-06)
  u = _mla(u, s, u.const(-0.0001981069071916863322258))
  u = _mla(u, s, u.const(0.00833307858556509017944336))
  u = _mla(u, s, u.const(-0.166666597127914428710938))
  u = _mla(s, u.e(BinaryOps.MUL, d), d)
  u = u.e(BinaryOps.MUL, u.const(-1.0))
  return u

def _xexp2(d: LazyBuffer) -> LazyBuffer:
  q = _rintk(d)
  s = d.e(BinaryOps.SUB, q.cast(d.dtype))
  u = d.const(0.1535920892e-3)
  u = _mla(u, s, d.const(0.1339262701e-2))
  u = _mla(u, s, d.const(0.9618384764e-2))
  u = _mla(u, s, d.const(0.5550347269e-1))
  u = _mla(u, s, d.const(0.2402264476e+0))
  u = _mla(u, s, d.const(0.6931471825e+0))
  u = _mla(u, s, d.const(0.1000000000e+1))
  u = _ldexp2kf(u, q)
  #u = d.e(BinaryOps.CMPNE, d.const(128.0)).e(TernaryOps.WHERE, d.const(math.inf), u)
  u = d.e(BinaryOps.CMPLT, d.const(128.0)).e(TernaryOps.WHERE, u, d.const(math.inf))
  u = d.e(BinaryOps.CMPLT, d.const(-150)).e(TernaryOps.WHERE, d.const(0.0), u)
  return u
  
class Sin(Function):
  def forward(self, x:LazyBuffer, fast_approx:bool=True) -> LazyBuffer:
    self.x = x
    self.fast_approx = fast_approx
    self.fast_approx = fast_approx or x.dtype == dtypes.float32 or x.dtype == dtypes.float64
    if self.fast_approx:
      assert x.dtype == dtypes.float32 or x.dtype == dtypes.float64, ""
      return _xsin(x)
    else:
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
    return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.e(BinaryOps.MUL, self.x.e(UnaryOps.RECIP))

class Exp(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = _xexp2(x.e(BinaryOps.MUL, x.const(1/math.log(2))))#_xexp2(x)#x.e(BinaryOps.MUL, x.const(1/math.log(2))).e(UnaryOps.EXP2)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return self.ret.e(BinaryOps.MUL, grad_output)

class Sqrt(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(UnaryOps.SQRT)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(BinaryOps.MUL, self.ret.e(BinaryOps.MUL, self.ret.const(2)).e(UnaryOps.RECIP))

# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.const(1).e(BinaryOps.ADD, x.e(BinaryOps.MUL, x.const(-1/math.log(2))).e(UnaryOps.EXP2)).e(UnaryOps.RECIP)
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
    return x.e(BinaryOps.MUL, y.e(UnaryOps.RECIP)) if not dtypes.is_int(x.dtype) else x.e(BinaryOps.IDIV, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output.e(BinaryOps.MUL, self.y.e(UnaryOps.RECIP)) if self.needs_input_grad[0] else None, \
           grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.x).e(BinaryOps.MUL, self.y.e(BinaryOps.MUL, self.y).e(UnaryOps.RECIP)) if self.needs_input_grad[1] else None  # noqa: E501

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
    return max_is_1s.e(BinaryOps.MUL, div.e(UnaryOps.RECIP)).cast(grad_output.dtype).e(BinaryOps.MUL, grad_output.expand(self.x.shape))

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
