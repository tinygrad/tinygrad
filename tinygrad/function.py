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
    self.ret = x.e(UnaryOps.RECIP)
    return self.ret
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.ret).e(BinaryOps.MUL, self.ret)

# ****** helper functions for fast approxs ***********
def _rintk(d: LazyBuffer) -> LazyBuffer:  # returns int32
  assert d.dtype in (dtypes.float32, dtypes.float64)
  return_t = dtypes.int32 if d.dtype == dtypes.float32 else dtypes.int64
  return d.e(BinaryOps.ADD, d.e(BinaryOps.CMPLT, d.const(0.0)).e(TernaryOps.WHERE, d.const(-0.5), d.const(0.5))).cast(return_t)


def _mla(x: LazyBuffer, y: LazyBuffer, z: LazyBuffer) -> LazyBuffer:
  return x.e(BinaryOps.MUL, y).e(BinaryOps.ADD, z)


def _xsin(d: LazyBuffer) -> LazyBuffer:
  assert d.dtype == dtypes.float32 or d.dtype == dtypes.float64
  fp32_p = dtypes.float32 == d.dtype
  trig_range = d.const(125.0 if fp32_p else 15.0)

  m_1_pi = 0.318309886183790671537767526745028724

  # abs(d)
  di = d.e(
    BinaryOps.MUL,
    d.e(BinaryOps.CMPNE, d.const(0)).e(TernaryOps.WHERE, d.e(BinaryOps.CMPLT, d.const(0)).e(TernaryOps.WHERE, d.const(-1), d.const(1)), d.const(0)),
  )

  qdh = None
  if not fp32_p:
    qdh = d.e(BinaryOps.MUL, d.const(m_1_pi / 16777216)).cast(dtypes.int64).cast(d.dtype).e(BinaryOps.MUL, d.const(16777216.0))

  def __lv1q(x: LazyBuffer) -> LazyBuffer:
    return _rintk(x.e(BinaryOps.MUL, d.const(m_1_pi))).cast(d.dtype)

  def __lv2q(x: LazyBuffer) -> LazyBuffer:
    assert qdh is not None
    return (
      _rintk(x.e(BinaryOps.MUL, d.const(m_1_pi))).cast(d.dtype) if fp32_p else _rintk(_mla(d, d.const(m_1_pi), qdh.e(UnaryOps.NEG))).cast(d.dtype)
    )

  q: LazyBuffer = (__lv1q(d) if fp32_p else di.e(BinaryOps.CMPLT, trig_range).e(TernaryOps.WHERE, __lv1q(d), __lv2q(d)))

  def __lv1(x: LazyBuffer) -> LazyBuffer:
    if fp32_p:
      d = _mla(q, x.const(-3.1414794921875), x)
      d = _mla(q, x.const(-0.00011315941810607910156), d)
      d = _mla(q, x.const(-1.9841872589410058936e-09), d)
      return d
    else:
      d = _mla(q, x.const(-3.141592653589793116), x)
      d = _mla(q, x.const(1.2246467991473532072e-16), d)
      return d

  def __lv2(x: LazyBuffer) -> LazyBuffer:
    if fp32_p:
      d = _mla(q, x.const(-3.1414794921875), x)
      d = _mla(q, x.const(-0.00011315941810607910156), d)
      d = _mla(q, x.const(-1.9841872589410058936e-09), d)
      d = _mla(q, x.const(-1.2154201256553420762e-10), d)
      return d
    else:
      assert qdh is not None
      d = _mla(qdh, x.const(-3.1415926218032836914), x)
      d = _mla(q, x.const(-3.1415926218032836914), d)
      d = _mla(qdh, x.const(-3.1786509424591713469e-08), d)
      d = _mla(q, x.const(-3.1786509424591713469e-08), d)
      d = _mla(qdh, x.const(-1.2246467864107188502e-16), d)
      d = _mla(q, x.const(-1.2246467864107188502e-16), d)
      d = _mla(qdh.e(BinaryOps.ADD, q), x.const(-1.2736634327021899816e-24), d)
      return d

  d = di.e(BinaryOps.CMPLT, trig_range).e(TernaryOps.WHERE, __lv1(d), __lv2(d))

  s = d.e(BinaryOps.MUL, d)
  a = q.cast(dtypes.int32).e(BinaryOps.MOD, d.const(2).cast(dtypes.int32)).cast(d.dtype)
  d = d.e(BinaryOps.MUL, a.e(BinaryOps.CMPNE, d.const(0)).e(TernaryOps.WHERE, d.const(-1), d.const(1)))

  u = None
  if fp32_p:
    u = d.const(2.6083159809786593541503e-06)
    u = _mla(u, s, u.const(-0.0001981069071916863322258))
    u = _mla(u, s, u.const(0.00833307858556509017944336))
    u = _mla(u, s, u.const(-0.166666597127914428710938))
    u = _mla(s, u.e(BinaryOps.MUL, d), d)
  else:
    s2 = s.e(BinaryOps.MUL, s)
    s4 = s2.e(BinaryOps.MUL, s2)

    def __poly4(x: LazyBuffer, x2: LazyBuffer, c3, c2, c1, c0) -> LazyBuffer:
      return _mla(x2, _mla(x, d.const(c3), d.const(c2)), _mla(x, d.const(c1), d.const(c0)))

    def __poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0) -> LazyBuffer:
      return _mla(x4, __poly4(x, x2, c7, c6, c5, c4), __poly4(x, x2, c3, c2, c1, c0))

    u = __poly8(
      s,
      s2,
      s4,
      -7.97255955009037868891952e-18,
      2.81009972710863200091251e-15,
      -7.64712219118158833288484e-13,
      1.60590430605664501629054e-10,
      -2.50521083763502045810755e-08,
      2.75573192239198747630416e-06,
      -0.000198412698412696162806809,
      0.00833333333333332974823815,
    )
    u = _mla(u, s, d.const(-0.166666666666666657414808))
    u = _mla(s, u.e(BinaryOps.MUL, d), d)
  return u


class Sin(Function):
  def forward(self, x: LazyBuffer) -> LazyBuffer:
    self.x = x
    fast_approx = x.dtype == dtypes.float32 or x.dtype == dtypes.float64
    # fast_approx=False
    if fast_approx:
      assert x.dtype == dtypes.float32 or x.dtype == dtypes.float64, ""
      return _xsin(x)
    else:
      return x.e(UnaryOps.SIN)

  def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
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
    self.ret = x.e(BinaryOps.MUL, x.const(1/math.log(2))).e(UnaryOps.EXP2)
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
