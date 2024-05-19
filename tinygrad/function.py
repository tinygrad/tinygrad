"""This is where the forwards and backwards passes live."""
import math
from typing import Tuple, List, Optional
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

class OldSin(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.e(UnaryOps.SIN)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad_output)

def _taylor(x:LazyBuffer, coefficients:List[float]) -> LazyBuffer:
  current_term = x.const(1)
  result = x.const(0)
  for i, coef in enumerate(coefficients):
    if i > 0: current_term = current_term.e(BinaryOps.MUL, x)
    result = result.e(BinaryOps.ADD, current_term.e(BinaryOps.MUL, x.const(coef)))
  return result

def _get_info(x:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer, LazyBuffer]:
  if x.dtype is dtypes.double:
    b = x.cast(dtypes.ulong, bitcast=True)
    int_repr = dtypes.long
    pow_shift = 2**52
    sig_shift = 2**12
    fix = 4607182418800017408
    bias = 1023
  elif x.dtype is dtypes.float:
    b = x.cast(dtypes.uint, bitcast=True)
    int_repr = dtypes.int
    pow_shift = 2**23
    sig_shift = 2**9
    fix = 0x3F800000
    bias = 127
  else:
    raise TypeError(f'{x.dtype} not supported.')

  b = b.e(BinaryOps.MUL, b.const(2)).e(BinaryOps.DIV, b.const(2))
  bpower = b.e(BinaryOps.DIV, b.const(pow_shift)).cast(int_repr, bitcast=True)
  power = bpower.e(BinaryOps.SUB, b.const(bias).cast(int_repr))
  bsig = b.e(BinaryOps.MUL, b.const(sig_shift)).e(BinaryOps.DIV, b.const(sig_shift))
  sig = bsig.e(BinaryOps.ADD, bsig.const(fix)).cast(x.dtype, bitcast=True)
  nan = bpower.e(BinaryOps.CMPEQ, bpower.const(sig_shift/2-1))
  return (power, sig, nan)

def _floor(x:LazyBuffer) -> LazyBuffer:
  x_dtype = x.dtype
  return x.cast(dtypes.ulong).cast(x_dtype)

class Sin(Function):
  coefficients = [
    -3.927706665244766e-14,   1.0000000000071227,
    -2.1372653510812573e-10, -0.16666666415735265,
    -1.5195658172408884e-08,  0.008333387485289051,
    -1.218355377507116e-07,  -0.00019823344523361102,
    -1.7392842459887678e-07,  2.864770129172753e-06,
    -4.1207696421241214e-08, -1.7470936444039315e-08
    ]

  four_div_pi = [
    (0xa2,       0xf9836e4e, 0x441529fc),
    (0xa2f9,     0x836e4e44, 0x1529fc27),
    (0xa2f983,   0x6e4e4415, 0x29fc2757),
    (0xa2f9836e, 0x4e441529, 0xfc2757d1),
    (0xf9836e4e, 0x441529fc, 0x2757d1f5),
    (0x836e4e44, 0x1529fc27, 0x57d1f534),
    (0x6e4e4415, 0x29fc2757, 0xd1f534dd),
    (0x4e441529, 0xfc2757d1, 0xf534ddc0),
    (0x441529fc, 0x2757d1f5, 0x34ddc0db),
    (0x1529fc27, 0x57d1f534, 0xddc0db62),
    (0x29fc2757, 0xd1f534dd, 0xc0db6295),
    (0xfc2757d1, 0xf534ddc0, 0xdb629599),
    (0x2757d1f5, 0x34ddc0db, 0x6295993c),
    (0x57d1f534, 0xddc0db62, 0x95993c43),
    (0xd1f534dd, 0xc0db6295, 0x993c4390),
    (0xf534ddc0, 0xdb629599, 0x3c439041)
  ]

  pi_63 = 3.4061215800865545e-19

  def small_red_ang(self, x):
    n = _floor(x.e(BinaryOps.MUL, x.const(1 / (2 * math.pi))))
    ang = x.e(BinaryOps.SUB, n.e(BinaryOps.MUL, x.const(2 * math.pi)))
    adj_bottom = ang.e(BinaryOps.CMPLT, ang.const(math.pi))
    ang = adj_bottom.e(TernaryOps.WHERE, ang, ang.e(BinaryOps.SUB, ang.const(math.pi)))
    adj_top = ang.e(BinaryOps.CMPLT, ang.const(math.pi/2))
    ang = adj_top.e(TernaryOps.WHERE, ang, ang.const(math.pi).e(BinaryOps.SUB, ang))
    return adj_bottom.e(TernaryOps.WHERE, ang, ang.e(UnaryOps.NEG))

  def big_red_ang(self, x):
    x_dtype = x.dtype
    xi = x.cast(dtypes.uint32, bitcast=True)
    index = xi.e(BinaryOps.MUL, xi.const(2**2)).e(BinaryOps.DIV, xi.const(2**28))
    shift = xi.e(BinaryOps.MUL, xi.const(2**6)).e(BinaryOps.DIV, xi.const(2**29))
    xi = xi.e(BinaryOps.MUL, xi.const(2**9)).e(BinaryOps.DIV, xi.const(2**9)).e(BinaryOps.ADD, xi.const(0x800000))
    xi = xi.e(BinaryOps.MUL, shift.cast(dtypes.float).e(UnaryOps.EXP2).cast(dtypes.uint32))

    pi0 = xi.const(0)
    pi1 = xi.const(0)
    pi2 = xi.const(0)
    for i in range(16):
      arr = index.e(BinaryOps.CMPEQ, index.const(i))
      pi0 = arr.e(TernaryOps.WHERE, index.const(self.four_div_pi[i][0]), pi0)
      pi1 = arr.e(TernaryOps.WHERE, index.const(self.four_div_pi[i][1]), pi1)
      pi2 = arr.e(TernaryOps.WHERE, index.const(self.four_div_pi[i][2]), pi2)

    res0 = xi.e(BinaryOps.MUL, pi0)
    res1 = xi.cast(dtypes.uint64).e(BinaryOps.MUL, pi1.cast(dtypes.uint64))
    res2 = xi.cast(dtypes.uint64).e(BinaryOps.MUL, pi2.cast(dtypes.uint64))
    upper_res0 = res0.cast(dtypes.uint64).e(BinaryOps.MUL, res0.cast(dtypes.uint64).const(2**32))
    lower_res2 = res2.e(BinaryOps.DIV, res2.const(2**32))
    res0 = upper_res0.e(BinaryOps.ADD, lower_res2)
    res0 = res0.e(BinaryOps.ADD, res1)

    n = res0.e(BinaryOps.ADD, res0.const(1).e(BinaryOps.MUL, res0.const(2**61))).e(BinaryOps.DIV, res0.const(2**62))
    res0 = res0.e(BinaryOps.SUB, n.e(BinaryOps.MUL, res0.const(2**62)))
    x = res0.cast(dtypes.int64, bitcast=True).cast(dtypes.double)
    x = x.e(BinaryOps.MUL, x.const(self.pi_63))

    n_mod_4 = n.e(BinaryOps.MOD, n.const(4))
    x = n_mod_4.e(BinaryOps.CMPEQ, n.const(1)).e(TernaryOps.WHERE, x.const(math.pi/2).e(BinaryOps.SUB, x), x)
    x = n_mod_4.e(BinaryOps.CMPEQ, n.const(2)).e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)
    x = n_mod_4.e(BinaryOps.CMPEQ, n.const(3)).e(TernaryOps.WHERE, x.e(BinaryOps.SUB, x.const(math.pi/2)), x)
    return x.cast(x_dtype)

  def red_ang(self, x:LazyBuffer) -> LazyBuffer:
    signs = x.e(BinaryOps.CMPLT, x.const(0))
    x = signs.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)
    size = x.e(BinaryOps.CMPLT, x.const(2**10))
    red_ang = size.e(TernaryOps.WHERE, self.small_red_ang(x), self.big_red_ang(x))
    red_ang = signs.e(TernaryOps.WHERE, red_ang.e(UnaryOps.NEG), red_ang)
    return red_ang

  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    x_dtype = x.dtype
    x = x.cast(dtypes.float)
    reduced_angles = self.red_ang(x)
    signs = reduced_angles.e(BinaryOps.CMPLT, reduced_angles.const(0))
    reduced_angles = signs.e(TernaryOps.WHERE, reduced_angles.e(UnaryOps.NEG), reduced_angles)
    t = _taylor(reduced_angles, self.coefficients)
    _, _, nan = _get_info(x)
    t = nan.e(TernaryOps.WHERE, x.const(math.nan), t)
    zero = x.e(BinaryOps.CMPEQ, x.const(0))
    t = zero.e(TernaryOps.WHERE, x.const(0), t)
    inf = x.e(BinaryOps.CMPEQ, x.const(math.inf))
    t = inf.e(TernaryOps.WHERE, t.const(math.nan), t)
    n_inf = x.e(BinaryOps.CMPEQ, x.const(-math.inf))
    t = n_inf.e(TernaryOps.WHERE, t.const(math.nan), t)
    return signs.e(TernaryOps.WHERE, t.e(UnaryOps.NEG), t).cast(x_dtype)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    x = self.x.e(BinaryOps.ADD, self.x.const(math.pi / 2))
    return self.forward(x)

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

class Sign(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.CMPEQ, x.const(0)).e(TernaryOps.WHERE, x.const(0),
                                              x.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, x.const(-1), x.const(1)))
  # backward always return 0 to match torch
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.const(0)

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
