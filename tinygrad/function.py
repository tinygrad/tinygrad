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

COEFF_SINF= [0.159154892, 5.112411827e-08, 3.626141271e-15, -2.036222915e-22,
            0.03415493667, 6.420638243e-09, 7.342738037e-17, 8.135951656e-24,
            0.03415493667, 6.420638243e-09, 7.342738037e-17, 8.135951656e-24,
            0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
            0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
            0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
            0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
            0.0009518179577, 1.342109202e-10, 1.791623576e-17, 1.518506657e-24,
            0.0009518179577, 1.342109202e-10, 1.791623576e-17, 1.518506657e-24,
            0.0004635368241, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
            0.0002193961991, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
            9.73258866e-05, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
            3.62907449e-05, 3.243700447e-12, 5.690024473e-19, 7.09405479e-26,
            5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
            5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
            5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
            1.958472239e-06, 5.152167755e-13, 1.3532163e-19, 1.92417627e-26,
            5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            2.132179588e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
            6.420638243e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
            6.420638243e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
            2.695347945e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
            8.327027956e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
            8.327027956e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
            3.670415083e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
            1.342109202e-10, 1.791623576e-17, 1.518506361e-24, 2.613904e-31,
            1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
            1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
            1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
            3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
            3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
            3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
            1.424711477e-12, 1.3532163e-19, 1.924175961e-26, 2.545416018e-33,
            5.152167755e-13, 1.3532163e-19, 1.924175961e-26, 2.545416018e-33,
            6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
            7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
            7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
            7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
            7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
            7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
            7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
            1.791623576e-17, 1.518506361e-24, 2.61390353e-31, 4.764937743e-38,
            1.791623576e-17, 1.518506361e-24, 2.61390353e-31, 4.764937743e-38,
            4.038449606e-18, -1.358545683e-25, -3.443243946e-32, 6.296048013e-40,
            4.038449606e-18, -1.358545683e-25, -3.443243946e-32, 6.296048013e-40,
            5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
            5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
            5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
            1.3532163e-19, 1.924175961e-26, 2.545415467e-33, 6.296048013e-40,
            1.3532163e-19, 1.924175961e-26, 2.545415467e-33, 6.296048013e-40,
            2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
            2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
            2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
            1.334890502e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
            6.572641438e-21, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
            0.05874381959, 1.222115387e-08, 7.693612965e-16, 1.792054435e-22,
            0.02749382704, 4.77057327e-09, 7.693612965e-16, 1.792054435e-22,
            0.01186883077, 1.045283415e-09, 3.252721926e-16, 7.332633139e-23,
            0.00405633077, 1.045283415e-09, 3.252721926e-16, 7.332633139e-23,
            0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
            0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
            0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
            0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
            0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
            2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
            2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
            2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
            1.275271279e-05, 1.183823005e-12, 1.161414894e-20, 1.291319272e-27,
            5.12331826e-06, 1.183823005e-12, 1.161414894e-20, 1.291319272e-27,
            1.308621904e-06, 2.743283031e-13, 1.161414894e-20, 1.291319272e-27,
            1.308621904e-06, 2.743283031e-13, 1.161414894e-20, 1.291319272e-27,
            3.549478151e-07, 4.695462769e-14, 1.161414894e-20, 1.291319272e-27,
            3.549478151e-07, 4.695462769e-14, 1.161414894e-20, 1.291319272e-27,
            1.165292645e-07, 1.853292503e-14, 4.837885366e-21, 1.291319272e-27,
            1.165292645e-07, 1.853292503e-14, 4.837885366e-21, 1.291319272e-27,
            5.69246339e-08, 4.322073705e-15, 1.449754789e-21, 7.962890365e-29,
            2.712231151e-08, 4.322073705e-15, 1.449754789e-21, 7.962890365e-29,
            1.222115387e-08, 7.693612965e-16, 1.792054182e-22, 2.91418027e-29,
            4.77057327e-09, 7.693612965e-16, 1.792054182e-22, 2.91418027e-29,
            1.045283415e-09, 3.252721926e-16, 7.332632508e-23, 3.898253736e-30,
            1.045283415e-09, 3.252721926e-16, 7.332632508e-23, 3.898253736e-30,
            1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
            1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
            1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
            1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
            5.575349904e-11, 6.083145782e-18, 5.344349223e-25, 1.511644828e-31,
            2.664967552e-11, -8.557475018e-19, -8.595036458e-26, -2.139883875e-32,
            1.209775682e-11, 2.61369883e-18, 5.344349223e-25, 1.511644828e-31,
            4.821800945e-12, 8.789757674e-19, 1.208447639e-25, 3.253064536e-33,
            1.183823005e-12, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
            1.183823005e-12, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
            2.743283031e-13, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
            0, 0, 0, 0]

def _div(x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.MUL, y.e(UnaryOps.RECIP))
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

def _rempif(x:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer, LazyBuffer]:
  ex = _ilogb2kf(x).e(BinaryOps.SUB, x.const(25).cast(dtypes.int32))
  k = ex.const(90 - 25)
  q = ex.e(BinaryOps.CMPNE, k).e(TernaryOps.WHERE, ex.e(BinaryOps.CMPLT, k).e(TernaryOps.WHERE, x.const(0), x.const(-64)), x.const(0)).cast(dtypes.int32)
  a = _ldexp3kf(x, q)

  ex = ex.e(BinaryOps.CMPLT, x.const(0).cast(dtypes.int32)).e(TernaryOps.WHERE, x.const(0).cast(dtypes.int32), ex)
  ex = ex.e(BinaryOps.MUL, x.const(4).cast(dtypes.int32))
  from tinygrad import Tensor
  lut = Tensor(np.array(COEFF_SINF))
  x_x, x_y = _dfmul_f2_f_f(a, lut[ex])
  di_d, di_i = _remisubf(x_x)
  q = di_i
  x_x = di_d
  x_x, x_y = _dfnormalize_f2_f2(x_x, x_y)
  y_x, y_y = _dfmul_f2_f_f(a, lut[ex.e(BinaryOps.ADD, ex.const(1))])
  x_x, x_y = _dfadd2_f2_f2_f2(x_x, x_y, y_x, y_y)
  di_d, di_i = _remisubf(x_x)
  q = q.e(BinaryOps.ADD, di_i)
  x_x = di_d
  x_x, x_y = _dfnormalize_f2_f2(x_x, x_y)
  y_x, y_y = _dfmul_f2_f2_f(lut[ex.e(BinaryOps.ADD, ex.const(2))], lut[ex.e(BinaryOps.ADD, ex.const(3))], a)
  x_x, x_y = _dfadd2_f2_f2_f2(x_x, x_y, y_x, y_y)
  x_x, x_y = _dfnormalize_f2_f2(x_x, x_y)
  x_x, x_y = _dfmul_f2_f2_f2(x_x, x_y, x_x.const(3.1415927410125732422*2), x_y.const(-8.7422776573475857731e-08*2))

  ret_dfx = _fabsfk(a).e(BinaryOps.CMPLT, a.const(0.7)).e(TernaryOps.WHERE, a, x_x)
  ret_dfy = _fabsfk(a).e(BinaryOps.CMPLT, a.const(0.7)).e(TernaryOps.WHERE, a.const(0), x_y)
  return ret_dfx, ret_dfy, q

def _xsind(d: LazyBuffer):
  assert False

# u10=true is the equivalent to Sleef_sinf_u10,
# u10=false is the equivalent to Sleef_sinf_u1.
def _xsinf(d: LazyBuffer, u10:bool = True) -> LazyBuffer:
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
  def __lv3q(x:LazyBuffer) -> LazyBuffer:
    return __lv2q(x)
    _, _, i= _rempif(x)
    return i.e(BinaryOps.AND, i.const(3)).e(BinaryOps.MUL, i.const(2)).e(BinaryOps.ADD, dfx.e(BinaryOps.CMPLT, i.const(0)).e(TernaryOps.WHERE, a.const(0), a.const(1))).e(BinaryOps.SHR, a.const(2))
      
  q = di.e(BinaryOps.CMPLT, TRIGRANGEMAX2f).e(
    TernaryOps.WHERE,
    __lv1q(d),
    di.e(BinaryOps.CMPLT, TRIGRANGEMAXf).e(TernaryOps.WHERE, __lv2q(d), __lv3q(d)) if u10 else __lv3q(d)
  )
  
  def __lv1(x:LazyBuffer) -> LazyBuffer:
    if u10:
      d = _mla(q, minus_PI_A2f, x)
      d = _mla(q, minus_PI_B2f, d)
      d = _mla(q, minus_PI_C2f, d)
      return d
    else:
      assert False
      
  def __lv2(x:LazyBuffer) -> LazyBuffer:
    if u10:
      d = _mla(q, minus_PI_A2f, x)
      d = _mla(q, minus_PI_B2f, d)
      d = _mla(q, minus_PI_C2f, d)
      d = _mla(q, minus_PI_D2f, d)
      return d
    else:
      assert False
      
  def __lv3(x:LazyBuffer) -> LazyBuffer:
    return __lv2(x)
    if u10:
      dfx, dfy, i = _rempif(x)
      return dfx.e(BinaryOps.ADD, dfy)
    else:
      assert False

  d = di.e(BinaryOps.CMPLT, TRIGRANGEMAX2f).e(
    TernaryOps.WHERE,
    __lv1(d),
    di.e(BinaryOps.CMPLT, TRIGRANGEMAXf).e(TernaryOps.WHERE, __lv2(d), __lv3(d)) if u10 else __lv3(d)
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

def __xsinf(d: LazyBuffer) -> LazyBuffer:
  M_1_PI = d.const(0.318309886183790671537767526745028724)
  minus_M_PI = d.const(-3.141592653589793238462643383279502884)
  
  q = None
  u, s, t = d, d, d
  q = _rintk(d.e(BinaryOps.MUL, M_1_PI)).cast(d.dtype)
  d = _mla(q, minus_M_PI, d)
  s = d.e(BinaryOps.MUL, d)
  u = d.const(-0.1881748176e-3)
  u = _mla(u, s, d.const(+0.8323502727e-2))
  u = _mla(u, s, d.const(-0.1666651368e+0))
  u = _mla(s.e(BinaryOps.MUL, d), u, d)
  return u
# follows: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8936472
# https://realtimecollisiondetection.net/blog/?p=9
class Sin(Function):
  def forward(self, x:LazyBuffer, fast_approx:bool=True) -> LazyBuffer:
    self.x = x
    self.fast_approx = fast_approx
    if fast_approx:
      # [WIP] Automatically set fast_approx = True if x is float32 or float64
      assert x.dtype == dtypes.float32 or x.dtype == dtypes.float64, ""
      if x.dtype == dtypes.float32:
        return _xsinf(x)
      
      if x.dtype == dtypes.float64:
        return _xsind(x)
      
      assert False
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
