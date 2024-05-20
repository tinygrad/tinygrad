"""This is where the forwards and backwards passes live."""

import math
from typing import Tuple, Optional, List
from tinygrad.helpers import argsort
from tinygrad.dtype import dtypes, DType, sum_acc_dtype
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import sint
import numpy as np
from tinygrad.device import Device


class Contiguous(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.contiguous()

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output


class ContiguousBackward(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.contiguous()


class Cast(Function):
    def forward(self, x: LazyBuffer, dtype: DType, bitcast: bool = False) -> LazyBuffer:
        self.input_dtype, self.bitcast = x.dtype, bitcast
        return x.cast(dtype, bitcast)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.cast(self.input_dtype, self.bitcast)


# ************* unary ops *************


class Neg(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.e(UnaryOps.NEG)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(UnaryOps.NEG)


class Reciprocal(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.const(1).e(BinaryOps.DIV, x)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return (
            grad_output.e(UnaryOps.NEG)
            .e(BinaryOps.MUL, self.ret)
            .e(BinaryOps.MUL, self.ret)
        )

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
        n = _floor(x.e(BinaryOps.DIV, x.const(2 * math.pi)))
        ang = x.e(BinaryOps.SUB, n.e(BinaryOps.MUL, x.const(2 * math.pi)))
        adj_bottom = ang.e(BinaryOps.CMPLT, ang.const(math.pi))
        ang = adj_bottom.e(TernaryOps.WHERE, ang, ang.e(BinaryOps.SUB, ang.const(math.pi)))
        adj_top = ang.e(BinaryOps.CMPLT, ang.const(math.pi/2))
        ang = adj_top.e(TernaryOps.WHERE, ang, ang.const(math.pi).e(BinaryOps.SUB, ang))
        return adj_bottom.e(TernaryOps.WHERE, ang, ang.e(UnaryOps.NEG))

    def big_red_ang(self, x):
        x_dtype = x.dtype
        x = x.cast(dtypes.float)
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
        if self.device == "METAL":
          red_ang = self.small_red_ang(x)
        else:
          red_ang = size.e(TernaryOps.WHERE, self.small_red_ang(x), self.big_red_ang(x))
        red_ang = signs.e(TernaryOps.WHERE, red_ang.e(UnaryOps.NEG), red_ang)
        return red_ang
    def _corr_coeff(self, x: LazyBuffer) -> LazyBuffer:
        # 1e13 - 2e13 - no correction
        # 2e13 - 5e13 - no correction
        # 5e13 - 7e13 - -0.015 correction
        # 7e13 - 1e14 - no correction
        # 1e14 - 2.2e14 - -0.009 correction
        # 2.2e14 - 2.7e14 - -0.005 correction
        # 2.7e14 - 5.6e14 - -0.03 correction
        # 5.6e14 - 8.8e14 - -0.04 correction
        # 8.8e14 - 1.1e15 - -0.06 correction
        # 1.1e15 - 1.7e15 - -0.01 correction
        # 1.7e15 - 2.2e15 - 0.01 correction
        # 2.2e15 - 3.5e15 - -0.1 correction
        # 3.5e15 - 4.5e15 - -0.2 correction
        # 4.5e15 - 7e15 - -0.4 correction
        # 7e15 - 9e15 - -0.15 correction
        # 9e15 - 1.36e16 - -0.07 correction

        # print("X: ")
        # print(__import__('tinygrad').Tensor(x).numpy())
        r = x.e(BinaryOps.CMPLT, x.const(5e13)).e(
            TernaryOps.WHERE, x.const(0), x.const(-0.015)
        )
        r = x.e(BinaryOps.CMPLT, x.const(6.9e13)).e(
            TernaryOps.WHERE, r, x.const(-0.005)
        )
        r = x.e(BinaryOps.CMPLT, x.const(1e14)).e(TernaryOps.WHERE, r, x.const(-0.009))
        r = x.e(BinaryOps.CMPLT, x.const(2.2e14)).e(
            TernaryOps.WHERE, r, x.const(-0.005)
        )
        r = x.e(BinaryOps.CMPLT, x.const(2.7e14)).e(TernaryOps.WHERE, r, x.const(-0.03))
        r = x.e(BinaryOps.CMPLT, x.const(5.6e14)).e(TernaryOps.WHERE, r, x.const(-0.04))
        r = x.e(BinaryOps.CMPLT, x.const(8.8e14)).e(TernaryOps.WHERE, r, x.const(-0.06))
        r = x.e(BinaryOps.CMPLT, x.const(1.1e15)).e(TernaryOps.WHERE, r, x.const(-0.01))
        r = x.e(BinaryOps.CMPLT, x.const(1.7e15)).e(TernaryOps.WHERE, r, x.const(0.01))
        r = x.e(BinaryOps.CMPLT, x.const(2.2e15)).e(TernaryOps.WHERE, r, x.const(-0.1))
        r = x.e(BinaryOps.CMPLT, x.const(3.5e15)).e(TernaryOps.WHERE, r, x.const(-0.2))
        r = x.e(BinaryOps.CMPLT, x.const(4.5e15)).e(TernaryOps.WHERE, r, x.const(-0.4))
        r = x.e(BinaryOps.CMPLT, x.const(7e15)).e(TernaryOps.WHERE, r, x.const(-0.15))
        r = x.e(BinaryOps.CMPLT, x.const(9e15)).e(TernaryOps.WHERE, r, x.const(-0.07))

        return r

    def _sin_grand(self, x: LazyBuffer) -> LazyBuffer:

        self.beginning_dtype = x.dtype
        if x.dtype not in (dtypes.double, dtypes.float): x = x.cast(dtypes.float32)
        _, _, nan = _get_info(x)
        # print(self.beginning_dtype)
        if Device.DEFAULT != "METAL":
            x = x.cast(dtypes.float64)
        else:
            x = x.cast(dtypes.float32)
        self.float_precision = x.dtype
        # print(x.dtype)
        # xsign = x.e(BinaryOps.CMPLT, x.const(0)).e(
        #     TernaryOps.WHERE, x.const(-1), x.const(1)
        # )

        # Compute normal sin if below 4e13, else use averaging
        # res = (
        #     self._abs(x)
        #     .e(BinaryOps.CMPLT, x.const(1e13))
        #     .e(TernaryOps.WHERE, self._sin(x), self._averaging_sin(x))
        # )
        # return self._sin(x).cast(self.beginning_dtype)
        # return self._averaging_sin(x)#.cast(self.beginning_dtype)
        # print(x.dtype)
        # res = self._averaging_sin(x)
        res = self._sin(x)#.cast(self.beginning_dtype)
        # _, _, nan = _get_info(x)
        res = nan.e(TernaryOps.WHERE, x.const(math.nan).cast(self.float_precision), res)

        # print(self.float_precision)
        pinf = x.const(float("inf")).cast(self.float_precision)
        # print(pinf.dtype)
        ninf = x.const(float("-inf")).cast(self.float_precision)
        # nan = x.const(float("nan")).cast(self.float_precision)
        # print(__import__('tinygrad').Tensor(nan).numpy())
        # print(ninf.dtype)
        # print(x.dtype)

        # res = x.e(BinaryOps.CMPEQ, nan).e(
        #     TernaryOps.WHERE, x.const(math.nan), res
        # )
        res = x.e(BinaryOps.CMPEQ, pinf).e(
            TernaryOps.WHERE, x.const(math.nan), res
        )
        res = x.e(BinaryOps.CMPEQ, ninf).e(
            TernaryOps.WHERE, x.const(math.nan), res
        )
        return res.cast(self.beginning_dtype)
        # print(res.dtype)
        cos = self._averaging_sin(x.e(BinaryOps.ADD, x.const(math.pi / 2)))
        # correction = cos.e(BinaryOps.MUL, cos.const(-0.005))
        # print(cos.dtype)
        correction = cos.e(BinaryOps.MUL, self._corr_coeff(x))
        # print("CORR COEFF: ")
        # print(__import__('tinygrad').Tensor(self._corr_coeff(x)).numpy())
        # print("CORRECTION: ")
        # print(__import__('tinygrad').Tensor(correction).numpy())
        res = res.e(BinaryOps.ADD, correction)
        # print(res.dtype)
        return res.cast(self.beginning_dtype)

    def _averaging_sin(self, x: LazyBuffer) -> LazyBuffer:
        # Compute 5 sines and average
        # offsets = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        # offsets = [o*10 for o in offsets]
        offsets = [0, 1, 2, 3]
        sines = [
            self._sin(x.e(BinaryOps.ADD, x.const(offset * 2 * math.pi)))
            for offset in offsets
        ]
        # x = x.cast(self.beginning_dtype)
        sum = x.const(0)
        for s in sines:
            sum = sum.e(BinaryOps.ADD, s)
        res = sum.e(BinaryOps.DIV, x.const(len(sines)))
        return res

    def _sin(self, x: LazyBuffer) -> LazyBuffer:
        # x = self.reduce_angle(x)
        x = self.red_ang(x)
        return _taylor(x, self.coefficients)
        # return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 30, x.const(1))
        # if self.beginning_dtype == dtypes.float32:
        #     return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 9, x.const(1))
        # else:
        #     return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 14, x.const(1))

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

    def _mod(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        def v1(x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
            y = y.cast(self.float_precision)
            x = x.cast(self.float_precision)
            return x.e(
                BinaryOps.SUB,
                x.e(BinaryOps.DIV, y)
                .cast(dtypes.int64)
                .cast(self.float_precision)
                .e(BinaryOps.MUL, y),
            )

        def v2(x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
            x = x.cast(self.float_precision)
            y = y.cast(self.float_precision)
            q = x.e(BinaryOps.DIV, y)
            q_floor = q.cast(dtypes.int64).cast(self.float_precision)
            diff = q.e(BinaryOps.SUB, q_floor)
            x = diff.e(BinaryOps.MUL, y)
            return x

        # Return v1 if x < 1e14, else return v2
        # return self._abs(x).e(BinaryOps.CMPLT, x.const(1e14)).e(
        #     TernaryOps.WHERE, v1(x, y), v2(x, y)
        # )
        # return v1(x, y)
        return v2(x, y)

    def _karatsuba_mul(
        self, a: LazyBuffer, b: LazyBuffer, c: LazyBuffer, d: LazyBuffer
    ) -> LazyBuffer:
        ac = a.e(BinaryOps.MUL, c)
        # print("AC: ")
        # print(__import__('tinygrad').Tensor(ac).numpy())
        bd = b.e(BinaryOps.MUL, d)
        # print("BD: ")
        # print(__import__('tinygrad').Tensor(bd).numpy())
        adbc = (
            a.e(BinaryOps.ADD, b)
            .e(BinaryOps.MUL, c.e(BinaryOps.ADD, d))
            .e(BinaryOps.SUB, ac)
            .e(BinaryOps.SUB, bd)
        )
        # print("ADBC: ")
        # print(__import__('tinygrad').Tensor(adbc).numpy())
        # ac, adbc, bd must be concatenated in this order to get the full number
        # i.e. ac * 10^2n + adbc * 10^n + bd, where n is the number of digits the initial number
        # was split by
        return ac, adbc, bd

    def _mod_2pi(self, x: LazyBuffer) -> LazyBuffer:
        # return self._mod(x, x.const(math.pi))
        # x = x.cast(self.float_precision)
        a = (
            x.e(BinaryOps.DIV, x.const(1e9))
            .cast(dtypes.int64)
            .cast(self.float_precision)
        )
        b = x.e(
            BinaryOps.SUB, a.e(BinaryOps.MUL, x.const(1e9).cast(self.float_precision))
        )
        c = x.const(31830988.0)
        d = x.const(618379068.3946033850086)

        ac, adbc, bd = self._karatsuba_mul(a, b, c, d)
        ac = ac.e(BinaryOps.MUL, x.const(1e1))
        adbc = adbc.e(BinaryOps.MUL, x.const(1e-8))
        bd = bd.e(BinaryOps.MUL, x.const(1e-17))
        rem = ac.e(BinaryOps.ADD, adbc).e(BinaryOps.ADD, bd)

        # print("REM: ")
        # print(__import__('tinygrad').Tensor(rem).numpy())
        # nearestremint = rem.e(BinaryOps.ADD, rem.const(0.5)).cast(dtypes.int64)
        nearestremint = rem.cast(dtypes.int64)

        floor = rem.cast(dtypes.int64).cast(self.float_precision)
        rem = rem.e(BinaryOps.SUB, floor)
        rem = rem.e(BinaryOps.MUL, rem.const(math.pi))

        # rem = self._mod(x, x.const(math.pi))
        # return rem

        nearestremint = nearestremint.e(BinaryOps.MOD, nearestremint.const(2))
        rem = nearestremint.e(BinaryOps.CMPEQ, nearestremint.const(1)).e(
            TernaryOps.WHERE, rem.e(UnaryOps.NEG), rem
        )

        # rem = nearestremint.e(BinaryOps.CMPEQ, nearestremint.const(2)) \
        # .e(TernaryOps.WHERE, rem.const(math.pi), rem)

        # return rem.e(BinaryOps.ADD, correction)
        return rem

    def reduce_angle(self, x: LazyBuffer) -> LazyBuffer:
        # x = x.e(BinaryOps.SUB, x.const(36323880599548.86))
        lt0 = x.e(BinaryOps.CMPLT, x.const(0))
        x = self._abs(x)
        x = lt0.e(TernaryOps.WHERE, x.e(BinaryOps.ADD, x.const(math.pi)), x)
        # x = self._mod(x, x.const(math.pi))
        x = self._mod_2pi(x)
        res = x.e(BinaryOps.CMPEQ, x.const(float("inf"))).e(
            TernaryOps.WHERE, x.const(math.nan), x
        )
        res = x.e(BinaryOps.CMPEQ, x.const(float("-inf"))).e(
            TernaryOps.WHERE, x.const(math.nan), res
        )
        # print("REDUCED ANGLE: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        return res

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return self._sin_grand(x)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self._sin_grand(self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x)).e(
            BinaryOps.MUL, grad_output
        )


# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(BinaryOps.MAX, x.const(0))
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return (
            self.ret.const(0)
            .e(BinaryOps.CMPLT, self.ret)
            .cast(grad_output.dtype)
            .e(BinaryOps.MUL, grad_output)
        )


class Log(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV, self.x)


class Exp(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(BinaryOps.MUL, x.const(1 / math.log(2))).e(UnaryOps.EXP2)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.e(BinaryOps.MUL, grad_output)


class Sqrt(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(
            BinaryOps.DIV, self.ret.e(BinaryOps.MUL, self.ret.const(2))
        )


# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.const(1).e(
            BinaryOps.DIV,
            x.const(1).e(
                BinaryOps.ADD,
                x.e(BinaryOps.MUL, x.const(-1 / math.log(2))).e(UnaryOps.EXP2),
            ),
        )
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.e(
            BinaryOps.MUL, self.ret.const(1).e(BinaryOps.SUB, self.ret)
        ).e(BinaryOps.MUL, grad_output)


class Sign(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.CMPEQ, x.const(0)).e(
            TernaryOps.WHERE,
            x.const(0),
            x.e(BinaryOps.CMPLT, x.const(0)).e(
                TernaryOps.WHERE, x.const(-1), x.const(1)
            ),
        )

    # backward always return 0 to match torch
    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.const(0)


# ************* binary ops *************


class Less(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.CMPLT, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return None, None


class Eq(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.CMPEQ, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return None, None


class Xor(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.XOR, y)


class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.ADD, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return grad_output if self.needs_input_grad[0] else None, (
            grad_output if self.needs_input_grad[1] else None
        )


class Sub(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.SUB, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return grad_output if self.needs_input_grad[0] else None, (
            grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None
        )


class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.MUL, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return (
            self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None
        ), (self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None)


class Div(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.DIV, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        return (
            grad_output.e(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None
        ), (
            grad_output.e(UnaryOps.NEG)
            .e(BinaryOps.MUL, self.x)
            .e(BinaryOps.DIV, self.y.e(BinaryOps.MUL, self.y))
            if self.needs_input_grad[1]
            else None
        )  # noqa: E501


# ************* ternary ops *************


class Where(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer, z: LazyBuffer) -> LazyBuffer:
        self.x = x
        return self.x.e(TernaryOps.WHERE, y, z)

    def backward(
        self, grad_output: LazyBuffer
    ) -> Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]:
        return (
            None,
            (
                self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0))
                if self.needs_input_grad[1]
                else None
            ),
            (
                self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output)
                if self.needs_input_grad[2]
                else None
            ),
        )


# ************* reduce ops *************


class Sum(Function):
    def forward(self, x: LazyBuffer, axis: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.r(ReduceOps.SUM, axis)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.expand(self.input_shape)


class Max(Function):
    def forward(self, x: LazyBuffer, axis: Tuple[int, ...]) -> LazyBuffer:
        self.x, self.ret, self.axis = x, x.r(ReduceOps.MAX, axis), axis
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # 1s in locations where the max was chosen (can be two locations)
        max_is_1s = self.x.e(BinaryOps.CMPEQ, self.ret.expand(self.x.shape)).cast(
            dtypes.float
        )
        div = max_is_1s.r(ReduceOps.SUM, self.axis).expand(self.x.shape)
        return (
            max_is_1s.e(BinaryOps.DIV, div)
            .cast(grad_output.dtype)
            .e(BinaryOps.MUL, grad_output.expand(self.x.shape))
        )


# ************* movement ops *************


# NOTE: this is sum in reverse
class Expand(Function):
    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.expanded_axis = tuple(
            i for i, (si, so) in enumerate(zip(x.shape, shape)) if si != so
        )
        return x.expand(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return (
            grad_output.cast(sum_acc_dtype(grad_output.dtype))
            .r(ReduceOps.SUM, self.expanded_axis)
            .cast(grad_output.dtype)
        )


class Reshape(Function):
    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.reshape(self.input_shape)


class Permute(Function):
    def forward(self, x: LazyBuffer, order: Tuple[int, ...]) -> LazyBuffer:
        self.input_order = order
        return x.permute(order)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.permute(argsort(self.input_order))


class Pad(Function):
    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[int, int], ...]) -> LazyBuffer:
        self.narg = tuple([(p[0], s + p[0]) for s, p in zip(x.shape, arg)])
        return x.pad(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.shrink(self.narg)


class Shrink(Function):
    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[sint, sint], ...]) -> LazyBuffer:
        self.narg = tuple([(p[0], s - p[1]) for s, p in zip(x.shape, arg)])
        return x.shrink(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.pad(self.narg)


class Flip(Function):
    def forward(self, x: LazyBuffer, axis: Tuple[int, ...]) -> LazyBuffer:
        self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])
        return x.stride(self.arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.stride(self.arg)
