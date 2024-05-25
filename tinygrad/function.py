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


def _taylor(x: LazyBuffer, coefficients: List[float]) -> LazyBuffer:
    current_term = x.const(1)
    result = x.const(0)
    for i, coef in enumerate(coefficients):
        if i > 0:
            current_term = current_term.e(BinaryOps.MUL, x)
        result = result.e(BinaryOps.ADD, current_term.e(BinaryOps.MUL, x.const(coef)))
    return result


def _get_info(x: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer, LazyBuffer]:
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
        raise TypeError(f"{x.dtype} not supported.")

    try:
        b = b.e(BinaryOps.MUL, b.const(2)).e(BinaryOps.DIV, b.const(2))
        bpower = b.e(BinaryOps.DIV, b.const(pow_shift)).cast(int_repr, bitcast=True)
        power = bpower.e(BinaryOps.SUB, b.const(bias).cast(int_repr))
        bsig = b.e(BinaryOps.MUL, b.const(sig_shift)).e(
            BinaryOps.DIV, b.const(sig_shift)
        )
        sig = bsig.e(BinaryOps.ADD, bsig.const(fix)).cast(x.dtype, bitcast=True)
        nan = bpower.e(BinaryOps.CMPEQ, bpower.const(sig_shift / 2 - 1))
        return (power, sig, nan)
    except Exception as _:
        return (b.const(0), b.const(0), b.const(0).cast(dtypes.bool))


def _floor(x: LazyBuffer) -> LazyBuffer:
    x_dtype = x.dtype
    return x.cast(dtypes.ulong).cast(x_dtype)


def _floor(x: LazyBuffer) -> LazyBuffer:
    x_dtype = x.dtype
    return x.cast(dtypes.ulong).cast(x_dtype)


class Sin(Function):
    coefficients = [
        -3.927706665244766e-14,
        1.0000000000071227,
        -2.1372653510812573e-10,
        -0.16666666415735265,
        -1.5195658172408884e-08,
        0.008333387485289051,
        -1.218355377507116e-07,
        -0.00019823344523361102,
        -1.7392842459887678e-07,
        2.864770129172753e-06,
        -4.1207696421241214e-08,
        -1.7470936444039315e-08,
    ]

    four_div_pi = [
        (0xA2, 0xF9836E4E, 0x441529FC),
        (0xA2F9, 0x836E4E44, 0x1529FC27),
        (0xA2F983, 0x6E4E4415, 0x29FC2757),
        (0xA2F9836E, 0x4E441529, 0xFC2757D1),
        (0xF9836E4E, 0x441529FC, 0x2757D1F5),
        (0x836E4E44, 0x1529FC27, 0x57D1F534),
        (0x6E4E4415, 0x29FC2757, 0xD1F534DD),
        (0x4E441529, 0xFC2757D1, 0xF534DDC0),
        (0x441529FC, 0x2757D1F5, 0x34DDC0DB),
        (0x1529FC27, 0x57D1F534, 0xDDC0DB62),
        (0x29FC2757, 0xD1F534DD, 0xC0DB6295),
        (0xFC2757D1, 0xF534DDC0, 0xDB629599),
        (0x2757D1F5, 0x34DDC0DB, 0x6295993C),
        (0x57D1F534, 0xDDC0DB62, 0x95993C43),
        (0xD1F534DD, 0xC0DB6295, 0x993C4390),
        (0xF534DDC0, 0xDB629599, 0x3C439041),
    ]

    pi_63 = 3.4061215800865545e-19

    def small_red_ang(self, x):
        n = _floor(x.e(BinaryOps.DIV, x.const(2 * math.pi)))
        ang = x.e(BinaryOps.SUB, n.e(BinaryOps.MUL, x.const(2 * math.pi)))
        adj_bottom = ang.e(BinaryOps.CMPLT, ang.const(math.pi))
        ang = adj_bottom.e(
            TernaryOps.WHERE, ang, ang.e(BinaryOps.SUB, ang.const(math.pi))
        )
        adj_top = ang.e(BinaryOps.CMPLT, ang.const(math.pi / 2))
        ang = adj_top.e(TernaryOps.WHERE, ang, ang.const(math.pi).e(BinaryOps.SUB, ang))
        return adj_bottom.e(TernaryOps.WHERE, ang, ang.e(UnaryOps.NEG))

    def big_red_ang(self, x):
        x_dtype = x.dtype
        x = x.cast(dtypes.float)
        xi = x.cast(dtypes.uint32, bitcast=True)
        index = xi.e(BinaryOps.MUL, xi.const(2**2)).e(BinaryOps.DIV, xi.const(2**28))
        shift = xi.e(BinaryOps.MUL, xi.const(2**6)).e(BinaryOps.DIV, xi.const(2**29))
        xi = (
            xi.e(BinaryOps.MUL, xi.const(2**9))
            .e(BinaryOps.DIV, xi.const(2**9))
            .e(BinaryOps.ADD, xi.const(0x800000))
        )
        xi = xi.e(
            BinaryOps.MUL, shift.cast(dtypes.float).e(UnaryOps.EXP2).cast(dtypes.uint32)
        )

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
        upper_res0 = res0.cast(dtypes.uint64).e(
            BinaryOps.MUL, res0.cast(dtypes.uint64).const(2**32)
        )
        lower_res2 = res2.e(BinaryOps.DIV, res2.const(2**32))
        res0 = upper_res0.e(BinaryOps.ADD, lower_res2)
        res0 = res0.e(BinaryOps.ADD, res1)

        n = res0.e(BinaryOps.ADD, res0.const(1).e(BinaryOps.MUL, res0.const(2**61))).e(
            BinaryOps.DIV, res0.const(2**62)
        )
        res0 = res0.e(BinaryOps.SUB, n.e(BinaryOps.MUL, res0.const(2**62)))
        x = res0.cast(dtypes.int64, bitcast=True).cast(dtypes.double)
        x = x.e(BinaryOps.MUL, x.const(self.pi_63))

        n_mod_4 = n.e(BinaryOps.MOD, n.const(4))
        x = n_mod_4.e(BinaryOps.CMPEQ, n.const(1)).e(
            TernaryOps.WHERE, x.const(math.pi / 2).e(BinaryOps.SUB, x), x
        )
        x = n_mod_4.e(BinaryOps.CMPEQ, n.const(2)).e(
            TernaryOps.WHERE, x.e(UnaryOps.NEG), x
        )
        x = n_mod_4.e(BinaryOps.CMPEQ, n.const(3)).e(
            TernaryOps.WHERE, x.e(BinaryOps.SUB, x.const(math.pi / 2)), x
        )
        return x.cast(x_dtype)

    def red_ang(self, x: LazyBuffer) -> LazyBuffer:
        signs = x.e(BinaryOps.CMPLT, x.const(0))
        x = signs.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)
        size = x.e(BinaryOps.CMPLT, x.const(2**10))
        if self.device == "METAL":
            red_ang = self.small_red_ang(x)
        else:
            red_ang = size.e(
                TernaryOps.WHERE, self.small_red_ang(x), self.big_red_ang(x)
            )
        red_ang = signs.e(TernaryOps.WHERE, red_ang.e(UnaryOps.NEG), red_ang)
        return red_ang

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(UnaryOps.SIN)
        # if x.size == 0: return x
        # x_dtype = x.dtype
        # if x.dtype not in (dtypes.double, dtypes.float): x = x.cast(dtypes.float32)
        # reduced_angles = self.red_ang(x)
        # signs = reduced_angles.e(BinaryOps.CMPLT, reduced_angles.const(0))
        # reduced_angles = signs.e(TernaryOps.WHERE, reduced_angles.e(UnaryOps.NEG), reduced_angles)
        # t = _taylor(reduced_angles, self.coefficients)
        # _, _, nan = _get_info(x)
        # t = nan.e(TernaryOps.WHERE, x.const(math.nan), t)
        # zero = x.e(BinaryOps.CMPEQ, x.const(0))
        # t = zero.e(TernaryOps.WHERE, x.const(0), t)
        # inf = x.e(BinaryOps.CMPEQ, x.const(math.inf))
        # t = inf.e(TernaryOps.WHERE, t.const(math.nan), t)
        # n_inf = x.e(BinaryOps.CMPEQ, x.const(-math.inf))
        # t = n_inf.e(TernaryOps.WHERE, t.const(math.nan), t)
        # return signs.e(TernaryOps.WHERE, t.e(UnaryOps.NEG), t).cast(x_dtype)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # x = self.x.const(math.pi/2).e(BinaryOps.SUB, self.x)
        # return self.forward(x).e(BinaryOps.MUL, grad_output)
        return (
            self.x.const(math.pi / 2)
            .e(BinaryOps.SUB, self.x)
            .e(UnaryOps.SIN)
            .e(BinaryOps.MUL, grad_output)
        )


class MySin(Function):
    coefficients = [
        -3.927706665244766e-14,
        1.0000000000071227,
        -2.1372653510812573e-10,
        -0.16666666415735265,
        -1.5195658172408884e-08,
        0.008333387485289051,
        -1.218355377507116e-07,
        -0.00019823344523361102,
        -1.7392842459887678e-07,
        2.864770129172753e-06,
        -4.1207696421241214e-08,
        -1.7470936444039315e-08,
    ]

    four_div_pi = [
        (0xA2, 0xF9836E4E, 0x441529FC),
        (0xA2F9, 0x836E4E44, 0x1529FC27),
        (0xA2F983, 0x6E4E4415, 0x29FC2757),
        (0xA2F9836E, 0x4E441529, 0xFC2757D1),
        (0xF9836E4E, 0x441529FC, 0x2757D1F5),
        (0x836E4E44, 0x1529FC27, 0x57D1F534),
        (0x6E4E4415, 0x29FC2757, 0xD1F534DD),
        (0x4E441529, 0xFC2757D1, 0xF534DDC0),
        (0x441529FC, 0x2757D1F5, 0x34DDC0DB),
        (0x1529FC27, 0x57D1F534, 0xDDC0DB62),
        (0x29FC2757, 0xD1F534DD, 0xC0DB6295),
        (0xFC2757D1, 0xF534DDC0, 0xDB629599),
        (0x2757D1F5, 0x34DDC0DB, 0x6295993C),
        (0x57D1F534, 0xDDC0DB62, 0x95993C43),
        (0xD1F534DD, 0xC0DB6295, 0x993C4390),
        (0xF534DDC0, 0xDB629599, 0x3C439041),
    ]

    pi_63 = 3.4061215800865545e-19

    def small_red_ang(self, x):
        n = _floor(x.e(BinaryOps.DIV, x.const(2 * math.pi)))
        ang = x.e(BinaryOps.SUB, n.e(BinaryOps.MUL, x.const(2 * math.pi)))
        adj_bottom = ang.e(BinaryOps.CMPLT, ang.const(math.pi))
        ang = adj_bottom.e(
            TernaryOps.WHERE, ang, ang.e(BinaryOps.SUB, ang.const(math.pi))
        )
        adj_top = ang.e(BinaryOps.CMPLT, ang.const(math.pi / 2))
        ang = adj_top.e(TernaryOps.WHERE, ang, ang.const(math.pi).e(BinaryOps.SUB, ang))
        return adj_bottom.e(TernaryOps.WHERE, ang, ang.e(UnaryOps.NEG))

    def big_red_ang(self, x):
        x_dtype = x.dtype
        x = x.cast(dtypes.float)
        xi = x.cast(dtypes.uint32, bitcast=True)
        index = xi.e(BinaryOps.MUL, xi.const(2**2)).e(BinaryOps.DIV, xi.const(2**28))
        shift = xi.e(BinaryOps.MUL, xi.const(2**6)).e(BinaryOps.DIV, xi.const(2**29))
        xi = (
            xi.e(BinaryOps.MUL, xi.const(2**9))
            .e(BinaryOps.DIV, xi.const(2**9))
            .e(BinaryOps.ADD, xi.const(0x800000))
        )
        xi = xi.e(
            BinaryOps.MUL, shift.cast(dtypes.float).e(UnaryOps.EXP2).cast(dtypes.uint32)
        )

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
        upper_res0 = res0.cast(dtypes.uint64).e(
            BinaryOps.MUL, res0.cast(dtypes.uint64).const(2**32)
        )
        lower_res2 = res2.e(BinaryOps.DIV, res2.const(2**32))
        res0 = upper_res0.e(BinaryOps.ADD, lower_res2)
        res0 = res0.e(BinaryOps.ADD, res1)

        n = res0.e(BinaryOps.ADD, res0.const(1).e(BinaryOps.MUL, res0.const(2**61))).e(
            BinaryOps.DIV, res0.const(2**62)
        )
        res0 = res0.e(BinaryOps.SUB, n.e(BinaryOps.MUL, res0.const(2**62)))
        x = res0.cast(dtypes.int64, bitcast=True).cast(dtypes.double)
        x = x.e(BinaryOps.MUL, x.const(self.pi_63))

        n_mod_4 = n.e(BinaryOps.MOD, n.const(4))
        x = n_mod_4.e(BinaryOps.CMPEQ, n.const(1)).e(
            TernaryOps.WHERE, x.const(math.pi / 2).e(BinaryOps.SUB, x), x
        )
        x = n_mod_4.e(BinaryOps.CMPEQ, n.const(2)).e(
            TernaryOps.WHERE, x.e(UnaryOps.NEG), x
        )
        x = n_mod_4.e(BinaryOps.CMPEQ, n.const(3)).e(
            TernaryOps.WHERE, x.e(BinaryOps.SUB, x.const(math.pi / 2)), x
        )
        return x.cast(x_dtype)

    def red_ang(self, x: LazyBuffer) -> LazyBuffer:
        signs = x.e(BinaryOps.CMPLT, x.const(0))
        x = signs.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)
        size = x.e(BinaryOps.CMPLT, x.const(2**10))
        if self.device == "METAL":
            red_ang = self.small_red_ang(x)
        else:
            red_ang = size.e(
                TernaryOps.WHERE, self.small_red_ang(x), self.big_red_ang(x)
            )
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
        if x.dtype not in (dtypes.double, dtypes.float):
            x = x.cast(dtypes.float32)
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
        res = self._sin(x)  # .cast(self.beginning_dtype)
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
        res = x.e(BinaryOps.CMPEQ, pinf).e(TernaryOps.WHERE, x.const(math.nan), res)
        res = x.e(BinaryOps.CMPEQ, ninf).e(TernaryOps.WHERE, x.const(math.nan), res)
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
    # def _pade(self, x: LazyBuffer) -> LazyBuffer:
    #     PQ = [-1.0171323192031956e-07, -8.974972614803067e-06, -0.00021569320345808662, -0.0022287096343596574, -0.011523228925209891, -0.02997122525322186, -0.03352213026479973, -0.038427405218536395, -0.30428083695014563, -1.0129765778842845, -1.200969213826871, -0.09639100181475165, -0.8843265383274049, -5.370061941354115, -5.066130614712678, 2.0073414590633094, 0.9698928140269265, -3.8600022677142456, 2.58620555984148, 1.6514031476066195, -8.252254129240626, 0.06224699227314931, 6.681924589077946, -4.629513219080865, -6.2307758823412405, -2.437614257288357, -6.571665880451656, -4.774951226934315, 0.7864858197983817, 1.0, 0.0]
    #     QC = [-1.683401215861629e-08, -2.2233784204689774e-06, -7.145980947116403e-05, -0.0009623854318982274, -0.006577947471552396, -0.024187052836199364, -0.0457752693637781, -0.04583686976823449, -0.14555829220584707, -0.7259367483528768, -1.5889614963900662, -1.2720682742743799, -0.2973848945342432, -3.1435886735004583, -7.9486882979352815, -4.281567449506767, 2.7946913135358447, -0.9799274703772372, -2.9267202153766196, 3.9246439573148297, -2.1919784020750295, -8.812108780672386, 3.538211985613214, 4.8688655541076935, -7.689102963891802, -7.102904975648057, -5.319153241666949, -8.983015312235345, -4.465041650368457, 1.2864858197983817, 1.0]
    #     ox = x
    #     Psum = x.const(0)
    #     Qsum = x.const(0)
    #     x = x.const(1)
    #     Pterms = []
    #     Qterms = []
    #     for p, q in zip(PQ[::-1], QC[::-1]):
    #         Pterms.append(x.e(BinaryOps.MUL, x.const(p)))
    #         Qterms.append(x.e(BinaryOps.MUL, x.const(q)))
    #         x = x.e(BinaryOps.MUL, ox)
    #
    #     for p, q in zip(Pterms, Qterms):
    #         Psum = Psum.e(BinaryOps.ADD, p)
    #         Qsum = Qsum.e(BinaryOps.ADD, q)
    #     return Psum.e(BinaryOps.DIV, Qsum)



    def forward(self, x: LazyBuffer) -> LazyBuffer:
        # x = x.e(BinaryOps.ADD, x.const(1))
        self.x = x
        return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))
        # self.beginning_dtype = x.dtype
        # if self.beginning_dtype == dtypes.float32 and self.device != "METAL":
        #     x = x.cast(dtypes.float64)
        #
        # # computed = x.e(BinaryOps.CMPLT, x.const(2)).e(
        # #     TernaryOps.WHERE, self._simple_approx(x), self._pade(x))
        # computed = self._pade(x)
        # computed = Exp().forward(x)

        # computed = computed.cast(self.beginning_dtype)
        # If x < -1, return NaN
        # computed = x.e(BinaryOps.CMPLT, x.const(1)).e(
            # TernaryOps.WHERE, computed.const(math.nan), computed
        # )

        # return computed

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV, self.x)


class Exp(Function):

    def _mod(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(
            BinaryOps.SUB,
            x.e(BinaryOps.DIV, y).cast(dtypes.int64).cast(x.dtype).e(BinaryOps.MUL, y),
        )

    def _abs(self, x: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.CMPLT, x.const(0)).e(
            TernaryOps.WHERE, x.e(UnaryOps.NEG), x)


        
    # def _pade(self, x: LazyBuffer, PC: list[float], QC: list[float]) -> LazyBuffer:
    def _pade(self, x: LazyBuffer) -> LazyBuffer:
        # Pade 7
        # PC = [5.781255783528314e-08, 3.2375032383864105e-06, 8.741258742934327e-05, 0.0014568764570696557, 0.01602564102704604, 0.11538461539064754, 0.5000000000118245, 1.0]
        # QC = [-5.781255779802232e-08, 3.237503236860197e-06, -8.741258739917715e-05, 0.001456876456709917, -0.01602564102435582, 0.11538461537882291, -0.4999999999881754, 1.0]

        # Pade 10
        PC = [1.4915521524114695e-12, 1.6407073718200185e-10, 8.859819752794675e-09, 3.071404154872436e-07, 7.524940113100287e-06, 0.0001354489208912936, 0.0018059855977014816, 0.01754385996556851, 0.11842105398892322, 0.500000002734005, 1.0]
        QC = [1.4915518992207407e-12, -1.6407071206103022e-10, 8.859818595453254e-09, -3.071403822471257e-07, 7.5249394551197106e-06, -0.00013544891159627956, 0.001805985504260442, -0.017543859323018895, 0.1184210512549182, -0.499999997265995, 1.0]

        # Pade 12
        # PC = [7.720255179077208e-16, 1.2043614264968927e-13, 9.273586521897406e-12, 4.6367932270546797e-10, 1.6692452501609867e-08, 4.5403458513140933e-07, 9.534723358411643e-06, 0.00015527973090666627, 0.0019409960211353716, 0.018115957285232354, 0.11956528277956602, 0.5000001329950453, 1.0]
        # QC = [7.720193367103758e-16, -1.2043515579017277e-13, 9.273516211983926e-12, -4.6367620824561017e-10, 1.6692355898639348e-08, -4.54032373790667e-07, 9.534685226131455e-06, -0.00015527923607996453, 0.001940991293178444, -0.01811592566347763, 0.11956514978452067, -0.4999998670049547, 1.0]

        # Pade 15
        # PC = [5.555436579284281e-21, 1.3125769210893678e-18, 1.5403901227001058e-16, 1.1868561690995733e-14, 6.692190361862563e-13, 2.916409677106555e-11, 1.0120390540692245e-09, 2.8407411743582775e-08, 6.48870399793012e-07, 1.2036730599655186e-05, 0.00017952743589247597, 0.002110706881493183, 0.018906257230122514, 0.12163057975674008, 0.5018697635563099, 1.0]
        # QC = [-4.478605805860244e-21, 1.0863415635646055e-18, -1.3053235772233647e-16, 1.0272111889614586e-14, -5.902904998127621e-13, 2.6167958543246226e-11, -9.222321652739181e-10, 2.6253510113724633e-08, -6.074293602608563e-07, 1.1401546217153162e-05, -0.00017190730848319345, 0.0020414456036890105, -0.018456107415129226, 0.11976081620043016, -0.49813023644369014, 1.0]

        # Pade 20
        # PC = [4.6637780272896875e-28, 1.3748145829270517e-25, 2.0678429841220108e-23, 2.0953791679834908e-21, 1.5954050482647834e-19, 9.657836328006502e-18, 4.80447918319304e-16, 2.004630036984425e-14, 7.104325973115042e-13, 2.1540905408061082e-11, 5.605722001145196e-10, 1.2516611717254843e-08, 2.389176673869676e-07, 3.870734354682331e-06, 5.262149025024091e-05, 0.0005901701545761858, 0.005325171382157336, 0.03720855266811149, 0.18917409413939507, 0.6233076919580327, 1.0]
        # QC = [-4.596395071180581e-30, 1.0400140559774975e-27, -9.200691857444309e-26, 1.7665692257725436e-24, 4.524980426094502e-22, -5.909657238678006e-20, 4.097308476599268e-18, -1.9320408770711587e-16, 6.50322000551547e-15, -1.5121849971817443e-13, 1.9982489493620867e-12, 3.769179189263304e-12, -6.716879499055917e-10, 3.763682790963229e-09, 5.236271142124743e-07, -2.192008517319955e-05, 0.00048571712407121536, -0.00697836215893382, 0.06586640218136229, -0.37669230804196724, 1.0]

        # Pade 25
        # PC = [5.420016188874444e-35, 2.190478474522844e-32, 4.353750978651839e-30, 5.734756455756635e-28, 5.641498546659708e-26, 4.413541919185761e-24, 2.851064055011928e-22, 1.5578415075746856e-20, 7.316574694612334e-19, 2.985704540205775e-17, 1.0661526434689396e-15, 3.3457611976954935e-14, 9.246324126031898e-13, 2.25059414168002e-11, 4.816762896326312e-10, 9.033221504798107e-09, 1.4764785804432456e-07, 2.0874888534864705e-06, 2.5268859466655203e-05, 0.0002583145551341067, 0.002188986263937045, 0.014984067649648715, 0.07976938603758893, 0.3108542287297197, 0.7926784746495348, 1.0]
        # QC = [-2.4392115725103084e-37, 8.305113443936566e-35, -1.3296200543574956e-32, 1.301665702770552e-30, -8.393982588980559e-29, 3.453329213865048e-27, -6.488614697836734e-26, -1.9147711945085106e-24, 1.847738870791667e-22, -5.754358000408214e-21, 2.5103464163079256e-20, 4.6859820641589556e-18, -1.6671207891800114e-16, 6.323299630774157e-16, 1.306754206047343e-13, -4.4812544632840956e-12, 3.354038334228988e-11, 2.2279060435462326e-09, -9.50323324071589e-08, 1.958906971016632e-06, -2.4490044806533024e-05, 0.0001953835353304822, -0.0014122720340299971, 0.01817575408018487, -0.20732152535046522, 1.0]

        # Pade 30
        # PC = [3.174762095014651e-43, 1.263811817120668e-40, 2.658055794651458e-38, 3.883024949574583e-36, 4.387406677622896e-34, 4.0568154379146483e-32, 3.1760823864584966e-30, 2.152815765767272e-28, 1.2828932159926686e-26, 6.794576362219056e-25, 3.2233733507145425e-23, 1.3773897092444197e-21, 5.322115916738285e-20, 1.864160966757357e-18, 5.927115816904798e-17, 1.7112268658344284e-15, 4.4833216357863006e-14, 1.0642119389244422e-12, 2.2827822584346438e-11, 4.408610736922463e-10, 7.627455623784314e-09, 1.174550495142883e-07, 1.5962976038929916e-06, 1.893847991776601e-05, 0.0001933307149243482, 0.0016655142800334362, 0.011784956877608896, 0.06581777187936474, 0.27224432096364365, 0.7419615234818666, 1.0]
        # QC = [-1.6450967139818605e-47, 7.083837883319338e-45, -1.4485633669730453e-42, 1.8497904314244254e-40, -1.627375956018421e-38, 1.0203926744598311e-36, -4.4884695672444623e-35, 1.2351339816691232e-33, -8.726182247724124e-33, -9.170082074829936e-31, 4.385918018403493e-29, -7.351396750962011e-28, -1.4056699049671058e-26, 1.3028331440402866e-24, -4.949101283272678e-23, 1.327044338712687e-21, -1.6968548038857145e-20, -7.208661425683472e-19, 5.711026248292443e-17, -2.0601609093590077e-15, 4.8444247051418994e-14, -7.80168384841284e-13, 1.1140384591689052e-11, -6.701138406495334e-10, 5.579382638755613e-08, -2.8800067559171957e-06, 9.575823308819763e-05, -0.0021124540100122403, 0.03028279748177699, -0.25803847651813333, 1.0]

        ox = x
        Psum = x.const(0)
        Qsum = x.const(0)
        x = x.const(1)
        Pterms = []
        Qterms = []
        for p, q in zip(PC[::-1], QC[::-1]):
            Pterms.append(x.e(BinaryOps.MUL, x.const(p)))
            
            Qterms.append(x.e(BinaryOps.MUL, x.const(q)))
            x = x.e(BinaryOps.MUL, ox)

        for p, q in zip(Pterms[::-1], Qterms[::-1]):
            Psum = Psum.e(BinaryOps.ADD, p)
            Qsum = Qsum.e(BinaryOps.ADD, q)
        return Psum.e(BinaryOps.DIV, Qsum)

    def _eq(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return self._abs(x.e(BinaryOps.SUB, y)).e(BinaryOps.CMPLT, x.const(0.001)).cast(x.dtype)
    
    def _revnorm(self, x: LazyBuffer) -> LazyBuffer:
        """
        Revert and normalize x, using arithmetic ops:
            if x is 0, return 1
            if x != 0, return x
        """
        return x.e(BinaryOps.ADD, self._eq(x, x.const(0)))
        

    def _exp_lowprec(self, x: LazyBuffer) -> LazyBuffer:
        sign = x.e(BinaryOps.CMPLT, x.const(0)).e(
            TernaryOps.WHERE,
            x.const(-1),
            x.const(1),
        )
        x = self._abs(x)
        res = self._pade(x)
        res = sign.e(BinaryOps.CMPEQ, sign.const(-1)).e(
            TernaryOps.WHERE, res.const(1).e(BinaryOps.DIV, res), res
        )
        return res

    def _divmod(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
            divres = x.e(BinaryOps.DIV, y).cast(dtypes.uint64).cast(x.dtype)
            modres = x.e(BinaryOps.SUB, divres.e(BinaryOps.MUL, y))
            return divres, modres

    def _exp(self, x: LazyBuffer) -> LazyBuffer:
        sign = x.e(BinaryOps.CMPLT, x.const(0)).e(
            TernaryOps.WHERE,
            x.const(-1),
            x.const(1),
        )
        x = self._abs(x)


        d50, m50 = self._divmod(x, x.const(50))
        res = self._revnorm(x.const(math.exp(50)).e(BinaryOps.MUL, d50))
        # print("RES after d50: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        d25, m25 = self._divmod(m50, x.const(25))
        res = res.e(BinaryOps.MUL, self._revnorm(x.const(math.exp(25)).e(BinaryOps.MUL, d25)))
        # print("RES after d25: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        d12, m12 = self._divmod(m25, x.const(12))
        res = res.e(BinaryOps.MUL, self._revnorm(x.const(math.exp(12)).e(BinaryOps.MUL, d12)))
        # print("RES after d12: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        d6, m6 = self._divmod(m12, x.const(6))
        res = res.e(BinaryOps.MUL, self._revnorm(x.const(math.exp(6)).e(BinaryOps.MUL, d6)))
        # print("RES after d6: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        d3, m3 = self._divmod(m6, x.const(3))
        res = res.e(BinaryOps.MUL, self._revnorm(x.const(math.exp(3)).e(BinaryOps.MUL, d3)))
        # print("RES after d3: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        rempade = self._pade(m3)
        # print("Pade: ")
        # print(__import__('tinygrad').Tensor(rempade).numpy())
        res = res.e(BinaryOps.MUL, rempade)
        # print("RES after pade: ")
        # print(__import__('tinygrad').Tensor(res).numpy())

        res = sign.e(BinaryOps.CMPEQ, sign.const(-1)).e(
            TernaryOps.WHERE, res.const(1).e(BinaryOps.DIV, res), res
        )
        return res

    # def zero_if_minus_inf(self, x: LazyBuffer) -> LazyBuffer:
    #     eq = x.e(BinaryOps.CMPEQ, x.const(float("-inf"))).cast(x.dtype)
    #     eq = x.const(1).e(BinaryOps.SUB, eq)
    #     return x.e(BinaryOps.MUL, eq)



    def forward(self, x: LazyBuffer) -> LazyBuffer:
        # self.ret = x.e(BinaryOps.MUL, x.const(1 / math.log(2))).e(UnaryOps.EXP2)
        # return self.ret

        initial_x = x
        pinf_t = x.const(88.722)
        ninf_t = x.const(-103.97539753975397)
        x = x.e(BinaryOps.CMPLT, ninf_t).e(TernaryOps.WHERE, x.const(0), x)
        # x = self.zero_if_minus_inf(x)


        # x = x.e(BinaryOps.MUL, x.const(1 / math.log(2)))

        self.beginning_dtype = x.dtype
        if self.beginning_dtype != dtypes.half and self.device != "METAL":
            x = x.cast(dtypes.float64)
        # x = x.cast(dtypes.float64)

        # if self.device != "METAL":
        #     x = x.cast(dtypes.float64)
        # print(self.beginning_dtype)
        # if self.device == "METAL":
        # x = x.cast(dtypes.float64)
        # print("X: ")
        # print(__import__('tinygrad').Tensor(x).numpy())
        # x = x.cast(dtypes.float64)
        # else:
        #     x = x.cast(dtypes.float64)

        # _, _, nan = _get_info(x)

        isnotnan = x.e(BinaryOps.CMPEQ, x)
        # print("X: ")
        # print(__import__('tinygrad').Tensor(x).numpy())
        # print("X dtype:")
        # print(x.dtype)
        #
        if self.beginning_dtype == dtypes.half:
            computed = self._exp_lowprec(x)
        else:
            computed = self._exp(x)
            # computed = self._exp_lowprec(x)

        computed = initial_x.e(BinaryOps.CMPLT, pinf_t).e(
            TernaryOps.WHERE, computed, computed.const(float("inf"))
        )
        computed = isnotnan.e(TernaryOps.WHERE, computed, x.const(float("nan")))
        computed = initial_x.e(BinaryOps.CMPLT, ninf_t).e(
            TernaryOps.WHERE, computed.const(0), computed
        )

        # print("COMPUTED: ")
        # print(__import__('tinygrad').Tensor(computed).numpy()[0])

        if not dtypes.is_int(self.beginning_dtype):
            computed = computed.cast(self.beginning_dtype)

        # print("RET: ")
        # print(__import__('tinygrad').Tensor(ret).numpy()[0])
        self.ret = computed#.cast(self.beginning_dtype)
        # print("RET: ")
        # print(__import__('tinygrad').Tensor(self.ret).numpy())
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
