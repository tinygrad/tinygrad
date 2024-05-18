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


class Sin(Function):
    def _sin_grand(self, x: LazyBuffer) -> LazyBuffer:
        self.beginning_dtype = x.dtype
        if Device.DEFAULT != "METAL":
            x = x.cast(dtypes.float64)
        else:
            x = x.cast(dtypes.float32)
        self.float_precision = x.dtype
        # xsign = x.e(BinaryOps.CMPLT, x.const(0)).e(
        #     TernaryOps.WHERE, x.const(-1), x.const(1)
        # )

        # Compute normal sin if below 4e13, else use averaging
        res = (
            self._abs(x)
            .e(BinaryOps.CMPLT, x.const(1e13))
            .e(TernaryOps.WHERE, self._sin(x), self._averaging_sin(x))
        )
        # return res
        # res = self._sin(x)
        # return res

        # cf1 = x.const(-0.002)
        # cf2 = x.const(-0.0075)
        # cf3 = x.const(-0.05)
        # cf4 = x.const(-0.05)
        # Choose correction factor based on x magnitude
        # cf = self._abs(x).e(BinaryOps.CMPLT, x.const(1e14)).e(TernaryOps.WHERE, cf1, cf2)
        # cf = self._abs(x) .e(BinaryOps.CMPLT, x.const(153e12)) .e(TernaryOps.WHERE, cf, cf3)
        # cf = self._abs(x).e(BinaryOps.CMPLT, x.const(1e15)).e(TernaryOps.WHERE, cf, cf4)

        # cf = x.const(-0.03)
        # cf = x.const(-0.00)
        # correction = self._sin(
        #     x.e(BinaryOps.ADD, x.const(math.pi / 2).e(BinaryOps.MUL, xsign))
        # ).e(BinaryOps.MUL, cf)

        # res = x.e(BinaryOps.CMPLT, x.const(1e14)).e(TernaryOps.WHERE, res, res.e(BinaryOps.ADD, correction))
        # res = x.e(BinaryOps.CMPLT, x.const(1e1)).e(TernaryOps.WHERE, res, res.e(BinaryOps.ADD, correction))
        # print("SIN: ")
        # print(__import__('tinygrad').Tensor(res).numpy())
        return res.cast(self.beginning_dtype)

    def _averaging_sin(self, x: LazyBuffer) -> LazyBuffer:
        # Compute 5 sines and average
        # offsets = [0]
        offsets = [-3, -2, -1, 0, 1, 2, 3]
        # offsets = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        # offsets = [i for i in range(-10, 11)]
        # offsets = [-2, -1, 0, 1, 2]
        sines = [ self._sin(x.e(BinaryOps.ADD, x.const(offset * 2 * math.pi))) for offset in offsets ]
        sum = x.const(0)
        for s in sines:
            sum = sum.e(BinaryOps.ADD, s)
        res = sum.e(BinaryOps.DIV, x.const(len(sines)))
        return res

    def _sin(self, x: LazyBuffer) -> LazyBuffer:
        x = self.reduce_angle(x)
        return self.horner_taylor_sin(x, x.e(BinaryOps.MUL, x), 30, x.const(1))

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
        return v1(x, y)
        # return v2(x, y)

    def _karatsuba_mul(self, a: LazyBuffer, b: LazyBuffer, c: LazyBuffer, d: LazyBuffer) -> LazyBuffer:
        ac = a.e(BinaryOps.MUL, c)
        # print("AC: ")
        # print(__import__('tinygrad').Tensor(ac).numpy())
        bd = b.e(BinaryOps.MUL, d)
        # print("BD: ")
        # print(__import__('tinygrad').Tensor(bd).numpy())
        adbc = a.e(BinaryOps.ADD, b).e(BinaryOps.MUL, c.e(BinaryOps.ADD, d)).e(BinaryOps.SUB, ac).e(BinaryOps.SUB, bd)
        # print("ADBC: ")
        # print(__import__('tinygrad').Tensor(adbc).numpy())
        # ac, adbc, bd must be concatenated in this order to get the full number
        # i.e. ac * 10^2n + adbc * 10^n + bd, where n is the number of digits the initial number
        # was split by
        return ac, adbc, bd

    def _mod_2pi(self, x: LazyBuffer) -> LazyBuffer:
        a = x.e(BinaryOps.DIV, x.const(1e9)).cast(dtypes.int64).cast(self.float_precision)
        print(self.float_precision)
        # a = x.e(BinaryOps.DIV, x.const(1e7)).cast(dtypes.uint64).cast(self.float_precision)
        # print("A: ")
        # print(__import__('tinygrad').Tensor(a).numpy())
        # a = a.e(BinaryOps.MUL, x.const(1e9))
        # print("A: ")
        # print(__import__('tinygrad').Tensor(a).numpy())
        # b = x.e(BinaryOps.SUB, a)
        b = x.e(BinaryOps.SUB, a.e(BinaryOps.MUL, x.const(1e9)))
        # b = x.e(BinaryOps.SUB, a.e(BinaryOps.MUL, x.const(1e7)))
        # print("B: ")
        # print(__import__('tinygrad').Tensor(b).numpy())
        c = x.const(1591549430.0)
        d = x.const(918953419.7301692504)


        # c = 15915494.0
        # d = 309189534.197301692504

        # c = 15915494.0
        # d = 309189534.197301692504
        # c = x.const(15915494.0)
        # d = x.const(309189534.197301692504)

        # c = x.const(159154.0)
        # d = x.const(9430918.9534197301692504)
        ac, adbc, bd = self._karatsuba_mul(a, b, c, d)

        # rem = result[0] * 1e-1 + result[1] * 1e-10 + result[2] * 1e-19
        ac = ac.e(BinaryOps.MUL, x.const(1e-1))
        # ac = ac.e(BinaryOps.MUL, x.const(1e1))
        # ac = ac.e(BinaryOps.MUL, x.const(1e5))
        adbc = adbc.e(BinaryOps.MUL, x.const(1e-10))
        # adbc = adbc.e(BinaryOps.MUL, x.const(1e-8))
        # adbc = adbc.e(BinaryOps.MUL, x.const(1e-4))
        bd = bd.e(BinaryOps.MUL, x.const(1e-19))
        # bd = bd.e(BinaryOps.MUL, x.const(1e-17))
        # bd = bd.e(BinaryOps.MUL, x.const(1e-13))
        rem = ac.e(BinaryOps.ADD, adbc).e(BinaryOps.ADD, bd)
        # print("REM: ")
        # print(__import__('tinygrad').Tensor(rem).numpy()[0])
        floor = rem.cast(dtypes.uint64).cast(self.float_precision)
        # print("FLOOR: ")
        # print(__import__('tinygrad').Tensor(floor).numpy()[0])
        # rem = rem.e(BinaryOps.SUB, rem.cast(dtypes.uint64).cast(self.float_precision))
        rem = rem.e(BinaryOps.SUB, floor)

        # print("REM: ")
        # print(__import__('tinygrad').Tensor(rem).numpy())
        rem = rem.e(BinaryOps.MUL, x.const(2*math.pi))

        # print("REM: ")
        # print(__import__('tinygrad').Tensor(rem).numpy())
        return rem

        fallback = self._mod(x, x.const(2 * math.pi))
        return fallback
        # return x.e(BinaryOps.CMPLT, x.const(1e15)).e(TernaryOps.WHERE, fallback, rem)
        # return self._abs(x).e(BinaryOps.CMPLT, x.const(0.25e15)).e(TernaryOps.WHERE, fallback, rem)
        # return self._abs(x).e(BinaryOps.CMPLT, x.const(0.25e15)).e(TernaryOps.WHERE, rem, fallback)
        return self._abs(x).e(BinaryOps.CMPLT, x.const(1e13)).e(TernaryOps.WHERE, fallback, rem)


    def reduce_angle(self, x: LazyBuffer) -> LazyBuffer:
        lt0 = x.e(BinaryOps.CMPLT, x.const(0))
        x = self._abs(x)
        x = lt0.e(TernaryOps.WHERE, x.e(BinaryOps.ADD, x.const(math.pi)), x)
        # x = self._mod(x, x.const(2 * math.pi))
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
