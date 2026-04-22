from __future__ import annotations
import functools, itertools
from typing import TYPE_CHECKING, Self, Sequence, Literal, get_args
from tinygrad.mixin.elementwise import ElementwiseMixin
from tinygrad.mixin.movement import MovementMixin
from tinygrad.mixin.reduce import ReduceMixin
from tinygrad.uop import Ops
from tinygrad.uop.ops import _broadcast_shape, resolve, smax, smin, identity_element
from tinygrad.dtype import ConstType, DTypeLike, Invalid, dtypes, least_upper_dtype, sum_acc_dtype, to_dtype
from tinygrad.helpers import argfix, ceildiv, flatten, flat_to_grouped, make_tuple, prod, resolve_pool_pads, round_up

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint

ReductionStr = Literal["mean", "sum", "none"]


class OpMixin(ElementwiseMixin, ReduceMixin):
  @staticmethod
  def unique_const(fill_value:ConstType, **kwargs): raise NotImplementedError("creation helpers are only supported on Tensor and UOp")

  @classmethod
  def full(cls, shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    return cls.unique_const(fill_value, **kwargs).reshape((1,)*len(new_shape := argfix(shape))).expand(new_shape)

  @classmethod
  def invalids(cls, *shape, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with Invalid.

    This is an alternative to Tensor.empty when you want an "anonymous" buffer.

    Eventually Tensor.empty will be replaced by this.
    """
    return cls.full(argfix(*shape), Invalid, **kwargs)

  @classmethod
  def zeros(cls, *shape, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return cls.full(argfix(*shape), 0.0, **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return cls.full(argfix(*shape), 1.0, **kwargs)

  @classmethod
  def arange(cls, start, stop=None, step=1, **kwargs) -> Self:
    """
    Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

    If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

    If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    if stop is None: stop, start = start, 0
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    lo, hi = (start, stop-step) if step > 0 else (stop-step, start)
    if lo < (dt:=to_dtype(dtype)).min or dt.max < hi: raise OverflowError(f"arange [{start}, {stop}) is not representable in dtype {dtype}")
    # NOTE: this matches numpy, torch raises RuntimeError if stop-start and step have different signs
    if (output_len:=ceildiv(stop-start, step)) <= 0: return cls.full((0,), 0, dtype=dtype, **kwargs)
    return (cls.full((output_len,), step, dtype=dtype, **kwargs)._cumalu(0, Ops.ADD) + (start - step)).cast(dtype)

  @classmethod
  def linspace(cls, start:int|float, stop:int|float, steps:int, **kwargs) -> Self:
    """
    Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(0, 10, 5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.linspace(-1, 1, 5).numpy())
    ```
    """
    if steps < 0: raise ValueError("number of steps must be non-negative")
    if (dtype := to_dtype(kwargs.pop("dtype", dtypes.default_float))) == dtypes.bool: raise ValueError("linspace with bool dtype is not supported")
    if steps == 1: return cls.full((1,), start, dtype=dtype, **kwargs)
    return (start + cls.arange(steps, dtype=dtypes.default_float, **kwargs) * ((stop - start) / (steps - 1))).cast(dtype)

  @classmethod
  def eye(cls, n:int, m:int|None=None, dtype:DTypeLike|None=None, device:str|tuple[str, ...]|None=None) -> Self:
    m_ = n if m is None else m
    if n < 0 or m_ < 0: raise ValueError(f"cannot have negative {n=}, {m_=}")
    out_dtype = to_dtype(dtype) if dtype is not None else dtypes.default_float
    return cls.arange(n, device=device).unsqueeze(-1).eq(cls.arange(m_, device=device)).cast(out_dtype)

  @classmethod
  def _tri(cls, r:sint, c:sint, diagonal=0, device:str|tuple[str, ...]|None=None) -> Self:
    return cls.arange(r, device=device).unsqueeze(-1) + diagonal <= cls.arange(c, device=device)

  def triu(self, diagonal:sint=0) -> Self:
    """
    Returns the upper triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(diagonal=-1).numpy())
    ```
    """
    return self._tri(self.shape[-2], self.shape[-1], diagonal, self.device).where(self, self.zeros_like())

  def tril(self, diagonal:sint=0) -> Self:
    """
    Returns the lower triangular part of the tensor, the other elements are set to 0.

    The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
    Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril(diagonal=-1).numpy())
    ```
    """
    return self._tri(self.shape[-2], self.shape[-1], diagonal+1, self.device).where(self.zeros_like(), self)

  def _pad_constant(self, pX, value:float) -> Self:
    # shrink first for negative pads, then pad with only non-negative values
    pX = tuple((0, 0) if p is None else p for p in pX)
    has_neg = not all(resolve(p >= 0) for p in flatten(pX))
    X = self.shrink(tuple((-smin(pB,0),smin(pA+s,s)) for (pB,pA),s in zip(pX, self.shape))) if has_neg else self
    pads = tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX) if has_neg else pX
    if value == 0: return MovementMixin.pad(X, pads)
    return MovementMixin.pad(X, pads) + MovementMixin.pad(X.ones_like(), pads).cast(dtypes.bool).where(0, value)

  def _broadcasted(self, y, reverse=False) -> tuple[Self, Self]:
    if not isinstance(y, type(self)): y = self.ufix(y)
    x, y = (self, y) if not reverse else (y, self)
    try:
      out_shape = _broadcast_shape(x.shape, y.shape)
      x, y = x._broadcast_to(out_shape), y._broadcast_to(out_shape)
    except RuntimeError: pass
    out_dtype = least_upper_dtype(x.dtype, y.dtype)
    return x.cast(out_dtype), y.cast(out_dtype)

  def dot(self, w:Self, dtype:DTypeLike|None=None) -> Self:
    """
    Performs dot product between two tensors.
    If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
    If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.

    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 0])
    print(a.dot(b).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.dot(b).numpy())
    ```
    """
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
    x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
    return (x*w).sum(-1, dtype=dtype).cast(least_upper_dtype(x.dtype, w.dtype) if dtype is None else to_dtype(dtype))

  def matmul(self, x:Self, reverse=False, dtype:DTypeLike|None=None) -> Self:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return x.dot(self, dtype=dtype) if reverse else self.dot(x, dtype=dtype)

  def __matmul__(self, x:Self) -> Self: return self.matmul(x)
  def __rmatmul__(self, x:Self) -> Self: return self.matmul(x, True)

  def min(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Returns the minimum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=1, keepdim=True).numpy())
    ```
    """
    return self._inverse().max(axis=axis, keepdim=keepdim)._inverse()

  def mean(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Returns the mean value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the mean is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=1).numpy())
    ```
    """
    output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
    numerator = self.cast(sum_acc_dtype(self.dtype)).sum(axis=axis, keepdim=keepdim)
    denominator = prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])
    return numerator.div(denominator).cast(output_dtype)  # type: ignore[arg-type]

  def var(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> Self:
    """
    Returns the variance of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=1).numpy())
    ```
    """
    squares = (self - self.mean(axis=axis, keepdim=True)).square()
    n = prod([si for si, so in zip(self.shape, squares.sum(axis=axis, keepdim=True).shape) if resolve(si != so)])
    reduced = squares.sum(axis=axis, keepdim=keepdim)
    denominator = reduced.const_like(n) - correction  # type: ignore[arg-type]
    # TODO: remove relu?
    return reduced.div(denominator.relu())

  def var_mean(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> tuple[Self, Self]:
    """
    Calculates the variance and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.var` and `Tensor.mean` to match `torch.var_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    var, mean = t.var_mean()
    print(var.numpy(), mean.numpy())
    ```
    """
    return self.var(axis, keepdim, correction), self.mean(axis, keepdim)

  def std(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> Self:
    """
    Returns the standard deviation of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the standard deviation is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=1).numpy())
    ```
    """
    return self.var(axis, keepdim, correction).sqrt()

  def std_mean(self, axis:int|Sequence[int]|None=None, keepdim=False, correction=1) -> tuple[Self, Self]:
    """
    Calculates the standard deviation and mean over the dimensions specified by dim.
    Syntactic sugar around `Tensor.std` and `Tensor.mean` to match `torch.std_mean`.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    std, mean = t.std_mean()
    print(std.numpy(), mean.numpy())
    ```
    """
    return self.std(axis, keepdim, correction), self.mean(axis, keepdim)

  def normalize(self, p:float=2.0, dim:int=1, eps:float=1e-12) -> Self:
    """
    Performs Lp normalization of the tensor along the specified dimension.

    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.normalize().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.normalize(p=1, dim=0).numpy())
    ```
    """
    if p == 0: return self / (self != 0).sum(dim, keepdim=True).maximum(eps)  # type: ignore[comparison-overlap]
    return self / self.abs().pow(p).sum(dim, keepdim=True).pow(1/p).maximum(eps)

  def logsumexp(self, axis=None, keepdim=False) -> Self:
    """
    Computes the log-sum-exp of the tensor along the specified axis or axes.

    The log-sum-exp function is a numerically stable way to compute the logarithm of the sum of exponentials.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the log-sum-exp is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().sum(axis=axis, keepdim=keepdim).log() + (m if keepdim else m.squeeze(axis))

  def _softmax(self, axis, dtype:DTypeLike|None=None) -> tuple[Self, Self, Self]:
    m = self - self.max(axis=axis, keepdim=True).detach()
    if dtype is not None: m = m.cast(to_dtype(dtype))
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1, dtype:DTypeLike|None=None) -> Self:
    """
    Applies the softmax function to the tensor along the specified axis.

    Rescales the elements of the tensor such that they lie in the range [0, 1] and sum to 1.

    You can pass in the `axis` keyword argument to control the axis along which the softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax(axis=0).numpy())
    ```
    """
    _, e, ss = self._softmax(axis, dtype)
    return e * ss.reciprocal()

  def log_softmax(self, axis=-1, dtype:DTypeLike|None=None) -> Self:
    """
    Applies the log-softmax function to the tensor along the specified axis.

    The log-softmax function is a numerically stable alternative to the softmax function in log space.

    You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax(axis=0).numpy())
    ```
    """
    m, _, ss = self._softmax(axis, dtype)
    return m - ss.log()

  def cat(self, *args:Self, dim:int=0) -> Self:
    """
    Concatenates self with other tensors in `args` along an axis specified by `dim`.
    All tensors must have the same shape except in the concatenating dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
    print(t0.cat(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.cat(t1, t2, dim=1).numpy())
    ```
    """
    dim = self._resolve_dim(dim)
    for arg in args: assert arg.ndim==self.ndim and all(ti==ai for i,(ti,ai) in enumerate(zip(self.shape, arg.shape)) if i!=dim)
    tensors = [self, *args]
    dim_cumsum = list(itertools.accumulate([t.shape[dim] for t in tensors], initial=0))
    padded = [t.pad(tuple((dim_cumsum[i], dim_cumsum[-1]-dim_cumsum[i+1]) if j==dim else None for j in range(t.ndim))) for i,t in enumerate(tensors)]
    return padded[0].usum(*padded[1:])

  def stack(self, *args:Self, dim:int=0) -> Self:
    """
    Concatenates self with other tensors in `args` along a new dimension specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])
    print(t0.stack(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.stack(t1, t2, dim=1).numpy())
    ```
    """
    # checks for shapes and number of dimensions delegated to cat
    unsqueezed = [t.unsqueeze(dim) for t in argfix(self, *args)]
    return unsqueezed[0].cat(*unsqueezed[1:], dim=dim)

  def _cumalu(self, axis:int, op:Ops) -> Self:
    assert self.shape[axis] != 0 and op in (Ops.ADD, Ops.MAX, Ops.MUL)
    pads = (None,)*(self.ndim-1) + ((self.shape[axis]-1, 0),)
    pooled = self.transpose(axis,-1)._pad_constant(pads, identity_element(op, self.dtype))._pool((self.shape[axis],))
    return getattr(pooled, {Ops.ADD: "sum", Ops.MAX: "max", Ops.MUL: "prod"}[op])(-1).transpose(axis, -1)

  def _split_cumalu(self, axis:int, op:Ops) -> Self:
    axis = self._resolve_dim(axis)
    if self.ndim == 0 or 0 in self.shape: return self
    # TODO: someday the optimizer will find this on its own
    # for now this is a two stage cumsum
    SPLIT = 256
    value = identity_element(op, self.dtype)
    if not isinstance(s:=self.shape[axis], int) or s <= SPLIT*2: return self._cumalu(axis, op)
    ret = self.transpose(axis,-1)._pad_constant((None,)*(self.ndim-1)+((round_up(s,SPLIT)-s,0),), value).unflatten(-1,(-1,SPLIT))._cumalu(-1, op)
    base = ret[..., -1]._cumalu(-1, op)._pad_constant((None,)*(ret.ndim-2) + ((1, -1),), value)
    base = base.unsqueeze(-1).expand(*base.shape, ret.shape[-1])
    def fix(x: Self) -> Self: return x.flatten(start_dim=-2)[..., -s:].transpose(axis,-1)
    return getattr(fix(ret), {Ops.ADD: "add", Ops.MAX: "maximum", Ops.MUL: "mul"}[op])(fix(base))

  def cumsum(self, axis:int=0) -> Self:
    """
    Computes the cumulative sum of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumsum(1).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.ADD)

  def cumprod(self, axis:int) -> Self:
    """
    Computes the cumulative product of the elements of the tensor along the specified `axis`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(1, 7).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumprod(axis=0).numpy())
    ```
    """
    return self._split_cumalu(axis, Ops.MUL)

  # helper function commonly used for indexing
  def _one_hot_along_dim(self, num_classes:sint, dim:int=-1) -> Self:
    from tinygrad.uop.ops import sint_to_uop
    if not dtypes.is_int(self.dtype): raise RuntimeError(f"_one_hot_along_dim expects int index tensor, getting {self.dtype}")
    offset = self.ndim - self._resolve_dim(dim) - 1
    dt = dtypes.int64 if sint_to_uop(num_classes).overflows(dtypes.int32) else dtypes.int32
    return self.eq(type(self).arange(num_classes, dtype=dt, device=self.device).reshape((num_classes,) + (1,) * offset))

  def one_hot(self, num_classes:int) -> Self:
    """
    Converts `self` to a one-hot tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, 3, 3, 4])
    print(t.one_hot(5).numpy())
    ```
    """
    if not dtypes.is_int(self.dtype): raise RuntimeError(f"expect integer dtype, getting {self.dtype=}")
    if num_classes < 0: raise ValueError(f"num_classes must be non-negative, got {num_classes}")
    return self[..., None]._one_hot_along_dim(num_classes).where(1, 0)

  # ***** functional nn ops *****

  def linear(self, weight:Self, bias:Self|None=None, dtype:DTypeLike|None=None) -> Self:
    """
    Applies a linear transformation to `self` using `weight` and `bias`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([1, 2])
    print(t.linear(weight, bias).numpy())
    ```
    """
    if dtype is not None:
      dt = to_dtype(dtype)
      return self.cast(dt).linear(weight.cast(dt), bias.cast(dt) if bias is not None else bias)
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def conv2d(self, weight:Self, bias:Self|None=None, groups=1, stride=1, dilation=1, padding:int|Sequence[int]=0,
             dtype:DTypeLike|None=None) -> Self:
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    padding_ = resolve_pool_pads(padding, len(HW))
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape),\
        f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    # conv2d is a pooling op (with padding, possibly negative — _pad_constant handles the shrink)
    x = self._pad_constant(((0,0),)*(self.ndim-len(HW)) + flat_to_grouped(padding_), 0.0)._pool(HW, stride, dilation)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW)\
      .permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])
    # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
    ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW))\
      .sum([-1-i for i in range(1+len(oyx))], keepdim=True, dtype=dtype).reshape(bs, cout, *oyx)
    return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

  def conv_transpose2d(self, weight:Self, bias:Self|None=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Self:
    """
    Applies a transposed convolution over a tensor with a given `weight` and optional `bias`.

    This function supports three different types of `padding`

    1. `int` (single value):
      Applies the same padding value uniformly to all spatial dimensions.

    2. `tuple[int, ...]` (length = number of spatial dimensions):
      Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

    3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
      Specifies explicit padding for each side of each spatial dimension in the form
      `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d transposed convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv_transpose2d(w).numpy())
    ```
    """
    x, w = self, weight.unflatten(0, (groups, -1)).transpose(1, 2).flip(*range(3, len(weight.shape)+1))
    HW = weight.shape[2:]
    padding = flat_to_grouped(resolve_pool_pads(padding, len(HW)))
    stride, dilation, output_padding = [make_tuple(x, len(HW)) for x in (stride, dilation, output_padding)]
    if any(s>1 for s in stride):
      # handle strides: (k) -> reshape -> (k,1) -> pad -> (k,s) -> reshape -> (k*s) -> shrink (k-(s-1))
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink_to(None, None, *[k-(s-1) for k,s in zip(x.shape[2:], stride)])
    padding = flatten((((k-1)*d-pB,(k-1)*d-pA+op) for k,d,(pB,pA),op in reversed(list(zip(HW, dilation, padding, output_padding)))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def layernorm(self, axis:int|tuple[int,...]=-1, eps:float=1e-5) -> Self:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    - Paper: https://arxiv.org/abs/1607.06450v1

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 10, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.layernorm()
    print(t.mean().item(), t.std().item())
    ```
    """
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Self|None, bias:Self|None, mean:Self, invstd:Self, axis:int|tuple[int, ...]=1) -> Self:
    """
    Applies Batch Normalization over a mini-batch of inputs.

    - Paper: https://arxiv.org/abs/1502.03167

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 4, 16, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.batchnorm(None, None, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
    print(t.mean().item(), t.std().item())
    ```
    """
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias is not None else ret

  # ***** loss ops *****

  def _do_reduction(self, reduction:ReductionStr="mean") -> Self:
    if reduction == "none": return self
    if reduction == "sum": return self.sum()
    if reduction == "mean": return self.mean()
    raise ValueError(f"{reduction=} must be one of {get_args(ReductionStr)}")

  def binary_crossentropy(self, Y:Self, reduction:ReductionStr="mean") -> Self:
    """
    Computes the binary cross-entropy loss between `self` and `Y`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0.1, 0.9, 0.2])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy(Y).item())
    ```
    """
    return (-Y*self.log() - (1-Y)*(1-self).log())._do_reduction(reduction)

  def binary_crossentropy_logits(self, Y:Self, reduction:ReductionStr="mean", pos_weight:Self|None=None) -> Self:
    """
    Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, -3])
    Y = Tensor([0, 1, 0])
    print(t.binary_crossentropy_logits(Y).item())
    ```
    """
    log_p, log_1_minus_p = self.logsigmoid(), (-self).logsigmoid()
    return (-((1 if pos_weight is None else pos_weight) * Y * log_p + (1-Y) * log_1_minus_p))._do_reduction(reduction)

  # ***** matrix ops *****

  def newton_schulz(self, steps:int, params:tuple[int, ...], eps:float=1.0e-7) -> Self:
    """
    Performs the newton-schulz algorithm for odd polynomials. The degree of the odd polynomial depends on the number of params.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(4, 4)
    print(t.newton_schulz(steps=5, params=(2,-1.5,0.5)).numpy())
    ```
    """
    assert self.ndim > 1, "NS only works for two or more dims"
    if self.shape[-2] > self.shape[-1]: return self.transpose(-2, -1).newton_schulz(steps, params, eps).transpose(-2, -1)
    G = self / (self.square().sum(axis=(-2, -1), keepdim=True).sqrt() + eps)
    for _ in range(steps):
      G = functools.reduce(lambda a, b: a + b, (p * functools.reduce(lambda x, y: (y @ y.transpose(-2, -1)) @ x, [G]*i, G)  # type: ignore[operator]
                                                 for i,p in enumerate(params)))
    return G

  # ***** tensor properties *****

  def nbytes(self) -> int:
    """
    Returns the total number of bytes of all elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([8, 9], dtype=dtypes.float)
    print(t.nbytes())
    ```
    """
    return int(self.numel()) * self.element_size()
