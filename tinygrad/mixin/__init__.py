import functools
from typing import Self, Sequence, Literal, get_args
from tinygrad.mixin.elementwise import ElementwiseMixin
from tinygrad.mixin.reduce import ReduceMixin
from tinygrad.uop.ops import _broadcast_shape, resolve
from tinygrad.dtype import DTypeLike, dtypes, least_upper_dtype, sum_acc_dtype, to_dtype
from tinygrad.helpers import argfix, prod

ReductionStr = Literal["mean", "sum", "none"]


class OpMixin(ElementwiseMixin, ReduceMixin):
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
