from tinygrad.tensor import Tensor
from tinygrad.helpers import prod, dtypes
from extra.onnx import safe_numpy
import numpy as np
import functools
from typing import Union, Tuple
import math

def Unsqueeze(data, axes):
  axes = [len(data.shape) + int(x) if x < 0 else int(x) for x in safe_numpy(axes)]
  ptr = 0
  new_shape = []
  for i in range(len(data.shape) + len(axes)):
    if i in axes: new_shape.append(1)
    else:
      new_shape.append(data.shape[ptr])
      ptr += 1
  return data.reshape(new_shape)

def Gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=0):
  ret = alpha * ((A.transpose() if transA == 1 else A) @ (B.transpose() if transB == 1 else B))
  if C is not None: ret += beta * (C if broadcast == 0 else C.reshape([-1 if i <  len(C.shape) else 1 for i in range(len(ret.shape))][::-1]))
  return ret

# TODO: this is copied from tinygrad/nn/__init__.py
# spatial is from opset 7 and has since been removed
def BatchNormalization(X, scale, B, input_mean, input_var, epsilon=1e-05, momentum=0.9, training_mode=0, spatial=1):
  if training_mode:
    x_detached = X.detach()
    current_mean = x_detached.mean(axis=(0,2,3))
    y = (x_detached - current_mean.reshape(shape=[1, -1, 1, 1]))
    current_var = (y*y).mean(axis=(0,2,3))
    current_invstd = current_var.add(epsilon).pow(-0.5)

    running_mean = input_mean * momentum + current_mean * (1 - momentum)
    running_var = input_var * momentum + current_var * (1 - momentum)

    return X.batchnorm(scale, B, current_mean, current_invstd), running_mean, running_var
  else:
    invstd = (input_var + epsilon)**-0.5
    return X.batchnorm(scale, B, input_mean, invstd)

def InstanceNormalization(x: Tensor, scale: Tensor, bias: Tensor, epsilon=1e-05):
  axis = tuple(range(2, len(x.shape)))
  mean = x.mean(axis=axis, keepdim=True)
  invstd = x.sub(mean).pow(2).mean(axis=axis, keepdim=True).add(epsilon).pow(-0.5)
  return x.sub(mean).mul(scale.reshape(shape=[-1, 1, 1])).mul(invstd).add(bias.reshape(shape=[-1, 1, 1]))

def LayerNormalization(x: Tensor, scale, bias, axis=-1, epsilon=1e-05, stash_type=1):
  assert stash_type == 1, "only float32 is supported"
  axis = tuple(i for i in range(axis if axis >= 0 else len(x.shape) + axis, len(x.shape)))
  mean = x.mean(axis=axis, keepdim=True)
  return x.layernorm(axis, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).pow(2).mean(axis=axis, keepdim=True).add(epsilon).sqrt().reciprocal()

def GroupNormalization(x: Tensor, scale: Tensor, bias: Tensor, num_groups, epsilon=1e-05):
  return x.reshape(x.shape[0], num_groups, -1).layernorm(axis=-1, eps=epsilon).mul(scale.unsqueeze(-1)).add(bias.unsqueeze(-1)).reshape(x.shape)

# onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
# numpy.pad: ((x1_begin, x1_end), (x2_begin, x2_end), ...)
def _format_padding(onnx_pads, ndims=None, axes=None):
  if ndims is None: ndims = len(onnx_pads) // 2
  if axes is None: axes = list(range(ndims))
  num_axes = len(axes)
  np_pads = [(0,0)] * ndims
  for i in range(num_axes):
    np_pads[axes[i]] = (onnx_pads[i], onnx_pads[i + num_axes])
  return np_pads

def _padding(X, pads=None, auto_pad="NOTSET", axes=None, constant_value=0.):
  assert auto_pad == "NOTSET"  # TODO: write this
  if pads is None: return X
  np_pads = _format_padding(pads, ndims=len(X.shape), axes=axes)
  zero_padded = X.pad(tuple(np_pads))
  constant_padder = Tensor(np.pad(np.zeros(X.shape, dtype=np.float32), np_pads, constant_values=constant_value), dtype=X.dtype)
  return zero_padded + constant_padder

def Pad(x: Tensor, pads: Union[Tensor, Tuple[int, ...]], constant_value: Tensor=None, axes: Tensor=None, mode="constant", value: float=0.):
  assert mode == "constant", f"WARNING: Pad mode {mode} not implemented"
  constant_value = value if constant_value is None else constant_value.numpy()
  seq_pads = list(pads) if isinstance(pads, tuple) else pads.numpy().astype(np.int32).tolist()
  seq_axes = axes.numpy().astype(np.int32).tolist() if axes is not None else None
  return _padding(x, seq_pads, axes=seq_axes, constant_value=constant_value)

def AveragePool(X, kernel_shape, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=1, pads=None, strides=1):
  assert ceil_mode == 0 and dilations == 1
  pixel_axes = tuple(range(len(X.shape)))[-2:]
  padding_included = _padding(X, pads, auto_pad, axes=pixel_axes).avg_pool2d(kernel_shape, stride=strides)
  if count_include_pad:
    return padding_included
  else:
    div = _padding(Tensor.ones(*X.shape), pads, auto_pad, axes=pixel_axes).avg_pool2d(kernel_shape, stride=strides)
    return padding_included / div

def MaxPool(X, kernel_shape, auto_pad="NOTSET", ceil_mode=0, dilations=1, pads=None, storage_order=0, strides=1):
  assert ceil_mode == 0 and storage_order == 0, f"WARNING: MaxPool ceil_mode {ceil_mode} and storage_order {storage_order} not implemented"
  return _padding(X, pads, auto_pad, constant_value=-np.inf, axes=tuple(range(len(X.shape)))[-2:]).max_pool2d(kernel_shape, stride=strides, dilation=dilations)

def Conv(X, W, B=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, strides=1):
  padding = [p for ps in zip(pads[:len(pads)//2][::-1], pads[len(pads)//2:][::-1]) for p in ps] if pads is not None else 0 # reorder padding
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=padding)

def ConvTranspose(X, W, B=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, output_shape=None, output_padding=0, strides=1):
  return X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=(pads[1], pads[3], pads[0], pads[2]) if pads is not None else 0, output_padding=output_padding)

# Reimplemented here because you need legacy RNG for passing ONNX tests.
def Dropout(data, ratio=0.5, training_mode=False, seed=None):
  if not training_mode: return data, Tensor.ones(*data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
  rng = np.random.RandomState(seed)
  ratio = ratio.lazydata.realize().toCPU()[0] if isinstance(ratio, Tensor) else ratio
  mask = Tensor((rng.random(data.shape) >= ratio), requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask

def Shape(data, end=None, start=0): return Tensor(list(data.shape)[start:end], dtype=dtypes.int64)
def Size(data): return prod(data if isinstance(data, list) else data.shape)

# TODO: this doesn't match Tensor.flatten behavior
def Flatten(input, axis=1):
  new_shape = (1, -1) if axis == 0 else (prod(input.shape[0:axis]), -1)
  return input.reshape(new_shape)

# TODO: abstract out the broadcast logic in tensor
def Expand(input, shape):
  x_shape, y_shape = input.shape, [int(x) for x in safe_numpy(shape)]
  # copied from _broadcasted
  x_shape, y_shape = [([1]*(max(len(x_shape), len(y_shape))-len(t_shape)) + list(t_shape)) for t_shape in [x_shape, y_shape]]
  shape_ret = tuple(max(sx, sy) for sx,sy in zip(x_shape, y_shape))
  return input.reshape(x_shape).expand(shape_ret)

def LRN(input, size, alpha=1e-4, beta=0.75, bias=1.0):
  bs, c, iy, ix = input.shape
  return input / input.mul(input).reshape(bs,1,c,iy*ix).pad2d((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1).reshape(bs,c,iy,ix).mul(alpha).add(bias).pow(beta)

def Identity(input): return input
def Neg(input): return -input
def Reciprocal(input): return input.reciprocal()
def Sqrt(input): return input.sqrt()
def Sign(input): return input.sign()
def Softsign(input): return input / (1+input.abs())
def Abs(input): return input.abs()
def Exp(input): return input.exp()
def Log(input): return input.log()
def Mish(input): return input.mish()
def HardSigmoid(input, alpha=0.2, beta=0.5): return (alpha*input + beta).clip(0, 1)
def HardSwish(input): return input * HardSigmoid(input, 1/6, 0.5)
def Celu(X, alpha=1.0): return X.relu() - (-alpha*(X/alpha).exp()+1).relu()
def Selu(X, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875): return gamma * (X.relu() - (-alpha*X.exp()+alpha).relu())
def Softplus(X): return X.softplus()
def PRelu(X:Tensor, slope:Tensor): return X.clip(0, float("inf")) + X.clip(float("-inf"), 0) * slope
def LeakyRelu(X, alpha=0.01): return X.leakyrelu(alpha)
def ThresholdedRelu(X, alpha=1.0): return (X-alpha).relu() + (X-alpha).relu().sign() * alpha
def Softmax_1(input, axis=1): return input.softmax(axis)
def Softmax_13(input, axis=-1): return input.softmax(axis)
Softmax = {1: Softmax_1, 13: Softmax_13}   # Softmax default axis changed
def LogSoftmax(input, axis=-1): return input.log_softmax(axis)
def Clip(input, min=None, max=None):
  if min is None: min = -3.4e38
  if max is None: max = 3.4e38
  return input.clip(min, max)

def Sin(x): return x.sin()
def Cos(x): return x.cos()
def Tan(x): return x.tan()
def Cosh(x): return (math.e ** x + math.e ** -x) / 2
def Sinh(x): return (math.e ** x - math.e ** -x) / 2
def Tanh(x): return Sinh(x) / Cosh(x)

def Less(x:Tensor,y:Tensor): return (x<y).cast(dtypes.bool)
def LessOrEqual(x:Tensor,y:Tensor): return (x<=y).cast(dtypes.bool)
def Greater(x:Tensor,y:Tensor): return (x>y).cast(dtypes.bool)
def GreaterOrEqual(x:Tensor,y:Tensor): return (x>=y).cast(dtypes.bool)
def Equal(x:Tensor,y:Tensor): return (x==y).cast(dtypes.bool)

def Max(*data_0): return functools.reduce(Tensor.maximum, data_0)
def Min(*data_0): return functools.reduce(Tensor.minimum, data_0)
def Sum(*data_0): return functools.reduce(Tensor.__add__, data_0)
def Mean(*data_0): return functools.reduce(Tensor.__add__, data_0) / len(data_0)

def _axes(axes, noop_with_empty_axes): return [int(x) for x in safe_numpy(axes)] if axes is not None else ([] if noop_with_empty_axes else None)

# ReduceProd would require a new llop
def ReduceMax(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMin(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSum(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMean(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSumSquare(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.square().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL1(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.abs().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL2(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.square().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).sqrt()
def ReduceLogSum(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).log()
def ReduceLogSumExp(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.exp().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).log()


def GlobalAveragePool(X): return X.mean(axis=tuple(range(2, len(X.shape))), keepdim=True)
def GlobalMaxPool(X): return X.max(axis=tuple(range(2, len(X.shape))), keepdim=True)
def OptionalHasElement(x: Tensor=None): return Tensor(x is not None and x.numel() > 0, dtype=dtypes.bool)
def OptionalGetElement(x: Tensor=None): return x if x is not None else Tensor([], dtype=dtypes.float32)

def Tile(input, repeats):
  repeats_ = [int(x) for x in safe_numpy(repeats)]
  new_shape = [x for i in range(len(input.shape)) for x in [1,input.shape[i]]]
  expand_shape = [x for r,s in zip(repeats_, input.shape) for x in [r,s]]
  final_shape = [r*s for r,s in zip(repeats_, input.shape)]
  return input.reshape(new_shape).expand(expand_shape).reshape(final_shape)

def Range(start, limit, delta): return Tensor.arange(safe_numpy(start)[0], safe_numpy(limit)[0], step=safe_numpy(delta)[0])
def Where(condition:Tensor,X:Tensor,Y:Tensor): return condition.where(X, Y).cast(X.dtype)

def And(x:Tensor, y:Tensor): return Where((x==y), x, Tensor.zeros(*x.shape)).cast(dtypes.bool)
def Or(x:Tensor, y:Tensor): return Where((x==y), x, Tensor.ones(*x.shape)).cast(dtypes.bool)
def Xor(x:Tensor, y:Tensor): return Where((x==y), Tensor.zeros(*x.shape), Tensor.ones(*x.shape)).cast(dtypes.bool)
def Not(x:Tensor): return Where((x==1), Tensor.zeros(*x.shape), Tensor.ones(*x.shape)).cast(dtypes.bool)

def Trilu(x: Tensor, k: Union[Tensor, int]=0, upper=1):
  k = int(k.numpy().item()) if k is not 0 else 0 # onnx passes k as a tensor int64 with one element, default is 0
  return x.triu(k) if upper else x.tril(k)

def ConstantOfShape(input, value:Tensor=None):
  if value is None: value=Tensor([0.0])
  shape = [int(x) for x in safe_numpy(input)]
  return Tensor.ones(*shape, dtype=value.dtype) * (value if shape[0]!=0 else 1)

# this is obviously wrong, but since we don't have types, it's better than nothing
def Cast(input, to):
  print(f"WARNING: attempting to cast to {to}")
  return input

# NOTE: since we only have one type, this is valid!
def CastLike(input, target_type):
  assert isinstance(target_type, Tensor), "can only CastLike Tensor"
  return input

def Binarizer(input, threshold=0.0): return input > threshold

def MeanVarianceNormalization(input, axis=(0, 2, 3)):
  data_mean = input.mean(axis=axis, keepdim=True)
  std = ((input**2).mean(axis=axis, keepdim=True) - data_mean**2).sqrt()
  return (input - data_mean) / (std + 1e-9)

def NegativeLogLikelihoodLoss(input, target, weight=None, ignore_index=None, reduction="mean"):
  N, C, i_shape = input.shape[0], input.shape[1], input.shape
  t_shape = target.shape
  if len(input.shape) != 3:
    input = input.reshape((N, C, -1))
    target = target.reshape((N, -1))
  if weight is not None:
    mask = target.unsqueeze(-1) == Tensor.arange(C,dtype=dtypes.int64).repeat((N, 1, 1))
    weight = (mask * weight).sum(axis=-1)
  if ignore_index is not None:
    cond = (target == ignore_index)
    weight = cond.where(0, weight) if weight is not None else cond.where(Tensor.zeros(*target.shape), 1)
  mask = target[:, None, :] ==  Tensor.arange(C).reshape([1, C] + [1]*(len(input.shape) -2))
  loss = (-mask * input).sum(axis=1) * (1 if weight is None else weight)
  if reduction == "mean": return loss.mean() if weight is None else loss.sum() / weight.sum()
  elif reduction == "sum": return loss.sum()
  return loss.reshape(t_shape) if len(i_shape) != 3 else loss

def OneHot(indices, depth, values, axis=-1):
  depth = int(safe_numpy(depth).item())
  indices, rank = (indices.cast(dtypes.float32) < 0).where(indices+depth, indices), len(indices.shape)
  if axis < 0: axis += rank + 1
  ls, rs = indices.shape[0:axis], indices.shape[axis: rank]
  cond = indices[:,None] == Tensor.arange(depth).reshape((1,) * len(ls) + (depth,) + (1,) * len(rs))
  return cond.where(values[1], values[0]).cast(values.dtype)

def Floor(x:Tensor): return x.floor()
def Ceil(x:Tensor): return x.ceil()
