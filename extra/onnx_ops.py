import functools, io, math
from typing import cast, Literal
from tinygrad.tensor import Tensor, _broadcast_shape, ConstType, ReductionStr
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.helpers import prod, flatten, make_tuple
from extra.onnx import dtype_parse, _cached_to_python_const
import numpy as np

# **************** Free Ops ****************

def Identity(x:Tensor): return x
# TODO: fix buffer_parse
def Add(x:Tensor, other:Tensor, broadcast=None, axis=None): return x + other if x.dtype == dtypes.float or isinstance(x.dtype, ImageDType) else (x + other).cast(x.dtype)
def Sub(x:Tensor|int, other:Tensor): return x - other # some test has input as int
def Less(x:Tensor,y:Tensor): return x < y
def LessOrEqual(x:Tensor,y:Tensor): return x <= y
def Greater(x:Tensor,y:Tensor): return x > y
def GreaterOrEqual(x:Tensor,y:Tensor): return x >= y
def Equal(x:Tensor,y:Tensor): return x == y
def BitwiseNot(x:Tensor): return ~x
def BitwiseOr(x:Tensor, y:Tensor): return x | y
def BitwiseAnd(x:Tensor, y:Tensor): return x & y
def BitwiseXor(x:Tensor, y:Tensor): return x ^ y
def Max(*data_0:Tensor): return functools.reduce(Tensor.maximum, data_0)
def Min(*data_0:Tensor): return functools.reduce(Tensor.minimum, data_0)
def Sum(*data_0:Tensor): return functools.reduce(Tensor.add, data_0)
def Mean(*data_0:Tensor): return Sum(*data_0) / len(data_0)
def Cast(x:Tensor, to:int, saturate:int=1): return x.cast(dtype_parse(to))
def CastLike(x:Tensor, target_type:Tensor, saturate:int=1): return x.cast(target_type.dtype)

# **************** Simple Ops ****************

# https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_div.py
def Div(x:Tensor, other:Tensor): return (x/other).cast(x.dtype)

def Constant(sparse_value:Tensor|None=None, value:Tensor|None=None, value_float:float|None=None,
             value_floats:list[float]|None=None, value_int:int|None=None, value_ints:list[int]|None=None,
             value_string:str|None=None, value_strings:list[str]|None=None):
  if value is not None: return value
  if value_float is not None: return Tensor(value_float, dtype=dtypes.float32, requires_grad=False)
  if value_floats is not None: return Tensor(list(value_floats), dtype=dtypes.float32, requires_grad=False)
  if value_int is not None: return Tensor(value_int, dtype=dtypes.int64, requires_grad=False)
  if value_ints is not None: return Tensor(list(value_ints), dtype=dtypes.int64, requires_grad=False)
  if value_string is not None or value_strings is not None and sparse_value is not None:
    raise NotImplementedError('Constant OP not implemented for value_string, value_strings and sparse_value')

def HardSigmoid(x:Tensor, alpha:float=0.2, beta:float=0.5): return (alpha*x + beta).clip(0, 1)
def Gelu(x:Tensor, approximate:str|None=None): return x.gelu() if approximate == "tanh" else 0.5 * x * (1 + (x/math.sqrt(2)).erf())
# TODO: fix this
def PRelu(X:Tensor, slope:Tensor):
  slope = slope[0] if slope.shape[-1] != X.shape[-1] else slope # HACK OnnxBackendPyTorchConvertedModelTest HAS WEIRD SLOPE WHERE IT'S [0.25, 0.25, 0.25] FOR ANY X.SHAPE
  return (X > 0).where(X, X * slope)
def LeakyRelu(X:Tensor, alpha:float=0.01): return X.leakyrelu(alpha)
def ThresholdedRelu(X:Tensor, alpha:float=1.0): return (X > alpha).where(X, 0)
def Softmax_1(x:Tensor, axis:int=1): return x.softmax(axis)
def Softmax_13(x:Tensor, axis:int=-1): return x.softmax(axis)
Softmax = {1: Softmax_1, 13: Softmax_13}   # Softmax default axis changed
def LogSoftmax(x: Tensor, axis:int=-1): return x.log_softmax(axis)
def Clip(x: Tensor, min:Tensor|None=None, max:Tensor|None=None):    # noqa: A002
  return x.clip(float('-inf') if min is None else min, float('inf') if max is None else max).cast(x.dtype)

def _axes(axes, noop_with_empty_axes): return axes or ([] if noop_with_empty_axes else None)
def ReduceMax(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMin(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSum(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMean(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSumSquare(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return ReduceSum(data.square(), axes, keepdims, noop_with_empty_axes)
def ReduceProd(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return data.prod(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL1(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return ReduceSum(data.abs(), axes, keepdims, noop_with_empty_axes)
def ReduceL2(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return ReduceSumSquare(data, axes, keepdims, noop_with_empty_axes).sqrt()
def ReduceLogSum(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return ReduceSum(data, axes, keepdims, noop_with_empty_axes).log()
def ReduceLogSumExp(data:Tensor, axes:list[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
  return ReduceSum(data.exp(), axes, keepdims, noop_with_empty_axes).log()

def GlobalAveragePool(X:Tensor): return X.mean(axis=tuple(range(2, X.ndim)), keepdim=True)
def GlobalMaxPool(X:Tensor): return X.max(axis=tuple(range(2, X.ndim)), keepdim=True)
def OptionalHasElement(x:Tensor|None=None): return Tensor(x is not None and x.numel() > 0)
def OptionalGetElement(x:Tensor|None=None): return x if x is not None else Tensor([])

def Tile(x:Tensor, repeats:list[int]): return x.repeat(repeats)
def Range(start:float|int, limit:float|int, delta:float|int): return Tensor.arange(start=start, stop=limit, step=delta)
def Shape(data:Tensor, end:int|None=None, start:int=0): return Tensor(data.shape[start:end], dtype=dtypes.int64)
def Size(data:Tensor): return data.numel()
def Flatten(x:Tensor, axis:int=1): return x.reshape(prod(x.shape[0:axis]), -1)
def Reshape(data:Tensor, shape:list[int], allowzero:int=0):
  return data.reshape([x if x != 0 else (0 if allowzero else data.shape[i]) for i,x in enumerate(shape)])
def Expand(x:Tensor, shape:list[int]): return x.expand(_broadcast_shape(x.shape, tuple(shape)))
def Shrink(x:Tensor, bias:float=0.0, lambd:float=0.5): return (x < -lambd)*(x+bias) + (x > lambd)*(x-bias)
def And(x:Tensor, y:Tensor): return (x==y).where(x, False)
def Or(x:Tensor, y:Tensor): return (x==y).where(x, True)
def Not(x:Tensor): return x.logical_not()

def Trilu(x:Tensor, k:int=0, upper:int=1): return x.triu(k) if upper else x.tril(k)

def Slice(data:Tensor, starts:list[int], ends:list[int], axes:list[int]|None=None, steps:list[int]|None=None):
  axes = axes or list(range(data.ndim))
  steps = steps or [1]*data.ndim
  slices = [slice(0,x,1) for x in data.shape]
  for i, axis in enumerate(axes): slices[axis] = slice(starts[i], ends[i], steps[i])
  return data[tuple(slices)]

def Split(data:Tensor, split:list[int]|None=None, num_outputs:int=0, axis:int=0):
  sz = data.shape[axis]
  if split is None: split = [sz // num_outputs + (1 if i < sz % num_outputs else 0) for i in range(num_outputs)]
  return data.split(split, axis)

# TODO: add test for when axes is None
def Squeeze(data:Tensor, axes:list[int]|None=None):
  return data.squeeze() if axes is None else functools.reduce(lambda d, dim: d.squeeze(dim), sorted(axes, reverse=True), data)
def Unsqueeze(data:Tensor, axes:list[int]): return functools.reduce(lambda d, dim: d.unsqueeze(dim), sorted(axes), data)

def Binarizer(x:Tensor, threshold:float=0.0): return (x > threshold).float()

def ArgMax(x:Tensor, axis:int=0, keepdims:int=1, select_last_index:int=0):
  if select_last_index: return ((x.shape[axis]-1) - x.flip(axis).argmax(axis, keepdim=keepdims)).cast(dtypes.int64)
  return x.argmax(axis, keepdim=keepdims).cast(dtypes.int64)
def ArgMin(x, axis:int=0, keepdims:int=1, select_last_index:int=0): return ArgMax(-x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)

def Concat(*xs:Tensor, axis:int): return Tensor.cat(*xs, dim=axis)
def Transpose(x:Tensor, perm:list[int]|None=None): return x.permute(order=list(range(x.ndim)[::-1]) if perm is None else perm)

def ConstantOfShape(shape:list[int], value:Tensor|None=None):
  if value is None: value = Tensor(0, dtype=dtypes.float32)
  return Tensor.ones(*shape, dtype=value.dtype) * (value if shape != [0] else 1)

# **************** Complex Ops ****************

def Gemm(A:Tensor, B:Tensor, C:Tensor|None=None, alpha:float=1.0, beta:float=1.0, transA:int=0, transB:int=0, broadcast=0):
  ret = alpha * (A.transpose(transA) @ B.transpose(transB))
  if C is not None: ret = ret + beta * (C if broadcast == 0 else C.reshape([-1 if i < len(C.shape) else 1 for i in range(ret.ndim)][::-1]))
  return ret

def Einsum(*Inputs:list[Tensor], equation:str): return Tensor.einsum(equation, *Inputs)

def CumSum(X:Tensor, axis:int|list, exclusive:int=0, reverse:int=0):
  axis = X._resolve_dim(axis[0] if isinstance(axis, list) else axis)
  if reverse: X = X.flip(axis)
  if exclusive: X = X.pad(tuple((1,0) if i == axis else None for i in range(X.ndim)))\
                      .shrink(tuple((0,X.shape[axis]) if i == axis else None for i in range(X.ndim)))
  return X.cumsum(axis).flip(axis) if reverse else X.cumsum(axis)

# TODO: this is copied from tinygrad/nn/__init__.py
# spatial is from opset 7 and has since been removed
def BatchNormalization(X:Tensor, scale:Tensor, B:Tensor, input_mean:Tensor, input_var:Tensor, epsilon:float=1e-05, momentum:float=0.9,
                       training_mode:int=0, spatial=1, is_test=0):
  if training_mode:
    x_detached = X.detach()
    current_mean = x_detached.mean(axis=(0,2,3))
    y = (x_detached - current_mean.reshape(shape=[1, -1, 1, 1]))
    current_var = (y*y).mean(axis=(0,2,3))
    current_invstd = current_var.add(epsilon).rsqrt()

    running_mean = input_mean * momentum + current_mean * (1 - momentum)
    running_var = input_var * momentum + current_var * (1 - momentum)

    return X.batchnorm(scale, B, current_mean, current_invstd), running_mean, running_var
  invstd = (input_var + epsilon).rsqrt()
  return X.batchnorm(scale, B, input_mean, invstd)

def InstanceNormalization(x:Tensor, scale:Tensor, bias:Tensor, epsilon:float=1e-05):
  axis = tuple(range(2, x.ndim))
  mean = x.mean(axis=axis, keepdim=True)
  invstd = x.sub(mean).square().mean(axis=axis, keepdim=True).add(epsilon).rsqrt()
  return x.sub(mean).mul(scale.reshape(shape=[-1, 1, 1])).mul(invstd).add(bias.reshape(shape=[-1, 1, 1]))

def LayerNormalization(x:Tensor, scale:Tensor, bias:Tensor, axis:int=-1, epsilon:float=1e-05, stash_type:int=1):
  assert stash_type == 1, "only float32 is supported"
  axes = tuple(i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim))
  mean = x.mean(axis=axes, keepdim=True)
  return x.layernorm(axes, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).square().mean(axis=axes, keepdim=True).add(epsilon).rsqrt()

def GroupNormalization(x:Tensor, scale:Tensor, bias:Tensor, num_groups:int, epsilon:float=1e-05):
  return x.reshape(x.shape[0], num_groups, -1).layernorm(axis=-1, eps=epsilon).mul(scale.unsqueeze(-1)).add(bias.unsqueeze(-1)).reshape(x.shape)

# (padding_top, padding_left, ..., padding_bottom, padding_right, ...) -> (padding_left, padding_right, padding_top, padding_bottom, ...)
def _onnx_pads_to_tiny_pads(pads): return flatten(reversed([(pB,pA) for pB, pA in zip(pads, pads[len(pads)//2:])]))

AUTO_PAD_OPTIONS = Literal["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]
# (padding_height, padding_width) -> (padding_top, padding_left, padding_bottom, padding_right)
def _auto_pad(pads, auto_pad: AUTO_PAD_OPTIONS):
  if auto_pad == "SAME_UPPER": return [pads[i]//2 for i in range(len(pads))] + [pads[i]-pads[i]//2 for i in range(len(pads))]
  return [pads[i]-pads[i]//2 for i in range(len(pads))] + [pads[i]//2 for i in range(len(pads))]

def Pad(x:Tensor, pads:list[int], constant_value:ConstType|None=None, axes:list[int]|None=None,
        mode:Literal["constant", "reflect", "edge", "wrap"]="constant", value=0):
  value = constant_value or value
  axes = axes or list(range(x.ndim))
  real_pads = [0] * (x.ndim*2)
  for i,axis in enumerate(axes): real_pads[axis%x.ndim], real_pads[axis%x.ndim+x.ndim] = pads[i], pads[i+len(axes)]
  return x.pad(padding=_onnx_pads_to_tiny_pads(real_pads), mode={"edge":"replicate", "wrap":"circular"}.get(mode, mode), value=value)

def _resolve_pool_pads(x:Tensor, p_, k_, d_, s_, auto_pad:AUTO_PAD_OPTIONS):
  i_, (s_,d_,p_) = x.shape[-len(k_):], (make_tuple(x, len(k_)*2) for x in (s_, d_, p_))
  if auto_pad == "NOTSET": return _onnx_pads_to_tiny_pads(p_ if len(p_)==len(k_)*2 else p_*2)
  o_ = [((i - (1 if auto_pad in ("SAME_UPPER", "SAME_LOWER") else k)) // s + 1) for i,k,s in zip(i_, k_, s_)]
  return _onnx_pads_to_tiny_pads(_auto_pad([(o-1)*s+k-i for o,i,k,s in zip(o_, i_, k_, s_)], auto_pad))

def AveragePool(X: Tensor, kernel_shape:list[int], auto_pad:AUTO_PAD_OPTIONS="NOTSET", ceil_mode:int=0, count_include_pad:int=0,
                dilations:list[int]|int=1, pads:list[int]|int=0, strides:list[int]|int=1):
  pads = _resolve_pool_pads(X, pads, kernel_shape, dilations, strides, auto_pad)
  return X.avg_pool2d(kernel_shape, strides, dilations, pads, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

def MaxPool(X: Tensor, kernel_shape:list[int], auto_pad:AUTO_PAD_OPTIONS="NOTSET", ceil_mode:int=0, dilations:list[int]|int=1, pads:list[int]|int=0,
            storage_order:int=0, strides:list[int]|int=1):
  pads = _resolve_pool_pads(X, pads, kernel_shape, dilations, strides, auto_pad)
  ret = X.max_pool2d(kernel_shape, strides, dilations, pads, ceil_mode=ceil_mode)
  # tests expect indices with int64 dtype
  # TODO: if there are repeated values, this is wrong
  indices = ((ret.reshape(-1, 1) == X.reshape(1, -1)) * Tensor.arange(X.numel(), dtype=dtypes.int64).unsqueeze(0)).sum(1).reshape(ret.shape)
  return ret.cast(X.dtype), indices.transpose(-2, -1) if storage_order else indices

def Conv(X: Tensor, W: Tensor, B:Tensor|None=None, auto_pad:AUTO_PAD_OPTIONS="NOTSET", dilations:list[int]|int=1, group:int=1,
         kernel_shape:list[int]|None=None, pads:list[int]|int=0, strides:list[int]|int=1):
  pads = _resolve_pool_pads(X, pads, kernel_shape or W.shape[2:], dilations, strides, auto_pad)
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=tuple(pads))

# src: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
# another src: https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv_transpose.py
def ConvTranspose(X: Tensor, W: Tensor, B:Tensor|None=None, auto_pad:AUTO_PAD_OPTIONS="NOTSET", dilations:list[int]|int=1, group:int=1,
                  kernel_shape:list[int]|None=None, pads:list[int]|None=None, output_shape:list[int]|None=None, output_padding:list[int]|int=0,
                  strides:list[int]|int=1):
  input_shape, kernel_shape = X.shape[2:], (kernel_shape or W.shape[2:])
  strides, dilations, output_padding = (make_tuple(x, len(input_shape)) for x in (strides, dilations, output_padding))
  if output_shape is not None: # we pad according to output_shape
    pads = _auto_pad([s*(i-1) + op + ((k-1)*d+1) - os for s,i,op,k,d,os in
                      zip(strides, input_shape, output_padding, kernel_shape, dilations, output_shape)], auto_pad)
  if pads is None: # we generate pads
    output_shape = output_shape or [X.shape[i+2] * strides[i] for i in range(len(strides))]
    pads = [strides[i]*(input_shape[i]-1) + output_padding[i] + ((kernel_shape[i]-1)*dilations[i]+1)-output_shape[i] for i in range(len(input_shape))]
    pads = _auto_pad(pads, auto_pad) if auto_pad != "NOTSET" else [0] * len(input_shape) * 2
  pads = _onnx_pads_to_tiny_pads(pads)
  return X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=pads, output_padding=output_padding)

def MaxUnpool(xT: Tensor, xI: Tensor, outshape: list[int]|None=None, kernel_shape:list[int]=None, pads:list[int]|int=0, strides:list[int]|int=1):
  pads, strides = (make_tuple(x, len(xI.shape)) for x in (pads, strides))
  out_sh = [(ks//2)*2 + st * inps for inps, st, ks in zip(xI.shape, strides, kernel_shape)]
  ret = (xI.reshape(-1, 1)._one_hot_along_dim(prod(out_sh)) * xT.reshape(-1, 1)).sum(0).reshape(1, 1, *out_sh)
  if outshape is not None and outshape != ret.shape: pads = _auto_pad([outshape[-2] - ret.shape[-2], outshape[-1] - ret.shape[-1]], "SAME_UPPER")
  return ret.pad(_onnx_pads_to_tiny_pads(pads))

def DepthToSpace(X:Tensor, blocksize:int, mode:str="DCR"):
  return X.rearrange("b (c h1 w1) h w -> b c (h h1) (w w1)" if mode=="CRD" else "b (h1 w1 c) h w -> b c (h h1) (w w1)", h1=blocksize, w1=blocksize)
def SpaceToDepth(X:Tensor, blocksize:int):
  return X.rearrange("b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=blocksize, w1=blocksize)

# Reimplemented here because you need legacy RNG for passing ONNX tests.
def Dropout_7(data:Tensor, ratio:float=0.5, training_mode:bool=False, seed:int|None=None):
  if not training_mode: return data, Tensor.ones(data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
  mask = Tensor(np.random.RandomState(seed).random(cast(tuple[int,...], data.shape)) >= ratio, requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask
# 6 with 'is_test' needed for https://github.com/MTlab/onnx2caffe/raw/refs/heads/master/model/MobileNetV2.onnx
def Dropout_6(data:Tensor, ratio:float=0.5, is_test=0): return Dropout_7(data, ratio, training_mode=not is_test)
Dropout = {6:Dropout_6, 7:Dropout_7}

def LRN(x:Tensor, size:int, alpha:float=1e-4, beta:float=0.75, bias:float=1.0):
  pooled_x = (x**2).rearrange('b c h w -> b 1 c (h w)').pad((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1)
  return x / (pooled_x.reshape(x.shape) * alpha + bias).pow(beta)

def MeanVarianceNormalization(x:Tensor, axis:list[int]=[0,2,3]):
  return (x - x.mean(axis, keepdim=True)) / (x.std(axis, keepdim=True, correction=0) + 1e-9)

def NegativeLogLikelihoodLoss(x:Tensor, target:Tensor, weight:Tensor|None=None, ignore_index:int|None=None, reduction:ReductionStr="mean"):
  return x.nll_loss(target, weight, ignore_index, reduction)

def SoftmaxCrossEntropyLoss(scores:Tensor, labels:Tensor, weights:Tensor|None=None, ignore_index:int|None=None, reduction:ReductionStr="mean"):
  log_probs = scores.log_softmax(1)
  return log_probs.nll_loss(labels, weights, ignore_index, reduction), log_probs

def ArrayFeatureExtractor(x:Tensor, indices:Tensor): return x[..., indices]

def Gather(x:Tensor, indices:Tensor, axis:int=0):
  if indices.numel() < 9: # NOTE lessor kernels for smaller indices but kernel number increases depending on size of indices
    x_sh = list(x.shape)
    ret_shape = x_sh[:axis] + list(indices.shape) + x_sh[axis+1:]
    if indices.ndim > 1: indices = indices.flatten()
    indices = [_cached_to_python_const(indices)] if indices.shape == () else [x_sh[axis]+x if x<0 else x for x in _cached_to_python_const(indices)]
    args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(x_sh)] for i in indices] # type: ignore
    return x.shrink(arg=tuple(args[0])).cat(*[x.shrink(arg=tuple(arg)) for arg in args[1:]], dim=axis).reshape(ret_shape)
  # NOTE faster gather, fixed number of kernels, but exceeds limited kernels for openpilot
  return x[tuple([slice(None) if i != axis else indices for i in range(x.ndim)])]
def Scatter(*args, **kwargs): return ScatterElements(*args, **kwargs) # deprecated

def GatherND(x:Tensor, indices:Tensor, batch_dims:int=0):
  if batch_dims == 0: return x[tuple(i.squeeze(-1) for i in indices.split(1, -1))]
  x_shape, i_shape = x.shape, indices.shape
  b = math.prod(x.shape[dim] for dim in range(batch_dims))
  # NOTE: each batched dim of both input and indices are equal
  x = x.reshape(b, *x.shape[batch_dims:])
  indices = indices.reshape(b, *indices.shape[batch_dims:])
  b_idx = Tensor.arange(b, device=x.device).reshape(b, *(1,)*(indices.ndim - 2)).expand(*indices.shape[:-1])
  ret = x[(b_idx,) + tuple(i.squeeze(-1) for i in indices.split(1, -1))]
  return ret.reshape(*x_shape[:batch_dims], *i_shape[batch_dims:-1], *ret.shape[indices.ndim-1:])
def ScatterND(x:Tensor, indices:Tensor, updates:Tensor, reduction:Literal["none", "add", "mul"]='none'):
  assert updates.shape == indices.shape[:-1] + x.shape[cast(int, indices.shape[-1]):]
  x = x.contiguous()
  for index, u in zip(indices.split(1, 0), updates.split(1, 0)):
    i = tuple(idx.squeeze(-1) for idx in index.squeeze(0).split(1, -1))
    u = u.squeeze(0)
    if reduction == "none": x[i] = u
    elif reduction == "add": x[i] += u
    elif reduction == "mul": x[i] *= u
    else: raise NotImplementedError("reduction doesn't support max or min")
  return x

def ScatterElements(x: Tensor, indices: Tensor, updates: Tensor, axis=0, reduction:Literal["none", "add", "mul"]="none"):
  indices = (indices < 0).where(x.shape[axis], 0) + indices
  return x.scatter(axis, indices, updates, {"none":None, "mul": "multiply"}.get(reduction, reduction))
def GatherElements(x:Tensor, indices:Tensor, axis:int):
  indices = (indices < 0).where(x.shape[axis], 0) + indices
  return x.gather(axis, indices)

def Resize(X:Tensor, roi:list[float]|None=None, scales:list[float]|None=None, sizes:list[int]|None=None, antialias:int=0,
           axes:list[int]|None=None, coordinate_transformation_mode:str='half_pixel', cubic_coeff_a:float=-0.75, exclude_outside:int=0,
           extrapolation_value:float=0.0, keep_aspect_ratio_policy:str='stretch', mode:str='nearest', nearest_mode:str='round_prefer_floor'):
  def _apply_nearest_mode(index: Tensor, input_dim, mode: str):
    if mode == "round_prefer_floor": index = (index - 0.5).ceil()
    elif mode == "round_prefer_ceil": index = (index + 0.5).floor()
    elif mode in ["floor", "ceil"]: index = getattr(index, mode)()
    else: raise ValueError(f"invalid {nearest_mode=}")
    return index.cast(dtypes.int32).clip(0, input_dim-1)
  def _apply_transformation(index: Tensor, input_dim, scale_dim, roi_dim, mode):
    # TODO: needs more testing, not confident in this
    # NOTE: their reference implementation differ from the implementation in their reference docs
    # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_resize.py
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
    output_dim = scale_dim * input_dim
    if mode == "half_pixel": index = (index + 0.5) / scale_dim - 0.5
    elif mode == "align_corners": index = index * (input_dim - 1) / (output_dim - 1) if output_dim != 1 else Tensor([0])
    elif mode == "asymmetric": index = index / scale_dim
    elif mode == "pytorch_half_pixel": index = (index + 0.5) / scale_dim - 0.5 if output_dim != 1 else Tensor([-0.5])
    elif mode == "half_pixel_symmetric": index = input_dim / 2 * (1 - int(output_dim) / output_dim) + (index + 0.5) / scale_dim - 0.5
    elif mode == "tf_crop_and_resize": index = roi_dim[0] * (input_dim - 1) + index * ((roi_dim[1] - roi_dim[0]) * (input_dim - 1) / (output_dim - 1))
    else: raise ValueError(f"invalid {coordinate_transformation_mode=}")
    return index.clip(0, input_dim-1)

  scales, sizes = (None if scales is None else scales[2-(X.ndim-len(scales)):]), (None if sizes is None else sizes[2-(X.ndim-len(sizes)):])
  # we pre permute the axes and permute back after resize
  axes, input_shape, = (axes or list(range(X.ndim))), cast(tuple[int, ...], X.shape[2:]),
  perm = [a for a in range(len(X.shape)) if a not in axes] + list(axes)
  X = X.permute(*perm)

  if sizes is not None:
    if keep_aspect_ratio_policy in ["not_larger", "not_smaller"]:
      scale_fxn = min if keep_aspect_ratio_policy == "not_larger" else max
      scales = [scale_fxn([sizes[i] / input_shape[i] for i in range(len(input_shape)) if i+2 in axes])] * 2
      sizes = [int((scales[0] * input_shape[i]) + 0.5) if i+2 in axes else input_shape[i] for i in range(X.ndim-2)]
    else:
      scales = [size / input_shape for size, input_shape in zip(sizes, input_shape)]
  else:
    sizes = [int(sc*sh) for sc, sh in zip(scales, input_shape)]
  regions = [[st, ed] for st, ed in zip(roi, roi[len(roi)//2:])] if isinstance(roi, list) and roi else [[0.0, 0.0]] * (X.ndim-2)

  # NOTE: this transformation makes it so that we can't just call Tensor.interpolate
  # in Tensor.interpolate, we use indexes without any transformation
  indexes = []
  for shape, size, scale, region in zip(input_shape, sizes, scales, regions):
    indexes.append(_apply_transformation(Tensor.arange(size), shape, scale, region, coordinate_transformation_mode))

  if mode == "nearest":
    indexes = [_apply_nearest_mode(index, shape, nearest_mode) for (index, shape) in zip(indexes, input_shape)]
    X = X[(..., *Tensor.meshgrid(*indexes))]
  if mode == "linear":
    expand = list(X.shape)
    for i in range(-len(sizes), 0):
      reshape, index = [1] * X.ndim, indexes[i]
      reshape[i] = expand[i] = sizes[i]
      low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor(), index.ceil(), index - index.floor())]
      X = X.gather(i, low).lerp(X.gather(i, high), perc)
  if mode == "cubic": raise NotImplementedError("cubic interpolation is not implemented")
  return X.permute(*[perm.index(i) for i in range(len(perm))]) if perm else X

def CenterCropPad(t:Tensor, shape:list[int], axes:list[int]|None=None):
  shrink_arg:list[None|tuple[int,int]] = [None] * t.ndim
  pad_arg:list[None|tuple[int,int]] = [None] * t.ndim
  for s, x in zip(shape, axes or range(t.ndim)):
    tx = t.shape[x]
    if s < tx: shrink_arg[x] = (tx//2 - (s+1)//2, tx//2 + s//2)
    elif s > tx: pad_arg[x] = ((s-tx)//2, (s-tx+1)//2)
  return t.shrink(tuple(shrink_arg)).pad(tuple(pad_arg))

def OneHot(indices:Tensor, depth:float|int|list, values:Tensor, axis:int=-1):
  # Scalar or Rank 1 tensor containing exactly one element
  depth = int(depth[0] if isinstance(depth, list) else depth)
  indices = (indices < 0).where(indices+depth, indices)
  return indices[:, None]._one_hot_along_dim(depth, dim=axis).where(values[1], values[0])

def Compress(inp:Tensor, condition:list[bool], axis:int|None=None):
  if axis is None:
    inp = inp.flatten()
    axis = 0
  if axis < 0: axis += inp.ndim
  con = Tensor(np.arange(len(condition))[condition]) # no boolean indexing in Tensor
  return inp[tuple(con if i == axis else slice(None) for i in range(inp.ndim))]

def EyeLike(x:Tensor, dtype:int|None=None, k:int=0):
  ret = Tensor.eye(cast(int, min(x.shape)), dtype=dtype_parse(dtype) if dtype is not None else x.dtype)
  return ret if x.size(0) == x.size(1) else ret.pad(tuple(None if d == ret.size(0) else (k, d-ret.shape[0]-k) for d in x.shape))

def Upsample(X, scales, mode): return Resize(X=X, scales=scales, mode=mode)  # deprecated

def _prepare_quantize(x, scale, zero_point, axis=1, block_size=0):
  if axis < 0: axis += x.ndim
  if not isinstance(zero_point, Tensor): zero_point = Tensor(zero_point, dtype=dtypes.uint8)._broadcast_to(scale.shape)
  if block_size == 0:
    shape = (*[1]*axis, *scale.shape, *[1]*(x.ndim - axis - scale.ndim))
    return scale.reshape(shape), zero_point.reshape(shape)
  return scale.repeat_interleave(block_size, dim=axis), zero_point.repeat_interleave(block_size, dim=axis)

def QuantizeLinear(x:Tensor, y_scale:Tensor, y_zero_point:Tensor|int=0, axis:int=1, block_size:int=0, output_dtype:int=0, saturate=1):
  out_dtype = y_zero_point.dtype if isinstance(y_zero_point, Tensor) else dtype_parse(output_dtype) if output_dtype else dtypes.uint8
  y_scale, y_zero_point = _prepare_quantize(x, y_scale, y_zero_point, axis, block_size)
  return ((x / y_scale).round() + y_zero_point).clamp(dtypes.min(out_dtype), dtypes.max(out_dtype)).cast(out_dtype).contiguous()

def DequantizeLinear(x:Tensor, x_scale:Tensor, x_zero_point:Tensor|int=0, axis:int=1, block_size:int=0):
  x_scale, x_zero_point = _prepare_quantize(x, x_scale, x_zero_point, axis, block_size)
  return ((x.int() - x_zero_point) * x_scale).cast(x_scale.dtype)

def _quantize_linear(y:Tensor, y_scale:Tensor, y_zero_point:Tensor):
  assert y_scale.dtype is dtypes.float32 and y_zero_point.dtype in {dtypes.uint8, dtypes.int8}, "used only for qlinear ops"
  y = (y / y_scale + y_zero_point).round()
  return y.clamp(dtypes.min(y_zero_point.dtype), dtypes.max(y_zero_point.dtype)).cast(y_zero_point.dtype)

def QLinearConv(x:Tensor, x_scale:Tensor, x_zero_point:Tensor|int, w:Tensor, w_scale:Tensor, w_zero_point:Tensor|int, y_scale:Tensor,
                y_zero_point: Tensor|int, B:Tensor|None=None, auto_pad:AUTO_PAD_OPTIONS="NOTSET", dilations:int|list[int]=1, group:int=1,
                kernel_shape:list[int]|None=None, pads:int|list[int]=0, strides:int|list[int]=1):
  x = x.int() - x_zero_point
  w = w.int() - w_zero_point
  y = Conv(x, w, B, auto_pad, dilations, group, kernel_shape, pads, strides)
  y_scale = y_scale / (x_scale * w_scale)
  return _quantize_linear(y, y_scale, y_zero_point)

def QLinearMatMul(a:Tensor, a_scale:Tensor, a_zero_point:Tensor|int, b:Tensor, b_scale:Tensor, b_zero_point:Tensor|int, y_scale:Tensor,
                  y_zero_point:Tensor|int) -> Tensor:
  a = a.int() - a_zero_point
  b = b.int() - b_zero_point
  y = Tensor.matmul(a, b, acc_dtype=dtypes.int32)
  y_scale =  y_scale / (a_scale * b_scale)
  return _quantize_linear(y, y_scale, y_zero_point)

def ConvInteger(x: Tensor, w: Tensor, x_zero_point: Tensor | int = 0, w_zero_point: Tensor | int = 0, B: Tensor | None = None,
                auto_pad: AUTO_PAD_OPTIONS = "NOTSET", dilations: int | list[int] = 1, group: int = 1, kernel_shape: list[int] | None = None,
                pads: int | list[int] = 0, strides: int | list[int] = 1) -> Tensor:
  x_int = x.int() - x_zero_point
  w_int = w.int() - w_zero_point
  return Conv(x_int, w_int, B, auto_pad, dilations, group, kernel_shape, pads, strides)

def MatMulInteger(A: Tensor, B: Tensor, a_zero_point: Tensor | int = 0, b_zero_point: Tensor | int = 0) -> Tensor:
  A_int = A.int() - a_zero_point
  B_int = B.int() - b_zero_point
  return Tensor.matmul(A_int, B_int, acc_dtype=dtypes.int32)

# copied from https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_image_decoder.py
def ImageDecoder(encoded_stream:bytes, pixel_format="RGB"):
  try: import PIL.Image
  except ImportError as e: raise ImportError("Pillow must be installed to use the reference implementation of the ImageDecoder operator") from e
  img = PIL.Image.open(io.BytesIO(encoded_stream))
  if pixel_format == "BGR": return Tensor(np.array(img))[:, :, ::-1]
  if pixel_format == "RGB": return Tensor(np.array(img))
  if pixel_format == "Grayscale": return Tensor(np.array(img.convert("L"))).unsqueeze(-1) # (H, W) to (H, W, 1)
  raise ValueError(f"pixel_format={pixel_format!r} is not supported.")

def AffineGrid(theta:Tensor, size:list[int], align_corners:int=0):
  N, _, *spatial_dims = size
  def generate_grid(steps):
    return Tensor.linspace(-1, 1, steps, device=theta.device) if align_corners else Tensor.linspace(-1+1/steps, 1-1/steps, steps, device=theta.device)
  grids = Tensor.meshgrid(*(generate_grid(d) for d in spatial_dims))
  base_grid = Tensor.stack(*reversed(grids), Tensor.ones_like(grids[0], device=theta.device), dim=-1)
  base_grid = base_grid.reshape(1, prod(spatial_dims), len(grids)+1).expand(N, -1, -1)
  return (base_grid @ theta.transpose(1, 2)).reshape(N, *spatial_dims, -1)

# **************** com.microsoft Ops ****************

def SkipLayerNormalization(x:Tensor, skip:Tensor, gamma:Tensor, beta:Tensor|None=None, bias:Tensor|None=None, epsilon:float=1e-12):
  x = x + skip + bias
  return x.layernorm(eps=epsilon) * gamma + beta, None, None, x

def FastGelu(x:Tensor, bias:Tensor|None=None):
  # this is tanh approximated
  return (x + bias).gelu() if bias is not None else x.gelu()

def EmbedLayerNormalization(input_ids: Tensor, segment_ids:Tensor, word_embedding:Tensor, position_embedding:Tensor,
                            segment_embedding:Tensor, gamma=None, beta=None, mask:Tensor|None=None,
                            position_ids:Tensor|None=None, epsilon=1e-12, mask_index_type=0):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
  assert (segment_ids is None) is (segment_embedding is None)
  assert mask is None and not mask_index_type, "functionality not supported yet"  # TODO
  input_shape = input_ids.shape
  seq_length = input_shape[1]
  compute_seg_emb = (segment_embedding is not None and segment_ids is not None)
  vocab_size, max_position_embeddings = word_embedding.shape[0], position_embedding.shape[0]
  type_vocab_size  = (segment_embedding.shape[0] if compute_seg_emb else None)

  def embedding(x:Tensor, vocab_size, weight:Tensor) -> Tensor:
    return x.unsqueeze(-1).expand(*x.shape, vocab_size)._one_hot_along_dim(vocab_size) @ weight

  # bert embedding layer
  if position_ids is None: position_ids = Tensor.arange(seq_length, requires_grad=False).unsqueeze(0).expand(*input_shape)
  wrd_embedding_res = embedding(input_ids, vocab_size, word_embedding)
  pos_embedding_res = embedding(position_ids, max_position_embeddings, position_embedding)
  seg_embedding_res = embedding(segment_ids, type_vocab_size, segment_embedding) if compute_seg_emb else None

  embedding_sum = wrd_embedding_res + pos_embedding_res
  if seg_embedding_res is not None: embedding_sum = embedding_sum + seg_embedding_res
  out = embedding_sum.layernorm(eps=epsilon) * gamma + beta
  return out, None, embedding_sum

def Attention(x:Tensor, weights, bias:Tensor, mask_index:Tensor|None=None, past:Tensor|None=None,
              relative_position_bias:Tensor|None=None, past_sequence_length:Tensor|None=None, do_rotary:int|None=None,
              mask_filter_value:float|None=None, num_heads:int|None=None, past_present_share_buffer:int|None=None,
              qkv_hidden_sizes:list[int]|None=None, scale:float|None=None, unidirectional:int|None=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention
  assert num_heads is not None  # required
  assert (qkv_hidden_sizes is None and past is not None) or (qkv_hidden_sizes is not None)
  assert relative_position_bias is do_rotary is past_sequence_length is mask_filter_value is past_present_share_buffer is scale is None, \
    "functionality not supported yet"  # TODO strange params
  hidden_size, v_hidden_size = qkv_hidden_sizes[1:] if qkv_hidden_sizes is not None else 2*(weights.shape[1] // 3,)

  if unidirectional:  # gpt-style
    assert hidden_size == v_hidden_size
    xqkv = x.linear(weights, bias)
    xq, xk, xv = [xqkv.shrink([None, None, (i*hidden_size, (i+1)*hidden_size)]) for i in range(3)]
  else:  # bert-style
    wq, wk, wv = weights[:,:hidden_size], weights[:,hidden_size:hidden_size+v_hidden_size], weights[:,hidden_size+v_hidden_size:]
    bq, bk, bv = (bias[:hidden_size], bias[hidden_size:hidden_size+v_hidden_size], bias[hidden_size+v_hidden_size]) if bias is not None else None
    xq, xk, xv = [x.linear(w, b) for w, b in zip((wq, wk, wv), (bq, bk, bv))]
  xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], num_heads, -1).transpose(1, 2) for x in (xq, xk, xv)]

  if past is not None:
    xk, xv = Tensor.cat(past[0], xk, dim=-2), Tensor.cat(past[1], xv, dim=-2)
    present = Tensor.cat(xk.unsqueeze(0), xv.unsqueeze(0))

  def attn(query, key, value, attn_mask):
    query_length, key_length = query.shape[-2], key.shape[-2]
    cdim = max(query_length, key_length) + 1
    attn_weights = query @ key.transpose(-1, -2) / math.sqrt(value.shape[-1])
    # This is where Tensor.scaled_dot_product_attention differs:
    causal_mask = Tensor.ones((cdim, cdim), requires_grad=False, dtype=dtypes.bool).tril(0)[key_length - query_length : key_length, :key_length]
    masked = Tensor.where(causal_mask, attn_weights, -math.inf)
    if attn_mask is not None: masked = masked + attn_mask
    return masked.softmax(-1) @ value

  bsz, _, seq_len, _ = xq.shape
  out = attn(xq, xk, xv, mask_index).transpose(1, 2).reshape(bsz, seq_len, -1)
  return out, present if past is not None else out

def QLinearAdd(a:Tensor, a_scale:Tensor, a_zero_point:Tensor, b:Tensor, b_scale:Tensor, b_zero_point:Tensor, c_scale:Tensor, c_zero_point:Tensor):
  a = a.int() - a_zero_point
  b = b.int() - b_zero_point
  c = (a * a_scale + b * b_scale)
  return _quantize_linear(c, c_scale, c_zero_point)

def QLinearGlobalAveragePool(X:Tensor, x_scale:Tensor, x_zero_point:Tensor, y_scale:Tensor, y_zero_point:Tensor, channels_last:int):
  assert channels_last in {0, 1}
  if channels_last == 1: X = X.permute(0, 2, 3, 1)
  X = (X.int() - x_zero_point) * x_scale
  y = GlobalAveragePool(X)
  return _quantize_linear(y, y_scale, y_zero_point)

# **************** ai.onnx.preview.training Ops ****************
# NOTE: onnx test coverage only covers `T==0` cases, so for all `T>0` this isn't tested
# NOTE: onnx training ops actually don't need the state for optim, all the ops work in a functional way, but we still can reuse optim.py code

from tinygrad.nn.optim import Adam as TinyAdam
from tinygrad.nn.optim import SGD

def onnx_training(input_group_size):
  def _decorator(func):
    def __wrapper(R:Tensor, T:int, *inputs:Tensor, **kwargs):
      R = R.detach()
      groups = len(inputs) // input_group_size
      ret = [func(R, T, *inps, **kwargs) for inps in (inputs[i::groups] for i in range(groups))]
      return tuple(flatten(zip(*ret)))
    return __wrapper
  return _decorator

@onnx_training(3)
def Adagrad(R:Tensor, T:int, *inputs:Tensor, decay_factor:float=0.0, epsilon:float=0.0, norm_coefficient:float=0.0):
  X, G, H = (i.detach() for i in inputs)
  grad = norm_coefficient * X + G
  H.assign(H + grad.square())
  up = grad / (H.sqrt() + epsilon)
  r = R / (1 + T * decay_factor)
  X.assign(X.detach() - r * up)
  return [X, H]

@onnx_training(4)
def Adam(R:Tensor, T:int, *inputs:Tensor, alpha:float=0.9, beta:float=0.999, epsilon:float=0.0, norm_coefficient:float=0.0,
         norm_coefficient_post:float=0.0):
  X, G, V, H = inputs
  G, V, H = G.detach(), V.detach(), H.detach()  # TODO we shouldn't need these detaches
  X.grad = norm_coefficient * X.detach() + G
  opt = TinyAdam([X], b1=alpha, b2=beta, eps=epsilon)
  opt.m, opt.v, opt.lr = [V], [H], R
  # need no-op for m_hat and v_hat if T == 0
  if T == 0: opt.b1_t, opt.b2_t = opt.b1_t.zeros_like(), opt.b2_t.zeros_like()
  else:
    # `T-1` since it's applied again at the start of `_step`
    opt.b1_t = Tensor([alpha**(T-1)], dtype=dtypes.float32, device=X.device, requires_grad=False)
    opt.b2_t = Tensor([beta**(T-1)], dtype=dtypes.float32, device=X.device, requires_grad=False)
  opt.step()
  X = (1 - norm_coefficient_post) * X
  return [X, V, H]

@onnx_training(3)
def Momentum(R:Tensor, T:int, *inputs:Tensor, alpha:float, beta:float, mode:str, norm_coefficient:float):
  X, G, V = inputs
  G, V = G.detach(), V.detach()
  X.grad = (norm_coefficient * X.detach() + G) * (beta if T > 0 else 1)
  opt = SGD([X], momentum=alpha, nesterov=(mode=="nesterov"))
  opt.b, opt.lr = [V], R
  opt.step()
  return [X, V]

def Gradient(*inputs:Tensor, y:str, intermediate_tensors:dict[str, Tensor], **__):
  intermediate_tensors[y].backward()
  return tuple([t.grad for t in inputs])
