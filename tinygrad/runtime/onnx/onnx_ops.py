import functools, io, math
from typing import Union, Tuple, Optional, List, Any, cast
from tinygrad.tensor import Tensor, _broadcast_shape, ConstType
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.helpers import prod, flatten, make_pair
from .onnx import DTYPE_MAP, to_python_const
import numpy as np

# TODO maybe don't cast hack things and instead raise exceptions
# TODO implement meshgrid

exact_tensor_methods = {"Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan", "Relu", "Sigmoid", "MatMul",
                  "Floor", "Ceil", "Softplus", "HardSwish", "Where", "Mul", "Sinh", "Cosh", "Tanh", "Softsign", "Asinh", "Acosh", "Atanh",
                  "Elu", "Celu", "Xor", "Round",}

# "Softmax_13": "softmax"
# NOTE: equivalent_tensor_methods turns opts all into positional args in case of arg name mismatch
equivalent_tensor_methods = {"Less": "__lt__", "Greater": "__gt__", "LessOrEqual": "__le__", "GreaterOrEqual": "__ge__",
                      "Equal": "__eq__", "LogSoftmax": "log_softmax", "Not": "logical_not", "LeakyRelu": "leakyrelu", "Selu": "selu", "Tile":"repeat",
                      "Range": "arange"}

equivalent_tensor_methods_exceptions = {"Concat": ("cat", {"axis": "dim"})}


# helper
# TODO maybe write helper for stuff like tuple((1,0) if i == axis else None for i in range(X.ndim)), easier to read, see it everywhere
# def dynamic_axis_value_tuple(main_value_fn: callable, default_value: Any, length: int, axis: int):
    # return tuple(main_value_fn(i) if i == axis else default_value for i in range(length))

# **************** Free Ops ****************

def Identity(x: Tensor): return x
# TODO: fix buffer_parse
def Add(x: Tensor, other: Tensor, broadcast=None, axis=None): return x + other if x.dtype == dtypes.float or isinstance(x.dtype, ImageDType) \
  else (x + other).cast(x.dtype)
def Sub(x: Union[Tensor, Any], other: Tensor): return x - other # some test has input as int
def Max(*data_0): return functools.reduce(Tensor.maximum, data_0)
def Min(*data_0): return functools.reduce(Tensor.minimum, data_0)
def Sum(*data_0): return functools.reduce(Tensor.add, data_0)
def Squeeze(data: Tensor, axes): return functools.reduce(lambda d, dim: d.squeeze(dim), sorted(axes), data)
def Unsqueeze(data: Tensor, axes): return functools.reduce(lambda d, dim: d.unsqueeze(dim), sorted(axes), data)
def Mean(*data_0): return Sum(*data_0) / len(data_0)
# NOTE: does not support saturate
def Cast(x: Tensor, to: int, saturate=1): return x.cast(DTYPE_MAP[to])
def CastLike(x: Tensor, target_type: Tensor, saturate=1): return x.cast(target_type.dtype)

# **************** Simple Ops ****************

# https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_div.py
def Div(x: Tensor, other: Tensor): return (x/other).cast(x.dtype)

# TODO: change if to dict match dtype
def Constant(value:Optional[Tensor]=None, value_float=None, value_floats=None, value_int=None, value_ints=None, value_string=None,value_strings=None):
  if value is not None: return value
  if value_float is not None: return Tensor(value_float, dtype=dtypes.float32, requires_grad=False)
  if value_floats is not None: return Tensor(list(value_floats), dtype=dtypes.float32, requires_grad=False)
  if value_int is not None: return Tensor(value_int, dtype=dtypes.int64, requires_grad=False)
  if value_ints is not None: return Tensor(list(value_ints), dtype=dtypes.int64, requires_grad=False)
  if value_string is not None or value_strings is not None: raise NotImplementedError('value_string or value_strings not implemented for Constant op')
def ConstantOfShape(shape:List[ConstType], value:Tensor): return Tensor.ones(*shape, dtype=value.dtype) * (value if shape != [0] else 1)

def HardSigmoid(x: Tensor, alpha=0.2, beta=0.5): return x.hardsigmoid(alpha, beta)
def Gelu(x:Tensor, approximate=None): return x.gelu() if approximate == "tanh" else 0.5 * x * (1 + Erf(x/math.sqrt(2)))
def PRelu(X:Tensor, slope:Tensor):
  # HACK OnnxBackendPyTorchConvertedModelTest HAS WEIRD SLOPE WHERE IT'S [0.25, 0.25, 0.25] FOR ANY X.SHAPE
  slope = slope[0] if slope.size(-1) != X.size(-1) else slope
  return (X > 0).where(X, X * slope)
def ThresholdedRelu(X: Tensor, alpha=1.0): return (X > alpha).where(X, 0)
def Softmax_1(x: Tensor, axis=1): return x.softmax(axis)
def Softmax_13(x: Tensor, axis=-1): return x.softmax(axis)
Softmax = {1: Softmax_1, 13: Softmax_13}   # Softmax default axis changed
def Clip(x: Tensor, min=None, max=None): # noqa: A002  cuz onnx just uses min and max as attribute names
  return x.clip(float('-inf') if min is None else min,float('inf') if max is None else max).cast(x.dtype)

def _axes(axes, noop_with_empty_axes):
  if axes is not None and not (isinstance(axes, Tensor) and axes.shape == (0,)): return to_python_const(axes)
  return [] if noop_with_empty_axes else None
def ReduceMax(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMin(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSum(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMean(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSumSquare(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data.square(), axes, keepdims,noop_with_empty_axes)
def ReduceProd(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.prod(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL1(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data.abs(), axes, keepdims, noop_with_empty_axes)
def ReduceL2(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSumSquare(data, axes, keepdims, noop_with_empty_axes).sqrt()
def ReduceLogSum(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data, axes, keepdims, noop_with_empty_axes).log()
def ReduceLogSumExp(data: Tensor, axes=None,keepdims=1,noop_with_empty_axes=0): return ReduceSum(data.exp(),axes,keepdims,noop_with_empty_axes).log()

def GlobalAveragePool(X: Tensor): return X.mean(axis=tuple(range(2, X.ndim)), keepdim=True)
def GlobalMaxPool(X: Tensor): return X.max(axis=tuple(range(2, X.ndim)), keepdim=True)
def OptionalHasElement(x: Optional[Tensor]=None): return Tensor(x is not None and x.numel() > 0)
def OptionalGetElement(x: Optional[Tensor]=None): return x if x is not None else Tensor([])

def Shape(data: Tensor, end=None, start=0): return Tensor(data.shape[start:end], dtype=dtypes.int64)
def Size(data: Union[Tensor, List]): return prod(data if isinstance(data, list) else data.shape)
def Flatten(x: Tensor, axis=1): return x.reshape(prod(x.shape[0:axis]), -1)
def Reshape(data: Tensor, shape, allowzero=0): return data.reshape([int(x) or (0 if allowzero else data.size(i)) for i, x in enumerate(shape)])
def Expand(x: Tensor, shape:List): return x.expand(_broadcast_shape(x.shape, tuple(shape)))
def Shrink(x: Tensor, bias=0.0, lambd=0.5): return (x < -lambd)*(x+bias) + (x > lambd)*(x-bias)
def And(x:Tensor, y:Tensor): return (x==y).where(x, False)
def Or(x:Tensor, y:Tensor): return (x==y).where(x, True)

def Asin(x): return Atan(x / (1 - x * x).sqrt())
def Acos(x: Tensor):
  negate = (x < 0)
  x = x.abs()
  ret = ((((-0.0187293 * x) + 0.0742610)*x - 0.2121144) * x + 1.5707288) * (1.0 - x).sqrt()
  ret = ret - 2 * negate * ret
  return negate * math.pi + ret
def Atan(y: Tensor):
  t1 = y.abs()
  t3 = (1 > t1).where(t1, t1.reciprocal())
  t4 = t3 * t3
  t0 = ((((-0.013480470 * t4 + 0.057477314) * t4 - 0.121239071) * t4 + 0.195635925) * t4 - 0.332994597) * t4 + 0.999995630
  t3 = t0 * t3
  t3 = (t1 > 1).where(1.570796327 - t3, t3)
  return y.sign() * t3

def Trilu(x: Tensor, k:int=0, upper=1): return x.triu(k) if upper else x.tril(k)

def Binarizer(x, threshold=0.0): return (x > threshold).float()

def ArgMax(x: Tensor, axis=0, keepdims=1, select_last_index=0):
  if select_last_index: return ((x.shape[axis]-1) - x.flip(axis).argmax(axis, keepdim=keepdims)).cast(dtypes.int64)
  return x.argmax(axis, keepdim=keepdims).cast(dtypes.int64)
def ArgMin(x, axis=0, keepdims=1, select_last_index=0): return ArgMax(-x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)

def Transpose(x: Tensor, perm=None): return x.permute(order=list(range(x.ndim)[::-1]) if perm is None else perm)

# **************** Complex Ops ****************

def Gemm(A: Tensor, B: Tensor, C: Optional[Tensor] = None, alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=0):
  ret = alpha * (A.transpose(transA) @ B.transpose(transB))
  if C is not None: ret = ret + beta * (C if broadcast == 0 else C.reshape([-1 if i < len(C.shape) else 1 for i in range(ret.ndim)][::-1]))
  return ret

def Einsum(*Inputs: List[Tensor], equation): return Tensor.einsum(equation, Inputs)

def CumSum(X:Tensor, axis:int, exclusive=0, reverse=0):
  if axis < 0: axis += X.ndim
  if reverse: X = X.flip(axis)
  if exclusive: X = X.pad(tuple((1,0) if i == axis else None for i in range(X.ndim)))\
                     .shrink(tuple((0,X.shape[axis]) if i == axis else None for i in range(X.ndim)))
  if reverse: return X.cumsum(axis).flip(axis)
  return X.cumsum(axis)

# TODO: this is copied from tinygrad/nn/__init__.py
# spatial is from opset 7 and has since been removed
def BatchNormalization(X: Tensor, scale, B, input_mean, input_var, epsilon=1e-05, momentum=0.9, training_mode=0, spatial=1, is_test=0):
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

def InstanceNormalization(x: Tensor, scale: Tensor, bias: Tensor, epsilon=1e-05):
  axis = tuple(range(2, x.ndim))
  mean = x.mean(axis=axis, keepdim=True)
  invstd = x.sub(mean).square().mean(axis=axis, keepdim=True).add(epsilon).rsqrt()
  return x.sub(mean).mul(scale.reshape(shape=[-1, 1, 1])).mul(invstd).add(bias.reshape(shape=[-1, 1, 1]))

def LayerNormalization(x: Tensor, scale, bias, axis=-1, epsilon=1e-05, stash_type=1):
  assert stash_type == 1, "only float32 is supported"
  axis = tuple(i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim))
  mean = x.mean(axis=axis, keepdim=True)
  return x.layernorm(axis, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).square().mean(axis=axis, keepdim=True).add(epsilon).rsqrt()

def GroupNormalization(x: Tensor, scale: Tensor, bias: Tensor, num_groups, epsilon=1e-05):
  return x.reshape(x.size(0), num_groups, -1).layernorm(axis=-1, eps=epsilon).mul(scale.unsqueeze(-1)).add(bias.unsqueeze(-1)).reshape(x.shape)

# TODO: rewrite all this padding crap
# Tensor._padding2d()
# there are 3 types of pads
# pad per axis: [padx, pady]
# onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
# tinygrad: ((x1_begin, x1_end), (x2_begin, x2_end), ...)
# pad per axis gets autopad split into onnx
def _format_padding(onnx_pads, ndims=None, axes=None) -> List[Tuple[int, int]]:
  axes = axes or list(range(ndims))
  np_pads = [(0,0)] * ndims
  for i in range(len(axes)): np_pads[axes[i]] = (onnx_pads[i], onnx_pads[i + len(axes)])
  return np_pads

def _padded(X: Tensor, pads=None, auto_pad="NOTSET", axes=None, constant_value=0., strides=None, kernel_shape=None, dilations=None, ceil_mode=0):
  if auto_pad != "NOTSET": pads = _auto_pad(X.shape[-len(kernel_shape):], auto_pad, strides, kernel_shape, dilations)
  elif ceil_mode:
    if strides is not None: strides = [strides]*len(kernel_shape) if isinstance(strides, int) else strides or [1]*len(kernel_shape)
    if dilations is not None: dilations = [1]*len(kernel_shape) if dilations == 1 else dilations
    out_spatial_shape = [math.ceil((sh - dil * (ker-1)-1)/st + 1) if ceil_mode else math.floor((sh - dil * (ker-1)-1)/st + 1)
                         for sh, st, ker, dil in zip(X.shape[-len(kernel_shape):], strides, kernel_shape, dilations)]
    pad_shape = [(osh-1)*st+((ks-1)*dil+1)-ish for osh, st, ks, dil, ish in
                 zip(out_spatial_shape, strides, kernel_shape, dilations, X.shape[-len(kernel_shape):])]
    pad_shape = [[sh//2, sh-sh//2] for sh in pad_shape]
    # ceil_mode case follows NOTE in https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    # so if any kernels start in right padded region, we decrease right pads to omit that kernel. Only omitting 1 kernel now.
    pad_shape = [[start,end-rpad] if (rpad := ks + st%(st-(((start+xs)%st)))) <= end else [start,end]
                 for (start,end), ks, st, xs in zip(pad_shape, kernel_shape, strides, X.shape[-len(kernel_shape):])]
    pad_shape = flatten(pad_shape)
    pads = pad_shape[::2] + pad_shape[1::2]
  if pads is None: return X
  pads = _format_padding(pads, ndims=len(X.shape), axes=axes)
  return X.pad(tuple(pads), value=constant_value)

def _auto_pad(img_shape, auto_pad, strides, kernel_shape, dilations):
  strides = [strides]*len(kernel_shape) if isinstance(strides, int) else strides or [1]*len(kernel_shape)
  dilations = [1]*len(kernel_shape) if dilations == 1 else dilations
  if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
    pad_shape = [(math.ceil(sh/st)-1)*st+((ks-1)*di+1)-sh for sh, st, ks, di in zip(img_shape, strides, kernel_shape, dilations)]
    pad_shape = flatten([[sh//2, sh-sh//2] for sh in pad_shape])
    return pad_shape[::2] + pad_shape[1::2] if auto_pad == "SAME_UPPER" else pad_shape[1::2] + pad_shape[::2]
  raise NotImplementedError(f"auto_pad={auto_pad} not implemented")

# [x1_begin, x2_begin, ..., x1_end, x2_end, ...] -> (padding_left, padding_right, ..., padding_top, padding_bottom, ...)
# NOTE: also works with 1D, 3D, or 1337D
def _onnx_pads_to_tiny_pads(pads): return flatten(reversed([(pb, pe) for pb, pe in zip(pads, pads[len(pads)//2:])]))

def _auto_pad(pads, auto_pad):
  return [pads[i]//2 for i in range(len(pads))] + [pads[i]-pads[i]//2 for i in range(len(pads))] if auto_pad == "SAME_UPPER" else \
         [pads[i]-pads[i]//2 for i in range(len(pads))] + [pads[i]//2 for i in range(len(pads))]
# def _auto_pad(img_shape, auto_pad, strides, kernel_shape, dilations):
#   strides = [strides]*len(kernel_shape) if isinstance(strides, int) else strides or [1]*len(kernel_shape)
#   dilations = [1]*len(kernel_shape) if dilations == 1 else dilations
#   if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
#     pad_shape = [(math.ceil(sh/st)-1)*st+((ks-1)*di+1)-sh for sh, st, ks, di in zip(img_shape, strides, kernel_shape, dilations)]
#     pad_shape = flatten([[sh//2, sh-sh//2] for sh in pad_shape])
#     return pad_shape[::2] + pad_shape[1::2] if auto_pad == "SAME_UPPER" else pad_shape[1::2] + pad_shape[::2]
#   raise NotImplementedError(f"auto_pad={auto_pad} not implemented")

def Pad(x:Tensor, pads:List[ConstType], constant_value:Optional[ConstType]=None, axes:Optional[List[ConstType]]=None, mode="constant",
        value:float=0.):
  constant_value = value if constant_value is None else float(constant_value)
  base_shape = x.shape
  formatted_pads = _format_padding(pads, ndims=len(x.shape), axes=axes)
  if mode == "wrap":
    repeat_args = [math.ceil(dim[0]/sh) + math.ceil(dim[1]/sh) + 1 for dim, sh in zip(formatted_pads, base_shape)]
    new_shape = [s*r for s,r in zip(base_shape, repeat_args)]
    shrink_args = [(sh-dim[0]%sh if dim[0]%sh != 0 else 0, nsh-(sh-dim[1]%sh if dim[1]%sh != 0 else 0))
                   for dim, sh, nsh in zip(formatted_pads, base_shape, new_shape)]
    return x.repeat(tuple(repeat_args)).shrink(tuple(shrink_args))
  if mode == "reflect":
    for i,s in enumerate(x.shape):
      if formatted_pads[i] != (0,0):
        xL = x.flip(i).shrink(tuple((s-formatted_pads[i][0]-1, s_-1) if i_ == i else None for i_,s_ in enumerate(x.shape)))
        xR = x.flip(i).shrink(tuple((1, formatted_pads[i][1]+1) if i_ == i else None for i_ in range(x.ndim)))
        x = xL.cat(x, xR, dim=i)
    return x
  if mode == "edge":
    for i,s in enumerate(x.shape):
      if formatted_pads[i] != (0,0):
        xL = x.shrink(tuple((0,1) if i_ == i else None for i_ in range(x.ndim))).expand([formatted_pads[i][0] if i_ == i else None for i_ in
                                                                                         range(x.ndim)])
        xR = x.shrink(tuple((s_-1, s_) if i_ == i else None for i_,s_ in enumerate(x.shape))).expand([formatted_pads[i][1] if i_ == i else None
                                                                                                      for i_ in range(x.ndim)])
        x = xL.cat(x, xR, dim=i)
    return x
  if mode == "constant": return _padded(x, pads, axes=axes, constant_value=constant_value)

def AveragePool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=1, pads=None, strides=1):
  pixel_axes = tuple(range(2, X.ndim))
  ret = _padded(X, pads, auto_pad, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations, ceil_mode=ceil_mode)
  ret = ret.avg_pool2d(kernel_shape, stride=strides, dilation=dilations)
  if count_include_pad: return ret
  div = _padded(Tensor.ones(X.shape), pads, auto_pad, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations,
                ceil_mode=ceil_mode).avg_pool2d(kernel_shape, stride=strides, dilation=dilations)
  return ret / div

def MaxPool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=0, dilations=1, pads=None, storage_order=0, strides=1):
  pixel_axes = tuple(range(2, X.ndim))
  ret = _padded(X, pads, auto_pad, constant_value=-math.inf, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations,
                ceil_mode=ceil_mode)
  ret = ret.max_pool2d(kernel_shape, stride=strides, dilation=dilations).cast(X.dtype)
  ret_len, X_len = ret.numel(), X.numel()
  indices = ((ret.flatten().unsqueeze(1).expand(ret_len, X_len) == X.flatten().unsqueeze(0).expand(ret_len, X_len)) * \
             Tensor.arange(X_len, dtype=dtypes.int64).unsqueeze(0).expand(ret_len, X_len)).sum(1).reshape(ret.shape)
  if storage_order: indices = indices.transpose(-2, -1)
  return ret, indices

def MaxUnpool(xT: Tensor, xI: Tensor, outshape: Optional[Tensor]=None, kernel_shape=None, pads=None, strides=None):
  out_sh = [(ks//2)*2 + st * inps for inps, st, ks in zip(xI.shape, strides, kernel_shape)]
  outlength = prod(out_sh)
  xI = xI.flatten().unsqueeze(1).expand(None, outlength)
  arange = Tensor.arange(outlength, requires_grad=False).reshape(1, outlength).expand(xI.shape)
  xT = xT.flatten().unsqueeze(1).expand(None, outlength)
  ret = ((xI == arange) * xT).sum(0).reshape([1, 1] + out_sh)
  if outshape is not None and outshape != ret.shape:
    diff = [outshape[2] - ret.shape[2], outshape[3] - ret.shape[3]]
    pad_args = [diff[0]//2, diff[1]//2, diff[0]-diff[0]//2, diff[1]-diff[1]//2]
    ret = ret.pad2d((pad_args[1], pad_args[3], pad_args[0], pad_args[2]))
  return ret

def Conv(X: Tensor, W: Tensor, B:Optional[Tensor]=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, strides=1):
  input_shape, kernel_shape = X.shape[2:], (kernel_shape or W.shape[2:])
  strides, dilations = (make_pair(x, len(input_shape)) for x in (strides, dilations))
  if auto_pad != "NOTSET":
    pads = _auto_pad([(math.ceil(sh/st)-1)*st+((ks-1)*di+1)-sh for sh, st, ks, di in zip(input_shape, strides, kernel_shape, dilations)], auto_pad)
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=_onnx_pads_to_tiny_pads(pads) if pads is not None else 0)

# TODO: their reference implementation and their documentation has different information
# ref: https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv_transpose.py
# doc: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
# the current implementation makes sense to geohotstan and pass tests, but differs from both ref and doc
def ConvTranspose(X: Tensor, W: Tensor, B:Optional[Tensor]=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None,
                  output_shape=None, output_padding=0, strides=1):
  input_shape, kernel_shape = X.shape[2:], (kernel_shape or W.shape[2:])
  strides, dilations, output_padding = (make_pair(x, len(input_shape)) for x in (strides, dilations, output_padding))
  if output_shape is not None: # we pad according to output_shape
    X = X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=0, output_padding=0)
    return X.pad((None, None, *((0, out-xs) for out, xs in zip(output_shape, X.shape[2:]))))  # TODO: unsure about this
  # NOTE the pads either from args or auto_pad have the format [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
  # this is asymmetrical padding and conv_transpose2d does not support it
  # padding for conv_transpose2d effectively "shrinks" the padding that goes into conv2d, so we just shrink it after
  if pads is None: # we generate asymmetrical pads
    output_shape = [X.shape[i+2] * strides[i] for i in range(len(strides))]
    pads = [strides[i]*(input_shape[i]-1) + output_padding[i] + ((kernel_shape[i]-1)*dilations[i]+1)-output_shape[i] for i in range(len(input_shape))]
    pads = [0,0] * len(input_shape) if auto_pad == "NOTSET" else _auto_pad(pads, auto_pad)
  X = X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=0, output_padding=output_padding)
  return X.pad2d(_onnx_pads_to_tiny_pads([-p for p in pads])) # neg it since we shrink
  # return X if pads is None else X.shrink((None, None, *((pl, X.size(i+2)-pr) for i,(pl,pr) in enumerate(zip(pads, pads[len(pads)//2:])))))

def DepthToSpace(X:Tensor, blocksize:int, mode:str="DCR"):
  b, c, h, w = X.shape
  if mode == "DCR":
    return X.reshape(b, blocksize, blocksize, c // (blocksize**2), h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // (blocksize**2), h * blocksize,
                                                                                                           w * blocksize)
  elif mode == "CRD":
    return X.reshape(b, c // (blocksize ** 2), blocksize, blocksize, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // (blocksize ** 2), h * blocksize,
                                                                                                             w * blocksize)

def SpaceToDepth(X:Tensor, blocksize:int):
  b, c, h, w = X.shape
  return X.reshape(b, c, h // blocksize, blocksize, w // blocksize, blocksize).permute(0, 3, 5, 1, 2, 4).reshape(b, c * (blocksize**2),
                                                                                                                 h // blocksize, w // blocksize)

# Reimplemented here because you need legacy RNG for passing ONNX tests.
def Dropout(data: Tensor, ratio=0.5, training_mode=False, seed=None):
  if not training_mode: return data, Tensor.ones(data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
  mask = Tensor(np.random.RandomState(seed).random(cast(Tuple[int,...], data.shape)) >= ratio, requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask

def LRN(x: Tensor, size, alpha=1e-4, beta=0.75, bias=1.0):
  bs, c, iy, ix = x.shape
  ret = x/x.mul(x).reshape(bs,1,c,iy*ix).pad2d((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1).reshape(bs,c,iy,ix).mul(alpha).add(bias).pow(beta)
  return ret

def MeanVarianceNormalization(x: Tensor, axis=(0, 2, 3)): return (x - x.mean(axis, keepdim=True)) / (x.std(axis, keepdim=True, correction=0) + 1e-9)

def NegativeLogLikelihoodLoss(x: Tensor, target: Tensor, weight=None, ignore_index=None, reduction="mean"):
  target_shape, x, target = target.shape, x.reshape(x.size(0), x.size(1), -1), target.reshape(x.size(0), -1)
  mask = Tensor.ones_like(target) if ignore_index is None else (target != ignore_index)
  weight = mask if weight is None else weight[target] * mask
  loss = -x.gather(1, target.unsqueeze(1)).squeeze(1) * mask * weight
  if reduction == "mean": return loss.sum() / weight.sum() if weight is not None else loss.sum() / mask.sum()
  elif reduction == "sum": return loss.sum()
  else: return loss.reshape(target_shape)

def SoftmaxCrossEntropyLoss(scores: Tensor, labels: Tensor, weights=None, ignore_index=None, reduction="mean"):
  _N, C, *s_dimensions = scores.shape
  if ignore_index is not None: labels = (labels == ignore_index).where(C+1, labels)
  mask = labels.unsqueeze(1) == Tensor.arange(C).reshape(1, C, *[1]*len(s_dimensions))
  y = scores.log_softmax(axis=1)
  loss = (mask * -y).sum(1)
  if weights is not None:
    weights = weights[labels, ...]
    loss = loss * weights
  if reduction == "mean": loss = loss.sum() / ((loss != 0).sum() if weights is None else weights.sum())
  elif reduction == "sum": loss = loss.sum()
  return loss, y

def ArrayFeatureExtractor(x: Tensor, indices: Tensor): return x[..., indices]
# TODO: is fuse_arange stuff working for this?
def Gather(x: Tensor, indices: Tensor, axis=0):
  if indices.numel() < 9: # NOTE lessor kernels for smaller indices but kernel number increases depending on size of indices
    x_sh = list(x.shape)
    ret_shape = x_sh[:axis] + list(indices.shape) + x_sh[axis+1:]
    if indices.ndim > 1: indices = indices.flatten()
    python_indices = cast(Union[List[int], int], to_python_const(indices))
    normalized_python_indices = [python_indices] if not isinstance(python_indices, list) else [x_sh[axis]+x if x<0 else x for x in python_indices]
    args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(x_sh)] for i in normalized_python_indices]
    return x.shrink(arg=tuple(args[0])).cat(*[x.shrink(arg=tuple(arg)) for arg in args[1:]], dim=axis).reshape(ret_shape)
  # NOTE faster gather, fixed number of kernels, but exceeds limited kernels for openpilot
  return x[tuple([slice(None) if i != axis else indices for i in range(x.ndim)])]
  # return x[tuple([slice(None) if i != axis else indices for i in range(x.ndim)])]
def GatherElements(x: Tensor, indices: Tensor, axis):
  indices = (indices < 0).where(x.shape[axis], 0) + indices
  return x.gather(axis, indices)

def Resize(X:Tensor, roi=None, scales=None, sizes=None, antialias=0, axes=None, coordinate_transformation_mode='half_pixel',
           cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0, keep_aspect_ratio_policy='stretch',
           mode='nearest', nearest_mode='round_prefer_floor'):
  def _apply_nearest_mode(index: Tensor, input_dim, mode: str):
    if mode == "round_prefer_floor": index = (index - 0.5).ceil()
    elif mode == "round_prefer_ceil": index = (index + 0.5).floor()
    elif mode == "floor": index = index.floor()
    elif mode == "ceil": index = index.ceil()
    else: raise ValueError(f"invalid {nearest_mode=}")
    return index.cast(dtypes.int32).clip(0, input_dim-1)
  def _apply_coordinate_transformation(index: Tensor, input_dim: int, scale_dim, roi_dim, sizes_frac, mode: str):
    # TODO: needs more testing, not confident in this
    # NOTE: their reference implementation differ from the implementation in their reference docs
    # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_resize.py
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
    output_dim = scale_dim * input_dim
    if mode == "half_pixel": index = (index + 0.5) / scale_dim - 0.5
    elif mode == "align_corners": index = index * (input_dim - 1) / (output_dim - 1) if output_dim != 1 else Tensor([0])
    elif mode == "asymmetric": index = index / scale_dim
    elif mode == "pytorch_half_pixel": index = (index + 0.5) / scale_dim - 0.5 if output_dim != 1 else Tensor([-0.5])
    elif mode == "half_pixel_symmetric":
      index = input_dim / 2 * (1 - int(output_dim) / sizes_frac) + (index + 0.5) / scale_dim - 0.5
    elif mode == "tf_crop_and_resize":
      index = roi_dim[0] * (input_dim - 1) + index * ((roi_dim[1] - roi_dim[0]) * (input_dim - 1) / (output_dim - 1))
    else: raise ValueError(f"invalid {coordinate_transformation_mode=}")
    return index.clip(0, input_dim-1)

  if axes is not None:
    perm = [a for a in range(len(X.shape)) if a not in axes] + list(axes)
    inverse_perm = [perm.index(i) for i in range(len(perm))]
    X = X.permute(*perm)
  else: axes, inverse_perm = list(range(X.ndim)), []
  if sizes is not None:
    sizes = [1]*(X.ndim - len(sizes)) + sizes
    scales = [sizes[i] / X.size(i) for i in range(X.ndim)]
    if keep_aspect_ratio_policy in ["not_larger", "not_smaller"]:
      scale_fxn = min if keep_aspect_ratio_policy == "not_larger" else max
      scale = scale_fxn(scale for i, scale in enumerate(scales) if i in axes)
      scales = [scale if i in axes else 1 for i in range(X.ndim)]
      sizes = [int((scale * X.size(i)) + 0.5) if i in axes else X.size(i) for i in range(X.ndim)]
    elif keep_aspect_ratio_policy != "stretch": raise ValueError(f"invalid {keep_aspect_ratio_policy=}")
  else:
    scales = [1]*(X.ndim - len(scales)) + scales
    sizes = [int(sc*sh) for sc, sh in zip(scales, X.shape)]

  sizes_frac = [sc*sh for sc, sh in zip(scales, X.shape)][2:]
  sizes, axes, scales, input_shape = (val[2:] if isinstance(val, list) else [None] * (X.ndim-2) for val in (sizes, axes, scales, list(X.shape)))
  roi = [[st, ed] for st, ed in zip(roi[:len(roi)//2], roi[len(roi)//2:])] if isinstance(roi, list) else [None] * (X.ndim-2)
  indexes = [Tensor.arange(shape, dtype=dtypes.default_float, device=X.device) for shape in sizes]
  indexes = [_apply_coordinate_transformation(*args, coordinate_transformation_mode) for args in zip(indexes, input_shape, scales, roi, sizes_frac)]
  if mode == "nearest":
    indexes = [_apply_nearest_mode(*args, nearest_mode) for args in zip(indexes, input_shape)]
    indexes = [idx.reshape(*(-1 if i == dim else 1 for i in range(len(sizes)))).expand(sizes) for dim, idx in enumerate(indexes)]
    X = X[(..., *indexes)]
  if mode == "linear":
    expand = list(X.shape)
    for i in range(-len(sizes), 0):
      reshape, index = [1] * X.ndim, indexes[i]
      reshape[i] = expand[i] = sizes[i]
      low, high, perc = [y.reshape(reshape).expand(expand) for y in (index.floor(), index.ceil(), index - index.floor())]
      X = X.gather(i, low).lerp(X.gather(i, high), perc)
  if mode == "cubic":
    raise NotImplementedError("cubic interpolation is not implemented")
  return X.permute(*inverse_perm) if inverse_perm else X
def Upsample(X, scales, mode): return Resize(X=X, scales=scales, mode=mode)

def CenterCropPad(t: Tensor, shape, axes=None):
  shrink_arg = [None] * t.ndim
  pad_arg = [None] * t.ndim
  for s, x in zip(shape, axes or range(t.ndim)):
    tx = t.shape[x]
    if s < tx: shrink_arg[x] = (tx//2 - (s+1)//2, tx//2 + s//2)
    elif s > tx: pad_arg[x] = ((s-tx)//2, (s-tx+1)//2)
  return t.shrink(tuple(shrink_arg)).pad(tuple(pad_arg))

def OneHot(indices: Tensor, depth, values, axis=-1):
  indices, rank = (indices < 0).where(indices+depth, indices), indices.ndim
  if axis < 0: axis += rank + 1
  ls, rs = indices.shape[0:axis], indices.shape[axis: rank]
  cond = indices[:,None] == Tensor.arange(int(depth)).reshape((1,) * len(ls) + (int(depth),) + (1,) * len(rs))
  return cond.where(values[1], values[0])

def Erf(x: Tensor):
  t = 1.0 / (1.0 + 0.3275911 * x.abs())
  term1 = 0.254829592 * t
  term2 = -0.284496736 * t ** 2
  term3 = 1.421413741 * t ** 3
  term4 = -1.453152027 * t ** 4
  term5 = 1.061405429 * t ** 5
  y = (term1 + term2 + term3 + term4 + term5)
  z = 1.0 - y * (-x * x).exp()
  return (x > 0).where(z, -z)

def Compress(inp: Tensor, condition, axis=None):
  if axis is None:
    inp = inp.flatten()
    axis = 0

  if axis < 0: axis += inp.ndim

  con = Tensor(np.arange(len(condition))[condition]) # TODO no boolean indexing in Tensor, pretty sure it's possible now...
  return inp[tuple(con if i == axis else slice(None) for i in range(inp.ndim))]

def EyeLike(x: Tensor, dtype: Optional[int]=None, k=0):
  tiny_dtype = x.dtype if dtype is None else DTYPE_MAP[dtype]
  dim = cast(int, min(x.shape))
  if x.size(0) == x.size(1): return Tensor.eye(dim, dtype=tiny_dtype)
  padarg = tuple(None if d == dim else (k, d-dim-k) for d in x.shape)
  return Tensor.eye(dim, dtype=tiny_dtype).pad(padarg)

def IsInf(x: Tensor, detect_negative=1, detect_positive=1):
  return (x == float("inf")) * bool(detect_positive) + (x == float("-inf")) * bool(detect_negative)

def DequantizeLinear(x: Tensor, x_scale: Tensor, x_zero_point: Union[Tensor, int] = 0, axis=1, block_size=0):
  if axis < 0: axis += x.ndim
  if not isinstance(x_zero_point, Tensor): x_zero_point = Tensor(x_zero_point)
  if block_size: x_zer, x_sc = x_zero_point.repeat_interleave(block_size, axis), x_scale.repeat_interleave(block_size, axis)
  else:
    shape = (*[1]*axis, *x_scale.shape, *[1]*(x.ndim - axis - x_scale.ndim))
    x_sc, x_zer = x_scale.reshape(shape), x_zero_point.reshape(shape)
  return ((x.float() - x_zer) * x_sc).cast(x_scale.dtype)

def IsNaN(x: Tensor): return x != x

# copied from https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_image_decoder.py
# without importing PIL we'll have to manually decode a bunch of image formats like PNG, JPEG, WebP, etc
# TODO maybe uint8 stuff may work?
def ImageDecoder(encoded_stream: bytes, pixel_format="RGB"):
  try: import PIL.Image
  except ImportError as e: raise ImportError("Pillow must be installed to use the reference implementation of the ImageDecoder operator") from e
  img = PIL.Image.open(io.BytesIO(encoded_stream))
  if pixel_format == "BGR": return Tensor(np.array(img))[:, :, ::-1]
  if pixel_format == "RGB": return Tensor(np.array(img))
  if pixel_format == "Grayscale": return Tensor(np.array(img.convert("L"))).unsqueeze(-1) # (H, W) to (H, W, 1)
  raise ValueError(f"pixel_format={pixel_format!r} is not supported.")

def AffineGrid(theta: Tensor, size, align_corners=0):
  _, _, *data_sz = size
  size_zeros, original_grid = Tensor.zeros(data_sz), Tensor.ones(data_sz)
  stackable = [original_grid]
  for dim, dim_sz in enumerate(data_sz):
    a = Tensor.arange(-1, 1.0001, 2/(dim_sz-1)) if align_corners == 1 else Tensor.arange(-1+1/dim_sz, 1, 2/dim_sz)
    if dim == 0: stackable = [a.reshape(dim_sz, *[1]*(len(data_sz)-1)) + size_zeros, *stackable]
    elif dim == 1: stackable = [a.reshape(1, dim_sz, *[1]*(len(data_sz)-2)) + size_zeros, *stackable]
    else: stackable = [a.reshape(1, dim_sz) + size_zeros, *stackable]
  original_grid = Tensor.stack(*stackable, dim=len(data_sz))
  transformed_grid = theta.matmul(original_grid.reshape(-1, len(data_sz)+1).transpose()).transpose(1, 2)
  return transformed_grid.reshape(size[0], *data_sz, theta.size(1))

# **************** com.microsoft Ops ****************

def SkipLayerNormalization(x:Tensor, skip:Tensor, gamma, beta:Optional[Tensor]=None, bias:Optional[Tensor]=None, epsilon=None):
  if epsilon is None: epsilon=1e-12
  x = x + skip + bias
  return x.layernorm(eps=epsilon) * gamma + beta, None, None, x

def FastGelu(x:Tensor, bias:Optional[Tensor]=None):
  # this is tanh approximated
  return (x + bias).gelu()

def EmbedLayerNormalization(input_ids: Tensor, segment_ids: Tensor, word_embedding:Tensor,
                            position_embedding:Tensor, segment_embedding:Tensor, gamma=None, beta=None,
                            mask:Optional[Tensor]=None, position_ids:Optional[Tensor]=None, epsilon=None, mask_index_type=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
  assert (segment_ids is None) is (segment_embedding is None)
  assert (mask is None) is (mask_index_type is None)
  assert mask is None, "functionality not supported yet"  # TODO
  input_shape = input_ids.shape
  seq_length = input_shape[1]
  compute_seg_emb = (segment_embedding is not None and segment_ids is not None)
  vocab_size, max_position_embeddings, type_vocab_size = word_embedding.size(0), position_embedding.size(0), (segment_embedding.size(0)
                                                                                                                if compute_seg_emb else None)

  def embedding(x:Tensor, vocab_size, weight:Tensor) -> Tensor:  # TODO from nn.Embedding. Could probably upstream this to Tensor
    vocab_counter = Tensor.arange(vocab_size, dtype=x.dtype, requires_grad=False).reshape(1, 1, vocab_size).expand(*x.shape, vocab_size)
    return (vocab_counter == x.unsqueeze(2).expand(*x.shape, vocab_size)) @ weight

  # bert embedding layer
  if epsilon is None: epsilon = 1e-12
  if position_ids is None: position_ids = Tensor.arange(seq_length, requires_grad=False).unsqueeze(0).expand(*input_shape)
  wrd_embedding_res = embedding(input_ids, vocab_size, word_embedding)
  pos_embedding_res = embedding(position_ids, max_position_embeddings, position_embedding)
  seg_embedding_res = embedding(segment_ids, type_vocab_size, segment_embedding) if compute_seg_emb else None

  embedding_sum = wrd_embedding_res + pos_embedding_res
  if seg_embedding_res is not None: embedding_sum = embedding_sum + seg_embedding_res
  out = embedding_sum.layernorm(eps=epsilon) * gamma + beta
  return out, None, embedding_sum

def Attention(x:Tensor, weights, bias:Tensor, mask_index:Optional[Tensor]=None, past:Optional[Tensor]=None,
              relative_position_bias:Optional[Tensor]=None, past_sequence_length:Optional[Tensor]=None, do_rotary=None, mask_filter_value=None,
              num_heads=None, past_present_share_buffer=None, qkv_hidden_sizes=None, scale=None, unidirectional=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention
  assert num_heads is not None  # required
  assert (qkv_hidden_sizes is None and past is not None) or (qkv_hidden_sizes is not None)
  assert relative_position_bias is do_rotary is past_sequence_length is mask_filter_value is past_present_share_buffer is scale is None, \
  "functionality not supported yet"  # TODO strange params
  hidden_size, v_hidden_size = qkv_hidden_sizes[1:] if qkv_hidden_sizes is not None else 2*(weights.size(1) // 3,)

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
  return out, present
