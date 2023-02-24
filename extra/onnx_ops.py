from tinygrad.tensor import Tensor
from extra.onnx import safe_numpy
import numpy as np

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

def Gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0):
  ret = alpha * ((A.transpose() if transA == 1 else A) @ (B.transpose() if transB == 1 else B))
  if C is not None: ret += beta * C
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

def _padding(pads=None, auto_pad="NOTSET"):
  assert auto_pad == "NOTSET"  # TODO: write this
  return (pads[1], pads[3], pads[0], pads[2]) if pads is not None else (0,0,0,0)

def AveragePool(X, kernel_shape, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=1, pads=None, strides=1):
  # TODO: the padding shouldn't be counted in the average! this is causing a test failure
  assert ceil_mode == 0 and count_include_pad == 0 and dilations == 1
  return X.pad2d(_padding(pads, auto_pad)).avg_pool2d(kernel_shape, stride=strides)

def MaxPool(X, kernel_shape, auto_pad="NOTSET", ceil_mode=0, dilations=1, pads=None, storage_order=0, strides=1):
  # TODO: the padding should be infinity, not 0!
  assert ceil_mode == 0 and storage_order == 0 and dilations == 1
  return X.pad2d(_padding(pads, auto_pad)).max_pool2d(kernel_shape, stride=strides)

def Conv(X, W, B=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, strides=1):
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=_padding(pads, auto_pad))

# TODO: copied from tensor.py
def Dropout(data, ratio=0.5, training_mode=False, seed=None):
  # TODO: mask should be a boolean tensor
  if not training_mode: return data, Tensor.ones(*data.shape)  # if mask is requested as output it will contain all ones.
  if seed is not None: Tensor.manual_seed(seed)
  _mask : np.ndarray = np.asarray(Tensor._rng.binomial(1, 1.0-ratio, size=data.shape), dtype=data.dtype)
  mask = Tensor(_mask, requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask