import numpy as np
import math
from .tensor import Function, Tensor

# ************* unary ops *************

class ReLU(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * (input >= 0)

class Log(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.log(input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output / input

class Exp(Function):
  def forward(ctx, input):
    ret = np.exp(input)
    ctx.save_for_backward(ret)
    return ret

  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return grad_output * ret

# ************* reduce ops *************

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input, axis)
    return np.array([input.sum()]) if axis is None else input.sum(axis=axis)

  def backward(ctx, grad_output):
    input, axis = ctx.saved_tensors
    if isinstance(axis, int): axis = [axis]
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    return grad_output.reshape(shape) + np.zeros_like(input)

class Max(Function):
  def forward(ctx, inp, axis=None):
    if isinstance(axis, int): axis = [axis]
    ret = np.amax(inp, axis=None if axis is None else tuple(axis), keepdims=True)
    ctx.save_for_backward(inp, axis, ret)
    if axis is not None:
      ret = ret.reshape([inp.shape[i] for i in range(len(inp.shape)) if i not in axis])
    return ret

  def backward(ctx, grad_output):
    input, axis, ret = ctx.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    ret2 = (input==ret.reshape(shape))
    div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True)
    return ret2*grad_output.reshape(shape)/div

# ************* binary ops *************

def unbroadcast(out, in_sh):
  # adjoint operation to broadcast is sum. Need to sum all axis with 1 = in_sh[i] < out.shape[i]
  sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
  return out.sum(axis=sum_axis).reshape(in_sh)

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return x+y

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_output, shape_x), unbroadcast(grad_output, shape_y)

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return x-y

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_output, shape_x), unbroadcast(-grad_output, shape_y)

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return unbroadcast(y*grad_output, x.shape), unbroadcast(x*grad_output, y.shape)

class Pow(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x ** y

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return unbroadcast(y * (x**(y-1.0)) * grad_output, x.shape), \
           unbroadcast((x**y) * np.log(x) * grad_output, y.shape)

# ************* movement ops *************

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.reshape(in_shape)

class Transpose(Function):
  def forward(ctx, x, order):
    ctx.save_for_backward(order)
    return np.transpose(x, order)

  def backward(ctx, x):
    return np.transpose(x, np.argsort(ctx.order))

def inner_slice(x, arg):
  padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
  x = np.pad(x, padding)
  slicee = [(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)]
  return x[tuple([slice(x[0], x[1], None) for x in slicee])]

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape)
    return inner_slice(x, arg)

  def backward(ctx, grad_output):
    shape, = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    return inner_slice(grad_output, narg)

# ************* processing ops *************

class Matmul(Function):
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input @ weight

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output @ np.swapaxes(weight, -2, -1)
    grad_weight = np.swapaxes(input, -2, -1) @ grad_output
    return grad_input, grad_weight

def pool2d(x, kernel_width, kernel_height, stride_w, stride_h, pad_w, pad_h):
  output_shape = (math.floor((x.shape[2] + 2 * pad_w - kernel_width)/stride_w) + 1, math.floor((x.shape[3] + 2 * pad_h - kernel_height)/stride_h) + 1)
  ret = np.zeros(shape=(x.shape[0], x.shape[1], (kernel_width * kernel_height) * (output_shape[0] * output_shape[1])), dtype=np.float32)
  indices = np.zeros(shape=(x.shape[0], x.shape[1], (output_shape[0] * output_shape[1])), dtype=np.float32)
  print("OUTPUT _SHAPE ", output_shape)
  ti = 0
  for n in range(x.shape[0]):
    for c in range(x.shape[1]):
      for ph in range(output_shape[1]):
        for pw in range(output_shape[0]):
          hstart = ph * stride_h - pad_h
          wstart = pw * stride_w - pad_w
          hend = min(hstart + kernel_height, x.shape[3])
          wend = min(wstart + kernel_width, x.shape[2])
          hstart = max(hstart, 0)
          wstart = max(wstart, 0)
          pool_index = ph + (pw * output_shape[1])
          maxval = -float("inf")
          for w in range(wstart, wend):
            for h in range(hstart, hend):
              index = w * x.shape[3] + h
              ret[n][c][pool_index * (kernel_width * kernel_height) + (ti % (kernel_width * kernel_height))] = x[n][c].flatten()[index]
              # indices[n][c][pool_index * (kernel_width * kernel_height) + (ti % (kernel_width * kernel_height))] = index
              if x[n][c].flatten()[index] > maxval:
                indices[n][c][pool_index] = index
                maxval = x[n][c].flatten()[index]
              ti = ti + 1
  ret = ret.reshape((x.shape[0], x.shape[1], -1, (kernel_width  * kernel_height)))
  return ret, indices

class AvgPool2D(Function):
  def forward(ctx, x, kernel_size=(2,2), stride=None, padding=(0, 0)):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    if stride is None: stride = kernel_size
    output_shape = ((x.shape[2] - kernel_size[0])//stride[0] + 1, (x.shape[3] - kernel_size[1])//stride[1] + 1)
    pools, indices = pool2d(x, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1])
    ctx.save_for_backward(x.shape, kernel_size, stride, padding)
    return pools.mean(axis=(3)).reshape(x.shape[0], x.shape[1], output_shape[0], output_shape[1])

  def backward(ctx, grad_output):
    bs, c, wx, hx = grad_output.shape
    in_shape, kernel_size, stride, padding = ctx.saved_tensors
    ret = np.zeros((bs, c, in_shape[2] * in_shape[3]))

    for batch in range(bs):
      for channel in range(c):
        for ph in range(hx):
          for pw in range(wx):
            hstart = ph * stride[1] - padding[1]
            wstart = pw * stride[0] - padding[0]
            hend = min(hstart + kernel_size[1], in_shape[3] + padding[1])
            wend = min(wstart + kernel_size[0], in_shape[2] + padding[0])
            pool_size = (hend - hstart) * (wend - wstart)
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)
            hend = min(hend, in_shape[3])
            wend = min(wend, in_shape[2])
            for h in range(hstart, hend):
              for w in range(wstart, wend):
                ret[batch][channel][h * in_shape[2] + w] = ret[batch][channel][h * in_shape[2] + w] + grad_output[batch][channel].flatten()[ph * wx + pw] / pool_size
    ret = ret.reshape(bs, c, in_shape[2], in_shape[3])
    return ret

class MaxPool2D(Function):
  def forward(ctx, x, kernel_size=(2,2), stride=None, padding=(0, 0)):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    if stride is None: stride = kernel_size
    output_shape = (math.floor((x.shape[2] + 2 * padding[0] - kernel_size[0])/stride[0]) + 1, math.floor((x.shape[3] + 2 * padding[1] - kernel_size[1])/stride[1]) + 1)
    pools, indices = pool2d(x, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1])

    """
    def arrange_indices(indices, max_indices):
      ret = np.zeros(shape=(x.shape[0], x.shape[1], indices.shape[2]), dtype=np.float32)
      for b in range(ret.shape[0]):
        for c in range(ret.shape[1]):
          for i in range(ret.shape[2]):
            ret[b][c][i] = indices[b][c][i][max_indices[b][c][i]]
      return ret
    
    max_indices = np.argmax(pools, axis=3)
    indices = arrange_indices(indices, max_indices) # Then, sort indices with max_indices
    """

    ctx.save_for_backward(indices, x.shape)
    return pools.max(axis=(3)).reshape(x.shape[0], x.shape[1], output_shape[0], output_shape[1])

  def backward(ctx, grad_output):
    bs, c, w, h = grad_output.shape
    indices, in_shape = ctx.saved_tensors
    ret = np.zeros((bs, c, in_shape[2] * in_shape[3]))
    for batch in range(bs):
      for channel in range(c):
        for ind in range(indices.shape[2]):
          ret[batch][channel][int(indices[batch][channel][ind])] = grad_output[batch][channel].flatten()[ind]
    
    ret = ret.reshape(bs, c, in_shape[2], in_shape[3])
    return ret

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    if isinstance(ctx.stride, int): ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_ = x.shape[0], x.shape[1]
    oy,ox = (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    gx = x.reshape(bs,ctx.groups,cin,x.shape[2],x.shape[3])
    tx = np.lib.stride_tricks.as_strided(gx,
      shape=(bs, ctx.groups, cin, oy, ox, H, W),
      strides=(*gx.strides[0:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:5]),
      writeable=False,
    )
    tw = w.reshape(ctx.groups, rcout, cin, H, W)
    ctx.save_for_backward(tx, tw, x.shape)

    ret = np.zeros((bs,ctx.groups,oy,ox,rcout),dtype=x.dtype)
    for g in range(ctx.groups):
      #ijYXyx,kjyx -> iYXk ->ikYX
      ret[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
    return np.moveaxis(ret,4,2).reshape(bs, cout, oy, ox)

  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    tx, tw, x_shape = ctx.saved_tensors
    _,rcout,cin,H,W = tw.shape
    ys,xs = ctx.stride
    OY,OX = x_shape[2:4]

    ggg = grad_output.reshape(bs,ctx.groups,rcout,oy,ox)

    gdw = np.zeros((ctx.groups,rcout,cin,H,W), dtype=tx.dtype)
    for g in range(ctx.groups):
      #'ikYX,ijYXyx -> kjyx'
      gdw[g] += np.tensordot(ggg[:,g], tx[:,g], ((0,2,3),(0,2,3)))

    # needs to be optimized
    gdx = np.zeros((bs,ctx.groups,cin,OY,OX), dtype=tx.dtype)
    for k in range(oy*ox):
      Y, X = k//ox, k%ox
      iY,iX = Y*ys, X*xs
      #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
      for g in range(ctx.groups):
        tg = np.dot(ggg[:,g,:,Y,X].reshape(bs, -1), tw[g].reshape(rcout, -1))
        gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))

    return gdx.reshape((bs, ctx.groups*cin, OY, OX)), gdw.reshape((ctx.groups*rcout, cin, H, W))
