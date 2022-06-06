import pyopencl as cl
import numpy as np
from tinygrad.helpers import binary_broadcast
from ..tensor import Function
from ..llops.gpu import GPUBuffer, buffer_new
from ..llops.gpu import unary_op, binary_op, reduce_op, perm_axis, inner_slice
from ..llops.gpu import matmul, conv, convdw, convdx

# ************* unary ops *************

class UnaryOp(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx.fop, input, buffer_new(input.shape))

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx.bop, input, grad_output, buffer_new(input.shape))

class ReLU(UnaryOp):
  fop = 'max(a, (float)0.)'
  bop = 'b * (a >= 0)'

class Log(UnaryOp):
  fop = 'log(a)'
  bop = 'b / a'

class Exp(UnaryOp):
  fop = 'exp(a)'
  bop = 'b * exp(a)'

# ************* reduce ops *************

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input.shape)
    return reduce_op("out += a", input, axis=axis)

  def backward(ctx, grad_output):
    shape_input, = ctx.saved_tensors
    # NOTE: the b buffer_new isn't used, since this is just for broadcast
    ret = buffer_new(shape_input)
    return binary_op('a', grad_output, ret, ret)

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = reduce_op("out = max(a,out)", input, axis=axis, start="-INFINITY")
    ctx.save_for_backward(input, axis, ret)
    return ret

  def backward(ctx, grad_output):
    input, axis, ret = ctx.saved_tensors
    ret2 = binary_op("1.0*(a==b)", input, ret, buffer_new(input.shape))
    div = reduce_op("out += a", ret2, axis=axis, start="1e-10")
    ret3 = binary_op("a/b", ret2, div, buffer_new(input.shape))
    return binary_op('a*b', ret3, grad_output, buffer_new(input.shape))

# ************* binary ops *************

def unbroadcast(out, in_sh):
  sum_axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  return reduce_op("out += a", out, sum_axis)

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op('a+b', x, y, buffer_new(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, grad_output
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_x, shape_x), unbroadcast(grad_y, shape_y)

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op('a-b', x, y, buffer_new(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, unary_op('-a', grad_output, buffer_new(grad_output.shape))
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_x, shape_x), unbroadcast(grad_y, shape_y)

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op('a*b', x, y, buffer_new(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op('a*b', y, grad_output, buffer_new(binary_broadcast(y.shape, grad_output.shape)))
    grad_y = binary_op('a*b', x, grad_output, buffer_new(binary_broadcast(x.shape, grad_output.shape)))
    return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

class Pow(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op('pow(a,b)', x, y, buffer_new(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x_inter = binary_op('b * (pow((float)a, (float)(b-1.0)))', x, y, buffer_new(grad_output.shape))
    grad_x = binary_op('a*b', grad_output, grad_x_inter, buffer_new(grad_output.shape))
    grad_y_inter = binary_op('pow(a, (float)b) * log(a);', x, y, buffer_new(grad_output.shape))
    grad_y = binary_op('a*b', grad_output, grad_y_inter, buffer_new(grad_output.shape))
    return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

# ************* movement ops *************

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    shape = tuple(-np.prod(x.shape) // np.prod(shape) if s == -1 else s for s in shape)
    r = GPUBuffer(shape, hostbuf=x)   # NOTE: this is not a copy
    assert np.prod(x.shape) == np.prod(r.shape)
    return r

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return GPUBuffer(in_shape, hostbuf=grad_output)

class Transpose(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    return perm_axis(x, order)

  def backward(ctx, grad_output):
    return perm_axis(grad_output, np.argsort(ctx.order))

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
    assert input.shape[-1] == weight.shape[-2]
    ret = buffer_new(list(input.shape[0:-1])+[weight.shape[-1]])
    ctx.save_for_backward(input, weight)
    return matmul(input, weight, ret)

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = buffer_new(input.shape)
    grad_weight = buffer_new(weight.shape)
    matmul(grad_output, weight, grad_input, transpose_b=True)
    matmul(input, grad_output, grad_weight, transpose_a=True)
    return grad_input, grad_weight

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    if isinstance(ctx.stride, int): ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    if cin*ctx.groups != cin_: raise Exception(f"Input Tensor shape {x.shape} does not match the shape of the weights {w.shape}. ({cin*ctx.groups} vs. {cin_})")
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    ctx.save_for_backward(x,w)

    # output buffer
    conv_args = H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs
    return conv(x, w, buffer_new((bs, cout, oy, ox)), conv_args)

  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    x, w = ctx.saved_tensors
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    conv_args = H, W, ctx.groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs
    dw = convdw(x, grad_output, buffer_new((cout, cin, H, W)), conv_args)
    dx = convdx(w, grad_output, buffer_new((bs, cin_, iy, ix), zero=True), conv_args)
    return dx, dw
