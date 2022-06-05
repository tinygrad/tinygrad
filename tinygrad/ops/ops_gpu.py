import pyopencl as cl
import numpy as np
from ..tensor import Function
from ..llops.gpu import GPUBuffer, clbuild, buffer_new, unary_op, binary_op, reduce_op, perm_axis, inner_slice, matmul, conv, convdw, convdx

i32 = np.int32

# ************* unary ops *************

class UnaryOp(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, ctx.fop, input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, ctx.bop, input, grad_output)

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
    return reduce_op(ctx, "out += a", "out", input, axis=axis)

  def backward(ctx, grad_output):
    shape_input, = ctx.saved_tensors
    # NOTE: the b buffer_new isn't used, since this is just for broadcast
    return binary_op(ctx, 'a', grad_output, buffer_new(ctx, shape_input))

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = reduce_op(ctx, "out = max(a,out)", "out", input, axis=axis, start="-INFINITY")
    ctx.save_for_backward(input, axis, ret)
    return ret

  def backward(ctx, grad_output):
    input, axis, ret = ctx.saved_tensors
    ret2 = binary_op(ctx, "1.0*(a==b)", input, ret)
    div = reduce_op(ctx, "out += a", "out+1e-10", ret2, axis=axis)
    ret3 = binary_op(ctx, "a/b", ret2, div)
    return binary_op(ctx, 'a*b', ret3, grad_output)

# ************* binary ops *************

def unbroadcast(ctx, out, in_sh):
  sum_axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  return reduce_op(ctx, "out += a", "out", out, sum_axis)

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op(ctx, 'a+b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, grad_output
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_x, shape_x), unbroadcast(ctx, grad_y, shape_y)

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op(ctx, 'a-b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, unary_op(ctx, '-a', grad_output)
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_x, shape_x), unbroadcast(ctx, grad_y, shape_y)

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'a*b', x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op(ctx, 'a*b', y, grad_output)
    grad_y = binary_op(ctx, 'a*b', x, grad_output)
    return unbroadcast(ctx, grad_x, x.shape), unbroadcast(ctx, grad_y, y.shape)

class Pow(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'pow(a,b)', x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x_inter = binary_op(ctx, 'b * (pow((float)a, (float)(b-1.0)))', x, y)
    grad_x = binary_op(ctx, 'a*b', grad_output, grad_x_inter)
    grad_y_inter = binary_op(ctx, 'pow(a, (float)b) * log(a);', x, y)
    grad_y = binary_op(ctx, 'a*b', grad_output, grad_y_inter)
    return unbroadcast(ctx, grad_x, x.shape), unbroadcast(ctx, grad_y, y.shape)

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
    return perm_axis(ctx, x, order)

  def backward(ctx, grad_output):
    return perm_axis(ctx, grad_output, np.argsort(ctx.order))

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape)
    return inner_slice(ctx, x, arg)

  def backward(ctx, grad_output):
    shape, = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    return inner_slice(ctx, grad_output, narg)

# ************* processing ops *************

class Matmul(Function):
  def forward(ctx, input, weight):
    assert input.shape[-1] == weight.shape[-2]
    ret = buffer_new(ctx, list(input.shape[0:-1])+[weight.shape[-1]])
    ctx.save_for_backward(input, weight)
    return matmul(input, weight, ret)

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = buffer_new(ctx, input.shape)
    grad_weight = buffer_new(ctx, weight.shape)
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
    return conv(x, w, buffer_new(ctx, (bs, cout, oy, ox)), conv_args)

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
    dw = convdw(x, grad_output, buffer_new(ctx, (cout, cin, H, W)), conv_args)
    dx = convdx(w, grad_output, buffer_new(ctx, (bs, cin_, iy, ix), zero=True), conv_args)
    return dx, dw
