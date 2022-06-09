import numpy as np
from tinygrad.helpers import binary_broadcast, UnaryOps, BinaryOps, ReduceOps
from tinygrad.tensor import Function

ll = None
def select_llops(ops):
  global ll
  import importlib
  ll = importlib.import_module(f".{ops}", f"tinygrad.llops")
  return ll.Buffer

# ************* unary ops *************

class UnaryOp(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return ll.unary_op(ctx.fop, input, ll.Buffer(input.shape))

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return ll.binary_op(ctx.bop, input, grad_output, ll.Buffer(input.shape))

class ReLU(UnaryOp):
  fop = UnaryOps.RELU

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = ll.Buffer(input.shape)
    ll.unary_op(UnaryOps.SIGN, input, ret)
    ll.unary_op(UnaryOps.RELU, ret, ret)
    return ll.binary_op(BinaryOps.MUL, ret, grad_output, ret)

class Log(UnaryOp):
  fop = UnaryOps.LOG
  bop = BinaryOps.DIV

class Exp(UnaryOp):
  def forward(ctx, input):
    ret = ll.unary_op(UnaryOps.EXP, input, ll.Buffer(input.shape))
    ctx.save_for_backward(ret)   # we save the output here, not the input
    return ret

  bop = BinaryOps.MUL

# ************* reduce ops *************

def reduce_shape(shape, axis):
  return [1 if i in axis else shape[i] for i in range(len(shape))]

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input.shape)
    return ll.reduce_op(ReduceOps.SUM, input, ll.Buffer(reduce_shape(input.shape, axis)))

  def backward(ctx, grad_output):
    shape_input, = ctx.saved_tensors
    # NOTE: the b Buffer isn't used, since this is just for broadcast
    ret = ll.Buffer(shape_input)
    return ll.binary_op(BinaryOps.A, grad_output, ret, ret)

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = ll.reduce_op(ReduceOps.MAX, input, ll.Buffer(reduce_shape(input.shape, axis)))
    ctx.save_for_backward(input, ret)
    return ret

  def backward(ctx, grad_output):
    input, ret = ctx.saved_tensors
    ret2 = ll.binary_op(BinaryOps.CMPEQ, input, ret, ll.Buffer(input.shape))
    div = ll.reduce_op(ReduceOps.SUM, ret2, ll.Buffer(grad_output.shape))
    ll.binary_op(BinaryOps.DIV, div, ret2, ret2)
    return ll.binary_op(BinaryOps.MUL, ret2, grad_output, ret2)

# ************* binary ops *************

def unbroadcast(out, in_sh):
  return ll.reduce_op(ReduceOps.SUM, out, ll.Buffer(in_sh))

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    buf = ll.Buffer(binary_broadcast(x.shape, y.shape))
    return ll.binary_op(BinaryOps.ADD, x, y, buf) #ll.Buffer(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_output, shape_x) if ctx.needs_input_grad[0] else None, \
           unbroadcast(grad_output, shape_y) if ctx.needs_input_grad[1] else None

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return ll.binary_op(BinaryOps.SUB, x, y, ll.Buffer(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    neg_grad_output = ll.unary_op(UnaryOps.NEG, grad_output, ll.Buffer(grad_output.shape))
    return unbroadcast(grad_output, shape_x) if ctx.needs_input_grad[0] else None, \
           unbroadcast(neg_grad_output, shape_y) if ctx.needs_input_grad[1] else None

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return ll.binary_op(BinaryOps.MUL, x, y, ll.Buffer(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    tmp = ll.Buffer(grad_output.shape)
    grad_x = unbroadcast(ll.binary_op(BinaryOps.MUL, y, grad_output, tmp), x.shape) if ctx.needs_input_grad[0] else None
    grad_y = unbroadcast(ll.binary_op(BinaryOps.MUL, x, grad_output, tmp), y.shape) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

class Pow(Function):
  def forward(ctx, x, y):
    ret = ll.Buffer(binary_broadcast(x.shape, y.shape))
    ctx.save_for_backward(x, y, ret)
    return ll.binary_op(BinaryOps.POW, x, y, ret)

  def backward(ctx, grad_output):
    x,y,powxy = ctx.saved_tensors
    tmp = ll.Buffer(grad_output.shape)
    ll.binary_op(BinaryOps.DIV, x, powxy, tmp)      # pow(x,y)/x
    ll.binary_op(BinaryOps.MUL, y, tmp, tmp)        # y * pow(x,y)/x
    grad_x = unbroadcast(ll.binary_op(BinaryOps.MUL, grad_output, tmp, tmp), x.shape) if ctx.needs_input_grad[0] else None
    log_x = ll.unary_op(UnaryOps.LOG, x, ll.Buffer(x.shape))
    ll.binary_op(BinaryOps.MUL, log_x, powxy, tmp)    # log(x) * pow(x,y)
    grad_y = unbroadcast(ll.binary_op(BinaryOps.MUL, grad_output, tmp, tmp), y.shape) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

# ************* movement ops *************

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    shape = tuple(-np.prod(x.shape) // np.prod(shape) if s == -1 else s for s in shape)
    return ll.reshape(x, shape) # NOTE: this is not a copy

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return ll.reshape(grad_output, in_shape)

class Transpose(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    ret = ll.Buffer([x.shape[i] for i in order])
    return ll.perm_axis(x, order, ret)

  def backward(ctx, grad_output):
    norder = np.argsort(ctx.order).tolist()
    ret = ll.Buffer([grad_output.shape[i] for i in norder])
    return ll.perm_axis(grad_output, norder, ret)

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape)
    ret = ll.Buffer([y[1]-y[0] for y in arg])
    return ll.inner_slice(x, arg, ret)

  def backward(ctx, grad_output):
    shape, = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    ret = ll.Buffer([y[1]-y[0] for y in narg])
    return ll.inner_slice(grad_output, narg, ret)

# ************* processing ops *************

class Matmul(Function):
  def forward(ctx, input, weight):
    assert input.shape[-1] == weight.shape[-2]
    ret = ll.Buffer(list(input.shape[0:-1])+[weight.shape[-1]])
    ctx.save_for_backward(input, weight)
    return ll.matmul(input, weight, ret)

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = ll.matmul(grad_output, weight, ll.Buffer(input.shape), transpose_b=True) if ctx.needs_input_grad[0] else None
    grad_weight = ll.matmul(input, grad_output, ll.Buffer(weight.shape), transpose_a=True) if ctx.needs_input_grad[1] else None
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
    conv_args = H, W, ctx.groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs
    return ll.conv(x, w, ll.Buffer((bs, cout, oy, ox)), conv_args)

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
    dx = ll.convdx(w, grad_output, ll.Buffer((bs, cin_, iy, ix)), conv_args) if ctx.needs_input_grad[0] else None
    dw = ll.convdw(x, grad_output, ll.Buffer((cout, cin, H, W)), conv_args) if ctx.needs_input_grad[1] else None
    return dx, dw