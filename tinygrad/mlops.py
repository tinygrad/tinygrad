import numpy as np    # TODO: remove this, it's used for np.prod and np.argsort
from tinygrad.helpers import binary_broadcast, get_conv_args, UnaryOps, BinaryOps, ReduceOps
from tinygrad.tensor import Function

# ************* unary ops *************

class _UnaryOp(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return ctx.op.unary_op(ctx.fop, input, ctx.buffer(input.shape))

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return ctx.op.binary_op(ctx.bop, input, grad_output, ctx.buffer(input.shape))

class ReLU(_UnaryOp):
  fop = UnaryOps.RELU

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = ctx.buffer(input.shape)
    ctx.op.unary_op(UnaryOps.SIGN, input, ret)
    ctx.op.unary_op(UnaryOps.RELU, ret, ret)
    return ctx.op.binary_op(BinaryOps.MUL, ret, grad_output, ret)

class Log(_UnaryOp):
  fop = UnaryOps.LOG
  bop = BinaryOps.DIV

class Exp(_UnaryOp):
  def forward(ctx, input):
    ret = ctx.op.unary_op(UnaryOps.EXP, input, ctx.buffer(input.shape))
    ctx.save_for_backward(ret)   # we save the output here, not the input
    return ret

  bop = BinaryOps.MUL

# ************* reduce ops *************

def reduce_shape(shape, axis):
  return [1 if i in axis else shape[i] for i in range(len(shape))]

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input.shape)
    return ctx.op.reduce_op(ReduceOps.SUM, input, ctx.buffer(reduce_shape(input.shape, axis)))

  def backward(ctx, grad_output):
    shape_input, = ctx.saved_tensors
    # NOTE: the b Buffer isn't used, since this is just for broadcast
    ret = ctx.buffer(shape_input)
    return ctx.op.binary_op(BinaryOps.A, grad_output, ret, ret)

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = ctx.op.reduce_op(ReduceOps.MAX, input, ctx.buffer(reduce_shape(input.shape, axis)))
    ctx.save_for_backward(input, ret)
    return ret

  def backward(ctx, grad_output):
    input, ret = ctx.saved_tensors
    ret2 = ctx.op.binary_op(BinaryOps.CMPEQ, input, ret, ctx.buffer(input.shape))
    div = ctx.op.reduce_op(ReduceOps.SUM, ret2, ctx.buffer(grad_output.shape))
    ctx.op.binary_op(BinaryOps.DIV, div, ret2, ret2)
    return ctx.op.binary_op(BinaryOps.MUL, ret2, grad_output, ret2)

# ************* binary ops *************

def unbroadcast(ctx, out, in_sh):
  return ctx.op.reduce_op(ReduceOps.SUM, out, ctx.buffer(in_sh))

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    buf = ctx.buffer(binary_broadcast(x.shape, y.shape))
    return ctx.op.binary_op(BinaryOps.ADD, x, y, buf) #ctx.buffer(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_output, shape_x) if ctx.needs_input_grad[0] else None, \
           unbroadcast(ctx, grad_output, shape_y) if ctx.needs_input_grad[1] else None

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return ctx.op.binary_op(BinaryOps.SUB, x, y, ctx.buffer(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    neg_grad_output = ctx.op.unary_op(UnaryOps.NEG, grad_output, ctx.buffer(grad_output.shape))
    return unbroadcast(ctx, grad_output, shape_x) if ctx.needs_input_grad[0] else None, \
           unbroadcast(ctx, neg_grad_output, shape_y) if ctx.needs_input_grad[1] else None

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return ctx.op.binary_op(BinaryOps.MUL, x, y, ctx.buffer(binary_broadcast(x.shape, y.shape)))

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    tmp = ctx.buffer(grad_output.shape)
    grad_x = unbroadcast(ctx, ctx.op.binary_op(BinaryOps.MUL, y, grad_output, tmp), x.shape) if ctx.needs_input_grad[0] else None
    grad_y = unbroadcast(ctx, ctx.op.binary_op(BinaryOps.MUL, x, grad_output, tmp), y.shape) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

class Pow(Function):
  def forward(ctx, x, y):
    ret = ctx.buffer(binary_broadcast(x.shape, y.shape))
    ctx.save_for_backward(x, y, ret)
    return ctx.op.binary_op(BinaryOps.POW, x, y, ret)

  def backward(ctx, grad_output):
    x,y,powxy = ctx.saved_tensors
    tmp = ctx.buffer(grad_output.shape)
    ctx.op.binary_op(BinaryOps.DIV, x, powxy, tmp)      # pow(x,y)/x
    ctx.op.binary_op(BinaryOps.MUL, y, tmp, tmp)        # y * pow(x,y)/x
    grad_x = unbroadcast(ctx, ctx.op.binary_op(BinaryOps.MUL, grad_output, tmp, tmp), x.shape) if ctx.needs_input_grad[0] else None
    log_x = ctx.op.unary_op(UnaryOps.LOG, x, ctx.buffer(x.shape))
    ctx.op.binary_op(BinaryOps.MUL, log_x, powxy, tmp)    # log(x) * pow(x,y)
    grad_y = unbroadcast(ctx, ctx.op.binary_op(BinaryOps.MUL, grad_output, tmp, tmp), y.shape) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

# ************* movement ops *************

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    shape = tuple(-np.prod(x.shape) // np.prod(shape) if s == -1 else s for s in shape)
    return ctx.op.reshape(x, shape) # NOTE: this is not a copy

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return ctx.op.reshape(grad_output, in_shape)

class Transpose(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    ret = ctx.buffer([x.shape[i] for i in order])
    return ctx.op.perm_axis(x, order, ret)

  def backward(ctx, grad_output):
    order, = ctx.saved_tensors
    norder = np.argsort(order).tolist()
    ret = ctx.buffer([grad_output.shape[i] for i in norder])
    return ctx.op.perm_axis(grad_output, norder, ret)

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape, arg)
    ret = ctx.buffer([y[1]-y[0] for y in arg])
    return ctx.op.inner_slice(x, arg, ret)

  def backward(ctx, grad_output):
    shape, arg = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(arg)]
    ret = ctx.buffer([y[1]-y[0] for y in narg])
    return ctx.op.inner_slice(grad_output, narg, ret)

# ************* processing ops *************

class Matmul(Function):
  def forward(ctx, input, weight):
    assert input.shape[-1] == weight.shape[-2]
    ret = ctx.buffer(list(input.shape[0:-1])+[weight.shape[-1]])
    ctx.save_for_backward(input, weight)
    return ctx.op.matmul(input, weight, ret)

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = ctx.op.matmul(grad_output, weight, ctx.buffer(input.shape), transpose_b=True) if ctx.needs_input_grad[0] else None
    grad_weight = ctx.op.matmul(input, grad_output, ctx.buffer(weight.shape), transpose_a=True) if ctx.needs_input_grad[1] else None
    return grad_input, grad_weight

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    C = get_conv_args(x.shape, w.shape, stride, groups)
    ctx.save_for_backward(x,w,(C.ys,C.xs), C.groups)
    return ctx.op.conv(x, w, ctx.buffer((C.bs, C.groups*C.rcout, C.oy, C.ox)), (C.ys,C.xs), C.groups)

  def backward(ctx, grad_output):
    x, w, stride, groups = ctx.saved_tensors
    dx = ctx.op.convdx(w, grad_output, ctx.buffer(x.shape), stride, groups) if ctx.needs_input_grad[0] else None
    dw = ctx.op.convdw(x, grad_output, ctx.buffer(w.shape), stride, groups) if ctx.needs_input_grad[1] else None
    return dx, dw