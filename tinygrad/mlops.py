import numpy as np    # TODO: remove this, it's used for np.prod and np.argsort
from tinygrad.helpers import prod, reduce_shape, get_conv_args
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.tensor import Function

# ************* unary ops *************

class _UnaryOp(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return ctx.unary_op(ctx.fop, input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return ctx.binary_op(ctx.bop, input, grad_output)

class ReLU(_UnaryOp):
  fop = UnaryOps.RELU

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = ctx.unary_op(UnaryOps.SIGN, input)
    ret = ctx.unary_op(UnaryOps.RELU, ret)
    return ctx.binary_op(BinaryOps.MUL, ret, grad_output)

class Log(_UnaryOp):
  fop = UnaryOps.LOG
  bop = BinaryOps.DIV

class Exp(_UnaryOp):
  def forward(ctx, input):
    ret = ctx.unary_op(UnaryOps.EXP, input)
    ctx.save_for_backward(ret)   # we save the output here, not the input
    return ret

  bop = BinaryOps.MUL

# ************* reduce ops *************

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input.shape)
    return ctx.reduce_op(ReduceOps.SUM, input, reduce_shape(input.shape, axis))

  def backward(ctx, grad_output):
    shape_input, = ctx.saved_tensors
    # NOTE: the b Buffer isn't used, since this is just for broadcast
    ret = ctx.buffer(shape_input)
    return ctx.binary_op(BinaryOps.A, grad_output, ret)

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = ctx.reduce_op(ReduceOps.MAX, input, reduce_shape(input.shape, axis))
    ctx.save_for_backward(input, ret)
    return ret

  def backward(ctx, grad_output):
    input, ret = ctx.saved_tensors
    ret2 = ctx.binary_op(BinaryOps.CMPEQ, input, ret)
    div = ctx.reduce_op(ReduceOps.SUM, ret2, grad_output.shape)
    ret2 = ctx.binary_op(BinaryOps.DIV, div, ret2)
    return ctx.binary_op(BinaryOps.MUL, ret2, grad_output)

# ************* binary ops *************

def unbroadcast(ctx, out, in_sh):
  return ctx.reduce_op(ReduceOps.SUM, out, in_sh)

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return ctx.binary_op(BinaryOps.ADD, x, y)

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_output, shape_x) if ctx.needs_input_grad[0] else None, \
           unbroadcast(ctx, grad_output, shape_y) if ctx.needs_input_grad[1] else None

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return ctx.binary_op(BinaryOps.SUB, x, y)

  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    neg_grad_output = ctx.unary_op(UnaryOps.NEG, grad_output)
    return unbroadcast(ctx, grad_output, shape_x) if ctx.needs_input_grad[0] else None, \
           unbroadcast(ctx, neg_grad_output, shape_y) if ctx.needs_input_grad[1] else None

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return ctx.binary_op(BinaryOps.MUL, x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = unbroadcast(ctx, ctx.binary_op(BinaryOps.MUL, y, grad_output), x.shape) if ctx.needs_input_grad[0] else None
    grad_y = unbroadcast(ctx, ctx.binary_op(BinaryOps.MUL, x, grad_output), y.shape) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

class Pow(Function):
  def forward(ctx, x, y):
    ret = ctx.binary_op(BinaryOps.POW, x, y)
    ctx.save_for_backward(x, y, ret)
    return ret

  def backward(ctx, grad_output):
    x,y,powxy = ctx.saved_tensors
    tmp = ctx.binary_op(BinaryOps.DIV, x, powxy)      # pow(x,y)/x
    tmp = ctx.binary_op(BinaryOps.MUL, y, tmp)        # y * pow(x,y)/x
    grad_x = unbroadcast(ctx, ctx.binary_op(BinaryOps.MUL, grad_output, tmp), x.shape) if ctx.needs_input_grad[0] else None
    tmp = ctx.binary_op(BinaryOps.MUL, ctx.unary_op(UnaryOps.LOG, x), powxy)  # log(x) * pow(x,y)
    grad_y = unbroadcast(ctx, ctx.binary_op(BinaryOps.MUL, grad_output, tmp), y.shape) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

# ************* movement ops *************

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    shape = tuple(-prod(x.shape) // prod(shape) if s == -1 else s for s in shape)
    return ctx.movement_op(MovementOps.RESHAPE, x, ctx.buffer(shape))

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return ctx.movement_op(MovementOps.RESHAPE, grad_output, ctx.buffer(in_shape))

class Permute(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    ret = ctx.buffer([x.shape[i] for i in order])
    return ctx.movement_op(MovementOps.PERMUTE, x, ret, order)

  def backward(ctx, grad_output):
    order, = ctx.saved_tensors
    norder = np.argsort(order).tolist()
    ret = ctx.buffer([grad_output.shape[i] for i in norder])
    return ctx.movement_op(MovementOps.PERMUTE, grad_output, ret, norder)

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape, arg)
    ret = ctx.buffer([y[1]-y[0] for y in arg])
    return ctx.movement_op(MovementOps.SLICE, x, ret, arg)

  def backward(ctx, grad_output):
    shape, arg = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(arg)]
    ret = ctx.buffer([y[1]-y[0] for y in narg])
    return ctx.movement_op(MovementOps.SLICE, grad_output, ret, narg)

# ************* processing ops *************

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    C = get_conv_args(x.shape, w.shape, stride, groups)
    ctx.save_for_backward(x,w,(C.ys,C.xs), C.groups)
    return ctx.processing_op(ProcessingOps.CONV, x, w, ctx.buffer((C.bs, C.groups*C.rcout, C.oy, C.ox)), (C.ys,C.xs), C.groups)

  def backward(ctx, grad_output):
    x, w, stride, groups = ctx.saved_tensors
    dx = ctx.processing_op(ProcessingOps.CONVT, grad_output, w, ctx.buffer(x.shape), stride, groups) if ctx.needs_input_grad[0] else None
    dw = ctx.processing_op(ProcessingOps.CONVDW, x, grad_output, ctx.buffer(w.shape), stride, groups) if ctx.needs_input_grad[1] else None
    return dx, dw