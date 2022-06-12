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
    return ctx.movement_op(MovementOps.RESHAPE, x, shape)

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return ctx.movement_op(MovementOps.RESHAPE, grad_output, in_shape)

class Permute(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    return ctx.movement_op(MovementOps.PERMUTE, x, order)

  def backward(ctx, grad_output):
    order, = ctx.saved_tensors
    norder = np.argsort(order).tolist()
    return ctx.movement_op(MovementOps.PERMUTE, grad_output, norder)

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape, arg)
    return ctx.movement_op(MovementOps.SLICE, x, arg)

  def backward(ctx, grad_output):
    shape, arg = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(arg)]
    return ctx.movement_op(MovementOps.SLICE, grad_output, narg)

# ************* processing ops *************

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    C = get_conv_args(x.shape, w.shape, stride, groups)
    ctx.save_for_backward(x,w,(C.ys,C.xs), C.groups)

    # opencl speed hacks
    # TODO: find a better way to id opencl
    if ctx.device == 2:
      print(x.shape, w.shape)

      # hack for multiples of 4
      if C.groups == 1 and C.cin % 4 != 0:
        to_add = 4 - (C.cin % 4)
        ws = [(0, s) for s in w.shape]
        ws[1] = (0, w.shape[1]+to_add)
        w = ctx.movement_op(MovementOps.SLICE, w, ws)

        xs = [(0, s) for s in x.shape]
        xs[1] = (0, x.shape[1]+to_add)
        x = ctx.movement_op(MovementOps.SLICE, x, xs)

        C = C._replace(cin = C.cin + to_add)

      # hack for multiples of 4
      added_output_shape = None
      if C.groups == 1 and C.cout % 4 != 0:
        to_add = 4 - (C.cout % 4)
        added_output_shape = to_add
        ws = [(0, s) for s in w.shape]
        ws[0] = (0, w.shape[0]+to_add)
        w = ctx.movement_op(MovementOps.SLICE, w, ws)
        C = C._replace(cout = C.cout + to_add, rcout = C.rcout + to_add)

      # packed
      assert (C.groups*C.cin) % 4 == 0
      x = ctx.movement_op(MovementOps.PERMUTE, x, (0,2,3,1))
      x = ctx.movement_op(MovementOps.RESHAPE, x, (C.bs*C.iy, C.ix*C.groups*C.cin//4, 4))

      assert C.cout % 4 == 0
      if C.cin == 1:
        # depthwise
        w = ctx.movement_op(MovementOps.RESHAPE, w, (C.cout//4,4,C.H*C.W))
        w = ctx.movement_op(MovementOps.PERMUTE, w, (0,2,1))
      else:
        w = ctx.movement_op(MovementOps.RESHAPE, w, (C.cout//4,4,C.cin//4,4,C.H,C.W))
        w = ctx.movement_op(MovementOps.PERMUTE, w, (0,4,2,5,1,3))
        w = ctx.movement_op(MovementOps.RESHAPE, w, (C.cout//4, C.H * C.cin//4 * C.W * 4, 4))

      out_shape = (C.bs*C.oy, C.ox*C.cout//4, 4)
      ret = ctx.processing_op(ProcessingOps.CONV, x, w, out_shape, C)
      ret = ctx.movement_op(MovementOps.RESHAPE, ret, (C.bs, C.oy, C.ox, C.cout))

      if added_output_shape is not None:
        xs = [(0, s) for s in ret.shape]
        xs[3] = (0, ret.shape[3]-added_output_shape)
        ret = ctx.movement_op(MovementOps.SLICE, ret, xs)

      ret = ctx.movement_op(MovementOps.PERMUTE, ret, (0,3,1,2))
      return ret
    else:
      return ctx.processing_op(ProcessingOps.CONV, x, w, (C.bs, C.groups*C.rcout, C.oy, C.ox), C)

  def backward(ctx, grad_output):
    x, w, stride, groups = ctx.saved_tensors
    C = get_conv_args(x.shape, w.shape, stride, groups)
    dx = ctx.processing_op(ProcessingOps.CONVT, grad_output, w, x.shape, C) if ctx.needs_input_grad[0] else None
    dw = ctx.processing_op(ProcessingOps.CONVDW, x, grad_output, w.shape, C) if ctx.needs_input_grad[1] else None
    return dx, dw