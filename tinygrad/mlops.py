import os
import numpy as np    # TODO: remove this, it's used for np.prod and np.argsort
from tinygrad.helpers import prod, reduce_shape, get_conv_args
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.tensor import Function

# ************* unary ops *************

class _UnaryOp(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.unary_op(ctx.fop)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return input.binary_op(ctx.bop, grad_output)

class ReLU(_UnaryOp):
  fop = UnaryOps.RELU

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = input.unary_op(UnaryOps.SIGN)
    ret = ret.unary_op(UnaryOps.RELU)
    return ret.binary_op(BinaryOps.MUL, grad_output)

class Log(_UnaryOp):
  fop = UnaryOps.LOG

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output.binary_op(BinaryOps.DIV, input)

class Exp(_UnaryOp):
  def forward(ctx, input):
    ret = input.unary_op(UnaryOps.EXP)
    ctx.save_for_backward(ret)   # we save the output here, not the input
    return ret

  bop = BinaryOps.MUL

# TODO: add Neg? confirm the optimizer on Sub good enough

# ************* reduce ops *************

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input.shape)
    return input.reduce_op(ReduceOps.SUM, reduce_shape(input.shape, axis))

  def backward(ctx, grad_output):
    shape_input, = ctx.saved_tensors
    return grad_output.movement_op(MovementOps.EXPAND, shape_input)

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = input.reduce_op(ReduceOps.MAX, reduce_shape(input.shape, axis))
    ctx.save_for_backward(input, ret)
    return ret

  def backward(ctx, grad_output):
    input, ret = ctx.saved_tensors

    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = input.binary_op(BinaryOps.CMPEQ, ret.movement_op(MovementOps.EXPAND, input.shape))

    # sum of locations, averaged
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape)
    div = div.movement_op(MovementOps.EXPAND, input.shape)
    max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

    grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, input.shape)
    return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded)

# ************* binary ops *************

class Add(Function):
  def forward(ctx, x, y):
    return x.binary_op(BinaryOps.ADD, y)

  def backward(ctx, grad_output):
    return grad_output if ctx.needs_input_grad[0] else None, \
           grad_output if ctx.needs_input_grad[1] else None

class Sub(Function):
  def forward(ctx, x, y):
    return x.binary_op(BinaryOps.SUB, y)

  def backward(ctx, grad_output):
    return grad_output if ctx.needs_input_grad[0] else None, \
           grad_output.unary_op(UnaryOps.NEG) if ctx.needs_input_grad[1] else None

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x.binary_op(BinaryOps.MUL, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = y.binary_op(BinaryOps.MUL, grad_output) if ctx.needs_input_grad[0] else None
    grad_y = x.binary_op(BinaryOps.MUL, grad_output) if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

# TODO: add Div? is the optimizer on Pow good enough?

class Pow(Function):
  def forward(ctx, x, y):
    ret = x.binary_op(BinaryOps.POW, y)
    ctx.save_for_backward(x, y, ret)
    return ret

  def backward(ctx, grad_output):
    x,y,powxy = ctx.saved_tensors
    grad_x, grad_y = None, None
    if ctx.needs_input_grad[0]:
      tmp = powxy.binary_op(BinaryOps.DIV, x)      # pow(x,y)/x
      tmp = y.binary_op(BinaryOps.MUL, tmp)        # y * pow(x,y)/x
      grad_x = grad_output.binary_op(BinaryOps.MUL, tmp)
    if ctx.needs_input_grad[1]:
      tmp = x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL, powxy)  # log(x) * pow(x,y)
      grad_y = grad_output.binary_op(BinaryOps.MUL, tmp)
    return grad_x, grad_y

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.movement_op(MovementOps.EXPAND, shape)

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.reduce_op(ReduceOps.SUM, in_shape)

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    shape = tuple(-prod(x.shape) // prod(shape) if s == -1 else s for s in shape)
    return x.movement_op(MovementOps.RESHAPE, shape)

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.movement_op(MovementOps.RESHAPE, in_shape)

class Permute(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    return x.movement_op(MovementOps.PERMUTE, order)

  def backward(ctx, grad_output):
    order, = ctx.saved_tensors
    norder = np.argsort(order).tolist()
    return grad_output.movement_op(MovementOps.PERMUTE, norder)

# TODO: merge Slice and Flip into Stride with the 3 arguments

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape, arg)
    return x.movement_op(MovementOps.SLICE, arg)

  def backward(ctx, grad_output):
    shape, arg = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(arg)]
    return grad_output.movement_op(MovementOps.SLICE, narg)

class Flip(Function):
  def forward(ctx, x, axis):
    ctx.save_for_backward(axis)
    return x.movement_op(MovementOps.FLIP, axis)

  def backward(ctx, grad_output):
    axis, = ctx.saved_tensors
    return grad_output.movement_op(MovementOps.FLIP, axis)

# ************* processing ops *************

class Conv2D(Function):
  def _conv(ctx, x, w, C):
    # TODO: this does NOT belong here
    # was pre/post processing for opencl
    return x.processing_op(ProcessingOps.CONV, w, C)

  def forward(ctx, x, w, stride=1, groups=1, dilation=1, padding=0):
    C = get_conv_args(x.shape, w.shape, stride, groups, dilation=dilation, padding=padding)
    ctx.save_for_backward(x,w,C)
    return ctx._conv(x, w, C)

  def backward(ctx, grad_output):
    x, w, C = ctx.saved_tensors
    dx, dw = None, None
    if ctx.needs_input_grad[0]:    # compute derivative of inputs using ProcessingOps.CONV (this is a transposed conv)
      xt = grad_output
      if C.sx > 1 or C.sy > 1:   # unstride. NOTE: this is really memory intensive for big strides.
        xt = xt.movement_op(MovementOps.RESHAPE, (grad_output.shape[0], grad_output.shape[1], grad_output.shape[2], 1, grad_output.shape[3], 1))
        xt = xt.movement_op(MovementOps.SLICE, ((0,xt.shape[0]), (0,xt.shape[1]), (0,xt.shape[2]), (0,C.sy), (0,xt.shape[4]), (0,C.sx)))
        xt = xt.movement_op(MovementOps.RESHAPE, (xt.shape[0], xt.shape[1], xt.shape[2]*C.sy, xt.shape[4]*C.sx))
      wt = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W))
      wt = wt.movement_op(MovementOps.FLIP, (3, 4))
      wt = wt.movement_op(MovementOps.PERMUTE, (0, 2, 1, 3, 4))
      wt = wt.movement_op(MovementOps.RESHAPE, (C.groups*C.cin, C.rcout, C.H, C.W))
      py, px = (C.H-1)*C.dy - C.py, (C.W-1)*C.dx - C.px
      py_ = x.shape[2] - xt.shape[2] + C.py
      px_ = x.shape[3] - xt.shape[3] + C.px
      Cdx = get_conv_args(xt.shape, wt.shape, dilation=(C.dy, C.dx), padding=(px, px_, py, py_), groups=C.groups)
      dx = ctx._conv(xt, wt, Cdx)

    if ctx.needs_input_grad[1]:   # compute derivative of weights using ProcessingOps.CONV
      xdw = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.iy, C.ix))
      xdw = xdw.movement_op(MovementOps.PERMUTE, (2,1,0,3,4))
      xdw = xdw.movement_op(MovementOps.RESHAPE, (C.cin, C.groups*C.bs, C.iy, C.ix))
      grad_output_dw = grad_output.movement_op(MovementOps.PERMUTE, (1,0,2,3))
      grad_output_dw = grad_output_dw.movement_op(MovementOps.RESHAPE, (C.cout, C.bs, C.oy, C.ox))
      py_ = (w.shape[2] - 1) * C.dy - xdw.shape[2] - C.py + C.sy * (grad_output_dw.shape[2]-1) + 1
      px_ = (w.shape[3] - 1) * C.dx - xdw.shape[3] - C.px + C.sx * (grad_output_dw.shape[3]-1) + 1
      Cdw = get_conv_args(xdw.shape, grad_output_dw.shape, padding=(C.px, px_, C.py, py_), stride=(C.dy, C.dx), dilation=(C.sy, C.sx), groups=C.groups)
      grad_weight = ctx._conv(xdw, grad_output_dw, Cdw)
      dw = grad_weight.movement_op(MovementOps.PERMUTE, (1,0,2,3))
    return dx, dw
