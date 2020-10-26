import numpy as np
from .tensor import Function, register

# ************* basic ops *************

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add)
    
class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register('dot', Dot)
register('matmul', Dot)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)


# ************* nn ops *************

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output * (input >= 0)
    return grad_input
register('relu', ReLU)

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)

  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.reshape(in_shape), None
register('reshape', Reshape)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      #return np.log(np.exp(x).sum(axis=1))
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)


# ************* conv ops *************

class Conv2D(Function):
  @staticmethod
  def forward(ctx, x, w, stride=1):
    if type(ctx.stride) == int:
      ctx.stride = (ctx.stride, ctx.stride)

    cout,cin,H,W = w.shape
    tw = w.reshape(cout, -1).T
    ys,xs = ctx.stride
    bs,oy,ox = x.shape[0], (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs

    ctx.save_for_backward(x, w)
    ret = np.zeros((bs, cout, oy, ox), dtype=w.dtype)
    for Y in range(oy):
      for X in range(ox):
        iY,iX = Y*ys, X*xs
        tx = x[:, :, iY:iY+H, iX:iX+W].reshape(bs, -1)
        ret[:, :, Y, X] = tx.dot(tw)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    x, w = ctx.saved_tensors
    cout,cin,H,W = w.shape
    tw = w.reshape(cout, -1)
    ys,xs = ctx.stride

    dx, dw = np.zeros_like(x), np.zeros_like(w)
    for Y in range(grad_output.shape[2]):
      for X in range(grad_output.shape[3]):
        iY,iX = Y*ys, X*xs
        gg = grad_output[:, :, Y, X]
        tx = x[:, :, iY:iY+H, iX:iX+W].reshape(x.shape[0], -1)
        dw += gg.T.dot(tx).reshape(dw.shape)
        dx[:, :, iY:iY+H, iX:iX+W] += gg.dot(tw).reshape(dx.shape[0], dx.shape[1], H, W)
    return dx, dw
register('conv2d', Conv2D)


# ************* pooling ops *************

def stack_for_pool(x, py, px):
  my, mx = (x.shape[2]//py)*py, (x.shape[3]//px)*px
  stack = []
  xup = x[:, :, :my, :mx]
  for Y in range(py):
    for X in range(px):
      stack.append(xup[:, :, Y::py, X::px][None])
  return np.concatenate(stack, axis=0)

def unstack_for_pool(fxn, s, py, px):
  my, mx = (s[2]//py)*py, (s[3]//px)*px
  for Y in range(py):
    for X in range(px):
      ll = fxn(Y*px+X)
      if X == 0 and Y == 0:
        ret = np.zeros(s, dtype=ll.dtype)
      ret[:, :, Y:my:py, X:mx:px] = ll
  return ret

class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, x, kernel_size=(2, 2)):
    stack = stack_for_pool(x, *kernel_size)
    idxs = np.argmax(stack, axis=0)
    ctx.save_for_backward(idxs, x.shape)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    idxs,s = ctx.saved_tensors
    return unstack_for_pool(
      lambda idx: grad_output * (idxs == idx),
      s, *ctx.kernel_size)
register('max_pool2d', MaxPool2D)

class AvgPool2D(Function):
  @staticmethod
  def forward(ctx, x, kernel_size=(2, 2)):
    stack = stack_for_pool(x, *kernel_size)
    ctx.save_for_backward(x.shape)
    return np.mean(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    s, = ctx.saved_tensors
    py, px = ctx.kernel_size
    return unstack_for_pool(
      lambda idx: grad_output/py/px,
      s, py, px)
register('avg_pool2d', AvgPool2D)

