import numpy as np
from tinygrad.tensor import Function, register

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


class Conv2D(Function):
  @staticmethod
  def forward(ctx, x, w):
    cout,cin,H,W = w.shape
    tw = w.reshape(cout, -1).T
    bs,oy,ox = x.shape[0], x.shape[2]-(H-1), x.shape[3]-(W-1)

    ctx.save_for_backward(x, w)
    ret = np.zeros((bs, cout, oy, ox), dtype=w.dtype)
    for Y in range(oy):
      for X in range(ox):
        tx = x[:, :, Y:Y+H, X:X+W].reshape(bs, -1)
        ret[:, :, Y, X] = tx.dot(tw)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    x, w = ctx.saved_tensors
    cout,cin,H,W = w.shape
    tw = w.reshape(cout, -1)

    dx, dw = np.zeros_like(x), np.zeros_like(w)
    for Y in range(grad_output.shape[2]):
      for X in range(grad_output.shape[3]):
        gg = grad_output[:, :, Y, X]
        tx = x[:, :, Y:Y+H, X:X+W].reshape(x.shape[0], -1)
        dw += gg.T.dot(tx).reshape(dw.shape)
        dx[:, :, Y:Y+H, X:X+W] += gg.dot(tw).reshape(dx.shape[0], dx.shape[1], H, W)
    return dx, dw
register('conv2d', Conv2D)

class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, x):
    my, mx = (x.shape[2]//2)*2, (x.shape[3]//2)*2
    stack = []
    xup = x[:, :, :my, :mx]
    for Y in range(2):
      for X in range(2):
        stack.append(xup[:, :, Y::2, X::2][None])
    stack = np.concatenate(stack, axis=0)
    idxs = np.argmax(stack, axis=0)
    ctx.save_for_backward(idxs, x.shape)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    idxs,s = ctx.saved_tensors
    my, mx = (s[2]//2)*2, (s[3]//2)*2
    ret = np.zeros(s, dtype=grad_output.dtype)
    for Y in range(2):
      for X in range(2):
        ret[:, :, Y:my:2, X:mx:2] = grad_output * (idxs == (Y*2+X))
    return ret
register('max_pool2d', MaxPool2D)

