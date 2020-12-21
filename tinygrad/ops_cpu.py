import warnings
import numpy as np
from .tensor import Function, register

# ************* basic ops *************
def unbroadcast(out, in_sh):
  # adjoint operation to broadcast is sum. Need to sum all axis with 1 = in_sh[i] < out.shape[i]
  sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
  return out.sum(axis=sum_axis).reshape(in_sh)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_output, shape_x), unbroadcast(grad_output, shape_y)
register('add', Add)

class Sub(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return x-y

  @staticmethod
  def backward(ctx, grad_output):
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_output, shape_x), unbroadcast(-grad_output, shape_y)
register('sub', Sub)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    return (lambda x,y: x*y)(*ctx.save_for_backward(x, y))

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return unbroadcast(y*grad_output, x.shape), unbroadcast(x*grad_output, y.shape)
register('mul', Mul)

class Pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    return (lambda x,y: x**y)(*ctx.save_for_backward(x, y))


  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return unbroadcast(y * (x**(y-1.0)) * grad_output, x.shape), \
           unbroadcast((x**y) * np.log(x) * grad_output, y.shape)
register('pow', Pow)

class Sum(Function):
  @staticmethod
  def forward(ctx, _input, axis=None):
    return (lambda i, a: np.array([i.sum()]) if a is None else _input.sum(axis=a))(*ctx.save_for_backward(_input, axis))

  @staticmethod
  def backward(ctx, grad_output):
    input, axis = ctx.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    return grad_output.reshape(shape) + np.zeros_like(input)
register('sum', Sum)


# ************* GEMM *************

class Dot(Function):
  @staticmethod
  def forward(ctx, _input, weight):
    return (lambda i, w: i.dot(w))(*ctx.save_for_backward(_input, weight))

  @staticmethod
  def backward(ctx, grad_output):
    _input, weight = ctx.saved_tensors
    return grad_output.dot(weight.T), _input.T.dot(grad_output)
register('dot', Dot)

# ************* simple ops *************

class Pad2D(Function):
  @staticmethod
  def forward(ctx, x, padding=None):
    return (lambda p: np.pad(x, ((0,0), (0,0), tuple(p[2:4]), tuple(p[0:2]))))(*ctx.save_for_backward(padding))


  @staticmethod
  def backward(ctx, grad_output):
    return (lambda p: grad_output[..., p[2]:-p[3], p[0]:-p[1]])(*ctx.saved_tensors)
register('pad2d', Pad2D)

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.reshape(*ctx.saved_tensors)
register('reshape', Reshape)


# ************* activation ops *************

class ReLU(Function):
  @staticmethod
  def forward(ctx, _input):
    return np.maximum(*ctx.save_for_backward(_input), 0)

  @staticmethod
  def backward(ctx, grad_output):
    return (lambda i: grad_output * (i >= 0))(*ctx.saved_tensors)
register('relu', ReLU)

def _exp_normalize(x, axis=None):
  y = np.exp(x - x.max(axis=axis, keepdims=True))
  return y / y.sum(axis=axis, keepdims=True)

class Sigmoid(Function):
  @staticmethod
  def forward(ctx, input):
    with np.warnings.catch_warnings():
      np.warnings.filterwarnings('ignore')
      ret = np.where(input >= 0,
        1/(1 + np.exp(-input)),
        np.exp(input)/(1 + np.exp(input))
      )
    return ctx.save_for_backward(ret)[0]

  @staticmethod
  def backward(ctx, grad_output):
    return (lambda r, gout: gout * (r * (1 - r)))(*ctx.saved_tensors, grad_output)
register('sigmoid', Sigmoid)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, _input):
    return np.log(*ctx.save_for_backward(_exp_normalize(_input, axis=1)))

  @staticmethod
  def backward(ctx, grad_output):
      return (lambda gout, s: gout - gout.sum(axis=1, keepdims=True)*s)(grad_output, *ctx.saved_tensors)
register('logsoftmax', LogSoftmax)


# ************* conv ops *************

class Conv2D(Function):
  @staticmethod
  def forward(ctx, x, w, stride=1, groups=1):
    if type(ctx.stride) == int:
      ctx.stride = (ctx.stride, ctx.stride)
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

  @staticmethod
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
register('conv2d', Conv2D)


# ************* pooling ops *************

def stack_for_pool(x, py, px):
  my, mx = (x.shape[2]//py)*py, (x.shape[3]//px)*px
  xup = x[:, :, :my, :mx]
  stack = [xup[:, :, k//px::py, k%px::px][None] for k in range(py*px)]
  return np.concatenate(stack, axis=0)

def unstack_for_pool(fxn, s, py, px):
  my, mx = (s[2]//py)*py, (s[3]//px)*px
  for k in range(py*px):
    Y, X = k//px, k%px
    ll = fxn(Y*px+X)
    if X == 0 and Y == 0:
      ret = np.zeros(s, dtype=ll.dtype)
    ret[:, :, Y:my:py, X:mx:px] = ll
  return ret

class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, x, kernel_size=(2, 2)):
    stack = stack_for_pool(x, *kernel_size)
    ctx.save_for_backward(np.argmax(stack, axis=0), x.shape)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    return (lambda idxs, s: unstack_for_pool(lambda idx: grad_output * (idxs == idx), s, *ctx.kernel_size))(*ctx.saved_tensors)
register('max_pool2d', MaxPool2D)

class AvgPool2D(Function):
  @staticmethod
  def forward(ctx, x, kernel_size=(2, 2)):
    ctx.save_for_backward(x.shape)
    return np.mean(stack_for_pool(x, *kernel_size), axis=0)

  @staticmethod
  def backward(ctx, grad_output):
    return (lambda py, px, s: unstack_for_pool(lambda idx: grad_output/py/px, s, py, px))(*ctx.kernel_size, *ctx.saved_tensors)

register('avg_pool2d', AvgPool2D)
