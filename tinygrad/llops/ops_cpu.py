import numpy as np
from tinygrad.helpers import UnaryOps, BinaryOps, ReduceOps

class Buffer(np.ndarray):
  def toCPU(x): return x
  def log(x): return np.log(x)
  def exp(x): return np.exp(x)
  def sign(x): return np.sign(x)
  def relu(x): return np.maximum(x, 0)
  def type(x, tt): return x.astype(tt)
  def custompad(x, padding): return np.pad(x, padding)
  def permute(x, order): return x.transpose(order)
  def expand(x, shp): return np.broadcast_to(x, shp)
  def amax(x, *args, **kwargs): return np.amax(x, *args, **kwargs)

  @staticmethod
  def fromCPU(x): return x

def unary_op(op, x, ret):
  if op == UnaryOps.RELU: ret[:] = x.relu()
  elif op == UnaryOps.EXP: ret[:] = x.exp()
  elif op == UnaryOps.LOG: ret[:] = x.log()
  elif op == UnaryOps.NEG: ret[:] = -x
  elif op == UnaryOps.SIGN: ret[:] = x.sign()
  else: raise Exception(f"{op} isn't supported")
  return ret

def binary_op(op, x, y, ret):
  if op == BinaryOps.ADD: ret[:] = x+y
  elif op == BinaryOps.SUB: ret[:] = x-y
  elif op == BinaryOps.MUL: ret[:] = x*y
  elif op == BinaryOps.DIV: ret[:] = y/x
  elif op == BinaryOps.POW: ret[:] = x**y
  elif op == BinaryOps.A: ret[:] = x
  elif op == BinaryOps.CMPEQ: ret[:] = 1.0*(x==y)
  else: raise Exception(f"{op} isn't supported")
  return ret

def reduce_op(op, inp, ret):
  if inp.shape == ret.shape:   # this is just a copy
    ret[:] = inp
    return ret
  if ret.shape == (1,): axis=tuple(range(len(inp.shape)))
  else: axis = tuple([i for i,(a,b) in enumerate(zip(inp.shape, ret.shape)) if a != b])
  if op == ReduceOps.SUM: ret[:] = inp.sum(axis, keepdims=True)
  elif op == ReduceOps.MAX: ret[:] = inp.amax(axis, keepdims=True)
  else: raise Exception(f"{op} isn't supported")
  return ret

def reshape(x, shape):
  assert np.prod(x.shape) == np.prod(shape)
  return x.reshape(shape)

def perm_axis(x, order, ret):
  ret[:] = x.permute(order)
  return ret

def inner_slice(x, arg, ret):
  padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
  x = x.custompad(padding)
  slicee = [(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)]
  ret[:] = x[tuple([slice(x[0], x[1], None) for x in slicee])]
  return ret

def matmul(a, b, c, transpose_a=False, transpose_b=False):
  if transpose_a: a = a.swapaxes(-2, -1)
  if transpose_b: b = b.swapaxes(-2, -1)
  c[:] = a @ b
  return c

def get_tx(x, conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args
  gx = x.reshape(bs,groups,cin,x.shape[2],x.shape[3])
  return np.lib.stride_tricks.as_strided(gx,
    shape=(bs, groups, cin, oy, ox, H, W),
    strides=(*gx.strides[0:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:5]),
    writeable=False,
  )

def conv(x,w,ret,conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args
  tx = get_tx(x, conv_args)
  tw = w.reshape(groups, rcout, cin, H, W)
  tmp = np.zeros((bs,groups,oy,ox,rcout),dtype=x.dtype)
  for g in range(groups):
    #ijYXyx,kjyx -> iYXk ->ikYX
    tmp[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
  ret[:] = np.moveaxis(tmp,4,2).reshape(bs, groups*rcout, oy, ox)
  return ret

def convdw(x,grad_output,dw,conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args
  tx = get_tx(x, conv_args)
  ggg = grad_output.reshape(bs,groups,rcout,oy,ox)
  gdw = dw.reshape((groups,rcout,cin,H,W))
  gdw[:] = 0
  for g in range(groups):
    #'ikYX,ijYXyx -> kjyx'
    gdw[g] += np.tensordot(ggg[:,g], tx[:,g], ((0,2,3),(0,2,3)))
  return dw

def convdx(w,grad_output,dx,conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args
  ggg = grad_output.reshape(bs,groups,rcout,oy,ox)
  tw = w.reshape(groups, rcout, cin, H, W)
  gdx = dx.reshape((bs,groups,cin,iy,ix))
  gdx[:] = 0
  for k in range(oy*ox):
    Y, X = k//ox, k%ox
    iY,iX = Y*ys, X*xs
    #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
    for g in range(groups):
      tg = np.dot(ggg[:,g,:,Y,X].reshape(bs, -1), tw[g].reshape(rcout, -1))
      gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))
  return dx
