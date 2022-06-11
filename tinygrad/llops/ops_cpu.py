import numpy as np
from tinygrad.helpers import get_conv_args, UnaryOps, BinaryOps, ReduceOps

class CPUBuffer(np.ndarray):
  def relu(x): return np.maximum(x, 0)
  def exp(x): return np.exp(x)
  def log(x): return np.log(x)
  def sign(x): return np.sign(x)
  def amax(x, *args, **kwargs): return np.amax(x, *args, **kwargs)
  def permute(x, order): return x.transpose(order)
  def custompad(x, padding): return np.pad(x, padding)

  @staticmethod
  def fromCPU(x): return x
  def toCPU(x): return x

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
  if inp.shape == ret.shape:   # this is just a copy, regardless of the reduce op
    ret[:] = inp
  else:
    if ret.shape == (1,):      # full reduce
      axis = tuple(range(len(inp.shape)))
    else:
      assert len(inp.shape) == len(ret.shape)
      axis = tuple([i for i,(a,b) in enumerate(zip(inp.shape, ret.shape)) if a != b])
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

def get_tx(x, C):
  gx = x.reshape(C.bs,C.groups,C.cin,x.shape[2],x.shape[3])
  return np.lib.stride_tricks.as_strided(gx,
    shape=(C.bs, C.groups, C.cin, C.oy, C.ox, C.H, C.W),
    strides=(*gx.strides[0:3], gx.strides[3]*C.ys, gx.strides[4]*C.xs, *gx.strides[3:5]),
    writeable=False,
  )

def conv(x,w,ret,stride,groups):
  C = get_conv_args(x.shape, w.shape, stride, groups)
  tx = get_tx(x, C)
  tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
  tmp = np.zeros((C.bs,C.groups,C.oy,C.ox,C.rcout),dtype=x.dtype)
  for g in range(C.groups):
    #ijYXyx,kjyx -> iYXk ->ikYX
    tmp[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
  ret[:] = np.moveaxis(tmp,4,2).reshape(C.bs, C.groups*C.rcout, C.oy, C.ox)
  return ret

def convdw(x,grad_output,dw,stride,groups):
  C = get_conv_args(x.shape, dw.shape, stride, groups)
  tx = get_tx(x, C)
  ggg = grad_output.reshape(C.bs, C.groups, C.rcout, C.oy, C.ox)
  gdw = dw.reshape((C.groups, C.rcout, C.cin, C.H, C.W))
  gdw[:] = 0
  for g in range(C.groups):
    #'ikYX,ijYXyx -> kjyx'
    gdw[g] += np.tensordot(ggg[:,g], tx[:,g], ((0,2,3),(0,2,3)))
  return dw

def convdx(w,grad_output,dx,stride,groups):
  C = get_conv_args(dx.shape, w.shape, stride, groups)
  ggg = grad_output.reshape(C.bs, C.groups, C.rcout, C.oy, C.ox)
  tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
  gdx = dx.reshape((C.bs, C.groups, C.cin, C.iy, C.ix))
  gdx[:] = 0
  for k in range(C.oy*C.ox):
    Y, X = k//C.ox, k%C.ox
    iY,iX = Y*C.ys, X*C.xs
    #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
    for g in range(C.groups):
      tg = np.dot(ggg[:,g,:,Y,X].reshape(C.bs, -1), tw[g].reshape(C.rcout, -1))
      gdx[:, g, :, iY:iY+C.H, iX:iX+C.W] += tg.reshape((C.bs, C.cin, C.H, C.W))
  return dx
