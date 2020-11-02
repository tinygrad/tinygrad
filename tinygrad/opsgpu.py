import numpy as np
from .tensor import Function, register, Tensor
import pyopencl as cl

def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void add(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()
    prg.add(ctx.cl_queue, [np.prod(ret.shape)], None, x, y, ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void mul(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] * b_g[gid];
    }
    """).build()
    prg.mul(ctx.cl_queue, [np.prod(ret.shape)], None, x, y, ret)
    ctx.save_for_backward(x, y, prg)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    x,y,prg = ctx.saved_tensors
    gx = buffer_like(ctx, x)
    gy = buffer_like(ctx, y)
    prg.mul(ctx.cl_queue, [gx.size//4], None, y, grad_output, gx)
    prg.mul(ctx.cl_queue, [gy.size//4], None, x, grad_output, gy)
    return gx, gy
register('mul', Mul, gpu=True)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    ret = buffer_new(ctx, (1,))
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void sum(
        __global const float *a_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[0] += a_g[gid];
    }
    """).build()
    prg.sum(ctx.cl_queue, [input.size//4], None, input, ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = Tensor(grad_output).cpu().data * np.ones(input.shape, dtype=input.dtype)
    return Tensor(ret).cuda().data
register('sum', Sum, gpu=True)

class Dot(Function):
  # TODO: write me!
  @staticmethod
  def forward(ctx, input, weight):
    assert input.shape[1] == weight.shape[0]
    isize = np.int32(input.shape[0])
    msize = np.int32(input.shape[1])
    osize = np.int32(weight.shape[1])
    one = np.int32(1)
    ret = buffer_new(ctx, (isize, osize))

    prg = cl.Program(ctx.cl_ctx, """
    __kernel void matmul(
        __global const float *input,
        __global const float *weight,
        __global float *res,
        int is0,
        int is1,
        int msize,
        int ws0,
        int ws1,
        int osize
        )
    {
      int X = get_global_id(0); // isize
      int Y = get_global_id(1); // osize
      
      float ret = 0.0;
      for (int x = 0; x < msize; x++) {
        ret += input[X * is0 + x * is1] * weight[Y * ws0 + x * ws1];
      }

      res[X * osize + Y] = ret;
    }
    """).build()
    ctx.save_for_backward(input, weight, prg)
    # (isize,msize) x (msize,osize) = (isize,osize)
    prg.matmul(ctx.cl_queue, [isize, osize], None,
      input, weight, ret,
      msize, one, msize, one, osize, osize)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, prg = ctx.saved_tensors
    isize = np.int32(input.shape[0])
    msize = np.int32(input.shape[1])
    osize = np.int32(weight.shape[1])
    one = np.int32(1)

    grad_input = buffer_like(ctx, input)
    grad_weight = buffer_like(ctx, weight)

    # (isize,osize) x (msize,osize) = (isize,msize)
    prg.matmul(ctx.cl_queue, [isize, msize], None,
      grad_output, weight, grad_input,
      osize, one, osize, osize, one, msize)

    # (isize,msize) x (isize,osize) = (msize,osize)
    prg.matmul(ctx.cl_queue, [msize, osize], None,
      input, grad_output, grad_weight,
      one, msize, isize, one, isize, osize)

    return grad_input, grad_weight
register('dot', Dot, gpu=True)
register('matmul', Dot, gpu=True)


# *** these two are unfinished, but until we fix the optimizer, it's useless ***

class ReLU(Function):
  @staticmethod
  def forward(ctx, x):
    ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void relu(
        __global const float *a_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = min(a_g[gid], (float)0.);
    }
    """).build()
    prg.relu(ctx.cl_queue, [np.prod(ret.shape)], None, x, ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output
register('relu', ReLU, gpu=True)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    return input

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output
register('logsoftmax', LogSoftmax, gpu=True)


