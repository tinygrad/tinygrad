import numpy as np
from .tensor import Function, register, Tensor
import pyopencl as cl

def buffer_new(ctx, shape):
  sz = 4
  for s in shape:
    sz *= s
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, sz)
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
    prg.add(ctx.cl_queue, [ret.size//4], None, x, y, ret)
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
    prg.mul(ctx.cl_queue, [ret.size//4], None, x, y, ret)
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
  def forward(ctx, x, y):
    pass

  @staticmethod
  def backward(ctx, grad_output):
    pass


