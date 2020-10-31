import numpy as np
from .tensor import Function, register
import pyopencl as cl

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    print(x,y)
    res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, x.size)
    # TODO: precompile on import?
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()
    prg.sum(ctx.cl_queue, [x.size], None, x, y, res_g)
    return res_g

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)
