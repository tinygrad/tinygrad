import numpy as np
from .tensor import Function, register, Tensor
import pyopencl as cl
import pyopencl.array as pycl_array
from pyopencl.reduction import ReductionKernel
import functools

def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

@functools.lru_cache
def clbuild(cl_ctx, prg):
  return cl.Program(cl_ctx, prg).build()

@functools.lru_cache
def cl_reduct_krnl_build(cl_ctx, *args, **kwargs):
  return ReductionKernel(cl_ctx, *args, **kwargs)

def binary_op(ctx, code, x, y):
  ret = buffer_like(ctx, x)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void add(
      __global const float *a_g, __global const float *b_g, __global float *res_g)
  {
    int gid = get_global_id(0);
    """+code+"""
  }
  """)
  prg.add(ctx.cl_queue, [np.prod(ret.shape)], None, x, y, ret)
  return ret

def unary_op(ctx, code, x):
  ret = buffer_like(ctx, x)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void relu(
      __global const float *a_g, __global float *res_g)
  {
    int gid = get_global_id(0);
    """+code+"""
  }
  """)
  prg.relu(ctx.cl_queue, [np.prod(ret.shape)], None, x, ret)
  return ret

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'res_g[gid] = a_g[gid] + b_g[gid];', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)

class Sub(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'res_g[gid] = a_g[gid] - b_g[gid];', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    not_grad_output = unary_op(ctx, 'res_g[gid] = -a_g[gid];', grad_output)
    return grad_output, not_grad_output
register('sub', Sub, gpu=True)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)

    # HACK
    if y.shape == (1,):
      return binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[0];', x, y)
    elif x.shape == y.shape:
      return binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', x, y)
    else:
      raise Exception("mismatched shapes %r %r" % (x.shape, y.shape))

    return ret

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', y, grad_output),\
           binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', x, grad_output)
register('mul', Mul, gpu=True)

class Pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'res_g[gid] = pow(a_g[gid], b_g[gid]);', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    gradx = binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', grad_output,
                      binary_op(ctx, 'res_g[gid] = b_g[gid] * (pow((float)a_g[gid], (float)(b_g[gid]-1.0)));', x, y))
    grady = binary_op(ctx, 'res_g[gid] = a_g[gid] * b_g[gid];', grad_output,
                      binary_op(ctx, 'res_g[gid] = pow((float)a_g[gid], (float)b_g[gid]) * log(a_g[gid]);', x, y))
    return gradx, grady
register('pow', Pow, gpu=True)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    krnl = cl_reduct_krnl_build(ctx.cl_ctx, np.float32, neutral="0", reduce_expr="a+b", 
      map_expr="x[i]", arguments="__global float *x")
    ret = krnl(pycl_array.Array(ctx.cl_queue, input.size, dtype=np.float32, data=input)).data
    ret.shape = (1,)
    ret.dtype = np.float32
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = b_g[0];', input, grad_output)  # Quick hack for fill
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

    prg = clbuild(ctx.cl_ctx, """
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
    """)
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


# *** these two are unfinished, optimizer fixed, fix this and TestMNIST.test_sgd_gpu should pass ***

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, 'res_g[gid] = max(a_g[gid], (float)0.);', input)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = a_g[gid] * (b_g[gid] >= 0);', grad_output, input)
register('relu', ReLU, gpu=True)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    return input

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output
register('logsoftmax', LogSoftmax, gpu=True)


