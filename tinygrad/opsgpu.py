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

    ret = buffer_new(ctx, (1,))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void sum(
        __global const float *a_g, int sz, __global float *res_g)
    {
      float out = 0.0;
      for (int x = 0; x < sz; x++) {
        out += a_g[x];
      }
      res_g[0] = out;
    }
    """)
    prg.sum(ctx.cl_queue, [input.shape[0]], None, input, np.int32(np.prod(input.shape)), ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = b_g[0];', input, grad_output)  # Quick hack for fill
register('sum', Sum, gpu=True)

class Dot(Function):
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
      one, msize, isize, one, osize, osize)

    return grad_input, grad_weight
register('dot', Dot, gpu=True)
register('matmul', Dot, gpu=True)

# ************* simple ops *************

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    x.shape = shape
    return x

  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    grad_output.shape = in_shape
    return grad_output
register('reshape', Reshape, gpu=True)

# ************* activation ops *************

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

class Sigmoid(Function):
  @staticmethod
  def forward(ctx, input):
    ret = unary_op(ctx, 'res_g[gid] = 1./(1+exp(-a_g[gid]));', input)
    ctx.save_for_backward(ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return binary_op(ctx, 'res_g[gid] = a_g[gid] * (b_g[gid] * (1 - b_g[gid]));', grad_output, ret)
register('sigmoid', Sigmoid, gpu=True)


# *** this is unfinished, fix this and TestMNIST.test_sgd_gpu should pass ***

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    lsum = buffer_new(ctx, (input.shape[0],))
    prg = clbuild(ctx.cl_ctx, """
    __kernel void logsoftmax(
        __global const float *a_g, int sz, __global float *res_g)
    {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      // TODO: stability with max
      float out = 0.0;
      for (int x = 0; x < sz; x++) {
        out += exp(a_g[gidsz+x]);
      }
      res_g[gid] = log(out);
    }
    """)
    prg.logsoftmax(ctx.cl_queue, [input.shape[0]], None, input, np.int32(input.shape[1]), lsum)

    output = buffer_like(ctx, input)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void lsmsub(
        __global const float *a_g, __global const float *b_g, int sz, __global float *res_g)
    {
      int gid = get_global_id(0);
      int gid2 = get_global_id(1);

      res_g[gid*sz + gid2] = a_g[gid*sz + gid2] - b_g[gid];
    }
    """)
    prg.lsmsub(ctx.cl_queue, [input.shape[0], input.shape[1]], None, input, lsum, np.int32(input.shape[1]), output)
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors

    grad_input = buffer_like(ctx, grad_output)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void lsmsub2(
        __global const float *grad_output, __global const float *output, int sz, __global float *grad_input)
    {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      int gid2 = get_global_id(1);

      // TODO: this is repeated in many kernels
      float acc = 0.0;
      for (int x = 0; x < sz; x++) {
        acc += grad_output[gidsz + x];
      }

      grad_input[gidsz + gid2] = grad_output[gidsz + gid2] - exp(output[gidsz + gid2]) * acc;
    }
    """)
    prg.lsmsub2(ctx.cl_queue, [grad_output.shape[0], grad_output.shape[1]], None,
      grad_output, output, np.int32(grad_output.shape[1]), grad_input)

    return grad_input
register('logsoftmax', LogSoftmax, gpu=True)

# ************* conv ops *************

class Conv2D(Function):
  @staticmethod
  def forward(ctx, x, w, stride=1, groups=1):
    if type(ctx.stride) == int:
      ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    # output buffer
    ret = buffer_new(ctx, (bs, cout, oy, ox))

    prg = clbuild(ctx.cl_ctx, """
    __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs) {

      int B = get_global_id(0);  // range 0-bs
      int Y = get_global_id(1);  // range 0-oy
      int X = get_global_id(2);  // range 0-ox
      int IY = Y*ys;
      int IX = X*xs;
      
      // input  = (bs, groups, cin, iy, ix)
      // weight = (groups, rcout, cin, H, W)
      // output = (bs, groups, rcout, oy, ox)
      for (int g = 0; g < groups; g++) {
        for (int c = 0; c < rcout; c++) {
          float acc = 0.0;
          for (int ci = 0; ci < cin; ci++) {
            for (int y = IY; y < IY+H; y++) {
              for (int x = IX; x < IX+W; x++) {
                acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + y*ix + x] * \
                  weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + (y-IY)*W + (x-IX)];
              }
            }
          }
          output[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] = acc;
        }
      }
    }
    """)

    prg.conv(ctx.cl_queue, [bs, oy, ox], None,
      x, w, ret,
      np.int32(H), np.int32(W),
      np.int32(groups), np.int32(rcout), np.int32(cin),
      np.int32(oy), np.int32(ox), 
      np.int32(iy), np.int32(ix),
      np.int32(ys), np.int32(xs)
    )
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    raise Exception("not implemented")

register('conv2d', Conv2D, gpu=True)

