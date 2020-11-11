import numpy as np
from .tensor import Function, register, Tensor
import pyopencl as cl
import functools

def buffer_new(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, 4*np.prod(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_zeros(ctx, shape):
  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(shape))
  res_g.shape = shape
  res_g.dtype = np.float32
  return res_g

def buffer_like(ctx, x):
  return buffer_new(ctx, x.shape)

@functools.lru_cache()
def clbuild(cl_ctx, prg):
  return cl.Program(cl_ctx, prg).build()

def uint2(x, y):
  return np.array((x,y), dtype=cl.cltypes.uint2)
def i32(x):
  return np.int32(x)

def cl_subsample_krnl_build(cl_ctx, iter_op, result_op, decls=''):
  prg = """
  __kernel void subsample(__global float *output, __global const float *input, uint2 osize, uint2 isize,
                          uint2 ksz, uint2 stride) {
    int3 gid = (int3)(get_global_id(2), get_global_id(1), get_global_id(0));
    int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
    """+decls+""";
    for (uint j=0; j<ksz.y; ++j) {
      for (uint i=0; i<ksz.x; ++i) {
        int iid  = (gid.x*stride.x+i) + isize.x*((gid.y*stride.y+j) + isize.y*gid.z);
        if (gid.x*stride.x+i < isize.x && gid.y*stride.y+j < isize.y) {
          """+iter_op+""";
        }
      }
    }
    output[oid] = """+result_op+""";
  }"""
  return clbuild(cl_ctx, prg)

def subsample_op(ctx, input, kernel_size, stride, iter_op, result_op, decls=''):
  py, px = stride
  N, C, Yin, Xin = input.shape
  Yout, Xout = (Yin-kernel_size[0])//py+1, (Xin-kernel_size[1])//px+1
  ret = buffer_zeros(ctx, (N, C, Yout, Xout))
  prg = cl_subsample_krnl_build(ctx.cl_ctx, iter_op, result_op, decls=decls)
  prg.subsample(ctx.cl_queue, (N*C, Yout, Xout), None,
                ret, input, uint2(Xout, Yout), uint2(Xin, Yin),
                uint2(*kernel_size[::-1]), uint2(px, py))
  ctx.data = np.empty((N, C, Yout, Xout)) # set shape expectation on tensor instance
  return ret

def cl_supsample_krnl_build(cl_ctx, result_op, decls=''):
  prg = """
  __kernel void supsample(__global float *output, __global const float *input, __global const void *input2,
                          uint2 osize, uint2 isize, uint2 ksz) {
    int3 gid = (int3)(get_global_id(2), get_global_id(1), get_global_id(0));
    int oid = gid.x + osize.x*(gid.y + osize.y*gid.z);
    int iid = (gid.x/ksz.x) + isize.x*((gid.y/ksz.y) + isize.y*gid.z);
    """+decls+""";
    if (gid.x/ksz.x < isize.x && gid.y/ksz.y < isize.y) {
      output[oid] = """+result_op+""";
    }
  }"""
  return clbuild(cl_ctx, prg)

def supersample_op(ctx, input, out_shape, kernel_size, result_op, decls='', input2=None):
  (N, C, Yin, Xin), (Yout, Xout) = input.shape, out_shape[2:]
  py,px = kernel_size
  ret = buffer_zeros(ctx, out_shape)
  prg = cl_supsample_krnl_build(ctx.cl_ctx, result_op, decls=decls)
  prg.supsample(ctx.cl_queue, (N*C, Yout, Xout), None,
                ret, input, input2, uint2(Xout, Yout), uint2(Xin, Yin), uint2(px, py))
  ctx.data = np.empty((N, C, Yout, Xout)) # set shape expectation on tensor instance
  return ret

def binary_op(ctx, code, x, y):
  if len(x.shape) != len(y.shape) and y.shape != (1,):
    raise Exception("shape mismatch in binop %s: %r %r" % (code, x.shape, y.shape))
  xdiv = 1
  ydiv = 1
  if x.shape != y.shape:
    # special case broadcasting
    # TODO: make general
    if len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and y.shape[2] == 1 and y.shape[3] == 1:
      ydiv = x.shape[2] * x.shape[3]
    elif len(y.shape) == 4 and x.shape[0:2] == y.shape[0:2] and x.shape[2] == 1 and x.shape[3] == 1:
      xdiv = y.shape[2] * y.shape[3]
    elif len(x.shape) == 2 and x.shape[0] == y.shape[0] and y.shape[1] == 1:
      ydiv = x.shape[1]
    elif np.prod(y.shape) == 1:
      ydiv = np.prod(x.shape)
    else:
      raise Exception("binary op shape mismatch: %r != %r" % (x.shape, y.shape))
  ret = buffer_like(ctx, x if np.prod(x.shape) >= np.prod(y.shape) else y)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void binop(__global const float *a_g, __global const float *b_g, __global float *res_g,
                      int xdiv, int ydiv) {
    int gid = get_global_id(0);
    float a = a_g[gid/xdiv];
    float b = b_g[gid/ydiv];
    res_g[gid] = """+code+""";
  }""")
  prg.binop(ctx.cl_queue, [np.prod(ret.shape)], None, x, y, ret, i32(xdiv), i32(ydiv))
  return ret

def unary_op(ctx, code, x):
  ret = buffer_like(ctx, x)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void unop(__global const float *a_g, __global float *res_g) {
    int gid = get_global_id(0);
    float a = a_g[gid];
    res_g[gid] = """+code+""";
  }""")
  prg.unop(ctx.cl_queue, [np.prod(ret.shape)], None, x, ret)
  return ret

def reduce_op(ctx, code, code2, input, osize):
  ret = buffer_new(ctx, osize)
  prg = clbuild(ctx.cl_ctx, """
  __kernel void reduce(__global const float *a_g, int sz, __global float *res_g) {
    int gid = get_global_id(0);
    float out = 0.0;
    for (int x = 0; x < sz; x++) {
      float a = a_g[gid*sz + x];
      """+code+""";
    }
    res_g[gid] = """+code2+""";
  }""")
  prg.reduce(ctx.cl_queue, osize, None, input, i32(np.prod(input.shape) // np.prod(osize)), ret)
  return ret

# ***** now for the ops themselves *****

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'a+b', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)

class Sub(Function):
  @staticmethod
  def forward(ctx, x, y):
    return binary_op(ctx, 'a-b', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, unary_op(ctx, '-a', grad_output)
register('sub', Sub, gpu=True)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'a*b', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return binary_op(ctx, 'a*b', y, grad_output),\
           binary_op(ctx, 'a*b', x, grad_output)
register('mul', Mul, gpu=True)

class Pow(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(ctx, 'pow(a,b)', x, y)

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    gradx = binary_op(ctx, 'a*b', grad_output,
                      binary_op(ctx, 'b * (pow((float)a, (float)(b-1.0)))', x, y))
    grady = binary_op(ctx, 'a*b', grad_output,
                      binary_op(ctx, 'pow(a, (float)b) * log(a);', x, y))
    return gradx, grady
register('pow', Pow, gpu=True)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return reduce_op(ctx, "out += a", "out", input, (1,))

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    ret = buffer_like(ctx, input)

    prg = clbuild(ctx.cl_ctx, """
    __kernel void fill(__global const float *a_g, __global float *res_g) {
      int gid = get_global_id(0);
      res_g[gid] = a_g[0];
    }""")
    prg.fill(ctx.cl_queue, [np.prod(ret.shape)], None, grad_output, ret)
    return ret

register('sum', Sum, gpu=True)

class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    assert input.shape[1] == weight.shape[0]
    isize, msize, osize = i32(input.shape[0]), i32(input.shape[1]), i32(weight.shape[1])
    ret = buffer_new(ctx, (isize, osize))

    prg = clbuild(ctx.cl_ctx, """
    __kernel void matmul(
        __global const float *input,
        __global const float *weight,
        __global float *res,
        int is0, int is1, int msize,
        int ws0, int ws1, int osize
   ) {
      int X = get_global_id(0); // isize
      int Y = get_global_id(1); // osize

      float ret = 0.0;
      for (int x = 0; x < msize; x++) {
        ret += input[X * is0 + x * is1] * weight[Y * ws0 + x * ws1];
      }

      res[X * osize + Y] = ret;
    }""")
    ctx.save_for_backward(input, weight, prg)
    # (isize,msize) x (msize,osize) = (isize,osize)
    prg.matmul(ctx.cl_queue, [isize, osize], None,
      input, weight, ret,
      msize, i32(1), msize, i32(1), osize, osize)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, prg = ctx.saved_tensors
    isize, msize, osize = i32(input.shape[0]), i32(input.shape[1]), i32(weight.shape[1])

    grad_input = buffer_like(ctx, input)
    grad_weight = buffer_like(ctx, weight)

    # (isize,osize) x (msize,osize) = (isize,msize)
    prg.matmul(ctx.cl_queue, [isize, msize], None,
      grad_output, weight, grad_input,
      osize, i32(1), osize, osize, i32(1), msize)

    # (isize,msize) x (isize,osize) = (msize,osize)
    prg.matmul(ctx.cl_queue, [msize, osize], None,
      input, grad_output, grad_weight,
      i32(1), msize, isize, i32(1), osize, osize)

    return grad_input, grad_weight
register('dot', Dot, gpu=True)
register('matmul', Dot, gpu=True)

# ************* simple ops *************

class Pad2D(Function):
  @staticmethod
  def forward(ctx, x, padding=None):
    bs,cin,iy,ix = x.shape
    oy,ox = iy+padding[2]+padding[3], ix+padding[0]+padding[1]
    ret = buffer_zeros(ctx, (bs, cin, oy, ox))

    prg = clbuild(ctx.cl_ctx, """
    __kernel void pad2d(__global const float *input, __global float *output,
                        int ipx, int ipy, int py, int px, int oy, int ox, int iy, int ix) {
      int BC = get_global_id(0);
      int Y = get_global_id(1);
      int X = get_global_id(2);

      int iptr = BC*iy*ix + (Y+ipy)*ix + ipx + X;
      int optr = BC*oy*ox + (Y+py)*ox + px + X;

      output[optr] = input[iptr];
    }""")
    ctx.save_for_backward(padding, prg)
    prg.pad2d(ctx.cl_queue, [bs*cin, iy, ix], None,
        x, ret,
        i32(0), i32(0), i32(padding[2]), i32(padding[0]),
        i32(oy), i32(ox), i32(iy), i32(ix)
      )
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    padding, prg = ctx.saved_tensors
    bs, cin, iy, ix = grad_output.shape
    oy, ox = iy - padding[2] - padding[3], ix - padding[0] - padding[1]
    ret = buffer_new(ctx, (bs, cin, oy, ox))
    prg.pad2d(ctx.cl_queue, [bs*cin, oy, ox], None,
              grad_output, ret,
              i32(padding[2]), i32(padding[0]), i32(0), i32(0),
              i32(oy), i32(ox), i32(iy), i32(ix)
              )
    return ret
register('pad2d', Pad2D, gpu=True)

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    ss = list(shape)

    # I'm sorry for this code
    tsum = 1
    for s in ss:
      if s != -1:
        tsum *= s
    for i,s in enumerate(ss):
      if s == -1:
        ss[i] = np.prod(x.shape) // tsum
    assert np.prod(x.shape) == np.prod(ss)
    x = unary_op(ctx, 'a', x)
    x.shape = tuple(ss)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    grad_output = unary_op(ctx, 'a', grad_output)
    grad_output.shape = in_shape
    return grad_output
register('reshape', Reshape, gpu=True)

# ************* activation ops *************

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(ctx, 'max(a, (float)0.)', input)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'a * (b >= 0)', grad_output, input)
register('relu', ReLU, gpu=True)

class Sigmoid(Function):
  @staticmethod
  def forward(ctx, input):
    ret = unary_op(ctx, '1./(1+exp(-a))', input)
    ctx.save_for_backward(ret)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return binary_op(ctx, 'a * (b * (1 - b));', grad_output, ret)
register('sigmoid', Sigmoid, gpu=True)

class AvgPool2D(Function):
  @staticmethod
  def forward(ctx, input, kernel_size=(2, 2)):
    ret = subsample_op(ctx, input, kernel_size, kernel_size, iter_op="sumval += input[iid]",
      result_op="sumval / (ksz.x * ksz.y)", decls="float sumval=0.f")
    ctx.save_for_backward(input.shape)
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    orig_shape, = ctx.saved_tensors
    return supersample_op(ctx, grad_output, orig_shape, ctx.kernel_size,
      result_op="input[iid] / (ksz.x * ksz.y)")
register('avg_pool2d', AvgPool2D, gpu=True)

class MaxPool2D(Function):
  @staticmethod
  def forward(ctx, input, kernel_size=(2, 2)):
    idxs = subsample_op(ctx, input, kernel_size, kernel_size,
      iter_op="if (input[iid]>maxval) { maxval = input[iid]; maxidx = j * ksz.x + i; }",
      result_op="(float)maxidx", decls="float maxval=-FLT_MAX; int maxidx=0")
    ctx.save_for_backward(idxs, input.shape)
    return subsample_op(ctx, input, kernel_size, kernel_size,
      iter_op="maxval = max(maxval, input[iid])",
      result_op="maxval", decls="float maxval = -FLT_MAX")

  @staticmethod
  def backward(ctx, grad_output):
    idxs, orig_shape = ctx.saved_tensors
    return supersample_op(ctx, grad_output, orig_shape, ctx.kernel_size,
      result_op="(maxidx == kernidx) * input[iid]",
      decls="int maxidx=((__global float*)input2)[iid]; int kernidx=(gid.x%ksz.x) + ksz.x*(gid.y%ksz.y)",
      input2=idxs)
register('max_pool2d', MaxPool2D, gpu=True)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    # TODO: stability?
    lsum = reduce_op(ctx, "out += exp(a)", "log(out)", input, (input.shape[0],1))
    output = binary_op(ctx, 'a-b', input, lsum)
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors

    grad_input = buffer_like(ctx, grad_output)
    prg = clbuild(ctx.cl_ctx, """
    __kernel void lsmsub2(__global const float *grad_output, __global const float *output, int sz,
                          __global float *grad_input) {
      int gid = get_global_id(0);
      int gidsz = gid*sz;
      int gid2 = get_global_id(1);

      // TODO: this is repeated in many kernels
      float acc = 0.0;
      for (int x = 0; x < sz; x++) {
        acc += grad_output[gidsz + x];
      }

      grad_input[gidsz + gid2] = grad_output[gidsz + gid2] - exp(output[gidsz + gid2]) * acc;
    }""")
    prg.lsmsub2(ctx.cl_queue, [grad_output.shape[0], grad_output.shape[1]], None,
      grad_output, output, i32(grad_output.shape[1]), grad_input)

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

    ctx.save_for_backward(x,w)

    # output buffer
    ret = buffer_new(ctx, (bs, cout, oy, ox))

    prg = clbuild(ctx.cl_ctx, """
    __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs) {

      int B = get_global_id(0)/(groups*rcout);  // range 0-bs
      int g = (get_global_id(0)/rcout)%groups;
      int c = get_global_id(0) % rcout;

      int Y = get_global_id(1);  // range 0-oy
      int X = get_global_id(2);  // range 0-ox
      int IY = Y*ys;
      int IX = X*xs;

      // input  = (bs, groups, cin, iy, ix)
      // weight = (groups, rcout, cin, H, W)
      // output = (bs, groups, rcout, oy, ox)
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
    }""")

    prg.conv(ctx.cl_queue, [bs*groups*rcout, oy, ox], None,
      x, w, ret,
      i32(H), i32(W),
      i32(groups), i32(rcout), i32(cin),
      i32(oy), i32(ox),
      i32(iy), i32(ix),
      i32(ys), i32(xs)
    )
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    x, w = ctx.saved_tensors
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    dx = buffer_zeros(ctx, (bs, cin_, iy, ix))
    dw = buffer_new(ctx, (cout, cin, H, W))

    prg = clbuild(ctx.cl_ctx, """
    __kernel void convw(__global const float *tensx, __global const float *ggg, __global float *dw,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

      int g = get_global_id(0)/(rcout*cin) ; // range 0-groups
      int c = (get_global_id(0)/(cin)) %rcout; // range 0-rcout
      int ci = get_global_id(0) % cin;        // range 0-cin
      int y = get_global_id(1);  // range 0-H
      int x = get_global_id(2);  // range 0-W

      // tensx  = (bs, groups*cin, iy, ix)
      // tensw = (groups*rcout, cin, H, W)
      // ggg = (bs, groups*rout, oy, ox)
      float acc = 0.0;
      for (int Y = 0; Y < oy; Y++) {
        for (int X = 0; X < ox; X++) {
          for (int B = 0; B < bs; B++) {
            acc += ggg[B*groups*rcout*oy*ox + +g*rcout*oy*ox + c*oy*ox + Y*ox + X]*tensx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x];
          }
        }
      }
      dw[get_global_id(0)*H*W + y*W + x] = acc;
    }
    __kernel void convx(__global const float *tensw, __global const float *ggg, __global float *dx,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

      int B = get_global_id(0);
      int g = get_global_id(1);
      int ci = get_global_id(2);

      for (int c = 0; c < rcout; c++) {
        for (int Y = 0; Y < oy; Y++) {
          for (int X = 0; X < ox; X++) {
            for (int y = 0; y < H; y++) {
              for (int x = 0; x < W; x++) {
                dx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x]+= ggg[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X]*tensw[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
              }
            }
          }
        }
      }
    }
    """)

    conv_args = i32(H), i32(W), i32(ctx.groups), i32(rcout), i32(cin), i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs), i32(bs)
    prg.convw(ctx.cl_queue, [ctx.groups*rcout*cin, H, W], None, x, grad_output, dw, *conv_args)
    prg.convx(ctx.cl_queue, [bs, ctx.groups, cin], None, w, grad_output, dx,*conv_args)
    return dx, dw
register('conv2d', Conv2D, gpu=True)
