# this is focused on speed
# it may not run everything

import pathlib
import numpy as np
from tinygrad.ops import ProcessingOps
from tinygrad.llops.ops_gpu import require_init_gpu, clbuild, sync, get_cl_queue, get_cl_ctx
from tinygrad.llops.ops_gpu import unary_op, binary_op, reduce_op, movement_op
from tinygrad.helpers import prod
import pyopencl as cl

def flip(x): return (x[1], x[0])
class OpenCLBuffer:
  def __init__(self, shape, hostbuf=None):
    require_init_gpu()
    self.shape = shape
    self.strides = [1]
    for d in self.shape[::-1][:-1]:
      self.strides = [d*self.strides[0]] + self.strides
    self.nhwc = False
    self.hostbuf = hostbuf
    self.cl = cl.Buffer(get_cl_ctx(), cl.mem_flags.READ_WRITE, 4*prod(shape)) 
    self.dtype = np.float32
    if hostbuf is not None:
      cl.enqueue_copy(get_cl_queue(), self.cl, hostbuf.astype(np.float32).ravel())

  @staticmethod
  def fromCPU(x):
    return OpenCLBuffer(x.shape, x)

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    sync()
    cl.enqueue_copy(get_cl_queue(), data, self.cl, is_blocking=True)
    return data
  
  def fromImage(self, img):
    assert flip(img.shape) == self.shape[0:2] and self.shape[2] == 4
    clbuild("from_image", """
      __kernel void from_image(
          read_only image2d_t in,
          __global float4 *out) {
        const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        int2 l;
        l.y = get_global_id(1);
        l.x = get_global_id(0);
        int W = get_image_width(in);
        out[l.y*W + l.x] = read_imagef(in, smp, l);
      }
    """)(img.shape, None, img, self.cl)

  def toImage(self, img):
    assert flip(img.shape) == self.shape[0:2] and self.shape[2] == 4
    clbuild("to_image", """
      __kernel void to_image(
          __global const float4 *in,
          write_only image2d_t out) {
        int2 l;
        l.y = get_global_id(1);
        l.x = get_global_id(0);
        int W = get_image_width(out);
        write_imagef(out, l, in[l.y*W + l.x]);
      }
    """)(img.shape, None, self.cl, img)

def load(x):
  with open(x) as f:
    ret = f.read()
  return ret

# input format is    N, H x W, C//4 x 4
# dweight format is  oc//4 x ch, cw x 4(oc)
# weight format is   oc//4 x ch, ic//4, cw, 4(oc) x 4(ic)
def conv(x,w,ret,C):
  print(x.shape, w.shape, ret.shape)
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

  xi = cl.Image(get_cl_ctx(), cl.mem_flags.READ_WRITE, fmt, shape=flip(x.shape))
  wi = cl.Image(get_cl_ctx(), cl.mem_flags.READ_WRITE, fmt, shape=flip(w.shape))
  reti = cl.Image(get_cl_ctx(), cl.mem_flags.READ_WRITE, fmt, shape=flip(ret.shape))
  x.toImage(xi)
  w.toImage(wi)
  options = ("-DDEPTHWISE",) if C.groups > 1 else tuple()
  conv_prg = clbuild("conv", load(pathlib.Path(__file__).parent.parent.parent / 'accel/opencl/conv.cl'), options)
  assert C.cin%4 == 0
  assert C.cout%4 == 0
  kernel_args = [C.cout//4, (C.ox+3)//4, C.oy]
  conv_args = [max(1, C.cin//4), C.groups*C.cin//4, C.cout//4, C.ox, C.W, C.H, 0, 0, C.xs, C.ys]
  print(conv_args, kernel_args)
  conv_prg(kernel_args, None, xi, wi, reti, *[np.int16(x) for x in conv_args])
  ret.fromImage(reti)
  #print(ret.toCPU())

def processing_op(op,a,b,ret,C):
  if op == ProcessingOps.CONV: conv(a,b,ret,C)
  else: raise Exception(f"{op} not implemented")

def test_image():
  x = OpenCLBuffer((5,8,4), np.random.randn(5,8,4).astype(np.float32))
  y = OpenCLBuffer((5,8,4), np.random.randn(5,8,4).astype(np.float32))
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
  xi = cl.Image(get_cl_ctx(), cl.mem_flags.READ_WRITE, fmt, shape=(x.shape[1], x.shape[0]))
  print(xi.shape)
  x.toImage(xi)
  y.fromImage(xi)
  assert np.allclose(x.toCPU(), y.toCPU())
  print(x.toCPU())
  print(y.toCPU())

if __name__ == "__main__":
  test_image()





