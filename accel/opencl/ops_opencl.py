# this is focused on speed
# it may not run everything

import pathlib
import numpy as np
from tinygrad.ops import MovementOps, ProcessingOps
from tinygrad.llops.ops_gpu import require_init_gpu, clbuild, sync, get_cl_queue, get_cl_ctx
from tinygrad.llops.ops_gpu import unary_op, binary_op, reduce_op, movement_op
from tinygrad.helpers import prod
import pyopencl as cl

def roundup(x, n=4): return (x+(n-1))//n * n
def flip(x): return (x[1], x[0])
class OpenCLBuffer:
  def __init__(self, shape, hostbuf=None):
    require_init_gpu()
    self.shape = tuple(shape)
    self._buf = None
    self._image = None
    self.dtype = np.float32
    if hostbuf is not None:
      # TODO: lazy?
      self._buf = cl.Buffer(get_cl_ctx(), cl.mem_flags.READ_WRITE, 4*roundup(prod(shape)))
      cl.enqueue_copy(get_cl_queue(), self._buf, hostbuf.astype(np.float32).ravel())

  @staticmethod
  def fromCPU(x):
    return OpenCLBuffer(x.shape, x)

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    sync()
    cl.enqueue_copy(get_cl_queue(), data, self.cl, is_blocking=True)
    return data

  @property
  def cl(self):
    if self._buf is None:
      self._buf = cl.Buffer(get_cl_ctx(), cl.mem_flags.READ_WRITE, 4*roundup(prod(self.shape)))
      if self._image is not None:
        print(f"converting {self.shape} back to buffer, image shape is {self._image.shape}")
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
        """)(self._image.shape, None, self._image, self._buf)
        self._image = None
    return self._buf

  @property
  def image(self):
    if self._image is None:
      assert self.shape[2] == 4 and len(self.shape) == 3
      fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
      self._image = cl.Image(get_cl_ctx(), cl.mem_flags.READ_WRITE, fmt, shape=flip(self.shape))
      if self._buf is not None:
        print(f"converting {self.shape} to image with shape {self._image.shape}")
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
        """)(self._image.shape, None, self._buf, self._image)
      self._buf = None
    return self._image

def load(x):
  with open(x) as f:
    ret = f.read()
  return ret

def conv(x,w,ret,C):
  print(x.shape, w.shape, ret.shape)
  options = []
  if C.cin == 1: options.append("-DDEPTHWISE")
  if C.bs > 1:
    options.append("-DBATCH")
    assert C.py == 0, "batched conv doesn't work with y-padding"
  conv_prg = clbuild("conv", load(pathlib.Path(__file__).parent.parent.parent / 'accel/opencl/conv.cl'), tuple(options))
  assert C.cout%4 == 0
  kernel_args = [C.cout//4, (C.ox+3)//4, C.bs*C.oy]
  conv_args = [max(1, C.cin//4), C.groups*C.cin//4, C.cout//4, C.ox, C.oy, C.iy, C.W, C.H, C.px, C.py, C.xs, C.ys, C.dx, C.dy]
  print(conv_args, kernel_args)
  conv_prg(kernel_args, None, x.image, w.image, ret.image, *[np.int16(x) for x in conv_args])

# input format is    N, H x W, C//4 x 4
# dweight format is  oc//4 x ch, cw x 4(oc)
# weight format is   oc//4 x ch, ic//4, cw, 4(oc) x 4(ic)
def processing_op(ctx,op,x,w,out_shape,C):
  assert op == ProcessingOps.CONV, f"{op} isn't supported"

  print(x.shape, w.shape)

  if C.bs > 1 and C.py > 0:
    # explictly add y-padding for batched inputs
    # N C H W
    x = ctx.movement_op(MovementOps.SLICE, x, ((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py), (0, x.shape[3])))
    C = C._replace(iy=C.iy + C.py*2, py=0)

  # hack for multiples of 4
  if C.cin % 4 != 0 and not (C.cin == 1 and C.groups%4 == 0):
    to_add = 4 - (C.cin % 4)
    ws = [(0, s) for s in w.shape]
    ws[1] = (0, w.shape[1]+to_add)
    w = ctx.movement_op(MovementOps.SLICE, w, ws)

    x = ctx.movement_op(MovementOps.RESHAPE, x, (C.bs, C.groups, C.cin, C.iy, C.ix))
    xs = [(0, s) for s in x.shape]
    xs[2] = (0, x.shape[2]+to_add)
    x = ctx.movement_op(MovementOps.SLICE, x, xs)
    C = C._replace(cin = C.cin + to_add)
    x = ctx.movement_op(MovementOps.RESHAPE, x, (C.bs, C.groups*C.cin, C.iy, C.ix))

  # hack for multiples of 4
  added_output_shape = None
  if C.groups == 1 and C.cout % 4 != 0:
    to_add = 4 - (C.cout % 4)
    added_output_shape = to_add
    ws = [(0, s) for s in w.shape]
    ws[0] = (0, w.shape[0]+to_add)
    w = ctx.movement_op(MovementOps.SLICE, w, ws)
    C = C._replace(cout = C.cout + to_add, rcout = C.rcout + to_add)

  # packed
  assert (C.groups*C.cin) % 4 == 0
  x = ctx.movement_op(MovementOps.PERMUTE, x, (0,2,3,1))
  x = ctx.movement_op(MovementOps.RESHAPE, x, (C.bs*C.iy, C.ix*C.groups*C.cin//4, 4))

  assert C.cout % 4 == 0
  if C.cin == 1:
    # depthwise
    w = ctx.movement_op(MovementOps.RESHAPE, w, (C.cout//4,4,C.H*C.W))
    w = ctx.movement_op(MovementOps.PERMUTE, w, (0,2,1))
  else:
    w = ctx.movement_op(MovementOps.RESHAPE, w, (C.cout//4,4,C.cin//4,4,C.H,C.W))
    w = ctx.movement_op(MovementOps.PERMUTE, w, (0,4,2,5,1,3))
    w = ctx.movement_op(MovementOps.RESHAPE, w, (C.cout//4, C.H * C.cin//4 * C.W * 4, 4))


  ret = ctx.buffer((C.bs*C.oy, C.ox*C.cout//4, 4))
  conv(x, w, ret, C)

  ret = ctx.movement_op(MovementOps.RESHAPE, ret, (C.bs, C.oy, C.ox, C.cout))

  if added_output_shape is not None:
    xs = [(0, s) for s in ret.shape]
    xs[3] = (0, ret.shape[3]-added_output_shape)
    ret = ctx.movement_op(MovementOps.SLICE, ret, xs)

  ret = ctx.movement_op(MovementOps.PERMUTE, ret, (0,3,1,2))
  return ret

def test_image():
  hostbuf = np.random.randn(5,8,4).astype(np.float32)
  x = OpenCLBuffer((5,8,4), hostbuf)
  assert np.allclose(x.toCPU(), hostbuf)
  print(x.image)
  assert np.allclose(x.toCPU(), hostbuf)

if __name__ == "__main__":
  test_image()





