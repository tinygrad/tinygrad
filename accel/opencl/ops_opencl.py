from __future__ import annotations
import pyopencl as cl
from tinygrad.llops.ops_gpu import GPUBuffer, get_cl_ctx, get_cl_queue, CLProgram
from tinygrad.ops import ProcessingOps
from tinygrad.helpers import prod, ConvArgs
from typing import List, Tuple, Optional
import numpy as np

import pathlib
def load(x):
   with open(x) as f:
     ret = f.read()
   return ret
CONV_SRC = load(pathlib.Path(__file__).parent.parent.parent / 'accel/opencl/conv.cl')

def get_replacements(prg_src, opencl_type):
  middle_code = []
  vv = "xyzw"
  for i in range(4):
    acc = f"outputValues[i].{vv[i%4]}"
    args = [x.split(" ")[-1].replace("*", "") for x in opencl_type]
    args = [f"(outputRow * get_image_width(output) + outputLocation.x)*4+{i}", acc]+args
    middle_code.append(f"{acc} = _ewop("+', '.join(args)+");\n")

  replacements = {}
  if len(opencl_type) != 0:
    replacements["//PREFIX"] = prg_src
    replacements["//ARGS"] = ","+','.join(opencl_type)
    replacements["//BINOP"] = ''.join(middle_code)
  return replacements

def roundup(x, n=4): return (x+(n-1))//n * n
class OpenCLBuffer(GPUBuffer):
  def __init__(self, shape, hostbuf:Optional[OpenCLBuffer]=None):
    super().__init__(shape, hostbuf)
    self._image = hostbuf._image if hostbuf is not None else None

  @staticmethod
  def fromCPU(x):
    ret = OpenCLBuffer(x.shape)
    # TODO: this is blocking even though we told it not to
    cl.enqueue_copy(get_cl_queue(), ret.cl, x.view(np.ndarray).astype(np.float32).ravel(), is_blocking=False)
    return ret

  @property
  def cl(self):
    if self._buf is None:
      self._buf = cl.Buffer(get_cl_ctx(), cl.mem_flags.READ_WRITE, 4*roundup(prod(self.shape)))
      if self._image is not None:
        assert prod(self.shape) == prod(self._image.shape)*4
        #print(f"converting {self.shape} back to buffer, image shape is {self._image.shape}")
        CLProgram("from_image", """
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
      self._image = cl.Image(get_cl_ctx(), cl.mem_flags.READ_WRITE, fmt, shape=(self.shape[1], self.shape[0]))
      if self._buf is not None:
        assert prod(self.shape) == prod(self._image.shape)*4
        #print(f"converting {self.shape} to image with shape {self._image.shape}")
        CLProgram("to_image", """
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

  def movement_op(x, op, arg):
    x.cl
    # TODO: call super after x.cl
    ret = type(x)(x.st, x)
    ret.shape = ret.st.movement_op(op, arg).shape
    return ret

  def processing_op(x, op, w, C:ConvArgs):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    return type(x)(C.out_shape)._processing_op_cl([("input", x), ("weight", w)], "acc", C)

  def _processing_op_cl(ret, bufs: List[Tuple[str, GPUBuffer]]=[], code:str="acc", C=None):
    assert bufs[0][0] == "input" and bufs[1][0] == "weight"
    x,w = bufs[0][1], bufs[1][1]
    ewbufs = bufs[2:]

    elementwise_prefix = '\n'.join([buf.contiguous_view(name) for name, buf in ewbufs])+ \
      "inline float _ewop("+','.join(["int gid", "float acc"]+[f"__global const float *{name}_g" for name, _ in ewbufs])+") {"+ \
      '\n'.join([f"float {name} = get_{name}({name}_g, gid);" for name, _ in ewbufs])+ \
      f"return {code}; }}"

    replacements = get_replacements(elementwise_prefix, [f"__global const float *{name}_g" for name, _ in ewbufs] )

    x, w = x.contiguous_op(), w.contiguous_op()
    options = []
    if C.cin == 1: options.append("-DDEPTHWISE")
    if C.bs > 1:
      options.append("-DBATCH")
      assert C.py == 0, "batched conv doesn't work with y-padding"
    assert C.cout%4 == 0
    conv_src = CONV_SRC
    for k,v in replacements.items():
      conv_src = conv_src.replace(k, v)
    #print(conv_src)
    conv_prg = CLProgram("image_conv", conv_src,
      options=tuple(options),
      argdtypes=tuple([None, None, None] + [np.int16]*15 + [None]*len(ewbufs))
    )
    conv_args = [max(1, C.cin//4), C.groups*C.cin//4, max(1, C.rcout//4), C.cout//4, C.ox, C.oy, C.iy, C.W, C.H, C.px, C.py, C.xs, C.ys, C.dx, C.dy]
    conv_prg([C.cout//4, (C.ox+3)//4, C.bs*C.oy], None, x.image, w.image, ret.image, *conv_args, *[buf.cl for _, buf in ewbufs])

    return ret