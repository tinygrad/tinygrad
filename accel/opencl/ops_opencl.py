# type: ignore

from __future__ import annotations
import os
from tinygrad.llops.ops_gpu import GPUBuffer, CL, CLProgram, CLBuffer
from tinygrad.ops import ProcessingOps
from tinygrad.helpers import prod, ConvArgs
from typing import List, Tuple, Optional, Dict
import numpy as np
import pyopencl as cl

import pathlib
def load(x):
   with open(x) as f:
     ret = f.read()
   return ret
CONV_SRC = load(pathlib.Path(__file__).parent.parent.parent / 'accel/opencl/conv.cl')
MATMUL_SRC = load(pathlib.Path(__file__).parent.parent.parent / 'accel/opencl/matmul.cl')

class ECL(CL):
  @staticmethod
  def image(shape):
    if int(os.getenv("FLOAT16", 0)):
      # HALF_FLOAT breaks tests
      fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT)
    else:
      fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    return cl.Image(CL().cl_ctx, cl.mem_flags.READ_WRITE, fmt, shape=shape)

def get_replacements(prg_src:str, opencl_type:List[str]) -> Dict[str, str]:
  middle_code = []

  """
  vv = "xyzw"
  for i in range(4):
    acc = f"outputValues[i].{vv[i%4]}"
    args = [x.split(" ")[-1].replace("*", "") for x in opencl_type]
    args = [f"(outputRow * get_image_width(output) + outputLocation.x)*4+{i}", acc]+args
    middle_code.append(f"{acc} = _ewop("+', '.join(args)+");\n")
  """
  acc = f"outputValues[i]"
  args = [x.split(" ")[-1].replace("*", "") for x in opencl_type]
  args = ["smp", "outputLocation", "(outputLocation.y * get_image_width(output) + outputLocation.x)*4", acc]+args
  middle_code.append(f"{acc} = _ewop("+', '.join(args)+");\n")

  replacements = {}
  if len(opencl_type) != 0:
    replacements["//PREFIX"] = prg_src
    replacements["//ARGS"] = ","+','.join(opencl_type)
    replacements["//BINOP"] = ''.join(middle_code)
  return replacements

def roundup(x, n=4): return (x+(n-1))//n * n
class OpenCLBuffer(GPUBuffer):
  def __init__(self, shape, hostbuf:Optional[OpenCLBuffer]=None, backing:Optional[np.ndarray]=None):
    self._image = hostbuf._image if hostbuf is not None else None
    super().__init__(shape, hostbuf, backing)

  @staticmethod
  def fromCPU(x): return OpenCLBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())
  
  def __repr__(self): return f"<OpenCLBuffer with shape {self.shape!r}>"

  @property
  def cl(self):
    if self._buf is None:
      if self.st.contiguous:
        self._buf = CLBuffer(4*roundup(prod(self.shape)))
        if self._backing is not None:
          CL.enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
          self._backing = None
      if self._image is not None:
        self._buf = CLBuffer(4*roundup(prod(self._image.shape)*4))
        if self._backing is not None:
          CL.enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
          self._backing = None
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
        """)(self._image.shape, None, self._image, self._buf.cl)
        self._image = None
    return self._buf.cl
  
  def is_image(self): return self._image is not None

  @property
  def image(self):
    if self._image is None:
      assert self.shape[2] == 4 and len(self.shape) == 3
      self._image = ECL.image(shape=(self.shape[1], self.shape[0]))
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
        """)(self._image.shape, None, self._buf.cl, self._image)
      self._buf = None
    return self._image

  seen = set()
  def _processing_op(ret, bufs: List[Tuple[str, OpenCLBuffer]]=[], code:str="acc", C=None):
    if C is None:
      # TODO: handle an opencl conv without the conv part
      return super()._processing_op(bufs, code, C)

    x = [x for x in bufs if x[0] == "input"][0][1]
    w = [x for x in bufs if x[0] == "weight"][0][1]
    ewbufs = [x for x in bufs if x[0] not in ["input", "weight"]]

    if (x,w) in OpenCLBuffer.seen:
      print("WARNING: recomputing CONV with", x, w)
    OpenCLBuffer.seen.add((x,w))

    fakebufs = []
    ewtypes = []
    getters = []
    for name, buf in ewbufs:
      if buf.is_image() and buf.shape == ret.shape and buf.st.contiguous:
        # use an image here
        ewtypes.append(f"read_only image2d_t {name}_g")
        getters.append(f"inline float4 get4_{name}(read_only image2d_t x, const sampler_t smp, int2 loc, int gid) {{ return read_imagef(x, smp, loc); }}")
      elif buf.st.contiguous:
        # use float4
        ewtypes.append(f"__global const float4 *{name}_g")
        getters.append(f"inline float4 get4_{name}(__global const float4 *x, const sampler_t smp, int2 loc, int gid) {{"+
          f"return x[gid/4]; }}")
      elif int(os.getenv("UNSAFE_FLOAT4", 0)):
        # use float4 indexed (HACK!)
        # TODO: work out when this is okay
        ewtypes.append(f"__global const float4 *{name}_g")
        getters.append(f"inline float4 get4_{name}(__global const float4 *x, const sampler_t smp, int2 loc, int gid) {{"+
          "int valid = 1; int idx = gid;"+buf.st.expr()+";"+
          f"return x[idx/4]; }}")
      else:
        # fallback to float
        view, unfolded = buf.contiguous_view_constant_fold(name)
        getters.append(view)
        if unfolded:
          ewtypes.append(f"__global const float *{name}_g")
          getters.append(f"inline float4 get4_{name}(__global const float *x, const sampler_t smp, int2 loc, int gid) {{"+
            f"return (float4)(get_{name}(x,gid+0), get_{name}(x,gid+1), get_{name}(x,gid+2), get_{name}(x,gid+3)); }}")
        else:
          fakebufs.append(name)
          getters.append(f"inline float4 get4_{name}(int gid) {{"+
            f"return (float4)(get_{name}(gid+0), get_{name}(gid+1), get_{name}(gid+2), get_{name}(gid+3)); }}")

    # remove fakebufs
    ewbufs = [x for x in ewbufs if x[0] not in fakebufs]

    elementwise_prefix = '\n'.join(getters)+ \
      "\n\ninline float4 _ewop("+','.join(["const sampler_t smp", "int2 loc", "int gid", "float4 acc"]+ewtypes)+") {\n"+ \
      ''.join([f"float4 {name} = get4_{name}(gid);\n" for name in fakebufs])+ \
      ''.join([f"float4 {name} = get4_{name}({name}_g, smp, loc, gid);\n" for name, _ in ewbufs])+ \
      f"return {code}; }}"

    replacements = get_replacements(elementwise_prefix, ewtypes)

    x, w = x.contiguous_op(), w.contiguous_op()
    options = []
    if C.bs > 1:
      options.append("-DBATCH")
      assert C.py == 0, "batched conv doesn't work with y-padding"
    if C.sx == 1 and C.sy == 1 and C.dx == 1 and C.dy == 1 and C.cin == 1: options.append("-DDEPTHWISE_UNSTRIDED")
    elif C.cin == 1: options.append("-DDEPTHWISE")
    if int(os.getenv("MATMUL", 0)) and C.groups == 1 and C.H == 1 and C.W == 1 and C.iy == 1 and C.ix == 1 and C.oy == 1 and C.ox == 1 and C.sx == 1 and C.sy == 1 and C.dx == 1 and C.dy == 1:
      options.append("-DMATMUL")
      # NOTE: this is not actually a matmul, it's a vector * matrix

      conv_args = []
      conv_short_names = ["numPackedInputChannelsForGroup", "totalNumPackedInputChannels", "numPackedOutputChannelsForGroup", "totalNumPackedOutputChannels", "numOutputColumns", "numOutputRows", "numInputRows"]
      conv_shorts = [max(1, C.cin//4), C.groups*C.cin//4, max(1, C.rcout//4), C.cout//4, C.ox, C.oy, C.iy]

      conv_src = MATMUL_SRC
      replacements["//SHORTS"] = ''.join([f"short {name} = {val};" for name,val in zip(conv_short_names, conv_shorts)])
      if "//BINOP" in replacements:
        replacements["//BINOP"] = replacements["//BINOP"].replace("outputValues[i]", "outputValues")
      for k,v in replacements.items():
        conv_src = conv_src.replace(k, v)

      #print(conv_src)
      conv_prg = CLProgram("matmul", conv_src,
        options=tuple(options),
        argdtypes=tuple([None, None, None, None] + [np.int16]*len(conv_args) + [None]*len(ewbufs))
      )
      global_work_size = [4, 16, C.cout//4]

      # must be even
      lw = CL.cl_ctx.devices[0].max_work_group_size // (global_work_size[0] * global_work_size[1])
      while global_work_size[2] % lw != 0:
        lw -= 1
      local_work_size = [4, global_work_size[1], lw]

      #print(global_work_size, local_work_size)
      conv_prg(global_work_size, local_work_size, cl.LocalMemory(4 * local_work_size[0] * local_work_size[1] * lw), x.image, w.image, ret.image, *conv_args, *[buf.image if 'image2d_t' in typ else buf.cl for typ, (_, buf) in zip(ewtypes, ewbufs)])
      return ret

    # this option is unused
    if C.H == 1 and C.W == 1: options.append("-DONLY_1X1_CONV")

    assert C.cout%4 == 0
    conv_src = CONV_SRC
    conv_short_names = ["filterSizeX", "filterSizeY", "paddingX", "paddingY", "strideX", "strideY", "dilationX", "dilationY"]
    conv_shorts = [C.W, C.H, C.px, C.py, C.sx, C.sy, C.dx, C.dy]
    conv_arg_names = ["numPackedInputChannelsForGroup", "totalNumPackedInputChannels", "numPackedOutputChannelsForGroup", "totalNumPackedOutputChannels", "numOutputColumns", "numOutputRows", "numInputRows"]
    conv_args = [max(1, C.cin//4), C.groups*C.cin//4, max(1, C.rcout//4), C.cout//4, C.ox, C.oy, C.iy]

    NUM_OUTPUTS = 4
    options.append(f"-DNUM_OUTPUTS={NUM_OUTPUTS}")

    # comment out for args
    conv_short_names += conv_arg_names
    conv_shorts += conv_args
    conv_args = []
    options.append("-DNOARGS")

    replacements["//SHORTS"] = ''.join([f"short {name} = {val};" for name,val in zip(conv_short_names, conv_shorts)])
    for k,v in replacements.items():
      conv_src = conv_src.replace(k, v)
    #print(conv_src)
    conv_prg = CLProgram("image_conv", conv_src,
      options=tuple(options),
      argdtypes=tuple([None, None, None] + [np.int16]*len(conv_args) + [None]*len(ewbufs))
    )
    global_work_size = [C.cout//4, (C.ox+NUM_OUTPUTS-1)//NUM_OUTPUTS, C.bs*C.oy]
    conv_prg(global_work_size, None, x.image, w.image, ret.image, *conv_args, *[buf.image if 'image2d_t' in typ else buf.cl for typ, (_, buf) in zip(ewtypes, ewbufs)])
    return ret

GPUBuffer = OpenCLBuffer
