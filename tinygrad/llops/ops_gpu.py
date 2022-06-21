from __future__ import annotations
import functools
import numpy as np
import pyopencl as cl
from typing import List, Tuple, Optional
from tinygrad.helpers import prod
from tinygrad.llops.ops_cpu import unary_op
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.shapetracker import ShapeTracker, View, strides_for_shape

cl_ctx, cl_queue = None, None
def get_cl_ctx(): return cl_ctx
def get_cl_queue(): return cl_queue
def require_init_gpu():
  global cl_ctx, cl_queue
  if cl_ctx is None:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:  # settle for CPU
      devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
    cl_ctx = cl.Context(devices=devices)
    cl_queue = cl.CommandQueue(cl_ctx)  # this is an in-order command queue

def roundup(x, n=4): return (x+(n-1))//n * n
class GPUBuffer:
  def __init__(self, shape, hostbuf:Optional[GPUBuffer]=None):
    require_init_gpu()
    self.st = ShapeTracker(shape)
    self.shape = self.st.shape
    if hostbuf is None:
      self._buf, self._image = None, None
    else:
      self._buf, self._image = hostbuf._buf, hostbuf._image

  def __repr__(self):
    return f"<GPUBuffer with shape {self.shape!r}>"

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

  @staticmethod
  def fromCPU(x):
    ret = GPUBuffer(x.shape)
    # TODO: this is blocking even though we told it not to
    cl.enqueue_copy(cl_queue, ret.cl, x.view(np.ndarray).astype(np.float32).ravel(), is_blocking=False)
    return ret

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    cl.enqueue_copy(cl_queue, data, contiguous(self).cl, is_blocking=True)
    return data

@functools.lru_cache(maxsize=None)
class CLProgram:
  def __init__(self, name, prg, options=tuple(), argdtypes=None):
    self.name = name
    self.built = cl.Program(cl_ctx, prg).build(options=options)
    self.clprg = self.built.__getattr__(name)
    if argdtypes is not None: self.clprg.set_scalar_arg_dtypes(argdtypes)
  def __call__(self, *args):
    #print(f"running {self.name} with {args[0]} count {len(args)-2}")
    self.clprg(cl_queue, *args)

def contiguous_view(name:str, x:GPUBuffer) -> str:
  return f"inline float get_{name}(__global const float *x, int gid) {{ int valid = 1; int idx = gid; {x.st.expr().replace('//', '/')}; return valid ? x[idx] : 0.0;}}"

def _processing_op(out_shape: Tuple[int], bufs: List[Tuple[str, GPUBuffer]]=[], code:str="acc", C=None):
  ret = GPUBuffer(out_shape)
  options = []

  if C is not None:
    ints = ''.join(f"int {x} = {getattr(C, x)};" for x in ["H", "W", "ys", "xs", "dx", "dy", "px", "py", "groups", "rcout", "cin"])
    params = [(f"int {x}", getattr(C, x)) for x in ["oy", "ox", "iy", "ix"]]
    if C.px == 0 and C.py == 0: options.append("-DALLVALID")
    if C.oy == 1 and C.ox == 1: options.append("-DONEBYONE")
    global_size = [C.bs*C.cout, C.oy, C.ox]
    assert bufs[0][0] == "input" and bufs[1][0] == "weight"
    ewbufs = bufs[2:]   # input and weight are consumed by the convs
    kernel_name = "conv"
  else:
    ints, params = '', []
    options.append("-DNOCONV")
    global_size = [prod(ret.shape), 1, 1]
    ewbufs = bufs
    kernel_name = "elementwise"

  elementwise_prefix = '\n'.join([contiguous_view(name, buf) for name, buf in ewbufs])+ \
    "inline float _ewop("+','.join(["int gid", "float acc"]+[f"__global const float *{name}_g" for name, _ in ewbufs])+") {"+ \
    '\n'.join([f"float {name} = get_{name}({name}_g, gid);" for name, _ in ewbufs])+ \
    f"return {code}; }}"

  conv_params = ["__global float* restrict output"] + \
                [f"__global const float *{name}_g" for name, _ in bufs] + \
                [x[0] for x in params]
  conv_prg = CLProgram(kernel_name, elementwise_prefix+f"__kernel void {kernel_name}("+','.join(conv_params)+""") {
    float acc = 0.0;
    int gid = get_global_id(0);
    """+ints+"""

#ifndef NOCONV
    int B = gid/(groups*rcout);  // range 0-bs
    int g = (gid/rcout)%groups;
    int c = gid % rcout;

#ifdef ONEBYONE
    int Y = 0;
    int X = 0;
#else
    int Y = get_global_id(1);  // range 0-oy
    int X = get_global_id(2);  // range 0-ox
    gid = gid*oy*ox + Y*ox + X;
#endif

    int IY = Y*ys;
    int IX = X*xs;

    for (int ci = 0; ci < cin; ci++) {
      for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) {
        int idx_y = y*dy + IY - py;
        int idx_x = x*dx + IX - px;
#ifdef ALLVALID
        acc += input_g[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + idx_y*ix + idx_x] * \
          weight_g[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
#else
        int valid = (idx_y >= 0 && idx_y < iy && idx_x >= 0 && idx_x < ix);
        acc += valid * input_g[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + clamp(idx_y, 0, iy-1)*ix + clamp(idx_x, 0, ix-1)] * \
          weight_g[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
#endif
      } }
    }
#endif

    output[gid] = _ewop("""+','.join(["gid", "acc"]+[f"{name}_g" for name, _ in ewbufs])+""");
  }""", options=tuple(options), argdtypes=tuple([None]*(1+len(bufs)) + [np.int32]*len(params)))
  conv_prg(global_size, None, ret.cl, *[buf.cl for _, buf in bufs], *[x[1] for x in params])
  return ret


# gpu ops

code_for_op = {
  UnaryOps.NOOP: "(A)", UnaryOps.RELU: "max(A, (float)0.)", UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.NEG: "(-(A))", UnaryOps.SIGN: "sign(A)",
  BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)", BinaryOps.DIV: "(B/A)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
}

def unary_op(op, x): return _processing_op(x.shape, [("A", x)], code_for_op[op])
def binary_op(op, x, y): return _processing_op(x.shape, [("A", x), ("B", y)], code_for_op[op])
def contiguous(x:GPUBuffer): return x if x.st.contiguous else unary_op(UnaryOps.NOOP, x)

def movement_op(op, x, arg):
  x.cl
  ret = GPUBuffer(x.st, x)
  ret.shape = ret.st.movement_op(op, arg).shape
  return ret

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

def _processing_op_cl(out_shape: Tuple[int], bufs: List[Tuple[str, GPUBuffer]]=[], code:str="acc", C=None):
  assert bufs[0][0] == "input" and bufs[1][0] == "weight"
  x,w = bufs[0][1], bufs[1][1]
  ewbufs = bufs[2:]

  elementwise_prefix = '\n'.join([contiguous_view(name, buf) for name, buf in ewbufs])+ \
    "inline float _ewop("+','.join(["int gid", "float acc"]+[f"__global const float *{name}_g" for name, _ in ewbufs])+") {"+ \
    '\n'.join([f"float {name} = get_{name}({name}_g, gid);" for name, _ in ewbufs])+ \
    f"return {code}; }}"

  replacements = get_replacements(elementwise_prefix, [f"__global const float *{name}_g" for name, _ in ewbufs] )

  x, w = contiguous(x), contiguous(w)
  ret = GPUBuffer(out_shape)
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

def processing_op(op, x, w, C):
  assert op == ProcessingOps.CONV, f"{op} isn't supported"
  return _processing_op_cl(C.out_shape, [("input", contiguous(x)), ("weight", contiguous(w))], "acc", C)

def reduce_op(op, x, new_shape):
  ret = GPUBuffer(new_shape)
  if op == ReduceOps.SUM: code, start = "out += a", "0.0"
  elif op == ReduceOps.MAX: code, start = "out = max(a,out)", "-INFINITY"
  else: raise Exception(f"{op} isn't supported")

  # reverse operation of expand, this validates inputs
  st = ShapeTracker(ret.shape).movement_op(MovementOps.EXPAND, x.shape)
  # this takes a ret index to an inp index, indexing 0 on the reduced strides
  view = View(ret.shape, strides_for_shape(x.shape))

  # generate loops with combined adjacent reduce axis
  acc = 1
  loop_start, loop_end = [], []
  for shp,stride in st.views[-1].shape_strides[::-1]:
    if stride == 0:
      loop_start.append(f"for (int axis_{len(loop_start)} = 0; axis_{len(loop_start)} < {shp}; axis_{len(loop_start)}++) {{")
      loop_end.append(f"idx += {acc}; }} idx -= {shp*acc};")
    acc *= shp

  # TODO: support multistage reduces
  CLProgram("reduce", contiguous_view('A', x)+"""
  __kernel void reduce(__global const float *a_g, __global float *res_g) {
    int gid = get_global_id(0); int idx = gid;"""+view.expr.replace('//', '/')+""";
    float out = """+start+""";\n"""+ \
      '\n'.join(loop_start[::-1])+"""
        float a = get_A(a_g, idx);
        """+code+""";\n"""+ \
      '\n'.join(loop_end)+"""
    res_g[gid] = out;
  }""")([prod(ret.shape)], None, x.cl, ret.cl)
  return ret
