from __future__ import annotations
import os
import functools
import numpy as np
import pyopencl as cl
from typing import List, Tuple, Optional
from tinygrad.helpers import prod, ConvArgs
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.shapetracker import ShapeTracker, View, strides_for_shape
from tinygrad.ops import DEBUG

class CL:
  CACHE = None
  def __init__(self):
    if getattr(CL, "cl_queue", None) is not None: return
    devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0:  # settle for CPU
      devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
    CL.cl_ctx = cl.Context(devices=[devices[int(os.getenv("CL_DEVICE", "0"))]])
    CL.cl_queue = cl.CommandQueue(self.cl_ctx)  # this is an in-order command queue

  @staticmethod
  def enqueue_copy(a, b, is_blocking=False):
    if CL.CACHE is not None: assert False, "can't copy while caching"
    cl.enqueue_copy(CL().cl_queue, a, b, is_blocking=is_blocking)

  @staticmethod
  def malloc(sz): return cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, sz)

@functools.lru_cache(maxsize=None)
class CLProgram:
  def __init__(self, name, prg, options=tuple(), argdtypes=None):
    self.name, self.prg = name, prg
    self.built = cl.Program(CL().cl_ctx, self.prg).build(options=options)
    self.clprg = self.built.__getattr__(self.name)
    if argdtypes is not None: self.clprg.set_scalar_arg_dtypes(argdtypes)
  def __call__(self, *args):
    if DEBUG >= 2: print(f"**** {self.name} {args[0]} {args[1]} ****")
    if DEBUG >= 3: print(self.prg)
    if CL.CACHE is not None: CL.CACHE.append((self, args))
    else: self.clprg(CL().cl_queue, *args)

# **** end CL wrappers ****

code_for_op = {
  UnaryOps.NOOP: "(A)", UnaryOps.RELU: "max(A, (float)0.)", UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.NEG: "(-(A))", UnaryOps.SIGN: "sign(A)",
  BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)", BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
}

class GPUBuffer:
  def __init__(self, shape, hostbuf:Optional[GPUBuffer]=None):
    self.st = ShapeTracker(shape)
    self.shape = self.st.shape
    self._buf = hostbuf._buf if hostbuf is not None else None
  
  @property
  def cl(self):
    if self._buf is None: self._buf = CL.malloc(4*prod(self.shape))
    return self._buf

  def __repr__(self):
    return f"<GPUBuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x):
    ret = GPUBuffer(x.shape)
    # TODO: this is blocking even though we told it not to
    CL.enqueue_copy(ret.cl, x.view(np.ndarray).astype(np.float32).ravel(), is_blocking=False)
    return ret

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    CL.enqueue_copy(data, self.contiguous_op().cl, is_blocking=True)
    return data

  def contiguous_view(x, name:str) -> str:
    return f"inline float get_{name}(__global const float *x, int gid) {{ int valid = 1; int idx = gid; {x.st.expr().replace('//', '/')}; return valid ? x[idx] : 0.0;}}"

  def unary_op(x, op:UnaryOps): return type(x)(x.shape)._processing_op([("A", x)], code_for_op[op])
  def binary_op(x, op:BinaryOps, y:GPUBuffer): return type(x)(x.shape)._processing_op([("A", x), ("B", y)], code_for_op[op])
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)

  def movement_op(x, op:MovementOps, arg) -> GPUBuffer:
    ret = type(x)(x.st, x)
    ret.shape = ret.st.movement_op(op, arg).shape
    return ret

  def processing_op(x, op:ProcessingOps, w:GPUBuffer, C:ConvArgs):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    return type(x)(C.out_shape)._processing_op([("input", x.contiguous_op()), ("weight", w.contiguous_op())], "acc", C)

  def reduce_op(x, op:ReduceOps, new_shape:Tuple[int]):
    ret = type(x)(new_shape)
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
    CLProgram("reduce", x.contiguous_view('A')+"""
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

  def _processing_op(ret, bufs: List[Tuple[str, GPUBuffer]]=[], code:str="acc", C:Optional[ConvArgs]=None) -> GPUBuffer:
    options = []
    if C is not None:
      ints = ''.join(f"int {x} = {getattr(C, x)};" for x in ["H", "W", "sy", "sx", "dx", "dy", "px", "py", "groups", "rcout", "cin"])
      params = [(f"int {x}", getattr(C, x)) for x in ["oy", "ox", "iy", "ix"]]
      if C.px == 0 and C.py == 0 and C.px_ == 0 and C.py_ == 0: options.append("-DALLVALID")
      if C.oy == 1 and C.ox == 1: options.append("-DONEBYONE")
      global_size = [C.bs*C.cout, C.oy, C.ox]
      assert bufs[0][0] == "input" and bufs[1][0] == "weight"
      ewbufs = bufs[2:]   # input and weight are consumed by the convs
      kernel_name = "conv"
      conv_src = """
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

      int IY = Y*sy;
      int IX = X*sx;

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
      """
    else:
      ints, params = '', []
      global_size = [prod(ret.shape), 1, 1]
      ewbufs = bufs
      kernel_name = "elementwise"
      conv_src = ""

    elementwise_prefix = '\n'.join([buf.contiguous_view(name) for name, buf in ewbufs])+ \
      "\n\ninline float _ewop("+','.join(["int gid", "float acc"]+[f"__global const float *{name}_g" for name, _ in ewbufs])+") {\n"+ \
      '\n'.join([f"float {name} = get_{name}({name}_g, gid);" for name, _ in ewbufs])+ \
      f"\nreturn {code}; }}"

    conv_params = ["__global float* restrict output"] + \
                  [f"__global const float *{name}_g" for name, _ in bufs] + \
                  [x[0] for x in params]
    conv_prg = CLProgram(kernel_name, elementwise_prefix+f"\n\n__kernel void {kernel_name}("+','.join(conv_params)+""") {
      float acc = 0.0;
      int gid = get_global_id(0);
      """+ints+conv_src+"""output[gid] = _ewop("""+','.join(["gid", "acc"]+[f"{name}_g" for name, _ in ewbufs])+""");
    }""", options=tuple(options), argdtypes=tuple([None]*(1+len(bufs)) + [np.int32]*len(params)))
    conv_prg(global_size, None, ret.cl, *[buf.cl for _, buf in bufs], *[x[1] for x in params])
    return ret





