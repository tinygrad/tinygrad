from __future__ import annotations
import os, functools
import numpy as np
import pyopencl as cl  # type: ignore
from typing import List, Tuple, Optional, Any, Union, Dict
from tinygrad.helpers import prod, ConvArgs
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.shapetracker import ShapeTracker, View, strides_for_shape
from tinygrad.ops import DEBUG

class CL:
  CACHE, kernel_count = None, 0
  cl_ctx : Optional[cl.Context] = None
  cl_queue : Optional[cl.CommandQueue] = None
  def __init__(self):
    if CL.cl_queue is not None: return
    devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0:  # settle for CPU
      devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
    CL.cl_ctx = cl.Context(devices=[devices[int(os.getenv("CL_DEVICE", "0"))]])
    CL.cl_queue = cl.CommandQueue(self.cl_ctx)  # this is an in-order command queue

  @staticmethod
  def enqueue_copy(a, b, is_blocking=False):
    if CL.CACHE is not None: assert False, "can't copy while caching"
    if DEBUG >= 1: print(f"**CL**      copy in {b.shape}" if isinstance(b, np.ndarray) else f"**CL**      copy OUT {a.shape}")
    cl.enqueue_copy(CL().cl_queue, a, b, is_blocking=is_blocking)

  @staticmethod
  def malloc(sz): return cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, sz)

@functools.lru_cache(maxsize=None)
class CLProgram:
  def __init__(self, name:str, prg:str, options=tuple(), argdtypes=None):
    self.name, self.prg = name, prg
    self.built = cl.Program(CL().cl_ctx, self.prg).build(options=options)
    self.clprg = self.built.__getattr__(self.name)
    if argdtypes is not None: self.clprg.set_scalar_arg_dtypes(argdtypes)
  def __call__(self, *args):
    CL.kernel_count += 1
    if DEBUG >= 1: print(f"**CL** {CL.kernel_count:4d} {self.name:20s} args {len(args[2:]):5d}  size {prod(args[0]):7d}  kernels {args[0]} {args[1]}")
    if DEBUG >= 3: print(self.prg)
    if CL.CACHE is not None: CL.CACHE.append((self, args))
    else: self.clprg(CL().cl_queue, *args)

# **** end CL wrappers ****

class GPUBuffer:
  code_for_op = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)", UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.SIGN: "sign(A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)", BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
  }

  def __init__(self, shape, hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None):
    self.st = ShapeTracker(shape)
    self.shape = self.st.shape
    self._buf : cl.Buffer = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if self._backing is not None and self._backing.shape != (1,): self.cl
  
  @property
  def cl(self):
    if self._buf is None: self._buf = CL.malloc(4*prod(self._base_shape))
    if self._backing is not None:
      CL.enqueue_copy(self._buf, self._backing, is_blocking=False)
      self._backing = None
    return self._buf

  def __repr__(self): return f"<GPUBuffer with shape {self.shape!r}>"
  def shapeTrackerView(self, st:ShapeTracker): return GPUBuffer(st, hostbuf=self)

  @staticmethod
  def fromCPU(x): return GPUBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    CL.enqueue_copy(data, self.contiguous_op().cl, is_blocking=True)
    return data

  def contiguous_view(x, name:str) -> str:
    return f"inline float get_{name}(__global const float *x, int gid) {{ int valid = 1; int idx = gid; {x.st.expr().replace('//', '/')}; return valid ? x[idx] : 0.0;}}"

  def contiguous_view_constant_fold(x, name:str) -> Tuple[str, bool]:
    if x._base_shape == (1,) and x._backing is not None:
      return f"inline float get_{name}(int gid) {{ int valid = 1; int idx = gid; {x.st.expr().replace('//', '/')}; return valid ? {x._backing[0]} : 0.0;}}", False
    else:
      return x.contiguous_view(name), True

  def unary_op(x, op:UnaryOps): return type(x)(x.shape)._processing_op([("A", x)], GPUBuffer.code_for_op[op])
  def binary_op(x, op:BinaryOps, y:GPUBuffer): return type(x)(x.shape)._processing_op([("A", x), ("B", y)], GPUBuffer.code_for_op[op])
  def contiguous_op(x): return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)

  def movement_op(x, op:MovementOps, arg) -> GPUBuffer:
    ret = type(x)(x.st, x)
    ret.shape = ret.st.movement_op(op, arg).shape
    return ret

  def processing_op(x, op:ProcessingOps, w:GPUBuffer, C:ConvArgs):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    return type(x)(C.out_shape)._processing_op([("input", x.contiguous_op()), ("weight", w.contiguous_op())], "acc", C)

  def reduce_op(x, op:ReduceOps, new_shape:Tuple[int, ...]):
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
    loop_start : List[str] = []
    loop_end : List[str] = []
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
    if C is not None:
      ints = ''.join(f"int {x} = {getattr(C, x)};" for x in ["H", "W", "sy", "sx", "dx", "dy", "px", "py", "groups", "rcout", "cin"])
      params = [(f"int {x}", getattr(C, x)) for x in ["oy", "ox", "iy", "ix"]]
      global_size = [C.bs*C.cout, C.oy, C.ox]
      assert bufs[0][0] == "input" and bufs[1][0] == "weight"
      assert bufs[0][1].st.contiguous and bufs[1][1].st.contiguous
      ewbufs = bufs[2:]   # input and weight are consumed by the convs
      kernel_name = "conv"
      conv_src = """
      int B = gid/(groups*rcout);  // range 0-bs
      int g = (gid/rcout)%groups;
      int c = gid % rcout;

      int Y = get_global_id(1);  // range 0-oy
      int X = get_global_id(2);  // range 0-ox
      gid = gid*oy*ox + Y*ox + X;

      for (int ci = 0; ci < cin; ci++) {
        for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) {
          int idx_y = y*dy + Y*sy - py;
          int idx_x = x*dx + X*sx - px;
          int valid = (idx_y >= 0 && idx_y < iy && idx_x >= 0 && idx_x < ix);
          acc += valid * input_g[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + clamp(idx_y, 0, iy-1)*ix + clamp(idx_x, 0, ix-1)] * \
            weight_g[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
        } }
      }
      """
    else:
      ints, params = '', []
      global_size = [prod(ret.shape), 1, 1]
      ewbufs = bufs
      kernel_name = "elementwise"
      conv_src = ""

    views = {name:buf.contiguous_view_constant_fold(name) for name, buf in ewbufs}
    elementwise_prefix = '\n'.join([x[0] for x in views.values()])+ \
      "\n\ninline float _ewop("+','.join(["int gid", "float acc"]+[f"__global const float *{name}_g" for name, _ in ewbufs if views[name][1]])+") {\n"+ \
      '\n'.join([f"float {name} = get_{name}({name}_g, gid);" if views[name][1] else f"float {name} = get_{name}(gid);" for name, _ in ewbufs])+ \
      f"\nreturn {code}; }}"

    buf_types = [f"__global const float *{name}_g" for name, _ in bufs if name not in views or views[name][1]] 
    conv_params = ["__global float* restrict output"] + buf_types + [x[0] for x in params]
    conv_prg = CLProgram(kernel_name, elementwise_prefix+f"\n\n__kernel void {kernel_name}("+','.join(conv_params)+""") {
      float acc = 0.0;
      int gid = get_global_id(0);
      """+ints+conv_src+"""output[gid] = _ewop("""+','.join(["gid", "acc"]+[f"{name}_g" for name, _ in ewbufs if name not in views or views[name][1]])+""");
    }""", argdtypes=tuple(None if i < 1+len(buf_types) else np.int32 for i in range(1+len(buf_types)+len(params))))
    conv_prg(global_size, None, ret.cl, *[buf.cl for name, buf in bufs if name not in views or views[name][1]], *[x[1] for x in params])
    return ret
