import ctypes
from typing import cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.runtime.support.c import init_c_var
from tinygrad.device import Device, MultiBuffer
from tinygrad.uop.ops import Ops, UPat, PatternMatcher
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args, cu_time_execution
from tinygrad.engine.realize import get_runner, unwrap_multi, resolve_params
from tinygrad.engine.jit import MultiGraphRunner

def encode_kernel(ctx, call, ast):
  gr, bufs, device_vars, replace, dev_idx = ctx
  prg = get_runner(bufs[0].device, ast)
  global_size, local_size = prg.p.launch_dims({v: 0 for v in gr.vars})
  deps, node = gr.new_node([b.base for b in bufs], prg.p.outs)
  c_args, vargs = encode_args([b._buf for b in bufs], [device_vars.get(x.expr, 0) for x in prg.p.vars])
  params = cuda.CUDA_KERNEL_NODE_PARAMS_v1(prg._prg.prg, *global_size, *local_size, 0,
                                           ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
  check(cuda.cuGraphAddKernelNode(ctypes.byref(node), gr.graph, deps, len(deps or ()), ctypes.byref(params)))
  gr.nodes.append((node, params, c_args, False, replace, dev_idx))

def encode_copy(ctx, call, ast):
  gr, bufs, _, replace, dev_idx = ctx
  dest, src = bufs[0], bufs[1]
  src_ctx = cast(CUDADevice, Device[src.device]).context
  deps, node = gr.new_node([dest.base, src.base], [0])
  params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                 dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                 WidthInBytes=dest.nbytes, Height=1, Depth=1)
  check(cuda.cuGraphAddMemcpyNode(ctypes.byref(node), gr.graph, deps, len(deps or ()), ctypes.byref(params), src_ctx))
  gr.nodes.append((node, params, src_ctx, True, [x for x in replace if x[0] < 2], dev_idx))

pm_encode = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat((Ops.SINK, Ops.PROGRAM), name="ast"),), name="call", allow_any_len=True), encode_kernel),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True), encode_copy),
])

class CUDAGraph(MultiGraphRunner):
  def __init__(self, linear, input_buffers, input_uops=()):
    super().__init__(linear, input_buffers, input_uops)
    self.graph = init_c_var(cuda.CUgraph, lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))
    self.nodes: list = []

    for call in self.linear.src:
      replace = [(p, b.arg) for p, b in enumerate(b for b in call.src[1:] if b.op is not Ops.BIND) if b.op is Ops.PARAM]
      for dev_idx, (bufs, device_vars) in enumerate(unwrap_multi(call, resolve_params(call, input_uops))):
        for b in bufs: b.ensure_allocated()
        pm_encode.rewrite(call, (self, bufs, device_vars, replace, dev_idx))

    self.instance = init_c_var(cuda.CUgraphExec, lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))
    self.updatable = sorted(set(j for j,n in enumerate(self.nodes) if n[4]) | self.var_vals_replace.keys() | self.launch_dims_replace.keys())

  def new_node(self, bufs, write):
    deps = self._access_resources(bufs, write, new_dependency=(node:=cuda.CUgraphNode()))
    return (cuda.CUgraphNode * len(deps))(*deps) if deps else None, node

  def __call__(self, input_buffers, var_vals, wait=False, input_uops=None):
    for j in self.updatable:
      _, params, c_args, is_copy, replace, dev_idx = self.nodes[j]
      for pos, iidx in replace:
        b = input_uops[iidx].buffer
        buf = b.bufs[dev_idx] if isinstance(b, MultiBuffer) else b
        if not is_copy: setattr(c_args, f'f{pos}', buf._buf)
        elif pos == 0: params.destDevice = buf._buf
        elif pos == 1: params.srcDevice = buf._buf
    for j, i, v in self.updated_vars(var_vals): setattr(self.nodes[j][2], f'v{i}', v)
    for j, gl, lc in self.updated_launch_dims(var_vals):
      p = self.nodes[j][1]
      p.blockDimX, p.blockDimY, p.blockDimZ, p.gridDimX, p.gridDimY, p.gridDimZ = *lc, *gl # type: ignore[misc]
    for j in self.updatable:
      node, params, extra, is_copy, _, _ = self.nodes[j]
      if is_copy: check(cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, ctypes.byref(params), extra))
      else: check(cuda.cuGraphExecKernelNodeSetParams(self.instance, node, ctypes.byref(params)))
    return cu_time_execution(lambda: check(cuda.cuGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))
