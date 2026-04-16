import ctypes
from typing import Any, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import dedup
from tinygrad.runtime.support.c import init_c_var
from tinygrad.device import Buffer, Device
from tinygrad.uop.ops import Ops
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args, cu_time_execution
from tinygrad.engine.realize import get_runner
from tinygrad.engine.jit import MultiGraphRunner, _unwrap_beam

class CUDAGraph(MultiGraphRunner):
  def __init__(self, linear, input_buffers:list[Buffer]):
    super().__init__(linear, input_buffers)
    self.jc_idx_with_updatable_bufs, self.updatable_nodes = dedup([x[0] for x in self.input_replace.keys()]), {}
    self.graph = init_c_var(cuda.CUgraph, lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))

    for j, call in enumerate(linear.src[0].src):
      bufs = [input_buffers[b.arg] if b.op is Ops.PARAM else b.buffer.ensure_allocated() for b in call.src[1:] if b.op is not Ops.BIND]
      dev_num, uast = int(bufs[0].device.split(":")[-1]) if ":" in bufs[0].device else 0, _unwrap_beam(call.src[0])

      if uast.op in (Ops.SINK, Ops.PROGRAM):
        prg, new_node = get_runner(bufs[0].device, call.src[0]), cuda.CUgraphNode()
        deps = self._access_resources([x.base for x in bufs], prg.p.outs, new_dependency=new_node)
        c_deps = (cuda.CUgraphNode*len(deps))(*deps) if deps else None
        c_args, vargs = encode_args([b._buf for b in bufs], [dev_num if x.expr == '_device_num' else 0 for x in prg.p.vars])
        global_size, local_size = prg.p.launch_dims({v: 0 for v in self.vars})
        kern_params = cuda.CUDA_KERNEL_NODE_PARAMS_v1(prg._prg.prg, *global_size, *local_size, 0,
                                                      ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
        check(cuda.cuGraphAddKernelNode(ctypes.byref(new_node), self.graph, c_deps, len(deps), ctypes.byref(kern_params)))
        if j in self.launch_dims_replace or j in self.var_vals_replace or j in self.jc_idx_with_updatable_bufs:
          self.updatable_nodes[j] = (new_node, kern_params, c_args, False)
      elif uast.op is Ops.COPY:
        dest, src, node = bufs[0], bufs[1], cuda.CUgraphNode()
        deps = self._access_resources([dest.base, src.base], [0], new_dependency=node)
        c_deps = (cuda.CUgraphNode*len(deps))(*deps) if deps else None
        cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                          dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                          WidthInBytes=dest.nbytes, Height=1, Depth=1)
        check(cuda.cuGraphAddMemcpyNode(ctypes.byref(node), self.graph, c_deps, len(deps), ctypes.byref(cp_params),
                                        cast(CUDADevice, Device[src.device]).context))
        if j in self.jc_idx_with_updatable_bufs: self.updatable_nodes[j] = (node, cp_params, cast(CUDADevice, Device[src.device]).context, True)

    self.instance = init_c_var(cuda.CUgraphExec, lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))

  def __call__(self, input_buffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float|None:
    # Update buffers in the c_args struct.
    for (j,i),input_idx in self.input_replace.items():
      if not self.updatable_nodes[j][3]: setattr(self.updatable_nodes[j][2], f'f{i}', input_buffers[input_idx]._buf)
      else:
        if i == 0: self.updatable_nodes[j][1].destDevice = input_buffers[input_idx]._buf
        elif i == 1: self.updatable_nodes[j][1].srcDevice = input_buffers[input_idx]._buf

    # Update var_vals in the c_args struct.
    for j, i, v in self.updated_vars(var_vals): setattr(self.updatable_nodes[j][2], f'v{i}', v)

    # Update launch dims in the kern_params struct.
    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      node = self.updatable_nodes[j][1]
      node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_dims, *global_dims # type: ignore[misc]

    # Update graph nodes with the updated structs.
    for node, c_node_params, c_args, is_copy in self.updatable_nodes.values():
      if not is_copy: check(cuda.cuGraphExecKernelNodeSetParams(self.instance, node, ctypes.byref(c_node_params)))
      else: check(cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, ctypes.byref(c_node_params), c_args))

    return cu_time_execution(lambda: check(cuda.cuGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))
