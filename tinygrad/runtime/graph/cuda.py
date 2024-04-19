import ctypes, collections
from typing import Any, Optional, Tuple, Dict, List, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import init_c_var, GraphException, getenv, colored
from tinygrad.device import CompiledRunner, Buffer, MultiDeviceJITGraph, BufferXfer, Device, BufferOptions
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args, cu_time_execution
from tinygrad.shape.symbolic import Variable
from tinygrad.engine.realize import ExecItem
from tinygrad.engine.jit import get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals

class CUDAGraph(MultiDeviceJITGraph):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    # Check all jit items are compatible.
    if not all(isinstance(ji.prg, CompiledRunner) or isinstance(ji.prg, BufferXfer) for ji in jit_cache): raise GraphException

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))
    self.updatable_nodes: Dict[int, Tuple[Any, Any, Any, bool]] = {} # Dict[jc index] = tuple(graph node, node params, input kernel params, is memcpy)

    self.graph = init_c_var(cuda.CUgraph(), lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))
    self.w_dependency_map: Dict[Any, Any] = {}
    self.r_dependency_map: Dict[Any, List[Any]] = collections.defaultdict(list)
    self.cpu_buffers = []

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledRunner):
        global_size, local_size = ji.prg.launch_dims(var_vals)

        new_node = cuda.CUgraphNode()
        deps = self.access_resources(ji.rawbufs[(outs:=ji.prg.outcount):], ji.rawbufs[:outs], new_dependency=new_node)
        c_deps = (cuda.CUgraphNode*len(deps))(*deps) if deps else None

        c_args, vargs = encode_args([cast(Buffer, x)._buf for x in ji.rawbufs], [var_vals[x] for x in ji.prg.vars])
        kern_params = cuda.CUDA_KERNEL_NODE_PARAMS(ji.prg.clprg.prg, *global_size, *local_size, 0, None, vargs)
        check(cuda.cuGraphAddKernelNode(ctypes.byref(new_node), self.graph, c_deps, len(deps), ctypes.byref(kern_params)))

        if j in self.jc_idxs_with_updatable_launch_dims or j in self.jc_idxs_with_updatable_var_vals or j in self.jc_idxs_with_updatable_rawbufs:
          self.updatable_nodes[j] = (new_node, kern_params, c_args, False)
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.rawbufs[0:2]]
        src_dev, dest_dev = cast(CUDADevice, Device[src.device]), cast(CUDADevice, Device[dest.device])
        node_from = cuda.CUgraphNode()
        deps = self.access_resources(read=[src], write=[dest], new_dependency=node_from)
        c_deps = (cuda.CUgraphNode*len(deps))(*deps) if deps else None
        if getenv("CUDA_P2P", int(CUDADevice.peer_access)):
          cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                            dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                            WidthInBytes=dest.nbytes, Height=1, Depth=1)
          check(cuda.cuGraphAddMemcpyNode(ctypes.byref(node_from), self.graph, c_deps, len(deps), ctypes.byref(cp_params), src_dev.context))
        else:
          self.cpu_buffers.append(cpu_buffer:=Buffer(device=src.device, dtype=src.dtype, size=src.size, options=BufferOptions(host=True)).allocate())

          node_to = cuda.CUgraphNode()
          cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                            dstMemoryType=cuda.CU_MEMORYTYPE_HOST, dstHost=cpu_buffer._buf, dstPitch=dest.nbytes, dstHeight=1,
                                            WidthInBytes=dest.nbytes, Height=1, Depth=1)
          check(cuda.cuGraphAddMemcpyNode(ctypes.byref(node_to), self.graph, c_deps, len(deps), ctypes.byref(cp_params), src_dev.context))
          cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_HOST, srcHost=cpu_buffer._buf, srcPitch=src.nbytes, srcHeight=1,
                                            dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                            WidthInBytes=dest.nbytes, Height=1, Depth=1)
          check(cuda.cuGraphAddMemcpyNode(ctypes.byref(node_from), self.graph, (cuda.CUgraphNode*1)(node_to), 1,
                                          ctypes.byref(cp_params), dest_dev.context))
        if j in self.jc_idxs_with_updatable_rawbufs: self.updatable_nodes[j] = (node_from, cp_params, src_dev.context, True)

    self.instance = init_c_var(cuda.CUgraphExec(), lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))

    # clear jit inputs to allow their memory to be freed/reused
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None
    super().__init__(colored(f"<batched {len(self.jit_cache)}>", "cyan"), "CUDA", *get_jit_stats(jit_cache))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Update rawbuffers in the c_args struct.
    for (j,i),input_idx in self.input_replace.items():
      if not self.updatable_nodes[j][3]: setattr(self.updatable_nodes[j][2], f'f{i}', input_rawbuffers[input_idx]._buf)
      else:
        if i == 0: self.updatable_nodes[j][1].destDevice = input_rawbuffers[input_idx]._buf
        elif i == 1: self.updatable_nodes[j][1].srcDevice = input_rawbuffers[input_idx]._buf

    # Update var_vals in the c_args struct.
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledRunner, self.jit_cache[j].prg).vars):
        setattr(self.updatable_nodes[j][2], f'v{i}', var_vals[v])

    # Update launch dims in the kern_params struct.
    for j in self.jc_idxs_with_updatable_launch_dims:
      self.set_kernel_node_launch_dims(self.updatable_nodes[j][1], *cast(CompiledRunner, self.jit_cache[j].prg).launch_dims(var_vals))

    # Update graph nodes with the updated structs.
    for node, c_node_params, c_args, is_copy in self.updatable_nodes.values():
      if not is_copy: check(cuda.cuGraphExecKernelNodeSetParams(self.instance, node, ctypes.byref(c_node_params)))
      else: check(cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, ctypes.byref(c_node_params), c_args))

    return cu_time_execution(lambda: check(cuda.cuGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))

  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]):
    node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_size, *global_size

  def access_resources(self, read, write, new_dependency):
    wait_nodes = []

    for rawbuf in read + write:
      if rawbuf._buf.value in self.w_dependency_map: wait_nodes.append(self.w_dependency_map[rawbuf._buf.value])
    for rawbuf in write:
      if rawbuf._buf.value in self.r_dependency_map: wait_nodes.extend(self.r_dependency_map.pop(rawbuf._buf.value))

    if new_dependency is not None:
      for rawbuf in read: self.r_dependency_map[rawbuf._buf.value].append(new_dependency)
      for rawbuf in write: self.w_dependency_map[rawbuf._buf.value] = new_dependency
    return {id(x):x for x in wait_nodes}.values()
