import ctypes, time
from typing import Any, Optional, Tuple, Dict, List, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import init_c_var, all_same, GraphException
from tinygrad.device import CompiledASTRunner, update_stats, Buffer, MultiDeviceJITGraph
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args
from tinygrad.shape.symbolic import Variable
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals

class CUDAGraph(MultiDeviceJITGraph):
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))
    self.updatable_nodes: Dict[int, Tuple[Compiled, Any, Any, Any]] = {} # Dict[jc index] = tuple(dev, graph node, node params, input kernel params)

    self.graphs: Dict[CUDADevice, cuda.CUgraph] = {}
    self.instances: Dict[CUDADevice, cuda.CUgraphExec] = {}
    self.prev_graph_node = {}

    # Check all jit items are compatible.
    compiled_devices = set()
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): compiled_devices.add(ji.prg.device)
      else: raise GraphException
    if any(not isinstance(d, CUDADevice) for d in compiled_devices): raise GraphException

    self.devices: List[CUDADevice] = list(compiled_devices) #type:ignore
    for dev in self.devices:
      self.graphs[dev] = init_c_var(cuda.CUgraph(), lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))

    for j,ji in enumerate(self.jit_cache):
      global_size, local_size = ji.prg.launch_dims(var_vals)

      c_args, vargs = encode_args([cast(Buffer, x)._buf for x in ji.rawbufs], [var_vals[x] for x in ji.prg.vars])
      kern_params = cuda.CUDA_KERNEL_NODE_PARAMS(ji.prg.clprg.prg, *global_size, *local_size, 0, None, vargs)

      c_deps = (cuda.CUgraphNode*1)(self.prev_graph_node[ji.prg.device]) if ji.prg.device in self.prev_graph_node else None
      check(cuda.cuGraphAddKernelNode(ctypes.byref(node := cuda.CUgraphNode()), self.graphs[ji.prg.device],
                                      c_deps, 1 if c_deps else 0, ctypes.byref(kern_params)))
      self.prev_graph_node[ji.prg.device] = node

      if j in self.jc_idxs_with_updatable_launch_dims or j in self.jc_idxs_with_updatable_var_vals or j in self.jc_idxs_with_updatable_rawbufs:
        self.updatable_nodes[j] = (ji.prg.device, node, kern_params, c_args)

    for dev, graph in self.graphs.items():
      self.instances[dev] = init_c_var(cuda.CUgraphExec(), lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), graph, None, None, 0)))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Update rawbuffers in the c_args struct.
    for (j,i),input_idx in self.input_replace.items():
      setattr(self.updatable_nodes[j][3], f'f{i}', input_rawbuffers[input_idx]._buf)

    # Update var_vals in the c_args struct.
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        setattr(self.updatable_nodes[j][3], f'f{len(self.jit_cache[j].rawbufs) + i}', var_vals[v])

    # Update launch dims in the kern_params struct.
    for j in self.jc_idxs_with_updatable_launch_dims:
      self.set_kernel_node_launch_dims(self.updatable_nodes[j][2], *cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals))

    # Update graph nodes with the updated structs.
    for dev, node, c_node_params, _ in self.updatable_nodes.values():
      check(cuda.cuGraphExecKernelNodeSetParams(self.instances[dev], node, ctypes.byref(c_node_params)))

    for dev,instance in self.instances.items():
      check(cuda.cuCtxSetCurrent(dev.context))
      check(cuda.cuGraphLaunch(instance, None))

    et = None
    if wait:
      st = time.perf_counter()
      for dev in self.devices:
        check(cuda.cuCtxSetCurrent(dev.context))
        check(cuda.cuCtxSynchronize())
      et = time.perf_counter() - st

    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device=f"CUDA")
    return et

  def __del__(self):
    for graph in self.graphs.values(): check(cuda.cuGraphDestroy(graph))
    for inst in self.instances.values(): check(cuda.cuGraphExecDestroy(inst))

  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]):
    node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_size, *global_size
