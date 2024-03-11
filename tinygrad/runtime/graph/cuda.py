import ctypes
from typing import Any, Optional, Tuple, Dict, List, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import init_c_var, encode_args_cuda_style, all_same, GraphException
from tinygrad.device import CompiledASTRunner, update_stats, Buffer
from tinygrad.runtime.ops_cuda import check, cu_time_execution
from tinygrad.shape.symbolic import Variable
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals

class CUDAGraph:
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    devices = [ji.prg.clprg.device if isinstance(ji.prg, CompiledASTRunner) else None for ji in jit_cache]
    if len(devices) == 0 or not all_same(devices) or devices[0] is None: raise GraphException
    self.device = devices[0]
    self.set_device()

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))
    self.updatable_nodes: Dict[int, Tuple[Any, Any, Any]] = {} # Dict[jc index] = tuple(graph node, node params, input kernel params)

    self.graph = self.graph_create()
    graph_node: Optional[ctypes._CData] = None

    for (j,i),input_name in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name]
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledASTRunner = cast(CompiledASTRunner, ji.prg)

      c_deps = (type(graph_node)*1)(*(graph_node,)) if graph_node is not None else None
      c_kernel_input_config, c_input_params = encode_args_cuda_style([cast(Buffer, x)._buf for x in ji.rawbufs], [var_vals[x] for x in prg.vars], *self.encode_args_info())  # noqa: E501
      c_node_params = self.build_kernel_node_params(prg, *cast(Tuple[List[int], List[int]], prg.launch_dims(var_vals)), c_kernel_input_config)
      graph_node = self.graph_add_kernel_node(self.graph, c_deps, c_node_params)

      if j in self.jc_idxs_with_updatable_launch_dims or j in self.jc_idxs_with_updatable_var_vals or j in self.jc_idxs_with_updatable_rawbufs:
        self.updatable_nodes[j] = (graph_node, c_node_params, c_input_params)

    self.instance = self.graph_instantiate(self.graph)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    self.set_device()
    # Update rawbuffers in the c_input_params struct.
    for (j,i),input_idx in self.input_replace.items():
      setattr(self.updatable_nodes[j][2], f'f{i}', input_rawbuffers[input_idx]._buf)

    # Update var_vals in the c_input_params struct.
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        setattr(self.updatable_nodes[j][2], f'f{len(self.jit_cache[j].rawbufs) + i}', var_vals[v])

    # Update launch dims in the c_node_params struct.
    for j in self.jc_idxs_with_updatable_launch_dims:
      self.set_kernel_node_launch_dims(self.updatable_nodes[j][1], *cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals))

    # Update graph nodes with the updated structs.
    for node, c_node_params, _ in self.updatable_nodes.values():
      self.graph_exec_kernel_node_set_params(self.instance, node, ctypes.byref(c_node_params))

    et = self.graph_launch(self.instance, None, wait=wait)
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device=f"<GPU>:{self.device}")
    return et

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))

  def set_device(self): pass
  def encode_args_info(self): return (cuda.CUdeviceptr_v2, (1,2,0))
  def graph_create(self): return init_c_var(cuda.CUgraph(), lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))
  def graph_instantiate(self, graph):
    return init_c_var(cuda.CUgraphExec(), lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), graph, None, None, 0)))
  def graph_add_kernel_node(self, graph, c_deps, c_node_params):
    return init_c_var(cuda.CUgraphNode(), lambda x: check(cuda.cuGraphAddKernelNode(ctypes.byref(x), graph, c_deps, ctypes.sizeof(c_deps)//8 if c_deps else 0, ctypes.byref(c_node_params))))  # noqa: E501
  def graph_launch(self, *args, wait=False): return cu_time_execution(lambda: check(cuda.cuGraphLaunch(*args)), enable=wait)
  def graph_exec_kernel_node_set_params(self, *args): return check(cuda.cuGraphExecKernelNodeSetParams(*args))
  def build_kernel_node_params(self, prg, global_size, local_size, c_kernel_config):
    return cuda.CUDA_KERNEL_NODE_PARAMS(prg.clprg.prg, *global_size, *local_size, 0, None, c_kernel_config)
  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]):
    node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_size, *global_size
