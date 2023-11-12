from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional, Set
import functools, itertools, time
from tinygrad.helpers import DEBUG, DType, merge_dicts, dtypes
from tinygrad.ops import RawBuffer, Device, ASTRunner
from tinygrad.tensor import Tensor
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, Node
from dataclasses import dataclass

from tinygrad.runtime.ops_metal import Metal, METAL, unwrap, Cocoa, RawMetalBuffer

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU", "LLVM"]

@dataclass(frozen=True)
class JitItem:
  prg: ASTRunner
  rawbufs: List[Optional[RawBuffer]]

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[JitItem] = []
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], ShapeTracker, DType]] = {}   # (kernel_number, buffer_number) -> (input_name, expected_shapetracker, expected_type)
    self.input_has_variable_dims: Set[int] = set()

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT.split(":")[0] not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device

    # all inputs are realized
    input_tensors: Dict[Union[int, str], Tensor] = {cast(Union[int, str], k):v.realize() for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}

    # get rawbuffers
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {k:(cast(RawBuffer, v.lazydata.realized), v.lazydata.st) for k,v in input_tensors.items()}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.lazydata.st.var_vals for arg in input_tensors.values()] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])

    if self.cnt >= 2:
      assert self.stored_vals == tuple(var_vals.keys()), "this didn't change"

      if hasattr(self, 'command_buffer'): self.command_buffer.waitUntilCompleted()

      # check validity and assign the inputs
      for (j,i),(input_name, expected_st, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name][0].dtype == expected_type, f"type mismatch in JIT, {input_rawbuffers[input_name][0].dtype} != {expected_type}"
        assert input_rawbuffers[input_name][1].unbind() == expected_st, f"ShapeTracker mismatch in JIT, {input_rawbuffers[input_name][1].unbind()} != {expected_st}"
        #self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name][0]
        self.icb_commands[j].setKernelBuffer_offset_atIndex_(input_rawbuffers[input_name][0]._buf, 0, i)
      for j in self.input_has_variable_dims:
        global_size, local_size = self.jit_cache[j].prg.launch_dims(var_vals)
        self.icb_commands[j].concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      self.int_buf_view[:] = list(var_vals.values())  # how to flush this cache?

      #print(dir(self.int_buf._buf))
      #print(self.int_buf_view)
      self.command_buffer = METAL.mtl_queue.commandBuffer()
      encoder = self.command_buffer.computeCommandEncoder()
      encoder.executeCommandsInBuffer_withRange_(self.icb, Cocoa.NSRange(0,len(self.jit_cache)))
      encoder.endEncoding()
      self.command_buffer.commit()
      METAL.mtl_buffers_in_flight.append(self.command_buffer)

    elif self.cnt == 1:
      CacheCollector.start(var_vals)
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = CacheCollector.finish()
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j,ji in enumerate(self.jit_cache):
        for i,a in enumerate(ji.rawbufs):
          if a in [v[0] for v in input_rawbuffers.values()]:
            self.input_replace[(j,i)] = [(k, v[1].unbind(), v[0].dtype) for k,v in input_rawbuffers.items() if v[0] == a][0]
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"

      # create metal batch exec
      self.stored_vals = tuple(var_vals.keys())
      self.int_buf = RawMetalBuffer(len(var_vals), dtypes.int32)
      self.int_buf_view = self.int_buf.buffer_view()
      icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
      icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
      icb_descriptor.setInheritBuffers_(False)
      icb_descriptor.setInheritPipelineState_(False)
      icb_descriptor.setMaxKernelBufferBindCount_(31)
      self.icb = METAL.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache), Metal.MTLResourceOptions(0))
      self.icb_commands = []
      for j,ji in enumerate(self.jit_cache):
        descriptor = Metal.MTLComputePipelineDescriptor.new()
        descriptor.setComputeFunction_(ji.prg.clprg.fxn)
        descriptor.setSupportIndirectCommandBuffers_(True)
        pipeline_state = unwrap(METAL.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None))
        icb_command = self.icb.indirectComputeCommandAtIndex_(j)
        icb_command.setComputePipelineState_(pipeline_state)
        for i,b in enumerate(ji.rawbufs):
          icb_command.setKernelBuffer_offset_atIndex_(b._buf, 0, i)
        for i,v in enumerate(getattr(ji.prg,"vars",[])):
          vals_idx = list(var_vals.keys()).index(v)
          icb_command.setKernelBuffer_offset_atIndex_(self.int_buf._buf, vals_idx*4, len(ji.rawbufs)+i)
        global_size, local_size = ji.prg.launch_dims(var_vals)
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
        if any(isinstance(x, Node) for x in ji.prg.global_size) or any(isinstance(x, Node) for x in ji.prg.local_size):
          self.input_has_variable_dims.add(j)
        icb_command.setBarrier()
        self.icb_commands.append(icb_command)

    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)

    # clear the inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None
    self.cnt += 1
    return self.ret

class _CacheCollector:
  def __init__(self):
    self.cache: Optional[List[JitItem]] = None
  def start(self, var_vals:Optional[Dict[Variable, int]]=None):
    self.cache = []
    self.var_vals = var_vals if var_vals is not None else {}
  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    for k,v in var_vals.items(): assert k in self.var_vals and self.var_vals[k] == v, f"var_vals {k} mismatch {v} != {self.var_vals.get(k)}"
    self.cache.append(JitItem(prg, rawbufs))
  def finish(self) -> List[JitItem]:
    if self.cache is None: return []
    ret = self.cache
    self.cache = None
    return ret
CacheCollector = _CacheCollector()
