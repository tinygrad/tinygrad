# pip3 install pyobjc-framework-Metal
import Metal # type: ignore
import numpy as np
from typing import List, Any
from tinygrad.ops import DEBUG, GlobalCounters

device = Metal.MTLCreateSystemDefaultDevice()
mtl_queue = device.newCommandQueue()
mtl_buffers_in_flight : List[Any] = []

def sync():
  global mtl_buffers_in_flight
  for cbuf in mtl_buffers_in_flight: cbuf.waitUntilCompleted()
  mtl_buffers_in_flight = []

class CLImage:
  def __init__(self, shape): raise NotImplementedError("Metal runtime doesn't support images")

class CLBuffer:
  def __init__(self, size): self._cl = device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
  def copyin(self, b:np.ndarray):
    # TODO: don't reallocate buffer!
    self._cl = device.newBufferWithBytes_length_options_(
      b.astype(np.float32).data,
      b.size*4,
      Metal.MTLResourceStorageModeShared)

  def toCPU(self):
    sync()
    return np.frombuffer(self._cl.contents().as_buffer(self._cl.length()), dtype=np.float32)

  # TODO: remove copyout everywhere
  def copyout(self, a:np.ndarray): np.copyto(a, self.toCPU().reshape(a.shape))

class CLProgram:
  kernel_prefix = "using namespace metal;\nkernel"
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  gid = [f"gid.{chr(120+i)}" for i in range(3)]
  def __init__(self, name:str, prg:str, op_estimate:int=0, mem_estimate:int=0):
    self.name, self.op_estimate, self.mem_estimate = name, op_estimate, mem_estimate
    options = Metal.MTLCompileOptions.alloc().init()
    if DEBUG >= 4: print("Metal compile", prg)
    self.library = device.newLibraryWithSource_options_error_(prg, options, None)
    assert self.library[0] is not None, str(self.library)
    self.fxn = self.library[0].newFunctionWithName_(name)
  def __call__(self, global_size, local_size, *args):
    global_size += [1] * (3-len(global_size))
    if local_size is None: local_size = []
    local_size += [1] * (3-len(local_size))
    if DEBUG >= 2: print("METAL launch", global_size, local_size)
    pipeline_state = device.newComputePipelineStateWithFunction_error_(self.fxn, None)
    assert pipeline_state[0] is not None, str(pipeline_state)
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_state[0])
    for i,a in enumerate(args):
      encoder.setBuffer_offset_atIndex_(a, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    mtl_buffers_in_flight.append(command_buffer)
    GlobalCounters.log_kernel(self.op_estimate, self.mem_estimate)
    return command_buffer
