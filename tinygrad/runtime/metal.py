# pip3 install pyobjc-framework-Metal
import Metal # type: ignore
import numpy as np
from typing import List, Any
from tinygrad.ops import DEBUG, GlobalCounters
from tinygrad.helpers import prod

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
  lid = [f"lid.{chr(120+i)}" for i in range(3)]
  def __init__(self, name:str, prg:str, op_estimate:int=0, mem_estimate:int=0):
    self.name, self.op_estimate, self.mem_estimate = name, op_estimate, mem_estimate
    options = Metal.MTLCompileOptions.alloc().init()
    if DEBUG >= 4: print("Metal compile", prg)
    # hacks to get LLVM
    if DEBUG >= 6:
      import os
      with open("/tmp/prog.metal", "w") as f:
        f.write(prg)
      os.system('rm -f /tmp/prog.air*')
      os.system('xcrun -sdk macosx metal -c /tmp/prog.metal -o /tmp/prog.air')
      os.system('/Users/kafka/Downloads/clang+llvm-15.0.7-arm64-apple-darwin22.0/bin/llvm-dis /tmp/prog.air')
      os.system('cat /tmp/prog.air.ll')
    self.library = device.newLibraryWithSource_options_error_(prg, options, None)
    assert self.library[0] is not None, str(self.library)
    self.fxn = self.library[0].newFunctionWithName_(name)
    # hacks to disassemble shader
    if DEBUG >= 5:
      import os
      arc, err = device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None)
      assert err is None, str(err)
      desc = Metal.MTLComputePipelineDescriptor.alloc().init()
      desc.setComputeFunction_(self.fxn)
      _, err = arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None)
      assert err is None, str(err)
      import Cocoa
      _, err = arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None)
      assert err is None, str(err)
      # https://github.com/dougallj/applegpu.git
      os.system("cd /Users/kafka/fun/m1/applegpu && python3 compiler_explorer.py /tmp/shader.bin")
  def __call__(self, global_size, local_size, *args):
    global_size += [1] * (3-len(global_size))
    if local_size is None: local_size = [32]
    local_size += [1] * (3-len(local_size))
    # TODO: only create this once for the program
    pipeline_state = device.newComputePipelineStateWithFunction_error_(self.fxn, None)
    assert pipeline_state[0] is not None, str(pipeline_state)
    assert prod(local_size) <= pipeline_state[0].maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {pipeline_state[0].maxTotalThreadsPerThreadgroup()} with exec width {pipeline_state[0].threadExecutionWidth()} memory length {pipeline_state[0].staticThreadgroupMemoryLength()}"
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_state[0])
    for i,a in enumerate(args):
      encoder.setBuffer_offset_atIndex_(a, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if DEBUG >= 2:
      command_buffer.waitUntilCompleted()
      et = command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
      print(f"METAL et {et*1e6:8.2f} us  {self.name:28s} launch {str(global_size):18s} {local_size}")
      GlobalCounters.time_sum += et
    else:
      mtl_buffers_in_flight.append(command_buffer)
    GlobalCounters.log_kernel(self.op_estimate, self.mem_estimate)
    return command_buffer
