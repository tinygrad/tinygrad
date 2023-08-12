# pip3 install pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
import os, subprocess, pathlib, functools
import Metal, Cocoa, libdispatch # type: ignore
from typing import List, Any
from tinygrad.codegen.linearizer import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
from tinygrad.helpers import prod, getenv, DEBUG, DType
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferMapped

METAL_XCODE = getenv("METAL_XCODE")

class _METAL:
  def __init__(self):
    self.mtl_buffers_in_flight: List[Any] = []
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueue()
  # TODO: is there a better way to do this?
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    self.mtl_buffers_in_flight.clear()
METAL = _METAL()

class RawMetalBuffer(RawBufferMapped):
  def __init__(self, size:int, dtype:DType):
    super().__init__(size, dtype, METAL.device.newBufferWithLength_options_(size*dtype.itemsize, Metal.MTLResourceStorageModeShared))
  def __del__(self):
    self._buf.release()
    super().__del__()
  def _buffer(self):
    METAL.synchronize()
    return self._buf.contents().as_buffer(self._buf.length())

def unwrap(x):
  ret, err = x
  assert err is None, str(err)
  return ret

class MetalProgram:
  def __init__(self, name:str, prg:str):
    if METAL_XCODE:
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      lib = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
      data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
      self.library = unwrap(METAL.device.newLibraryWithData_error_(data, None))
    else:
      options = Metal.MTLCompileOptions.alloc().init()
      self.library = unwrap(METAL.device.newLibraryWithSource_options_error_(prg, options, None))
    self.fxn = self.library.newFunctionWithName_(name)
    # hacks to disassemble shader
    if DEBUG >= 5:
      arc = unwrap(METAL.device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None))
      desc = Metal.MTLComputePipelineDescriptor.alloc().init()
      desc.setComputeFunction_(self.fxn)
      unwrap(arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None))
      unwrap(arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None))
      # clone https://github.com/dougallj/applegpu.git in tinygrad/disassemblers
      os.system(f"cd {pathlib.Path(__file__).parent.parent.parent}/disassemblers/applegpu && python3 compiler_explorer.py /tmp/shader.bin")
    self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def __call__(self, global_size, local_size, *bufs, wait=False):
    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    METAL.mtl_buffers_in_flight.append(command_buffer)

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(
  kernel_prefix = "#include <metal_stdlib>\nusing namespace metal;\nkernel", buffer_prefix = "device ", smem_prefix = "threadgroup ",
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);", float4 = "float4", uses_ptr_arithmetic=True,
  gid = [f"gid.{chr(120+i)}" for i in range(3)], lid = [f"lid.{chr(120+i)}" for i in range(3)],
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']))
MetalBuffer = Compiled(RawMetalBuffer, LinearizerOptions(), renderer, MetalProgram, METAL.synchronize)
