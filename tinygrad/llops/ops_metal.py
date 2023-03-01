# pip3 install pyobjc-framework-Metal pyobjc-framework-libdispatch
import os, subprocess, pathlib
import Metal, Cocoa, libdispatch # type: ignore
import numpy as np
from typing import List, Any
from tinygrad.ops import GlobalCounters
from tinygrad.helpers import prod, getenv, DEBUG
from tinygrad.ops import CompiledBuffer, RawBuffer

METAL_XCODE = getenv("METAL_XCODE")

class METAL:
  device = None
  mtl_queue = None
  mtl_buffers_in_flight : List[Any] = []
  def __init__(self):
    if METAL.device is not None: return
    METAL.device = Metal.MTLCreateSystemDefaultDevice()
    METAL.mtl_queue = METAL.device.newCommandQueue()

class RawMetalBuffer(RawBuffer):
  def __init__(self, size): self._cl = METAL().device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
  def __del__(self): self._cl.release()

  def _as_np(self): return np.frombuffer(self._cl.contents().as_buffer(self._cl.length()), dtype=np.float32)

  def copyin(self, b:np.ndarray):
    np.copyto(self._as_np(), b.reshape(-1).data)

  def copyout(self, a:np.ndarray):
    for cbuf in METAL.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    METAL.mtl_buffers_in_flight = []
    np.copyto(a, self._as_np().reshape(a.shape))

class MetalProgram:
  def __init__(self, name:str, prg:str):
    if DEBUG >= 6:  # dump llvm
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
      dis = subprocess.check_output(['/Users/kafka/Downloads/clang+llvm-15.0.7-arm64-apple-darwin22.0/bin/llvm-dis'], input=air)
      print(dis.decode('utf-8'))
    if METAL_XCODE:
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
      lib = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
      data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
      self.library, err = METAL().device.newLibraryWithData_error_(data, None)
    else:
      options = Metal.MTLCompileOptions.alloc().init()
      self.library, err = METAL().device.newLibraryWithSource_options_error_(prg, options, None)
    assert err is None, str(err)
    self.fxn = self.library.newFunctionWithName_(name)
    # hacks to disassemble shader
    if DEBUG >= 5:
      arc, err = METAL().device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None)
      assert err is None, str(err)
      desc = Metal.MTLComputePipelineDescriptor.alloc().init()
      desc.setComputeFunction_(self.fxn)
      _, err = arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None)
      assert err is None, str(err)
      _, err = arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None)
      assert err is None, str(err)
      # clone https://github.com/dougallj/applegpu.git in the root of tinygrad
      os.system(f"cd {pathlib.Path(__file__).parent.parent.parent}/applegpu && python3 compiler_explorer.py /tmp/shader.bin")
    self.pipeline_state, err = METAL().device.newComputePipelineStateWithFunction_error_(self.fxn, None)
    assert err is None, str(err)

  def __call__(self, global_size, local_size, *bufs, wait=False):
    global_size += [1] * (3-len(global_size))
    if local_size is None: local_size = [32]
    local_size += [1] * (3-len(local_size))

    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = METAL().mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a._cl, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    else:
      METAL().mtl_buffers_in_flight.append(command_buffer)

from tinygrad.compiler.gpu import GPUCodegen, GPULanguage

metal_lang = GPULanguage(
  kernel_prefix = "#include <metal_stdlib>\nusing namespace metal;\nkernel", buffer_prefix = "device ", smem_prefix = "threadgroup ",
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);", float4 = "float4",
  gid = [f"gid.{chr(120+i)}" for i in range(3)], lid = [f"lid.{chr(120+i)}" for i in range(3)],
  extra_args = ['uint3 gid [[thread_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]'])

class MetalBuffer(CompiledBuffer):
  @staticmethod
  def create_raw_buffer(shape): return RawMetalBuffer(4*prod(shape))
  @staticmethod
  def compile(ast, output_buffer):
    k = GPUCodegen(ast, output_buffer, metal_lang)
    return (k.codegen().build(MetalProgram), k.bufs, k.ret)