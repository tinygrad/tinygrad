# pip3 install pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
import os, subprocess, pathlib
import numpy as np
import Metal, Cocoa, libdispatch # type: ignore
from typing import List, Any
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage
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
  def __init__(self, size:int, dtype:DType): super().__init__(size, dtype, METAL.device.newBufferWithLength_options_(size*dtype.itemsize, Metal.MTLResourceStorageModeShared))
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
    global_size += [1] * (3-len(global_size))
    if local_size is None: local_size = [32]
    local_size += [1] * (3-len(local_size))

    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    else:
      METAL.mtl_buffers_in_flight.append(command_buffer)

class Float8x8DType(DType):
  def __new__(cls, strides):
    return super().__new__(cls, 100, 64, "float8x8", np.float32)
  def __init__(self, strides):
    self.strides = strides
    super().__init__()
  def __repr__(self): return f"dtypes.{self.name}({self.strides})"


float4 = DType(100, 4*4, "float4", np.float32)

class MetalCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "#include <metal_stdlib>\nusing namespace metal;\nkernel", buffer_prefix = "device ", smem_prefix = "threadgroup ",
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);", float4 = "float4",
    gid = [f"gid.{chr(120+i)}" for i in range(3)], lid = [f"lid.{chr(120+i)}" for i in range(3)],
    extra_args = ['uint3 gid [[thread_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]'])

  def hand_coded_optimizations(self):
    if self.sts[0].shape == (1024, 1024, 1) or self.sts[0].shape == (512, 512, 1):
      # Metal supports a simdgroup_float8x8 type
      """
      self.shift_to(0, amount=8, insert_before=2)
      self.shift_to(1, amount=2) # per kernel
      self.upcast()
      self.shift_to(1, amount=4, insert_before=3)
      self.reshape_and_permute(lambda x: x[0:2]+(x[2]*x[3],)+x[4:], None)
      self.local_non_reduce = 1
      self.shift_to(3, amount=8) # per kernel
      self.upcast()
      """

      amt = 8
      self.shift_to(0, amount=amt)
      self.upcast()
      self.shift_to(1, amount=amt)
      self.upcast()
      self.shift_to(2, amount=amt)
      self.upcast()

      self.shift_to(0, amount=2) #, insert_before=3)
      self.upcast()

      """
      from tinygrad.shape.shapetracker import View
      for j in range(len(self.bufs)):
        s = self.sts[j].shape
        st = self.sts[j].strides
        axes = [i for i,(s,st) in enumerate(zip(s, st)) if s == 4 and st == 1 and i >= self.shape_len-self.upcasted]
        if len(axes) == 1:
          self.dtypes[j] = float4
          self.sts[j].views[-1] = View(tuple(1 if i in axes else x for i,x in enumerate(s)),
                                      tuple(0 if i in axes else x for i,x in enumerate(st)),
                                      self.sts[j].views[-1].offset)
        axes = [i for i,(s,st) in enumerate(zip(s, st)) if s == 8 and st != 0 and i >= self.shape_len-self.upcasted]
        #axes = [i for i,(s,st) in enumerate(zip(s, st)) if s == 8 and i >= self.shape_len-self.upcasted]
        #axes = [self.shape_len-3, self.shape_len-2, self.shape_len-1]
        if len(axes) == 2:
          self.dtypes[j] = Float8x8DType([st[i] for i in axes])
          self.sts[j].views[-1] = View(tuple(1 if i in axes else x for i,x in enumerate(s)),
                                       tuple(0 if i in axes else x for i,x in enumerate(st)),
                                       self.sts[j].views[-1].offset)
        #self.simplify_ones()
        #self.upcasted -= 3
      """

MetalBuffer = Compiled(RawMetalBuffer, MetalCodegen, MetalProgram, METAL.synchronize)
