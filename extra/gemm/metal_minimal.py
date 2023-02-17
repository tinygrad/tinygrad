# pip3 install pyobjc-framework-Metal
import Metal
import numpy as np

device = Metal.MTLCreateSystemDefaultDevice()
mtl_queue = device.newCommandQueue()

N = 2048

a = device.newBufferWithLength_options_(N*N*4, Metal.MTLResourceStorageModeShared)
b = device.newBufferWithLength_options_(N*N*4, Metal.MTLResourceStorageModeShared)

program = """
#include <metal_stdlib>
using namespace metal;

kernel void test(const device float *in [[ buffer(0) ]],
                device float  *out [[ buffer(1) ]],
                uint id [[ thread_position_in_grid ]]) {
    //out[id] = in[id];
    out[id] = 4;
}
"""

options = Metal.MTLCompileOptions.alloc().init()
library = device.newLibraryWithSource_options_error_(program, options, None)
fxn = library[0].newFunctionWithName_("test")

pipeline_state = device.newComputePipelineStateWithFunction_error_(fxn, None)

command_buffer = mtl_queue.commandBuffer()
encoder = command_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline_state[0])
encoder.setBuffer_offset_atIndex_(a, 0, 0)
encoder.setBuffer_offset_atIndex_(b, 0, 1)
encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(N*N,1,1), Metal.MTLSize(1024,1,1))
encoder.endEncoding()
command_buffer.commit()
command_buffer.waitUntilCompleted()
print("her")
nb = np.frombuffer(b''.join(b.contents()[0:0x10]), dtype=np.float32)
print(nb)
