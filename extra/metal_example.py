# https://gist.github.com/awni/87f49147b13b7e119d36401e7c678a49
import numpy as np
from tinygrad.helpers import to_mv
from tinygrad.runtime.support.metal import Metal, libdispatch

# Get the default GPU device
device = Metal.MTLCreateSystemDefaultDevice()

# Make a command queue to encode command buffers to
command_queue = device.newCommandQueue()

# Compile the source code into a library
library, err = device.newLibraryWithSource_options_error_(r"""
[[kernel]] void add(
    device const float* a,
    device const float* b,
    device float* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = a[index] + b[index];
}
""", None, None)

if err:
    print(err)
    exit(1)

ns_data = library.libraryDataContents()
lib_data = to_mv(ns_data.bytes(), ns_data.length()).tobytes()
del library

data = libdispatch.dispatch_data_create(lib_data, len(lib_data), None, None)
library, err = device.newLibraryWithData_error_(data, None)

if err:
    print(err)
    exit(1)

# Get the compiled "add" kernel
function = library.newFunctionWithName_("add")
kernel, err = device.newComputePipelineStateWithFunction_error_(function, None)
if err:
    print(err)
    exit(1)

# Make the command buffer and compute command encoder
command_buffer = command_queue.commandBuffer()
compute_encoder = command_buffer.computeCommandEncoder()

# Setup the problem data
n = 4096
a = np.random.uniform(size=(n,)).astype(np.float32)
b = np.random.uniform(size=(n,)).astype(np.float32)


def np_to_mtl_buffer(x):
    opts = Metal.MTLResourceStorageModeShared
    return device.newBufferWithBytes_length_options_(
        memoryview(x).tobytes(), x.nbytes, opts,
    )

def mtl_buffer(size):
    opts = Metal.MTLResourceStorageModeShared
    return device.newBufferWithLength_options_(size, opts)

def mtl_buffer_to_np(buf):
    return np.frombuffer(to_mv(buf.contents(), buf.length()), dtype=np.float32)

# Dispatch the kernel with the correct number of threads
compute_encoder.setComputePipelineState_(kernel)

grid_dims = Metal.MTLSize(n, 1, 1)
group_dims = Metal.MTLSize(1024, 1, 1)

a_buf = np_to_mtl_buffer(a)
b_buf = np_to_mtl_buffer(b)
c_buf = mtl_buffer(a.nbytes)

compute_encoder.setBuffer_offset_atIndex_(a_buf, 0, 0)
compute_encoder.setBuffer_offset_atIndex_(b_buf, 0, 1)
compute_encoder.setBuffer_offset_atIndex_(c_buf, 0, 2)

# print(compute_encoder.methods_info["dispatchThreads:threadsPerThreadgroup:"])
compute_encoder.dispatchThreads_threadsPerThreadgroup_(grid_dims, group_dims)

# End the encoding and commit the buffer
compute_encoder.endEncoding()
command_buffer.commit()

# Wait until the computation is finished
command_buffer.waitUntilCompleted()
c = mtl_buffer_to_np(c_buf)

print(a + b)
print(c)
