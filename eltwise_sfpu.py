# This is a python implementation of the eltwise_sfpu example from tt-metal
# Python accesses the C++ API of tt-metalium using cppyy
# C++ source this is based on: https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.cpp
# Tutorial: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/eltwise_sfpu.html

import os
os.environ["TT_METAL_HOME"] = "/root/tt-metal/"
os.environ["ARCH_NAME"] = "wormhole_b0"

import numpy as np
import cppyy
import cppyy.ll

cppyy.ll.set_signals_as_exception(True)
# cppyy.set_debug(True)

def rp(path): return os.path.join(os.environ.get("TT_METAL_HOME"), path)

cppyy.load_library(rp("./build/tt_metal/libtt_metal.so"))

cppyy.add_include_path(rp("."))
cppyy.add_include_path(rp("./.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f"))
cppyy.add_include_path(rp("./tt_metal"))
cppyy.add_include_path(rp("./tt_metal/api"))
cppyy.add_include_path(rp("./tt_metal/api/tt-metalium"))
cppyy.add_include_path(rp("./tt_metal/common"))
cppyy.add_include_path(rp("./tt_metal/hostdevcommon/api"))
cppyy.add_include_path(rp("./tt_metal/impl"))
cppyy.add_include_path(rp("./tt_metal/impl/dispatch"))
cppyy.add_include_path(rp("./tt_metal/third_party/umd/device/api"))
cppyy.add_include_path(rp("./tt_stl"))
cppyy.add_include_path(rp("./tt_stl/tt_stl"))

cppyy.include(rp('./tt_metal/api/tt-metalium/kernel.hpp'))
cppyy.include(rp('./tt_metal/api/tt-metalium/command_queue.hpp'))
cppyy.include(rp('./tt_metal/impl/dispatch/hardware_command_queue.hpp'))
cppyy.include(rp('./tt_metal/api/tt-metalium/host_api.hpp'))
cppyy.include(rp('./tt_metal/api/tt-metalium/bfloat16.hpp'))
cppyy.include(rp('./tt_metal/api/tt-metalium/device.hpp'))


gbl = cppyy.gbl
tt = gbl.tt
tt_metal = gbl.tt.tt_metal

device_id = 0
device = tt_metal.CreateDevice(device_id)

cq = device.command_queue()
program = tt_metal.CreateProgram()

core = tt.umd.xy_pair(0, 0)

num_element = 64 * 1024
single_tile_size = 2 * 1024
num_tiles = 64
dram_buffer_size = single_tile_size * num_tiles

dram_config = tt_metal.InterleavedBufferConfig(device, dram_buffer_size, dram_buffer_size, tt_metal.BufferType.DRAM, tt_metal.TensorMemoryLayout.INTERLEAVED)

src0_dram_buffer = tt_metal.CreateBuffer(dram_config)
dram_buffer_src0_addr = src0_dram_buffer.address()

dst_dram_buffer = tt_metal.CreateBuffer(dram_config)
dram_buffer_dst_addr = dst_dram_buffer.address()

src0_bank_id = 0
dst_bank_id = 0

# Use circular buffers to set input and output buffers that the compute engine will use.

src0_cb_index = tt.CBIndex.c_0
num_input_tiles = 2
cb_src0_config = tt_metal.CircularBufferConfig(num_input_tiles * single_tile_size, {src0_cb_index: tt.DataFormat.Float16_b}) \
    .set_page_size(src0_cb_index, single_tile_size)
cb_src0 = tt_metal.CreateCircularBuffer(program, core, cb_src0_config)

output_cb_index = tt.CBIndex.c_16
num_output_tiles = 2
cb_output_config = tt_metal.CircularBufferConfig(num_output_tiles * single_tile_size, {output_cb_index: tt.DataFormat.Float16_b}) \
    .set_page_size(output_cb_index, single_tile_size)
cb_output = tt_metal.CreateCircularBuffer(program, core, cb_output_config)

# Specify data movement kernels for reading/writing data to/from DRAM.

data_movement_config_R1 = tt_metal.DataMovementConfig()
data_movement_config_R1.processor = tt_metal.DataMovementProcessor.RISCV_1
data_movement_config_R1.noc = tt_metal.NOC.RISCV_1_default
unary_reader_kernel_id = tt_metal.CreateKernel(
    program,
    "tt_metal/kernels/dataflow/reader_unary.cpp",
    core,
    data_movement_config_R1
)

data_movement_config_R0 = tt_metal.DataMovementConfig()
data_movement_config_R0.processor = tt_metal.DataMovementProcessor.RISCV_0
data_movement_config_R0.noc = tt_metal.NOC.RISCV_0_default
unary_writer_kernel_id = tt_metal.CreateKernel(
    program,
    "tt_metal/kernels/dataflow/writer_unary.cpp",
    core,
    data_movement_config_R0
)

# Set the parameters that the compute kernel will use.

compute_kernel_args = [num_tiles, 1]
math_approx_mode = False

sfpu_defines = {
    "SFPU_OP_EXP_INCLUDE": "1",
    "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"
}

compute_config = tt_metal.ComputeConfig()
compute_config.math_approx_mode = math_approx_mode
compute_config.compile_args = compute_kernel_args
compute_config.defines = sfpu_defines
eltwise_sfpu_kernel_id = tt_metal.CreateKernel(
    program,
    "tt_metal/kernels/compute/eltwise_sfpu.cpp",
    core,
    compute_config
)

# Create source data and write to DRAM.

src0_vec = gbl.create_random_vector_of_bfloat16(dram_buffer_size, 1, 42)
src0_data = src0_vec.data() # Turn into LowLevelView, cppyy infers with the memory layout

tt_metal.EnqueueWriteBuffer(cq, src0_dram_buffer, src0_data, False)

# Configure program and runtime kernel arguments, then execute.

tt_metal.SetRuntimeArgs(program, unary_reader_kernel_id, core, [src0_dram_buffer.address(), src0_bank_id, num_tiles])
tt_metal.SetRuntimeArgs(program, unary_writer_kernel_id, core, [dst_dram_buffer.address(), dst_bank_id, num_tiles])

tt_metal.EnqueueProgram(cq, program, False)
tt_metal.Finish(cq)

# Read the result and compare to a golden result. Record pass/fail and teardown.

result_ptr = gbl.calloc(num_element, 2) # Create LowLevelView to have a pointer (vector doesn't work with cppyy)

tt_metal.EnqueueReadBuffer(cq, dst_dram_buffer, result_ptr, True)

# Convert the result to numpy and compare with the expected result.

def bfloat16_to_float32(binary_data):
    # bfloat16 is just the upper 16 bits of a float32
    binary_data = np.asarray(binary_data, dtype=np.uint16)
    uint32_data = binary_data.astype(np.uint32) << 16
    float_array = uint32_data.view(np.float32)
    return float_array

v_tt_buf = np.frombuffer(result_ptr, dtype=np.uint16, count=num_element)
v_tt = bfloat16_to_float32(v_tt_buf)
print(v_tt)

v_src0 = bfloat16_to_float32(np.frombuffer(src0_data, dtype=np.uint16, count=num_element))
v_ref = np.exp(v_src0)
print(v_ref)

test_pass = np.allclose(v_tt, v_ref, rtol=0.02, atol=0.02)
print(f"Result correct: {test_pass}")

tt_metal.CloseDevice(device)
gbl.free(result_ptr)
