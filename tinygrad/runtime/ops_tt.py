from __future__ import annotations
import functools
from tinygrad.helpers import mv_address
from tinygrad.device import Compiled, Compiler, Allocator, BufferSpec, CPUProgram
from tinygrad.ops import UOp
from tinygrad.renderer import Renderer

import os
os.environ["TT_METAL_HOME"] = "/root/tt-metal/"
os.environ["ARCH_NAME"] = "wormhole_b0"

import cppyy
import cppyy.ll

cppyy.ll.set_signals_as_exception(True)
# cppyy.set_debug(True)

def load_tt():
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
    return cppyy.gbl


gbl = load_tt()
tt = gbl.tt
tt_metal = gbl.tt.tt_metal

def tt_time_execution(cb, enable=False) -> float|None:
  if not enable: return cb()
  cb()
  return 0 * 1e-3


class TTRenderer(Renderer):
  device = "TT"
  supports_float4 = False
  has_local = False
  global_max = None

  def render(self, uops: list[UOp]) -> str:
    reader = """
#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
"""

    writer = """
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);

        cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
"""

    compute = """
#define SFPU_OP_EXP_INCLUDE 1
#define SFPU_OP_CHAIN_0 

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_16);

            cb_pop_front(tt::CBIndex::c_0, 1);
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
"""
    return "[SEP]".join([reader, writer, compute])

class TTProgram:
  def __init__(self, dev:TTDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib

    reader_src, writer_src, compute_src = lib.decode("utf-8").split("[SEP]")

    self.prog = tt_metal.CreateProgram()
    self.core = tt.umd.xy_pair(0, 0)

    self.single_tile_size = 2 * 1024
    self.num_tiles = 64

    src0_cb_index = tt.CBIndex.c_0
    num_input_tiles = 2
    cb_src0_config = tt_metal.CircularBufferConfig(num_input_tiles * self.single_tile_size, {src0_cb_index: tt.DataFormat.Float16_b}) \
        .set_page_size(src0_cb_index, self.single_tile_size)
    cb_src0 = tt_metal.CreateCircularBuffer(self.prog, self.core, cb_src0_config)

    output_cb_index = tt.CBIndex.c_16
    num_output_tiles = 2
    cb_output_config = tt_metal.CircularBufferConfig(num_output_tiles * self.single_tile_size, {output_cb_index: tt.DataFormat.Float16_b}) \
        .set_page_size(output_cb_index, self.single_tile_size)
    cb_output = tt_metal.CreateCircularBuffer(self.prog, self.core, cb_output_config)


    data_movement_config_R1 = tt_metal.DataMovementConfig()
    data_movement_config_R1.processor = tt_metal.DataMovementProcessor.RISCV_1
    data_movement_config_R1.noc = tt_metal.NOC.RISCV_1_default
    self.unary_reader_kernel_id = tt_metal.CreateKernelFromString(
        self.prog,
        reader_src,
        self.core,
        data_movement_config_R1
    )


    data_movement_config_R0 = tt_metal.DataMovementConfig()
    data_movement_config_R0.processor = tt_metal.DataMovementProcessor.RISCV_0
    data_movement_config_R0.noc = tt_metal.NOC.RISCV_0_default
    self.unary_writer_kernel_id = tt_metal.CreateKernelFromString(
        self.prog,
        writer_src,
        self.core,
        data_movement_config_R0
    )


    compute_config = tt_metal.ComputeConfig()
    compute_config.math_approx_mode = False
    compute_config.compile_args = [self.num_tiles, 1]
    self.eltwise_sfpu_kernel_id = tt_metal.CreateKernelFromString(
        self.prog,
        compute_src,
        self.core,
        compute_config
    )


  def __del__(self):
    pass

  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)

    src0_bank_id = 0
    dst_bank_id = 0

    tt_metal.SetRuntimeArgs(self.prog, self.unary_reader_kernel_id, self.core, [args[0].address(), src0_bank_id, self.num_tiles])
    tt_metal.SetRuntimeArgs(self.prog, self.unary_writer_kernel_id, self.core, [args[1].address(), dst_bank_id, self.num_tiles])

    return tt_time_execution(lambda: tt_metal.EnqueueProgram(self.dev.cq, self.prog, False), enable=wait)


class TTAllocator(Allocator):
  def __init__(self, dev:TTDevice):
    self.dev = dev
    super().__init__()
  def _alloc(self, size:int, options:BufferSpec):
    # TODO: investigate how big to set the page size
    dram_config = tt_metal.InterleavedBufferConfig(self.dev.dev, size, size, tt_metal.BufferType.DRAM, tt_metal.TensorMemoryLayout.INTERLEAVED)
    return tt_metal.CreateBuffer(dram_config)
  def _free(self, opaque, options:BufferSpec): 
    # TODO: implement
    pass
  def _copyin(self, dest, src:memoryview): 
    tt_metal.EnqueueWriteBuffer(self.dev.cq, dest, self._memview_ptr(src), False)
  def _copyout(self, dest:memoryview, src):
    sync = False # TODO: Does this needs to be sync?
    tt_metal.EnqueueReadBuffer(self.dev.cq, src, self._memview_ptr(dest), sync)
  def _memview_ptr(self, mem:memoryview): return cppyy.ll.reinterpret_cast["void*"](mv_address(mem))


class TTDevice(Compiled):
  def __init__(self, device:str): 
    device_id = int(device.split(":")[1]) if ":" in device else 0
    self.dev = tt_metal.CreateDevice(device_id)
    self.cq = self.dev.command_queue()
    super().__init__(device, TTAllocator(self), TTRenderer(), Compiler(), functools.partial(TTProgram, self))

  def synchronize(self):
    tt_metal.Finish(self.cq)

  # tt_metal.CloseDevice(device)
