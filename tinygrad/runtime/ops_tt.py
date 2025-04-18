from __future__ import annotations
import functools
from tinygrad.helpers import mv_address
from tinygrad.device import Compiled, Compiler, Allocator, BufferSpec, CPUProgram
from tinygrad.ops import Ops, UOp
from tinygrad.renderer import Renderer
import math

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

TILE_SIZE = 32 * 32 # 32x32 Tile

def tt_time_execution(cb, enable=False) -> float|None:
  if not enable: return cb()
  cb()
  return 0 * 1e-3

def find_nodes(ast:UOp, op:Ops, acc: list[UOp]) -> list[UOp]:
  if ast.op == op:
    acc.append(ast)
  for src in ast.src:
    find_nodes(src, op, acc)
  return acc


class TTRenderer(Renderer):
  device = "TT"
  supports_float4 = False
  has_local = False
  global_max = None


  def reader_builder(self, loads, num_tiles) -> str:
    def bl(builder): return '\n'.join([builder(i) for i in range(len(loads))])
    return f"""
#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {{
{bl(lambda i: f'''
constexpr uint32_t cb_id_{i} = {loads[i].src[0].arg};
uint32_t src_addr_{i}  = get_arg_val<uint32_t>({2*i});
uint32_t bank_id_{i} = get_arg_val<uint32_t>({2*i+1});
''')}

constexpr uint32_t ublock_size_tiles = 1; // ublocks size defined in tiles
{bl(lambda i: f'uint32_t ublock_size_bytes_{i} = get_tile_size(cb_id_{i}) * ublock_size_tiles;')}

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
// TODO: Dynamic sizes and strides
for (uint32_t i = 0; i < {num_tiles}; i += ublock_size_tiles) {{
{bl(lambda i: f'cb_reserve_back(cb_id_{i}, ublock_size_tiles);')}
{bl(lambda i: f'uint64_t src_noc_addr_{i} = get_noc_addr_from_bank_id<true>(bank_id_{i}, src_addr_{i});')}

{bl(lambda i: f'noc_async_read(src_noc_addr_{i}, get_write_ptr(cb_id_{i}), ublock_size_bytes_{i});')}

noc_async_read_barrier();

{bl(lambda i: f'cb_push_back(cb_id_{i}, ublock_size_tiles);')}
{bl(lambda i: f'src_addr_{i} += ublock_size_bytes_{i};')}
}}
}}
"""

  def writer_builder(self, num_tiles) -> str:
    return f"""
#include "dataflow_api.h"

void kernel_main() {{
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
  
    constexpr uint32_t cb_id_out0 = 0;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < {num_tiles}; i += ublock_size_tiles) {{
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);

        cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }}
}}
"""

  code_for_op: dict = {
    Ops.SQRT: "sqrt_tile", Ops.RECIP: "recip_tile", Ops.NEG: "negative_tile",
    Ops.EXP2: "", Ops.LOG2: "", Ops.SIN: "",
    Ops.AND: "bitwise_and_tile", Ops.XOR: "bitwise_xor_tile", Ops.OR: "bitwise_or_tile",
    Ops.ADD: "add_tiles", Ops.SUB: "", Ops.MUL: "mul_tiles",
    Ops.MOD: "", Ops.IDIV: "", Ops.CMPNE: "",
    Ops.SHR: "right_shift_tile", Ops.SHL: "left_shift_tile", Ops.CMPLT: "",
    Ops.WHERE: "" }


  def render_ast(self, ast:UOp) -> str:
    loads = find_nodes(ast, Ops.LOAD, [])
    stores = find_nodes(ast, Ops.STORE, [])


    # TODO: support different buffer lengths
    # TODO: what happens on non divisible by TILE_SIZE?
    num_tiles = math.ceil(loads[0].size / TILE_SIZE)

    binary_op = stores[0].src[2]

    compute = f"""
#include "compute_kernel_api/eltwise_binary.h"
#include <cstdint>
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {{
void MAIN {{
    uint32_t per_core_block_cnt = {num_tiles};
    uint32_t per_core_block_size = 1;

    constexpr auto cb_inp0 = tt::CBIndex::c_1;
    constexpr auto cb_inp1 = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_0;

    binary_op_init_common(cb_inp0, cb_inp1, cb_out0);

    {self.code_for_op[binary_op.op]}_init(cb_inp0, cb_inp1);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {{
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();

        for (uint32_t i = 0; i < per_core_block_size; ++i) {{
            {self.code_for_op[binary_op.op]}(cb_inp0, cb_inp1, i, i, i);
        }}
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {{
            pack_tile(i, cb_out0);
        }}
        tile_regs_release();

        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }}
}}
}}
"""

    reader = self.reader_builder(loads, num_tiles)
    writer = self.writer_builder(num_tiles)

    cbs_read = [f"{op.op}:{op.src[0].arg}:{op.dtype.name}:{op.dtype.itemsize}" for op in loads]
    cbs_write = [f"{op.op}:{op.src[0].arg}:{op.src[2].dtype.name}:{op.src[2].dtype.itemsize}" for op in stores]
    cbs = ",".join(cbs_write + cbs_read)

    return "[SEP]".join([reader, writer, compute, cbs])

class TTProgram:
  def __init__(self, dev:TTDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib

    reader_src, writer_src, compute_src, cbs = lib.decode("utf-8").split("[SEP]")

    self.prog = tt_metal.CreateProgram()
    self.core = tt.umd.xy_pair(0, 0)

    buffer_tile_count = 2 # 2 Amount of tiles the buffer can hold, with more tiles compute and data movement can work more independently
    cb_dtype_map = {"float": tt.DataFormat.Float32}
    self.cbs = cbs.split(",")
    for cb in self.cbs:
      op_type, cb_id, cb_dtype, cb_itemsize = cb.split(":")

      tile_size = TILE_SIZE * int(cb_itemsize)
      cb_config = tt_metal.CircularBufferConfig(buffer_tile_count * tile_size, {int(cb_id): cb_dtype_map[cb_dtype]}) \
          .set_page_size(int(cb_id), tile_size)
      cb_src = tt_metal.CreateCircularBuffer(self.prog, self.core, cb_config)

    data_movement_config_R1 = tt_metal.DataMovementConfig()
    data_movement_config_R1.processor = tt_metal.DataMovementProcessor.RISCV_1
    data_movement_config_R1.noc = tt_metal.NOC.RISCV_1_default
    self.unary_reader_kernel_id = tt_metal.CreateKernelFromString(self.prog, reader_src, self.core, data_movement_config_R1)

    data_movement_config_R0 = tt_metal.DataMovementConfig()
    data_movement_config_R0.processor = tt_metal.DataMovementProcessor.RISCV_0
    data_movement_config_R0.noc = tt_metal.NOC.RISCV_0_default
    self.unary_writer_kernel_id = tt_metal.CreateKernelFromString(self.prog, writer_src, self.core, data_movement_config_R0)


    compute_config = tt_metal.ComputeConfig()
    compute_config.math_approx_mode = False
    compute_config.fp32_dest_acc_en = True
    compute_config.math_fidelity = gbl.MathFidelity.HiFi4
    self.eltwise_sfpu_kernel_id = tt_metal.CreateKernelFromString(self.prog, compute_src, self.core, compute_config)


  def __del__(self):
    pass

  def __call__(self, *bufs, vals=(), wait=False):
    # args = list(bufs) + list(vals)
    src_bank_id, dst_bank_id = 0, 0

    reader_args = []
    for i in range(1, len(bufs)): reader_args.extend([bufs[i].address(), src_bank_id])
    tt_metal.SetRuntimeArgs(self.prog, self.unary_reader_kernel_id, self.core, reader_args)
    tt_metal.SetRuntimeArgs(self.prog, self.unary_writer_kernel_id, self.core, [bufs[0].address(), dst_bank_id])

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
    tt_metal.EnqueueReadBuffer(self.dev.cq, src, self._memview_ptr(dest), False)
    # TODO: Why do we need sync here?
    self.dev.synchronize()
  def _memview_ptr(self, mem:memoryview): return cppyy.ll.reinterpret_cast["void*"](mv_address(mem))


class TTDevice(Compiled):
  def __init__(self, device:str): 
    device_id = int(device.split(":")[1]) if ":" in device else 0
    self.dev = tt_metal.CreateDevice(device_id)
    self.cq = self.dev.command_queue()
    super().__init__(device, TTAllocator(self), TTRenderer(), Compiler(), functools.partial(TTProgram, self))

  def synchronize(self):
    tt_metal.Finish(self.cq)

  def finalize(self):
    tt_metal.CloseDevice(self.dev)
