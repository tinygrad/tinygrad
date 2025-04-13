from __future__ import annotations
import ctypes
import functools
import platform, subprocess, sys
from tinygrad.helpers import capstone_flatdump, getenv, from_mv, mv_address
from tinygrad.device import Compiled, Compiler, Allocator, BufferSpec, CPUProgram
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.renderer.cstyle import ClangRenderer

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


class ClangJITCompiler(Compiler):
  def __init__(self, cachekey="compile_clang_jit"): super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # -fno-math-errno is required for __builtin_sqrt to become an instruction instead of a function call
    # x18 is a reserved platform register. It is clobbered on context switch in macos and is used to store TEB pointer in windows on arm, don't use it
    target = 'x86_64' if sys.platform == 'win32' else platform.machine()
    args = ['-march=native', f'--target={target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib', '-fno-ident']
    arch_args = ['-ffixed-x18'] if target == 'arm64' else []
    obj = subprocess.check_output([getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
    return jit_loader(obj)

  def disassemble(self, lib:bytes): return capstone_flatdump(lib)

class TTProgram:
  def __init__(self, dev:TTDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib

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
    self.unary_reader_kernel_id = tt_metal.CreateKernel(
        self.prog,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        self.core,
        data_movement_config_R1
    )

    data_movement_config_R0 = tt_metal.DataMovementConfig()
    data_movement_config_R0.processor = tt_metal.DataMovementProcessor.RISCV_0
    data_movement_config_R0.noc = tt_metal.NOC.RISCV_0_default
    self.unary_writer_kernel_id = tt_metal.CreateKernel(
        self.prog,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        self.core,
        data_movement_config_R0
    )


    sfpu_defines = {
        "SFPU_OP_EXP_INCLUDE": "1",
        "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"
    }
    compute_config = tt_metal.ComputeConfig()
    compute_config.math_approx_mode = False
    compute_config.compile_args = [self.num_tiles, 1]
    compute_config.defines = sfpu_defines
    # Use CreateKernelFromString
    self.eltwise_sfpu_kernel_id = tt_metal.CreateKernel(
        self.prog,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
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
    super().__init__(device, TTAllocator(self), ClangRenderer(), ClangJITCompiler(), functools.partial(TTProgram, self))

  def synchronize(self):
    tt_metal.Finish(self.cq)

  # tt_metal.CloseDevice(device)
