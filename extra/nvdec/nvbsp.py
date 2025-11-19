from __future__ import annotations
from hexdump import hexdump
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref
assert sys.platform != 'win32'
from typing import cast, ClassVar
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQProgram, HCQSignal, BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, MOCKGPU, hcq_filter_visible_devices
from tinygrad.uop.ops import sint
from tinygrad.device import BufferSpec, CompilerPairT
from tinygrad.helpers import getenv, mv_address, round_up, data64, data64_le, prod, OSX, to_mv, hi32, lo32, suppress_finalizing
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, PTXCompiler, NVPTXCompiler, NVCompiler
from tinygrad.runtime.support.compiler_mesa import NAKCompiler
from tinygrad.runtime.autogen import nv_570, nv_580, pci, mesa
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import System, PCIIfaceBase, MAP_FIXED
from tinygrad.renderer.nir import NAKRenderer

import ctypes, queue
from tinygrad.runtime.ops_cuda import CUDADevice
from tinygrad.runtime.autogen import cuda, nvcuvid
from tinygrad.helpers import DEBUG, init_c_var, getenv

from PIL import Image
from tinygrad import dtypes
from tinygrad.tensor import Tensor

if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo

import cv2, numpy as np, mmap, os
from tinygrad.device import Device
from pathlib import Path

from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, nv_gpu, NVKIface, nv_iowr, BumpAllocator, GPFifo
from tinygrad.runtime.support.system import System, PCIIfaceBase, MAP_FIXED, HCQBuffer, MMIOInterface, FileIOInterface

class NVVideoQueue(NVCommandQueue):
  def decode_hevc_chunk(self, buf_in_data, luma_chroma_bufs:list,
                        coloc_buf, filter_buf,
                        scaling_list, tile_sizes,
                        status_buf, pic_desc, pic_idx):
    self.nvm(4, nv_gpu.NVC9B0_SET_APPLICATION_ID, 7)
    self.nvm(4, nv_gpu.NVC9B0_SET_CONTROL_PARAMS, 0x52057)
    self.nvm(4, nv_gpu.NVC9B0_SET_DRV_PIC_SETUP_OFFSET, pic_desc.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_IN_BUF_BASE_OFFSET, buf_in_data.va_addr >> 8)
    for i in range(8):
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_LUMA_OFFSET0 + i*4, luma_chroma_bufs[i].va_addr >> 8)
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_CHROMA_OFFSET0 + i*4, (luma_chroma_bufs[i].va_addr + 0x1fe000) >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_COLOC_DATA_OFFSET, coloc_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_NVDEC_STATUS_OFFSET, status_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_INDEX, pic_idx)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_SCALING_LIST_OFFSET, scaling_list.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_TILE_SIZES_OFFSET, tile_sizes.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_FILTER_BUFFER_OFFSET, filter_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_INTRA_TOP_BUF_OFFSET, (filter_buf.va_addr + 0x8d7200) >> 8)
    self.nvm(4, nv_gpu.NVC9B0_EXECUTE, 0)
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.vid_gpfifo)

class FakeNV(HCQCompiled[HCQSignal]):
  def __init__(self):
    self.iface = NVKIface(self, 0)

    self.vmem_alloc = BumpAllocator(size=0x10000000000, base=0x100000000)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())

    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL,
      nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(offset=0, limit=562949953421311, hVASpace=0))

    self.usermode, self.gpu_mmio = self.iface.setup_usermode()

    # self.sysmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_SYSTEM,
    #   nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.virtmem, type=0, flags=114945, attr=713031680, attr2=67108873,
    #                                      format=6, size=163840, alignment=4096, offset=0, limit=163839))

    notifier = self.nv_sys_alloc(163840)
    gpfifo_area = self.nv_sys_alloc(163840)

    self.cmdq_page:HCQBuffer = self.nv_sys_alloc(0x200000)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=cast(int, self.cmdq_page.va_addr), wrap=True)
    self.cmdq = self.cmdq_page.cpu_view().view(fmt='I')

    # self.notifier = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_ALLOC_MEMORY,
    #   nv_gpu.NV_NOTIFIER_ALLOC_PARAMETERS(hObjectError=self.sysmem.meta.hMemory, size=4096, count=1))

    # AMPERE_CHANNEL_GPFIFO_A
    self.vid_gpfifo = self.iface.rm_alloc(self.nvdevice, nv_gpu.AMPERE_CHANNEL_GPFIFO_A,
      nv_gpu.NV_CHANNEL_ALLOC_PARAMS(hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem, gpFifoOffset=gpfifo_area.meta.dmaOffset,
        gpFifoEntries=2048, flags=32, engineType=19,
        hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.meta.hMemory), userdOffset=(ctypes.c_uint64*8)(2048*8+0)))

    self.iface.rm_control(self.vid_gpfifo, nv_gpu.NVA06F_CTRL_CMD_BIND, nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=19))

    self.iface.rm_control(self.vid_gpfifo, nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    ws_token_params = self.iface.rm_control(self.vid_gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))

    self.iface.rm_alloc(self.vid_gpfifo, nv_gpu.NVC9B0_VIDEO_DECODER)
    print("video decoder inited")

    self.vid_gpfifo = GPFifo(ring=gpfifo_area.cpu_view().view(0, 2048*8, fmt='Q'), entries_count=2048, token=ws_token_params.workSubmitToken,
                  controls=nv_gpu.AmpereAControlGPFifo.from_address(gpfifo_area.cpu_view().addr + 0 + 2048 * 8))

    self.arch = "sm_89"
    # compilers:list[CompilerPairT] = [(functools.partial(NVRenderer, self.arch),functools.partial(CUDACompiler if MOCKGPU else NVCompiler, self.arch)),
      # (functools.partial(PTXRenderer, self.arch, device="NV"), functools.partial(PTXCompiler if MOCKGPU else NVPTXCompiler, self.arch))]
    # super().__init__(device, NVAllocator(self), compilers, functools.partial(NVProgram, self), HCQSignal, NVComputeQueue, NVCopyQueue)

    NVVideoQueue().setup(copy_class=nv_gpu.NVC9B0_VIDEO_DECODER).submit(self)
    import time
    time.sleep(2)
    hexdump(to_mv(gpfifo_area.cpu_view().addr + 0 + 2048 * 8, 0x100))
    hexdump(to_mv(notifier.cpu_view().addr + 0, 0x20))

  def nv_sys_alloc(self, size):
    sysmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_SYSTEM,
      nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.virtmem, type=0, flags=114945, attr=713031680, attr2=67108873,
                                         format=6, size=size, alignment=4096, offset=0, limit=size-1))

    # map dma
    made = nv_gpu.NVOS46_PARAMETERS(hClient=self.iface.root, hDevice=self.nvdevice, hDma=self.virtmem, hMemory=sysmem, length=size, flags=272,
      dmaOffset=0)
    nv_iowr(self.iface.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY_DMA, made)
    if made.status != 0: raise RuntimeError(f"nv_sys_alloc 1 returned {get_error_str(made.status)}")
    made_d = made
    
    # map cpu
    fd_dev = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev.fd,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.iface.root, hDevice=self.nvdevice, hMemory=sysmem, length=size, flags=50364416))
    nv_iowr(self.iface.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"nv_sys_alloc 2 returned {get_error_str(made.params.status)}")

    va_base = fd_dev.mmap(0x0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, 0)
    return HCQBuffer(made_d.dmaOffset, size, meta=made_d, view=MMIOInterface(va_base, size, fmt='B'), owner=self)

  # def nv_virt_alloc(self, off, size):
  #   NVKIface.host_object_enumerator += 1

  #   virt = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV50_MEMORY_VIRTUAL,
  #     nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=1886745448, type=6, flags=573440, width=0, height=0, pitch=0, attr=369098752, attr2=5, format=6,
  #      comprCovg=0, zcullCovg=0, rangeLo=0x1000000000, rangeHi=0x1ffffffffffff, size=size, alignment=65536, offset=off, limit=size-1, 
  #      address=None, ctagOffset=0, hVASpace=0, internalflags=0, tag=0, numaNode=0))

  #   made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.iface.root, hObjectParent=self.dev.nvdevice, flags=flags,
  #     hClass=nv_gpu.NV01_MEMORY_SYSTEM, pMemory=0, limit=size-1), fd=self.iface.fd_ctl.fd)
  #   nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)

  #   hex(print(made.pMemory))
    # self.iface.fd_ctl.mmap(0x0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, 0)


if __name__ == "__main__":
  dev = FakeNV()

