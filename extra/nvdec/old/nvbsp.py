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

import cv2, numpy as np

if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo, dump_struct_ext

import cv2, numpy as np, mmap, os
from tinygrad.device import Device
from pathlib import Path

from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, nv_gpu, NVKIface, nv_iowr, BumpAllocator, GPFifo, HCQSignal, NVProgram, NVComputeQueue, NVCopyQueue
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
    for i in range(3):
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

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_D, 0)
    # self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
    #          (1 << 0) | (1 << 20) | (1 << 24) | (1 << 25)) # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.vid_gpfifo)

class FakeNVAllocator(HCQAllocator['NVDevice']):
  def __init__(self, dev):
    super().__init__(dev, batch_cnt=0)
    self.supports_copy_from_disk = False

  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev.nv_sys_alloc(size)

  def _free(self, opaque:HCQBuffer, options:BufferSpec):
    self.dev.synchronize()

class FakeNV(HCQCompiled[HCQSignal]):
  def __init__(self):
    device = "NV:0"
    self.iface = NVKIface(self, 0)

    # self.vmem_alloc = BumpAllocator(size=0x10000000000, base=0x100000000)

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
    compilers:list[CompilerPairT] = [(functools.partial(NVRenderer, self.arch),functools.partial(CUDACompiler if MOCKGPU else NVCompiler, self.arch)),
      (functools.partial(PTXRenderer, self.arch, device="NV"), functools.partial(PTXCompiler if MOCKGPU else NVPTXCompiler, self.arch))]
    super().__init__(device, FakeNVAllocator(self), compilers, functools.partial(NVProgram, self), HCQSignal, NVComputeQueue, NVCopyQueue)

    NVVideoQueue().setup(copy_class=nv_gpu.NVC9B0_VIDEO_DECODER).signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()

    hexdump(to_mv(gpfifo_area.cpu_view().addr + 0 + 2048 * 8, 0x100))
    hexdump(to_mv(notifier.cpu_view().addr + 0, 0x20))

  def nv_sys_alloc(self, size):
    sysmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_SYSTEM,
      nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.virtmem, type=0, flags=114945, attr=713031680, attr2=67108873,
                                         format=6, size=size, alignment=65536, offset=0, limit=size-1))

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

    va_base = fd_dev.mmap(made_d.dmaOffset, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, 0)
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

def desc_struct(dev):
  pic1 = dev.allocator.alloc(0x1000)
  pic1_desc = nv_gpu.nvdec_hevc_pic_s.from_address(pic1.cpu_view().addr)
  # pic1_desc.stream_len = 264596

  # pic1_desc.gptimer_timeout_value = 81600000
  # pic1_desc.tileformat = 1
  # pic1_desc.sw_start_code_e = 1
  
  pic1_desc.framestride[0] = 1920
  pic1_desc.framestride[1] = 1920

  pic1_desc.colMvBuffersize = 510
  pic1_desc.HevcSaoBufferOffset = 2584
  pic1_desc.HevcBsdCtrlOffset = 23256

  # pic1_desc.pic_width_in_luma_samples  = 1920
  # pic1_desc.pic_height_in_luma_samples = 1080

  # pic1_desc.chroma_format_idc = 1
  # pic1_desc.bit_depth_luma = 8
  # pic1_desc.bit_depth_chroma = 8

  # pic1_desc.log2_min_luma_coding_block_size = 3
  # pic1_desc.log2_max_luma_coding_block_size = 6
  # pic1_desc.log2_min_transform_block_size   = 2
  # pic1_desc.log2_max_transform_block_size   = 5

  # pic1_desc.sample_adaptive_offset_enabled_flag = 1
  # pic1_desc.sps_temporal_mvp_enabled_flag       = 1
  # pic1_desc.strong_intra_smoothing_enabled_flag = 1

  # pic1_desc.sign_data_hiding_enabled_flag = 1

  # pic1_desc.num_ref_idx_l0_default_active = 1
  # pic1_desc.num_ref_idx_l1_default_active = 1

  # pic1_desc.init_qp = 26

  # pic1_desc.cu_qp_delta_enabled_flag = 1
  # pic1_desc.diff_cu_qp_delta_depth   = 1

  # pic1_desc.weighted_pred_flag = 1
  # pic1_desc.entropy_coding_sync_enabled_flag = 1

  # pic1_desc.loop_filter_across_tiles_enabled_flag  = 1
  # pic1_desc.loop_filter_across_slices_enabled_flag = 1

  # pic1_desc.log2_parallel_merge_level = 2

  # pic1_desc.IDR_picture_flag = 1
  # pic1_desc.RAP_picture_flag = 1

  pic1_desc.pattern_id = 2

  # pic1_desc.ecdma_cfg.log2_max_pic_order_cnt_lsb_minus4 = 4

  pic1_desc.v1.hevc_main10_444_ext.log2MaxTransformSkipSize = 2
  pic1_desc.v1.hevc_main10_444_ext.HevcFltAboveOffset = 23902
  pic1_desc.v1.hevc_main10_444_ext.HevcSaoAboveOffset = 32130

  pic1_desc.v3.slice_ec_mv_type     = 1
  pic1_desc.v3.slice_ec_slice_type  = 1
  pic1_desc.v3.HevcSliceEdgeOffset  = 32130
  return pic1, pic1_desc

def patch_pic2(desc):
  desc.stream_len=51302
  desc.num_ref_frames=1
  desc.RefDiffPicOrderCnts[0] = 4
  desc.IDR_picture_flag=0
  desc.RAP_picture_flag=0
  desc.curr_pic_idx=1
  desc.sw_hdr_skip_length=19
  # desc.ecdma_cfg.num_bits_short_term_ref_pics_in_slice=10
  desc.v3.slice_ec_slice_type=0

def patch_pic3(desc):
  desc.stream_len=1910
  desc.num_ref_frames=2
  
  desc.initreflistidxl0=(ctypes.c_ubyte * 16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
  desc.initreflistidxl1=(ctypes.c_ubyte * 16)(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
  desc.RefDiffPicOrderCnts=(ctypes.c_int16 * 16)(2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

  desc.curr_pic_idx=2
  desc.sw_hdr_skip_length=23
  # desc.ecdma_cfg.num_bits_short_term_ref_pics_in_slice=14

def get_tiled_offset(x, y, width):
  GOB_WIDTH = 64
  GOB_HEIGHT = 8
  GOB_SIZE = 512
  BLOCK_HEIGHT_GOBS = 2  

  gob_x = x // GOB_WIDTH
  gob_y = y // GOB_HEIGHT

  super_block_y = gob_y // BLOCK_HEIGHT_GOBS
  gob_y_in_block = gob_y % BLOCK_HEIGHT_GOBS

  stride_gobs = width // GOB_WIDTH

  gob_index = (super_block_y * stride_gobs + gob_x) * BLOCK_HEIGHT_GOBS + gob_y_in_block
  base_offset = gob_index * GOB_SIZE

  lx = x % GOB_WIDTH
  ly = y % GOB_HEIGHT

  swizzle_offset = (
      (lx & 0x0F)       |  # Bits 0-3: Lower 4 bits of X
      ((ly & 0x03) << 4)|  # Bits 4-5: Lower 2 bits of Y
      ((lx & 0x10) << 2)|  # Bit 6:    Bit 4 of X
      ((ly & 0x04) << 5)|  # Bit 7:    Bit 2 of Y
      ((lx & 0x20) << 3)   # Bit 8:    Bit 5 of X
  )

  return base_offset + swizzle_offset

def copyout(luma_chroma_buf):
  chroma_offset=0x1fe000 # 0x220000 # 0x1fe000
  src_pitch=0x780 # 0x800
  fw=0x780
  lh=0x438
  ch=0x21c

  sz = fw * (lh + ch)
  host_buffer = (ctypes.c_ubyte * sz)()
  def copy_2d(height, width, src_offset, dst_offset):
    for y in range(height):
      for x in range(width):
        src_idx = src_offset + get_tiled_offset(x, y, width)
        dst_idx = dst_offset + y * width + x
        host_buffer[dst_idx] = luma_chroma_buf.cpu_view()[src_idx]

  copy_2d(lh, fw, 0, 0) # luma
  copy_2d(ch, fw, chroma_offset, lh * fw) # chroma

  # print("luma")
  # hexdump(bytes(host_buffer)[:lh*fw])
  # print("chroma")
  # hexdump(bytes(host_buffer)[lh*fw:lh*fw+64])

  return bytes(host_buffer)

if __name__ == "__main__":
  dev = FakeNV()

  with Path("/home/nimlgen/tinygrad/extra/nvdec/big_buck_bunny.hevc").open("rb") as f:
    file = f.read()

  # intra bufs struct
  buf_in_data = dev.allocator.alloc(0x3fc000)
  luma_chroma_bufs = [dev.allocator.alloc(0x2fd000) for i in range(3)]
  coloc_buf = dev.allocator.alloc(0x400000)
  filter_buf = dev.allocator.alloc(0xa00000)
  scaling_list = dev.allocator.alloc(0x1000)
  tile_sizes = dev.allocator.alloc(0x1000)
  status_buf = dev.allocator.alloc(0x1000)
  status_desc = nv_gpu.nvdec_status_s.from_address(status_buf.cpu_view().addr)

  # desc struct
  pic1, pic1_desc = desc_struct(dev)

  # init fixed data
  tile_cnt = 63*16
  scaling_list.cpu_view()[:tile_cnt] = bytes([0x10]*tile_cnt)
  scaling_list.cpu_view()[8:16] = bytes([0x0]*8)

  # 1E 00 11 00 00 00 00 00  00 00 00 00 00 00 00 00
  tile_sizes.cpu_view()[:4] = bytes([0x1e,0x0,0x11,0x0])

  # copy in all data
  data = file[0x974:0x974+0x3fc000]
  buf_in_data.cpu_view()[:len(data)] = data

  _dump_gpfifo("before decode", iden=True)
  
  print("Decoding...")
  NVVideoQueue().wait(dev.timeline_signal, dev.timeline_value - 1) \
                .decode_hevc_chunk(buf_in_data, luma_chroma_bufs, coloc_buf, filter_buf, scaling_list, tile_sizes, status_buf, pic1, pic_idx=0) \
                .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()

  _dump_gpfifo("aft decode", iden=True)

  print("Decoding done!")
  hexdump(luma_chroma_bufs[0].cpu_view()[:64])
  dump_struct_ext(status_desc)
  dump_struct_ext(status_desc.hevc)
  dump_struct_ext(pic1_desc)

  patch_pic2(pic1_desc)
  data = file[0x41308:0x41308+0x3fc000]
  buf_in_data.cpu_view()[:len(data)] = data

  NVVideoQueue().wait(dev.timeline_signal, dev.timeline_value - 1) \
                .decode_hevc_chunk(buf_in_data, luma_chroma_bufs, coloc_buf, filter_buf, scaling_list, tile_sizes, status_buf, pic1, pic_idx=1) \
                .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()

  _dump_gpfifo("aft decode", iden=True)

  print("Decoding done!")
  hexdump(luma_chroma_bufs[1].cpu_view()[:64])
  dump_struct_ext(status_desc)
  dump_struct_ext(status_desc.hevc)
  dump_struct_ext(pic1_desc)


  # 0004db60

  patch_pic3(pic1_desc)
  data = file[0x4db6e:0x4db6e+0x3fc000]
  buf_in_data.cpu_view()[:len(data)] = data

  NVVideoQueue().wait(dev.timeline_signal, dev.timeline_value - 1) \
                .decode_hevc_chunk(buf_in_data, luma_chroma_bufs, coloc_buf, filter_buf, scaling_list, tile_sizes, status_buf, pic1, pic_idx=2) \
                .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()

  _dump_gpfifo("aft decode", iden=True)

  print("Decoding done!")
  hexdump(luma_chroma_bufs[2].cpu_view()[:64])
  dump_struct_ext(status_desc)
  dump_struct_ext(status_desc.hevc)
  dump_struct_ext(pic1_desc)

  hexdump(luma_chroma_bufs[0].cpu_view()[:64])

  for i in range(1):
    img1 = copyout(luma_chroma_bufs[i])
    w, h, ch = 1920, 1080, 540
    total = h + ch
    frame = np.frombuffer(img1, dtype=np.uint8).reshape((total, w))
    cv2.imwrite(f"extra/nvdec/out/nv_frame_{i}.png", cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12))
