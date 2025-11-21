import ctypes, queue
from tinygrad.runtime.ops_cuda import CUDADevice
from tinygrad.runtime.autogen import cuda, nvcuvid
from tinygrad.helpers import DEBUG, init_c_var, getenv

from PIL import Image
from tinygrad import dtypes
from tinygrad.tensor import Tensor

if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo

import cv2, numpy as np
from tinygrad.device import Device
from pathlib import Path

from tinygrad.runtime.ops_nv import NVDevice, NVCommandQueue, nv_gpu

def desc_struct(dev):
  pic1 = dev.allocator.alloc(0x1000, cpu_access=True)
  pic1_desc = nv_gpu.nvdec_hevc_pic_s.from_address(pic1.cpu_view().addr)
  pic1_desc.stream_len = 264596

  pic1_desc.framestride[0] = 1920
  pic1_desc.framestride[1] = 1920

  pic1_desc.colMvBuffersize = 510
  pic1_desc.HevcSaoBufferOffset = 2584
  pic1_desc.HevcBsdCtrlOffset = 23256

  pic1_desc.pic_width_in_luma_samples  = 1920
  pic1_desc.pic_height_in_luma_samples = 1080

  pic1_desc.chroma_format_idc = 1
  pic1_desc.bit_depth_luma = 8
  pic1_desc.bit_depth_chroma = 8

  pic1_desc.log2_min_luma_coding_block_size = 3
  pic1_desc.log2_max_luma_coding_block_size = 6
  pic1_desc.log2_min_transform_block_size   = 2
  pic1_desc.log2_max_transform_block_size   = 5

  pic1_desc.sample_adaptive_offset_enabled_flag = 1
  pic1_desc.sps_temporal_mvp_enabled_flag       = 1
  pic1_desc.strong_intra_smoothing_enabled_flag = 1

  pic1_desc.sign_data_hiding_enabled_flag = 1

  pic1_desc.num_ref_idx_l0_default_active = 1
  pic1_desc.num_ref_idx_l1_default_active = 1

  pic1_desc.init_qp = 26

  pic1_desc.cu_qp_delta_enabled_flag = 1
  pic1_desc.diff_cu_qp_delta_depth   = 1

  pic1_desc.weighted_pred_flag = 1
  pic1_desc.entropy_coding_sync_enabled_flag = 1

  pic1_desc.loop_filter_across_tiles_enabled_flag  = 1
  pic1_desc.loop_filter_across_slices_enabled_flag = 1

  pic1_desc.log2_parallel_merge_level = 2

  pic1_desc.IDR_picture_flag = 1
  pic1_desc.RAP_picture_flag = 1

  pic1_desc.pattern_id = 2

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
  desc.num_bits_short_term_ref_pics_in_slice=10
  desc.v3.slice_ec_slice_type=0

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

if __name__ == "__main__":
  with Path("/home/nimlgen/tinygrad/extra/nvdec/big_buck_bunny.hevc").open("rb") as f:
    file = f.read()

  dev = NVDevice("")

  # copy class to use subc 4.
  NVVideoQueue().wait(dev.timeline_signal, dev.timeline_value - 1) \
                .setup(copy_class=NVC9B0_VIDEO_DECODER) \
                .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()

  # intra bufs struct
  buf_in_data = dev.allocator.alloc(0x3fc000, cpu_access=True)
  luma_chroma_bufs = [dev.allocator.alloc(0x2fd000, cpu_access=True) for i in range(8)]
  coloc_buf = dev.allocator.alloc(0x400000, cpu_access=True)
  filter_buf = dev.allocator.alloc(0xa00000, cpu_access=True)
  scaling_list = dev.allocator.alloc(0x1000, cpu_access=True)
  tile_sizes = dev.allocator.alloc(0x1000, cpu_access=True)
  status_buf = dev.allocator.alloc(0x1000, cpu_access=True)
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
  buf_in_data.cpu_view()[:] = file[0x972:0x972+0x3fc000]

  print("Decoding...")
  NVVideoQueue().wait(dev.timeline_signal, dev.timeline_value - 1) \
                .decode_hevc_chunk(buf_in_data, luma_chroma_bufs, coloc_buf, filter_buf, scaling_list, tile_sizes, status_buf, pic1) \
                .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize()

  print("Decoding done!")
  hexdump(luma_chroma_bufs[0].cpu_view()[:64])
  hexdump(status_buf.cpu_view()[:64])
