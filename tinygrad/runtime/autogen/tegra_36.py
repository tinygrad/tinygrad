# mypy: disable-error-code="empty-body"
# generated from L4T r36.4.4 public_sources.tbz2
#   nvgpu uapi:  nvgpu/include/uapi/linux/{nvgpu.h, nvgpu-ctrl.h, nvgpu-as.h}
#   nvmap uapi:  nvidia-oot/include/uapi/linux/nvmap.h
from __future__ import annotations
import ctypes
from typing import Annotated, Literal
from tinygrad.runtime.support.c import _IO, _IOW, _IOWR
from tinygrad.runtime.support import c

# nvgpu-ctrl.h: struct nvgpu_gpu_characteristics
@c.record
class nvgpu_gpu_characteristics(c.Struct):
  SIZE = 328
  arch: Annotated[Annotated[int, ctypes.c_uint32], 0]
  impl: Annotated[Annotated[int, ctypes.c_uint32], 4]
  rev: Annotated[Annotated[int, ctypes.c_uint32], 8]
  num_gpc: Annotated[Annotated[int, ctypes.c_uint32], 12]
  numa_domain_id: Annotated[Annotated[int, ctypes.c_int32], 16]
  L2_cache_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  on_board_video_memory_size: Annotated[Annotated[int, ctypes.c_uint64], 32]
  num_tpc_per_gpc: Annotated[Annotated[int, ctypes.c_uint32], 40]
  bus_type: Annotated[Annotated[int, ctypes.c_uint32], 44]
  big_page_size: Annotated[Annotated[int, ctypes.c_uint32], 48]
  compression_page_size: Annotated[Annotated[int, ctypes.c_uint32], 52]
  pde_coverage_bit_count: Annotated[Annotated[int, ctypes.c_uint32], 56]
  available_big_page_sizes: Annotated[Annotated[int, ctypes.c_uint32], 60]
  flags: Annotated[Annotated[int, ctypes.c_uint64], 64]
  twod_class: Annotated[Annotated[int, ctypes.c_uint32], 72]
  threed_class: Annotated[Annotated[int, ctypes.c_uint32], 76]
  compute_class: Annotated[Annotated[int, ctypes.c_uint32], 80]
  gpfifo_class: Annotated[Annotated[int, ctypes.c_uint32], 84]
  inline_to_memory_class: Annotated[Annotated[int, ctypes.c_uint32], 88]
  dma_copy_class: Annotated[Annotated[int, ctypes.c_uint32], 92]
  gpc_mask: Annotated[Annotated[int, ctypes.c_uint32], 96]
  sm_arch_sm_version: Annotated[Annotated[int, ctypes.c_uint32], 100]
  sm_arch_spa_version: Annotated[Annotated[int, ctypes.c_uint32], 104]
  sm_arch_warp_count: Annotated[Annotated[int, ctypes.c_uint32], 108]
  gpu_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 112]
  tsg_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 114]
  dbg_gpu_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 116]
  ioctl_channel_nr_last: Annotated[Annotated[int, ctypes.c_int16], 118]
  as_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 120]
  gpu_va_bit_count: Annotated[Annotated[int, ctypes.c_uint8], 122]
  reserved: Annotated[Annotated[int, ctypes.c_uint8], 123]
  max_fbps_count: Annotated[Annotated[int, ctypes.c_uint32], 124]
  fbp_en_mask: Annotated[Annotated[int, ctypes.c_uint32], 128]
  emc_en_mask: Annotated[Annotated[int, ctypes.c_uint32], 132]
  max_ltc_per_fbp: Annotated[Annotated[int, ctypes.c_uint32], 136]
  max_lts_per_ltc: Annotated[Annotated[int, ctypes.c_uint32], 140]
  max_tex_per_tpc: Annotated[Annotated[int, ctypes.c_uint32], 144]
  max_gpc_count: Annotated[Annotated[int, ctypes.c_uint32], 148]
  rop_l2_en_mask_DEPRECATED: Annotated[c.Array[ctypes.c_uint32, Literal[2]], 152]
  chipname: Annotated[c.Array[ctypes.c_uint8, Literal[8]], 160]
  gr_compbit_store_base_hw: Annotated[Annotated[int, ctypes.c_uint64], 168]
  gr_gobs_per_comptagline_per_slice: Annotated[Annotated[int, ctypes.c_uint32], 176]
  num_ltc: Annotated[Annotated[int, ctypes.c_uint32], 180]
  lts_per_ltc: Annotated[Annotated[int, ctypes.c_uint32], 184]
  cbc_cache_line_size: Annotated[Annotated[int, ctypes.c_uint32], 188]
  cbc_comptags_per_line: Annotated[Annotated[int, ctypes.c_uint32], 192]
  map_buffer_batch_limit: Annotated[Annotated[int, ctypes.c_uint32], 196]
  max_freq: Annotated[Annotated[int, ctypes.c_uint64], 200]
  graphics_preemption_mode_flags: Annotated[Annotated[int, ctypes.c_uint32], 208]
  compute_preemption_mode_flags: Annotated[Annotated[int, ctypes.c_uint32], 212]
  default_graphics_preempt_mode: Annotated[Annotated[int, ctypes.c_uint32], 216]
  default_compute_preempt_mode: Annotated[Annotated[int, ctypes.c_uint32], 220]
  local_video_memory_size: Annotated[Annotated[int, ctypes.c_uint64], 224]
  pci_vendor_id: Annotated[Annotated[int, ctypes.c_uint16], 232]
  pci_device_id: Annotated[Annotated[int, ctypes.c_uint16], 234]
  pci_subsystem_vendor_id: Annotated[Annotated[int, ctypes.c_uint16], 236]
  pci_subsystem_device_id: Annotated[Annotated[int, ctypes.c_uint16], 238]
  pci_class: Annotated[Annotated[int, ctypes.c_uint16], 240]
  pci_revision: Annotated[Annotated[int, ctypes.c_uint8], 242]
  vbios_oem_version: Annotated[Annotated[int, ctypes.c_uint8], 243]
  vbios_version: Annotated[Annotated[int, ctypes.c_uint32], 244]
  reg_ops_limit: Annotated[Annotated[int, ctypes.c_uint32], 248]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 252]
  event_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 256]
  pad: Annotated[Annotated[int, ctypes.c_uint16], 258]
  max_css_buffer_size: Annotated[Annotated[int, ctypes.c_uint32], 260]
  ctxsw_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 264]
  prof_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 266]
  nvs_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 268]
  reserved2: Annotated[c.Array[ctypes.c_uint8, Literal[2]], 270]
  max_ctxsw_ring_buffer_size: Annotated[Annotated[int, ctypes.c_uint32], 272]
  reserved3: Annotated[Annotated[int, ctypes.c_uint32], 276]
  per_device_identifier: Annotated[Annotated[int, ctypes.c_uint64], 280]
  num_ppc_per_gpc: Annotated[Annotated[int, ctypes.c_uint32], 288]
  max_veid_count_per_tsg: Annotated[Annotated[int, ctypes.c_uint32], 292]
  num_sub_partition_per_fbpa: Annotated[Annotated[int, ctypes.c_uint32], 296]
  gpu_instance_id: Annotated[Annotated[int, ctypes.c_uint32], 300]
  gr_instance_id: Annotated[Annotated[int, ctypes.c_uint32], 304]
  max_gpfifo_entries: Annotated[Annotated[int, ctypes.c_uint32], 308]
  max_dbg_tsg_timeslice: Annotated[Annotated[int, ctypes.c_uint32], 312]
  reserved5: Annotated[Annotated[int, ctypes.c_uint32], 316]
  device_instance_id: Annotated[Annotated[int, ctypes.c_uint64], 320]
@c.record
class nvgpu_gpu_get_characteristics(c.Struct):
  SIZE = 16
  buf_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class nvgpu_alloc_as_args(c.Struct):
  SIZE = 64
  big_page_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  as_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  va_range_start: Annotated[Annotated[int, ctypes.c_uint64], 16]
  va_range_end: Annotated[Annotated[int, ctypes.c_uint64], 24]
  va_range_split: Annotated[Annotated[int, ctypes.c_uint64], 32]
  padding: Annotated[c.Array[ctypes.c_uint32, Literal[6]], 40]
@c.record
class nvgpu_gpu_open_tsg_args(c.Struct):
  SIZE = 24
  tsg_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  token: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  subctx_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class nvgpu_gpu_open_channel_args(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]

# nvgpu-as.h
@c.record
class nvgpu_as_bind_channel_args(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class nvgpu_as_alloc_space_args(c.Struct):
  SIZE = 32
  pages: Annotated[Annotated[int, ctypes.c_uint64], 0]
  page_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  padding: Annotated[c.Array[ctypes.c_uint32, Literal[2]], 24]
@c.record
class nvgpu_as_map_buffer_ex_args(c.Struct):
  SIZE = 40
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  compr_kind: Annotated[Annotated[int, ctypes.c_int16], 4]
  incompr_kind: Annotated[Annotated[int, ctypes.c_int16], 6]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 8]
  page_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  buffer_offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  mapping_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 32]
@c.record
class nvgpu_as_unmap_buffer_args(c.Struct):
  SIZE = 8
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]

# nvgpu.h: TSG ioctls
@c.record
class nvgpu_tsg_bind_channel_ex_args(c.Struct):
  SIZE = 24
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  subcontext_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  reserved: Annotated[c.Array[ctypes.c_uint8, Literal[16]], 8]
@c.record
class nvgpu_tsg_create_subcontext_args(c.Struct):
  SIZE = 16
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  as_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  veid: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]

# nvgpu.h: channel ioctls
@c.record
class nvgpu_alloc_obj_ctx_args(c.Struct):
  SIZE = 16
  class_num: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  obj_id: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class nvgpu_channel_setup_bind_args(c.Struct):
  SIZE = 104
  num_gpfifo_entries: Annotated[Annotated[int, ctypes.c_uint32], 0]
  num_inflight_jobs: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  userd_dmabuf_fd: Annotated[Annotated[int, ctypes.c_int32], 12]
  gpfifo_dmabuf_fd: Annotated[Annotated[int, ctypes.c_int32], 16]
  work_submit_token: Annotated[Annotated[int, ctypes.c_uint32], 20]
  userd_dmabuf_offset: Annotated[Annotated[int, ctypes.c_uint64], 24]
  gpfifo_dmabuf_offset: Annotated[Annotated[int, ctypes.c_uint64], 32]
  gpfifo_gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 40]
  userd_gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 48]
  usermode_mmio_gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 56]
  reserved: Annotated[c.Array[ctypes.c_uint32, Literal[9]], 64]
@c.record
class nvgpu_channel_wdt_args(c.Struct):
  SIZE = 8
  wdt_status: Annotated[Annotated[int, ctypes.c_uint32], 0]
  timeout_ms: Annotated[Annotated[int, ctypes.c_uint32], 4]

# nvmap.h
@c.record
class nvmap_create_handle(c.Struct):
  SIZE = 8
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class nvmap_alloc_handle(c.Struct):
  SIZE = 20
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  heap_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  align: Annotated[Annotated[int, ctypes.c_uint32], 12]
  numa_nid: Annotated[Annotated[int, ctypes.c_int32], 16]

# ioctl magic bytes
NVGPU_GPU_IOCTL_MAGIC = ord('G')
NVGPU_AS_IOCTL_MAGIC = ord('A')
NVGPU_TSG_IOCTL_MAGIC = ord('T')
NVGPU_IOCTL_MAGIC = ord('H')
NVMAP_IOC_MAGIC = ord('N')

# nvgpu-ctrl.h ioctls
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 5, nvgpu_gpu_get_characteristics)
NVGPU_GPU_IOCTL_ALLOC_AS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 8, nvgpu_alloc_as_args)
NVGPU_GPU_IOCTL_OPEN_TSG = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 9, nvgpu_gpu_open_tsg_args)
NVGPU_GPU_IOCTL_OPEN_CHANNEL = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 11, nvgpu_gpu_open_channel_args)

# nvgpu-as.h ioctls
NVGPU_AS_IOCTL_BIND_CHANNEL = _IOWR(NVGPU_AS_IOCTL_MAGIC, 1, nvgpu_as_bind_channel_args)
NVGPU_AS_IOCTL_UNMAP_BUFFER = _IOWR(NVGPU_AS_IOCTL_MAGIC, 5, nvgpu_as_unmap_buffer_args)
NVGPU_AS_IOCTL_ALLOC_SPACE = _IOWR(NVGPU_AS_IOCTL_MAGIC, 6, nvgpu_as_alloc_space_args)
NVGPU_AS_IOCTL_MAP_BUFFER_EX = _IOWR(NVGPU_AS_IOCTL_MAGIC, 7, nvgpu_as_map_buffer_ex_args)

# nvgpu.h TSG ioctls
NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 11, nvgpu_tsg_bind_channel_ex_args)
NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 18, nvgpu_tsg_create_subcontext_args)

# nvgpu.h channel ioctls
NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _IOWR(NVGPU_IOCTL_MAGIC, 108, nvgpu_alloc_obj_ctx_args)
NVGPU_IOCTL_CHANNEL_WDT = _IOW(NVGPU_IOCTL_MAGIC, 119, nvgpu_channel_wdt_args)
NVGPU_IOCTL_CHANNEL_SETUP_BIND = _IOWR(NVGPU_IOCTL_MAGIC, 128, nvgpu_channel_setup_bind_args)

# nvmap.h ioctls
NVMAP_IOC_CREATE = _IOWR(NVMAP_IOC_MAGIC, 0, nvmap_create_handle)
NVMAP_IOC_ALLOC = _IOW(NVMAP_IOC_MAGIC, 3, nvmap_alloc_handle)
NVMAP_IOC_FREE = _IO(NVMAP_IOC_MAGIC, 4)
NVMAP_IOC_GET_FD = _IOWR(NVMAP_IOC_MAGIC, 15, nvmap_create_handle)

# nvmap.h constants
NVMAP_HEAP_IOVMM = 1 << 30
NVMAP_HANDLE_UNCACHEABLE = 0
NVMAP_HANDLE_WRITE_COMBINE = 1
NVMAP_HANDLE_INNER_CACHEABLE = 2
NVMAP_HANDLE_CACHEABLE = 3

# nvgpu.h channel setup_bind flags
NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC = 1 << 1
NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT = 1 << 3

c.init_records()
