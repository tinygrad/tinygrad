# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_nvgpu_tsg_bind_channel_ex_args(c.Struct):
  SIZE = 24
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  subcontext_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 8]
@c.record
class struct_nvgpu_tsg_bind_scheduling_domain_args(c.Struct):
  SIZE = 32
  domain_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  reserved0: Annotated[Annotated[int, ctypes.c_int32], 4]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[3]], 8]
@c.record
class struct_nvgpu_tsg_sm_error_state_record(c.Struct):
  SIZE = 24
  global_esr: Annotated[Annotated[int, ctypes.c_uint32], 0]
  warp_esr: Annotated[Annotated[int, ctypes.c_uint32], 4]
  warp_esr_pc: Annotated[Annotated[int, ctypes.c_uint64], 8]
  global_esr_report_mask: Annotated[Annotated[int, ctypes.c_uint32], 16]
  warp_esr_report_mask: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_nvgpu_tsg_read_single_sm_error_state_args(c.Struct):
  SIZE = 24
  sm_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  record_mem: Annotated[Annotated[int, ctypes.c_uint64], 8]
  record_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_tsg_read_all_sm_error_state_args(c.Struct):
  SIZE = 24
  num_sm: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  buffer_mem: Annotated[Annotated[int, ctypes.c_uint64], 8]
  buffer_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_tsg_l2_max_ways_evict_last_args(c.Struct):
  SIZE = 8
  max_ways: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_tsg_set_l2_sector_promotion_args(c.Struct):
  SIZE = 8
  promotion_flag: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_tsg_create_subcontext_args(c.Struct):
  SIZE = 16
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  as_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  veid: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_tsg_delete_subcontext_args(c.Struct):
  SIZE = 8
  veid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_tsg_get_share_token_args(c.Struct):
  SIZE = 24
  source_device_instance_id: Annotated[Annotated[int, ctypes.c_uint64], 0]
  target_device_instance_id: Annotated[Annotated[int, ctypes.c_uint64], 8]
  share_token: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_tsg_revoke_share_token_args(c.Struct):
  SIZE = 24
  source_device_instance_id: Annotated[Annotated[int, ctypes.c_uint64], 0]
  target_device_instance_id: Annotated[Annotated[int, ctypes.c_uint64], 8]
  share_token: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_dbg_gpu_bind_channel_args(c.Struct):
  SIZE = 8
  channel_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad0: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[1]], 4]
@c.record
class struct_nvgpu_dbg_gpu_reg_op(c.Struct):
  SIZE = 32
  op: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  type: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  status: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  quad: Annotated[Annotated[int, ctypes.c_ubyte], 3]
  group_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sub_group_mask: Annotated[Annotated[int, ctypes.c_uint32], 8]
  offset: Annotated[Annotated[int, ctypes.c_uint32], 12]
  value_lo: Annotated[Annotated[int, ctypes.c_uint32], 16]
  value_hi: Annotated[Annotated[int, ctypes.c_uint32], 20]
  and_n_mask_lo: Annotated[Annotated[int, ctypes.c_uint32], 24]
  and_n_mask_hi: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_nvgpu_dbg_gpu_exec_reg_ops_args(c.Struct):
  SIZE = 16
  ops: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_ops: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gr_ctx_resident: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_dbg_gpu_events_ctrl_args(c.Struct):
  SIZE = 8
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad0: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[1]], 4]
@c.record
class struct_nvgpu_dbg_gpu_powergate_args(c.Struct):
  SIZE = 4
  mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_dbg_gpu_smpc_ctxsw_mode_args(c.Struct):
  SIZE = 4
  mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_dbg_gpu_suspend_resume_all_sms_args(c.Struct):
  SIZE = 4
  mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_dbg_gpu_perfbuf_map_args(c.Struct):
  SIZE = 24
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  mapping_size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_dbg_gpu_perfbuf_unmap_args(c.Struct):
  SIZE = 8
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_dbg_gpu_pc_sampling_args(c.Struct):
  SIZE = 8
  enable: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad0: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[1]], 4]
@c.record
class struct_nvgpu_dbg_gpu_timeout_args(c.Struct):
  SIZE = 8
  enable: Annotated[Annotated[int, ctypes.c_uint32], 0]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_set_next_stop_trigger_type_args(c.Struct):
  SIZE = 8
  broadcast: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_hwpm_ctxsw_mode_args(c.Struct):
  SIZE = 8
  mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_sm_error_state_record(c.Struct):
  SIZE = 24
  hww_global_esr: Annotated[Annotated[int, ctypes.c_uint32], 0]
  hww_warp_esr: Annotated[Annotated[int, ctypes.c_uint32], 4]
  hww_warp_esr_pc: Annotated[Annotated[int, ctypes.c_uint64], 8]
  hww_global_esr_report_mask: Annotated[Annotated[int, ctypes.c_uint32], 16]
  hww_warp_esr_report_mask: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_nvgpu_dbg_gpu_read_single_sm_error_state_args(c.Struct):
  SIZE = 24
  sm_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sm_error_state_record_mem: Annotated[Annotated[int, ctypes.c_uint64], 8]
  sm_error_state_record_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_dbg_gpu_clear_single_sm_error_state_args(c.Struct):
  SIZE = 8
  sm_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_unbind_channel_args(c.Struct):
  SIZE = 8
  channel_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad0: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[1]], 4]
@c.record
class struct_nvgpu_dbg_gpu_suspend_resume_contexts_args(c.Struct):
  SIZE = 16
  action: Annotated[Annotated[int, ctypes.c_uint32], 0]
  is_resident_context: Annotated[Annotated[int, ctypes.c_uint32], 4]
  resident_context_fd: Annotated[Annotated[int, ctypes.c_int32], 8]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_dbg_gpu_access_fb_memory_args(c.Struct):
  SIZE = 32
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  buffer: Annotated[Annotated[int, ctypes.c_uint64], 16]
  size: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_nvgpu_dbg_gpu_profiler_obj_mgt_args(c.Struct):
  SIZE = 8
  profiler_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_profiler_reserve_args(c.Struct):
  SIZE = 8
  profiler_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  acquire: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_set_sm_exception_type_mask_args(c.Struct):
  SIZE = 8
  exception_type_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_cycle_stats_args(c.Struct):
  SIZE = 8
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_cycle_stats_snapshot_args(c.Struct):
  SIZE = 16
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 4]
  extra: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_dbg_gpu_set_ctx_mmu_debug_mode_args(c.Struct):
  SIZE = 8
  mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_get_gr_context_size_args(c.Struct):
  SIZE = 8
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_dbg_gpu_get_gr_context_args(c.Struct):
  SIZE = 16
  buffer: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_dbg_gpu_get_mappings_entry(c.Struct):
  SIZE = 16
  gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_dbg_gpu_get_mappings_args(c.Struct):
  SIZE = 32
  va_lo: Annotated[Annotated[int, ctypes.c_uint64], 0]
  va_hi: Annotated[Annotated[int, ctypes.c_uint64], 8]
  ops_buffer: Annotated[Annotated[int, ctypes.c_uint64], 16]
  count: Annotated[Annotated[int, ctypes.c_uint32], 24]
  has_more: Annotated[Annotated[int, ctypes.c_ubyte], 28]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
@c.record
class struct_nvgpu_dbg_gpu_va_access_entry(c.Struct):
  SIZE = 24
  gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  data: Annotated[Annotated[int, ctypes.c_uint64], 8]
  size: Annotated[Annotated[int, ctypes.c_uint32], 16]
  valid: Annotated[Annotated[int, ctypes.c_ubyte], 20]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 21]
@c.record
class struct_nvgpu_dbg_gpu_va_access_args(c.Struct):
  SIZE = 16
  ops_buf: Annotated[Annotated[int, ctypes.c_uint64], 0]
  count: Annotated[Annotated[int, ctypes.c_uint32], 8]
  cmd: Annotated[Annotated[int, ctypes.c_ubyte], 12]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 13]
@c.record
class struct_nvgpu_sched_exit_wait_for_errbar_args(c.Struct):
  SIZE = 4
  enable: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_profiler_bind_context_args(c.Struct):
  SIZE = 8
  tsg_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_profiler_reserve_pm_resource_args(c.Struct):
  SIZE = 16
  resource: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 8]
@c.record
class struct_nvgpu_profiler_release_pm_resource_args(c.Struct):
  SIZE = 8
  resource: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_profiler_alloc_pma_stream_args(c.Struct):
  SIZE = 48
  pma_buffer_map_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  pma_buffer_offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pma_buffer_va: Annotated[Annotated[int, ctypes.c_uint64], 16]
  pma_buffer_fd: Annotated[Annotated[int, ctypes.c_int32], 24]
  pma_bytes_available_buffer_fd: Annotated[Annotated[int, ctypes.c_int32], 28]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  pma_channel_id: Annotated[Annotated[int, ctypes.c_uint32], 36]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 40]
@c.record
class struct_nvgpu_profiler_free_pma_stream_args(c.Struct):
  SIZE = 12
  pma_channel_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 4]
@c.record
class struct_nvgpu_profiler_pma_stream_update_get_put_args(c.Struct):
  SIZE = 40
  bytes_consumed: Annotated[Annotated[int, ctypes.c_uint64], 0]
  bytes_available: Annotated[Annotated[int, ctypes.c_uint64], 8]
  put_ptr: Annotated[Annotated[int, ctypes.c_uint64], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pma_channel_id: Annotated[Annotated[int, ctypes.c_uint32], 28]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 32]
@c.record
class struct_nvgpu_profiler_reg_op(c.Struct):
  SIZE = 24
  op: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  status: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  offset: Annotated[Annotated[int, ctypes.c_uint32], 4]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
  and_n_mask: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_profiler_exec_reg_ops_args(c.Struct):
  SIZE = 32
  mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  count: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ops: Annotated[Annotated[int, ctypes.c_uint64], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 20]
@c.record
class struct_nvgpu_profiler_vab_range_checker(c.Struct):
  SIZE = 16
  start_phys_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  granularity_shift: Annotated[Annotated[int, ctypes.c_ubyte], 8]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 9]
@c.record
class struct_nvgpu_profiler_vab_reserve_args(c.Struct):
  SIZE = 16
  vab_mode: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1]
  num_range_checkers: Annotated[Annotated[int, ctypes.c_uint32], 4]
  range_checkers_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_profiler_vab_flush_state_args(c.Struct):
  SIZE = 16
  buffer_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  buffer_size: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpfifo(c.Struct):
  SIZE = 8
  entry0: Annotated[Annotated[int, ctypes.c_uint32], 0]
  entry1: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_get_param_args(c.Struct):
  SIZE = 4
  value: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_channel_open_args(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  _in: Annotated[struct_nvgpu_channel_open_args_in, 0]
  out: Annotated[struct_nvgpu_channel_open_args_out, 0]
@c.record
class struct_nvgpu_channel_open_args_in(c.Struct):
  SIZE = 4
  runlist_id: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_nvgpu_channel_open_args_out(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_nvgpu_set_nvmap_fd_args(c.Struct):
  SIZE = 4
  fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_alloc_obj_ctx_args(c.Struct):
  SIZE = 16
  class_num: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  obj_id: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_alloc_gpfifo_ex_args(c.Struct):
  SIZE = 32
  num_entries: Annotated[Annotated[int, ctypes.c_uint32], 0]
  num_inflight_jobs: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[5]], 12]
@c.record
class struct_nvgpu_channel_setup_bind_args(c.Struct):
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
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[9]], 64]
@c.record
class struct_nvgpu_fence(c.Struct):
  SIZE = 8
  id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  value: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_submit_gpfifo_args(c.Struct):
  SIZE = 24
  gpfifo: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_entries: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  fence: Annotated[struct_nvgpu_fence, 16]
@c.record
class struct_nvgpu_wait_args(c.Struct):
  SIZE = 24
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  timeout: Annotated[Annotated[int, ctypes.c_uint32], 4]
  condition: Annotated[struct_nvgpu_wait_args_condition, 8]
@c.record
class struct_nvgpu_wait_args_condition(c.Struct):
  SIZE = 16
  notifier: Annotated[struct_nvgpu_wait_args_condition_notifier, 0]
  semaphore: Annotated[struct_nvgpu_wait_args_condition_semaphore, 0]
@c.record
class struct_nvgpu_wait_args_condition_notifier(c.Struct):
  SIZE = 16
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint32], 4]
  padding1: Annotated[Annotated[int, ctypes.c_uint32], 8]
  padding2: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_wait_args_condition_semaphore(c.Struct):
  SIZE = 16
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint32], 4]
  payload: Annotated[Annotated[int, ctypes.c_uint32], 8]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_set_timeout_args(c.Struct):
  SIZE = 4
  timeout: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_set_timeout_ex_args(c.Struct):
  SIZE = 8
  timeout: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_zcull_bind_args(c.Struct):
  SIZE = 16
  gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  mode: Annotated[Annotated[int, ctypes.c_uint32], 8]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_set_error_notifier(c.Struct):
  SIZE = 24
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  mem: Annotated[Annotated[int, ctypes.c_uint32], 16]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_nvgpu_notification(c.Struct):
  SIZE = 16
  time_stamp: Annotated[struct_nvgpu_notification_time_stamp, 0]
  info32: Annotated[Annotated[int, ctypes.c_uint32], 8]
  info16: Annotated[Annotated[int, ctypes.c_uint16], 12]
  status: Annotated[Annotated[int, ctypes.c_uint16], 14]
@c.record
class struct_nvgpu_notification_time_stamp(c.Struct):
  SIZE = 8
  nanoseconds: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 0]
@c.record
class struct_nvgpu_channel_wdt_args(c.Struct):
  SIZE = 8
  wdt_status: Annotated[Annotated[int, ctypes.c_uint32], 0]
  timeout_ms: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_runlist_interleave_args(c.Struct):
  SIZE = 8
  level: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_timeslice_args(c.Struct):
  SIZE = 8
  timeslice_us: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_event_id_ctrl_args(c.Struct):
  SIZE = 16
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  event_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  event_fd: Annotated[Annotated[int, ctypes.c_int32], 8]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_preemption_mode_args(c.Struct):
  SIZE = 8
  graphics_preempt_mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  compute_preempt_mode: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_boosted_ctx_args(c.Struct):
  SIZE = 8
  boost: Annotated[Annotated[int, ctypes.c_uint32], 0]
  padding: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_get_user_syncpoint_args(c.Struct):
  SIZE = 16
  gpu_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  syncpoint_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  syncpoint_max: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_reschedule_runlist_args(c.Struct):
  SIZE = 4
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_ctxsw_trace_entry(c.Struct):
  SIZE = 24
  tag: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  vmid: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  seqno: Annotated[Annotated[int, ctypes.c_uint16], 2]
  context_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  pid: Annotated[Annotated[int, ctypes.c_uint64], 8]
  timestamp: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_ctxsw_ring_header(c.Struct):
  SIZE = 32
  magic: Annotated[Annotated[int, ctypes.c_uint32], 0]
  version: Annotated[Annotated[int, ctypes.c_uint32], 4]
  num_ents: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ent_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  drop_count: Annotated[Annotated[int, ctypes.c_uint32], 16]
  write_seqno: Annotated[Annotated[int, ctypes.c_uint32], 20]
  write_idx: Annotated[Annotated[int, ctypes.c_uint32], 24]
  read_idx: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_nvgpu_ctxsw_ring_setup_args(c.Struct):
  SIZE = 4
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_ctxsw_trace_filter(c.Struct):
  SIZE = 32
  tag_bits: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[4]], 0]
@c.record
class struct_nvgpu_ctxsw_trace_filter_args(c.Struct):
  SIZE = 32
  filter: Annotated[struct_nvgpu_ctxsw_trace_filter, 0]
@c.record
class struct_nvgpu_sched_get_tsgs_args(c.Struct):
  SIZE = 16
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  buffer: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_sched_get_tsgs_by_pid_args(c.Struct):
  SIZE = 24
  pid: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  buffer: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_sched_tsg_get_params_args(c.Struct):
  SIZE = 32
  tsgid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  timeslice: Annotated[Annotated[int, ctypes.c_uint32], 4]
  runlist_interleave: Annotated[Annotated[int, ctypes.c_uint32], 8]
  graphics_preempt_mode: Annotated[Annotated[int, ctypes.c_uint32], 12]
  compute_preempt_mode: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pid: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_nvgpu_sched_tsg_timeslice_args(c.Struct):
  SIZE = 8
  tsgid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  timeslice: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_sched_tsg_runlist_interleave_args(c.Struct):
  SIZE = 8
  tsgid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  runlist_interleave: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_sched_api_version_args(c.Struct):
  SIZE = 4
  version: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_sched_tsg_refcount_args(c.Struct):
  SIZE = 4
  tsgid: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_sched_event_arg(c.Struct):
  SIZE = 16
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 0]
  status: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_zcull_get_ctx_size_args(c.Struct):
  SIZE = 4
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_gpu_zcull_get_info_args(c.Struct):
  SIZE = 40
  width_align_pixels: Annotated[Annotated[int, ctypes.c_uint32], 0]
  height_align_pixels: Annotated[Annotated[int, ctypes.c_uint32], 4]
  pixel_squares_by_aliquots: Annotated[Annotated[int, ctypes.c_uint32], 8]
  aliquot_total: Annotated[Annotated[int, ctypes.c_uint32], 12]
  region_byte_multiplier: Annotated[Annotated[int, ctypes.c_uint32], 16]
  region_header_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
  subregion_header_size: Annotated[Annotated[int, ctypes.c_uint32], 24]
  subregion_width_align_pixels: Annotated[Annotated[int, ctypes.c_uint32], 28]
  subregion_height_align_pixels: Annotated[Annotated[int, ctypes.c_uint32], 32]
  subregion_count: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_nvgpu_gpu_zbc_set_table_args(c.Struct):
  SIZE = 48
  color_ds: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 0]
  color_l2: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 16]
  depth: Annotated[Annotated[int, ctypes.c_uint32], 32]
  stencil: Annotated[Annotated[int, ctypes.c_uint32], 36]
  format: Annotated[Annotated[int, ctypes.c_uint32], 40]
  type: Annotated[Annotated[int, ctypes.c_uint32], 44]
@c.record
class struct_nvgpu_gpu_zbc_query_table_args(c.Struct):
  SIZE = 56
  color_ds: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 0]
  color_l2: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 16]
  depth: Annotated[Annotated[int, ctypes.c_uint32], 32]
  stencil: Annotated[Annotated[int, ctypes.c_uint32], 36]
  ref_cnt: Annotated[Annotated[int, ctypes.c_uint32], 40]
  format: Annotated[Annotated[int, ctypes.c_uint32], 44]
  type: Annotated[Annotated[int, ctypes.c_uint32], 48]
  index_size: Annotated[Annotated[int, ctypes.c_uint32], 52]
@c.record
class struct_nvgpu_gpu_characteristics(c.Struct):
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
  gpu_va_bit_count: Annotated[Annotated[int, ctypes.c_ubyte], 122]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 123]
  max_fbps_count: Annotated[Annotated[int, ctypes.c_uint32], 124]
  fbp_en_mask: Annotated[Annotated[int, ctypes.c_uint32], 128]
  emc_en_mask: Annotated[Annotated[int, ctypes.c_uint32], 132]
  max_ltc_per_fbp: Annotated[Annotated[int, ctypes.c_uint32], 136]
  max_lts_per_ltc: Annotated[Annotated[int, ctypes.c_uint32], 140]
  max_tex_per_tpc: Annotated[Annotated[int, ctypes.c_uint32], 144]
  max_gpc_count: Annotated[Annotated[int, ctypes.c_uint32], 148]
  rop_l2_en_mask_DEPRECATED: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 152]
  chipname: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
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
  pci_revision: Annotated[Annotated[int, ctypes.c_ubyte], 242]
  vbios_oem_version: Annotated[Annotated[int, ctypes.c_ubyte], 243]
  vbios_version: Annotated[Annotated[int, ctypes.c_uint32], 244]
  reg_ops_limit: Annotated[Annotated[int, ctypes.c_uint32], 248]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 252]
  event_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 256]
  pad: Annotated[Annotated[int, ctypes.c_uint16], 258]
  max_css_buffer_size: Annotated[Annotated[int, ctypes.c_uint32], 260]
  ctxsw_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 264]
  prof_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 266]
  nvs_ioctl_nr_last: Annotated[Annotated[int, ctypes.c_int16], 268]
  reserved2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 270]
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
class struct_nvgpu_gpu_get_characteristics(c.Struct):
  SIZE = 16
  gpu_characteristics_buf_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_characteristics_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_prepare_compressible_read_args(c.Struct):
  SIZE = 80
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  request_compbits: Annotated[Annotated[int, ctypes.c_uint32], 4]
  valid_compbits: Annotated[Annotated[int, ctypes.c_uint32], 4]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  compbits_hoffset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  compbits_voffset: Annotated[Annotated[int, ctypes.c_uint64], 24]
  width: Annotated[Annotated[int, ctypes.c_uint32], 32]
  height: Annotated[Annotated[int, ctypes.c_uint32], 36]
  block_height_log2: Annotated[Annotated[int, ctypes.c_uint32], 40]
  submit_flags: Annotated[Annotated[int, ctypes.c_uint32], 44]
  fence: Annotated[struct_nvgpu_gpu_prepare_compressible_read_args_fence, 48]
  zbc_color: Annotated[Annotated[int, ctypes.c_uint32], 56]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 60]
  scatterbuffer_offset: Annotated[Annotated[int, ctypes.c_uint64], 64]
  reserved2: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 72]
@c.record
class struct_nvgpu_gpu_prepare_compressible_read_args_fence(c.Struct):
  SIZE = 8
  syncpt_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  syncpt_value: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_nvgpu_gpu_mark_compressible_write_args(c.Struct):
  SIZE = 32
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  valid_compbits: Annotated[Annotated[int, ctypes.c_uint32], 4]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  zbc_color: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 20]
@c.record
class struct_nvgpu_alloc_as_args(c.Struct):
  SIZE = 64
  big_page_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  as_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  va_range_start: Annotated[Annotated[int, ctypes.c_uint64], 16]
  va_range_end: Annotated[Annotated[int, ctypes.c_uint64], 24]
  va_range_split: Annotated[Annotated[int, ctypes.c_uint64], 32]
  padding: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[6]], 40]
@c.record
class struct_nvgpu_gpu_open_tsg_args(c.Struct):
  SIZE = 24
  tsg_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  source_device_instance_id: Annotated[Annotated[int, ctypes.c_uint64], 8]
  share_token: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_gpu_get_tpc_masks_args(c.Struct):
  SIZE = 16
  mask_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  mask_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_get_gpc_physical_map_args(c.Struct):
  SIZE = 16
  map_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  physical_gpc_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_get_gpc_logical_map_args(c.Struct):
  SIZE = 16
  map_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  logical_gpc_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_open_channel_args(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  _in: Annotated[struct_nvgpu_gpu_open_channel_args_in, 0]
  out: Annotated[struct_nvgpu_gpu_open_channel_args_out, 0]
@c.record
class struct_nvgpu_gpu_open_channel_args_in(c.Struct):
  SIZE = 4
  runlist_id: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_nvgpu_gpu_open_channel_args_out(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_nvgpu_gpu_l2_fb_args(c.Struct):
  SIZE = 5
  l2_flush: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 0]
  l2_invalidate: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 1]
  fb_flush: Annotated[Annotated[int, ctypes.c_uint32], 0, 1, 2]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 1]
@c.record
class struct_nvgpu_gpu_mmu_debug_mode_args(c.Struct):
  SIZE = 8
  state: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_gpu_sm_debug_mode_args(c.Struct):
  SIZE = 16
  channel_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  enable: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sms: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_warpstate(c.Struct):
  SIZE = 48
  valid_warps: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[2]], 0]
  trapped_warps: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[2]], 16]
  paused_warps: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[2]], 32]
@c.record
class struct_nvgpu_gpu_wait_pause_args(c.Struct):
  SIZE = 8
  pwarpstate: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_gpu_tpc_exception_en_status_args(c.Struct):
  SIZE = 8
  tpc_exception_en_sm_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_gpu_num_vsms(c.Struct):
  SIZE = 8
  num_vsms: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_nvgpu_gpu_vsms_mapping_entry(c.Struct):
  SIZE = 6
  gpc_logical_index: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  gpc_virtual_index: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  tpc_local_logical_index: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  tpc_global_logical_index: Annotated[Annotated[int, ctypes.c_ubyte], 3]
  sm_local_id: Annotated[Annotated[int, ctypes.c_ubyte], 4]
  tpc_migratable_index: Annotated[Annotated[int, ctypes.c_ubyte], 5]
@c.record
class struct_nvgpu_gpu_vsms_mapping(c.Struct):
  SIZE = 8
  vsms_map_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_gpu_get_buffer_info_args(c.Struct):
  SIZE = 24
  _in: Annotated[struct_nvgpu_gpu_get_buffer_info_args_in, 0]
  out: Annotated[struct_nvgpu_gpu_get_buffer_info_args_out, 0]
@c.record
class struct_nvgpu_gpu_get_buffer_info_args_in(c.Struct):
  SIZE = 16
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  metadata_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
  metadata_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_get_buffer_info_args_out(c.Struct):
  SIZE = 24
  flags: Annotated[Annotated[int, ctypes.c_uint64], 0]
  metadata_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_gpu_get_cpu_time_correlation_sample(c.Struct):
  SIZE = 16
  cpu_timestamp: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_timestamp: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_get_cpu_time_correlation_info_args(c.Struct):
  SIZE = 264
  samples: Annotated[c.Array[struct_nvgpu_gpu_get_cpu_time_correlation_sample, Literal[16]], 0]
  count: Annotated[Annotated[int, ctypes.c_uint32], 256]
  source_id: Annotated[Annotated[int, ctypes.c_uint32], 260]
@c.record
class struct_nvgpu_gpu_get_gpu_time_args(c.Struct):
  SIZE = 16
  gpu_timestamp: Annotated[Annotated[int, ctypes.c_uint64], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_get_engine_info_item(c.Struct):
  SIZE = 16
  engine_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  engine_instance: Annotated[Annotated[int, ctypes.c_uint32], 4]
  runlist_id: Annotated[Annotated[int, ctypes.c_int32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_gpu_get_engine_info_args(c.Struct):
  SIZE = 16
  engine_info_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  engine_info_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_alloc_vidmem_args(c.Struct):
  SIZE = 32
  _in: Annotated[struct_nvgpu_gpu_alloc_vidmem_args_in, 0]
  out: Annotated[struct_nvgpu_gpu_alloc_vidmem_args_out, 0]
@c.record
class struct_nvgpu_gpu_alloc_vidmem_args_in(c.Struct):
  SIZE = 32
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  memtag: Annotated[Annotated[int, ctypes.c_uint16], 12]
  reserved0: Annotated[Annotated[int, ctypes.c_uint16], 14]
  alignment: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved1: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 20]
@c.record
class struct_nvgpu_gpu_alloc_vidmem_args_out(c.Struct):
  SIZE = 4
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_nvgpu_gpu_clk_range(c.Struct):
  SIZE = 24
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  clk_domain: Annotated[Annotated[int, ctypes.c_uint32], 4]
  min_hz: Annotated[Annotated[int, ctypes.c_uint64], 8]
  max_hz: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_gpu_clk_range_args(c.Struct):
  SIZE = 16
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad0: Annotated[Annotated[int, ctypes.c_uint16], 4]
  num_entries: Annotated[Annotated[int, ctypes.c_uint16], 6]
  clk_range_entries: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_clk_vf_point(c.Struct):
  SIZE = 8
  freq_hz: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_gpu_clk_vf_points_args(c.Struct):
  SIZE = 24
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  clk_domain: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_entries: Annotated[Annotated[int, ctypes.c_uint16], 8]
  num_entries: Annotated[Annotated[int, ctypes.c_uint16], 10]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  clk_vf_point_entries: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_gpu_clk_info(c.Struct):
  SIZE = 16
  flags: Annotated[Annotated[int, ctypes.c_uint16], 0]
  clk_type: Annotated[Annotated[int, ctypes.c_uint16], 2]
  clk_domain: Annotated[Annotated[int, ctypes.c_uint32], 4]
  freq_hz: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_clk_get_info_args(c.Struct):
  SIZE = 16
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  clk_type: Annotated[Annotated[int, ctypes.c_uint16], 4]
  num_entries: Annotated[Annotated[int, ctypes.c_uint16], 6]
  clk_info_entries: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_clk_set_info_args(c.Struct):
  SIZE = 24
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad0: Annotated[Annotated[int, ctypes.c_uint16], 4]
  num_entries: Annotated[Annotated[int, ctypes.c_uint16], 6]
  clk_info_entries: Annotated[Annotated[int, ctypes.c_uint64], 8]
  completion_fd: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_nvgpu_gpu_get_event_fd_args(c.Struct):
  SIZE = 8
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  event_fd: Annotated[Annotated[int, ctypes.c_int32], 4]
@c.record
class struct_nvgpu_gpu_get_memory_state_args(c.Struct):
  SIZE = 40
  total_free_bytes: Annotated[Annotated[int, ctypes.c_uint64], 0]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[4]], 8]
@c.record
class struct_nvgpu_gpu_get_fbp_l2_masks_args(c.Struct):
  SIZE = 16
  mask_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  mask_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_get_voltage_args(c.Struct):
  SIZE = 16
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 0]
  which: Annotated[Annotated[int, ctypes.c_uint32], 8]
  voltage: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_gpu_get_current_args(c.Struct):
  SIZE = 16
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 0]
  currnt: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_gpu_get_power_args(c.Struct):
  SIZE = 16
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 0]
  power: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_gpu_get_temperature_args(c.Struct):
  SIZE = 16
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 0]
  temp_f24_8: Annotated[Annotated[int, ctypes.c_int32], 12]
@c.record
class struct_nvgpu_gpu_set_therm_alert_limit_args(c.Struct):
  SIZE = 16
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 0]
  temp_f24_8: Annotated[Annotated[int, ctypes.c_int32], 12]
@c.record
class struct_nvgpu_gpu_set_deterministic_opts_args(c.Struct):
  SIZE = 16
  num_channels: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  channels: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvgpu_gpu_register_buffer_args(c.Struct):
  SIZE = 24
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  comptags_alloc_control: Annotated[Annotated[int, ctypes.c_ubyte], 4]
  reserved0: Annotated[Annotated[int, ctypes.c_ubyte], 5]
  reserved1: Annotated[Annotated[int, ctypes.c_uint16], 6]
  metadata_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  metadata_size: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_nvgpu32_as_alloc_space_args(c.Struct):
  SIZE = 24
  pages: Annotated[Annotated[int, ctypes.c_uint32], 0]
  page_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  o_a: Annotated[struct_nvgpu32_as_alloc_space_args_o_a, 16]
@c.record
class struct_nvgpu32_as_alloc_space_args_o_a(c.Struct):
  SIZE = 8
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  align: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_as_alloc_space_args(c.Struct):
  SIZE = 32
  pages: Annotated[Annotated[int, ctypes.c_uint64], 0]
  page_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  o_a: Annotated[struct_nvgpu_as_alloc_space_args_o_a, 16]
  padding: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 24]
@c.record
class struct_nvgpu_as_alloc_space_args_o_a(c.Struct):
  SIZE = 8
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  align: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_as_free_space_args(c.Struct):
  SIZE = 32
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  pages: Annotated[Annotated[int, ctypes.c_uint64], 8]
  page_size: Annotated[Annotated[int, ctypes.c_uint32], 16]
  padding: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 20]
@c.record
class struct_nvgpu_as_bind_channel_args(c.Struct):
  SIZE = 4
  channel_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvgpu_as_map_buffer_ex_args(c.Struct):
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
class struct_nvgpu_as_get_buffer_compbits_info_args(c.Struct):
  SIZE = 32
  mapping_gva: Annotated[Annotated[int, ctypes.c_uint64], 0]
  compbits_win_size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  compbits_win_ctagline: Annotated[Annotated[int, ctypes.c_uint32], 16]
  mapping_ctagline: Annotated[Annotated[int, ctypes.c_uint32], 20]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_nvgpu_as_map_buffer_compbits_args(c.Struct):
  SIZE = 40
  mapping_gva: Annotated[Annotated[int, ctypes.c_uint64], 0]
  compbits_win_gva: Annotated[Annotated[int, ctypes.c_uint64], 8]
  mapping_iova: Annotated[Annotated[int, ctypes.c_uint64], 16]
  mapping_iova_buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 16]
  mapping_iova_buf_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_nvgpu_as_unmap_buffer_args(c.Struct):
  SIZE = 8
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvgpu_as_va_region(c.Struct):
  SIZE = 24
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  page_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  pages: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvgpu_as_get_va_regions_args(c.Struct):
  SIZE = 16
  buf_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  buf_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_as_map_buffer_batch_args(c.Struct):
  SIZE = 32
  unmaps: Annotated[Annotated[int, ctypes.c_uint64], 0]
  maps: Annotated[Annotated[int, ctypes.c_uint64], 8]
  num_unmaps: Annotated[Annotated[int, ctypes.c_uint32], 16]
  num_maps: Annotated[Annotated[int, ctypes.c_uint32], 20]
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_nvgpu_as_get_sync_ro_map_args(c.Struct):
  SIZE = 16
  base_gpuva: Annotated[Annotated[int, ctypes.c_uint64], 0]
  sync_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  num_syncpoints: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvgpu_as_mapping_modify_args(c.Struct):
  SIZE = 32
  compr_kind: Annotated[Annotated[int, ctypes.c_int16], 0]
  incompr_kind: Annotated[Annotated[int, ctypes.c_int16], 2]
  buffer_offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  buffer_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
  map_address: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_nvgpu_as_remap_op(c.Struct):
  SIZE = 40
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  compr_kind: Annotated[Annotated[int, ctypes.c_int16], 4]
  incompr_kind: Annotated[Annotated[int, ctypes.c_int16], 6]
  mem_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_int32], 12]
  mem_offset_in_pages: Annotated[Annotated[int, ctypes.c_uint64], 16]
  virt_offset_in_pages: Annotated[Annotated[int, ctypes.c_uint64], 24]
  num_pages: Annotated[Annotated[int, ctypes.c_uint64], 32]
@c.record
class struct_nvgpu_as_remap_args(c.Struct):
  SIZE = 16
  ops: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_ops: Annotated[Annotated[int, ctypes.c_uint32], 8]
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
NVMAP_HANDLE_PARAM_SIZE = _anonenum0.define('NVMAP_HANDLE_PARAM_SIZE', 1)
NVMAP_HANDLE_PARAM_ALIGNMENT = _anonenum0.define('NVMAP_HANDLE_PARAM_ALIGNMENT', 2)
NVMAP_HANDLE_PARAM_BASE = _anonenum0.define('NVMAP_HANDLE_PARAM_BASE', 3)
NVMAP_HANDLE_PARAM_HEAP = _anonenum0.define('NVMAP_HANDLE_PARAM_HEAP', 4)
NVMAP_HANDLE_PARAM_KIND = _anonenum0.define('NVMAP_HANDLE_PARAM_KIND', 5)
NVMAP_HANDLE_PARAM_COMPR = _anonenum0.define('NVMAP_HANDLE_PARAM_COMPR', 6)

class _anonenum1(Annotated[int, ctypes.c_uint32], c.Enum): pass
NVMAP_CACHE_OP_WB = _anonenum1.define('NVMAP_CACHE_OP_WB', 0)
NVMAP_CACHE_OP_INV = _anonenum1.define('NVMAP_CACHE_OP_INV', 1)
NVMAP_CACHE_OP_WB_INV = _anonenum1.define('NVMAP_CACHE_OP_WB_INV', 2)

class _anonenum2(Annotated[int, ctypes.c_uint32], c.Enum): pass
NVMAP_PAGES_UNRESERVE = _anonenum2.define('NVMAP_PAGES_UNRESERVE', 0)
NVMAP_PAGES_RESERVE = _anonenum2.define('NVMAP_PAGES_RESERVE', 1)
NVMAP_INSERT_PAGES_ON_UNRESERVE = _anonenum2.define('NVMAP_INSERT_PAGES_ON_UNRESERVE', 2)
NVMAP_PAGES_PROT_AND_CLEAN = _anonenum2.define('NVMAP_PAGES_PROT_AND_CLEAN', 3)

@c.record
class struct_nvmap_create_handle(c.Struct):
  SIZE = 8
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ivm_id: Annotated[Annotated[int, ctypes.c_uint64], 0]
  ivm_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size64: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle64: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_nvmap_create_handle_from_va(c.Struct):
  SIZE = 24
  va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  size64: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_nvmap_gup_test(c.Struct):
  SIZE = 16
  va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  result: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_nvmap_alloc_handle(c.Struct):
  SIZE = 20
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  heap_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  align: Annotated[Annotated[int, ctypes.c_uint32], 12]
  numa_nid: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_nvmap_alloc_ivm_handle(c.Struct):
  SIZE = 20
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  heap_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  align: Annotated[Annotated[int, ctypes.c_uint32], 12]
  peer: Annotated[Annotated[int, ctypes.c_uint32], 16]
@c.record
class struct_nvmap_rw_handle(c.Struct):
  SIZE = 56
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  elem_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  hmem_stride: Annotated[Annotated[int, ctypes.c_uint64], 32]
  user_stride: Annotated[Annotated[int, ctypes.c_uint64], 40]
  count: Annotated[Annotated[int, ctypes.c_uint64], 48]
@c.record
class struct_nvmap_handle_param(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  param: Annotated[Annotated[int, ctypes.c_uint32], 4]
  result: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvmap_cache_op(c.Struct):
  SIZE = 24
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  len: Annotated[Annotated[int, ctypes.c_uint32], 12]
  op: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_nvmap_cache_op_64(c.Struct):
  SIZE = 32
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  len: Annotated[Annotated[int, ctypes.c_uint64], 16]
  op: Annotated[Annotated[int, ctypes.c_int32], 24]
@c.record
class struct_nvmap_cache_op_list(c.Struct):
  SIZE = 32
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  offsets: Annotated[Annotated[int, ctypes.c_uint64], 8]
  sizes: Annotated[Annotated[int, ctypes.c_uint64], 16]
  nr: Annotated[Annotated[int, ctypes.c_uint32], 24]
  op: Annotated[Annotated[int, ctypes.c_int32], 28]
@c.record
class struct_nvmap_debugfs_handles_header(c.Struct):
  SIZE = 1
  version: Annotated[Annotated[int, ctypes.c_ubyte], 0]
@c.record
class struct_nvmap_debugfs_handles_entry(c.Struct):
  SIZE = 32
  base: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  share_count: Annotated[Annotated[int, ctypes.c_uint32], 20]
  mapped_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_nvmap_set_tag_label(c.Struct):
  SIZE = 16
  tag: Annotated[Annotated[int, ctypes.c_uint32], 0]
  len: Annotated[Annotated[int, ctypes.c_uint32], 4]
  addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvmap_available_heaps(c.Struct):
  SIZE = 8
  heaps: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_nvmap_heap_size(c.Struct):
  SIZE = 16
  heap: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_nvmap_sciipc_map(c.Struct):
  SIZE = 32
  auth_token: Annotated[Annotated[int, ctypes.c_uint64], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  sci_ipc_id: Annotated[Annotated[int, ctypes.c_uint64], 16]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 24]
@c.record
class struct_nvmap_handle_parameters(c.Struct):
  SIZE = 72
  contig: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  import_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  heap_number: Annotated[Annotated[int, ctypes.c_uint32], 12]
  access_flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  heap: Annotated[Annotated[int, ctypes.c_uint64], 24]
  align: Annotated[Annotated[int, ctypes.c_uint64], 32]
  coherency: Annotated[Annotated[int, ctypes.c_uint64], 40]
  size: Annotated[Annotated[int, ctypes.c_uint64], 48]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 56]
  serial_id: Annotated[Annotated[int, ctypes.c_uint64], 64]
@c.record
class struct_nvmap_query_heap_params(c.Struct):
  SIZE = 48
  heap_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  contig: Annotated[Annotated[int, ctypes.c_ubyte], 8]
  total: Annotated[Annotated[int, ctypes.c_uint64], 16]
  free: Annotated[Annotated[int, ctypes.c_uint64], 24]
  largest_free_block: Annotated[Annotated[int, ctypes.c_uint64], 32]
  granule_size: Annotated[Annotated[int, ctypes.c_uint32], 40]
@c.record
class struct_nvmap_duplicate_handle(c.Struct):
  SIZE = 12
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  access_flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  dup_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_nvmap_fd_for_range_from_list(c.Struct):
  SIZE = 40
  handles: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 0]
  num_handles: Annotated[Annotated[int, ctypes.c_uint32], 8]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  fd: Annotated[Annotated[int, ctypes.c_int32], 32]
c.init_records()
NVGPU_TSG_IOCTL_MAGIC = 'T' # type: ignore
NVGPU_GPU_IOCTL_TSG_L2_SECTOR_PROMOTE_FLAG_NONE = (1 << 0) # type: ignore
NVGPU_GPU_IOCTL_TSG_L2_SECTOR_PROMOTE_FLAG_64B = (1 << 1) # type: ignore
NVGPU_GPU_IOCTL_TSG_L2_SECTOR_PROMOTE_FLAG_128B = (1 << 2) # type: ignore
NVGPU_TSG_SUBCONTEXT_TYPE_SYNC = (0x0) # type: ignore
NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC = (0x1) # type: ignore
NVGPU_TSG_IOCTL_BIND_CHANNEL = _IOW(NVGPU_TSG_IOCTL_MAGIC, 1, int) # type: ignore
NVGPU_TSG_IOCTL_UNBIND_CHANNEL = _IOW(NVGPU_TSG_IOCTL_MAGIC, 2, int) # type: ignore
NVGPU_IOCTL_TSG_ENABLE = _IO(NVGPU_TSG_IOCTL_MAGIC, 3) # type: ignore
NVGPU_IOCTL_TSG_DISABLE = _IO(NVGPU_TSG_IOCTL_MAGIC, 4) # type: ignore
NVGPU_IOCTL_TSG_PREEMPT = _IO(NVGPU_TSG_IOCTL_MAGIC, 5) # type: ignore
NVGPU_IOCTL_TSG_EVENT_ID_CTRL = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 7, struct_nvgpu_event_id_ctrl_args) # type: ignore
NVGPU_IOCTL_TSG_SET_RUNLIST_INTERLEAVE = _IOW(NVGPU_TSG_IOCTL_MAGIC, 8, struct_nvgpu_runlist_interleave_args) # type: ignore
NVGPU_IOCTL_TSG_SET_TIMESLICE = _IOW(NVGPU_TSG_IOCTL_MAGIC, 9, struct_nvgpu_timeslice_args) # type: ignore
NVGPU_IOCTL_TSG_GET_TIMESLICE = _IOR(NVGPU_TSG_IOCTL_MAGIC, 10, struct_nvgpu_timeslice_args) # type: ignore
NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 11, struct_nvgpu_tsg_bind_channel_ex_args) # type: ignore
NVGPU_TSG_IOCTL_READ_SINGLE_SM_ERROR_STATE = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 12, struct_nvgpu_tsg_read_single_sm_error_state_args) # type: ignore
NVGPU_TSG_IOCTL_SET_L2_MAX_WAYS_EVICT_LAST = _IOW(NVGPU_TSG_IOCTL_MAGIC, 13, struct_nvgpu_tsg_l2_max_ways_evict_last_args) # type: ignore
NVGPU_TSG_IOCTL_GET_L2_MAX_WAYS_EVICT_LAST = _IOR(NVGPU_TSG_IOCTL_MAGIC, 14, struct_nvgpu_tsg_l2_max_ways_evict_last_args) # type: ignore
NVGPU_TSG_IOCTL_SET_L2_SECTOR_PROMOTION = _IOW(NVGPU_TSG_IOCTL_MAGIC, 15, struct_nvgpu_tsg_set_l2_sector_promotion_args) # type: ignore
NVGPU_TSG_IOCTL_BIND_SCHEDULING_DOMAIN = _IOW(NVGPU_TSG_IOCTL_MAGIC, 16, struct_nvgpu_tsg_bind_scheduling_domain_args) # type: ignore
NVGPU_TSG_IOCTL_READ_ALL_SM_ERROR_STATES = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 17, struct_nvgpu_tsg_read_all_sm_error_state_args) # type: ignore
NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 18, struct_nvgpu_tsg_create_subcontext_args) # type: ignore
NVGPU_TSG_IOCTL_DELETE_SUBCONTEXT = _IOW(NVGPU_TSG_IOCTL_MAGIC, 19, struct_nvgpu_tsg_delete_subcontext_args) # type: ignore
NVGPU_TSG_IOCTL_GET_SHARE_TOKEN = _IOWR(NVGPU_TSG_IOCTL_MAGIC, 20, struct_nvgpu_tsg_get_share_token_args) # type: ignore
NVGPU_TSG_IOCTL_REVOKE_SHARE_TOKEN = _IOW(NVGPU_TSG_IOCTL_MAGIC, 21, struct_nvgpu_tsg_revoke_share_token_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_MAGIC = 'D' # type: ignore
NVGPU_DBG_GPU_IOCTL_BIND_CHANNEL = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 1, struct_nvgpu_dbg_gpu_bind_channel_args) # type: ignore
NVGPU_DBG_GPU_REG_OP_READ_32 = (0x00000000) # type: ignore
NVGPU_DBG_GPU_REG_OP_WRITE_32 = (0x00000001) # type: ignore
NVGPU_DBG_GPU_REG_OP_READ_64 = (0x00000002) # type: ignore
NVGPU_DBG_GPU_REG_OP_WRITE_64 = (0x00000003) # type: ignore
NVGPU_DBG_GPU_REG_OP_READ_08 = (0x00000004) # type: ignore
NVGPU_DBG_GPU_REG_OP_WRITE_08 = (0x00000005) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GLOBAL = (0x00000000) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX = (0x00000001) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_TPC = (0x00000002) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_SM = (0x00000004) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_CROP = (0x00000008) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_ZROP = (0x00000010) # type: ignore
NVGPU_DBG_GPU_REG_OP_TYPE_GR_CTX_QUAD = (0x00000040) # type: ignore
NVGPU_DBG_GPU_REG_OP_STATUS_SUCCESS = (0x00000000) # type: ignore
NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_OP = (0x00000001) # type: ignore
NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_TYPE = (0x00000002) # type: ignore
NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_OFFSET = (0x00000004) # type: ignore
NVGPU_DBG_GPU_REG_OP_STATUS_UNSUPPORTED_OP = (0x00000008) # type: ignore
NVGPU_DBG_GPU_REG_OP_STATUS_INVALID_MASK = (0x00000010) # type: ignore
NVGPU_DBG_GPU_IOCTL_REG_OPS = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 2, struct_nvgpu_dbg_gpu_exec_reg_ops_args) # type: ignore
NVGPU_DBG_GPU_EVENTS_CTRL_CMD_DISABLE = (0x00000000) # type: ignore
NVGPU_DBG_GPU_EVENTS_CTRL_CMD_ENABLE = (0x00000001) # type: ignore
NVGPU_DBG_GPU_EVENTS_CTRL_CMD_CLEAR = (0x00000002) # type: ignore
NVGPU_DBG_GPU_IOCTL_EVENTS_CTRL = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 3, struct_nvgpu_dbg_gpu_events_ctrl_args) # type: ignore
NVGPU_DBG_GPU_POWERGATE_MODE_ENABLE = 1 # type: ignore
NVGPU_DBG_GPU_POWERGATE_MODE_DISABLE = 2 # type: ignore
NVGPU_DBG_GPU_IOCTL_POWERGATE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 4, struct_nvgpu_dbg_gpu_powergate_args) # type: ignore
NVGPU_DBG_GPU_SMPC_CTXSW_MODE_NO_CTXSW = (0x00000000) # type: ignore
NVGPU_DBG_GPU_SMPC_CTXSW_MODE_CTXSW = (0x00000001) # type: ignore
NVGPU_DBG_GPU_IOCTL_SMPC_CTXSW_MODE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 5, struct_nvgpu_dbg_gpu_smpc_ctxsw_mode_args) # type: ignore
NVGPU_DBG_GPU_SUSPEND_ALL_SMS = 1 # type: ignore
NVGPU_DBG_GPU_RESUME_ALL_SMS = 2 # type: ignore
NVGPU_DBG_GPU_IOCTL_SUSPEND_RESUME_ALL_SMS = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 6, struct_nvgpu_dbg_gpu_suspend_resume_all_sms_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_PERFBUF_MAP = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 7, struct_nvgpu_dbg_gpu_perfbuf_map_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_PERFBUF_UNMAP = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 8, struct_nvgpu_dbg_gpu_perfbuf_unmap_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_PC_SAMPLING_DISABLE = 0 # type: ignore
NVGPU_DBG_GPU_IOCTL_PC_SAMPLING_ENABLE = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_PC_SAMPLING = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC,  9, struct_nvgpu_dbg_gpu_pc_sampling_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_TIMEOUT_ENABLE = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_TIMEOUT_DISABLE = 0 # type: ignore
NVGPU_DBG_GPU_IOCTL_TIMEOUT = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 10, struct_nvgpu_dbg_gpu_timeout_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_GET_TIMEOUT = _IOR(NVGPU_DBG_GPU_IOCTL_MAGIC, 11, struct_nvgpu_dbg_gpu_timeout_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_SET_NEXT_STOP_TRIGGER_TYPE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 12, struct_nvgpu_dbg_gpu_set_next_stop_trigger_type_args) # type: ignore
NVGPU_DBG_GPU_HWPM_CTXSW_MODE_NO_CTXSW = (0x00000000) # type: ignore
NVGPU_DBG_GPU_HWPM_CTXSW_MODE_CTXSW = (0x00000001) # type: ignore
NVGPU_DBG_GPU_HWPM_CTXSW_MODE_STREAM_OUT_CTXSW = (0x00000002) # type: ignore
NVGPU_DBG_GPU_IOCTL_HWPM_CTXSW_MODE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 13, struct_nvgpu_dbg_gpu_hwpm_ctxsw_mode_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_READ_SINGLE_SM_ERROR_STATE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 14, struct_nvgpu_dbg_gpu_read_single_sm_error_state_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_CLEAR_SINGLE_SM_ERROR_STATE = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 15, struct_nvgpu_dbg_gpu_clear_single_sm_error_state_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_UNBIND_CHANNEL = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 17, struct_nvgpu_dbg_gpu_unbind_channel_args) # type: ignore
NVGPU_DBG_GPU_SUSPEND_ALL_CONTEXTS = 1 # type: ignore
NVGPU_DBG_GPU_RESUME_ALL_CONTEXTS = 2 # type: ignore
NVGPU_DBG_GPU_IOCTL_SUSPEND_RESUME_CONTEXTS = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 18, struct_nvgpu_dbg_gpu_suspend_resume_contexts_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_ACCESS_FB_MEMORY_CMD_READ = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_ACCESS_FB_MEMORY_CMD_WRITE = 2 # type: ignore
NVGPU_DBG_GPU_IOCTL_ACCESS_FB_MEMORY = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 19, struct_nvgpu_dbg_gpu_access_fb_memory_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_PROFILER_ALLOCATE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 20, struct_nvgpu_dbg_gpu_profiler_obj_mgt_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_PROFILER_FREE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 21, struct_nvgpu_dbg_gpu_profiler_obj_mgt_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_PROFILER_RESERVE = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 22, struct_nvgpu_dbg_gpu_profiler_reserve_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK_NONE = (0x0) # type: ignore
NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK_FATAL = (0x1 << 0) # type: ignore
NVGPU_DBG_GPU_IOCTL_SET_SM_EXCEPTION_TYPE_MASK = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 23, struct_nvgpu_dbg_gpu_set_sm_exception_type_mask_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_CYCLE_STATS = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 24, struct_nvgpu_dbg_gpu_cycle_stats_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT_CMD_FLUSH = 0 # type: ignore
NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT_CMD_ATTACH = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT_CMD_DETACH = 2 # type: ignore
NVGPU_DBG_GPU_IOCTL_CYCLE_STATS_SNAPSHOT = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 25, struct_nvgpu_dbg_gpu_cycle_stats_snapshot_args) # type: ignore
NVGPU_DBG_GPU_CTX_MMU_DEBUG_MODE_DISABLED = 0 # type: ignore
NVGPU_DBG_GPU_CTX_MMU_DEBUG_MODE_ENABLED = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_SET_CTX_MMU_DEBUG_MODE = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 26, struct_nvgpu_dbg_gpu_set_ctx_mmu_debug_mode_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_GET_GR_CONTEXT_SIZE = _IOR(NVGPU_DBG_GPU_IOCTL_MAGIC, 27, struct_nvgpu_dbg_gpu_get_gr_context_size_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_GET_GR_CONTEXT = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 28, struct_nvgpu_dbg_gpu_get_gr_context_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_TSG_SET_TIMESLICE = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 29, struct_nvgpu_timeslice_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_TSG_GET_TIMESLICE = _IOR(NVGPU_DBG_GPU_IOCTL_MAGIC, 30, struct_nvgpu_timeslice_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_ACCESS_GPUVA_CMD_READ = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_ACCESS_GPUVA_CMD_WRITE = 2 # type: ignore
NVGPU_DBG_GPU_IOCTL_GET_MAPPINGS = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 31, struct_nvgpu_dbg_gpu_get_mappings_args) # type: ignore
NVGPU_DBG_GPU_IOCTL_ACCESS_GPU_VA = _IOWR(NVGPU_DBG_GPU_IOCTL_MAGIC, 32, struct_nvgpu_dbg_gpu_va_access_args) # type: ignore
NVGPU_DBG_GPU_SCHED_EXIT_WAIT_FOR_ERRBAR_DISABLED = 0 # type: ignore
NVGPU_DBG_GPU_SCHED_EXIT_WAIT_FOR_ERRBAR_ENABLED = 1 # type: ignore
NVGPU_DBG_GPU_IOCTL_SET_SCHED_EXIT_WAIT_FOR_ERRBAR = _IOW(NVGPU_DBG_GPU_IOCTL_MAGIC, 33, struct_nvgpu_sched_exit_wait_for_errbar_args) # type: ignore
NVGPU_PROFILER_IOCTL_MAGIC = 'P' # type: ignore
NVGPU_PROFILER_PM_RESOURCE_ARG_HWPM_LEGACY = 0 # type: ignore
NVGPU_PROFILER_PM_RESOURCE_ARG_SMPC = 1 # type: ignore
NVGPU_PROFILER_PM_RESOURCE_ARG_PC_SAMPLER = 2 # type: ignore
NVGPU_PROFILER_PM_RESOURCE_ARG_HES_CWD = 3 # type: ignore
NVGPU_PROFILER_RESERVE_PM_RESOURCE_ARG_FLAG_CTXSW = (1 << 0) # type: ignore
NVGPU_PROFILER_ALLOC_PMA_STREAM_ARG_FLAG_CTXSW = (1 << 0) # type: ignore
NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_UPDATE_AVAILABLE_BYTES = (1 << 0) # type: ignore
NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_WAIT_FOR_UPDATE = (1 << 1) # type: ignore
NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_RETURN_PUT_PTR = (1 << 2) # type: ignore
NVGPU_PROFILER_PMA_STREAM_UPDATE_GET_PUT_ARG_FLAG_OVERFLOW_TRIGGERED = (1 << 3) # type: ignore
NVGPU_PROFILER_EXEC_REG_OPS_ARG_MODE_ALL_OR_NONE = 0 # type: ignore
NVGPU_PROFILER_EXEC_REG_OPS_ARG_MODE_CONTINUE_ON_ERROR = 1 # type: ignore
NVGPU_PROFILER_EXEC_REG_OPS_ARG_FLAG_ALL_PASSED = (1 << 0) # type: ignore
NVGPU_PROFILER_EXEC_REG_OPS_ARG_FLAG_DIRECT_OPS = (1 << 1) # type: ignore
NVGPU_PROFILER_VAB_RANGE_CHECKER_MODE_ACCESS = 1 # type: ignore
NVGPU_PROFILER_VAB_RANGE_CHECKER_MODE_DIRTY = 2 # type: ignore
NVGPU_PROFILER_IOCTL_BIND_CONTEXT = _IOW(NVGPU_PROFILER_IOCTL_MAGIC, 1, struct_nvgpu_profiler_bind_context_args) # type: ignore
NVGPU_PROFILER_IOCTL_RESERVE_PM_RESOURCE = _IOW(NVGPU_PROFILER_IOCTL_MAGIC, 2, struct_nvgpu_profiler_reserve_pm_resource_args) # type: ignore
NVGPU_PROFILER_IOCTL_RELEASE_PM_RESOURCE = _IOW(NVGPU_PROFILER_IOCTL_MAGIC, 3, struct_nvgpu_profiler_release_pm_resource_args) # type: ignore
NVGPU_PROFILER_IOCTL_ALLOC_PMA_STREAM = _IOWR(NVGPU_PROFILER_IOCTL_MAGIC, 4, struct_nvgpu_profiler_alloc_pma_stream_args) # type: ignore
NVGPU_PROFILER_IOCTL_FREE_PMA_STREAM = _IOW(NVGPU_PROFILER_IOCTL_MAGIC, 5, struct_nvgpu_profiler_free_pma_stream_args) # type: ignore
NVGPU_PROFILER_IOCTL_BIND_PM_RESOURCES = _IO(NVGPU_PROFILER_IOCTL_MAGIC, 6) # type: ignore
NVGPU_PROFILER_IOCTL_UNBIND_PM_RESOURCES = _IO(NVGPU_PROFILER_IOCTL_MAGIC, 7) # type: ignore
NVGPU_PROFILER_IOCTL_PMA_STREAM_UPDATE_GET_PUT = _IOWR(NVGPU_PROFILER_IOCTL_MAGIC, 8, struct_nvgpu_profiler_pma_stream_update_get_put_args) # type: ignore
NVGPU_PROFILER_IOCTL_EXEC_REG_OPS = _IOWR(NVGPU_PROFILER_IOCTL_MAGIC, 9, struct_nvgpu_profiler_exec_reg_ops_args) # type: ignore
NVGPU_PROFILER_IOCTL_UNBIND_CONTEXT = _IO(NVGPU_PROFILER_IOCTL_MAGIC, 10) # type: ignore
NVGPU_PROFILER_IOCTL_VAB_RESERVE = _IOW(NVGPU_PROFILER_IOCTL_MAGIC, 11, struct_nvgpu_profiler_vab_reserve_args) # type: ignore
NVGPU_PROFILER_IOCTL_VAB_RELEASE = _IO(NVGPU_PROFILER_IOCTL_MAGIC, 12) # type: ignore
NVGPU_PROFILER_IOCTL_VAB_FLUSH_STATE = _IOW(NVGPU_PROFILER_IOCTL_MAGIC, 13, struct_nvgpu_profiler_vab_flush_state_args) # type: ignore
NVGPU_IOCTL_MAGIC = 'H' # type: ignore
NVGPU_TIMEOUT_FLAG_DISABLE_DUMP = 0 # type: ignore
NVGPU_ALLOC_OBJ_FLAGS_LOCKBOOST_ZERO = (1 << 0) # type: ignore
NVGPU_ALLOC_OBJ_FLAGS_GFXP = (1 << 1) # type: ignore
NVGPU_ALLOC_OBJ_FLAGS_CILP = (1 << 2) # type: ignore
NVGPU_ALLOC_GPFIFO_EX_FLAGS_VPR_ENABLED = (1 << 0) # type: ignore
NVGPU_ALLOC_GPFIFO_EX_FLAGS_DETERMINISTIC = (1 << 1) # type: ignore
NVGPU_CHANNEL_SETUP_BIND_FLAGS_VPR_ENABLED = (1 << 0) # type: ignore
NVGPU_CHANNEL_SETUP_BIND_FLAGS_DETERMINISTIC = (1 << 1) # type: ignore
NVGPU_CHANNEL_SETUP_BIND_FLAGS_REPLAYABLE_FAULTS_ENABLE = (1 << 2) # type: ignore
NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT = (1 << 3) # type: ignore
NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_GPU_MAP_RESOURCES_SUPPORT = (1 << 4) # type: ignore
NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_WAIT = (1 << 0) # type: ignore
NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_GET = (1 << 1) # type: ignore
NVGPU_SUBMIT_GPFIFO_FLAGS_HW_FORMAT = (1 << 2) # type: ignore
NVGPU_SUBMIT_GPFIFO_FLAGS_SYNC_FENCE = (1 << 3) # type: ignore
NVGPU_SUBMIT_GPFIFO_FLAGS_SUPPRESS_WFI = (1 << 4) # type: ignore
NVGPU_SUBMIT_GPFIFO_FLAGS_SKIP_BUFFER_REFCOUNTING = (1 << 5) # type: ignore
NVGPU_WAIT_TYPE_NOTIFIER = 0x0 # type: ignore
NVGPU_WAIT_TYPE_SEMAPHORE = 0x1 # type: ignore
NVGPU_ZCULL_MODE_GLOBAL = 0 # type: ignore
NVGPU_ZCULL_MODE_NO_CTXSW = 1 # type: ignore
NVGPU_ZCULL_MODE_SEPARATE_BUFFER = 2 # type: ignore
NVGPU_ZCULL_MODE_PART_OF_REGULAR_BUF = 3 # type: ignore
NVGPU_CHANNEL_FIFO_ERROR_IDLE_TIMEOUT = 8 # type: ignore
NVGPU_CHANNEL_GR_ERROR_SW_METHOD = 12 # type: ignore
NVGPU_CHANNEL_GR_ERROR_SW_NOTIFY = 13 # type: ignore
NVGPU_CHANNEL_GR_EXCEPTION = 13 # type: ignore
NVGPU_CHANNEL_GR_SEMAPHORE_TIMEOUT = 24 # type: ignore
NVGPU_CHANNEL_GR_ILLEGAL_NOTIFY = 25 # type: ignore
NVGPU_CHANNEL_FIFO_ERROR_MMU_ERR_FLT = 31 # type: ignore
NVGPU_CHANNEL_PBDMA_ERROR = 32 # type: ignore
NVGPU_CHANNEL_FECS_ERR_UNIMP_FIRMWARE_METHOD = 37 # type: ignore
NVGPU_CHANNEL_RESETCHANNEL_VERIF_ERROR = 43 # type: ignore
NVGPU_CHANNEL_PBDMA_PUSHBUFFER_CRC_MISMATCH = 80 # type: ignore
NVGPU_CHANNEL_SUBMIT_TIMEOUT = 1 # type: ignore
NVGPU_IOCTL_CHANNEL_DISABLE_WDT = (1 << 0) # type: ignore
NVGPU_IOCTL_CHANNEL_ENABLE_WDT = (1 << 1) # type: ignore
NVGPU_IOCTL_CHANNEL_WDT_FLAG_SET_TIMEOUT = (1 << 2) # type: ignore
NVGPU_IOCTL_CHANNEL_WDT_FLAG_DISABLE_DUMP = (1 << 3) # type: ignore
NVGPU_RUNLIST_INTERLEAVE_LEVEL_LOW = 0 # type: ignore
NVGPU_RUNLIST_INTERLEAVE_LEVEL_MEDIUM = 1 # type: ignore
NVGPU_RUNLIST_INTERLEAVE_LEVEL_HIGH = 2 # type: ignore
NVGPU_RUNLIST_INTERLEAVE_NUM_LEVELS = 3 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_BPT_INT = 0 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_BPT_PAUSE = 1 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_BLOCKING_SYNC = 2 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_CILP_PREEMPTION_STARTED = 3 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_CILP_PREEMPTION_COMPLETE = 4 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_GR_SEMAPHORE_WRITE_AWAKEN = 5 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_MAX = 6 # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_CMD_ENABLE = 1 # type: ignore
NVGPU_GRAPHICS_PREEMPTION_MODE_WFI = (1 << 0) # type: ignore
NVGPU_GRAPHICS_PREEMPTION_MODE_GFXP = (1 << 1) # type: ignore
NVGPU_COMPUTE_PREEMPTION_MODE_WFI = (1 << 0) # type: ignore
NVGPU_COMPUTE_PREEMPTION_MODE_CTA = (1 << 1) # type: ignore
NVGPU_COMPUTE_PREEMPTION_MODE_CILP = (1 << 2) # type: ignore
NVGPU_BOOSTED_CTX_MODE_NORMAL = (0) # type: ignore
NVGPU_BOOSTED_CTX_MODE_BOOSTED_EXECUTION = (1) # type: ignore
NVGPU_RESCHEDULE_RUNLIST_PREEMPT_NEXT = (1 << 0) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_NVMAP_FD = _IOW(NVGPU_IOCTL_MAGIC, 5, struct_nvgpu_set_nvmap_fd_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_TIMEOUT = _IOW(NVGPU_IOCTL_MAGIC, 11, struct_nvgpu_set_timeout_args) # type: ignore
NVGPU_IOCTL_CHANNEL_GET_TIMEDOUT = _IOR(NVGPU_IOCTL_MAGIC, 12, struct_nvgpu_get_param_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_TIMEOUT_EX = _IOWR(NVGPU_IOCTL_MAGIC, 18, struct_nvgpu_set_timeout_ex_args) # type: ignore
NVGPU_IOCTL_CHANNEL_WAIT = _IOWR(NVGPU_IOCTL_MAGIC, 102, struct_nvgpu_wait_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO = _IOWR(NVGPU_IOCTL_MAGIC, 107, struct_nvgpu_submit_gpfifo_args) # type: ignore
NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _IOWR(NVGPU_IOCTL_MAGIC, 108, struct_nvgpu_alloc_obj_ctx_args) # type: ignore
NVGPU_IOCTL_CHANNEL_ZCULL_BIND = _IOWR(NVGPU_IOCTL_MAGIC, 110, struct_nvgpu_zcull_bind_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER = _IOWR(NVGPU_IOCTL_MAGIC, 111, struct_nvgpu_set_error_notifier) # type: ignore
NVGPU_IOCTL_CHANNEL_OPEN = _IOR(NVGPU_IOCTL_MAGIC,  112, struct_nvgpu_channel_open_args) # type: ignore
NVGPU_IOCTL_CHANNEL_ENABLE = _IO(NVGPU_IOCTL_MAGIC,  113) # type: ignore
NVGPU_IOCTL_CHANNEL_DISABLE = _IO(NVGPU_IOCTL_MAGIC,  114) # type: ignore
NVGPU_IOCTL_CHANNEL_PREEMPT = _IO(NVGPU_IOCTL_MAGIC,  115) # type: ignore
NVGPU_IOCTL_CHANNEL_FORCE_RESET = _IO(NVGPU_IOCTL_MAGIC,  116) # type: ignore
NVGPU_IOCTL_CHANNEL_EVENT_ID_CTRL = _IOWR(NVGPU_IOCTL_MAGIC, 117, struct_nvgpu_event_id_ctrl_args) # type: ignore
NVGPU_IOCTL_CHANNEL_WDT = _IOW(NVGPU_IOCTL_MAGIC, 119, struct_nvgpu_channel_wdt_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_RUNLIST_INTERLEAVE = _IOW(NVGPU_IOCTL_MAGIC, 120, struct_nvgpu_runlist_interleave_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_PREEMPTION_MODE = _IOW(NVGPU_IOCTL_MAGIC, 122, struct_nvgpu_preemption_mode_args) # type: ignore
NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX = _IOW(NVGPU_IOCTL_MAGIC, 123, struct_nvgpu_alloc_gpfifo_ex_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SET_BOOSTED_CTX = _IOW(NVGPU_IOCTL_MAGIC, 124, struct_nvgpu_boosted_ctx_args) # type: ignore
NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT = _IOR(NVGPU_IOCTL_MAGIC, 126, struct_nvgpu_get_user_syncpoint_args) # type: ignore
NVGPU_IOCTL_CHANNEL_RESCHEDULE_RUNLIST = _IOW(NVGPU_IOCTL_MAGIC, 127, struct_nvgpu_reschedule_runlist_args) # type: ignore
NVGPU_IOCTL_CHANNEL_SETUP_BIND = _IOWR(NVGPU_IOCTL_MAGIC, 128, struct_nvgpu_channel_setup_bind_args) # type: ignore
NVGPU_CTXSW_IOCTL_MAGIC = 'C' # type: ignore
NVGPU_CTXSW_TAG_SOF = 0x00 # type: ignore
NVGPU_CTXSW_TAG_CTXSW_REQ_BY_HOST = 0x01 # type: ignore
NVGPU_CTXSW_TAG_FE_ACK = 0x02 # type: ignore
NVGPU_CTXSW_TAG_FE_ACK_WFI = 0x0a # type: ignore
NVGPU_CTXSW_TAG_FE_ACK_GFXP = 0x0b # type: ignore
NVGPU_CTXSW_TAG_FE_ACK_CTAP = 0x0c # type: ignore
NVGPU_CTXSW_TAG_FE_ACK_CILP = 0x0d # type: ignore
NVGPU_CTXSW_TAG_SAVE_END = 0x03 # type: ignore
NVGPU_CTXSW_TAG_RESTORE_START = 0x04 # type: ignore
NVGPU_CTXSW_TAG_CONTEXT_START = 0x05 # type: ignore
NVGPU_CTXSW_TAG_ENGINE_RESET = 0xfe # type: ignore
NVGPU_CTXSW_TAG_INVALID_TIMESTAMP = 0xff # type: ignore
NVGPU_CTXSW_TAG_LAST = NVGPU_CTXSW_TAG_INVALID_TIMESTAMP # type: ignore
NVGPU_CTXSW_RING_HEADER_MAGIC = 0x7000fade # type: ignore
NVGPU_CTXSW_RING_HEADER_VERSION = 0 # type: ignore
NVGPU_CTXSW_FILTER_SIZE = (NVGPU_CTXSW_TAG_LAST + 1) # type: ignore
NVGPU_CTXSW_IOCTL_TRACE_ENABLE = _IO(NVGPU_CTXSW_IOCTL_MAGIC, 1) # type: ignore
NVGPU_CTXSW_IOCTL_TRACE_DISABLE = _IO(NVGPU_CTXSW_IOCTL_MAGIC, 2) # type: ignore
NVGPU_CTXSW_IOCTL_RING_SETUP = _IOWR(NVGPU_CTXSW_IOCTL_MAGIC, 3, struct_nvgpu_ctxsw_ring_setup_args) # type: ignore
NVGPU_CTXSW_IOCTL_SET_FILTER = _IOW(NVGPU_CTXSW_IOCTL_MAGIC, 4, struct_nvgpu_ctxsw_trace_filter_args) # type: ignore
NVGPU_CTXSW_IOCTL_GET_FILTER = _IOR(NVGPU_CTXSW_IOCTL_MAGIC, 5, struct_nvgpu_ctxsw_trace_filter_args) # type: ignore
NVGPU_CTXSW_IOCTL_POLL = _IO(NVGPU_CTXSW_IOCTL_MAGIC, 6) # type: ignore
NVGPU_SCHED_IOCTL_MAGIC = 'S' # type: ignore
NVGPU_SCHED_IOCTL_GET_TSGS = _IOWR(NVGPU_SCHED_IOCTL_MAGIC, 1, struct_nvgpu_sched_get_tsgs_args) # type: ignore
NVGPU_SCHED_IOCTL_GET_RECENT_TSGS = _IOWR(NVGPU_SCHED_IOCTL_MAGIC, 2, struct_nvgpu_sched_get_tsgs_args) # type: ignore
NVGPU_SCHED_IOCTL_GET_TSGS_BY_PID = _IOWR(NVGPU_SCHED_IOCTL_MAGIC, 3, struct_nvgpu_sched_get_tsgs_by_pid_args) # type: ignore
NVGPU_SCHED_IOCTL_TSG_GET_PARAMS = _IOWR(NVGPU_SCHED_IOCTL_MAGIC, 4, struct_nvgpu_sched_tsg_get_params_args) # type: ignore
NVGPU_SCHED_IOCTL_TSG_SET_TIMESLICE = _IOW(NVGPU_SCHED_IOCTL_MAGIC, 5, struct_nvgpu_sched_tsg_timeslice_args) # type: ignore
NVGPU_SCHED_IOCTL_TSG_SET_RUNLIST_INTERLEAVE = _IOW(NVGPU_SCHED_IOCTL_MAGIC, 6, struct_nvgpu_sched_tsg_runlist_interleave_args) # type: ignore
NVGPU_SCHED_IOCTL_LOCK_CONTROL = _IO(NVGPU_SCHED_IOCTL_MAGIC, 7) # type: ignore
NVGPU_SCHED_IOCTL_UNLOCK_CONTROL = _IO(NVGPU_SCHED_IOCTL_MAGIC, 8) # type: ignore
NVGPU_SCHED_IOCTL_GET_API_VERSION = _IOR(NVGPU_SCHED_IOCTL_MAGIC, 9, struct_nvgpu_sched_api_version_args) # type: ignore
NVGPU_SCHED_IOCTL_GET_TSG = _IOW(NVGPU_SCHED_IOCTL_MAGIC, 10, struct_nvgpu_sched_tsg_refcount_args) # type: ignore
NVGPU_SCHED_IOCTL_PUT_TSG = _IOW(NVGPU_SCHED_IOCTL_MAGIC, 11, struct_nvgpu_sched_tsg_refcount_args) # type: ignore
NVGPU_SCHED_STATUS_TSG_OPEN = (1 << 0) # type: ignore
NVGPU_SCHED_API_VERSION = 1 # type: ignore
NVGPU_GPU_IOCTL_MAGIC = 'G' # type: ignore
NVGPU_ZBC_COLOR_VALUE_SIZE = 4 # type: ignore
NVGPU_ZBC_TYPE_INVALID = 0 # type: ignore
NVGPU_ZBC_TYPE_COLOR = 1 # type: ignore
NVGPU_ZBC_TYPE_DEPTH = 2 # type: ignore
NVGPU_ZBC_TYPE_STENCIL = 3 # type: ignore
NVGPU_GPU_ARCH_GK100 = 0x000000E0 # type: ignore
NVGPU_GPU_ARCH_GM200 = 0x00000120 # type: ignore
NVGPU_GPU_ARCH_GP100 = 0x00000130 # type: ignore
NVGPU_GPU_ARCH_GV110 = 0x00000150 # type: ignore
NVGPU_GPU_ARCH_GV100 = 0x00000140 # type: ignore
NVGPU_GPU_IMPL_GK20A = 0x0000000A # type: ignore
NVGPU_GPU_IMPL_GM204 = 0x00000004 # type: ignore
NVGPU_GPU_IMPL_GM206 = 0x00000006 # type: ignore
NVGPU_GPU_IMPL_GM20B = 0x0000000B # type: ignore
NVGPU_GPU_IMPL_GM20B_B = 0x0000000E # type: ignore
NVGPU_GPU_IMPL_GP104 = 0x00000004 # type: ignore
NVGPU_GPU_IMPL_GP106 = 0x00000006 # type: ignore
NVGPU_GPU_IMPL_GP10B = 0x0000000B # type: ignore
NVGPU_GPU_IMPL_GV11B = 0x0000000B # type: ignore
NVGPU_GPU_IMPL_GV100 = 0x00000000 # type: ignore
NVGPU_GPU_BUS_TYPE_NONE = 0 # type: ignore
NVGPU_GPU_BUS_TYPE_AXI = 32 # type: ignore
NVGPU_GPU_FLAGS_HAS_SYNCPOINTS = (1 << 0) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SPARSE_ALLOCS = (1 << 2) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SYNC_FENCE_FDS = (1 << 3) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_CYCLE_STATS = (1 << 4) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_CYCLE_STATS_SNAPSHOT = (1 << 6) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_TSG = (1 << 8) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_CLOCK_CONTROLS = (1 << 9) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_GET_VOLTAGE = (1 << 10) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_GET_CURRENT = (1 << 11) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_GET_POWER = (1 << 12) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_GET_TEMPERATURE = (1 << 13) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SET_THERM_ALERT_LIMIT = (1 << 14) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_DEVICE_EVENTS = (1 << 15) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_FECS_CTXSW_TRACE = (1 << 16) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_MAP_COMPBITS = (1 << 17) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_SUBMIT_NO_JOBTRACKING = (1 << 18) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_SUBMIT_FULL = (1 << 19) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_IO_COHERENCE = (1 << 20) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_RESCHEDULE_RUNLIST = (1 << 21) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_TSG_SUBCONTEXTS = (1 << 22) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_DETERMINISTIC_OPTS = (1 << 24) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SCG = (1 << 25) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SYNCPOINT_ADDRESS = (1 << 26) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_VPR = (1 << 27) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_USER_SYNCPOINT = (1 << 28) # type: ignore
NVGPU_GPU_FLAGS_CAN_RAILGATE = (1 << 29) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_USERMODE_SUBMIT = (1 << 30) # type: ignore
NVGPU_GPU_FLAGS_DRIVER_REDUCED_PROFILE = (1 << 31) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SET_CTX_MMU_DEBUG_MODE = (1 << 32) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_FAULT_RECOVERY = (1 << 33) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_MAPPING_MODIFY = (1 << 34) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_REMAP = (1 << 35) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_COMPRESSION = (1 << 36) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SM_TTU = (1 << 37) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_POST_L2_COMPRESSION = (1 << 38) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_MAP_ACCESS_TYPE = (1 << 39) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_2D = (1 << 40) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_3D = (1 << 41) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_COMPUTE = (1 << 42) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_I2M = (1 << 43) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_ZBC = (1 << 44) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_PROFILER_V2_DEVICE = (1 << 46) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_PROFILER_V2_CONTEXT = (1 << 47) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_SMPC_GLOBAL_MODE = (1 << 48) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_GET_GR_CONTEXT = (1 << 49) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_BUFFER_METADATA = (1 << 50) # type: ignore
NVGPU_GPU_FLAGS_L2_MAX_WAYS_EVICT_LAST_ENABLED = (1 << 51) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_VAB = (1 << 52) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_NVS = (1 << 53) # type: ignore
NVGPU_GPU_FLAGS_SCHED_EXIT_WAIT_FOR_ERRBAR_SUPPORTED = (1 << 55) # type: ignore
NVGPU_GPU_FLAGS_MULTI_PROCESS_TSG_SHARING = (1 << 56) # type: ignore
NVGPU_GPU_FLAGS_ECC_ENABLED_SM_LRF = (1 << 60) # type: ignore
NVGPU_GPU_FLAGS_SUPPORT_GPU_MMIO = (1 << 57) # type: ignore
NVGPU_GPU_FLAGS_ECC_ENABLED_SM_SHM = (1 << 61) # type: ignore
NVGPU_GPU_FLAGS_ECC_ENABLED_TEX = (1 << 62) # type: ignore
NVGPU_GPU_FLAGS_ECC_ENABLED_LTC = (1 << 63) # type: ignore
NVGPU_GPU_FLAGS_ALL_ECC_ENABLED = (NVGPU_GPU_FLAGS_ECC_ENABLED_SM_LRF | NVGPU_GPU_FLAGS_ECC_ENABLED_SM_SHM | NVGPU_GPU_FLAGS_ECC_ENABLED_TEX    | NVGPU_GPU_FLAGS_ECC_ENABLED_LTC) # type: ignore
NVGPU_GPU_CHARACTERISTICS_NO_NUMA_INFO = (-1) # type: ignore
NVGPU_GPU_COMPBITS_NONE = 0 # type: ignore
NVGPU_GPU_COMPBITS_GPU = (1 << 0) # type: ignore
NVGPU_GPU_COMPBITS_CDEH = (1 << 1) # type: ignore
NVGPU_GPU_COMPBITS_CDEV = (1 << 2) # type: ignore
NVGPU_GPU_IOCTL_ALLOC_AS_FLAGS_UNIFIED_VA = (1 << 1) # type: ignore
NVGPU_GPU_BUFFER_INFO_FLAGS_METADATA_REGISTERED = (1 << 0) # type: ignore
NVGPU_GPU_BUFFER_INFO_FLAGS_COMPTAGS_ALLOCATED = (1 << 1) # type: ignore
NVGPU_GPU_BUFFER_INFO_FLAGS_MUTABLE_METADATA = (1 << 2) # type: ignore
NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_MAX_COUNT = 16 # type: ignore
NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_SRC_ID_TSC = 1 # type: ignore
NVGPU_GPU_GET_CPU_TIME_CORRELATION_INFO_SRC_ID_OSTIME = 2 # type: ignore
NVGPU_GPU_ENGINE_ID_GR = 0 # type: ignore
NVGPU_GPU_ENGINE_ID_GR_COPY = 1 # type: ignore
NVGPU_GPU_ENGINE_ID_ASYNC_COPY = 2 # type: ignore
NVGPU_GPU_ENGINE_ID_NVENC = 5 # type: ignore
NVGPU_GPU_ENGINE_ID_OFA = 6 # type: ignore
NVGPU_GPU_ENGINE_ID_NVDEC = 7 # type: ignore
NVGPU_GPU_ENGINE_ID_NVJPG = 8 # type: ignore
NVGPU_GPU_ALLOC_VIDMEM_FLAG_CONTIGUOUS = (1 << 0) # type: ignore
NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_NOT_MAPPABLE = (0 << 1) # type: ignore
NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_WRITE_COMBINE = (1 << 1) # type: ignore
NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_CACHED = (2 << 1) # type: ignore
NVGPU_GPU_ALLOC_VIDMEM_FLAG_CPU_MASK = (7 << 1) # type: ignore
NVGPU_GPU_ALLOC_VIDMEM_FLAG_VPR = (1 << 4) # type: ignore
NVGPU_GPU_CLK_DOMAIN_MCLK = (0) # type: ignore
NVGPU_GPU_CLK_DOMAIN_GPCCLK = (1) # type: ignore
NVGPU_GPU_CLK_FLAG_SPECIFIC_DOMAINS = (1 << 0) # type: ignore
NVGPU_GPU_CLK_TYPE_TARGET = 1 # type: ignore
NVGPU_GPU_CLK_TYPE_ACTUAL = 2 # type: ignore
NVGPU_GPU_CLK_TYPE_EFFECTIVE = 3 # type: ignore
NVGPU_GPU_VOLTAGE_CORE = 1 # type: ignore
NVGPU_GPU_VOLTAGE_SRAM = 2 # type: ignore
NVGPU_GPU_VOLTAGE_BUS = 3 # type: ignore
NVGPU_GPU_SET_DETERMINISTIC_OPTS_FLAGS_ALLOW_RAILGATING = (1 << 0) # type: ignore
NVGPU_GPU_SET_DETERMINISTIC_OPTS_FLAGS_DISALLOW_RAILGATING = (1 << 1) # type: ignore
NVGPU_GPU_COMPTAGS_ALLOC_NONE = 0 # type: ignore
NVGPU_GPU_COMPTAGS_ALLOC_REQUESTED = 1 # type: ignore
NVGPU_GPU_COMPTAGS_ALLOC_REQUIRED = 2 # type: ignore
NVGPU_GPU_REGISTER_BUFFER_FLAGS_COMPTAGS_ALLOCATED = (1 << 0) # type: ignore
NVGPU_GPU_REGISTER_BUFFER_FLAGS_MUTABLE = (1 << 1) # type: ignore
NVGPU_GPU_REGISTER_BUFFER_FLAGS_MODIFY = (1 << 2) # type: ignore
NVGPU_GPU_REGISTER_BUFFER_METADATA_MAX_SIZE = 256 # type: ignore
NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE = _IOR(NVGPU_GPU_IOCTL_MAGIC, 1, struct_nvgpu_gpu_zcull_get_ctx_size_args) # type: ignore
NVGPU_GPU_IOCTL_ZCULL_GET_INFO = _IOR(NVGPU_GPU_IOCTL_MAGIC, 2, struct_nvgpu_gpu_zcull_get_info_args) # type: ignore
NVGPU_GPU_IOCTL_ZBC_SET_TABLE = _IOW(NVGPU_GPU_IOCTL_MAGIC, 3, struct_nvgpu_gpu_zbc_set_table_args) # type: ignore
NVGPU_GPU_IOCTL_ZBC_QUERY_TABLE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 4, struct_nvgpu_gpu_zbc_query_table_args) # type: ignore
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 5, struct_nvgpu_gpu_get_characteristics) # type: ignore
NVGPU_GPU_IOCTL_PREPARE_COMPRESSIBLE_READ = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 6, struct_nvgpu_gpu_prepare_compressible_read_args) # type: ignore
NVGPU_GPU_IOCTL_MARK_COMPRESSIBLE_WRITE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 7, struct_nvgpu_gpu_mark_compressible_write_args) # type: ignore
NVGPU_GPU_IOCTL_ALLOC_AS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 8, struct_nvgpu_alloc_as_args) # type: ignore
NVGPU_GPU_IOCTL_OPEN_TSG = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 9, struct_nvgpu_gpu_open_tsg_args) # type: ignore
NVGPU_GPU_IOCTL_GET_TPC_MASKS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 10, struct_nvgpu_gpu_get_tpc_masks_args) # type: ignore
NVGPU_GPU_IOCTL_OPEN_CHANNEL = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 11, struct_nvgpu_gpu_open_channel_args) # type: ignore
NVGPU_GPU_IOCTL_FLUSH_L2 = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 12, struct_nvgpu_gpu_l2_fb_args) # type: ignore
NVGPU_GPU_IOCTL_SET_MMUDEBUG_MODE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 14, struct_nvgpu_gpu_mmu_debug_mode_args) # type: ignore
NVGPU_GPU_IOCTL_SET_SM_DEBUG_MODE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 15, struct_nvgpu_gpu_sm_debug_mode_args) # type: ignore
NVGPU_GPU_IOCTL_WAIT_FOR_PAUSE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 16, struct_nvgpu_gpu_wait_pause_args) # type: ignore
NVGPU_GPU_IOCTL_GET_TPC_EXCEPTION_EN_STATUS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 17, struct_nvgpu_gpu_tpc_exception_en_status_args) # type: ignore
NVGPU_GPU_IOCTL_NUM_VSMS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 18, struct_nvgpu_gpu_num_vsms) # type: ignore
NVGPU_GPU_IOCTL_VSMS_MAPPING = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 19, struct_nvgpu_gpu_vsms_mapping) # type: ignore
NVGPU_GPU_IOCTL_RESUME_FROM_PAUSE = _IO(NVGPU_GPU_IOCTL_MAGIC, 21) # type: ignore
NVGPU_GPU_IOCTL_TRIGGER_SUSPEND = _IO(NVGPU_GPU_IOCTL_MAGIC, 22) # type: ignore
NVGPU_GPU_IOCTL_CLEAR_SM_ERRORS = _IO(NVGPU_GPU_IOCTL_MAGIC, 23) # type: ignore
NVGPU_GPU_IOCTL_GET_CPU_TIME_CORRELATION_INFO = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 24, struct_nvgpu_gpu_get_cpu_time_correlation_info_args) # type: ignore
NVGPU_GPU_IOCTL_GET_GPU_TIME = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 25, struct_nvgpu_gpu_get_gpu_time_args) # type: ignore
NVGPU_GPU_IOCTL_GET_ENGINE_INFO = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 26, struct_nvgpu_gpu_get_engine_info_args) # type: ignore
NVGPU_GPU_IOCTL_ALLOC_VIDMEM = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 27, struct_nvgpu_gpu_alloc_vidmem_args) # type: ignore
NVGPU_GPU_IOCTL_CLK_GET_RANGE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 28, struct_nvgpu_gpu_clk_range_args) # type: ignore
NVGPU_GPU_IOCTL_CLK_GET_VF_POINTS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 29, struct_nvgpu_gpu_clk_vf_points_args) # type: ignore
NVGPU_GPU_IOCTL_CLK_GET_INFO = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 30, struct_nvgpu_gpu_clk_get_info_args) # type: ignore
NVGPU_GPU_IOCTL_CLK_SET_INFO = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 31, struct_nvgpu_gpu_clk_set_info_args) # type: ignore
NVGPU_GPU_IOCTL_GET_EVENT_FD = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 32, struct_nvgpu_gpu_get_event_fd_args) # type: ignore
NVGPU_GPU_IOCTL_GET_MEMORY_STATE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 33, struct_nvgpu_gpu_get_memory_state_args) # type: ignore
NVGPU_GPU_IOCTL_GET_VOLTAGE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 34, struct_nvgpu_gpu_get_voltage_args) # type: ignore
NVGPU_GPU_IOCTL_GET_CURRENT = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 35, struct_nvgpu_gpu_get_current_args) # type: ignore
NVGPU_GPU_IOCTL_GET_POWER = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 36, struct_nvgpu_gpu_get_power_args) # type: ignore
NVGPU_GPU_IOCTL_GET_TEMPERATURE = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 37, struct_nvgpu_gpu_get_temperature_args) # type: ignore
NVGPU_GPU_IOCTL_GET_FBP_L2_MASKS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 38, struct_nvgpu_gpu_get_fbp_l2_masks_args) # type: ignore
NVGPU_GPU_IOCTL_SET_THERM_ALERT_LIMIT = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 39, struct_nvgpu_gpu_set_therm_alert_limit_args) # type: ignore
NVGPU_GPU_IOCTL_SET_DETERMINISTIC_OPTS = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 40, struct_nvgpu_gpu_set_deterministic_opts_args) # type: ignore
NVGPU_GPU_IOCTL_REGISTER_BUFFER = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 41, struct_nvgpu_gpu_register_buffer_args) # type: ignore
NVGPU_GPU_IOCTL_GET_BUFFER_INFO = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 42, struct_nvgpu_gpu_get_buffer_info_args) # type: ignore
NVGPU_GPU_IOCTL_GET_GPC_LOCAL_TO_PHYSICAL_MAP = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 43, struct_nvgpu_gpu_get_gpc_physical_map_args) # type: ignore
NVGPU_GPU_IOCTL_GET_GPC_LOCAL_TO_LOGICAL_MAP = _IOWR(NVGPU_GPU_IOCTL_MAGIC, 44, struct_nvgpu_gpu_get_gpc_logical_map_args) # type: ignore
NVGPU_AS_IOCTL_MAGIC = 'A' # type: ignore
NVGPU_AS_ALLOC_SPACE_FLAGS_FIXED_OFFSET = 0x1 # type: ignore
NVGPU_AS_ALLOC_SPACE_FLAGS_SPARSE = 0x2 # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_FIXED_OFFSET = (1 << 0) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_CACHEABLE = (1 << 2) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_UNMAPPED_PTE = (1 << 5) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_MAPPABLE_COMPBITS = (1 << 6) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_L3_ALLOC = (1 << 7) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_SYSTEM_COHERENT = (1 << 9) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_TEGRA_RAW = (1 << 12) # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_ACCESS_BITMASK_OFFSET = 10 # type: ignore
NVGPU_AS_MAP_BUFFER_FLAGS_ACCESS_BITMASK_SIZE = 2 # type: ignore
NVGPU_AS_MAP_BUFFER_ACCESS_DEFAULT = 0 # type: ignore
NVGPU_AS_MAP_BUFFER_ACCESS_READ_ONLY = 1 # type: ignore
NVGPU_AS_MAP_BUFFER_ACCESS_READ_WRITE = 2 # type: ignore
NV_KIND_INVALID = -1 # type: ignore
NVGPU_AS_GET_BUFFER_COMPBITS_INFO_FLAGS_HAS_COMPBITS = (1 << 0) # type: ignore
NVGPU_AS_GET_BUFFER_COMPBITS_INFO_FLAGS_MAPPABLE = (1 << 1) # type: ignore
NVGPU_AS_GET_BUFFER_COMPBITS_INFO_FLAGS_DISCONTIG_IOVA = (1 << 2) # type: ignore
NVGPU_AS_MAP_BUFFER_COMPBITS_FLAGS_FIXED_OFFSET = (1 << 0) # type: ignore
NVGPU_AS_REMAP_OP_FLAGS_CACHEABLE = (1 << 2) # type: ignore
NVGPU_AS_REMAP_OP_FLAGS_ACCESS_NO_WRITE = (1 << 10) # type: ignore
NVGPU_AS_REMAP_OP_FLAGS_PAGESIZE_4K = (1 << 15) # type: ignore
NVGPU_AS_REMAP_OP_FLAGS_PAGESIZE_64K = (1 << 16) # type: ignore
NVGPU_AS_REMAP_OP_FLAGS_PAGESIZE_128K = (1 << 17) # type: ignore
NVGPU_AS_IOCTL_BIND_CHANNEL = _IOWR(NVGPU_AS_IOCTL_MAGIC, 1, struct_nvgpu_as_bind_channel_args) # type: ignore
NVGPU32_AS_IOCTL_ALLOC_SPACE = _IOWR(NVGPU_AS_IOCTL_MAGIC, 2, struct_nvgpu32_as_alloc_space_args) # type: ignore
NVGPU_AS_IOCTL_FREE_SPACE = _IOWR(NVGPU_AS_IOCTL_MAGIC, 3, struct_nvgpu_as_free_space_args) # type: ignore
NVGPU_AS_IOCTL_UNMAP_BUFFER = _IOWR(NVGPU_AS_IOCTL_MAGIC, 5, struct_nvgpu_as_unmap_buffer_args) # type: ignore
NVGPU_AS_IOCTL_ALLOC_SPACE = _IOWR(NVGPU_AS_IOCTL_MAGIC, 6, struct_nvgpu_as_alloc_space_args) # type: ignore
NVGPU_AS_IOCTL_MAP_BUFFER_EX = _IOWR(NVGPU_AS_IOCTL_MAGIC, 7, struct_nvgpu_as_map_buffer_ex_args) # type: ignore
NVGPU_AS_IOCTL_GET_VA_REGIONS = _IOWR(NVGPU_AS_IOCTL_MAGIC, 8, struct_nvgpu_as_get_va_regions_args) # type: ignore
NVGPU_AS_IOCTL_GET_BUFFER_COMPBITS_INFO = _IOWR(NVGPU_AS_IOCTL_MAGIC, 9, struct_nvgpu_as_get_buffer_compbits_info_args) # type: ignore
NVGPU_AS_IOCTL_MAP_BUFFER_COMPBITS = _IOWR(NVGPU_AS_IOCTL_MAGIC, 10, struct_nvgpu_as_map_buffer_compbits_args) # type: ignore
NVGPU_AS_IOCTL_MAP_BUFFER_BATCH = _IOWR(NVGPU_AS_IOCTL_MAGIC, 11, struct_nvgpu_as_map_buffer_batch_args) # type: ignore
NVGPU_AS_IOCTL_GET_SYNC_RO_MAP = _IOR(NVGPU_AS_IOCTL_MAGIC,  12, struct_nvgpu_as_get_sync_ro_map_args) # type: ignore
NVGPU_AS_IOCTL_MAPPING_MODIFY = _IOWR(NVGPU_AS_IOCTL_MAGIC,  13, struct_nvgpu_as_mapping_modify_args) # type: ignore
NVGPU_AS_IOCTL_REMAP = _IOWR(NVGPU_AS_IOCTL_MAGIC, 14, struct_nvgpu_as_remap_args) # type: ignore
NVMAP_ELEM_SIZE_U64 = (1 << 31) # type: ignore
NVMAP_IOC_MAGIC = 'N' # type: ignore
NVMAP_IOC_CREATE = _IOWR(NVMAP_IOC_MAGIC, 0, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_CREATE_64 = _IOWR(NVMAP_IOC_MAGIC, 1, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_FROM_ID = _IOWR(NVMAP_IOC_MAGIC, 2, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_ALLOC = _IOW(NVMAP_IOC_MAGIC, 3, struct_nvmap_alloc_handle) # type: ignore
NVMAP_IOC_FREE = _IO(NVMAP_IOC_MAGIC, 4) # type: ignore
NVMAP_IOC_WRITE = _IOW(NVMAP_IOC_MAGIC, 6, struct_nvmap_rw_handle) # type: ignore
NVMAP_IOC_READ = _IOW(NVMAP_IOC_MAGIC, 7, struct_nvmap_rw_handle) # type: ignore
NVMAP_IOC_PARAM = _IOWR(NVMAP_IOC_MAGIC, 8, struct_nvmap_handle_param) # type: ignore
NVMAP_IOC_CACHE = _IOW(NVMAP_IOC_MAGIC, 12, struct_nvmap_cache_op) # type: ignore
NVMAP_IOC_CACHE_64 = _IOW(NVMAP_IOC_MAGIC, 12, struct_nvmap_cache_op_64) # type: ignore
NVMAP_IOC_GET_ID = _IOWR(NVMAP_IOC_MAGIC, 13, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_GET_FD = _IOWR(NVMAP_IOC_MAGIC, 15, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_FROM_FD = _IOWR(NVMAP_IOC_MAGIC, 16, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_CACHE_LIST = _IOW(NVMAP_IOC_MAGIC, 17, struct_nvmap_cache_op_list) # type: ignore
NVMAP_IOC_FROM_IVC_ID = _IOWR(NVMAP_IOC_MAGIC, 19, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_GET_IVC_ID = _IOWR(NVMAP_IOC_MAGIC, 20, struct_nvmap_create_handle) # type: ignore
NVMAP_IOC_FROM_VA = _IOWR(NVMAP_IOC_MAGIC, 22, struct_nvmap_create_handle_from_va) # type: ignore
NVMAP_IOC_GUP_TEST = _IOWR(NVMAP_IOC_MAGIC, 23, struct_nvmap_gup_test) # type: ignore
NVMAP_IOC_SET_TAG_LABEL = _IOW(NVMAP_IOC_MAGIC, 24, struct_nvmap_set_tag_label) # type: ignore
NVMAP_IOC_GET_AVAILABLE_HEAPS = _IOR(NVMAP_IOC_MAGIC, 25, struct_nvmap_available_heaps) # type: ignore
NVMAP_IOC_GET_HEAP_SIZE = _IOR(NVMAP_IOC_MAGIC, 26, struct_nvmap_heap_size) # type: ignore
NVMAP_IOC_PARAMETERS = _IOR(NVMAP_IOC_MAGIC, 27, struct_nvmap_handle_parameters) # type: ignore
NVMAP_IOC_ALLOC_IVM = _IOW(NVMAP_IOC_MAGIC, 101, struct_nvmap_alloc_ivm_handle) # type: ignore
NVMAP_IOC_GET_SCIIPCID = _IOR(NVMAP_IOC_MAGIC, 103, struct_nvmap_sciipc_map) # type: ignore
NVMAP_IOC_HANDLE_FROM_SCIIPCID = _IOR(NVMAP_IOC_MAGIC, 104, struct_nvmap_sciipc_map) # type: ignore
NVMAP_IOC_QUERY_HEAP_PARAMS = _IOR(NVMAP_IOC_MAGIC, 105, struct_nvmap_query_heap_params) # type: ignore
NVMAP_IOC_DUP_HANDLE = _IOWR(NVMAP_IOC_MAGIC, 106, struct_nvmap_duplicate_handle) # type: ignore
NVMAP_IOC_GET_FD_FOR_RANGE_FROM_LIST = _IOR(NVMAP_IOC_MAGIC, 107, struct_nvmap_fd_for_range_from_list) # type: ignore
NVMAP_HEAP_IOVMM = (1 << 30) # type: ignore
NVMAP_HANDLE_UNCACHEABLE = (0 << 0) # type: ignore
NVMAP_HANDLE_WRITE_COMBINE = (1 << 0) # type: ignore
NVMAP_HANDLE_INNER_CACHEABLE = (2 << 0) # type: ignore
NVMAP_HANDLE_CACHEABLE = (3 << 0) # type: ignore