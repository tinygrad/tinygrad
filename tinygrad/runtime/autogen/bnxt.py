# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_hwrm_cmd_hdr(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_cmd_hdr.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_resp_hdr(c.Struct):
  SIZE = 8
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
struct_hwrm_resp_hdr.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6)])
@c.record
class struct_tlv(c.Struct):
  SIZE = 8
  cmd_discr: int
  reserved_8b: int
  flags: int
  tlv_type: int
  length: int
struct_tlv.register_fields([('cmd_discr', ctypes.c_uint16, 0), ('reserved_8b', ctypes.c_ubyte, 2), ('flags', ctypes.c_ubyte, 3), ('tlv_type', ctypes.c_uint16, 4), ('length', ctypes.c_uint16, 6)])
@c.record
class struct_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_output(c.Struct):
  SIZE = 8
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
struct_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6)])
@c.record
class struct_hwrm_short_input(c.Struct):
  SIZE = 16
  req_type: int
  signature: int
  target_id: int
  size: int
  req_addr: int
struct_hwrm_short_input.register_fields([('req_type', ctypes.c_uint16, 0), ('signature', ctypes.c_uint16, 2), ('target_id', ctypes.c_uint16, 4), ('size', ctypes.c_uint16, 6), ('req_addr', ctypes.c_uint64, 8)])
@c.record
class struct_cmd_nums(c.Struct):
  SIZE = 8
  req_type: int
  unused_0: c.Array[ctypes.c_uint16, Literal[3]]
struct_cmd_nums.register_fields([('req_type', ctypes.c_uint16, 0), ('unused_0', c.Array[ctypes.c_uint16, Literal[3]], 2)])
@c.record
class struct_ret_codes(c.Struct):
  SIZE = 8
  error_code: int
  unused_0: c.Array[ctypes.c_uint16, Literal[3]]
struct_ret_codes.register_fields([('error_code', ctypes.c_uint16, 0), ('unused_0', c.Array[ctypes.c_uint16, Literal[3]], 2)])
@c.record
class struct_hwrm_err_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  opaque_0: int
  opaque_1: int
  cmd_err: int
  valid: int
struct_hwrm_err_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('opaque_0', ctypes.c_uint32, 8), ('opaque_1', ctypes.c_uint16, 12), ('cmd_err', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ver_get_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  hwrm_intf_maj: int
  hwrm_intf_min: int
  hwrm_intf_upd: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
struct_hwrm_ver_get_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('hwrm_intf_maj', ctypes.c_ubyte, 16), ('hwrm_intf_min', ctypes.c_ubyte, 17), ('hwrm_intf_upd', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19)])
@c.record
class struct_hwrm_ver_get_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  hwrm_intf_maj_8b: int
  hwrm_intf_min_8b: int
  hwrm_intf_upd_8b: int
  hwrm_intf_rsvd_8b: int
  hwrm_fw_maj_8b: int
  hwrm_fw_min_8b: int
  hwrm_fw_bld_8b: int
  hwrm_fw_rsvd_8b: int
  mgmt_fw_maj_8b: int
  mgmt_fw_min_8b: int
  mgmt_fw_bld_8b: int
  mgmt_fw_rsvd_8b: int
  netctrl_fw_maj_8b: int
  netctrl_fw_min_8b: int
  netctrl_fw_bld_8b: int
  netctrl_fw_rsvd_8b: int
  dev_caps_cfg: int
  roce_fw_maj_8b: int
  roce_fw_min_8b: int
  roce_fw_bld_8b: int
  roce_fw_rsvd_8b: int
  hwrm_fw_name: c.Array[ctypes.c_char, Literal[16]]
  mgmt_fw_name: c.Array[ctypes.c_char, Literal[16]]
  netctrl_fw_name: c.Array[ctypes.c_char, Literal[16]]
  active_pkg_name: c.Array[ctypes.c_char, Literal[16]]
  roce_fw_name: c.Array[ctypes.c_char, Literal[16]]
  chip_num: int
  chip_rev: int
  chip_metal: int
  chip_bond_id: int
  chip_platform_type: int
  max_req_win_len: int
  max_resp_len: int
  def_req_timeout: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  always_1: int
  hwrm_intf_major: int
  hwrm_intf_minor: int
  hwrm_intf_build: int
  hwrm_intf_patch: int
  hwrm_fw_major: int
  hwrm_fw_minor: int
  hwrm_fw_build: int
  hwrm_fw_patch: int
  mgmt_fw_major: int
  mgmt_fw_minor: int
  mgmt_fw_build: int
  mgmt_fw_patch: int
  netctrl_fw_major: int
  netctrl_fw_minor: int
  netctrl_fw_build: int
  netctrl_fw_patch: int
  roce_fw_major: int
  roce_fw_minor: int
  roce_fw_build: int
  roce_fw_patch: int
  max_ext_req_len: int
  max_req_timeout: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_ver_get_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('hwrm_intf_maj_8b', ctypes.c_ubyte, 8), ('hwrm_intf_min_8b', ctypes.c_ubyte, 9), ('hwrm_intf_upd_8b', ctypes.c_ubyte, 10), ('hwrm_intf_rsvd_8b', ctypes.c_ubyte, 11), ('hwrm_fw_maj_8b', ctypes.c_ubyte, 12), ('hwrm_fw_min_8b', ctypes.c_ubyte, 13), ('hwrm_fw_bld_8b', ctypes.c_ubyte, 14), ('hwrm_fw_rsvd_8b', ctypes.c_ubyte, 15), ('mgmt_fw_maj_8b', ctypes.c_ubyte, 16), ('mgmt_fw_min_8b', ctypes.c_ubyte, 17), ('mgmt_fw_bld_8b', ctypes.c_ubyte, 18), ('mgmt_fw_rsvd_8b', ctypes.c_ubyte, 19), ('netctrl_fw_maj_8b', ctypes.c_ubyte, 20), ('netctrl_fw_min_8b', ctypes.c_ubyte, 21), ('netctrl_fw_bld_8b', ctypes.c_ubyte, 22), ('netctrl_fw_rsvd_8b', ctypes.c_ubyte, 23), ('dev_caps_cfg', ctypes.c_uint32, 24), ('roce_fw_maj_8b', ctypes.c_ubyte, 28), ('roce_fw_min_8b', ctypes.c_ubyte, 29), ('roce_fw_bld_8b', ctypes.c_ubyte, 30), ('roce_fw_rsvd_8b', ctypes.c_ubyte, 31), ('hwrm_fw_name', c.Array[ctypes.c_char, Literal[16]], 32), ('mgmt_fw_name', c.Array[ctypes.c_char, Literal[16]], 48), ('netctrl_fw_name', c.Array[ctypes.c_char, Literal[16]], 64), ('active_pkg_name', c.Array[ctypes.c_char, Literal[16]], 80), ('roce_fw_name', c.Array[ctypes.c_char, Literal[16]], 96), ('chip_num', ctypes.c_uint16, 112), ('chip_rev', ctypes.c_ubyte, 114), ('chip_metal', ctypes.c_ubyte, 115), ('chip_bond_id', ctypes.c_ubyte, 116), ('chip_platform_type', ctypes.c_ubyte, 117), ('max_req_win_len', ctypes.c_uint16, 118), ('max_resp_len', ctypes.c_uint16, 120), ('def_req_timeout', ctypes.c_uint16, 122), ('flags', ctypes.c_ubyte, 124), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 125), ('always_1', ctypes.c_ubyte, 127), ('hwrm_intf_major', ctypes.c_uint16, 128), ('hwrm_intf_minor', ctypes.c_uint16, 130), ('hwrm_intf_build', ctypes.c_uint16, 132), ('hwrm_intf_patch', ctypes.c_uint16, 134), ('hwrm_fw_major', ctypes.c_uint16, 136), ('hwrm_fw_minor', ctypes.c_uint16, 138), ('hwrm_fw_build', ctypes.c_uint16, 140), ('hwrm_fw_patch', ctypes.c_uint16, 142), ('mgmt_fw_major', ctypes.c_uint16, 144), ('mgmt_fw_minor', ctypes.c_uint16, 146), ('mgmt_fw_build', ctypes.c_uint16, 148), ('mgmt_fw_patch', ctypes.c_uint16, 150), ('netctrl_fw_major', ctypes.c_uint16, 152), ('netctrl_fw_minor', ctypes.c_uint16, 154), ('netctrl_fw_build', ctypes.c_uint16, 156), ('netctrl_fw_patch', ctypes.c_uint16, 158), ('roce_fw_major', ctypes.c_uint16, 160), ('roce_fw_minor', ctypes.c_uint16, 162), ('roce_fw_build', ctypes.c_uint16, 164), ('roce_fw_patch', ctypes.c_uint16, 166), ('max_ext_req_len', ctypes.c_uint16, 168), ('max_req_timeout', ctypes.c_uint16, 170), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 172), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_eject_cmpl(c.Struct):
  SIZE = 16
  type: int
  len: int
  opaque: int
  v: int
  reserved16: int
  unused_2: int
struct_eject_cmpl.register_fields([('type', ctypes.c_uint16, 0), ('len', ctypes.c_uint16, 2), ('opaque', ctypes.c_uint32, 4), ('v', ctypes.c_uint16, 8), ('reserved16', ctypes.c_uint16, 10), ('unused_2', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_cmpl(c.Struct):
  SIZE = 16
  type: int
  sequence_id: int
  unused_1: int
  v: int
  unused_3: int
struct_hwrm_cmpl.register_fields([('type', ctypes.c_uint16, 0), ('sequence_id', ctypes.c_uint16, 2), ('unused_1', ctypes.c_uint32, 4), ('v', ctypes.c_uint32, 8), ('unused_3', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_fwd_req_cmpl(c.Struct):
  SIZE = 16
  req_len_type: int
  source_id: int
  unused0: int
  req_buf_addr_v: c.Array[ctypes.c_uint32, Literal[2]]
struct_hwrm_fwd_req_cmpl.register_fields([('req_len_type', ctypes.c_uint16, 0), ('source_id', ctypes.c_uint16, 2), ('unused0', ctypes.c_uint32, 4), ('req_buf_addr_v', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_hwrm_fwd_resp_cmpl(c.Struct):
  SIZE = 16
  type: int
  source_id: int
  resp_len: int
  unused_1: int
  resp_buf_addr_v: c.Array[ctypes.c_uint32, Literal[2]]
struct_hwrm_fwd_resp_cmpl.register_fields([('type', ctypes.c_uint16, 0), ('source_id', ctypes.c_uint16, 2), ('resp_len', ctypes.c_uint16, 4), ('unused_1', ctypes.c_uint16, 6), ('resp_buf_addr_v', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_hwrm_async_event_cmpl(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_link_status_change(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_link_status_change.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_port_conn_not_allowed(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_port_conn_not_allowed.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_link_speed_cfg_change(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_link_speed_cfg_change.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_reset_notify(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_reset_notify.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_recovery(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_recovery.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_ring_monitor_msg(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_ring_monitor_msg.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_vf_cfg_change(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_vf_cfg_change.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_default_vnic_change(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_default_vnic_change.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_hw_flow_aged(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_hw_flow_aged.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_eem_cache_flush_req(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_eem_cache_flush_req.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_eem_cache_flush_done(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_eem_cache_flush_done.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_deferred_response(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_deferred_response.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_echo_request(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_echo_request.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_phc_update(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_phc_update.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_pps_timestamp(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_pps_timestamp.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_dbg_buf_producer(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_dbg_buf_producer.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_hwrm_error(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_hwrm_error.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_base(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_base.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_pause_storm(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_pause_storm.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_invalid_signal(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_invalid_signal.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_nvm(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_nvm.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_doorbell_drop_threshold(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_doorbell_drop_threshold.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_thermal(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_thermal.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_async_event_cmpl_error_report_dual_data_rate_not_supported(c.Struct):
  SIZE = 16
  type: int
  event_id: int
  event_data2: int
  opaque_v: int
  timestamp_lo: int
  timestamp_hi: int
  event_data1: int
struct_hwrm_async_event_cmpl_error_report_dual_data_rate_not_supported.register_fields([('type', ctypes.c_uint16, 0), ('event_id', ctypes.c_uint16, 2), ('event_data2', ctypes.c_uint32, 4), ('opaque_v', ctypes.c_ubyte, 8), ('timestamp_lo', ctypes.c_ubyte, 9), ('timestamp_hi', ctypes.c_uint16, 10), ('event_data1', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_func_reset_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  vf_id: int
  func_reset_level: int
  unused_0: int
struct_hwrm_func_reset_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('vf_id', ctypes.c_uint16, 20), ('func_reset_level', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_func_reset_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_reset_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_getfid_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  pci_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
struct_hwrm_func_getfid_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('pci_id', ctypes.c_uint16, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 22)])
@c.record
class struct_hwrm_func_getfid_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_func_getfid_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_vf_alloc_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  first_vf_id: int
  num_vfs: int
struct_hwrm_func_vf_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('first_vf_id', ctypes.c_uint16, 20), ('num_vfs', ctypes.c_uint16, 22)])
@c.record
class struct_hwrm_func_vf_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  first_vf_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_func_vf_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('first_vf_id', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_vf_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  first_vf_id: int
  num_vfs: int
struct_hwrm_func_vf_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('first_vf_id', ctypes.c_uint16, 20), ('num_vfs', ctypes.c_uint16, 22)])
@c.record
class struct_hwrm_func_vf_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_vf_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_vf_cfg_input(c.Struct):
  SIZE = 72
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  mtu: int
  guest_vlan: int
  async_event_cr: int
  dflt_mac_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  flags: int
  num_rsscos_ctxs: int
  num_cmpl_rings: int
  num_tx_rings: int
  num_rx_rings: int
  num_l2_ctxs: int
  num_vnics: int
  num_stat_ctxs: int
  num_hw_ring_grps: int
  num_ktls_tx_key_ctxs: int
  num_ktls_rx_key_ctxs: int
  num_msix: int
  unused: c.Array[ctypes.c_ubyte, Literal[2]]
  num_quic_tx_key_ctxs: int
  num_quic_rx_key_ctxs: int
struct_hwrm_func_vf_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('mtu', ctypes.c_uint16, 20), ('guest_vlan', ctypes.c_uint16, 22), ('async_event_cr', ctypes.c_uint16, 24), ('dflt_mac_addr', c.Array[ctypes.c_ubyte, Literal[6]], 26), ('flags', ctypes.c_uint32, 32), ('num_rsscos_ctxs', ctypes.c_uint16, 36), ('num_cmpl_rings', ctypes.c_uint16, 38), ('num_tx_rings', ctypes.c_uint16, 40), ('num_rx_rings', ctypes.c_uint16, 42), ('num_l2_ctxs', ctypes.c_uint16, 44), ('num_vnics', ctypes.c_uint16, 46), ('num_stat_ctxs', ctypes.c_uint16, 48), ('num_hw_ring_grps', ctypes.c_uint16, 50), ('num_ktls_tx_key_ctxs', ctypes.c_uint32, 52), ('num_ktls_rx_key_ctxs', ctypes.c_uint32, 56), ('num_msix', ctypes.c_uint16, 60), ('unused', c.Array[ctypes.c_ubyte, Literal[2]], 62), ('num_quic_tx_key_ctxs', ctypes.c_uint32, 64), ('num_quic_rx_key_ctxs', ctypes.c_uint32, 68)])
@c.record
class struct_hwrm_func_vf_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_vf_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_qcaps_output(c.Struct):
  SIZE = 144
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  port_id: int
  flags: int
  mac_address: c.Array[ctypes.c_ubyte, Literal[6]]
  max_rsscos_ctx: int
  max_cmpl_rings: int
  max_tx_rings: int
  max_rx_rings: int
  max_l2_ctxs: int
  max_vnics: int
  first_vf_id: int
  max_vfs: int
  max_stat_ctx: int
  max_encap_records: int
  max_decap_records: int
  max_tx_em_flows: int
  max_tx_wm_flows: int
  max_rx_em_flows: int
  max_rx_wm_flows: int
  max_mcast_filters: int
  max_flow_id: int
  max_hw_ring_grps: int
  max_sp_tx_rings: int
  max_msix_vfs: int
  flags_ext: int
  max_schqs: int
  mpc_chnls_cap: int
  max_key_ctxs_alloc: int
  flags_ext2: int
  tunnel_disable_flag: int
  xid_partition_cap: int
  device_serial_number: c.Array[ctypes.c_ubyte, Literal[8]]
  ctxs_per_partition: int
  max_tso_segs: int
  roce_vf_max_av: int
  roce_vf_max_cq: int
  roce_vf_max_mrw: int
  roce_vf_max_qp: int
  roce_vf_max_srq: int
  roce_vf_max_gid: int
  flags_ext3: int
  max_roce_vfs: int
  max_crypto_rx_flow_filters: int
  unused_3: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('port_id', ctypes.c_uint16, 10), ('flags', ctypes.c_uint32, 12), ('mac_address', c.Array[ctypes.c_ubyte, Literal[6]], 16), ('max_rsscos_ctx', ctypes.c_uint16, 22), ('max_cmpl_rings', ctypes.c_uint16, 24), ('max_tx_rings', ctypes.c_uint16, 26), ('max_rx_rings', ctypes.c_uint16, 28), ('max_l2_ctxs', ctypes.c_uint16, 30), ('max_vnics', ctypes.c_uint16, 32), ('first_vf_id', ctypes.c_uint16, 34), ('max_vfs', ctypes.c_uint16, 36), ('max_stat_ctx', ctypes.c_uint16, 38), ('max_encap_records', ctypes.c_uint32, 40), ('max_decap_records', ctypes.c_uint32, 44), ('max_tx_em_flows', ctypes.c_uint32, 48), ('max_tx_wm_flows', ctypes.c_uint32, 52), ('max_rx_em_flows', ctypes.c_uint32, 56), ('max_rx_wm_flows', ctypes.c_uint32, 60), ('max_mcast_filters', ctypes.c_uint32, 64), ('max_flow_id', ctypes.c_uint32, 68), ('max_hw_ring_grps', ctypes.c_uint32, 72), ('max_sp_tx_rings', ctypes.c_uint16, 76), ('max_msix_vfs', ctypes.c_uint16, 78), ('flags_ext', ctypes.c_uint32, 80), ('max_schqs', ctypes.c_ubyte, 84), ('mpc_chnls_cap', ctypes.c_ubyte, 85), ('max_key_ctxs_alloc', ctypes.c_uint16, 86), ('flags_ext2', ctypes.c_uint32, 88), ('tunnel_disable_flag', ctypes.c_uint16, 92), ('xid_partition_cap', ctypes.c_uint16, 94), ('device_serial_number', c.Array[ctypes.c_ubyte, Literal[8]], 96), ('ctxs_per_partition', ctypes.c_uint16, 104), ('max_tso_segs', ctypes.c_uint16, 106), ('roce_vf_max_av', ctypes.c_uint32, 108), ('roce_vf_max_cq', ctypes.c_uint32, 112), ('roce_vf_max_mrw', ctypes.c_uint32, 116), ('roce_vf_max_qp', ctypes.c_uint32, 120), ('roce_vf_max_srq', ctypes.c_uint32, 124), ('roce_vf_max_gid', ctypes.c_uint32, 128), ('flags_ext3', ctypes.c_uint32, 132), ('max_roce_vfs', ctypes.c_uint16, 136), ('max_crypto_rx_flow_filters', ctypes.c_uint16, 138), ('unused_3', c.Array[ctypes.c_ubyte, Literal[3]], 140), ('valid', ctypes.c_ubyte, 143)])
@c.record
class struct_hwrm_func_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_qcfg_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  port_id: int
  vlan: int
  flags: int
  mac_address: c.Array[ctypes.c_ubyte, Literal[6]]
  pci_id: int
  alloc_rsscos_ctx: int
  alloc_cmpl_rings: int
  alloc_tx_rings: int
  alloc_rx_rings: int
  alloc_l2_ctx: int
  alloc_vnics: int
  admin_mtu: int
  mru: int
  stat_ctx_id: int
  port_partition_type: int
  port_pf_cnt: int
  dflt_vnic_id: int
  max_mtu_configured: int
  min_bw: int
  max_bw: int
  evb_mode: int
  options: int
  alloc_vfs: int
  alloc_mcast_filters: int
  alloc_hw_ring_grps: int
  alloc_sp_tx_rings: int
  alloc_stat_ctx: int
  alloc_msix: int
  registered_vfs: int
  l2_doorbell_bar_size_kb: int
  active_endpoints: int
  always_1: int
  reset_addr_poll: int
  legacy_l2_db_size_kb: int
  svif_info: int
  mpc_chnls: int
  db_page_size: int
  roce_vnic_id: int
  partition_min_bw: int
  partition_max_bw: int
  host_mtu: int
  flags2: int
  stag_vid: int
  port_kdnet_mode: int
  kdnet_pcie_function: int
  port_kdnet_fid: int
  unused_5: int
  roce_bidi_opt_mode: int
  num_ktls_tx_key_ctxs: int
  num_ktls_rx_key_ctxs: int
  lag_id: int
  parif: int
  fw_lag_id: int
  unused_6: int
  num_quic_tx_key_ctxs: int
  num_quic_rx_key_ctxs: int
  roce_max_av_per_vf: int
  roce_max_cq_per_vf: int
  roce_max_mrw_per_vf: int
  roce_max_qp_per_vf: int
  roce_max_srq_per_vf: int
  roce_max_gid_per_vf: int
  xid_partition_cfg: int
  mirror_vnic_id: int
  max_link_width: int
  max_link_speed: int
  negotiated_link_width: int
  negotiated_link_speed: int
  unused_7: c.Array[ctypes.c_ubyte, Literal[2]]
  pcie_compliance: int
  unused_8: int
  l2_db_multi_page_size_kb: int
  unused_9: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_func_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('port_id', ctypes.c_uint16, 10), ('vlan', ctypes.c_uint16, 12), ('flags', ctypes.c_uint16, 14), ('mac_address', c.Array[ctypes.c_ubyte, Literal[6]], 16), ('pci_id', ctypes.c_uint16, 22), ('alloc_rsscos_ctx', ctypes.c_uint16, 24), ('alloc_cmpl_rings', ctypes.c_uint16, 26), ('alloc_tx_rings', ctypes.c_uint16, 28), ('alloc_rx_rings', ctypes.c_uint16, 30), ('alloc_l2_ctx', ctypes.c_uint16, 32), ('alloc_vnics', ctypes.c_uint16, 34), ('admin_mtu', ctypes.c_uint16, 36), ('mru', ctypes.c_uint16, 38), ('stat_ctx_id', ctypes.c_uint16, 40), ('port_partition_type', ctypes.c_ubyte, 42), ('port_pf_cnt', ctypes.c_ubyte, 43), ('dflt_vnic_id', ctypes.c_uint16, 44), ('max_mtu_configured', ctypes.c_uint16, 46), ('min_bw', ctypes.c_uint32, 48), ('max_bw', ctypes.c_uint32, 52), ('evb_mode', ctypes.c_ubyte, 56), ('options', ctypes.c_ubyte, 57), ('alloc_vfs', ctypes.c_uint16, 58), ('alloc_mcast_filters', ctypes.c_uint32, 60), ('alloc_hw_ring_grps', ctypes.c_uint32, 64), ('alloc_sp_tx_rings', ctypes.c_uint16, 68), ('alloc_stat_ctx', ctypes.c_uint16, 70), ('alloc_msix', ctypes.c_uint16, 72), ('registered_vfs', ctypes.c_uint16, 74), ('l2_doorbell_bar_size_kb', ctypes.c_uint16, 76), ('active_endpoints', ctypes.c_ubyte, 78), ('always_1', ctypes.c_ubyte, 79), ('reset_addr_poll', ctypes.c_uint32, 80), ('legacy_l2_db_size_kb', ctypes.c_uint16, 84), ('svif_info', ctypes.c_uint16, 86), ('mpc_chnls', ctypes.c_ubyte, 88), ('db_page_size', ctypes.c_ubyte, 89), ('roce_vnic_id', ctypes.c_uint16, 90), ('partition_min_bw', ctypes.c_uint32, 92), ('partition_max_bw', ctypes.c_uint32, 96), ('host_mtu', ctypes.c_uint16, 100), ('flags2', ctypes.c_uint16, 102), ('stag_vid', ctypes.c_uint16, 104), ('port_kdnet_mode', ctypes.c_ubyte, 106), ('kdnet_pcie_function', ctypes.c_ubyte, 107), ('port_kdnet_fid', ctypes.c_uint16, 108), ('unused_5', ctypes.c_ubyte, 110), ('roce_bidi_opt_mode', ctypes.c_ubyte, 111), ('num_ktls_tx_key_ctxs', ctypes.c_uint32, 112), ('num_ktls_rx_key_ctxs', ctypes.c_uint32, 116), ('lag_id', ctypes.c_ubyte, 120), ('parif', ctypes.c_ubyte, 121), ('fw_lag_id', ctypes.c_ubyte, 122), ('unused_6', ctypes.c_ubyte, 123), ('num_quic_tx_key_ctxs', ctypes.c_uint32, 124), ('num_quic_rx_key_ctxs', ctypes.c_uint32, 128), ('roce_max_av_per_vf', ctypes.c_uint32, 132), ('roce_max_cq_per_vf', ctypes.c_uint32, 136), ('roce_max_mrw_per_vf', ctypes.c_uint32, 140), ('roce_max_qp_per_vf', ctypes.c_uint32, 144), ('roce_max_srq_per_vf', ctypes.c_uint32, 148), ('roce_max_gid_per_vf', ctypes.c_uint32, 152), ('xid_partition_cfg', ctypes.c_uint16, 156), ('mirror_vnic_id', ctypes.c_uint16, 158), ('max_link_width', ctypes.c_ubyte, 160), ('max_link_speed', ctypes.c_ubyte, 161), ('negotiated_link_width', ctypes.c_ubyte, 162), ('negotiated_link_speed', ctypes.c_ubyte, 163), ('unused_7', c.Array[ctypes.c_ubyte, Literal[2]], 164), ('pcie_compliance', ctypes.c_ubyte, 166), ('unused_8', ctypes.c_ubyte, 167), ('l2_db_multi_page_size_kb', ctypes.c_uint16, 168), ('unused_9', c.Array[ctypes.c_ubyte, Literal[5]], 170), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_hwrm_func_cfg_input(c.Struct):
  SIZE = 160
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  num_msix: int
  flags: int
  enables: int
  admin_mtu: int
  mru: int
  num_rsscos_ctxs: int
  num_cmpl_rings: int
  num_tx_rings: int
  num_rx_rings: int
  num_l2_ctxs: int
  num_vnics: int
  num_stat_ctxs: int
  num_hw_ring_grps: int
  dflt_mac_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  dflt_vlan: int
  dflt_ip_addr: c.Array[ctypes.c_uint32, Literal[4]]
  min_bw: int
  max_bw: int
  async_event_cr: int
  vlan_antispoof_mode: int
  allowed_vlan_pris: int
  evb_mode: int
  options: int
  num_mcast_filters: int
  schq_id: int
  mpc_chnls: int
  partition_min_bw: int
  partition_max_bw: int
  tpid: int
  host_mtu: int
  flags2: int
  enables2: int
  port_kdnet_mode: int
  db_page_size: int
  physical_slot_number: int
  num_ktls_tx_key_ctxs: int
  num_ktls_rx_key_ctxs: int
  num_quic_tx_key_ctxs: int
  num_quic_rx_key_ctxs: int
  roce_max_av_per_vf: int
  roce_max_cq_per_vf: int
  roce_max_mrw_per_vf: int
  roce_max_qp_per_vf: int
  roce_max_srq_per_vf: int
  roce_max_gid_per_vf: int
  xid_partition_cfg: int
  pcie_compliance: int
  unused_2: int
struct_hwrm_func_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('num_msix', ctypes.c_uint16, 18), ('flags', ctypes.c_uint32, 20), ('enables', ctypes.c_uint32, 24), ('admin_mtu', ctypes.c_uint16, 28), ('mru', ctypes.c_uint16, 30), ('num_rsscos_ctxs', ctypes.c_uint16, 32), ('num_cmpl_rings', ctypes.c_uint16, 34), ('num_tx_rings', ctypes.c_uint16, 36), ('num_rx_rings', ctypes.c_uint16, 38), ('num_l2_ctxs', ctypes.c_uint16, 40), ('num_vnics', ctypes.c_uint16, 42), ('num_stat_ctxs', ctypes.c_uint16, 44), ('num_hw_ring_grps', ctypes.c_uint16, 46), ('dflt_mac_addr', c.Array[ctypes.c_ubyte, Literal[6]], 48), ('dflt_vlan', ctypes.c_uint16, 54), ('dflt_ip_addr', c.Array[ctypes.c_uint32, Literal[4]], 56), ('min_bw', ctypes.c_uint32, 72), ('max_bw', ctypes.c_uint32, 76), ('async_event_cr', ctypes.c_uint16, 80), ('vlan_antispoof_mode', ctypes.c_ubyte, 82), ('allowed_vlan_pris', ctypes.c_ubyte, 83), ('evb_mode', ctypes.c_ubyte, 84), ('options', ctypes.c_ubyte, 85), ('num_mcast_filters', ctypes.c_uint16, 86), ('schq_id', ctypes.c_uint16, 88), ('mpc_chnls', ctypes.c_uint16, 90), ('partition_min_bw', ctypes.c_uint32, 92), ('partition_max_bw', ctypes.c_uint32, 96), ('tpid', ctypes.c_uint16, 100), ('host_mtu', ctypes.c_uint16, 102), ('flags2', ctypes.c_uint32, 104), ('enables2', ctypes.c_uint32, 108), ('port_kdnet_mode', ctypes.c_ubyte, 112), ('db_page_size', ctypes.c_ubyte, 113), ('physical_slot_number', ctypes.c_uint16, 114), ('num_ktls_tx_key_ctxs', ctypes.c_uint32, 116), ('num_ktls_rx_key_ctxs', ctypes.c_uint32, 120), ('num_quic_tx_key_ctxs', ctypes.c_uint32, 124), ('num_quic_rx_key_ctxs', ctypes.c_uint32, 128), ('roce_max_av_per_vf', ctypes.c_uint32, 132), ('roce_max_cq_per_vf', ctypes.c_uint32, 136), ('roce_max_mrw_per_vf', ctypes.c_uint32, 140), ('roce_max_qp_per_vf', ctypes.c_uint32, 144), ('roce_max_srq_per_vf', ctypes.c_uint32, 148), ('roce_max_gid_per_vf', ctypes.c_uint32, 152), ('xid_partition_cfg', ctypes.c_uint16, 156), ('pcie_compliance', ctypes.c_ubyte, 158), ('unused_2', ctypes.c_ubyte, 159)])
@c.record
class struct_hwrm_func_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_cfg_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_func_cfg_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_func_qstats_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
struct_hwrm_func_qstats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('flags', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19)])
@c.record
class struct_hwrm_func_qstats_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tx_ucast_pkts: int
  tx_mcast_pkts: int
  tx_bcast_pkts: int
  tx_discard_pkts: int
  tx_drop_pkts: int
  tx_ucast_bytes: int
  tx_mcast_bytes: int
  tx_bcast_bytes: int
  rx_ucast_pkts: int
  rx_mcast_pkts: int
  rx_bcast_pkts: int
  rx_discard_pkts: int
  rx_drop_pkts: int
  rx_ucast_bytes: int
  rx_mcast_bytes: int
  rx_bcast_bytes: int
  rx_agg_pkts: int
  rx_agg_bytes: int
  rx_agg_events: int
  rx_agg_aborts: int
  clear_seq: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_func_qstats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tx_ucast_pkts', ctypes.c_uint64, 8), ('tx_mcast_pkts', ctypes.c_uint64, 16), ('tx_bcast_pkts', ctypes.c_uint64, 24), ('tx_discard_pkts', ctypes.c_uint64, 32), ('tx_drop_pkts', ctypes.c_uint64, 40), ('tx_ucast_bytes', ctypes.c_uint64, 48), ('tx_mcast_bytes', ctypes.c_uint64, 56), ('tx_bcast_bytes', ctypes.c_uint64, 64), ('rx_ucast_pkts', ctypes.c_uint64, 72), ('rx_mcast_pkts', ctypes.c_uint64, 80), ('rx_bcast_pkts', ctypes.c_uint64, 88), ('rx_discard_pkts', ctypes.c_uint64, 96), ('rx_drop_pkts', ctypes.c_uint64, 104), ('rx_ucast_bytes', ctypes.c_uint64, 112), ('rx_mcast_bytes', ctypes.c_uint64, 120), ('rx_bcast_bytes', ctypes.c_uint64, 128), ('rx_agg_pkts', ctypes.c_uint64, 136), ('rx_agg_bytes', ctypes.c_uint64, 144), ('rx_agg_events', ctypes.c_uint64, 152), ('rx_agg_aborts', ctypes.c_uint64, 160), ('clear_seq', ctypes.c_ubyte, 168), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 169), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_hwrm_func_qstats_ext_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[1]]
  enables: int
  schq_id: int
  traffic_class: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_func_qstats_ext_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('flags', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[1]], 19), ('enables', ctypes.c_uint32, 20), ('schq_id', ctypes.c_uint16, 24), ('traffic_class', ctypes.c_uint16, 26), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 28)])
@c.record
class struct_hwrm_func_qstats_ext_output(c.Struct):
  SIZE = 192
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  rx_ucast_pkts: int
  rx_mcast_pkts: int
  rx_bcast_pkts: int
  rx_discard_pkts: int
  rx_error_pkts: int
  rx_ucast_bytes: int
  rx_mcast_bytes: int
  rx_bcast_bytes: int
  tx_ucast_pkts: int
  tx_mcast_pkts: int
  tx_bcast_pkts: int
  tx_error_pkts: int
  tx_discard_pkts: int
  tx_ucast_bytes: int
  tx_mcast_bytes: int
  tx_bcast_bytes: int
  rx_tpa_eligible_pkt: int
  rx_tpa_eligible_bytes: int
  rx_tpa_pkt: int
  rx_tpa_bytes: int
  rx_tpa_errors: int
  rx_tpa_events: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_qstats_ext_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('rx_ucast_pkts', ctypes.c_uint64, 8), ('rx_mcast_pkts', ctypes.c_uint64, 16), ('rx_bcast_pkts', ctypes.c_uint64, 24), ('rx_discard_pkts', ctypes.c_uint64, 32), ('rx_error_pkts', ctypes.c_uint64, 40), ('rx_ucast_bytes', ctypes.c_uint64, 48), ('rx_mcast_bytes', ctypes.c_uint64, 56), ('rx_bcast_bytes', ctypes.c_uint64, 64), ('tx_ucast_pkts', ctypes.c_uint64, 72), ('tx_mcast_pkts', ctypes.c_uint64, 80), ('tx_bcast_pkts', ctypes.c_uint64, 88), ('tx_error_pkts', ctypes.c_uint64, 96), ('tx_discard_pkts', ctypes.c_uint64, 104), ('tx_ucast_bytes', ctypes.c_uint64, 112), ('tx_mcast_bytes', ctypes.c_uint64, 120), ('tx_bcast_bytes', ctypes.c_uint64, 128), ('rx_tpa_eligible_pkt', ctypes.c_uint64, 136), ('rx_tpa_eligible_bytes', ctypes.c_uint64, 144), ('rx_tpa_pkt', ctypes.c_uint64, 152), ('rx_tpa_bytes', ctypes.c_uint64, 160), ('rx_tpa_errors', ctypes.c_uint64, 168), ('rx_tpa_events', ctypes.c_uint64, 176), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 184), ('valid', ctypes.c_ubyte, 191)])
@c.record
class struct_hwrm_func_clr_stats_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_clr_stats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_clr_stats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_clr_stats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_vf_resc_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vf_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_vf_resc_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vf_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_vf_resc_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_vf_resc_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_drv_rgtr_input(c.Struct):
  SIZE = 112
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  os_type: int
  ver_maj_8b: int
  ver_min_8b: int
  ver_upd_8b: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  timestamp: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
  vf_req_fwd: c.Array[ctypes.c_uint32, Literal[8]]
  async_event_fwd: c.Array[ctypes.c_uint32, Literal[8]]
  ver_maj: int
  ver_min: int
  ver_upd: int
  ver_patch: int
struct_hwrm_func_drv_rgtr_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('os_type', ctypes.c_uint16, 24), ('ver_maj_8b', ctypes.c_ubyte, 26), ('ver_min_8b', ctypes.c_ubyte, 27), ('ver_upd_8b', ctypes.c_ubyte, 28), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 29), ('timestamp', ctypes.c_uint32, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 36), ('vf_req_fwd', c.Array[ctypes.c_uint32, Literal[8]], 40), ('async_event_fwd', c.Array[ctypes.c_uint32, Literal[8]], 72), ('ver_maj', ctypes.c_uint16, 104), ('ver_min', ctypes.c_uint16, 106), ('ver_upd', ctypes.c_uint16, 108), ('ver_patch', ctypes.c_uint16, 110)])
@c.record
class struct_hwrm_func_drv_rgtr_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_drv_rgtr_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_drv_unrgtr_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_func_drv_unrgtr_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_func_drv_unrgtr_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_drv_unrgtr_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_buf_rgtr_input(c.Struct):
  SIZE = 128
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  vf_id: int
  req_buf_num_pages: int
  req_buf_page_size: int
  req_buf_len: int
  resp_buf_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  req_buf_page_addr0: int
  req_buf_page_addr1: int
  req_buf_page_addr2: int
  req_buf_page_addr3: int
  req_buf_page_addr4: int
  req_buf_page_addr5: int
  req_buf_page_addr6: int
  req_buf_page_addr7: int
  req_buf_page_addr8: int
  req_buf_page_addr9: int
  error_buf_addr: int
  resp_buf_addr: int
struct_hwrm_func_buf_rgtr_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('vf_id', ctypes.c_uint16, 20), ('req_buf_num_pages', ctypes.c_uint16, 22), ('req_buf_page_size', ctypes.c_uint16, 24), ('req_buf_len', ctypes.c_uint16, 26), ('resp_buf_len', ctypes.c_uint16, 28), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 30), ('req_buf_page_addr0', ctypes.c_uint64, 32), ('req_buf_page_addr1', ctypes.c_uint64, 40), ('req_buf_page_addr2', ctypes.c_uint64, 48), ('req_buf_page_addr3', ctypes.c_uint64, 56), ('req_buf_page_addr4', ctypes.c_uint64, 64), ('req_buf_page_addr5', ctypes.c_uint64, 72), ('req_buf_page_addr6', ctypes.c_uint64, 80), ('req_buf_page_addr7', ctypes.c_uint64, 88), ('req_buf_page_addr8', ctypes.c_uint64, 96), ('req_buf_page_addr9', ctypes.c_uint64, 104), ('error_buf_addr', ctypes.c_uint64, 112), ('resp_buf_addr', ctypes.c_uint64, 120)])
@c.record
class struct_hwrm_func_buf_rgtr_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_buf_rgtr_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_drv_qver_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  reserved: int
  fid: int
  driver_type: int
  unused_0: int
struct_hwrm_func_drv_qver_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('reserved', ctypes.c_uint32, 16), ('fid', ctypes.c_uint16, 20), ('driver_type', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_func_drv_qver_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  os_type: int
  ver_maj_8b: int
  ver_min_8b: int
  ver_upd_8b: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  ver_maj: int
  ver_min: int
  ver_upd: int
  ver_patch: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_drv_qver_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('os_type', ctypes.c_uint16, 8), ('ver_maj_8b', ctypes.c_ubyte, 10), ('ver_min_8b', ctypes.c_ubyte, 11), ('ver_upd_8b', ctypes.c_ubyte, 12), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 13), ('ver_maj', ctypes.c_uint16, 16), ('ver_min', ctypes.c_uint16, 18), ('ver_upd', ctypes.c_uint16, 20), ('ver_patch', ctypes.c_uint16, 22), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 24), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_func_resource_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_resource_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_resource_qcaps_output(c.Struct):
  SIZE = 88
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  max_vfs: int
  max_msix: int
  vf_reservation_strategy: int
  min_rsscos_ctx: int
  max_rsscos_ctx: int
  min_cmpl_rings: int
  max_cmpl_rings: int
  min_tx_rings: int
  max_tx_rings: int
  min_rx_rings: int
  max_rx_rings: int
  min_l2_ctxs: int
  max_l2_ctxs: int
  min_vnics: int
  max_vnics: int
  min_stat_ctx: int
  max_stat_ctx: int
  min_hw_ring_grps: int
  max_hw_ring_grps: int
  max_tx_scheduler_inputs: int
  flags: int
  min_msix: int
  min_ktls_tx_key_ctxs: int
  max_ktls_tx_key_ctxs: int
  min_ktls_rx_key_ctxs: int
  max_ktls_rx_key_ctxs: int
  min_quic_tx_key_ctxs: int
  max_quic_tx_key_ctxs: int
  min_quic_rx_key_ctxs: int
  max_quic_rx_key_ctxs: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_resource_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('max_vfs', ctypes.c_uint16, 8), ('max_msix', ctypes.c_uint16, 10), ('vf_reservation_strategy', ctypes.c_uint16, 12), ('min_rsscos_ctx', ctypes.c_uint16, 14), ('max_rsscos_ctx', ctypes.c_uint16, 16), ('min_cmpl_rings', ctypes.c_uint16, 18), ('max_cmpl_rings', ctypes.c_uint16, 20), ('min_tx_rings', ctypes.c_uint16, 22), ('max_tx_rings', ctypes.c_uint16, 24), ('min_rx_rings', ctypes.c_uint16, 26), ('max_rx_rings', ctypes.c_uint16, 28), ('min_l2_ctxs', ctypes.c_uint16, 30), ('max_l2_ctxs', ctypes.c_uint16, 32), ('min_vnics', ctypes.c_uint16, 34), ('max_vnics', ctypes.c_uint16, 36), ('min_stat_ctx', ctypes.c_uint16, 38), ('max_stat_ctx', ctypes.c_uint16, 40), ('min_hw_ring_grps', ctypes.c_uint16, 42), ('max_hw_ring_grps', ctypes.c_uint16, 44), ('max_tx_scheduler_inputs', ctypes.c_uint16, 46), ('flags', ctypes.c_uint16, 48), ('min_msix', ctypes.c_uint16, 50), ('min_ktls_tx_key_ctxs', ctypes.c_uint32, 52), ('max_ktls_tx_key_ctxs', ctypes.c_uint32, 56), ('min_ktls_rx_key_ctxs', ctypes.c_uint32, 60), ('max_ktls_rx_key_ctxs', ctypes.c_uint32, 64), ('min_quic_tx_key_ctxs', ctypes.c_uint32, 68), ('max_quic_tx_key_ctxs', ctypes.c_uint32, 72), ('min_quic_rx_key_ctxs', ctypes.c_uint32, 76), ('max_quic_rx_key_ctxs', ctypes.c_uint32, 80), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 84), ('valid', ctypes.c_ubyte, 87)])
@c.record
class struct_hwrm_func_vf_resource_cfg_input(c.Struct):
  SIZE = 88
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vf_id: int
  max_msix: int
  min_rsscos_ctx: int
  max_rsscos_ctx: int
  min_cmpl_rings: int
  max_cmpl_rings: int
  min_tx_rings: int
  max_tx_rings: int
  min_rx_rings: int
  max_rx_rings: int
  min_l2_ctxs: int
  max_l2_ctxs: int
  min_vnics: int
  max_vnics: int
  min_stat_ctx: int
  max_stat_ctx: int
  min_hw_ring_grps: int
  max_hw_ring_grps: int
  flags: int
  min_msix: int
  min_ktls_tx_key_ctxs: int
  max_ktls_tx_key_ctxs: int
  min_ktls_rx_key_ctxs: int
  max_ktls_rx_key_ctxs: int
  min_quic_tx_key_ctxs: int
  max_quic_tx_key_ctxs: int
  min_quic_rx_key_ctxs: int
  max_quic_rx_key_ctxs: int
struct_hwrm_func_vf_resource_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vf_id', ctypes.c_uint16, 16), ('max_msix', ctypes.c_uint16, 18), ('min_rsscos_ctx', ctypes.c_uint16, 20), ('max_rsscos_ctx', ctypes.c_uint16, 22), ('min_cmpl_rings', ctypes.c_uint16, 24), ('max_cmpl_rings', ctypes.c_uint16, 26), ('min_tx_rings', ctypes.c_uint16, 28), ('max_tx_rings', ctypes.c_uint16, 30), ('min_rx_rings', ctypes.c_uint16, 32), ('max_rx_rings', ctypes.c_uint16, 34), ('min_l2_ctxs', ctypes.c_uint16, 36), ('max_l2_ctxs', ctypes.c_uint16, 38), ('min_vnics', ctypes.c_uint16, 40), ('max_vnics', ctypes.c_uint16, 42), ('min_stat_ctx', ctypes.c_uint16, 44), ('max_stat_ctx', ctypes.c_uint16, 46), ('min_hw_ring_grps', ctypes.c_uint16, 48), ('max_hw_ring_grps', ctypes.c_uint16, 50), ('flags', ctypes.c_uint16, 52), ('min_msix', ctypes.c_uint16, 54), ('min_ktls_tx_key_ctxs', ctypes.c_uint32, 56), ('max_ktls_tx_key_ctxs', ctypes.c_uint32, 60), ('min_ktls_rx_key_ctxs', ctypes.c_uint32, 64), ('max_ktls_rx_key_ctxs', ctypes.c_uint32, 68), ('min_quic_tx_key_ctxs', ctypes.c_uint32, 72), ('max_quic_tx_key_ctxs', ctypes.c_uint32, 76), ('min_quic_rx_key_ctxs', ctypes.c_uint32, 80), ('max_quic_rx_key_ctxs', ctypes.c_uint32, 84)])
@c.record
class struct_hwrm_func_vf_resource_cfg_output(c.Struct):
  SIZE = 48
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  reserved_rsscos_ctx: int
  reserved_cmpl_rings: int
  reserved_tx_rings: int
  reserved_rx_rings: int
  reserved_l2_ctxs: int
  reserved_vnics: int
  reserved_stat_ctx: int
  reserved_hw_ring_grps: int
  reserved_ktls_tx_key_ctxs: int
  reserved_ktls_rx_key_ctxs: int
  reserved_quic_tx_key_ctxs: int
  reserved_quic_rx_key_ctxs: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_vf_resource_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('reserved_rsscos_ctx', ctypes.c_uint16, 8), ('reserved_cmpl_rings', ctypes.c_uint16, 10), ('reserved_tx_rings', ctypes.c_uint16, 12), ('reserved_rx_rings', ctypes.c_uint16, 14), ('reserved_l2_ctxs', ctypes.c_uint16, 16), ('reserved_vnics', ctypes.c_uint16, 18), ('reserved_stat_ctx', ctypes.c_uint16, 20), ('reserved_hw_ring_grps', ctypes.c_uint16, 22), ('reserved_ktls_tx_key_ctxs', ctypes.c_uint32, 24), ('reserved_ktls_rx_key_ctxs', ctypes.c_uint32, 28), ('reserved_quic_tx_key_ctxs', ctypes.c_uint32, 32), ('reserved_quic_rx_key_ctxs', ctypes.c_uint32, 36), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 40), ('valid', ctypes.c_ubyte, 47)])
@c.record
class struct_hwrm_func_backing_store_qcaps_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_func_backing_store_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_func_backing_store_qcaps_output(c.Struct):
  SIZE = 104
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  qp_max_entries: int
  qp_min_qp1_entries: int
  qp_max_l2_entries: int
  qp_entry_size: int
  srq_max_l2_entries: int
  srq_max_entries: int
  srq_entry_size: int
  cq_max_l2_entries: int
  cq_max_entries: int
  cq_entry_size: int
  vnic_max_vnic_entries: int
  vnic_max_ring_table_entries: int
  vnic_entry_size: int
  stat_max_entries: int
  stat_entry_size: int
  tqm_entry_size: int
  tqm_min_entries_per_ring: int
  tqm_max_entries_per_ring: int
  mrav_max_entries: int
  mrav_entry_size: int
  tim_entry_size: int
  tim_max_entries: int
  mrav_num_entries_units: int
  tqm_entries_multiple: int
  ctx_kind_initializer: int
  ctx_init_mask: int
  qp_init_offset: int
  srq_init_offset: int
  cq_init_offset: int
  vnic_init_offset: int
  tqm_fp_rings_count: int
  stat_init_offset: int
  mrav_init_offset: int
  tqm_fp_rings_count_ext: int
  tkc_init_offset: int
  rkc_init_offset: int
  tkc_entry_size: int
  rkc_entry_size: int
  tkc_max_entries: int
  rkc_max_entries: int
  fast_qpmd_qp_num_entries: int
  rsvd1: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_func_backing_store_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('qp_max_entries', ctypes.c_uint32, 8), ('qp_min_qp1_entries', ctypes.c_uint16, 12), ('qp_max_l2_entries', ctypes.c_uint16, 14), ('qp_entry_size', ctypes.c_uint16, 16), ('srq_max_l2_entries', ctypes.c_uint16, 18), ('srq_max_entries', ctypes.c_uint32, 20), ('srq_entry_size', ctypes.c_uint16, 24), ('cq_max_l2_entries', ctypes.c_uint16, 26), ('cq_max_entries', ctypes.c_uint32, 28), ('cq_entry_size', ctypes.c_uint16, 32), ('vnic_max_vnic_entries', ctypes.c_uint16, 34), ('vnic_max_ring_table_entries', ctypes.c_uint16, 36), ('vnic_entry_size', ctypes.c_uint16, 38), ('stat_max_entries', ctypes.c_uint32, 40), ('stat_entry_size', ctypes.c_uint16, 44), ('tqm_entry_size', ctypes.c_uint16, 46), ('tqm_min_entries_per_ring', ctypes.c_uint32, 48), ('tqm_max_entries_per_ring', ctypes.c_uint32, 52), ('mrav_max_entries', ctypes.c_uint32, 56), ('mrav_entry_size', ctypes.c_uint16, 60), ('tim_entry_size', ctypes.c_uint16, 62), ('tim_max_entries', ctypes.c_uint32, 64), ('mrav_num_entries_units', ctypes.c_uint16, 68), ('tqm_entries_multiple', ctypes.c_ubyte, 70), ('ctx_kind_initializer', ctypes.c_ubyte, 71), ('ctx_init_mask', ctypes.c_uint16, 72), ('qp_init_offset', ctypes.c_ubyte, 74), ('srq_init_offset', ctypes.c_ubyte, 75), ('cq_init_offset', ctypes.c_ubyte, 76), ('vnic_init_offset', ctypes.c_ubyte, 77), ('tqm_fp_rings_count', ctypes.c_ubyte, 78), ('stat_init_offset', ctypes.c_ubyte, 79), ('mrav_init_offset', ctypes.c_ubyte, 80), ('tqm_fp_rings_count_ext', ctypes.c_ubyte, 81), ('tkc_init_offset', ctypes.c_ubyte, 82), ('rkc_init_offset', ctypes.c_ubyte, 83), ('tkc_entry_size', ctypes.c_uint16, 84), ('rkc_entry_size', ctypes.c_uint16, 86), ('tkc_max_entries', ctypes.c_uint32, 88), ('rkc_max_entries', ctypes.c_uint32, 92), ('fast_qpmd_qp_num_entries', ctypes.c_uint16, 96), ('rsvd1', c.Array[ctypes.c_ubyte, Literal[5]], 98), ('valid', ctypes.c_ubyte, 103)])
@c.record
class struct_tqm_fp_ring_cfg(c.Struct):
  SIZE = 16
  tqm_ring_pg_size_tqm_ring_lvl: int
  unused: c.Array[ctypes.c_ubyte, Literal[3]]
  tqm_ring_num_entries: int
  tqm_ring_page_dir: int
struct_tqm_fp_ring_cfg.register_fields([('tqm_ring_pg_size_tqm_ring_lvl', ctypes.c_ubyte, 0), ('unused', c.Array[ctypes.c_ubyte, Literal[3]], 1), ('tqm_ring_num_entries', ctypes.c_uint32, 4), ('tqm_ring_page_dir', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_func_backing_store_cfg_input(c.Struct):
  SIZE = 336
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  qpc_pg_size_qpc_lvl: int
  srq_pg_size_srq_lvl: int
  cq_pg_size_cq_lvl: int
  vnic_pg_size_vnic_lvl: int
  stat_pg_size_stat_lvl: int
  tqm_sp_pg_size_tqm_sp_lvl: int
  tqm_ring0_pg_size_tqm_ring0_lvl: int
  tqm_ring1_pg_size_tqm_ring1_lvl: int
  tqm_ring2_pg_size_tqm_ring2_lvl: int
  tqm_ring3_pg_size_tqm_ring3_lvl: int
  tqm_ring4_pg_size_tqm_ring4_lvl: int
  tqm_ring5_pg_size_tqm_ring5_lvl: int
  tqm_ring6_pg_size_tqm_ring6_lvl: int
  tqm_ring7_pg_size_tqm_ring7_lvl: int
  mrav_pg_size_mrav_lvl: int
  tim_pg_size_tim_lvl: int
  qpc_page_dir: int
  srq_page_dir: int
  cq_page_dir: int
  vnic_page_dir: int
  stat_page_dir: int
  tqm_sp_page_dir: int
  tqm_ring0_page_dir: int
  tqm_ring1_page_dir: int
  tqm_ring2_page_dir: int
  tqm_ring3_page_dir: int
  tqm_ring4_page_dir: int
  tqm_ring5_page_dir: int
  tqm_ring6_page_dir: int
  tqm_ring7_page_dir: int
  mrav_page_dir: int
  tim_page_dir: int
  qp_num_entries: int
  srq_num_entries: int
  cq_num_entries: int
  stat_num_entries: int
  tqm_sp_num_entries: int
  tqm_ring0_num_entries: int
  tqm_ring1_num_entries: int
  tqm_ring2_num_entries: int
  tqm_ring3_num_entries: int
  tqm_ring4_num_entries: int
  tqm_ring5_num_entries: int
  tqm_ring6_num_entries: int
  tqm_ring7_num_entries: int
  mrav_num_entries: int
  tim_num_entries: int
  qp_num_qp1_entries: int
  qp_num_l2_entries: int
  qp_entry_size: int
  srq_num_l2_entries: int
  srq_entry_size: int
  cq_num_l2_entries: int
  cq_entry_size: int
  vnic_num_vnic_entries: int
  vnic_num_ring_table_entries: int
  vnic_entry_size: int
  stat_entry_size: int
  tqm_entry_size: int
  mrav_entry_size: int
  tim_entry_size: int
  tqm_ring8_pg_size_tqm_ring_lvl: int
  ring8_unused: c.Array[ctypes.c_ubyte, Literal[3]]
  tqm_ring8_num_entries: int
  tqm_ring8_page_dir: int
  tqm_ring9_pg_size_tqm_ring_lvl: int
  ring9_unused: c.Array[ctypes.c_ubyte, Literal[3]]
  tqm_ring9_num_entries: int
  tqm_ring9_page_dir: int
  tqm_ring10_pg_size_tqm_ring_lvl: int
  ring10_unused: c.Array[ctypes.c_ubyte, Literal[3]]
  tqm_ring10_num_entries: int
  tqm_ring10_page_dir: int
  tkc_num_entries: int
  rkc_num_entries: int
  tkc_page_dir: int
  rkc_page_dir: int
  tkc_entry_size: int
  rkc_entry_size: int
  tkc_pg_size_tkc_lvl: int
  rkc_pg_size_rkc_lvl: int
  qp_num_fast_qpmd_entries: int
struct_hwrm_func_backing_store_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('qpc_pg_size_qpc_lvl', ctypes.c_ubyte, 24), ('srq_pg_size_srq_lvl', ctypes.c_ubyte, 25), ('cq_pg_size_cq_lvl', ctypes.c_ubyte, 26), ('vnic_pg_size_vnic_lvl', ctypes.c_ubyte, 27), ('stat_pg_size_stat_lvl', ctypes.c_ubyte, 28), ('tqm_sp_pg_size_tqm_sp_lvl', ctypes.c_ubyte, 29), ('tqm_ring0_pg_size_tqm_ring0_lvl', ctypes.c_ubyte, 30), ('tqm_ring1_pg_size_tqm_ring1_lvl', ctypes.c_ubyte, 31), ('tqm_ring2_pg_size_tqm_ring2_lvl', ctypes.c_ubyte, 32), ('tqm_ring3_pg_size_tqm_ring3_lvl', ctypes.c_ubyte, 33), ('tqm_ring4_pg_size_tqm_ring4_lvl', ctypes.c_ubyte, 34), ('tqm_ring5_pg_size_tqm_ring5_lvl', ctypes.c_ubyte, 35), ('tqm_ring6_pg_size_tqm_ring6_lvl', ctypes.c_ubyte, 36), ('tqm_ring7_pg_size_tqm_ring7_lvl', ctypes.c_ubyte, 37), ('mrav_pg_size_mrav_lvl', ctypes.c_ubyte, 38), ('tim_pg_size_tim_lvl', ctypes.c_ubyte, 39), ('qpc_page_dir', ctypes.c_uint64, 40), ('srq_page_dir', ctypes.c_uint64, 48), ('cq_page_dir', ctypes.c_uint64, 56), ('vnic_page_dir', ctypes.c_uint64, 64), ('stat_page_dir', ctypes.c_uint64, 72), ('tqm_sp_page_dir', ctypes.c_uint64, 80), ('tqm_ring0_page_dir', ctypes.c_uint64, 88), ('tqm_ring1_page_dir', ctypes.c_uint64, 96), ('tqm_ring2_page_dir', ctypes.c_uint64, 104), ('tqm_ring3_page_dir', ctypes.c_uint64, 112), ('tqm_ring4_page_dir', ctypes.c_uint64, 120), ('tqm_ring5_page_dir', ctypes.c_uint64, 128), ('tqm_ring6_page_dir', ctypes.c_uint64, 136), ('tqm_ring7_page_dir', ctypes.c_uint64, 144), ('mrav_page_dir', ctypes.c_uint64, 152), ('tim_page_dir', ctypes.c_uint64, 160), ('qp_num_entries', ctypes.c_uint32, 168), ('srq_num_entries', ctypes.c_uint32, 172), ('cq_num_entries', ctypes.c_uint32, 176), ('stat_num_entries', ctypes.c_uint32, 180), ('tqm_sp_num_entries', ctypes.c_uint32, 184), ('tqm_ring0_num_entries', ctypes.c_uint32, 188), ('tqm_ring1_num_entries', ctypes.c_uint32, 192), ('tqm_ring2_num_entries', ctypes.c_uint32, 196), ('tqm_ring3_num_entries', ctypes.c_uint32, 200), ('tqm_ring4_num_entries', ctypes.c_uint32, 204), ('tqm_ring5_num_entries', ctypes.c_uint32, 208), ('tqm_ring6_num_entries', ctypes.c_uint32, 212), ('tqm_ring7_num_entries', ctypes.c_uint32, 216), ('mrav_num_entries', ctypes.c_uint32, 220), ('tim_num_entries', ctypes.c_uint32, 224), ('qp_num_qp1_entries', ctypes.c_uint16, 228), ('qp_num_l2_entries', ctypes.c_uint16, 230), ('qp_entry_size', ctypes.c_uint16, 232), ('srq_num_l2_entries', ctypes.c_uint16, 234), ('srq_entry_size', ctypes.c_uint16, 236), ('cq_num_l2_entries', ctypes.c_uint16, 238), ('cq_entry_size', ctypes.c_uint16, 240), ('vnic_num_vnic_entries', ctypes.c_uint16, 242), ('vnic_num_ring_table_entries', ctypes.c_uint16, 244), ('vnic_entry_size', ctypes.c_uint16, 246), ('stat_entry_size', ctypes.c_uint16, 248), ('tqm_entry_size', ctypes.c_uint16, 250), ('mrav_entry_size', ctypes.c_uint16, 252), ('tim_entry_size', ctypes.c_uint16, 254), ('tqm_ring8_pg_size_tqm_ring_lvl', ctypes.c_ubyte, 256), ('ring8_unused', c.Array[ctypes.c_ubyte, Literal[3]], 257), ('tqm_ring8_num_entries', ctypes.c_uint32, 260), ('tqm_ring8_page_dir', ctypes.c_uint64, 264), ('tqm_ring9_pg_size_tqm_ring_lvl', ctypes.c_ubyte, 272), ('ring9_unused', c.Array[ctypes.c_ubyte, Literal[3]], 273), ('tqm_ring9_num_entries', ctypes.c_uint32, 276), ('tqm_ring9_page_dir', ctypes.c_uint64, 280), ('tqm_ring10_pg_size_tqm_ring_lvl', ctypes.c_ubyte, 288), ('ring10_unused', c.Array[ctypes.c_ubyte, Literal[3]], 289), ('tqm_ring10_num_entries', ctypes.c_uint32, 292), ('tqm_ring10_page_dir', ctypes.c_uint64, 296), ('tkc_num_entries', ctypes.c_uint32, 304), ('rkc_num_entries', ctypes.c_uint32, 308), ('tkc_page_dir', ctypes.c_uint64, 312), ('rkc_page_dir', ctypes.c_uint64, 320), ('tkc_entry_size', ctypes.c_uint16, 328), ('rkc_entry_size', ctypes.c_uint16, 330), ('tkc_pg_size_tkc_lvl', ctypes.c_ubyte, 332), ('rkc_pg_size_rkc_lvl', ctypes.c_ubyte, 333), ('qp_num_fast_qpmd_entries', ctypes.c_uint16, 334)])
@c.record
class struct_hwrm_func_backing_store_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_backing_store_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_error_recovery_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[8]]
struct_hwrm_error_recovery_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[8]], 16)])
@c.record
class struct_hwrm_error_recovery_qcfg_output(c.Struct):
  SIZE = 208
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  driver_polling_freq: int
  master_func_wait_period: int
  normal_func_wait_period: int
  master_func_wait_period_after_reset: int
  max_bailout_time_after_reset: int
  fw_health_status_reg: int
  fw_heartbeat_reg: int
  fw_reset_cnt_reg: int
  reset_inprogress_reg: int
  reset_inprogress_reg_mask: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  reg_array_cnt: int
  reset_reg: c.Array[ctypes.c_uint32, Literal[16]]
  reset_reg_val: c.Array[ctypes.c_uint32, Literal[16]]
  delay_after_reset: c.Array[ctypes.c_ubyte, Literal[16]]
  err_recovery_cnt_reg: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_error_recovery_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('driver_polling_freq', ctypes.c_uint32, 12), ('master_func_wait_period', ctypes.c_uint32, 16), ('normal_func_wait_period', ctypes.c_uint32, 20), ('master_func_wait_period_after_reset', ctypes.c_uint32, 24), ('max_bailout_time_after_reset', ctypes.c_uint32, 28), ('fw_health_status_reg', ctypes.c_uint32, 32), ('fw_heartbeat_reg', ctypes.c_uint32, 36), ('fw_reset_cnt_reg', ctypes.c_uint32, 40), ('reset_inprogress_reg', ctypes.c_uint32, 44), ('reset_inprogress_reg_mask', ctypes.c_uint32, 48), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 52), ('reg_array_cnt', ctypes.c_ubyte, 55), ('reset_reg', c.Array[ctypes.c_uint32, Literal[16]], 56), ('reset_reg_val', c.Array[ctypes.c_uint32, Literal[16]], 120), ('delay_after_reset', c.Array[ctypes.c_ubyte, Literal[16]], 184), ('err_recovery_cnt_reg', ctypes.c_uint32, 200), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 204), ('valid', ctypes.c_ubyte, 207)])
@c.record
class struct_hwrm_func_echo_response_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  event_data1: int
  event_data2: int
struct_hwrm_func_echo_response_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('event_data1', ctypes.c_uint32, 16), ('event_data2', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_func_echo_response_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_echo_response_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_ptp_pin_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[8]]
struct_hwrm_func_ptp_pin_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[8]], 16)])
@c.record
class struct_hwrm_func_ptp_pin_qcfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  num_pins: int
  state: int
  pin0_usage: int
  pin1_usage: int
  pin2_usage: int
  pin3_usage: int
  unused_0: int
  valid: int
struct_hwrm_func_ptp_pin_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('num_pins', ctypes.c_ubyte, 8), ('state', ctypes.c_ubyte, 9), ('pin0_usage', ctypes.c_ubyte, 10), ('pin1_usage', ctypes.c_ubyte, 11), ('pin2_usage', ctypes.c_ubyte, 12), ('pin3_usage', ctypes.c_ubyte, 13), ('unused_0', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_ptp_pin_cfg_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  pin0_state: int
  pin0_usage: int
  pin1_state: int
  pin1_usage: int
  pin2_state: int
  pin2_usage: int
  pin3_state: int
  pin3_usage: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_func_ptp_pin_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('pin0_state', ctypes.c_ubyte, 20), ('pin0_usage', ctypes.c_ubyte, 21), ('pin1_state', ctypes.c_ubyte, 22), ('pin1_usage', ctypes.c_ubyte, 23), ('pin2_state', ctypes.c_ubyte, 24), ('pin2_usage', ctypes.c_ubyte, 25), ('pin3_state', ctypes.c_ubyte, 26), ('pin3_usage', ctypes.c_ubyte, 27), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 28)])
@c.record
class struct_hwrm_func_ptp_pin_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_ptp_pin_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_ptp_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  ptp_pps_event: int
  ptp_freq_adj_dll_source: int
  ptp_freq_adj_dll_phase: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  ptp_freq_adj_ext_period: int
  ptp_freq_adj_ext_up: int
  ptp_freq_adj_ext_phase_lower: int
  ptp_freq_adj_ext_phase_upper: int
  ptp_set_time: int
struct_hwrm_func_ptp_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint16, 16), ('ptp_pps_event', ctypes.c_ubyte, 18), ('ptp_freq_adj_dll_source', ctypes.c_ubyte, 19), ('ptp_freq_adj_dll_phase', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21), ('ptp_freq_adj_ext_period', ctypes.c_uint32, 24), ('ptp_freq_adj_ext_up', ctypes.c_uint32, 28), ('ptp_freq_adj_ext_phase_lower', ctypes.c_uint32, 32), ('ptp_freq_adj_ext_phase_upper', ctypes.c_uint32, 36), ('ptp_set_time', ctypes.c_uint64, 40)])
@c.record
class struct_hwrm_func_ptp_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_ptp_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_ptp_ts_query_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_func_ptp_ts_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_func_ptp_ts_query_output(c.Struct):
  SIZE = 40
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  pps_event_ts: int
  ptm_local_ts: int
  ptm_system_ts: int
  ptm_link_delay: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_ptp_ts_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('pps_event_ts', ctypes.c_uint64, 8), ('ptm_local_ts', ctypes.c_uint64, 16), ('ptm_system_ts', ctypes.c_uint64, 24), ('ptm_link_delay', ctypes.c_uint32, 32), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 36), ('valid', ctypes.c_ubyte, 39)])
@c.record
class struct_hwrm_func_ptp_ext_cfg_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  phc_master_fid: int
  phc_sec_fid: int
  phc_sec_mode: int
  unused_0: int
  failover_timer: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_func_ptp_ext_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint16, 16), ('phc_master_fid', ctypes.c_uint16, 18), ('phc_sec_fid', ctypes.c_uint16, 20), ('phc_sec_mode', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23), ('failover_timer', ctypes.c_uint32, 24), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 28)])
@c.record
class struct_hwrm_func_ptp_ext_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_ptp_ext_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_ptp_ext_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[8]]
struct_hwrm_func_ptp_ext_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[8]], 16)])
@c.record
class struct_hwrm_func_ptp_ext_qcfg_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  phc_master_fid: int
  phc_sec_fid: int
  phc_active_fid0: int
  phc_active_fid1: int
  last_failover_event: int
  from_fid: int
  to_fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_ptp_ext_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('phc_master_fid', ctypes.c_uint16, 8), ('phc_sec_fid', ctypes.c_uint16, 10), ('phc_active_fid0', ctypes.c_uint16, 12), ('phc_active_fid1', ctypes.c_uint16, 14), ('last_failover_event', ctypes.c_uint32, 16), ('from_fid', ctypes.c_uint16, 20), ('to_fid', ctypes.c_uint16, 22), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 24), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_func_backing_store_cfg_v2_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  type: int
  instance: int
  flags: int
  page_dir: int
  num_entries: int
  entry_size: int
  page_size_pbl_level: int
  subtype_valid_cnt: int
  split_entry_0: int
  split_entry_1: int
  split_entry_2: int
  split_entry_3: int
  enables: int
  next_bs_offset: int
struct_hwrm_func_backing_store_cfg_v2_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('type', ctypes.c_uint16, 16), ('instance', ctypes.c_uint16, 18), ('flags', ctypes.c_uint32, 20), ('page_dir', ctypes.c_uint64, 24), ('num_entries', ctypes.c_uint32, 32), ('entry_size', ctypes.c_uint16, 36), ('page_size_pbl_level', ctypes.c_ubyte, 38), ('subtype_valid_cnt', ctypes.c_ubyte, 39), ('split_entry_0', ctypes.c_uint32, 40), ('split_entry_1', ctypes.c_uint32, 44), ('split_entry_2', ctypes.c_uint32, 48), ('split_entry_3', ctypes.c_uint32, 52), ('enables', ctypes.c_uint32, 56), ('next_bs_offset', ctypes.c_uint32, 60)])
@c.record
class struct_hwrm_func_backing_store_cfg_v2_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  rsvd0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_backing_store_cfg_v2_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('rsvd0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_func_backing_store_qcfg_v2_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  type: int
  instance: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_func_backing_store_qcfg_v2_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('type', ctypes.c_uint16, 16), ('instance', ctypes.c_uint16, 18), ('rsvd', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_func_backing_store_qcfg_v2_output(c.Struct):
  SIZE = 56
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  type: int
  instance: int
  flags: int
  page_dir: int
  num_entries: int
  page_size_pbl_level: int
  subtype_valid_cnt: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[2]]
  split_entry_0: int
  split_entry_1: int
  split_entry_2: int
  split_entry_3: int
  rsvd2: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_backing_store_qcfg_v2_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('type', ctypes.c_uint16, 8), ('instance', ctypes.c_uint16, 10), ('flags', ctypes.c_uint32, 12), ('page_dir', ctypes.c_uint64, 16), ('num_entries', ctypes.c_uint32, 24), ('page_size_pbl_level', ctypes.c_ubyte, 28), ('subtype_valid_cnt', ctypes.c_ubyte, 29), ('rsvd', c.Array[ctypes.c_ubyte, Literal[2]], 30), ('split_entry_0', ctypes.c_uint32, 32), ('split_entry_1', ctypes.c_uint32, 36), ('split_entry_2', ctypes.c_uint32, 40), ('split_entry_3', ctypes.c_uint32, 44), ('rsvd2', c.Array[ctypes.c_ubyte, Literal[7]], 48), ('valid', ctypes.c_ubyte, 55)])
@c.record
class struct_qpc_split_entries(c.Struct):
  SIZE = 16
  qp_num_l2_entries: int
  qp_num_qp1_entries: int
  qp_num_fast_qpmd_entries: int
  rsvd: int
struct_qpc_split_entries.register_fields([('qp_num_l2_entries', ctypes.c_uint32, 0), ('qp_num_qp1_entries', ctypes.c_uint32, 4), ('qp_num_fast_qpmd_entries', ctypes.c_uint32, 8), ('rsvd', ctypes.c_uint32, 12)])
@c.record
class struct_srq_split_entries(c.Struct):
  SIZE = 16
  srq_num_l2_entries: int
  rsvd: int
  rsvd2: c.Array[ctypes.c_uint32, Literal[2]]
struct_srq_split_entries.register_fields([('srq_num_l2_entries', ctypes.c_uint32, 0), ('rsvd', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_cq_split_entries(c.Struct):
  SIZE = 16
  cq_num_l2_entries: int
  rsvd: int
  rsvd2: c.Array[ctypes.c_uint32, Literal[2]]
struct_cq_split_entries.register_fields([('cq_num_l2_entries', ctypes.c_uint32, 0), ('rsvd', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_vnic_split_entries(c.Struct):
  SIZE = 16
  vnic_num_vnic_entries: int
  rsvd: int
  rsvd2: c.Array[ctypes.c_uint32, Literal[2]]
struct_vnic_split_entries.register_fields([('vnic_num_vnic_entries', ctypes.c_uint32, 0), ('rsvd', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_mrav_split_entries(c.Struct):
  SIZE = 16
  mrav_num_av_entries: int
  rsvd: int
  rsvd2: c.Array[ctypes.c_uint32, Literal[2]]
struct_mrav_split_entries.register_fields([('mrav_num_av_entries', ctypes.c_uint32, 0), ('rsvd', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_ts_split_entries(c.Struct):
  SIZE = 16
  region_num_entries: int
  tsid: int
  lkup_static_bkt_cnt_exp: c.Array[ctypes.c_ubyte, Literal[2]]
  locked: int
  rsvd2: c.Array[ctypes.c_uint32, Literal[2]]
struct_ts_split_entries.register_fields([('region_num_entries', ctypes.c_uint32, 0), ('tsid', ctypes.c_ubyte, 4), ('lkup_static_bkt_cnt_exp', c.Array[ctypes.c_ubyte, Literal[2]], 5), ('locked', ctypes.c_ubyte, 7), ('rsvd2', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_ck_split_entries(c.Struct):
  SIZE = 16
  num_quic_entries: int
  rsvd: int
  rsvd2: c.Array[ctypes.c_uint32, Literal[2]]
struct_ck_split_entries.register_fields([('num_quic_entries', ctypes.c_uint32, 0), ('rsvd', ctypes.c_uint32, 4), ('rsvd2', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_hwrm_func_backing_store_qcaps_v2_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  type: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_func_backing_store_qcaps_v2_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('type', ctypes.c_uint16, 16), ('rsvd', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_func_backing_store_qcaps_v2_output(c.Struct):
  SIZE = 56
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  type: int
  entry_size: int
  flags: int
  instance_bit_map: int
  ctx_init_value: int
  ctx_init_offset: int
  entry_multiple: int
  rsvd: int
  max_num_entries: int
  min_num_entries: int
  next_valid_type: int
  subtype_valid_cnt: int
  exact_cnt_bit_map: int
  split_entry_0: int
  split_entry_1: int
  split_entry_2: int
  split_entry_3: int
  max_instance_count: int
  rsvd3: int
  valid: int
struct_hwrm_func_backing_store_qcaps_v2_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('type', ctypes.c_uint16, 8), ('entry_size', ctypes.c_uint16, 10), ('flags', ctypes.c_uint32, 12), ('instance_bit_map', ctypes.c_uint32, 16), ('ctx_init_value', ctypes.c_ubyte, 20), ('ctx_init_offset', ctypes.c_ubyte, 21), ('entry_multiple', ctypes.c_ubyte, 22), ('rsvd', ctypes.c_ubyte, 23), ('max_num_entries', ctypes.c_uint32, 24), ('min_num_entries', ctypes.c_uint32, 28), ('next_valid_type', ctypes.c_uint16, 32), ('subtype_valid_cnt', ctypes.c_ubyte, 34), ('exact_cnt_bit_map', ctypes.c_ubyte, 35), ('split_entry_0', ctypes.c_uint32, 36), ('split_entry_1', ctypes.c_uint32, 40), ('split_entry_2', ctypes.c_uint32, 44), ('split_entry_3', ctypes.c_uint32, 48), ('max_instance_count', ctypes.c_uint16, 52), ('rsvd3', ctypes.c_ubyte, 54), ('valid', ctypes.c_ubyte, 55)])
@c.record
class struct_hwrm_func_dbr_pacing_qcfg_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_func_dbr_pacing_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_func_dbr_pacing_qcfg_output(c.Struct):
  SIZE = 64
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  dbr_stat_db_fifo_reg: int
  dbr_stat_db_fifo_reg_watermark_mask: int
  dbr_stat_db_fifo_reg_watermark_shift: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  dbr_stat_db_fifo_reg_fifo_room_mask: int
  dbr_stat_db_fifo_reg_fifo_room_shift: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[3]]
  dbr_throttling_aeq_arm_reg: int
  dbr_throttling_aeq_arm_reg_val: int
  unused_3: c.Array[ctypes.c_ubyte, Literal[3]]
  dbr_stat_db_max_fifo_depth: int
  primary_nq_id: int
  pacing_threshold: int
  unused_4: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_func_dbr_pacing_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 9), ('dbr_stat_db_fifo_reg', ctypes.c_uint32, 16), ('dbr_stat_db_fifo_reg_watermark_mask', ctypes.c_uint32, 20), ('dbr_stat_db_fifo_reg_watermark_shift', ctypes.c_ubyte, 24), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 25), ('dbr_stat_db_fifo_reg_fifo_room_mask', ctypes.c_uint32, 28), ('dbr_stat_db_fifo_reg_fifo_room_shift', ctypes.c_ubyte, 32), ('unused_2', c.Array[ctypes.c_ubyte, Literal[3]], 33), ('dbr_throttling_aeq_arm_reg', ctypes.c_uint32, 36), ('dbr_throttling_aeq_arm_reg_val', ctypes.c_ubyte, 40), ('unused_3', c.Array[ctypes.c_ubyte, Literal[3]], 41), ('dbr_stat_db_max_fifo_depth', ctypes.c_uint32, 44), ('primary_nq_id', ctypes.c_uint32, 48), ('pacing_threshold', ctypes.c_uint32, 52), ('unused_4', c.Array[ctypes.c_ubyte, Literal[7]], 56), ('valid', ctypes.c_ubyte, 63)])
@c.record
class struct_hwrm_func_drv_if_change_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused: int
struct_hwrm_func_drv_if_change_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('unused', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_func_drv_if_change_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_func_drv_if_change_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_phy_cfg_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  force_link_speed: int
  auto_mode: int
  auto_duplex: int
  auto_pause: int
  mgmt_flag: int
  auto_link_speed: int
  auto_link_speed_mask: int
  wirespeed: int
  lpbk: int
  force_pause: int
  unused_1: int
  preemphasis: int
  eee_link_speed_mask: int
  force_pam4_link_speed: int
  tx_lpi_timer: int
  auto_link_pam4_speed_mask: int
  force_link_speeds2: int
  auto_link_speeds2_mask: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_phy_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('force_link_speed', ctypes.c_uint16, 26), ('auto_mode', ctypes.c_ubyte, 28), ('auto_duplex', ctypes.c_ubyte, 29), ('auto_pause', ctypes.c_ubyte, 30), ('mgmt_flag', ctypes.c_ubyte, 31), ('auto_link_speed', ctypes.c_uint16, 32), ('auto_link_speed_mask', ctypes.c_uint16, 34), ('wirespeed', ctypes.c_ubyte, 36), ('lpbk', ctypes.c_ubyte, 37), ('force_pause', ctypes.c_ubyte, 38), ('unused_1', ctypes.c_ubyte, 39), ('preemphasis', ctypes.c_uint32, 40), ('eee_link_speed_mask', ctypes.c_uint16, 44), ('force_pam4_link_speed', ctypes.c_uint16, 46), ('tx_lpi_timer', ctypes.c_uint32, 48), ('auto_link_pam4_speed_mask', ctypes.c_uint16, 52), ('force_link_speeds2', ctypes.c_uint16, 54), ('auto_link_speeds2_mask', ctypes.c_uint16, 56), ('unused_2', c.Array[ctypes.c_ubyte, Literal[6]], 58)])
@c.record
class struct_hwrm_port_phy_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_phy_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_phy_cfg_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_port_phy_cfg_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_port_phy_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_phy_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_phy_qcfg_output(c.Struct):
  SIZE = 104
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  link: int
  active_fec_signal_mode: int
  link_speed: int
  duplex_cfg: int
  pause: int
  support_speeds: int
  force_link_speed: int
  auto_mode: int
  auto_pause: int
  auto_link_speed: int
  auto_link_speed_mask: int
  wirespeed: int
  lpbk: int
  force_pause: int
  module_status: int
  preemphasis: int
  phy_maj: int
  phy_min: int
  phy_bld: int
  phy_type: int
  media_type: int
  xcvr_pkg_type: int
  eee_config_phy_addr: int
  parallel_detect: int
  link_partner_adv_speeds: int
  link_partner_adv_auto_mode: int
  link_partner_adv_pause: int
  adv_eee_link_speed_mask: int
  link_partner_adv_eee_link_speed_mask: int
  xcvr_identifier_type_tx_lpi_timer: int
  fec_cfg: int
  duplex_state: int
  option_flags: int
  phy_vendor_name: c.Array[ctypes.c_char, Literal[16]]
  phy_vendor_partnumber: c.Array[ctypes.c_char, Literal[16]]
  support_pam4_speeds: int
  force_pam4_link_speed: int
  auto_pam4_link_speed_mask: int
  link_partner_pam4_adv_speeds: int
  link_down_reason: int
  support_speeds2: int
  force_link_speeds2: int
  auto_link_speeds2: int
  active_lanes: int
  valid: int
struct_hwrm_port_phy_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('link', ctypes.c_ubyte, 8), ('active_fec_signal_mode', ctypes.c_ubyte, 9), ('link_speed', ctypes.c_uint16, 10), ('duplex_cfg', ctypes.c_ubyte, 12), ('pause', ctypes.c_ubyte, 13), ('support_speeds', ctypes.c_uint16, 14), ('force_link_speed', ctypes.c_uint16, 16), ('auto_mode', ctypes.c_ubyte, 18), ('auto_pause', ctypes.c_ubyte, 19), ('auto_link_speed', ctypes.c_uint16, 20), ('auto_link_speed_mask', ctypes.c_uint16, 22), ('wirespeed', ctypes.c_ubyte, 24), ('lpbk', ctypes.c_ubyte, 25), ('force_pause', ctypes.c_ubyte, 26), ('module_status', ctypes.c_ubyte, 27), ('preemphasis', ctypes.c_uint32, 28), ('phy_maj', ctypes.c_ubyte, 32), ('phy_min', ctypes.c_ubyte, 33), ('phy_bld', ctypes.c_ubyte, 34), ('phy_type', ctypes.c_ubyte, 35), ('media_type', ctypes.c_ubyte, 36), ('xcvr_pkg_type', ctypes.c_ubyte, 37), ('eee_config_phy_addr', ctypes.c_ubyte, 38), ('parallel_detect', ctypes.c_ubyte, 39), ('link_partner_adv_speeds', ctypes.c_uint16, 40), ('link_partner_adv_auto_mode', ctypes.c_ubyte, 42), ('link_partner_adv_pause', ctypes.c_ubyte, 43), ('adv_eee_link_speed_mask', ctypes.c_uint16, 44), ('link_partner_adv_eee_link_speed_mask', ctypes.c_uint16, 46), ('xcvr_identifier_type_tx_lpi_timer', ctypes.c_uint32, 48), ('fec_cfg', ctypes.c_uint16, 52), ('duplex_state', ctypes.c_ubyte, 54), ('option_flags', ctypes.c_ubyte, 55), ('phy_vendor_name', c.Array[ctypes.c_char, Literal[16]], 56), ('phy_vendor_partnumber', c.Array[ctypes.c_char, Literal[16]], 72), ('support_pam4_speeds', ctypes.c_uint16, 88), ('force_pam4_link_speed', ctypes.c_uint16, 90), ('auto_pam4_link_speed_mask', ctypes.c_uint16, 92), ('link_partner_pam4_adv_speeds', ctypes.c_ubyte, 94), ('link_down_reason', ctypes.c_ubyte, 95), ('support_speeds2', ctypes.c_uint16, 96), ('force_link_speeds2', ctypes.c_uint16, 98), ('auto_link_speeds2', ctypes.c_uint16, 100), ('active_lanes', ctypes.c_ubyte, 102), ('valid', ctypes.c_ubyte, 103)])
@c.record
class struct_hwrm_port_mac_cfg_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  ipg: int
  lpbk: int
  vlan_pri2cos_map_pri: int
  reserved1: int
  tunnel_pri2cos_map_pri: int
  dscp2pri_map_pri: int
  rx_ts_capture_ptp_msg_type: int
  tx_ts_capture_ptp_msg_type: int
  cos_field_cfg: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  ptp_freq_adj_ppb: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  ptp_load_control: int
  ptp_adj_phase: int
struct_hwrm_port_mac_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('ipg', ctypes.c_ubyte, 26), ('lpbk', ctypes.c_ubyte, 27), ('vlan_pri2cos_map_pri', ctypes.c_ubyte, 28), ('reserved1', ctypes.c_ubyte, 29), ('tunnel_pri2cos_map_pri', ctypes.c_ubyte, 30), ('dscp2pri_map_pri', ctypes.c_ubyte, 31), ('rx_ts_capture_ptp_msg_type', ctypes.c_uint16, 32), ('tx_ts_capture_ptp_msg_type', ctypes.c_uint16, 34), ('cos_field_cfg', ctypes.c_ubyte, 36), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 37), ('ptp_freq_adj_ppb', ctypes.c_uint32, 40), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 44), ('ptp_load_control', ctypes.c_ubyte, 47), ('ptp_adj_phase', ctypes.c_uint64, 48)])
@c.record
class struct_hwrm_port_mac_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  mru: int
  mtu: int
  ipg: int
  lpbk: int
  unused_0: int
  valid: int
struct_hwrm_port_mac_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('mru', ctypes.c_uint16, 8), ('mtu', ctypes.c_uint16, 10), ('ipg', ctypes.c_ubyte, 12), ('lpbk', ctypes.c_ubyte, 13), ('unused_0', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_mac_ptp_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_mac_ptp_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_mac_ptp_qcfg_output(c.Struct):
  SIZE = 88
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  rx_ts_reg_off_lower: int
  rx_ts_reg_off_upper: int
  rx_ts_reg_off_seq_id: int
  rx_ts_reg_off_src_id_0: int
  rx_ts_reg_off_src_id_1: int
  rx_ts_reg_off_src_id_2: int
  rx_ts_reg_off_domain_id: int
  rx_ts_reg_off_fifo: int
  rx_ts_reg_off_fifo_adv: int
  rx_ts_reg_off_granularity: int
  tx_ts_reg_off_lower: int
  tx_ts_reg_off_upper: int
  tx_ts_reg_off_seq_id: int
  tx_ts_reg_off_fifo: int
  tx_ts_reg_off_granularity: int
  ts_ref_clock_reg_lower: int
  ts_ref_clock_reg_upper: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_mac_ptp_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 9), ('rx_ts_reg_off_lower', ctypes.c_uint32, 12), ('rx_ts_reg_off_upper', ctypes.c_uint32, 16), ('rx_ts_reg_off_seq_id', ctypes.c_uint32, 20), ('rx_ts_reg_off_src_id_0', ctypes.c_uint32, 24), ('rx_ts_reg_off_src_id_1', ctypes.c_uint32, 28), ('rx_ts_reg_off_src_id_2', ctypes.c_uint32, 32), ('rx_ts_reg_off_domain_id', ctypes.c_uint32, 36), ('rx_ts_reg_off_fifo', ctypes.c_uint32, 40), ('rx_ts_reg_off_fifo_adv', ctypes.c_uint32, 44), ('rx_ts_reg_off_granularity', ctypes.c_uint32, 48), ('tx_ts_reg_off_lower', ctypes.c_uint32, 52), ('tx_ts_reg_off_upper', ctypes.c_uint32, 56), ('tx_ts_reg_off_seq_id', ctypes.c_uint32, 60), ('tx_ts_reg_off_fifo', ctypes.c_uint32, 64), ('tx_ts_reg_off_granularity', ctypes.c_uint32, 68), ('ts_ref_clock_reg_lower', ctypes.c_uint32, 72), ('ts_ref_clock_reg_upper', ctypes.c_uint32, 76), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 80), ('valid', ctypes.c_ubyte, 87)])
@c.record
class struct_tx_port_stats(c.Struct):
  SIZE = 408
  tx_64b_frames: int
  tx_65b_127b_frames: int
  tx_128b_255b_frames: int
  tx_256b_511b_frames: int
  tx_512b_1023b_frames: int
  tx_1024b_1518b_frames: int
  tx_good_vlan_frames: int
  tx_1519b_2047b_frames: int
  tx_2048b_4095b_frames: int
  tx_4096b_9216b_frames: int
  tx_9217b_16383b_frames: int
  tx_good_frames: int
  tx_total_frames: int
  tx_ucast_frames: int
  tx_mcast_frames: int
  tx_bcast_frames: int
  tx_pause_frames: int
  tx_pfc_frames: int
  tx_jabber_frames: int
  tx_fcs_err_frames: int
  tx_control_frames: int
  tx_oversz_frames: int
  tx_single_dfrl_frames: int
  tx_multi_dfrl_frames: int
  tx_single_coll_frames: int
  tx_multi_coll_frames: int
  tx_late_coll_frames: int
  tx_excessive_coll_frames: int
  tx_frag_frames: int
  tx_err: int
  tx_tagged_frames: int
  tx_dbl_tagged_frames: int
  tx_runt_frames: int
  tx_fifo_underruns: int
  tx_pfc_ena_frames_pri0: int
  tx_pfc_ena_frames_pri1: int
  tx_pfc_ena_frames_pri2: int
  tx_pfc_ena_frames_pri3: int
  tx_pfc_ena_frames_pri4: int
  tx_pfc_ena_frames_pri5: int
  tx_pfc_ena_frames_pri6: int
  tx_pfc_ena_frames_pri7: int
  tx_eee_lpi_events: int
  tx_eee_lpi_duration: int
  tx_llfc_logical_msgs: int
  tx_hcfc_msgs: int
  tx_total_collisions: int
  tx_bytes: int
  tx_xthol_frames: int
  tx_stat_discard: int
  tx_stat_error: int
struct_tx_port_stats.register_fields([('tx_64b_frames', ctypes.c_uint64, 0), ('tx_65b_127b_frames', ctypes.c_uint64, 8), ('tx_128b_255b_frames', ctypes.c_uint64, 16), ('tx_256b_511b_frames', ctypes.c_uint64, 24), ('tx_512b_1023b_frames', ctypes.c_uint64, 32), ('tx_1024b_1518b_frames', ctypes.c_uint64, 40), ('tx_good_vlan_frames', ctypes.c_uint64, 48), ('tx_1519b_2047b_frames', ctypes.c_uint64, 56), ('tx_2048b_4095b_frames', ctypes.c_uint64, 64), ('tx_4096b_9216b_frames', ctypes.c_uint64, 72), ('tx_9217b_16383b_frames', ctypes.c_uint64, 80), ('tx_good_frames', ctypes.c_uint64, 88), ('tx_total_frames', ctypes.c_uint64, 96), ('tx_ucast_frames', ctypes.c_uint64, 104), ('tx_mcast_frames', ctypes.c_uint64, 112), ('tx_bcast_frames', ctypes.c_uint64, 120), ('tx_pause_frames', ctypes.c_uint64, 128), ('tx_pfc_frames', ctypes.c_uint64, 136), ('tx_jabber_frames', ctypes.c_uint64, 144), ('tx_fcs_err_frames', ctypes.c_uint64, 152), ('tx_control_frames', ctypes.c_uint64, 160), ('tx_oversz_frames', ctypes.c_uint64, 168), ('tx_single_dfrl_frames', ctypes.c_uint64, 176), ('tx_multi_dfrl_frames', ctypes.c_uint64, 184), ('tx_single_coll_frames', ctypes.c_uint64, 192), ('tx_multi_coll_frames', ctypes.c_uint64, 200), ('tx_late_coll_frames', ctypes.c_uint64, 208), ('tx_excessive_coll_frames', ctypes.c_uint64, 216), ('tx_frag_frames', ctypes.c_uint64, 224), ('tx_err', ctypes.c_uint64, 232), ('tx_tagged_frames', ctypes.c_uint64, 240), ('tx_dbl_tagged_frames', ctypes.c_uint64, 248), ('tx_runt_frames', ctypes.c_uint64, 256), ('tx_fifo_underruns', ctypes.c_uint64, 264), ('tx_pfc_ena_frames_pri0', ctypes.c_uint64, 272), ('tx_pfc_ena_frames_pri1', ctypes.c_uint64, 280), ('tx_pfc_ena_frames_pri2', ctypes.c_uint64, 288), ('tx_pfc_ena_frames_pri3', ctypes.c_uint64, 296), ('tx_pfc_ena_frames_pri4', ctypes.c_uint64, 304), ('tx_pfc_ena_frames_pri5', ctypes.c_uint64, 312), ('tx_pfc_ena_frames_pri6', ctypes.c_uint64, 320), ('tx_pfc_ena_frames_pri7', ctypes.c_uint64, 328), ('tx_eee_lpi_events', ctypes.c_uint64, 336), ('tx_eee_lpi_duration', ctypes.c_uint64, 344), ('tx_llfc_logical_msgs', ctypes.c_uint64, 352), ('tx_hcfc_msgs', ctypes.c_uint64, 360), ('tx_total_collisions', ctypes.c_uint64, 368), ('tx_bytes', ctypes.c_uint64, 376), ('tx_xthol_frames', ctypes.c_uint64, 384), ('tx_stat_discard', ctypes.c_uint64, 392), ('tx_stat_error', ctypes.c_uint64, 400)])
@c.record
class struct_rx_port_stats(c.Struct):
  SIZE = 528
  rx_64b_frames: int
  rx_65b_127b_frames: int
  rx_128b_255b_frames: int
  rx_256b_511b_frames: int
  rx_512b_1023b_frames: int
  rx_1024b_1518b_frames: int
  rx_good_vlan_frames: int
  rx_1519b_2047b_frames: int
  rx_2048b_4095b_frames: int
  rx_4096b_9216b_frames: int
  rx_9217b_16383b_frames: int
  rx_total_frames: int
  rx_ucast_frames: int
  rx_mcast_frames: int
  rx_bcast_frames: int
  rx_fcs_err_frames: int
  rx_ctrl_frames: int
  rx_pause_frames: int
  rx_pfc_frames: int
  rx_unsupported_opcode_frames: int
  rx_unsupported_da_pausepfc_frames: int
  rx_wrong_sa_frames: int
  rx_align_err_frames: int
  rx_oor_len_frames: int
  rx_code_err_frames: int
  rx_false_carrier_frames: int
  rx_ovrsz_frames: int
  rx_jbr_frames: int
  rx_mtu_err_frames: int
  rx_match_crc_frames: int
  rx_promiscuous_frames: int
  rx_tagged_frames: int
  rx_double_tagged_frames: int
  rx_trunc_frames: int
  rx_good_frames: int
  rx_pfc_xon2xoff_frames_pri0: int
  rx_pfc_xon2xoff_frames_pri1: int
  rx_pfc_xon2xoff_frames_pri2: int
  rx_pfc_xon2xoff_frames_pri3: int
  rx_pfc_xon2xoff_frames_pri4: int
  rx_pfc_xon2xoff_frames_pri5: int
  rx_pfc_xon2xoff_frames_pri6: int
  rx_pfc_xon2xoff_frames_pri7: int
  rx_pfc_ena_frames_pri0: int
  rx_pfc_ena_frames_pri1: int
  rx_pfc_ena_frames_pri2: int
  rx_pfc_ena_frames_pri3: int
  rx_pfc_ena_frames_pri4: int
  rx_pfc_ena_frames_pri5: int
  rx_pfc_ena_frames_pri6: int
  rx_pfc_ena_frames_pri7: int
  rx_sch_crc_err_frames: int
  rx_undrsz_frames: int
  rx_frag_frames: int
  rx_eee_lpi_events: int
  rx_eee_lpi_duration: int
  rx_llfc_physical_msgs: int
  rx_llfc_logical_msgs: int
  rx_llfc_msgs_with_crc_err: int
  rx_hcfc_msgs: int
  rx_hcfc_msgs_with_crc_err: int
  rx_bytes: int
  rx_runt_bytes: int
  rx_runt_frames: int
  rx_stat_discard: int
  rx_stat_err: int
struct_rx_port_stats.register_fields([('rx_64b_frames', ctypes.c_uint64, 0), ('rx_65b_127b_frames', ctypes.c_uint64, 8), ('rx_128b_255b_frames', ctypes.c_uint64, 16), ('rx_256b_511b_frames', ctypes.c_uint64, 24), ('rx_512b_1023b_frames', ctypes.c_uint64, 32), ('rx_1024b_1518b_frames', ctypes.c_uint64, 40), ('rx_good_vlan_frames', ctypes.c_uint64, 48), ('rx_1519b_2047b_frames', ctypes.c_uint64, 56), ('rx_2048b_4095b_frames', ctypes.c_uint64, 64), ('rx_4096b_9216b_frames', ctypes.c_uint64, 72), ('rx_9217b_16383b_frames', ctypes.c_uint64, 80), ('rx_total_frames', ctypes.c_uint64, 88), ('rx_ucast_frames', ctypes.c_uint64, 96), ('rx_mcast_frames', ctypes.c_uint64, 104), ('rx_bcast_frames', ctypes.c_uint64, 112), ('rx_fcs_err_frames', ctypes.c_uint64, 120), ('rx_ctrl_frames', ctypes.c_uint64, 128), ('rx_pause_frames', ctypes.c_uint64, 136), ('rx_pfc_frames', ctypes.c_uint64, 144), ('rx_unsupported_opcode_frames', ctypes.c_uint64, 152), ('rx_unsupported_da_pausepfc_frames', ctypes.c_uint64, 160), ('rx_wrong_sa_frames', ctypes.c_uint64, 168), ('rx_align_err_frames', ctypes.c_uint64, 176), ('rx_oor_len_frames', ctypes.c_uint64, 184), ('rx_code_err_frames', ctypes.c_uint64, 192), ('rx_false_carrier_frames', ctypes.c_uint64, 200), ('rx_ovrsz_frames', ctypes.c_uint64, 208), ('rx_jbr_frames', ctypes.c_uint64, 216), ('rx_mtu_err_frames', ctypes.c_uint64, 224), ('rx_match_crc_frames', ctypes.c_uint64, 232), ('rx_promiscuous_frames', ctypes.c_uint64, 240), ('rx_tagged_frames', ctypes.c_uint64, 248), ('rx_double_tagged_frames', ctypes.c_uint64, 256), ('rx_trunc_frames', ctypes.c_uint64, 264), ('rx_good_frames', ctypes.c_uint64, 272), ('rx_pfc_xon2xoff_frames_pri0', ctypes.c_uint64, 280), ('rx_pfc_xon2xoff_frames_pri1', ctypes.c_uint64, 288), ('rx_pfc_xon2xoff_frames_pri2', ctypes.c_uint64, 296), ('rx_pfc_xon2xoff_frames_pri3', ctypes.c_uint64, 304), ('rx_pfc_xon2xoff_frames_pri4', ctypes.c_uint64, 312), ('rx_pfc_xon2xoff_frames_pri5', ctypes.c_uint64, 320), ('rx_pfc_xon2xoff_frames_pri6', ctypes.c_uint64, 328), ('rx_pfc_xon2xoff_frames_pri7', ctypes.c_uint64, 336), ('rx_pfc_ena_frames_pri0', ctypes.c_uint64, 344), ('rx_pfc_ena_frames_pri1', ctypes.c_uint64, 352), ('rx_pfc_ena_frames_pri2', ctypes.c_uint64, 360), ('rx_pfc_ena_frames_pri3', ctypes.c_uint64, 368), ('rx_pfc_ena_frames_pri4', ctypes.c_uint64, 376), ('rx_pfc_ena_frames_pri5', ctypes.c_uint64, 384), ('rx_pfc_ena_frames_pri6', ctypes.c_uint64, 392), ('rx_pfc_ena_frames_pri7', ctypes.c_uint64, 400), ('rx_sch_crc_err_frames', ctypes.c_uint64, 408), ('rx_undrsz_frames', ctypes.c_uint64, 416), ('rx_frag_frames', ctypes.c_uint64, 424), ('rx_eee_lpi_events', ctypes.c_uint64, 432), ('rx_eee_lpi_duration', ctypes.c_uint64, 440), ('rx_llfc_physical_msgs', ctypes.c_uint64, 448), ('rx_llfc_logical_msgs', ctypes.c_uint64, 456), ('rx_llfc_msgs_with_crc_err', ctypes.c_uint64, 464), ('rx_hcfc_msgs', ctypes.c_uint64, 472), ('rx_hcfc_msgs_with_crc_err', ctypes.c_uint64, 480), ('rx_bytes', ctypes.c_uint64, 488), ('rx_runt_bytes', ctypes.c_uint64, 496), ('rx_runt_frames', ctypes.c_uint64, 504), ('rx_stat_discard', ctypes.c_uint64, 512), ('rx_stat_err', ctypes.c_uint64, 520)])
@c.record
class struct_hwrm_port_qstats_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  tx_stat_host_addr: int
  rx_stat_host_addr: int
struct_hwrm_port_qstats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('flags', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19), ('tx_stat_host_addr', ctypes.c_uint64, 24), ('rx_stat_host_addr', ctypes.c_uint64, 32)])
@c.record
class struct_hwrm_port_qstats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tx_stat_size: int
  rx_stat_size: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  valid: int
struct_hwrm_port_qstats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tx_stat_size', ctypes.c_uint16, 8), ('rx_stat_size', ctypes.c_uint16, 10), ('flags', ctypes.c_ubyte, 12), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 13), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_tx_port_stats_ext(c.Struct):
  SIZE = 256
  tx_bytes_cos0: int
  tx_bytes_cos1: int
  tx_bytes_cos2: int
  tx_bytes_cos3: int
  tx_bytes_cos4: int
  tx_bytes_cos5: int
  tx_bytes_cos6: int
  tx_bytes_cos7: int
  tx_packets_cos0: int
  tx_packets_cos1: int
  tx_packets_cos2: int
  tx_packets_cos3: int
  tx_packets_cos4: int
  tx_packets_cos5: int
  tx_packets_cos6: int
  tx_packets_cos7: int
  pfc_pri0_tx_duration_us: int
  pfc_pri0_tx_transitions: int
  pfc_pri1_tx_duration_us: int
  pfc_pri1_tx_transitions: int
  pfc_pri2_tx_duration_us: int
  pfc_pri2_tx_transitions: int
  pfc_pri3_tx_duration_us: int
  pfc_pri3_tx_transitions: int
  pfc_pri4_tx_duration_us: int
  pfc_pri4_tx_transitions: int
  pfc_pri5_tx_duration_us: int
  pfc_pri5_tx_transitions: int
  pfc_pri6_tx_duration_us: int
  pfc_pri6_tx_transitions: int
  pfc_pri7_tx_duration_us: int
  pfc_pri7_tx_transitions: int
struct_tx_port_stats_ext.register_fields([('tx_bytes_cos0', ctypes.c_uint64, 0), ('tx_bytes_cos1', ctypes.c_uint64, 8), ('tx_bytes_cos2', ctypes.c_uint64, 16), ('tx_bytes_cos3', ctypes.c_uint64, 24), ('tx_bytes_cos4', ctypes.c_uint64, 32), ('tx_bytes_cos5', ctypes.c_uint64, 40), ('tx_bytes_cos6', ctypes.c_uint64, 48), ('tx_bytes_cos7', ctypes.c_uint64, 56), ('tx_packets_cos0', ctypes.c_uint64, 64), ('tx_packets_cos1', ctypes.c_uint64, 72), ('tx_packets_cos2', ctypes.c_uint64, 80), ('tx_packets_cos3', ctypes.c_uint64, 88), ('tx_packets_cos4', ctypes.c_uint64, 96), ('tx_packets_cos5', ctypes.c_uint64, 104), ('tx_packets_cos6', ctypes.c_uint64, 112), ('tx_packets_cos7', ctypes.c_uint64, 120), ('pfc_pri0_tx_duration_us', ctypes.c_uint64, 128), ('pfc_pri0_tx_transitions', ctypes.c_uint64, 136), ('pfc_pri1_tx_duration_us', ctypes.c_uint64, 144), ('pfc_pri1_tx_transitions', ctypes.c_uint64, 152), ('pfc_pri2_tx_duration_us', ctypes.c_uint64, 160), ('pfc_pri2_tx_transitions', ctypes.c_uint64, 168), ('pfc_pri3_tx_duration_us', ctypes.c_uint64, 176), ('pfc_pri3_tx_transitions', ctypes.c_uint64, 184), ('pfc_pri4_tx_duration_us', ctypes.c_uint64, 192), ('pfc_pri4_tx_transitions', ctypes.c_uint64, 200), ('pfc_pri5_tx_duration_us', ctypes.c_uint64, 208), ('pfc_pri5_tx_transitions', ctypes.c_uint64, 216), ('pfc_pri6_tx_duration_us', ctypes.c_uint64, 224), ('pfc_pri6_tx_transitions', ctypes.c_uint64, 232), ('pfc_pri7_tx_duration_us', ctypes.c_uint64, 240), ('pfc_pri7_tx_transitions', ctypes.c_uint64, 248)])
@c.record
class struct_rx_port_stats_ext(c.Struct):
  SIZE = 488
  link_down_events: int
  continuous_pause_events: int
  resume_pause_events: int
  continuous_roce_pause_events: int
  resume_roce_pause_events: int
  rx_bytes_cos0: int
  rx_bytes_cos1: int
  rx_bytes_cos2: int
  rx_bytes_cos3: int
  rx_bytes_cos4: int
  rx_bytes_cos5: int
  rx_bytes_cos6: int
  rx_bytes_cos7: int
  rx_packets_cos0: int
  rx_packets_cos1: int
  rx_packets_cos2: int
  rx_packets_cos3: int
  rx_packets_cos4: int
  rx_packets_cos5: int
  rx_packets_cos6: int
  rx_packets_cos7: int
  pfc_pri0_rx_duration_us: int
  pfc_pri0_rx_transitions: int
  pfc_pri1_rx_duration_us: int
  pfc_pri1_rx_transitions: int
  pfc_pri2_rx_duration_us: int
  pfc_pri2_rx_transitions: int
  pfc_pri3_rx_duration_us: int
  pfc_pri3_rx_transitions: int
  pfc_pri4_rx_duration_us: int
  pfc_pri4_rx_transitions: int
  pfc_pri5_rx_duration_us: int
  pfc_pri5_rx_transitions: int
  pfc_pri6_rx_duration_us: int
  pfc_pri6_rx_transitions: int
  pfc_pri7_rx_duration_us: int
  pfc_pri7_rx_transitions: int
  rx_bits: int
  rx_buffer_passed_threshold: int
  rx_pcs_symbol_err: int
  rx_corrected_bits: int
  rx_discard_bytes_cos0: int
  rx_discard_bytes_cos1: int
  rx_discard_bytes_cos2: int
  rx_discard_bytes_cos3: int
  rx_discard_bytes_cos4: int
  rx_discard_bytes_cos5: int
  rx_discard_bytes_cos6: int
  rx_discard_bytes_cos7: int
  rx_discard_packets_cos0: int
  rx_discard_packets_cos1: int
  rx_discard_packets_cos2: int
  rx_discard_packets_cos3: int
  rx_discard_packets_cos4: int
  rx_discard_packets_cos5: int
  rx_discard_packets_cos6: int
  rx_discard_packets_cos7: int
  rx_fec_corrected_blocks: int
  rx_fec_uncorrectable_blocks: int
  rx_filter_miss: int
  rx_fec_symbol_err: int
struct_rx_port_stats_ext.register_fields([('link_down_events', ctypes.c_uint64, 0), ('continuous_pause_events', ctypes.c_uint64, 8), ('resume_pause_events', ctypes.c_uint64, 16), ('continuous_roce_pause_events', ctypes.c_uint64, 24), ('resume_roce_pause_events', ctypes.c_uint64, 32), ('rx_bytes_cos0', ctypes.c_uint64, 40), ('rx_bytes_cos1', ctypes.c_uint64, 48), ('rx_bytes_cos2', ctypes.c_uint64, 56), ('rx_bytes_cos3', ctypes.c_uint64, 64), ('rx_bytes_cos4', ctypes.c_uint64, 72), ('rx_bytes_cos5', ctypes.c_uint64, 80), ('rx_bytes_cos6', ctypes.c_uint64, 88), ('rx_bytes_cos7', ctypes.c_uint64, 96), ('rx_packets_cos0', ctypes.c_uint64, 104), ('rx_packets_cos1', ctypes.c_uint64, 112), ('rx_packets_cos2', ctypes.c_uint64, 120), ('rx_packets_cos3', ctypes.c_uint64, 128), ('rx_packets_cos4', ctypes.c_uint64, 136), ('rx_packets_cos5', ctypes.c_uint64, 144), ('rx_packets_cos6', ctypes.c_uint64, 152), ('rx_packets_cos7', ctypes.c_uint64, 160), ('pfc_pri0_rx_duration_us', ctypes.c_uint64, 168), ('pfc_pri0_rx_transitions', ctypes.c_uint64, 176), ('pfc_pri1_rx_duration_us', ctypes.c_uint64, 184), ('pfc_pri1_rx_transitions', ctypes.c_uint64, 192), ('pfc_pri2_rx_duration_us', ctypes.c_uint64, 200), ('pfc_pri2_rx_transitions', ctypes.c_uint64, 208), ('pfc_pri3_rx_duration_us', ctypes.c_uint64, 216), ('pfc_pri3_rx_transitions', ctypes.c_uint64, 224), ('pfc_pri4_rx_duration_us', ctypes.c_uint64, 232), ('pfc_pri4_rx_transitions', ctypes.c_uint64, 240), ('pfc_pri5_rx_duration_us', ctypes.c_uint64, 248), ('pfc_pri5_rx_transitions', ctypes.c_uint64, 256), ('pfc_pri6_rx_duration_us', ctypes.c_uint64, 264), ('pfc_pri6_rx_transitions', ctypes.c_uint64, 272), ('pfc_pri7_rx_duration_us', ctypes.c_uint64, 280), ('pfc_pri7_rx_transitions', ctypes.c_uint64, 288), ('rx_bits', ctypes.c_uint64, 296), ('rx_buffer_passed_threshold', ctypes.c_uint64, 304), ('rx_pcs_symbol_err', ctypes.c_uint64, 312), ('rx_corrected_bits', ctypes.c_uint64, 320), ('rx_discard_bytes_cos0', ctypes.c_uint64, 328), ('rx_discard_bytes_cos1', ctypes.c_uint64, 336), ('rx_discard_bytes_cos2', ctypes.c_uint64, 344), ('rx_discard_bytes_cos3', ctypes.c_uint64, 352), ('rx_discard_bytes_cos4', ctypes.c_uint64, 360), ('rx_discard_bytes_cos5', ctypes.c_uint64, 368), ('rx_discard_bytes_cos6', ctypes.c_uint64, 376), ('rx_discard_bytes_cos7', ctypes.c_uint64, 384), ('rx_discard_packets_cos0', ctypes.c_uint64, 392), ('rx_discard_packets_cos1', ctypes.c_uint64, 400), ('rx_discard_packets_cos2', ctypes.c_uint64, 408), ('rx_discard_packets_cos3', ctypes.c_uint64, 416), ('rx_discard_packets_cos4', ctypes.c_uint64, 424), ('rx_discard_packets_cos5', ctypes.c_uint64, 432), ('rx_discard_packets_cos6', ctypes.c_uint64, 440), ('rx_discard_packets_cos7', ctypes.c_uint64, 448), ('rx_fec_corrected_blocks', ctypes.c_uint64, 456), ('rx_fec_uncorrectable_blocks', ctypes.c_uint64, 464), ('rx_filter_miss', ctypes.c_uint64, 472), ('rx_fec_symbol_err', ctypes.c_uint64, 480)])
@c.record
class struct_hwrm_port_qstats_ext_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  tx_stat_size: int
  rx_stat_size: int
  flags: int
  unused_0: int
  tx_stat_host_addr: int
  rx_stat_host_addr: int
struct_hwrm_port_qstats_ext_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('tx_stat_size', ctypes.c_uint16, 18), ('rx_stat_size', ctypes.c_uint16, 20), ('flags', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23), ('tx_stat_host_addr', ctypes.c_uint64, 24), ('rx_stat_host_addr', ctypes.c_uint64, 32)])
@c.record
class struct_hwrm_port_qstats_ext_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tx_stat_size: int
  rx_stat_size: int
  total_active_cos_queues: int
  flags: int
  valid: int
struct_hwrm_port_qstats_ext_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tx_stat_size', ctypes.c_uint16, 8), ('rx_stat_size', ctypes.c_uint16, 10), ('total_active_cos_queues', ctypes.c_uint16, 12), ('flags', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_lpbk_qstats_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  lpbk_stat_size: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  lpbk_stat_host_addr: int
struct_hwrm_port_lpbk_qstats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('lpbk_stat_size', ctypes.c_uint16, 16), ('flags', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19), ('lpbk_stat_host_addr', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_port_lpbk_qstats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  lpbk_stat_size: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_port_lpbk_qstats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('lpbk_stat_size', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_port_lpbk_stats(c.Struct):
  SIZE = 80
  lpbk_ucast_frames: int
  lpbk_mcast_frames: int
  lpbk_bcast_frames: int
  lpbk_ucast_bytes: int
  lpbk_mcast_bytes: int
  lpbk_bcast_bytes: int
  lpbk_tx_discards: int
  lpbk_tx_errors: int
  lpbk_rx_discards: int
  lpbk_rx_errors: int
struct_port_lpbk_stats.register_fields([('lpbk_ucast_frames', ctypes.c_uint64, 0), ('lpbk_mcast_frames', ctypes.c_uint64, 8), ('lpbk_bcast_frames', ctypes.c_uint64, 16), ('lpbk_ucast_bytes', ctypes.c_uint64, 24), ('lpbk_mcast_bytes', ctypes.c_uint64, 32), ('lpbk_bcast_bytes', ctypes.c_uint64, 40), ('lpbk_tx_discards', ctypes.c_uint64, 48), ('lpbk_tx_errors', ctypes.c_uint64, 56), ('lpbk_rx_discards', ctypes.c_uint64, 64), ('lpbk_rx_errors', ctypes.c_uint64, 72)])
@c.record
class struct_hwrm_port_ecn_qstats_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  ecn_stat_buf_size: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  ecn_stat_host_addr: int
struct_hwrm_port_ecn_qstats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('ecn_stat_buf_size', ctypes.c_uint16, 18), ('flags', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21), ('ecn_stat_host_addr', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_port_ecn_qstats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  ecn_stat_buf_size: int
  mark_en: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  valid: int
struct_hwrm_port_ecn_qstats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('ecn_stat_buf_size', ctypes.c_uint16, 8), ('mark_en', ctypes.c_ubyte, 10), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 11), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_port_stats_ecn(c.Struct):
  SIZE = 64
  mark_cnt_cos0: int
  mark_cnt_cos1: int
  mark_cnt_cos2: int
  mark_cnt_cos3: int
  mark_cnt_cos4: int
  mark_cnt_cos5: int
  mark_cnt_cos6: int
  mark_cnt_cos7: int
struct_port_stats_ecn.register_fields([('mark_cnt_cos0', ctypes.c_uint64, 0), ('mark_cnt_cos1', ctypes.c_uint64, 8), ('mark_cnt_cos2', ctypes.c_uint64, 16), ('mark_cnt_cos3', ctypes.c_uint64, 24), ('mark_cnt_cos4', ctypes.c_uint64, 32), ('mark_cnt_cos5', ctypes.c_uint64, 40), ('mark_cnt_cos6', ctypes.c_uint64, 48), ('mark_cnt_cos7', ctypes.c_uint64, 56)])
@c.record
class struct_hwrm_port_clr_stats_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
struct_hwrm_port_clr_stats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('flags', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19)])
@c.record
class struct_hwrm_port_clr_stats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_clr_stats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_lpbk_clr_stats_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_lpbk_clr_stats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_lpbk_clr_stats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_lpbk_clr_stats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_ts_query_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  enables: int
  ts_req_timeout: int
  ptp_seq_id: int
  ptp_hdr_offset: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_ts_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('port_id', ctypes.c_uint16, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 22), ('enables', ctypes.c_uint16, 24), ('ts_req_timeout', ctypes.c_uint16, 26), ('ptp_seq_id', ctypes.c_uint32, 28), ('ptp_hdr_offset', ctypes.c_uint16, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 34)])
@c.record
class struct_hwrm_port_ts_query_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  ptp_msg_ts: int
  ptp_msg_seqid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_port_ts_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('ptp_msg_ts', ctypes.c_uint64, 8), ('ptp_msg_seqid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 18), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_port_phy_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_phy_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_phy_qcaps_output(c.Struct):
  SIZE = 40
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  port_cnt: int
  supported_speeds_force_mode: int
  supported_speeds_auto_mode: int
  supported_speeds_eee_mode: int
  tx_lpi_timer_low: int
  valid_tx_lpi_timer_high: int
  supported_pam4_speeds_auto_mode: int
  supported_pam4_speeds_force_mode: int
  flags2: int
  internal_port_cnt: int
  unused_0: int
  supported_speeds2_force_mode: int
  supported_speeds2_auto_mode: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_port_phy_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('port_cnt', ctypes.c_ubyte, 9), ('supported_speeds_force_mode', ctypes.c_uint16, 10), ('supported_speeds_auto_mode', ctypes.c_uint16, 12), ('supported_speeds_eee_mode', ctypes.c_uint16, 14), ('tx_lpi_timer_low', ctypes.c_uint32, 16), ('valid_tx_lpi_timer_high', ctypes.c_uint32, 20), ('supported_pam4_speeds_auto_mode', ctypes.c_uint16, 24), ('supported_pam4_speeds_force_mode', ctypes.c_uint16, 26), ('flags2', ctypes.c_uint16, 28), ('internal_port_cnt', ctypes.c_ubyte, 30), ('unused_0', ctypes.c_ubyte, 31), ('supported_speeds2_force_mode', ctypes.c_uint16, 32), ('supported_speeds2_auto_mode', ctypes.c_uint16, 34), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 36), ('valid', ctypes.c_ubyte, 39)])
@c.record
class struct_hwrm_port_phy_i2c_write_input(c.Struct):
  SIZE = 104
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  i2c_slave_addr: int
  bank_number: int
  page_number: int
  page_offset: int
  data_length: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
  data: c.Array[ctypes.c_uint32, Literal[16]]
struct_hwrm_port_phy_i2c_write_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('i2c_slave_addr', ctypes.c_ubyte, 26), ('bank_number', ctypes.c_ubyte, 27), ('page_number', ctypes.c_uint16, 28), ('page_offset', ctypes.c_uint16, 30), ('data_length', ctypes.c_ubyte, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 33), ('data', c.Array[ctypes.c_uint32, Literal[16]], 40)])
@c.record
class struct_hwrm_port_phy_i2c_write_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_phy_i2c_write_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_phy_i2c_read_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  i2c_slave_addr: int
  bank_number: int
  page_number: int
  page_offset: int
  data_length: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_port_phy_i2c_read_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('i2c_slave_addr', ctypes.c_ubyte, 26), ('bank_number', ctypes.c_ubyte, 27), ('page_number', ctypes.c_uint16, 28), ('page_offset', ctypes.c_uint16, 30), ('data_length', ctypes.c_ubyte, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 33)])
@c.record
class struct_hwrm_port_phy_i2c_read_output(c.Struct):
  SIZE = 80
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  data: c.Array[ctypes.c_uint32, Literal[16]]
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_phy_i2c_read_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('data', c.Array[ctypes.c_uint32, Literal[16]], 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 72), ('valid', ctypes.c_ubyte, 79)])
@c.record
class struct_hwrm_port_phy_mdio_write_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  unused_0: c.Array[ctypes.c_uint32, Literal[2]]
  port_id: int
  phy_addr: int
  dev_addr: int
  reg_addr: int
  reg_data: int
  cl45_mdio: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_port_phy_mdio_write_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('unused_0', c.Array[ctypes.c_uint32, Literal[2]], 16), ('port_id', ctypes.c_uint16, 24), ('phy_addr', ctypes.c_ubyte, 26), ('dev_addr', ctypes.c_ubyte, 27), ('reg_addr', ctypes.c_uint16, 28), ('reg_data', ctypes.c_uint16, 30), ('cl45_mdio', ctypes.c_ubyte, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 33)])
@c.record
class struct_hwrm_port_phy_mdio_write_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_phy_mdio_write_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_phy_mdio_read_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  unused_0: c.Array[ctypes.c_uint32, Literal[2]]
  port_id: int
  phy_addr: int
  dev_addr: int
  reg_addr: int
  cl45_mdio: int
  unused_1: int
struct_hwrm_port_phy_mdio_read_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('unused_0', c.Array[ctypes.c_uint32, Literal[2]], 16), ('port_id', ctypes.c_uint16, 24), ('phy_addr', ctypes.c_ubyte, 26), ('dev_addr', ctypes.c_ubyte, 27), ('reg_addr', ctypes.c_uint16, 28), ('cl45_mdio', ctypes.c_ubyte, 30), ('unused_1', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_port_phy_mdio_read_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  reg_data: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_port_phy_mdio_read_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('reg_data', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_led_cfg_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  port_id: int
  num_leds: int
  rsvd: int
  led0_id: int
  led0_state: int
  led0_color: int
  unused_0: int
  led0_blink_on: int
  led0_blink_off: int
  led0_group_id: int
  rsvd0: int
  led1_id: int
  led1_state: int
  led1_color: int
  unused_1: int
  led1_blink_on: int
  led1_blink_off: int
  led1_group_id: int
  rsvd1: int
  led2_id: int
  led2_state: int
  led2_color: int
  unused_2: int
  led2_blink_on: int
  led2_blink_off: int
  led2_group_id: int
  rsvd2: int
  led3_id: int
  led3_state: int
  led3_color: int
  unused_3: int
  led3_blink_on: int
  led3_blink_off: int
  led3_group_id: int
  rsvd3: int
struct_hwrm_port_led_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('port_id', ctypes.c_uint16, 20), ('num_leds', ctypes.c_ubyte, 22), ('rsvd', ctypes.c_ubyte, 23), ('led0_id', ctypes.c_ubyte, 24), ('led0_state', ctypes.c_ubyte, 25), ('led0_color', ctypes.c_ubyte, 26), ('unused_0', ctypes.c_ubyte, 27), ('led0_blink_on', ctypes.c_uint16, 28), ('led0_blink_off', ctypes.c_uint16, 30), ('led0_group_id', ctypes.c_ubyte, 32), ('rsvd0', ctypes.c_ubyte, 33), ('led1_id', ctypes.c_ubyte, 34), ('led1_state', ctypes.c_ubyte, 35), ('led1_color', ctypes.c_ubyte, 36), ('unused_1', ctypes.c_ubyte, 37), ('led1_blink_on', ctypes.c_uint16, 38), ('led1_blink_off', ctypes.c_uint16, 40), ('led1_group_id', ctypes.c_ubyte, 42), ('rsvd1', ctypes.c_ubyte, 43), ('led2_id', ctypes.c_ubyte, 44), ('led2_state', ctypes.c_ubyte, 45), ('led2_color', ctypes.c_ubyte, 46), ('unused_2', ctypes.c_ubyte, 47), ('led2_blink_on', ctypes.c_uint16, 48), ('led2_blink_off', ctypes.c_uint16, 50), ('led2_group_id', ctypes.c_ubyte, 52), ('rsvd2', ctypes.c_ubyte, 53), ('led3_id', ctypes.c_ubyte, 54), ('led3_state', ctypes.c_ubyte, 55), ('led3_color', ctypes.c_ubyte, 56), ('unused_3', ctypes.c_ubyte, 57), ('led3_blink_on', ctypes.c_uint16, 58), ('led3_blink_off', ctypes.c_uint16, 60), ('led3_group_id', ctypes.c_ubyte, 62), ('rsvd3', ctypes.c_ubyte, 63)])
@c.record
class struct_hwrm_port_led_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_port_led_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_port_led_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_led_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_led_qcfg_output(c.Struct):
  SIZE = 56
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  num_leds: int
  led0_id: int
  led0_type: int
  led0_state: int
  led0_color: int
  unused_0: int
  led0_blink_on: int
  led0_blink_off: int
  led0_group_id: int
  led1_id: int
  led1_type: int
  led1_state: int
  led1_color: int
  unused_1: int
  led1_blink_on: int
  led1_blink_off: int
  led1_group_id: int
  led2_id: int
  led2_type: int
  led2_state: int
  led2_color: int
  unused_2: int
  led2_blink_on: int
  led2_blink_off: int
  led2_group_id: int
  led3_id: int
  led3_type: int
  led3_state: int
  led3_color: int
  unused_3: int
  led3_blink_on: int
  led3_blink_off: int
  led3_group_id: int
  unused_4: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_port_led_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('num_leds', ctypes.c_ubyte, 8), ('led0_id', ctypes.c_ubyte, 9), ('led0_type', ctypes.c_ubyte, 10), ('led0_state', ctypes.c_ubyte, 11), ('led0_color', ctypes.c_ubyte, 12), ('unused_0', ctypes.c_ubyte, 13), ('led0_blink_on', ctypes.c_uint16, 14), ('led0_blink_off', ctypes.c_uint16, 16), ('led0_group_id', ctypes.c_ubyte, 18), ('led1_id', ctypes.c_ubyte, 19), ('led1_type', ctypes.c_ubyte, 20), ('led1_state', ctypes.c_ubyte, 21), ('led1_color', ctypes.c_ubyte, 22), ('unused_1', ctypes.c_ubyte, 23), ('led1_blink_on', ctypes.c_uint16, 24), ('led1_blink_off', ctypes.c_uint16, 26), ('led1_group_id', ctypes.c_ubyte, 28), ('led2_id', ctypes.c_ubyte, 29), ('led2_type', ctypes.c_ubyte, 30), ('led2_state', ctypes.c_ubyte, 31), ('led2_color', ctypes.c_ubyte, 32), ('unused_2', ctypes.c_ubyte, 33), ('led2_blink_on', ctypes.c_uint16, 34), ('led2_blink_off', ctypes.c_uint16, 36), ('led2_group_id', ctypes.c_ubyte, 38), ('led3_id', ctypes.c_ubyte, 39), ('led3_type', ctypes.c_ubyte, 40), ('led3_state', ctypes.c_ubyte, 41), ('led3_color', ctypes.c_ubyte, 42), ('unused_3', ctypes.c_ubyte, 43), ('led3_blink_on', ctypes.c_uint16, 44), ('led3_blink_off', ctypes.c_uint16, 46), ('led3_group_id', ctypes.c_ubyte, 48), ('unused_4', c.Array[ctypes.c_ubyte, Literal[6]], 49), ('valid', ctypes.c_ubyte, 55)])
@c.record
class struct_hwrm_port_led_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_led_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_led_qcaps_output(c.Struct):
  SIZE = 48
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  num_leds: int
  unused: c.Array[ctypes.c_ubyte, Literal[3]]
  led0_id: int
  led0_type: int
  led0_group_id: int
  unused_0: int
  led0_state_caps: int
  led0_color_caps: int
  led1_id: int
  led1_type: int
  led1_group_id: int
  unused_1: int
  led1_state_caps: int
  led1_color_caps: int
  led2_id: int
  led2_type: int
  led2_group_id: int
  unused_2: int
  led2_state_caps: int
  led2_color_caps: int
  led3_id: int
  led3_type: int
  led3_group_id: int
  unused_3: int
  led3_state_caps: int
  led3_color_caps: int
  unused_4: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_port_led_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('num_leds', ctypes.c_ubyte, 8), ('unused', c.Array[ctypes.c_ubyte, Literal[3]], 9), ('led0_id', ctypes.c_ubyte, 12), ('led0_type', ctypes.c_ubyte, 13), ('led0_group_id', ctypes.c_ubyte, 14), ('unused_0', ctypes.c_ubyte, 15), ('led0_state_caps', ctypes.c_uint16, 16), ('led0_color_caps', ctypes.c_uint16, 18), ('led1_id', ctypes.c_ubyte, 20), ('led1_type', ctypes.c_ubyte, 21), ('led1_group_id', ctypes.c_ubyte, 22), ('unused_1', ctypes.c_ubyte, 23), ('led1_state_caps', ctypes.c_uint16, 24), ('led1_color_caps', ctypes.c_uint16, 26), ('led2_id', ctypes.c_ubyte, 28), ('led2_type', ctypes.c_ubyte, 29), ('led2_group_id', ctypes.c_ubyte, 30), ('unused_2', ctypes.c_ubyte, 31), ('led2_state_caps', ctypes.c_uint16, 32), ('led2_color_caps', ctypes.c_uint16, 34), ('led3_id', ctypes.c_ubyte, 36), ('led3_type', ctypes.c_ubyte, 37), ('led3_group_id', ctypes.c_ubyte, 38), ('unused_3', ctypes.c_ubyte, 39), ('led3_state_caps', ctypes.c_uint16, 40), ('led3_color_caps', ctypes.c_uint16, 42), ('unused_4', c.Array[ctypes.c_ubyte, Literal[3]], 44), ('valid', ctypes.c_ubyte, 47)])
@c.record
class struct_hwrm_port_mac_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_port_mac_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_port_mac_qcaps_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_port_mac_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 9), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_qportcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  port_id: int
  drv_qmap_cap: int
  unused_0: int
struct_hwrm_queue_qportcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('port_id', ctypes.c_uint16, 20), ('drv_qmap_cap', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_queue_qportcfg_output(c.Struct):
  SIZE = 168
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  max_configurable_queues: int
  max_configurable_lossless_queues: int
  queue_cfg_allowed: int
  queue_cfg_info: int
  queue_pfcenable_cfg_allowed: int
  queue_pri2cos_cfg_allowed: int
  queue_cos2bw_cfg_allowed: int
  queue_id0: int
  queue_id0_service_profile: int
  queue_id1: int
  queue_id1_service_profile: int
  queue_id2: int
  queue_id2_service_profile: int
  queue_id3: int
  queue_id3_service_profile: int
  queue_id4: int
  queue_id4_service_profile: int
  queue_id5: int
  queue_id5_service_profile: int
  queue_id6: int
  queue_id6_service_profile: int
  queue_id7: int
  queue_id7_service_profile: int
  queue_id0_service_profile_type: int
  qid0_name: c.Array[ctypes.c_char, Literal[16]]
  qid1_name: c.Array[ctypes.c_char, Literal[16]]
  qid2_name: c.Array[ctypes.c_char, Literal[16]]
  qid3_name: c.Array[ctypes.c_char, Literal[16]]
  qid4_name: c.Array[ctypes.c_char, Literal[16]]
  qid5_name: c.Array[ctypes.c_char, Literal[16]]
  qid6_name: c.Array[ctypes.c_char, Literal[16]]
  qid7_name: c.Array[ctypes.c_char, Literal[16]]
  queue_id1_service_profile_type: int
  queue_id2_service_profile_type: int
  queue_id3_service_profile_type: int
  queue_id4_service_profile_type: int
  queue_id5_service_profile_type: int
  queue_id6_service_profile_type: int
  queue_id7_service_profile_type: int
  valid: int
struct_hwrm_queue_qportcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('max_configurable_queues', ctypes.c_ubyte, 8), ('max_configurable_lossless_queues', ctypes.c_ubyte, 9), ('queue_cfg_allowed', ctypes.c_ubyte, 10), ('queue_cfg_info', ctypes.c_ubyte, 11), ('queue_pfcenable_cfg_allowed', ctypes.c_ubyte, 12), ('queue_pri2cos_cfg_allowed', ctypes.c_ubyte, 13), ('queue_cos2bw_cfg_allowed', ctypes.c_ubyte, 14), ('queue_id0', ctypes.c_ubyte, 15), ('queue_id0_service_profile', ctypes.c_ubyte, 16), ('queue_id1', ctypes.c_ubyte, 17), ('queue_id1_service_profile', ctypes.c_ubyte, 18), ('queue_id2', ctypes.c_ubyte, 19), ('queue_id2_service_profile', ctypes.c_ubyte, 20), ('queue_id3', ctypes.c_ubyte, 21), ('queue_id3_service_profile', ctypes.c_ubyte, 22), ('queue_id4', ctypes.c_ubyte, 23), ('queue_id4_service_profile', ctypes.c_ubyte, 24), ('queue_id5', ctypes.c_ubyte, 25), ('queue_id5_service_profile', ctypes.c_ubyte, 26), ('queue_id6', ctypes.c_ubyte, 27), ('queue_id6_service_profile', ctypes.c_ubyte, 28), ('queue_id7', ctypes.c_ubyte, 29), ('queue_id7_service_profile', ctypes.c_ubyte, 30), ('queue_id0_service_profile_type', ctypes.c_ubyte, 31), ('qid0_name', c.Array[ctypes.c_char, Literal[16]], 32), ('qid1_name', c.Array[ctypes.c_char, Literal[16]], 48), ('qid2_name', c.Array[ctypes.c_char, Literal[16]], 64), ('qid3_name', c.Array[ctypes.c_char, Literal[16]], 80), ('qid4_name', c.Array[ctypes.c_char, Literal[16]], 96), ('qid5_name', c.Array[ctypes.c_char, Literal[16]], 112), ('qid6_name', c.Array[ctypes.c_char, Literal[16]], 128), ('qid7_name', c.Array[ctypes.c_char, Literal[16]], 144), ('queue_id1_service_profile_type', ctypes.c_ubyte, 160), ('queue_id2_service_profile_type', ctypes.c_ubyte, 161), ('queue_id3_service_profile_type', ctypes.c_ubyte, 162), ('queue_id4_service_profile_type', ctypes.c_ubyte, 163), ('queue_id5_service_profile_type', ctypes.c_ubyte, 164), ('queue_id6_service_profile_type', ctypes.c_ubyte, 165), ('queue_id7_service_profile_type', ctypes.c_ubyte, 166), ('valid', ctypes.c_ubyte, 167)])
@c.record
class struct_hwrm_queue_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  queue_id: int
struct_hwrm_queue_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('queue_id', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_queue_qcfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  queue_len: int
  service_profile: int
  queue_cfg_info: int
  unused_0: int
  valid: int
struct_hwrm_queue_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('queue_len', ctypes.c_uint32, 8), ('service_profile', ctypes.c_ubyte, 12), ('queue_cfg_info', ctypes.c_ubyte, 13), ('unused_0', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_cfg_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  queue_id: int
  dflt_len: int
  service_profile: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_queue_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('queue_id', ctypes.c_uint32, 24), ('dflt_len', ctypes.c_uint32, 28), ('service_profile', ctypes.c_ubyte, 32), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 33)])
@c.record
class struct_hwrm_queue_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_queue_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_pfcenable_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_queue_pfcenable_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_queue_pfcenable_qcfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_queue_pfcenable_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_pfcenable_cfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
struct_hwrm_queue_pfcenable_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('port_id', ctypes.c_uint16, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 22)])
@c.record
class struct_hwrm_queue_pfcenable_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_queue_pfcenable_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_pri2cos_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
struct_hwrm_queue_pri2cos_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('port_id', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21)])
@c.record
class struct_hwrm_queue_pri2cos_qcfg_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  pri0_cos_queue_id: int
  pri1_cos_queue_id: int
  pri2_cos_queue_id: int
  pri3_cos_queue_id: int
  pri4_cos_queue_id: int
  pri5_cos_queue_id: int
  pri6_cos_queue_id: int
  pri7_cos_queue_id: int
  queue_cfg_info: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_queue_pri2cos_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('pri0_cos_queue_id', ctypes.c_ubyte, 8), ('pri1_cos_queue_id', ctypes.c_ubyte, 9), ('pri2_cos_queue_id', ctypes.c_ubyte, 10), ('pri3_cos_queue_id', ctypes.c_ubyte, 11), ('pri4_cos_queue_id', ctypes.c_ubyte, 12), ('pri5_cos_queue_id', ctypes.c_ubyte, 13), ('pri6_cos_queue_id', ctypes.c_ubyte, 14), ('pri7_cos_queue_id', ctypes.c_ubyte, 15), ('queue_cfg_info', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 17), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_queue_pri2cos_cfg_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  pri0_cos_queue_id: int
  pri1_cos_queue_id: int
  pri2_cos_queue_id: int
  pri3_cos_queue_id: int
  pri4_cos_queue_id: int
  pri5_cos_queue_id: int
  pri6_cos_queue_id: int
  pri7_cos_queue_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_queue_pri2cos_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_ubyte, 24), ('pri0_cos_queue_id', ctypes.c_ubyte, 25), ('pri1_cos_queue_id', ctypes.c_ubyte, 26), ('pri2_cos_queue_id', ctypes.c_ubyte, 27), ('pri3_cos_queue_id', ctypes.c_ubyte, 28), ('pri4_cos_queue_id', ctypes.c_ubyte, 29), ('pri5_cos_queue_id', ctypes.c_ubyte, 30), ('pri6_cos_queue_id', ctypes.c_ubyte, 31), ('pri7_cos_queue_id', ctypes.c_ubyte, 32), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 33)])
@c.record
class struct_hwrm_queue_pri2cos_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_queue_pri2cos_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_cos2bw_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_queue_cos2bw_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_queue_cos2bw_qcfg_output(c.Struct):
  SIZE = 48
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  queue_id0: int
  unused_0: int
  unused_1: int
  queue_id0_min_bw: int
  queue_id0_max_bw: int
  queue_id0_tsa_assign: int
  queue_id0_pri_lvl: int
  queue_id0_bw_weight: int
  __packed: struct_hwrm_queue_cos2bw_qcfg_output___packed
  unused_2: c.Array[ctypes.c_ubyte, Literal[4]]
  valid: int
@c.record
class struct_hwrm_queue_cos2bw_qcfg_output___packed(c.Struct):
  SIZE = 16
  queue_id: int
  queue_id_min_bw: int
  queue_id_max_bw: int
  queue_id_tsa_assign: int
  queue_id_pri_lvl: int
  queue_id_bw_weight: int
struct_hwrm_queue_cos2bw_qcfg_output___packed.register_fields([('queue_id', ctypes.c_ubyte, 0), ('queue_id_min_bw', ctypes.c_uint32, 4), ('queue_id_max_bw', ctypes.c_uint32, 8), ('queue_id_tsa_assign', ctypes.c_ubyte, 12), ('queue_id_pri_lvl', ctypes.c_ubyte, 13), ('queue_id_bw_weight', ctypes.c_ubyte, 14)])
struct_hwrm_queue_cos2bw_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('queue_id0', ctypes.c_ubyte, 8), ('unused_0', ctypes.c_ubyte, 9), ('unused_1', ctypes.c_uint16, 10), ('queue_id0_min_bw', ctypes.c_uint32, 12), ('queue_id0_max_bw', ctypes.c_uint32, 16), ('queue_id0_tsa_assign', ctypes.c_ubyte, 20), ('queue_id0_pri_lvl', ctypes.c_ubyte, 21), ('queue_id0_bw_weight', ctypes.c_ubyte, 22), ('__packed', struct_hwrm_queue_cos2bw_qcfg_output___packed, 24), ('unused_2', c.Array[ctypes.c_ubyte, Literal[4]], 40), ('valid', ctypes.c_ubyte, 44)])
@c.record
class struct_hwrm_queue_cos2bw_cfg_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  queue_id0: int
  unused_0: int
  queue_id0_min_bw: int
  queue_id0_max_bw: int
  queue_id0_tsa_assign: int
  queue_id0_pri_lvl: int
  queue_id0_bw_weight: int
  __packed: struct_hwrm_queue_cos2bw_cfg_input___packed
  unused_1: c.Array[ctypes.c_ubyte, Literal[5]]
@c.record
class struct_hwrm_queue_cos2bw_cfg_input___packed(c.Struct):
  SIZE = 16
  queue_id: int
  queue_id_min_bw: int
  queue_id_max_bw: int
  queue_id_tsa_assign: int
  queue_id_pri_lvl: int
  queue_id_bw_weight: int
struct_hwrm_queue_cos2bw_cfg_input___packed.register_fields([('queue_id', ctypes.c_ubyte, 0), ('queue_id_min_bw', ctypes.c_uint32, 4), ('queue_id_max_bw', ctypes.c_uint32, 8), ('queue_id_tsa_assign', ctypes.c_ubyte, 12), ('queue_id_pri_lvl', ctypes.c_ubyte, 13), ('queue_id_bw_weight', ctypes.c_ubyte, 14)])
struct_hwrm_queue_cos2bw_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('queue_id0', ctypes.c_ubyte, 26), ('unused_0', ctypes.c_ubyte, 27), ('queue_id0_min_bw', ctypes.c_uint32, 28), ('queue_id0_max_bw', ctypes.c_uint32, 32), ('queue_id0_tsa_assign', ctypes.c_ubyte, 36), ('queue_id0_pri_lvl', ctypes.c_ubyte, 37), ('queue_id0_bw_weight', ctypes.c_ubyte, 38), ('__packed', struct_hwrm_queue_cos2bw_cfg_input___packed, 40), ('unused_1', c.Array[ctypes.c_ubyte, Literal[5]], 56)])
@c.record
class struct_hwrm_queue_cos2bw_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_queue_cos2bw_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_dscp_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_queue_dscp_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 17)])
@c.record
class struct_hwrm_queue_dscp_qcaps_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  num_dscp_bits: int
  unused_0: int
  max_entries: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_queue_dscp_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('num_dscp_bits', ctypes.c_ubyte, 8), ('unused_0', ctypes.c_ubyte, 9), ('max_entries', ctypes.c_uint16, 10), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_dscp2pri_qcfg_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  dest_data_addr: int
  port_id: int
  unused_0: int
  dest_data_buffer_size: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_queue_dscp2pri_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('dest_data_addr', ctypes.c_uint64, 16), ('port_id', ctypes.c_ubyte, 24), ('unused_0', ctypes.c_ubyte, 25), ('dest_data_buffer_size', ctypes.c_uint16, 26), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 28)])
@c.record
class struct_hwrm_queue_dscp2pri_qcfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  entry_cnt: int
  default_pri: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  valid: int
struct_hwrm_queue_dscp2pri_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('entry_cnt', ctypes.c_uint16, 8), ('default_pri', ctypes.c_ubyte, 10), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 11), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_dscp2pri_cfg_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  src_data_addr: int
  flags: int
  enables: int
  port_id: int
  default_pri: int
  entry_cnt: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_queue_dscp2pri_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('src_data_addr', ctypes.c_uint64, 16), ('flags', ctypes.c_uint32, 24), ('enables', ctypes.c_uint32, 28), ('port_id', ctypes.c_ubyte, 32), ('default_pri', ctypes.c_ubyte, 33), ('entry_cnt', ctypes.c_uint16, 34), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 36)])
@c.record
class struct_hwrm_queue_dscp2pri_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_queue_dscp2pri_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_pfcwd_timeout_qcaps_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_queue_pfcwd_timeout_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_queue_pfcwd_timeout_qcaps_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  max_pfcwd_timeout: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_queue_pfcwd_timeout_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('max_pfcwd_timeout', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_pfcwd_timeout_cfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  pfcwd_timeout_value: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_queue_pfcwd_timeout_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('pfcwd_timeout_value', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_queue_pfcwd_timeout_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_queue_pfcwd_timeout_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_queue_pfcwd_timeout_qcfg_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_queue_pfcwd_timeout_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_queue_pfcwd_timeout_qcfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  pfcwd_timeout_value: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_queue_pfcwd_timeout_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('pfcwd_timeout_value', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_alloc_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  virtio_net_fid: int
  vnic_id: int
struct_hwrm_vnic_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('virtio_net_fid', ctypes.c_uint16, 20), ('vnic_id', ctypes.c_uint16, 22)])
@c.record
class struct_hwrm_vnic_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  vnic_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_vnic_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('vnic_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_update_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vnic_id: int
  enables: int
  vnic_state: int
  metadata_format_type: int
  mru: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_update_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vnic_id', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('vnic_state', ctypes.c_ubyte, 24), ('metadata_format_type', ctypes.c_ubyte, 25), ('mru', ctypes.c_uint16, 26), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 28)])
@c.record
class struct_hwrm_vnic_update_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_update_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vnic_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vnic_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_vnic_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  vnic_id: int
  dflt_ring_grp: int
  rss_rule: int
  cos_rule: int
  lb_rule: int
  mru: int
  default_rx_ring_id: int
  default_cmpl_ring_id: int
  queue_id: int
  rx_csum_v2_mode: int
  l2_cqe_mode: int
  raw_qp_id: int
struct_hwrm_vnic_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('vnic_id', ctypes.c_uint16, 24), ('dflt_ring_grp', ctypes.c_uint16, 26), ('rss_rule', ctypes.c_uint16, 28), ('cos_rule', ctypes.c_uint16, 30), ('lb_rule', ctypes.c_uint16, 32), ('mru', ctypes.c_uint16, 34), ('default_rx_ring_id', ctypes.c_uint16, 36), ('default_cmpl_ring_id', ctypes.c_uint16, 38), ('queue_id', ctypes.c_uint16, 40), ('rx_csum_v2_mode', ctypes.c_ubyte, 42), ('l2_cqe_mode', ctypes.c_ubyte, 43), ('raw_qp_id', ctypes.c_uint32, 44)])
@c.record
class struct_hwrm_vnic_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_vnic_qcaps_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  mru: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  flags: int
  max_aggs_supported: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_vnic_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('mru', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 10), ('flags', ctypes.c_uint32, 12), ('max_aggs_supported', ctypes.c_uint16, 16), ('unused_1', c.Array[ctypes.c_ubyte, Literal[5]], 18), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_vnic_tpa_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  vnic_id: int
  max_agg_segs: int
  max_aggs: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  max_agg_timer: int
  min_agg_len: int
  tnl_tpa_en_bitmap: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_tpa_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('vnic_id', ctypes.c_uint16, 24), ('max_agg_segs', ctypes.c_uint16, 26), ('max_aggs', ctypes.c_uint16, 28), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 30), ('max_agg_timer', ctypes.c_uint32, 32), ('min_agg_len', ctypes.c_uint32, 36), ('tnl_tpa_en_bitmap', ctypes.c_uint32, 40), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 44)])
@c.record
class struct_hwrm_vnic_tpa_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_tpa_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_tpa_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vnic_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_vnic_tpa_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vnic_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_vnic_tpa_qcfg_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  max_agg_segs: int
  max_aggs: int
  max_agg_timer: int
  min_agg_len: int
  tnl_tpa_en_bitmap: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_vnic_tpa_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('max_agg_segs', ctypes.c_uint16, 12), ('max_aggs', ctypes.c_uint16, 14), ('max_agg_timer', ctypes.c_uint32, 16), ('min_agg_len', ctypes.c_uint32, 20), ('tnl_tpa_en_bitmap', ctypes.c_uint32, 24), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 28), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_vnic_rss_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  hash_type: int
  vnic_id: int
  ring_table_pair_index: int
  hash_mode_flags: int
  ring_grp_tbl_addr: int
  hash_key_tbl_addr: int
  rss_ctx_idx: int
  flags: int
  ring_select_mode: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_rss_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('hash_type', ctypes.c_uint32, 16), ('vnic_id', ctypes.c_uint16, 20), ('ring_table_pair_index', ctypes.c_ubyte, 22), ('hash_mode_flags', ctypes.c_ubyte, 23), ('ring_grp_tbl_addr', ctypes.c_uint64, 24), ('hash_key_tbl_addr', ctypes.c_uint64, 32), ('rss_ctx_idx', ctypes.c_uint16, 40), ('flags', ctypes.c_ubyte, 42), ('ring_select_mode', ctypes.c_ubyte, 43), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 44)])
@c.record
class struct_hwrm_vnic_rss_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_rss_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_rss_cfg_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_vnic_rss_cfg_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_vnic_rss_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  rss_ctx_idx: int
  vnic_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_rss_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('rss_ctx_idx', ctypes.c_uint16, 16), ('vnic_id', ctypes.c_uint16, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_vnic_rss_qcfg_output(c.Struct):
  SIZE = 64
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  hash_type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  hash_key: c.Array[ctypes.c_uint32, Literal[10]]
  hash_mode_flags: int
  ring_select_mode: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_vnic_rss_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('hash_type', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 12), ('hash_key', c.Array[ctypes.c_uint32, Literal[10]], 16), ('hash_mode_flags', ctypes.c_ubyte, 56), ('ring_select_mode', ctypes.c_ubyte, 57), ('unused_1', c.Array[ctypes.c_ubyte, Literal[5]], 58), ('valid', ctypes.c_ubyte, 63)])
@c.record
class struct_hwrm_vnic_plcmodes_cfg_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  vnic_id: int
  jumbo_thresh: int
  hds_offset: int
  hds_threshold: int
  max_bds: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_vnic_plcmodes_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('vnic_id', ctypes.c_uint32, 24), ('jumbo_thresh', ctypes.c_uint16, 28), ('hds_offset', ctypes.c_uint16, 30), ('hds_threshold', ctypes.c_uint16, 32), ('max_bds', ctypes.c_uint16, 34), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 36)])
@c.record
class struct_hwrm_vnic_plcmodes_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_plcmodes_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_plcmodes_cfg_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_vnic_plcmodes_cfg_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_vnic_rss_cos_lb_ctx_alloc_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_vnic_rss_cos_lb_ctx_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_vnic_rss_cos_lb_ctx_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  rss_cos_lb_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_vnic_rss_cos_lb_ctx_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('rss_cos_lb_ctx_id', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vnic_rss_cos_lb_ctx_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  rss_cos_lb_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_vnic_rss_cos_lb_ctx_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('rss_cos_lb_ctx_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_vnic_rss_cos_lb_ctx_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_vnic_rss_cos_lb_ctx_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_alloc_input(c.Struct):
  SIZE = 96
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  ring_type: int
  cmpl_coal_cnt: int
  flags: int
  page_tbl_addr: int
  fbo: int
  page_size: int
  page_tbl_depth: int
  schq_id: int
  length: int
  logical_id: int
  cmpl_ring_id: int
  queue_id: int
  rx_buf_size: int
  rx_ring_id: int
  nq_ring_id: int
  ring_arb_cfg: int
  steering_tag: int
  reserved3: int
  stat_ctx_id: int
  reserved4: int
  max_bw: int
  int_mode: int
  mpc_chnls_type: int
  rx_rate_profile_sel: int
  unused_4: int
  cq_handle: int
  dpi: int
  unused_5: c.Array[ctypes.c_uint16, Literal[3]]
struct_hwrm_ring_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('ring_type', ctypes.c_ubyte, 20), ('cmpl_coal_cnt', ctypes.c_ubyte, 21), ('flags', ctypes.c_uint16, 22), ('page_tbl_addr', ctypes.c_uint64, 24), ('fbo', ctypes.c_uint32, 32), ('page_size', ctypes.c_ubyte, 36), ('page_tbl_depth', ctypes.c_ubyte, 37), ('schq_id', ctypes.c_uint16, 38), ('length', ctypes.c_uint32, 40), ('logical_id', ctypes.c_uint16, 44), ('cmpl_ring_id', ctypes.c_uint16, 46), ('queue_id', ctypes.c_uint16, 48), ('rx_buf_size', ctypes.c_uint16, 50), ('rx_ring_id', ctypes.c_uint16, 52), ('nq_ring_id', ctypes.c_uint16, 54), ('ring_arb_cfg', ctypes.c_uint16, 56), ('steering_tag', ctypes.c_uint16, 58), ('reserved3', ctypes.c_uint32, 60), ('stat_ctx_id', ctypes.c_uint32, 64), ('reserved4', ctypes.c_uint32, 68), ('max_bw', ctypes.c_uint32, 72), ('int_mode', ctypes.c_ubyte, 76), ('mpc_chnls_type', ctypes.c_ubyte, 77), ('rx_rate_profile_sel', ctypes.c_ubyte, 78), ('unused_4', ctypes.c_ubyte, 79), ('cq_handle', ctypes.c_uint64, 80), ('dpi', ctypes.c_uint16, 88), ('unused_5', c.Array[ctypes.c_uint16, Literal[3]], 90)])
@c.record
class struct_hwrm_ring_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  ring_id: int
  logical_ring_id: int
  push_buffer_index: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  valid: int
struct_hwrm_ring_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('ring_id', ctypes.c_uint16, 8), ('logical_ring_id', ctypes.c_uint16, 10), ('push_buffer_index', ctypes.c_ubyte, 12), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 13), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_free_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ring_type: int
  flags: int
  ring_id: int
  prod_idx: int
  opaque: int
  unused_1: int
struct_hwrm_ring_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ring_type', ctypes.c_ubyte, 16), ('flags', ctypes.c_ubyte, 17), ('ring_id', ctypes.c_uint16, 18), ('prod_idx', ctypes.c_uint32, 20), ('opaque', ctypes.c_uint32, 24), ('unused_1', ctypes.c_uint32, 28)])
@c.record
class struct_hwrm_ring_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_ring_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_reset_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ring_type: int
  unused_0: int
  ring_id: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_ring_reset_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ring_type', ctypes.c_ubyte, 16), ('unused_0', ctypes.c_ubyte, 17), ('ring_id', ctypes.c_uint16, 18), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_ring_reset_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  push_buffer_index: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  consumer_idx: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_ring_reset_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('push_buffer_index', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 9), ('consumer_idx', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_aggint_qcaps_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_ring_aggint_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_ring_aggint_qcaps_output(c.Struct):
  SIZE = 48
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  cmpl_params: int
  nq_params: int
  num_cmpl_dma_aggr_min: int
  num_cmpl_dma_aggr_max: int
  num_cmpl_dma_aggr_during_int_min: int
  num_cmpl_dma_aggr_during_int_max: int
  cmpl_aggr_dma_tmr_min: int
  cmpl_aggr_dma_tmr_max: int
  cmpl_aggr_dma_tmr_during_int_min: int
  cmpl_aggr_dma_tmr_during_int_max: int
  int_lat_tmr_min_min: int
  int_lat_tmr_min_max: int
  int_lat_tmr_max_min: int
  int_lat_tmr_max_max: int
  num_cmpl_aggr_int_min: int
  num_cmpl_aggr_int_max: int
  timer_units: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[1]]
  valid: int
struct_hwrm_ring_aggint_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('cmpl_params', ctypes.c_uint32, 8), ('nq_params', ctypes.c_uint32, 12), ('num_cmpl_dma_aggr_min', ctypes.c_uint16, 16), ('num_cmpl_dma_aggr_max', ctypes.c_uint16, 18), ('num_cmpl_dma_aggr_during_int_min', ctypes.c_uint16, 20), ('num_cmpl_dma_aggr_during_int_max', ctypes.c_uint16, 22), ('cmpl_aggr_dma_tmr_min', ctypes.c_uint16, 24), ('cmpl_aggr_dma_tmr_max', ctypes.c_uint16, 26), ('cmpl_aggr_dma_tmr_during_int_min', ctypes.c_uint16, 28), ('cmpl_aggr_dma_tmr_during_int_max', ctypes.c_uint16, 30), ('int_lat_tmr_min_min', ctypes.c_uint16, 32), ('int_lat_tmr_min_max', ctypes.c_uint16, 34), ('int_lat_tmr_max_min', ctypes.c_uint16, 36), ('int_lat_tmr_max_max', ctypes.c_uint16, 38), ('num_cmpl_aggr_int_min', ctypes.c_uint16, 40), ('num_cmpl_aggr_int_max', ctypes.c_uint16, 42), ('timer_units', ctypes.c_uint16, 44), ('unused_0', c.Array[ctypes.c_ubyte, Literal[1]], 46), ('valid', ctypes.c_ubyte, 47)])
@c.record
class struct_hwrm_ring_cmpl_ring_qaggint_params_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ring_id: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_ring_cmpl_ring_qaggint_params_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ring_id', ctypes.c_uint16, 16), ('flags', ctypes.c_uint16, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_ring_cmpl_ring_qaggint_params_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  num_cmpl_dma_aggr: int
  num_cmpl_dma_aggr_during_int: int
  cmpl_aggr_dma_tmr: int
  cmpl_aggr_dma_tmr_during_int: int
  int_lat_tmr_min: int
  int_lat_tmr_max: int
  num_cmpl_aggr_int: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_ring_cmpl_ring_qaggint_params_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint16, 8), ('num_cmpl_dma_aggr', ctypes.c_uint16, 10), ('num_cmpl_dma_aggr_during_int', ctypes.c_uint16, 12), ('cmpl_aggr_dma_tmr', ctypes.c_uint16, 14), ('cmpl_aggr_dma_tmr_during_int', ctypes.c_uint16, 16), ('int_lat_tmr_min', ctypes.c_uint16, 18), ('int_lat_tmr_max', ctypes.c_uint16, 20), ('num_cmpl_aggr_int', ctypes.c_uint16, 22), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 24), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_ring_cmpl_ring_cfg_aggint_params_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ring_id: int
  flags: int
  num_cmpl_dma_aggr: int
  num_cmpl_dma_aggr_during_int: int
  cmpl_aggr_dma_tmr: int
  cmpl_aggr_dma_tmr_during_int: int
  int_lat_tmr_min: int
  int_lat_tmr_max: int
  num_cmpl_aggr_int: int
  enables: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_ring_cmpl_ring_cfg_aggint_params_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ring_id', ctypes.c_uint16, 16), ('flags', ctypes.c_uint16, 18), ('num_cmpl_dma_aggr', ctypes.c_uint16, 20), ('num_cmpl_dma_aggr_during_int', ctypes.c_uint16, 22), ('cmpl_aggr_dma_tmr', ctypes.c_uint16, 24), ('cmpl_aggr_dma_tmr_during_int', ctypes.c_uint16, 26), ('int_lat_tmr_min', ctypes.c_uint16, 28), ('int_lat_tmr_max', ctypes.c_uint16, 30), ('num_cmpl_aggr_int', ctypes.c_uint16, 32), ('enables', ctypes.c_uint16, 34), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 36)])
@c.record
class struct_hwrm_ring_cmpl_ring_cfg_aggint_params_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_ring_cmpl_ring_cfg_aggint_params_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_grp_alloc_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  cr: int
  rr: int
  ar: int
  sc: int
struct_hwrm_ring_grp_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('cr', ctypes.c_uint16, 16), ('rr', ctypes.c_uint16, 18), ('ar', ctypes.c_uint16, 20), ('sc', ctypes.c_uint16, 22)])
@c.record
class struct_hwrm_ring_grp_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  ring_group_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_ring_grp_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('ring_group_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_ring_grp_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ring_group_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_ring_grp_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ring_group_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_ring_grp_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_ring_grp_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_l2_filter_alloc_input(c.Struct):
  SIZE = 96
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  l2_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  num_vlans: int
  t_num_vlans: int
  l2_addr_mask: c.Array[ctypes.c_ubyte, Literal[6]]
  l2_ovlan: int
  l2_ovlan_mask: int
  l2_ivlan: int
  l2_ivlan_mask: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[2]]
  t_l2_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  unused_2: c.Array[ctypes.c_ubyte, Literal[2]]
  t_l2_addr_mask: c.Array[ctypes.c_ubyte, Literal[6]]
  t_l2_ovlan: int
  t_l2_ovlan_mask: int
  t_l2_ivlan: int
  t_l2_ivlan_mask: int
  src_type: int
  unused_3: int
  src_id: int
  tunnel_type: int
  unused_4: int
  dst_id: int
  mirror_vnic_id: int
  pri_hint: int
  unused_5: int
  unused_6: int
  l2_filter_id_hint: int
struct_hwrm_cfa_l2_filter_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('l2_addr', c.Array[ctypes.c_ubyte, Literal[6]], 24), ('num_vlans', ctypes.c_ubyte, 30), ('t_num_vlans', ctypes.c_ubyte, 31), ('l2_addr_mask', c.Array[ctypes.c_ubyte, Literal[6]], 32), ('l2_ovlan', ctypes.c_uint16, 38), ('l2_ovlan_mask', ctypes.c_uint16, 40), ('l2_ivlan', ctypes.c_uint16, 42), ('l2_ivlan_mask', ctypes.c_uint16, 44), ('unused_1', c.Array[ctypes.c_ubyte, Literal[2]], 46), ('t_l2_addr', c.Array[ctypes.c_ubyte, Literal[6]], 48), ('unused_2', c.Array[ctypes.c_ubyte, Literal[2]], 54), ('t_l2_addr_mask', c.Array[ctypes.c_ubyte, Literal[6]], 56), ('t_l2_ovlan', ctypes.c_uint16, 62), ('t_l2_ovlan_mask', ctypes.c_uint16, 64), ('t_l2_ivlan', ctypes.c_uint16, 66), ('t_l2_ivlan_mask', ctypes.c_uint16, 68), ('src_type', ctypes.c_ubyte, 70), ('unused_3', ctypes.c_ubyte, 71), ('src_id', ctypes.c_uint32, 72), ('tunnel_type', ctypes.c_ubyte, 76), ('unused_4', ctypes.c_ubyte, 77), ('dst_id', ctypes.c_uint16, 78), ('mirror_vnic_id', ctypes.c_uint16, 80), ('pri_hint', ctypes.c_ubyte, 82), ('unused_5', ctypes.c_ubyte, 83), ('unused_6', ctypes.c_uint32, 84), ('l2_filter_id_hint', ctypes.c_uint64, 88)])
@c.record
class struct_hwrm_cfa_l2_filter_alloc_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  l2_filter_id: int
  flow_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_l2_filter_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('l2_filter_id', ctypes.c_uint64, 8), ('flow_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 20), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_cfa_l2_filter_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  l2_filter_id: int
struct_hwrm_cfa_l2_filter_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('l2_filter_id', ctypes.c_uint64, 16)])
@c.record
class struct_hwrm_cfa_l2_filter_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_l2_filter_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_l2_filter_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  l2_filter_id: int
  dst_id: int
  new_mirror_vnic_id: int
  prof_func: int
  l2_context_id: int
struct_hwrm_cfa_l2_filter_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('l2_filter_id', ctypes.c_uint64, 24), ('dst_id', ctypes.c_uint32, 32), ('new_mirror_vnic_id', ctypes.c_uint32, 36), ('prof_func', ctypes.c_uint32, 40), ('l2_context_id', ctypes.c_uint32, 44)])
@c.record
class struct_hwrm_cfa_l2_filter_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_l2_filter_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_l2_set_rx_mask_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vnic_id: int
  mask: int
  mc_tbl_addr: int
  num_mc_entries: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  vlan_tag_tbl_addr: int
  num_vlan_tags: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_cfa_l2_set_rx_mask_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vnic_id', ctypes.c_uint32, 16), ('mask', ctypes.c_uint32, 20), ('mc_tbl_addr', ctypes.c_uint64, 24), ('num_mc_entries', ctypes.c_uint32, 32), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 36), ('vlan_tag_tbl_addr', ctypes.c_uint64, 40), ('num_vlan_tags', ctypes.c_uint32, 48), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 52)])
@c.record
class struct_hwrm_cfa_l2_set_rx_mask_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_l2_set_rx_mask_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_l2_set_rx_mask_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_cfa_l2_set_rx_mask_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_cfa_tunnel_filter_alloc_input(c.Struct):
  SIZE = 88
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  l2_filter_id: int
  l2_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  l2_ivlan: int
  l3_addr: c.Array[ctypes.c_uint32, Literal[4]]
  t_l3_addr: c.Array[ctypes.c_uint32, Literal[4]]
  l3_addr_type: int
  t_l3_addr_type: int
  tunnel_type: int
  tunnel_flags: int
  vni: int
  dst_vnic_id: int
  mirror_vnic_id: int
struct_hwrm_cfa_tunnel_filter_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('l2_filter_id', ctypes.c_uint64, 24), ('l2_addr', c.Array[ctypes.c_ubyte, Literal[6]], 32), ('l2_ivlan', ctypes.c_uint16, 38), ('l3_addr', c.Array[ctypes.c_uint32, Literal[4]], 40), ('t_l3_addr', c.Array[ctypes.c_uint32, Literal[4]], 56), ('l3_addr_type', ctypes.c_ubyte, 72), ('t_l3_addr_type', ctypes.c_ubyte, 73), ('tunnel_type', ctypes.c_ubyte, 74), ('tunnel_flags', ctypes.c_ubyte, 75), ('vni', ctypes.c_uint32, 76), ('dst_vnic_id', ctypes.c_uint32, 80), ('mirror_vnic_id', ctypes.c_uint32, 84)])
@c.record
class struct_hwrm_cfa_tunnel_filter_alloc_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tunnel_filter_id: int
  flow_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_tunnel_filter_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tunnel_filter_id', ctypes.c_uint64, 8), ('flow_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 20), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_cfa_tunnel_filter_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  tunnel_filter_id: int
struct_hwrm_cfa_tunnel_filter_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('tunnel_filter_id', ctypes.c_uint64, 16)])
@c.record
class struct_hwrm_cfa_tunnel_filter_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_tunnel_filter_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_vxlan_ipv4_hdr(c.Struct):
  SIZE = 16
  ver_hlen: int
  tos: int
  ip_id: int
  flags_frag_offset: int
  ttl: int
  protocol: int
  src_ip_addr: int
  dest_ip_addr: int
struct_hwrm_vxlan_ipv4_hdr.register_fields([('ver_hlen', ctypes.c_ubyte, 0), ('tos', ctypes.c_ubyte, 1), ('ip_id', ctypes.c_uint16, 2), ('flags_frag_offset', ctypes.c_uint16, 4), ('ttl', ctypes.c_ubyte, 6), ('protocol', ctypes.c_ubyte, 7), ('src_ip_addr', ctypes.c_uint32, 8), ('dest_ip_addr', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_vxlan_ipv6_hdr(c.Struct):
  SIZE = 40
  ver_tc_flow_label: int
  payload_len: int
  next_hdr: int
  ttl: int
  src_ip_addr: c.Array[ctypes.c_uint32, Literal[4]]
  dest_ip_addr: c.Array[ctypes.c_uint32, Literal[4]]
struct_hwrm_vxlan_ipv6_hdr.register_fields([('ver_tc_flow_label', ctypes.c_uint32, 0), ('payload_len', ctypes.c_uint16, 4), ('next_hdr', ctypes.c_ubyte, 6), ('ttl', ctypes.c_ubyte, 7), ('src_ip_addr', c.Array[ctypes.c_uint32, Literal[4]], 8), ('dest_ip_addr', c.Array[ctypes.c_uint32, Literal[4]], 24)])
@c.record
class struct_hwrm_cfa_encap_data_vxlan(c.Struct):
  SIZE = 80
  src_mac_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  unused_0: int
  dst_mac_addr: c.Array[ctypes.c_ubyte, Literal[6]]
  num_vlan_tags: int
  unused_1: int
  ovlan_tpid: int
  ovlan_tci: int
  ivlan_tpid: int
  ivlan_tci: int
  l3: c.Array[ctypes.c_uint32, Literal[10]]
  src_port: int
  dst_port: int
  vni: int
  hdr_rsvd0: c.Array[ctypes.c_ubyte, Literal[3]]
  hdr_rsvd1: int
  hdr_flags: int
  unused: c.Array[ctypes.c_ubyte, Literal[3]]
struct_hwrm_cfa_encap_data_vxlan.register_fields([('src_mac_addr', c.Array[ctypes.c_ubyte, Literal[6]], 0), ('unused_0', ctypes.c_uint16, 6), ('dst_mac_addr', c.Array[ctypes.c_ubyte, Literal[6]], 8), ('num_vlan_tags', ctypes.c_ubyte, 14), ('unused_1', ctypes.c_ubyte, 15), ('ovlan_tpid', ctypes.c_uint16, 16), ('ovlan_tci', ctypes.c_uint16, 18), ('ivlan_tpid', ctypes.c_uint16, 20), ('ivlan_tci', ctypes.c_uint16, 22), ('l3', c.Array[ctypes.c_uint32, Literal[10]], 24), ('src_port', ctypes.c_uint16, 64), ('dst_port', ctypes.c_uint16, 66), ('vni', ctypes.c_uint32, 68), ('hdr_rsvd0', c.Array[ctypes.c_ubyte, Literal[3]], 72), ('hdr_rsvd1', ctypes.c_ubyte, 75), ('hdr_flags', ctypes.c_ubyte, 76), ('unused', c.Array[ctypes.c_ubyte, Literal[3]], 77)])
@c.record
class struct_hwrm_cfa_encap_record_alloc_input(c.Struct):
  SIZE = 104
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  encap_type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  encap_data: c.Array[ctypes.c_uint32, Literal[20]]
struct_hwrm_cfa_encap_record_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('encap_type', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21), ('encap_data', c.Array[ctypes.c_uint32, Literal[20]], 24)])
@c.record
class struct_hwrm_cfa_encap_record_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  encap_record_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_encap_record_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('encap_record_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_encap_record_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  encap_record_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_cfa_encap_record_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('encap_record_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_cfa_encap_record_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_encap_record_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_ntuple_filter_alloc_input(c.Struct):
  SIZE = 128
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  l2_filter_id: int
  src_macaddr: c.Array[ctypes.c_ubyte, Literal[6]]
  ethertype: int
  ip_addr_type: int
  ip_protocol: int
  dst_id: int
  rfs_ring_tbl_idx: int
  tunnel_type: int
  pri_hint: int
  src_ipaddr: c.Array[ctypes.c_uint32, Literal[4]]
  src_ipaddr_mask: c.Array[ctypes.c_uint32, Literal[4]]
  dst_ipaddr: c.Array[ctypes.c_uint32, Literal[4]]
  dst_ipaddr_mask: c.Array[ctypes.c_uint32, Literal[4]]
  src_port: int
  src_port_mask: int
  dst_port: int
  dst_port_mask: int
  ntuple_filter_id_hint: int
struct_hwrm_cfa_ntuple_filter_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('l2_filter_id', ctypes.c_uint64, 24), ('src_macaddr', c.Array[ctypes.c_ubyte, Literal[6]], 32), ('ethertype', ctypes.c_uint16, 38), ('ip_addr_type', ctypes.c_ubyte, 40), ('ip_protocol', ctypes.c_ubyte, 41), ('dst_id', ctypes.c_uint16, 42), ('rfs_ring_tbl_idx', ctypes.c_uint16, 44), ('tunnel_type', ctypes.c_ubyte, 46), ('pri_hint', ctypes.c_ubyte, 47), ('src_ipaddr', c.Array[ctypes.c_uint32, Literal[4]], 48), ('src_ipaddr_mask', c.Array[ctypes.c_uint32, Literal[4]], 64), ('dst_ipaddr', c.Array[ctypes.c_uint32, Literal[4]], 80), ('dst_ipaddr_mask', c.Array[ctypes.c_uint32, Literal[4]], 96), ('src_port', ctypes.c_uint16, 112), ('src_port_mask', ctypes.c_uint16, 114), ('dst_port', ctypes.c_uint16, 116), ('dst_port_mask', ctypes.c_uint16, 118), ('ntuple_filter_id_hint', ctypes.c_uint64, 120)])
@c.record
class struct_hwrm_cfa_ntuple_filter_alloc_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  ntuple_filter_id: int
  flow_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_ntuple_filter_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('ntuple_filter_id', ctypes.c_uint64, 8), ('flow_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 20), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_cfa_ntuple_filter_alloc_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_cfa_ntuple_filter_alloc_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_cfa_ntuple_filter_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ntuple_filter_id: int
struct_hwrm_cfa_ntuple_filter_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ntuple_filter_id', ctypes.c_uint64, 16)])
@c.record
class struct_hwrm_cfa_ntuple_filter_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_ntuple_filter_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_ntuple_filter_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  flags: int
  ntuple_filter_id: int
  new_dst_id: int
  new_mirror_vnic_id: int
  new_meter_instance_id: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_cfa_ntuple_filter_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('ntuple_filter_id', ctypes.c_uint64, 24), ('new_dst_id', ctypes.c_uint32, 32), ('new_mirror_vnic_id', ctypes.c_uint32, 36), ('new_meter_instance_id', ctypes.c_uint16, 40), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 42)])
@c.record
class struct_hwrm_cfa_ntuple_filter_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_ntuple_filter_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_decap_filter_alloc_input(c.Struct):
  SIZE = 104
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  tunnel_id: int
  tunnel_type: int
  unused_0: int
  unused_1: int
  src_macaddr: c.Array[ctypes.c_ubyte, Literal[6]]
  unused_2: c.Array[ctypes.c_ubyte, Literal[2]]
  dst_macaddr: c.Array[ctypes.c_ubyte, Literal[6]]
  ovlan_vid: int
  ivlan_vid: int
  t_ovlan_vid: int
  t_ivlan_vid: int
  ethertype: int
  ip_addr_type: int
  ip_protocol: int
  unused_3: int
  unused_4: int
  src_ipaddr: c.Array[ctypes.c_uint32, Literal[4]]
  dst_ipaddr: c.Array[ctypes.c_uint32, Literal[4]]
  src_port: int
  dst_port: int
  dst_id: int
  l2_ctxt_ref_id: int
struct_hwrm_cfa_decap_filter_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('tunnel_id', ctypes.c_uint32, 24), ('tunnel_type', ctypes.c_ubyte, 28), ('unused_0', ctypes.c_ubyte, 29), ('unused_1', ctypes.c_uint16, 30), ('src_macaddr', c.Array[ctypes.c_ubyte, Literal[6]], 32), ('unused_2', c.Array[ctypes.c_ubyte, Literal[2]], 38), ('dst_macaddr', c.Array[ctypes.c_ubyte, Literal[6]], 40), ('ovlan_vid', ctypes.c_uint16, 46), ('ivlan_vid', ctypes.c_uint16, 48), ('t_ovlan_vid', ctypes.c_uint16, 50), ('t_ivlan_vid', ctypes.c_uint16, 52), ('ethertype', ctypes.c_uint16, 54), ('ip_addr_type', ctypes.c_ubyte, 56), ('ip_protocol', ctypes.c_ubyte, 57), ('unused_3', ctypes.c_uint16, 58), ('unused_4', ctypes.c_uint32, 60), ('src_ipaddr', c.Array[ctypes.c_uint32, Literal[4]], 64), ('dst_ipaddr', c.Array[ctypes.c_uint32, Literal[4]], 80), ('src_port', ctypes.c_uint16, 96), ('dst_port', ctypes.c_uint16, 98), ('dst_id', ctypes.c_uint16, 100), ('l2_ctxt_ref_id', ctypes.c_uint16, 102)])
@c.record
class struct_hwrm_cfa_decap_filter_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  decap_filter_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_decap_filter_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('decap_filter_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_decap_filter_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  decap_filter_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_cfa_decap_filter_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('decap_filter_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_cfa_decap_filter_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_decap_filter_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_flow_alloc_input(c.Struct):
  SIZE = 128
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  src_fid: int
  tunnel_handle: int
  action_flags: int
  dst_fid: int
  l2_rewrite_vlan_tpid: int
  l2_rewrite_vlan_tci: int
  act_meter_id: int
  ref_flow_handle: int
  ethertype: int
  outer_vlan_tci: int
  dmac: c.Array[ctypes.c_uint16, Literal[3]]
  inner_vlan_tci: int
  smac: c.Array[ctypes.c_uint16, Literal[3]]
  ip_dst_mask_len: int
  ip_src_mask_len: int
  ip_dst: c.Array[ctypes.c_uint32, Literal[4]]
  ip_src: c.Array[ctypes.c_uint32, Literal[4]]
  l4_src_port: int
  l4_src_port_mask: int
  l4_dst_port: int
  l4_dst_port_mask: int
  nat_ip_address: c.Array[ctypes.c_uint32, Literal[4]]
  l2_rewrite_dmac: c.Array[ctypes.c_uint16, Literal[3]]
  nat_port: int
  l2_rewrite_smac: c.Array[ctypes.c_uint16, Literal[3]]
  ip_proto: int
  tunnel_type: int
struct_hwrm_cfa_flow_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint16, 16), ('src_fid', ctypes.c_uint16, 18), ('tunnel_handle', ctypes.c_uint32, 20), ('action_flags', ctypes.c_uint16, 24), ('dst_fid', ctypes.c_uint16, 26), ('l2_rewrite_vlan_tpid', ctypes.c_uint16, 28), ('l2_rewrite_vlan_tci', ctypes.c_uint16, 30), ('act_meter_id', ctypes.c_uint16, 32), ('ref_flow_handle', ctypes.c_uint16, 34), ('ethertype', ctypes.c_uint16, 36), ('outer_vlan_tci', ctypes.c_uint16, 38), ('dmac', c.Array[ctypes.c_uint16, Literal[3]], 40), ('inner_vlan_tci', ctypes.c_uint16, 46), ('smac', c.Array[ctypes.c_uint16, Literal[3]], 48), ('ip_dst_mask_len', ctypes.c_ubyte, 54), ('ip_src_mask_len', ctypes.c_ubyte, 55), ('ip_dst', c.Array[ctypes.c_uint32, Literal[4]], 56), ('ip_src', c.Array[ctypes.c_uint32, Literal[4]], 72), ('l4_src_port', ctypes.c_uint16, 88), ('l4_src_port_mask', ctypes.c_uint16, 90), ('l4_dst_port', ctypes.c_uint16, 92), ('l4_dst_port_mask', ctypes.c_uint16, 94), ('nat_ip_address', c.Array[ctypes.c_uint32, Literal[4]], 96), ('l2_rewrite_dmac', c.Array[ctypes.c_uint16, Literal[3]], 112), ('nat_port', ctypes.c_uint16, 118), ('l2_rewrite_smac', c.Array[ctypes.c_uint16, Literal[3]], 120), ('ip_proto', ctypes.c_ubyte, 126), ('tunnel_type', ctypes.c_ubyte, 127)])
@c.record
class struct_hwrm_cfa_flow_alloc_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flow_handle: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  flow_id: int
  ext_flow_handle: int
  flow_counter_id: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_flow_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flow_handle', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 10), ('flow_id', ctypes.c_uint32, 12), ('ext_flow_handle', ctypes.c_uint64, 16), ('flow_counter_id', ctypes.c_uint32, 24), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 28), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_cfa_flow_alloc_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_cfa_flow_alloc_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_cfa_flow_free_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flow_handle: int
  unused_0: int
  flow_counter_id: int
  ext_flow_handle: int
struct_hwrm_cfa_flow_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flow_handle', ctypes.c_uint16, 16), ('unused_0', ctypes.c_uint16, 18), ('flow_counter_id', ctypes.c_uint32, 20), ('ext_flow_handle', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_cfa_flow_free_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  packet: int
  byte: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_flow_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('packet', ctypes.c_uint64, 8), ('byte', ctypes.c_uint64, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 24), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_cfa_flow_info_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flow_handle: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  ext_flow_handle: int
struct_hwrm_cfa_flow_info_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flow_handle', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18), ('ext_flow_handle', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_cfa_flow_info_output(c.Struct):
  SIZE = 704
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  profile: int
  src_fid: int
  dst_fid: int
  l2_ctxt_id: int
  em_info: int
  tcam_info: int
  vfp_tcam_info: int
  ar_id: int
  flow_handle: int
  tunnel_handle: int
  flow_timer: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  flow_key_data: c.Array[ctypes.c_uint32, Literal[130]]
  flow_action_info: c.Array[ctypes.c_uint32, Literal[30]]
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_flow_info_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('profile', ctypes.c_ubyte, 9), ('src_fid', ctypes.c_uint16, 10), ('dst_fid', ctypes.c_uint16, 12), ('l2_ctxt_id', ctypes.c_uint16, 14), ('em_info', ctypes.c_uint64, 16), ('tcam_info', ctypes.c_uint64, 24), ('vfp_tcam_info', ctypes.c_uint64, 32), ('ar_id', ctypes.c_uint16, 40), ('flow_handle', ctypes.c_uint16, 42), ('tunnel_handle', ctypes.c_uint32, 44), ('flow_timer', ctypes.c_uint16, 48), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 50), ('flow_key_data', c.Array[ctypes.c_uint32, Literal[130]], 56), ('flow_action_info', c.Array[ctypes.c_uint32, Literal[30]], 576), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 696), ('valid', ctypes.c_ubyte, 703)])
@c.record
class struct_hwrm_cfa_flow_stats_input(c.Struct):
  SIZE = 80
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  num_flows: int
  flow_handle_0: int
  flow_handle_1: int
  flow_handle_2: int
  flow_handle_3: int
  flow_handle_4: int
  flow_handle_5: int
  flow_handle_6: int
  flow_handle_7: int
  flow_handle_8: int
  flow_handle_9: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  flow_id_0: int
  flow_id_1: int
  flow_id_2: int
  flow_id_3: int
  flow_id_4: int
  flow_id_5: int
  flow_id_6: int
  flow_id_7: int
  flow_id_8: int
  flow_id_9: int
struct_hwrm_cfa_flow_stats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('num_flows', ctypes.c_uint16, 16), ('flow_handle_0', ctypes.c_uint16, 18), ('flow_handle_1', ctypes.c_uint16, 20), ('flow_handle_2', ctypes.c_uint16, 22), ('flow_handle_3', ctypes.c_uint16, 24), ('flow_handle_4', ctypes.c_uint16, 26), ('flow_handle_5', ctypes.c_uint16, 28), ('flow_handle_6', ctypes.c_uint16, 30), ('flow_handle_7', ctypes.c_uint16, 32), ('flow_handle_8', ctypes.c_uint16, 34), ('flow_handle_9', ctypes.c_uint16, 36), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 38), ('flow_id_0', ctypes.c_uint32, 40), ('flow_id_1', ctypes.c_uint32, 44), ('flow_id_2', ctypes.c_uint32, 48), ('flow_id_3', ctypes.c_uint32, 52), ('flow_id_4', ctypes.c_uint32, 56), ('flow_id_5', ctypes.c_uint32, 60), ('flow_id_6', ctypes.c_uint32, 64), ('flow_id_7', ctypes.c_uint32, 68), ('flow_id_8', ctypes.c_uint32, 72), ('flow_id_9', ctypes.c_uint32, 76)])
@c.record
class struct_hwrm_cfa_flow_stats_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  packet_0: int
  packet_1: int
  packet_2: int
  packet_3: int
  packet_4: int
  packet_5: int
  packet_6: int
  packet_7: int
  packet_8: int
  packet_9: int
  byte_0: int
  byte_1: int
  byte_2: int
  byte_3: int
  byte_4: int
  byte_5: int
  byte_6: int
  byte_7: int
  byte_8: int
  byte_9: int
  flow_hits: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_cfa_flow_stats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('packet_0', ctypes.c_uint64, 8), ('packet_1', ctypes.c_uint64, 16), ('packet_2', ctypes.c_uint64, 24), ('packet_3', ctypes.c_uint64, 32), ('packet_4', ctypes.c_uint64, 40), ('packet_5', ctypes.c_uint64, 48), ('packet_6', ctypes.c_uint64, 56), ('packet_7', ctypes.c_uint64, 64), ('packet_8', ctypes.c_uint64, 72), ('packet_9', ctypes.c_uint64, 80), ('byte_0', ctypes.c_uint64, 88), ('byte_1', ctypes.c_uint64, 96), ('byte_2', ctypes.c_uint64, 104), ('byte_3', ctypes.c_uint64, 112), ('byte_4', ctypes.c_uint64, 120), ('byte_5', ctypes.c_uint64, 128), ('byte_6', ctypes.c_uint64, 136), ('byte_7', ctypes.c_uint64, 144), ('byte_8', ctypes.c_uint64, 152), ('byte_9', ctypes.c_uint64, 160), ('flow_hits', ctypes.c_uint16, 168), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 170), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_hwrm_cfa_vfr_alloc_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vf_id: int
  reserved: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  vfr_name: c.Array[ctypes.c_char, Literal[32]]
struct_hwrm_cfa_vfr_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vf_id', ctypes.c_uint16, 16), ('reserved', ctypes.c_uint16, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20), ('vfr_name', c.Array[ctypes.c_char, Literal[32]], 24)])
@c.record
class struct_hwrm_cfa_vfr_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  rx_cfa_code: int
  tx_cfa_action: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_vfr_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('rx_cfa_code', ctypes.c_uint16, 8), ('tx_cfa_action', ctypes.c_uint16, 10), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_vfr_free_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  vfr_name: c.Array[ctypes.c_char, Literal[32]]
  vf_id: int
  reserved: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_cfa_vfr_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('vfr_name', c.Array[ctypes.c_char, Literal[32]], 16), ('vf_id', ctypes.c_uint16, 48), ('reserved', ctypes.c_uint16, 50), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 52)])
@c.record
class struct_hwrm_cfa_vfr_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_vfr_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_eem_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: int
struct_hwrm_cfa_eem_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_cfa_eem_qcaps_output(c.Struct):
  SIZE = 40
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: int
  supported: int
  max_entries_supported: int
  key_entry_size: int
  record_entry_size: int
  efc_entry_size: int
  fid_entry_size: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_eem_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('unused_0', ctypes.c_uint32, 12), ('supported', ctypes.c_uint32, 16), ('max_entries_supported', ctypes.c_uint32, 20), ('key_entry_size', ctypes.c_uint16, 24), ('record_entry_size', ctypes.c_uint16, 26), ('efc_entry_size', ctypes.c_uint16, 28), ('fid_entry_size', ctypes.c_uint16, 30), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 32), ('valid', ctypes.c_ubyte, 39)])
@c.record
class struct_hwrm_cfa_eem_cfg_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  group_id: int
  unused_0: int
  num_entries: int
  unused_1: int
  key0_ctx_id: int
  key1_ctx_id: int
  record_ctx_id: int
  efc_ctx_id: int
  fid_ctx_id: int
  unused_2: int
  unused_3: int
struct_hwrm_cfa_eem_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('group_id', ctypes.c_uint16, 20), ('unused_0', ctypes.c_uint16, 22), ('num_entries', ctypes.c_uint32, 24), ('unused_1', ctypes.c_uint32, 28), ('key0_ctx_id', ctypes.c_uint16, 32), ('key1_ctx_id', ctypes.c_uint16, 34), ('record_ctx_id', ctypes.c_uint16, 36), ('efc_ctx_id', ctypes.c_uint16, 38), ('fid_ctx_id', ctypes.c_uint16, 40), ('unused_2', ctypes.c_uint16, 42), ('unused_3', ctypes.c_uint32, 44)])
@c.record
class struct_hwrm_cfa_eem_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_eem_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_eem_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: int
struct_hwrm_cfa_eem_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_cfa_eem_qcfg_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  num_entries: int
  key0_ctx_id: int
  key1_ctx_id: int
  record_ctx_id: int
  efc_ctx_id: int
  fid_ctx_id: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_cfa_eem_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('num_entries', ctypes.c_uint32, 12), ('key0_ctx_id', ctypes.c_uint16, 16), ('key1_ctx_id', ctypes.c_uint16, 18), ('record_ctx_id', ctypes.c_uint16, 20), ('efc_ctx_id', ctypes.c_uint16, 22), ('fid_ctx_id', ctypes.c_uint16, 24), ('unused_2', c.Array[ctypes.c_ubyte, Literal[5]], 26), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_cfa_eem_op_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: int
  op: int
struct_hwrm_cfa_eem_op_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint16, 20), ('op', ctypes.c_uint16, 22)])
@c.record
class struct_hwrm_cfa_eem_op_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_cfa_eem_op_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_cfa_adv_flow_mgnt_qcaps_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  unused_0: c.Array[ctypes.c_uint32, Literal[4]]
struct_hwrm_cfa_adv_flow_mgnt_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('unused_0', c.Array[ctypes.c_uint32, Literal[4]], 16)])
@c.record
class struct_hwrm_cfa_adv_flow_mgnt_qcaps_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_cfa_adv_flow_mgnt_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_tunnel_dst_port_query_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  tunnel_type: int
  tunnel_next_proto: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_tunnel_dst_port_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('tunnel_type', ctypes.c_ubyte, 16), ('tunnel_next_proto', ctypes.c_ubyte, 17), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_tunnel_dst_port_query_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tunnel_dst_port_id: int
  tunnel_dst_port_val: int
  upar_in_use: int
  status: int
  unused_0: int
  valid: int
struct_hwrm_tunnel_dst_port_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tunnel_dst_port_id', ctypes.c_uint16, 8), ('tunnel_dst_port_val', ctypes.c_uint16, 10), ('upar_in_use', ctypes.c_ubyte, 12), ('status', ctypes.c_ubyte, 13), ('unused_0', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_tunnel_dst_port_alloc_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  tunnel_type: int
  tunnel_next_proto: int
  tunnel_dst_port_val: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_tunnel_dst_port_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('tunnel_type', ctypes.c_ubyte, 16), ('tunnel_next_proto', ctypes.c_ubyte, 17), ('tunnel_dst_port_val', ctypes.c_uint16, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_tunnel_dst_port_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tunnel_dst_port_id: int
  error_info: int
  upar_in_use: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_tunnel_dst_port_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tunnel_dst_port_id', ctypes.c_uint16, 8), ('error_info', ctypes.c_ubyte, 10), ('upar_in_use', ctypes.c_ubyte, 11), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_tunnel_dst_port_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  tunnel_type: int
  tunnel_next_proto: int
  tunnel_dst_port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_tunnel_dst_port_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('tunnel_type', ctypes.c_ubyte, 16), ('tunnel_next_proto', ctypes.c_ubyte, 17), ('tunnel_dst_port_id', ctypes.c_uint16, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_tunnel_dst_port_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  error_info: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_tunnel_dst_port_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('error_info', ctypes.c_ubyte, 8), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 9), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_ctx_hw_stats(c.Struct):
  SIZE = 160
  rx_ucast_pkts: int
  rx_mcast_pkts: int
  rx_bcast_pkts: int
  rx_discard_pkts: int
  rx_error_pkts: int
  rx_ucast_bytes: int
  rx_mcast_bytes: int
  rx_bcast_bytes: int
  tx_ucast_pkts: int
  tx_mcast_pkts: int
  tx_bcast_pkts: int
  tx_error_pkts: int
  tx_discard_pkts: int
  tx_ucast_bytes: int
  tx_mcast_bytes: int
  tx_bcast_bytes: int
  tpa_pkts: int
  tpa_bytes: int
  tpa_events: int
  tpa_aborts: int
struct_ctx_hw_stats.register_fields([('rx_ucast_pkts', ctypes.c_uint64, 0), ('rx_mcast_pkts', ctypes.c_uint64, 8), ('rx_bcast_pkts', ctypes.c_uint64, 16), ('rx_discard_pkts', ctypes.c_uint64, 24), ('rx_error_pkts', ctypes.c_uint64, 32), ('rx_ucast_bytes', ctypes.c_uint64, 40), ('rx_mcast_bytes', ctypes.c_uint64, 48), ('rx_bcast_bytes', ctypes.c_uint64, 56), ('tx_ucast_pkts', ctypes.c_uint64, 64), ('tx_mcast_pkts', ctypes.c_uint64, 72), ('tx_bcast_pkts', ctypes.c_uint64, 80), ('tx_error_pkts', ctypes.c_uint64, 88), ('tx_discard_pkts', ctypes.c_uint64, 96), ('tx_ucast_bytes', ctypes.c_uint64, 104), ('tx_mcast_bytes', ctypes.c_uint64, 112), ('tx_bcast_bytes', ctypes.c_uint64, 120), ('tpa_pkts', ctypes.c_uint64, 128), ('tpa_bytes', ctypes.c_uint64, 136), ('tpa_events', ctypes.c_uint64, 144), ('tpa_aborts', ctypes.c_uint64, 152)])
@c.record
class struct_ctx_hw_stats_ext(c.Struct):
  SIZE = 176
  rx_ucast_pkts: int
  rx_mcast_pkts: int
  rx_bcast_pkts: int
  rx_discard_pkts: int
  rx_error_pkts: int
  rx_ucast_bytes: int
  rx_mcast_bytes: int
  rx_bcast_bytes: int
  tx_ucast_pkts: int
  tx_mcast_pkts: int
  tx_bcast_pkts: int
  tx_error_pkts: int
  tx_discard_pkts: int
  tx_ucast_bytes: int
  tx_mcast_bytes: int
  tx_bcast_bytes: int
  rx_tpa_eligible_pkt: int
  rx_tpa_eligible_bytes: int
  rx_tpa_pkt: int
  rx_tpa_bytes: int
  rx_tpa_errors: int
  rx_tpa_events: int
struct_ctx_hw_stats_ext.register_fields([('rx_ucast_pkts', ctypes.c_uint64, 0), ('rx_mcast_pkts', ctypes.c_uint64, 8), ('rx_bcast_pkts', ctypes.c_uint64, 16), ('rx_discard_pkts', ctypes.c_uint64, 24), ('rx_error_pkts', ctypes.c_uint64, 32), ('rx_ucast_bytes', ctypes.c_uint64, 40), ('rx_mcast_bytes', ctypes.c_uint64, 48), ('rx_bcast_bytes', ctypes.c_uint64, 56), ('tx_ucast_pkts', ctypes.c_uint64, 64), ('tx_mcast_pkts', ctypes.c_uint64, 72), ('tx_bcast_pkts', ctypes.c_uint64, 80), ('tx_error_pkts', ctypes.c_uint64, 88), ('tx_discard_pkts', ctypes.c_uint64, 96), ('tx_ucast_bytes', ctypes.c_uint64, 104), ('tx_mcast_bytes', ctypes.c_uint64, 112), ('tx_bcast_bytes', ctypes.c_uint64, 120), ('rx_tpa_eligible_pkt', ctypes.c_uint64, 128), ('rx_tpa_eligible_bytes', ctypes.c_uint64, 136), ('rx_tpa_pkt', ctypes.c_uint64, 144), ('rx_tpa_bytes', ctypes.c_uint64, 152), ('rx_tpa_errors', ctypes.c_uint64, 160), ('rx_tpa_events', ctypes.c_uint64, 168)])
@c.record
class struct_hwrm_stat_ctx_alloc_input(c.Struct):
  SIZE = 48
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  stats_dma_addr: int
  update_period_ms: int
  stat_ctx_flags: int
  unused_0: int
  stats_dma_length: int
  flags: int
  steering_tag: int
  stat_ctx_id: int
  alloc_seq_id: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_stat_ctx_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('stats_dma_addr', ctypes.c_uint64, 16), ('update_period_ms', ctypes.c_uint32, 24), ('stat_ctx_flags', ctypes.c_ubyte, 28), ('unused_0', ctypes.c_ubyte, 29), ('stats_dma_length', ctypes.c_uint16, 30), ('flags', ctypes.c_uint16, 32), ('steering_tag', ctypes.c_uint16, 34), ('stat_ctx_id', ctypes.c_uint32, 36), ('alloc_seq_id', ctypes.c_uint16, 40), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 42)])
@c.record
class struct_hwrm_stat_ctx_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  stat_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_stat_ctx_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('stat_ctx_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_stat_ctx_free_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  stat_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_stat_ctx_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('stat_ctx_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_stat_ctx_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  stat_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_stat_ctx_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('stat_ctx_id', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_stat_ctx_query_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  stat_ctx_id: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
struct_hwrm_stat_ctx_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('stat_ctx_id', ctypes.c_uint32, 16), ('flags', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21)])
@c.record
class struct_hwrm_stat_ctx_query_output(c.Struct):
  SIZE = 176
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  tx_ucast_pkts: int
  tx_mcast_pkts: int
  tx_bcast_pkts: int
  tx_discard_pkts: int
  tx_error_pkts: int
  tx_ucast_bytes: int
  tx_mcast_bytes: int
  tx_bcast_bytes: int
  rx_ucast_pkts: int
  rx_mcast_pkts: int
  rx_bcast_pkts: int
  rx_discard_pkts: int
  rx_error_pkts: int
  rx_ucast_bytes: int
  rx_mcast_bytes: int
  rx_bcast_bytes: int
  rx_agg_pkts: int
  rx_agg_bytes: int
  rx_agg_events: int
  rx_agg_aborts: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_stat_ctx_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('tx_ucast_pkts', ctypes.c_uint64, 8), ('tx_mcast_pkts', ctypes.c_uint64, 16), ('tx_bcast_pkts', ctypes.c_uint64, 24), ('tx_discard_pkts', ctypes.c_uint64, 32), ('tx_error_pkts', ctypes.c_uint64, 40), ('tx_ucast_bytes', ctypes.c_uint64, 48), ('tx_mcast_bytes', ctypes.c_uint64, 56), ('tx_bcast_bytes', ctypes.c_uint64, 64), ('rx_ucast_pkts', ctypes.c_uint64, 72), ('rx_mcast_pkts', ctypes.c_uint64, 80), ('rx_bcast_pkts', ctypes.c_uint64, 88), ('rx_discard_pkts', ctypes.c_uint64, 96), ('rx_error_pkts', ctypes.c_uint64, 104), ('rx_ucast_bytes', ctypes.c_uint64, 112), ('rx_mcast_bytes', ctypes.c_uint64, 120), ('rx_bcast_bytes', ctypes.c_uint64, 128), ('rx_agg_pkts', ctypes.c_uint64, 136), ('rx_agg_bytes', ctypes.c_uint64, 144), ('rx_agg_events', ctypes.c_uint64, 152), ('rx_agg_aborts', ctypes.c_uint64, 160), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 168), ('valid', ctypes.c_ubyte, 175)])
@c.record
class struct_hwrm_stat_ext_ctx_query_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  stat_ctx_id: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
struct_hwrm_stat_ext_ctx_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('stat_ctx_id', ctypes.c_uint32, 16), ('flags', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21)])
@c.record
class struct_hwrm_stat_ext_ctx_query_output(c.Struct):
  SIZE = 192
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  rx_ucast_pkts: int
  rx_mcast_pkts: int
  rx_bcast_pkts: int
  rx_discard_pkts: int
  rx_error_pkts: int
  rx_ucast_bytes: int
  rx_mcast_bytes: int
  rx_bcast_bytes: int
  tx_ucast_pkts: int
  tx_mcast_pkts: int
  tx_bcast_pkts: int
  tx_error_pkts: int
  tx_discard_pkts: int
  tx_ucast_bytes: int
  tx_mcast_bytes: int
  tx_bcast_bytes: int
  rx_tpa_eligible_pkt: int
  rx_tpa_eligible_bytes: int
  rx_tpa_pkt: int
  rx_tpa_bytes: int
  rx_tpa_errors: int
  rx_tpa_events: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_stat_ext_ctx_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('rx_ucast_pkts', ctypes.c_uint64, 8), ('rx_mcast_pkts', ctypes.c_uint64, 16), ('rx_bcast_pkts', ctypes.c_uint64, 24), ('rx_discard_pkts', ctypes.c_uint64, 32), ('rx_error_pkts', ctypes.c_uint64, 40), ('rx_ucast_bytes', ctypes.c_uint64, 48), ('rx_mcast_bytes', ctypes.c_uint64, 56), ('rx_bcast_bytes', ctypes.c_uint64, 64), ('tx_ucast_pkts', ctypes.c_uint64, 72), ('tx_mcast_pkts', ctypes.c_uint64, 80), ('tx_bcast_pkts', ctypes.c_uint64, 88), ('tx_error_pkts', ctypes.c_uint64, 96), ('tx_discard_pkts', ctypes.c_uint64, 104), ('tx_ucast_bytes', ctypes.c_uint64, 112), ('tx_mcast_bytes', ctypes.c_uint64, 120), ('tx_bcast_bytes', ctypes.c_uint64, 128), ('rx_tpa_eligible_pkt', ctypes.c_uint64, 136), ('rx_tpa_eligible_bytes', ctypes.c_uint64, 144), ('rx_tpa_pkt', ctypes.c_uint64, 152), ('rx_tpa_bytes', ctypes.c_uint64, 160), ('rx_tpa_errors', ctypes.c_uint64, 168), ('rx_tpa_events', ctypes.c_uint64, 176), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 184), ('valid', ctypes.c_ubyte, 191)])
@c.record
class struct_hwrm_stat_ctx_clr_stats_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  stat_ctx_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_stat_ctx_clr_stats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('stat_ctx_id', ctypes.c_uint32, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_stat_ctx_clr_stats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_stat_ctx_clr_stats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_pcie_qstats_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  pcie_stat_size: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  pcie_stat_host_addr: int
struct_hwrm_pcie_qstats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('pcie_stat_size', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18), ('pcie_stat_host_addr', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_pcie_qstats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  pcie_stat_size: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_pcie_qstats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('pcie_stat_size', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_pcie_ctx_hw_stats(c.Struct):
  SIZE = 96
  pcie_pl_signal_integrity: int
  pcie_dl_signal_integrity: int
  pcie_tl_signal_integrity: int
  pcie_link_integrity: int
  pcie_tx_traffic_rate: int
  pcie_rx_traffic_rate: int
  pcie_tx_dllp_statistics: int
  pcie_rx_dllp_statistics: int
  pcie_equalization_time: int
  pcie_ltssm_histogram: c.Array[ctypes.c_uint32, Literal[4]]
  pcie_recovery_histogram: int
struct_pcie_ctx_hw_stats.register_fields([('pcie_pl_signal_integrity', ctypes.c_uint64, 0), ('pcie_dl_signal_integrity', ctypes.c_uint64, 8), ('pcie_tl_signal_integrity', ctypes.c_uint64, 16), ('pcie_link_integrity', ctypes.c_uint64, 24), ('pcie_tx_traffic_rate', ctypes.c_uint64, 32), ('pcie_rx_traffic_rate', ctypes.c_uint64, 40), ('pcie_tx_dllp_statistics', ctypes.c_uint64, 48), ('pcie_rx_dllp_statistics', ctypes.c_uint64, 56), ('pcie_equalization_time', ctypes.c_uint64, 64), ('pcie_ltssm_histogram', c.Array[ctypes.c_uint32, Literal[4]], 72), ('pcie_recovery_histogram', ctypes.c_uint64, 88)])
@c.record
class struct_pcie_ctx_hw_stats_v2(c.Struct):
  SIZE = 568
  pcie_pl_signal_integrity: int
  pcie_dl_signal_integrity: int
  pcie_tl_signal_integrity: int
  pcie_link_integrity: int
  pcie_tx_traffic_rate: int
  pcie_rx_traffic_rate: int
  pcie_tx_dllp_statistics: int
  pcie_rx_dllp_statistics: int
  pcie_equalization_time: int
  pcie_ltssm_histogram: c.Array[ctypes.c_uint32, Literal[4]]
  pcie_recovery_histogram: int
  pcie_tl_credit_nph_histogram: c.Array[ctypes.c_uint32, Literal[8]]
  pcie_tl_credit_ph_histogram: c.Array[ctypes.c_uint32, Literal[8]]
  pcie_tl_credit_pd_histogram: c.Array[ctypes.c_uint32, Literal[8]]
  pcie_cmpl_latest_times: c.Array[ctypes.c_uint32, Literal[4]]
  pcie_cmpl_longest_time: int
  pcie_cmpl_shortest_time: int
  unused_0: c.Array[ctypes.c_uint32, Literal[2]]
  pcie_cmpl_latest_headers: c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]]
  pcie_cmpl_longest_headers: c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]]
  pcie_cmpl_shortest_headers: c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]]
  pcie_wr_latency_histogram: c.Array[ctypes.c_uint32, Literal[12]]
  pcie_wr_latency_all_normal_count: int
  unused_1: int
  pcie_posted_packet_count: int
  pcie_non_posted_packet_count: int
  pcie_other_packet_count: int
  pcie_blocked_packet_count: int
  pcie_cmpl_packet_count: int
  pcie_rd_latency_histogram: c.Array[ctypes.c_uint32, Literal[12]]
  pcie_rd_latency_all_normal_count: int
  unused_2: int
struct_pcie_ctx_hw_stats_v2.register_fields([('pcie_pl_signal_integrity', ctypes.c_uint64, 0), ('pcie_dl_signal_integrity', ctypes.c_uint64, 8), ('pcie_tl_signal_integrity', ctypes.c_uint64, 16), ('pcie_link_integrity', ctypes.c_uint64, 24), ('pcie_tx_traffic_rate', ctypes.c_uint64, 32), ('pcie_rx_traffic_rate', ctypes.c_uint64, 40), ('pcie_tx_dllp_statistics', ctypes.c_uint64, 48), ('pcie_rx_dllp_statistics', ctypes.c_uint64, 56), ('pcie_equalization_time', ctypes.c_uint64, 64), ('pcie_ltssm_histogram', c.Array[ctypes.c_uint32, Literal[4]], 72), ('pcie_recovery_histogram', ctypes.c_uint64, 88), ('pcie_tl_credit_nph_histogram', c.Array[ctypes.c_uint32, Literal[8]], 96), ('pcie_tl_credit_ph_histogram', c.Array[ctypes.c_uint32, Literal[8]], 128), ('pcie_tl_credit_pd_histogram', c.Array[ctypes.c_uint32, Literal[8]], 160), ('pcie_cmpl_latest_times', c.Array[ctypes.c_uint32, Literal[4]], 192), ('pcie_cmpl_longest_time', ctypes.c_uint32, 208), ('pcie_cmpl_shortest_time', ctypes.c_uint32, 212), ('unused_0', c.Array[ctypes.c_uint32, Literal[2]], 216), ('pcie_cmpl_latest_headers', c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]], 224), ('pcie_cmpl_longest_headers', c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]], 288), ('pcie_cmpl_shortest_headers', c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]], 352), ('pcie_wr_latency_histogram', c.Array[ctypes.c_uint32, Literal[12]], 416), ('pcie_wr_latency_all_normal_count', ctypes.c_uint32, 464), ('unused_1', ctypes.c_uint32, 468), ('pcie_posted_packet_count', ctypes.c_uint64, 472), ('pcie_non_posted_packet_count', ctypes.c_uint64, 480), ('pcie_other_packet_count', ctypes.c_uint64, 488), ('pcie_blocked_packet_count', ctypes.c_uint64, 496), ('pcie_cmpl_packet_count', ctypes.c_uint64, 504), ('pcie_rd_latency_histogram', c.Array[ctypes.c_uint32, Literal[12]], 512), ('pcie_rd_latency_all_normal_count', ctypes.c_uint32, 560), ('unused_2', ctypes.c_uint32, 564)])
@c.record
class struct_hwrm_stat_generic_qstats_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  generic_stat_size: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  generic_stat_host_addr: int
struct_hwrm_stat_generic_qstats_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('generic_stat_size', ctypes.c_uint16, 16), ('flags', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 19), ('generic_stat_host_addr', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_stat_generic_qstats_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  generic_stat_size: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_stat_generic_qstats_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('generic_stat_size', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_generic_sw_hw_stats(c.Struct):
  SIZE = 184
  pcie_statistics_tx_tlp: int
  pcie_statistics_rx_tlp: int
  pcie_credit_fc_hdr_posted: int
  pcie_credit_fc_hdr_nonposted: int
  pcie_credit_fc_hdr_cmpl: int
  pcie_credit_fc_data_posted: int
  pcie_credit_fc_data_nonposted: int
  pcie_credit_fc_data_cmpl: int
  pcie_credit_fc_tgt_nonposted: int
  pcie_credit_fc_tgt_data_posted: int
  pcie_credit_fc_tgt_hdr_posted: int
  pcie_credit_fc_cmpl_hdr_posted: int
  pcie_credit_fc_cmpl_data_posted: int
  pcie_cmpl_longest: int
  pcie_cmpl_shortest: int
  cache_miss_count_cfcq: int
  cache_miss_count_cfcs: int
  cache_miss_count_cfcc: int
  cache_miss_count_cfcm: int
  hw_db_recov_dbs_dropped: int
  hw_db_recov_drops_serviced: int
  hw_db_recov_dbs_recovered: int
  hw_db_recov_oo_drop_count: int
struct_generic_sw_hw_stats.register_fields([('pcie_statistics_tx_tlp', ctypes.c_uint64, 0), ('pcie_statistics_rx_tlp', ctypes.c_uint64, 8), ('pcie_credit_fc_hdr_posted', ctypes.c_uint64, 16), ('pcie_credit_fc_hdr_nonposted', ctypes.c_uint64, 24), ('pcie_credit_fc_hdr_cmpl', ctypes.c_uint64, 32), ('pcie_credit_fc_data_posted', ctypes.c_uint64, 40), ('pcie_credit_fc_data_nonposted', ctypes.c_uint64, 48), ('pcie_credit_fc_data_cmpl', ctypes.c_uint64, 56), ('pcie_credit_fc_tgt_nonposted', ctypes.c_uint64, 64), ('pcie_credit_fc_tgt_data_posted', ctypes.c_uint64, 72), ('pcie_credit_fc_tgt_hdr_posted', ctypes.c_uint64, 80), ('pcie_credit_fc_cmpl_hdr_posted', ctypes.c_uint64, 88), ('pcie_credit_fc_cmpl_data_posted', ctypes.c_uint64, 96), ('pcie_cmpl_longest', ctypes.c_uint64, 104), ('pcie_cmpl_shortest', ctypes.c_uint64, 112), ('cache_miss_count_cfcq', ctypes.c_uint64, 120), ('cache_miss_count_cfcs', ctypes.c_uint64, 128), ('cache_miss_count_cfcc', ctypes.c_uint64, 136), ('cache_miss_count_cfcm', ctypes.c_uint64, 144), ('hw_db_recov_dbs_dropped', ctypes.c_uint64, 152), ('hw_db_recov_drops_serviced', ctypes.c_uint64, 160), ('hw_db_recov_dbs_recovered', ctypes.c_uint64, 168), ('hw_db_recov_oo_drop_count', ctypes.c_uint64, 176)])
@c.record
class struct_hwrm_fw_reset_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  embedded_proc_type: int
  selfrst_status: int
  host_idx: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_fw_reset_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('embedded_proc_type', ctypes.c_ubyte, 16), ('selfrst_status', ctypes.c_ubyte, 17), ('host_idx', ctypes.c_ubyte, 18), ('flags', ctypes.c_ubyte, 19), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20)])
@c.record
class struct_hwrm_fw_reset_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  selfrst_status: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_fw_reset_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('selfrst_status', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 9), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fw_qstatus_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  embedded_proc_type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_fw_qstatus_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('embedded_proc_type', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 17)])
@c.record
class struct_hwrm_fw_qstatus_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  selfrst_status: int
  nvm_option_action_status: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_fw_qstatus_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('selfrst_status', ctypes.c_ubyte, 8), ('nvm_option_action_status', ctypes.c_ubyte, 9), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fw_set_time_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  year: int
  month: int
  day: int
  hour: int
  minute: int
  second: int
  unused_0: int
  millisecond: int
  zone: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_fw_set_time_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('year', ctypes.c_uint16, 16), ('month', ctypes.c_ubyte, 18), ('day', ctypes.c_ubyte, 19), ('hour', ctypes.c_ubyte, 20), ('minute', ctypes.c_ubyte, 21), ('second', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23), ('millisecond', ctypes.c_uint16, 24), ('zone', ctypes.c_uint16, 26), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 28)])
@c.record
class struct_hwrm_fw_set_time_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_fw_set_time_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_struct_hdr(c.Struct):
  SIZE = 16
  struct_id: int
  len: int
  version: int
  count: int
  subtype: int
  next_offset: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_struct_hdr.register_fields([('struct_id', ctypes.c_uint16, 0), ('len', ctypes.c_uint16, 2), ('version', ctypes.c_ubyte, 4), ('count', ctypes.c_ubyte, 5), ('subtype', ctypes.c_uint16, 6), ('next_offset', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_hwrm_struct_data_dcbx_app(c.Struct):
  SIZE = 8
  protocol_id: int
  protocol_selector: int
  priority: int
  valid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
struct_hwrm_struct_data_dcbx_app.register_fields([('protocol_id', ctypes.c_uint16, 0), ('protocol_selector', ctypes.c_ubyte, 2), ('priority', ctypes.c_ubyte, 3), ('valid', ctypes.c_ubyte, 4), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 5)])
@c.record
class struct_hwrm_fw_set_structured_data_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  src_data_addr: int
  data_len: int
  hdr_cnt: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
struct_hwrm_fw_set_structured_data_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('src_data_addr', ctypes.c_uint64, 16), ('data_len', ctypes.c_uint16, 24), ('hdr_cnt', ctypes.c_ubyte, 26), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 27)])
@c.record
class struct_hwrm_fw_set_structured_data_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_fw_set_structured_data_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fw_set_structured_data_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_fw_set_structured_data_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_fw_get_structured_data_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  dest_data_addr: int
  data_len: int
  structure_id: int
  subtype: int
  count: int
  unused_0: int
struct_hwrm_fw_get_structured_data_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('dest_data_addr', ctypes.c_uint64, 16), ('data_len', ctypes.c_uint16, 24), ('structure_id', ctypes.c_uint16, 26), ('subtype', ctypes.c_uint16, 28), ('count', ctypes.c_ubyte, 30), ('unused_0', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_fw_get_structured_data_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  hdr_cnt: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_fw_get_structured_data_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('hdr_cnt', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 9), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fw_get_structured_data_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_fw_get_structured_data_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_fw_livepatch_query_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fw_target: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_fw_livepatch_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fw_target', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 17)])
@c.record
class struct_hwrm_fw_livepatch_query_output(c.Struct):
  SIZE = 80
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  install_ver: c.Array[ctypes.c_char, Literal[32]]
  active_ver: c.Array[ctypes.c_char, Literal[32]]
  status_flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_fw_livepatch_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('install_ver', c.Array[ctypes.c_char, Literal[32]], 8), ('active_ver', c.Array[ctypes.c_char, Literal[32]], 40), ('status_flags', ctypes.c_uint16, 72), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 74), ('valid', ctypes.c_ubyte, 79)])
@c.record
class struct_hwrm_fw_livepatch_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  opcode: int
  fw_target: int
  loadtype: int
  flags: int
  patch_len: int
  host_addr: int
struct_hwrm_fw_livepatch_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('opcode', ctypes.c_ubyte, 16), ('fw_target', ctypes.c_ubyte, 17), ('loadtype', ctypes.c_ubyte, 18), ('flags', ctypes.c_ubyte, 19), ('patch_len', ctypes.c_uint32, 20), ('host_addr', ctypes.c_uint64, 24)])
@c.record
class struct_hwrm_fw_livepatch_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_fw_livepatch_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fw_livepatch_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_fw_livepatch_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_exec_fwd_resp_input(c.Struct):
  SIZE = 128
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  encap_request: c.Array[ctypes.c_uint32, Literal[26]]
  encap_resp_target_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_exec_fwd_resp_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('encap_request', c.Array[ctypes.c_uint32, Literal[26]], 16), ('encap_resp_target_id', ctypes.c_uint16, 120), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 122)])
@c.record
class struct_hwrm_exec_fwd_resp_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_exec_fwd_resp_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_reject_fwd_resp_input(c.Struct):
  SIZE = 128
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  encap_request: c.Array[ctypes.c_uint32, Literal[26]]
  encap_resp_target_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_reject_fwd_resp_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('encap_request', c.Array[ctypes.c_uint32, Literal[26]], 16), ('encap_resp_target_id', ctypes.c_uint16, 120), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 122)])
@c.record
class struct_hwrm_reject_fwd_resp_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_reject_fwd_resp_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fwd_resp_input(c.Struct):
  SIZE = 128
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  encap_resp_target_id: int
  encap_resp_cmpl_ring: int
  encap_resp_len: int
  unused_0: int
  unused_1: int
  encap_resp_addr: int
  encap_resp: c.Array[ctypes.c_uint32, Literal[24]]
struct_hwrm_fwd_resp_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('encap_resp_target_id', ctypes.c_uint16, 16), ('encap_resp_cmpl_ring', ctypes.c_uint16, 18), ('encap_resp_len', ctypes.c_uint16, 20), ('unused_0', ctypes.c_ubyte, 22), ('unused_1', ctypes.c_ubyte, 23), ('encap_resp_addr', ctypes.c_uint64, 24), ('encap_resp', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_hwrm_fwd_resp_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_fwd_resp_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_fwd_async_event_cmpl_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  encap_async_event_target_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  encap_async_event_cmpl: c.Array[ctypes.c_uint32, Literal[4]]
struct_hwrm_fwd_async_event_cmpl_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('encap_async_event_target_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18), ('encap_async_event_cmpl', c.Array[ctypes.c_uint32, Literal[4]], 24)])
@c.record
class struct_hwrm_fwd_async_event_cmpl_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_fwd_async_event_cmpl_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_temp_monitor_query_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_temp_monitor_query_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_temp_monitor_query_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  temp: int
  phy_temp: int
  om_temp: int
  flags: int
  temp2: int
  phy_temp2: int
  om_temp2: int
  warn_threshold: int
  critical_threshold: int
  fatal_threshold: int
  shutdown_threshold: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  valid: int
struct_hwrm_temp_monitor_query_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('temp', ctypes.c_ubyte, 8), ('phy_temp', ctypes.c_ubyte, 9), ('om_temp', ctypes.c_ubyte, 10), ('flags', ctypes.c_ubyte, 11), ('temp2', ctypes.c_ubyte, 12), ('phy_temp2', ctypes.c_ubyte, 13), ('om_temp2', ctypes.c_ubyte, 14), ('warn_threshold', ctypes.c_ubyte, 15), ('critical_threshold', ctypes.c_ubyte, 16), ('fatal_threshold', ctypes.c_ubyte, 17), ('shutdown_threshold', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 19), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_wol_filter_alloc_input(c.Struct):
  SIZE = 64
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  wol_type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  mac_address: c.Array[ctypes.c_ubyte, Literal[6]]
  pattern_offset: int
  pattern_buf_size: int
  pattern_mask_size: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
  pattern_buf_addr: int
  pattern_mask_addr: int
struct_hwrm_wol_filter_alloc_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('wol_type', ctypes.c_ubyte, 26), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 27), ('mac_address', c.Array[ctypes.c_ubyte, Literal[6]], 32), ('pattern_offset', ctypes.c_uint16, 38), ('pattern_buf_size', ctypes.c_uint16, 40), ('pattern_mask_size', ctypes.c_uint16, 42), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 44), ('pattern_buf_addr', ctypes.c_uint64, 48), ('pattern_mask_addr', ctypes.c_uint64, 56)])
@c.record
class struct_hwrm_wol_filter_alloc_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  wol_filter_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_wol_filter_alloc_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('wol_filter_id', ctypes.c_ubyte, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 9), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_wol_filter_free_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  enables: int
  port_id: int
  wol_filter_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
struct_hwrm_wol_filter_free_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('enables', ctypes.c_uint32, 20), ('port_id', ctypes.c_uint16, 24), ('wol_filter_id', ctypes.c_ubyte, 26), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 27)])
@c.record
class struct_hwrm_wol_filter_free_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_wol_filter_free_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_wol_filter_qcfg_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  handle: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  pattern_buf_addr: int
  pattern_buf_size: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
  pattern_mask_addr: int
  pattern_mask_size: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_wol_filter_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('handle', ctypes.c_uint16, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 20), ('pattern_buf_addr', ctypes.c_uint64, 24), ('pattern_buf_size', ctypes.c_uint16, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 34), ('pattern_mask_addr', ctypes.c_uint64, 40), ('pattern_mask_size', ctypes.c_uint16, 48), ('unused_2', c.Array[ctypes.c_ubyte, Literal[6]], 50)])
@c.record
class struct_hwrm_wol_filter_qcfg_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  next_handle: int
  wol_filter_id: int
  wol_type: int
  unused_0: int
  mac_address: c.Array[ctypes.c_ubyte, Literal[6]]
  pattern_offset: int
  pattern_size: int
  pattern_mask_size: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_wol_filter_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('next_handle', ctypes.c_uint16, 8), ('wol_filter_id', ctypes.c_ubyte, 10), ('wol_type', ctypes.c_ubyte, 11), ('unused_0', ctypes.c_uint32, 12), ('mac_address', c.Array[ctypes.c_ubyte, Literal[6]], 16), ('pattern_offset', ctypes.c_uint16, 22), ('pattern_size', ctypes.c_uint16, 24), ('pattern_mask_size', ctypes.c_uint16, 26), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 28), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_wol_reason_qcfg_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  port_id: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
  wol_pkt_buf_addr: int
  wol_pkt_buf_size: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_wol_reason_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('port_id', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18), ('wol_pkt_buf_addr', ctypes.c_uint64, 24), ('wol_pkt_buf_size', ctypes.c_uint16, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[6]], 34)])
@c.record
class struct_hwrm_wol_reason_qcfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  wol_filter_id: int
  wol_reason: int
  wol_pkt_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  valid: int
struct_hwrm_wol_reason_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('wol_filter_id', ctypes.c_ubyte, 8), ('wol_reason', ctypes.c_ubyte, 9), ('wol_pkt_len', ctypes.c_ubyte, 10), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 11), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_dbg_read_direct_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_dest_addr: int
  read_addr: int
  read_len32: int
struct_hwrm_dbg_read_direct_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_dest_addr', ctypes.c_uint64, 16), ('read_addr', ctypes.c_uint32, 24), ('read_len32', ctypes.c_uint32, 28)])
@c.record
class struct_hwrm_dbg_read_direct_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  crc32: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_dbg_read_direct_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('crc32', ctypes.c_uint32, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_dbg_qcaps_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_dbg_qcaps_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_dbg_qcaps_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  coredump_component_disable_caps: int
  flags: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_dbg_qcaps_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 10), ('coredump_component_disable_caps', ctypes.c_uint32, 12), ('flags', ctypes.c_uint32, 16), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 20), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_dbg_qcfg_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  fid: int
  flags: int
  coredump_component_disable_flags: int
struct_hwrm_dbg_qcfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('fid', ctypes.c_uint16, 16), ('flags', ctypes.c_uint16, 18), ('coredump_component_disable_flags', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_dbg_qcfg_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  fid: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  coredump_size: int
  flags: int
  async_cmpl_ring: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[2]]
  crashdump_size: int
  unused_3: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_dbg_qcfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('fid', ctypes.c_uint16, 8), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 10), ('coredump_size', ctypes.c_uint32, 12), ('flags', ctypes.c_uint32, 16), ('async_cmpl_ring', ctypes.c_uint16, 20), ('unused_2', c.Array[ctypes.c_ubyte, Literal[2]], 22), ('crashdump_size', ctypes.c_uint32, 24), ('unused_3', c.Array[ctypes.c_ubyte, Literal[3]], 28), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_dbg_crashdump_medium_cfg_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  output_dest_flags: int
  pg_size_lvl: int
  size: int
  coredump_component_disable_flags: int
  unused_0: int
  pbl: int
struct_hwrm_dbg_crashdump_medium_cfg_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('output_dest_flags', ctypes.c_uint16, 16), ('pg_size_lvl', ctypes.c_uint16, 18), ('size', ctypes.c_uint32, 20), ('coredump_component_disable_flags', ctypes.c_uint32, 24), ('unused_0', ctypes.c_uint32, 28), ('pbl', ctypes.c_uint64, 32)])
@c.record
class struct_hwrm_dbg_crashdump_medium_cfg_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_dbg_crashdump_medium_cfg_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_coredump_segment_record(c.Struct):
  SIZE = 16
  component_id: int
  segment_id: int
  max_instances: int
  version_hi: int
  version_low: int
  seg_flags: int
  compress_flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  segment_len: int
struct_coredump_segment_record.register_fields([('component_id', ctypes.c_uint16, 0), ('segment_id', ctypes.c_uint16, 2), ('max_instances', ctypes.c_uint16, 4), ('version_hi', ctypes.c_ubyte, 6), ('version_low', ctypes.c_ubyte, 7), ('seg_flags', ctypes.c_ubyte, 8), ('compress_flags', ctypes.c_ubyte, 9), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 10), ('segment_len', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_dbg_coredump_list_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_dest_addr: int
  host_buf_len: int
  seq_no: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[1]]
struct_hwrm_dbg_coredump_list_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_dest_addr', ctypes.c_uint64, 16), ('host_buf_len', ctypes.c_uint32, 24), ('seq_no', ctypes.c_uint16, 28), ('flags', ctypes.c_ubyte, 30), ('unused_0', c.Array[ctypes.c_ubyte, Literal[1]], 31)])
@c.record
class struct_hwrm_dbg_coredump_list_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: int
  total_segments: int
  data_len: int
  unused_1: int
  valid: int
struct_hwrm_dbg_coredump_list_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('unused_0', ctypes.c_ubyte, 9), ('total_segments', ctypes.c_uint16, 10), ('data_len', ctypes.c_uint16, 12), ('unused_1', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_dbg_coredump_initiate_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  component_id: int
  segment_id: int
  instance: int
  unused_0: int
  seg_flags: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_dbg_coredump_initiate_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('component_id', ctypes.c_uint16, 16), ('segment_id', ctypes.c_uint16, 18), ('instance', ctypes.c_uint16, 20), ('unused_0', ctypes.c_uint16, 22), ('seg_flags', ctypes.c_ubyte, 24), ('unused_1', c.Array[ctypes.c_ubyte, Literal[7]], 25)])
@c.record
class struct_hwrm_dbg_coredump_initiate_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_dbg_coredump_initiate_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_coredump_data_hdr(c.Struct):
  SIZE = 16
  address: int
  flags_length: int
  instance: int
  next_offset: int
struct_coredump_data_hdr.register_fields([('address', ctypes.c_uint32, 0), ('flags_length', ctypes.c_uint32, 4), ('instance', ctypes.c_uint32, 8), ('next_offset', ctypes.c_uint32, 12)])
@c.record
class struct_hwrm_dbg_coredump_retrieve_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_dest_addr: int
  host_buf_len: int
  unused_0: int
  component_id: int
  segment_id: int
  instance: int
  unused_1: int
  seg_flags: int
  unused_2: int
  unused_3: int
  unused_4: int
  seq_no: int
  unused_5: int
struct_hwrm_dbg_coredump_retrieve_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_dest_addr', ctypes.c_uint64, 16), ('host_buf_len', ctypes.c_uint32, 24), ('unused_0', ctypes.c_uint32, 28), ('component_id', ctypes.c_uint16, 32), ('segment_id', ctypes.c_uint16, 34), ('instance', ctypes.c_uint16, 36), ('unused_1', ctypes.c_uint16, 38), ('seg_flags', ctypes.c_ubyte, 40), ('unused_2', ctypes.c_ubyte, 41), ('unused_3', ctypes.c_uint16, 42), ('unused_4', ctypes.c_uint32, 44), ('seq_no', ctypes.c_uint32, 48), ('unused_5', ctypes.c_uint32, 52)])
@c.record
class struct_hwrm_dbg_coredump_retrieve_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  flags: int
  unused_0: int
  data_len: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_dbg_coredump_retrieve_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('flags', ctypes.c_ubyte, 8), ('unused_0', ctypes.c_ubyte, 9), ('data_len', ctypes.c_uint16, 10), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_dbg_ring_info_get_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  ring_type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  fw_ring_id: int
struct_hwrm_dbg_ring_info_get_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('ring_type', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 17), ('fw_ring_id', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_dbg_ring_info_get_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  producer_index: int
  consumer_index: int
  cag_vector_ctrl: int
  st_tag: int
  unused_0: int
  valid: int
struct_hwrm_dbg_ring_info_get_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('producer_index', ctypes.c_uint32, 8), ('consumer_index', ctypes.c_uint32, 12), ('cag_vector_ctrl', ctypes.c_uint32, 16), ('st_tag', ctypes.c_uint16, 20), ('unused_0', ctypes.c_ubyte, 22), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_dbg_log_buffer_flush_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  type: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[2]]
  flags: int
struct_hwrm_dbg_log_buffer_flush_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('type', ctypes.c_uint16, 16), ('unused_1', c.Array[ctypes.c_ubyte, Literal[2]], 18), ('flags', ctypes.c_uint32, 20)])
@c.record
class struct_hwrm_dbg_log_buffer_flush_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  current_buffer_offset: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_dbg_log_buffer_flush_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('current_buffer_offset', ctypes.c_uint32, 8), ('unused_1', c.Array[ctypes.c_ubyte, Literal[3]], 12), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_read_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_dest_addr: int
  dir_idx: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  offset: int
  len: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_nvm_read_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_dest_addr', ctypes.c_uint64, 16), ('dir_idx', ctypes.c_uint16, 24), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 26), ('offset', ctypes.c_uint32, 28), ('len', ctypes.c_uint32, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 36)])
@c.record
class struct_hwrm_nvm_read_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_read_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_get_dir_entries_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_dest_addr: int
struct_hwrm_nvm_get_dir_entries_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_dest_addr', ctypes.c_uint64, 16)])
@c.record
class struct_hwrm_nvm_get_dir_entries_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_get_dir_entries_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_get_dir_info_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_nvm_get_dir_info_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_nvm_get_dir_info_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  entries: int
  entry_length: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_get_dir_info_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('entries', ctypes.c_uint32, 8), ('entry_length', ctypes.c_uint32, 12), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 16), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_nvm_write_input(c.Struct):
  SIZE = 56
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_src_addr: int
  dir_type: int
  dir_ordinal: int
  dir_ext: int
  dir_attr: int
  dir_data_length: int
  option: int
  flags: int
  dir_item_length: int
  offset: int
  len: int
  unused_0: int
struct_hwrm_nvm_write_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_src_addr', ctypes.c_uint64, 16), ('dir_type', ctypes.c_uint16, 24), ('dir_ordinal', ctypes.c_uint16, 26), ('dir_ext', ctypes.c_uint16, 28), ('dir_attr', ctypes.c_uint16, 30), ('dir_data_length', ctypes.c_uint32, 32), ('option', ctypes.c_uint16, 36), ('flags', ctypes.c_uint16, 38), ('dir_item_length', ctypes.c_uint32, 40), ('offset', ctypes.c_uint32, 44), ('len', ctypes.c_uint32, 48), ('unused_0', ctypes.c_uint32, 52)])
@c.record
class struct_hwrm_nvm_write_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  dir_item_length: int
  dir_idx: int
  unused_0: int
  valid: int
struct_hwrm_nvm_write_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('dir_item_length', ctypes.c_uint32, 8), ('dir_idx', ctypes.c_uint16, 12), ('unused_0', ctypes.c_ubyte, 14), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_write_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_nvm_write_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_nvm_modify_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  host_src_addr: int
  dir_idx: int
  flags: int
  offset: int
  len: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[4]]
struct_hwrm_nvm_modify_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('host_src_addr', ctypes.c_uint64, 16), ('dir_idx', ctypes.c_uint16, 24), ('flags', ctypes.c_uint16, 26), ('offset', ctypes.c_uint32, 28), ('len', ctypes.c_uint32, 32), ('unused_1', c.Array[ctypes.c_ubyte, Literal[4]], 36)])
@c.record
class struct_hwrm_nvm_modify_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_modify_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_find_dir_entry_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  dir_idx: int
  dir_type: int
  dir_ordinal: int
  dir_ext: int
  opt_ordinal: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
struct_hwrm_nvm_find_dir_entry_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('dir_idx', ctypes.c_uint16, 20), ('dir_type', ctypes.c_uint16, 22), ('dir_ordinal', ctypes.c_uint16, 24), ('dir_ext', ctypes.c_uint16, 26), ('opt_ordinal', ctypes.c_ubyte, 28), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 29)])
@c.record
class struct_hwrm_nvm_find_dir_entry_output(c.Struct):
  SIZE = 32
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  dir_item_length: int
  dir_data_length: int
  fw_ver: int
  dir_ordinal: int
  dir_idx: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_find_dir_entry_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('dir_item_length', ctypes.c_uint32, 8), ('dir_data_length', ctypes.c_uint32, 12), ('fw_ver', ctypes.c_uint32, 16), ('dir_ordinal', ctypes.c_uint16, 20), ('dir_idx', ctypes.c_uint16, 22), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 24), ('valid', ctypes.c_ubyte, 31)])
@c.record
class struct_hwrm_nvm_erase_dir_entry_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  dir_idx: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_hwrm_nvm_erase_dir_entry_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('dir_idx', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_hwrm_nvm_erase_dir_entry_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_erase_dir_entry_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_get_dev_info_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_nvm_get_dev_info_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 17)])
@c.record
class struct_hwrm_nvm_get_dev_info_output(c.Struct):
  SIZE = 96
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  manufacturer_id: int
  device_id: int
  sector_size: int
  nvram_size: int
  reserved_size: int
  available_size: int
  nvm_cfg_ver_maj: int
  nvm_cfg_ver_min: int
  nvm_cfg_ver_upd: int
  flags: int
  pkg_name: c.Array[ctypes.c_char, Literal[16]]
  hwrm_fw_major: int
  hwrm_fw_minor: int
  hwrm_fw_build: int
  hwrm_fw_patch: int
  mgmt_fw_major: int
  mgmt_fw_minor: int
  mgmt_fw_build: int
  mgmt_fw_patch: int
  roce_fw_major: int
  roce_fw_minor: int
  roce_fw_build: int
  roce_fw_patch: int
  netctrl_fw_major: int
  netctrl_fw_minor: int
  netctrl_fw_build: int
  netctrl_fw_patch: int
  srt2_fw_major: int
  srt2_fw_minor: int
  srt2_fw_build: int
  srt2_fw_patch: int
  security_soc_fw_major: int
  security_soc_fw_minor: int
  security_soc_fw_build: int
  security_soc_fw_patch: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
  valid: int
struct_hwrm_nvm_get_dev_info_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('manufacturer_id', ctypes.c_uint16, 8), ('device_id', ctypes.c_uint16, 10), ('sector_size', ctypes.c_uint32, 12), ('nvram_size', ctypes.c_uint32, 16), ('reserved_size', ctypes.c_uint32, 20), ('available_size', ctypes.c_uint32, 24), ('nvm_cfg_ver_maj', ctypes.c_ubyte, 28), ('nvm_cfg_ver_min', ctypes.c_ubyte, 29), ('nvm_cfg_ver_upd', ctypes.c_ubyte, 30), ('flags', ctypes.c_ubyte, 31), ('pkg_name', c.Array[ctypes.c_char, Literal[16]], 32), ('hwrm_fw_major', ctypes.c_uint16, 48), ('hwrm_fw_minor', ctypes.c_uint16, 50), ('hwrm_fw_build', ctypes.c_uint16, 52), ('hwrm_fw_patch', ctypes.c_uint16, 54), ('mgmt_fw_major', ctypes.c_uint16, 56), ('mgmt_fw_minor', ctypes.c_uint16, 58), ('mgmt_fw_build', ctypes.c_uint16, 60), ('mgmt_fw_patch', ctypes.c_uint16, 62), ('roce_fw_major', ctypes.c_uint16, 64), ('roce_fw_minor', ctypes.c_uint16, 66), ('roce_fw_build', ctypes.c_uint16, 68), ('roce_fw_patch', ctypes.c_uint16, 70), ('netctrl_fw_major', ctypes.c_uint16, 72), ('netctrl_fw_minor', ctypes.c_uint16, 74), ('netctrl_fw_build', ctypes.c_uint16, 76), ('netctrl_fw_patch', ctypes.c_uint16, 78), ('srt2_fw_major', ctypes.c_uint16, 80), ('srt2_fw_minor', ctypes.c_uint16, 82), ('srt2_fw_build', ctypes.c_uint16, 84), ('srt2_fw_patch', ctypes.c_uint16, 86), ('security_soc_fw_major', ctypes.c_ubyte, 88), ('security_soc_fw_minor', ctypes.c_ubyte, 89), ('security_soc_fw_build', ctypes.c_ubyte, 90), ('security_soc_fw_patch', ctypes.c_ubyte, 91), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 92), ('valid', ctypes.c_ubyte, 95)])
@c.record
class struct_hwrm_nvm_mod_dir_entry_input(c.Struct):
  SIZE = 32
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  enables: int
  dir_idx: int
  dir_ordinal: int
  dir_ext: int
  dir_attr: int
  checksum: int
struct_hwrm_nvm_mod_dir_entry_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('enables', ctypes.c_uint32, 16), ('dir_idx', ctypes.c_uint16, 20), ('dir_ordinal', ctypes.c_uint16, 22), ('dir_ext', ctypes.c_uint16, 24), ('dir_attr', ctypes.c_uint16, 26), ('checksum', ctypes.c_uint32, 28)])
@c.record
class struct_hwrm_nvm_mod_dir_entry_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_mod_dir_entry_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_verify_update_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  dir_type: int
  dir_ordinal: int
  dir_ext: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
struct_hwrm_nvm_verify_update_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('dir_type', ctypes.c_uint16, 16), ('dir_ordinal', ctypes.c_uint16, 18), ('dir_ext', ctypes.c_uint16, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 22)])
@c.record
class struct_hwrm_nvm_verify_update_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_verify_update_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_install_update_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  install_type: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
struct_hwrm_nvm_install_update_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('install_type', ctypes.c_uint32, 16), ('flags', ctypes.c_uint16, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 22)])
@c.record
class struct_hwrm_nvm_install_update_output(c.Struct):
  SIZE = 24
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  installed_items: int
  result: int
  problem_item: int
  reset_required: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[4]]
  valid: int
struct_hwrm_nvm_install_update_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('installed_items', ctypes.c_uint64, 8), ('result', ctypes.c_ubyte, 16), ('problem_item', ctypes.c_ubyte, 17), ('reset_required', ctypes.c_ubyte, 18), ('unused_0', c.Array[ctypes.c_ubyte, Literal[4]], 19), ('valid', ctypes.c_ubyte, 23)])
@c.record
class struct_hwrm_nvm_install_update_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_nvm_install_update_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_nvm_get_variable_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  dest_data_addr: int
  data_len: int
  option_num: int
  dimensions: int
  index_0: int
  index_1: int
  index_2: int
  index_3: int
  flags: int
  unused_0: int
struct_hwrm_nvm_get_variable_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('dest_data_addr', ctypes.c_uint64, 16), ('data_len', ctypes.c_uint16, 24), ('option_num', ctypes.c_uint16, 26), ('dimensions', ctypes.c_uint16, 28), ('index_0', ctypes.c_uint16, 30), ('index_1', ctypes.c_uint16, 32), ('index_2', ctypes.c_uint16, 34), ('index_3', ctypes.c_uint16, 36), ('flags', ctypes.c_ubyte, 38), ('unused_0', ctypes.c_ubyte, 39)])
@c.record
class struct_hwrm_nvm_get_variable_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  data_len: int
  option_num: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[2]]
  valid: int
struct_hwrm_nvm_get_variable_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('data_len', ctypes.c_uint16, 8), ('option_num', ctypes.c_uint16, 10), ('flags', ctypes.c_ubyte, 12), ('unused_0', c.Array[ctypes.c_ubyte, Literal[2]], 13), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_get_variable_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_nvm_get_variable_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_nvm_set_variable_input(c.Struct):
  SIZE = 40
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  src_data_addr: int
  data_len: int
  option_num: int
  dimensions: int
  index_0: int
  index_1: int
  index_2: int
  index_3: int
  flags: int
  unused_0: int
struct_hwrm_nvm_set_variable_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('src_data_addr', ctypes.c_uint64, 16), ('data_len', ctypes.c_uint16, 24), ('option_num', ctypes.c_uint16, 26), ('dimensions', ctypes.c_uint16, 28), ('index_0', ctypes.c_uint16, 30), ('index_1', ctypes.c_uint16, 32), ('index_2', ctypes.c_uint16, 34), ('index_3', ctypes.c_uint16, 36), ('flags', ctypes.c_ubyte, 38), ('unused_0', ctypes.c_ubyte, 39)])
@c.record
class struct_hwrm_nvm_set_variable_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_nvm_set_variable_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_nvm_set_variable_cmd_err(c.Struct):
  SIZE = 8
  code: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_nvm_set_variable_cmd_err.register_fields([('code', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_hwrm_selftest_qlist_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_selftest_qlist_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_selftest_qlist_output(c.Struct):
  SIZE = 280
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  num_tests: int
  available_tests: int
  offline_tests: int
  unused_0: int
  test_timeout: int
  unused_1: c.Array[ctypes.c_ubyte, Literal[2]]
  test_name: c.Array[c.Array[ctypes.c_char, Literal[32]], Literal[8]]
  eyescope_target_BER_support: int
  unused_2: c.Array[ctypes.c_ubyte, Literal[6]]
  valid: int
struct_hwrm_selftest_qlist_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('num_tests', ctypes.c_ubyte, 8), ('available_tests', ctypes.c_ubyte, 9), ('offline_tests', ctypes.c_ubyte, 10), ('unused_0', ctypes.c_ubyte, 11), ('test_timeout', ctypes.c_uint16, 12), ('unused_1', c.Array[ctypes.c_ubyte, Literal[2]], 14), ('test_name', c.Array[c.Array[ctypes.c_char, Literal[32]], Literal[8]], 16), ('eyescope_target_BER_support', ctypes.c_ubyte, 272), ('unused_2', c.Array[ctypes.c_ubyte, Literal[6]], 273), ('valid', ctypes.c_ubyte, 279)])
@c.record
class struct_hwrm_selftest_exec_input(c.Struct):
  SIZE = 24
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
  flags: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_hwrm_selftest_exec_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8), ('flags', ctypes.c_ubyte, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 17)])
@c.record
class struct_hwrm_selftest_exec_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  requested_tests: int
  test_success: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[5]]
  valid: int
struct_hwrm_selftest_exec_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('requested_tests', ctypes.c_ubyte, 8), ('test_success', ctypes.c_ubyte, 9), ('unused_0', c.Array[ctypes.c_ubyte, Literal[5]], 10), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_hwrm_selftest_irq_input(c.Struct):
  SIZE = 16
  req_type: int
  cmpl_ring: int
  seq_id: int
  target_id: int
  resp_addr: int
struct_hwrm_selftest_irq_input.register_fields([('req_type', ctypes.c_uint16, 0), ('cmpl_ring', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('target_id', ctypes.c_uint16, 6), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_hwrm_selftest_irq_output(c.Struct):
  SIZE = 16
  error_code: int
  req_type: int
  seq_id: int
  resp_len: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
  valid: int
struct_hwrm_selftest_irq_output.register_fields([('error_code', ctypes.c_uint16, 0), ('req_type', ctypes.c_uint16, 2), ('seq_id', ctypes.c_uint16, 4), ('resp_len', ctypes.c_uint16, 6), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 8), ('valid', ctypes.c_ubyte, 15)])
@c.record
class struct_dbc_dbc(c.Struct):
  SIZE = 8
  index: int
  type_path_xid: int
struct_dbc_dbc.register_fields([('index', ctypes.c_uint32, 0), ('type_path_xid', ctypes.c_uint32, 4)])
@c.record
class struct_db_push_start(c.Struct):
  SIZE = 8
  db: int
struct_db_push_start.register_fields([('db', ctypes.c_uint64, 0)])
@c.record
class struct_db_push_end(c.Struct):
  SIZE = 8
  db: int
struct_db_push_end.register_fields([('db', ctypes.c_uint64, 0)])
@c.record
class struct_db_push_info(c.Struct):
  SIZE = 8
  push_size_push_index: int
  reserved32: int
struct_db_push_info.register_fields([('push_size_push_index', ctypes.c_uint32, 0), ('reserved32', ctypes.c_uint32, 4)])
@c.record
class struct_fw_status_reg(c.Struct):
  SIZE = 4
  fw_status: int
struct_fw_status_reg.register_fields([('fw_status', ctypes.c_uint32, 0)])
@c.record
class struct_hcomm_status(c.Struct):
  SIZE = 8
  sig_ver: int
  fw_status_loc: int
struct_hcomm_status.register_fields([('sig_ver', ctypes.c_uint32, 0), ('fw_status_loc', ctypes.c_uint32, 4)])
@c.record
class struct_tx_doorbell(c.Struct):
  SIZE = 4
  key_idx: int
struct_tx_doorbell.register_fields([('key_idx', ctypes.c_uint32, 0)])
@c.record
class struct_rx_doorbell(c.Struct):
  SIZE = 4
  key_idx: int
struct_rx_doorbell.register_fields([('key_idx', ctypes.c_uint32, 0)])
@c.record
class struct_cmpl_doorbell(c.Struct):
  SIZE = 4
  key_mask_valid_idx: int
struct_cmpl_doorbell.register_fields([('key_mask_valid_idx', ctypes.c_uint32, 0)])
@c.record
class struct_status_doorbell(c.Struct):
  SIZE = 4
  key_idx: int
struct_status_doorbell.register_fields([('key_idx', ctypes.c_uint32, 0)])
@c.record
class struct_cmdq_init(c.Struct):
  SIZE = 16
  cmdq_pbl: int
  cmdq_size_cmdq_lvl: int
  creq_ring_id: int
  prod_idx: int
struct_cmdq_init.register_fields([('cmdq_pbl', ctypes.c_uint64, 0), ('cmdq_size_cmdq_lvl', ctypes.c_uint16, 8), ('creq_ring_id', ctypes.c_uint16, 10), ('prod_idx', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_base(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_base.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_base(c.Struct):
  SIZE = 16
  type: int
  reserved56: c.Array[ctypes.c_ubyte, Literal[7]]
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_base.register_fields([('type', ctypes.c_ubyte, 0), ('reserved56', c.Array[ctypes.c_ubyte, Literal[7]], 1), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_query_version(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_query_version.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_query_version_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  fw_maj: int
  fw_minor: int
  fw_bld: int
  fw_rsvd: int
  v: int
  event: int
  reserved16: int
  intf_maj: int
  intf_minor: int
  intf_bld: int
  intf_rsvd: int
struct_creq_query_version_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('fw_maj', ctypes.c_ubyte, 4), ('fw_minor', ctypes.c_ubyte, 5), ('fw_bld', ctypes.c_ubyte, 6), ('fw_rsvd', ctypes.c_ubyte, 7), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved16', ctypes.c_uint16, 10), ('intf_maj', ctypes.c_ubyte, 12), ('intf_minor', ctypes.c_ubyte, 13), ('intf_bld', ctypes.c_ubyte, 14), ('intf_rsvd', ctypes.c_ubyte, 15)])
@c.record
class struct_cmdq_initialize_fw(c.Struct):
  SIZE = 112
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qpc_pg_size_qpc_lvl: int
  mrw_pg_size_mrw_lvl: int
  srq_pg_size_srq_lvl: int
  cq_pg_size_cq_lvl: int
  tqm_pg_size_tqm_lvl: int
  tim_pg_size_tim_lvl: int
  log2_dbr_pg_size: int
  qpc_page_dir: int
  mrw_page_dir: int
  srq_page_dir: int
  cq_page_dir: int
  tqm_page_dir: int
  tim_page_dir: int
  number_of_qp: int
  number_of_mrw: int
  number_of_srq: int
  number_of_cq: int
  max_qp_per_vf: int
  max_mrw_per_vf: int
  max_srq_per_vf: int
  max_cq_per_vf: int
  max_gid_per_vf: int
  stat_ctx_id: int
struct_cmdq_initialize_fw.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qpc_pg_size_qpc_lvl', ctypes.c_ubyte, 16), ('mrw_pg_size_mrw_lvl', ctypes.c_ubyte, 17), ('srq_pg_size_srq_lvl', ctypes.c_ubyte, 18), ('cq_pg_size_cq_lvl', ctypes.c_ubyte, 19), ('tqm_pg_size_tqm_lvl', ctypes.c_ubyte, 20), ('tim_pg_size_tim_lvl', ctypes.c_ubyte, 21), ('log2_dbr_pg_size', ctypes.c_uint16, 22), ('qpc_page_dir', ctypes.c_uint64, 24), ('mrw_page_dir', ctypes.c_uint64, 32), ('srq_page_dir', ctypes.c_uint64, 40), ('cq_page_dir', ctypes.c_uint64, 48), ('tqm_page_dir', ctypes.c_uint64, 56), ('tim_page_dir', ctypes.c_uint64, 64), ('number_of_qp', ctypes.c_uint32, 72), ('number_of_mrw', ctypes.c_uint32, 76), ('number_of_srq', ctypes.c_uint32, 80), ('number_of_cq', ctypes.c_uint32, 84), ('max_qp_per_vf', ctypes.c_uint32, 88), ('max_mrw_per_vf', ctypes.c_uint32, 92), ('max_srq_per_vf', ctypes.c_uint32, 96), ('max_cq_per_vf', ctypes.c_uint32, 100), ('max_gid_per_vf', ctypes.c_uint32, 104), ('stat_ctx_id', ctypes.c_uint32, 108)])
@c.record
class struct_creq_initialize_fw_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_initialize_fw_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_deinitialize_fw(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_deinitialize_fw.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_deinitialize_fw_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_deinitialize_fw_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_create_qp(c.Struct):
  SIZE = 104
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qp_handle: int
  qp_flags: int
  type: int
  sq_pg_size_sq_lvl: int
  rq_pg_size_rq_lvl: int
  unused_0: int
  dpi: int
  sq_size: int
  rq_size: int
  sq_fwo_sq_sge: int
  rq_fwo_rq_sge: int
  scq_cid: int
  rcq_cid: int
  srq_cid: int
  pd_id: int
  sq_pbl: int
  rq_pbl: int
  irrq_addr: int
  orrq_addr: int
  request_xid: int
  steering_tag: int
  reserved16: int
struct_cmdq_create_qp.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qp_handle', ctypes.c_uint64, 16), ('qp_flags', ctypes.c_uint32, 24), ('type', ctypes.c_ubyte, 28), ('sq_pg_size_sq_lvl', ctypes.c_ubyte, 29), ('rq_pg_size_rq_lvl', ctypes.c_ubyte, 30), ('unused_0', ctypes.c_ubyte, 31), ('dpi', ctypes.c_uint32, 32), ('sq_size', ctypes.c_uint32, 36), ('rq_size', ctypes.c_uint32, 40), ('sq_fwo_sq_sge', ctypes.c_uint16, 44), ('rq_fwo_rq_sge', ctypes.c_uint16, 46), ('scq_cid', ctypes.c_uint32, 48), ('rcq_cid', ctypes.c_uint32, 52), ('srq_cid', ctypes.c_uint32, 56), ('pd_id', ctypes.c_uint32, 60), ('sq_pbl', ctypes.c_uint64, 64), ('rq_pbl', ctypes.c_uint64, 72), ('irrq_addr', ctypes.c_uint64, 80), ('orrq_addr', ctypes.c_uint64, 88), ('request_xid', ctypes.c_uint32, 96), ('steering_tag', ctypes.c_uint16, 100), ('reserved16', ctypes.c_uint16, 102)])
@c.record
class struct_creq_create_qp_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  optimized_transmit_enabled: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[5]]
struct_creq_create_qp_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('optimized_transmit_enabled', ctypes.c_ubyte, 10), ('reserved48', c.Array[ctypes.c_ubyte, Literal[5]], 11)])
@c.record
class struct_cmdq_destroy_qp(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qp_cid: int
  unused_0: int
struct_cmdq_destroy_qp.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qp_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_destroy_qp_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_destroy_qp_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_modify_qp(c.Struct):
  SIZE = 144
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  qp_type: int
  resp_addr: int
  modify_mask: int
  qp_cid: int
  network_type_en_sqd_async_notify_new_state: int
  access: int
  pkey: int
  qkey: int
  dgid: c.Array[ctypes.c_uint32, Literal[4]]
  flow_label: int
  sgid_index: int
  hop_limit: int
  traffic_class: int
  dest_mac: c.Array[ctypes.c_uint16, Literal[3]]
  tos_dscp_tos_ecn: int
  path_mtu_pingpong_push_enable: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  min_rnr_timer: int
  rq_psn: int
  sq_psn: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  enable_cc: int
  sq_size: int
  rq_size: int
  sq_sge: int
  rq_sge: int
  max_inline_data: int
  dest_qp_id: int
  pingpong_push_dpi: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan_pcp_vlan_dei_vlan_id: int
  irrq_addr: int
  orrq_addr: int
  ext_modify_mask: int
  ext_stats_ctx_id: int
  schq_id: int
  unused_0: int
  reserved32: int
struct_cmdq_modify_qp.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('qp_type', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('modify_mask', ctypes.c_uint32, 16), ('qp_cid', ctypes.c_uint32, 20), ('network_type_en_sqd_async_notify_new_state', ctypes.c_ubyte, 24), ('access', ctypes.c_ubyte, 25), ('pkey', ctypes.c_uint16, 26), ('qkey', ctypes.c_uint32, 28), ('dgid', c.Array[ctypes.c_uint32, Literal[4]], 32), ('flow_label', ctypes.c_uint32, 48), ('sgid_index', ctypes.c_uint16, 52), ('hop_limit', ctypes.c_ubyte, 54), ('traffic_class', ctypes.c_ubyte, 55), ('dest_mac', c.Array[ctypes.c_uint16, Literal[3]], 56), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 62), ('path_mtu_pingpong_push_enable', ctypes.c_ubyte, 63), ('timeout', ctypes.c_ubyte, 64), ('retry_cnt', ctypes.c_ubyte, 65), ('rnr_retry', ctypes.c_ubyte, 66), ('min_rnr_timer', ctypes.c_ubyte, 67), ('rq_psn', ctypes.c_uint32, 68), ('sq_psn', ctypes.c_uint32, 72), ('max_rd_atomic', ctypes.c_ubyte, 76), ('max_dest_rd_atomic', ctypes.c_ubyte, 77), ('enable_cc', ctypes.c_uint16, 78), ('sq_size', ctypes.c_uint32, 80), ('rq_size', ctypes.c_uint32, 84), ('sq_sge', ctypes.c_uint16, 88), ('rq_sge', ctypes.c_uint16, 90), ('max_inline_data', ctypes.c_uint32, 92), ('dest_qp_id', ctypes.c_uint32, 96), ('pingpong_push_dpi', ctypes.c_uint32, 100), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 104), ('vlan_pcp_vlan_dei_vlan_id', ctypes.c_uint16, 110), ('irrq_addr', ctypes.c_uint64, 112), ('orrq_addr', ctypes.c_uint64, 120), ('ext_modify_mask', ctypes.c_uint32, 128), ('ext_stats_ctx_id', ctypes.c_uint32, 132), ('schq_id', ctypes.c_uint16, 136), ('unused_0', ctypes.c_uint16, 138), ('reserved32', ctypes.c_uint32, 140)])
@c.record
class struct_creq_modify_qp_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  pingpong_push_state_index_enabled: int
  reserved8: int
  lag_src_mac: int
struct_creq_modify_qp_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('pingpong_push_state_index_enabled', ctypes.c_ubyte, 10), ('reserved8', ctypes.c_ubyte, 11), ('lag_src_mac', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_query_qp(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qp_cid: int
  unused_0: int
struct_cmdq_query_qp.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qp_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_query_qp_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_qp_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_qp_resp_sb(c.Struct):
  SIZE = 104
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  xid: int
  en_sqd_async_notify_state: int
  access: int
  pkey: int
  qkey: int
  udp_src_port: int
  reserved16: int
  dgid: c.Array[ctypes.c_uint32, Literal[4]]
  flow_label: int
  sgid_index: int
  hop_limit: int
  traffic_class: int
  dest_mac: c.Array[ctypes.c_uint16, Literal[3]]
  path_mtu_dest_vlan_id: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  min_rnr_timer: int
  rq_psn: int
  sq_psn: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  tos_dscp_tos_ecn: int
  enable_cc: int
  sq_size: int
  rq_size: int
  sq_sge: int
  rq_sge: int
  max_inline_data: int
  dest_qp_id: int
  port_id: int
  unused_0: int
  stat_collection_id: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan_pcp_vlan_dei_vlan_id: int
struct_creq_query_qp_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('xid', ctypes.c_uint32, 8), ('en_sqd_async_notify_state', ctypes.c_ubyte, 12), ('access', ctypes.c_ubyte, 13), ('pkey', ctypes.c_uint16, 14), ('qkey', ctypes.c_uint32, 16), ('udp_src_port', ctypes.c_uint16, 20), ('reserved16', ctypes.c_uint16, 22), ('dgid', c.Array[ctypes.c_uint32, Literal[4]], 24), ('flow_label', ctypes.c_uint32, 40), ('sgid_index', ctypes.c_uint16, 44), ('hop_limit', ctypes.c_ubyte, 46), ('traffic_class', ctypes.c_ubyte, 47), ('dest_mac', c.Array[ctypes.c_uint16, Literal[3]], 48), ('path_mtu_dest_vlan_id', ctypes.c_uint16, 54), ('timeout', ctypes.c_ubyte, 56), ('retry_cnt', ctypes.c_ubyte, 57), ('rnr_retry', ctypes.c_ubyte, 58), ('min_rnr_timer', ctypes.c_ubyte, 59), ('rq_psn', ctypes.c_uint32, 60), ('sq_psn', ctypes.c_uint32, 64), ('max_rd_atomic', ctypes.c_ubyte, 68), ('max_dest_rd_atomic', ctypes.c_ubyte, 69), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 70), ('enable_cc', ctypes.c_ubyte, 71), ('sq_size', ctypes.c_uint32, 72), ('rq_size', ctypes.c_uint32, 76), ('sq_sge', ctypes.c_uint16, 80), ('rq_sge', ctypes.c_uint16, 82), ('max_inline_data', ctypes.c_uint32, 84), ('dest_qp_id', ctypes.c_uint32, 88), ('port_id', ctypes.c_uint16, 92), ('unused_0', ctypes.c_ubyte, 94), ('stat_collection_id', ctypes.c_ubyte, 95), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 96), ('vlan_pcp_vlan_dei_vlan_id', ctypes.c_uint16, 102)])
@c.record
class struct_cmdq_query_qp_extend(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  num_qps: int
  resp_addr: int
  function_id: int
  current_index: int
struct_cmdq_query_qp_extend.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('num_qps', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('function_id', ctypes.c_uint32, 16), ('current_index', ctypes.c_uint32, 20)])
@c.record
class struct_creq_query_qp_extend_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved16: int
  current_index: int
struct_creq_query_qp_extend_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved16', ctypes.c_uint16, 10), ('current_index', ctypes.c_uint32, 12)])
@c.record
class struct_creq_query_qp_extend_resp_sb(c.Struct):
  SIZE = 48
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  xid: int
  state: int
  reserved_8: int
  port_id: int
  qkey: int
  sgid_index: int
  network_type: int
  unused_0: int
  dgid: c.Array[ctypes.c_uint32, Literal[4]]
  dest_qp_id: int
  stat_collection_id: int
  reservred_8: int
  reserved_16: int
struct_creq_query_qp_extend_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('xid', ctypes.c_uint32, 8), ('state', ctypes.c_ubyte, 12), ('reserved_8', ctypes.c_ubyte, 13), ('port_id', ctypes.c_uint16, 14), ('qkey', ctypes.c_uint32, 16), ('sgid_index', ctypes.c_uint16, 20), ('network_type', ctypes.c_ubyte, 22), ('unused_0', ctypes.c_ubyte, 23), ('dgid', c.Array[ctypes.c_uint32, Literal[4]], 24), ('dest_qp_id', ctypes.c_uint32, 40), ('stat_collection_id', ctypes.c_ubyte, 44), ('reservred_8', ctypes.c_ubyte, 45), ('reserved_16', ctypes.c_uint16, 46)])
@c.record
class struct_creq_query_qp_extend_resp_sb_tlv(c.Struct):
  SIZE = 64
  cmd_discr: int
  reserved_8b: int
  tlv_flags: int
  tlv_type: int
  length: int
  total_size: int
  reserved56: c.Array[ctypes.c_ubyte, Literal[7]]
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  xid: int
  state: int
  reserved_8: int
  port_id: int
  qkey: int
  sgid_index: int
  network_type: int
  unused_0: int
  dgid: c.Array[ctypes.c_uint32, Literal[4]]
  dest_qp_id: int
  stat_collection_id: int
  reservred_8: int
  reserved_16: int
struct_creq_query_qp_extend_resp_sb_tlv.register_fields([('cmd_discr', ctypes.c_uint16, 0), ('reserved_8b', ctypes.c_ubyte, 2), ('tlv_flags', ctypes.c_ubyte, 3), ('tlv_type', ctypes.c_uint16, 4), ('length', ctypes.c_uint16, 6), ('total_size', ctypes.c_ubyte, 8), ('reserved56', c.Array[ctypes.c_ubyte, Literal[7]], 9), ('opcode', ctypes.c_ubyte, 16), ('status', ctypes.c_ubyte, 17), ('cookie', ctypes.c_uint16, 18), ('flags', ctypes.c_uint16, 20), ('resp_size', ctypes.c_ubyte, 22), ('reserved8', ctypes.c_ubyte, 23), ('xid', ctypes.c_uint32, 24), ('state', ctypes.c_ubyte, 28), ('reserved_8', ctypes.c_ubyte, 29), ('port_id', ctypes.c_uint16, 30), ('qkey', ctypes.c_uint32, 32), ('sgid_index', ctypes.c_uint16, 36), ('network_type', ctypes.c_ubyte, 38), ('unused_0', ctypes.c_ubyte, 39), ('dgid', c.Array[ctypes.c_uint32, Literal[4]], 40), ('dest_qp_id', ctypes.c_uint32, 56), ('stat_collection_id', ctypes.c_ubyte, 60), ('reservred_8', ctypes.c_ubyte, 61), ('reserved_16', ctypes.c_uint16, 62)])
@c.record
class struct_cmdq_create_srq(c.Struct):
  SIZE = 56
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  srq_handle: int
  pg_size_lvl: int
  eventq_id: int
  srq_size: int
  srq_fwo: int
  dpi: int
  pd_id: int
  pbl: int
  steering_tag: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_cmdq_create_srq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('srq_handle', ctypes.c_uint64, 16), ('pg_size_lvl', ctypes.c_uint16, 24), ('eventq_id', ctypes.c_uint16, 26), ('srq_size', ctypes.c_uint16, 28), ('srq_fwo', ctypes.c_uint16, 30), ('dpi', ctypes.c_uint32, 32), ('pd_id', ctypes.c_uint32, 36), ('pbl', ctypes.c_uint64, 40), ('steering_tag', ctypes.c_uint16, 48), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 50)])
@c.record
class struct_creq_create_srq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_create_srq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_destroy_srq(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  srq_cid: int
  unused_0: int
struct_cmdq_destroy_srq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('srq_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_destroy_srq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  enable_for_arm: c.Array[ctypes.c_uint16, Literal[3]]
struct_creq_destroy_srq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('enable_for_arm', c.Array[ctypes.c_uint16, Literal[3]], 10)])
@c.record
class struct_cmdq_query_srq(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  srq_cid: int
  unused_0: int
struct_cmdq_query_srq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('srq_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_query_srq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_srq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_srq_resp_sb(c.Struct):
  SIZE = 32
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  xid: int
  srq_limit: int
  reserved16: int
  data: c.Array[ctypes.c_uint32, Literal[4]]
struct_creq_query_srq_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('xid', ctypes.c_uint32, 8), ('srq_limit', ctypes.c_uint16, 12), ('reserved16', ctypes.c_uint16, 14), ('data', c.Array[ctypes.c_uint32, Literal[4]], 16)])
@c.record
class struct_cmdq_create_cq(c.Struct):
  SIZE = 64
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  cq_handle: int
  pg_size_lvl: int
  cq_fco_cnq_id: int
  dpi: int
  cq_size: int
  pbl: int
  steering_tag: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[2]]
  coalescing: int
  reserved64: int
struct_cmdq_create_cq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('cq_handle', ctypes.c_uint64, 16), ('pg_size_lvl', ctypes.c_uint32, 24), ('cq_fco_cnq_id', ctypes.c_uint32, 28), ('dpi', ctypes.c_uint32, 32), ('cq_size', ctypes.c_uint32, 36), ('pbl', ctypes.c_uint64, 40), ('steering_tag', ctypes.c_uint16, 48), ('reserved48', c.Array[ctypes.c_ubyte, Literal[2]], 50), ('coalescing', ctypes.c_uint32, 52), ('reserved64', ctypes.c_uint64, 56)])
@c.record
class struct_creq_create_cq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_create_cq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_destroy_cq(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  cq_cid: int
  unused_0: int
struct_cmdq_destroy_cq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('cq_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_destroy_cq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  cq_arm_lvl: int
  total_cnq_events: int
  reserved16: int
struct_creq_destroy_cq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('cq_arm_lvl', ctypes.c_uint16, 10), ('total_cnq_events', ctypes.c_uint16, 12), ('reserved16', ctypes.c_uint16, 14)])
@c.record
class struct_cmdq_resize_cq(c.Struct):
  SIZE = 40
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  cq_cid: int
  new_cq_size_pg_size_lvl: int
  new_pbl: int
  new_cq_fco: int
  unused_0: int
struct_cmdq_resize_cq.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('cq_cid', ctypes.c_uint32, 16), ('new_cq_size_pg_size_lvl', ctypes.c_uint32, 20), ('new_pbl', ctypes.c_uint64, 24), ('new_cq_fco', ctypes.c_uint32, 32), ('unused_0', ctypes.c_uint32, 36)])
@c.record
class struct_creq_resize_cq_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_resize_cq_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_allocate_mrw(c.Struct):
  SIZE = 32
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  mrw_handle: int
  mrw_flags: int
  access: int
  steering_tag: int
  pd_id: int
struct_cmdq_allocate_mrw.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('mrw_handle', ctypes.c_uint64, 16), ('mrw_flags', ctypes.c_ubyte, 24), ('access', ctypes.c_ubyte, 25), ('steering_tag', ctypes.c_uint16, 26), ('pd_id', ctypes.c_uint32, 28)])
@c.record
class struct_creq_allocate_mrw_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_allocate_mrw_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_deallocate_key(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  mrw_flags: int
  unused24: c.Array[ctypes.c_ubyte, Literal[3]]
  key: int
struct_cmdq_deallocate_key.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('mrw_flags', ctypes.c_ubyte, 16), ('unused24', c.Array[ctypes.c_ubyte, Literal[3]], 17), ('key', ctypes.c_uint32, 20)])
@c.record
class struct_creq_deallocate_key_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved16: int
  bound_window_info: int
struct_creq_deallocate_key_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved16', ctypes.c_uint16, 10), ('bound_window_info', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_register_mr(c.Struct):
  SIZE = 56
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  log2_pg_size_lvl: int
  access: int
  log2_pbl_pg_size: int
  key: int
  pbl: int
  va: int
  mr_size: int
  steering_tag: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_cmdq_register_mr.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('log2_pg_size_lvl', ctypes.c_ubyte, 16), ('access', ctypes.c_ubyte, 17), ('log2_pbl_pg_size', ctypes.c_uint16, 18), ('key', ctypes.c_uint32, 20), ('pbl', ctypes.c_uint64, 24), ('va', ctypes.c_uint64, 32), ('mr_size', ctypes.c_uint64, 40), ('steering_tag', ctypes.c_uint16, 48), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 50)])
@c.record
class struct_creq_register_mr_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_register_mr_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_deregister_mr(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  lkey: int
  unused_0: int
struct_cmdq_deregister_mr.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('lkey', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_deregister_mr_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved16: int
  bound_windows: int
struct_creq_deregister_mr_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved16', ctypes.c_uint16, 10), ('bound_windows', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_add_gid(c.Struct):
  SIZE = 48
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  gid: c.Array[ctypes.c_uint32, Literal[4]]
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan: int
  ipid: int
  stats_ctx: int
  unused_0: int
struct_cmdq_add_gid.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('gid', c.Array[ctypes.c_uint32, Literal[4]], 16), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 32), ('vlan', ctypes.c_uint16, 38), ('ipid', ctypes.c_uint16, 40), ('stats_ctx', ctypes.c_uint16, 42), ('unused_0', ctypes.c_uint32, 44)])
@c.record
class struct_creq_add_gid_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_add_gid_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_delete_gid(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  gid_index: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[6]]
struct_cmdq_delete_gid.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('gid_index', ctypes.c_uint16, 16), ('unused_0', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_creq_delete_gid_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_delete_gid_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_modify_gid(c.Struct):
  SIZE = 48
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  gid: c.Array[ctypes.c_uint32, Literal[4]]
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan: int
  ipid: int
  gid_index: int
  stats_ctx: int
  unused_0: int
struct_cmdq_modify_gid.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('gid', c.Array[ctypes.c_uint32, Literal[4]], 16), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 32), ('vlan', ctypes.c_uint16, 38), ('ipid', ctypes.c_uint16, 40), ('gid_index', ctypes.c_uint16, 42), ('stats_ctx', ctypes.c_uint16, 44), ('unused_0', ctypes.c_uint16, 46)])
@c.record
class struct_creq_modify_gid_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_modify_gid_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_query_gid(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  gid_index: int
  unused16: c.Array[ctypes.c_ubyte, Literal[6]]
struct_cmdq_query_gid.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('gid_index', ctypes.c_uint16, 16), ('unused16', c.Array[ctypes.c_ubyte, Literal[6]], 18)])
@c.record
class struct_creq_query_gid_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_gid_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_gid_resp_sb(c.Struct):
  SIZE = 40
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  gid: c.Array[ctypes.c_uint32, Literal[4]]
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  vlan: int
  ipid: int
  gid_index: int
  unused_0: int
struct_creq_query_gid_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('gid', c.Array[ctypes.c_uint32, Literal[4]], 8), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 24), ('vlan', ctypes.c_uint16, 30), ('ipid', ctypes.c_uint16, 32), ('gid_index', ctypes.c_uint16, 34), ('unused_0', ctypes.c_uint32, 36)])
@c.record
class struct_cmdq_create_qp1(c.Struct):
  SIZE = 80
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qp_handle: int
  qp_flags: int
  type: int
  sq_pg_size_sq_lvl: int
  rq_pg_size_rq_lvl: int
  unused_0: int
  dpi: int
  sq_size: int
  rq_size: int
  sq_fwo_sq_sge: int
  rq_fwo_rq_sge: int
  scq_cid: int
  rcq_cid: int
  srq_cid: int
  pd_id: int
  sq_pbl: int
  rq_pbl: int
struct_cmdq_create_qp1.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qp_handle', ctypes.c_uint64, 16), ('qp_flags', ctypes.c_uint32, 24), ('type', ctypes.c_ubyte, 28), ('sq_pg_size_sq_lvl', ctypes.c_ubyte, 29), ('rq_pg_size_rq_lvl', ctypes.c_ubyte, 30), ('unused_0', ctypes.c_ubyte, 31), ('dpi', ctypes.c_uint32, 32), ('sq_size', ctypes.c_uint32, 36), ('rq_size', ctypes.c_uint32, 40), ('sq_fwo_sq_sge', ctypes.c_uint16, 44), ('rq_fwo_rq_sge', ctypes.c_uint16, 46), ('scq_cid', ctypes.c_uint32, 48), ('rcq_cid', ctypes.c_uint32, 52), ('srq_cid', ctypes.c_uint32, 56), ('pd_id', ctypes.c_uint32, 60), ('sq_pbl', ctypes.c_uint64, 64), ('rq_pbl', ctypes.c_uint64, 72)])
@c.record
class struct_creq_create_qp1_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_create_qp1_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_destroy_qp1(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  qp1_cid: int
  unused_0: int
struct_cmdq_destroy_qp1.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('qp1_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_destroy_qp1_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_destroy_qp1_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_create_ah(c.Struct):
  SIZE = 64
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  ah_handle: int
  dgid: c.Array[ctypes.c_uint32, Literal[4]]
  type: int
  hop_limit: int
  sgid_index: int
  dest_vlan_id_flow_label: int
  pd_id: int
  unused_0: int
  dest_mac: c.Array[ctypes.c_uint16, Literal[3]]
  traffic_class: int
  enable_cc: int
struct_cmdq_create_ah.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('ah_handle', ctypes.c_uint64, 16), ('dgid', c.Array[ctypes.c_uint32, Literal[4]], 24), ('type', ctypes.c_ubyte, 40), ('hop_limit', ctypes.c_ubyte, 41), ('sgid_index', ctypes.c_uint16, 42), ('dest_vlan_id_flow_label', ctypes.c_uint32, 44), ('pd_id', ctypes.c_uint32, 48), ('unused_0', ctypes.c_uint32, 52), ('dest_mac', c.Array[ctypes.c_uint16, Literal[3]], 56), ('traffic_class', ctypes.c_ubyte, 62), ('enable_cc', ctypes.c_ubyte, 63)])
@c.record
class struct_creq_create_ah_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_create_ah_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_destroy_ah(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  ah_cid: int
  unused_0: int
struct_cmdq_destroy_ah.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('ah_cid', ctypes.c_uint32, 16), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_destroy_ah_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_destroy_ah_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_query_roce_stats(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  collection_id: int
  resp_addr: int
  function_id: int
  reserved32: int
struct_cmdq_query_roce_stats.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('collection_id', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('function_id', ctypes.c_uint32, 16), ('reserved32', ctypes.c_uint32, 20)])
@c.record
class struct_creq_query_roce_stats_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_roce_stats_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_roce_stats_resp_sb(c.Struct):
  SIZE = 368
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  rsvd: int
  num_counters: int
  rsvd1: int
  to_retransmits: int
  seq_err_naks_rcvd: int
  max_retry_exceeded: int
  rnr_naks_rcvd: int
  missing_resp: int
  unrecoverable_err: int
  bad_resp_err: int
  local_qp_op_err: int
  local_protection_err: int
  mem_mgmt_op_err: int
  remote_invalid_req_err: int
  remote_access_err: int
  remote_op_err: int
  dup_req: int
  res_exceed_max: int
  res_length_mismatch: int
  res_exceeds_wqe: int
  res_opcode_err: int
  res_rx_invalid_rkey: int
  res_rx_domain_err: int
  res_rx_no_perm: int
  res_rx_range_err: int
  res_tx_invalid_rkey: int
  res_tx_domain_err: int
  res_tx_no_perm: int
  res_tx_range_err: int
  res_irrq_oflow: int
  res_unsup_opcode: int
  res_unaligned_atomic: int
  res_rem_inv_err: int
  res_mem_error: int
  res_srq_err: int
  res_cmp_err: int
  res_invalid_dup_rkey: int
  res_wqe_format_err: int
  res_cq_load_err: int
  res_srq_load_err: int
  res_tx_pci_err: int
  res_rx_pci_err: int
  res_oos_drop_count: int
  active_qp_count_p0: int
  active_qp_count_p1: int
  active_qp_count_p2: int
  active_qp_count_p3: int
struct_creq_query_roce_stats_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('rsvd', ctypes.c_ubyte, 7), ('num_counters', ctypes.c_uint32, 8), ('rsvd1', ctypes.c_uint32, 12), ('to_retransmits', ctypes.c_uint64, 16), ('seq_err_naks_rcvd', ctypes.c_uint64, 24), ('max_retry_exceeded', ctypes.c_uint64, 32), ('rnr_naks_rcvd', ctypes.c_uint64, 40), ('missing_resp', ctypes.c_uint64, 48), ('unrecoverable_err', ctypes.c_uint64, 56), ('bad_resp_err', ctypes.c_uint64, 64), ('local_qp_op_err', ctypes.c_uint64, 72), ('local_protection_err', ctypes.c_uint64, 80), ('mem_mgmt_op_err', ctypes.c_uint64, 88), ('remote_invalid_req_err', ctypes.c_uint64, 96), ('remote_access_err', ctypes.c_uint64, 104), ('remote_op_err', ctypes.c_uint64, 112), ('dup_req', ctypes.c_uint64, 120), ('res_exceed_max', ctypes.c_uint64, 128), ('res_length_mismatch', ctypes.c_uint64, 136), ('res_exceeds_wqe', ctypes.c_uint64, 144), ('res_opcode_err', ctypes.c_uint64, 152), ('res_rx_invalid_rkey', ctypes.c_uint64, 160), ('res_rx_domain_err', ctypes.c_uint64, 168), ('res_rx_no_perm', ctypes.c_uint64, 176), ('res_rx_range_err', ctypes.c_uint64, 184), ('res_tx_invalid_rkey', ctypes.c_uint64, 192), ('res_tx_domain_err', ctypes.c_uint64, 200), ('res_tx_no_perm', ctypes.c_uint64, 208), ('res_tx_range_err', ctypes.c_uint64, 216), ('res_irrq_oflow', ctypes.c_uint64, 224), ('res_unsup_opcode', ctypes.c_uint64, 232), ('res_unaligned_atomic', ctypes.c_uint64, 240), ('res_rem_inv_err', ctypes.c_uint64, 248), ('res_mem_error', ctypes.c_uint64, 256), ('res_srq_err', ctypes.c_uint64, 264), ('res_cmp_err', ctypes.c_uint64, 272), ('res_invalid_dup_rkey', ctypes.c_uint64, 280), ('res_wqe_format_err', ctypes.c_uint64, 288), ('res_cq_load_err', ctypes.c_uint64, 296), ('res_srq_load_err', ctypes.c_uint64, 304), ('res_tx_pci_err', ctypes.c_uint64, 312), ('res_rx_pci_err', ctypes.c_uint64, 320), ('res_oos_drop_count', ctypes.c_uint64, 328), ('active_qp_count_p0', ctypes.c_uint64, 336), ('active_qp_count_p1', ctypes.c_uint64, 344), ('active_qp_count_p2', ctypes.c_uint64, 352), ('active_qp_count_p3', ctypes.c_uint64, 360)])
@c.record
class struct_cmdq_query_roce_stats_ext(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  collection_id: int
  resp_addr: int
  function_id: int
  reserved32: int
struct_cmdq_query_roce_stats_ext.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('collection_id', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('function_id', ctypes.c_uint32, 16), ('reserved32', ctypes.c_uint32, 20)])
@c.record
class struct_creq_query_roce_stats_ext_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_roce_stats_ext_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_roce_stats_ext_resp_sb(c.Struct):
  SIZE = 232
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  rsvd: int
  tx_atomic_req_pkts: int
  tx_read_req_pkts: int
  tx_read_res_pkts: int
  tx_write_req_pkts: int
  tx_send_req_pkts: int
  tx_roce_pkts: int
  tx_roce_bytes: int
  rx_atomic_req_pkts: int
  rx_read_req_pkts: int
  rx_read_res_pkts: int
  rx_write_req_pkts: int
  rx_send_req_pkts: int
  rx_roce_pkts: int
  rx_roce_bytes: int
  rx_roce_good_pkts: int
  rx_roce_good_bytes: int
  rx_out_of_buffer_pkts: int
  rx_out_of_sequence_pkts: int
  tx_cnp_pkts: int
  rx_cnp_pkts: int
  rx_ecn_marked_pkts: int
  tx_cnp_bytes: int
  rx_cnp_bytes: int
  seq_err_naks_rcvd: int
  rnr_naks_rcvd: int
  missing_resp: int
  to_retransmit: int
  dup_req: int
struct_creq_query_roce_stats_ext_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('rsvd', ctypes.c_ubyte, 7), ('tx_atomic_req_pkts', ctypes.c_uint64, 8), ('tx_read_req_pkts', ctypes.c_uint64, 16), ('tx_read_res_pkts', ctypes.c_uint64, 24), ('tx_write_req_pkts', ctypes.c_uint64, 32), ('tx_send_req_pkts', ctypes.c_uint64, 40), ('tx_roce_pkts', ctypes.c_uint64, 48), ('tx_roce_bytes', ctypes.c_uint64, 56), ('rx_atomic_req_pkts', ctypes.c_uint64, 64), ('rx_read_req_pkts', ctypes.c_uint64, 72), ('rx_read_res_pkts', ctypes.c_uint64, 80), ('rx_write_req_pkts', ctypes.c_uint64, 88), ('rx_send_req_pkts', ctypes.c_uint64, 96), ('rx_roce_pkts', ctypes.c_uint64, 104), ('rx_roce_bytes', ctypes.c_uint64, 112), ('rx_roce_good_pkts', ctypes.c_uint64, 120), ('rx_roce_good_bytes', ctypes.c_uint64, 128), ('rx_out_of_buffer_pkts', ctypes.c_uint64, 136), ('rx_out_of_sequence_pkts', ctypes.c_uint64, 144), ('tx_cnp_pkts', ctypes.c_uint64, 152), ('rx_cnp_pkts', ctypes.c_uint64, 160), ('rx_ecn_marked_pkts', ctypes.c_uint64, 168), ('tx_cnp_bytes', ctypes.c_uint64, 176), ('rx_cnp_bytes', ctypes.c_uint64, 184), ('seq_err_naks_rcvd', ctypes.c_uint64, 192), ('rnr_naks_rcvd', ctypes.c_uint64, 200), ('missing_resp', ctypes.c_uint64, 208), ('to_retransmit', ctypes.c_uint64, 216), ('dup_req', ctypes.c_uint64, 224)])
@c.record
class struct_cmdq_roce_mirror_cfg(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  mirror_flags: int
  rsvd: c.Array[ctypes.c_ubyte, Literal[7]]
struct_cmdq_roce_mirror_cfg.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('mirror_flags', ctypes.c_ubyte, 16), ('rsvd', c.Array[ctypes.c_ubyte, Literal[7]], 17)])
@c.record
class struct_creq_roce_mirror_cfg_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_roce_mirror_cfg_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_query_func(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_query_func.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_query_func_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_func_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_func_resp_sb(c.Struct):
  SIZE = 160
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  max_mr_size: int
  max_qp: int
  max_qp_wr: int
  dev_cap_flags: int
  max_cq: int
  max_cqe: int
  max_pd: int
  max_sge: int
  max_srq_sge: int
  max_qp_rd_atom: int
  max_qp_init_rd_atom: int
  max_mr: int
  max_mw: int
  max_raw_eth_qp: int
  max_ah: int
  max_fmr: int
  max_srq_wr: int
  max_pkeys: int
  max_inline_data: int
  max_map_per_fmr: int
  l2_db_space_size: int
  max_srq: int
  max_gid: int
  tqm_alloc_reqs: c.Array[ctypes.c_uint32, Literal[12]]
  max_dpi: int
  max_sge_var_wqe: int
  dev_cap_ext_flags: int
  max_inline_data_var_wqe: int
  start_qid: int
  max_msn_table_size: int
  reserved8_1: int
  dev_cap_ext_flags_2: int
  max_xp_qp_size: int
  create_qp_batch_size: int
  destroy_qp_batch_size: int
  max_srq_ext: int
  reserved64: int
struct_creq_query_func_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('max_mr_size', ctypes.c_uint64, 8), ('max_qp', ctypes.c_uint32, 16), ('max_qp_wr', ctypes.c_uint16, 20), ('dev_cap_flags', ctypes.c_uint16, 22), ('max_cq', ctypes.c_uint32, 24), ('max_cqe', ctypes.c_uint32, 28), ('max_pd', ctypes.c_uint32, 32), ('max_sge', ctypes.c_ubyte, 36), ('max_srq_sge', ctypes.c_ubyte, 37), ('max_qp_rd_atom', ctypes.c_ubyte, 38), ('max_qp_init_rd_atom', ctypes.c_ubyte, 39), ('max_mr', ctypes.c_uint32, 40), ('max_mw', ctypes.c_uint32, 44), ('max_raw_eth_qp', ctypes.c_uint32, 48), ('max_ah', ctypes.c_uint32, 52), ('max_fmr', ctypes.c_uint32, 56), ('max_srq_wr', ctypes.c_uint32, 60), ('max_pkeys', ctypes.c_uint32, 64), ('max_inline_data', ctypes.c_uint32, 68), ('max_map_per_fmr', ctypes.c_ubyte, 72), ('l2_db_space_size', ctypes.c_ubyte, 73), ('max_srq', ctypes.c_uint16, 74), ('max_gid', ctypes.c_uint32, 76), ('tqm_alloc_reqs', c.Array[ctypes.c_uint32, Literal[12]], 80), ('max_dpi', ctypes.c_uint32, 128), ('max_sge_var_wqe', ctypes.c_ubyte, 132), ('dev_cap_ext_flags', ctypes.c_ubyte, 133), ('max_inline_data_var_wqe', ctypes.c_uint16, 134), ('start_qid', ctypes.c_uint32, 136), ('max_msn_table_size', ctypes.c_ubyte, 140), ('reserved8_1', ctypes.c_ubyte, 141), ('dev_cap_ext_flags_2', ctypes.c_uint16, 142), ('max_xp_qp_size', ctypes.c_uint16, 144), ('create_qp_batch_size', ctypes.c_uint16, 146), ('destroy_qp_batch_size', ctypes.c_uint16, 148), ('max_srq_ext', ctypes.c_uint16, 150), ('reserved64', ctypes.c_uint64, 152)])
@c.record
class struct_cmdq_set_func_resources(c.Struct):
  SIZE = 56
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  number_of_qp: int
  number_of_mrw: int
  number_of_srq: int
  number_of_cq: int
  max_qp_per_vf: int
  max_mrw_per_vf: int
  max_srq_per_vf: int
  max_cq_per_vf: int
  max_gid_per_vf: int
  stat_ctx_id: int
struct_cmdq_set_func_resources.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('number_of_qp', ctypes.c_uint32, 16), ('number_of_mrw', ctypes.c_uint32, 20), ('number_of_srq', ctypes.c_uint32, 24), ('number_of_cq', ctypes.c_uint32, 28), ('max_qp_per_vf', ctypes.c_uint32, 32), ('max_mrw_per_vf', ctypes.c_uint32, 36), ('max_srq_per_vf', ctypes.c_uint32, 40), ('max_cq_per_vf', ctypes.c_uint32, 44), ('max_gid_per_vf', ctypes.c_uint32, 48), ('stat_ctx_id', ctypes.c_uint32, 52)])
@c.record
class struct_creq_set_func_resources_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_set_func_resources_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_read_context(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  xid: int
  type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[3]]
struct_cmdq_read_context.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('xid', ctypes.c_uint32, 16), ('type', ctypes.c_ubyte, 20), ('unused_0', c.Array[ctypes.c_ubyte, Literal[3]], 21)])
@c.record
class struct_creq_read_context(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved16: int
  reserved_32: int
struct_creq_read_context.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved16', ctypes.c_uint16, 10), ('reserved_32', ctypes.c_uint32, 12)])
@c.record
class struct_cmdq_map_tc_to_cos(c.Struct):
  SIZE = 24
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  cos0: int
  cos1: int
  unused_0: int
struct_cmdq_map_tc_to_cos.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('cos0', ctypes.c_uint16, 16), ('cos1', ctypes.c_uint16, 18), ('unused_0', ctypes.c_uint32, 20)])
@c.record
class struct_creq_map_tc_to_cos_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_map_tc_to_cos_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_query_roce_cc(c.Struct):
  SIZE = 16
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
struct_cmdq_query_roce_cc.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8)])
@c.record
class struct_creq_query_roce_cc_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  size: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_query_roce_cc_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('size', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_query_roce_cc_resp_sb(c.Struct):
  SIZE = 32
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  enable_cc: int
  tos_dscp_tos_ecn: int
  g: int
  num_phases_per_state: int
  init_cr: int
  init_tr: int
  alt_vlan_pcp: int
  alt_tos_dscp: int
  cc_mode: int
  tx_queue: int
  rtt: int
  tcp_cp: int
  inactivity_th: int
  pkts_per_phase: int
  time_per_phase: int
  reserved32: int
struct_creq_query_roce_cc_resp_sb.register_fields([('opcode', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('flags', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('enable_cc', ctypes.c_ubyte, 8), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 9), ('g', ctypes.c_ubyte, 10), ('num_phases_per_state', ctypes.c_ubyte, 11), ('init_cr', ctypes.c_uint16, 12), ('init_tr', ctypes.c_uint16, 14), ('alt_vlan_pcp', ctypes.c_ubyte, 16), ('alt_tos_dscp', ctypes.c_ubyte, 17), ('cc_mode', ctypes.c_ubyte, 18), ('tx_queue', ctypes.c_ubyte, 19), ('rtt', ctypes.c_uint16, 20), ('tcp_cp', ctypes.c_uint16, 22), ('inactivity_th', ctypes.c_uint16, 24), ('pkts_per_phase', ctypes.c_ubyte, 26), ('time_per_phase', ctypes.c_ubyte, 27), ('reserved32', ctypes.c_uint32, 28)])
@c.record
class struct_creq_query_roce_cc_resp_sb_tlv(c.Struct):
  SIZE = 48
  cmd_discr: int
  reserved_8b: int
  tlv_flags: int
  tlv_type: int
  length: int
  total_size: int
  reserved56: c.Array[ctypes.c_ubyte, Literal[7]]
  opcode: int
  status: int
  cookie: int
  flags: int
  resp_size: int
  reserved8: int
  enable_cc: int
  tos_dscp_tos_ecn: int
  g: int
  num_phases_per_state: int
  init_cr: int
  init_tr: int
  alt_vlan_pcp: int
  alt_tos_dscp: int
  cc_mode: int
  tx_queue: int
  rtt: int
  tcp_cp: int
  inactivity_th: int
  pkts_per_phase: int
  time_per_phase: int
  reserved32: int
struct_creq_query_roce_cc_resp_sb_tlv.register_fields([('cmd_discr', ctypes.c_uint16, 0), ('reserved_8b', ctypes.c_ubyte, 2), ('tlv_flags', ctypes.c_ubyte, 3), ('tlv_type', ctypes.c_uint16, 4), ('length', ctypes.c_uint16, 6), ('total_size', ctypes.c_ubyte, 8), ('reserved56', c.Array[ctypes.c_ubyte, Literal[7]], 9), ('opcode', ctypes.c_ubyte, 16), ('status', ctypes.c_ubyte, 17), ('cookie', ctypes.c_uint16, 18), ('flags', ctypes.c_uint16, 20), ('resp_size', ctypes.c_ubyte, 22), ('reserved8', ctypes.c_ubyte, 23), ('enable_cc', ctypes.c_ubyte, 24), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 25), ('g', ctypes.c_ubyte, 26), ('num_phases_per_state', ctypes.c_ubyte, 27), ('init_cr', ctypes.c_uint16, 28), ('init_tr', ctypes.c_uint16, 30), ('alt_vlan_pcp', ctypes.c_ubyte, 32), ('alt_tos_dscp', ctypes.c_ubyte, 33), ('cc_mode', ctypes.c_ubyte, 34), ('tx_queue', ctypes.c_ubyte, 35), ('rtt', ctypes.c_uint16, 36), ('tcp_cp', ctypes.c_uint16, 38), ('inactivity_th', ctypes.c_uint16, 40), ('pkts_per_phase', ctypes.c_ubyte, 42), ('time_per_phase', ctypes.c_ubyte, 43), ('reserved32', ctypes.c_uint32, 44)])
@c.record
class struct_creq_query_roce_cc_gen1_resp_sb_tlv(c.Struct):
  SIZE = 88
  cmd_discr: int
  reserved_8b: int
  tlv_flags: int
  tlv_type: int
  length: int
  reserved64: int
  inactivity_th_hi: int
  min_time_between_cnps: int
  init_cp: int
  tr_update_mode: int
  tr_update_cycles: int
  fr_num_rtts: int
  ai_rate_increase: int
  reduction_relax_rtts_th: int
  additional_relax_cr_th: int
  cr_min_th: int
  bw_avg_weight: int
  actual_cr_factor: int
  max_cp_cr_th: int
  cp_bias_en: int
  cp_bias: int
  cnp_ecn: int
  rtt_jitter_en: int
  link_bytes_per_usec: int
  reset_cc_cr_th: int
  cr_width: int
  quota_period_min: int
  quota_period_max: int
  quota_period_abs_max: int
  tr_lower_bound: int
  cr_prob_factor: int
  tr_prob_factor: int
  fairness_cr_th: int
  red_div: int
  cnp_ratio_th: int
  exp_ai_rtts: int
  exp_ai_cr_cp_ratio: int
  use_rate_table: int
  cp_exp_update_th: int
  high_exp_ai_rtts_th1: int
  high_exp_ai_rtts_th2: int
  actual_cr_cong_free_rtts_th: int
  severe_cong_cr_th1: int
  severe_cong_cr_th2: int
  link64B_per_rtt: int
  cc_ack_bytes: int
  reduce_init_en: int
  reduce_init_cong_free_rtts_th: int
  random_no_red_en: int
  actual_cr_shift_correction_en: int
  quota_period_adjust_en: int
  reserved: c.Array[ctypes.c_ubyte, Literal[5]]
struct_creq_query_roce_cc_gen1_resp_sb_tlv.register_fields([('cmd_discr', ctypes.c_uint16, 0), ('reserved_8b', ctypes.c_ubyte, 2), ('tlv_flags', ctypes.c_ubyte, 3), ('tlv_type', ctypes.c_uint16, 4), ('length', ctypes.c_uint16, 6), ('reserved64', ctypes.c_uint64, 8), ('inactivity_th_hi', ctypes.c_uint16, 16), ('min_time_between_cnps', ctypes.c_uint16, 18), ('init_cp', ctypes.c_uint16, 20), ('tr_update_mode', ctypes.c_ubyte, 22), ('tr_update_cycles', ctypes.c_ubyte, 23), ('fr_num_rtts', ctypes.c_ubyte, 24), ('ai_rate_increase', ctypes.c_ubyte, 25), ('reduction_relax_rtts_th', ctypes.c_uint16, 26), ('additional_relax_cr_th', ctypes.c_uint16, 28), ('cr_min_th', ctypes.c_uint16, 30), ('bw_avg_weight', ctypes.c_ubyte, 32), ('actual_cr_factor', ctypes.c_ubyte, 33), ('max_cp_cr_th', ctypes.c_uint16, 34), ('cp_bias_en', ctypes.c_ubyte, 36), ('cp_bias', ctypes.c_ubyte, 37), ('cnp_ecn', ctypes.c_ubyte, 38), ('rtt_jitter_en', ctypes.c_ubyte, 39), ('link_bytes_per_usec', ctypes.c_uint16, 40), ('reset_cc_cr_th', ctypes.c_uint16, 42), ('cr_width', ctypes.c_ubyte, 44), ('quota_period_min', ctypes.c_ubyte, 45), ('quota_period_max', ctypes.c_ubyte, 46), ('quota_period_abs_max', ctypes.c_ubyte, 47), ('tr_lower_bound', ctypes.c_uint16, 48), ('cr_prob_factor', ctypes.c_ubyte, 50), ('tr_prob_factor', ctypes.c_ubyte, 51), ('fairness_cr_th', ctypes.c_uint16, 52), ('red_div', ctypes.c_ubyte, 54), ('cnp_ratio_th', ctypes.c_ubyte, 55), ('exp_ai_rtts', ctypes.c_uint16, 56), ('exp_ai_cr_cp_ratio', ctypes.c_ubyte, 58), ('use_rate_table', ctypes.c_ubyte, 59), ('cp_exp_update_th', ctypes.c_uint16, 60), ('high_exp_ai_rtts_th1', ctypes.c_uint16, 62), ('high_exp_ai_rtts_th2', ctypes.c_uint16, 64), ('actual_cr_cong_free_rtts_th', ctypes.c_uint16, 66), ('severe_cong_cr_th1', ctypes.c_uint16, 68), ('severe_cong_cr_th2', ctypes.c_uint16, 70), ('link64B_per_rtt', ctypes.c_uint32, 72), ('cc_ack_bytes', ctypes.c_ubyte, 76), ('reduce_init_en', ctypes.c_ubyte, 77), ('reduce_init_cong_free_rtts_th', ctypes.c_uint16, 78), ('random_no_red_en', ctypes.c_ubyte, 80), ('actual_cr_shift_correction_en', ctypes.c_ubyte, 81), ('quota_period_adjust_en', ctypes.c_ubyte, 82), ('reserved', c.Array[ctypes.c_ubyte, Literal[5]], 83)])
@c.record
class struct_cmdq_modify_roce_cc(c.Struct):
  SIZE = 56
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  modify_mask: int
  enable_cc: int
  g: int
  num_phases_per_state: int
  pkts_per_phase: int
  init_cr: int
  init_tr: int
  tos_dscp_tos_ecn: int
  alt_vlan_pcp: int
  alt_tos_dscp: int
  rtt: int
  tcp_cp: int
  cc_mode: int
  tx_queue: int
  inactivity_th: int
  time_per_phase: int
  reserved8_1: int
  reserved16: int
  reserved32: int
  reserved64: int
struct_cmdq_modify_roce_cc.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('modify_mask', ctypes.c_uint32, 16), ('enable_cc', ctypes.c_ubyte, 20), ('g', ctypes.c_ubyte, 21), ('num_phases_per_state', ctypes.c_ubyte, 22), ('pkts_per_phase', ctypes.c_ubyte, 23), ('init_cr', ctypes.c_uint16, 24), ('init_tr', ctypes.c_uint16, 26), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 28), ('alt_vlan_pcp', ctypes.c_ubyte, 29), ('alt_tos_dscp', ctypes.c_uint16, 30), ('rtt', ctypes.c_uint16, 32), ('tcp_cp', ctypes.c_uint16, 34), ('cc_mode', ctypes.c_ubyte, 36), ('tx_queue', ctypes.c_ubyte, 37), ('inactivity_th', ctypes.c_uint16, 38), ('time_per_phase', ctypes.c_ubyte, 40), ('reserved8_1', ctypes.c_ubyte, 41), ('reserved16', ctypes.c_uint16, 42), ('reserved32', ctypes.c_uint32, 44), ('reserved64', ctypes.c_uint64, 48)])
@c.record
class struct_cmdq_modify_roce_cc_tlv(c.Struct):
  SIZE = 80
  cmd_discr: int
  reserved_8b: int
  tlv_flags: int
  tlv_type: int
  length: int
  total_size: int
  reserved56: c.Array[ctypes.c_ubyte, Literal[7]]
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  modify_mask: int
  enable_cc: int
  g: int
  num_phases_per_state: int
  pkts_per_phase: int
  init_cr: int
  init_tr: int
  tos_dscp_tos_ecn: int
  alt_vlan_pcp: int
  alt_tos_dscp: int
  rtt: int
  tcp_cp: int
  cc_mode: int
  tx_queue: int
  inactivity_th: int
  time_per_phase: int
  reserved8_1: int
  reserved16: int
  reserved32: int
  reserved64: int
  reservedtlvpad: int
struct_cmdq_modify_roce_cc_tlv.register_fields([('cmd_discr', ctypes.c_uint16, 0), ('reserved_8b', ctypes.c_ubyte, 2), ('tlv_flags', ctypes.c_ubyte, 3), ('tlv_type', ctypes.c_uint16, 4), ('length', ctypes.c_uint16, 6), ('total_size', ctypes.c_ubyte, 8), ('reserved56', c.Array[ctypes.c_ubyte, Literal[7]], 9), ('opcode', ctypes.c_ubyte, 16), ('cmd_size', ctypes.c_ubyte, 17), ('flags', ctypes.c_uint16, 18), ('cookie', ctypes.c_uint16, 20), ('resp_size', ctypes.c_ubyte, 22), ('reserved8', ctypes.c_ubyte, 23), ('resp_addr', ctypes.c_uint64, 24), ('modify_mask', ctypes.c_uint32, 32), ('enable_cc', ctypes.c_ubyte, 36), ('g', ctypes.c_ubyte, 37), ('num_phases_per_state', ctypes.c_ubyte, 38), ('pkts_per_phase', ctypes.c_ubyte, 39), ('init_cr', ctypes.c_uint16, 40), ('init_tr', ctypes.c_uint16, 42), ('tos_dscp_tos_ecn', ctypes.c_ubyte, 44), ('alt_vlan_pcp', ctypes.c_ubyte, 45), ('alt_tos_dscp', ctypes.c_uint16, 46), ('rtt', ctypes.c_uint16, 48), ('tcp_cp', ctypes.c_uint16, 50), ('cc_mode', ctypes.c_ubyte, 52), ('tx_queue', ctypes.c_ubyte, 53), ('inactivity_th', ctypes.c_uint16, 54), ('time_per_phase', ctypes.c_ubyte, 56), ('reserved8_1', ctypes.c_ubyte, 57), ('reserved16', ctypes.c_uint16, 58), ('reserved32', ctypes.c_uint32, 60), ('reserved64', ctypes.c_uint64, 64), ('reservedtlvpad', ctypes.c_uint64, 72)])
@c.record
class struct_cmdq_modify_roce_cc_gen1_tlv(c.Struct):
  SIZE = 96
  cmd_discr: int
  reserved_8b: int
  tlv_flags: int
  tlv_type: int
  length: int
  reserved64: int
  modify_mask: int
  inactivity_th_hi: int
  min_time_between_cnps: int
  init_cp: int
  tr_update_mode: int
  tr_update_cycles: int
  fr_num_rtts: int
  ai_rate_increase: int
  reduction_relax_rtts_th: int
  additional_relax_cr_th: int
  cr_min_th: int
  bw_avg_weight: int
  actual_cr_factor: int
  max_cp_cr_th: int
  cp_bias_en: int
  cp_bias: int
  cnp_ecn: int
  rtt_jitter_en: int
  link_bytes_per_usec: int
  reset_cc_cr_th: int
  cr_width: int
  quota_period_min: int
  quota_period_max: int
  quota_period_abs_max: int
  tr_lower_bound: int
  cr_prob_factor: int
  tr_prob_factor: int
  fairness_cr_th: int
  red_div: int
  cnp_ratio_th: int
  exp_ai_rtts: int
  exp_ai_cr_cp_ratio: int
  use_rate_table: int
  cp_exp_update_th: int
  high_exp_ai_rtts_th1: int
  high_exp_ai_rtts_th2: int
  actual_cr_cong_free_rtts_th: int
  severe_cong_cr_th1: int
  severe_cong_cr_th2: int
  link64B_per_rtt: int
  cc_ack_bytes: int
  reduce_init_en: int
  reduce_init_cong_free_rtts_th: int
  random_no_red_en: int
  actual_cr_shift_correction_en: int
  quota_period_adjust_en: int
  reserved: c.Array[ctypes.c_ubyte, Literal[5]]
struct_cmdq_modify_roce_cc_gen1_tlv.register_fields([('cmd_discr', ctypes.c_uint16, 0), ('reserved_8b', ctypes.c_ubyte, 2), ('tlv_flags', ctypes.c_ubyte, 3), ('tlv_type', ctypes.c_uint16, 4), ('length', ctypes.c_uint16, 6), ('reserved64', ctypes.c_uint64, 8), ('modify_mask', ctypes.c_uint64, 16), ('inactivity_th_hi', ctypes.c_uint16, 24), ('min_time_between_cnps', ctypes.c_uint16, 26), ('init_cp', ctypes.c_uint16, 28), ('tr_update_mode', ctypes.c_ubyte, 30), ('tr_update_cycles', ctypes.c_ubyte, 31), ('fr_num_rtts', ctypes.c_ubyte, 32), ('ai_rate_increase', ctypes.c_ubyte, 33), ('reduction_relax_rtts_th', ctypes.c_uint16, 34), ('additional_relax_cr_th', ctypes.c_uint16, 36), ('cr_min_th', ctypes.c_uint16, 38), ('bw_avg_weight', ctypes.c_ubyte, 40), ('actual_cr_factor', ctypes.c_ubyte, 41), ('max_cp_cr_th', ctypes.c_uint16, 42), ('cp_bias_en', ctypes.c_ubyte, 44), ('cp_bias', ctypes.c_ubyte, 45), ('cnp_ecn', ctypes.c_ubyte, 46), ('rtt_jitter_en', ctypes.c_ubyte, 47), ('link_bytes_per_usec', ctypes.c_uint16, 48), ('reset_cc_cr_th', ctypes.c_uint16, 50), ('cr_width', ctypes.c_ubyte, 52), ('quota_period_min', ctypes.c_ubyte, 53), ('quota_period_max', ctypes.c_ubyte, 54), ('quota_period_abs_max', ctypes.c_ubyte, 55), ('tr_lower_bound', ctypes.c_uint16, 56), ('cr_prob_factor', ctypes.c_ubyte, 58), ('tr_prob_factor', ctypes.c_ubyte, 59), ('fairness_cr_th', ctypes.c_uint16, 60), ('red_div', ctypes.c_ubyte, 62), ('cnp_ratio_th', ctypes.c_ubyte, 63), ('exp_ai_rtts', ctypes.c_uint16, 64), ('exp_ai_cr_cp_ratio', ctypes.c_ubyte, 66), ('use_rate_table', ctypes.c_ubyte, 67), ('cp_exp_update_th', ctypes.c_uint16, 68), ('high_exp_ai_rtts_th1', ctypes.c_uint16, 70), ('high_exp_ai_rtts_th2', ctypes.c_uint16, 72), ('actual_cr_cong_free_rtts_th', ctypes.c_uint16, 74), ('severe_cong_cr_th1', ctypes.c_uint16, 76), ('severe_cong_cr_th2', ctypes.c_uint16, 78), ('link64B_per_rtt', ctypes.c_uint32, 80), ('cc_ack_bytes', ctypes.c_ubyte, 84), ('reduce_init_en', ctypes.c_ubyte, 85), ('reduce_init_cong_free_rtts_th', ctypes.c_uint16, 86), ('random_no_red_en', ctypes.c_ubyte, 88), ('actual_cr_shift_correction_en', ctypes.c_ubyte, 89), ('quota_period_adjust_en', ctypes.c_ubyte, 90), ('reserved', c.Array[ctypes.c_ubyte, Literal[5]], 91)])
@c.record
class struct_creq_modify_roce_cc_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_modify_roce_cc_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_cmdq_set_link_aggr_mode_cc(c.Struct):
  SIZE = 40
  opcode: int
  cmd_size: int
  flags: int
  cookie: int
  resp_size: int
  reserved8: int
  resp_addr: int
  modify_mask: int
  aggr_enable: int
  active_port_map: int
  member_port_map: int
  link_aggr_mode: int
  stat_ctx_id: c.Array[ctypes.c_uint16, Literal[4]]
  rsvd1: int
struct_cmdq_set_link_aggr_mode_cc.register_fields([('opcode', ctypes.c_ubyte, 0), ('cmd_size', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('cookie', ctypes.c_uint16, 4), ('resp_size', ctypes.c_ubyte, 6), ('reserved8', ctypes.c_ubyte, 7), ('resp_addr', ctypes.c_uint64, 8), ('modify_mask', ctypes.c_uint32, 16), ('aggr_enable', ctypes.c_ubyte, 20), ('active_port_map', ctypes.c_ubyte, 21), ('member_port_map', ctypes.c_ubyte, 22), ('link_aggr_mode', ctypes.c_ubyte, 23), ('stat_ctx_id', c.Array[ctypes.c_uint16, Literal[4]], 24), ('rsvd1', ctypes.c_uint64, 32)])
@c.record
class struct_creq_set_link_aggr_mode_resources_resp(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_set_link_aggr_mode_resources_resp.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_func_event(c.Struct):
  SIZE = 16
  type: int
  reserved56: c.Array[ctypes.c_ubyte, Literal[7]]
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_func_event.register_fields([('type', ctypes.c_ubyte, 0), ('reserved56', c.Array[ctypes.c_ubyte, Literal[7]], 1), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_qp_event(c.Struct):
  SIZE = 16
  type: int
  status: int
  cookie: int
  reserved32: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_qp_event.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cookie', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_creq_qp_error_notification(c.Struct):
  SIZE = 16
  type: int
  status: int
  req_slow_path_state: int
  req_err_state_reason: int
  xid: int
  v: int
  event: int
  res_slow_path_state: int
  res_err_state_reason: int
  sq_cons_idx: int
  rq_cons_idx: int
struct_creq_qp_error_notification.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('req_slow_path_state', ctypes.c_ubyte, 2), ('req_err_state_reason', ctypes.c_ubyte, 3), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('res_slow_path_state', ctypes.c_ubyte, 10), ('res_err_state_reason', ctypes.c_ubyte, 11), ('sq_cons_idx', ctypes.c_uint16, 12), ('rq_cons_idx', ctypes.c_uint16, 14)])
@c.record
class struct_creq_cq_error_notification(c.Struct):
  SIZE = 16
  type: int
  status: int
  cq_err_reason: int
  reserved8: int
  xid: int
  v: int
  event: int
  reserved48: c.Array[ctypes.c_ubyte, Literal[6]]
struct_creq_cq_error_notification.register_fields([('type', ctypes.c_ubyte, 0), ('status', ctypes.c_ubyte, 1), ('cq_err_reason', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('xid', ctypes.c_uint32, 4), ('v', ctypes.c_ubyte, 8), ('event', ctypes.c_ubyte, 9), ('reserved48', c.Array[ctypes.c_ubyte, Literal[6]], 10)])
@c.record
class struct_sq_base(c.Struct):
  SIZE = 8
  wqe_type: int
  unused_0: c.Array[ctypes.c_ubyte, Literal[7]]
struct_sq_base.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('unused_0', c.Array[ctypes.c_ubyte, Literal[7]], 1)])
@c.record
class struct_sq_sge(c.Struct):
  SIZE = 16
  va_or_pa: int
  l_key: int
  size: int
struct_sq_sge.register_fields([('va_or_pa', ctypes.c_uint64, 0), ('l_key', ctypes.c_uint32, 8), ('size', ctypes.c_uint32, 12)])
@c.record
class struct_sq_psn_search(c.Struct):
  SIZE = 8
  opcode_start_psn: int
  flags_next_psn: int
struct_sq_psn_search.register_fields([('opcode_start_psn', ctypes.c_uint32, 0), ('flags_next_psn', ctypes.c_uint32, 4)])
@c.record
class struct_sq_psn_search_ext(c.Struct):
  SIZE = 16
  opcode_start_psn: int
  flags_next_psn: int
  start_slot_idx: int
  reserved16: int
  reserved32: int
struct_sq_psn_search_ext.register_fields([('opcode_start_psn', ctypes.c_uint32, 0), ('flags_next_psn', ctypes.c_uint32, 4), ('start_slot_idx', ctypes.c_uint16, 8), ('reserved16', ctypes.c_uint16, 10), ('reserved32', ctypes.c_uint32, 12)])
@c.record
class struct_sq_msn_search(c.Struct):
  SIZE = 8
  start_idx_next_psn_start_psn: int
struct_sq_msn_search.register_fields([('start_idx_next_psn_start_psn', ctypes.c_uint64, 0)])
@c.record
class struct_sq_send(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8_1: int
  inv_key_or_imm_data: int
  length: int
  q_key: int
  dst_qp: int
  avid: int
  reserved32: int
  timestamp: int
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_send.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8_1', ctypes.c_ubyte, 3), ('inv_key_or_imm_data', ctypes.c_uint32, 4), ('length', ctypes.c_uint32, 8), ('q_key', ctypes.c_uint32, 12), ('dst_qp', ctypes.c_uint32, 16), ('avid', ctypes.c_uint32, 20), ('reserved32', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_send_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8_1: int
  inv_key_or_imm_data: int
  length: int
  q_key: int
  dst_qp: int
  avid: int
  reserved32: int
  timestamp: int
struct_sq_send_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8_1', ctypes.c_ubyte, 3), ('inv_key_or_imm_data', ctypes.c_uint32, 4), ('length', ctypes.c_uint32, 8), ('q_key', ctypes.c_uint32, 12), ('dst_qp', ctypes.c_uint32, 16), ('avid', ctypes.c_uint32, 20), ('reserved32', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28)])
@c.record
class struct_sq_send_raweth_qp1(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  lflags: int
  cfa_action: int
  length: int
  reserved32_1: int
  cfa_meta: int
  reserved32_2: int
  reserved32_3: int
  timestamp: int
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_send_raweth_qp1.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('lflags', ctypes.c_uint16, 4), ('cfa_action', ctypes.c_uint16, 6), ('length', ctypes.c_uint32, 8), ('reserved32_1', ctypes.c_uint32, 12), ('cfa_meta', ctypes.c_uint32, 16), ('reserved32_2', ctypes.c_uint32, 20), ('reserved32_3', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_send_raweth_qp1_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  lflags: int
  cfa_action: int
  length: int
  reserved32_1: int
  cfa_meta: int
  reserved32_2: int
  reserved32_3: int
  timestamp: int
struct_sq_send_raweth_qp1_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('lflags', ctypes.c_uint16, 4), ('cfa_action', ctypes.c_uint16, 6), ('length', ctypes.c_uint32, 8), ('reserved32_1', ctypes.c_uint32, 12), ('cfa_meta', ctypes.c_uint32, 16), ('reserved32_2', ctypes.c_uint32, 20), ('reserved32_3', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28)])
@c.record
class struct_sq_rdma(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  imm_data: int
  length: int
  reserved32_1: int
  remote_va: int
  remote_key: int
  timestamp: int
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_rdma.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('imm_data', ctypes.c_uint32, 4), ('length', ctypes.c_uint32, 8), ('reserved32_1', ctypes.c_uint32, 12), ('remote_va', ctypes.c_uint64, 16), ('remote_key', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_rdma_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  imm_data: int
  length: int
  reserved32_1: int
  remote_va: int
  remote_key: int
  timestamp: int
struct_sq_rdma_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('imm_data', ctypes.c_uint32, 4), ('length', ctypes.c_uint32, 8), ('reserved32_1', ctypes.c_uint32, 12), ('remote_va', ctypes.c_uint64, 16), ('remote_key', ctypes.c_uint32, 24), ('timestamp', ctypes.c_uint32, 28)])
@c.record
class struct_sq_atomic(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  reserved16: int
  remote_key: int
  remote_va: int
  swap_data: int
  cmp_data: int
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_atomic.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('reserved16', ctypes.c_uint16, 2), ('remote_key', ctypes.c_uint32, 4), ('remote_va', ctypes.c_uint64, 8), ('swap_data', ctypes.c_uint64, 16), ('cmp_data', ctypes.c_uint64, 24), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_atomic_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  reserved16: int
  remote_key: int
  remote_va: int
  swap_data: int
  cmp_data: int
struct_sq_atomic_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('reserved16', ctypes.c_uint16, 2), ('remote_key', ctypes.c_uint32, 4), ('remote_va', ctypes.c_uint64, 8), ('swap_data', ctypes.c_uint64, 16), ('cmp_data', ctypes.c_uint64, 24)])
@c.record
class struct_sq_localinvalidate(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  reserved16: int
  inv_l_key: int
  reserved64: int
  reserved128: c.Array[ctypes.c_ubyte, Literal[16]]
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_localinvalidate.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('reserved16', ctypes.c_uint16, 2), ('inv_l_key', ctypes.c_uint32, 4), ('reserved64', ctypes.c_uint64, 8), ('reserved128', c.Array[ctypes.c_ubyte, Literal[16]], 16), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_localinvalidate_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  reserved16: int
  inv_l_key: int
  reserved64: int
  reserved128: c.Array[ctypes.c_ubyte, Literal[16]]
struct_sq_localinvalidate_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('reserved16', ctypes.c_uint16, 2), ('inv_l_key', ctypes.c_uint32, 4), ('reserved64', ctypes.c_uint64, 8), ('reserved128', c.Array[ctypes.c_ubyte, Literal[16]], 16)])
@c.record
class struct_sq_fr_pmr(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  access_cntl: int
  zero_based_page_size_log: int
  l_key: int
  length: c.Array[ctypes.c_ubyte, Literal[5]]
  reserved8_1: int
  reserved8_2: int
  numlevels_pbl_page_size_log: int
  pblptr: int
  va: int
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_fr_pmr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('access_cntl', ctypes.c_ubyte, 2), ('zero_based_page_size_log', ctypes.c_ubyte, 3), ('l_key', ctypes.c_uint32, 4), ('length', c.Array[ctypes.c_ubyte, Literal[5]], 8), ('reserved8_1', ctypes.c_ubyte, 13), ('reserved8_2', ctypes.c_ubyte, 14), ('numlevels_pbl_page_size_log', ctypes.c_ubyte, 15), ('pblptr', ctypes.c_uint64, 16), ('va', ctypes.c_uint64, 24), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_fr_pmr_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  access_cntl: int
  zero_based_page_size_log: int
  l_key: int
  length: c.Array[ctypes.c_ubyte, Literal[5]]
  reserved8_1: int
  reserved8_2: int
  numlevels_pbl_page_size_log: int
  pblptr: int
  va: int
struct_sq_fr_pmr_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('access_cntl', ctypes.c_ubyte, 2), ('zero_based_page_size_log', ctypes.c_ubyte, 3), ('l_key', ctypes.c_uint32, 4), ('length', c.Array[ctypes.c_ubyte, Literal[5]], 8), ('reserved8_1', ctypes.c_ubyte, 13), ('reserved8_2', ctypes.c_ubyte, 14), ('numlevels_pbl_page_size_log', ctypes.c_ubyte, 15), ('pblptr', ctypes.c_uint64, 16), ('va', ctypes.c_uint64, 24)])
@c.record
class struct_sq_bind(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  access_cntl: int
  reserved8_1: int
  mw_type_zero_based: int
  reserved8_2: int
  reserved16: int
  parent_l_key: int
  l_key: int
  va: int
  length: c.Array[ctypes.c_ubyte, Literal[5]]
  reserved24: c.Array[ctypes.c_ubyte, Literal[3]]
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_sq_bind.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('access_cntl', ctypes.c_ubyte, 2), ('reserved8_1', ctypes.c_ubyte, 3), ('mw_type_zero_based', ctypes.c_ubyte, 4), ('reserved8_2', ctypes.c_ubyte, 5), ('reserved16', ctypes.c_uint16, 6), ('parent_l_key', ctypes.c_uint32, 8), ('l_key', ctypes.c_uint32, 12), ('va', ctypes.c_uint64, 16), ('length', c.Array[ctypes.c_ubyte, Literal[5]], 24), ('reserved24', c.Array[ctypes.c_ubyte, Literal[3]], 29), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_sq_bind_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  access_cntl: int
  reserved8_1: int
  mw_type_zero_based: int
  reserved8_2: int
  reserved16: int
  parent_l_key: int
  l_key: int
  va: int
  length: c.Array[ctypes.c_ubyte, Literal[5]]
  reserved24: c.Array[ctypes.c_ubyte, Literal[3]]
struct_sq_bind_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('access_cntl', ctypes.c_ubyte, 2), ('reserved8_1', ctypes.c_ubyte, 3), ('mw_type_zero_based', ctypes.c_ubyte, 4), ('reserved8_2', ctypes.c_ubyte, 5), ('reserved16', ctypes.c_uint16, 6), ('parent_l_key', ctypes.c_uint32, 8), ('l_key', ctypes.c_uint32, 12), ('va', ctypes.c_uint64, 16), ('length', c.Array[ctypes.c_ubyte, Literal[5]], 24), ('reserved24', c.Array[ctypes.c_ubyte, Literal[3]], 29)])
@c.record
class struct_rq_wqe(c.Struct):
  SIZE = 128
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  reserved32: int
  wr_id: c.Array[ctypes.c_uint32, Literal[2]]
  reserved128: c.Array[ctypes.c_ubyte, Literal[16]]
  data: c.Array[ctypes.c_uint32, Literal[24]]
struct_rq_wqe.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('reserved32', ctypes.c_uint32, 4), ('wr_id', c.Array[ctypes.c_uint32, Literal[2]], 8), ('reserved128', c.Array[ctypes.c_ubyte, Literal[16]], 16), ('data', c.Array[ctypes.c_uint32, Literal[24]], 32)])
@c.record
class struct_rq_wqe_hdr(c.Struct):
  SIZE = 32
  wqe_type: int
  flags: int
  wqe_size: int
  reserved8: int
  reserved32: int
  wr_id: c.Array[ctypes.c_uint32, Literal[2]]
  reserved128: c.Array[ctypes.c_ubyte, Literal[16]]
struct_rq_wqe_hdr.register_fields([('wqe_type', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('wqe_size', ctypes.c_ubyte, 2), ('reserved8', ctypes.c_ubyte, 3), ('reserved32', ctypes.c_uint32, 4), ('wr_id', c.Array[ctypes.c_uint32, Literal[2]], 8), ('reserved128', c.Array[ctypes.c_ubyte, Literal[16]], 16)])
@c.record
class struct_cq_base(c.Struct):
  SIZE = 32
  reserved64_1: int
  reserved64_2: int
  reserved64_3: int
  cqe_type_toggle: int
  status: int
  reserved16: int
  opaque: int
struct_cq_base.register_fields([('reserved64_1', ctypes.c_uint64, 0), ('reserved64_2', ctypes.c_uint64, 8), ('reserved64_3', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('reserved16', ctypes.c_uint16, 26), ('opaque', ctypes.c_uint32, 28)])
@c.record
class struct_cq_req(c.Struct):
  SIZE = 32
  qp_handle: int
  sq_cons_idx: int
  reserved16_1: int
  reserved32_2: int
  reserved64: int
  cqe_type_toggle: int
  status: int
  reserved16_2: int
  reserved32_1: int
struct_cq_req.register_fields([('qp_handle', ctypes.c_uint64, 0), ('sq_cons_idx', ctypes.c_uint16, 8), ('reserved16_1', ctypes.c_uint16, 10), ('reserved32_2', ctypes.c_uint32, 12), ('reserved64', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('reserved16_2', ctypes.c_uint16, 26), ('reserved32_1', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_rc(c.Struct):
  SIZE = 32
  length: int
  imm_data_or_inv_r_key: int
  qp_handle: int
  mr_handle: int
  cqe_type_toggle: int
  status: int
  flags: int
  srq_or_rq_wr_id: int
struct_cq_res_rc.register_fields([('length', ctypes.c_uint32, 0), ('imm_data_or_inv_r_key', ctypes.c_uint32, 4), ('qp_handle', ctypes.c_uint64, 8), ('mr_handle', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_ud(c.Struct):
  SIZE = 32
  length: int
  cfa_metadata: int
  imm_data: int
  qp_handle: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  src_qp_low: int
  cqe_type_toggle: int
  status: int
  flags: int
  src_qp_high_srq_or_rq_wr_id: int
struct_cq_res_ud.register_fields([('length', ctypes.c_uint16, 0), ('cfa_metadata', ctypes.c_uint16, 2), ('imm_data', ctypes.c_uint32, 4), ('qp_handle', ctypes.c_uint64, 8), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 16), ('src_qp_low', ctypes.c_uint16, 22), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('src_qp_high_srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_ud_v2(c.Struct):
  SIZE = 32
  length: int
  cfa_metadata0: int
  imm_data: int
  qp_handle: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  src_qp_low: int
  cqe_type_toggle: int
  status: int
  flags: int
  src_qp_high_srq_or_rq_wr_id: int
struct_cq_res_ud_v2.register_fields([('length', ctypes.c_uint16, 0), ('cfa_metadata0', ctypes.c_uint16, 2), ('imm_data', ctypes.c_uint32, 4), ('qp_handle', ctypes.c_uint64, 8), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 16), ('src_qp_low', ctypes.c_uint16, 22), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('src_qp_high_srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_ud_cfa(c.Struct):
  SIZE = 32
  length: int
  cfa_code: int
  imm_data: int
  qid: int
  cfa_metadata: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  src_qp_low: int
  cqe_type_toggle: int
  status: int
  flags: int
  src_qp_high_srq_or_rq_wr_id: int
struct_cq_res_ud_cfa.register_fields([('length', ctypes.c_uint16, 0), ('cfa_code', ctypes.c_uint16, 2), ('imm_data', ctypes.c_uint32, 4), ('qid', ctypes.c_uint32, 8), ('cfa_metadata', ctypes.c_uint32, 12), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 16), ('src_qp_low', ctypes.c_uint16, 22), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('src_qp_high_srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_ud_cfa_v2(c.Struct):
  SIZE = 32
  length: int
  cfa_metadata0: int
  imm_data: int
  qid: int
  cfa_metadata2: int
  src_mac: c.Array[ctypes.c_uint16, Literal[3]]
  src_qp_low: int
  cqe_type_toggle: int
  status: int
  flags: int
  src_qp_high_srq_or_rq_wr_id: int
struct_cq_res_ud_cfa_v2.register_fields([('length', ctypes.c_uint16, 0), ('cfa_metadata0', ctypes.c_uint16, 2), ('imm_data', ctypes.c_uint32, 4), ('qid', ctypes.c_uint32, 8), ('cfa_metadata2', ctypes.c_uint32, 12), ('src_mac', c.Array[ctypes.c_uint16, Literal[3]], 16), ('src_qp_low', ctypes.c_uint16, 22), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('src_qp_high_srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_raweth_qp1(c.Struct):
  SIZE = 32
  length: int
  raweth_qp1_flags: int
  raweth_qp1_errors: int
  raweth_qp1_cfa_code: int
  qp_handle: int
  raweth_qp1_flags2: int
  raweth_qp1_metadata: int
  cqe_type_toggle: int
  status: int
  flags: int
  raweth_qp1_payload_offset_srq_or_rq_wr_id: int
struct_cq_res_raweth_qp1.register_fields([('length', ctypes.c_uint16, 0), ('raweth_qp1_flags', ctypes.c_uint16, 2), ('raweth_qp1_errors', ctypes.c_uint16, 4), ('raweth_qp1_cfa_code', ctypes.c_uint16, 6), ('qp_handle', ctypes.c_uint64, 8), ('raweth_qp1_flags2', ctypes.c_uint32, 16), ('raweth_qp1_metadata', ctypes.c_uint32, 20), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('raweth_qp1_payload_offset_srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_res_raweth_qp1_v2(c.Struct):
  SIZE = 32
  length: int
  raweth_qp1_flags: int
  raweth_qp1_errors: int
  cfa_metadata0: int
  qp_handle: int
  raweth_qp1_flags2: int
  cfa_metadata2: int
  cqe_type_toggle: int
  status: int
  flags: int
  raweth_qp1_payload_offset_srq_or_rq_wr_id: int
struct_cq_res_raweth_qp1_v2.register_fields([('length', ctypes.c_uint16, 0), ('raweth_qp1_flags', ctypes.c_uint16, 2), ('raweth_qp1_errors', ctypes.c_uint16, 4), ('cfa_metadata0', ctypes.c_uint16, 6), ('qp_handle', ctypes.c_uint64, 8), ('raweth_qp1_flags2', ctypes.c_uint32, 16), ('cfa_metadata2', ctypes.c_uint32, 20), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('flags', ctypes.c_uint16, 26), ('raweth_qp1_payload_offset_srq_or_rq_wr_id', ctypes.c_uint32, 28)])
@c.record
class struct_cq_terminal(c.Struct):
  SIZE = 32
  qp_handle: int
  sq_cons_idx: int
  rq_cons_idx: int
  reserved32_1: int
  reserved64_3: int
  cqe_type_toggle: int
  status: int
  reserved16: int
  reserved32_2: int
struct_cq_terminal.register_fields([('qp_handle', ctypes.c_uint64, 0), ('sq_cons_idx', ctypes.c_uint16, 8), ('rq_cons_idx', ctypes.c_uint16, 10), ('reserved32_1', ctypes.c_uint32, 12), ('reserved64_3', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('reserved16', ctypes.c_uint16, 26), ('reserved32_2', ctypes.c_uint32, 28)])
@c.record
class struct_cq_cutoff(c.Struct):
  SIZE = 32
  reserved64_1: int
  reserved64_2: int
  reserved64_3: int
  cqe_type_toggle: int
  status: int
  reserved16: int
  reserved32: int
struct_cq_cutoff.register_fields([('reserved64_1', ctypes.c_uint64, 0), ('reserved64_2', ctypes.c_uint64, 8), ('reserved64_3', ctypes.c_uint64, 16), ('cqe_type_toggle', ctypes.c_ubyte, 24), ('status', ctypes.c_ubyte, 25), ('reserved16', ctypes.c_uint16, 26), ('reserved32', ctypes.c_uint32, 28)])
@c.record
class struct_nq_base(c.Struct):
  SIZE = 16
  info10_type: int
  info16: int
  info32: int
  info63_v: c.Array[ctypes.c_uint32, Literal[2]]
struct_nq_base.register_fields([('info10_type', ctypes.c_uint16, 0), ('info16', ctypes.c_uint16, 2), ('info32', ctypes.c_uint32, 4), ('info63_v', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_nq_cn(c.Struct):
  SIZE = 16
  type: int
  reserved16: int
  cq_handle_low: int
  v: int
  cq_handle_high: int
struct_nq_cn.register_fields([('type', ctypes.c_uint16, 0), ('reserved16', ctypes.c_uint16, 2), ('cq_handle_low', ctypes.c_uint32, 4), ('v', ctypes.c_uint32, 8), ('cq_handle_high', ctypes.c_uint32, 12)])
@c.record
class struct_nq_srq_event(c.Struct):
  SIZE = 16
  type: int
  event: int
  reserved16: int
  srq_handle_low: int
  v: int
  srq_handle_high: int
struct_nq_srq_event.register_fields([('type', ctypes.c_ubyte, 0), ('event', ctypes.c_ubyte, 1), ('reserved16', ctypes.c_uint16, 2), ('srq_handle_low', ctypes.c_uint32, 4), ('v', ctypes.c_uint32, 8), ('srq_handle_high', ctypes.c_uint32, 12)])
@c.record
class struct_nq_dbq_event(c.Struct):
  SIZE = 16
  type: int
  event: int
  db_pfid: int
  db_dpi: int
  v: int
  db_type_db_xid: int
struct_nq_dbq_event.register_fields([('type', ctypes.c_ubyte, 0), ('event', ctypes.c_ubyte, 1), ('db_pfid', ctypes.c_uint16, 2), ('db_dpi', ctypes.c_uint32, 4), ('v', ctypes.c_uint32, 8), ('db_type_db_xid', ctypes.c_uint32, 12)])
@c.record
class struct_xrrq_irrq(c.Struct):
  SIZE = 32
  credits_type: int
  reserved16: int
  reserved32: int
  psn: int
  msn: int
  va_or_atomic_result: int
  rdma_r_key: int
  length: int
struct_xrrq_irrq.register_fields([('credits_type', ctypes.c_uint16, 0), ('reserved16', ctypes.c_uint16, 2), ('reserved32', ctypes.c_uint32, 4), ('psn', ctypes.c_uint32, 8), ('msn', ctypes.c_uint32, 12), ('va_or_atomic_result', ctypes.c_uint64, 16), ('rdma_r_key', ctypes.c_uint32, 24), ('length', ctypes.c_uint32, 28)])
@c.record
class struct_xrrq_orrq(c.Struct):
  SIZE = 32
  num_sges_type: int
  reserved16: int
  length: int
  psn: int
  end_psn: int
  first_sge_phy_or_sing_sge_va: int
  single_sge_l_key: int
  single_sge_size: int
struct_xrrq_orrq.register_fields([('num_sges_type', ctypes.c_uint16, 0), ('reserved16', ctypes.c_uint16, 2), ('length', ctypes.c_uint32, 4), ('psn', ctypes.c_uint32, 8), ('end_psn', ctypes.c_uint32, 12), ('first_sge_phy_or_sing_sge_va', ctypes.c_uint64, 16), ('single_sge_l_key', ctypes.c_uint32, 24), ('single_sge_size', ctypes.c_uint32, 28)])
@c.record
class struct_ptu_pte(c.Struct):
  SIZE = 8
  page_next_to_last_last_valid: c.Array[ctypes.c_uint32, Literal[2]]
struct_ptu_pte.register_fields([('page_next_to_last_last_valid', c.Array[ctypes.c_uint32, Literal[2]], 0)])
@c.record
class struct_ptu_pde(c.Struct):
  SIZE = 8
  page_valid: c.Array[ctypes.c_uint32, Literal[2]]
struct_ptu_pde.register_fields([('page_valid', c.Array[ctypes.c_uint32, Literal[2]], 0)])
@c.record
class struct_bnxt_qplib_cmdqe(c.Struct):
  SIZE = 16
  data: c.Array[ctypes.c_ubyte, Literal[16]]
struct_bnxt_qplib_cmdqe.register_fields([('data', c.Array[ctypes.c_ubyte, Literal[16]], 0)])
@c.record
class struct_bnxt_qplib_crsbe(c.Struct):
  SIZE = 1024
  data: c.Array[ctypes.c_ubyte, Literal[1024]]
struct_bnxt_qplib_crsbe.register_fields([('data', c.Array[ctypes.c_ubyte, Literal[1024]], 0)])
class struct_bnxt_qplib_rcfw(c.Struct): pass
aeq_handler_t: TypeAlias = c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_bnxt_qplib_rcfw], ctypes.c_void_p, ctypes.c_void_p]]
class struct_bnxt_qplib_crsqe(c.Struct): pass
class struct_bnxt_qplib_rcfw_sbuf(c.Struct): pass
@c.record
class struct_bnxt_qplib_qp_node(c.Struct):
  SIZE = 16
  qp_id: int
  qp_handle: ctypes.c_void_p
struct_bnxt_qplib_qp_node.register_fields([('qp_id', ctypes.c_uint32, 0), ('qp_handle', ctypes.c_void_p, 8)])
class struct_bnxt_qplib_cmdq_mbox(c.Struct): pass
class struct_bnxt_qplib_cmdq_ctx(c.Struct): pass
class struct_bnxt_qplib_creq_db(c.Struct): pass
@c.record
class struct_bnxt_qplib_creq_stat(c.Struct):
  SIZE = 16
  creq_qp_event_processed: int
  creq_func_event_processed: int
struct_bnxt_qplib_creq_stat.register_fields([('creq_qp_event_processed', ctypes.c_uint64, 0), ('creq_func_event_processed', ctypes.c_uint64, 8)])
class struct_bnxt_qplib_creq_ctx(c.Struct): pass
@c.record
class struct_bnxt_qplib_cmdqmsg(c.Struct):
  SIZE = 40
  req: c.POINTER[struct_cmdq_base]
  resp: c.POINTER[struct_creq_base]
  sb: ctypes.c_void_p
  req_sz: int
  res_sz: int
  block: int
struct_bnxt_qplib_cmdqmsg.register_fields([('req', c.POINTER[struct_cmdq_base], 0), ('resp', c.POINTER[struct_creq_base], 8), ('sb', ctypes.c_void_p, 16), ('req_sz', ctypes.c_uint32, 24), ('res_sz', ctypes.c_uint32, 28), ('block', ctypes.c_ubyte, 32)])
class struct_bnxt_qplib_gid(c.Struct): pass
class struct_bnxt_qplib_drv_modes(c.Struct): pass
enum_bnxt_re_toggle_modes: dict[int, str] = {(BNXT_QPLIB_CQ_TOGGLE_BIT:=1): 'BNXT_QPLIB_CQ_TOGGLE_BIT', (BNXT_QPLIB_SRQ_TOGGLE_BIT:=2): 'BNXT_QPLIB_SRQ_TOGGLE_BIT'}
class struct_bnxt_qplib_chip_ctx(c.Struct): pass
@c.record
class struct_bnxt_qplib_db_pacing_data(c.Struct):
  SIZE = 32
  do_pacing: int
  pacing_th: int
  alarm_th: int
  fifo_max_depth: int
  fifo_room_mask: int
  fifo_room_shift: int
  grc_reg_offset: int
  dev_err_state: int
struct_bnxt_qplib_db_pacing_data.register_fields([('do_pacing', ctypes.c_uint32, 0), ('pacing_th', ctypes.c_uint32, 4), ('alarm_th', ctypes.c_uint32, 8), ('fifo_max_depth', ctypes.c_uint32, 12), ('fifo_room_mask', ctypes.c_uint32, 16), ('fifo_room_shift', ctypes.c_uint32, 20), ('grc_reg_offset', ctypes.c_uint32, 24), ('dev_err_state', ctypes.c_uint32, 28)])
enum_bnxt_qplib_hwq_type: dict[int, str] = {(HWQ_TYPE_CTX:=0): 'HWQ_TYPE_CTX', (HWQ_TYPE_QUEUE:=1): 'HWQ_TYPE_QUEUE', (HWQ_TYPE_L2_CMPL:=2): 'HWQ_TYPE_L2_CMPL', (HWQ_TYPE_MR:=3): 'HWQ_TYPE_MR'}
enum_bnxt_qplib_pbl_lvl: dict[int, str] = {(PBL_LVL_0:=0): 'PBL_LVL_0', (PBL_LVL_1:=1): 'PBL_LVL_1', (PBL_LVL_2:=2): 'PBL_LVL_2', (PBL_LVL_MAX:=3): 'PBL_LVL_MAX'}
enum_bnxt_qplib_hwrm_pg_size: dict[int, str] = {(BNXT_QPLIB_HWRM_PG_SIZE_4K:=0): 'BNXT_QPLIB_HWRM_PG_SIZE_4K', (BNXT_QPLIB_HWRM_PG_SIZE_8K:=1): 'BNXT_QPLIB_HWRM_PG_SIZE_8K', (BNXT_QPLIB_HWRM_PG_SIZE_64K:=2): 'BNXT_QPLIB_HWRM_PG_SIZE_64K', (BNXT_QPLIB_HWRM_PG_SIZE_2M:=3): 'BNXT_QPLIB_HWRM_PG_SIZE_2M', (BNXT_QPLIB_HWRM_PG_SIZE_8M:=4): 'BNXT_QPLIB_HWRM_PG_SIZE_8M', (BNXT_QPLIB_HWRM_PG_SIZE_1G:=5): 'BNXT_QPLIB_HWRM_PG_SIZE_1G'}
class struct_bnxt_qplib_reg_desc(c.Struct): pass
class struct_bnxt_qplib_pbl(c.Struct): pass
class struct_bnxt_qplib_sg_info(c.Struct): pass
@c.record
class struct_bnxt_qplib_hwq_attr(c.Struct):
  SIZE = 40
  res: c.POINTER[struct_bnxt_qplib_res]
  sginfo: c.POINTER[struct_bnxt_qplib_sg_info]
  type: int
  depth: int
  stride: int
  aux_stride: int
  aux_depth: int
class struct_bnxt_qplib_res(c.Struct): pass
struct_bnxt_qplib_hwq_attr.register_fields([('res', c.POINTER[struct_bnxt_qplib_res], 0), ('sginfo', c.POINTER[struct_bnxt_qplib_sg_info], 8), ('type', ctypes.c_uint32, 16), ('depth', ctypes.c_uint32, 20), ('stride', ctypes.c_uint32, 24), ('aux_stride', ctypes.c_uint32, 28), ('aux_depth', ctypes.c_uint32, 32)])
class struct_bnxt_qplib_hwq(c.Struct): pass
class struct_bnxt_qplib_db_info(c.Struct): pass
enum_bnxt_qplib_db_info_flags_mask: dict[int, str] = {(BNXT_QPLIB_FLAG_EPOCH_CONS_SHIFT:=0): 'BNXT_QPLIB_FLAG_EPOCH_CONS_SHIFT', (BNXT_QPLIB_FLAG_EPOCH_PROD_SHIFT:=1): 'BNXT_QPLIB_FLAG_EPOCH_PROD_SHIFT', (BNXT_QPLIB_FLAG_EPOCH_CONS_MASK:=1): 'BNXT_QPLIB_FLAG_EPOCH_CONS_MASK', (BNXT_QPLIB_FLAG_EPOCH_PROD_MASK:=2): 'BNXT_QPLIB_FLAG_EPOCH_PROD_MASK'}
enum_bnxt_qplib_db_epoch_flag_shift: dict[int, str] = {(BNXT_QPLIB_DB_EPOCH_CONS_SHIFT:=24): 'BNXT_QPLIB_DB_EPOCH_CONS_SHIFT', (BNXT_QPLIB_DB_EPOCH_PROD_SHIFT:=23): 'BNXT_QPLIB_DB_EPOCH_PROD_SHIFT'}
@c.record
class struct_bnxt_qplib_pd_tbl(c.Struct):
  SIZE = 16
  tbl: c.POINTER[ctypes.c_uint64]
  max: int
struct_bnxt_qplib_pd_tbl.register_fields([('tbl', c.POINTER[ctypes.c_uint64], 0), ('max', ctypes.c_uint32, 8)])
class struct_bnxt_qplib_sgid_tbl(c.Struct): pass
_anonenum0: dict[int, str] = {(BNXT_QPLIB_DPI_TYPE_KERNEL:=0): 'BNXT_QPLIB_DPI_TYPE_KERNEL', (BNXT_QPLIB_DPI_TYPE_UC:=1): 'BNXT_QPLIB_DPI_TYPE_UC', (BNXT_QPLIB_DPI_TYPE_WC:=2): 'BNXT_QPLIB_DPI_TYPE_WC'}
class struct_bnxt_qplib_dpi(c.Struct): pass
class struct_bnxt_qplib_dpi_tbl(c.Struct): pass
class struct_bnxt_qplib_stats(c.Struct): pass
@c.record
class struct_bnxt_qplib_vf_res(c.Struct):
  SIZE = 20
  max_qp_per_vf: int
  max_mrw_per_vf: int
  max_srq_per_vf: int
  max_cq_per_vf: int
  max_gid_per_vf: int
struct_bnxt_qplib_vf_res.register_fields([('max_qp_per_vf', ctypes.c_uint32, 0), ('max_mrw_per_vf', ctypes.c_uint32, 4), ('max_srq_per_vf', ctypes.c_uint32, 8), ('max_cq_per_vf', ctypes.c_uint32, 12), ('max_gid_per_vf', ctypes.c_uint32, 16)])
class struct_bnxt_qplib_tqm_ctx(c.Struct): pass
class struct_bnxt_qplib_ctx(c.Struct): pass
class struct_bnxt_qplib_pd(c.Struct): pass
class struct_bnxt_qplib_dev_attr(c.Struct): pass
enum_bnxt_hwrm_ctx_flags: dict[int, str] = {(BNXT_HWRM_INTERNAL_CTX_OWNED:=0): 'BNXT_HWRM_INTERNAL_CTX_OWNED', (BNXT_HWRM_INTERNAL_RESP_DIRTY:=1): 'BNXT_HWRM_INTERNAL_RESP_DIRTY', (BNXT_HWRM_CTX_SILENT:=2): 'BNXT_HWRM_CTX_SILENT', (BNXT_HWRM_FULL_WAIT:=3): 'BNXT_HWRM_FULL_WAIT'}
class struct_bnxt_hwrm_ctx(c.Struct): pass
enum_bnxt_hwrm_wait_state: dict[int, str] = {(BNXT_HWRM_PENDING:=0): 'BNXT_HWRM_PENDING', (BNXT_HWRM_DEFERRED:=1): 'BNXT_HWRM_DEFERRED', (BNXT_HWRM_COMPLETE:=2): 'BNXT_HWRM_COMPLETE', (BNXT_HWRM_CANCELLED:=3): 'BNXT_HWRM_CANCELLED'}
enum_bnxt_hwrm_chnl: dict[int, str] = {(BNXT_HWRM_CHNL_CHIMP:=0): 'BNXT_HWRM_CHNL_CHIMP', (BNXT_HWRM_CHNL_KONG:=1): 'BNXT_HWRM_CHNL_KONG'}
class struct_bnxt_hwrm_wait_token(c.Struct): pass
CMD_DISCR_TLV_ENCAP = 0x8000
CMD_DISCR_LAST = CMD_DISCR_TLV_ENCAP
TLV_TYPE_HWRM_REQUEST = 0x1
TLV_TYPE_HWRM_RESPONSE = 0x2
TLV_TYPE_ROCE_SP_COMMAND = 0x3
TLV_TYPE_QUERY_ROCE_CC_GEN1 = 0x4
TLV_TYPE_MODIFY_ROCE_CC_GEN1 = 0x5
TLV_TYPE_QUERY_ROCE_CC_GEN2 = 0x6
TLV_TYPE_MODIFY_ROCE_CC_GEN2 = 0x7
TLV_TYPE_QUERY_ROCE_CC_GEN1_EXT = 0x8
TLV_TYPE_MODIFY_ROCE_CC_GEN1_EXT = 0x9
TLV_TYPE_QUERY_ROCE_CC_GEN2_EXT = 0xa
TLV_TYPE_MODIFY_ROCE_CC_GEN2_EXT = 0xb
TLV_TYPE_ENGINE_CKV_ALIAS_ECC_PUBLIC_KEY = 0x8001
TLV_TYPE_ENGINE_CKV_IV = 0x8003
TLV_TYPE_ENGINE_CKV_AUTH_TAG = 0x8004
TLV_TYPE_ENGINE_CKV_CIPHERTEXT = 0x8005
TLV_TYPE_ENGINE_CKV_HOST_ALGORITHMS = 0x8006
TLV_TYPE_ENGINE_CKV_HOST_ECC_PUBLIC_KEY = 0x8007
TLV_TYPE_ENGINE_CKV_ECDSA_SIGNATURE = 0x8008
TLV_TYPE_ENGINE_CKV_FW_ECC_PUBLIC_KEY = 0x8009
TLV_TYPE_ENGINE_CKV_FW_ALGORITHMS = 0x800a
TLV_TYPE_LAST = TLV_TYPE_ENGINE_CKV_FW_ALGORITHMS
TLV_FLAGS_MORE = 0x1
TLV_FLAGS_MORE_LAST = 0x0
TLV_FLAGS_MORE_NOT_LAST = 0x1
TLV_FLAGS_REQUIRED = 0x2
TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
TLV_FLAGS_REQUIRED_LAST = TLV_FLAGS_REQUIRED_YES
SHORT_REQ_SIGNATURE_SHORT_CMD = 0x4321
SHORT_REQ_SIGNATURE_LAST = SHORT_REQ_SIGNATURE_SHORT_CMD
SHORT_REQ_TARGET_ID_DEFAULT = 0x0
SHORT_REQ_TARGET_ID_TOOLS = 0xfffd
SHORT_REQ_TARGET_ID_LAST = SHORT_REQ_TARGET_ID_TOOLS
HWRM_VER_GET = 0x0
HWRM_FUNC_ECHO_RESPONSE = 0xb
HWRM_ERROR_RECOVERY_QCFG = 0xc
HWRM_FUNC_DRV_IF_CHANGE = 0xd
HWRM_FUNC_BUF_UNRGTR = 0xe
HWRM_FUNC_VF_CFG = 0xf
HWRM_RESERVED1 = 0x10
HWRM_FUNC_RESET = 0x11
HWRM_FUNC_GETFID = 0x12
HWRM_FUNC_VF_ALLOC = 0x13
HWRM_FUNC_VF_FREE = 0x14
HWRM_FUNC_QCAPS = 0x15
HWRM_FUNC_QCFG = 0x16
HWRM_FUNC_CFG = 0x17
HWRM_FUNC_QSTATS = 0x18
HWRM_FUNC_CLR_STATS = 0x19
HWRM_FUNC_DRV_UNRGTR = 0x1a
HWRM_FUNC_VF_RESC_FREE = 0x1b
HWRM_FUNC_VF_VNIC_IDS_QUERY = 0x1c
HWRM_FUNC_DRV_RGTR = 0x1d
HWRM_FUNC_DRV_QVER = 0x1e
HWRM_FUNC_BUF_RGTR = 0x1f
HWRM_PORT_PHY_CFG = 0x20
HWRM_PORT_MAC_CFG = 0x21
HWRM_PORT_TS_QUERY = 0x22
HWRM_PORT_QSTATS = 0x23
HWRM_PORT_LPBK_QSTATS = 0x24
HWRM_PORT_CLR_STATS = 0x25
HWRM_PORT_LPBK_CLR_STATS = 0x26
HWRM_PORT_PHY_QCFG = 0x27
HWRM_PORT_MAC_QCFG = 0x28
HWRM_PORT_MAC_PTP_QCFG = 0x29
HWRM_PORT_PHY_QCAPS = 0x2a
HWRM_PORT_PHY_I2C_WRITE = 0x2b
HWRM_PORT_PHY_I2C_READ = 0x2c
HWRM_PORT_LED_CFG = 0x2d
HWRM_PORT_LED_QCFG = 0x2e
HWRM_PORT_LED_QCAPS = 0x2f
HWRM_QUEUE_QPORTCFG = 0x30
HWRM_QUEUE_QCFG = 0x31
HWRM_QUEUE_CFG = 0x32
HWRM_FUNC_VLAN_CFG = 0x33
HWRM_FUNC_VLAN_QCFG = 0x34
HWRM_QUEUE_PFCENABLE_QCFG = 0x35
HWRM_QUEUE_PFCENABLE_CFG = 0x36
HWRM_QUEUE_PRI2COS_QCFG = 0x37
HWRM_QUEUE_PRI2COS_CFG = 0x38
HWRM_QUEUE_COS2BW_QCFG = 0x39
HWRM_QUEUE_COS2BW_CFG = 0x3a
HWRM_QUEUE_DSCP_QCAPS = 0x3b
HWRM_QUEUE_DSCP2PRI_QCFG = 0x3c
HWRM_QUEUE_DSCP2PRI_CFG = 0x3d
HWRM_VNIC_ALLOC = 0x40
HWRM_VNIC_FREE = 0x41
HWRM_VNIC_CFG = 0x42
HWRM_VNIC_QCFG = 0x43
HWRM_VNIC_TPA_CFG = 0x44
HWRM_VNIC_TPA_QCFG = 0x45
HWRM_VNIC_RSS_CFG = 0x46
HWRM_VNIC_RSS_QCFG = 0x47
HWRM_VNIC_PLCMODES_CFG = 0x48
HWRM_VNIC_PLCMODES_QCFG = 0x49
HWRM_VNIC_QCAPS = 0x4a
HWRM_VNIC_UPDATE = 0x4b
HWRM_RING_ALLOC = 0x50
HWRM_RING_FREE = 0x51
HWRM_RING_CMPL_RING_QAGGINT_PARAMS = 0x52
HWRM_RING_CMPL_RING_CFG_AGGINT_PARAMS = 0x53
HWRM_RING_AGGINT_QCAPS = 0x54
HWRM_RING_SCHQ_ALLOC = 0x55
HWRM_RING_SCHQ_CFG = 0x56
HWRM_RING_SCHQ_FREE = 0x57
HWRM_RING_RESET = 0x5e
HWRM_RING_GRP_ALLOC = 0x60
HWRM_RING_GRP_FREE = 0x61
HWRM_RING_CFG = 0x62
HWRM_RING_QCFG = 0x63
HWRM_RESERVED5 = 0x64
HWRM_RESERVED6 = 0x65
HWRM_VNIC_RSS_COS_LB_CTX_ALLOC = 0x70
HWRM_VNIC_RSS_COS_LB_CTX_FREE = 0x71
HWRM_QUEUE_MPLS_QCAPS = 0x80
HWRM_QUEUE_MPLSTC2PRI_QCFG = 0x81
HWRM_QUEUE_MPLSTC2PRI_CFG = 0x82
HWRM_QUEUE_VLANPRI_QCAPS = 0x83
HWRM_QUEUE_VLANPRI2PRI_QCFG = 0x84
HWRM_QUEUE_VLANPRI2PRI_CFG = 0x85
HWRM_QUEUE_GLOBAL_CFG = 0x86
HWRM_QUEUE_GLOBAL_QCFG = 0x87
HWRM_QUEUE_ADPTV_QOS_RX_FEATURE_QCFG = 0x88
HWRM_QUEUE_ADPTV_QOS_RX_FEATURE_CFG = 0x89
HWRM_QUEUE_ADPTV_QOS_TX_FEATURE_QCFG = 0x8a
HWRM_QUEUE_ADPTV_QOS_TX_FEATURE_CFG = 0x8b
HWRM_QUEUE_QCAPS = 0x8c
HWRM_QUEUE_ADPTV_QOS_RX_TUNING_QCFG = 0x8d
HWRM_QUEUE_ADPTV_QOS_RX_TUNING_CFG = 0x8e
HWRM_QUEUE_ADPTV_QOS_TX_TUNING_QCFG = 0x8f
HWRM_CFA_L2_FILTER_ALLOC = 0x90
HWRM_CFA_L2_FILTER_FREE = 0x91
HWRM_CFA_L2_FILTER_CFG = 0x92
HWRM_CFA_L2_SET_RX_MASK = 0x93
HWRM_CFA_VLAN_ANTISPOOF_CFG = 0x94
HWRM_CFA_TUNNEL_FILTER_ALLOC = 0x95
HWRM_CFA_TUNNEL_FILTER_FREE = 0x96
HWRM_CFA_ENCAP_RECORD_ALLOC = 0x97
HWRM_CFA_ENCAP_RECORD_FREE = 0x98
HWRM_CFA_NTUPLE_FILTER_ALLOC = 0x99
HWRM_CFA_NTUPLE_FILTER_FREE = 0x9a
HWRM_CFA_NTUPLE_FILTER_CFG = 0x9b
HWRM_CFA_EM_FLOW_ALLOC = 0x9c
HWRM_CFA_EM_FLOW_FREE = 0x9d
HWRM_CFA_EM_FLOW_CFG = 0x9e
HWRM_TUNNEL_DST_PORT_QUERY = 0xa0
HWRM_TUNNEL_DST_PORT_ALLOC = 0xa1
HWRM_TUNNEL_DST_PORT_FREE = 0xa2
HWRM_QUEUE_ADPTV_QOS_TX_TUNING_CFG = 0xa3
HWRM_STAT_CTX_ENG_QUERY = 0xaf
HWRM_STAT_CTX_ALLOC = 0xb0
HWRM_STAT_CTX_FREE = 0xb1
HWRM_STAT_CTX_QUERY = 0xb2
HWRM_STAT_CTX_CLR_STATS = 0xb3
HWRM_PORT_QSTATS_EXT = 0xb4
HWRM_PORT_PHY_MDIO_WRITE = 0xb5
HWRM_PORT_PHY_MDIO_READ = 0xb6
HWRM_PORT_PHY_MDIO_BUS_ACQUIRE = 0xb7
HWRM_PORT_PHY_MDIO_BUS_RELEASE = 0xb8
HWRM_PORT_QSTATS_EXT_PFC_WD = 0xb9
HWRM_RESERVED7 = 0xba
HWRM_PORT_TX_FIR_CFG = 0xbb
HWRM_PORT_TX_FIR_QCFG = 0xbc
HWRM_PORT_ECN_QSTATS = 0xbd
HWRM_FW_LIVEPATCH_QUERY = 0xbe
HWRM_FW_LIVEPATCH = 0xbf
HWRM_FW_RESET = 0xc0
HWRM_FW_QSTATUS = 0xc1
HWRM_FW_HEALTH_CHECK = 0xc2
HWRM_FW_SYNC = 0xc3
HWRM_FW_STATE_QCAPS = 0xc4
HWRM_FW_STATE_QUIESCE = 0xc5
HWRM_FW_STATE_BACKUP = 0xc6
HWRM_FW_STATE_RESTORE = 0xc7
HWRM_FW_SET_TIME = 0xc8
HWRM_FW_GET_TIME = 0xc9
HWRM_FW_SET_STRUCTURED_DATA = 0xca
HWRM_FW_GET_STRUCTURED_DATA = 0xcb
HWRM_FW_IPC_MAILBOX = 0xcc
HWRM_FW_ECN_CFG = 0xcd
HWRM_FW_ECN_QCFG = 0xce
HWRM_FW_SECURE_CFG = 0xcf
HWRM_EXEC_FWD_RESP = 0xd0
HWRM_REJECT_FWD_RESP = 0xd1
HWRM_FWD_RESP = 0xd2
HWRM_FWD_ASYNC_EVENT_CMPL = 0xd3
HWRM_OEM_CMD = 0xd4
HWRM_PORT_PRBS_TEST = 0xd5
HWRM_PORT_SFP_SIDEBAND_CFG = 0xd6
HWRM_PORT_SFP_SIDEBAND_QCFG = 0xd7
HWRM_FW_STATE_UNQUIESCE = 0xd8
HWRM_PORT_DSC_DUMP = 0xd9
HWRM_PORT_EP_TX_QCFG = 0xda
HWRM_PORT_EP_TX_CFG = 0xdb
HWRM_PORT_CFG = 0xdc
HWRM_PORT_QCFG = 0xdd
HWRM_PORT_MAC_QCAPS = 0xdf
HWRM_TEMP_MONITOR_QUERY = 0xe0
HWRM_REG_POWER_QUERY = 0xe1
HWRM_CORE_FREQUENCY_QUERY = 0xe2
HWRM_REG_POWER_HISTOGRAM = 0xe3
HWRM_MONITOR_PAX_HISTOGRAM_START = 0xe4
HWRM_MONITOR_PAX_HISTOGRAM_COLLECT = 0xe5
HWRM_STAT_QUERY_ROCE_STATS = 0xe6
HWRM_STAT_QUERY_ROCE_STATS_EXT = 0xe7
HWRM_WOL_FILTER_ALLOC = 0xf0
HWRM_WOL_FILTER_FREE = 0xf1
HWRM_WOL_FILTER_QCFG = 0xf2
HWRM_WOL_REASON_QCFG = 0xf3
HWRM_CFA_METER_QCAPS = 0xf4
HWRM_CFA_METER_PROFILE_ALLOC = 0xf5
HWRM_CFA_METER_PROFILE_FREE = 0xf6
HWRM_CFA_METER_PROFILE_CFG = 0xf7
HWRM_CFA_METER_INSTANCE_ALLOC = 0xf8
HWRM_CFA_METER_INSTANCE_FREE = 0xf9
HWRM_CFA_METER_INSTANCE_CFG = 0xfa
HWRM_CFA_VFR_ALLOC = 0xfd
HWRM_CFA_VFR_FREE = 0xfe
HWRM_CFA_VF_PAIR_ALLOC = 0x100
HWRM_CFA_VF_PAIR_FREE = 0x101
HWRM_CFA_VF_PAIR_INFO = 0x102
HWRM_CFA_FLOW_ALLOC = 0x103
HWRM_CFA_FLOW_FREE = 0x104
HWRM_CFA_FLOW_FLUSH = 0x105
HWRM_CFA_FLOW_STATS = 0x106
HWRM_CFA_FLOW_INFO = 0x107
HWRM_CFA_DECAP_FILTER_ALLOC = 0x108
HWRM_CFA_DECAP_FILTER_FREE = 0x109
HWRM_CFA_VLAN_ANTISPOOF_QCFG = 0x10a
HWRM_CFA_REDIRECT_TUNNEL_TYPE_ALLOC = 0x10b
HWRM_CFA_REDIRECT_TUNNEL_TYPE_FREE = 0x10c
HWRM_CFA_PAIR_ALLOC = 0x10d
HWRM_CFA_PAIR_FREE = 0x10e
HWRM_CFA_PAIR_INFO = 0x10f
HWRM_FW_IPC_MSG = 0x110
HWRM_CFA_REDIRECT_TUNNEL_TYPE_INFO = 0x111
HWRM_CFA_REDIRECT_QUERY_TUNNEL_TYPE = 0x112
HWRM_CFA_FLOW_AGING_TIMER_RESET = 0x113
HWRM_CFA_FLOW_AGING_CFG = 0x114
HWRM_CFA_FLOW_AGING_QCFG = 0x115
HWRM_CFA_FLOW_AGING_QCAPS = 0x116
HWRM_CFA_CTX_MEM_RGTR = 0x117
HWRM_CFA_CTX_MEM_UNRGTR = 0x118
HWRM_CFA_CTX_MEM_QCTX = 0x119
HWRM_CFA_CTX_MEM_QCAPS = 0x11a
HWRM_CFA_COUNTER_QCAPS = 0x11b
HWRM_CFA_COUNTER_CFG = 0x11c
HWRM_CFA_COUNTER_QCFG = 0x11d
HWRM_CFA_COUNTER_QSTATS = 0x11e
HWRM_CFA_TCP_FLAG_PROCESS_QCFG = 0x11f
HWRM_CFA_EEM_QCAPS = 0x120
HWRM_CFA_EEM_CFG = 0x121
HWRM_CFA_EEM_QCFG = 0x122
HWRM_CFA_EEM_OP = 0x123
HWRM_CFA_ADV_FLOW_MGNT_QCAPS = 0x124
HWRM_CFA_TFLIB = 0x125
HWRM_CFA_LAG_GROUP_MEMBER_RGTR = 0x126
HWRM_CFA_LAG_GROUP_MEMBER_UNRGTR = 0x127
HWRM_CFA_TLS_FILTER_ALLOC = 0x128
HWRM_CFA_TLS_FILTER_FREE = 0x129
HWRM_CFA_RELEASE_AFM_FUNC = 0x12a
HWRM_ENGINE_CKV_STATUS = 0x12e
HWRM_ENGINE_CKV_CKEK_ADD = 0x12f
HWRM_ENGINE_CKV_CKEK_DELETE = 0x130
HWRM_ENGINE_CKV_KEY_ADD = 0x131
HWRM_ENGINE_CKV_KEY_DELETE = 0x132
HWRM_ENGINE_CKV_FLUSH = 0x133
HWRM_ENGINE_CKV_RNG_GET = 0x134
HWRM_ENGINE_CKV_KEY_GEN = 0x135
HWRM_ENGINE_CKV_KEY_LABEL_CFG = 0x136
HWRM_ENGINE_CKV_KEY_LABEL_QCFG = 0x137
HWRM_ENGINE_QG_CONFIG_QUERY = 0x13c
HWRM_ENGINE_QG_QUERY = 0x13d
HWRM_ENGINE_QG_METER_PROFILE_CONFIG_QUERY = 0x13e
HWRM_ENGINE_QG_METER_PROFILE_QUERY = 0x13f
HWRM_ENGINE_QG_METER_PROFILE_ALLOC = 0x140
HWRM_ENGINE_QG_METER_PROFILE_FREE = 0x141
HWRM_ENGINE_QG_METER_QUERY = 0x142
HWRM_ENGINE_QG_METER_BIND = 0x143
HWRM_ENGINE_QG_METER_UNBIND = 0x144
HWRM_ENGINE_QG_FUNC_BIND = 0x145
HWRM_ENGINE_SG_CONFIG_QUERY = 0x146
HWRM_ENGINE_SG_QUERY = 0x147
HWRM_ENGINE_SG_METER_QUERY = 0x148
HWRM_ENGINE_SG_METER_CONFIG = 0x149
HWRM_ENGINE_SG_QG_BIND = 0x14a
HWRM_ENGINE_QG_SG_UNBIND = 0x14b
HWRM_ENGINE_CONFIG_QUERY = 0x154
HWRM_ENGINE_STATS_CONFIG = 0x155
HWRM_ENGINE_STATS_CLEAR = 0x156
HWRM_ENGINE_STATS_QUERY = 0x157
HWRM_ENGINE_STATS_QUERY_CONTINUOUS_ERROR = 0x158
HWRM_ENGINE_RQ_ALLOC = 0x15e
HWRM_ENGINE_RQ_FREE = 0x15f
HWRM_ENGINE_CQ_ALLOC = 0x160
HWRM_ENGINE_CQ_FREE = 0x161
HWRM_ENGINE_NQ_ALLOC = 0x162
HWRM_ENGINE_NQ_FREE = 0x163
HWRM_ENGINE_ON_DIE_RQE_CREDITS = 0x164
HWRM_ENGINE_FUNC_QCFG = 0x165
HWRM_FUNC_RESOURCE_QCAPS = 0x190
HWRM_FUNC_VF_RESOURCE_CFG = 0x191
HWRM_FUNC_BACKING_STORE_QCAPS = 0x192
HWRM_FUNC_BACKING_STORE_CFG = 0x193
HWRM_FUNC_BACKING_STORE_QCFG = 0x194
HWRM_FUNC_VF_BW_CFG = 0x195
HWRM_FUNC_VF_BW_QCFG = 0x196
HWRM_FUNC_HOST_PF_IDS_QUERY = 0x197
HWRM_FUNC_QSTATS_EXT = 0x198
HWRM_STAT_EXT_CTX_QUERY = 0x199
HWRM_FUNC_SPD_CFG = 0x19a
HWRM_FUNC_SPD_QCFG = 0x19b
HWRM_FUNC_PTP_PIN_QCFG = 0x19c
HWRM_FUNC_PTP_PIN_CFG = 0x19d
HWRM_FUNC_PTP_CFG = 0x19e
HWRM_FUNC_PTP_TS_QUERY = 0x19f
HWRM_FUNC_PTP_EXT_CFG = 0x1a0
HWRM_FUNC_PTP_EXT_QCFG = 0x1a1
HWRM_FUNC_KEY_CTX_ALLOC = 0x1a2
HWRM_FUNC_BACKING_STORE_CFG_V2 = 0x1a3
HWRM_FUNC_BACKING_STORE_QCFG_V2 = 0x1a4
HWRM_FUNC_DBR_PACING_CFG = 0x1a5
HWRM_FUNC_DBR_PACING_QCFG = 0x1a6
HWRM_FUNC_DBR_PACING_BROADCAST_EVENT = 0x1a7
HWRM_FUNC_BACKING_STORE_QCAPS_V2 = 0x1a8
HWRM_FUNC_DBR_PACING_NQLIST_QUERY = 0x1a9
HWRM_FUNC_DBR_RECOVERY_COMPLETED = 0x1aa
HWRM_FUNC_SYNCE_CFG = 0x1ab
HWRM_FUNC_SYNCE_QCFG = 0x1ac
HWRM_FUNC_KEY_CTX_FREE = 0x1ad
HWRM_FUNC_LAG_MODE_CFG = 0x1ae
HWRM_FUNC_LAG_MODE_QCFG = 0x1af
HWRM_FUNC_LAG_CREATE = 0x1b0
HWRM_FUNC_LAG_UPDATE = 0x1b1
HWRM_FUNC_LAG_FREE = 0x1b2
HWRM_FUNC_LAG_QCFG = 0x1b3
HWRM_FUNC_TTX_PACING_RATE_PROF_QUERY = 0x1c3
HWRM_FUNC_TTX_PACING_RATE_QUERY = 0x1c4
HWRM_SELFTEST_QLIST = 0x200
HWRM_SELFTEST_EXEC = 0x201
HWRM_SELFTEST_IRQ = 0x202
HWRM_SELFTEST_RETRIEVE_SERDES_DATA = 0x203
HWRM_PCIE_QSTATS = 0x204
HWRM_MFG_FRU_WRITE_CONTROL = 0x205
HWRM_MFG_TIMERS_QUERY = 0x206
HWRM_MFG_OTP_CFG = 0x207
HWRM_MFG_OTP_QCFG = 0x208
HWRM_MFG_HDMA_TEST = 0x209
HWRM_MFG_FRU_EEPROM_WRITE = 0x20a
HWRM_MFG_FRU_EEPROM_READ = 0x20b
HWRM_MFG_SOC_IMAGE = 0x20c
HWRM_MFG_SOC_QSTATUS = 0x20d
HWRM_MFG_PARAM_CRITICAL_DATA_FINALIZE = 0x20e
HWRM_MFG_PARAM_CRITICAL_DATA_READ = 0x20f
HWRM_MFG_PARAM_CRITICAL_DATA_HEALTH = 0x210
HWRM_MFG_PRVSN_EXPORT_CSR = 0x211
HWRM_MFG_PRVSN_IMPORT_CERT = 0x212
HWRM_MFG_PRVSN_GET_STATE = 0x213
HWRM_MFG_GET_NVM_MEASUREMENT = 0x214
HWRM_MFG_PSOC_QSTATUS = 0x215
HWRM_MFG_SELFTEST_QLIST = 0x216
HWRM_MFG_SELFTEST_EXEC = 0x217
HWRM_STAT_GENERIC_QSTATS = 0x218
HWRM_MFG_PRVSN_EXPORT_CERT = 0x219
HWRM_STAT_DB_ERROR_QSTATS = 0x21a
HWRM_MFG_TESTS = 0x21b
HWRM_MFG_WRITE_CERT_NVM = 0x21c
HWRM_PORT_POE_CFG = 0x230
HWRM_PORT_POE_QCFG = 0x231
HWRM_PORT_PHY_FDRSTAT = 0x232
HWRM_UDCC_QCAPS = 0x258
HWRM_UDCC_CFG = 0x259
HWRM_UDCC_QCFG = 0x25a
HWRM_UDCC_SESSION_CFG = 0x25b
HWRM_UDCC_SESSION_QCFG = 0x25c
HWRM_UDCC_SESSION_QUERY = 0x25d
HWRM_UDCC_COMP_CFG = 0x25e
HWRM_UDCC_COMP_QCFG = 0x25f
HWRM_UDCC_COMP_QUERY = 0x260
HWRM_QUEUE_PFCWD_TIMEOUT_QCAPS = 0x261
HWRM_QUEUE_PFCWD_TIMEOUT_CFG = 0x262
HWRM_QUEUE_PFCWD_TIMEOUT_QCFG = 0x263
HWRM_QUEUE_ADPTV_QOS_RX_QCFG = 0x264
HWRM_QUEUE_ADPTV_QOS_TX_QCFG = 0x265
HWRM_TF = 0x2bc
HWRM_TF_VERSION_GET = 0x2bd
HWRM_TF_SESSION_OPEN = 0x2c6
HWRM_TF_SESSION_REGISTER = 0x2c8
HWRM_TF_SESSION_UNREGISTER = 0x2c9
HWRM_TF_SESSION_CLOSE = 0x2ca
HWRM_TF_SESSION_QCFG = 0x2cb
HWRM_TF_SESSION_RESC_QCAPS = 0x2cc
HWRM_TF_SESSION_RESC_ALLOC = 0x2cd
HWRM_TF_SESSION_RESC_FREE = 0x2ce
HWRM_TF_SESSION_RESC_FLUSH = 0x2cf
HWRM_TF_SESSION_RESC_INFO = 0x2d0
HWRM_TF_SESSION_HOTUP_STATE_SET = 0x2d1
HWRM_TF_SESSION_HOTUP_STATE_GET = 0x2d2
HWRM_TF_TBL_TYPE_GET = 0x2da
HWRM_TF_TBL_TYPE_SET = 0x2db
HWRM_TF_TBL_TYPE_BULK_GET = 0x2dc
HWRM_TF_EM_INSERT = 0x2ea
HWRM_TF_EM_DELETE = 0x2eb
HWRM_TF_EM_HASH_INSERT = 0x2ec
HWRM_TF_EM_MOVE = 0x2ed
HWRM_TF_TCAM_SET = 0x2f8
HWRM_TF_TCAM_GET = 0x2f9
HWRM_TF_TCAM_MOVE = 0x2fa
HWRM_TF_TCAM_FREE = 0x2fb
HWRM_TF_GLOBAL_CFG_SET = 0x2fc
HWRM_TF_GLOBAL_CFG_GET = 0x2fd
HWRM_TF_IF_TBL_SET = 0x2fe
HWRM_TF_IF_TBL_GET = 0x2ff
HWRM_TF_RESC_USAGE_SET = 0x300
HWRM_TF_RESC_USAGE_QUERY = 0x301
HWRM_TF_TBL_TYPE_ALLOC = 0x302
HWRM_TF_TBL_TYPE_FREE = 0x303
HWRM_TFC_TBL_SCOPE_QCAPS = 0x380
HWRM_TFC_TBL_SCOPE_ID_ALLOC = 0x381
HWRM_TFC_TBL_SCOPE_CONFIG = 0x382
HWRM_TFC_TBL_SCOPE_DECONFIG = 0x383
HWRM_TFC_TBL_SCOPE_FID_ADD = 0x384
HWRM_TFC_TBL_SCOPE_FID_REM = 0x385
HWRM_TFC_TBL_SCOPE_POOL_ALLOC = 0x386
HWRM_TFC_TBL_SCOPE_POOL_FREE = 0x387
HWRM_TFC_SESSION_ID_ALLOC = 0x388
HWRM_TFC_SESSION_FID_ADD = 0x389
HWRM_TFC_SESSION_FID_REM = 0x38a
HWRM_TFC_IDENT_ALLOC = 0x38b
HWRM_TFC_IDENT_FREE = 0x38c
HWRM_TFC_IDX_TBL_ALLOC = 0x38d
HWRM_TFC_IDX_TBL_ALLOC_SET = 0x38e
HWRM_TFC_IDX_TBL_SET = 0x38f
HWRM_TFC_IDX_TBL_GET = 0x390
HWRM_TFC_IDX_TBL_FREE = 0x391
HWRM_TFC_GLOBAL_ID_ALLOC = 0x392
HWRM_TFC_TCAM_SET = 0x393
HWRM_TFC_TCAM_GET = 0x394
HWRM_TFC_TCAM_ALLOC = 0x395
HWRM_TFC_TCAM_ALLOC_SET = 0x396
HWRM_TFC_TCAM_FREE = 0x397
HWRM_TFC_IF_TBL_SET = 0x398
HWRM_TFC_IF_TBL_GET = 0x399
HWRM_TFC_TBL_SCOPE_CONFIG_GET = 0x39a
HWRM_TFC_RESC_USAGE_QUERY = 0x39b
HWRM_TFC_GLOBAL_ID_FREE = 0x39c
HWRM_TFC_TCAM_PRI_UPDATE = 0x39d
HWRM_TFC_HOT_UPGRADE_PROCESS = 0x3a0
HWRM_SV = 0x400
HWRM_DBG_SERDES_TEST = 0xff0e
HWRM_DBG_LOG_BUFFER_FLUSH = 0xff0f
HWRM_DBG_READ_DIRECT = 0xff10
HWRM_DBG_READ_INDIRECT = 0xff11
HWRM_DBG_WRITE_DIRECT = 0xff12
HWRM_DBG_WRITE_INDIRECT = 0xff13
HWRM_DBG_DUMP = 0xff14
HWRM_DBG_ERASE_NVM = 0xff15
HWRM_DBG_CFG = 0xff16
HWRM_DBG_COREDUMP_LIST = 0xff17
HWRM_DBG_COREDUMP_INITIATE = 0xff18
HWRM_DBG_COREDUMP_RETRIEVE = 0xff19
HWRM_DBG_FW_CLI = 0xff1a
HWRM_DBG_I2C_CMD = 0xff1b
HWRM_DBG_RING_INFO_GET = 0xff1c
HWRM_DBG_CRASHDUMP_HEADER = 0xff1d
HWRM_DBG_CRASHDUMP_ERASE = 0xff1e
HWRM_DBG_DRV_TRACE = 0xff1f
HWRM_DBG_QCAPS = 0xff20
HWRM_DBG_QCFG = 0xff21
HWRM_DBG_CRASHDUMP_MEDIUM_CFG = 0xff22
HWRM_DBG_USEQ_ALLOC = 0xff23
HWRM_DBG_USEQ_FREE = 0xff24
HWRM_DBG_USEQ_FLUSH = 0xff25
HWRM_DBG_USEQ_QCAPS = 0xff26
HWRM_DBG_USEQ_CW_CFG = 0xff27
HWRM_DBG_USEQ_SCHED_CFG = 0xff28
HWRM_DBG_USEQ_RUN = 0xff29
HWRM_DBG_USEQ_DELIVERY_REQ = 0xff2a
HWRM_DBG_USEQ_RESP_HDR = 0xff2b
HWRM_DBG_COREDUMP_CAPTURE = 0xff2c
HWRM_DBG_PTRACE = 0xff2d
HWRM_DBG_SIM_CABLE_STATE = 0xff2e
HWRM_DBG_TOKEN_QUERY_AUTH_IDS = 0xff2f
HWRM_DBG_TOKEN_CFG = 0xff30
HWRM_NVM_GET_VPD_FIELD_INFO = 0xffea
HWRM_NVM_SET_VPD_FIELD_INFO = 0xffeb
HWRM_NVM_DEFRAG = 0xffec
HWRM_NVM_REQ_ARBITRATION = 0xffed
HWRM_NVM_FACTORY_DEFAULTS = 0xffee
HWRM_NVM_VALIDATE_OPTION = 0xffef
HWRM_NVM_FLUSH = 0xfff0
HWRM_NVM_GET_VARIABLE = 0xfff1
HWRM_NVM_SET_VARIABLE = 0xfff2
HWRM_NVM_INSTALL_UPDATE = 0xfff3
HWRM_NVM_MODIFY = 0xfff4
HWRM_NVM_VERIFY_UPDATE = 0xfff5
HWRM_NVM_GET_DEV_INFO = 0xfff6
HWRM_NVM_ERASE_DIR_ENTRY = 0xfff7
HWRM_NVM_MOD_DIR_ENTRY = 0xfff8
HWRM_NVM_FIND_DIR_ENTRY = 0xfff9
HWRM_NVM_GET_DIR_ENTRIES = 0xfffa
HWRM_NVM_GET_DIR_INFO = 0xfffb
HWRM_NVM_RAW_DUMP = 0xfffc
HWRM_NVM_READ = 0xfffd
HWRM_NVM_WRITE = 0xfffe
HWRM_NVM_RAW_WRITE_BLK = 0xffff
HWRM_LAST = HWRM_NVM_RAW_WRITE_BLK
HWRM_ERR_CODE_SUCCESS = 0x0
HWRM_ERR_CODE_FAIL = 0x1
HWRM_ERR_CODE_INVALID_PARAMS = 0x2
HWRM_ERR_CODE_RESOURCE_ACCESS_DENIED = 0x3
HWRM_ERR_CODE_RESOURCE_ALLOC_ERROR = 0x4
HWRM_ERR_CODE_INVALID_FLAGS = 0x5
HWRM_ERR_CODE_INVALID_ENABLES = 0x6
HWRM_ERR_CODE_UNSUPPORTED_TLV = 0x7
HWRM_ERR_CODE_NO_BUFFER = 0x8
HWRM_ERR_CODE_UNSUPPORTED_OPTION_ERR = 0x9
HWRM_ERR_CODE_HOT_RESET_PROGRESS = 0xa
HWRM_ERR_CODE_HOT_RESET_FAIL = 0xb
HWRM_ERR_CODE_NO_FLOW_COUNTER_DURING_ALLOC = 0xc
HWRM_ERR_CODE_KEY_HASH_COLLISION = 0xd
HWRM_ERR_CODE_KEY_ALREADY_EXISTS = 0xe
HWRM_ERR_CODE_HWRM_ERROR = 0xf
HWRM_ERR_CODE_BUSY = 0x10
HWRM_ERR_CODE_RESOURCE_LOCKED = 0x11
HWRM_ERR_CODE_PF_UNAVAILABLE = 0x12
HWRM_ERR_CODE_ENTITY_NOT_PRESENT = 0x13
HWRM_ERR_CODE_SECURE_SOC_ERROR = 0x14
HWRM_ERR_CODE_TLV_ENCAPSULATED_RESPONSE = 0x8000
HWRM_ERR_CODE_UNKNOWN_ERR = 0xfffe
HWRM_ERR_CODE_CMD_NOT_SUPPORTED = 0xffff
HWRM_ERR_CODE_LAST = HWRM_ERR_CODE_CMD_NOT_SUPPORTED
HWRM_MAX_REQ_LEN = 128
HWRM_MAX_RESP_LEN = 704
HW_HASH_INDEX_SIZE = 0x80
HW_HASH_KEY_SIZE = 40
HWRM_RESP_VALID_KEY = 1
HWRM_TARGET_ID_BONO = 0xFFF8
HWRM_TARGET_ID_KONG = 0xFFF9
HWRM_TARGET_ID_APE = 0xFFFA
HWRM_TARGET_ID_TOOLS = 0xFFFD
HWRM_VERSION_MAJOR = 1
HWRM_VERSION_MINOR = 10
HWRM_VERSION_UPDATE = 3
HWRM_VERSION_RSVD = 133
HWRM_VERSION_STR = "1.10.3.133"
VER_GET_RESP_DEV_CAPS_CFG_SECURE_FW_UPD_SUPPORTED = 0x1
VER_GET_RESP_DEV_CAPS_CFG_FW_DCBX_AGENT_SUPPORTED = 0x2
VER_GET_RESP_DEV_CAPS_CFG_SHORT_CMD_SUPPORTED = 0x4
VER_GET_RESP_DEV_CAPS_CFG_SHORT_CMD_REQUIRED = 0x8
VER_GET_RESP_DEV_CAPS_CFG_KONG_MB_CHNL_SUPPORTED = 0x10
VER_GET_RESP_DEV_CAPS_CFG_FLOW_HANDLE_64BIT_SUPPORTED = 0x20
VER_GET_RESP_DEV_CAPS_CFG_L2_FILTER_TYPES_ROCE_OR_L2_SUPPORTED = 0x40
VER_GET_RESP_DEV_CAPS_CFG_VIRTIO_VSWITCH_OFFLOAD_SUPPORTED = 0x80
VER_GET_RESP_DEV_CAPS_CFG_TRUSTED_VF_SUPPORTED = 0x100
VER_GET_RESP_DEV_CAPS_CFG_FLOW_AGING_SUPPORTED = 0x200
VER_GET_RESP_DEV_CAPS_CFG_ADV_FLOW_COUNTERS_SUPPORTED = 0x400
VER_GET_RESP_DEV_CAPS_CFG_CFA_EEM_SUPPORTED = 0x800
VER_GET_RESP_DEV_CAPS_CFG_CFA_ADV_FLOW_MGNT_SUPPORTED = 0x1000
VER_GET_RESP_DEV_CAPS_CFG_CFA_TFLIB_SUPPORTED = 0x2000
VER_GET_RESP_DEV_CAPS_CFG_CFA_TRUFLOW_SUPPORTED = 0x4000
VER_GET_RESP_DEV_CAPS_CFG_SECURE_BOOT_CAPABLE = 0x8000
VER_GET_RESP_DEV_CAPS_CFG_SECURE_SOC_CAPABLE = 0x10000
VER_GET_RESP_DEV_CAPS_CFG_DEBUG_TOKEN_SUPPORTED = 0x20000
VER_GET_RESP_CHIP_PLATFORM_TYPE_ASIC = 0x0
VER_GET_RESP_CHIP_PLATFORM_TYPE_FPGA = 0x1
VER_GET_RESP_CHIP_PLATFORM_TYPE_PALLADIUM = 0x2
VER_GET_RESP_CHIP_PLATFORM_TYPE_LAST = VER_GET_RESP_CHIP_PLATFORM_TYPE_PALLADIUM
VER_GET_RESP_FLAGS_DEV_NOT_RDY = 0x1
VER_GET_RESP_FLAGS_EXT_VER_AVAIL = 0x2
VER_GET_RESP_FLAGS_DEV_NOT_RDY_BACKING_STORE = 0x4
EJECT_CMPL_TYPE_MASK = 0x3f
EJECT_CMPL_TYPE_SFT = 0
EJECT_CMPL_TYPE_STAT_EJECT = 0x1a
EJECT_CMPL_TYPE_LAST = EJECT_CMPL_TYPE_STAT_EJECT
EJECT_CMPL_FLAGS_MASK = 0xffc0
EJECT_CMPL_FLAGS_SFT = 6
EJECT_CMPL_FLAGS_ERROR = 0x40
EJECT_CMPL_V = 0x1
EJECT_CMPL_ERRORS_MASK = 0xfffe
EJECT_CMPL_ERRORS_SFT = 1
EJECT_CMPL_ERRORS_BUFFER_ERROR_MASK = 0xe
EJECT_CMPL_ERRORS_BUFFER_ERROR_SFT = 1
EJECT_CMPL_ERRORS_BUFFER_ERROR_NO_BUFFER = (0x0 << 1)
EJECT_CMPL_ERRORS_BUFFER_ERROR_DID_NOT_FIT = (0x1 << 1)
EJECT_CMPL_ERRORS_BUFFER_ERROR_BAD_FORMAT = (0x3 << 1)
EJECT_CMPL_ERRORS_BUFFER_ERROR_FLUSH = (0x5 << 1)
EJECT_CMPL_ERRORS_BUFFER_ERROR_LAST = EJECT_CMPL_ERRORS_BUFFER_ERROR_FLUSH
CMPL_TYPE_MASK = 0x3f
CMPL_TYPE_SFT = 0
CMPL_TYPE_HWRM_DONE = 0x20
CMPL_TYPE_LAST = CMPL_TYPE_HWRM_DONE
CMPL_V = 0x1
FWD_REQ_CMPL_TYPE_MASK = 0x3f
FWD_REQ_CMPL_TYPE_SFT = 0
FWD_REQ_CMPL_TYPE_HWRM_FWD_REQ = 0x22
FWD_REQ_CMPL_TYPE_LAST = FWD_REQ_CMPL_TYPE_HWRM_FWD_REQ
FWD_REQ_CMPL_REQ_LEN_MASK = 0xffc0
FWD_REQ_CMPL_REQ_LEN_SFT = 6
FWD_REQ_CMPL_V = 0x1
FWD_REQ_CMPL_REQ_BUF_ADDR_MASK = 0xfffffffe
FWD_REQ_CMPL_REQ_BUF_ADDR_SFT = 1
FWD_RESP_CMPL_TYPE_MASK = 0x3f
FWD_RESP_CMPL_TYPE_SFT = 0
FWD_RESP_CMPL_TYPE_HWRM_FWD_RESP = 0x24
FWD_RESP_CMPL_TYPE_LAST = FWD_RESP_CMPL_TYPE_HWRM_FWD_RESP
FWD_RESP_CMPL_V = 0x1
FWD_RESP_CMPL_RESP_BUF_ADDR_MASK = 0xfffffffe
FWD_RESP_CMPL_RESP_BUF_ADDR_SFT = 1
ASYNC_EVENT_CMPL_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_TYPE_SFT = 0
ASYNC_EVENT_CMPL_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_TYPE_LAST = ASYNC_EVENT_CMPL_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_EVENT_ID_LINK_STATUS_CHANGE = 0x0
ASYNC_EVENT_CMPL_EVENT_ID_LINK_MTU_CHANGE = 0x1
ASYNC_EVENT_CMPL_EVENT_ID_LINK_SPEED_CHANGE = 0x2
ASYNC_EVENT_CMPL_EVENT_ID_DCB_CONFIG_CHANGE = 0x3
ASYNC_EVENT_CMPL_EVENT_ID_PORT_CONN_NOT_ALLOWED = 0x4
ASYNC_EVENT_CMPL_EVENT_ID_LINK_SPEED_CFG_NOT_ALLOWED = 0x5
ASYNC_EVENT_CMPL_EVENT_ID_LINK_SPEED_CFG_CHANGE = 0x6
ASYNC_EVENT_CMPL_EVENT_ID_PORT_PHY_CFG_CHANGE = 0x7
ASYNC_EVENT_CMPL_EVENT_ID_RESET_NOTIFY = 0x8
ASYNC_EVENT_CMPL_EVENT_ID_ERROR_RECOVERY = 0x9
ASYNC_EVENT_CMPL_EVENT_ID_RING_MONITOR_MSG = 0xa
ASYNC_EVENT_CMPL_EVENT_ID_FUNC_DRVR_UNLOAD = 0x10
ASYNC_EVENT_CMPL_EVENT_ID_FUNC_DRVR_LOAD = 0x11
ASYNC_EVENT_CMPL_EVENT_ID_FUNC_FLR_PROC_CMPLT = 0x12
ASYNC_EVENT_CMPL_EVENT_ID_PF_DRVR_UNLOAD = 0x20
ASYNC_EVENT_CMPL_EVENT_ID_PF_DRVR_LOAD = 0x21
ASYNC_EVENT_CMPL_EVENT_ID_VF_FLR = 0x30
ASYNC_EVENT_CMPL_EVENT_ID_VF_MAC_ADDR_CHANGE = 0x31
ASYNC_EVENT_CMPL_EVENT_ID_PF_VF_COMM_STATUS_CHANGE = 0x32
ASYNC_EVENT_CMPL_EVENT_ID_VF_CFG_CHANGE = 0x33
ASYNC_EVENT_CMPL_EVENT_ID_LLFC_PFC_CHANGE = 0x34
ASYNC_EVENT_CMPL_EVENT_ID_DEFAULT_VNIC_CHANGE = 0x35
ASYNC_EVENT_CMPL_EVENT_ID_HW_FLOW_AGED = 0x36
ASYNC_EVENT_CMPL_EVENT_ID_DEBUG_NOTIFICATION = 0x37
ASYNC_EVENT_CMPL_EVENT_ID_EEM_CACHE_FLUSH_REQ = 0x38
ASYNC_EVENT_CMPL_EVENT_ID_EEM_CACHE_FLUSH_DONE = 0x39
ASYNC_EVENT_CMPL_EVENT_ID_TCP_FLAG_ACTION_CHANGE = 0x3a
ASYNC_EVENT_CMPL_EVENT_ID_EEM_FLOW_ACTIVE = 0x3b
ASYNC_EVENT_CMPL_EVENT_ID_EEM_CFG_CHANGE = 0x3c
ASYNC_EVENT_CMPL_EVENT_ID_TFLIB_DEFAULT_VNIC_CHANGE = 0x3d
ASYNC_EVENT_CMPL_EVENT_ID_TFLIB_LINK_STATUS_CHANGE = 0x3e
ASYNC_EVENT_CMPL_EVENT_ID_QUIESCE_DONE = 0x3f
ASYNC_EVENT_CMPL_EVENT_ID_DEFERRED_RESPONSE = 0x40
ASYNC_EVENT_CMPL_EVENT_ID_PFC_WATCHDOG_CFG_CHANGE = 0x41
ASYNC_EVENT_CMPL_EVENT_ID_ECHO_REQUEST = 0x42
ASYNC_EVENT_CMPL_EVENT_ID_PHC_UPDATE = 0x43
ASYNC_EVENT_CMPL_EVENT_ID_PPS_TIMESTAMP = 0x44
ASYNC_EVENT_CMPL_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_EVENT_ID_DOORBELL_PACING_THRESHOLD = 0x46
ASYNC_EVENT_CMPL_EVENT_ID_RSS_CHANGE = 0x47
ASYNC_EVENT_CMPL_EVENT_ID_DOORBELL_PACING_NQ_UPDATE = 0x48
ASYNC_EVENT_CMPL_EVENT_ID_HW_DOORBELL_RECOVERY_READ_ERROR = 0x49
ASYNC_EVENT_CMPL_EVENT_ID_CTX_ERROR = 0x4a
ASYNC_EVENT_CMPL_EVENT_ID_UDCC_SESSION_CHANGE = 0x4b
ASYNC_EVENT_CMPL_EVENT_ID_DBG_BUF_PRODUCER = 0x4c
ASYNC_EVENT_CMPL_EVENT_ID_PEER_MMAP_CHANGE = 0x4d
ASYNC_EVENT_CMPL_EVENT_ID_REPRESENTOR_PAIR_CHANGE = 0x4e
ASYNC_EVENT_CMPL_EVENT_ID_VF_STAT_CHANGE = 0x4f
ASYNC_EVENT_CMPL_EVENT_ID_HOST_COREDUMP = 0x50
ASYNC_EVENT_CMPL_EVENT_ID_ADPTV_QOS = 0x51
ASYNC_EVENT_CMPL_EVENT_ID_MAX_RGTR_EVENT_ID = 0x52
ASYNC_EVENT_CMPL_EVENT_ID_FW_TRACE_MSG = 0xfe
ASYNC_EVENT_CMPL_EVENT_ID_HWRM_ERROR = 0xff
ASYNC_EVENT_CMPL_EVENT_ID_LAST = ASYNC_EVENT_CMPL_EVENT_ID_HWRM_ERROR
ASYNC_EVENT_CMPL_V = 0x1
ASYNC_EVENT_CMPL_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_TYPE_LAST = ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_ID_LINK_STATUS_CHANGE = 0x0
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_ID_LINK_STATUS_CHANGE
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_V = 0x1
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_LINK_CHANGE = 0x1
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_LINK_CHANGE_DOWN = 0x0
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_LINK_CHANGE_UP = 0x1
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_LINK_CHANGE_LAST = ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_LINK_CHANGE_UP
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_PORT_MASK = 0xe
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_PORT_SFT = 1
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_PORT_ID_MASK = 0xffff0
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_PORT_ID_SFT = 4
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_PF_ID_MASK = 0xff00000
ASYNC_EVENT_CMPL_LINK_STATUS_CHANGE_EVENT_DATA1_PF_ID_SFT = 20
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_TYPE_SFT = 0
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_TYPE_LAST = ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_ID_PORT_CONN_NOT_ALLOWED = 0x4
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_ID_LAST = ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_ID_PORT_CONN_NOT_ALLOWED
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_V = 0x1
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_PORT_ID_MASK = 0xffff
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_PORT_ID_SFT = 0
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_MASK = 0xff0000
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_SFT = 16
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_NONE = (0x0 << 16)
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_DISABLETX = (0x1 << 16)
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_WARNINGMSG = (0x2 << 16)
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_PWRDOWN = (0x3 << 16)
ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_LAST = ASYNC_EVENT_CMPL_PORT_CONN_NOT_ALLOWED_EVENT_DATA1_ENFORCEMENT_POLICY_PWRDOWN
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_TYPE_LAST = ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_ID_LINK_SPEED_CFG_CHANGE = 0x6
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_ID_LINK_SPEED_CFG_CHANGE
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_V = 0x1
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_DATA1_PORT_ID_MASK = 0xffff
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_DATA1_PORT_ID_SFT = 0
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_DATA1_SUPPORTED_LINK_SPEEDS_CHANGE = 0x10000
ASYNC_EVENT_CMPL_LINK_SPEED_CFG_CHANGE_EVENT_DATA1_ILLEGAL_LINK_SPEED_CFG = 0x20000
ASYNC_EVENT_CMPL_RESET_NOTIFY_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_RESET_NOTIFY_TYPE_SFT = 0
ASYNC_EVENT_CMPL_RESET_NOTIFY_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_RESET_NOTIFY_TYPE_LAST = ASYNC_EVENT_CMPL_RESET_NOTIFY_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_ID_RESET_NOTIFY = 0x8
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_ID_LAST = ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_ID_RESET_NOTIFY
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA2_FW_STATUS_CODE_MASK = 0xffff
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA2_FW_STATUS_CODE_SFT = 0
ASYNC_EVENT_CMPL_RESET_NOTIFY_V = 0x1
ASYNC_EVENT_CMPL_RESET_NOTIFY_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_RESET_NOTIFY_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DRIVER_ACTION_MASK = 0xff
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DRIVER_ACTION_SFT = 0
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DRIVER_ACTION_DRIVER_STOP_TX_QUEUE = 0x1
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DRIVER_ACTION_DRIVER_IFDOWN = 0x2
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DRIVER_ACTION_LAST = ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DRIVER_ACTION_DRIVER_IFDOWN
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_MASK = 0xff00
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_SFT = 8
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_MANAGEMENT_RESET_REQUEST = (0x1 << 8)
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_FW_EXCEPTION_FATAL = (0x2 << 8)
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_FW_EXCEPTION_NON_FATAL = (0x3 << 8)
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_FAST_RESET = (0x4 << 8)
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_FW_ACTIVATION = (0x5 << 8)
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_LAST = ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_REASON_CODE_FW_ACTIVATION
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DELAY_IN_100MS_TICKS_MASK = 0xffff0000
ASYNC_EVENT_CMPL_RESET_NOTIFY_EVENT_DATA1_DELAY_IN_100MS_TICKS_SFT = 16
ASYNC_EVENT_CMPL_ERROR_RECOVERY_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_RECOVERY_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_RECOVERY_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_RECOVERY_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_RECOVERY_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_ID_ERROR_RECOVERY = 0x9
ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_ID_ERROR_RECOVERY
ASYNC_EVENT_CMPL_ERROR_RECOVERY_V = 0x1
ASYNC_EVENT_CMPL_ERROR_RECOVERY_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_RECOVERY_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_SFT = 0
ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_MASTER_FUNC = 0x1
ASYNC_EVENT_CMPL_ERROR_RECOVERY_EVENT_DATA1_FLAGS_RECOVERY_ENABLED = 0x2
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_TYPE_SFT = 0
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_TYPE_LAST = ASYNC_EVENT_CMPL_RING_MONITOR_MSG_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_ID_RING_MONITOR_MSG = 0xa
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_ID_LAST = ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_ID_RING_MONITOR_MSG
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_SFT = 0
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_TX = 0x0
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_RX = 0x1
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_CMPL = 0x2
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_LAST = ASYNC_EVENT_CMPL_RING_MONITOR_MSG_EVENT_DATA2_DISABLE_RING_TYPE_CMPL
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_V = 0x1
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_RING_MONITOR_MSG_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_TYPE_LAST = ASYNC_EVENT_CMPL_VF_CFG_CHANGE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_ID_VF_CFG_CHANGE = 0x33
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_ID_VF_CFG_CHANGE
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA2_VF_ID_MASK = 0xffff
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA2_VF_ID_SFT = 0
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_V = 0x1
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA1_MTU_CHANGE = 0x1
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA1_MRU_CHANGE = 0x2
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA1_DFLT_MAC_ADDR_CHANGE = 0x4
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA1_DFLT_VLAN_CHANGE = 0x8
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA1_TRUSTED_VF_CFG_CHANGE = 0x10
ASYNC_EVENT_CMPL_VF_CFG_CHANGE_EVENT_DATA1_TF_OWNERSHIP_RELEASE = 0x20
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_TYPE_LAST = ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_UNUSED1_MASK = 0xffc0
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_UNUSED1_SFT = 6
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_ID_ALLOC_FREE_NOTIFICATION = 0x35
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_ID_ALLOC_FREE_NOTIFICATION
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_V = 0x1
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_DEF_VNIC_STATE_MASK = 0x3
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_DEF_VNIC_STATE_SFT = 0
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_DEF_VNIC_STATE_DEF_VNIC_ALLOC = 0x1
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_DEF_VNIC_STATE_DEF_VNIC_FREE = 0x2
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_DEF_VNIC_STATE_LAST = ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_DEF_VNIC_STATE_DEF_VNIC_FREE
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_PF_ID_MASK = 0x3fc
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_PF_ID_SFT = 2
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_VF_ID_MASK = 0x3fffc00
ASYNC_EVENT_CMPL_DEFAULT_VNIC_CHANGE_EVENT_DATA1_VF_ID_SFT = 10
ASYNC_EVENT_CMPL_HW_FLOW_AGED_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_HW_FLOW_AGED_TYPE_SFT = 0
ASYNC_EVENT_CMPL_HW_FLOW_AGED_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_HW_FLOW_AGED_TYPE_LAST = ASYNC_EVENT_CMPL_HW_FLOW_AGED_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_ID_HW_FLOW_AGED = 0x36
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_ID_LAST = ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_ID_HW_FLOW_AGED
ASYNC_EVENT_CMPL_HW_FLOW_AGED_V = 0x1
ASYNC_EVENT_CMPL_HW_FLOW_AGED_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_HW_FLOW_AGED_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_ID_MASK = 0x7fffffff
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_ID_SFT = 0
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_DIRECTION = 0x80000000
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_DIRECTION_RX = (0x0 << 31)
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_DIRECTION_TX = (0x1 << 31)
ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_DIRECTION_LAST = ASYNC_EVENT_CMPL_HW_FLOW_AGED_EVENT_DATA1_FLOW_DIRECTION_TX
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_TYPE_SFT = 0
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_TYPE_LAST = ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_EVENT_ID_EEM_CACHE_FLUSH_REQ = 0x38
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_EVENT_ID_LAST = ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_EVENT_ID_EEM_CACHE_FLUSH_REQ
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_V = 0x1
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_REQ_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_TYPE_LAST = ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_EVENT_ID_EEM_CACHE_FLUSH_DONE = 0x39
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_EVENT_ID_EEM_CACHE_FLUSH_DONE
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_V = 0x1
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_EVENT_DATA1_FID_MASK = 0xffff
ASYNC_EVENT_CMPL_EEM_CACHE_FLUSH_DONE_EVENT_DATA1_FID_SFT = 0
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_TYPE_LAST = ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_EVENT_ID_DEFERRED_RESPONSE = 0x40
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_EVENT_ID_DEFERRED_RESPONSE
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_EVENT_DATA2_SEQ_ID_MASK = 0xffff
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_EVENT_DATA2_SEQ_ID_SFT = 0
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_V = 0x1
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_DEFERRED_RESPONSE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ECHO_REQUEST_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ECHO_REQUEST_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ECHO_REQUEST_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ECHO_REQUEST_TYPE_LAST = ASYNC_EVENT_CMPL_ECHO_REQUEST_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ECHO_REQUEST_EVENT_ID_ECHO_REQUEST = 0x42
ASYNC_EVENT_CMPL_ECHO_REQUEST_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ECHO_REQUEST_EVENT_ID_ECHO_REQUEST
ASYNC_EVENT_CMPL_ECHO_REQUEST_V = 0x1
ASYNC_EVENT_CMPL_ECHO_REQUEST_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ECHO_REQUEST_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_PHC_UPDATE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_PHC_UPDATE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_PHC_UPDATE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_PHC_UPDATE_TYPE_LAST = ASYNC_EVENT_CMPL_PHC_UPDATE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_ID_PHC_UPDATE = 0x43
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_ID_PHC_UPDATE
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA2_PHC_MASTER_FID_MASK = 0xffff
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA2_PHC_MASTER_FID_SFT = 0
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA2_PHC_SEC_FID_MASK = 0xffff0000
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA2_PHC_SEC_FID_SFT = 16
ASYNC_EVENT_CMPL_PHC_UPDATE_V = 0x1
ASYNC_EVENT_CMPL_PHC_UPDATE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_PHC_UPDATE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_MASK = 0xf
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_SFT = 0
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_PHC_MASTER = 0x1
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_PHC_SECONDARY = 0x2
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_PHC_FAILOVER = 0x3
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_PHC_RTC_UPDATE = 0x4
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_LAST = ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_FLAGS_PHC_RTC_UPDATE
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_PHC_TIME_MSB_MASK = 0xffff0
ASYNC_EVENT_CMPL_PHC_UPDATE_EVENT_DATA1_PHC_TIME_MSB_SFT = 4
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_TYPE_SFT = 0
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_TYPE_LAST = ASYNC_EVENT_CMPL_PPS_TIMESTAMP_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_ID_PPS_TIMESTAMP = 0x44
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_ID_LAST = ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_ID_PPS_TIMESTAMP
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_EVENT_TYPE = 0x1
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_EVENT_TYPE_INTERNAL = 0x0
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_EVENT_TYPE_EXTERNAL = 0x1
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_EVENT_TYPE_LAST = ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_EVENT_TYPE_EXTERNAL
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_PIN_NUMBER_MASK = 0xe
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_PIN_NUMBER_SFT = 1
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_PPS_TIMESTAMP_UPPER_MASK = 0xffff0
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA2_PPS_TIMESTAMP_UPPER_SFT = 4
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_V = 0x1
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA1_PPS_TIMESTAMP_LOWER_MASK = 0xffffffff
ASYNC_EVENT_CMPL_PPS_TIMESTAMP_EVENT_DATA1_PPS_TIMESTAMP_LOWER_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_TYPE_SFT = 0
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_TYPE_LAST = ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_ID_DBG_BUF_PRODUCER = 0x4c
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_ID_LAST = ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_ID_DBG_BUF_PRODUCER
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA2_CURR_OFF_MASK = 0xffffffff
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA2_CURR_OFF_SFT = 0
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_V = 0x1
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_MASK = 0xffff
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_SFT = 0
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_SRT_TRACE = 0x0
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_SRT2_TRACE = 0x1
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_CRT_TRACE = 0x2
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_CRT2_TRACE = 0x3
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_RIGP0_TRACE = 0x4
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_L2_HWRM_TRACE = 0x5
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_ROCE_HWRM_TRACE = 0x6
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_CA0_TRACE = 0x7
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_CA1_TRACE = 0x8
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_CA2_TRACE = 0x9
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_RIGP1_TRACE = 0xa
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_AFM_KONG_HWRM_TRACE = 0xb
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_ERR_QPC_TRACE = 0xc
ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_LAST = ASYNC_EVENT_CMPL_DBG_BUF_PRODUCER_EVENT_DATA1_TYPE_ERR_QPC_TRACE
ASYNC_EVENT_CMPL_HWRM_ERROR_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_HWRM_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_HWRM_ERROR_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_HWRM_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_HWRM_ERROR_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_ID_HWRM_ERROR = 0xff
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_ID_LAST = ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_ID_HWRM_ERROR
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_MASK = 0xff
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_SFT = 0
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_WARNING = 0x0
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_NONFATAL = 0x1
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_FATAL = 0x2
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_LAST = ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA2_SEVERITY_FATAL
ASYNC_EVENT_CMPL_HWRM_ERROR_V = 0x1
ASYNC_EVENT_CMPL_HWRM_ERROR_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_HWRM_ERROR_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_HWRM_ERROR_EVENT_DATA1_TIMESTAMP = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_RESERVED = 0x0
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_PAUSE_STORM = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_INVALID_SIGNAL = 0x2
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_NVM = 0x3
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_DOORBELL_DROP_THRESHOLD = 0x4
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_THERMAL_THRESHOLD = 0x5
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_DUAL_DATA_RATE_NOT_SUPPORTED = 0x6
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_DUP_UDCC_SES = 0x7
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_DB_DROP = 0x8
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_MD_TEMP = 0x9
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_VNIC_ERR = 0xa
ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_BASE_EVENT_DATA1_ERROR_TYPE_VNIC_ERR
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_DATA1_ERROR_TYPE_PAUSE_STORM = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_PAUSE_STORM_EVENT_DATA1_ERROR_TYPE_PAUSE_STORM
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA2_PIN_ID_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA2_PIN_ID_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA1_ERROR_TYPE_INVALID_SIGNAL = 0x2
ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_INVALID_SIGNAL_EVENT_DATA1_ERROR_TYPE_INVALID_SIGNAL
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA2_ERR_ADDR_MASK = 0xffffffff
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA2_ERR_ADDR_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_ERROR_TYPE_NVM_ERROR = 0x3
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_ERROR_TYPE_NVM_ERROR
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_NVM_ERR_TYPE_MASK = 0xff00
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_NVM_ERR_TYPE_SFT = 8
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_NVM_ERR_TYPE_WRITE = (0x1 << 8)
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_NVM_ERR_TYPE_ERASE = (0x2 << 8)
ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_NVM_ERR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_NVM_EVENT_DATA1_NVM_ERR_TYPE_ERASE
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_ERROR_TYPE_DOORBELL_DROP_THRESHOLD = 0x4
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_ERROR_TYPE_DOORBELL_DROP_THRESHOLD
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_EPOCH_MASK = 0xffffff00
ASYNC_EVENT_CMPL_ERROR_REPORT_DOORBELL_DROP_THRESHOLD_EVENT_DATA1_EPOCH_SFT = 8
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA2_CURRENT_TEMP_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA2_CURRENT_TEMP_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA2_THRESHOLD_TEMP_MASK = 0xff00
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA2_THRESHOLD_TEMP_SFT = 8
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_ERROR_TYPE_THERMAL_EVENT = 0x5
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_ERROR_TYPE_THERMAL_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_MASK = 0x700
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_SFT = 8
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_WARN = (0x0 << 8)
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_CRITICAL = (0x1 << 8)
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_FATAL = (0x2 << 8)
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_SHUTDOWN = (0x3 << 8)
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_THRESHOLD_TYPE_SHUTDOWN
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_TRANSITION_DIR = 0x800
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_TRANSITION_DIR_DECREASING = (0x0 << 11)
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_TRANSITION_DIR_INCREASING = (0x1 << 11)
ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_TRANSITION_DIR_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_THERMAL_EVENT_DATA1_TRANSITION_DIR_INCREASING
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_TYPE_MASK = 0x3f
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_TYPE_HWRM_ASYNC_EVENT = 0x2e
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_TYPE_HWRM_ASYNC_EVENT
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_ID_ERROR_REPORT = 0x45
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_ID_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_ID_ERROR_REPORT
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_V = 0x1
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_OPAQUE_MASK = 0xfe
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_OPAQUE_SFT = 1
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_DATA1_ERROR_TYPE_MASK = 0xff
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_DATA1_ERROR_TYPE_SFT = 0
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_DATA1_ERROR_TYPE_DUAL_DATA_RATE_NOT_SUPPORTED = 0x6
ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_DATA1_ERROR_TYPE_LAST = ASYNC_EVENT_CMPL_ERROR_REPORT_DUAL_DATA_RATE_NOT_SUPPORTED_EVENT_DATA1_ERROR_TYPE_DUAL_DATA_RATE_NOT_SUPPORTED
FUNC_RESET_REQ_ENABLES_VF_ID_VALID = 0x1
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETALL = 0x0
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETME = 0x1
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETCHILDREN = 0x2
FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETVF = 0x3
FUNC_RESET_REQ_FUNC_RESET_LEVEL_LAST = FUNC_RESET_REQ_FUNC_RESET_LEVEL_RESETVF
FUNC_GETFID_REQ_ENABLES_PCI_ID = 0x1
FUNC_VF_ALLOC_REQ_ENABLES_FIRST_VF_ID = 0x1
FUNC_VF_FREE_REQ_ENABLES_FIRST_VF_ID = 0x1
FUNC_VF_CFG_REQ_ENABLES_MTU = 0x1
FUNC_VF_CFG_REQ_ENABLES_GUEST_VLAN = 0x2
FUNC_VF_CFG_REQ_ENABLES_ASYNC_EVENT_CR = 0x4
FUNC_VF_CFG_REQ_ENABLES_DFLT_MAC_ADDR = 0x8
FUNC_VF_CFG_REQ_ENABLES_NUM_RSSCOS_CTXS = 0x10
FUNC_VF_CFG_REQ_ENABLES_NUM_CMPL_RINGS = 0x20
FUNC_VF_CFG_REQ_ENABLES_NUM_TX_RINGS = 0x40
FUNC_VF_CFG_REQ_ENABLES_NUM_RX_RINGS = 0x80
FUNC_VF_CFG_REQ_ENABLES_NUM_L2_CTXS = 0x100
FUNC_VF_CFG_REQ_ENABLES_NUM_VNICS = 0x200
FUNC_VF_CFG_REQ_ENABLES_NUM_STAT_CTXS = 0x400
FUNC_VF_CFG_REQ_ENABLES_NUM_HW_RING_GRPS = 0x800
FUNC_VF_CFG_REQ_ENABLES_NUM_KTLS_TX_KEY_CTXS = 0x1000
FUNC_VF_CFG_REQ_ENABLES_NUM_KTLS_RX_KEY_CTXS = 0x2000
FUNC_VF_CFG_REQ_ENABLES_NUM_QUIC_TX_KEY_CTXS = 0x4000
FUNC_VF_CFG_REQ_ENABLES_NUM_QUIC_RX_KEY_CTXS = 0x8000
FUNC_VF_CFG_REQ_FLAGS_TX_ASSETS_TEST = 0x1
FUNC_VF_CFG_REQ_FLAGS_RX_ASSETS_TEST = 0x2
FUNC_VF_CFG_REQ_FLAGS_CMPL_ASSETS_TEST = 0x4
FUNC_VF_CFG_REQ_FLAGS_RSSCOS_CTX_ASSETS_TEST = 0x8
FUNC_VF_CFG_REQ_FLAGS_RING_GRP_ASSETS_TEST = 0x10
FUNC_VF_CFG_REQ_FLAGS_STAT_CTX_ASSETS_TEST = 0x20
FUNC_VF_CFG_REQ_FLAGS_VNIC_ASSETS_TEST = 0x40
FUNC_VF_CFG_REQ_FLAGS_L2_CTX_ASSETS_TEST = 0x80
FUNC_VF_CFG_REQ_FLAGS_PPP_PUSH_MODE_ENABLE = 0x100
FUNC_VF_CFG_REQ_FLAGS_PPP_PUSH_MODE_DISABLE = 0x200
FUNC_QCAPS_RESP_FLAGS_PUSH_MODE_SUPPORTED = 0x1
FUNC_QCAPS_RESP_FLAGS_GLOBAL_MSIX_AUTOMASKING = 0x2
FUNC_QCAPS_RESP_FLAGS_PTP_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_ROCE_V1_SUPPORTED = 0x8
FUNC_QCAPS_RESP_FLAGS_ROCE_V2_SUPPORTED = 0x10
FUNC_QCAPS_RESP_FLAGS_WOL_MAGICPKT_SUPPORTED = 0x20
FUNC_QCAPS_RESP_FLAGS_WOL_BMP_SUPPORTED = 0x40
FUNC_QCAPS_RESP_FLAGS_TX_RING_RL_SUPPORTED = 0x80
FUNC_QCAPS_RESP_FLAGS_TX_BW_CFG_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_VF_TX_RING_RL_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_VF_BW_CFG_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_STD_TX_RING_MODE_SUPPORTED = 0x800
FUNC_QCAPS_RESP_FLAGS_GENEVE_TUN_FLAGS_SUPPORTED = 0x1000
FUNC_QCAPS_RESP_FLAGS_NVGRE_TUN_FLAGS_SUPPORTED = 0x2000
FUNC_QCAPS_RESP_FLAGS_GRE_TUN_FLAGS_SUPPORTED = 0x4000
FUNC_QCAPS_RESP_FLAGS_MPLS_TUN_FLAGS_SUPPORTED = 0x8000
FUNC_QCAPS_RESP_FLAGS_PCIE_STATS_SUPPORTED = 0x10000
FUNC_QCAPS_RESP_FLAGS_ADOPTED_PF_SUPPORTED = 0x20000
FUNC_QCAPS_RESP_FLAGS_ADMIN_PF_SUPPORTED = 0x40000
FUNC_QCAPS_RESP_FLAGS_LINK_ADMIN_STATUS_SUPPORTED = 0x80000
FUNC_QCAPS_RESP_FLAGS_WCB_PUSH_MODE = 0x100000
FUNC_QCAPS_RESP_FLAGS_DYNAMIC_TX_RING_ALLOC = 0x200000
FUNC_QCAPS_RESP_FLAGS_HOT_RESET_CAPABLE = 0x400000
FUNC_QCAPS_RESP_FLAGS_ERROR_RECOVERY_CAPABLE = 0x800000
FUNC_QCAPS_RESP_FLAGS_EXT_STATS_SUPPORTED = 0x1000000
FUNC_QCAPS_RESP_FLAGS_ERR_RECOVER_RELOAD = 0x2000000
FUNC_QCAPS_RESP_FLAGS_NOTIFY_VF_DEF_VNIC_CHNG_SUPPORTED = 0x4000000
FUNC_QCAPS_RESP_FLAGS_VLAN_ACCELERATION_TX_DISABLED = 0x8000000
FUNC_QCAPS_RESP_FLAGS_COREDUMP_CMD_SUPPORTED = 0x10000000
FUNC_QCAPS_RESP_FLAGS_CRASHDUMP_CMD_SUPPORTED = 0x20000000
FUNC_QCAPS_RESP_FLAGS_PFC_WD_STATS_SUPPORTED = 0x40000000
FUNC_QCAPS_RESP_FLAGS_DBG_QCAPS_CMD_SUPPORTED = 0x80000000
FUNC_QCAPS_RESP_FLAGS_EXT_ECN_MARK_SUPPORTED = 0x1
FUNC_QCAPS_RESP_FLAGS_EXT_ECN_STATS_SUPPORTED = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT_EXT_HW_STATS_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_EXT_HOT_RESET_IF_SUPPORT = 0x8
FUNC_QCAPS_RESP_FLAGS_EXT_PROXY_MODE_SUPPORT = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT_TX_PROXY_SRC_INTF_OVERRIDE_SUPPORT = 0x20
FUNC_QCAPS_RESP_FLAGS_EXT_SCHQ_SUPPORTED = 0x40
FUNC_QCAPS_RESP_FLAGS_EXT_PPP_PUSH_MODE_SUPPORTED = 0x80
FUNC_QCAPS_RESP_FLAGS_EXT_EVB_MODE_CFG_NOT_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_EXT_SOC_SPD_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_EXT_FW_LIVEPATCH_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_EXT_FAST_RESET_CAPABLE = 0x800
FUNC_QCAPS_RESP_FLAGS_EXT_TX_METADATA_CFG_CAPABLE = 0x1000
FUNC_QCAPS_RESP_FLAGS_EXT_NVM_OPTION_ACTION_SUPPORTED = 0x2000
FUNC_QCAPS_RESP_FLAGS_EXT_BD_METADATA_SUPPORTED = 0x4000
FUNC_QCAPS_RESP_FLAGS_EXT_ECHO_REQUEST_SUPPORTED = 0x8000
FUNC_QCAPS_RESP_FLAGS_EXT_NPAR_1_2_SUPPORTED = 0x10000
FUNC_QCAPS_RESP_FLAGS_EXT_PTP_PTM_SUPPORTED = 0x20000
FUNC_QCAPS_RESP_FLAGS_EXT_PTP_PPS_SUPPORTED = 0x40000
FUNC_QCAPS_RESP_FLAGS_EXT_VF_CFG_ASYNC_FOR_PF_SUPPORTED = 0x80000
FUNC_QCAPS_RESP_FLAGS_EXT_PARTITION_BW_SUPPORTED = 0x100000
FUNC_QCAPS_RESP_FLAGS_EXT_DFLT_VLAN_TPID_PCP_SUPPORTED = 0x200000
FUNC_QCAPS_RESP_FLAGS_EXT_KTLS_SUPPORTED = 0x400000
FUNC_QCAPS_RESP_FLAGS_EXT_EP_RATE_CONTROL = 0x800000
FUNC_QCAPS_RESP_FLAGS_EXT_MIN_BW_SUPPORTED = 0x1000000
FUNC_QCAPS_RESP_FLAGS_EXT_TX_COAL_CMPL_CAP = 0x2000000
FUNC_QCAPS_RESP_FLAGS_EXT_BS_V2_SUPPORTED = 0x4000000
FUNC_QCAPS_RESP_FLAGS_EXT_BS_V2_REQUIRED = 0x8000000
FUNC_QCAPS_RESP_FLAGS_EXT_PTP_64BIT_RTC_SUPPORTED = 0x10000000
FUNC_QCAPS_RESP_FLAGS_EXT_DBR_PACING_SUPPORTED = 0x20000000
FUNC_QCAPS_RESP_FLAGS_EXT_HW_DBR_DROP_RECOV_SUPPORTED = 0x40000000
FUNC_QCAPS_RESP_FLAGS_EXT_DISABLE_CQ_OVERFLOW_DETECTION_SUPPORTED = 0x80000000
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_TCE = 0x1
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_RCE = 0x2
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_TE_CFA = 0x4
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_RE_CFA = 0x8
FUNC_QCAPS_RESP_MPC_CHNLS_CAP_PRIMATE = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT2_RX_ALL_PKTS_TIMESTAMPS_SUPPORTED = 0x1
FUNC_QCAPS_RESP_FLAGS_EXT2_QUIC_SUPPORTED = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT2_KDNET_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_EXT2_DBR_PACING_EXT_SUPPORTED = 0x8
FUNC_QCAPS_RESP_FLAGS_EXT2_SW_DBR_DROP_RECOVERY_SUPPORTED = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT2_GENERIC_STATS_SUPPORTED = 0x20
FUNC_QCAPS_RESP_FLAGS_EXT2_UDP_GSO_SUPPORTED = 0x40
FUNC_QCAPS_RESP_FLAGS_EXT2_SYNCE_SUPPORTED = 0x80
FUNC_QCAPS_RESP_FLAGS_EXT2_DBR_PACING_V0_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_EXT2_TX_PKT_TS_CMPL_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_EXT2_HW_LAG_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_EXT2_ON_CHIP_CTX_SUPPORTED = 0x800
FUNC_QCAPS_RESP_FLAGS_EXT2_STEERING_TAG_SUPPORTED = 0x1000
FUNC_QCAPS_RESP_FLAGS_EXT2_ENHANCED_VF_SCALE_SUPPORTED = 0x2000
FUNC_QCAPS_RESP_FLAGS_EXT2_KEY_XID_PARTITION_SUPPORTED = 0x4000
FUNC_QCAPS_RESP_FLAGS_EXT2_CONCURRENT_KTLS_QUIC_SUPPORTED = 0x8000
FUNC_QCAPS_RESP_FLAGS_EXT2_SCHQ_CROSS_TC_CAP_SUPPORTED = 0x10000
FUNC_QCAPS_RESP_FLAGS_EXT2_SCHQ_PER_TC_CAP_SUPPORTED = 0x20000
FUNC_QCAPS_RESP_FLAGS_EXT2_SCHQ_PER_TC_RESERVATION_SUPPORTED = 0x40000
FUNC_QCAPS_RESP_FLAGS_EXT2_DB_ERROR_STATS_SUPPORTED = 0x80000
FUNC_QCAPS_RESP_FLAGS_EXT2_ROCE_VF_RESOURCE_MGMT_SUPPORTED = 0x100000
FUNC_QCAPS_RESP_FLAGS_EXT2_UDCC_SUPPORTED = 0x200000
FUNC_QCAPS_RESP_FLAGS_EXT2_TIMED_TX_SO_TXTIME_SUPPORTED = 0x400000
FUNC_QCAPS_RESP_FLAGS_EXT2_SW_MAX_RESOURCE_LIMITS_SUPPORTED = 0x800000
FUNC_QCAPS_RESP_FLAGS_EXT2_TF_INGRESS_NIC_FLOW_SUPPORTED = 0x1000000
FUNC_QCAPS_RESP_FLAGS_EXT2_LPBK_STATS_SUPPORTED = 0x2000000
FUNC_QCAPS_RESP_FLAGS_EXT2_TF_EGRESS_NIC_FLOW_SUPPORTED = 0x4000000
FUNC_QCAPS_RESP_FLAGS_EXT2_MULTI_LOSSLESS_QUEUES_SUPPORTED = 0x8000000
FUNC_QCAPS_RESP_FLAGS_EXT2_PEER_MMAP_SUPPORTED = 0x10000000
FUNC_QCAPS_RESP_FLAGS_EXT2_TIMED_TX_PACING_SUPPORTED = 0x20000000
FUNC_QCAPS_RESP_FLAGS_EXT2_VF_STAT_EJECTION_SUPPORTED = 0x40000000
FUNC_QCAPS_RESP_FLAGS_EXT2_HOST_COREDUMP_SUPPORTED = 0x80000000
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_VXLAN = 0x1
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_NGE = 0x2
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_NVGRE = 0x4
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_L2GRE = 0x8
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_GRE = 0x10
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_IPINIP = 0x20
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_MPLS = 0x40
FUNC_QCAPS_RESP_TUNNEL_DISABLE_FLAG_DISABLE_PPPOE = 0x80
FUNC_QCAPS_RESP_XID_PARTITION_CAP_TX_CK = 0x1
FUNC_QCAPS_RESP_XID_PARTITION_CAP_RX_CK = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT3_RM_RSV_WHILE_ALLOC_CAP = 0x1
FUNC_QCAPS_RESP_FLAGS_EXT3_REQUIRE_L2_FILTER = 0x2
FUNC_QCAPS_RESP_FLAGS_EXT3_MAX_ROCE_VFS_SUPPORTED = 0x4
FUNC_QCAPS_RESP_FLAGS_EXT3_RX_RATE_PROFILE_SEL_SUPPORTED = 0x8
FUNC_QCAPS_RESP_FLAGS_EXT3_BIDI_OPT_SUPPORTED = 0x10
FUNC_QCAPS_RESP_FLAGS_EXT3_MIRROR_ON_ROCE_SUPPORTED = 0x20
FUNC_QCAPS_RESP_FLAGS_EXT3_ROCE_VF_DYN_ALLOC_SUPPORT = 0x40
FUNC_QCAPS_RESP_FLAGS_EXT3_CHANGE_UDP_SRCPORT_SUPPORT = 0x80
FUNC_QCAPS_RESP_FLAGS_EXT3_PCIE_COMPLIANCE_SUPPORTED = 0x100
FUNC_QCAPS_RESP_FLAGS_EXT3_MULTI_L2_DB_SUPPORTED = 0x200
FUNC_QCAPS_RESP_FLAGS_EXT3_PCIE_SECURE_ATS_SUPPORTED = 0x400
FUNC_QCAPS_RESP_FLAGS_EXT3_MBUF_STATS_SUPPORTED = 0x800
FUNC_QCFG_RESP_FLAGS_OOB_WOL_MAGICPKT_ENABLED = 0x1
FUNC_QCFG_RESP_FLAGS_OOB_WOL_BMP_ENABLED = 0x2
FUNC_QCFG_RESP_FLAGS_FW_DCBX_AGENT_ENABLED = 0x4
FUNC_QCFG_RESP_FLAGS_STD_TX_RING_MODE_ENABLED = 0x8
FUNC_QCFG_RESP_FLAGS_FW_LLDP_AGENT_ENABLED = 0x10
FUNC_QCFG_RESP_FLAGS_MULTI_HOST = 0x20
FUNC_QCFG_RESP_FLAGS_TRUSTED_VF = 0x40
FUNC_QCFG_RESP_FLAGS_SECURE_MODE_ENABLED = 0x80
FUNC_QCFG_RESP_FLAGS_PREBOOT_LEGACY_L2_RINGS = 0x100
FUNC_QCFG_RESP_FLAGS_HOT_RESET_ALLOWED = 0x200
FUNC_QCFG_RESP_FLAGS_PPP_PUSH_MODE_ENABLED = 0x400
FUNC_QCFG_RESP_FLAGS_RING_MONITOR_ENABLED = 0x800
FUNC_QCFG_RESP_FLAGS_FAST_RESET_ALLOWED = 0x1000
FUNC_QCFG_RESP_FLAGS_MULTI_ROOT = 0x2000
FUNC_QCFG_RESP_FLAGS_ENABLE_RDMA_SRIOV = 0x4000
FUNC_QCFG_RESP_FLAGS_ROCE_VNIC_ID_VALID = 0x8000
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_SPF = 0x0
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_MPFS = 0x1
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR1_0 = 0x2
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR1_5 = 0x3
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR2_0 = 0x4
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_NPAR1_2 = 0x5
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_UNKNOWN = 0xff
FUNC_QCFG_RESP_PORT_PARTITION_TYPE_LAST = FUNC_QCFG_RESP_PORT_PARTITION_TYPE_UNKNOWN
FUNC_QCFG_RESP_PORT_PF_CNT_UNAVAIL = 0x0
FUNC_QCFG_RESP_PORT_PF_CNT_LAST = FUNC_QCFG_RESP_PORT_PF_CNT_UNAVAIL
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_MIN_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_MIN_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_MIN_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_MIN_BW_SCALE_LAST = FUNC_QCFG_RESP_MIN_BW_SCALE_BYTES
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_MIN_BW_BW_VALUE_UNIT_INVALID
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_MAX_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_MAX_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_MAX_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_MAX_BW_SCALE_LAST = FUNC_QCFG_RESP_MAX_BW_SCALE_BYTES
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_MAX_BW_BW_VALUE_UNIT_INVALID
FUNC_QCFG_RESP_EVB_MODE_NO_EVB = 0x0
FUNC_QCFG_RESP_EVB_MODE_VEB = 0x1
FUNC_QCFG_RESP_EVB_MODE_VEPA = 0x2
FUNC_QCFG_RESP_EVB_MODE_LAST = FUNC_QCFG_RESP_EVB_MODE_VEPA
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_MASK = 0x3
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SFT = 0
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SIZE_64 = 0x0
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SIZE_128 = 0x1
FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_LAST = FUNC_QCFG_RESP_OPTIONS_CACHE_LINESIZE_SIZE_128
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_MASK = 0xc
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_SFT = 2
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_FORCED_DOWN = (0x0 << 2)
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_FORCED_UP = (0x1 << 2)
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_AUTO = (0x2 << 2)
FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_LAST = FUNC_QCFG_RESP_OPTIONS_LINK_ADMIN_STATE_AUTO
FUNC_QCFG_RESP_OPTIONS_RSVD_MASK = 0xf0
FUNC_QCFG_RESP_OPTIONS_RSVD_SFT = 4
FUNC_QCFG_RESP_SVIF_INFO_SVIF_MASK = 0x7fff
FUNC_QCFG_RESP_SVIF_INFO_SVIF_SFT = 0
FUNC_QCFG_RESP_SVIF_INFO_SVIF_VALID = 0x8000
FUNC_QCFG_RESP_MPC_CHNLS_TCE_ENABLED = 0x1
FUNC_QCFG_RESP_MPC_CHNLS_RCE_ENABLED = 0x2
FUNC_QCFG_RESP_MPC_CHNLS_TE_CFA_ENABLED = 0x4
FUNC_QCFG_RESP_MPC_CHNLS_RE_CFA_ENABLED = 0x8
FUNC_QCFG_RESP_MPC_CHNLS_PRIMATE_ENABLED = 0x10
FUNC_QCFG_RESP_DB_PAGE_SIZE_4KB = 0x0
FUNC_QCFG_RESP_DB_PAGE_SIZE_8KB = 0x1
FUNC_QCFG_RESP_DB_PAGE_SIZE_16KB = 0x2
FUNC_QCFG_RESP_DB_PAGE_SIZE_32KB = 0x3
FUNC_QCFG_RESP_DB_PAGE_SIZE_64KB = 0x4
FUNC_QCFG_RESP_DB_PAGE_SIZE_128KB = 0x5
FUNC_QCFG_RESP_DB_PAGE_SIZE_256KB = 0x6
FUNC_QCFG_RESP_DB_PAGE_SIZE_512KB = 0x7
FUNC_QCFG_RESP_DB_PAGE_SIZE_1MB = 0x8
FUNC_QCFG_RESP_DB_PAGE_SIZE_2MB = 0x9
FUNC_QCFG_RESP_DB_PAGE_SIZE_4MB = 0xa
FUNC_QCFG_RESP_DB_PAGE_SIZE_LAST = FUNC_QCFG_RESP_DB_PAGE_SIZE_4MB
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_LAST = FUNC_QCFG_RESP_PARTITION_MIN_BW_SCALE_BYTES
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_PARTITION_MIN_BW_BW_VALUE_UNIT_PERCENT1_100
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_MASK = 0xfffffff
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_SFT = 0
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE = 0x10000000
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_BITS = (0x0 << 28)
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_BYTES = (0x1 << 28)
FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_LAST = FUNC_QCFG_RESP_PARTITION_MAX_BW_SCALE_BYTES
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_SFT = 29
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_LAST = FUNC_QCFG_RESP_PARTITION_MAX_BW_BW_VALUE_UNIT_PERCENT1_100
FUNC_QCFG_RESP_FLAGS2_SRIOV_DSCP_INSERT_ENABLED = 0x1
FUNC_QCFG_RESP_PORT_KDNET_MODE_DISABLED = 0x0
FUNC_QCFG_RESP_PORT_KDNET_MODE_ENABLED = 0x1
FUNC_QCFG_RESP_PORT_KDNET_MODE_LAST = FUNC_QCFG_RESP_PORT_KDNET_MODE_ENABLED
FUNC_QCFG_RESP_ROCE_BIDI_OPT_MODE_DISABLED = 0x1
FUNC_QCFG_RESP_ROCE_BIDI_OPT_MODE_DEDICATED = 0x2
FUNC_QCFG_RESP_ROCE_BIDI_OPT_MODE_SHARED = 0x4
FUNC_QCFG_RESP_XID_PARTITION_CFG_TX_CK = 0x1
FUNC_QCFG_RESP_XID_PARTITION_CFG_RX_CK = 0x2
FUNC_QCFG_RESP_MAX_LINK_WIDTH_UNKNOWN = 0x0
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X1 = 0x1
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X2 = 0x2
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X4 = 0x4
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X8 = 0x8
FUNC_QCFG_RESP_MAX_LINK_WIDTH_X16 = 0x10
FUNC_QCFG_RESP_MAX_LINK_WIDTH_LAST = FUNC_QCFG_RESP_MAX_LINK_WIDTH_X16
FUNC_QCFG_RESP_MAX_LINK_SPEED_UNKNOWN = 0x0
FUNC_QCFG_RESP_MAX_LINK_SPEED_G1 = 0x1
FUNC_QCFG_RESP_MAX_LINK_SPEED_G2 = 0x2
FUNC_QCFG_RESP_MAX_LINK_SPEED_G3 = 0x3
FUNC_QCFG_RESP_MAX_LINK_SPEED_G4 = 0x4
FUNC_QCFG_RESP_MAX_LINK_SPEED_G5 = 0x5
FUNC_QCFG_RESP_MAX_LINK_SPEED_LAST = FUNC_QCFG_RESP_MAX_LINK_SPEED_G5
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_UNKNOWN = 0x0
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X1 = 0x1
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X2 = 0x2
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X4 = 0x4
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X8 = 0x8
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X16 = 0x10
FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_LAST = FUNC_QCFG_RESP_NEGOTIATED_LINK_WIDTH_X16
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_UNKNOWN = 0x0
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G1 = 0x1
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G2 = 0x2
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G3 = 0x3
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G4 = 0x4
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G5 = 0x5
FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_LAST = FUNC_QCFG_RESP_NEGOTIATED_LINK_SPEED_G5
FUNC_CFG_REQ_FLAGS_SRC_MAC_ADDR_CHECK_DISABLE = 0x1
FUNC_CFG_REQ_FLAGS_SRC_MAC_ADDR_CHECK_ENABLE = 0x2
FUNC_CFG_REQ_FLAGS_RSVD_MASK = 0x1fc
FUNC_CFG_REQ_FLAGS_RSVD_SFT = 2
FUNC_CFG_REQ_FLAGS_STD_TX_RING_MODE_ENABLE = 0x200
FUNC_CFG_REQ_FLAGS_STD_TX_RING_MODE_DISABLE = 0x400
FUNC_CFG_REQ_FLAGS_VIRT_MAC_PERSIST = 0x800
FUNC_CFG_REQ_FLAGS_NO_AUTOCLEAR_STATISTIC = 0x1000
FUNC_CFG_REQ_FLAGS_TX_ASSETS_TEST = 0x2000
FUNC_CFG_REQ_FLAGS_RX_ASSETS_TEST = 0x4000
FUNC_CFG_REQ_FLAGS_CMPL_ASSETS_TEST = 0x8000
FUNC_CFG_REQ_FLAGS_RSSCOS_CTX_ASSETS_TEST = 0x10000
FUNC_CFG_REQ_FLAGS_RING_GRP_ASSETS_TEST = 0x20000
FUNC_CFG_REQ_FLAGS_STAT_CTX_ASSETS_TEST = 0x40000
FUNC_CFG_REQ_FLAGS_VNIC_ASSETS_TEST = 0x80000
FUNC_CFG_REQ_FLAGS_L2_CTX_ASSETS_TEST = 0x100000
FUNC_CFG_REQ_FLAGS_TRUSTED_VF_ENABLE = 0x200000
FUNC_CFG_REQ_FLAGS_DYNAMIC_TX_RING_ALLOC = 0x400000
FUNC_CFG_REQ_FLAGS_NQ_ASSETS_TEST = 0x800000
FUNC_CFG_REQ_FLAGS_TRUSTED_VF_DISABLE = 0x1000000
FUNC_CFG_REQ_FLAGS_PREBOOT_LEGACY_L2_RINGS = 0x2000000
FUNC_CFG_REQ_FLAGS_HOT_RESET_IF_EN_DIS = 0x4000000
FUNC_CFG_REQ_FLAGS_PPP_PUSH_MODE_ENABLE = 0x8000000
FUNC_CFG_REQ_FLAGS_PPP_PUSH_MODE_DISABLE = 0x10000000
FUNC_CFG_REQ_FLAGS_BD_METADATA_ENABLE = 0x20000000
FUNC_CFG_REQ_FLAGS_BD_METADATA_DISABLE = 0x40000000
FUNC_CFG_REQ_ENABLES_ADMIN_MTU = 0x1
FUNC_CFG_REQ_ENABLES_MRU = 0x2
FUNC_CFG_REQ_ENABLES_NUM_RSSCOS_CTXS = 0x4
FUNC_CFG_REQ_ENABLES_NUM_CMPL_RINGS = 0x8
FUNC_CFG_REQ_ENABLES_NUM_TX_RINGS = 0x10
FUNC_CFG_REQ_ENABLES_NUM_RX_RINGS = 0x20
FUNC_CFG_REQ_ENABLES_NUM_L2_CTXS = 0x40
FUNC_CFG_REQ_ENABLES_NUM_VNICS = 0x80
FUNC_CFG_REQ_ENABLES_NUM_STAT_CTXS = 0x100
FUNC_CFG_REQ_ENABLES_DFLT_MAC_ADDR = 0x200
FUNC_CFG_REQ_ENABLES_DFLT_VLAN = 0x400
FUNC_CFG_REQ_ENABLES_DFLT_IP_ADDR = 0x800
FUNC_CFG_REQ_ENABLES_MIN_BW = 0x1000
FUNC_CFG_REQ_ENABLES_MAX_BW = 0x2000
FUNC_CFG_REQ_ENABLES_ASYNC_EVENT_CR = 0x4000
FUNC_CFG_REQ_ENABLES_VLAN_ANTISPOOF_MODE = 0x8000
FUNC_CFG_REQ_ENABLES_ALLOWED_VLAN_PRIS = 0x10000
FUNC_CFG_REQ_ENABLES_EVB_MODE = 0x20000
FUNC_CFG_REQ_ENABLES_NUM_MCAST_FILTERS = 0x40000
FUNC_CFG_REQ_ENABLES_NUM_HW_RING_GRPS = 0x80000
FUNC_CFG_REQ_ENABLES_CACHE_LINESIZE = 0x100000
FUNC_CFG_REQ_ENABLES_NUM_MSIX = 0x200000
FUNC_CFG_REQ_ENABLES_ADMIN_LINK_STATE = 0x400000
FUNC_CFG_REQ_ENABLES_HOT_RESET_IF_SUPPORT = 0x800000
FUNC_CFG_REQ_ENABLES_SCHQ_ID = 0x1000000
FUNC_CFG_REQ_ENABLES_MPC_CHNLS = 0x2000000
FUNC_CFG_REQ_ENABLES_PARTITION_MIN_BW = 0x4000000
FUNC_CFG_REQ_ENABLES_PARTITION_MAX_BW = 0x8000000
FUNC_CFG_REQ_ENABLES_TPID = 0x10000000
FUNC_CFG_REQ_ENABLES_HOST_MTU = 0x20000000
FUNC_CFG_REQ_ENABLES_KTLS_TX_KEY_CTXS = 0x40000000
FUNC_CFG_REQ_ENABLES_KTLS_RX_KEY_CTXS = 0x80000000
FUNC_CFG_REQ_MIN_BW_BW_VALUE_MASK = 0xfffffff
FUNC_CFG_REQ_MIN_BW_BW_VALUE_SFT = 0
FUNC_CFG_REQ_MIN_BW_SCALE = 0x10000000
FUNC_CFG_REQ_MIN_BW_SCALE_BITS = (0x0 << 28)
FUNC_CFG_REQ_MIN_BW_SCALE_BYTES = (0x1 << 28)
FUNC_CFG_REQ_MIN_BW_SCALE_LAST = FUNC_CFG_REQ_MIN_BW_SCALE_BYTES
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_SFT = 29
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_LAST = FUNC_CFG_REQ_MIN_BW_BW_VALUE_UNIT_INVALID
FUNC_CFG_REQ_MAX_BW_BW_VALUE_MASK = 0xfffffff
FUNC_CFG_REQ_MAX_BW_BW_VALUE_SFT = 0
FUNC_CFG_REQ_MAX_BW_SCALE = 0x10000000
FUNC_CFG_REQ_MAX_BW_SCALE_BITS = (0x0 << 28)
FUNC_CFG_REQ_MAX_BW_SCALE_BYTES = (0x1 << 28)
FUNC_CFG_REQ_MAX_BW_SCALE_LAST = FUNC_CFG_REQ_MAX_BW_SCALE_BYTES
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_SFT = 29
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_LAST = FUNC_CFG_REQ_MAX_BW_BW_VALUE_UNIT_INVALID
FUNC_CFG_REQ_VLAN_ANTISPOOF_MODE_NOCHECK = 0x0
FUNC_CFG_REQ_VLAN_ANTISPOOF_MODE_VALIDATE_VLAN = 0x1
FUNC_CFG_REQ_VLAN_ANTISPOOF_MODE_INSERT_IF_VLANDNE = 0x2
FUNC_CFG_REQ_VLAN_ANTISPOOF_MODE_INSERT_OR_OVERRIDE_VLAN = 0x3
FUNC_CFG_REQ_VLAN_ANTISPOOF_MODE_LAST = FUNC_CFG_REQ_VLAN_ANTISPOOF_MODE_INSERT_OR_OVERRIDE_VLAN
FUNC_CFG_REQ_EVB_MODE_NO_EVB = 0x0
FUNC_CFG_REQ_EVB_MODE_VEB = 0x1
FUNC_CFG_REQ_EVB_MODE_VEPA = 0x2
FUNC_CFG_REQ_EVB_MODE_LAST = FUNC_CFG_REQ_EVB_MODE_VEPA
FUNC_CFG_REQ_OPTIONS_CACHE_LINESIZE_MASK = 0x3
FUNC_CFG_REQ_OPTIONS_CACHE_LINESIZE_SFT = 0
FUNC_CFG_REQ_OPTIONS_CACHE_LINESIZE_SIZE_64 = 0x0
FUNC_CFG_REQ_OPTIONS_CACHE_LINESIZE_SIZE_128 = 0x1
FUNC_CFG_REQ_OPTIONS_CACHE_LINESIZE_LAST = FUNC_CFG_REQ_OPTIONS_CACHE_LINESIZE_SIZE_128
FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_MASK = 0xc
FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_SFT = 2
FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_FORCED_DOWN = (0x0 << 2)
FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_FORCED_UP = (0x1 << 2)
FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_AUTO = (0x2 << 2)
FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_LAST = FUNC_CFG_REQ_OPTIONS_LINK_ADMIN_STATE_AUTO
FUNC_CFG_REQ_OPTIONS_RSVD_MASK = 0xf0
FUNC_CFG_REQ_OPTIONS_RSVD_SFT = 4
FUNC_CFG_REQ_MPC_CHNLS_TCE_ENABLE = 0x1
FUNC_CFG_REQ_MPC_CHNLS_TCE_DISABLE = 0x2
FUNC_CFG_REQ_MPC_CHNLS_RCE_ENABLE = 0x4
FUNC_CFG_REQ_MPC_CHNLS_RCE_DISABLE = 0x8
FUNC_CFG_REQ_MPC_CHNLS_TE_CFA_ENABLE = 0x10
FUNC_CFG_REQ_MPC_CHNLS_TE_CFA_DISABLE = 0x20
FUNC_CFG_REQ_MPC_CHNLS_RE_CFA_ENABLE = 0x40
FUNC_CFG_REQ_MPC_CHNLS_RE_CFA_DISABLE = 0x80
FUNC_CFG_REQ_MPC_CHNLS_PRIMATE_ENABLE = 0x100
FUNC_CFG_REQ_MPC_CHNLS_PRIMATE_DISABLE = 0x200
FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_MASK = 0xfffffff
FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_SFT = 0
FUNC_CFG_REQ_PARTITION_MIN_BW_SCALE = 0x10000000
FUNC_CFG_REQ_PARTITION_MIN_BW_SCALE_BITS = (0x0 << 28)
FUNC_CFG_REQ_PARTITION_MIN_BW_SCALE_BYTES = (0x1 << 28)
FUNC_CFG_REQ_PARTITION_MIN_BW_SCALE_LAST = FUNC_CFG_REQ_PARTITION_MIN_BW_SCALE_BYTES
FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_UNIT_SFT = 29
FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_UNIT_LAST = FUNC_CFG_REQ_PARTITION_MIN_BW_BW_VALUE_UNIT_PERCENT1_100
FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_MASK = 0xfffffff
FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_SFT = 0
FUNC_CFG_REQ_PARTITION_MAX_BW_SCALE = 0x10000000
FUNC_CFG_REQ_PARTITION_MAX_BW_SCALE_BITS = (0x0 << 28)
FUNC_CFG_REQ_PARTITION_MAX_BW_SCALE_BYTES = (0x1 << 28)
FUNC_CFG_REQ_PARTITION_MAX_BW_SCALE_LAST = FUNC_CFG_REQ_PARTITION_MAX_BW_SCALE_BYTES
FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_UNIT_SFT = 29
FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_UNIT_LAST = FUNC_CFG_REQ_PARTITION_MAX_BW_BW_VALUE_UNIT_PERCENT1_100
FUNC_CFG_REQ_FLAGS2_KTLS_KEY_CTX_ASSETS_TEST = 0x1
FUNC_CFG_REQ_FLAGS2_QUIC_KEY_CTX_ASSETS_TEST = 0x2
FUNC_CFG_REQ_ENABLES2_KDNET = 0x1
FUNC_CFG_REQ_ENABLES2_DB_PAGE_SIZE = 0x2
FUNC_CFG_REQ_ENABLES2_QUIC_TX_KEY_CTXS = 0x4
FUNC_CFG_REQ_ENABLES2_QUIC_RX_KEY_CTXS = 0x8
FUNC_CFG_REQ_ENABLES2_ROCE_MAX_AV_PER_VF = 0x10
FUNC_CFG_REQ_ENABLES2_ROCE_MAX_CQ_PER_VF = 0x20
FUNC_CFG_REQ_ENABLES2_ROCE_MAX_MRW_PER_VF = 0x40
FUNC_CFG_REQ_ENABLES2_ROCE_MAX_QP_PER_VF = 0x80
FUNC_CFG_REQ_ENABLES2_ROCE_MAX_SRQ_PER_VF = 0x100
FUNC_CFG_REQ_ENABLES2_ROCE_MAX_GID_PER_VF = 0x200
FUNC_CFG_REQ_ENABLES2_XID_PARTITION_CFG = 0x400
FUNC_CFG_REQ_ENABLES2_PHYSICAL_SLOT_NUMBER = 0x800
FUNC_CFG_REQ_ENABLES2_PCIE_COMPLIANCE = 0x1000
FUNC_CFG_REQ_PORT_KDNET_MODE_DISABLED = 0x0
FUNC_CFG_REQ_PORT_KDNET_MODE_ENABLED = 0x1
FUNC_CFG_REQ_PORT_KDNET_MODE_LAST = FUNC_CFG_REQ_PORT_KDNET_MODE_ENABLED
FUNC_CFG_REQ_DB_PAGE_SIZE_4KB = 0x0
FUNC_CFG_REQ_DB_PAGE_SIZE_8KB = 0x1
FUNC_CFG_REQ_DB_PAGE_SIZE_16KB = 0x2
FUNC_CFG_REQ_DB_PAGE_SIZE_32KB = 0x3
FUNC_CFG_REQ_DB_PAGE_SIZE_64KB = 0x4
FUNC_CFG_REQ_DB_PAGE_SIZE_128KB = 0x5
FUNC_CFG_REQ_DB_PAGE_SIZE_256KB = 0x6
FUNC_CFG_REQ_DB_PAGE_SIZE_512KB = 0x7
FUNC_CFG_REQ_DB_PAGE_SIZE_1MB = 0x8
FUNC_CFG_REQ_DB_PAGE_SIZE_2MB = 0x9
FUNC_CFG_REQ_DB_PAGE_SIZE_4MB = 0xa
FUNC_CFG_REQ_DB_PAGE_SIZE_LAST = FUNC_CFG_REQ_DB_PAGE_SIZE_4MB
FUNC_CFG_REQ_XID_PARTITION_CFG_TX_CK = 0x1
FUNC_CFG_REQ_XID_PARTITION_CFG_RX_CK = 0x2
FUNC_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
FUNC_CFG_CMD_ERR_CODE_PARTITION_BW_OUT_OF_RANGE = 0x1
FUNC_CFG_CMD_ERR_CODE_NPAR_PARTITION_DOWN_FAILED = 0x2
FUNC_CFG_CMD_ERR_CODE_TPID_SET_DFLT_VLAN_NOT_SET = 0x3
FUNC_CFG_CMD_ERR_CODE_RES_ARRAY_ALLOC_FAILED = 0x4
FUNC_CFG_CMD_ERR_CODE_TX_RING_ASSET_TEST_FAILED = 0x5
FUNC_CFG_CMD_ERR_CODE_TX_RING_RES_UPDATE_FAILED = 0x6
FUNC_CFG_CMD_ERR_CODE_APPLY_MAX_BW_FAILED = 0x7
FUNC_CFG_CMD_ERR_CODE_ENABLE_EVB_FAILED = 0x8
FUNC_CFG_CMD_ERR_CODE_RSS_CTXT_ASSET_TEST_FAILED = 0x9
FUNC_CFG_CMD_ERR_CODE_RSS_CTXT_RES_UPDATE_FAILED = 0xa
FUNC_CFG_CMD_ERR_CODE_CMPL_RING_ASSET_TEST_FAILED = 0xb
FUNC_CFG_CMD_ERR_CODE_CMPL_RING_RES_UPDATE_FAILED = 0xc
FUNC_CFG_CMD_ERR_CODE_NQ_ASSET_TEST_FAILED = 0xd
FUNC_CFG_CMD_ERR_CODE_NQ_RES_UPDATE_FAILED = 0xe
FUNC_CFG_CMD_ERR_CODE_RX_RING_ASSET_TEST_FAILED = 0xf
FUNC_CFG_CMD_ERR_CODE_RX_RING_RES_UPDATE_FAILED = 0x10
FUNC_CFG_CMD_ERR_CODE_VNIC_ASSET_TEST_FAILED = 0x11
FUNC_CFG_CMD_ERR_CODE_VNIC_RES_UPDATE_FAILED = 0x12
FUNC_CFG_CMD_ERR_CODE_FAILED_TO_START_STATS_THREAD = 0x13
FUNC_CFG_CMD_ERR_CODE_RDMA_SRIOV_DISABLED = 0x14
FUNC_CFG_CMD_ERR_CODE_TX_KTLS_DISABLED = 0x15
FUNC_CFG_CMD_ERR_CODE_TX_KTLS_ASSET_TEST_FAILED = 0x16
FUNC_CFG_CMD_ERR_CODE_TX_KTLS_RES_UPDATE_FAILED = 0x17
FUNC_CFG_CMD_ERR_CODE_RX_KTLS_DISABLED = 0x18
FUNC_CFG_CMD_ERR_CODE_RX_KTLS_ASSET_TEST_FAILED = 0x19
FUNC_CFG_CMD_ERR_CODE_RX_KTLS_RES_UPDATE_FAILED = 0x1a
FUNC_CFG_CMD_ERR_CODE_TX_QUIC_DISABLED = 0x1b
FUNC_CFG_CMD_ERR_CODE_TX_QUIC_ASSET_TEST_FAILED = 0x1c
FUNC_CFG_CMD_ERR_CODE_TX_QUIC_RES_UPDATE_FAILED = 0x1d
FUNC_CFG_CMD_ERR_CODE_RX_QUIC_DISABLED = 0x1e
FUNC_CFG_CMD_ERR_CODE_RX_QUIC_ASSET_TEST_FAILED = 0x1f
FUNC_CFG_CMD_ERR_CODE_RX_QUIC_RES_UPDATE_FAILED = 0x20
FUNC_CFG_CMD_ERR_CODE_INVALID_KDNET_MODE = 0x21
FUNC_CFG_CMD_ERR_CODE_SCHQ_CFG_FAIL = 0x22
FUNC_CFG_CMD_ERR_CODE_LAST = FUNC_CFG_CMD_ERR_CODE_SCHQ_CFG_FAIL
FUNC_QSTATS_REQ_FLAGS_ROCE_ONLY = 0x1
FUNC_QSTATS_REQ_FLAGS_COUNTER_MASK = 0x2
FUNC_QSTATS_REQ_FLAGS_L2_ONLY = 0x4
FUNC_QSTATS_EXT_REQ_FLAGS_ROCE_ONLY = 0x1
FUNC_QSTATS_EXT_REQ_FLAGS_COUNTER_MASK = 0x2
FUNC_QSTATS_EXT_REQ_ENABLES_SCHQ_ID = 0x1
FUNC_DRV_RGTR_REQ_FLAGS_FWD_ALL_MODE = 0x1
FUNC_DRV_RGTR_REQ_FLAGS_FWD_NONE_MODE = 0x2
FUNC_DRV_RGTR_REQ_FLAGS_16BIT_VER_MODE = 0x4
FUNC_DRV_RGTR_REQ_FLAGS_FLOW_HANDLE_64BIT_MODE = 0x8
FUNC_DRV_RGTR_REQ_FLAGS_HOT_RESET_SUPPORT = 0x10
FUNC_DRV_RGTR_REQ_FLAGS_ERROR_RECOVERY_SUPPORT = 0x20
FUNC_DRV_RGTR_REQ_FLAGS_MASTER_SUPPORT = 0x40
FUNC_DRV_RGTR_REQ_FLAGS_FAST_RESET_SUPPORT = 0x80
FUNC_DRV_RGTR_REQ_FLAGS_RSS_STRICT_HASH_TYPE_SUPPORT = 0x100
FUNC_DRV_RGTR_REQ_FLAGS_NPAR_1_2_SUPPORT = 0x200
FUNC_DRV_RGTR_REQ_FLAGS_ASYM_QUEUE_CFG_SUPPORT = 0x400
FUNC_DRV_RGTR_REQ_FLAGS_TF_INGRESS_NIC_FLOW_MODE = 0x800
FUNC_DRV_RGTR_REQ_FLAGS_TF_EGRESS_NIC_FLOW_MODE = 0x1000
FUNC_DRV_RGTR_REQ_ENABLES_OS_TYPE = 0x1
FUNC_DRV_RGTR_REQ_ENABLES_VER = 0x2
FUNC_DRV_RGTR_REQ_ENABLES_TIMESTAMP = 0x4
FUNC_DRV_RGTR_REQ_ENABLES_VF_REQ_FWD = 0x8
FUNC_DRV_RGTR_REQ_ENABLES_ASYNC_EVENT_FWD = 0x10
FUNC_DRV_RGTR_REQ_OS_TYPE_UNKNOWN = 0x0
FUNC_DRV_RGTR_REQ_OS_TYPE_OTHER = 0x1
FUNC_DRV_RGTR_REQ_OS_TYPE_MSDOS = 0xe
FUNC_DRV_RGTR_REQ_OS_TYPE_WINDOWS = 0x12
FUNC_DRV_RGTR_REQ_OS_TYPE_SOLARIS = 0x1d
FUNC_DRV_RGTR_REQ_OS_TYPE_LINUX = 0x24
FUNC_DRV_RGTR_REQ_OS_TYPE_FREEBSD = 0x2a
FUNC_DRV_RGTR_REQ_OS_TYPE_ESXI = 0x68
FUNC_DRV_RGTR_REQ_OS_TYPE_WIN864 = 0x73
FUNC_DRV_RGTR_REQ_OS_TYPE_WIN2012R2 = 0x74
FUNC_DRV_RGTR_REQ_OS_TYPE_UEFI = 0x8000
FUNC_DRV_RGTR_REQ_OS_TYPE_LAST = FUNC_DRV_RGTR_REQ_OS_TYPE_UEFI
FUNC_DRV_RGTR_RESP_FLAGS_IF_CHANGE_SUPPORTED = 0x1
FUNC_DRV_UNRGTR_REQ_FLAGS_PREPARE_FOR_SHUTDOWN = 0x1
FUNC_BUF_RGTR_REQ_ENABLES_VF_ID = 0x1
FUNC_BUF_RGTR_REQ_ENABLES_ERR_BUF_ADDR = 0x2
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_16B = 0x4
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_4K = 0xc
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_8K = 0xd
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_64K = 0x10
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_2M = 0x15
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_4M = 0x16
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_1G = 0x1e
FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_LAST = FUNC_BUF_RGTR_REQ_REQ_BUF_PAGE_SIZE_1G
FUNC_DRV_QVER_REQ_DRIVER_TYPE_L2 = 0x0
FUNC_DRV_QVER_REQ_DRIVER_TYPE_ROCE = 0x1
FUNC_DRV_QVER_REQ_DRIVER_TYPE_LAST = FUNC_DRV_QVER_REQ_DRIVER_TYPE_ROCE
FUNC_DRV_QVER_RESP_OS_TYPE_UNKNOWN = 0x0
FUNC_DRV_QVER_RESP_OS_TYPE_OTHER = 0x1
FUNC_DRV_QVER_RESP_OS_TYPE_MSDOS = 0xe
FUNC_DRV_QVER_RESP_OS_TYPE_WINDOWS = 0x12
FUNC_DRV_QVER_RESP_OS_TYPE_SOLARIS = 0x1d
FUNC_DRV_QVER_RESP_OS_TYPE_LINUX = 0x24
FUNC_DRV_QVER_RESP_OS_TYPE_FREEBSD = 0x2a
FUNC_DRV_QVER_RESP_OS_TYPE_ESXI = 0x68
FUNC_DRV_QVER_RESP_OS_TYPE_WIN864 = 0x73
FUNC_DRV_QVER_RESP_OS_TYPE_WIN2012R2 = 0x74
FUNC_DRV_QVER_RESP_OS_TYPE_UEFI = 0x8000
FUNC_DRV_QVER_RESP_OS_TYPE_LAST = FUNC_DRV_QVER_RESP_OS_TYPE_UEFI
FUNC_RESOURCE_QCAPS_RESP_VF_RESERVATION_STRATEGY_MAXIMAL = 0x0
FUNC_RESOURCE_QCAPS_RESP_VF_RESERVATION_STRATEGY_MINIMAL = 0x1
FUNC_RESOURCE_QCAPS_RESP_VF_RESERVATION_STRATEGY_MINIMAL_STATIC = 0x2
FUNC_RESOURCE_QCAPS_RESP_VF_RESERVATION_STRATEGY_LAST = FUNC_RESOURCE_QCAPS_RESP_VF_RESERVATION_STRATEGY_MINIMAL_STATIC
FUNC_RESOURCE_QCAPS_RESP_FLAGS_MIN_GUARANTEED = 0x1
FUNC_VF_RESOURCE_CFG_REQ_FLAGS_MIN_GUARANTEED = 0x1
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_QP = 0x1
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_SRQ = 0x2
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_CQ = 0x4
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_VNIC = 0x8
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_STAT = 0x10
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_MRAV = 0x20
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_TKC = 0x40
FUNC_BACKING_STORE_QCAPS_RESP_CTX_INIT_MASK_RKC = 0x80
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_MASK = 0xf
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_SFT = 0
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_LVL_0 = 0x0
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_LVL_1 = 0x1
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_LVL_2 = 0x2
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_LAST = TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_LVL_LVL_2
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_MASK = 0xf0
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_SFT = 4
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_LAST = TQM_FP_RING_CFG_TQM_RING_CFG_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_FLAGS_PREBOOT_MODE = 0x1
FUNC_BACKING_STORE_CFG_REQ_FLAGS_MRAV_RESERVATION_SPLIT = 0x2
FUNC_BACKING_STORE_CFG_REQ_ENABLES_QP = 0x1
FUNC_BACKING_STORE_CFG_REQ_ENABLES_SRQ = 0x2
FUNC_BACKING_STORE_CFG_REQ_ENABLES_CQ = 0x4
FUNC_BACKING_STORE_CFG_REQ_ENABLES_VNIC = 0x8
FUNC_BACKING_STORE_CFG_REQ_ENABLES_STAT = 0x10
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_SP = 0x20
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING0 = 0x40
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING1 = 0x80
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING2 = 0x100
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING3 = 0x200
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING4 = 0x400
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING5 = 0x800
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING6 = 0x1000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING7 = 0x2000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_MRAV = 0x4000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TIM = 0x8000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING8 = 0x10000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING9 = 0x20000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TQM_RING10 = 0x40000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_TKC = 0x80000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_RKC = 0x100000
FUNC_BACKING_STORE_CFG_REQ_ENABLES_QP_FAST_QPMD = 0x200000
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_QPC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_QPC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_SRQ_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_SRQ_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_CQ_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_CQ_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_VNIC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_VNIC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_STAT_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_STAT_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_SP_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_SP_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING0_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING1_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING2_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING3_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING4_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING5_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING6_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TQM_RING7_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_MRAV_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_MRAV_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TIM_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TIM_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RING8_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RING9_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RING10_TQM_RING_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_TKC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_TKC_PG_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_MASK = 0xf
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_SFT = 0
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LAST = FUNC_BACKING_STORE_CFG_REQ_RKC_LVL_LVL_2
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_LAST = FUNC_BACKING_STORE_CFG_REQ_RKC_PG_SIZE_PG_1G
ERROR_RECOVERY_QCFG_RESP_FLAGS_HOST = 0x1
ERROR_RECOVERY_QCFG_RESP_FLAGS_CO_CPU = 0x2
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_MASK = 0x3
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_SFT = 0
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_PCIE_CFG = 0x0
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_GRC = 0x1
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_BAR0 = 0x2
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_BAR1 = 0x3
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_LAST = ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SPACE_BAR1
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_MASK = 0xfffffffc
ERROR_RECOVERY_QCFG_RESP_FW_HEALTH_STATUS_REG_ADDR_SFT = 2
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_MASK = 0x3
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_SFT = 0
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_PCIE_CFG = 0x0
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_GRC = 0x1
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_BAR0 = 0x2
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_BAR1 = 0x3
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_LAST = ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SPACE_BAR1
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_MASK = 0xfffffffc
ERROR_RECOVERY_QCFG_RESP_FW_HEARTBEAT_REG_ADDR_SFT = 2
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_MASK = 0x3
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_SFT = 0
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_PCIE_CFG = 0x0
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_GRC = 0x1
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_BAR0 = 0x2
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_BAR1 = 0x3
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_LAST = ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SPACE_BAR1
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_MASK = 0xfffffffc
ERROR_RECOVERY_QCFG_RESP_FW_RESET_CNT_REG_ADDR_SFT = 2
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_MASK = 0x3
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_SFT = 0
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_PCIE_CFG = 0x0
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_GRC = 0x1
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_BAR0 = 0x2
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_BAR1 = 0x3
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_LAST = ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SPACE_BAR1
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_MASK = 0xfffffffc
ERROR_RECOVERY_QCFG_RESP_RESET_INPROGRESS_REG_ADDR_SFT = 2
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_MASK = 0x3
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_SFT = 0
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_PCIE_CFG = 0x0
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_GRC = 0x1
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_BAR0 = 0x2
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_BAR1 = 0x3
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_LAST = ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SPACE_BAR1
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_MASK = 0xfffffffc
ERROR_RECOVERY_QCFG_RESP_RESET_REG_ADDR_SFT = 2
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_MASK = 0x3
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_SFT = 0
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_PCIE_CFG = 0x0
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_GRC = 0x1
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_BAR0 = 0x2
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_BAR1 = 0x3
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_LAST = ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SPACE_BAR1
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_MASK = 0xfffffffc
ERROR_RECOVERY_QCFG_RESP_ERR_RECOVERY_CNT_REG_ADDR_SFT = 2
FUNC_PTP_PIN_QCFG_RESP_STATE_PIN0_ENABLED = 0x1
FUNC_PTP_PIN_QCFG_RESP_STATE_PIN1_ENABLED = 0x2
FUNC_PTP_PIN_QCFG_RESP_STATE_PIN2_ENABLED = 0x4
FUNC_PTP_PIN_QCFG_RESP_STATE_PIN3_ENABLED = 0x8
FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_NONE = 0x0
FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_LAST = FUNC_PTP_PIN_QCFG_RESP_PIN0_USAGE_SYNC_OUT
FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_NONE = 0x0
FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_LAST = FUNC_PTP_PIN_QCFG_RESP_PIN1_USAGE_SYNC_OUT
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_NONE = 0x0
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_SYNCE_PRIMARY_CLOCK_OUT = 0x5
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_SYNCE_SECONDARY_CLOCK_OUT = 0x6
FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_LAST = FUNC_PTP_PIN_QCFG_RESP_PIN2_USAGE_SYNCE_SECONDARY_CLOCK_OUT
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_NONE = 0x0
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_SYNCE_PRIMARY_CLOCK_OUT = 0x5
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_SYNCE_SECONDARY_CLOCK_OUT = 0x6
FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_LAST = FUNC_PTP_PIN_QCFG_RESP_PIN3_USAGE_SYNCE_SECONDARY_CLOCK_OUT
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN0_STATE = 0x1
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN0_USAGE = 0x2
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN1_STATE = 0x4
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN1_USAGE = 0x8
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN2_STATE = 0x10
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN2_USAGE = 0x20
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN3_STATE = 0x40
FUNC_PTP_PIN_CFG_REQ_ENABLES_PIN3_USAGE = 0x80
FUNC_PTP_PIN_CFG_REQ_PIN0_STATE_DISABLED = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN0_STATE_ENABLED = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN0_STATE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN0_STATE_ENABLED
FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_NONE = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN0_USAGE_SYNC_OUT
FUNC_PTP_PIN_CFG_REQ_PIN1_STATE_DISABLED = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN1_STATE_ENABLED = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN1_STATE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN1_STATE_ENABLED
FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_NONE = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN1_USAGE_SYNC_OUT
FUNC_PTP_PIN_CFG_REQ_PIN2_STATE_DISABLED = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN2_STATE_ENABLED = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN2_STATE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN2_STATE_ENABLED
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_NONE = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_SYNCE_PRIMARY_CLOCK_OUT = 0x5
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_SYNCE_SECONDARY_CLOCK_OUT = 0x6
FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN2_USAGE_SYNCE_SECONDARY_CLOCK_OUT
FUNC_PTP_PIN_CFG_REQ_PIN3_STATE_DISABLED = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN3_STATE_ENABLED = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN3_STATE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN3_STATE_ENABLED
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_NONE = 0x0
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_PPS_IN = 0x1
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_PPS_OUT = 0x2
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_SYNC_IN = 0x3
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_SYNC_OUT = 0x4
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_SYNCE_PRIMARY_CLOCK_OUT = 0x5
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_SYNCE_SECONDARY_CLOCK_OUT = 0x6
FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_LAST = FUNC_PTP_PIN_CFG_REQ_PIN3_USAGE_SYNCE_SECONDARY_CLOCK_OUT
FUNC_PTP_CFG_REQ_ENABLES_PTP_PPS_EVENT = 0x1
FUNC_PTP_CFG_REQ_ENABLES_PTP_FREQ_ADJ_DLL_SOURCE = 0x2
FUNC_PTP_CFG_REQ_ENABLES_PTP_FREQ_ADJ_DLL_PHASE = 0x4
FUNC_PTP_CFG_REQ_ENABLES_PTP_FREQ_ADJ_EXT_PERIOD = 0x8
FUNC_PTP_CFG_REQ_ENABLES_PTP_FREQ_ADJ_EXT_UP = 0x10
FUNC_PTP_CFG_REQ_ENABLES_PTP_FREQ_ADJ_EXT_PHASE = 0x20
FUNC_PTP_CFG_REQ_ENABLES_PTP_SET_TIME = 0x40
FUNC_PTP_CFG_REQ_PTP_PPS_EVENT_INTERNAL = 0x1
FUNC_PTP_CFG_REQ_PTP_PPS_EVENT_EXTERNAL = 0x2
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_NONE = 0x0
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_TSIO_0 = 0x1
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_TSIO_1 = 0x2
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_TSIO_2 = 0x3
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_TSIO_3 = 0x4
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_PORT_0 = 0x5
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_PORT_1 = 0x6
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_PORT_2 = 0x7
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_PORT_3 = 0x8
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_INVALID = 0xff
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_LAST = FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_SOURCE_INVALID
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_NONE = 0x0
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_4K = 0x1
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_8K = 0x2
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_10M = 0x3
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_25M = 0x4
FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_LAST = FUNC_PTP_CFG_REQ_PTP_FREQ_ADJ_DLL_PHASE_25M
FUNC_PTP_TS_QUERY_REQ_FLAGS_PPS_TIME = 0x1
FUNC_PTP_TS_QUERY_REQ_FLAGS_PTM_TIME = 0x2
FUNC_PTP_EXT_CFG_REQ_ENABLES_PHC_MASTER_FID = 0x1
FUNC_PTP_EXT_CFG_REQ_ENABLES_PHC_SEC_FID = 0x2
FUNC_PTP_EXT_CFG_REQ_ENABLES_PHC_SEC_MODE = 0x4
FUNC_PTP_EXT_CFG_REQ_ENABLES_FAILOVER_TIMER = 0x8
FUNC_PTP_EXT_CFG_REQ_PHC_SEC_MODE_SWITCH = 0x0
FUNC_PTP_EXT_CFG_REQ_PHC_SEC_MODE_ALL = 0x1
FUNC_PTP_EXT_CFG_REQ_PHC_SEC_MODE_PF_ONLY = 0x2
FUNC_PTP_EXT_CFG_REQ_PHC_SEC_MODE_LAST = FUNC_PTP_EXT_CFG_REQ_PHC_SEC_MODE_PF_ONLY
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_QP = 0x0
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CQ = 0x2
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_STAT = 0x4
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TIM = 0xf
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_LAST = FUNC_BACKING_STORE_CFG_V2_REQ_TYPE_INVALID
FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_PREBOOT_MODE = 0x1
FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE = 0x2
FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_EXTEND = 0x4
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_MASK = 0xf
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_SFT = 0
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_0 = 0x0
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_1 = 0x1
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_2 = 0x2
FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LAST = FUNC_BACKING_STORE_CFG_V2_REQ_PBL_LEVEL_LVL_2
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_SFT = 4
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_LAST = FUNC_BACKING_STORE_CFG_V2_REQ_PAGE_SIZE_PG_1G
FUNC_BACKING_STORE_CFG_V2_REQ_ENABLES_NEXT_BS_OFFSET = 0x1
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_XID_PARTITION_TABLE = 0x1d
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_LAST = FUNC_BACKING_STORE_QCFG_V2_REQ_TYPE_INVALID
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_ERR_QPC_TRACE = 0x2a
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_LAST = FUNC_BACKING_STORE_QCFG_V2_RESP_TYPE_INVALID
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_MASK = 0xf
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_SFT = 0
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_0 = 0x0
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_1 = 0x1
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_2 = 0x2
FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LAST = FUNC_BACKING_STORE_QCFG_V2_RESP_PBL_LEVEL_LVL_2
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_MASK = 0xf0
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_SFT = 4
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_4K = (0x0 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_8K = (0x1 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_64K = (0x2 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_2M = (0x3 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_8M = (0x4 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_1G = (0x5 << 4)
FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_LAST = FUNC_BACKING_STORE_QCFG_V2_RESP_PAGE_SIZE_PG_1G
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_LAST = FUNC_BACKING_STORE_QCAPS_V2_REQ_TYPE_INVALID
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_QP = 0x0
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRQ = 0x1
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CQ = 0x2
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_VNIC = 0x3
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_STAT = 0x4
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SP_TQM_RING = 0x5
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_FP_TQM_RING = 0x6
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_MRAV = 0xe
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TIM = 0xf
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TX_CK = 0x13
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RX_CK = 0x14
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_MP_TQM_RING = 0x15
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SQ_DB_SHADOW = 0x16
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RQ_DB_SHADOW = 0x17
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRQ_DB_SHADOW = 0x18
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CQ_DB_SHADOW = 0x19
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TBL_SCOPE = 0x1c
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_XID_PARTITION = 0x1d
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRT_TRACE = 0x1e
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_SRT2_TRACE = 0x1f
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CRT_TRACE = 0x20
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CRT2_TRACE = 0x21
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RIGP0_TRACE = 0x22
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_L2_HWRM_TRACE = 0x23
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_ROCE_HWRM_TRACE = 0x24
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_TTX_PACING_TQM_RING = 0x25
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CA0_TRACE = 0x26
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CA1_TRACE = 0x27
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_CA2_TRACE = 0x28
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_RIGP1_TRACE = 0x29
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_AFM_KONG_HWRM_TRACE = 0x2a
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_ERR_QPC_TRACE = 0x2b
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_INVALID = 0xffff
FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_LAST = FUNC_BACKING_STORE_QCAPS_V2_RESP_TYPE_INVALID
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_ENABLE_CTX_KIND_INIT = 0x1
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_TYPE_VALID = 0x2
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_DRIVER_MANAGED_MEMORY = 0x4
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_ROCE_QP_PSEUDO_STATIC_ALLOC = 0x8
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_FW_DBG_TRACE = 0x10
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_FW_BIN_DBG_TRACE = 0x20
FUNC_BACKING_STORE_QCAPS_V2_RESP_FLAGS_NEXT_BS_OFFSET = 0x40
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_0_EXACT = 0x1
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_1_EXACT = 0x2
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_2_EXACT = 0x4
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_SPLIT_ENTRY_3_EXACT = 0x8
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_UNUSED_MASK = 0xf0
FUNC_BACKING_STORE_QCAPS_V2_RESP_EXACT_CNT_BIT_MAP_UNUSED_SFT = 4
FUNC_DBR_PACING_QCFG_RESP_FLAGS_DBR_NQ_EVENT_ENABLED = 0x1
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_MASK = 0x3
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_SFT = 0
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_PCIE_CFG = 0x0
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_GRC = 0x1
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_BAR0 = 0x2
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_BAR1 = 0x3
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_LAST = FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SPACE_BAR1
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_MASK = 0xfffffffc
FUNC_DBR_PACING_QCFG_RESP_DBR_STAT_DB_FIFO_REG_ADDR_SFT = 2
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_MASK = 0x3
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_SFT = 0
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_PCIE_CFG = 0x0
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_GRC = 0x1
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_BAR0 = 0x2
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_BAR1 = 0x3
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_LAST = FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SPACE_BAR1
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_MASK = 0xfffffffc
FUNC_DBR_PACING_QCFG_RESP_DBR_THROTTLING_AEQ_ARM_REG_ADDR_SFT = 2
FUNC_DRV_IF_CHANGE_REQ_FLAGS_UP = 0x1
FUNC_DRV_IF_CHANGE_RESP_FLAGS_RESC_CHANGE = 0x1
FUNC_DRV_IF_CHANGE_RESP_FLAGS_HOT_FW_RESET_DONE = 0x2
FUNC_DRV_IF_CHANGE_RESP_FLAGS_CAPS_CHANGE = 0x4
PORT_PHY_CFG_REQ_FLAGS_RESET_PHY = 0x1
PORT_PHY_CFG_REQ_FLAGS_DEPRECATED = 0x2
PORT_PHY_CFG_REQ_FLAGS_FORCE = 0x4
PORT_PHY_CFG_REQ_FLAGS_RESTART_AUTONEG = 0x8
PORT_PHY_CFG_REQ_FLAGS_EEE_ENABLE = 0x10
PORT_PHY_CFG_REQ_FLAGS_EEE_DISABLE = 0x20
PORT_PHY_CFG_REQ_FLAGS_EEE_TX_LPI_ENABLE = 0x40
PORT_PHY_CFG_REQ_FLAGS_EEE_TX_LPI_DISABLE = 0x80
PORT_PHY_CFG_REQ_FLAGS_FEC_AUTONEG_ENABLE = 0x100
PORT_PHY_CFG_REQ_FLAGS_FEC_AUTONEG_DISABLE = 0x200
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE74_ENABLE = 0x400
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE74_DISABLE = 0x800
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE91_ENABLE = 0x1000
PORT_PHY_CFG_REQ_FLAGS_FEC_CLAUSE91_DISABLE = 0x2000
PORT_PHY_CFG_REQ_FLAGS_FORCE_LINK_DWN = 0x4000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_1XN_ENABLE = 0x8000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_1XN_DISABLE = 0x10000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_IEEE_ENABLE = 0x20000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS544_IEEE_DISABLE = 0x40000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_1XN_ENABLE = 0x80000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_1XN_DISABLE = 0x100000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_IEEE_ENABLE = 0x200000
PORT_PHY_CFG_REQ_FLAGS_FEC_RS272_IEEE_DISABLE = 0x400000
PORT_PHY_CFG_REQ_FLAGS_LINK_TRAINING_ENABLE = 0x800000
PORT_PHY_CFG_REQ_FLAGS_LINK_TRAINING_DISABLE = 0x1000000
PORT_PHY_CFG_REQ_FLAGS_PRECODING_ENABLE = 0x2000000
PORT_PHY_CFG_REQ_FLAGS_PRECODING_DISABLE = 0x4000000
PORT_PHY_CFG_REQ_ENABLES_AUTO_MODE = 0x1
PORT_PHY_CFG_REQ_ENABLES_AUTO_DUPLEX = 0x2
PORT_PHY_CFG_REQ_ENABLES_AUTO_PAUSE = 0x4
PORT_PHY_CFG_REQ_ENABLES_AUTO_LINK_SPEED = 0x8
PORT_PHY_CFG_REQ_ENABLES_AUTO_LINK_SPEED_MASK = 0x10
PORT_PHY_CFG_REQ_ENABLES_WIRESPEED = 0x20
PORT_PHY_CFG_REQ_ENABLES_LPBK = 0x40
PORT_PHY_CFG_REQ_ENABLES_PREEMPHASIS = 0x80
PORT_PHY_CFG_REQ_ENABLES_FORCE_PAUSE = 0x100
PORT_PHY_CFG_REQ_ENABLES_EEE_LINK_SPEED_MASK = 0x200
PORT_PHY_CFG_REQ_ENABLES_TX_LPI_TIMER = 0x400
PORT_PHY_CFG_REQ_ENABLES_FORCE_PAM4_LINK_SPEED = 0x800
PORT_PHY_CFG_REQ_ENABLES_AUTO_PAM4_LINK_SPEED_MASK = 0x1000
PORT_PHY_CFG_REQ_ENABLES_FORCE_LINK_SPEEDS2 = 0x2000
PORT_PHY_CFG_REQ_ENABLES_AUTO_LINK_SPEEDS2_MASK = 0x4000
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_100MB = 0x1
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_1GB = 0xa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_2GB = 0x14
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_2_5GB = 0x19
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_10GB = 0x64
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_20GB = 0xc8
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_25GB = 0xfa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_40GB = 0x190
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_50GB = 0x1f4
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_100GB = 0x3e8
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_10MB = 0xffff
PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_LAST = PORT_PHY_CFG_REQ_FORCE_LINK_SPEED_10MB
PORT_PHY_CFG_REQ_AUTO_MODE_NONE = 0x0
PORT_PHY_CFG_REQ_AUTO_MODE_ALL_SPEEDS = 0x1
PORT_PHY_CFG_REQ_AUTO_MODE_ONE_SPEED = 0x2
PORT_PHY_CFG_REQ_AUTO_MODE_ONE_OR_BELOW = 0x3
PORT_PHY_CFG_REQ_AUTO_MODE_SPEED_MASK = 0x4
PORT_PHY_CFG_REQ_AUTO_MODE_LAST = PORT_PHY_CFG_REQ_AUTO_MODE_SPEED_MASK
PORT_PHY_CFG_REQ_AUTO_DUPLEX_HALF = 0x0
PORT_PHY_CFG_REQ_AUTO_DUPLEX_FULL = 0x1
PORT_PHY_CFG_REQ_AUTO_DUPLEX_BOTH = 0x2
PORT_PHY_CFG_REQ_AUTO_DUPLEX_LAST = PORT_PHY_CFG_REQ_AUTO_DUPLEX_BOTH
PORT_PHY_CFG_REQ_AUTO_PAUSE_TX = 0x1
PORT_PHY_CFG_REQ_AUTO_PAUSE_RX = 0x2
PORT_PHY_CFG_REQ_AUTO_PAUSE_AUTONEG_PAUSE = 0x4
PORT_PHY_CFG_REQ_MGMT_FLAG_LINK_RELEASE = 0x1
PORT_PHY_CFG_REQ_MGMT_FLAG_MGMT_VALID = 0x80
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_100MB = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_1GB = 0xa
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_2GB = 0x14
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_2_5GB = 0x19
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_10GB = 0x64
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_20GB = 0xc8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_25GB = 0xfa
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_40GB = 0x190
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_50GB = 0x1f4
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_100GB = 0x3e8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_10MB = 0xffff
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_LAST = PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_10MB
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_100MBHD = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_1GBHD = 0x4
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_2GB = 0x10
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_2_5GB = 0x20
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_20GB = 0x80
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_25GB = 0x100
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_40GB = 0x200
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_50GB = 0x400
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_100GB = 0x800
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_10MBHD = 0x1000
PORT_PHY_CFG_REQ_AUTO_LINK_SPEED_MASK_10MB = 0x2000
PORT_PHY_CFG_REQ_WIRESPEED_OFF = 0x0
PORT_PHY_CFG_REQ_WIRESPEED_ON = 0x1
PORT_PHY_CFG_REQ_WIRESPEED_LAST = PORT_PHY_CFG_REQ_WIRESPEED_ON
PORT_PHY_CFG_REQ_LPBK_NONE = 0x0
PORT_PHY_CFG_REQ_LPBK_LOCAL = 0x1
PORT_PHY_CFG_REQ_LPBK_REMOTE = 0x2
PORT_PHY_CFG_REQ_LPBK_EXTERNAL = 0x3
PORT_PHY_CFG_REQ_LPBK_LAST = PORT_PHY_CFG_REQ_LPBK_EXTERNAL
PORT_PHY_CFG_REQ_FORCE_PAUSE_TX = 0x1
PORT_PHY_CFG_REQ_FORCE_PAUSE_RX = 0x2
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD1 = 0x1
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD2 = 0x4
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD3 = 0x10
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_RSVD4 = 0x20
PORT_PHY_CFG_REQ_EEE_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_50GB = 0x1f4
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_100GB = 0x3e8
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_200GB = 0x7d0
PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_LAST = PORT_PHY_CFG_REQ_FORCE_PAM4_LINK_SPEED_200GB
PORT_PHY_CFG_REQ_TX_LPI_TIMER_MASK = 0xffffff
PORT_PHY_CFG_REQ_TX_LPI_TIMER_SFT = 0
PORT_PHY_CFG_REQ_AUTO_LINK_PAM4_SPEED_MASK_50G = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_PAM4_SPEED_MASK_100G = 0x2
PORT_PHY_CFG_REQ_AUTO_LINK_PAM4_SPEED_MASK_200G = 0x4
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_1GB = 0xa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_10GB = 0x64
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_25GB = 0xfa
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_40GB = 0x190
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_50GB = 0x1f4
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_100GB = 0x3e8
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_50GB_PAM4_56 = 0x1f5
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_100GB_PAM4_56 = 0x3e9
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_200GB_PAM4_56 = 0x7d1
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_400GB_PAM4_56 = 0xfa1
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_100GB_PAM4_112 = 0x3ea
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_200GB_PAM4_112 = 0x7d2
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_400GB_PAM4_112 = 0xfa2
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_800GB_PAM4_112 = 0x1f42
PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_LAST = PORT_PHY_CFG_REQ_FORCE_LINK_SPEEDS2_800GB_PAM4_112
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_1GB = 0x1
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_10GB = 0x2
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_25GB = 0x4
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_40GB = 0x8
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_50GB = 0x10
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_100GB = 0x20
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_50GB_PAM4_56 = 0x40
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_100GB_PAM4_56 = 0x80
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_200GB_PAM4_56 = 0x100
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_400GB_PAM4_56 = 0x200
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_100GB_PAM4_112 = 0x400
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_200GB_PAM4_112 = 0x800
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_400GB_PAM4_112 = 0x1000
PORT_PHY_CFG_REQ_AUTO_LINK_SPEEDS2_MASK_800GB_PAM4_112 = 0x2000
PORT_PHY_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
PORT_PHY_CFG_CMD_ERR_CODE_ILLEGAL_SPEED = 0x1
PORT_PHY_CFG_CMD_ERR_CODE_RETRY = 0x2
PORT_PHY_CFG_CMD_ERR_CODE_LAST = PORT_PHY_CFG_CMD_ERR_CODE_RETRY
PORT_PHY_QCFG_RESP_LINK_NO_LINK = 0x0
PORT_PHY_QCFG_RESP_LINK_SIGNAL = 0x1
PORT_PHY_QCFG_RESP_LINK_LINK = 0x2
PORT_PHY_QCFG_RESP_LINK_LAST = PORT_PHY_QCFG_RESP_LINK_LINK
PORT_PHY_QCFG_RESP_SIGNAL_MODE_MASK = 0xf
PORT_PHY_QCFG_RESP_SIGNAL_MODE_SFT = 0
PORT_PHY_QCFG_RESP_SIGNAL_MODE_NRZ = 0x0
PORT_PHY_QCFG_RESP_SIGNAL_MODE_PAM4 = 0x1
PORT_PHY_QCFG_RESP_SIGNAL_MODE_PAM4_112 = 0x2
PORT_PHY_QCFG_RESP_SIGNAL_MODE_LAST = PORT_PHY_QCFG_RESP_SIGNAL_MODE_PAM4_112
PORT_PHY_QCFG_RESP_ACTIVE_FEC_MASK = 0xf0
PORT_PHY_QCFG_RESP_ACTIVE_FEC_SFT = 4
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_NONE_ACTIVE = (0x0 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_CLAUSE74_ACTIVE = (0x1 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_CLAUSE91_ACTIVE = (0x2 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_RS544_1XN_ACTIVE = (0x3 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_RS544_IEEE_ACTIVE = (0x4 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_RS272_1XN_ACTIVE = (0x5 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_RS272_IEEE_ACTIVE = (0x6 << 4)
PORT_PHY_QCFG_RESP_ACTIVE_FEC_LAST = PORT_PHY_QCFG_RESP_ACTIVE_FEC_FEC_RS272_IEEE_ACTIVE
PORT_PHY_QCFG_RESP_LINK_SPEED_100MB = 0x1
PORT_PHY_QCFG_RESP_LINK_SPEED_1GB = 0xa
PORT_PHY_QCFG_RESP_LINK_SPEED_2GB = 0x14
PORT_PHY_QCFG_RESP_LINK_SPEED_2_5GB = 0x19
PORT_PHY_QCFG_RESP_LINK_SPEED_10GB = 0x64
PORT_PHY_QCFG_RESP_LINK_SPEED_20GB = 0xc8
PORT_PHY_QCFG_RESP_LINK_SPEED_25GB = 0xfa
PORT_PHY_QCFG_RESP_LINK_SPEED_40GB = 0x190
PORT_PHY_QCFG_RESP_LINK_SPEED_50GB = 0x1f4
PORT_PHY_QCFG_RESP_LINK_SPEED_100GB = 0x3e8
PORT_PHY_QCFG_RESP_LINK_SPEED_200GB = 0x7d0
PORT_PHY_QCFG_RESP_LINK_SPEED_400GB = 0xfa0
PORT_PHY_QCFG_RESP_LINK_SPEED_800GB = 0x1f40
PORT_PHY_QCFG_RESP_LINK_SPEED_10MB = 0xffff
PORT_PHY_QCFG_RESP_LINK_SPEED_LAST = PORT_PHY_QCFG_RESP_LINK_SPEED_10MB
PORT_PHY_QCFG_RESP_DUPLEX_CFG_HALF = 0x0
PORT_PHY_QCFG_RESP_DUPLEX_CFG_FULL = 0x1
PORT_PHY_QCFG_RESP_DUPLEX_CFG_LAST = PORT_PHY_QCFG_RESP_DUPLEX_CFG_FULL
PORT_PHY_QCFG_RESP_PAUSE_TX = 0x1
PORT_PHY_QCFG_RESP_PAUSE_RX = 0x2
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_100MBHD = 0x1
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_100MB = 0x2
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_1GBHD = 0x4
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_1GB = 0x8
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_2GB = 0x10
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_2_5GB = 0x20
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_10GB = 0x40
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_20GB = 0x80
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_25GB = 0x100
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_40GB = 0x200
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_50GB = 0x400
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_100GB = 0x800
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_10MBHD = 0x1000
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS_10MB = 0x2000
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_100MB = 0x1
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_1GB = 0xa
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_2GB = 0x14
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_2_5GB = 0x19
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_10GB = 0x64
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_20GB = 0xc8
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_25GB = 0xfa
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_40GB = 0x190
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_50GB = 0x1f4
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_100GB = 0x3e8
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_10MB = 0xffff
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_LAST = PORT_PHY_QCFG_RESP_FORCE_LINK_SPEED_10MB
PORT_PHY_QCFG_RESP_AUTO_MODE_NONE = 0x0
PORT_PHY_QCFG_RESP_AUTO_MODE_ALL_SPEEDS = 0x1
PORT_PHY_QCFG_RESP_AUTO_MODE_ONE_SPEED = 0x2
PORT_PHY_QCFG_RESP_AUTO_MODE_ONE_OR_BELOW = 0x3
PORT_PHY_QCFG_RESP_AUTO_MODE_SPEED_MASK = 0x4
PORT_PHY_QCFG_RESP_AUTO_MODE_LAST = PORT_PHY_QCFG_RESP_AUTO_MODE_SPEED_MASK
PORT_PHY_QCFG_RESP_AUTO_PAUSE_TX = 0x1
PORT_PHY_QCFG_RESP_AUTO_PAUSE_RX = 0x2
PORT_PHY_QCFG_RESP_AUTO_PAUSE_AUTONEG_PAUSE = 0x4
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_100MB = 0x1
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_1GB = 0xa
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_2GB = 0x14
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_2_5GB = 0x19
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_10GB = 0x64
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_20GB = 0xc8
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_25GB = 0xfa
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_40GB = 0x190
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_50GB = 0x1f4
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_100GB = 0x3e8
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_10MB = 0xffff
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_LAST = PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_10MB
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_100MBHD = 0x1
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_1GBHD = 0x4
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_2GB = 0x10
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_2_5GB = 0x20
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_20GB = 0x80
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_25GB = 0x100
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_40GB = 0x200
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_50GB = 0x400
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_100GB = 0x800
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_10MBHD = 0x1000
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEED_MASK_10MB = 0x2000
PORT_PHY_QCFG_RESP_WIRESPEED_OFF = 0x0
PORT_PHY_QCFG_RESP_WIRESPEED_ON = 0x1
PORT_PHY_QCFG_RESP_WIRESPEED_LAST = PORT_PHY_QCFG_RESP_WIRESPEED_ON
PORT_PHY_QCFG_RESP_LPBK_NONE = 0x0
PORT_PHY_QCFG_RESP_LPBK_LOCAL = 0x1
PORT_PHY_QCFG_RESP_LPBK_REMOTE = 0x2
PORT_PHY_QCFG_RESP_LPBK_EXTERNAL = 0x3
PORT_PHY_QCFG_RESP_LPBK_LAST = PORT_PHY_QCFG_RESP_LPBK_EXTERNAL
PORT_PHY_QCFG_RESP_FORCE_PAUSE_TX = 0x1
PORT_PHY_QCFG_RESP_FORCE_PAUSE_RX = 0x2
PORT_PHY_QCFG_RESP_MODULE_STATUS_NONE = 0x0
PORT_PHY_QCFG_RESP_MODULE_STATUS_DISABLETX = 0x1
PORT_PHY_QCFG_RESP_MODULE_STATUS_WARNINGMSG = 0x2
PORT_PHY_QCFG_RESP_MODULE_STATUS_PWRDOWN = 0x3
PORT_PHY_QCFG_RESP_MODULE_STATUS_NOTINSERTED = 0x4
PORT_PHY_QCFG_RESP_MODULE_STATUS_CURRENTFAULT = 0x5
PORT_PHY_QCFG_RESP_MODULE_STATUS_OVERHEATED = 0x6
PORT_PHY_QCFG_RESP_MODULE_STATUS_NOTAPPLICABLE = 0xff
PORT_PHY_QCFG_RESP_MODULE_STATUS_LAST = PORT_PHY_QCFG_RESP_MODULE_STATUS_NOTAPPLICABLE
PORT_PHY_QCFG_RESP_PHY_TYPE_UNKNOWN = 0x0
PORT_PHY_QCFG_RESP_PHY_TYPE_BASECR = 0x1
PORT_PHY_QCFG_RESP_PHY_TYPE_BASEKR4 = 0x2
PORT_PHY_QCFG_RESP_PHY_TYPE_BASELR = 0x3
PORT_PHY_QCFG_RESP_PHY_TYPE_BASESR = 0x4
PORT_PHY_QCFG_RESP_PHY_TYPE_BASEKR2 = 0x5
PORT_PHY_QCFG_RESP_PHY_TYPE_BASEKX = 0x6
PORT_PHY_QCFG_RESP_PHY_TYPE_BASEKR = 0x7
PORT_PHY_QCFG_RESP_PHY_TYPE_BASET = 0x8
PORT_PHY_QCFG_RESP_PHY_TYPE_BASETE = 0x9
PORT_PHY_QCFG_RESP_PHY_TYPE_SGMIIEXTPHY = 0xa
PORT_PHY_QCFG_RESP_PHY_TYPE_25G_BASECR_CA_L = 0xb
PORT_PHY_QCFG_RESP_PHY_TYPE_25G_BASECR_CA_S = 0xc
PORT_PHY_QCFG_RESP_PHY_TYPE_25G_BASECR_CA_N = 0xd
PORT_PHY_QCFG_RESP_PHY_TYPE_25G_BASESR = 0xe
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASECR4 = 0xf
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASESR4 = 0x10
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASELR4 = 0x11
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASEER4 = 0x12
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASESR10 = 0x13
PORT_PHY_QCFG_RESP_PHY_TYPE_40G_BASECR4 = 0x14
PORT_PHY_QCFG_RESP_PHY_TYPE_40G_BASESR4 = 0x15
PORT_PHY_QCFG_RESP_PHY_TYPE_40G_BASELR4 = 0x16
PORT_PHY_QCFG_RESP_PHY_TYPE_40G_BASEER4 = 0x17
PORT_PHY_QCFG_RESP_PHY_TYPE_40G_ACTIVE_CABLE = 0x18
PORT_PHY_QCFG_RESP_PHY_TYPE_1G_BASET = 0x19
PORT_PHY_QCFG_RESP_PHY_TYPE_1G_BASESX = 0x1a
PORT_PHY_QCFG_RESP_PHY_TYPE_1G_BASECX = 0x1b
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASECR4 = 0x1c
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASESR4 = 0x1d
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASELR4 = 0x1e
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASEER4 = 0x1f
PORT_PHY_QCFG_RESP_PHY_TYPE_50G_BASECR = 0x20
PORT_PHY_QCFG_RESP_PHY_TYPE_50G_BASESR = 0x21
PORT_PHY_QCFG_RESP_PHY_TYPE_50G_BASELR = 0x22
PORT_PHY_QCFG_RESP_PHY_TYPE_50G_BASEER = 0x23
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASECR2 = 0x24
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASESR2 = 0x25
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASELR2 = 0x26
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASEER2 = 0x27
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASECR = 0x28
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASESR = 0x29
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASELR = 0x2a
PORT_PHY_QCFG_RESP_PHY_TYPE_100G_BASEER = 0x2b
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASECR2 = 0x2c
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASESR2 = 0x2d
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASELR2 = 0x2e
PORT_PHY_QCFG_RESP_PHY_TYPE_200G_BASEER2 = 0x2f
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASECR8 = 0x30
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASESR8 = 0x31
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASELR8 = 0x32
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASEER8 = 0x33
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASECR4 = 0x34
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASESR4 = 0x35
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASELR4 = 0x36
PORT_PHY_QCFG_RESP_PHY_TYPE_400G_BASEER4 = 0x37
PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASECR8 = 0x38
PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASESR8 = 0x39
PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASELR8 = 0x3a
PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASEER8 = 0x3b
PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASEFR8 = 0x3c
PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASEDR8 = 0x3d
PORT_PHY_QCFG_RESP_PHY_TYPE_LAST = PORT_PHY_QCFG_RESP_PHY_TYPE_800G_BASEDR8
PORT_PHY_QCFG_RESP_MEDIA_TYPE_UNKNOWN = 0x0
PORT_PHY_QCFG_RESP_MEDIA_TYPE_TP = 0x1
PORT_PHY_QCFG_RESP_MEDIA_TYPE_DAC = 0x2
PORT_PHY_QCFG_RESP_MEDIA_TYPE_FIBRE = 0x3
PORT_PHY_QCFG_RESP_MEDIA_TYPE_BACKPLANE = 0x4
PORT_PHY_QCFG_RESP_MEDIA_TYPE_LAST = PORT_PHY_QCFG_RESP_MEDIA_TYPE_BACKPLANE
PORT_PHY_QCFG_RESP_XCVR_PKG_TYPE_XCVR_INTERNAL = 0x1
PORT_PHY_QCFG_RESP_XCVR_PKG_TYPE_XCVR_EXTERNAL = 0x2
PORT_PHY_QCFG_RESP_XCVR_PKG_TYPE_LAST = PORT_PHY_QCFG_RESP_XCVR_PKG_TYPE_XCVR_EXTERNAL
PORT_PHY_QCFG_RESP_PHY_ADDR_MASK = 0x1f
PORT_PHY_QCFG_RESP_PHY_ADDR_SFT = 0
PORT_PHY_QCFG_RESP_EEE_CONFIG_MASK = 0xe0
PORT_PHY_QCFG_RESP_EEE_CONFIG_SFT = 5
PORT_PHY_QCFG_RESP_EEE_CONFIG_EEE_ENABLED = 0x20
PORT_PHY_QCFG_RESP_EEE_CONFIG_EEE_ACTIVE = 0x40
PORT_PHY_QCFG_RESP_EEE_CONFIG_EEE_TX_LPI = 0x80
PORT_PHY_QCFG_RESP_PARALLEL_DETECT = 0x1
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_100MBHD = 0x1
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_100MB = 0x2
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_1GBHD = 0x4
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_1GB = 0x8
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_2GB = 0x10
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_2_5GB = 0x20
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_10GB = 0x40
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_20GB = 0x80
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_25GB = 0x100
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_40GB = 0x200
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_50GB = 0x400
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_100GB = 0x800
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_10MBHD = 0x1000
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_SPEEDS_10MB = 0x2000
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_NONE = 0x0
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_ALL_SPEEDS = 0x1
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_ONE_SPEED = 0x2
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_ONE_OR_BELOW = 0x3
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_SPEED_MASK = 0x4
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_LAST = PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_AUTO_MODE_SPEED_MASK
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_PAUSE_TX = 0x1
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_PAUSE_RX = 0x2
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_RSVD1 = 0x1
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_RSVD2 = 0x4
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_RSVD3 = 0x10
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_RSVD4 = 0x20
PORT_PHY_QCFG_RESP_ADV_EEE_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_RSVD1 = 0x1
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_100MB = 0x2
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_RSVD2 = 0x4
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_1GB = 0x8
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_RSVD3 = 0x10
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_RSVD4 = 0x20
PORT_PHY_QCFG_RESP_LINK_PARTNER_ADV_EEE_LINK_SPEED_MASK_10GB = 0x40
PORT_PHY_QCFG_RESP_TX_LPI_TIMER_MASK = 0xffffff
PORT_PHY_QCFG_RESP_TX_LPI_TIMER_SFT = 0
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_MASK = 0xff000000
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_SFT = 24
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_UNKNOWN = (0x0 << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_SFP = (0x3 << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_QSFP = (0xc << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_QSFPPLUS = (0xd << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_QSFP28 = (0x11 << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_QSFPDD = (0x18 << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_QSFP112 = (0x1e << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_SFPDD = (0x1f << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_CSFP = (0x20 << 24)
PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_LAST = PORT_PHY_QCFG_RESP_XCVR_IDENTIFIER_TYPE_CSFP
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_NONE_SUPPORTED = 0x1
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_AUTONEG_SUPPORTED = 0x2
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_AUTONEG_ENABLED = 0x4
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_CLAUSE74_SUPPORTED = 0x8
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_CLAUSE74_ENABLED = 0x10
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_CLAUSE91_SUPPORTED = 0x20
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_CLAUSE91_ENABLED = 0x40
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS544_1XN_SUPPORTED = 0x80
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS544_1XN_ENABLED = 0x100
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS544_IEEE_SUPPORTED = 0x200
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS544_IEEE_ENABLED = 0x400
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS272_1XN_SUPPORTED = 0x800
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS272_1XN_ENABLED = 0x1000
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS272_IEEE_SUPPORTED = 0x2000
PORT_PHY_QCFG_RESP_FEC_CFG_FEC_RS272_IEEE_ENABLED = 0x4000
PORT_PHY_QCFG_RESP_DUPLEX_STATE_HALF = 0x0
PORT_PHY_QCFG_RESP_DUPLEX_STATE_FULL = 0x1
PORT_PHY_QCFG_RESP_DUPLEX_STATE_LAST = PORT_PHY_QCFG_RESP_DUPLEX_STATE_FULL
PORT_PHY_QCFG_RESP_OPTION_FLAGS_MEDIA_AUTO_DETECT = 0x1
PORT_PHY_QCFG_RESP_OPTION_FLAGS_SIGNAL_MODE_KNOWN = 0x2
PORT_PHY_QCFG_RESP_OPTION_FLAGS_SPEEDS2_SUPPORTED = 0x4
PORT_PHY_QCFG_RESP_OPTION_FLAGS_LINK_TRAINING = 0x8
PORT_PHY_QCFG_RESP_OPTION_FLAGS_PRECODING = 0x10
PORT_PHY_QCFG_RESP_SUPPORT_PAM4_SPEEDS_50G = 0x1
PORT_PHY_QCFG_RESP_SUPPORT_PAM4_SPEEDS_100G = 0x2
PORT_PHY_QCFG_RESP_SUPPORT_PAM4_SPEEDS_200G = 0x4
PORT_PHY_QCFG_RESP_FORCE_PAM4_LINK_SPEED_50GB = 0x1f4
PORT_PHY_QCFG_RESP_FORCE_PAM4_LINK_SPEED_100GB = 0x3e8
PORT_PHY_QCFG_RESP_FORCE_PAM4_LINK_SPEED_200GB = 0x7d0
PORT_PHY_QCFG_RESP_FORCE_PAM4_LINK_SPEED_LAST = PORT_PHY_QCFG_RESP_FORCE_PAM4_LINK_SPEED_200GB
PORT_PHY_QCFG_RESP_AUTO_PAM4_LINK_SPEED_MASK_50G = 0x1
PORT_PHY_QCFG_RESP_AUTO_PAM4_LINK_SPEED_MASK_100G = 0x2
PORT_PHY_QCFG_RESP_AUTO_PAM4_LINK_SPEED_MASK_200G = 0x4
PORT_PHY_QCFG_RESP_LINK_PARTNER_PAM4_ADV_SPEEDS_50GB = 0x1
PORT_PHY_QCFG_RESP_LINK_PARTNER_PAM4_ADV_SPEEDS_100GB = 0x2
PORT_PHY_QCFG_RESP_LINK_PARTNER_PAM4_ADV_SPEEDS_200GB = 0x4
PORT_PHY_QCFG_RESP_LINK_DOWN_REASON_RF = 0x1
PORT_PHY_QCFG_RESP_LINK_DOWN_REASON_OTP_SPEED_VIOLATION = 0x2
PORT_PHY_QCFG_RESP_LINK_DOWN_REASON_CABLE_REMOVED = 0x4
PORT_PHY_QCFG_RESP_LINK_DOWN_REASON_MODULE_FAULT = 0x8
PORT_PHY_QCFG_RESP_LINK_DOWN_REASON_BMC_REQUEST = 0x10
PORT_PHY_QCFG_RESP_LINK_DOWN_REASON_TX_LASER_DISABLED = 0x20
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_1GB = 0x1
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_10GB = 0x2
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_25GB = 0x4
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_40GB = 0x8
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_50GB = 0x10
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_100GB = 0x20
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_50GB_PAM4_56 = 0x40
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_100GB_PAM4_56 = 0x80
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_200GB_PAM4_56 = 0x100
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_400GB_PAM4_56 = 0x200
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_100GB_PAM4_112 = 0x400
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_200GB_PAM4_112 = 0x800
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_400GB_PAM4_112 = 0x1000
PORT_PHY_QCFG_RESP_SUPPORT_SPEEDS2_800GB_PAM4_112 = 0x2000
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_1GB = 0xa
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_10GB = 0x64
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_25GB = 0xfa
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_40GB = 0x190
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_50GB = 0x1f4
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_100GB = 0x3e8
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_50GB_PAM4_56 = 0x1f5
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_100GB_PAM4_56 = 0x3e9
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_200GB_PAM4_56 = 0x7d1
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_400GB_PAM4_56 = 0xfa1
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_100GB_PAM4_112 = 0x3ea
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_200GB_PAM4_112 = 0x7d2
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_400GB_PAM4_112 = 0xfa2
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_800GB_PAM4_112 = 0x1f42
PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_LAST = PORT_PHY_QCFG_RESP_FORCE_LINK_SPEEDS2_800GB_PAM4_112
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_1GB = 0x1
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_10GB = 0x2
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_25GB = 0x4
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_40GB = 0x8
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_50GB = 0x10
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_100GB = 0x20
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_50GB_PAM4_56 = 0x40
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_100GB_PAM4_56 = 0x80
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_200GB_PAM4_56 = 0x100
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_400GB_PAM4_56 = 0x200
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_100GB_PAM4_112 = 0x400
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_200GB_PAM4_112 = 0x800
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_400GB_PAM4_112 = 0x1000
PORT_PHY_QCFG_RESP_AUTO_LINK_SPEEDS2_800GB_PAM4_112 = 0x2000
PORT_MAC_CFG_REQ_FLAGS_MATCH_LINK = 0x1
PORT_MAC_CFG_REQ_FLAGS_VLAN_PRI2COS_ENABLE = 0x2
PORT_MAC_CFG_REQ_FLAGS_TUNNEL_PRI2COS_ENABLE = 0x4
PORT_MAC_CFG_REQ_FLAGS_IP_DSCP2COS_ENABLE = 0x8
PORT_MAC_CFG_REQ_FLAGS_PTP_RX_TS_CAPTURE_ENABLE = 0x10
PORT_MAC_CFG_REQ_FLAGS_PTP_RX_TS_CAPTURE_DISABLE = 0x20
PORT_MAC_CFG_REQ_FLAGS_PTP_TX_TS_CAPTURE_ENABLE = 0x40
PORT_MAC_CFG_REQ_FLAGS_PTP_TX_TS_CAPTURE_DISABLE = 0x80
PORT_MAC_CFG_REQ_FLAGS_OOB_WOL_ENABLE = 0x100
PORT_MAC_CFG_REQ_FLAGS_OOB_WOL_DISABLE = 0x200
PORT_MAC_CFG_REQ_FLAGS_VLAN_PRI2COS_DISABLE = 0x400
PORT_MAC_CFG_REQ_FLAGS_TUNNEL_PRI2COS_DISABLE = 0x800
PORT_MAC_CFG_REQ_FLAGS_IP_DSCP2COS_DISABLE = 0x1000
PORT_MAC_CFG_REQ_FLAGS_PTP_ONE_STEP_TX_TS = 0x2000
PORT_MAC_CFG_REQ_FLAGS_ALL_RX_TS_CAPTURE_ENABLE = 0x4000
PORT_MAC_CFG_REQ_FLAGS_ALL_RX_TS_CAPTURE_DISABLE = 0x8000
PORT_MAC_CFG_REQ_ENABLES_IPG = 0x1
PORT_MAC_CFG_REQ_ENABLES_LPBK = 0x2
PORT_MAC_CFG_REQ_ENABLES_VLAN_PRI2COS_MAP_PRI = 0x4
PORT_MAC_CFG_REQ_ENABLES_TUNNEL_PRI2COS_MAP_PRI = 0x10
PORT_MAC_CFG_REQ_ENABLES_DSCP2COS_MAP_PRI = 0x20
PORT_MAC_CFG_REQ_ENABLES_RX_TS_CAPTURE_PTP_MSG_TYPE = 0x40
PORT_MAC_CFG_REQ_ENABLES_TX_TS_CAPTURE_PTP_MSG_TYPE = 0x80
PORT_MAC_CFG_REQ_ENABLES_COS_FIELD_CFG = 0x100
PORT_MAC_CFG_REQ_ENABLES_PTP_FREQ_ADJ_PPB = 0x200
PORT_MAC_CFG_REQ_ENABLES_PTP_ADJ_PHASE = 0x400
PORT_MAC_CFG_REQ_ENABLES_PTP_LOAD_CONTROL = 0x800
PORT_MAC_CFG_REQ_LPBK_NONE = 0x0
PORT_MAC_CFG_REQ_LPBK_LOCAL = 0x1
PORT_MAC_CFG_REQ_LPBK_REMOTE = 0x2
PORT_MAC_CFG_REQ_LPBK_LAST = PORT_MAC_CFG_REQ_LPBK_REMOTE
PORT_MAC_CFG_REQ_COS_FIELD_CFG_RSVD1 = 0x1
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_MASK = 0x6
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_SFT = 1
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_INNERMOST = (0x0 << 1)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_OUTER = (0x1 << 1)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_OUTERMOST = (0x2 << 1)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_UNSPECIFIED = (0x3 << 1)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_LAST = PORT_MAC_CFG_REQ_COS_FIELD_CFG_VLAN_PRI_SEL_UNSPECIFIED
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_MASK = 0x18
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_SFT = 3
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_INNERMOST = (0x0 << 3)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_OUTER = (0x1 << 3)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_OUTERMOST = (0x2 << 3)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_UNSPECIFIED = (0x3 << 3)
PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_LAST = PORT_MAC_CFG_REQ_COS_FIELD_CFG_T_VLAN_PRI_SEL_UNSPECIFIED
PORT_MAC_CFG_REQ_COS_FIELD_CFG_DEFAULT_COS_MASK = 0xe0
PORT_MAC_CFG_REQ_COS_FIELD_CFG_DEFAULT_COS_SFT = 5
PORT_MAC_CFG_REQ_PTP_LOAD_CONTROL_NONE = 0x0
PORT_MAC_CFG_REQ_PTP_LOAD_CONTROL_IMMEDIATE = 0x1
PORT_MAC_CFG_REQ_PTP_LOAD_CONTROL_PPS_EVENT = 0x2
PORT_MAC_CFG_REQ_PTP_LOAD_CONTROL_LAST = PORT_MAC_CFG_REQ_PTP_LOAD_CONTROL_PPS_EVENT
PORT_MAC_CFG_RESP_LPBK_NONE = 0x0
PORT_MAC_CFG_RESP_LPBK_LOCAL = 0x1
PORT_MAC_CFG_RESP_LPBK_REMOTE = 0x2
PORT_MAC_CFG_RESP_LPBK_LAST = PORT_MAC_CFG_RESP_LPBK_REMOTE
PORT_MAC_PTP_QCFG_RESP_FLAGS_DIRECT_ACCESS = 0x1
PORT_MAC_PTP_QCFG_RESP_FLAGS_ONE_STEP_TX_TS = 0x4
PORT_MAC_PTP_QCFG_RESP_FLAGS_HWRM_ACCESS = 0x8
PORT_MAC_PTP_QCFG_RESP_FLAGS_PARTIAL_DIRECT_ACCESS_REF_CLOCK = 0x10
PORT_MAC_PTP_QCFG_RESP_FLAGS_RTC_CONFIGURED = 0x20
PORT_MAC_PTP_QCFG_RESP_FLAGS_64B_PHC_TIME = 0x40
PORT_QSTATS_REQ_FLAGS_COUNTER_MASK = 0x1
PORT_QSTATS_RESP_FLAGS_CLEARED = 0x1
PORT_QSTATS_EXT_REQ_FLAGS_COUNTER_MASK = 0x1
PORT_QSTATS_EXT_RESP_FLAGS_CLEAR_ROCE_COUNTERS_SUPPORTED = 0x1
PORT_QSTATS_EXT_RESP_FLAGS_CLEARED = 0x2
PORT_LPBK_QSTATS_REQ_FLAGS_COUNTER_MASK = 0x1
PORT_ECN_QSTATS_REQ_FLAGS_COUNTER_MASK = 0x1
PORT_CLR_STATS_REQ_FLAGS_ROCE_COUNTERS = 0x1
PORT_TS_QUERY_REQ_FLAGS_PATH = 0x1
PORT_TS_QUERY_REQ_FLAGS_PATH_TX = 0x0
PORT_TS_QUERY_REQ_FLAGS_PATH_RX = 0x1
PORT_TS_QUERY_REQ_FLAGS_PATH_LAST = PORT_TS_QUERY_REQ_FLAGS_PATH_RX
PORT_TS_QUERY_REQ_FLAGS_CURRENT_TIME = 0x2
PORT_TS_QUERY_REQ_ENABLES_TS_REQ_TIMEOUT = 0x1
PORT_TS_QUERY_REQ_ENABLES_PTP_SEQ_ID = 0x2
PORT_TS_QUERY_REQ_ENABLES_PTP_HDR_OFFSET = 0x4
PORT_PHY_QCAPS_RESP_FLAGS_EEE_SUPPORTED = 0x1
PORT_PHY_QCAPS_RESP_FLAGS_EXTERNAL_LPBK_SUPPORTED = 0x2
PORT_PHY_QCAPS_RESP_FLAGS_AUTONEG_LPBK_SUPPORTED = 0x4
PORT_PHY_QCAPS_RESP_FLAGS_SHARED_PHY_CFG_SUPPORTED = 0x8
PORT_PHY_QCAPS_RESP_FLAGS_CUMULATIVE_COUNTERS_ON_RESET = 0x10
PORT_PHY_QCAPS_RESP_FLAGS_LOCAL_LPBK_NOT_SUPPORTED = 0x20
PORT_PHY_QCAPS_RESP_FLAGS_FW_MANAGED_LINK_DOWN = 0x40
PORT_PHY_QCAPS_RESP_FLAGS_NO_FCS = 0x80
PORT_PHY_QCAPS_RESP_PORT_CNT_UNKNOWN = 0x0
PORT_PHY_QCAPS_RESP_PORT_CNT_1 = 0x1
PORT_PHY_QCAPS_RESP_PORT_CNT_2 = 0x2
PORT_PHY_QCAPS_RESP_PORT_CNT_3 = 0x3
PORT_PHY_QCAPS_RESP_PORT_CNT_4 = 0x4
PORT_PHY_QCAPS_RESP_PORT_CNT_12 = 0xc
PORT_PHY_QCAPS_RESP_PORT_CNT_LAST = PORT_PHY_QCAPS_RESP_PORT_CNT_12
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_100MBHD = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_100MB = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_1GBHD = 0x4
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_1GB = 0x8
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_2GB = 0x10
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_2_5GB = 0x20
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_10GB = 0x40
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_20GB = 0x80
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_25GB = 0x100
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_40GB = 0x200
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_50GB = 0x400
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_100GB = 0x800
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_10MBHD = 0x1000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_FORCE_MODE_10MB = 0x2000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_100MBHD = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_100MB = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_1GBHD = 0x4
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_1GB = 0x8
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_2GB = 0x10
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_2_5GB = 0x20
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_10GB = 0x40
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_20GB = 0x80
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_25GB = 0x100
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_40GB = 0x200
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_50GB = 0x400
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_100GB = 0x800
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_10MBHD = 0x1000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_AUTO_MODE_10MB = 0x2000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_RSVD1 = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_100MB = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_RSVD2 = 0x4
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_1GB = 0x8
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_RSVD3 = 0x10
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_RSVD4 = 0x20
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS_EEE_MODE_10GB = 0x40
PORT_PHY_QCAPS_RESP_TX_LPI_TIMER_LOW_MASK = 0xffffff
PORT_PHY_QCAPS_RESP_TX_LPI_TIMER_LOW_SFT = 0
PORT_PHY_QCAPS_RESP_RSVD2_MASK = 0xff000000
PORT_PHY_QCAPS_RESP_RSVD2_SFT = 24
PORT_PHY_QCAPS_RESP_TX_LPI_TIMER_HIGH_MASK = 0xffffff
PORT_PHY_QCAPS_RESP_TX_LPI_TIMER_HIGH_SFT = 0
PORT_PHY_QCAPS_RESP_RSVD_MASK = 0xff000000
PORT_PHY_QCAPS_RESP_RSVD_SFT = 24
PORT_PHY_QCAPS_RESP_SUPPORTED_PAM4_SPEEDS_AUTO_MODE_50G = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_PAM4_SPEEDS_AUTO_MODE_100G = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_PAM4_SPEEDS_AUTO_MODE_200G = 0x4
PORT_PHY_QCAPS_RESP_SUPPORTED_PAM4_SPEEDS_FORCE_MODE_50G = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_PAM4_SPEEDS_FORCE_MODE_100G = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_PAM4_SPEEDS_FORCE_MODE_200G = 0x4
PORT_PHY_QCAPS_RESP_FLAGS2_PAUSE_UNSUPPORTED = 0x1
PORT_PHY_QCAPS_RESP_FLAGS2_PFC_UNSUPPORTED = 0x2
PORT_PHY_QCAPS_RESP_FLAGS2_BANK_ADDR_SUPPORTED = 0x4
PORT_PHY_QCAPS_RESP_FLAGS2_SPEEDS2_SUPPORTED = 0x8
PORT_PHY_QCAPS_RESP_FLAGS2_REMOTE_LPBK_UNSUPPORTED = 0x10
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_1GB = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_10GB = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_25GB = 0x4
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_40GB = 0x8
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_50GB = 0x10
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_100GB = 0x20
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_50GB_PAM4_56 = 0x40
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_100GB_PAM4_56 = 0x80
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_200GB_PAM4_56 = 0x100
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_400GB_PAM4_56 = 0x200
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_100GB_PAM4_112 = 0x400
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_200GB_PAM4_112 = 0x800
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_400GB_PAM4_112 = 0x1000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_FORCE_MODE_800GB_PAM4_112 = 0x2000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_1GB = 0x1
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_10GB = 0x2
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_25GB = 0x4
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_40GB = 0x8
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_50GB = 0x10
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_100GB = 0x20
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_50GB_PAM4_56 = 0x40
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_100GB_PAM4_56 = 0x80
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_200GB_PAM4_56 = 0x100
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_400GB_PAM4_56 = 0x200
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_100GB_PAM4_112 = 0x400
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_200GB_PAM4_112 = 0x800
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_400GB_PAM4_112 = 0x1000
PORT_PHY_QCAPS_RESP_SUPPORTED_SPEEDS2_AUTO_MODE_800GB_PAM4_112 = 0x2000
PORT_PHY_I2C_WRITE_REQ_ENABLES_PAGE_OFFSET = 0x1
PORT_PHY_I2C_WRITE_REQ_ENABLES_BANK_NUMBER = 0x2
PORT_PHY_I2C_READ_REQ_ENABLES_PAGE_OFFSET = 0x1
PORT_PHY_I2C_READ_REQ_ENABLES_BANK_NUMBER = 0x2
PORT_LED_CFG_REQ_ENABLES_LED0_ID = 0x1
PORT_LED_CFG_REQ_ENABLES_LED0_STATE = 0x2
PORT_LED_CFG_REQ_ENABLES_LED0_COLOR = 0x4
PORT_LED_CFG_REQ_ENABLES_LED0_BLINK_ON = 0x8
PORT_LED_CFG_REQ_ENABLES_LED0_BLINK_OFF = 0x10
PORT_LED_CFG_REQ_ENABLES_LED0_GROUP_ID = 0x20
PORT_LED_CFG_REQ_ENABLES_LED1_ID = 0x40
PORT_LED_CFG_REQ_ENABLES_LED1_STATE = 0x80
PORT_LED_CFG_REQ_ENABLES_LED1_COLOR = 0x100
PORT_LED_CFG_REQ_ENABLES_LED1_BLINK_ON = 0x200
PORT_LED_CFG_REQ_ENABLES_LED1_BLINK_OFF = 0x400
PORT_LED_CFG_REQ_ENABLES_LED1_GROUP_ID = 0x800
PORT_LED_CFG_REQ_ENABLES_LED2_ID = 0x1000
PORT_LED_CFG_REQ_ENABLES_LED2_STATE = 0x2000
PORT_LED_CFG_REQ_ENABLES_LED2_COLOR = 0x4000
PORT_LED_CFG_REQ_ENABLES_LED2_BLINK_ON = 0x8000
PORT_LED_CFG_REQ_ENABLES_LED2_BLINK_OFF = 0x10000
PORT_LED_CFG_REQ_ENABLES_LED2_GROUP_ID = 0x20000
PORT_LED_CFG_REQ_ENABLES_LED3_ID = 0x40000
PORT_LED_CFG_REQ_ENABLES_LED3_STATE = 0x80000
PORT_LED_CFG_REQ_ENABLES_LED3_COLOR = 0x100000
PORT_LED_CFG_REQ_ENABLES_LED3_BLINK_ON = 0x200000
PORT_LED_CFG_REQ_ENABLES_LED3_BLINK_OFF = 0x400000
PORT_LED_CFG_REQ_ENABLES_LED3_GROUP_ID = 0x800000
PORT_LED_CFG_REQ_LED0_STATE_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED0_STATE_OFF = 0x1
PORT_LED_CFG_REQ_LED0_STATE_ON = 0x2
PORT_LED_CFG_REQ_LED0_STATE_BLINK = 0x3
PORT_LED_CFG_REQ_LED0_STATE_BLINKALT = 0x4
PORT_LED_CFG_REQ_LED0_STATE_LAST = PORT_LED_CFG_REQ_LED0_STATE_BLINKALT
PORT_LED_CFG_REQ_LED0_COLOR_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED0_COLOR_AMBER = 0x1
PORT_LED_CFG_REQ_LED0_COLOR_GREEN = 0x2
PORT_LED_CFG_REQ_LED0_COLOR_GREENAMBER = 0x3
PORT_LED_CFG_REQ_LED0_COLOR_LAST = PORT_LED_CFG_REQ_LED0_COLOR_GREENAMBER
PORT_LED_CFG_REQ_LED1_STATE_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED1_STATE_OFF = 0x1
PORT_LED_CFG_REQ_LED1_STATE_ON = 0x2
PORT_LED_CFG_REQ_LED1_STATE_BLINK = 0x3
PORT_LED_CFG_REQ_LED1_STATE_BLINKALT = 0x4
PORT_LED_CFG_REQ_LED1_STATE_LAST = PORT_LED_CFG_REQ_LED1_STATE_BLINKALT
PORT_LED_CFG_REQ_LED1_COLOR_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED1_COLOR_AMBER = 0x1
PORT_LED_CFG_REQ_LED1_COLOR_GREEN = 0x2
PORT_LED_CFG_REQ_LED1_COLOR_GREENAMBER = 0x3
PORT_LED_CFG_REQ_LED1_COLOR_LAST = PORT_LED_CFG_REQ_LED1_COLOR_GREENAMBER
PORT_LED_CFG_REQ_LED2_STATE_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED2_STATE_OFF = 0x1
PORT_LED_CFG_REQ_LED2_STATE_ON = 0x2
PORT_LED_CFG_REQ_LED2_STATE_BLINK = 0x3
PORT_LED_CFG_REQ_LED2_STATE_BLINKALT = 0x4
PORT_LED_CFG_REQ_LED2_STATE_LAST = PORT_LED_CFG_REQ_LED2_STATE_BLINKALT
PORT_LED_CFG_REQ_LED2_COLOR_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED2_COLOR_AMBER = 0x1
PORT_LED_CFG_REQ_LED2_COLOR_GREEN = 0x2
PORT_LED_CFG_REQ_LED2_COLOR_GREENAMBER = 0x3
PORT_LED_CFG_REQ_LED2_COLOR_LAST = PORT_LED_CFG_REQ_LED2_COLOR_GREENAMBER
PORT_LED_CFG_REQ_LED3_STATE_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED3_STATE_OFF = 0x1
PORT_LED_CFG_REQ_LED3_STATE_ON = 0x2
PORT_LED_CFG_REQ_LED3_STATE_BLINK = 0x3
PORT_LED_CFG_REQ_LED3_STATE_BLINKALT = 0x4
PORT_LED_CFG_REQ_LED3_STATE_LAST = PORT_LED_CFG_REQ_LED3_STATE_BLINKALT
PORT_LED_CFG_REQ_LED3_COLOR_DEFAULT = 0x0
PORT_LED_CFG_REQ_LED3_COLOR_AMBER = 0x1
PORT_LED_CFG_REQ_LED3_COLOR_GREEN = 0x2
PORT_LED_CFG_REQ_LED3_COLOR_GREENAMBER = 0x3
PORT_LED_CFG_REQ_LED3_COLOR_LAST = PORT_LED_CFG_REQ_LED3_COLOR_GREENAMBER
PORT_LED_QCFG_RESP_LED0_TYPE_SPEED = 0x0
PORT_LED_QCFG_RESP_LED0_TYPE_ACTIVITY = 0x1
PORT_LED_QCFG_RESP_LED0_TYPE_INVALID = 0xff
PORT_LED_QCFG_RESP_LED0_TYPE_LAST = PORT_LED_QCFG_RESP_LED0_TYPE_INVALID
PORT_LED_QCFG_RESP_LED0_STATE_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED0_STATE_OFF = 0x1
PORT_LED_QCFG_RESP_LED0_STATE_ON = 0x2
PORT_LED_QCFG_RESP_LED0_STATE_BLINK = 0x3
PORT_LED_QCFG_RESP_LED0_STATE_BLINKALT = 0x4
PORT_LED_QCFG_RESP_LED0_STATE_LAST = PORT_LED_QCFG_RESP_LED0_STATE_BLINKALT
PORT_LED_QCFG_RESP_LED0_COLOR_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED0_COLOR_AMBER = 0x1
PORT_LED_QCFG_RESP_LED0_COLOR_GREEN = 0x2
PORT_LED_QCFG_RESP_LED0_COLOR_GREENAMBER = 0x3
PORT_LED_QCFG_RESP_LED0_COLOR_LAST = PORT_LED_QCFG_RESP_LED0_COLOR_GREENAMBER
PORT_LED_QCFG_RESP_LED1_TYPE_SPEED = 0x0
PORT_LED_QCFG_RESP_LED1_TYPE_ACTIVITY = 0x1
PORT_LED_QCFG_RESP_LED1_TYPE_INVALID = 0xff
PORT_LED_QCFG_RESP_LED1_TYPE_LAST = PORT_LED_QCFG_RESP_LED1_TYPE_INVALID
PORT_LED_QCFG_RESP_LED1_STATE_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED1_STATE_OFF = 0x1
PORT_LED_QCFG_RESP_LED1_STATE_ON = 0x2
PORT_LED_QCFG_RESP_LED1_STATE_BLINK = 0x3
PORT_LED_QCFG_RESP_LED1_STATE_BLINKALT = 0x4
PORT_LED_QCFG_RESP_LED1_STATE_LAST = PORT_LED_QCFG_RESP_LED1_STATE_BLINKALT
PORT_LED_QCFG_RESP_LED1_COLOR_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED1_COLOR_AMBER = 0x1
PORT_LED_QCFG_RESP_LED1_COLOR_GREEN = 0x2
PORT_LED_QCFG_RESP_LED1_COLOR_GREENAMBER = 0x3
PORT_LED_QCFG_RESP_LED1_COLOR_LAST = PORT_LED_QCFG_RESP_LED1_COLOR_GREENAMBER
PORT_LED_QCFG_RESP_LED2_TYPE_SPEED = 0x0
PORT_LED_QCFG_RESP_LED2_TYPE_ACTIVITY = 0x1
PORT_LED_QCFG_RESP_LED2_TYPE_INVALID = 0xff
PORT_LED_QCFG_RESP_LED2_TYPE_LAST = PORT_LED_QCFG_RESP_LED2_TYPE_INVALID
PORT_LED_QCFG_RESP_LED2_STATE_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED2_STATE_OFF = 0x1
PORT_LED_QCFG_RESP_LED2_STATE_ON = 0x2
PORT_LED_QCFG_RESP_LED2_STATE_BLINK = 0x3
PORT_LED_QCFG_RESP_LED2_STATE_BLINKALT = 0x4
PORT_LED_QCFG_RESP_LED2_STATE_LAST = PORT_LED_QCFG_RESP_LED2_STATE_BLINKALT
PORT_LED_QCFG_RESP_LED2_COLOR_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED2_COLOR_AMBER = 0x1
PORT_LED_QCFG_RESP_LED2_COLOR_GREEN = 0x2
PORT_LED_QCFG_RESP_LED2_COLOR_GREENAMBER = 0x3
PORT_LED_QCFG_RESP_LED2_COLOR_LAST = PORT_LED_QCFG_RESP_LED2_COLOR_GREENAMBER
PORT_LED_QCFG_RESP_LED3_TYPE_SPEED = 0x0
PORT_LED_QCFG_RESP_LED3_TYPE_ACTIVITY = 0x1
PORT_LED_QCFG_RESP_LED3_TYPE_INVALID = 0xff
PORT_LED_QCFG_RESP_LED3_TYPE_LAST = PORT_LED_QCFG_RESP_LED3_TYPE_INVALID
PORT_LED_QCFG_RESP_LED3_STATE_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED3_STATE_OFF = 0x1
PORT_LED_QCFG_RESP_LED3_STATE_ON = 0x2
PORT_LED_QCFG_RESP_LED3_STATE_BLINK = 0x3
PORT_LED_QCFG_RESP_LED3_STATE_BLINKALT = 0x4
PORT_LED_QCFG_RESP_LED3_STATE_LAST = PORT_LED_QCFG_RESP_LED3_STATE_BLINKALT
PORT_LED_QCFG_RESP_LED3_COLOR_DEFAULT = 0x0
PORT_LED_QCFG_RESP_LED3_COLOR_AMBER = 0x1
PORT_LED_QCFG_RESP_LED3_COLOR_GREEN = 0x2
PORT_LED_QCFG_RESP_LED3_COLOR_GREENAMBER = 0x3
PORT_LED_QCFG_RESP_LED3_COLOR_LAST = PORT_LED_QCFG_RESP_LED3_COLOR_GREENAMBER
PORT_LED_QCAPS_RESP_LED0_TYPE_SPEED = 0x0
PORT_LED_QCAPS_RESP_LED0_TYPE_ACTIVITY = 0x1
PORT_LED_QCAPS_RESP_LED0_TYPE_INVALID = 0xff
PORT_LED_QCAPS_RESP_LED0_TYPE_LAST = PORT_LED_QCAPS_RESP_LED0_TYPE_INVALID
PORT_LED_QCAPS_RESP_LED0_STATE_CAPS_ENABLED = 0x1
PORT_LED_QCAPS_RESP_LED0_STATE_CAPS_OFF_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED0_STATE_CAPS_ON_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED0_STATE_CAPS_BLINK_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED0_STATE_CAPS_BLINK_ALT_SUPPORTED = 0x10
PORT_LED_QCAPS_RESP_LED0_COLOR_CAPS_RSVD = 0x1
PORT_LED_QCAPS_RESP_LED0_COLOR_CAPS_AMBER_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED0_COLOR_CAPS_GREEN_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED0_COLOR_CAPS_GRNAMB_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED1_TYPE_SPEED = 0x0
PORT_LED_QCAPS_RESP_LED1_TYPE_ACTIVITY = 0x1
PORT_LED_QCAPS_RESP_LED1_TYPE_INVALID = 0xff
PORT_LED_QCAPS_RESP_LED1_TYPE_LAST = PORT_LED_QCAPS_RESP_LED1_TYPE_INVALID
PORT_LED_QCAPS_RESP_LED1_STATE_CAPS_ENABLED = 0x1
PORT_LED_QCAPS_RESP_LED1_STATE_CAPS_OFF_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED1_STATE_CAPS_ON_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED1_STATE_CAPS_BLINK_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED1_STATE_CAPS_BLINK_ALT_SUPPORTED = 0x10
PORT_LED_QCAPS_RESP_LED1_COLOR_CAPS_RSVD = 0x1
PORT_LED_QCAPS_RESP_LED1_COLOR_CAPS_AMBER_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED1_COLOR_CAPS_GREEN_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED1_COLOR_CAPS_GRNAMB_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED2_TYPE_SPEED = 0x0
PORT_LED_QCAPS_RESP_LED2_TYPE_ACTIVITY = 0x1
PORT_LED_QCAPS_RESP_LED2_TYPE_INVALID = 0xff
PORT_LED_QCAPS_RESP_LED2_TYPE_LAST = PORT_LED_QCAPS_RESP_LED2_TYPE_INVALID
PORT_LED_QCAPS_RESP_LED2_STATE_CAPS_ENABLED = 0x1
PORT_LED_QCAPS_RESP_LED2_STATE_CAPS_OFF_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED2_STATE_CAPS_ON_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED2_STATE_CAPS_BLINK_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED2_STATE_CAPS_BLINK_ALT_SUPPORTED = 0x10
PORT_LED_QCAPS_RESP_LED2_COLOR_CAPS_RSVD = 0x1
PORT_LED_QCAPS_RESP_LED2_COLOR_CAPS_AMBER_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED2_COLOR_CAPS_GREEN_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED2_COLOR_CAPS_GRNAMB_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED3_TYPE_SPEED = 0x0
PORT_LED_QCAPS_RESP_LED3_TYPE_ACTIVITY = 0x1
PORT_LED_QCAPS_RESP_LED3_TYPE_INVALID = 0xff
PORT_LED_QCAPS_RESP_LED3_TYPE_LAST = PORT_LED_QCAPS_RESP_LED3_TYPE_INVALID
PORT_LED_QCAPS_RESP_LED3_STATE_CAPS_ENABLED = 0x1
PORT_LED_QCAPS_RESP_LED3_STATE_CAPS_OFF_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED3_STATE_CAPS_ON_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED3_STATE_CAPS_BLINK_SUPPORTED = 0x8
PORT_LED_QCAPS_RESP_LED3_STATE_CAPS_BLINK_ALT_SUPPORTED = 0x10
PORT_LED_QCAPS_RESP_LED3_COLOR_CAPS_RSVD = 0x1
PORT_LED_QCAPS_RESP_LED3_COLOR_CAPS_AMBER_SUPPORTED = 0x2
PORT_LED_QCAPS_RESP_LED3_COLOR_CAPS_GREEN_SUPPORTED = 0x4
PORT_LED_QCAPS_RESP_LED3_COLOR_CAPS_GRNAMB_SUPPORTED = 0x8
PORT_MAC_QCAPS_RESP_FLAGS_LOCAL_LPBK_NOT_SUPPORTED = 0x1
PORT_MAC_QCAPS_RESP_FLAGS_REMOTE_LPBK_SUPPORTED = 0x2
QUEUE_QPORTCFG_REQ_FLAGS_PATH = 0x1
QUEUE_QPORTCFG_REQ_FLAGS_PATH_TX = 0x0
QUEUE_QPORTCFG_REQ_FLAGS_PATH_RX = 0x1
QUEUE_QPORTCFG_REQ_FLAGS_PATH_LAST = QUEUE_QPORTCFG_REQ_FLAGS_PATH_RX
QUEUE_QPORTCFG_REQ_DRV_QMAP_CAP_DISABLED = 0x0
QUEUE_QPORTCFG_REQ_DRV_QMAP_CAP_ENABLED = 0x1
QUEUE_QPORTCFG_REQ_DRV_QMAP_CAP_LAST = QUEUE_QPORTCFG_REQ_DRV_QMAP_CAP_ENABLED
QUEUE_QPORTCFG_RESP_QUEUE_CFG_INFO_ASYM_CFG = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_CFG_INFO_USE_PROFILE_TYPE = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_LOSSLESS_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_LOSSY_ROCE_CNP = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_LOSSLESS_NIC = 0x3
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_LAST = QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_UNKNOWN
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID0_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID1_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID2_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID3_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID4_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID5_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID6_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_TYPE_ROCE = 0x1
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_TYPE_NIC = 0x2
QUEUE_QPORTCFG_RESP_QUEUE_ID7_SERVICE_PROFILE_TYPE_CNP = 0x4
QUEUE_QCFG_REQ_FLAGS_PATH = 0x1
QUEUE_QCFG_REQ_FLAGS_PATH_TX = 0x0
QUEUE_QCFG_REQ_FLAGS_PATH_RX = 0x1
QUEUE_QCFG_REQ_FLAGS_PATH_LAST = QUEUE_QCFG_REQ_FLAGS_PATH_RX
QUEUE_QCFG_RESP_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_QCFG_RESP_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_QCFG_RESP_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_QCFG_RESP_SERVICE_PROFILE_LAST = QUEUE_QCFG_RESP_SERVICE_PROFILE_UNKNOWN
QUEUE_QCFG_RESP_QUEUE_CFG_INFO_ASYM_CFG = 0x1
QUEUE_CFG_REQ_FLAGS_PATH_MASK = 0x3
QUEUE_CFG_REQ_FLAGS_PATH_SFT = 0
QUEUE_CFG_REQ_FLAGS_PATH_TX = 0x0
QUEUE_CFG_REQ_FLAGS_PATH_RX = 0x1
QUEUE_CFG_REQ_FLAGS_PATH_BIDIR = 0x2
QUEUE_CFG_REQ_FLAGS_PATH_LAST = QUEUE_CFG_REQ_FLAGS_PATH_BIDIR
QUEUE_CFG_REQ_ENABLES_DFLT_LEN = 0x1
QUEUE_CFG_REQ_ENABLES_SERVICE_PROFILE = 0x2
QUEUE_CFG_REQ_SERVICE_PROFILE_LOSSY = 0x0
QUEUE_CFG_REQ_SERVICE_PROFILE_LOSSLESS = 0x1
QUEUE_CFG_REQ_SERVICE_PROFILE_UNKNOWN = 0xff
QUEUE_CFG_REQ_SERVICE_PROFILE_LAST = QUEUE_CFG_REQ_SERVICE_PROFILE_UNKNOWN
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI0_PFC_ENABLED = 0x1
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI1_PFC_ENABLED = 0x2
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI2_PFC_ENABLED = 0x4
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI3_PFC_ENABLED = 0x8
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI4_PFC_ENABLED = 0x10
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI5_PFC_ENABLED = 0x20
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI6_PFC_ENABLED = 0x40
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI7_PFC_ENABLED = 0x80
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI0_PFC_WATCHDOG_ENABLED = 0x100
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI1_PFC_WATCHDOG_ENABLED = 0x200
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI2_PFC_WATCHDOG_ENABLED = 0x400
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI3_PFC_WATCHDOG_ENABLED = 0x800
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI4_PFC_WATCHDOG_ENABLED = 0x1000
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI5_PFC_WATCHDOG_ENABLED = 0x2000
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI6_PFC_WATCHDOG_ENABLED = 0x4000
QUEUE_PFCENABLE_QCFG_RESP_FLAGS_PRI7_PFC_WATCHDOG_ENABLED = 0x8000
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI0_PFC_ENABLED = 0x1
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI1_PFC_ENABLED = 0x2
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI2_PFC_ENABLED = 0x4
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI3_PFC_ENABLED = 0x8
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI4_PFC_ENABLED = 0x10
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI5_PFC_ENABLED = 0x20
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI6_PFC_ENABLED = 0x40
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI7_PFC_ENABLED = 0x80
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI0_PFC_WATCHDOG_ENABLED = 0x100
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI1_PFC_WATCHDOG_ENABLED = 0x200
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI2_PFC_WATCHDOG_ENABLED = 0x400
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI3_PFC_WATCHDOG_ENABLED = 0x800
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI4_PFC_WATCHDOG_ENABLED = 0x1000
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI5_PFC_WATCHDOG_ENABLED = 0x2000
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI6_PFC_WATCHDOG_ENABLED = 0x4000
QUEUE_PFCENABLE_CFG_REQ_FLAGS_PRI7_PFC_WATCHDOG_ENABLED = 0x8000
QUEUE_PRI2COS_QCFG_REQ_FLAGS_PATH = 0x1
QUEUE_PRI2COS_QCFG_REQ_FLAGS_PATH_TX = 0x0
QUEUE_PRI2COS_QCFG_REQ_FLAGS_PATH_RX = 0x1
QUEUE_PRI2COS_QCFG_REQ_FLAGS_PATH_LAST = QUEUE_PRI2COS_QCFG_REQ_FLAGS_PATH_RX
QUEUE_PRI2COS_QCFG_REQ_FLAGS_IVLAN = 0x2
QUEUE_PRI2COS_QCFG_RESP_QUEUE_CFG_INFO_ASYM_CFG = 0x1
QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_MASK = 0x3
QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_SFT = 0
QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_TX = 0x0
QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_RX = 0x1
QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_BIDIR = 0x2
QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_LAST = QUEUE_PRI2COS_CFG_REQ_FLAGS_PATH_BIDIR
QUEUE_PRI2COS_CFG_REQ_FLAGS_IVLAN = 0x4
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI0_COS_QUEUE_ID = 0x1
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI1_COS_QUEUE_ID = 0x2
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI2_COS_QUEUE_ID = 0x4
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI3_COS_QUEUE_ID = 0x8
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI4_COS_QUEUE_ID = 0x10
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI5_COS_QUEUE_ID = 0x20
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI6_COS_QUEUE_ID = 0x40
QUEUE_PRI2COS_CFG_REQ_ENABLES_PRI7_COS_QUEUE_ID = 0x80
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_SCALE = 0x10000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_SCALE_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_SCALE_BYTES
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_SCALE = 0x10000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_SCALE_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_SCALE_BYTES
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_TSA_ASSIGN_SP = 0x0
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_TSA_ASSIGN_ETS = 0x1
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_TSA_ASSIGN_RESERVED_FIRST = 0x2
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID0_TSA_ASSIGN_RESERVED_LAST = 0xff
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_SCALE = 0x10000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_SCALE_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_SCALE_BYTES
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_SCALE = 0x10000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_SCALE_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_SCALE_BYTES
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_TSA_ASSIGN_SP = 0x0
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_TSA_ASSIGN_ETS = 0x1
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_TSA_ASSIGN_RESERVED_FIRST = 0x2
QUEUE_COS2BW_QCFG_RESP_QUEUE_ID_TSA_ASSIGN_RESERVED_LAST = 0xff
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID0_VALID = 0x1
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID1_VALID = 0x2
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID2_VALID = 0x4
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID3_VALID = 0x8
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID4_VALID = 0x10
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID5_VALID = 0x20
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID6_VALID = 0x40
QUEUE_COS2BW_CFG_REQ_ENABLES_COS_QUEUE_ID7_VALID = 0x80
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_SCALE = 0x10000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_SCALE_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_SCALE_BYTES
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MIN_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_SCALE = 0x10000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_SCALE_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_SCALE_BYTES
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_MAX_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_TSA_ASSIGN_SP = 0x0
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_TSA_ASSIGN_ETS = 0x1
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_TSA_ASSIGN_RESERVED_FIRST = 0x2
QUEUE_COS2BW_CFG_REQ_QUEUE_ID0_TSA_ASSIGN_RESERVED_LAST = 0xff
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_SCALE = 0x10000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_SCALE_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_SCALE_BYTES
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MIN_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_MASK = 0xfffffff
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_SFT = 0
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_SCALE = 0x10000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_SCALE_BITS = (0x0 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_SCALE_BYTES = (0x1 << 28)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_SCALE_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_SCALE_BYTES
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_SFT = 29
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_LAST = QUEUE_COS2BW_CFG_REQ_QUEUE_ID_MAX_BW_BW_VALUE_UNIT_INVALID
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_TSA_ASSIGN_SP = 0x0
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_TSA_ASSIGN_ETS = 0x1
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_TSA_ASSIGN_RESERVED_FIRST = 0x2
QUEUE_COS2BW_CFG_REQ_QUEUE_ID_TSA_ASSIGN_RESERVED_LAST = 0xff
QUEUE_DSCP2PRI_CFG_REQ_FLAGS_USE_HW_DEFAULT_PRI = 0x1
QUEUE_DSCP2PRI_CFG_REQ_ENABLES_DEFAULT_PRI = 0x1
VNIC_ALLOC_REQ_FLAGS_DEFAULT = 0x1
VNIC_ALLOC_REQ_FLAGS_VIRTIO_NET_FID_VALID = 0x2
VNIC_ALLOC_REQ_FLAGS_VNIC_ID_VALID = 0x4
VNIC_UPDATE_REQ_ENABLES_VNIC_STATE_VALID = 0x1
VNIC_UPDATE_REQ_ENABLES_MRU_VALID = 0x2
VNIC_UPDATE_REQ_ENABLES_METADATA_FORMAT_TYPE_VALID = 0x4
VNIC_UPDATE_REQ_VNIC_STATE_NORMAL = 0x0
VNIC_UPDATE_REQ_VNIC_STATE_DROP = 0x1
VNIC_UPDATE_REQ_VNIC_STATE_LAST = VNIC_UPDATE_REQ_VNIC_STATE_DROP
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_0 = 0x0
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_1 = 0x1
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_2 = 0x2
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_3 = 0x3
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_4 = 0x4
VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_LAST = VNIC_UPDATE_REQ_METADATA_FORMAT_TYPE_4
VNIC_CFG_REQ_FLAGS_DEFAULT = 0x1
VNIC_CFG_REQ_FLAGS_VLAN_STRIP_MODE = 0x2
VNIC_CFG_REQ_FLAGS_BD_STALL_MODE = 0x4
VNIC_CFG_REQ_FLAGS_ROCE_DUAL_VNIC_MODE = 0x8
VNIC_CFG_REQ_FLAGS_ROCE_ONLY_VNIC_MODE = 0x10
VNIC_CFG_REQ_FLAGS_RSS_DFLT_CR_MODE = 0x20
VNIC_CFG_REQ_FLAGS_ROCE_MIRRORING_CAPABLE_VNIC_MODE = 0x40
VNIC_CFG_REQ_FLAGS_PORTCOS_MAPPING_MODE = 0x80
VNIC_CFG_REQ_ENABLES_DFLT_RING_GRP = 0x1
VNIC_CFG_REQ_ENABLES_RSS_RULE = 0x2
VNIC_CFG_REQ_ENABLES_COS_RULE = 0x4
VNIC_CFG_REQ_ENABLES_LB_RULE = 0x8
VNIC_CFG_REQ_ENABLES_MRU = 0x10
VNIC_CFG_REQ_ENABLES_DEFAULT_RX_RING_ID = 0x20
VNIC_CFG_REQ_ENABLES_DEFAULT_CMPL_RING_ID = 0x40
VNIC_CFG_REQ_ENABLES_QUEUE_ID = 0x80
VNIC_CFG_REQ_ENABLES_RX_CSUM_V2_MODE = 0x100
VNIC_CFG_REQ_ENABLES_L2_CQE_MODE = 0x200
VNIC_CFG_REQ_ENABLES_RAW_QP_ID = 0x400
VNIC_CFG_REQ_RX_CSUM_V2_MODE_DEFAULT = 0x0
VNIC_CFG_REQ_RX_CSUM_V2_MODE_ALL_OK = 0x1
VNIC_CFG_REQ_RX_CSUM_V2_MODE_MAX = 0x2
VNIC_CFG_REQ_RX_CSUM_V2_MODE_LAST = VNIC_CFG_REQ_RX_CSUM_V2_MODE_MAX
VNIC_CFG_REQ_L2_CQE_MODE_DEFAULT = 0x0
VNIC_CFG_REQ_L2_CQE_MODE_COMPRESSED = 0x1
VNIC_CFG_REQ_L2_CQE_MODE_MIXED = 0x2
VNIC_CFG_REQ_L2_CQE_MODE_LAST = VNIC_CFG_REQ_L2_CQE_MODE_MIXED
VNIC_QCAPS_RESP_FLAGS_UNUSED = 0x1
VNIC_QCAPS_RESP_FLAGS_VLAN_STRIP_CAP = 0x2
VNIC_QCAPS_RESP_FLAGS_BD_STALL_CAP = 0x4
VNIC_QCAPS_RESP_FLAGS_ROCE_DUAL_VNIC_CAP = 0x8
VNIC_QCAPS_RESP_FLAGS_ROCE_ONLY_VNIC_CAP = 0x10
VNIC_QCAPS_RESP_FLAGS_RSS_DFLT_CR_CAP = 0x20
VNIC_QCAPS_RESP_FLAGS_ROCE_MIRRORING_CAPABLE_VNIC_CAP = 0x40
VNIC_QCAPS_RESP_FLAGS_OUTERMOST_RSS_CAP = 0x80
VNIC_QCAPS_RESP_FLAGS_COS_ASSIGNMENT_CAP = 0x100
VNIC_QCAPS_RESP_FLAGS_RX_CMPL_V2_CAP = 0x200
VNIC_QCAPS_RESP_FLAGS_VNIC_STATE_CAP = 0x400
VNIC_QCAPS_RESP_FLAGS_VIRTIO_NET_VNIC_ALLOC_CAP = 0x800
VNIC_QCAPS_RESP_FLAGS_METADATA_FORMAT_CAP = 0x1000
VNIC_QCAPS_RESP_FLAGS_RSS_STRICT_HASH_TYPE_CAP = 0x2000
VNIC_QCAPS_RESP_FLAGS_RSS_HASH_TYPE_DELTA_CAP = 0x4000
VNIC_QCAPS_RESP_FLAGS_RING_SELECT_MODE_TOEPLITZ_CAP = 0x8000
VNIC_QCAPS_RESP_FLAGS_RING_SELECT_MODE_XOR_CAP = 0x10000
VNIC_QCAPS_RESP_FLAGS_RING_SELECT_MODE_TOEPLITZ_CHKSM_CAP = 0x20000
VNIC_QCAPS_RESP_FLAGS_RSS_IPV6_FLOW_LABEL_CAP = 0x40000
VNIC_QCAPS_RESP_FLAGS_RX_CMPL_V3_CAP = 0x80000
VNIC_QCAPS_RESP_FLAGS_L2_CQE_MODE_CAP = 0x100000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_AH_SPI_IPV4_CAP = 0x200000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_ESP_SPI_IPV4_CAP = 0x400000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_AH_SPI_IPV6_CAP = 0x800000
VNIC_QCAPS_RESP_FLAGS_RSS_IPSEC_ESP_SPI_IPV6_CAP = 0x1000000
VNIC_QCAPS_RESP_FLAGS_OUTERMOST_RSS_TRUSTED_VF_CAP = 0x2000000
VNIC_QCAPS_RESP_FLAGS_PORTCOS_MAPPING_MODE = 0x4000000
VNIC_QCAPS_RESP_FLAGS_RSS_PROF_TCAM_MODE_ENABLED = 0x8000000
VNIC_QCAPS_RESP_FLAGS_VNIC_RSS_HASH_MODE_CAP = 0x10000000
VNIC_QCAPS_RESP_FLAGS_HW_TUNNEL_TPA_CAP = 0x20000000
VNIC_QCAPS_RESP_FLAGS_RE_FLUSH_CAP = 0x40000000
VNIC_TPA_CFG_REQ_FLAGS_TPA = 0x1
VNIC_TPA_CFG_REQ_FLAGS_ENCAP_TPA = 0x2
VNIC_TPA_CFG_REQ_FLAGS_RSC_WND_UPDATE = 0x4
VNIC_TPA_CFG_REQ_FLAGS_GRO = 0x8
VNIC_TPA_CFG_REQ_FLAGS_AGG_WITH_ECN = 0x10
VNIC_TPA_CFG_REQ_FLAGS_AGG_WITH_SAME_GRE_SEQ = 0x20
VNIC_TPA_CFG_REQ_FLAGS_GRO_IPID_CHECK = 0x40
VNIC_TPA_CFG_REQ_FLAGS_GRO_TTL_CHECK = 0x80
VNIC_TPA_CFG_REQ_FLAGS_AGG_PACK_AS_GRO = 0x100
VNIC_TPA_CFG_REQ_ENABLES_MAX_AGG_SEGS = 0x1
VNIC_TPA_CFG_REQ_ENABLES_MAX_AGGS = 0x2
VNIC_TPA_CFG_REQ_ENABLES_MAX_AGG_TIMER = 0x4
VNIC_TPA_CFG_REQ_ENABLES_MIN_AGG_LEN = 0x8
VNIC_TPA_CFG_REQ_ENABLES_TNL_TPA_EN = 0x10
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_1 = 0x0
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_2 = 0x1
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_4 = 0x2
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_8 = 0x3
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_MAX = 0x1f
VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_LAST = VNIC_TPA_CFG_REQ_MAX_AGG_SEGS_MAX
VNIC_TPA_CFG_REQ_MAX_AGGS_1 = 0x0
VNIC_TPA_CFG_REQ_MAX_AGGS_2 = 0x1
VNIC_TPA_CFG_REQ_MAX_AGGS_4 = 0x2
VNIC_TPA_CFG_REQ_MAX_AGGS_8 = 0x3
VNIC_TPA_CFG_REQ_MAX_AGGS_16 = 0x4
VNIC_TPA_CFG_REQ_MAX_AGGS_MAX = 0x7
VNIC_TPA_CFG_REQ_MAX_AGGS_LAST = VNIC_TPA_CFG_REQ_MAX_AGGS_MAX
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_VXLAN = 0x1
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_GENEVE = 0x2
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_NVGRE = 0x4
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_GRE = 0x8
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_IPV4 = 0x10
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_IPV6 = 0x20
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_VXLAN_GPE = 0x40
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_VXLAN_CUST1 = 0x80
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_GRE_CUST1 = 0x100
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR1 = 0x200
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR2 = 0x400
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR3 = 0x800
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR4 = 0x1000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR5 = 0x2000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR6 = 0x4000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR7 = 0x8000
VNIC_TPA_CFG_REQ_TNL_TPA_EN_BITMAP_UPAR8 = 0x10000
VNIC_TPA_QCFG_RESP_FLAGS_TPA = 0x1
VNIC_TPA_QCFG_RESP_FLAGS_ENCAP_TPA = 0x2
VNIC_TPA_QCFG_RESP_FLAGS_RSC_WND_UPDATE = 0x4
VNIC_TPA_QCFG_RESP_FLAGS_GRO = 0x8
VNIC_TPA_QCFG_RESP_FLAGS_AGG_WITH_ECN = 0x10
VNIC_TPA_QCFG_RESP_FLAGS_AGG_WITH_SAME_GRE_SEQ = 0x20
VNIC_TPA_QCFG_RESP_FLAGS_GRO_IPID_CHECK = 0x40
VNIC_TPA_QCFG_RESP_FLAGS_GRO_TTL_CHECK = 0x80
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_1 = 0x0
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_2 = 0x1
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_4 = 0x2
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_8 = 0x3
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_MAX = 0x1f
VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_LAST = VNIC_TPA_QCFG_RESP_MAX_AGG_SEGS_MAX
VNIC_TPA_QCFG_RESP_MAX_AGGS_1 = 0x0
VNIC_TPA_QCFG_RESP_MAX_AGGS_2 = 0x1
VNIC_TPA_QCFG_RESP_MAX_AGGS_4 = 0x2
VNIC_TPA_QCFG_RESP_MAX_AGGS_8 = 0x3
VNIC_TPA_QCFG_RESP_MAX_AGGS_16 = 0x4
VNIC_TPA_QCFG_RESP_MAX_AGGS_MAX = 0x7
VNIC_TPA_QCFG_RESP_MAX_AGGS_LAST = VNIC_TPA_QCFG_RESP_MAX_AGGS_MAX
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_VXLAN = 0x1
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_GENEVE = 0x2
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_NVGRE = 0x4
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_GRE = 0x8
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_IPV4 = 0x10
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_IPV6 = 0x20
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_VXLAN_GPE = 0x40
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_VXLAN_CUST1 = 0x80
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_GRE_CUST1 = 0x100
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR1 = 0x200
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR2 = 0x400
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR3 = 0x800
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR4 = 0x1000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR5 = 0x2000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR6 = 0x4000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR7 = 0x8000
VNIC_TPA_QCFG_RESP_TNL_TPA_EN_BITMAP_UPAR8 = 0x10000
VNIC_RSS_CFG_REQ_HASH_TYPE_IPV4 = 0x1
VNIC_RSS_CFG_REQ_HASH_TYPE_TCP_IPV4 = 0x2
VNIC_RSS_CFG_REQ_HASH_TYPE_UDP_IPV4 = 0x4
VNIC_RSS_CFG_REQ_HASH_TYPE_IPV6 = 0x8
VNIC_RSS_CFG_REQ_HASH_TYPE_TCP_IPV6 = 0x10
VNIC_RSS_CFG_REQ_HASH_TYPE_UDP_IPV6 = 0x20
VNIC_RSS_CFG_REQ_HASH_TYPE_IPV6_FLOW_LABEL = 0x40
VNIC_RSS_CFG_REQ_HASH_TYPE_AH_SPI_IPV4 = 0x80
VNIC_RSS_CFG_REQ_HASH_TYPE_ESP_SPI_IPV4 = 0x100
VNIC_RSS_CFG_REQ_HASH_TYPE_AH_SPI_IPV6 = 0x200
VNIC_RSS_CFG_REQ_HASH_TYPE_ESP_SPI_IPV6 = 0x400
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_DEFAULT = 0x1
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_INNERMOST_4 = 0x2
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_INNERMOST_2 = 0x4
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_OUTERMOST_4 = 0x8
VNIC_RSS_CFG_REQ_HASH_MODE_FLAGS_OUTERMOST_2 = 0x10
VNIC_RSS_CFG_REQ_FLAGS_HASH_TYPE_INCLUDE = 0x1
VNIC_RSS_CFG_REQ_FLAGS_HASH_TYPE_EXCLUDE = 0x2
VNIC_RSS_CFG_REQ_FLAGS_IPSEC_HASH_TYPE_CFG_SUPPORT = 0x4
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_TOEPLITZ = 0x0
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_XOR = 0x1
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_TOEPLITZ_CHECKSUM = 0x2
VNIC_RSS_CFG_REQ_RING_SELECT_MODE_LAST = VNIC_RSS_CFG_REQ_RING_SELECT_MODE_TOEPLITZ_CHECKSUM
VNIC_RSS_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
VNIC_RSS_CFG_CMD_ERR_CODE_INTERFACE_NOT_READY = 0x1
VNIC_RSS_CFG_CMD_ERR_CODE_UNABLE_TO_GET_RSS_CFG = 0x2
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_TYPE_UNSUPPORTED = 0x3
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_TYPE_ERR = 0x4
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_MODE_FAIL = 0x5
VNIC_RSS_CFG_CMD_ERR_CODE_RING_GRP_TABLE_ALLOC_ERR = 0x6
VNIC_RSS_CFG_CMD_ERR_CODE_HASH_KEY_ALLOC_ERR = 0x7
VNIC_RSS_CFG_CMD_ERR_CODE_DMA_FAILED = 0x8
VNIC_RSS_CFG_CMD_ERR_CODE_RX_RING_ALLOC_ERR = 0x9
VNIC_RSS_CFG_CMD_ERR_CODE_CMPL_RING_ALLOC_ERR = 0xa
VNIC_RSS_CFG_CMD_ERR_CODE_HW_SET_RSS_FAILED = 0xb
VNIC_RSS_CFG_CMD_ERR_CODE_CTX_INVALID = 0xc
VNIC_RSS_CFG_CMD_ERR_CODE_VNIC_INVALID = 0xd
VNIC_RSS_CFG_CMD_ERR_CODE_VNIC_RING_TABLE_PAIR_INVALID = 0xe
VNIC_RSS_CFG_CMD_ERR_CODE_LAST = VNIC_RSS_CFG_CMD_ERR_CODE_VNIC_RING_TABLE_PAIR_INVALID
VNIC_RSS_QCFG_RESP_HASH_TYPE_IPV4 = 0x1
VNIC_RSS_QCFG_RESP_HASH_TYPE_TCP_IPV4 = 0x2
VNIC_RSS_QCFG_RESP_HASH_TYPE_UDP_IPV4 = 0x4
VNIC_RSS_QCFG_RESP_HASH_TYPE_IPV6 = 0x8
VNIC_RSS_QCFG_RESP_HASH_TYPE_TCP_IPV6 = 0x10
VNIC_RSS_QCFG_RESP_HASH_TYPE_UDP_IPV6 = 0x20
VNIC_RSS_QCFG_RESP_HASH_TYPE_IPV6_FLOW_LABEL = 0x40
VNIC_RSS_QCFG_RESP_HASH_TYPE_AH_SPI_IPV4 = 0x80
VNIC_RSS_QCFG_RESP_HASH_TYPE_ESP_SPI_IPV4 = 0x100
VNIC_RSS_QCFG_RESP_HASH_TYPE_AH_SPI_IPV6 = 0x200
VNIC_RSS_QCFG_RESP_HASH_TYPE_ESP_SPI_IPV6 = 0x400
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_DEFAULT = 0x1
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_INNERMOST_4 = 0x2
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_INNERMOST_2 = 0x4
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_OUTERMOST_4 = 0x8
VNIC_RSS_QCFG_RESP_HASH_MODE_FLAGS_OUTERMOST_2 = 0x10
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_TOEPLITZ = 0x0
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_XOR = 0x1
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_TOEPLITZ_CHECKSUM = 0x2
VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_LAST = VNIC_RSS_QCFG_RESP_RING_SELECT_MODE_TOEPLITZ_CHECKSUM
VNIC_PLCMODES_CFG_REQ_FLAGS_REGULAR_PLACEMENT = 0x1
VNIC_PLCMODES_CFG_REQ_FLAGS_JUMBO_PLACEMENT = 0x2
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_IPV4 = 0x4
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_IPV6 = 0x8
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_FCOE = 0x10
VNIC_PLCMODES_CFG_REQ_FLAGS_HDS_ROCE = 0x20
VNIC_PLCMODES_CFG_REQ_FLAGS_VIRTIO_PLACEMENT = 0x40
VNIC_PLCMODES_CFG_REQ_ENABLES_JUMBO_THRESH_VALID = 0x1
VNIC_PLCMODES_CFG_REQ_ENABLES_HDS_OFFSET_VALID = 0x2
VNIC_PLCMODES_CFG_REQ_ENABLES_HDS_THRESHOLD_VALID = 0x4
VNIC_PLCMODES_CFG_REQ_ENABLES_MAX_BDS_VALID = 0x8
VNIC_PLCMODES_CFG_CMD_ERR_CODE_UNKNOWN = 0x0
VNIC_PLCMODES_CFG_CMD_ERR_CODE_INVALID_HDS_THRESHOLD = 0x1
VNIC_PLCMODES_CFG_CMD_ERR_CODE_LAST = VNIC_PLCMODES_CFG_CMD_ERR_CODE_INVALID_HDS_THRESHOLD
RING_ALLOC_REQ_ENABLES_RING_ARB_CFG = 0x2
RING_ALLOC_REQ_ENABLES_STAT_CTX_ID_VALID = 0x8
RING_ALLOC_REQ_ENABLES_MAX_BW_VALID = 0x20
RING_ALLOC_REQ_ENABLES_RX_RING_ID_VALID = 0x40
RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID = 0x80
RING_ALLOC_REQ_ENABLES_RX_BUF_SIZE_VALID = 0x100
RING_ALLOC_REQ_ENABLES_SCHQ_ID = 0x200
RING_ALLOC_REQ_ENABLES_MPC_CHNLS_TYPE = 0x400
RING_ALLOC_REQ_ENABLES_STEERING_TAG_VALID = 0x800
RING_ALLOC_REQ_ENABLES_RX_RATE_PROFILE_VALID = 0x1000
RING_ALLOC_REQ_ENABLES_DPI_VALID = 0x2000
RING_ALLOC_REQ_RING_TYPE_L2_CMPL = 0x0
RING_ALLOC_REQ_RING_TYPE_TX = 0x1
RING_ALLOC_REQ_RING_TYPE_RX = 0x2
RING_ALLOC_REQ_RING_TYPE_ROCE_CMPL = 0x3
RING_ALLOC_REQ_RING_TYPE_RX_AGG = 0x4
RING_ALLOC_REQ_RING_TYPE_NQ = 0x5
RING_ALLOC_REQ_RING_TYPE_LAST = RING_ALLOC_REQ_RING_TYPE_NQ
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_OFF = 0x0
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_4 = 0x1
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_8 = 0x2
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_12 = 0x3
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_16 = 0x4
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_24 = 0x5
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_32 = 0x6
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_48 = 0x7
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_64 = 0x8
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_96 = 0x9
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_128 = 0xa
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_192 = 0xb
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_256 = 0xc
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_320 = 0xd
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_384 = 0xe
RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_MAX = 0xf
RING_ALLOC_REQ_CMPL_COAL_CNT_LAST = RING_ALLOC_REQ_CMPL_COAL_CNT_COAL_MAX
RING_ALLOC_REQ_FLAGS_RX_SOP_PAD = 0x1
RING_ALLOC_REQ_FLAGS_DISABLE_CQ_OVERFLOW_DETECTION = 0x2
RING_ALLOC_REQ_FLAGS_NQ_DBR_PACING = 0x4
RING_ALLOC_REQ_FLAGS_TX_PKT_TS_CMPL_ENABLE = 0x8
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_MASK = 0xf
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_SFT = 0
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_SP = 0x1
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_WFQ = 0x2
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_LAST = RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_WFQ
RING_ALLOC_REQ_RING_ARB_CFG_RSVD_MASK = 0xf0
RING_ALLOC_REQ_RING_ARB_CFG_RSVD_SFT = 4
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_PARAM_MASK = 0xff00
RING_ALLOC_REQ_RING_ARB_CFG_ARB_POLICY_PARAM_SFT = 8
RING_ALLOC_REQ_MAX_BW_BW_VALUE_MASK = 0xfffffff
RING_ALLOC_REQ_MAX_BW_BW_VALUE_SFT = 0
RING_ALLOC_REQ_MAX_BW_SCALE = 0x10000000
RING_ALLOC_REQ_MAX_BW_SCALE_BITS = (0x0 << 28)
RING_ALLOC_REQ_MAX_BW_SCALE_BYTES = (0x1 << 28)
RING_ALLOC_REQ_MAX_BW_SCALE_LAST = RING_ALLOC_REQ_MAX_BW_SCALE_BYTES
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_MASK = 0xe0000000
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_SFT = 29
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_MEGA = (0x0 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_KILO = (0x2 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_BASE = (0x4 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_GIGA = (0x6 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_PERCENT1_100 = (0x1 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_INVALID = (0x7 << 29)
RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_LAST = RING_ALLOC_REQ_MAX_BW_BW_VALUE_UNIT_INVALID
RING_ALLOC_REQ_INT_MODE_LEGACY = 0x0
RING_ALLOC_REQ_INT_MODE_RSVD = 0x1
RING_ALLOC_REQ_INT_MODE_MSIX = 0x2
RING_ALLOC_REQ_INT_MODE_POLL = 0x3
RING_ALLOC_REQ_INT_MODE_LAST = RING_ALLOC_REQ_INT_MODE_POLL
RING_ALLOC_REQ_MPC_CHNLS_TYPE_TCE = 0x0
RING_ALLOC_REQ_MPC_CHNLS_TYPE_RCE = 0x1
RING_ALLOC_REQ_MPC_CHNLS_TYPE_TE_CFA = 0x2
RING_ALLOC_REQ_MPC_CHNLS_TYPE_RE_CFA = 0x3
RING_ALLOC_REQ_MPC_CHNLS_TYPE_PRIMATE = 0x4
RING_ALLOC_REQ_MPC_CHNLS_TYPE_LAST = RING_ALLOC_REQ_MPC_CHNLS_TYPE_PRIMATE
RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_DEFAULT = 0x0
RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_POLL_MODE = 0x1
RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_LAST = RING_ALLOC_REQ_RX_RATE_PROFILE_SEL_POLL_MODE
RING_ALLOC_RESP_PUSH_BUFFER_INDEX_PING_BUFFER = 0x0
RING_ALLOC_RESP_PUSH_BUFFER_INDEX_PONG_BUFFER = 0x1
RING_ALLOC_RESP_PUSH_BUFFER_INDEX_LAST = RING_ALLOC_RESP_PUSH_BUFFER_INDEX_PONG_BUFFER
RING_FREE_REQ_RING_TYPE_L2_CMPL = 0x0
RING_FREE_REQ_RING_TYPE_TX = 0x1
RING_FREE_REQ_RING_TYPE_RX = 0x2
RING_FREE_REQ_RING_TYPE_ROCE_CMPL = 0x3
RING_FREE_REQ_RING_TYPE_RX_AGG = 0x4
RING_FREE_REQ_RING_TYPE_NQ = 0x5
RING_FREE_REQ_RING_TYPE_LAST = RING_FREE_REQ_RING_TYPE_NQ
RING_FREE_REQ_FLAGS_VIRTIO_RING_VALID = 0x1
RING_FREE_REQ_FLAGS_LAST = RING_FREE_REQ_FLAGS_VIRTIO_RING_VALID
RING_RESET_REQ_RING_TYPE_L2_CMPL = 0x0
RING_RESET_REQ_RING_TYPE_TX = 0x1
RING_RESET_REQ_RING_TYPE_RX = 0x2
RING_RESET_REQ_RING_TYPE_ROCE_CMPL = 0x3
RING_RESET_REQ_RING_TYPE_RX_RING_GRP = 0x6
RING_RESET_REQ_RING_TYPE_LAST = RING_RESET_REQ_RING_TYPE_RX_RING_GRP
RING_RESET_RESP_PUSH_BUFFER_INDEX_PING_BUFFER = 0x0
RING_RESET_RESP_PUSH_BUFFER_INDEX_PONG_BUFFER = 0x1
RING_RESET_RESP_PUSH_BUFFER_INDEX_LAST = RING_RESET_RESP_PUSH_BUFFER_INDEX_PONG_BUFFER
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_INT_LAT_TMR_MIN = 0x1
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_INT_LAT_TMR_MAX = 0x2
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_TIMER_RESET = 0x4
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_RING_IDLE = 0x8
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_NUM_CMPL_DMA_AGGR = 0x10
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_NUM_CMPL_DMA_AGGR_DURING_INT = 0x20
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_CMPL_AGGR_DMA_TMR = 0x40
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_CMPL_AGGR_DMA_TMR_DURING_INT = 0x80
RING_AGGINT_QCAPS_RESP_CMPL_PARAMS_NUM_CMPL_AGGR_INT = 0x100
RING_AGGINT_QCAPS_RESP_NQ_PARAMS_INT_LAT_TMR_MIN = 0x1
RING_CMPL_RING_QAGGINT_PARAMS_REQ_FLAGS_UNUSED_0_MASK = 0x3
RING_CMPL_RING_QAGGINT_PARAMS_REQ_FLAGS_UNUSED_0_SFT = 0
RING_CMPL_RING_QAGGINT_PARAMS_REQ_FLAGS_IS_NQ = 0x4
RING_CMPL_RING_QAGGINT_PARAMS_RESP_FLAGS_TIMER_RESET = 0x1
RING_CMPL_RING_QAGGINT_PARAMS_RESP_FLAGS_RING_IDLE = 0x2
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_FLAGS_TIMER_RESET = 0x1
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_FLAGS_RING_IDLE = 0x2
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_FLAGS_IS_NQ = 0x4
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_ENABLES_NUM_CMPL_DMA_AGGR = 0x1
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_ENABLES_NUM_CMPL_DMA_AGGR_DURING_INT = 0x2
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_ENABLES_CMPL_AGGR_DMA_TMR = 0x4
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_ENABLES_INT_LAT_TMR_MIN = 0x8
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_ENABLES_INT_LAT_TMR_MAX = 0x10
RING_CMPL_RING_CFG_AGGINT_PARAMS_REQ_ENABLES_NUM_CMPL_AGGR_INT = 0x20
DEFAULT_FLOW_ID = 0xFFFFFFFF
ROCEV1_FLOW_ID = 0xFFFFFFFE
ROCEV2_FLOW_ID = 0xFFFFFFFD
ROCEV2_CNP_FLOW_ID = 0xFFFFFFFC
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH = 0x1
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_TX = 0x0
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX = 0x1
CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_LAST = CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX
CFA_L2_FILTER_ALLOC_REQ_FLAGS_LOOPBACK = 0x2
CFA_L2_FILTER_ALLOC_REQ_FLAGS_DROP = 0x4
CFA_L2_FILTER_ALLOC_REQ_FLAGS_OUTERMOST = 0x8
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_MASK = 0x30
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_SFT = 4
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_NO_ROCE_L2 = (0x0 << 4)
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_L2 = (0x1 << 4)
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_ROCE = (0x2 << 4)
CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_LAST = CFA_L2_FILTER_ALLOC_REQ_FLAGS_TRAFFIC_ROCE
CFA_L2_FILTER_ALLOC_REQ_FLAGS_XDP_DISABLE = 0x40
CFA_L2_FILTER_ALLOC_REQ_FLAGS_SOURCE_VALID = 0x80
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR = 0x1
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR_MASK = 0x2
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_OVLAN = 0x4
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_OVLAN_MASK = 0x8
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_IVLAN = 0x10
CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_IVLAN_MASK = 0x20
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_ADDR = 0x40
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_ADDR_MASK = 0x80
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_OVLAN = 0x100
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_OVLAN_MASK = 0x200
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_IVLAN = 0x400
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_L2_IVLAN_MASK = 0x800
CFA_L2_FILTER_ALLOC_REQ_ENABLES_SRC_TYPE = 0x1000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_SRC_ID = 0x2000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_TUNNEL_TYPE = 0x4000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_DST_ID = 0x8000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_MIRROR_VNIC_ID = 0x10000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_NUM_VLANS = 0x20000
CFA_L2_FILTER_ALLOC_REQ_ENABLES_T_NUM_VLANS = 0x40000
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_NPORT = 0x0
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_PF = 0x1
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_VF = 0x2
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_VNIC = 0x3
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_KONG = 0x4
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_APE = 0x5
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_BONO = 0x6
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_TANG = 0x7
CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_LAST = CFA_L2_FILTER_ALLOC_REQ_SRC_TYPE_TANG
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_NONTUNNEL = 0x0
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_NVGRE = 0x2
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2GRE = 0x3
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPIP = 0x4
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_MPLS = 0x6
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_STT = 0x7
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE = 0x8
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL = 0xff
CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_LAST = CFA_L2_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_NO_PREFER = 0x0
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_ABOVE_FILTER = 0x1
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_BELOW_FILTER = 0x2
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_MAX = 0x3
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_MIN = 0x4
CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_LAST = CFA_L2_FILTER_ALLOC_REQ_PRI_HINT_MIN
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_VALUE_MASK = 0x3fffffff
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_VALUE_SFT = 0
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE = 0x40000000
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_INT = (0x0 << 30)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT = (0x1 << 30)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_LAST = CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR = 0x80000000
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_RX = (0x0 << 31)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX = (0x1 << 31)
CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_LAST = CFA_L2_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH = 0x1
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_TX = 0x0
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_RX = 0x1
CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_LAST = CFA_L2_FILTER_CFG_REQ_FLAGS_PATH_RX
CFA_L2_FILTER_CFG_REQ_FLAGS_DROP = 0x2
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_MASK = 0xc
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_SFT = 2
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_NO_ROCE_L2 = (0x0 << 2)
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_L2 = (0x1 << 2)
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_ROCE = (0x2 << 2)
CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_LAST = CFA_L2_FILTER_CFG_REQ_FLAGS_TRAFFIC_ROCE
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_MASK = 0x30
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_SFT = 4
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_NO_UPDATE = (0x0 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_BYPASS_LKUP = (0x1 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_ENABLE_LKUP = (0x2 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_RESTORE_FW_OP = (0x3 << 4)
CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_LAST = CFA_L2_FILTER_CFG_REQ_FLAGS_REMAP_OP_RESTORE_FW_OP
CFA_L2_FILTER_CFG_REQ_ENABLES_DST_ID = 0x1
CFA_L2_FILTER_CFG_REQ_ENABLES_NEW_MIRROR_VNIC_ID = 0x2
CFA_L2_FILTER_CFG_REQ_ENABLES_PROF_FUNC = 0x4
CFA_L2_FILTER_CFG_REQ_ENABLES_L2_CONTEXT_ID = 0x8
CFA_L2_SET_RX_MASK_REQ_MASK_MCAST = 0x2
CFA_L2_SET_RX_MASK_REQ_MASK_ALL_MCAST = 0x4
CFA_L2_SET_RX_MASK_REQ_MASK_BCAST = 0x8
CFA_L2_SET_RX_MASK_REQ_MASK_PROMISCUOUS = 0x10
CFA_L2_SET_RX_MASK_REQ_MASK_OUTERMOST = 0x20
CFA_L2_SET_RX_MASK_REQ_MASK_VLANONLY = 0x40
CFA_L2_SET_RX_MASK_REQ_MASK_VLAN_NONVLAN = 0x80
CFA_L2_SET_RX_MASK_REQ_MASK_ANYVLAN_NONVLAN = 0x100
CFA_L2_SET_RX_MASK_CMD_ERR_CODE_UNKNOWN = 0x0
CFA_L2_SET_RX_MASK_CMD_ERR_CODE_NTUPLE_FILTER_CONFLICT_ERR = 0x1
CFA_L2_SET_RX_MASK_CMD_ERR_CODE_MAX_VLAN_TAGS = 0x2
CFA_L2_SET_RX_MASK_CMD_ERR_CODE_INVALID_VNIC_ID = 0x3
CFA_L2_SET_RX_MASK_CMD_ERR_CODE_INVALID_ACTION = 0x4
CFA_L2_SET_RX_MASK_CMD_ERR_CODE_LAST = CFA_L2_SET_RX_MASK_CMD_ERR_CODE_INVALID_ACTION
CFA_TUNNEL_FILTER_ALLOC_REQ_FLAGS_LOOPBACK = 0x1
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_L2_FILTER_ID = 0x1
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_L2_ADDR = 0x2
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_L2_IVLAN = 0x4
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_L3_ADDR = 0x8
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_L3_ADDR_TYPE = 0x10
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_T_L3_ADDR_TYPE = 0x20
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_T_L3_ADDR = 0x40
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_TUNNEL_TYPE = 0x80
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_VNI = 0x100
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_DST_VNIC_ID = 0x200
CFA_TUNNEL_FILTER_ALLOC_REQ_ENABLES_MIRROR_VNIC_ID = 0x400
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_NONTUNNEL = 0x0
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_NVGRE = 0x2
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2GRE = 0x3
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPIP = 0x4
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_MPLS = 0x6
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_STT = 0x7
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE = 0x8
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL = 0xff
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_LAST = CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_FLAGS_TUN_FLAGS_OAM_CHECKSUM_EXPLHDR = 0x1
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_FLAGS_TUN_FLAGS_CRITICAL_OPT_S1 = 0x2
CFA_TUNNEL_FILTER_ALLOC_REQ_TUNNEL_FLAGS_TUN_FLAGS_EXTHDR_SEQNUM_S0 = 0x4
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_VALUE_MASK = 0x3fffffff
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_VALUE_SFT = 0
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_TYPE = 0x40000000
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_TYPE_INT = (0x0 << 30)
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT = (0x1 << 30)
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_TYPE_LAST = CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_DIR = 0x80000000
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_DIR_RX = (0x0 << 31)
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX = (0x1 << 31)
CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_DIR_LAST = CFA_TUNNEL_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX
VXLAN_IPV4_HDR_VER_HLEN_HEADER_LENGTH_MASK = 0xf
VXLAN_IPV4_HDR_VER_HLEN_HEADER_LENGTH_SFT = 0
VXLAN_IPV4_HDR_VER_HLEN_VERSION_MASK = 0xf0
VXLAN_IPV4_HDR_VER_HLEN_VERSION_SFT = 4
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_VER_SFT = 0x1c
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_VER_MASK = 0xf0000000
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_TC_SFT = 0x14
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_TC_MASK = 0xff00000
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_FLOW_LABEL_SFT = 0x0
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_FLOW_LABEL_MASK = 0xfffff
VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_LAST = VXLAN_IPV6_HDR_VER_TC_FLOW_LABEL_FLOW_LABEL_MASK
CFA_ENCAP_DATA_VXLAN_L3_VER_MASK = 0xf
CFA_ENCAP_DATA_VXLAN_L3_VER_IPV4 = 0x4
CFA_ENCAP_DATA_VXLAN_L3_VER_IPV6 = 0x6
CFA_ENCAP_DATA_VXLAN_L3_LAST = CFA_ENCAP_DATA_VXLAN_L3_VER_IPV6
CFA_ENCAP_RECORD_ALLOC_REQ_FLAGS_LOOPBACK = 0x1
CFA_ENCAP_RECORD_ALLOC_REQ_FLAGS_EXTERNAL = 0x2
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_VXLAN = 0x1
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_NVGRE = 0x2
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_L2GRE = 0x3
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_IPIP = 0x4
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_GENEVE = 0x5
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_MPLS = 0x6
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_VLAN = 0x7
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_IPGRE = 0x8
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_VXLAN_V4 = 0x9
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_IPGRE_V1 = 0xa
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_L2_ETYPE = 0xb
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_VXLAN_GPE_V6 = 0xc
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_VXLAN_GPE = 0x10
CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_LAST = CFA_ENCAP_RECORD_ALLOC_REQ_ENCAP_TYPE_VXLAN_GPE
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_LOOPBACK = 0x1
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_DROP = 0x2
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_METER = 0x4
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_DEST_FID = 0x8
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_ARP_REPLY = 0x10
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_DEST_RFS_RING_IDX = 0x20
CFA_NTUPLE_FILTER_ALLOC_REQ_FLAGS_NO_L2_CONTEXT = 0x40
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_L2_FILTER_ID = 0x1
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_ETHERTYPE = 0x2
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_TUNNEL_TYPE = 0x4
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_SRC_MACADDR = 0x8
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_IPADDR_TYPE = 0x10
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_SRC_IPADDR = 0x20
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_SRC_IPADDR_MASK = 0x40
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_DST_IPADDR = 0x80
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_DST_IPADDR_MASK = 0x100
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_IP_PROTOCOL = 0x200
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_SRC_PORT = 0x400
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_SRC_PORT_MASK = 0x800
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_DST_PORT = 0x1000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_DST_PORT_MASK = 0x2000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_PRI_HINT = 0x4000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_NTUPLE_FILTER_ID = 0x8000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_DST_ID = 0x10000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_MIRROR_VNIC_ID = 0x20000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_DST_MACADDR = 0x40000
CFA_NTUPLE_FILTER_ALLOC_REQ_ENABLES_RFS_RING_TBL_IDX = 0x80000
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_ADDR_TYPE_UNKNOWN = 0x0
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_ADDR_TYPE_IPV4 = 0x4
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_ADDR_TYPE_IPV6 = 0x6
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_ADDR_TYPE_LAST = CFA_NTUPLE_FILTER_ALLOC_REQ_IP_ADDR_TYPE_IPV6
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_UNKNOWN = 0x0
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_TCP = 0x6
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_UDP = 0x11
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_ICMP = 0x1
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_ICMPV6 = 0x3a
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_RSVD = 0xff
CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_LAST = CFA_NTUPLE_FILTER_ALLOC_REQ_IP_PROTOCOL_RSVD
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_NONTUNNEL = 0x0
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_NVGRE = 0x2
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2GRE = 0x3
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPIP = 0x4
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_MPLS = 0x6
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_STT = 0x7
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE = 0x8
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL = 0xff
CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_LAST = CFA_NTUPLE_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL
CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_NO_PREFER = 0x0
CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_ABOVE = 0x1
CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_BELOW = 0x2
CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_HIGHEST = 0x3
CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_LOWEST = 0x4
CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_LAST = CFA_NTUPLE_FILTER_ALLOC_REQ_PRI_HINT_LOWEST
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_VALUE_MASK = 0x3fffffff
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_VALUE_SFT = 0
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_TYPE = 0x40000000
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_TYPE_INT = (0x0 << 30)
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT = (0x1 << 30)
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_TYPE_LAST = CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_TYPE_EXT
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_DIR = 0x80000000
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_DIR_RX = (0x0 << 31)
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX = (0x1 << 31)
CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_DIR_LAST = CFA_NTUPLE_FILTER_ALLOC_RESP_FLOW_ID_DIR_TX
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_UNKNOWN = 0x0
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_ZERO_MAC = 0x65
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_BC_MC_MAC = 0x66
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_VNIC = 0x67
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_PF_FID = 0x68
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_L2_CTXT_ID = 0x69
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_NULL_L2_CTXT_CFG = 0x6a
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_NULL_L2_DATA_FLD = 0x6b
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_CFA_LAYOUT = 0x6c
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_L2_CTXT_ALLOC_FAIL = 0x6d
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_ROCE_FLOW_ERR = 0x6e
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_OWNER_FID = 0x6f
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_ZERO_REF_CNT = 0x70
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_FLOW_TYPE = 0x71
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_IVLAN = 0x72
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_MAX_VLAN_ID = 0x73
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_TNL_REQ = 0x74
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_L2_ADDR = 0x75
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_L2_IVLAN = 0x76
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_L3_ADDR = 0x77
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_L3_ADDR_TYPE = 0x78
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_T_L3_ADDR_TYPE = 0x79
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_DST_VNIC_ID = 0x7a
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_VNI = 0x7b
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_DST_ID = 0x7c
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_FAIL_ROCE_L2_FLOW = 0x7d
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_NPAR_VLAN = 0x7e
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_ATSP_ADD = 0x7f
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_DFLT_VLAN_FAIL = 0x80
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_INVALID_L3_TYPE = 0x81
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_VAL_FAIL_TNL_FLOW = 0x82
CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_LAST = CFA_NTUPLE_FILTER_ALLOC_CMD_ERR_CODE_VAL_FAIL_TNL_FLOW
CFA_NTUPLE_FILTER_CFG_REQ_ENABLES_NEW_DST_ID = 0x1
CFA_NTUPLE_FILTER_CFG_REQ_ENABLES_NEW_MIRROR_VNIC_ID = 0x2
CFA_NTUPLE_FILTER_CFG_REQ_ENABLES_NEW_METER_INSTANCE_ID = 0x4
CFA_NTUPLE_FILTER_CFG_REQ_FLAGS_DEST_FID = 0x1
CFA_NTUPLE_FILTER_CFG_REQ_FLAGS_DEST_RFS_RING_IDX = 0x2
CFA_NTUPLE_FILTER_CFG_REQ_FLAGS_NO_L2_CONTEXT = 0x4
CFA_NTUPLE_FILTER_CFG_REQ_NEW_METER_INSTANCE_ID_INVALID = 0xffff
CFA_NTUPLE_FILTER_CFG_REQ_NEW_METER_INSTANCE_ID_LAST = CFA_NTUPLE_FILTER_CFG_REQ_NEW_METER_INSTANCE_ID_INVALID
CFA_DECAP_FILTER_ALLOC_REQ_FLAGS_OVS_TUNNEL = 0x1
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_TUNNEL_TYPE = 0x1
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_TUNNEL_ID = 0x2
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_SRC_MACADDR = 0x4
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_DST_MACADDR = 0x8
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_OVLAN_VID = 0x10
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_IVLAN_VID = 0x20
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_T_OVLAN_VID = 0x40
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_T_IVLAN_VID = 0x80
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_ETHERTYPE = 0x100
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_SRC_IPADDR = 0x200
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_DST_IPADDR = 0x400
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_IPADDR_TYPE = 0x800
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_IP_PROTOCOL = 0x1000
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_SRC_PORT = 0x2000
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_DST_PORT = 0x4000
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_DST_ID = 0x8000
CFA_DECAP_FILTER_ALLOC_REQ_ENABLES_MIRROR_VNIC_ID = 0x10000
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_NONTUNNEL = 0x0
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_NVGRE = 0x2
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2GRE = 0x3
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPIP = 0x4
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_MPLS = 0x6
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_STT = 0x7
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE = 0x8
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL = 0xff
CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_LAST = CFA_DECAP_FILTER_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL
CFA_DECAP_FILTER_ALLOC_REQ_IP_ADDR_TYPE_UNKNOWN = 0x0
CFA_DECAP_FILTER_ALLOC_REQ_IP_ADDR_TYPE_IPV4 = 0x4
CFA_DECAP_FILTER_ALLOC_REQ_IP_ADDR_TYPE_IPV6 = 0x6
CFA_DECAP_FILTER_ALLOC_REQ_IP_ADDR_TYPE_LAST = CFA_DECAP_FILTER_ALLOC_REQ_IP_ADDR_TYPE_IPV6
CFA_DECAP_FILTER_ALLOC_REQ_IP_PROTOCOL_UNKNOWN = 0x0
CFA_DECAP_FILTER_ALLOC_REQ_IP_PROTOCOL_TCP = 0x6
CFA_DECAP_FILTER_ALLOC_REQ_IP_PROTOCOL_UDP = 0x11
CFA_DECAP_FILTER_ALLOC_REQ_IP_PROTOCOL_LAST = CFA_DECAP_FILTER_ALLOC_REQ_IP_PROTOCOL_UDP
CFA_FLOW_ALLOC_REQ_FLAGS_TUNNEL = 0x1
CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_MASK = 0x6
CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_SFT = 1
CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_NONE = (0x0 << 1)
CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_ONE = (0x1 << 1)
CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_TWO = (0x2 << 1)
CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_LAST = CFA_FLOW_ALLOC_REQ_FLAGS_NUM_VLAN_TWO
CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_MASK = 0x38
CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_SFT = 3
CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_L2 = (0x0 << 3)
CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_IPV4 = (0x1 << 3)
CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_IPV6 = (0x2 << 3)
CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_LAST = CFA_FLOW_ALLOC_REQ_FLAGS_FLOWTYPE_IPV6
CFA_FLOW_ALLOC_REQ_FLAGS_PATH_TX = 0x40
CFA_FLOW_ALLOC_REQ_FLAGS_PATH_RX = 0x80
CFA_FLOW_ALLOC_REQ_FLAGS_MATCH_VXLAN_IP_VNI = 0x100
CFA_FLOW_ALLOC_REQ_FLAGS_VHOST_ID_USE_VLAN = 0x200
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_FWD = 0x1
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_RECYCLE = 0x2
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_DROP = 0x4
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_METER = 0x8
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_TUNNEL = 0x10
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_NAT_SRC = 0x20
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_NAT_DEST = 0x40
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_NAT_IPV4_ADDRESS = 0x80
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_L2_HEADER_REWRITE = 0x100
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_TTL_DECREMENT = 0x200
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_TUNNEL_IP = 0x400
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_FLOW_AGING_ENABLED = 0x800
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_PRI_HINT = 0x1000
CFA_FLOW_ALLOC_REQ_ACTION_FLAGS_NO_FLOW_COUNTER_ALLOC = 0x2000
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_NONTUNNEL = 0x0
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_NVGRE = 0x2
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_L2GRE = 0x3
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_IPIP = 0x4
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_MPLS = 0x6
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_STT = 0x7
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_IPGRE = 0x8
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL = 0xff
CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_LAST = CFA_FLOW_ALLOC_REQ_TUNNEL_TYPE_ANYTUNNEL
CFA_FLOW_ALLOC_RESP_FLOW_ID_VALUE_MASK = 0x3fffffff
CFA_FLOW_ALLOC_RESP_FLOW_ID_VALUE_SFT = 0
CFA_FLOW_ALLOC_RESP_FLOW_ID_TYPE = 0x40000000
CFA_FLOW_ALLOC_RESP_FLOW_ID_TYPE_INT = (0x0 << 30)
CFA_FLOW_ALLOC_RESP_FLOW_ID_TYPE_EXT = (0x1 << 30)
CFA_FLOW_ALLOC_RESP_FLOW_ID_TYPE_LAST = CFA_FLOW_ALLOC_RESP_FLOW_ID_TYPE_EXT
CFA_FLOW_ALLOC_RESP_FLOW_ID_DIR = 0x80000000
CFA_FLOW_ALLOC_RESP_FLOW_ID_DIR_RX = (0x0 << 31)
CFA_FLOW_ALLOC_RESP_FLOW_ID_DIR_TX = (0x1 << 31)
CFA_FLOW_ALLOC_RESP_FLOW_ID_DIR_LAST = CFA_FLOW_ALLOC_RESP_FLOW_ID_DIR_TX
CFA_FLOW_ALLOC_CMD_ERR_CODE_UNKNOWN = 0x0
CFA_FLOW_ALLOC_CMD_ERR_CODE_L2_CONTEXT_TCAM = 0x1
CFA_FLOW_ALLOC_CMD_ERR_CODE_ACTION_RECORD = 0x2
CFA_FLOW_ALLOC_CMD_ERR_CODE_FLOW_COUNTER = 0x3
CFA_FLOW_ALLOC_CMD_ERR_CODE_WILD_CARD_TCAM = 0x4
CFA_FLOW_ALLOC_CMD_ERR_CODE_HASH_COLLISION = 0x5
CFA_FLOW_ALLOC_CMD_ERR_CODE_KEY_EXISTS = 0x6
CFA_FLOW_ALLOC_CMD_ERR_CODE_FLOW_CTXT_DB = 0x7
CFA_FLOW_ALLOC_CMD_ERR_CODE_LAST = CFA_FLOW_ALLOC_CMD_ERR_CODE_FLOW_CTXT_DB
CFA_FLOW_INFO_REQ_FLOW_HANDLE_MAX_MASK = 0xfff
CFA_FLOW_INFO_REQ_FLOW_HANDLE_CNP_CNT = 0x1000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_ROCEV1_CNT = 0x2000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_NIC_TX = 0x3000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_ROCEV2_CNT = 0x4000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_DIR_RX = 0x8000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_CNP_CNT_RX = 0x9000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_ROCEV1_CNT_RX = 0xa000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_NIC_RX = 0xb000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_ROCEV2_CNT_RX = 0xc000
CFA_FLOW_INFO_REQ_FLOW_HANDLE_LAST = CFA_FLOW_INFO_REQ_FLOW_HANDLE_ROCEV2_CNT_RX
CFA_FLOW_INFO_RESP_FLAGS_PATH_TX = 0x1
CFA_FLOW_INFO_RESP_FLAGS_PATH_RX = 0x2
CFA_EEM_QCAPS_REQ_FLAGS_PATH_TX = 0x1
CFA_EEM_QCAPS_REQ_FLAGS_PATH_RX = 0x2
CFA_EEM_QCAPS_REQ_FLAGS_PREFERRED_OFFLOAD = 0x4
CFA_EEM_QCAPS_RESP_FLAGS_PATH_TX = 0x1
CFA_EEM_QCAPS_RESP_FLAGS_PATH_RX = 0x2
CFA_EEM_QCAPS_RESP_FLAGS_CENTRALIZED_MEMORY_MODEL_SUPPORTED = 0x4
CFA_EEM_QCAPS_RESP_FLAGS_DETACHED_CENTRALIZED_MEMORY_MODEL_SUPPORTED = 0x8
CFA_EEM_QCAPS_RESP_SUPPORTED_KEY0_TABLE = 0x1
CFA_EEM_QCAPS_RESP_SUPPORTED_KEY1_TABLE = 0x2
CFA_EEM_QCAPS_RESP_SUPPORTED_EXTERNAL_RECORD_TABLE = 0x4
CFA_EEM_QCAPS_RESP_SUPPORTED_EXTERNAL_FLOW_COUNTERS_TABLE = 0x8
CFA_EEM_QCAPS_RESP_SUPPORTED_FID_TABLE = 0x10
CFA_EEM_CFG_REQ_FLAGS_PATH_TX = 0x1
CFA_EEM_CFG_REQ_FLAGS_PATH_RX = 0x2
CFA_EEM_CFG_REQ_FLAGS_PREFERRED_OFFLOAD = 0x4
CFA_EEM_CFG_REQ_FLAGS_SECONDARY_PF = 0x8
CFA_EEM_QCFG_REQ_FLAGS_PATH_TX = 0x1
CFA_EEM_QCFG_REQ_FLAGS_PATH_RX = 0x2
CFA_EEM_QCFG_RESP_FLAGS_PATH_TX = 0x1
CFA_EEM_QCFG_RESP_FLAGS_PATH_RX = 0x2
CFA_EEM_QCFG_RESP_FLAGS_PREFERRED_OFFLOAD = 0x4
CFA_EEM_OP_REQ_FLAGS_PATH_TX = 0x1
CFA_EEM_OP_REQ_FLAGS_PATH_RX = 0x2
CFA_EEM_OP_REQ_OP_RESERVED = 0x0
CFA_EEM_OP_REQ_OP_EEM_DISABLE = 0x1
CFA_EEM_OP_REQ_OP_EEM_ENABLE = 0x2
CFA_EEM_OP_REQ_OP_EEM_CLEANUP = 0x3
CFA_EEM_OP_REQ_OP_LAST = CFA_EEM_OP_REQ_OP_EEM_CLEANUP
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_FLOW_HND_16BIT_SUPPORTED = 0x1
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_FLOW_HND_64BIT_SUPPORTED = 0x2
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_FLOW_BATCH_DELETE_SUPPORTED = 0x4
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_FLOW_RESET_ALL_SUPPORTED = 0x8
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_NTUPLE_FLOW_DEST_FUNC_SUPPORTED = 0x10
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_TX_EEM_FLOW_SUPPORTED = 0x20
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_RX_EEM_FLOW_SUPPORTED = 0x40
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_FLOW_COUNTER_ALLOC_SUPPORTED = 0x80
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_RFS_RING_TBL_IDX_SUPPORTED = 0x100
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_UNTAGGED_VLAN_SUPPORTED = 0x200
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_XDP_SUPPORTED = 0x400
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_L2_HEADER_SOURCE_FIELDS_SUPPORTED = 0x800
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_NTUPLE_FLOW_RX_ARP_SUPPORTED = 0x1000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_RFS_RING_TBL_IDX_V2_SUPPORTED = 0x2000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_NTUPLE_FLOW_RX_ETHERTYPE_IP_SUPPORTED = 0x4000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_TRUFLOW_CAPABLE = 0x8000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_L2_FILTER_TRAFFIC_TYPE_L2_ROCE_SUPPORTED = 0x10000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_LAG_SUPPORTED = 0x20000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_NTUPLE_FLOW_NO_L2CTX_SUPPORTED = 0x40000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_NIC_FLOW_STATS_SUPPORTED = 0x80000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_NTUPLE_FLOW_RX_EXT_IP_PROTO_SUPPORTED = 0x100000
CFA_ADV_FLOW_MGNT_QCAPS_RESP_FLAGS_RFS_RING_TBL_IDX_V3_SUPPORTED = 0x200000
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_VXLAN = 0x1
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_GENEVE = 0x5
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_CUSTOM_GRE = 0xd
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ECPRI = 0xe
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_SRV6 = 0xf
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_GRE = 0x11
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR = 0x12
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES01 = 0x13
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES02 = 0x14
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES03 = 0x15
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES04 = 0x16
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES05 = 0x17
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES06 = 0x18
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES07 = 0x19
TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_LAST = TUNNEL_DST_PORT_QUERY_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES07
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR0 = 0x1
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR1 = 0x2
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR2 = 0x4
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR3 = 0x8
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR4 = 0x10
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR5 = 0x20
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR6 = 0x40
TUNNEL_DST_PORT_QUERY_RESP_UPAR_IN_USE_UPAR7 = 0x80
TUNNEL_DST_PORT_QUERY_RESP_STATUS_CHIP_LEVEL = 0x1
TUNNEL_DST_PORT_QUERY_RESP_STATUS_FUNC_LEVEL = 0x2
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_VXLAN = 0x1
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_GENEVE = 0x5
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_CUSTOM_GRE = 0xd
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ECPRI = 0xe
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_SRV6 = 0xf
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_GRE = 0x11
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR = 0x12
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES01 = 0x13
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES02 = 0x14
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES03 = 0x15
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES04 = 0x16
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES05 = 0x17
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES06 = 0x18
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES07 = 0x19
TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_LAST = TUNNEL_DST_PORT_ALLOC_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES07
TUNNEL_DST_PORT_ALLOC_RESP_ERROR_INFO_SUCCESS = 0x0
TUNNEL_DST_PORT_ALLOC_RESP_ERROR_INFO_ERR_ALLOCATED = 0x1
TUNNEL_DST_PORT_ALLOC_RESP_ERROR_INFO_ERR_NO_RESOURCE = 0x2
TUNNEL_DST_PORT_ALLOC_RESP_ERROR_INFO_ERR_ENABLED = 0x3
TUNNEL_DST_PORT_ALLOC_RESP_ERROR_INFO_LAST = TUNNEL_DST_PORT_ALLOC_RESP_ERROR_INFO_ERR_ENABLED
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR0 = 0x1
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR1 = 0x2
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR2 = 0x4
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR3 = 0x8
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR4 = 0x10
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR5 = 0x20
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR6 = 0x40
TUNNEL_DST_PORT_ALLOC_RESP_UPAR_IN_USE_UPAR7 = 0x80
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_VXLAN = 0x1
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_GENEVE = 0x5
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_VXLAN_V4 = 0x9
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_IPGRE_V1 = 0xa
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_L2_ETYPE = 0xb
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_VXLAN_GPE_V6 = 0xc
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_CUSTOM_GRE = 0xd
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ECPRI = 0xe
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_SRV6 = 0xf
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_VXLAN_GPE = 0x10
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_GRE = 0x11
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR = 0x12
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES01 = 0x13
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES02 = 0x14
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES03 = 0x15
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES04 = 0x16
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES05 = 0x17
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES06 = 0x18
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES07 = 0x19
TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_LAST = TUNNEL_DST_PORT_FREE_REQ_TUNNEL_TYPE_ULP_DYN_UPAR_RES07
TUNNEL_DST_PORT_FREE_RESP_ERROR_INFO_SUCCESS = 0x0
TUNNEL_DST_PORT_FREE_RESP_ERROR_INFO_ERR_NOT_OWNER = 0x1
TUNNEL_DST_PORT_FREE_RESP_ERROR_INFO_ERR_NOT_ALLOCATED = 0x2
TUNNEL_DST_PORT_FREE_RESP_ERROR_INFO_LAST = TUNNEL_DST_PORT_FREE_RESP_ERROR_INFO_ERR_NOT_ALLOCATED
STAT_CTX_ALLOC_REQ_STAT_CTX_FLAGS_ROCE = 0x1
STAT_CTX_ALLOC_REQ_STAT_CTX_FLAGS_DUP_HOST_BUF = 0x2
STAT_CTX_ALLOC_REQ_FLAGS_STEERING_TAG_VALID = 0x1
STAT_CTX_QUERY_REQ_FLAGS_COUNTER_MASK = 0x1
STAT_EXT_CTX_QUERY_REQ_FLAGS_COUNTER_MASK = 0x1
STAT_GENERIC_QSTATS_REQ_FLAGS_COUNTER_MASK = 0x1
FW_RESET_REQ_EMBEDDED_PROC_TYPE_BOOT = 0x0
FW_RESET_REQ_EMBEDDED_PROC_TYPE_MGMT = 0x1
FW_RESET_REQ_EMBEDDED_PROC_TYPE_NETCTRL = 0x2
FW_RESET_REQ_EMBEDDED_PROC_TYPE_ROCE = 0x3
FW_RESET_REQ_EMBEDDED_PROC_TYPE_HOST = 0x4
FW_RESET_REQ_EMBEDDED_PROC_TYPE_AP = 0x5
FW_RESET_REQ_EMBEDDED_PROC_TYPE_CHIP = 0x6
FW_RESET_REQ_EMBEDDED_PROC_TYPE_HOST_RESOURCE_REINIT = 0x7
FW_RESET_REQ_EMBEDDED_PROC_TYPE_IMPACTLESS_ACTIVATION = 0x8
FW_RESET_REQ_EMBEDDED_PROC_TYPE_LAST = FW_RESET_REQ_EMBEDDED_PROC_TYPE_IMPACTLESS_ACTIVATION
FW_RESET_REQ_SELFRST_STATUS_SELFRSTNONE = 0x0
FW_RESET_REQ_SELFRST_STATUS_SELFRSTASAP = 0x1
FW_RESET_REQ_SELFRST_STATUS_SELFRSTPCIERST = 0x2
FW_RESET_REQ_SELFRST_STATUS_SELFRSTIMMEDIATE = 0x3
FW_RESET_REQ_SELFRST_STATUS_LAST = FW_RESET_REQ_SELFRST_STATUS_SELFRSTIMMEDIATE
FW_RESET_REQ_FLAGS_RESET_GRACEFUL = 0x1
FW_RESET_REQ_FLAGS_FW_ACTIVATION = 0x2
FW_RESET_RESP_SELFRST_STATUS_SELFRSTNONE = 0x0
FW_RESET_RESP_SELFRST_STATUS_SELFRSTASAP = 0x1
FW_RESET_RESP_SELFRST_STATUS_SELFRSTPCIERST = 0x2
FW_RESET_RESP_SELFRST_STATUS_SELFRSTIMMEDIATE = 0x3
FW_RESET_RESP_SELFRST_STATUS_LAST = FW_RESET_RESP_SELFRST_STATUS_SELFRSTIMMEDIATE
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_BOOT = 0x0
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_MGMT = 0x1
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_NETCTRL = 0x2
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_ROCE = 0x3
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_HOST = 0x4
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_AP = 0x5
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_CHIP = 0x6
FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_LAST = FW_QSTATUS_REQ_EMBEDDED_PROC_TYPE_CHIP
FW_QSTATUS_RESP_SELFRST_STATUS_SELFRSTNONE = 0x0
FW_QSTATUS_RESP_SELFRST_STATUS_SELFRSTASAP = 0x1
FW_QSTATUS_RESP_SELFRST_STATUS_SELFRSTPCIERST = 0x2
FW_QSTATUS_RESP_SELFRST_STATUS_SELFRSTPOWER = 0x3
FW_QSTATUS_RESP_SELFRST_STATUS_LAST = FW_QSTATUS_RESP_SELFRST_STATUS_SELFRSTPOWER
FW_QSTATUS_RESP_NVM_OPTION_ACTION_STATUS_NVMOPT_ACTION_NONE = 0x0
FW_QSTATUS_RESP_NVM_OPTION_ACTION_STATUS_NVMOPT_ACTION_HOTRESET = 0x1
FW_QSTATUS_RESP_NVM_OPTION_ACTION_STATUS_NVMOPT_ACTION_WARMBOOT = 0x2
FW_QSTATUS_RESP_NVM_OPTION_ACTION_STATUS_NVMOPT_ACTION_COLDBOOT = 0x3
FW_QSTATUS_RESP_NVM_OPTION_ACTION_STATUS_LAST = FW_QSTATUS_RESP_NVM_OPTION_ACTION_STATUS_NVMOPT_ACTION_COLDBOOT
FW_SET_TIME_REQ_YEAR_UNKNOWN = 0x0
FW_SET_TIME_REQ_YEAR_LAST = FW_SET_TIME_REQ_YEAR_UNKNOWN
FW_SET_TIME_REQ_ZONE_UTC = 0
FW_SET_TIME_REQ_ZONE_UNKNOWN = 65535
FW_SET_TIME_REQ_ZONE_LAST = FW_SET_TIME_REQ_ZONE_UNKNOWN
STRUCT_HDR_STRUCT_ID_LLDP_CFG = 0x41b
STRUCT_HDR_STRUCT_ID_DCBX_ETS = 0x41d
STRUCT_HDR_STRUCT_ID_DCBX_PFC = 0x41f
STRUCT_HDR_STRUCT_ID_DCBX_APP = 0x421
STRUCT_HDR_STRUCT_ID_DCBX_FEATURE_STATE = 0x422
STRUCT_HDR_STRUCT_ID_LLDP_GENERIC = 0x424
STRUCT_HDR_STRUCT_ID_LLDP_DEVICE = 0x426
STRUCT_HDR_STRUCT_ID_POWER_BKUP = 0x427
STRUCT_HDR_STRUCT_ID_PEER_MMAP = 0x429
STRUCT_HDR_STRUCT_ID_AFM_OPAQUE = 0x1
STRUCT_HDR_STRUCT_ID_PORT_DESCRIPTION = 0xa
STRUCT_HDR_STRUCT_ID_RSS_V2 = 0x64
STRUCT_HDR_STRUCT_ID_MSIX_PER_VF = 0xc8
STRUCT_HDR_STRUCT_ID_UDCC_RTT_BUCKET_COUNT = 0x12c
STRUCT_HDR_STRUCT_ID_UDCC_RTT_BUCKET_BOUND = 0x12d
STRUCT_HDR_STRUCT_ID_DBG_TOKEN_CLAIMS = 0x190
STRUCT_HDR_STRUCT_ID_LAST = STRUCT_HDR_STRUCT_ID_DBG_TOKEN_CLAIMS
STRUCT_HDR_VERSION_0 = 0x0
STRUCT_HDR_VERSION_1 = 0x1
STRUCT_HDR_VERSION_LAST = STRUCT_HDR_VERSION_1
STRUCT_HDR_NEXT_OFFSET_LAST = 0x0
STRUCT_DATA_DCBX_APP_PROTOCOL_SELECTOR_ETHER_TYPE = 0x1
STRUCT_DATA_DCBX_APP_PROTOCOL_SELECTOR_TCP_PORT = 0x2
STRUCT_DATA_DCBX_APP_PROTOCOL_SELECTOR_UDP_PORT = 0x3
STRUCT_DATA_DCBX_APP_PROTOCOL_SELECTOR_TCP_UDP_PORT = 0x4
STRUCT_DATA_DCBX_APP_PROTOCOL_SELECTOR_LAST = STRUCT_DATA_DCBX_APP_PROTOCOL_SELECTOR_TCP_UDP_PORT
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_UNKNOWN = 0x0
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_BAD_HDR_CNT = 0x1
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_BAD_FMT = 0x2
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_BAD_ID = 0x3
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_ALREADY_ADDED = 0x4
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_INST_IN_PROG = 0x5
FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_LAST = FW_SET_STRUCTURED_DATA_CMD_ERR_CODE_INST_IN_PROG
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_UNUSED = 0x0
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_ALL = 0xffff
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_NEAR_BRIDGE_ADMIN = 0x100
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_NEAR_BRIDGE_PEER = 0x101
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_NEAR_BRIDGE_OPERATIONAL = 0x102
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_NON_TPMR_ADMIN = 0x200
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_NON_TPMR_PEER = 0x201
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_NON_TPMR_OPERATIONAL = 0x202
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_HOST_OPERATIONAL = 0x300
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_CLAIMS_SUPPORTED = 0x320
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_CLAIMS_ACTIVE = 0x321
FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_LAST = FW_GET_STRUCTURED_DATA_REQ_SUBTYPE_CLAIMS_ACTIVE
FW_GET_STRUCTURED_DATA_CMD_ERR_CODE_UNKNOWN = 0x0
FW_GET_STRUCTURED_DATA_CMD_ERR_CODE_BAD_ID = 0x3
FW_GET_STRUCTURED_DATA_CMD_ERR_CODE_LAST = FW_GET_STRUCTURED_DATA_CMD_ERR_CODE_BAD_ID
FW_LIVEPATCH_QUERY_REQ_FW_TARGET_COMMON_FW = 0x1
FW_LIVEPATCH_QUERY_REQ_FW_TARGET_SECURE_FW = 0x2
FW_LIVEPATCH_QUERY_REQ_FW_TARGET_LAST = FW_LIVEPATCH_QUERY_REQ_FW_TARGET_SECURE_FW
FW_LIVEPATCH_QUERY_RESP_STATUS_FLAGS_INSTALL = 0x1
FW_LIVEPATCH_QUERY_RESP_STATUS_FLAGS_ACTIVE = 0x2
FW_LIVEPATCH_REQ_OPCODE_ACTIVATE = 0x1
FW_LIVEPATCH_REQ_OPCODE_DEACTIVATE = 0x2
FW_LIVEPATCH_REQ_OPCODE_LAST = FW_LIVEPATCH_REQ_OPCODE_DEACTIVATE
FW_LIVEPATCH_REQ_FW_TARGET_COMMON_FW = 0x1
FW_LIVEPATCH_REQ_FW_TARGET_SECURE_FW = 0x2
FW_LIVEPATCH_REQ_FW_TARGET_LAST = FW_LIVEPATCH_REQ_FW_TARGET_SECURE_FW
FW_LIVEPATCH_REQ_LOADTYPE_NVM_INSTALL = 0x1
FW_LIVEPATCH_REQ_LOADTYPE_MEMORY_DIRECT = 0x2
FW_LIVEPATCH_REQ_LOADTYPE_LAST = FW_LIVEPATCH_REQ_LOADTYPE_MEMORY_DIRECT
FW_LIVEPATCH_CMD_ERR_CODE_UNKNOWN = 0x0
FW_LIVEPATCH_CMD_ERR_CODE_INVALID_OPCODE = 0x1
FW_LIVEPATCH_CMD_ERR_CODE_INVALID_TARGET = 0x2
FW_LIVEPATCH_CMD_ERR_CODE_NOT_SUPPORTED = 0x3
FW_LIVEPATCH_CMD_ERR_CODE_NOT_INSTALLED = 0x4
FW_LIVEPATCH_CMD_ERR_CODE_NOT_PATCHED = 0x5
FW_LIVEPATCH_CMD_ERR_CODE_AUTH_FAIL = 0x6
FW_LIVEPATCH_CMD_ERR_CODE_INVALID_HEADER = 0x7
FW_LIVEPATCH_CMD_ERR_CODE_INVALID_SIZE = 0x8
FW_LIVEPATCH_CMD_ERR_CODE_ALREADY_PATCHED = 0x9
FW_LIVEPATCH_CMD_ERR_CODE_LAST = FW_LIVEPATCH_CMD_ERR_CODE_ALREADY_PATCHED
TEMP_MONITOR_QUERY_RESP_FLAGS_TEMP_NOT_AVAILABLE = 0x1
TEMP_MONITOR_QUERY_RESP_FLAGS_PHY_TEMP_NOT_AVAILABLE = 0x2
TEMP_MONITOR_QUERY_RESP_FLAGS_OM_NOT_PRESENT = 0x4
TEMP_MONITOR_QUERY_RESP_FLAGS_OM_TEMP_NOT_AVAILABLE = 0x8
TEMP_MONITOR_QUERY_RESP_FLAGS_EXT_TEMP_FIELDS_AVAILABLE = 0x10
TEMP_MONITOR_QUERY_RESP_FLAGS_THRESHOLD_VALUES_AVAILABLE = 0x20
WOL_FILTER_ALLOC_REQ_ENABLES_MAC_ADDRESS = 0x1
WOL_FILTER_ALLOC_REQ_ENABLES_PATTERN_OFFSET = 0x2
WOL_FILTER_ALLOC_REQ_ENABLES_PATTERN_BUF_SIZE = 0x4
WOL_FILTER_ALLOC_REQ_ENABLES_PATTERN_BUF_ADDR = 0x8
WOL_FILTER_ALLOC_REQ_ENABLES_PATTERN_MASK_ADDR = 0x10
WOL_FILTER_ALLOC_REQ_ENABLES_PATTERN_MASK_SIZE = 0x20
WOL_FILTER_ALLOC_REQ_WOL_TYPE_MAGICPKT = 0x0
WOL_FILTER_ALLOC_REQ_WOL_TYPE_BMP = 0x1
WOL_FILTER_ALLOC_REQ_WOL_TYPE_INVALID = 0xff
WOL_FILTER_ALLOC_REQ_WOL_TYPE_LAST = WOL_FILTER_ALLOC_REQ_WOL_TYPE_INVALID
WOL_FILTER_FREE_REQ_FLAGS_FREE_ALL_WOL_FILTERS = 0x1
WOL_FILTER_FREE_REQ_ENABLES_WOL_FILTER_ID = 0x1
WOL_FILTER_QCFG_RESP_WOL_TYPE_MAGICPKT = 0x0
WOL_FILTER_QCFG_RESP_WOL_TYPE_BMP = 0x1
WOL_FILTER_QCFG_RESP_WOL_TYPE_INVALID = 0xff
WOL_FILTER_QCFG_RESP_WOL_TYPE_LAST = WOL_FILTER_QCFG_RESP_WOL_TYPE_INVALID
WOL_REASON_QCFG_RESP_WOL_REASON_MAGICPKT = 0x0
WOL_REASON_QCFG_RESP_WOL_REASON_BMP = 0x1
WOL_REASON_QCFG_RESP_WOL_REASON_INVALID = 0xff
WOL_REASON_QCFG_RESP_WOL_REASON_LAST = WOL_REASON_QCFG_RESP_WOL_REASON_INVALID
DBG_QCAPS_RESP_COREDUMP_COMPONENT_DISABLE_CAPS_NVRAM = 0x1
DBG_QCAPS_RESP_FLAGS_CRASHDUMP_NVM = 0x1
DBG_QCAPS_RESP_FLAGS_CRASHDUMP_HOST_DDR = 0x2
DBG_QCAPS_RESP_FLAGS_CRASHDUMP_SOC_DDR = 0x4
DBG_QCAPS_RESP_FLAGS_USEQ = 0x8
DBG_QCAPS_RESP_FLAGS_COREDUMP_HOST_DDR = 0x10
DBG_QCAPS_RESP_FLAGS_COREDUMP_HOST_CAPTURE = 0x20
DBG_QCAPS_RESP_FLAGS_PTRACE = 0x40
DBG_QCAPS_RESP_FLAGS_REG_ACCESS_RESTRICTED = 0x80
DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_MASK = 0x3
DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_SFT = 0
DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_DEST_NVM = 0x0
DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_DEST_HOST_DDR = 0x1
DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_DEST_SOC_DDR = 0x2
DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_LAST = DBG_QCFG_REQ_FLAGS_CRASHDUMP_SIZE_FOR_DEST_DEST_SOC_DDR
DBG_QCFG_REQ_COREDUMP_COMPONENT_DISABLE_FLAGS_NVRAM = 0x1
DBG_QCFG_RESP_FLAGS_UART_LOG = 0x1
DBG_QCFG_RESP_FLAGS_UART_LOG_SECONDARY = 0x2
DBG_QCFG_RESP_FLAGS_FW_TRACE = 0x4
DBG_QCFG_RESP_FLAGS_FW_TRACE_SECONDARY = 0x8
DBG_QCFG_RESP_FLAGS_DEBUG_NOTIFY = 0x10
DBG_QCFG_RESP_FLAGS_JTAG_DEBUG = 0x20
DBG_CRASHDUMP_MEDIUM_CFG_REQ_TYPE_DDR = 0x1
DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_MASK = 0x3
DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_SFT = 0
DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_LVL_0 = 0x0
DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_LVL_1 = 0x1
DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_LVL_2 = 0x2
DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_LAST = DBG_CRASHDUMP_MEDIUM_CFG_REQ_LVL_LVL_2
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_MASK = 0x1c
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_SFT = 2
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_4K = (0x0 << 2)
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_8K = (0x1 << 2)
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_64K = (0x2 << 2)
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_2M = (0x3 << 2)
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_8M = (0x4 << 2)
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_1G = (0x5 << 2)
DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_LAST = DBG_CRASHDUMP_MEDIUM_CFG_REQ_PG_SIZE_PG_1G
DBG_CRASHDUMP_MEDIUM_CFG_REQ_UNUSED11_MASK = 0xffe0
DBG_CRASHDUMP_MEDIUM_CFG_REQ_UNUSED11_SFT = 5
DBG_CRASHDUMP_MEDIUM_CFG_REQ_NVRAM = 0x1
SFLAG_COMPRESSED_ZLIB = 0x1
DBG_COREDUMP_LIST_REQ_FLAGS_CRASHDUMP = 0x1
DBG_COREDUMP_LIST_RESP_FLAGS_MORE = 0x1
DBG_COREDUMP_INITIATE_REQ_SEG_FLAGS_LIVE_DATA = 0x1
DBG_COREDUMP_INITIATE_REQ_SEG_FLAGS_CRASH_DATA = 0x2
DBG_COREDUMP_INITIATE_REQ_SEG_FLAGS_COLLECT_CTX_L1_CACHE = 0x4
COREDUMP_DATA_HDR_FLAGS_LENGTH_ACTUAL_LEN_MASK = 0xffffff
COREDUMP_DATA_HDR_FLAGS_LENGTH_ACTUAL_LEN_SFT = 0
COREDUMP_DATA_HDR_FLAGS_LENGTH_INDIRECT_ACCESS = 0x1000000
DBG_COREDUMP_RETRIEVE_RESP_FLAGS_MORE = 0x1
DBG_RING_INFO_GET_REQ_RING_TYPE_L2_CMPL = 0x0
DBG_RING_INFO_GET_REQ_RING_TYPE_TX = 0x1
DBG_RING_INFO_GET_REQ_RING_TYPE_RX = 0x2
DBG_RING_INFO_GET_REQ_RING_TYPE_NQ = 0x3
DBG_RING_INFO_GET_REQ_RING_TYPE_LAST = DBG_RING_INFO_GET_REQ_RING_TYPE_NQ
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_SRT_TRACE = 0x0
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_SRT2_TRACE = 0x1
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_CRT_TRACE = 0x2
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_CRT2_TRACE = 0x3
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_RIGP0_TRACE = 0x4
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_L2_HWRM_TRACE = 0x5
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_ROCE_HWRM_TRACE = 0x6
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_CA0_TRACE = 0x7
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_CA1_TRACE = 0x8
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_CA2_TRACE = 0x9
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_RIGP1_TRACE = 0xa
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_AFM_KONG_HWRM_TRACE = 0xb
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_ERR_QPC_TRACE = 0xc
DBG_LOG_BUFFER_FLUSH_REQ_TYPE_LAST = DBG_LOG_BUFFER_FLUSH_REQ_TYPE_ERR_QPC_TRACE
DBG_LOG_BUFFER_FLUSH_REQ_FLAGS_FLUSH_ALL_BUFFERS = 0x1
NVM_WRITE_REQ_FLAGS_KEEP_ORIG_ACTIVE_IMG = 0x1
NVM_WRITE_REQ_FLAGS_BATCH_MODE = 0x2
NVM_WRITE_REQ_FLAGS_BATCH_LAST = 0x4
NVM_WRITE_REQ_FLAGS_SKIP_CRID_CHECK = 0x8
NVM_WRITE_CMD_ERR_CODE_UNKNOWN = 0x0
NVM_WRITE_CMD_ERR_CODE_FRAG_ERR = 0x1
NVM_WRITE_CMD_ERR_CODE_NO_SPACE = 0x2
NVM_WRITE_CMD_ERR_CODE_WRITE_FAILED = 0x3
NVM_WRITE_CMD_ERR_CODE_REQD_ERASE_FAILED = 0x4
NVM_WRITE_CMD_ERR_CODE_VERIFY_FAILED = 0x5
NVM_WRITE_CMD_ERR_CODE_INVALID_HEADER = 0x6
NVM_WRITE_CMD_ERR_CODE_UPDATE_DIGEST_FAILED = 0x7
NVM_WRITE_CMD_ERR_CODE_LAST = NVM_WRITE_CMD_ERR_CODE_UPDATE_DIGEST_FAILED
NVM_MODIFY_REQ_FLAGS_BATCH_MODE = 0x1
NVM_MODIFY_REQ_FLAGS_BATCH_LAST = 0x2
NVM_FIND_DIR_ENTRY_REQ_ENABLES_DIR_IDX_VALID = 0x1
NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_MASK = 0x3
NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_SFT = 0
NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_EQ = 0x0
NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_GE = 0x1
NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_GT = 0x2
NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_LAST = NVM_FIND_DIR_ENTRY_REQ_OPT_ORDINAL_GT
NVM_GET_DEV_INFO_REQ_FLAGS_SECURITY_SOC_NVM = 0x1
NVM_GET_DEV_INFO_RESP_FLAGS_FW_VER_VALID = 0x1
NVM_MOD_DIR_ENTRY_REQ_ENABLES_CHECKSUM = 0x1
NVM_INSTALL_UPDATE_REQ_INSTALL_TYPE_NORMAL = 0x0
NVM_INSTALL_UPDATE_REQ_INSTALL_TYPE_ALL = 0xffffffff
NVM_INSTALL_UPDATE_REQ_INSTALL_TYPE_LAST = NVM_INSTALL_UPDATE_REQ_INSTALL_TYPE_ALL
NVM_INSTALL_UPDATE_REQ_FLAGS_ERASE_UNUSED_SPACE = 0x1
NVM_INSTALL_UPDATE_REQ_FLAGS_REMOVE_UNUSED_PKG = 0x2
NVM_INSTALL_UPDATE_REQ_FLAGS_ALLOWED_TO_DEFRAG = 0x4
NVM_INSTALL_UPDATE_REQ_FLAGS_VERIFY_ONLY = 0x8
NVM_INSTALL_UPDATE_RESP_RESULT_SUCCESS = 0x0
NVM_INSTALL_UPDATE_RESP_RESULT_FAILURE = 0xff
NVM_INSTALL_UPDATE_RESP_RESULT_MALLOC_FAILURE = 0xfd
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_INDEX_PARAMETER = 0xfb
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_TYPE_PARAMETER = 0xf3
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_PREREQUISITE = 0xf2
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_FILE_HEADER = 0xec
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_SIGNATURE = 0xeb
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_PROP_STREAM = 0xea
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_PROP_LENGTH = 0xe9
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_MANIFEST = 0xe8
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_TRAILER = 0xe7
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_CHECKSUM = 0xe6
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_ITEM_CHECKSUM = 0xe5
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_DATA_LENGTH = 0xe4
NVM_INSTALL_UPDATE_RESP_RESULT_INVALID_DIRECTIVE = 0xe1
NVM_INSTALL_UPDATE_RESP_RESULT_UNSUPPORTED_CHIP_REV = 0xce
NVM_INSTALL_UPDATE_RESP_RESULT_UNSUPPORTED_DEVICE_ID = 0xcd
NVM_INSTALL_UPDATE_RESP_RESULT_UNSUPPORTED_SUBSYS_VENDOR = 0xcc
NVM_INSTALL_UPDATE_RESP_RESULT_UNSUPPORTED_SUBSYS_ID = 0xcb
NVM_INSTALL_UPDATE_RESP_RESULT_UNSUPPORTED_PLATFORM = 0xc5
NVM_INSTALL_UPDATE_RESP_RESULT_DUPLICATE_ITEM = 0xc4
NVM_INSTALL_UPDATE_RESP_RESULT_ZERO_LENGTH_ITEM = 0xc3
NVM_INSTALL_UPDATE_RESP_RESULT_INSTALL_CHECKSUM_ERROR = 0xb9
NVM_INSTALL_UPDATE_RESP_RESULT_INSTALL_DATA_ERROR = 0xb8
NVM_INSTALL_UPDATE_RESP_RESULT_INSTALL_AUTHENTICATION_ERROR = 0xb7
NVM_INSTALL_UPDATE_RESP_RESULT_ITEM_NOT_FOUND = 0xb0
NVM_INSTALL_UPDATE_RESP_RESULT_ITEM_LOCKED = 0xa7
NVM_INSTALL_UPDATE_RESP_RESULT_LAST = NVM_INSTALL_UPDATE_RESP_RESULT_ITEM_LOCKED
NVM_INSTALL_UPDATE_RESP_PROBLEM_ITEM_NONE = 0x0
NVM_INSTALL_UPDATE_RESP_PROBLEM_ITEM_PACKAGE = 0xff
NVM_INSTALL_UPDATE_RESP_PROBLEM_ITEM_LAST = NVM_INSTALL_UPDATE_RESP_PROBLEM_ITEM_PACKAGE
NVM_INSTALL_UPDATE_RESP_RESET_REQUIRED_NONE = 0x0
NVM_INSTALL_UPDATE_RESP_RESET_REQUIRED_PCI = 0x1
NVM_INSTALL_UPDATE_RESP_RESET_REQUIRED_POWER = 0x2
NVM_INSTALL_UPDATE_RESP_RESET_REQUIRED_LAST = NVM_INSTALL_UPDATE_RESP_RESET_REQUIRED_POWER
NVM_INSTALL_UPDATE_CMD_ERR_CODE_UNKNOWN = 0x0
NVM_INSTALL_UPDATE_CMD_ERR_CODE_FRAG_ERR = 0x1
NVM_INSTALL_UPDATE_CMD_ERR_CODE_NO_SPACE = 0x2
NVM_INSTALL_UPDATE_CMD_ERR_CODE_ANTI_ROLLBACK = 0x3
NVM_INSTALL_UPDATE_CMD_ERR_CODE_NO_VOLTREG_SUPPORT = 0x4
NVM_INSTALL_UPDATE_CMD_ERR_CODE_DEFRAG_FAILED = 0x5
NVM_INSTALL_UPDATE_CMD_ERR_CODE_UNKNOWN_DIR_ERR = 0x6
NVM_INSTALL_UPDATE_CMD_ERR_CODE_LAST = NVM_INSTALL_UPDATE_CMD_ERR_CODE_UNKNOWN_DIR_ERR
NVM_GET_VARIABLE_REQ_OPTION_NUM_RSVD_0 = 0x0
NVM_GET_VARIABLE_REQ_OPTION_NUM_RSVD_FFFF = 0xffff
NVM_GET_VARIABLE_REQ_OPTION_NUM_LAST = NVM_GET_VARIABLE_REQ_OPTION_NUM_RSVD_FFFF
NVM_GET_VARIABLE_REQ_FLAGS_FACTORY_DFLT = 0x1
NVM_GET_VARIABLE_REQ_FLAGS_VALIDATE_OPT_VALUE = 0x2
NVM_GET_VARIABLE_RESP_OPTION_NUM_RSVD_0 = 0x0
NVM_GET_VARIABLE_RESP_OPTION_NUM_RSVD_FFFF = 0xffff
NVM_GET_VARIABLE_RESP_OPTION_NUM_LAST = NVM_GET_VARIABLE_RESP_OPTION_NUM_RSVD_FFFF
NVM_GET_VARIABLE_RESP_FLAGS_VALIDATE_OPT_VALUE = 0x1
NVM_GET_VARIABLE_CMD_ERR_CODE_UNKNOWN = 0x0
NVM_GET_VARIABLE_CMD_ERR_CODE_VAR_NOT_EXIST = 0x1
NVM_GET_VARIABLE_CMD_ERR_CODE_CORRUPT_VAR = 0x2
NVM_GET_VARIABLE_CMD_ERR_CODE_LEN_TOO_SHORT = 0x3
NVM_GET_VARIABLE_CMD_ERR_CODE_INDEX_INVALID = 0x4
NVM_GET_VARIABLE_CMD_ERR_CODE_ACCESS_DENIED = 0x5
NVM_GET_VARIABLE_CMD_ERR_CODE_CB_FAILED = 0x6
NVM_GET_VARIABLE_CMD_ERR_CODE_INVALID_DATA_LEN = 0x7
NVM_GET_VARIABLE_CMD_ERR_CODE_NO_MEM = 0x8
NVM_GET_VARIABLE_CMD_ERR_CODE_LAST = NVM_GET_VARIABLE_CMD_ERR_CODE_NO_MEM
NVM_SET_VARIABLE_REQ_OPTION_NUM_RSVD_0 = 0x0
NVM_SET_VARIABLE_REQ_OPTION_NUM_RSVD_FFFF = 0xffff
NVM_SET_VARIABLE_REQ_OPTION_NUM_LAST = NVM_SET_VARIABLE_REQ_OPTION_NUM_RSVD_FFFF
NVM_SET_VARIABLE_REQ_FLAGS_FORCE_FLUSH = 0x1
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_MASK = 0xe
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_SFT = 1
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_NONE = (0x0 << 1)
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_HMAC_SHA1 = (0x1 << 1)
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_AES256 = (0x2 << 1)
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_HMAC_SHA1_AUTH = (0x3 << 1)
NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_LAST = NVM_SET_VARIABLE_REQ_FLAGS_ENCRYPT_MODE_HMAC_SHA1_AUTH
NVM_SET_VARIABLE_REQ_FLAGS_FLAGS_UNUSED_0_MASK = 0x70
NVM_SET_VARIABLE_REQ_FLAGS_FLAGS_UNUSED_0_SFT = 4
NVM_SET_VARIABLE_REQ_FLAGS_FACTORY_DEFAULT = 0x80
NVM_SET_VARIABLE_CMD_ERR_CODE_UNKNOWN = 0x0
NVM_SET_VARIABLE_CMD_ERR_CODE_VAR_NOT_EXIST = 0x1
NVM_SET_VARIABLE_CMD_ERR_CODE_CORRUPT_VAR = 0x2
NVM_SET_VARIABLE_CMD_ERR_CODE_LEN_TOO_SHORT = 0x3
NVM_SET_VARIABLE_CMD_ERR_CODE_ACTION_NOT_SUPPORTED = 0x4
NVM_SET_VARIABLE_CMD_ERR_CODE_INDEX_INVALID = 0x5
NVM_SET_VARIABLE_CMD_ERR_CODE_ACCESS_DENIED = 0x6
NVM_SET_VARIABLE_CMD_ERR_CODE_CB_FAILED = 0x7
NVM_SET_VARIABLE_CMD_ERR_CODE_INVALID_DATA_LEN = 0x8
NVM_SET_VARIABLE_CMD_ERR_CODE_NO_MEM = 0x9
NVM_SET_VARIABLE_CMD_ERR_CODE_LAST = NVM_SET_VARIABLE_CMD_ERR_CODE_NO_MEM
SELFTEST_QLIST_RESP_AVAILABLE_TESTS_NVM_TEST = 0x1
SELFTEST_QLIST_RESP_AVAILABLE_TESTS_LINK_TEST = 0x2
SELFTEST_QLIST_RESP_AVAILABLE_TESTS_REGISTER_TEST = 0x4
SELFTEST_QLIST_RESP_AVAILABLE_TESTS_MEMORY_TEST = 0x8
SELFTEST_QLIST_RESP_AVAILABLE_TESTS_PCIE_SERDES_TEST = 0x10
SELFTEST_QLIST_RESP_AVAILABLE_TESTS_ETHERNET_SERDES_TEST = 0x20
SELFTEST_QLIST_RESP_OFFLINE_TESTS_NVM_TEST = 0x1
SELFTEST_QLIST_RESP_OFFLINE_TESTS_LINK_TEST = 0x2
SELFTEST_QLIST_RESP_OFFLINE_TESTS_REGISTER_TEST = 0x4
SELFTEST_QLIST_RESP_OFFLINE_TESTS_MEMORY_TEST = 0x8
SELFTEST_QLIST_RESP_OFFLINE_TESTS_PCIE_SERDES_TEST = 0x10
SELFTEST_QLIST_RESP_OFFLINE_TESTS_ETHERNET_SERDES_TEST = 0x20
SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_BER_1E8_SUPPORTED = 0x0
SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_BER_1E9_SUPPORTED = 0x1
SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_BER_1E10_SUPPORTED = 0x2
SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_BER_1E11_SUPPORTED = 0x3
SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_BER_1E12_SUPPORTED = 0x4
SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_LAST = SELFTEST_QLIST_RESP_EYESCOPE_TARGET_BER_SUPPORT_BER_1E12_SUPPORTED
SELFTEST_EXEC_REQ_FLAGS_NVM_TEST = 0x1
SELFTEST_EXEC_REQ_FLAGS_LINK_TEST = 0x2
SELFTEST_EXEC_REQ_FLAGS_REGISTER_TEST = 0x4
SELFTEST_EXEC_REQ_FLAGS_MEMORY_TEST = 0x8
SELFTEST_EXEC_REQ_FLAGS_PCIE_SERDES_TEST = 0x10
SELFTEST_EXEC_REQ_FLAGS_ETHERNET_SERDES_TEST = 0x20
SELFTEST_EXEC_RESP_REQUESTED_TESTS_NVM_TEST = 0x1
SELFTEST_EXEC_RESP_REQUESTED_TESTS_LINK_TEST = 0x2
SELFTEST_EXEC_RESP_REQUESTED_TESTS_REGISTER_TEST = 0x4
SELFTEST_EXEC_RESP_REQUESTED_TESTS_MEMORY_TEST = 0x8
SELFTEST_EXEC_RESP_REQUESTED_TESTS_PCIE_SERDES_TEST = 0x10
SELFTEST_EXEC_RESP_REQUESTED_TESTS_ETHERNET_SERDES_TEST = 0x20
SELFTEST_EXEC_RESP_TEST_SUCCESS_NVM_TEST = 0x1
SELFTEST_EXEC_RESP_TEST_SUCCESS_LINK_TEST = 0x2
SELFTEST_EXEC_RESP_TEST_SUCCESS_REGISTER_TEST = 0x4
SELFTEST_EXEC_RESP_TEST_SUCCESS_MEMORY_TEST = 0x8
SELFTEST_EXEC_RESP_TEST_SUCCESS_PCIE_SERDES_TEST = 0x10
SELFTEST_EXEC_RESP_TEST_SUCCESS_ETHERNET_SERDES_TEST = 0x20
DBC_DBC_INDEX_MASK = 0xffffff
DBC_DBC_INDEX_SFT = 0
DBC_DBC_EPOCH = 0x1000000
DBC_DBC_TOGGLE_MASK = 0x6000000
DBC_DBC_TOGGLE_SFT = 25
DBC_DBC_XID_MASK = 0xfffff
DBC_DBC_XID_SFT = 0
DBC_DBC_PATH_MASK = 0x3000000
DBC_DBC_PATH_SFT = 24
DBC_DBC_PATH_ROCE = (0x0 << 24)
DBC_DBC_PATH_L2 = (0x1 << 24)
DBC_DBC_PATH_ENGINE = (0x2 << 24)
DBC_DBC_PATH_LAST = DBC_DBC_PATH_ENGINE
DBC_DBC_VALID = 0x4000000
DBC_DBC_DEBUG_TRACE = 0x8000000
DBC_DBC_TYPE_MASK = 0xf0000000
DBC_DBC_TYPE_SFT = 28
DBC_DBC_TYPE_SQ = (0x0 << 28)
DBC_DBC_TYPE_RQ = (0x1 << 28)
DBC_DBC_TYPE_SRQ = (0x2 << 28)
DBC_DBC_TYPE_SRQ_ARM = (0x3 << 28)
DBC_DBC_TYPE_CQ = (0x4 << 28)
DBC_DBC_TYPE_CQ_ARMSE = (0x5 << 28)
DBC_DBC_TYPE_CQ_ARMALL = (0x6 << 28)
DBC_DBC_TYPE_CQ_ARMENA = (0x7 << 28)
DBC_DBC_TYPE_SRQ_ARMENA = (0x8 << 28)
DBC_DBC_TYPE_CQ_CUTOFF_ACK = (0x9 << 28)
DBC_DBC_TYPE_NQ = (0xa << 28)
DBC_DBC_TYPE_NQ_ARM = (0xb << 28)
DBC_DBC_TYPE_NQ_MASK = (0xe << 28)
DBC_DBC_TYPE_NULL = (0xf << 28)
DBC_DBC_TYPE_LAST = DBC_DBC_TYPE_NULL
DB_PUSH_START_DB_INDEX_MASK = 0xffffff
DB_PUSH_START_DB_INDEX_SFT = 0
DB_PUSH_START_DB_PI_LO_MASK = 0xff000000
DB_PUSH_START_DB_PI_LO_SFT = 24
DB_PUSH_START_DB_XID_MASK = 0xfffff00000000
DB_PUSH_START_DB_XID_SFT = 32
DB_PUSH_START_DB_PI_HI_MASK = 0xf0000000000000
DB_PUSH_START_DB_PI_HI_SFT = 52
DB_PUSH_START_DB_TYPE_MASK = 0xf000000000000000
DB_PUSH_START_DB_TYPE_SFT = 60
DB_PUSH_START_DB_TYPE_PUSH_START = (0xc << 60)
DB_PUSH_START_DB_TYPE_PUSH_END = (0xd << 60)
DB_PUSH_START_DB_TYPE_LAST = DB_PUSH_START_DB_TYPE_PUSH_END
DB_PUSH_END_DB_INDEX_MASK = 0xffffff
DB_PUSH_END_DB_INDEX_SFT = 0
DB_PUSH_END_DB_PI_LO_MASK = 0xff000000
DB_PUSH_END_DB_PI_LO_SFT = 24
DB_PUSH_END_DB_XID_MASK = 0xfffff00000000
DB_PUSH_END_DB_XID_SFT = 32
DB_PUSH_END_DB_PI_HI_MASK = 0xf0000000000000
DB_PUSH_END_DB_PI_HI_SFT = 52
DB_PUSH_END_DB_PATH_MASK = 0x300000000000000
DB_PUSH_END_DB_PATH_SFT = 56
DB_PUSH_END_DB_PATH_ROCE = (0x0 << 56)
DB_PUSH_END_DB_PATH_L2 = (0x1 << 56)
DB_PUSH_END_DB_PATH_ENGINE = (0x2 << 56)
DB_PUSH_END_DB_PATH_LAST = DB_PUSH_END_DB_PATH_ENGINE
DB_PUSH_END_DB_DEBUG_TRACE = 0x800000000000000
DB_PUSH_END_DB_TYPE_MASK = 0xf000000000000000
DB_PUSH_END_DB_TYPE_SFT = 60
DB_PUSH_END_DB_TYPE_PUSH_START = (0xc << 60)
DB_PUSH_END_DB_TYPE_PUSH_END = (0xd << 60)
DB_PUSH_END_DB_TYPE_LAST = DB_PUSH_END_DB_TYPE_PUSH_END
DB_PUSH_INFO_PUSH_INDEX_MASK = 0xffffff
DB_PUSH_INFO_PUSH_INDEX_SFT = 0
DB_PUSH_INFO_PUSH_SIZE_MASK = 0x1f000000
DB_PUSH_INFO_PUSH_SIZE_SFT = 24
FW_STATUS_REG_CODE_MASK = 0xffff
FW_STATUS_REG_CODE_SFT = 0
FW_STATUS_REG_CODE_READY = 0x8000
FW_STATUS_REG_CODE_LAST = FW_STATUS_REG_CODE_READY
FW_STATUS_REG_IMAGE_DEGRADED = 0x10000
FW_STATUS_REG_RECOVERABLE = 0x20000
FW_STATUS_REG_CRASHDUMP_ONGOING = 0x40000
FW_STATUS_REG_CRASHDUMP_COMPLETE = 0x80000
FW_STATUS_REG_SHUTDOWN = 0x100000
FW_STATUS_REG_CRASHED_NO_MASTER = 0x200000
FW_STATUS_REG_RECOVERING = 0x400000
FW_STATUS_REG_MANU_DEBUG_STATUS = 0x800000
HCOMM_STATUS_VER_MASK = 0xff
HCOMM_STATUS_VER_SFT = 0
HCOMM_STATUS_VER_LATEST = 0x1
HCOMM_STATUS_VER_LAST = HCOMM_STATUS_VER_LATEST
HCOMM_STATUS_SIGNATURE_MASK = 0xffffff00
HCOMM_STATUS_SIGNATURE_SFT = 8
HCOMM_STATUS_SIGNATURE_VAL = (0x484353 << 8)
HCOMM_STATUS_SIGNATURE_LAST = HCOMM_STATUS_SIGNATURE_VAL
HCOMM_STATUS_TRUE_ADDR_SPACE_MASK = 0x3
HCOMM_STATUS_TRUE_ADDR_SPACE_SFT = 0
HCOMM_STATUS_FW_STATUS_LOC_ADDR_SPACE_PCIE_CFG = 0x0
HCOMM_STATUS_FW_STATUS_LOC_ADDR_SPACE_GRC = 0x1
HCOMM_STATUS_FW_STATUS_LOC_ADDR_SPACE_BAR0 = 0x2
HCOMM_STATUS_FW_STATUS_LOC_ADDR_SPACE_BAR1 = 0x3
HCOMM_STATUS_FW_STATUS_LOC_ADDR_SPACE_LAST = HCOMM_STATUS_FW_STATUS_LOC_ADDR_SPACE_BAR1
HCOMM_STATUS_TRUE_OFFSET_MASK = 0xfffffffc
HCOMM_STATUS_TRUE_OFFSET_SFT = 2
HCOMM_STATUS_STRUCT_LOC = 0x31001F0
TX_DOORBELL_IDX_MASK = 0xffffff
TX_DOORBELL_IDX_SFT = 0
TX_DOORBELL_KEY_MASK = 0xf0000000
TX_DOORBELL_KEY_SFT = 28
TX_DOORBELL_KEY_TX = (0x0 << 28)
TX_DOORBELL_KEY_LAST = TX_DOORBELL_KEY_TX
RX_DOORBELL_IDX_MASK = 0xffffff
RX_DOORBELL_IDX_SFT = 0
RX_DOORBELL_KEY_MASK = 0xf0000000
RX_DOORBELL_KEY_SFT = 28
RX_DOORBELL_KEY_RX = (0x1 << 28)
RX_DOORBELL_KEY_LAST = RX_DOORBELL_KEY_RX
CMPL_DOORBELL_IDX_MASK = 0xffffff
CMPL_DOORBELL_IDX_SFT = 0
CMPL_DOORBELL_IDX_VALID = 0x4000000
CMPL_DOORBELL_MASK = 0x8000000
CMPL_DOORBELL_KEY_MASK = 0xf0000000
CMPL_DOORBELL_KEY_SFT = 28
CMPL_DOORBELL_KEY_CMPL = (0x2 << 28)
CMPL_DOORBELL_KEY_LAST = CMPL_DOORBELL_KEY_CMPL
STATUS_DOORBELL_IDX_MASK = 0xffffff
STATUS_DOORBELL_IDX_SFT = 0
STATUS_DOORBELL_KEY_MASK = 0xf0000000
STATUS_DOORBELL_KEY_SFT = 28
STATUS_DOORBELL_KEY_STAT = (0x3 << 28)
STATUS_DOORBELL_KEY_LAST = STATUS_DOORBELL_KEY_STAT
CMDQ_INIT_CMDQ_LVL_MASK = 0x3
CMDQ_INIT_CMDQ_LVL_SFT = 0
CMDQ_INIT_CMDQ_SIZE_MASK = 0xfffc
CMDQ_INIT_CMDQ_SIZE_SFT = 2
CMDQ_BASE_OPCODE_CREATE_QP = 0x1
CMDQ_BASE_OPCODE_DESTROY_QP = 0x2
CMDQ_BASE_OPCODE_MODIFY_QP = 0x3
CMDQ_BASE_OPCODE_QUERY_QP = 0x4
CMDQ_BASE_OPCODE_CREATE_SRQ = 0x5
CMDQ_BASE_OPCODE_DESTROY_SRQ = 0x6
CMDQ_BASE_OPCODE_QUERY_SRQ = 0x8
CMDQ_BASE_OPCODE_CREATE_CQ = 0x9
CMDQ_BASE_OPCODE_DESTROY_CQ = 0xa
CMDQ_BASE_OPCODE_RESIZE_CQ = 0xc
CMDQ_BASE_OPCODE_ALLOCATE_MRW = 0xd
CMDQ_BASE_OPCODE_DEALLOCATE_KEY = 0xe
CMDQ_BASE_OPCODE_REGISTER_MR = 0xf
CMDQ_BASE_OPCODE_DEREGISTER_MR = 0x10
CMDQ_BASE_OPCODE_ADD_GID = 0x11
CMDQ_BASE_OPCODE_DELETE_GID = 0x12
CMDQ_BASE_OPCODE_MODIFY_GID = 0x17
CMDQ_BASE_OPCODE_QUERY_GID = 0x18
CMDQ_BASE_OPCODE_CREATE_QP1 = 0x13
CMDQ_BASE_OPCODE_DESTROY_QP1 = 0x14
CMDQ_BASE_OPCODE_CREATE_AH = 0x15
CMDQ_BASE_OPCODE_DESTROY_AH = 0x16
CMDQ_BASE_OPCODE_INITIALIZE_FW = 0x80
CMDQ_BASE_OPCODE_DEINITIALIZE_FW = 0x81
CMDQ_BASE_OPCODE_STOP_FUNC = 0x82
CMDQ_BASE_OPCODE_QUERY_FUNC = 0x83
CMDQ_BASE_OPCODE_SET_FUNC_RESOURCES = 0x84
CMDQ_BASE_OPCODE_READ_CONTEXT = 0x85
CMDQ_BASE_OPCODE_VF_BACKCHANNEL_REQUEST = 0x86
CMDQ_BASE_OPCODE_READ_VF_MEMORY = 0x87
CMDQ_BASE_OPCODE_COMPLETE_VF_REQUEST = 0x88
CMDQ_BASE_OPCODE_EXTEND_CONTEXT_ARRRAY = 0x89
CMDQ_BASE_OPCODE_MAP_TC_TO_COS = 0x8a
CMDQ_BASE_OPCODE_QUERY_VERSION = 0x8b
CMDQ_BASE_OPCODE_MODIFY_ROCE_CC = 0x8c
CMDQ_BASE_OPCODE_QUERY_ROCE_CC = 0x8d
CMDQ_BASE_OPCODE_QUERY_ROCE_STATS = 0x8e
CMDQ_BASE_OPCODE_SET_LINK_AGGR_MODE = 0x8f
CMDQ_BASE_OPCODE_MODIFY_CQ = 0x90
CMDQ_BASE_OPCODE_QUERY_QP_EXTEND = 0x91
CMDQ_BASE_OPCODE_QUERY_ROCE_STATS_EXT = 0x92
CMDQ_BASE_OPCODE_ROCE_MIRROR_CFG = 0x99
CMDQ_BASE_OPCODE_LAST = CMDQ_BASE_OPCODE_ROCE_MIRROR_CFG
CREQ_BASE_TYPE_MASK = 0x3f
CREQ_BASE_TYPE_SFT = 0
CREQ_BASE_TYPE_QP_EVENT = 0x38
CREQ_BASE_TYPE_FUNC_EVENT = 0x3a
CREQ_BASE_TYPE_LAST = CREQ_BASE_TYPE_FUNC_EVENT
CREQ_BASE_V = 0x1
CMDQ_QUERY_VERSION_OPCODE_QUERY_VERSION = 0x8b
CMDQ_QUERY_VERSION_OPCODE_LAST = CMDQ_QUERY_VERSION_OPCODE_QUERY_VERSION
CREQ_QUERY_VERSION_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_VERSION_RESP_TYPE_SFT = 0
CREQ_QUERY_VERSION_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_VERSION_RESP_TYPE_LAST = CREQ_QUERY_VERSION_RESP_TYPE_QP_EVENT
CREQ_QUERY_VERSION_RESP_V = 0x1
CREQ_QUERY_VERSION_RESP_EVENT_QUERY_VERSION = 0x8b
CREQ_QUERY_VERSION_RESP_EVENT_LAST = CREQ_QUERY_VERSION_RESP_EVENT_QUERY_VERSION
CMDQ_INITIALIZE_FW_OPCODE_INITIALIZE_FW = 0x80
CMDQ_INITIALIZE_FW_OPCODE_LAST = CMDQ_INITIALIZE_FW_OPCODE_INITIALIZE_FW
CMDQ_INITIALIZE_FW_FLAGS_MRAV_RESERVATION_SPLIT = 0x1
CMDQ_INITIALIZE_FW_FLAGS_HW_REQUESTER_RETX_SUPPORTED = 0x2
CMDQ_INITIALIZE_FW_FLAGS_OPTIMIZE_MODIFY_QP_SUPPORTED = 0x8
CMDQ_INITIALIZE_FW_FLAGS_L2_VF_RESOURCE_MGMT = 0x10
CMDQ_INITIALIZE_FW_FLAGS_MIRROR_ON_ROCE_SUPPORTED = 0x80
CMDQ_INITIALIZE_FW_QPC_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_QPC_LVL_SFT = 0
CMDQ_INITIALIZE_FW_QPC_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_QPC_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_QPC_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_QPC_LVL_LAST = CMDQ_INITIALIZE_FW_QPC_LVL_LVL_2
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_QPC_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_QPC_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_MRW_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_MRW_LVL_SFT = 0
CMDQ_INITIALIZE_FW_MRW_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_MRW_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_MRW_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_MRW_LVL_LAST = CMDQ_INITIALIZE_FW_MRW_LVL_LVL_2
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_MRW_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_MRW_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_SRQ_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_SRQ_LVL_SFT = 0
CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_SRQ_LVL_LAST = CMDQ_INITIALIZE_FW_SRQ_LVL_LVL_2
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_SRQ_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_CQ_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_CQ_LVL_SFT = 0
CMDQ_INITIALIZE_FW_CQ_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_CQ_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_CQ_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_CQ_LVL_LAST = CMDQ_INITIALIZE_FW_CQ_LVL_LVL_2
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_CQ_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_CQ_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_TQM_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_TQM_LVL_SFT = 0
CMDQ_INITIALIZE_FW_TQM_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_TQM_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_TQM_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_TQM_LVL_LAST = CMDQ_INITIALIZE_FW_TQM_LVL_LVL_2
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_TQM_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_TQM_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_TIM_LVL_MASK = 0xf
CMDQ_INITIALIZE_FW_TIM_LVL_SFT = 0
CMDQ_INITIALIZE_FW_TIM_LVL_LVL_0 = 0x0
CMDQ_INITIALIZE_FW_TIM_LVL_LVL_1 = 0x1
CMDQ_INITIALIZE_FW_TIM_LVL_LVL_2 = 0x2
CMDQ_INITIALIZE_FW_TIM_LVL_LAST = CMDQ_INITIALIZE_FW_TIM_LVL_LVL_2
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_MASK = 0xf0
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_SFT = 4
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_INITIALIZE_FW_TIM_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_TIM_PG_SIZE_PG_1G
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_MASK = 0xf
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_SFT = 0
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_4K = 0x0
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_8K = 0x1
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_16K = 0x2
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_32K = 0x3
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_64K = 0x4
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_128K = 0x5
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_256K = 0x6
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_512K = 0x7
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_1M = 0x8
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_2M = 0x9
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_4M = 0xa
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_8M = 0xb
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_16M = 0xc
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_32M = 0xd
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_64M = 0xe
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_128M = 0xf
CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_LAST = CMDQ_INITIALIZE_FW_LOG2_DBR_PG_SIZE_PG_128M
CMDQ_INITIALIZE_FW_RSVD_MASK = 0xfff0
CMDQ_INITIALIZE_FW_RSVD_SFT = 4
CREQ_INITIALIZE_FW_RESP_TYPE_MASK = 0x3f
CREQ_INITIALIZE_FW_RESP_TYPE_SFT = 0
CREQ_INITIALIZE_FW_RESP_TYPE_QP_EVENT = 0x38
CREQ_INITIALIZE_FW_RESP_TYPE_LAST = CREQ_INITIALIZE_FW_RESP_TYPE_QP_EVENT
CREQ_INITIALIZE_FW_RESP_V = 0x1
CREQ_INITIALIZE_FW_RESP_EVENT_INITIALIZE_FW = 0x80
CREQ_INITIALIZE_FW_RESP_EVENT_LAST = CREQ_INITIALIZE_FW_RESP_EVENT_INITIALIZE_FW
CMDQ_DEINITIALIZE_FW_OPCODE_DEINITIALIZE_FW = 0x81
CMDQ_DEINITIALIZE_FW_OPCODE_LAST = CMDQ_DEINITIALIZE_FW_OPCODE_DEINITIALIZE_FW
CREQ_DEINITIALIZE_FW_RESP_TYPE_MASK = 0x3f
CREQ_DEINITIALIZE_FW_RESP_TYPE_SFT = 0
CREQ_DEINITIALIZE_FW_RESP_TYPE_QP_EVENT = 0x38
CREQ_DEINITIALIZE_FW_RESP_TYPE_LAST = CREQ_DEINITIALIZE_FW_RESP_TYPE_QP_EVENT
CREQ_DEINITIALIZE_FW_RESP_V = 0x1
CREQ_DEINITIALIZE_FW_RESP_EVENT_DEINITIALIZE_FW = 0x81
CREQ_DEINITIALIZE_FW_RESP_EVENT_LAST = CREQ_DEINITIALIZE_FW_RESP_EVENT_DEINITIALIZE_FW
CMDQ_CREATE_QP_OPCODE_CREATE_QP = 0x1
CMDQ_CREATE_QP_OPCODE_LAST = CMDQ_CREATE_QP_OPCODE_CREATE_QP
CMDQ_CREATE_QP_QP_FLAGS_SRQ_USED = 0x1
CMDQ_CREATE_QP_QP_FLAGS_FORCE_COMPLETION = 0x2
CMDQ_CREATE_QP_QP_FLAGS_RESERVED_LKEY_ENABLE = 0x4
CMDQ_CREATE_QP_QP_FLAGS_FR_PMR_ENABLED = 0x8
CMDQ_CREATE_QP_QP_FLAGS_VARIABLE_SIZED_WQE_ENABLED = 0x10
CMDQ_CREATE_QP_QP_FLAGS_OPTIMIZED_TRANSMIT_ENABLED = 0x20
CMDQ_CREATE_QP_QP_FLAGS_RESPONDER_UD_CQE_WITH_CFA = 0x40
CMDQ_CREATE_QP_QP_FLAGS_EXT_STATS_ENABLED = 0x80
CMDQ_CREATE_QP_QP_FLAGS_EXPRESS_MODE_ENABLED = 0x100
CMDQ_CREATE_QP_QP_FLAGS_STEERING_TAG_VALID = 0x200
CMDQ_CREATE_QP_QP_FLAGS_RDMA_READ_OR_ATOMICS_USED = 0x400
CMDQ_CREATE_QP_QP_FLAGS_LAST = CMDQ_CREATE_QP_QP_FLAGS_RDMA_READ_OR_ATOMICS_USED
CMDQ_CREATE_QP_TYPE_RC = 0x2
CMDQ_CREATE_QP_TYPE_UD = 0x4
CMDQ_CREATE_QP_TYPE_RAW_ETHERTYPE = 0x6
CMDQ_CREATE_QP_TYPE_GSI = 0x7
CMDQ_CREATE_QP_TYPE_LAST = CMDQ_CREATE_QP_TYPE_GSI
CMDQ_CREATE_QP_SQ_LVL_MASK = 0xf
CMDQ_CREATE_QP_SQ_LVL_SFT = 0
CMDQ_CREATE_QP_SQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP_SQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP_SQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP_SQ_LVL_LAST = CMDQ_CREATE_QP_SQ_LVL_LVL_2
CMDQ_CREATE_QP_SQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP_SQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP_SQ_PG_SIZE_LAST = CMDQ_CREATE_QP_SQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP_RQ_LVL_MASK = 0xf
CMDQ_CREATE_QP_RQ_LVL_SFT = 0
CMDQ_CREATE_QP_RQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP_RQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP_RQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP_RQ_LVL_LAST = CMDQ_CREATE_QP_RQ_LVL_LVL_2
CMDQ_CREATE_QP_RQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP_RQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP_RQ_PG_SIZE_LAST = CMDQ_CREATE_QP_RQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP_SQ_SGE_MASK = 0xf
CMDQ_CREATE_QP_SQ_SGE_SFT = 0
CMDQ_CREATE_QP_SQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP_SQ_FWO_SFT = 4
CMDQ_CREATE_QP_RQ_SGE_MASK = 0xf
CMDQ_CREATE_QP_RQ_SGE_SFT = 0
CMDQ_CREATE_QP_RQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP_RQ_FWO_SFT = 4
CREQ_CREATE_QP_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_QP_RESP_TYPE_SFT = 0
CREQ_CREATE_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_QP_RESP_TYPE_LAST = CREQ_CREATE_QP_RESP_TYPE_QP_EVENT
CREQ_CREATE_QP_RESP_V = 0x1
CREQ_CREATE_QP_RESP_EVENT_CREATE_QP = 0x1
CREQ_CREATE_QP_RESP_EVENT_LAST = CREQ_CREATE_QP_RESP_EVENT_CREATE_QP
CMDQ_DESTROY_QP_OPCODE_DESTROY_QP = 0x2
CMDQ_DESTROY_QP_OPCODE_LAST = CMDQ_DESTROY_QP_OPCODE_DESTROY_QP
CREQ_DESTROY_QP_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_QP_RESP_TYPE_SFT = 0
CREQ_DESTROY_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_QP_RESP_TYPE_LAST = CREQ_DESTROY_QP_RESP_TYPE_QP_EVENT
CREQ_DESTROY_QP_RESP_V = 0x1
CREQ_DESTROY_QP_RESP_EVENT_DESTROY_QP = 0x2
CREQ_DESTROY_QP_RESP_EVENT_LAST = CREQ_DESTROY_QP_RESP_EVENT_DESTROY_QP
CMDQ_MODIFY_QP_OPCODE_MODIFY_QP = 0x3
CMDQ_MODIFY_QP_OPCODE_LAST = CMDQ_MODIFY_QP_OPCODE_MODIFY_QP
CMDQ_MODIFY_QP_FLAGS_SRQ_USED = 0x1
CMDQ_MODIFY_QP_QP_TYPE_RC = 0x2
CMDQ_MODIFY_QP_QP_TYPE_UD = 0x4
CMDQ_MODIFY_QP_QP_TYPE_RAW_ETHERTYPE = 0x6
CMDQ_MODIFY_QP_QP_TYPE_GSI = 0x7
CMDQ_MODIFY_QP_QP_TYPE_LAST = CMDQ_MODIFY_QP_QP_TYPE_GSI
CMDQ_MODIFY_QP_MODIFY_MASK_STATE = 0x1
CMDQ_MODIFY_QP_MODIFY_MASK_EN_SQD_ASYNC_NOTIFY = 0x2
CMDQ_MODIFY_QP_MODIFY_MASK_ACCESS = 0x4
CMDQ_MODIFY_QP_MODIFY_MASK_PKEY = 0x8
CMDQ_MODIFY_QP_MODIFY_MASK_QKEY = 0x10
CMDQ_MODIFY_QP_MODIFY_MASK_DGID = 0x20
CMDQ_MODIFY_QP_MODIFY_MASK_FLOW_LABEL = 0x40
CMDQ_MODIFY_QP_MODIFY_MASK_SGID_INDEX = 0x80
CMDQ_MODIFY_QP_MODIFY_MASK_HOP_LIMIT = 0x100
CMDQ_MODIFY_QP_MODIFY_MASK_TRAFFIC_CLASS = 0x200
CMDQ_MODIFY_QP_MODIFY_MASK_DEST_MAC = 0x400
CMDQ_MODIFY_QP_MODIFY_MASK_PINGPONG_PUSH_MODE = 0x800
CMDQ_MODIFY_QP_MODIFY_MASK_PATH_MTU = 0x1000
CMDQ_MODIFY_QP_MODIFY_MASK_TIMEOUT = 0x2000
CMDQ_MODIFY_QP_MODIFY_MASK_RETRY_CNT = 0x4000
CMDQ_MODIFY_QP_MODIFY_MASK_RNR_RETRY = 0x8000
CMDQ_MODIFY_QP_MODIFY_MASK_RQ_PSN = 0x10000
CMDQ_MODIFY_QP_MODIFY_MASK_MAX_RD_ATOMIC = 0x20000
CMDQ_MODIFY_QP_MODIFY_MASK_MIN_RNR_TIMER = 0x40000
CMDQ_MODIFY_QP_MODIFY_MASK_SQ_PSN = 0x80000
CMDQ_MODIFY_QP_MODIFY_MASK_MAX_DEST_RD_ATOMIC = 0x100000
CMDQ_MODIFY_QP_MODIFY_MASK_SQ_SIZE = 0x200000
CMDQ_MODIFY_QP_MODIFY_MASK_RQ_SIZE = 0x400000
CMDQ_MODIFY_QP_MODIFY_MASK_SQ_SGE = 0x800000
CMDQ_MODIFY_QP_MODIFY_MASK_RQ_SGE = 0x1000000
CMDQ_MODIFY_QP_MODIFY_MASK_MAX_INLINE_DATA = 0x2000000
CMDQ_MODIFY_QP_MODIFY_MASK_DEST_QP_ID = 0x4000000
CMDQ_MODIFY_QP_MODIFY_MASK_SRC_MAC = 0x8000000
CMDQ_MODIFY_QP_MODIFY_MASK_VLAN_ID = 0x10000000
CMDQ_MODIFY_QP_MODIFY_MASK_ENABLE_CC = 0x20000000
CMDQ_MODIFY_QP_MODIFY_MASK_TOS_ECN = 0x40000000
CMDQ_MODIFY_QP_MODIFY_MASK_TOS_DSCP = 0x80000000
CMDQ_MODIFY_QP_NEW_STATE_MASK = 0xf
CMDQ_MODIFY_QP_NEW_STATE_SFT = 0
CMDQ_MODIFY_QP_NEW_STATE_RESET = 0x0
CMDQ_MODIFY_QP_NEW_STATE_INIT = 0x1
CMDQ_MODIFY_QP_NEW_STATE_RTR = 0x2
CMDQ_MODIFY_QP_NEW_STATE_RTS = 0x3
CMDQ_MODIFY_QP_NEW_STATE_SQD = 0x4
CMDQ_MODIFY_QP_NEW_STATE_SQE = 0x5
CMDQ_MODIFY_QP_NEW_STATE_ERR = 0x6
CMDQ_MODIFY_QP_NEW_STATE_LAST = CMDQ_MODIFY_QP_NEW_STATE_ERR
CMDQ_MODIFY_QP_EN_SQD_ASYNC_NOTIFY = 0x10
CMDQ_MODIFY_QP_UNUSED1 = 0x20
CMDQ_MODIFY_QP_NETWORK_TYPE_MASK = 0xc0
CMDQ_MODIFY_QP_NETWORK_TYPE_SFT = 6
CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV1 = (0x0 << 6)
CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV4 = (0x2 << 6)
CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV6 = (0x3 << 6)
CMDQ_MODIFY_QP_NETWORK_TYPE_LAST = CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV6
CMDQ_MODIFY_QP_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_MASK = 0xff
CMDQ_MODIFY_QP_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_SFT = 0
CMDQ_MODIFY_QP_ACCESS_LOCAL_WRITE = 0x1
CMDQ_MODIFY_QP_ACCESS_REMOTE_WRITE = 0x2
CMDQ_MODIFY_QP_ACCESS_REMOTE_READ = 0x4
CMDQ_MODIFY_QP_ACCESS_REMOTE_ATOMIC = 0x8
CMDQ_MODIFY_QP_TOS_ECN_MASK = 0x3
CMDQ_MODIFY_QP_TOS_ECN_SFT = 0
CMDQ_MODIFY_QP_TOS_DSCP_MASK = 0xfc
CMDQ_MODIFY_QP_TOS_DSCP_SFT = 2
CMDQ_MODIFY_QP_PINGPONG_PUSH_ENABLE = 0x1
CMDQ_MODIFY_QP_UNUSED3_MASK = 0xe
CMDQ_MODIFY_QP_UNUSED3_SFT = 1
CMDQ_MODIFY_QP_PATH_MTU_MASK = 0xf0
CMDQ_MODIFY_QP_PATH_MTU_SFT = 4
CMDQ_MODIFY_QP_PATH_MTU_MTU_256 = (0x0 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_512 = (0x1 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_1024 = (0x2 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_2048 = (0x3 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_4096 = (0x4 << 4)
CMDQ_MODIFY_QP_PATH_MTU_MTU_8192 = (0x5 << 4)
CMDQ_MODIFY_QP_PATH_MTU_LAST = CMDQ_MODIFY_QP_PATH_MTU_MTU_8192
CMDQ_MODIFY_QP_ENABLE_CC = 0x1
CMDQ_MODIFY_QP_UNUSED15_MASK = 0xfffe
CMDQ_MODIFY_QP_UNUSED15_SFT = 1
CMDQ_MODIFY_QP_VLAN_ID_MASK = 0xfff
CMDQ_MODIFY_QP_VLAN_ID_SFT = 0
CMDQ_MODIFY_QP_VLAN_DEI = 0x1000
CMDQ_MODIFY_QP_VLAN_PCP_MASK = 0xe000
CMDQ_MODIFY_QP_VLAN_PCP_SFT = 13
CMDQ_MODIFY_QP_EXT_MODIFY_MASK_EXT_STATS_CTX = 0x1
CMDQ_MODIFY_QP_EXT_MODIFY_MASK_SCHQ_ID_VALID = 0x2
CREQ_MODIFY_QP_RESP_TYPE_MASK = 0x3f
CREQ_MODIFY_QP_RESP_TYPE_SFT = 0
CREQ_MODIFY_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_MODIFY_QP_RESP_TYPE_LAST = CREQ_MODIFY_QP_RESP_TYPE_QP_EVENT
CREQ_MODIFY_QP_RESP_V = 0x1
CREQ_MODIFY_QP_RESP_EVENT_MODIFY_QP = 0x3
CREQ_MODIFY_QP_RESP_EVENT_LAST = CREQ_MODIFY_QP_RESP_EVENT_MODIFY_QP
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_ENABLED = 0x1
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_INDEX_MASK = 0xe
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_INDEX_SFT = 1
CREQ_MODIFY_QP_RESP_PINGPONG_PUSH_STATE = 0x10
CMDQ_QUERY_QP_OPCODE_QUERY_QP = 0x4
CMDQ_QUERY_QP_OPCODE_LAST = CMDQ_QUERY_QP_OPCODE_QUERY_QP
CREQ_QUERY_QP_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_QP_RESP_TYPE_SFT = 0
CREQ_QUERY_QP_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_QP_RESP_TYPE_LAST = CREQ_QUERY_QP_RESP_TYPE_QP_EVENT
CREQ_QUERY_QP_RESP_V = 0x1
CREQ_QUERY_QP_RESP_EVENT_QUERY_QP = 0x4
CREQ_QUERY_QP_RESP_EVENT_LAST = CREQ_QUERY_QP_RESP_EVENT_QUERY_QP
CREQ_QUERY_QP_RESP_SB_OPCODE_QUERY_QP = 0x4
CREQ_QUERY_QP_RESP_SB_OPCODE_LAST = CREQ_QUERY_QP_RESP_SB_OPCODE_QUERY_QP
CREQ_QUERY_QP_RESP_SB_STATE_MASK = 0xf
CREQ_QUERY_QP_RESP_SB_STATE_SFT = 0
CREQ_QUERY_QP_RESP_SB_STATE_RESET = 0x0
CREQ_QUERY_QP_RESP_SB_STATE_INIT = 0x1
CREQ_QUERY_QP_RESP_SB_STATE_RTR = 0x2
CREQ_QUERY_QP_RESP_SB_STATE_RTS = 0x3
CREQ_QUERY_QP_RESP_SB_STATE_SQD = 0x4
CREQ_QUERY_QP_RESP_SB_STATE_SQE = 0x5
CREQ_QUERY_QP_RESP_SB_STATE_ERR = 0x6
CREQ_QUERY_QP_RESP_SB_STATE_LAST = CREQ_QUERY_QP_RESP_SB_STATE_ERR
CREQ_QUERY_QP_RESP_SB_EN_SQD_ASYNC_NOTIFY = 0x10
CREQ_QUERY_QP_RESP_SB_UNUSED3_MASK = 0xe0
CREQ_QUERY_QP_RESP_SB_UNUSED3_SFT = 5
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_MASK = 0xff
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_ATOMIC_REMOTE_READ_REMOTE_WRITE_LOCAL_WRITE_SFT = 0
CREQ_QUERY_QP_RESP_SB_ACCESS_LOCAL_WRITE = 0x1
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_WRITE = 0x2
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_READ = 0x4
CREQ_QUERY_QP_RESP_SB_ACCESS_REMOTE_ATOMIC = 0x8
CREQ_QUERY_QP_RESP_SB_DEST_VLAN_ID_MASK = 0xfff
CREQ_QUERY_QP_RESP_SB_DEST_VLAN_ID_SFT = 0
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MASK = 0xf000
CREQ_QUERY_QP_RESP_SB_PATH_MTU_SFT = 12
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_256 = (0x0 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_512 = (0x1 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_1024 = (0x2 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_2048 = (0x3 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_4096 = (0x4 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_8192 = (0x5 << 12)
CREQ_QUERY_QP_RESP_SB_PATH_MTU_LAST = CREQ_QUERY_QP_RESP_SB_PATH_MTU_MTU_8192
CREQ_QUERY_QP_RESP_SB_TOS_ECN_MASK = 0x3
CREQ_QUERY_QP_RESP_SB_TOS_ECN_SFT = 0
CREQ_QUERY_QP_RESP_SB_TOS_DSCP_MASK = 0xfc
CREQ_QUERY_QP_RESP_SB_TOS_DSCP_SFT = 2
CREQ_QUERY_QP_RESP_SB_ENABLE_CC = 0x1
CREQ_QUERY_QP_RESP_SB_VLAN_ID_MASK = 0xfff
CREQ_QUERY_QP_RESP_SB_VLAN_ID_SFT = 0
CREQ_QUERY_QP_RESP_SB_VLAN_DEI = 0x1000
CREQ_QUERY_QP_RESP_SB_VLAN_PCP_MASK = 0xe000
CREQ_QUERY_QP_RESP_SB_VLAN_PCP_SFT = 13
CMDQ_QUERY_QP_EXTEND_OPCODE_QUERY_QP_EXTEND = 0x91
CMDQ_QUERY_QP_EXTEND_OPCODE_LAST = CMDQ_QUERY_QP_EXTEND_OPCODE_QUERY_QP_EXTEND
CMDQ_QUERY_QP_EXTEND_PF_NUM_MASK = 0xff
CMDQ_QUERY_QP_EXTEND_PF_NUM_SFT = 0
CMDQ_QUERY_QP_EXTEND_VF_NUM_MASK = 0xffff00
CMDQ_QUERY_QP_EXTEND_VF_NUM_SFT = 8
CMDQ_QUERY_QP_EXTEND_VF_VALID = 0x1000000
CREQ_QUERY_QP_EXTEND_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_QP_EXTEND_RESP_TYPE_SFT = 0
CREQ_QUERY_QP_EXTEND_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_QP_EXTEND_RESP_TYPE_LAST = CREQ_QUERY_QP_EXTEND_RESP_TYPE_QP_EVENT
CREQ_QUERY_QP_EXTEND_RESP_V = 0x1
CREQ_QUERY_QP_EXTEND_RESP_EVENT_QUERY_QP_EXTEND = 0x91
CREQ_QUERY_QP_EXTEND_RESP_EVENT_LAST = CREQ_QUERY_QP_EXTEND_RESP_EVENT_QUERY_QP_EXTEND
CREQ_QUERY_QP_EXTEND_RESP_SB_OPCODE_QUERY_QP_EXTEND = 0x91
CREQ_QUERY_QP_EXTEND_RESP_SB_OPCODE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_OPCODE_QUERY_QP_EXTEND
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_MASK = 0xf
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_SFT = 0
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_RESET = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_INIT = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_RTR = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_RTS = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_SQD = 0x4
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_SQE = 0x5
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_ERR = 0x6
CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_STATE_ERR
CREQ_QUERY_QP_EXTEND_RESP_SB_UNUSED4_MASK = 0xf0
CREQ_QUERY_QP_EXTEND_RESP_SB_UNUSED4_SFT = 4
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV1 = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV2_IPV4 = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV2_IPV6 = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_NETWORK_TYPE_ROCEV2_IPV6
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_MORE = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_MORE_LAST = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_OPCODE_QUERY_QP_EXTEND = 0x91
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_OPCODE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_OPCODE_QUERY_QP_EXTEND
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_MASK = 0xf
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_SFT = 0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_RESET = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_INIT = 0x1
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_RTR = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_RTS = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_SQD = 0x4
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_SQE = 0x5
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_ERR = 0x6
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_STATE_ERR
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_UNUSED4_MASK = 0xf0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_UNUSED4_SFT = 4
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV1 = 0x0
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV2_IPV4 = 0x2
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV2_IPV6 = 0x3
CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_LAST = CREQ_QUERY_QP_EXTEND_RESP_SB_TLV_NETWORK_TYPE_ROCEV2_IPV6
CMDQ_CREATE_SRQ_OPCODE_CREATE_SRQ = 0x5
CMDQ_CREATE_SRQ_OPCODE_LAST = CMDQ_CREATE_SRQ_OPCODE_CREATE_SRQ
CMDQ_CREATE_SRQ_FLAGS_STEERING_TAG_VALID = 0x1
CMDQ_CREATE_SRQ_LVL_MASK = 0x3
CMDQ_CREATE_SRQ_LVL_SFT = 0
CMDQ_CREATE_SRQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_SRQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_SRQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_SRQ_LVL_LAST = CMDQ_CREATE_SRQ_LVL_LVL_2
CMDQ_CREATE_SRQ_PG_SIZE_MASK = 0x1c
CMDQ_CREATE_SRQ_PG_SIZE_SFT = 2
CMDQ_CREATE_SRQ_PG_SIZE_PG_4K = (0x0 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_8K = (0x1 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_64K = (0x2 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_2M = (0x3 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_8M = (0x4 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_PG_1G = (0x5 << 2)
CMDQ_CREATE_SRQ_PG_SIZE_LAST = CMDQ_CREATE_SRQ_PG_SIZE_PG_1G
CMDQ_CREATE_SRQ_UNUSED11_MASK = 0xffe0
CMDQ_CREATE_SRQ_UNUSED11_SFT = 5
CMDQ_CREATE_SRQ_EVENTQ_ID_MASK = 0xfff
CMDQ_CREATE_SRQ_EVENTQ_ID_SFT = 0
CMDQ_CREATE_SRQ_UNUSED4_MASK = 0xf000
CMDQ_CREATE_SRQ_UNUSED4_SFT = 12
CREQ_CREATE_SRQ_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_SRQ_RESP_TYPE_SFT = 0
CREQ_CREATE_SRQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_SRQ_RESP_TYPE_LAST = CREQ_CREATE_SRQ_RESP_TYPE_QP_EVENT
CREQ_CREATE_SRQ_RESP_V = 0x1
CREQ_CREATE_SRQ_RESP_EVENT_CREATE_SRQ = 0x5
CREQ_CREATE_SRQ_RESP_EVENT_LAST = CREQ_CREATE_SRQ_RESP_EVENT_CREATE_SRQ
CMDQ_DESTROY_SRQ_OPCODE_DESTROY_SRQ = 0x6
CMDQ_DESTROY_SRQ_OPCODE_LAST = CMDQ_DESTROY_SRQ_OPCODE_DESTROY_SRQ
CREQ_DESTROY_SRQ_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_SRQ_RESP_TYPE_SFT = 0
CREQ_DESTROY_SRQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_SRQ_RESP_TYPE_LAST = CREQ_DESTROY_SRQ_RESP_TYPE_QP_EVENT
CREQ_DESTROY_SRQ_RESP_V = 0x1
CREQ_DESTROY_SRQ_RESP_EVENT_DESTROY_SRQ = 0x6
CREQ_DESTROY_SRQ_RESP_EVENT_LAST = CREQ_DESTROY_SRQ_RESP_EVENT_DESTROY_SRQ
CREQ_DESTROY_SRQ_RESP_UNUSED0_MASK = 0xffff
CREQ_DESTROY_SRQ_RESP_UNUSED0_SFT = 0
CREQ_DESTROY_SRQ_RESP_ENABLE_FOR_ARM_MASK = 0x30000
CREQ_DESTROY_SRQ_RESP_ENABLE_FOR_ARM_SFT = 16
CMDQ_QUERY_SRQ_OPCODE_QUERY_SRQ = 0x8
CMDQ_QUERY_SRQ_OPCODE_LAST = CMDQ_QUERY_SRQ_OPCODE_QUERY_SRQ
CREQ_QUERY_SRQ_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_SRQ_RESP_TYPE_SFT = 0
CREQ_QUERY_SRQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_SRQ_RESP_TYPE_LAST = CREQ_QUERY_SRQ_RESP_TYPE_QP_EVENT
CREQ_QUERY_SRQ_RESP_V = 0x1
CREQ_QUERY_SRQ_RESP_EVENT_QUERY_SRQ = 0x8
CREQ_QUERY_SRQ_RESP_EVENT_LAST = CREQ_QUERY_SRQ_RESP_EVENT_QUERY_SRQ
CREQ_QUERY_SRQ_RESP_SB_OPCODE_QUERY_SRQ = 0x8
CREQ_QUERY_SRQ_RESP_SB_OPCODE_LAST = CREQ_QUERY_SRQ_RESP_SB_OPCODE_QUERY_SRQ
CMDQ_CREATE_CQ_OPCODE_CREATE_CQ = 0x9
CMDQ_CREATE_CQ_OPCODE_LAST = CMDQ_CREATE_CQ_OPCODE_CREATE_CQ
CMDQ_CREATE_CQ_FLAGS_DISABLE_CQ_OVERFLOW_DETECTION = 0x1
CMDQ_CREATE_CQ_FLAGS_STEERING_TAG_VALID = 0x2
CMDQ_CREATE_CQ_FLAGS_INFINITE_CQ_MODE = 0x4
CMDQ_CREATE_CQ_FLAGS_COALESCING_VALID = 0x8
CMDQ_CREATE_CQ_LVL_MASK = 0x3
CMDQ_CREATE_CQ_LVL_SFT = 0
CMDQ_CREATE_CQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_CQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_CQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_CQ_LVL_LAST = CMDQ_CREATE_CQ_LVL_LVL_2
CMDQ_CREATE_CQ_PG_SIZE_MASK = 0x1c
CMDQ_CREATE_CQ_PG_SIZE_SFT = 2
CMDQ_CREATE_CQ_PG_SIZE_PG_4K = (0x0 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_8K = (0x1 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_64K = (0x2 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_2M = (0x3 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_8M = (0x4 << 2)
CMDQ_CREATE_CQ_PG_SIZE_PG_1G = (0x5 << 2)
CMDQ_CREATE_CQ_PG_SIZE_LAST = CMDQ_CREATE_CQ_PG_SIZE_PG_1G
CMDQ_CREATE_CQ_UNUSED27_MASK = 0xffffffe0
CMDQ_CREATE_CQ_UNUSED27_SFT = 5
CMDQ_CREATE_CQ_CNQ_ID_MASK = 0xfff
CMDQ_CREATE_CQ_CNQ_ID_SFT = 0
CMDQ_CREATE_CQ_CQ_FCO_MASK = 0xfffff000
CMDQ_CREATE_CQ_CQ_FCO_SFT = 12
CMDQ_CREATE_CQ_BUF_MAXTIME_MASK = 0x1ff
CMDQ_CREATE_CQ_BUF_MAXTIME_SFT = 0
CMDQ_CREATE_CQ_NORMAL_MAXBUF_MASK = 0x3e00
CMDQ_CREATE_CQ_NORMAL_MAXBUF_SFT = 9
CMDQ_CREATE_CQ_DURING_MAXBUF_MASK = 0x7c000
CMDQ_CREATE_CQ_DURING_MAXBUF_SFT = 14
CMDQ_CREATE_CQ_ENABLE_RING_IDLE_MODE = 0x80000
CMDQ_CREATE_CQ_UNUSED12_MASK = 0xfff00000
CMDQ_CREATE_CQ_UNUSED12_SFT = 20
CREQ_CREATE_CQ_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_CQ_RESP_TYPE_SFT = 0
CREQ_CREATE_CQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_CQ_RESP_TYPE_LAST = CREQ_CREATE_CQ_RESP_TYPE_QP_EVENT
CREQ_CREATE_CQ_RESP_V = 0x1
CREQ_CREATE_CQ_RESP_EVENT_CREATE_CQ = 0x9
CREQ_CREATE_CQ_RESP_EVENT_LAST = CREQ_CREATE_CQ_RESP_EVENT_CREATE_CQ
CMDQ_DESTROY_CQ_OPCODE_DESTROY_CQ = 0xa
CMDQ_DESTROY_CQ_OPCODE_LAST = CMDQ_DESTROY_CQ_OPCODE_DESTROY_CQ
CREQ_DESTROY_CQ_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_CQ_RESP_TYPE_SFT = 0
CREQ_DESTROY_CQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_CQ_RESP_TYPE_LAST = CREQ_DESTROY_CQ_RESP_TYPE_QP_EVENT
CREQ_DESTROY_CQ_RESP_V = 0x1
CREQ_DESTROY_CQ_RESP_EVENT_DESTROY_CQ = 0xa
CREQ_DESTROY_CQ_RESP_EVENT_LAST = CREQ_DESTROY_CQ_RESP_EVENT_DESTROY_CQ
CREQ_DESTROY_CQ_RESP_CQ_ARM_LVL_MASK = 0x3
CREQ_DESTROY_CQ_RESP_CQ_ARM_LVL_SFT = 0
CMDQ_RESIZE_CQ_OPCODE_RESIZE_CQ = 0xc
CMDQ_RESIZE_CQ_OPCODE_LAST = CMDQ_RESIZE_CQ_OPCODE_RESIZE_CQ
CMDQ_RESIZE_CQ_LVL_MASK = 0x3
CMDQ_RESIZE_CQ_LVL_SFT = 0
CMDQ_RESIZE_CQ_LVL_LVL_0 = 0x0
CMDQ_RESIZE_CQ_LVL_LVL_1 = 0x1
CMDQ_RESIZE_CQ_LVL_LVL_2 = 0x2
CMDQ_RESIZE_CQ_LVL_LAST = CMDQ_RESIZE_CQ_LVL_LVL_2
CMDQ_RESIZE_CQ_PG_SIZE_MASK = 0x1c
CMDQ_RESIZE_CQ_PG_SIZE_SFT = 2
CMDQ_RESIZE_CQ_PG_SIZE_PG_4K = (0x0 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_8K = (0x1 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_64K = (0x2 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_2M = (0x3 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_8M = (0x4 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_PG_1G = (0x5 << 2)
CMDQ_RESIZE_CQ_PG_SIZE_LAST = CMDQ_RESIZE_CQ_PG_SIZE_PG_1G
CMDQ_RESIZE_CQ_NEW_CQ_SIZE_MASK = 0x1fffffe0
CMDQ_RESIZE_CQ_NEW_CQ_SIZE_SFT = 5
CREQ_RESIZE_CQ_RESP_TYPE_MASK = 0x3f
CREQ_RESIZE_CQ_RESP_TYPE_SFT = 0
CREQ_RESIZE_CQ_RESP_TYPE_QP_EVENT = 0x38
CREQ_RESIZE_CQ_RESP_TYPE_LAST = CREQ_RESIZE_CQ_RESP_TYPE_QP_EVENT
CREQ_RESIZE_CQ_RESP_V = 0x1
CREQ_RESIZE_CQ_RESP_EVENT_RESIZE_CQ = 0xc
CREQ_RESIZE_CQ_RESP_EVENT_LAST = CREQ_RESIZE_CQ_RESP_EVENT_RESIZE_CQ
CMDQ_ALLOCATE_MRW_OPCODE_ALLOCATE_MRW = 0xd
CMDQ_ALLOCATE_MRW_OPCODE_LAST = CMDQ_ALLOCATE_MRW_OPCODE_ALLOCATE_MRW
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MASK = 0xf
CMDQ_ALLOCATE_MRW_MRW_FLAGS_SFT = 0
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MR = 0x0
CMDQ_ALLOCATE_MRW_MRW_FLAGS_PMR = 0x1
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE1 = 0x2
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE2A = 0x3
CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE2B = 0x4
CMDQ_ALLOCATE_MRW_MRW_FLAGS_LAST = CMDQ_ALLOCATE_MRW_MRW_FLAGS_MW_TYPE2B
CMDQ_ALLOCATE_MRW_STEERING_TAG_VALID = 0x10
CMDQ_ALLOCATE_MRW_UNUSED4_MASK = 0xe0
CMDQ_ALLOCATE_MRW_UNUSED4_SFT = 5
CMDQ_ALLOCATE_MRW_ACCESS_CONSUMER_OWNED_KEY = 0x20
CREQ_ALLOCATE_MRW_RESP_TYPE_MASK = 0x3f
CREQ_ALLOCATE_MRW_RESP_TYPE_SFT = 0
CREQ_ALLOCATE_MRW_RESP_TYPE_QP_EVENT = 0x38
CREQ_ALLOCATE_MRW_RESP_TYPE_LAST = CREQ_ALLOCATE_MRW_RESP_TYPE_QP_EVENT
CREQ_ALLOCATE_MRW_RESP_V = 0x1
CREQ_ALLOCATE_MRW_RESP_EVENT_ALLOCATE_MRW = 0xd
CREQ_ALLOCATE_MRW_RESP_EVENT_LAST = CREQ_ALLOCATE_MRW_RESP_EVENT_ALLOCATE_MRW
CMDQ_DEALLOCATE_KEY_OPCODE_DEALLOCATE_KEY = 0xe
CMDQ_DEALLOCATE_KEY_OPCODE_LAST = CMDQ_DEALLOCATE_KEY_OPCODE_DEALLOCATE_KEY
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MASK = 0xf
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_SFT = 0
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MR = 0x0
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_PMR = 0x1
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE1 = 0x2
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE2A = 0x3
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE2B = 0x4
CMDQ_DEALLOCATE_KEY_MRW_FLAGS_LAST = CMDQ_DEALLOCATE_KEY_MRW_FLAGS_MW_TYPE2B
CMDQ_DEALLOCATE_KEY_UNUSED4_MASK = 0xf0
CMDQ_DEALLOCATE_KEY_UNUSED4_SFT = 4
CREQ_DEALLOCATE_KEY_RESP_TYPE_MASK = 0x3f
CREQ_DEALLOCATE_KEY_RESP_TYPE_SFT = 0
CREQ_DEALLOCATE_KEY_RESP_TYPE_QP_EVENT = 0x38
CREQ_DEALLOCATE_KEY_RESP_TYPE_LAST = CREQ_DEALLOCATE_KEY_RESP_TYPE_QP_EVENT
CREQ_DEALLOCATE_KEY_RESP_V = 0x1
CREQ_DEALLOCATE_KEY_RESP_EVENT_DEALLOCATE_KEY = 0xe
CREQ_DEALLOCATE_KEY_RESP_EVENT_LAST = CREQ_DEALLOCATE_KEY_RESP_EVENT_DEALLOCATE_KEY
CMDQ_REGISTER_MR_OPCODE_REGISTER_MR = 0xf
CMDQ_REGISTER_MR_OPCODE_LAST = CMDQ_REGISTER_MR_OPCODE_REGISTER_MR
CMDQ_REGISTER_MR_FLAGS_ALLOC_MR = 0x1
CMDQ_REGISTER_MR_FLAGS_STEERING_TAG_VALID = 0x2
CMDQ_REGISTER_MR_FLAGS_ENABLE_RO = 0x4
CMDQ_REGISTER_MR_LVL_MASK = 0x3
CMDQ_REGISTER_MR_LVL_SFT = 0
CMDQ_REGISTER_MR_LVL_LVL_0 = 0x0
CMDQ_REGISTER_MR_LVL_LVL_1 = 0x1
CMDQ_REGISTER_MR_LVL_LVL_2 = 0x2
CMDQ_REGISTER_MR_LVL_LAST = CMDQ_REGISTER_MR_LVL_LVL_2
CMDQ_REGISTER_MR_LOG2_PG_SIZE_MASK = 0x7c
CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT = 2
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_4K = (0xc << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_8K = (0xd << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_64K = (0x10 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_256K = (0x12 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_1M = (0x14 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_2M = (0x15 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_4M = (0x16 << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_1G = (0x1e << 2)
CMDQ_REGISTER_MR_LOG2_PG_SIZE_LAST = CMDQ_REGISTER_MR_LOG2_PG_SIZE_PG_1G
CMDQ_REGISTER_MR_UNUSED1 = 0x80
CMDQ_REGISTER_MR_ACCESS_LOCAL_WRITE = 0x1
CMDQ_REGISTER_MR_ACCESS_REMOTE_READ = 0x2
CMDQ_REGISTER_MR_ACCESS_REMOTE_WRITE = 0x4
CMDQ_REGISTER_MR_ACCESS_REMOTE_ATOMIC = 0x8
CMDQ_REGISTER_MR_ACCESS_MW_BIND = 0x10
CMDQ_REGISTER_MR_ACCESS_ZERO_BASED = 0x20
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_MASK = 0x1f
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_SFT = 0
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_4K = 0xc
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_8K = 0xd
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_64K = 0x10
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_256K = 0x12
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_1M = 0x14
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_2M = 0x15
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_4M = 0x16
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_1G = 0x1e
CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_LAST = CMDQ_REGISTER_MR_LOG2_PBL_PG_SIZE_PG_1G
CMDQ_REGISTER_MR_UNUSED11_MASK = 0xffe0
CMDQ_REGISTER_MR_UNUSED11_SFT = 5
CREQ_REGISTER_MR_RESP_TYPE_MASK = 0x3f
CREQ_REGISTER_MR_RESP_TYPE_SFT = 0
CREQ_REGISTER_MR_RESP_TYPE_QP_EVENT = 0x38
CREQ_REGISTER_MR_RESP_TYPE_LAST = CREQ_REGISTER_MR_RESP_TYPE_QP_EVENT
CREQ_REGISTER_MR_RESP_V = 0x1
CREQ_REGISTER_MR_RESP_EVENT_REGISTER_MR = 0xf
CREQ_REGISTER_MR_RESP_EVENT_LAST = CREQ_REGISTER_MR_RESP_EVENT_REGISTER_MR
CMDQ_DEREGISTER_MR_OPCODE_DEREGISTER_MR = 0x10
CMDQ_DEREGISTER_MR_OPCODE_LAST = CMDQ_DEREGISTER_MR_OPCODE_DEREGISTER_MR
CREQ_DEREGISTER_MR_RESP_TYPE_MASK = 0x3f
CREQ_DEREGISTER_MR_RESP_TYPE_SFT = 0
CREQ_DEREGISTER_MR_RESP_TYPE_QP_EVENT = 0x38
CREQ_DEREGISTER_MR_RESP_TYPE_LAST = CREQ_DEREGISTER_MR_RESP_TYPE_QP_EVENT
CREQ_DEREGISTER_MR_RESP_V = 0x1
CREQ_DEREGISTER_MR_RESP_EVENT_DEREGISTER_MR = 0x10
CREQ_DEREGISTER_MR_RESP_EVENT_LAST = CREQ_DEREGISTER_MR_RESP_EVENT_DEREGISTER_MR
CMDQ_ADD_GID_OPCODE_ADD_GID = 0x11
CMDQ_ADD_GID_OPCODE_LAST = CMDQ_ADD_GID_OPCODE_ADD_GID
CMDQ_ADD_GID_VLAN_VLAN_EN_TPID_VLAN_ID_MASK = 0xffff
CMDQ_ADD_GID_VLAN_VLAN_EN_TPID_VLAN_ID_SFT = 0
CMDQ_ADD_GID_VLAN_VLAN_ID_MASK = 0xfff
CMDQ_ADD_GID_VLAN_VLAN_ID_SFT = 0
CMDQ_ADD_GID_VLAN_TPID_MASK = 0x7000
CMDQ_ADD_GID_VLAN_TPID_SFT = 12
CMDQ_ADD_GID_VLAN_TPID_TPID_88A8 = (0x0 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_8100 = (0x1 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_9100 = (0x2 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_9200 = (0x3 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_9300 = (0x4 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_CFG1 = (0x5 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_CFG2 = (0x6 << 12)
CMDQ_ADD_GID_VLAN_TPID_TPID_CFG3 = (0x7 << 12)
CMDQ_ADD_GID_VLAN_TPID_LAST = CMDQ_ADD_GID_VLAN_TPID_TPID_CFG3
CMDQ_ADD_GID_VLAN_VLAN_EN = 0x8000
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_VALID_STATS_CTX_ID_MASK = 0xffff
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_VALID_STATS_CTX_ID_SFT = 0
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_ID_MASK = 0x7fff
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_ID_SFT = 0
CMDQ_ADD_GID_STATS_CTX_STATS_CTX_VALID = 0x8000
CREQ_ADD_GID_RESP_TYPE_MASK = 0x3f
CREQ_ADD_GID_RESP_TYPE_SFT = 0
CREQ_ADD_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_ADD_GID_RESP_TYPE_LAST = CREQ_ADD_GID_RESP_TYPE_QP_EVENT
CREQ_ADD_GID_RESP_V = 0x1
CREQ_ADD_GID_RESP_EVENT_ADD_GID = 0x11
CREQ_ADD_GID_RESP_EVENT_LAST = CREQ_ADD_GID_RESP_EVENT_ADD_GID
CMDQ_DELETE_GID_OPCODE_DELETE_GID = 0x12
CMDQ_DELETE_GID_OPCODE_LAST = CMDQ_DELETE_GID_OPCODE_DELETE_GID
CREQ_DELETE_GID_RESP_TYPE_MASK = 0x3f
CREQ_DELETE_GID_RESP_TYPE_SFT = 0
CREQ_DELETE_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_DELETE_GID_RESP_TYPE_LAST = CREQ_DELETE_GID_RESP_TYPE_QP_EVENT
CREQ_DELETE_GID_RESP_V = 0x1
CREQ_DELETE_GID_RESP_EVENT_DELETE_GID = 0x12
CREQ_DELETE_GID_RESP_EVENT_LAST = CREQ_DELETE_GID_RESP_EVENT_DELETE_GID
CMDQ_MODIFY_GID_OPCODE_MODIFY_GID = 0x17
CMDQ_MODIFY_GID_OPCODE_LAST = CMDQ_MODIFY_GID_OPCODE_MODIFY_GID
CMDQ_MODIFY_GID_VLAN_VLAN_ID_MASK = 0xfff
CMDQ_MODIFY_GID_VLAN_VLAN_ID_SFT = 0
CMDQ_MODIFY_GID_VLAN_TPID_MASK = 0x7000
CMDQ_MODIFY_GID_VLAN_TPID_SFT = 12
CMDQ_MODIFY_GID_VLAN_TPID_TPID_88A8 = (0x0 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_8100 = (0x1 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_9100 = (0x2 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_9200 = (0x3 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_9300 = (0x4 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG1 = (0x5 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG2 = (0x6 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG3 = (0x7 << 12)
CMDQ_MODIFY_GID_VLAN_TPID_LAST = CMDQ_MODIFY_GID_VLAN_TPID_TPID_CFG3
CMDQ_MODIFY_GID_VLAN_VLAN_EN = 0x8000
CMDQ_MODIFY_GID_STATS_CTX_STATS_CTX_ID_MASK = 0x7fff
CMDQ_MODIFY_GID_STATS_CTX_STATS_CTX_ID_SFT = 0
CMDQ_MODIFY_GID_STATS_CTX_STATS_CTX_VALID = 0x8000
CREQ_MODIFY_GID_RESP_TYPE_MASK = 0x3f
CREQ_MODIFY_GID_RESP_TYPE_SFT = 0
CREQ_MODIFY_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_MODIFY_GID_RESP_TYPE_LAST = CREQ_MODIFY_GID_RESP_TYPE_QP_EVENT
CREQ_MODIFY_GID_RESP_V = 0x1
CREQ_MODIFY_GID_RESP_EVENT_ADD_GID = 0x11
CREQ_MODIFY_GID_RESP_EVENT_LAST = CREQ_MODIFY_GID_RESP_EVENT_ADD_GID
CMDQ_QUERY_GID_OPCODE_QUERY_GID = 0x18
CMDQ_QUERY_GID_OPCODE_LAST = CMDQ_QUERY_GID_OPCODE_QUERY_GID
CREQ_QUERY_GID_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_GID_RESP_TYPE_SFT = 0
CREQ_QUERY_GID_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_GID_RESP_TYPE_LAST = CREQ_QUERY_GID_RESP_TYPE_QP_EVENT
CREQ_QUERY_GID_RESP_V = 0x1
CREQ_QUERY_GID_RESP_EVENT_QUERY_GID = 0x18
CREQ_QUERY_GID_RESP_EVENT_LAST = CREQ_QUERY_GID_RESP_EVENT_QUERY_GID
CREQ_QUERY_GID_RESP_SB_OPCODE_QUERY_GID = 0x18
CREQ_QUERY_GID_RESP_SB_OPCODE_LAST = CREQ_QUERY_GID_RESP_SB_OPCODE_QUERY_GID
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_EN_TPID_VLAN_ID_MASK = 0xffff
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_EN_TPID_VLAN_ID_SFT = 0
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_ID_MASK = 0xfff
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_ID_SFT = 0
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_MASK = 0x7000
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_SFT = 12
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_88A8 = (0x0 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_8100 = (0x1 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_9100 = (0x2 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_9200 = (0x3 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_9300 = (0x4 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG1 = (0x5 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG2 = (0x6 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG3 = (0x7 << 12)
CREQ_QUERY_GID_RESP_SB_VLAN_TPID_LAST = CREQ_QUERY_GID_RESP_SB_VLAN_TPID_TPID_CFG3
CREQ_QUERY_GID_RESP_SB_VLAN_VLAN_EN = 0x8000
CMDQ_CREATE_QP1_OPCODE_CREATE_QP1 = 0x13
CMDQ_CREATE_QP1_OPCODE_LAST = CMDQ_CREATE_QP1_OPCODE_CREATE_QP1
CMDQ_CREATE_QP1_QP_FLAGS_SRQ_USED = 0x1
CMDQ_CREATE_QP1_QP_FLAGS_FORCE_COMPLETION = 0x2
CMDQ_CREATE_QP1_QP_FLAGS_RESERVED_LKEY_ENABLE = 0x4
CMDQ_CREATE_QP1_QP_FLAGS_LAST = CMDQ_CREATE_QP1_QP_FLAGS_RESERVED_LKEY_ENABLE
CMDQ_CREATE_QP1_TYPE_GSI = 0x1
CMDQ_CREATE_QP1_TYPE_LAST = CMDQ_CREATE_QP1_TYPE_GSI
CMDQ_CREATE_QP1_SQ_LVL_MASK = 0xf
CMDQ_CREATE_QP1_SQ_LVL_SFT = 0
CMDQ_CREATE_QP1_SQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP1_SQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP1_SQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP1_SQ_LVL_LAST = CMDQ_CREATE_QP1_SQ_LVL_LVL_2
CMDQ_CREATE_QP1_SQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP1_SQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP1_SQ_PG_SIZE_LAST = CMDQ_CREATE_QP1_SQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP1_RQ_LVL_MASK = 0xf
CMDQ_CREATE_QP1_RQ_LVL_SFT = 0
CMDQ_CREATE_QP1_RQ_LVL_LVL_0 = 0x0
CMDQ_CREATE_QP1_RQ_LVL_LVL_1 = 0x1
CMDQ_CREATE_QP1_RQ_LVL_LVL_2 = 0x2
CMDQ_CREATE_QP1_RQ_LVL_LAST = CMDQ_CREATE_QP1_RQ_LVL_LVL_2
CMDQ_CREATE_QP1_RQ_PG_SIZE_MASK = 0xf0
CMDQ_CREATE_QP1_RQ_PG_SIZE_SFT = 4
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_4K = (0x0 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_8K = (0x1 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_64K = (0x2 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_2M = (0x3 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_8M = (0x4 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_1G = (0x5 << 4)
CMDQ_CREATE_QP1_RQ_PG_SIZE_LAST = CMDQ_CREATE_QP1_RQ_PG_SIZE_PG_1G
CMDQ_CREATE_QP1_SQ_SGE_MASK = 0xf
CMDQ_CREATE_QP1_SQ_SGE_SFT = 0
CMDQ_CREATE_QP1_SQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP1_SQ_FWO_SFT = 4
CMDQ_CREATE_QP1_RQ_SGE_MASK = 0xf
CMDQ_CREATE_QP1_RQ_SGE_SFT = 0
CMDQ_CREATE_QP1_RQ_FWO_MASK = 0xfff0
CMDQ_CREATE_QP1_RQ_FWO_SFT = 4
CREQ_CREATE_QP1_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_QP1_RESP_TYPE_SFT = 0
CREQ_CREATE_QP1_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_QP1_RESP_TYPE_LAST = CREQ_CREATE_QP1_RESP_TYPE_QP_EVENT
CREQ_CREATE_QP1_RESP_V = 0x1
CREQ_CREATE_QP1_RESP_EVENT_CREATE_QP1 = 0x13
CREQ_CREATE_QP1_RESP_EVENT_LAST = CREQ_CREATE_QP1_RESP_EVENT_CREATE_QP1
CMDQ_DESTROY_QP1_OPCODE_DESTROY_QP1 = 0x14
CMDQ_DESTROY_QP1_OPCODE_LAST = CMDQ_DESTROY_QP1_OPCODE_DESTROY_QP1
CREQ_DESTROY_QP1_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_QP1_RESP_TYPE_SFT = 0
CREQ_DESTROY_QP1_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_QP1_RESP_TYPE_LAST = CREQ_DESTROY_QP1_RESP_TYPE_QP_EVENT
CREQ_DESTROY_QP1_RESP_V = 0x1
CREQ_DESTROY_QP1_RESP_EVENT_DESTROY_QP1 = 0x14
CREQ_DESTROY_QP1_RESP_EVENT_LAST = CREQ_DESTROY_QP1_RESP_EVENT_DESTROY_QP1
CMDQ_CREATE_AH_OPCODE_CREATE_AH = 0x15
CMDQ_CREATE_AH_OPCODE_LAST = CMDQ_CREATE_AH_OPCODE_CREATE_AH
CMDQ_CREATE_AH_TYPE_V1 = 0x0
CMDQ_CREATE_AH_TYPE_V2IPV4 = 0x2
CMDQ_CREATE_AH_TYPE_V2IPV6 = 0x3
CMDQ_CREATE_AH_TYPE_LAST = CMDQ_CREATE_AH_TYPE_V2IPV6
CMDQ_CREATE_AH_FLOW_LABEL_MASK = 0xfffff
CMDQ_CREATE_AH_FLOW_LABEL_SFT = 0
CMDQ_CREATE_AH_DEST_VLAN_ID_MASK = 0xfff00000
CMDQ_CREATE_AH_DEST_VLAN_ID_SFT = 20
CMDQ_CREATE_AH_ENABLE_CC = 0x1
CREQ_CREATE_AH_RESP_TYPE_MASK = 0x3f
CREQ_CREATE_AH_RESP_TYPE_SFT = 0
CREQ_CREATE_AH_RESP_TYPE_QP_EVENT = 0x38
CREQ_CREATE_AH_RESP_TYPE_LAST = CREQ_CREATE_AH_RESP_TYPE_QP_EVENT
CREQ_CREATE_AH_RESP_V = 0x1
CREQ_CREATE_AH_RESP_EVENT_CREATE_AH = 0x15
CREQ_CREATE_AH_RESP_EVENT_LAST = CREQ_CREATE_AH_RESP_EVENT_CREATE_AH
CMDQ_DESTROY_AH_OPCODE_DESTROY_AH = 0x16
CMDQ_DESTROY_AH_OPCODE_LAST = CMDQ_DESTROY_AH_OPCODE_DESTROY_AH
CREQ_DESTROY_AH_RESP_TYPE_MASK = 0x3f
CREQ_DESTROY_AH_RESP_TYPE_SFT = 0
CREQ_DESTROY_AH_RESP_TYPE_QP_EVENT = 0x38
CREQ_DESTROY_AH_RESP_TYPE_LAST = CREQ_DESTROY_AH_RESP_TYPE_QP_EVENT
CREQ_DESTROY_AH_RESP_V = 0x1
CREQ_DESTROY_AH_RESP_EVENT_DESTROY_AH = 0x16
CREQ_DESTROY_AH_RESP_EVENT_LAST = CREQ_DESTROY_AH_RESP_EVENT_DESTROY_AH
CMDQ_QUERY_ROCE_STATS_OPCODE_QUERY_ROCE_STATS = 0x8e
CMDQ_QUERY_ROCE_STATS_OPCODE_LAST = CMDQ_QUERY_ROCE_STATS_OPCODE_QUERY_ROCE_STATS
CMDQ_QUERY_ROCE_STATS_FLAGS_COLLECTION_ID = 0x1
CMDQ_QUERY_ROCE_STATS_FLAGS_FUNCTION_ID = 0x2
CMDQ_QUERY_ROCE_STATS_PF_NUM_MASK = 0xff
CMDQ_QUERY_ROCE_STATS_PF_NUM_SFT = 0
CMDQ_QUERY_ROCE_STATS_VF_NUM_MASK = 0xffff00
CMDQ_QUERY_ROCE_STATS_VF_NUM_SFT = 8
CMDQ_QUERY_ROCE_STATS_VF_VALID = 0x1000000
CREQ_QUERY_ROCE_STATS_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_ROCE_STATS_RESP_TYPE_SFT = 0
CREQ_QUERY_ROCE_STATS_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_ROCE_STATS_RESP_TYPE_LAST = CREQ_QUERY_ROCE_STATS_RESP_TYPE_QP_EVENT
CREQ_QUERY_ROCE_STATS_RESP_V = 0x1
CREQ_QUERY_ROCE_STATS_RESP_EVENT_QUERY_ROCE_STATS = 0x8e
CREQ_QUERY_ROCE_STATS_RESP_EVENT_LAST = CREQ_QUERY_ROCE_STATS_RESP_EVENT_QUERY_ROCE_STATS
CREQ_QUERY_ROCE_STATS_RESP_SB_OPCODE_QUERY_ROCE_STATS = 0x8e
CREQ_QUERY_ROCE_STATS_RESP_SB_OPCODE_LAST = CREQ_QUERY_ROCE_STATS_RESP_SB_OPCODE_QUERY_ROCE_STATS
CMDQ_QUERY_ROCE_STATS_EXT_OPCODE_QUERY_ROCE_STATS = 0x92
CMDQ_QUERY_ROCE_STATS_EXT_OPCODE_LAST = CMDQ_QUERY_ROCE_STATS_EXT_OPCODE_QUERY_ROCE_STATS
CMDQ_QUERY_ROCE_STATS_EXT_FLAGS_COLLECTION_ID = 0x1
CMDQ_QUERY_ROCE_STATS_EXT_FLAGS_FUNCTION_ID = 0x2
CMDQ_QUERY_ROCE_STATS_EXT_PF_NUM_MASK = 0xff
CMDQ_QUERY_ROCE_STATS_EXT_PF_NUM_SFT = 0
CMDQ_QUERY_ROCE_STATS_EXT_VF_NUM_MASK = 0xffff00
CMDQ_QUERY_ROCE_STATS_EXT_VF_NUM_SFT = 8
CMDQ_QUERY_ROCE_STATS_EXT_VF_VALID = 0x1000000
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_SFT = 0
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_LAST = CREQ_QUERY_ROCE_STATS_EXT_RESP_TYPE_QP_EVENT
CREQ_QUERY_ROCE_STATS_EXT_RESP_V = 0x1
CREQ_QUERY_ROCE_STATS_EXT_RESP_EVENT_QUERY_ROCE_STATS_EXT = 0x92
CREQ_QUERY_ROCE_STATS_EXT_RESP_EVENT_LAST = CREQ_QUERY_ROCE_STATS_EXT_RESP_EVENT_QUERY_ROCE_STATS_EXT
CREQ_QUERY_ROCE_STATS_EXT_RESP_SB_OPCODE_QUERY_ROCE_STATS_EXT = 0x92
CREQ_QUERY_ROCE_STATS_EXT_RESP_SB_OPCODE_LAST = CREQ_QUERY_ROCE_STATS_EXT_RESP_SB_OPCODE_QUERY_ROCE_STATS_EXT
CMDQ_ROCE_MIRROR_CFG_OPCODE_ROCE_MIRROR_CFG = 0x99
CMDQ_ROCE_MIRROR_CFG_OPCODE_LAST = CMDQ_ROCE_MIRROR_CFG_OPCODE_ROCE_MIRROR_CFG
CMDQ_ROCE_MIRROR_CFG_MIRROR_ENABLE = 0x1
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_MASK = 0x3f
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_SFT = 0
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_QP_EVENT = 0x38
CREQ_ROCE_MIRROR_CFG_RESP_TYPE_LAST = CREQ_ROCE_MIRROR_CFG_RESP_TYPE_QP_EVENT
CREQ_ROCE_MIRROR_CFG_RESP_V = 0x1
CREQ_ROCE_MIRROR_CFG_RESP_EVENT_ROCE_MIRROR_CFG = 0x99
CREQ_ROCE_MIRROR_CFG_RESP_EVENT_LAST = CREQ_ROCE_MIRROR_CFG_RESP_EVENT_ROCE_MIRROR_CFG
CMDQ_QUERY_FUNC_OPCODE_QUERY_FUNC = 0x83
CMDQ_QUERY_FUNC_OPCODE_LAST = CMDQ_QUERY_FUNC_OPCODE_QUERY_FUNC
CREQ_QUERY_FUNC_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_FUNC_RESP_TYPE_SFT = 0
CREQ_QUERY_FUNC_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_FUNC_RESP_TYPE_LAST = CREQ_QUERY_FUNC_RESP_TYPE_QP_EVENT
CREQ_QUERY_FUNC_RESP_V = 0x1
CREQ_QUERY_FUNC_RESP_EVENT_QUERY_FUNC = 0x83
CREQ_QUERY_FUNC_RESP_EVENT_LAST = CREQ_QUERY_FUNC_RESP_EVENT_QUERY_FUNC
CREQ_QUERY_FUNC_RESP_SB_OPCODE_QUERY_FUNC = 0x83
CREQ_QUERY_FUNC_RESP_SB_OPCODE_LAST = CREQ_QUERY_FUNC_RESP_SB_OPCODE_QUERY_FUNC
CREQ_QUERY_FUNC_RESP_SB_RESIZE_QP = 0x1
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_MASK = 0xe
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_SFT = 1
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN0 = (0x0 << 1)
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN1 = (0x1 << 1)
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN1_EXT = (0x2 << 1)
CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_LAST = CREQ_QUERY_FUNC_RESP_SB_CC_GENERATION_CC_GEN1_EXT
CREQ_QUERY_FUNC_RESP_SB_EXT_STATS = 0x10
CREQ_QUERY_FUNC_RESP_SB_MR_REGISTER_ALLOC = 0x20
CREQ_QUERY_FUNC_RESP_SB_OPTIMIZED_TRANSMIT_ENABLED = 0x40
CREQ_QUERY_FUNC_RESP_SB_CQE_V2 = 0x80
CREQ_QUERY_FUNC_RESP_SB_PINGPONG_PUSH_MODE = 0x100
CREQ_QUERY_FUNC_RESP_SB_HW_REQUESTER_RETX_ENABLED = 0x200
CREQ_QUERY_FUNC_RESP_SB_HW_RESPONDER_RETX_ENABLED = 0x400
CREQ_QUERY_FUNC_RESP_SB_ATOMIC_OPS_NOT_SUPPORTED = 0x1
CREQ_QUERY_FUNC_RESP_SB_DRV_VERSION_RGTR_SUPPORTED = 0x2
CREQ_QUERY_FUNC_RESP_SB_CREATE_QP_BATCH_SUPPORTED = 0x4
CREQ_QUERY_FUNC_RESP_SB_DESTROY_QP_BATCH_SUPPORTED = 0x8
CREQ_QUERY_FUNC_RESP_SB_ROCE_STATS_EXT_CTX_SUPPORTED = 0x10
CREQ_QUERY_FUNC_RESP_SB_CREATE_SRQ_SGE_SUPPORTED = 0x20
CREQ_QUERY_FUNC_RESP_SB_FIXED_SIZE_WQE_DISABLED = 0x40
CREQ_QUERY_FUNC_RESP_SB_DCN_SUPPORTED = 0x80
CREQ_QUERY_FUNC_RESP_SB_OPTIMIZE_MODIFY_QP_SUPPORTED = 0x1
CREQ_QUERY_FUNC_RESP_SB_CHANGE_UDP_SRC_PORT_WQE_SUPPORTED = 0x2
CREQ_QUERY_FUNC_RESP_SB_CQ_COALESCING_SUPPORTED = 0x4
CREQ_QUERY_FUNC_RESP_SB_MEMORY_REGION_RO_SUPPORTED = 0x8
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_MASK = 0x30
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_SFT = 4
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_HOST_PSN_TABLE = (0x0 << 4)
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_HOST_MSN_TABLE = (0x1 << 4)
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_IQM_MSN_TABLE = (0x2 << 4)
CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_LAST = CREQ_QUERY_FUNC_RESP_SB_REQ_RETRANSMISSION_SUPPORT_IQM_MSN_TABLE
CREQ_QUERY_FUNC_RESP_SB_MAX_SRQ_EXTENDED = 0x40
CREQ_QUERY_FUNC_RESP_SB_MIN_RNR_RTR_RTS_OPT_SUPPORTED = 0x1000
CMDQ_SET_FUNC_RESOURCES_OPCODE_SET_FUNC_RESOURCES = 0x84
CMDQ_SET_FUNC_RESOURCES_OPCODE_LAST = CMDQ_SET_FUNC_RESOURCES_OPCODE_SET_FUNC_RESOURCES
CMDQ_SET_FUNC_RESOURCES_FLAGS_MRAV_RESERVATION_SPLIT = 0x1
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_MASK = 0x3f
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_SFT = 0
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_QP_EVENT = 0x38
CREQ_SET_FUNC_RESOURCES_RESP_TYPE_LAST = CREQ_SET_FUNC_RESOURCES_RESP_TYPE_QP_EVENT
CREQ_SET_FUNC_RESOURCES_RESP_V = 0x1
CREQ_SET_FUNC_RESOURCES_RESP_EVENT_SET_FUNC_RESOURCES = 0x84
CREQ_SET_FUNC_RESOURCES_RESP_EVENT_LAST = CREQ_SET_FUNC_RESOURCES_RESP_EVENT_SET_FUNC_RESOURCES
CMDQ_READ_CONTEXT_OPCODE_READ_CONTEXT = 0x85
CMDQ_READ_CONTEXT_OPCODE_LAST = CMDQ_READ_CONTEXT_OPCODE_READ_CONTEXT
CMDQ_READ_CONTEXT_TYPE_QPC = 0x0
CMDQ_READ_CONTEXT_TYPE_CQ = 0x1
CMDQ_READ_CONTEXT_TYPE_MRW = 0x2
CMDQ_READ_CONTEXT_TYPE_SRQ = 0x3
CMDQ_READ_CONTEXT_TYPE_LAST = CMDQ_READ_CONTEXT_TYPE_SRQ
CREQ_READ_CONTEXT_TYPE_MASK = 0x3f
CREQ_READ_CONTEXT_TYPE_SFT = 0
CREQ_READ_CONTEXT_TYPE_QP_EVENT = 0x38
CREQ_READ_CONTEXT_TYPE_LAST = CREQ_READ_CONTEXT_TYPE_QP_EVENT
CREQ_READ_CONTEXT_V = 0x1
CREQ_READ_CONTEXT_EVENT_READ_CONTEXT = 0x85
CREQ_READ_CONTEXT_EVENT_LAST = CREQ_READ_CONTEXT_EVENT_READ_CONTEXT
CMDQ_MAP_TC_TO_COS_OPCODE_MAP_TC_TO_COS = 0x8a
CMDQ_MAP_TC_TO_COS_OPCODE_LAST = CMDQ_MAP_TC_TO_COS_OPCODE_MAP_TC_TO_COS
CMDQ_MAP_TC_TO_COS_COS0_NO_CHANGE = 0xffff
CMDQ_MAP_TC_TO_COS_COS0_LAST = CMDQ_MAP_TC_TO_COS_COS0_NO_CHANGE
CMDQ_MAP_TC_TO_COS_COS1_DISABLE = 0x8000
CMDQ_MAP_TC_TO_COS_COS1_NO_CHANGE = 0xffff
CMDQ_MAP_TC_TO_COS_COS1_LAST = CMDQ_MAP_TC_TO_COS_COS1_NO_CHANGE
CREQ_MAP_TC_TO_COS_RESP_TYPE_MASK = 0x3f
CREQ_MAP_TC_TO_COS_RESP_TYPE_SFT = 0
CREQ_MAP_TC_TO_COS_RESP_TYPE_QP_EVENT = 0x38
CREQ_MAP_TC_TO_COS_RESP_TYPE_LAST = CREQ_MAP_TC_TO_COS_RESP_TYPE_QP_EVENT
CREQ_MAP_TC_TO_COS_RESP_V = 0x1
CREQ_MAP_TC_TO_COS_RESP_EVENT_MAP_TC_TO_COS = 0x8a
CREQ_MAP_TC_TO_COS_RESP_EVENT_LAST = CREQ_MAP_TC_TO_COS_RESP_EVENT_MAP_TC_TO_COS
CMDQ_QUERY_ROCE_CC_OPCODE_QUERY_ROCE_CC = 0x8d
CMDQ_QUERY_ROCE_CC_OPCODE_LAST = CMDQ_QUERY_ROCE_CC_OPCODE_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_TYPE_MASK = 0x3f
CREQ_QUERY_ROCE_CC_RESP_TYPE_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_TYPE_QP_EVENT = 0x38
CREQ_QUERY_ROCE_CC_RESP_TYPE_LAST = CREQ_QUERY_ROCE_CC_RESP_TYPE_QP_EVENT
CREQ_QUERY_ROCE_CC_RESP_V = 0x1
CREQ_QUERY_ROCE_CC_RESP_EVENT_QUERY_ROCE_CC = 0x8d
CREQ_QUERY_ROCE_CC_RESP_EVENT_LAST = CREQ_QUERY_ROCE_CC_RESP_EVENT_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_SB_OPCODE_QUERY_ROCE_CC = 0x8d
CREQ_QUERY_ROCE_CC_RESP_SB_OPCODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_OPCODE_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_SB_ENABLE_CC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_UNUSED7_MASK = 0xfe
CREQ_QUERY_ROCE_CC_RESP_SB_UNUSED7_SFT = 1
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_ECN_MASK = 0x3
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_ECN_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_DSCP_MASK = 0xfc
CREQ_QUERY_ROCE_CC_RESP_SB_TOS_DSCP_SFT = 2
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_VLAN_PCP_MASK = 0x7
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_VLAN_PCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD1_MASK = 0xf8
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD1_SFT = 3
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_TOS_DSCP_MASK = 0x3f
CREQ_QUERY_ROCE_CC_RESP_SB_ALT_TOS_DSCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD4_MASK = 0xc0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD4_SFT = 6
CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_DCTCP = 0x0
CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_PROBABILISTIC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_CC_MODE_PROBABILISTIC
CREQ_QUERY_ROCE_CC_RESP_SB_RTT_MASK = 0x3fff
CREQ_QUERY_ROCE_CC_RESP_SB_RTT_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD5_MASK = 0xc000
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD5_SFT = 14
CREQ_QUERY_ROCE_CC_RESP_SB_TCP_CP_MASK = 0x3ff
CREQ_QUERY_ROCE_CC_RESP_SB_TCP_CP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD6_MASK = 0xfc00
CREQ_QUERY_ROCE_CC_RESP_SB_RSVD6_SFT = 10
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_MORE = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_MORE_LAST = 0x0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED = 0x2
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_OPCODE_QUERY_ROCE_CC = 0x8d
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_OPCODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_TLV_OPCODE_QUERY_ROCE_CC
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ENABLE_CC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_UNUSED7_MASK = 0xfe
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_UNUSED7_SFT = 1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_ECN_MASK = 0x3
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_ECN_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_DSCP_MASK = 0xfc
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TOS_DSCP_SFT = 2
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_VLAN_PCP_MASK = 0x7
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_VLAN_PCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD1_MASK = 0xf8
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD1_SFT = 3
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_TOS_DSCP_MASK = 0x3f
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_ALT_TOS_DSCP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD4_MASK = 0xc0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD4_SFT = 6
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_DCTCP = 0x0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_PROBABILISTIC = 0x1
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_LAST = CREQ_QUERY_ROCE_CC_RESP_SB_TLV_CC_MODE_PROBABILISTIC
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RTT_MASK = 0x3fff
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RTT_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD5_MASK = 0xc000
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD5_SFT = 14
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TCP_CP_MASK = 0x3ff
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_TCP_CP_SFT = 0
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD6_MASK = 0xfc00
CREQ_QUERY_ROCE_CC_RESP_SB_TLV_RSVD6_SFT = 10
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_MORE = 0x1
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_MORE_LAST = 0x0
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED = 0x2
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_LAST = CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_TLV_FLAGS_REQUIRED_YES
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_NOT_ECT = 0x0
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_ECT_1 = 0x1
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_ECT_0 = 0x2
CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_LAST = CREQ_QUERY_ROCE_CC_GEN1_RESP_SB_TLV_CNP_ECN_ECT_0
CMDQ_MODIFY_ROCE_CC_OPCODE_MODIFY_ROCE_CC = 0x8c
CMDQ_MODIFY_ROCE_CC_OPCODE_LAST = CMDQ_MODIFY_ROCE_CC_OPCODE_MODIFY_ROCE_CC
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_G = 0x2
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_NUMPHASEPERSTATE = 0x4
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_INIT_CR = 0x8
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_INIT_TR = 0x10
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TOS_ECN = 0x20
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TOS_DSCP = 0x40
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_ALT_VLAN_PCP = 0x80
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_ALT_TOS_DSCP = 0x100
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_RTT = 0x200
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_CC_MODE = 0x400
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TCP_CP = 0x800
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TX_QUEUE = 0x1000
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_INACTIVITY_CP = 0x2000
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_TIME_PER_PHASE = 0x4000
CMDQ_MODIFY_ROCE_CC_MODIFY_MASK_PKTS_PER_PHASE = 0x8000
CMDQ_MODIFY_ROCE_CC_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_RSVD1_MASK = 0xfe
CMDQ_MODIFY_ROCE_CC_RSVD1_SFT = 1
CMDQ_MODIFY_ROCE_CC_TOS_ECN_MASK = 0x3
CMDQ_MODIFY_ROCE_CC_TOS_ECN_SFT = 0
CMDQ_MODIFY_ROCE_CC_TOS_DSCP_MASK = 0xfc
CMDQ_MODIFY_ROCE_CC_TOS_DSCP_SFT = 2
CMDQ_MODIFY_ROCE_CC_ALT_VLAN_PCP_MASK = 0x7
CMDQ_MODIFY_ROCE_CC_ALT_VLAN_PCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD3_MASK = 0xf8
CMDQ_MODIFY_ROCE_CC_RSVD3_SFT = 3
CMDQ_MODIFY_ROCE_CC_ALT_TOS_DSCP_MASK = 0x3f
CMDQ_MODIFY_ROCE_CC_ALT_TOS_DSCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD4_MASK = 0xffc0
CMDQ_MODIFY_ROCE_CC_RSVD4_SFT = 6
CMDQ_MODIFY_ROCE_CC_RTT_MASK = 0x3fff
CMDQ_MODIFY_ROCE_CC_RTT_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD5_MASK = 0xc000
CMDQ_MODIFY_ROCE_CC_RSVD5_SFT = 14
CMDQ_MODIFY_ROCE_CC_TCP_CP_MASK = 0x3ff
CMDQ_MODIFY_ROCE_CC_TCP_CP_SFT = 0
CMDQ_MODIFY_ROCE_CC_RSVD6_MASK = 0xfc00
CMDQ_MODIFY_ROCE_CC_RSVD6_SFT = 10
CMDQ_MODIFY_ROCE_CC_CC_MODE_DCTCP_CC_MODE = 0x0
CMDQ_MODIFY_ROCE_CC_CC_MODE_PROBABILISTIC_CC_MODE = 0x1
CMDQ_MODIFY_ROCE_CC_CC_MODE_LAST = CMDQ_MODIFY_ROCE_CC_CC_MODE_PROBABILISTIC_CC_MODE
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_MORE = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_MORE_LAST = 0x0
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED = 0x2
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_LAST = CMDQ_MODIFY_ROCE_CC_TLV_TLV_FLAGS_REQUIRED_YES
CMDQ_MODIFY_ROCE_CC_TLV_OPCODE_MODIFY_ROCE_CC = 0x8c
CMDQ_MODIFY_ROCE_CC_TLV_OPCODE_LAST = CMDQ_MODIFY_ROCE_CC_TLV_OPCODE_MODIFY_ROCE_CC
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_G = 0x2
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_NUMPHASEPERSTATE = 0x4
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_INIT_CR = 0x8
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_INIT_TR = 0x10
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TOS_ECN = 0x20
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TOS_DSCP = 0x40
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_ALT_VLAN_PCP = 0x80
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_ALT_TOS_DSCP = 0x100
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_RTT = 0x200
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_CC_MODE = 0x400
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TCP_CP = 0x800
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TX_QUEUE = 0x1000
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_INACTIVITY_CP = 0x2000
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_TIME_PER_PHASE = 0x4000
CMDQ_MODIFY_ROCE_CC_TLV_MODIFY_MASK_PKTS_PER_PHASE = 0x8000
CMDQ_MODIFY_ROCE_CC_TLV_ENABLE_CC = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_RSVD1_MASK = 0xfe
CMDQ_MODIFY_ROCE_CC_TLV_RSVD1_SFT = 1
CMDQ_MODIFY_ROCE_CC_TLV_TOS_ECN_MASK = 0x3
CMDQ_MODIFY_ROCE_CC_TLV_TOS_ECN_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_TOS_DSCP_MASK = 0xfc
CMDQ_MODIFY_ROCE_CC_TLV_TOS_DSCP_SFT = 2
CMDQ_MODIFY_ROCE_CC_TLV_ALT_VLAN_PCP_MASK = 0x7
CMDQ_MODIFY_ROCE_CC_TLV_ALT_VLAN_PCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD3_MASK = 0xf8
CMDQ_MODIFY_ROCE_CC_TLV_RSVD3_SFT = 3
CMDQ_MODIFY_ROCE_CC_TLV_ALT_TOS_DSCP_MASK = 0x3f
CMDQ_MODIFY_ROCE_CC_TLV_ALT_TOS_DSCP_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD4_MASK = 0xffc0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD4_SFT = 6
CMDQ_MODIFY_ROCE_CC_TLV_RTT_MASK = 0x3fff
CMDQ_MODIFY_ROCE_CC_TLV_RTT_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD5_MASK = 0xc000
CMDQ_MODIFY_ROCE_CC_TLV_RSVD5_SFT = 14
CMDQ_MODIFY_ROCE_CC_TLV_TCP_CP_MASK = 0x3ff
CMDQ_MODIFY_ROCE_CC_TLV_TCP_CP_SFT = 0
CMDQ_MODIFY_ROCE_CC_TLV_RSVD6_MASK = 0xfc00
CMDQ_MODIFY_ROCE_CC_TLV_RSVD6_SFT = 10
CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_DCTCP_CC_MODE = 0x0
CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_PROBABILISTIC_CC_MODE = 0x1
CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_LAST = CMDQ_MODIFY_ROCE_CC_TLV_CC_MODE_PROBABILISTIC_CC_MODE
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_MORE = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_MORE_LAST = 0x0
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_MORE_NOT_LAST = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED = 0x2
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_NO = (0x0 << 1)
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_YES = (0x1 << 1)
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_LAST = CMDQ_MODIFY_ROCE_CC_GEN1_TLV_TLV_FLAGS_REQUIRED_YES
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_MIN_TIME_BETWEEN_CNPS = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_INIT_CP = 0x2
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_UPDATE_MODE = 0x4
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_UPDATE_CYCLES = 0x8
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_FR_NUM_RTTS = 0x10
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_AI_RATE_INCREASE = 0x20
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_REDUCTION_RELAX_RTTS_TH = 0x40
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ADDITIONAL_RELAX_CR_TH = 0x80
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CR_MIN_TH = 0x100
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_BW_AVG_WEIGHT = 0x200
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ACTUAL_CR_FACTOR = 0x400
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_MAX_CP_CR_TH = 0x800
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CP_BIAS_EN = 0x1000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CP_BIAS = 0x2000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CNP_ECN = 0x4000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RTT_JITTER_EN = 0x8000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_LINK_BYTES_PER_USEC = 0x10000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RESET_CC_CR_TH = 0x20000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CR_WIDTH = 0x40000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_MIN = 0x80000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_MAX = 0x100000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_ABS_MAX = 0x200000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_LOWER_BOUND = 0x400000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CR_PROB_FACTOR = 0x800000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_TR_PROB_FACTOR = 0x1000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_FAIRNESS_CR_TH = 0x2000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RED_DIV = 0x4000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CNP_RATIO_TH = 0x8000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_EXP_AI_RTTS = 0x10000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_EXP_AI_CR_CP_RATIO = 0x20000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CP_EXP_UPDATE_TH = 0x40000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_HIGH_EXP_AI_RTTS_TH1 = 0x80000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_HIGH_EXP_AI_RTTS_TH2 = 0x100000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_USE_RATE_TABLE = 0x200000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_LINK64B_PER_RTT = 0x400000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ACTUAL_CR_CONG_FREE_RTTS_TH = 0x800000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_SEVERE_CONG_CR_TH1 = 0x1000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_SEVERE_CONG_CR_TH2 = 0x2000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_CC_ACK_BYTES = 0x4000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_REDUCE_INIT_EN = 0x8000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_REDUCE_INIT_CONG_FREE_RTTS_TH = 0x10000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_RANDOM_NO_RED_EN = 0x20000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_ACTUAL_CR_SHIFT_CORRECTION_EN = 0x40000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_MODIFY_MASK_QUOTA_PERIOD_ADJUST_EN = 0x80000000000
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_NOT_ECT = 0x0
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_ECT_1 = 0x1
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_ECT_0 = 0x2
CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_LAST = CMDQ_MODIFY_ROCE_CC_GEN1_TLV_CNP_ECN_ECT_0
CREQ_MODIFY_ROCE_CC_RESP_TYPE_MASK = 0x3f
CREQ_MODIFY_ROCE_CC_RESP_TYPE_SFT = 0
CREQ_MODIFY_ROCE_CC_RESP_TYPE_QP_EVENT = 0x38
CREQ_MODIFY_ROCE_CC_RESP_TYPE_LAST = CREQ_MODIFY_ROCE_CC_RESP_TYPE_QP_EVENT
CREQ_MODIFY_ROCE_CC_RESP_V = 0x1
CREQ_MODIFY_ROCE_CC_RESP_EVENT_MODIFY_ROCE_CC = 0x8c
CREQ_MODIFY_ROCE_CC_RESP_EVENT_LAST = CREQ_MODIFY_ROCE_CC_RESP_EVENT_MODIFY_ROCE_CC
CMDQ_SET_LINK_AGGR_MODE_OPCODE_SET_LINK_AGGR_MODE = 0x8f
CMDQ_SET_LINK_AGGR_MODE_OPCODE_LAST = CMDQ_SET_LINK_AGGR_MODE_OPCODE_SET_LINK_AGGR_MODE
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_AGGR_EN = 0x1
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_ACTIVE_PORT_MAP = 0x2
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_MEMBER_PORT_MAP = 0x4
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_AGGR_MODE = 0x8
CMDQ_SET_LINK_AGGR_MODE_MODIFY_MASK_STAT_CTX_ID = 0x10
CMDQ_SET_LINK_AGGR_MODE_AGGR_ENABLE = 0x1
CMDQ_SET_LINK_AGGR_MODE_RSVD1_MASK = 0xfe
CMDQ_SET_LINK_AGGR_MODE_RSVD1_SFT = 1
CMDQ_SET_LINK_AGGR_MODE_ACTIVE_PORT_MAP_MASK = 0xf
CMDQ_SET_LINK_AGGR_MODE_ACTIVE_PORT_MAP_SFT = 0
CMDQ_SET_LINK_AGGR_MODE_RSVD2_MASK = 0xf0
CMDQ_SET_LINK_AGGR_MODE_RSVD2_SFT = 4
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_ACTIVE_ACTIVE = 0x1
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_ACTIVE_BACKUP = 0x2
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_BALANCE_XOR = 0x3
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_802_3_AD = 0x4
CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_LAST = CMDQ_SET_LINK_AGGR_MODE_AGGR_MODE_802_3_AD
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_MASK = 0x3f
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_SFT = 0
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_QP_EVENT = 0x38
CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_LAST = CREQ_SET_LINK_AGGR_MODE_RESP_TYPE_QP_EVENT
CREQ_SET_LINK_AGGR_MODE_RESP_V = 0x1
CREQ_SET_LINK_AGGR_MODE_RESP_EVENT_SET_LINK_AGGR_MODE = 0x8f
CREQ_SET_LINK_AGGR_MODE_RESP_EVENT_LAST = CREQ_SET_LINK_AGGR_MODE_RESP_EVENT_SET_LINK_AGGR_MODE
CREQ_FUNC_EVENT_TYPE_MASK = 0x3f
CREQ_FUNC_EVENT_TYPE_SFT = 0
CREQ_FUNC_EVENT_TYPE_FUNC_EVENT = 0x3a
CREQ_FUNC_EVENT_TYPE_LAST = CREQ_FUNC_EVENT_TYPE_FUNC_EVENT
CREQ_FUNC_EVENT_V = 0x1
CREQ_FUNC_EVENT_EVENT_TX_WQE_ERROR = 0x1
CREQ_FUNC_EVENT_EVENT_TX_DATA_ERROR = 0x2
CREQ_FUNC_EVENT_EVENT_RX_WQE_ERROR = 0x3
CREQ_FUNC_EVENT_EVENT_RX_DATA_ERROR = 0x4
CREQ_FUNC_EVENT_EVENT_CQ_ERROR = 0x5
CREQ_FUNC_EVENT_EVENT_TQM_ERROR = 0x6
CREQ_FUNC_EVENT_EVENT_CFCQ_ERROR = 0x7
CREQ_FUNC_EVENT_EVENT_CFCS_ERROR = 0x8
CREQ_FUNC_EVENT_EVENT_CFCC_ERROR = 0x9
CREQ_FUNC_EVENT_EVENT_CFCM_ERROR = 0xa
CREQ_FUNC_EVENT_EVENT_TIM_ERROR = 0xb
CREQ_FUNC_EVENT_EVENT_VF_COMM_REQUEST = 0x80
CREQ_FUNC_EVENT_EVENT_RESOURCE_EXHAUSTED = 0x81
CREQ_FUNC_EVENT_EVENT_LAST = CREQ_FUNC_EVENT_EVENT_RESOURCE_EXHAUSTED
CREQ_QP_EVENT_TYPE_MASK = 0x3f
CREQ_QP_EVENT_TYPE_SFT = 0
CREQ_QP_EVENT_TYPE_QP_EVENT = 0x38
CREQ_QP_EVENT_TYPE_LAST = CREQ_QP_EVENT_TYPE_QP_EVENT
CREQ_QP_EVENT_STATUS_SUCCESS = 0x0
CREQ_QP_EVENT_STATUS_FAIL = 0x1
CREQ_QP_EVENT_STATUS_RESOURCES = 0x2
CREQ_QP_EVENT_STATUS_INVALID_CMD = 0x3
CREQ_QP_EVENT_STATUS_NOT_IMPLEMENTED = 0x4
CREQ_QP_EVENT_STATUS_INVALID_PARAMETER = 0x5
CREQ_QP_EVENT_STATUS_HARDWARE_ERROR = 0x6
CREQ_QP_EVENT_STATUS_INTERNAL_ERROR = 0x7
CREQ_QP_EVENT_STATUS_LAST = CREQ_QP_EVENT_STATUS_INTERNAL_ERROR
CREQ_QP_EVENT_V = 0x1
CREQ_QP_EVENT_EVENT_CREATE_QP = 0x1
CREQ_QP_EVENT_EVENT_DESTROY_QP = 0x2
CREQ_QP_EVENT_EVENT_MODIFY_QP = 0x3
CREQ_QP_EVENT_EVENT_QUERY_QP = 0x4
CREQ_QP_EVENT_EVENT_CREATE_SRQ = 0x5
CREQ_QP_EVENT_EVENT_DESTROY_SRQ = 0x6
CREQ_QP_EVENT_EVENT_QUERY_SRQ = 0x8
CREQ_QP_EVENT_EVENT_CREATE_CQ = 0x9
CREQ_QP_EVENT_EVENT_DESTROY_CQ = 0xa
CREQ_QP_EVENT_EVENT_RESIZE_CQ = 0xc
CREQ_QP_EVENT_EVENT_ALLOCATE_MRW = 0xd
CREQ_QP_EVENT_EVENT_DEALLOCATE_KEY = 0xe
CREQ_QP_EVENT_EVENT_REGISTER_MR = 0xf
CREQ_QP_EVENT_EVENT_DEREGISTER_MR = 0x10
CREQ_QP_EVENT_EVENT_ADD_GID = 0x11
CREQ_QP_EVENT_EVENT_DELETE_GID = 0x12
CREQ_QP_EVENT_EVENT_MODIFY_GID = 0x17
CREQ_QP_EVENT_EVENT_QUERY_GID = 0x18
CREQ_QP_EVENT_EVENT_CREATE_QP1 = 0x13
CREQ_QP_EVENT_EVENT_DESTROY_QP1 = 0x14
CREQ_QP_EVENT_EVENT_CREATE_AH = 0x15
CREQ_QP_EVENT_EVENT_DESTROY_AH = 0x16
CREQ_QP_EVENT_EVENT_INITIALIZE_FW = 0x80
CREQ_QP_EVENT_EVENT_DEINITIALIZE_FW = 0x81
CREQ_QP_EVENT_EVENT_STOP_FUNC = 0x82
CREQ_QP_EVENT_EVENT_QUERY_FUNC = 0x83
CREQ_QP_EVENT_EVENT_SET_FUNC_RESOURCES = 0x84
CREQ_QP_EVENT_EVENT_READ_CONTEXT = 0x85
CREQ_QP_EVENT_EVENT_MAP_TC_TO_COS = 0x8a
CREQ_QP_EVENT_EVENT_QUERY_VERSION = 0x8b
CREQ_QP_EVENT_EVENT_MODIFY_CC = 0x8c
CREQ_QP_EVENT_EVENT_QUERY_CC = 0x8d
CREQ_QP_EVENT_EVENT_QUERY_ROCE_STATS = 0x8e
CREQ_QP_EVENT_EVENT_SET_LINK_AGGR_MODE = 0x8f
CREQ_QP_EVENT_EVENT_QUERY_QP_EXTEND = 0x91
CREQ_QP_EVENT_EVENT_QP_ERROR_NOTIFICATION = 0xc0
CREQ_QP_EVENT_EVENT_CQ_ERROR_NOTIFICATION = 0xc1
CREQ_QP_EVENT_EVENT_LAST = CREQ_QP_EVENT_EVENT_CQ_ERROR_NOTIFICATION
CREQ_QP_ERROR_NOTIFICATION_TYPE_MASK = 0x3f
CREQ_QP_ERROR_NOTIFICATION_TYPE_SFT = 0
CREQ_QP_ERROR_NOTIFICATION_TYPE_QP_EVENT = 0x38
CREQ_QP_ERROR_NOTIFICATION_TYPE_LAST = CREQ_QP_ERROR_NOTIFICATION_TYPE_QP_EVENT
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_NO_ERROR = 0X0
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_OPCODE_ERROR = 0X1
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_TIMEOUT_RETRY_LIMIT = 0X2
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RNR_TIMEOUT_RETRY_LIMIT = 0X3
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_1 = 0X4
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_2 = 0X5
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_3 = 0X6
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_NAK_ARRIVAL_4 = 0X7
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RX_MEMORY_ERROR = 0X8
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_TX_MEMORY_ERROR = 0X9
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_READ_RESP_LENGTH = 0XA
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_INVALID_READ_RESP = 0XB
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ILLEGAL_BIND = 0XC
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ILLEGAL_FAST_REG = 0XD
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ILLEGAL_INVALIDATE = 0XE
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_CMP_ERROR = 0XF
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RETRAN_LOCAL_ERROR = 0X10
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_WQE_FORMAT_ERROR = 0X11
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_ORRQ_FORMAT_ERROR = 0X12
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_INVALID_AVID_ERROR = 0X13
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_AV_DOMAIN_ERROR = 0X14
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_CQ_LOAD_ERROR = 0X15
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_SERV_TYPE_ERROR = 0X16
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_INVALID_OP_ERROR = 0X17
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_TX_PCI_ERROR = 0X18
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RX_PCI_ERROR = 0X19
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_PROD_WQE_MSMTCH_ERROR = 0X1A
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_PSN_RANGE_CHECK_ERROR = 0X1B
CREQ_QP_ERROR_NOTIFICATION_REQ_ERR_STATE_REASON_REQ_RETX_SETUP_ERROR = 0X1C
CREQ_QP_ERROR_NOTIFICATION_V = 0x1
CREQ_QP_ERROR_NOTIFICATION_EVENT_QP_ERROR_NOTIFICATION = 0xc0
CREQ_QP_ERROR_NOTIFICATION_EVENT_LAST = CREQ_QP_ERROR_NOTIFICATION_EVENT_QP_ERROR_NOTIFICATION
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_NO_ERROR = 0x0
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_EXCEED_MAX = 0x1
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_PAYLOAD_LENGTH_MISMATCH = 0x2
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_EXCEEDS_WQE = 0x3
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_OPCODE_ERROR = 0x4
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_PSN_SEQ_ERROR_RETRY_LIMIT = 0x5
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_INVALID_R_KEY = 0x6
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_DOMAIN_ERROR = 0x7
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_NO_PERMISSION = 0x8
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_RANGE_ERROR = 0x9
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_INVALID_R_KEY = 0xa
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_DOMAIN_ERROR = 0xb
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_NO_PERMISSION = 0xc
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_RANGE_ERROR = 0xd
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_IRRQ_OFLOW = 0xe
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_UNSUPPORTED_OPCODE = 0xf
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_UNALIGN_ATOMIC = 0x10
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_REM_INVALIDATE = 0x11
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_MEMORY_ERROR = 0x12
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_SRQ_ERROR = 0x13
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_CMP_ERROR = 0x14
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_INVALID_DUP_RKEY = 0x15
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_WQE_FORMAT_ERROR = 0x16
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_IRRQ_FORMAT_ERROR = 0x17
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_CQ_LOAD_ERROR = 0x18
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_SRQ_LOAD_ERROR = 0x19
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_TX_PCI_ERROR = 0x1b
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_RX_PCI_ERROR = 0x1c
CREQ_QP_ERROR_NOTIFICATION_RES_ERR_STATE_REASON_RES_PSN_NOT_FOUND = 0x1d
CREQ_CQ_ERROR_NOTIFICATION_TYPE_MASK = 0x3f
CREQ_CQ_ERROR_NOTIFICATION_TYPE_SFT = 0
CREQ_CQ_ERROR_NOTIFICATION_TYPE_CQ_EVENT = 0x38
CREQ_CQ_ERROR_NOTIFICATION_TYPE_LAST = CREQ_CQ_ERROR_NOTIFICATION_TYPE_CQ_EVENT
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_REQ_CQ_INVALID_ERROR = 0x1
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_REQ_CQ_OVERFLOW_ERROR = 0x2
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_REQ_CQ_LOAD_ERROR = 0x3
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_INVALID_ERROR = 0x4
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_OVERFLOW_ERROR = 0x5
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_LOAD_ERROR = 0x6
CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_LAST = CREQ_CQ_ERROR_NOTIFICATION_CQ_ERR_REASON_RES_CQ_LOAD_ERROR
CREQ_CQ_ERROR_NOTIFICATION_V = 0x1
CREQ_CQ_ERROR_NOTIFICATION_EVENT_CQ_ERROR_NOTIFICATION = 0xc1
CREQ_CQ_ERROR_NOTIFICATION_EVENT_LAST = CREQ_CQ_ERROR_NOTIFICATION_EVENT_CQ_ERROR_NOTIFICATION
SQ_BASE_WQE_TYPE_SEND = 0x0
SQ_BASE_WQE_TYPE_SEND_W_IMMEAD = 0x1
SQ_BASE_WQE_TYPE_SEND_W_INVALID = 0x2
SQ_BASE_WQE_TYPE_WRITE_WQE = 0x4
SQ_BASE_WQE_TYPE_WRITE_W_IMMEAD = 0x5
SQ_BASE_WQE_TYPE_READ_WQE = 0x6
SQ_BASE_WQE_TYPE_ATOMIC_CS = 0x8
SQ_BASE_WQE_TYPE_ATOMIC_FA = 0xb
SQ_BASE_WQE_TYPE_LOCAL_INVALID = 0xc
SQ_BASE_WQE_TYPE_FR_PMR = 0xd
SQ_BASE_WQE_TYPE_BIND = 0xe
SQ_BASE_WQE_TYPE_FR_PPMR = 0xf
SQ_BASE_WQE_TYPE_LAST = SQ_BASE_WQE_TYPE_FR_PPMR
SQ_PSN_SEARCH_START_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_START_PSN_SFT = 0
SQ_PSN_SEARCH_OPCODE_MASK = 0xff000000
SQ_PSN_SEARCH_OPCODE_SFT = 24
SQ_PSN_SEARCH_NEXT_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_NEXT_PSN_SFT = 0
SQ_PSN_SEARCH_FLAGS_MASK = 0xff000000
SQ_PSN_SEARCH_FLAGS_SFT = 24
SQ_PSN_SEARCH_EXT_START_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_EXT_START_PSN_SFT = 0
SQ_PSN_SEARCH_EXT_OPCODE_MASK = 0xff000000
SQ_PSN_SEARCH_EXT_OPCODE_SFT = 24
SQ_PSN_SEARCH_EXT_NEXT_PSN_MASK = 0xffffff
SQ_PSN_SEARCH_EXT_NEXT_PSN_SFT = 0
SQ_PSN_SEARCH_EXT_FLAGS_MASK = 0xff000000
SQ_PSN_SEARCH_EXT_FLAGS_SFT = 24
SQ_MSN_SEARCH_START_PSN_MASK = 0xffffff
SQ_MSN_SEARCH_START_PSN_SFT = 0
SQ_MSN_SEARCH_NEXT_PSN_MASK = 0xffffff000000
SQ_MSN_SEARCH_NEXT_PSN_SFT = 24
SQ_MSN_SEARCH_START_IDX_MASK = 0xffff000000000000
SQ_MSN_SEARCH_START_IDX_SFT = 48
SQ_SEND_WQE_TYPE_SEND = 0x0
SQ_SEND_WQE_TYPE_SEND_W_IMMEAD = 0x1
SQ_SEND_WQE_TYPE_SEND_W_INVALID = 0x2
SQ_SEND_WQE_TYPE_LAST = SQ_SEND_WQE_TYPE_SEND_W_INVALID
SQ_SEND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_FLAGS_UC_FENCE = 0x4
SQ_SEND_FLAGS_SE = 0x8
SQ_SEND_FLAGS_INLINE = 0x10
SQ_SEND_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_DST_QP_MASK = 0xffffff
SQ_SEND_DST_QP_SFT = 0
SQ_SEND_AVID_MASK = 0xfffff
SQ_SEND_AVID_SFT = 0
SQ_SEND_TIMESTAMP_MASK = 0xffffff
SQ_SEND_TIMESTAMP_SFT = 0
SQ_SEND_HDR_WQE_TYPE_SEND = 0x0
SQ_SEND_HDR_WQE_TYPE_SEND_W_IMMEAD = 0x1
SQ_SEND_HDR_WQE_TYPE_SEND_W_INVALID = 0x2
SQ_SEND_HDR_WQE_TYPE_LAST = SQ_SEND_HDR_WQE_TYPE_SEND_W_INVALID
SQ_SEND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_HDR_FLAGS_UC_FENCE = 0x4
SQ_SEND_HDR_FLAGS_SE = 0x8
SQ_SEND_HDR_FLAGS_INLINE = 0x10
SQ_SEND_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_HDR_DST_QP_MASK = 0xffffff
SQ_SEND_HDR_DST_QP_SFT = 0
SQ_SEND_HDR_AVID_MASK = 0xfffff
SQ_SEND_HDR_AVID_SFT = 0
SQ_SEND_HDR_TIMESTAMP_MASK = 0xffffff
SQ_SEND_HDR_TIMESTAMP_SFT = 0
SQ_SEND_RAWETH_QP1_WQE_TYPE_SEND = 0x0
SQ_SEND_RAWETH_QP1_WQE_TYPE_LAST = SQ_SEND_RAWETH_QP1_WQE_TYPE_SEND
SQ_SEND_RAWETH_QP1_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_RAWETH_QP1_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_RAWETH_QP1_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_RAWETH_QP1_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_RAWETH_QP1_FLAGS_UC_FENCE = 0x4
SQ_SEND_RAWETH_QP1_FLAGS_SE = 0x8
SQ_SEND_RAWETH_QP1_FLAGS_INLINE = 0x10
SQ_SEND_RAWETH_QP1_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_RAWETH_QP1_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_RAWETH_QP1_LFLAGS_TCP_UDP_CHKSUM = 0x1
SQ_SEND_RAWETH_QP1_LFLAGS_IP_CHKSUM = 0x2
SQ_SEND_RAWETH_QP1_LFLAGS_NOCRC = 0x4
SQ_SEND_RAWETH_QP1_LFLAGS_STAMP = 0x8
SQ_SEND_RAWETH_QP1_LFLAGS_T_IP_CHKSUM = 0x10
SQ_SEND_RAWETH_QP1_LFLAGS_ROCE_CRC = 0x100
SQ_SEND_RAWETH_QP1_LFLAGS_FCOE_CRC = 0x200
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_VID_MASK = 0xfff
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_VID_SFT = 0
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_DE = 0x1000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_PRI_MASK = 0xe000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_PRI_SFT = 13
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_MASK = 0x70000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_SFT = 16
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID88A8 = (0x0 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID8100 = (0x1 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID9100 = (0x2 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID9200 = (0x3 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPID9300 = (0x4 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPIDCFG = (0x5 << 16)
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_LAST = SQ_SEND_RAWETH_QP1_CFA_META_VLAN_TPID_TPIDCFG
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_RESERVED_MASK = 0xff80000
SQ_SEND_RAWETH_QP1_CFA_META_VLAN_RESERVED_SFT = 19
SQ_SEND_RAWETH_QP1_CFA_META_KEY_MASK = 0xf0000000
SQ_SEND_RAWETH_QP1_CFA_META_KEY_SFT = 28
SQ_SEND_RAWETH_QP1_CFA_META_KEY_NONE = (0x0 << 28)
SQ_SEND_RAWETH_QP1_CFA_META_KEY_VLAN_TAG = (0x1 << 28)
SQ_SEND_RAWETH_QP1_CFA_META_KEY_LAST = SQ_SEND_RAWETH_QP1_CFA_META_KEY_VLAN_TAG
SQ_SEND_RAWETH_QP1_TIMESTAMP_MASK = 0xffffff
SQ_SEND_RAWETH_QP1_TIMESTAMP_SFT = 0
SQ_SEND_RAWETH_QP1_HDR_WQE_TYPE_SEND = 0x0
SQ_SEND_RAWETH_QP1_HDR_WQE_TYPE_LAST = SQ_SEND_RAWETH_QP1_HDR_WQE_TYPE_SEND
SQ_SEND_RAWETH_QP1_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_SEND_RAWETH_QP1_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_SEND_RAWETH_QP1_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_SEND_RAWETH_QP1_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_SEND_RAWETH_QP1_HDR_FLAGS_UC_FENCE = 0x4
SQ_SEND_RAWETH_QP1_HDR_FLAGS_SE = 0x8
SQ_SEND_RAWETH_QP1_HDR_FLAGS_INLINE = 0x10
SQ_SEND_RAWETH_QP1_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_SEND_RAWETH_QP1_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_TCP_UDP_CHKSUM = 0x1
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_IP_CHKSUM = 0x2
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_NOCRC = 0x4
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_STAMP = 0x8
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_T_IP_CHKSUM = 0x10
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_ROCE_CRC = 0x100
SQ_SEND_RAWETH_QP1_HDR_LFLAGS_FCOE_CRC = 0x200
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_VID_MASK = 0xfff
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_VID_SFT = 0
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_DE = 0x1000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_PRI_MASK = 0xe000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_PRI_SFT = 13
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_MASK = 0x70000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_SFT = 16
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID88A8 = (0x0 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID8100 = (0x1 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID9100 = (0x2 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID9200 = (0x3 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPID9300 = (0x4 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPIDCFG = (0x5 << 16)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_LAST = SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_TPID_TPIDCFG
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_RESERVED_MASK = 0xff80000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_VLAN_RESERVED_SFT = 19
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_MASK = 0xf0000000
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_SFT = 28
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_NONE = (0x0 << 28)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_VLAN_TAG = (0x1 << 28)
SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_LAST = SQ_SEND_RAWETH_QP1_HDR_CFA_META_KEY_VLAN_TAG
SQ_SEND_RAWETH_QP1_HDR_TIMESTAMP_MASK = 0xffffff
SQ_SEND_RAWETH_QP1_HDR_TIMESTAMP_SFT = 0
SQ_RDMA_WQE_TYPE_WRITE_WQE = 0x4
SQ_RDMA_WQE_TYPE_WRITE_W_IMMEAD = 0x5
SQ_RDMA_WQE_TYPE_READ_WQE = 0x6
SQ_RDMA_WQE_TYPE_LAST = SQ_RDMA_WQE_TYPE_READ_WQE
SQ_RDMA_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_RDMA_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_RDMA_FLAGS_SIGNAL_COMP = 0x1
SQ_RDMA_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_RDMA_FLAGS_UC_FENCE = 0x4
SQ_RDMA_FLAGS_SE = 0x8
SQ_RDMA_FLAGS_INLINE = 0x10
SQ_RDMA_FLAGS_WQE_TS_EN = 0x20
SQ_RDMA_FLAGS_DEBUG_TRACE = 0x40
SQ_RDMA_TIMESTAMP_MASK = 0xffffff
SQ_RDMA_TIMESTAMP_SFT = 0
SQ_RDMA_HDR_WQE_TYPE_WRITE_WQE = 0x4
SQ_RDMA_HDR_WQE_TYPE_WRITE_W_IMMEAD = 0x5
SQ_RDMA_HDR_WQE_TYPE_READ_WQE = 0x6
SQ_RDMA_HDR_WQE_TYPE_LAST = SQ_RDMA_HDR_WQE_TYPE_READ_WQE
SQ_RDMA_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_RDMA_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_RDMA_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_RDMA_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_RDMA_HDR_FLAGS_UC_FENCE = 0x4
SQ_RDMA_HDR_FLAGS_SE = 0x8
SQ_RDMA_HDR_FLAGS_INLINE = 0x10
SQ_RDMA_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_RDMA_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_RDMA_HDR_TIMESTAMP_MASK = 0xffffff
SQ_RDMA_HDR_TIMESTAMP_SFT = 0
SQ_ATOMIC_WQE_TYPE_ATOMIC_CS = 0x8
SQ_ATOMIC_WQE_TYPE_ATOMIC_FA = 0xb
SQ_ATOMIC_WQE_TYPE_LAST = SQ_ATOMIC_WQE_TYPE_ATOMIC_FA
SQ_ATOMIC_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_ATOMIC_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_ATOMIC_FLAGS_SIGNAL_COMP = 0x1
SQ_ATOMIC_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_ATOMIC_FLAGS_UC_FENCE = 0x4
SQ_ATOMIC_FLAGS_SE = 0x8
SQ_ATOMIC_FLAGS_INLINE = 0x10
SQ_ATOMIC_FLAGS_WQE_TS_EN = 0x20
SQ_ATOMIC_FLAGS_DEBUG_TRACE = 0x40
SQ_ATOMIC_HDR_WQE_TYPE_ATOMIC_CS = 0x8
SQ_ATOMIC_HDR_WQE_TYPE_ATOMIC_FA = 0xb
SQ_ATOMIC_HDR_WQE_TYPE_LAST = SQ_ATOMIC_HDR_WQE_TYPE_ATOMIC_FA
SQ_ATOMIC_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_ATOMIC_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_ATOMIC_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_ATOMIC_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_ATOMIC_HDR_FLAGS_UC_FENCE = 0x4
SQ_ATOMIC_HDR_FLAGS_SE = 0x8
SQ_ATOMIC_HDR_FLAGS_INLINE = 0x10
SQ_ATOMIC_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_ATOMIC_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_LOCALINVALIDATE_WQE_TYPE_LOCAL_INVALID = 0xc
SQ_LOCALINVALIDATE_WQE_TYPE_LAST = SQ_LOCALINVALIDATE_WQE_TYPE_LOCAL_INVALID
SQ_LOCALINVALIDATE_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_LOCALINVALIDATE_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_LOCALINVALIDATE_FLAGS_SIGNAL_COMP = 0x1
SQ_LOCALINVALIDATE_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_LOCALINVALIDATE_FLAGS_UC_FENCE = 0x4
SQ_LOCALINVALIDATE_FLAGS_SE = 0x8
SQ_LOCALINVALIDATE_FLAGS_INLINE = 0x10
SQ_LOCALINVALIDATE_FLAGS_WQE_TS_EN = 0x20
SQ_LOCALINVALIDATE_FLAGS_DEBUG_TRACE = 0x40
SQ_LOCALINVALIDATE_HDR_WQE_TYPE_LOCAL_INVALID = 0xc
SQ_LOCALINVALIDATE_HDR_WQE_TYPE_LAST = SQ_LOCALINVALIDATE_HDR_WQE_TYPE_LOCAL_INVALID
SQ_LOCALINVALIDATE_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_LOCALINVALIDATE_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_LOCALINVALIDATE_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_LOCALINVALIDATE_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_LOCALINVALIDATE_HDR_FLAGS_UC_FENCE = 0x4
SQ_LOCALINVALIDATE_HDR_FLAGS_SE = 0x8
SQ_LOCALINVALIDATE_HDR_FLAGS_INLINE = 0x10
SQ_LOCALINVALIDATE_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_LOCALINVALIDATE_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_FR_PMR_WQE_TYPE_FR_PMR = 0xd
SQ_FR_PMR_WQE_TYPE_LAST = SQ_FR_PMR_WQE_TYPE_FR_PMR
SQ_FR_PMR_FLAGS_SIGNAL_COMP = 0x1
SQ_FR_PMR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_FR_PMR_FLAGS_UC_FENCE = 0x4
SQ_FR_PMR_FLAGS_SE = 0x8
SQ_FR_PMR_FLAGS_INLINE = 0x10
SQ_FR_PMR_FLAGS_WQE_TS_EN = 0x20
SQ_FR_PMR_FLAGS_DEBUG_TRACE = 0x40
SQ_FR_PMR_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_FR_PMR_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_FR_PMR_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_FR_PMR_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_FR_PMR_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_FR_PMR_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_ZERO_BASED = 0x20
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_PBL_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_PBL_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_NUMLEVELS_MASK = 0xc0
SQ_FR_PMR_NUMLEVELS_SFT = 6
SQ_FR_PMR_NUMLEVELS_PHYSICAL = (0x0 << 6)
SQ_FR_PMR_NUMLEVELS_LAYER1 = (0x1 << 6)
SQ_FR_PMR_NUMLEVELS_LAYER2 = (0x2 << 6)
SQ_FR_PMR_NUMLEVELS_LAST = SQ_FR_PMR_NUMLEVELS_LAYER2
SQ_FR_PMR_HDR_WQE_TYPE_FR_PMR = 0xd
SQ_FR_PMR_HDR_WQE_TYPE_LAST = SQ_FR_PMR_HDR_WQE_TYPE_FR_PMR
SQ_FR_PMR_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_FR_PMR_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_FR_PMR_HDR_FLAGS_UC_FENCE = 0x4
SQ_FR_PMR_HDR_FLAGS_SE = 0x8
SQ_FR_PMR_HDR_FLAGS_INLINE = 0x10
SQ_FR_PMR_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_FR_PMR_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_FR_PMR_HDR_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_FR_PMR_HDR_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_FR_PMR_HDR_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_FR_PMR_HDR_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_FR_PMR_HDR_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_HDR_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_HDR_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_HDR_ZERO_BASED = 0x20
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_MASK = 0x1f
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_SFT = 0
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4K = 0x0
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8K = 0x1
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_16K = 0x2
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_32K = 0x3
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_64K = 0x4
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_128K = 0x5
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_256K = 0x6
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_512K = 0x7
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_1M = 0x8
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_2M = 0x9
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4M = 0xa
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8M = 0xb
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_16M = 0xc
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_32M = 0xd
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_64M = 0xe
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_128M = 0xf
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_256M = 0x10
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_512M = 0x11
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_1G = 0x12
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_2G = 0x13
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4G = 0x14
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8G = 0x15
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_16G = 0x16
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_32G = 0x17
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_64G = 0x18
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_128G = 0x19
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_256G = 0x1a
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_512G = 0x1b
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_1T = 0x1c
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_2T = 0x1d
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_4T = 0x1e
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8T = 0x1f
SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_LAST = SQ_FR_PMR_HDR_PBL_PAGE_SIZE_LOG_PGSZ_8T
SQ_FR_PMR_HDR_NUMLEVELS_MASK = 0xc0
SQ_FR_PMR_HDR_NUMLEVELS_SFT = 6
SQ_FR_PMR_HDR_NUMLEVELS_PHYSICAL = (0x0 << 6)
SQ_FR_PMR_HDR_NUMLEVELS_LAYER1 = (0x1 << 6)
SQ_FR_PMR_HDR_NUMLEVELS_LAYER2 = (0x2 << 6)
SQ_FR_PMR_HDR_NUMLEVELS_LAST = SQ_FR_PMR_HDR_NUMLEVELS_LAYER2
SQ_BIND_WQE_TYPE_BIND = 0xe
SQ_BIND_WQE_TYPE_LAST = SQ_BIND_WQE_TYPE_BIND
SQ_BIND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_BIND_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_BIND_FLAGS_SIGNAL_COMP = 0x1
SQ_BIND_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_BIND_FLAGS_UC_FENCE = 0x4
SQ_BIND_FLAGS_SE = 0x8
SQ_BIND_FLAGS_INLINE = 0x10
SQ_BIND_FLAGS_WQE_TS_EN = 0x20
SQ_BIND_FLAGS_DEBUG_TRACE = 0x40
SQ_BIND_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_MASK = 0xff
SQ_BIND_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_SFT = 0
SQ_BIND_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_BIND_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_BIND_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_BIND_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_BIND_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_BIND_ZERO_BASED = 0x1
SQ_BIND_MW_TYPE = 0x2
SQ_BIND_MW_TYPE_TYPE1 = (0x0 << 1)
SQ_BIND_MW_TYPE_TYPE2 = (0x1 << 1)
SQ_BIND_MW_TYPE_LAST = SQ_BIND_MW_TYPE_TYPE2
SQ_BIND_HDR_WQE_TYPE_BIND = 0xe
SQ_BIND_HDR_WQE_TYPE_LAST = SQ_BIND_HDR_WQE_TYPE_BIND
SQ_BIND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_MASK = 0xff
SQ_BIND_HDR_FLAGS_INLINE_SE_UC_FENCE_RD_OR_ATOMIC_FENCE_SIGNAL_COMP_SFT = 0
SQ_BIND_HDR_FLAGS_SIGNAL_COMP = 0x1
SQ_BIND_HDR_FLAGS_RD_OR_ATOMIC_FENCE = 0x2
SQ_BIND_HDR_FLAGS_UC_FENCE = 0x4
SQ_BIND_HDR_FLAGS_SE = 0x8
SQ_BIND_HDR_FLAGS_INLINE = 0x10
SQ_BIND_HDR_FLAGS_WQE_TS_EN = 0x20
SQ_BIND_HDR_FLAGS_DEBUG_TRACE = 0x40
SQ_BIND_HDR_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_MASK = 0xff
SQ_BIND_HDR_ACCESS_CNTL_WINDOW_BIND_REMOTE_ATOMIC_REMOTE_WRITE_REMOTE_READ_LOCAL_WRITE_SFT = 0
SQ_BIND_HDR_ACCESS_CNTL_LOCAL_WRITE = 0x1
SQ_BIND_HDR_ACCESS_CNTL_REMOTE_READ = 0x2
SQ_BIND_HDR_ACCESS_CNTL_REMOTE_WRITE = 0x4
SQ_BIND_HDR_ACCESS_CNTL_REMOTE_ATOMIC = 0x8
SQ_BIND_HDR_ACCESS_CNTL_WINDOW_BIND = 0x10
SQ_BIND_HDR_ZERO_BASED = 0x1
SQ_BIND_HDR_MW_TYPE = 0x2
SQ_BIND_HDR_MW_TYPE_TYPE1 = (0x0 << 1)
SQ_BIND_HDR_MW_TYPE_TYPE2 = (0x1 << 1)
SQ_BIND_HDR_MW_TYPE_LAST = SQ_BIND_HDR_MW_TYPE_TYPE2
RQ_WQE_WQE_TYPE_RCV = 0x80
RQ_WQE_WQE_TYPE_LAST = RQ_WQE_WQE_TYPE_RCV
RQ_WQE_WR_ID_MASK = 0xfffff
RQ_WQE_WR_ID_SFT = 0
RQ_WQE_HDR_WQE_TYPE_RCV = 0x80
RQ_WQE_HDR_WQE_TYPE_LAST = RQ_WQE_HDR_WQE_TYPE_RCV
RQ_WQE_HDR_WR_ID_MASK = 0xfffff
RQ_WQE_HDR_WR_ID_SFT = 0
CQ_BASE_TOGGLE = 0x1
CQ_BASE_CQE_TYPE_MASK = 0x1e
CQ_BASE_CQE_TYPE_SFT = 1
CQ_BASE_CQE_TYPE_REQ = (0x0 << 1)
CQ_BASE_CQE_TYPE_RES_RC = (0x1 << 1)
CQ_BASE_CQE_TYPE_RES_UD = (0x2 << 1)
CQ_BASE_CQE_TYPE_RES_RAWETH_QP1 = (0x3 << 1)
CQ_BASE_CQE_TYPE_RES_UD_CFA = (0x4 << 1)
CQ_BASE_CQE_TYPE_REQ_V3 = (0x8 << 1)
CQ_BASE_CQE_TYPE_RES_RC_V3 = (0x9 << 1)
CQ_BASE_CQE_TYPE_RES_UD_V3 = (0xa << 1)
CQ_BASE_CQE_TYPE_RES_RAWETH_QP1_V3 = (0xb << 1)
CQ_BASE_CQE_TYPE_RES_UD_CFA_V3 = (0xc << 1)
CQ_BASE_CQE_TYPE_NO_OP = (0xd << 1)
CQ_BASE_CQE_TYPE_TERMINAL = (0xe << 1)
CQ_BASE_CQE_TYPE_CUT_OFF = (0xf << 1)
CQ_BASE_CQE_TYPE_LAST = CQ_BASE_CQE_TYPE_CUT_OFF
CQ_BASE_STATUS_OK = 0x0
CQ_BASE_STATUS_BAD_RESPONSE_ERR = 0x1
CQ_BASE_STATUS_LOCAL_LENGTH_ERR = 0x2
CQ_BASE_STATUS_HW_LOCAL_LENGTH_ERR = 0x3
CQ_BASE_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_BASE_STATUS_LOCAL_PROTECTION_ERR = 0x5
CQ_BASE_STATUS_LOCAL_ACCESS_ERROR = 0x6
CQ_BASE_STATUS_MEMORY_MGT_OPERATION_ERR = 0x7
CQ_BASE_STATUS_REMOTE_INVALID_REQUEST_ERR = 0x8
CQ_BASE_STATUS_REMOTE_ACCESS_ERR = 0x9
CQ_BASE_STATUS_REMOTE_OPERATION_ERR = 0xa
CQ_BASE_STATUS_RNR_NAK_RETRY_CNT_ERR = 0xb
CQ_BASE_STATUS_TRANSPORT_RETRY_CNT_ERR = 0xc
CQ_BASE_STATUS_WORK_REQUEST_FLUSHED_ERR = 0xd
CQ_BASE_STATUS_HW_FLUSH_ERR = 0xe
CQ_BASE_STATUS_OVERFLOW_ERR = 0xf
CQ_BASE_STATUS_LAST = CQ_BASE_STATUS_OVERFLOW_ERR
CQ_REQ_TOGGLE = 0x1
CQ_REQ_CQE_TYPE_MASK = 0x1e
CQ_REQ_CQE_TYPE_SFT = 1
CQ_REQ_CQE_TYPE_REQ = (0x0 << 1)
CQ_REQ_CQE_TYPE_LAST = CQ_REQ_CQE_TYPE_REQ
CQ_REQ_PUSH = 0x20
CQ_REQ_STATUS_OK = 0x0
CQ_REQ_STATUS_BAD_RESPONSE_ERR = 0x1
CQ_REQ_STATUS_LOCAL_LENGTH_ERR = 0x2
CQ_REQ_STATUS_LOCAL_QP_OPERATION_ERR = 0x3
CQ_REQ_STATUS_LOCAL_PROTECTION_ERR = 0x4
CQ_REQ_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_REQ_STATUS_REMOTE_INVALID_REQUEST_ERR = 0x6
CQ_REQ_STATUS_REMOTE_ACCESS_ERR = 0x7
CQ_REQ_STATUS_REMOTE_OPERATION_ERR = 0x8
CQ_REQ_STATUS_RNR_NAK_RETRY_CNT_ERR = 0x9
CQ_REQ_STATUS_TRANSPORT_RETRY_CNT_ERR = 0xa
CQ_REQ_STATUS_WORK_REQUEST_FLUSHED_ERR = 0xb
CQ_REQ_STATUS_LAST = CQ_REQ_STATUS_WORK_REQUEST_FLUSHED_ERR
CQ_RES_RC_TOGGLE = 0x1
CQ_RES_RC_CQE_TYPE_MASK = 0x1e
CQ_RES_RC_CQE_TYPE_SFT = 1
CQ_RES_RC_CQE_TYPE_RES_RC = (0x1 << 1)
CQ_RES_RC_CQE_TYPE_LAST = CQ_RES_RC_CQE_TYPE_RES_RC
CQ_RES_RC_STATUS_OK = 0x0
CQ_RES_RC_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_RC_STATUS_LOCAL_LENGTH_ERR = 0x2
CQ_RES_RC_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_RC_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_RC_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_RC_STATUS_REMOTE_INVALID_REQUEST_ERR = 0x6
CQ_RES_RC_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_RC_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_RC_STATUS_LAST = CQ_RES_RC_STATUS_HW_FLUSH_ERR
CQ_RES_RC_FLAGS_SRQ = 0x1
CQ_RES_RC_FLAGS_SRQ_RQ = 0x0
CQ_RES_RC_FLAGS_SRQ_SRQ = 0x1
CQ_RES_RC_FLAGS_SRQ_LAST = CQ_RES_RC_FLAGS_SRQ_SRQ
CQ_RES_RC_FLAGS_IMM = 0x2
CQ_RES_RC_FLAGS_INV = 0x4
CQ_RES_RC_FLAGS_RDMA = 0x8
CQ_RES_RC_FLAGS_RDMA_SEND = (0x0 << 3)
CQ_RES_RC_FLAGS_RDMA_RDMA_WRITE = (0x1 << 3)
CQ_RES_RC_FLAGS_RDMA_LAST = CQ_RES_RC_FLAGS_RDMA_RDMA_WRITE
CQ_RES_RC_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_RC_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_LENGTH_MASK = 0x3fff
CQ_RES_UD_LENGTH_SFT = 0
CQ_RES_UD_CFA_METADATA_VID_MASK = 0xfff
CQ_RES_UD_CFA_METADATA_VID_SFT = 0
CQ_RES_UD_CFA_METADATA_DE = 0x1000
CQ_RES_UD_CFA_METADATA_PRI_MASK = 0xe000
CQ_RES_UD_CFA_METADATA_PRI_SFT = 13
CQ_RES_UD_TOGGLE = 0x1
CQ_RES_UD_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_CQE_TYPE_SFT = 1
CQ_RES_UD_CQE_TYPE_RES_UD = (0x2 << 1)
CQ_RES_UD_CQE_TYPE_LAST = CQ_RES_UD_CQE_TYPE_RES_UD
CQ_RES_UD_STATUS_OK = 0x0
CQ_RES_UD_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_STATUS_LAST = CQ_RES_UD_STATUS_HW_FLUSH_ERR
CQ_RES_UD_FLAGS_SRQ = 0x1
CQ_RES_UD_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_FLAGS_SRQ_LAST = CQ_RES_UD_FLAGS_SRQ_SRQ
CQ_RES_UD_FLAGS_IMM = 0x2
CQ_RES_UD_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_VLAN = (0x1 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_FLAGS_META_FORMAT_LAST = CQ_RES_UD_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_FLAGS_EXT_META_FORMAT_MASK = 0xc00
CQ_RES_UD_FLAGS_EXT_META_FORMAT_SFT = 10
CQ_RES_UD_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_SRC_QP_HIGH_SFT = 24
CQ_RES_UD_V2_LENGTH_MASK = 0x3fff
CQ_RES_UD_V2_LENGTH_SFT = 0
CQ_RES_UD_V2_CFA_METADATA0_VID_MASK = 0xfff
CQ_RES_UD_V2_CFA_METADATA0_VID_SFT = 0
CQ_RES_UD_V2_CFA_METADATA0_DE = 0x1000
CQ_RES_UD_V2_CFA_METADATA0_PRI_MASK = 0xe000
CQ_RES_UD_V2_CFA_METADATA0_PRI_SFT = 13
CQ_RES_UD_V2_TOGGLE = 0x1
CQ_RES_UD_V2_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_V2_CQE_TYPE_SFT = 1
CQ_RES_UD_V2_CQE_TYPE_RES_UD = (0x2 << 1)
CQ_RES_UD_V2_CQE_TYPE_LAST = CQ_RES_UD_V2_CQE_TYPE_RES_UD
CQ_RES_UD_V2_STATUS_OK = 0x0
CQ_RES_UD_V2_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_V2_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_V2_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_V2_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_V2_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_V2_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_V2_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_V2_STATUS_LAST = CQ_RES_UD_V2_STATUS_HW_FLUSH_ERR
CQ_RES_UD_V2_FLAGS_SRQ = 0x1
CQ_RES_UD_V2_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_V2_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_V2_FLAGS_SRQ_LAST = CQ_RES_UD_V2_FLAGS_SRQ_SRQ
CQ_RES_UD_V2_FLAGS_IMM = 0x2
CQ_RES_UD_V2_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_V2_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_V2_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_V2_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_V2_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_V2_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_ACT_REC_PTR = (0x1 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_V2_FLAGS_META_FORMAT_LAST = CQ_RES_UD_V2_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_V2_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_V2_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_V2_CFA_METADATA1_MASK = 0xf00000
CQ_RES_UD_V2_CFA_METADATA1_SFT = 20
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_MASK = 0x700000
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_SFT = 20
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID88A8 = (0x0 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID8100 = (0x1 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID9100 = (0x2 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID9200 = (0x3 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPID9300 = (0x4 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPIDCFG = (0x5 << 20)
CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_LAST = CQ_RES_UD_V2_CFA_METADATA1_TPID_SEL_TPIDCFG
CQ_RES_UD_V2_CFA_METADATA1_VALID = 0x800000
CQ_RES_UD_V2_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_V2_SRC_QP_HIGH_SFT = 24
CQ_RES_UD_CFA_LENGTH_MASK = 0x3fff
CQ_RES_UD_CFA_LENGTH_SFT = 0
CQ_RES_UD_CFA_QID_MASK = 0xfffff
CQ_RES_UD_CFA_QID_SFT = 0
CQ_RES_UD_CFA_CFA_METADATA_VID_MASK = 0xfff
CQ_RES_UD_CFA_CFA_METADATA_VID_SFT = 0
CQ_RES_UD_CFA_CFA_METADATA_DE = 0x1000
CQ_RES_UD_CFA_CFA_METADATA_PRI_MASK = 0xe000
CQ_RES_UD_CFA_CFA_METADATA_PRI_SFT = 13
CQ_RES_UD_CFA_CFA_METADATA_TPID_MASK = 0xffff0000
CQ_RES_UD_CFA_CFA_METADATA_TPID_SFT = 16
CQ_RES_UD_CFA_TOGGLE = 0x1
CQ_RES_UD_CFA_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_CFA_CQE_TYPE_SFT = 1
CQ_RES_UD_CFA_CQE_TYPE_RES_UD_CFA = (0x4 << 1)
CQ_RES_UD_CFA_CQE_TYPE_LAST = CQ_RES_UD_CFA_CQE_TYPE_RES_UD_CFA
CQ_RES_UD_CFA_STATUS_OK = 0x0
CQ_RES_UD_CFA_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_CFA_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_CFA_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_CFA_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_CFA_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_CFA_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_CFA_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_CFA_STATUS_LAST = CQ_RES_UD_CFA_STATUS_HW_FLUSH_ERR
CQ_RES_UD_CFA_FLAGS_SRQ = 0x1
CQ_RES_UD_CFA_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_CFA_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_CFA_FLAGS_SRQ_LAST = CQ_RES_UD_CFA_FLAGS_SRQ_SRQ
CQ_RES_UD_CFA_FLAGS_IMM = 0x2
CQ_RES_UD_CFA_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_CFA_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_CFA_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_CFA_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_CFA_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_CFA_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_VLAN = (0x1 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_CFA_FLAGS_META_FORMAT_LAST = CQ_RES_UD_CFA_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_CFA_FLAGS_EXT_META_FORMAT_MASK = 0xc00
CQ_RES_UD_CFA_FLAGS_EXT_META_FORMAT_SFT = 10
CQ_RES_UD_CFA_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_CFA_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_CFA_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_CFA_SRC_QP_HIGH_SFT = 24
CQ_RES_UD_CFA_V2_LENGTH_MASK = 0x3fff
CQ_RES_UD_CFA_V2_LENGTH_SFT = 0
CQ_RES_UD_CFA_V2_CFA_METADATA0_VID_MASK = 0xfff
CQ_RES_UD_CFA_V2_CFA_METADATA0_VID_SFT = 0
CQ_RES_UD_CFA_V2_CFA_METADATA0_DE = 0x1000
CQ_RES_UD_CFA_V2_CFA_METADATA0_PRI_MASK = 0xe000
CQ_RES_UD_CFA_V2_CFA_METADATA0_PRI_SFT = 13
CQ_RES_UD_CFA_V2_QID_MASK = 0xfffff
CQ_RES_UD_CFA_V2_QID_SFT = 0
CQ_RES_UD_CFA_V2_TOGGLE = 0x1
CQ_RES_UD_CFA_V2_CQE_TYPE_MASK = 0x1e
CQ_RES_UD_CFA_V2_CQE_TYPE_SFT = 1
CQ_RES_UD_CFA_V2_CQE_TYPE_RES_UD_CFA = (0x4 << 1)
CQ_RES_UD_CFA_V2_CQE_TYPE_LAST = CQ_RES_UD_CFA_V2_CQE_TYPE_RES_UD_CFA
CQ_RES_UD_CFA_V2_STATUS_OK = 0x0
CQ_RES_UD_CFA_V2_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_UD_CFA_V2_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_UD_CFA_V2_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_UD_CFA_V2_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_UD_CFA_V2_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_UD_CFA_V2_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_UD_CFA_V2_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_UD_CFA_V2_STATUS_LAST = CQ_RES_UD_CFA_V2_STATUS_HW_FLUSH_ERR
CQ_RES_UD_CFA_V2_FLAGS_SRQ = 0x1
CQ_RES_UD_CFA_V2_FLAGS_SRQ_RQ = 0x0
CQ_RES_UD_CFA_V2_FLAGS_SRQ_SRQ = 0x1
CQ_RES_UD_CFA_V2_FLAGS_SRQ_LAST = CQ_RES_UD_CFA_V2_FLAGS_SRQ_SRQ
CQ_RES_UD_CFA_V2_FLAGS_IMM = 0x2
CQ_RES_UD_CFA_V2_FLAGS_UNUSED_MASK = 0xc
CQ_RES_UD_CFA_V2_FLAGS_UNUSED_SFT = 2
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_MASK = 0x30
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_SFT = 4
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V1 = (0x0 << 4)
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V2IPV4 = (0x2 << 4)
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V2IPV6 = (0x3 << 4)
CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_LAST = CQ_RES_UD_CFA_V2_FLAGS_ROCE_IP_VER_V2IPV6
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_MASK = 0x3c0
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_SFT = 6
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_NONE = (0x0 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_ACT_REC_PTR = (0x1 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_TUNNEL_ID = (0x2 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_CHDR_DATA = (0x3 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_HDR_OFFSET = (0x4 << 6)
CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_LAST = CQ_RES_UD_CFA_V2_FLAGS_META_FORMAT_HDR_OFFSET
CQ_RES_UD_CFA_V2_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_UD_CFA_V2_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_UD_CFA_V2_CFA_METADATA1_MASK = 0xf00000
CQ_RES_UD_CFA_V2_CFA_METADATA1_SFT = 20
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_MASK = 0x700000
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_SFT = 20
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID88A8 = (0x0 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID8100 = (0x1 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID9100 = (0x2 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID9200 = (0x3 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPID9300 = (0x4 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPIDCFG = (0x5 << 20)
CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_LAST = CQ_RES_UD_CFA_V2_CFA_METADATA1_TPID_SEL_TPIDCFG
CQ_RES_UD_CFA_V2_CFA_METADATA1_VALID = 0x800000
CQ_RES_UD_CFA_V2_SRC_QP_HIGH_MASK = 0xff000000
CQ_RES_UD_CFA_V2_SRC_QP_HIGH_SFT = 24
CQ_RES_RAWETH_QP1_LENGTH_MASK = 0x3fff
CQ_RES_RAWETH_QP1_LENGTH_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_MASK = 0x3ff
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ERROR = 0x1
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_MASK = 0x3c0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_SFT = 6
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_NOT_KNOWN = (0x0 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_IP = (0x1 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_TCP = (0x2 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_UDP = (0x3 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_FCOE = (0x4 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_ROCE = (0x5 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_ICMP = (0x7 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_PTP_WO_TIMESTAMP = (0x8 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP = (0x9 << 6)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_IP_CS_ERROR = 0x10
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_L4_CS_ERROR = 0x20
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_IP_CS_ERROR = 0x40
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_L4_CS_ERROR = 0x80
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_CRC_ERROR = 0x100
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_MASK = 0xe00
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_SFT = 9
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_NO_ERROR = (0x0 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_VERSION = (0x1 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_HDR_LEN = (0x2 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_TUNNEL_TOTAL_ERROR = (0x3 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_IP_TOTAL_ERROR = (0x4 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_UDP_TOTAL_ERROR = (0x5 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL = (0x6 << 9)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_MASK = 0xf000
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_SFT = 12
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_NO_ERROR = (0x0 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_VERSION = (0x1 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_HDR_LEN = (0x2 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_TTL = (0x3 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_IP_TOTAL_ERROR = (0x4 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_UDP_TOTAL_ERROR = (0x5 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN = (0x6 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN_TOO_SMALL = (0x7 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN = (0x8 << 12)
CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_IP_CS_CALC = 0x1
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_L4_CS_CALC = 0x2
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_T_IP_CS_CALC = 0x4
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_T_L4_CS_CALC = 0x8
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_MASK = 0xf0
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_SFT = 4
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_NONE = (0x0 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_VLAN = (0x1 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_TUNNEL_ID = (0x2 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_CHDR_DATA = (0x3 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET = (0x4 << 4)
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_LAST = CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_IP_TYPE = 0x100
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_CALC = 0x200
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_EXT_META_FORMAT_MASK = 0xc00
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_EXT_META_FORMAT_SFT = 10
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_MASK = 0xffff0000
CQ_RES_RAWETH_QP1_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_SFT = 16
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_DE_VID_MASK = 0xffff
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_DE_VID_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_VID_MASK = 0xfff
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_VID_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_DE = 0x1000
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_MASK = 0xe000
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_PRI_SFT = 13
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_TPID_MASK = 0xffff0000
CQ_RES_RAWETH_QP1_RAWETH_QP1_METADATA_TPID_SFT = 16
CQ_RES_RAWETH_QP1_TOGGLE = 0x1
CQ_RES_RAWETH_QP1_CQE_TYPE_MASK = 0x1e
CQ_RES_RAWETH_QP1_CQE_TYPE_SFT = 1
CQ_RES_RAWETH_QP1_CQE_TYPE_RES_RAWETH_QP1 = (0x3 << 1)
CQ_RES_RAWETH_QP1_CQE_TYPE_LAST = CQ_RES_RAWETH_QP1_CQE_TYPE_RES_RAWETH_QP1
CQ_RES_RAWETH_QP1_STATUS_OK = 0x0
CQ_RES_RAWETH_QP1_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_RAWETH_QP1_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_RAWETH_QP1_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_RAWETH_QP1_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_RAWETH_QP1_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_RAWETH_QP1_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_RAWETH_QP1_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_RAWETH_QP1_STATUS_LAST = CQ_RES_RAWETH_QP1_STATUS_HW_FLUSH_ERR
CQ_RES_RAWETH_QP1_FLAGS_SRQ = 0x1
CQ_RES_RAWETH_QP1_FLAGS_SRQ_RQ = 0x0
CQ_RES_RAWETH_QP1_FLAGS_SRQ_SRQ = 0x1
CQ_RES_RAWETH_QP1_FLAGS_SRQ_LAST = CQ_RES_RAWETH_QP1_FLAGS_SRQ_SRQ
CQ_RES_RAWETH_QP1_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_RAWETH_QP1_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_RAWETH_QP1_RAWETH_QP1_PAYLOAD_OFFSET_MASK = 0xff000000
CQ_RES_RAWETH_QP1_RAWETH_QP1_PAYLOAD_OFFSET_SFT = 24
CQ_RES_RAWETH_QP1_V2_LENGTH_MASK = 0x3fff
CQ_RES_RAWETH_QP1_V2_LENGTH_SFT = 0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_MASK = 0x3ff
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_SFT = 0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ERROR = 0x1
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_MASK = 0x3c0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_SFT = 6
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_NOT_KNOWN = (0x0 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_IP = (0x1 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_TCP = (0x2 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_UDP = (0x3 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_FCOE = (0x4 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_ROCE = (0x5 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_ICMP = (0x7 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_PTP_WO_TIMESTAMP = (0x8 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP = (0x9 << 6)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS_ITYPE_PTP_W_TIMESTAMP
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_IP_CS_ERROR = 0x10
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_L4_CS_ERROR = 0x20
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_IP_CS_ERROR = 0x40
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_L4_CS_ERROR = 0x80
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_CRC_ERROR = 0x100
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_MASK = 0xe00
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_SFT = 9
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_NO_ERROR = (0x0 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_VERSION = (0x1 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_HDR_LEN = (0x2 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_TUNNEL_TOTAL_ERROR = (0x3 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_IP_TOTAL_ERROR = (0x4 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_UDP_TOTAL_ERROR = (0x5 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL = (0x6 << 9)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_T_PKT_ERROR_T_L3_BAD_TTL
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_MASK = 0xf000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_SFT = 12
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_NO_ERROR = (0x0 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_VERSION = (0x1 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_HDR_LEN = (0x2 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L3_BAD_TTL = (0x3 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_IP_TOTAL_ERROR = (0x4 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_UDP_TOTAL_ERROR = (0x5 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN = (0x6 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_HDR_LEN_TOO_SMALL = (0x7 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN = (0x8 << 12)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_ERRORS_PKT_ERROR_L4_BAD_OPT_LEN
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_VID_MASK = 0xfff
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_VID_SFT = 0
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_DE = 0x1000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_PRI_MASK = 0xe000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA0_PRI_SFT = 13
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_CS_ALL_OK_MODE = 0x8
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_MASK = 0xf0
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_SFT = 4
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_NONE = (0x0 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_ACT_REC_PTR = (0x1 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_TUNNEL_ID = (0x2 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_CHDR_DATA = (0x3 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET = (0x4 << 4)
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_LAST = CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_META_FORMAT_HDR_OFFSET
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_IP_TYPE = 0x100
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_CALC = 0x200
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_CS_OK_MASK = 0xfc00
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_CS_OK_SFT = 10
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_MASK = 0xffff0000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_FLAGS2_COMPLETE_CHECKSUM_SFT = 16
CQ_RES_RAWETH_QP1_V2_TOGGLE = 0x1
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_MASK = 0x1e
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_SFT = 1
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_RES_RAWETH_QP1 = (0x3 << 1)
CQ_RES_RAWETH_QP1_V2_CQE_TYPE_LAST = CQ_RES_RAWETH_QP1_V2_CQE_TYPE_RES_RAWETH_QP1
CQ_RES_RAWETH_QP1_V2_STATUS_OK = 0x0
CQ_RES_RAWETH_QP1_V2_STATUS_LOCAL_ACCESS_ERROR = 0x1
CQ_RES_RAWETH_QP1_V2_STATUS_HW_LOCAL_LENGTH_ERR = 0x2
CQ_RES_RAWETH_QP1_V2_STATUS_LOCAL_PROTECTION_ERR = 0x3
CQ_RES_RAWETH_QP1_V2_STATUS_LOCAL_QP_OPERATION_ERR = 0x4
CQ_RES_RAWETH_QP1_V2_STATUS_MEMORY_MGT_OPERATION_ERR = 0x5
CQ_RES_RAWETH_QP1_V2_STATUS_WORK_REQUEST_FLUSHED_ERR = 0x7
CQ_RES_RAWETH_QP1_V2_STATUS_HW_FLUSH_ERR = 0x8
CQ_RES_RAWETH_QP1_V2_STATUS_LAST = CQ_RES_RAWETH_QP1_V2_STATUS_HW_FLUSH_ERR
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ = 0x1
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_RQ = 0x0
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_SRQ = 0x1
CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_LAST = CQ_RES_RAWETH_QP1_V2_FLAGS_SRQ_SRQ
CQ_RES_RAWETH_QP1_V2_SRQ_OR_RQ_WR_ID_MASK = 0xfffff
CQ_RES_RAWETH_QP1_V2_SRQ_OR_RQ_WR_ID_SFT = 0
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_MASK = 0xf00000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_SFT = 20
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_MASK = 0x700000
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_SFT = 20
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID88A8 = (0x0 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID8100 = (0x1 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID9100 = (0x2 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID9200 = (0x3 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPID9300 = (0x4 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPIDCFG = (0x5 << 20)
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_LAST = CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_TPID_SEL_TPIDCFG
CQ_RES_RAWETH_QP1_V2_CFA_METADATA1_VALID = 0x800000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_PAYLOAD_OFFSET_MASK = 0xff000000
CQ_RES_RAWETH_QP1_V2_RAWETH_QP1_PAYLOAD_OFFSET_SFT = 24
CQ_TERMINAL_TOGGLE = 0x1
CQ_TERMINAL_CQE_TYPE_MASK = 0x1e
CQ_TERMINAL_CQE_TYPE_SFT = 1
CQ_TERMINAL_CQE_TYPE_TERMINAL = (0xe << 1)
CQ_TERMINAL_CQE_TYPE_LAST = CQ_TERMINAL_CQE_TYPE_TERMINAL
CQ_TERMINAL_STATUS_OK = 0x0
CQ_TERMINAL_STATUS_LAST = CQ_TERMINAL_STATUS_OK
CQ_CUTOFF_TOGGLE = 0x1
CQ_CUTOFF_CQE_TYPE_MASK = 0x1e
CQ_CUTOFF_CQE_TYPE_SFT = 1
CQ_CUTOFF_CQE_TYPE_CUT_OFF = (0xf << 1)
CQ_CUTOFF_CQE_TYPE_LAST = CQ_CUTOFF_CQE_TYPE_CUT_OFF
CQ_CUTOFF_RESIZE_TOGGLE_MASK = 0x60
CQ_CUTOFF_RESIZE_TOGGLE_SFT = 5
CQ_CUTOFF_STATUS_OK = 0x0
CQ_CUTOFF_STATUS_LAST = CQ_CUTOFF_STATUS_OK
NQ_BASE_TYPE_MASK = 0x3f
NQ_BASE_TYPE_SFT = 0
NQ_BASE_TYPE_CQ_NOTIFICATION = 0x30
NQ_BASE_TYPE_SRQ_EVENT = 0x32
NQ_BASE_TYPE_DBQ_EVENT = 0x34
NQ_BASE_TYPE_QP_EVENT = 0x38
NQ_BASE_TYPE_FUNC_EVENT = 0x3a
NQ_BASE_TYPE_LAST = NQ_BASE_TYPE_FUNC_EVENT
NQ_BASE_INFO10_MASK = 0xffc0
NQ_BASE_INFO10_SFT = 6
NQ_BASE_V = 0x1
NQ_BASE_INFO63_MASK = 0xfffffffe
NQ_BASE_INFO63_SFT = 1
NQ_CN_TYPE_MASK = 0x3f
NQ_CN_TYPE_SFT = 0
NQ_CN_TYPE_CQ_NOTIFICATION = 0x30
NQ_CN_TYPE_LAST = NQ_CN_TYPE_CQ_NOTIFICATION
NQ_CN_TOGGLE_MASK = 0xc0
NQ_CN_TOGGLE_SFT = 6
NQ_CN_V = 0x1
NQ_SRQ_EVENT_TYPE_MASK = 0x3f
NQ_SRQ_EVENT_TYPE_SFT = 0
NQ_SRQ_EVENT_TYPE_SRQ_EVENT = 0x32
NQ_SRQ_EVENT_TYPE_LAST = NQ_SRQ_EVENT_TYPE_SRQ_EVENT
NQ_SRQ_EVENT_TOGGLE_MASK = 0xc0
NQ_SRQ_EVENT_TOGGLE_SFT = 6
NQ_SRQ_EVENT_EVENT_SRQ_THRESHOLD_EVENT = 0x1
NQ_SRQ_EVENT_EVENT_LAST = NQ_SRQ_EVENT_EVENT_SRQ_THRESHOLD_EVENT
NQ_SRQ_EVENT_V = 0x1
NQ_DBQ_EVENT_TYPE_MASK = 0x3f
NQ_DBQ_EVENT_TYPE_SFT = 0
NQ_DBQ_EVENT_TYPE_DBQ_EVENT = 0x34
NQ_DBQ_EVENT_TYPE_LAST = NQ_DBQ_EVENT_TYPE_DBQ_EVENT
NQ_DBQ_EVENT_EVENT_DBQ_THRESHOLD_EVENT = 0x1
NQ_DBQ_EVENT_EVENT_LAST = NQ_DBQ_EVENT_EVENT_DBQ_THRESHOLD_EVENT
NQ_DBQ_EVENT_DB_PFID_MASK = 0xf
NQ_DBQ_EVENT_DB_PFID_SFT = 0
NQ_DBQ_EVENT_DB_DPI_MASK = 0xfffff
NQ_DBQ_EVENT_DB_DPI_SFT = 0
NQ_DBQ_EVENT_V = 0x1
NQ_DBQ_EVENT_DB_XID_MASK = 0xfffff
NQ_DBQ_EVENT_DB_XID_SFT = 0
NQ_DBQ_EVENT_DB_TYPE_MASK = 0xf0000000
NQ_DBQ_EVENT_DB_TYPE_SFT = 28
XRRQ_IRRQ_TYPE = 0x1
XRRQ_IRRQ_TYPE_READ_REQ = 0x0
XRRQ_IRRQ_TYPE_ATOMIC_REQ = 0x1
XRRQ_IRRQ_TYPE_LAST = XRRQ_IRRQ_TYPE_ATOMIC_REQ
XRRQ_IRRQ_CREDITS_MASK = 0xf800
XRRQ_IRRQ_CREDITS_SFT = 11
XRRQ_IRRQ_PSN_MASK = 0xffffff
XRRQ_IRRQ_PSN_SFT = 0
XRRQ_IRRQ_MSN_MASK = 0xffffff
XRRQ_IRRQ_MSN_SFT = 0
XRRQ_ORRQ_TYPE = 0x1
XRRQ_ORRQ_TYPE_READ_REQ = 0x0
XRRQ_ORRQ_TYPE_ATOMIC_REQ = 0x1
XRRQ_ORRQ_TYPE_LAST = XRRQ_ORRQ_TYPE_ATOMIC_REQ
XRRQ_ORRQ_NUM_SGES_MASK = 0xf800
XRRQ_ORRQ_NUM_SGES_SFT = 11
XRRQ_ORRQ_PSN_MASK = 0xffffff
XRRQ_ORRQ_PSN_SFT = 0
XRRQ_ORRQ_END_PSN_MASK = 0xffffff
XRRQ_ORRQ_END_PSN_SFT = 0
PTU_PTE_VALID = 0x1
PTU_PTE_LAST = 0x2
PTU_PTE_NEXT_TO_LAST = 0x4
PTU_PTE_UNUSED_MASK = 0xff8
PTU_PTE_UNUSED_SFT = 3
PTU_PTE_PAGE_MASK = 0xfffff000
PTU_PTE_PAGE_SFT = 12
PTU_PDE_VALID = 0x1
PTU_PDE_UNUSED_MASK = 0xffe
PTU_PDE_UNUSED_SFT = 1
PTU_PDE_PAGE_MASK = 0xfffff000
PTU_PDE_PAGE_SFT = 12
RCFW_CMDQ_TRIG_VAL = 1
RCFW_COMM_PCI_BAR_REGION = 0
RCFW_COMM_CONS_PCI_BAR_REGION = 2
RCFW_COMM_BASE_OFFSET = 0x600
RCFW_PF_VF_COMM_PROD_OFFSET = 0xc
RCFW_COMM_TRIG_OFFSET = 0x100
RCFW_COMM_SIZE = 0x104
RCFW_DBR_PCI_BAR_REGION = 2
RCFW_DBR_BASE_PAGE_SHIFT = 12
RCFW_FW_STALL_MAX_TIMEOUT = 40
RCFW_CMD_NON_BLOCKING_SHADOW_QD = 64
RCFW_CMD_WAIT_TIME_MS = 20000
BNXT_QPLIB_CMDQE_MAX_CNT = 8192
BNXT_QPLIB_CMDQE_BYTES = lambda depth: ((depth) * BNXT_QPLIB_CMDQE_UNITS) # type: ignore
RCFW_MAX_COOKIE_VALUE = (BNXT_QPLIB_CMDQE_MAX_CNT - 1)
RCFW_CMD_IS_BLOCKING = 0x8000
HWRM_VERSION_DEV_ATTR_MAX_DPI = 0x1000A0000000D
HWRM_VERSION_READ_CTX = 0x1000A00030012
BNXT_QPLIB_CREQE_MAX_CNT = (64 * 1024)
BNXT_QPLIB_CREQE_UNITS = 16
CREQ_ENTRY_POLL_BUDGET = 0x100
BNXT_QPLIB_OOS_COUNT_MASK = 0xFFFFFFFF
FIRMWARE_INITIALIZED_FLAG = (0)
FIRMWARE_FIRST_FLAG = (31)
FIRMWARE_STALL_DETECTED = (3)
ERR_DEVICE_DETACHED = (4)
CHIP_NUM_57508 = 0x1750
CHIP_NUM_57504 = 0x1751
CHIP_NUM_57502 = 0x1752
CHIP_NUM_58818 = 0xd818
CHIP_NUM_57608 = 0x1760
BNXT_RE_MAX_QPC_COUNT = (64 * 1024)
BNXT_RE_MAX_MRW_COUNT = (64 * 1024)
BNXT_RE_MAX_SRQC_COUNT = (64 * 1024)
BNXT_RE_MAX_CQ_COUNT = (64 * 1024)
BNXT_RE_MAX_MRW_COUNT_64K = (64 * 1024)
BNXT_RE_MAX_MRW_COUNT_256K = (256 * 1024)
BNXT_QPLIB_DBR_VALID = (0x1 << 26)
BNXT_QPLIB_DBR_EPOCH_SHIFT = 24
BNXT_QPLIB_DBR_TOGGLE_SHIFT = 25
BNXT_QPLIB_DBR_PF_DB_OFFSET = 0x10000
BNXT_QPLIB_DBR_VF_DB_OFFSET = 0x4000
PTR_PG = lambda x: (((x) & ~PTR_MAX_IDX_PER_PG) / PTR_CNT_PER_PG) # type: ignore
PTR_IDX = lambda x: ((x) & PTR_MAX_IDX_PER_PG) # type: ignore
MAX_PBL_LVL_0_PGS = 1
MAX_PBL_LVL_1_PGS = 512
MAX_PBL_LVL_1_PGS_SHIFT = 9
MAX_PBL_LVL_1_PGS_FOR_LVL_2 = 256
MAX_PBL_LVL_2_PGS = (256 * 512)
MAX_PDL_LVL_SHIFT = 9
ROCE_PG_SIZE_4K = (4 * 1024)
ROCE_PG_SIZE_8K = (8 * 1024)
ROCE_PG_SIZE_64K = (64 * 1024)
ROCE_PG_SIZE_2M = (2 * 1024 * 1024)
ROCE_PG_SIZE_8M = (8 * 1024 * 1024)
ROCE_PG_SIZE_1G = (1024 * 1024 * 1024)
BNXT_QPLIB_MAX_QP_CTX_ENTRY_SIZE = 448
BNXT_QPLIB_MAX_SRQ_CTX_ENTRY_SIZE = 64
BNXT_QPLIB_MAX_CQ_CTX_ENTRY_SIZE = 64
BNXT_QPLIB_MAX_MRW_CTX_ENTRY_SIZE = 128
MAX_TQM_ALLOC_REQ = 48
MAX_TQM_ALLOC_BLK_SIZE = 8
to_bnxt_qplib = lambda ptr,type,member: container_of(ptr, type, member) # type: ignore
BNXT_QPLIB_INIT_DBHDR = lambda xid,type,indx,toggle: (((u64)(((xid) & DBC_DBC_XID_MASK) | DBC_DBC_PATH_ROCE | (type) | BNXT_QPLIB_DBR_VALID) << 32) | (indx) | (((u32)(toggle)) << (BNXT_QPLIB_DBR_TOGGLE_SHIFT))) # type: ignore
BNXT_RE_HW_RETX = lambda a: _is_hw_retx_supported((a)) # type: ignore
HWRM_API_FLAGS = (BNXT_HWRM_CTX_SILENT | BNXT_HWRM_FULL_WAIT)
HWRM_CMD_MAX_TIMEOUT = 60000
SHORT_HWRM_CMD_TIMEOUT = 20
BNXT_HWRM_TARGET = 0xffff
BNXT_HWRM_NO_CMPL_RING = -1
BNXT_HWRM_REQ_MAX_SIZE = 128
BNXT_HWRM_DMA_ALIGN = 16
BNXT_HWRM_SENTINEL = 0xb6e1f68a12e9a7eb
HWRM_SHORT_MIN_TIMEOUT = 3
HWRM_SHORT_MAX_TIMEOUT = 10
HWRM_SHORT_TIMEOUT_COUNTER = 5
HWRM_MIN_TIMEOUT = 25
HWRM_MAX_TIMEOUT = 40
HWRM_VALID_BIT_DELAY_USEC = 50000