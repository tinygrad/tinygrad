# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
enum_ROGUE_CDMCTRL_BLOCK_TYPE: dict[int, str] = {(ROGUE_CDMCTRL_BLOCK_TYPE_COMPUTE_KERNEL:=0): 'ROGUE_CDMCTRL_BLOCK_TYPE_COMPUTE_KERNEL', (ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_LINK:=1): 'ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_LINK', (ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_TERMINATE:=2): 'ROGUE_CDMCTRL_BLOCK_TYPE_STREAM_TERMINATE'}
enum_ROGUE_CDMCTRL_USC_TARGET: dict[int, str] = {(ROGUE_CDMCTRL_USC_TARGET_ALL:=0): 'ROGUE_CDMCTRL_USC_TARGET_ALL', (ROGUE_CDMCTRL_USC_TARGET_ANY:=1): 'ROGUE_CDMCTRL_USC_TARGET_ANY'}
enum_ROGUE_CDMCTRL_SD_TYPE: dict[int, str] = {(ROGUE_CDMCTRL_SD_TYPE_NONE:=0): 'ROGUE_CDMCTRL_SD_TYPE_NONE', (ROGUE_CDMCTRL_SD_TYPE_PDS:=1): 'ROGUE_CDMCTRL_SD_TYPE_PDS', (ROGUE_CDMCTRL_SD_TYPE_USC:=2): 'ROGUE_CDMCTRL_SD_TYPE_USC'}
@c.record
class struct_ROGUE_CDMCTRL_KERNEL0(c.Struct):
  SIZE = 32
  block_type: int
  indirect_present: bool
  global_offsets_present: bool
  event_object_present: bool
  usc_common_size: int
  usc_unified_size: int
  pds_temp_size: int
  pds_data_size: int
  usc_target: int
  fence: bool
uint32_t: TypeAlias = ctypes.c_uint32
struct_ROGUE_CDMCTRL_KERNEL0.register_fields([('block_type', ctypes.c_uint32, 0), ('indirect_present', ctypes.c_bool, 4), ('global_offsets_present', ctypes.c_bool, 5), ('event_object_present', ctypes.c_bool, 6), ('usc_common_size', uint32_t, 8), ('usc_unified_size', uint32_t, 12), ('pds_temp_size', uint32_t, 16), ('pds_data_size', uint32_t, 20), ('usc_target', ctypes.c_uint32, 24), ('fence', ctypes.c_bool, 28)])
class struct_ROGUE_CDMCTRL_KERNEL1(c.Struct): pass
class struct_ROGUE_CDMCTRL_KERNEL2(c.Struct): pass
@c.record
class struct_ROGUE_CDMCTRL_KERNEL3(c.Struct):
  SIZE = 4
  workgroup_x: int
struct_ROGUE_CDMCTRL_KERNEL3.register_fields([('workgroup_x', uint32_t, 0)])
@c.record
class struct_ROGUE_CDMCTRL_KERNEL4(c.Struct):
  SIZE = 4
  workgroup_y: int
struct_ROGUE_CDMCTRL_KERNEL4.register_fields([('workgroup_y', uint32_t, 0)])
@c.record
class struct_ROGUE_CDMCTRL_KERNEL5(c.Struct):
  SIZE = 4
  workgroup_z: int
struct_ROGUE_CDMCTRL_KERNEL5.register_fields([('workgroup_z', uint32_t, 0)])
class struct_ROGUE_CDMCTRL_KERNEL6(c.Struct): pass
class struct_ROGUE_CDMCTRL_KERNEL7(c.Struct): pass
@c.record
class struct_ROGUE_CDMCTRL_KERNEL8(c.Struct):
  SIZE = 16
  max_instances: int
  workgroup_size_x: int
  workgroup_size_y: int
  workgroup_size_z: int
struct_ROGUE_CDMCTRL_KERNEL8.register_fields([('max_instances', uint32_t, 0), ('workgroup_size_x', uint32_t, 4), ('workgroup_size_y', uint32_t, 8), ('workgroup_size_z', uint32_t, 12)])
@c.record
class struct_ROGUE_CDMCTRL_KERNEL9(c.Struct):
  SIZE = 4
  global_offset_x: int
struct_ROGUE_CDMCTRL_KERNEL9.register_fields([('global_offset_x', uint32_t, 0)])
@c.record
class struct_ROGUE_CDMCTRL_KERNEL10(c.Struct):
  SIZE = 4
  global_offset_y: int
struct_ROGUE_CDMCTRL_KERNEL10.register_fields([('global_offset_y', uint32_t, 0)])
@c.record
class struct_ROGUE_CDMCTRL_KERNEL11(c.Struct):
  SIZE = 4
  global_offset_z: int
struct_ROGUE_CDMCTRL_KERNEL11.register_fields([('global_offset_z', uint32_t, 0)])
class struct_ROGUE_CDMCTRL_STREAM_LINK0(c.Struct): pass
class struct_ROGUE_CDMCTRL_STREAM_LINK1(c.Struct): pass
@c.record
class struct_ROGUE_CDMCTRL_STREAM_TERMINATE(c.Struct):
  SIZE = 4
  block_type: int
struct_ROGUE_CDMCTRL_STREAM_TERMINATE.register_fields([('block_type', ctypes.c_uint32, 0)])
ROGUE_CDMCTRL_GUARD_SIZE_DEFAULT = 64
ROGUE_CDMCTRL_KERNEL0_length = 1
ROGUE_CDMCTRL_KERNEL0_USC_COMMON_SIZE_UNIT_SIZE = 64
ROGUE_CDMCTRL_KERNEL0_USC_COMMON_SIZE_MAX_SIZE = 256
ROGUE_CDMCTRL_KERNEL0_USC_UNIFIED_SIZE_UNIT_SIZE = 16
ROGUE_CDMCTRL_KERNEL0_PDS_TEMP_SIZE_UNIT_SIZE = 16
ROGUE_CDMCTRL_KERNEL0_PDS_DATA_SIZE_UNIT_SIZE = 16
ROGUE_CDMCTRL_KERNEL1_length = 1
ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT = 16
ROGUE_CDMCTRL_KERNEL2_length = 1
ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT = 16
ROGUE_CDMCTRL_KERNEL3_length = 1
ROGUE_CDMCTRL_KERNEL4_length = 1
ROGUE_CDMCTRL_KERNEL5_length = 1
ROGUE_CDMCTRL_KERNEL6_length = 1
ROGUE_CDMCTRL_KERNEL6_INDIRECT_ADDRMSB_ALIGNMENT = 4294967296
ROGUE_CDMCTRL_KERNEL7_length = 1
ROGUE_CDMCTRL_KERNEL7_INDIRECT_ADDRLSB_ALIGNMENT = 4
ROGUE_CDMCTRL_KERNEL8_length = 1
ROGUE_CDMCTRL_KERNEL8_MAX_INSTANCES_MAX_SIZE = 31
ROGUE_CDMCTRL_KERNEL9_length = 1
ROGUE_CDMCTRL_KERNEL10_length = 1
ROGUE_CDMCTRL_KERNEL11_length = 1
ROGUE_CDMCTRL_STREAM_LINK0_length = 1
ROGUE_CDMCTRL_STREAM_LINK0_LINK_ADDRMSB_ALIGNMENT = 4294967296
ROGUE_CDMCTRL_STREAM_LINK1_length = 1
ROGUE_CDMCTRL_STREAM_LINK1_LINK_ADDRLSB_ALIGNMENT = 4
ROGUE_CDMCTRL_STREAM_TERMINATE_length = 1