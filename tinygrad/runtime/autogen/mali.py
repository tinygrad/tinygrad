# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal
from tinygrad.runtime.support import c
from tinygrad.runtime.support.c import record

@record
class struct_pan_ubo_word(c.Struct): SIZE = 4
struct_pan_ubo_word.register_fields([('ubo', ctypes.c_uint16, 0), ('offset', ctypes.c_uint16, 2)])
@record
class struct_pan_shader_info(c.Struct): SIZE = 1496
struct_pan_shader_info.register_fields([
  ('stage', ctypes.c_uint32, 0), ('work_reg_count', ctypes.c_uint32, 4), ('tls_size', ctypes.c_uint32, 8),
  ('wls_size', ctypes.c_uint32, 12), ('preload', ctypes.c_uint64, 128), ('allow_merging_workgroups', ctypes.c_ubyte, 136),
  ('contains_barrier', ctypes.c_ubyte, 168), ('writes_global', ctypes.c_ubyte, 170),
  ('sampler_count', ctypes.c_uint32, 188), ('texture_count', ctypes.c_uint32, 192), ('ubo_count', ctypes.c_uint32, 196),
  ('attributes_read_count', ctypes.c_uint32, 200), ('attribute_count', ctypes.c_uint32, 204),
  ('attributes_read', ctypes.c_uint32, 208), ('push_count', ctypes.c_uint32, 820),
  ('push_words', c.Array[struct_pan_ubo_word, Literal[128]], 824), ('ubo_mask', ctypes.c_uint32, 1336)])
