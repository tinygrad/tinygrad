import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def, amdgpu_psp_gfx_if
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class Firmware:
  def __init__(self, adev, path, header_t):
    self.adev = adev
    self.path = path
    self.blob = memoryview(bytearray(open(self.path, "rb").read()))
    
    self.common_header = amdgpu_2.struct_common_firmware_header.from_buffer(self.blob[:ctypes.sizeof(amdgpu_2.struct_common_firmware_header)])
    self.header = header_t.from_buffer(self.blob[:ctypes.sizeof(header_t)])

    self.gpu_addr_fw_desc = {}

  def load_fw(self, typ, offset, size):
    if typ not in self.gpu_addr_fw_desc: self.gpu_addr_fw_desc[typ] = (typ, self.blob[offset:offset+size])
    return self.gpu_addr_fw_desc[typ]

  def smu_psp_desc(self):
    return self.load_fw(amdgpu_psp_gfx_if.GFX_FW_TYPE_SMU, self.header.header.ucode_array_offset_bytes, self.header.header.ucode_size_bytes)

  def cpv2_code_psp_desc(self, hdr):
    return self.load_fw(hdr, self.common_header.ucode_array_offset_bytes, self.header.ucode_size_bytes)

  def cpv2_data_psp_desc(self, hdr):
    return self.load_fw(hdr, self.header.data_offset_bytes, self.header.data_size_bytes)

  def mes_code_psp_desc(self, hdr):
    return self.load_fw(hdr, self.header.mes_ucode_offset_bytes, self.header.mes_ucode_size_bytes)

  def mes_data_psp_desc(self, hdr):
    return self.load_fw(hdr, self.header.mes_ucode_data_offset_bytes, self.header.mes_ucode_data_size_bytes)

  def imu_psp_desc(self, vv):
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_IMU_I:
      return self.load_fw(vv, self.header.header.ucode_array_offset_bytes, self.header.imu_iram_ucode_size_bytes)
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_IMU_D:
      return self.load_fw(vv, self.header.header.ucode_array_offset_bytes + self.header.imu_iram_ucode_size_bytes, self.header.imu_dram_ucode_size_bytes)
    assert False

  def rlc_v2_psp_desc(self, vv):
    assert vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_G
    hdr = amdgpu_2.struct_rlc_firmware_header_v2_0.from_buffer(self.blob[:ctypes.sizeof(amdgpu_2.struct_rlc_firmware_header_v2_0)])
    return self.load_fw(vv, self.common_header.ucode_array_offset_bytes, hdr.header.ucode_size_bytes)

  def rlc_v2_1_psp_desc(self, vv):
    hdr = amdgpu_2.struct_rlc_firmware_header_v2_1.from_buffer(self.blob[:ctypes.sizeof(amdgpu_2.struct_rlc_firmware_header_v2_1)])
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM:
      return self.load_fw(vv, hdr.save_restore_list_gpm_offset_bytes, hdr.save_restore_list_gpm_size_bytes)
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM:
      return self.load_fw(vv, hdr.save_restore_list_srm_offset_bytes, hdr.save_restore_list_srm_size_bytes)
    assert False

  def rlc_v2_2_psp_desc(self, vv):
    hdr = amdgpu_2.struct_rlc_firmware_header_v2_2.from_buffer(self.blob[:ctypes.sizeof(amdgpu_2.struct_rlc_firmware_header_v2_2)])
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_IRAM:
      return self.load_fw(vv, hdr.rlc_iram_ucode_offset_bytes, hdr.rlc_iram_ucode_size_bytes)
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_DRAM_BOOT:
      return self.load_fw(vv, hdr.rlc_dram_ucode_offset_bytes, hdr.rlc_dram_ucode_size_bytes)
    assert False

  def rlc_v2_3_psp_desc(self, vv):
    hdr = amdgpu_2.struct_rlc_firmware_header_v2_3.from_buffer(self.blob[:ctypes.sizeof(amdgpu_2.struct_rlc_firmware_header_v2_3)])
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_P:
      return self.load_fw(vv, hdr.rlcp_ucode_offset_bytes, hdr.rlcp_ucode_size_bytes)
    if vv == amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_V:
      return self.load_fw(vv, hdr.rlcv_ucode_offset_bytes, hdr.rlcv_ucode_size_bytes)
    assert False
