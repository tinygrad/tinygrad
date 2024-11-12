import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def, amdgpu_psp_gfx_if
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class FWEntry:
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

class Firmware:
  SOS_PATH = "/lib/firmware/amdgpu/psp_13_0_0_sos.bin"
  TA_PATH = "/lib/firmware/amdgpu/psp_13_0_0_ta.bin"
  SMU_PATH = "/lib/firmware/amdgpu/smu_13_0_0.bin"
  PFP_PATH = "/lib/firmware/amdgpu/gc_11_0_0_pfp.bin"
  ME_PATH = "/lib/firmware/amdgpu/gc_11_0_0_me.bin"
  RLC_PATH = "/lib/firmware/amdgpu/gc_11_0_0_rlc.bin"
  MEC_PATH = "/lib/firmware/amdgpu/gc_11_0_0_mec.bin"
  MES_2_PATH = "/lib/firmware/amdgpu/gc_11_0_0_mes_2.bin"
  MES1_PATH = "/lib/firmware/amdgpu/gc_11_0_0_mes1.bin" # KIQ
  IMU_PATH = "/lib/firmware/amdgpu/gc_11_0_0_imu.bin"

  def __init__(self, adev):
    self.adev = adev
    self.init_sos()

    self.smu_fw = FWEntry(self.adev, self.SMU_PATH, amdgpu_2.struct_smc_firmware_header_v1_0)
    self.smu_psp_desc = self.smu_fw.smu_psp_desc()

    self.pfp_fw = FWEntry(self.adev, self.PFP_PATH, amdgpu_2.struct_gfx_firmware_header_v2_0)
    self.me_fw = FWEntry(self.adev, self.ME_PATH, amdgpu_2.struct_gfx_firmware_header_v2_0)
    self.mec_fw = FWEntry(self.adev, self.MEC_PATH, amdgpu_2.struct_gfx_firmware_header_v2_0)
    self.mes_fw = FWEntry(self.adev, self.MES_2_PATH, amdgpu_2.struct_mes_firmware_header_v1_0)
    self.mes_kiq_fw = FWEntry(self.adev, self.MES1_PATH, amdgpu_2.struct_mes_firmware_header_v1_0)

    self.rlc_fw = FWEntry(self.adev, self.RLC_PATH, amdgpu_2.struct_rlc_firmware_header_v2_0)
    self.imu_fw = FWEntry(self.adev, self.IMU_PATH, amdgpu_2.struct_imu_firmware_header_v1_0)

    self.psp_descs = [
      self.pfp_fw.cpv2_code_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_PFP),
      self.me_fw.cpv2_code_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_ME),
      self.mec_fw.cpv2_code_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_MEC),

      self.pfp_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_PFP_P0_STACK),
      self.pfp_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_PFP_P1_STACK),

      self.me_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_ME_P0_STACK),
      self.me_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_ME_P1_STACK),

      self.mec_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_MEC_P0_STACK),
      self.mec_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_MEC_P1_STACK),
      self.mec_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_MEC_P2_STACK),
      self.mec_fw.cpv2_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RS64_MEC_P3_STACK),

      self.mes_fw.mes_code_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_CP_MES),
      self.mes_fw.mes_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_MES_STACK),

      self.mes_kiq_fw.mes_code_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_CP_MES_KIQ),
      self.mes_kiq_fw.mes_data_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_MES_KIQ_STACK),

      self.imu_fw.imu_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_IMU_I),
      self.imu_fw.imu_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_IMU_D),

      self.rlc_fw.rlc_v2_1_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM),
      self.rlc_fw.rlc_v2_1_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM),

      self.rlc_fw.rlc_v2_2_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_IRAM),
      self.rlc_fw.rlc_v2_2_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_DRAM_BOOT),

      self.rlc_fw.rlc_v2_3_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_P),
      self.rlc_fw.rlc_v2_3_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_V),

      self.rlc_fw.rlc_v2_psp_desc(amdgpu_psp_gfx_if.GFX_FW_TYPE_RLC_G),
    ]

  def init_sos(self):
    with open(self.SOS_PATH, "rb") as f: sos_fw = memoryview(bytearray(f.read()))
    sos_hdr = amdgpu_2.struct_psp_firmware_header_v1_0.from_address(mv_address(sos_fw))

    assert sos_hdr.header.header_version_major == 2
    sos_hdr = amdgpu_2.struct_psp_firmware_header_v2_0.from_address(mv_address(sos_fw))

    assert sos_hdr.header.header_version_minor == 0
    fw_bin = sos_hdr.psp_fw_bin

    self.sos_fw_infos = {}
    for fw_i in range(sos_hdr.psp_fw_bin_count):
      fw_bin_desc = amdgpu_2.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(amdgpu_2.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw_infos[fw_bin_desc.fw_type] = sos_fw[ucode_start_offset:ucode_start_offset+fw_bin_desc.size_bytes]

    self.sos_fw = sos_fw
    self.sos_hdr = sos_hdr
