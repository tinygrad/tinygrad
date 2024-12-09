import os, ctypes
from typing import Tuple, Dict, List
from tinygrad.runtime.autogen.am import am
from tinygrad.helpers import mv_address

def load_fw(path:str, *headers):
  with open(path, "rb") as f: blob = memoryview(bytearray(f.read()))
  return tuple([blob] + [hdr.from_address(mv_address(blob)) for hdr in headers])

def psp_desc(typ:int, blob:memoryview, offset:int, size:int) -> Tuple[int, memoryview]: return (typ, blob[offset:offset+size])

class Firmware:
  SOS_PATH = "/lib/firmware/amdgpu/psp_13_0_0_sos.bin"
  TA_PATH = "/lib/firmware/amdgpu/psp_13_0_0_ta.bin"
  SMU_PATH = "/lib/firmware/amdgpu/smu_13_0_0.bin"
  PFP_PATH = "/lib/firmware/amdgpu/gc_11_0_0_pfp.bin"
  ME_PATH = "/lib/firmware/amdgpu/gc_11_0_0_me.bin"
  RLC_PATH = "/lib/firmware/amdgpu/gc_11_0_0_rlc.bin"
  MEC_PATH = "/lib/firmware/amdgpu/gc_11_0_0_mec.bin"
  IMU_PATH = "/lib/firmware/amdgpu/gc_11_0_0_imu.bin"
  SDMA_PATH = "/lib/firmware/amdgpu/sdma_6_0_0.bin"

  def __init__(self, adev):
    self.init_sos_fw()
    self.init_fw()

  def init_fw(self):
    self.ucode_start: Dict[str, int] = {}
    self.descs: List[Tuple[int, memoryview]] = []

    blob, hdr = load_fw(self.SMU_PATH, am.struct_smc_firmware_header_v1_0)
    self.smu_psp_desc = psp_desc(am.GFX_FW_TYPE_SMU, blob, hdr.header.ucode_array_offset_bytes, hdr.header.ucode_size_bytes)

    # SDMA firmware
    blob, hdr = load_fw(self.SDMA_PATH, am.struct_sdma_firmware_header_v2_0)
    self.descs += [psp_desc(am.GFX_FW_TYPE_SDMA_UCODE_TH0, blob, hdr.header.ucode_array_offset_bytes, hdr.ctx_ucode_size_bytes)]
    self.descs += [psp_desc(am.GFX_FW_TYPE_SDMA_UCODE_TH1, blob, hdr.ctl_ucode_offset, hdr.ctl_ucode_size_bytes)]

    # PFP, ME, MEC firmware
    # Code
    for (fw_path, fw_name) in [(self.PFP_PATH, 'PFP'), (self.ME_PATH, 'ME'), (self.MEC_PATH, 'MEC')]:
      blob, hdr = load_fw(fw_path, am.struct_gfx_firmware_header_v2_0)
      self.descs += [psp_desc(getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}'), blob, hdr.header.ucode_array_offset_bytes, hdr.ucode_size_bytes)]

    # Stack
    for (fw_path, fw_name, fw_cnt) in [(self.PFP_PATH, 'PFP', 2), (self.ME_PATH, 'ME', 2), (self.MEC_PATH, 'MEC', 4)]:
      blob, hdr = load_fw(fw_path, am.struct_gfx_firmware_header_v2_0)
      fw_types = [getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}_P{fwnun}_STACK') for fwnun in range(fw_cnt)]
      self.descs += [psp_desc(typ, blob, hdr.data_offset_bytes, hdr.data_size_bytes) for typ in fw_types]
      self.ucode_start[fw_name] = hdr.ucode_start_addr_lo | (hdr.ucode_start_addr_hi << 32)

    # IMU firmware
    blob, hdr = load_fw(self.IMU_PATH, am.struct_imu_firmware_header_v1_0)
    imu_i_off, imu_i_sz, imu_d_sz = hdr.header.ucode_array_offset_bytes, hdr.imu_iram_ucode_size_bytes, hdr.imu_dram_ucode_size_bytes
    self.descs += [psp_desc(am.GFX_FW_TYPE_IMU_I, blob, imu_i_off, imu_i_sz)]
    self.descs += [psp_desc(am.GFX_FW_TYPE_IMU_D, blob, imu_i_off + imu_i_sz, imu_d_sz)]

    # RLC firmware
    blob, hdr0, hdr1, hdr2, hdr3 = load_fw(self.RLC_PATH, am.struct_rlc_firmware_header_v2_0, am.struct_rlc_firmware_header_v2_1,
      am.struct_rlc_firmware_header_v2_2, am.struct_rlc_firmware_header_v2_3)

    for mem in ['GPM', 'SRM']:
      off, sz = getattr(hdr1, f'save_restore_list_{mem.lower()}_offset_bytes'), getattr(hdr1, f'save_restore_list_{mem.lower()}_size_bytes')
      self.descs += [psp_desc(getattr(am, f'GFX_FW_TYPE_RLC_RESTORE_LIST_{mem}_MEM'), blob, off, sz)]

    for mem,fmem in [('IRAM', 'iram'), ('DRAM_BOOT', 'dram')]:
      off, sz = getattr(hdr2, f'rlc_{fmem}_ucode_offset_bytes'), getattr(hdr2, f'rlc_{fmem}_ucode_size_bytes')
      self.descs += [psp_desc(getattr(am, f'GFX_FW_TYPE_RLC_{mem}'), blob, off, sz)]

    for mem in ['P', 'V']:
      off, sz = getattr(hdr3, f'rlc{mem.lower()}_ucode_offset_bytes'), getattr(hdr3, f'rlc{mem.lower()}_ucode_size_bytes')
      self.descs += [psp_desc(getattr(am, f'GFX_FW_TYPE_RLC_{mem}'), blob, off, sz)]

    self.descs += [psp_desc(am.GFX_FW_TYPE_RLC_G, blob, hdr0.header.ucode_array_offset_bytes, hdr0.header.ucode_size_bytes)]

  def init_sos_fw(self):
    self.sos_fw = {}

    blob, sos_hdr = load_fw(self.SOS_PATH, am.struct_psp_firmware_header_v2_0)
    for fw_i in range(sos_hdr.psp_fw_bin_count):
      fw_bin_desc = am.struct_psp_fw_bin_desc.from_address(ctypes.addressof(sos_hdr.psp_fw_bin) + fw_i * ctypes.sizeof(am.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw[fw_bin_desc.fw_type] = blob[ucode_start_offset:ucode_start_offset+fw_bin_desc.size_bytes]
