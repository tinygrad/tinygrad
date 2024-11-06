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
    if typ not in self.gpu_addr_fw_desc:
      gpu_paddr = self.adev.vmm.alloc_vram(size, f"fw_{typ}")
      gpu_paddr_mv = self.adev.vmm.vram_to_cpu_mv(gpu_paddr, size)
      gpu_paddr_mv[:] = self.blob[offset:offset+size]
      self.gpu_addr_fw_desc[typ] = (typ, gpu_paddr, size)

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
