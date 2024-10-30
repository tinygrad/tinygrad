import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0_offset, amdgpu_psp_gfx_if
from tinygrad.helpers import to_mv, mv_address

class PSP_IP:
  SOS_PATH = "/lib/firmware/amdgpu/psp_13_0_0_sos.bin"
  TA_PATH = "/lib/firmware/amdgpu/psp_13_0_0_ta.bin"

  def __init__(self, adev):
    self.adev = adev
    self.init_sos()
    self.init_ta()
    self.load_fw()

  def init_sos(self):
    sos_fw = memoryview(bytearray(open(self.SOS_PATH, "rb").read()))
    sos_hdr = amdgpu_2.struct_psp_firmware_header_v1_0.from_address(mv_address(sos_fw))

    assert sos_hdr.header.header_version_major == 2
    sos_hdr = amdgpu_2.struct_psp_firmware_header_v2_0.from_address(mv_address(sos_fw))

    assert sos_hdr.header.header_version_minor == 0
    fw_bin = sos_hdr.psp_fw_bin

    self.sos_fw_infos = []
    for fw_i in range(sos_hdr.psp_fw_bin_count):
      fw_bin_desc = amdgpu_2.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(amdgpu_2.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw_infos.append((fw_bin_desc.fw_type, ucode_start_offset))

    self.sos_fw = sos_fw
    self.sos_hdr = sos_hdr

  def init_ta(self):
    ta_fw = memoryview(bytearray(open(self.TA_PATH, "rb").read()))
    ta_hdr = amdgpu_2.struct_common_firmware_header.from_address(mv_address(ta_fw))
    assert ta_hdr.header_version_major == 2

    ta_hdr = amdgpu_2.struct_ta_firmware_header_v2_0.from_address(mv_address(ta_fw))

    fw_bin = ta_hdr.ta_fw_bin
    self.ta_fw_infos = []
    for fw_i in range(ta_hdr.ta_fw_bin_count):
      fw_bin_desc = amdgpu_2.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(amdgpu_2.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + ta_hdr.header.ucode_array_offset_bytes
      self.ta_fw_infos.append((fw_bin_desc.fw_type, ucode_start_offset))

    self.ta_fw = ta_fw
    self.ta_hdr = ta_hdr

  def load_fw(self):
    self.fence_buf = self.adev.vmm.alloc_vram(0x1000, "psp_fence_buf")
    self.cmd_buf = self.adev.vmm.alloc_vram(0x1000, "psp_cmd_buf")
    self.ring_mem = self.adev.vmm.alloc_vram(0x1000, "psp_ring_mem")

    ctypes.memset(self.adev.vmm.vram_to_cpu_addr(self.fence_buf, 0x1000), 0, 0x1000)

    self.hw_start()

  def is_sos_alive(self):
    sol = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_81, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_81_BASE_IDX)
    return sol != 0x0

  def init_sos_version(self):
    self.sos_fw_version = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_58, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_58_BASE_IDX)
    
  def ring_create(self):
    # Wait till the ring is ready
    reg = 0
    while reg & 0x80000000 != 0x80000000:
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64_BASE_IDX)

    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_69, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_69_BASE_IDX, self.ring_mem & 0xffffffff)
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_70, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_70_BASE_IDX, self.ring_mem >> 32)
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_71, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_71_BASE_IDX, 0x1000)

    ring_type = 2 << 16 # PSP_RING_TYPE__KM = 2. Kernel mode ring
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64_BASE_IDX, ring_type)
    print(self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64_BASE_IDX))

    # there might be handshake issue with hardware which needs delay
    time.sleep(100 / 1000) # 20 ms orignally

    # Wait for response flag
    reg = 0
    while reg & 0x80000000 != 0x80000000:
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64_BASE_IDX)

    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64_BASE_IDX, amdgpu_psp_gfx_if.GFX_CTRL_CMD_ID_DESTROY_RINGS)
    time.sleep(100 / 1000) # 20 ms orignally

    reg = 0
    while reg & 0x80000000 != 0x80000000:
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_64_BASE_IDX)
      print(reg)

    print("sOS ring created")

  def prep_load_ip_fw_cmd_buf(self, phys_addr):
    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.adev.vmm.vram_to_cpu_addr(self.cmd_buf, 0x1000), 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.adev.vmm.vram_to_cpu_addr(self.cmd_buf))
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_LOAD_IP_FW
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = phys_addr & 0xffffffff
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi = phys_addr >> 32
    # psp_gfx_cmd_resp
    # TODO
    pass

  def prep_boot_config_get(self):
    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.adev.vmm.vram_to_cpu_addr(self.cmd_buf, 0x1000), 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.adev.vmm.vram_to_cpu_addr(self.cmd_buf))
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_BOOT_CFG
    cmd.cmd.boot_cfg.sub_cmd = amdgpu_psp_gfx_if.BOOTCFG_CMD_GET

  def ring_get_wptr(self):
    return self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_67, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_67_BASE_IDX)
  
  def ring_set_wptr(self, wptr):
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_67, amdgpu_mp_13_0_0_offset.regMP0_SMN_C2PMSG_67_BASE_IDX, wptr)

  def cmd_submit_buf(self):
    # Write only first command
    # TODO: fix this
    prev_wptr = self.ring_get_wptr()

    ctypes.memset(self.adev.vmm.vram_to_cpu_addr(self.ring_mem), 0, ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame))
    write_loc = amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame.from_address(self.adev.vmm.vram_to_cpu_addr(self.ring_mem))
    write_loc.cmd_buf_addr_hi = self.cmd_buf >> 32
    write_loc.cmd_buf_addr_lo = self.cmd_buf & 0xffffffff
    write_loc.fence_addr_hi = self.fence_buf >> 32
    write_loc.fence_addr_lo = self.fence_buf & 0xffffffff
    write_loc.fence_value = 0x1

    print(prev_wptr)
    self.ring_set_wptr(prev_wptr + ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame) // 4)
    
    fence_view = to_mv(self.adev.vmm.vram_to_cpu_addr(self.fence_buf), 4).cast('I')
    while fence_view[0] != 0x1:
      # WREG32_SOC15_NO_KIQ(HDP, 0, mmHDP_READ_CACHE_INVALIDATE, 1);
      self.adev.wreg_ip("HDP", 0, 0x00d1, 0x0, 1)
      print("now", fence_view[0])
  
  def execute_ip_fw_load(self):
    pass

  
  def load_smu_fw(self):
    self.prep_boot_config_get()
    self.cmd_submit_buf()
    pass

  def hw_start(self):
    self.bootloader_load_sos()
    self.ring_create()

    # For ASICs with DF Cstate management centralized to PMFW, TMR setup should be performed after PMFW loaded and before other non-psp firmware loaded.
    self.load_smu_fw()
  
  def bootloader_load_sos(self):
    if (self.is_sos_alive()):
      self.init_sos_version()
      print(f"sOS alive, version {self.sos_fw_version}")
      return 0

    assert False, "TODO: Init from bootloader"
