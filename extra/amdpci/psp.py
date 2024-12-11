import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0, amdgpu_psp_gfx_if
from tinygrad.helpers import to_mv, mv_address, colored
from extra.amdpci.firmware import Firmware

class PSP_IP:
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

  def init(self):
    print("PSP init")
    self.prep_fw()
    self.init_sos()
    self.init_ta()
    self.load_fw()

  def prep_fw(self):
    self.smu_fw = Firmware(self.adev, self.SMU_PATH, amdgpu_2.struct_smc_firmware_header_v1_0)
    self.smu_psp_desc = self.smu_fw.smu_psp_desc()

    self.pfp_fw = Firmware(self.adev, self.PFP_PATH, amdgpu_2.struct_gfx_firmware_header_v2_0)
    self.me_fw = Firmware(self.adev, self.ME_PATH, amdgpu_2.struct_gfx_firmware_header_v2_0)
    self.mec_fw = Firmware(self.adev, self.MEC_PATH, amdgpu_2.struct_gfx_firmware_header_v2_0)
    self.mes_fw = Firmware(self.adev, self.MES_2_PATH, amdgpu_2.struct_mes_firmware_header_v1_0)
    self.mes_kiq_fw = Firmware(self.adev, self.MES1_PATH, amdgpu_2.struct_mes_firmware_header_v1_0)

    self.rlc_fw = Firmware(self.adev, self.RLC_PATH, amdgpu_2.struct_rlc_firmware_header_v2_0)
    self.imu_fw = Firmware(self.adev, self.IMU_PATH, amdgpu_2.struct_imu_firmware_header_v1_0)

    self.fw_list = [
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
    sos_fw = memoryview(bytearray(open(self.SOS_PATH, "rb").read()))
    sos_hdr = amdgpu_2.struct_psp_firmware_header_v1_0.from_address(mv_address(sos_fw))

    assert sos_hdr.header.header_version_major == 2
    sos_hdr = amdgpu_2.struct_psp_firmware_header_v2_0.from_address(mv_address(sos_fw))

    assert sos_hdr.header.header_version_minor == 0
    fw_bin = sos_hdr.psp_fw_bin

    # print(sos_hdr.psp_fw_bin_count)

    self.sos_fw_infos = {}
    for fw_i in range(sos_hdr.psp_fw_bin_count):
      fw_bin_desc = amdgpu_2.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(amdgpu_2.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw_infos[fw_bin_desc.fw_type] = sos_fw[ucode_start_offset:ucode_start_offset+fw_bin_desc.size_bytes]
      # print(fw_i, hex(fw_bin_desc.size_bytes), hex(len(self.sos_fw_infos[fw_bin_desc.fw_type])), hex(self.sos_fw_infos[fw_bin_desc.fw_type].cast('I')[0]))
      # print("\t", hex(fw_bin_desc.offset_bytes), hex(sos_hdr.header.ucode_array_offset_bytes))

    # print(self.sos_fw_infos)
    self.sos_fw = sos_fw
    self.sos_hdr = sos_hdr

  def init_ta(self):
    ta_fw = memoryview(bytearray(open(self.TA_PATH, "rb").read()
    ))
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
    PSP_1_MEG = 0x100000
    self.msg1_vaddr = self.adev.vmm.alloc_vram(PSP_1_MEG, "psp_msg1_buf", align=(1<<20))
    self.msg1_paddr = self.adev.vmm.vaddr_to_paddr(self.msg1_vaddr)
    self.msg1_mc_addr = self.adev.vmm.paddr_to_mc(self.msg1_paddr)
    self.msg1_cpu_addr = self.adev.vmm.paddr_to_cpu_addr(self.msg1_paddr)
    self.msg1_view = to_mv(self.msg1_cpu_addr, PSP_1_MEG)

    self.fence_vaddr = self.adev.vmm.alloc_vram(0x1000, "psp_fence_buf")
    self.fence_paddr = self.adev.vmm.vaddr_to_paddr(self.fence_vaddr)
    self.fence_mc_addr = self.adev.vmm.paddr_to_mc(self.fence_paddr)
    self.fence_view = to_mv(self.adev.vmm.paddr_to_cpu_addr(self.fence_paddr), 4).cast('I')

    self.cmd_vaddr = self.adev.vmm.alloc_vram(0x1000, "psp_cmd_buf")
    self.cmd_paddr = self.adev.vmm.vaddr_to_paddr(self.cmd_vaddr)
    self.cmd_mc_addr = self.adev.vmm.paddr_to_mc(self.cmd_paddr)
    self.cmd_cpu_addr = self.adev.vmm.paddr_to_cpu_addr(self.cmd_paddr)

    self.ring_vaddr = self.adev.vmm.alloc_vram(0x10000, "psp_ring_mem") # a bit bigger, no wrap around for this ring
    self.ring_paddr = self.adev.vmm.vaddr_to_paddr(self.ring_vaddr)
    self.ring_mc_addr = self.adev.vmm.paddr_to_mc(self.ring_paddr)
    self.ring_cpu_addr = self.adev.vmm.paddr_to_cpu_addr(self.ring_paddr)

    ctypes.memset(self.adev.vmm.paddr_to_cpu_addr(self.fence_paddr), 0, 0x1000)

    self.hw_start()

  def is_sos_alive(self):
    sol = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_81, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_81_BASE_IDX)
    return sol != 0x0

  def init_sos_version(self):
    self.sos_fw_version = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_58, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_58_BASE_IDX)
    
  def ring_create(self):
    # Remove all rings, will setup our new rings...
    # self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64_BASE_IDX, amdgpu_psp_gfx_if.GFX_CTRL_CMD_ID_DESTROY_RINGS)
    # time.sleep(100 / 1000) # 20 ms orignally

    reg = 0
    while reg & 0x8000FFFF != 0x80000000:
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64_BASE_IDX)

    # Wait till the ring is ready
    reg = 0
    while reg & 0x80000000 != 0x80000000:
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64_BASE_IDX)

    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_69, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_69_BASE_IDX, self.ring_mc_addr & 0xffffffff)
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_70, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_70_BASE_IDX, self.ring_mc_addr >> 32)
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_71, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_71_BASE_IDX, 0x1000)

    ring_type = 2 << 16 # PSP_RING_TYPE__KM = 2. Kernel mode ring
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64_BASE_IDX, ring_type)
    # print(self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64_BASE_IDX))

    # there might be handshake issue with hardware which needs delay
    time.sleep(100 / 1000) # 20 ms orignally

    # Wait for response flag
    reg = 0
    while reg & 0x8000FFFF != 0x80000000: # last 16 bits are status, should be 0
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_64_BASE_IDX)

    print("sOS ring created")

  def prep_load_ip_fw_cmd_buf(self, psp_desc):
    fw_type, fwm = psp_desc
    print('PSP: issue load ip fw:', fw_type, hex(len(fwm)))

    ctypes.memset(self.msg1_cpu_addr, 0, len(self.msg1_view))
    self.msg1_view[:len(fwm)] = fwm

    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.cmd_cpu_addr, 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_cpu_addr)
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_LOAD_IP_FW
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = self.msg1_mc_addr & 0xffffffff
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi = self.msg1_mc_addr >> 32
    cmd.cmd.cmd_load_ip_fw.fw_size = len(self.msg1_view)
    cmd.cmd.cmd_load_ip_fw.fw_type = fw_type

  def prep_tmr_cmd_buf(self):
    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.cmd_cpu_addr, 0, 0x1000)

    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_cpu_addr)
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_SETUP_TMR
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = self.tmr_mc_addr & 0xffffffff
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi = self.tmr_mc_addr >> 32
    cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = self.tmr_paddr & 0xffffffff
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi = self.tmr_paddr >> 32
    cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
    cmd.cmd.cmd_setup_tmr.buf_size = self.tmr_size

  def prep_load_toc_cmd_buf(self, toc_size):
    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.cmd_cpu_addr, 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_cpu_addr)
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_LOAD_TOC
    cmd.cmd.cmd_load_toc.toc_phy_addr_lo = self.msg1_mc_addr & 0xffffffff
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi = self.msg1_mc_addr >> 32
    cmd.cmd.cmd_load_toc.toc_size = toc_size

  def prep_boot_config_get(self):
    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.cmd_cpu_addr, 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_cpu_addr)
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_BOOT_CFG
    cmd.cmd.boot_cfg.sub_cmd = amdgpu_psp_gfx_if.BOOTCFG_CMD_GET

  def ring_get_wptr(self):
    return self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_67, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_67_BASE_IDX)
  
  def ring_set_wptr(self, wptr):
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_67, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_67_BASE_IDX, wptr)

  def cmd_submit_buf(self):
    prev_wptr = self.ring_get_wptr()
    ring_entry_addr = self.ring_cpu_addr + prev_wptr * 4

    ctypes.memset(ring_entry_addr, 0, ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame))
    write_loc = amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame.from_address(ring_entry_addr)
    write_loc.cmd_buf_addr_hi = self.cmd_mc_addr >> 32
    write_loc.cmd_buf_addr_lo = self.cmd_mc_addr & 0xffffffff
    write_loc.fence_addr_hi = self.fence_mc_addr >> 32
    write_loc.fence_addr_lo = self.fence_mc_addr & 0xffffffff
    write_loc.fence_value = prev_wptr

    self.ring_set_wptr(prev_wptr + ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame) // 4)
    
    smth = self.fence_view[0]
    while smth != prev_wptr:
      self.adev.wreg_ip("HDP", 0, 0x00d1, 0x0, 1)
      smth = self.fence_view[0]
    time.sleep(0.05)

    resp = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_cpu_addr)
    if resp.resp.status != 0: print(colored(f"PSP command failed {resp.cmd_id} {resp.resp.status}", "red"))
    else: 
      print(colored(f"PSP command success {resp.cmd_id}", "green"))
      # print(hex(resp.resp.fw_addr_lo))
      # print(hex(resp.resp.fw_addr_hi))
    return resp

  def load_smu_fw(self):
    self.prep_tmr_cmd_buf()
    self.cmd_submit_buf()

    self.prep_load_ip_fw_cmd_buf(self.smu_psp_desc)
    self.cmd_submit_buf()

  def load_toc(self):
    ctypes.memset(self.msg1_cpu_addr, 0, len(self.msg1_view))
    fwm = self.sos_fw_infos[amdgpu_2.PSP_FW_TYPE_PSP_TOC]
    self.msg1_view[:len(fwm)] = fwm

    self.prep_load_toc_cmd_buf(len(fwm))
    resp = self.cmd_submit_buf()

    self.tmr_size = resp.resp.tmr_size
    print("PSP: TOC loaded")
    print("TMR size: ", hex(self.tmr_size))

  def tmr_init(self):
    self.load_toc()
    self.tmr_vaddr = self.adev.vmm.alloc_vram(self.tmr_size, "psp_tmr", align=0x100000) # psp tmr
    self.tmr_paddr = self.adev.vmm.vaddr_to_paddr(self.tmr_vaddr)
    self.tmr_mc_addr = self.adev.vmm.paddr_to_mc(self.tmr_paddr)
  
  def load_tmr(self):
    self.prep_tmr_cmd_buf()
    self.cmd_submit_buf()

  def rlc_autoload_start(self):
    assert ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp) == 1024
    ctypes.memset(self.cmd_cpu_addr, 0, 0x1000)

    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_cpu_addr)
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_AUTOLOAD_RLC
    self.cmd_submit_buf()

  def hw_start(self):

    components_load_order = [
      (amdgpu_2.PSP_FW_TYPE_PSP_KDB, 0x80000),
      (amdgpu_2.PSP_FW_TYPE_PSP_KDB, 0x10000000),
      (amdgpu_2.PSP_FW_TYPE_PSP_SYS_DRV, 0x10000),
      (amdgpu_2.PSP_FW_TYPE_PSP_SOC_DRV, 0xB0000),
      (amdgpu_2.PSP_FW_TYPE_PSP_INTF_DRV, 0xD0000),
      (amdgpu_2.PSP_FW_TYPE_PSP_DBG_DRV, 0xC0000),
      (amdgpu_2.PSP_FW_TYPE_PSP_RAS_DRV, 0xE0000),
    ]
    for fw, compid in components_load_order: self.bootloader_load_component(fw, compid)
    self.bootloader_load_sos()
  
    self.ring_create()

    self.tmr_init()

    # For ASICs with DF Cstate management centralized to PMFW, TMR setup should be performed after PMFW loaded and before other non-psp firmware loaded.
    self.load_smu_fw()
    self.load_tmr()

    for fw in self.fw_list:
      self.prep_load_ip_fw_cmd_buf(fw)
      self.cmd_submit_buf()

    self.rlc_autoload_start()

  def wait_for_bootloader(self):
    for i in range(1000):
      reg = self.adev.rreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_35, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_35_BASE_IDX)
      # print(reg)
      if (reg & 0xffffffff) == 0x80000000: return 0
      time.sleep(0.01)
    raise RuntimeError("PSP: bootloader timeout")

  def bootloader_load_component(self, fw, compid):
    if fw not in self.sos_fw_infos: return 0
    if self.is_sos_alive(): return 0

    self.wait_for_bootloader()

    ctypes.memset(self.msg1_cpu_addr, 0, len(self.msg1_view))
    fwm = self.sos_fw_infos[fw]
    self.msg1_view[:len(fwm)] = fwm

    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_36, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_36_BASE_IDX, self.msg1_mc_addr >> 20)
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_35, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_35_BASE_IDX, compid)

    print(f"PSP: loading fw: {fw} {compid}")

    return self.wait_for_bootloader()
  
  def bootloader_load_sos(self):
    if (self.is_sos_alive()):
      self.init_sos_version()
      print(f"sOS alive, version {self.sos_fw_version}")
      return 0

    self.wait_for_bootloader()

    ctypes.memset(self.msg1_cpu_addr, 0, len(self.msg1_view))
    fwm = self.sos_fw_infos[amdgpu_2.PSP_FW_TYPE_PSP_SOS]
    self.msg1_view[:len(fwm)] = fwm

    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_36, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_36_BASE_IDX, self.msg1_mc_addr >> 20)
    self.adev.wreg_ip("MP0", 0, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_35, amdgpu_mp_13_0_0.regMP0_SMN_C2PMSG_35_BASE_IDX, 0x20000)

    # ret = psp_wait_for(psp, SOC15_REG_OFFSET(MP0, 0, regMP0_SMN_C2PMSG_81),
		# 	   RREG32_SOC15(MP0, 0, regMP0_SMN_C2PMSG_81),
		# 	   0, true);

    while not self.is_sos_alive(): time.sleep(0.01)
    self.init_sos_version()
    print(f"sOS alive, version {self.sos_fw_version}")
