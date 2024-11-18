import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0, amdgpu_psp_gfx_if
from tinygrad.helpers import to_mv, mv_address, colored
from extra.amdpci.firmware import Firmware

class PSP_IP:
  def __init__(self, adev):
    self.adev = adev

    self.msg1_pm = self.adev.mm.palloc(0x100000, align=0x100000)
    self.fence_pm = self.adev.mm.palloc(0x1000)
    self.cmd_pm = self.adev.mm.palloc(0x1000)
    self.ring_pm = self.adev.mm.palloc(0x10000)

  def is_sos_alive(self): return self.adev.regMP0_SMN_C2PMSG_81.read() != 0x0

  def init(self):
    print("PSP init")

    # Load sOS components
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

    # Load sOS itself
    self.bootloader_load_component(amdgpu_2.PSP_FW_TYPE_PSP_SOS, 0x20000)
    while not self.is_sos_alive(): time.sleep(0.01)
    self.sos_fw_version = self.adev.regMP0_SMN_C2PMSG_58.read()

    self.ring_create()
    self.tmr_init()

    self.load_ip_fw_cmd(self.adev.fw.smu_psp_desc)
    self.tmr_cmd()

    for psp_desc in self.adev.fw.descs: self.load_ip_fw_cmd(psp_desc)
    self.rlc_autoload_cmd()

  def wait_for_bootloader(self): self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_35, mask=0xFFFFFFFF, value=0x80000000)

  def prep_msg1(self, data):
    ctypes.memset(self.msg1_pm.cpu_addr(), 0, self.msg1_pm.size)
    self.msg1_pm.cpu_view()[:len(data)] = data

  def bootloader_load_component(self, fw, compid):
    if fw not in self.adev.fw.sos_fw: return 0
    if self.is_sos_alive(): return 0

    self.wait_for_bootloader()

    self.prep_msg1(self.adev.fw.sos_fw[fw])
    self.adev.regMP0_SMN_C2PMSG_36.write(self.msg1_pm.mc_addr() >> 20)
    self.adev.regMP0_SMN_C2PMSG_35.write(compid)

    return self.wait_for_bootloader()

  def ring_create(self):
    # Wait untill the sOS is ready
    self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_64, mask=0x80000000, value=0x80000000)

    self.adev.regMP0_SMN_C2PMSG_69.write(self.ring_pm.mc_addr() & 0xffffffff)
    self.adev.regMP0_SMN_C2PMSG_70.write(self.ring_pm.mc_addr() >> 32)
    self.adev.regMP0_SMN_C2PMSG_71.write(self.ring_pm.size)
    self.adev.regMP0_SMN_C2PMSG_64.write(2 << 16) # PSP_RING_TYPE__KM = 2. Kernel mode ring

    # there might be handshake issue with hardware which needs delay
    time.sleep(100 / 1000)

    self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_64, mask=0x8000FFFF, value=0x80000000)
    print("sOS ring created")

  def ring_set_wptr(self, wptr): self.adev.regMP0_SMN_C2PMSG_67.write(wptr)
  def ring_get_wptr(self): return self.adev.regMP0_SMN_C2PMSG_67.read()
  def ring_submit(self):
    prev_wptr = self.ring_get_wptr()
    ring_entry_addr = self.ring_pm.cpu_addr() + prev_wptr * 4

    ctypes.memset(ring_entry_addr, 0, ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame))
    write_loc = amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame.from_address(ring_entry_addr)
    write_loc.cmd_buf_addr_hi = self.cmd_pm.mc_addr() >> 32
    write_loc.cmd_buf_addr_lo = self.cmd_pm.mc_addr() & 0xffffffff
    write_loc.fence_addr_hi = self.fence_pm.mc_addr() >> 32
    write_loc.fence_addr_lo = self.fence_pm.mc_addr() & 0xffffffff
    write_loc.fence_value = prev_wptr

    self.ring_set_wptr(prev_wptr + ctypes.sizeof(amdgpu_psp_gfx_if.struct_psp_gfx_rb_frame) // 4)

    while self.fence_pm.cpu_view().cast('I')[0] != prev_wptr: self.adev.wreg(self.adev.reg_off("HDP", 0, 0x00d1, 0x0), 1)
    time.sleep(0.05)

    resp = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    if resp.resp.status != 0: print(colored(f"PSP command failed {resp.cmd_id} {resp.resp.status}", "red"))
    # else: print(colored(f"PSP command success {resp.cmd_id}", "green"))
    return resp

  def load_ip_fw_cmd(self, psp_desc):
    fw_type, fw_bytes = psp_desc

    self.prep_msg1(fw_bytes)
    ctypes.memset(self.cmd_pm.cpu_addr(), 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_LOAD_IP_FW
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = self.msg1_pm.mc_addr() & 0xffffffff
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi = self.msg1_pm.mc_addr() >> 32
    cmd.cmd.cmd_load_ip_fw.fw_size = len(fw_bytes)
    cmd.cmd.cmd_load_ip_fw.fw_type = fw_type
    return self.ring_submit()

  def tmr_cmd(self):
    ctypes.memset(self.cmd_pm.cpu_addr(), 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_SETUP_TMR
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = self.tmr_pm.mc_addr() & 0xffffffff
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi = self.tmr_pm.mc_addr() >> 32
    cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = self.tmr_pm.paddr & 0xffffffff
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi = self.tmr_pm.paddr >> 32
    cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
    cmd.cmd.cmd_setup_tmr.buf_size = self.tmr_pm.size
    return self.ring_submit()

  def load_toc_cmd(self, toc_size):
    ctypes.memset(self.cmd_pm.cpu_addr(), 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_LOAD_TOC
    cmd.cmd.cmd_load_toc.toc_phy_addr_lo = self.msg1_pm.mc_addr() & 0xffffffff
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi = self.msg1_pm.mc_addr() >> 32
    cmd.cmd.cmd_load_toc.toc_size = toc_size
    return self.ring_submit()

  def rlc_autoload_cmd(self):
    ctypes.memset(self.cmd_pm.cpu_addr(), 0, 0x1000)
    cmd = amdgpu_psp_gfx_if.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    cmd.cmd_id = amdgpu_psp_gfx_if.GFX_CMD_ID_AUTOLOAD_RLC
    self.ring_submit()
  
  def tmr_init(self):
    # Load toc
    self.prep_msg1(fwm:=self.adev.fw.sos_fw[amdgpu_2.PSP_FW_TYPE_PSP_TOC])
    resp = self.load_toc_cmd(len(fwm))

    self.tmr_pm = self.adev.mm.palloc(resp.resp.tmr_size, align=0x100000)
