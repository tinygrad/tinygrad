from __future__ import annotations
import ctypes, time
from tinygrad.runtime.autogen.am import am, gc_11_0_0, smu_v13_0_0
from tinygrad.runtime.support.am.amring import AMRing

class AM_IP:
  def __init__(self, adev): self.adev = adev
  def init(self): raise NotImplementedError("IP block init must be implemeted")

class AM_SOC21(AM_IP):
  def init(self):
    self.adev.regRCC_DEV0_EPF2_STRAP2.update(strap_no_soft_reset_dev0_f2=0x0)
    self.adev.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN.write(0x1)

class AM_GMC(AM_IP):
  def __init__(self, adev):
    super().__init__(adev)

    # Memory controller aperture
    self.mc_base = self.adev.regMMMC_VM_FB_LOCATION_BASE.read() << 24
    self.mc_end = self.mc_base + self.adev.mm.vram_size - 1

    # VM aperture
    self.vm_base = 0x7F0000000000
    self.vm_end = self.vm_base + (512 * (1 << 30)) - 1

    self.memscratch_pm = self.adev.mm.palloc(0x1000)
    self.dummy_page_pm = self.adev.mm.palloc(0x1000)
    self.hub_initted = {"MM": False, "GC": False}

  def init(self): self.init_hub("MM")

  def flush_hdp(self): self.adev.wreg(0x1fc00, 0x0) # TODO: write up!
  def flush_tlb(self, ip:Union["MM", "GC"], vmid, flush_type=0):
    self.flush_hdp()

    # Check if the hub is initialized
    if not self.hub_initted[ip]: return

    if ip == "MM": x = self.adev.wait_reg(self.adev.regMMVM_INVALIDATE_ENG17_SEM, mask=0x1, value=0x1)

    self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_REQ").write(flush_type=flush_type, per_vmid_invalidate_req=(1 << vmid), invalidate_l2_ptes=1,
      invalidate_l2_pde0=1, invalidate_l2_pde1=1, invalidate_l2_pde2=1, invalidate_l1_ptes=1, clear_protection_fault_status_addr=0)

    self.adev.wait_reg(self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_ACK"), mask=(1 << vmid), value=(1 << vmid))

    if ip == "MM":
      self.adev.regMMVM_INVALIDATE_ENG17_SEM.write(0x0)
      self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.update(reserved_cache_private_invalidation=1)

      # Read back the register to ensure the invalidation is complete
      self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.read()

  def enable_vm_addressing(self, page_table, ip:Union["MM", "GC"], vmid):
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", page_table.pm.paddr | 1)
    self.adev.reg(f"reg{ip}VM_CONTEXT{vmid}_CNTL").write(0x1fffe00, enable_context=1, page_table_depth=2)

  def init_hub(self, ip:Union["MM", "GC"]):
    # Init system apertures
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT0_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", self.vm_base >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT0_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", self.vm_end >> 12)

    self.adev.reg(f"reg{ip}MC_VM_AGP_BASE").write(0)
    self.adev.reg(f"reg{ip}MC_VM_AGP_BOT").write(0xffffffffffff >> 24) # disable AGP
    self.adev.reg(f"reg{ip}MC_VM_AGP_TOP").write(0)

    self.adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_LOW_ADDR").write(self.mc_base >> 18)
    self.adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(self.mc_end >> 18)
    self.adev.wreg_pair(f"reg{ip}MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR", "_LSB", "_MSB", self.memscratch_pm.paddr >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_L2_PROTECTION_FAULT_DEFAULT_ADDR", "_LO32", "_HI32", self.dummy_page_pm.paddr >> 12)

    self.adev.reg(f"reg{ip}VM_L2_PROTECTION_FAULT_CNTL2").update(active_page_migration_pte_read_retry=1)

    # Init TLB and cache
    self.adev.reg(f"reg{ip}MC_VM_MX_L1_TLB_CNTL").update(enable_l1_tlb=1, system_access_mode=3, enable_advanced_driver_model=1,
                                                         system_aperture_unmapped_access=0, eco_bits=0, mtype=am.MTYPE_UC)

    self.adev.reg(f"reg{ip}VM_L2_CNTL").update(enable_l2_cache=1, enable_l2_fragment_processing=0, enable_default_page_out_to_system_memory=1,
      l2_pde0_cache_tag_generation_mode=0, pde_fault_classification=0, context1_identity_access_mode=1, identity_mode_fragment_size=0)
    self.adev.reg(f"reg{ip}VM_L2_CNTL2").update(invalidate_all_l1_tlbs=1, invalidate_l2_cache=1)
    self.adev.reg(f"reg{ip}VM_L2_CNTL3").write(bank_select=9, l2_cache_bigk_fragment_size=6,l2_cache_4k_associativity=1,l2_cache_bigk_associativity=1)
    self.adev.reg(f"reg{ip}VM_L2_CNTL4").write(l2_cache_4k_partition_count=1)
    self.adev.reg(f"reg{ip}VM_L2_CNTL5").write(walker_priority_client_id=0x1ff)

    self.enable_vm_addressing(self.adev.mm.root_pt, ip, vmid=0)

    # Disable identity aperture
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", "_LO32", "_HI32", 0xfffffffff)
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", "_LO32", "_HI32", 0x0)
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "_LO32", "_HI32", 0x0)

    for eng_i in range(18): self.adev.wreg_pair(f"reg{ip}VM_INVALIDATE_ENG{eng_i}_ADDR_RANGE", "_LO32", "_HI32", 0x1fffffffff)
    self.hub_initted[ip] = True

class AM_SMU(AM_IP):
  def init(self):
    self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_RunDcBtc, 0, poll=True)
    self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_EnableAllSmuFeatures, 0, poll=True)

    self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_SetWorkloadMask, 0x24, poll=True)
    for clck in [0x00000C94, 0x000204E1, 0x000105DC, 0x00050B76, 0x00070B76, 0x00040898, 0x00060898, 0x000308FD]:
      self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_SetSoftMinByFreq, clck, poll=True)
      self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_SetSoftMaxByFreq, clck, poll=True)

  def mode1_reset(self): self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_Mode1Reset, 0, poll=True)

  def _smu_cmn_poll_stat(self): self.adev.wait_reg(self.adev.mmMP1_SMN_C2PMSG_90, mask=0xFFFFFFFF, value=1)
  def _smu_cmn_send_msg(self, msg, param=0):
    self.adev.mmMP1_SMN_C2PMSG_90.write(0) # resp reg
    self.adev.mmMP1_SMN_C2PMSG_82.write(param)
    self.adev.mmMP1_SMN_C2PMSG_66.write(msg)

  def _smu_cmn_send_smc_msg_with_param(self, msg, param, poll=True, read_back_arg=False):
    if poll: self._smu_cmn_poll_stat()

    self._smu_cmn_send_msg(msg, param)
    self._smu_cmn_poll_stat()
    return self.adev.rreg(self.adev.mmMP1_SMN_C2PMSG_82) if read_back_arg else None

class AM_GFX(AM_IP):
  def init(self):
    self._wait_for_rlc_autoload()
    self._config_gfx_rs64()
    self.adev.gmc.init_hub("GC")

    # NOTE: Golden reg for gfx11. No values for this reg provided. The kernel just ors 0x20000000 to this reg.
    self.adev.regTCP_CNTL.write(self.adev.regTCP_CNTL.read() | 0x20000000)

    self.adev.regGRBM_CNTL.update(read_timeout=0xff)
    for i in range(0, 16):
      self._grbm_select(vmid=i)
      self.adev.regSH_MEM_CONFIG.write(address_mode=am.SH_MEM_ADDRESS_MODE_64, alignment_mode=am.SH_MEM_ALIGNMENT_MODE_UNALIGNED, initial_inst_prefetch=3)

      # Configure apertures:
      # LDS:         0x60000000'00000000 - 0x60000001'00000000 (4GB)
      # Scratch:     0x60000001'00000000 - 0x60000002'00000000 (4GB)
      # GPUVM:       0x60010000'00000000 - 0x60020000'00000000 (1TB)
      self.adev.regSH_MEM_BASES.write(shared_base=0x1, private_base=0x2)
    self._grbm_select()

    # Setup doorbels. TODO: write up
    self.adev.regS2A_DOORBELL_ENTRY_0_CTRL.write(0x30000007)
    self.adev.regS2A_DOORBELL_ENTRY_3_CTRL.write(0x3000000d)
    self.adev.regCP_RB_DOORBELL_RANGE_LOWER.write(0x458)
    self.adev.regCP_RB_DOORBELL_RANGE_UPPER.write(0x7f8)
    self.adev.regCP_MEC_DOORBELL_RANGE_LOWER.write(0x0)
    self.adev.regCP_MEC_DOORBELL_RANGE_UPPER.write(0x450)

    # Enable MEC
    self.adev.regCP_MEC_RS64_CNTL.update(mec_invalidate_icache=0, mec_pipe0_reset=0, mec_pipe1_reset=0, mec_pipe2_reset=0, mec_pipe3_reset=0,
                                         mec_pipe0_active=1, mec_pipe1_active=1, mec_pipe2_active=1, mec_pipe3_active=1, mec_halt=0)

    # NOTE: Wait for MEC to be ready. The kernel does udelay as well.
    time.sleep(0.005)

    # TODO: better to move out this.
    self.kcq_ring = AMRing(self.adev, size=0x100000, me=1, pipe=0, queue=1, vmid=0, doorbell_index=(0x3 << 1))
    self.load_ring(self.kcq_ring)
    time.sleep(2)

  def load_ring(self, ring:AMRing):
    self._grbm_select(me=1, pipe=ring.pipe, queue=ring.queue)

    for i, reg in enumerate(range(self.adev.regCP_MQD_BASE_ADDR.reg_off, self.adev.regCP_HQD_PQ_WPTR_HI.reg_off + 1)):
      self.adev.wreg(reg, ring.mqd_mv.cast('I')[0x80 + i])

    self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.write(ring.mqd.cp_hqd_pq_doorbell_control)
    self.adev.regCP_HQD_ACTIVE.write(0x1)

    self._grbm_select()

  def _grbm_select(self, me=0, pipe=0, queue=0, vmid=0): self.adev.regGRBM_GFX_CNTL.write(meid=me, pipeid=pipe, vmid=vmid, queueid=queue)

  def _wait_for_rlc_autoload(self):
    while True:
      bootload_ready = (self.adev.regRLC_RLCS_BOOTLOAD_STATUS.read() & gc_11_0_0.RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) != 0
      if self.adev.regCP_STAT.read() == 0 and bootload_ready: break

  def _config_gfx_rs64(self):
    for pipe in range(2):
      self._grbm_select(pipe=pipe)
      self.adev.wreg_pair("regCP_PFP_PRGRM_CNTR_START", "", "_HI", self.adev.fw.ucode_start['PFP'] >> 2)
    self._grbm_select()
    self.adev.regCP_ME_CNTL.update(pfp_pipe0_reset=1, pfp_pipe1_reset=1)
    self.adev.regCP_ME_CNTL.update(pfp_pipe0_reset=0, pfp_pipe1_reset=0)
    
    for pipe in range(2):
      self._grbm_select(pipe=pipe)
      self.adev.wreg_pair("regCP_ME_PRGRM_CNTR_START", "", "_HI", self.adev.fw.ucode_start['ME'] >> 2)
    self._grbm_select()
    self.adev.regCP_ME_CNTL.update(me_pipe0_reset=1, me_pipe1_reset=1)
    self.adev.regCP_ME_CNTL.update(me_pipe0_reset=0, me_pipe1_reset=0)

    for pipe in range(4):
      self._grbm_select(me=1, pipe=pipe)
      self.adev.wreg_pair("regCP_MEC_RS64_PRGRM_CNTR_START", "", "_HI", self.adev.fw.ucode_start['MEC'] >> 2)
    self._grbm_select()
    self.adev.regCP_MEC_RS64_CNTL.update(mec_pipe0_reset=1, mec_pipe1_reset=1, mec_pipe2_reset=1, mec_pipe3_reset=1)
    self.adev.regCP_MEC_RS64_CNTL.update(mec_pipe0_reset=0, mec_pipe1_reset=0, mec_pipe2_reset=0, mec_pipe3_reset=0)

class AM_PSP(AM_IP):
  def __init__(self, adev):
    super().__init__(adev)

    self.msg1_pm = self.adev.mm.palloc(0x100000, align=0x100000)
    self.fence_pm = self.adev.mm.palloc(0x1000)
    self.cmd_pm = self.adev.mm.palloc(0x1000)
    self.ring_pm = self.adev.mm.palloc(0x10000)

  def is_sos_alive(self): return self.adev.regMP0_SMN_C2PMSG_81.read() != 0x0
  def init(self):
    # Load sOS components. TODO: parse compid
    components_load_order = [
      (am.PSP_FW_TYPE_PSP_KDB, 0x80000),
      (am.PSP_FW_TYPE_PSP_KDB, 0x10000000),
      (am.PSP_FW_TYPE_PSP_SYS_DRV, 0x10000),
      (am.PSP_FW_TYPE_PSP_SOC_DRV, 0xB0000),
      (am.PSP_FW_TYPE_PSP_INTF_DRV, 0xD0000),
      (am.PSP_FW_TYPE_PSP_DBG_DRV, 0xC0000),
      (am.PSP_FW_TYPE_PSP_RAS_DRV, 0xE0000),
      (am.PSP_FW_TYPE_PSP_SOS, 0x20000),
    ]
    for fw, compid in components_load_order: self._bootloader_load_component(fw, compid)
    while not self.is_sos_alive(): time.sleep(0.01)

    self._ring_create()
    self._tmr_init()

    # SMU fw should be loaded before TMR.
    self._load_ip_fw_cmd(self.adev.fw.smu_psp_desc)
    self._tmr_load_cmd()

    for psp_desc in self.adev.fw.descs: self._load_ip_fw_cmd(psp_desc)
    self._rlc_autoload_cmd()

  def _wait_for_bootloader(self): self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_35, mask=0xFFFFFFFF, value=0x80000000)

  def _prep_msg1(self, data):
    ctypes.memset(self.msg1_pm.cpu_addr(), 0, self.msg1_pm.size)
    self.msg1_pm.cpu_view()[:len(data)] = data
    self.adev.gmc.flush_hdp()

  def _bootloader_load_component(self, fw, compid):
    if fw not in self.adev.fw.sos_fw: return 0

    self._wait_for_bootloader()

    self._prep_msg1(self.adev.fw.sos_fw[fw])
    self.adev.regMP0_SMN_C2PMSG_36.write(self.msg1_pm.mc_addr() >> 20)
    self.adev.regMP0_SMN_C2PMSG_35.write(compid)

    return self._wait_for_bootloader()

  def _tmr_init(self):
    # Load TOC and calculate TMR size
    self._prep_msg1(fwm:=self.adev.fw.sos_fw[am.PSP_FW_TYPE_PSP_TOC])
    resp = self._load_toc_cmd(len(fwm))

    self.tmr_pm = self.adev.mm.palloc(resp.resp.tmr_size, align=0x100000)

  def _ring_create(self):
    # Wait until the sOS is ready
    self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_64, mask=0x80000000, value=0x80000000)

    self.adev.wreg_pair("regMP0_SMN_C2PMSG", "_69", "_70", self.ring_pm.mc_addr())
    self.adev.regMP0_SMN_C2PMSG_71.write(self.ring_pm.size)
    self.adev.regMP0_SMN_C2PMSG_64.write(2 << 16) # PSP_RING_TYPE__KM = 2. Kernel mode ring

    # There might be handshake issue with hardware which needs delay
    time.sleep(100 / 1000)

    self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_64, mask=0x8000FFFF, value=0x80000000)
  
  def _ring_submit(self):
    prev_wptr = self.adev.regMP0_SMN_C2PMSG_67.read()
    ring_entry_addr = self.ring_pm.cpu_addr() + prev_wptr * 4

    ctypes.memset(ring_entry_addr, 0, ctypes.sizeof(am.struct_psp_gfx_rb_frame))
    write_loc = am.struct_psp_gfx_rb_frame.from_address(ring_entry_addr)
    write_loc.cmd_buf_addr_hi = self.cmd_pm.mc_addr() >> 32
    write_loc.cmd_buf_addr_lo = self.cmd_pm.mc_addr() & 0xffffffff
    write_loc.fence_addr_hi = self.fence_pm.mc_addr() >> 32
    write_loc.fence_addr_lo = self.fence_pm.mc_addr() & 0xffffffff
    write_loc.fence_value = prev_wptr

    # Move the wptr
    self.adev.regMP0_SMN_C2PMSG_67.write(prev_wptr + ctypes.sizeof(am.struct_psp_gfx_rb_frame) // 4)

    while self.fence_pm.cpu_view().cast('I')[0] != prev_wptr: pass #self.adev.wreg(self.adev.reg_off("HDP", 0, 0x00d1, 0x0), 1)
    time.sleep(0.05)

    resp = am.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    if resp.resp.status != 0: raise RuntimeError(f"PSP command failed {resp.cmd_id} {resp.resp.status}")

    return resp

  def _prep_ring_cmd(self, hdr): 
    ctypes.memset(self.cmd_pm.cpu_addr(), 0, 0x1000)
    cmd = am.struct_psp_gfx_cmd_resp.from_address(self.cmd_pm.cpu_addr())
    cmd.cmd_id = hdr
    return cmd

  def _load_ip_fw_cmd(self, psp_desc):
    fw_type, fw_bytes = psp_desc

    self._prep_msg1(fw_bytes)
    cmd = self._prep_ring_cmd(am.GFX_CMD_ID_LOAD_IP_FW)
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = self.msg1_pm.mc_addr() & 0xffffffff
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi = self.msg1_pm.mc_addr() >> 32
    cmd.cmd.cmd_load_ip_fw.fw_size = len(fw_bytes)
    cmd.cmd.cmd_load_ip_fw.fw_type = fw_type
    return self._ring_submit()

  def _tmr_load_cmd(self):
    cmd = self._prep_ring_cmd(am.GFX_CMD_ID_SETUP_TMR)
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = self.tmr_pm.mc_addr() & 0xffffffff
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi = self.tmr_pm.mc_addr() >> 32
    cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = self.tmr_pm.paddr & 0xffffffff
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi = self.tmr_pm.paddr >> 32
    cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
    cmd.cmd.cmd_setup_tmr.buf_size = self.tmr_pm.size
    return self._ring_submit()

  def _load_toc_cmd(self, toc_size):
    cmd = self._prep_ring_cmd(am.GFX_CMD_ID_LOAD_TOC)
    cmd.cmd.cmd_load_toc.toc_phy_addr_lo = self.msg1_pm.mc_addr() & 0xffffffff
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi = self.msg1_pm.mc_addr() >> 32
    cmd.cmd.cmd_load_toc.toc_size = toc_size
    return self._ring_submit()

  def _rlc_autoload_cmd(self):
    self._prep_ring_cmd(am.GFX_CMD_ID_AUTOLOAD_RLC)
    return self._ring_submit()
