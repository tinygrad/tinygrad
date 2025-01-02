from __future__ import annotations
import ctypes, time
from typing import Literal
from tinygrad.runtime.autogen import libpciaccess
from tinygrad.runtime.autogen.am import am, gc_11_0_0, smu_v13_0_0
from tinygrad.helpers import to_mv, data64, lo32, hi32

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
    self.vm_base = self.adev.mm.va_allocator.base
    self.vm_end = self.vm_base + self.adev.mm.va_allocator.size - 1

    self.memscratch_pm = self.adev.mm.palloc(0x1000)
    self.dummy_page_pm = self.adev.mm.palloc(0x1000)
    self.hub_initted = {"MM": False, "GC": False}

  def init(self): self.init_hub("MM")

  def flush_hdp(self): self.adev.regBIF_BX_PF0_GPU_HDP_FLUSH_REQ.write(0xffffffff)
  def flush_tlb(self, ip:Literal["MM", "GC"], vmid, flush_type=0):
    self.flush_hdp()

    # Can't issue TLB invalidation if the hub isn't initialized.
    if not self.hub_initted[ip]: return

    if ip == "MM": self.adev.wait_reg(self.adev.regMMVM_INVALIDATE_ENG17_SEM, mask=0x1, value=0x1)

    self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_REQ").write(flush_type=flush_type, per_vmid_invalidate_req=(1 << vmid), invalidate_l2_ptes=1,
      invalidate_l2_pde0=1, invalidate_l2_pde1=1, invalidate_l2_pde2=1, invalidate_l1_ptes=1, clear_protection_fault_status_addr=0)

    self.adev.wait_reg(self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_ACK"), mask=(1 << vmid), value=(1 << vmid))

    if ip == "MM":
      self.adev.regMMVM_INVALIDATE_ENG17_SEM.write(0x0)
      self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.update(reserved_cache_private_invalidation=1)

      # Read back the register to ensure the invalidation is complete
      self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.read()

  def enable_vm_addressing(self, page_table, ip:Literal["MM", "GC"], vmid):
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", self.vm_base >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", self.vm_end >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", page_table.pm.paddr | 1)
    self.adev.reg(f"reg{ip}VM_CONTEXT{vmid}_CNTL").write(0x1fffe00, enable_context=1, page_table_depth=(3 - page_table.lv))

  def init_hub(self, ip:Literal["MM", "GC"]):
    # Init system apertures
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

    self.enable_vm_addressing(self.adev.mm.root_page_table, ip, vmid=0)

    # Disable identity aperture
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", "_LO32", "_HI32", 0xfffffffff)
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", "_LO32", "_HI32", 0x0)
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "_LO32", "_HI32", 0x0)

    for eng_i in range(18): self.adev.wreg_pair(f"reg{ip}VM_INVALIDATE_ENG{eng_i}_ADDR_RANGE", "_LO32", "_HI32", 0x1fffffffff)
    self.hub_initted[ip] = True

  def on_interrupt(self):
    for ip in ["MM", "GC"]:
      st, va = self.adev.reg(f'reg{ip}VM_L2_PROTECTION_FAULT_STATUS').read(), self.adev.reg(f'reg{ip}VM_L2_PROTECTION_FAULT_ADDR_LO32').read()
      va = (va | (self.adev.reg(f'reg{ip}VM_L2_PROTECTION_FAULT_ADDR_HI32').read()) << 32) << 12
      if self.adev.reg(f"reg{ip}VM_L2_PROTECTION_FAULT_STATUS").read(): raise RuntimeError(f"{ip}VM_L2_PROTECTION_FAULT_STATUS: {st:#x} {va:#x}")

class AM_SMU(AM_IP):
  def init(self):
    self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_EnableAllSmuFeatures, 0, poll=True)

    for clck in [0x00000C94, 0x000204E1, 0x000105DC, 0x00050B76, 0x00070B76, 0x00040898, 0x00060898, 0x000308FD]:
      self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_SetSoftMinByFreq, clck, poll=True)
      self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_SetSoftMaxByFreq, clck, poll=True)

  def mode1_reset(self):
    self._smu_cmn_send_smc_msg_with_param(smu_v13_0_0.PPSMC_MSG_Mode1Reset, 0, poll=True)
    time.sleep(0.5)

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
    self.adev.regRLC_SRM_CNTL.update(srm_enable=1, auto_incr_addr=1)

    self.adev.regGRBM_CNTL.update(read_timeout=0xff)
    for i in range(0, 16):
      self._grbm_select(vmid=i)
      self.adev.regSH_MEM_CONFIG.write(address_mode=am.SH_MEM_ADDRESS_MODE_64, alignment_mode=am.SH_MEM_ALIGNMENT_MODE_UNALIGNED,
                                       initial_inst_prefetch=3)

      # Configure apertures:
      # LDS:         0x10000000'00000000 - 0x10000001'00000000 (4GB)
      # Scratch:     0x20000000'00000000 - 0x20000001'00000000 (4GB)
      self.adev.regSH_MEM_BASES.write(shared_base=0x1, private_base=0x2)
    self._grbm_select()

    # Configure MEC doorbell range
    self.adev.regCP_MEC_DOORBELL_RANGE_LOWER.write(0x0)
    self.adev.regCP_MEC_DOORBELL_RANGE_UPPER.write(0x450)

    # Enable MEC
    self.adev.regCP_MEC_RS64_CNTL.update(mec_invalidate_icache=0, mec_pipe0_reset=0, mec_pipe1_reset=0, mec_pipe2_reset=0, mec_pipe3_reset=0,
                                         mec_pipe0_active=1, mec_pipe1_active=1, mec_pipe2_active=1, mec_pipe3_active=1, mec_halt=0)

    # NOTE: Wait for MEC to be ready. The kernel does udelay here as well.
    time.sleep(0.5)

  def setup_ring(self, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, eop_addr:int, eop_size:int, doorbell:int, pipe:int, queue:int):
    mqd = self.adev.mm.valloc(0x1000, uncached=True, contigous=True)

    mqd_struct = am.struct_v11_compute_mqd(header=0xC0310800, cp_mqd_base_addr_lo=lo32(mqd.va_addr), cp_mqd_base_addr_hi=hi32(mqd.va_addr),
      cp_hqd_persistent_state=self.adev.regCP_HQD_PERSISTENT_STATE.build(preload_size=0x55, preload_req=1),
      cp_hqd_pipe_priority=0x2, cp_hqd_queue_priority=0xf, cp_hqd_quantum=0x111,
      cp_hqd_pq_base_lo=lo32(ring_addr>>8), cp_hqd_pq_base_hi=hi32(ring_addr>>8),
      cp_hqd_pq_rptr_report_addr_lo=lo32(rptr_addr), cp_hqd_pq_rptr_report_addr_hi=hi32(rptr_addr),
      cp_hqd_pq_wptr_poll_addr_lo=lo32(wptr_addr), cp_hqd_pq_wptr_poll_addr_hi=hi32(wptr_addr),
      cp_hqd_pq_doorbell_control=self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.build(doorbell_offset=doorbell*2, doorbell_en=1),
      cp_hqd_pq_control=self.adev.regCP_HQD_PQ_CONTROL.build(rptr_block_size=5, unord_dispatch=1, queue_size=(ring_size//4).bit_length()-2),
      cp_hqd_ib_control=self.adev.regCP_HQD_IB_CONTROL.build(min_ib_avail_size=0x3), cp_hqd_hq_status0=0x20004000,
      cp_mqd_control=self.adev.regCP_MQD_CONTROL.build(priv_state=1), cp_hqd_vmid=0,
      cp_hqd_eop_base_addr_lo=lo32(eop_addr>>8), cp_hqd_eop_base_addr_hi=hi32(eop_addr>>8),
      cp_hqd_eop_control=self.adev.regCP_HQD_EOP_CONTROL.build(eop_size=(eop_size//4).bit_length()-2))

    # Copy mqd into memory
    ctypes.memmove(mqd.cpu_addr, ctypes.addressof(mqd_struct), ctypes.sizeof(mqd_struct))
    self.adev.gmc.flush_hdp()

    self._grbm_select(me=1, pipe=pipe, queue=queue)

    mqd_st_mv = to_mv(ctypes.addressof(mqd_struct), ctypes.sizeof(mqd_struct)).cast('I')
    for i, reg in enumerate(range(self.adev.regCP_MQD_BASE_ADDR.reg_off, self.adev.regCP_HQD_PQ_WPTR_HI.reg_off + 1)):
      self.adev.wreg(reg, mqd_st_mv[0x80 + i])
    self.adev.regCP_HQD_ACTIVE.write(0x1)

    self._grbm_select()

    self.adev.reg(f"regCP_ME1_PIPE{pipe}_INT_CNTL").update(time_stamp_int_enable=1, generic0_int_enable=1)

  def set_clockgating_state(self):
    self.adev.regRLC_SAFE_MODE.write(message=1, cmd=1)
    self.adev.wait_reg(self.adev.regRLC_SAFE_MODE, mask=0x1, value=0x0)

    self.adev.regRLC_CGCG_CGLS_CTRL.update(cgcg_gfx_idle_threshold=0x36, cgcg_en=1, cgls_rep_compansat_delay=0xf, cgls_en=1)

    self.adev.regCP_RB_WPTR_POLL_CNTL.update(poll_frequency=0x100, idle_poll_count=0x90)
    self.adev.regCP_INT_CNTL.update(cntx_busy_int_enable=1, cntx_empty_int_enable=1, cmp_busy_int_enable=1, gfx_idle_int_enable=1)
    self.adev.regSDMA0_RLC_CGCG_CTRL.update(cgcg_int_enable=1)

    self.adev.regRLC_CGTT_MGCG_OVERRIDE.update(perfmon_clock_state=0, gfxip_fgcg_override=0, gfxip_repeater_fgcg_override=0,
      grbm_cgtt_sclk_override=0, rlc_cgtt_sclk_override=0, gfxip_mgcg_override=0, gfxip_cgls_override=0)

    self.adev.regRLC_SAFE_MODE.write(message=0, cmd=1)

  def _grbm_select(self, me=0, pipe=0, queue=0, vmid=0): self.adev.regGRBM_GFX_CNTL.write(meid=me, pipeid=pipe, vmid=vmid, queueid=queue)

  def _wait_for_rlc_autoload(self):
    while True:
      bootload_ready = (self.adev.regRLC_RLCS_BOOTLOAD_STATUS.read() & gc_11_0_0.RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) != 0
      if self.adev.regCP_STAT.read() == 0 and bootload_ready: break

  def _config_gfx_rs64(self):
    def _config_helper(eng_name, cntl_reg, eng_reg, pipe_cnt, me=0):
      for pipe in range(pipe_cnt):
        self._grbm_select(me=me, pipe=pipe)
        self.adev.wreg_pair(f"regCP_{eng_reg}_PRGRM_CNTR_START", "", "_HI", self.adev.fw.ucode_start[eng_name] >> 2)
      self._grbm_select()
      self.adev.reg(f"regCP_{cntl_reg}_CNTL").update(**{f"{eng_name.lower()}_pipe{pipe}_reset": 1 for pipe in range(pipe_cnt)})
      self.adev.reg(f"regCP_{cntl_reg}_CNTL").update(**{f"{eng_name.lower()}_pipe{pipe}_reset": 0 for pipe in range(pipe_cnt)})

    _config_helper(eng_name="PFP", cntl_reg="ME", eng_reg="PFP", pipe_cnt=2)
    _config_helper(eng_name="ME", cntl_reg="ME", eng_reg="ME", pipe_cnt=2)
    _config_helper(eng_name="MEC", cntl_reg="MEC_RS64", eng_reg="MEC_RS64", pipe_cnt=4, me=1)

class AM_IH(AM_IP):
  def interrupt_handler(self):
    ring_vm, rwptr_vm, suf, _ = self.rings[0]
    wptr = to_mv(rwptr_vm.cpu_addr, 8).cast('Q')[0]

    if self.adev.reg(f"regIH_RB_WPTR{suf}").read(rb_overflow=1):
      self.adev.reg(f"regIH_RB_WPTR{suf}").update(rb_overflow=0)
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(wptr_overflow_clear=1)
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(wptr_overflow_clear=0)
    self.adev.regIH_RB_RPTR.write(wptr % ring_vm.size)

  def init(self):
    self.rings = [(self.adev.mm.valloc(1 << 20, uncached=True, contigous=True), self.adev.mm.valloc(0x1000, uncached=True, contigous=True), "", 0),
      (self.adev.mm.valloc(1 << 20, uncached=True, contigous=True), self.adev.mm.valloc(0x1000, uncached=True, contigous=True), "_RING1", 1)]

    for ring_vm, rwptr_vm, suf, ring_id in self.rings:
      self.adev.wreg_pair("regIH_RB_BASE", suf, f"_HI{suf}", ring_vm.va_addr >> 8)

      self.adev.reg(f"regIH_RB_CNTL{suf}").write(mc_space=4, wptr_overflow_clear=1, rb_size=(ring_vm.size//4).bit_length(),
        mc_snoop=1, mc_ro=0, mc_vmid=0, **({'wptr_overflow_enable': 1, 'rptr_rearm': 1} if ring_id == 0 else {'rb_full_drain_enable': 1}))

      if ring_id == 0: self.adev.wreg_pair("regIH_RB_WPTR_ADDR", "_LO", "_HI", rwptr_vm.va_addr)

      self.adev.reg(f"regIH_RB_WPTR{suf}").write(0)
      self.adev.reg(f"regIH_RB_RPTR{suf}").write(0)

      self.adev.reg(f"regIH_DOORBELL_RPTR{suf}").write(((am.AMDGPU_NAVI10_DOORBELL_IH + ring_id) * 2), enable=1)

    self.adev.regIH_STORM_CLIENT_LIST_CNTL.update(client18_is_storm_client=1)
    self.adev.regIH_INT_FLOOD_CNTL.update(flood_cntl_enable=1)
    self.adev.regIH_MSI_STORM_CTRL.update(delay=3)

    libpciaccess.pci_device_cfg_read_u16(self.adev.pcidev, ctypes.byref(val:=ctypes.c_uint16()), libpciaccess.PCI_COMMAND)
    libpciaccess.pci_device_cfg_write_u16(self.adev.pcidev, val.value | libpciaccess.PCI_COMMAND_MASTER, libpciaccess.PCI_COMMAND)

    # toggle interrupts
    for _, rwptr_vm, suf, ring_id in self.rings:
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(rb_enable=1, **({'enable_intr': 1} if ring_id == 0 else {}))

class AM_SDMA(AM_IP):
  def setup_ring(self, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, doorbell:int, pipe:int, queue:int):
    # Stop if something is running...
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_RB_CNTL").update(rb_enable=0)
    while not self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_CONTEXT_STATUS").read(idle=1): pass

    # Setup the ring
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_RPTR", "", "_HI", 0)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_WPTR", "", "_HI", 0)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_BASE", "", "_HI", ring_addr >> 8)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_RPTR_ADDR", "_LO", "_HI", rptr_addr)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_WPTR_POLL_ADDR", "_LO", "_HI", wptr_addr)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_DOORBELL_OFFSET").update(offset=doorbell * 2)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_DOORBELL").update(enable=1)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_RB_CNTL").write(rb_vmid=0, rptr_writeback_enable=1, rptr_writeback_timer=4,
      f32_wptr_poll_enable=1, rb_size=(ring_size//4).bit_length()-1, rb_enable=1, rb_priv=1)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_IB_CNTL").update(ib_enable=1)

  def init(self):
    self.adev.regSDMA0_SEM_WAIT_FAIL_TIMER_CNTL.write(0x0)
    self.adev.regSDMA0_WATCHDOG_CNTL.update(queue_hang_count=100) # 10s, 100ms per unit
    self.adev.regSDMA0_UTCL1_CNTL.update(resp_mode=3, redo_delay=9)
    self.adev.regSDMA0_UTCL1_PAGE.update(rd_l2_policy=0x2, wr_l2_policy=0x3, llc_noalloc=1) # rd=noa, wr=bypass
    self.adev.regSDMA0_F32_CNTL.update(halt=0, th1_reset=0)
    self.adev.regSDMA0_CNTL.update(ctxempty_int_enable=1, trap_enable=1)

class AM_PSP(AM_IP):
  def __init__(self, adev):
    super().__init__(adev)

    self.msg1_pm = self.adev.mm.palloc(am.PSP_1_MEG, align=am.PSP_1_MEG)
    self.cmd_pm = self.adev.mm.palloc(am.PSP_CMD_BUFFER_SIZE)
    self.fence_pm = self.adev.mm.palloc(am.PSP_FENCE_BUFFER_SIZE)
    self.ring_pm = self.adev.mm.palloc(0x10000)

  def is_sos_alive(self): return self.adev.regMP0_SMN_C2PMSG_81.read() != 0x0
  def init(self):
    sos_components_load_order = [
      (am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE), (am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_TOS_SPL_TABLE),
      (am.PSP_FW_TYPE_PSP_SYS_DRV, am.PSP_BL__LOAD_SYSDRV), (am.PSP_FW_TYPE_PSP_SOC_DRV, am.PSP_BL__LOAD_SOCDRV),
      (am.PSP_FW_TYPE_PSP_INTF_DRV, am.PSP_BL__LOAD_INTFDRV), (am.PSP_FW_TYPE_PSP_DBG_DRV, am.PSP_BL__LOAD_DBGDRV),
      (am.PSP_FW_TYPE_PSP_RAS_DRV, am.PSP_BL__LOAD_RASDRV), (am.PSP_FW_TYPE_PSP_SOS, am.PSP_BL__LOAD_SOSDRV)]

    for fw, compid in sos_components_load_order: self._bootloader_load_component(fw, compid)
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

    self.tmr_pm = self.adev.mm.palloc(resp.resp.tmr_size, align=am.PSP_TMR_ALIGNMENT)

  def _ring_create(self):
    # Wait until the sOS is ready
    self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_64, mask=0x80000000, value=0x80000000)

    self.adev.wreg_pair("regMP0_SMN_C2PMSG", "_69", "_70", self.ring_pm.mc_addr())
    self.adev.regMP0_SMN_C2PMSG_71.write(self.ring_pm.size)
    self.adev.regMP0_SMN_C2PMSG_64.write(am.PSP_RING_TYPE__KM << 16)

    # There might be handshake issue with hardware which needs delay
    time.sleep(0.1)

    self.adev.wait_reg(self.adev.regMP0_SMN_C2PMSG_64, mask=0x8000FFFF, value=0x80000000)

  def _ring_submit(self):
    prev_wptr = self.adev.regMP0_SMN_C2PMSG_67.read()
    ring_entry_addr = self.ring_pm.cpu_addr() + prev_wptr * 4

    ctypes.memset(ring_entry_addr, 0, ctypes.sizeof(am.struct_psp_gfx_rb_frame))
    write_loc = am.struct_psp_gfx_rb_frame.from_address(ring_entry_addr)
    write_loc.cmd_buf_addr_hi, write_loc.cmd_buf_addr_lo = data64(self.cmd_pm.mc_addr())
    write_loc.fence_addr_hi, write_loc.fence_addr_lo = data64(self.fence_pm.mc_addr())
    write_loc.fence_value = prev_wptr

    # Move the wptr
    self.adev.regMP0_SMN_C2PMSG_67.write(prev_wptr + ctypes.sizeof(am.struct_psp_gfx_rb_frame) // 4)

    while self.fence_pm.cpu_view().cast('I')[0] != prev_wptr: pass
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
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi, cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = data64(self.msg1_pm.mc_addr())
    cmd.cmd.cmd_load_ip_fw.fw_size = len(fw_bytes)
    cmd.cmd.cmd_load_ip_fw.fw_type = fw_type
    return self._ring_submit()

  def _tmr_load_cmd(self):
    cmd = self._prep_ring_cmd(am.GFX_CMD_ID_SETUP_TMR)
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi, cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = data64(self.tmr_pm.mc_addr())
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi, cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = data64(self.tmr_pm.paddr)
    cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
    cmd.cmd.cmd_setup_tmr.buf_size = self.tmr_pm.size
    return self._ring_submit()

  def _load_toc_cmd(self, toc_size):
    cmd = self._prep_ring_cmd(am.GFX_CMD_ID_LOAD_TOC)
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi, cmd.cmd.cmd_load_toc.toc_phy_addr_lo = data64(self.msg1_pm.mc_addr())
    cmd.cmd.cmd_load_toc.toc_size = toc_size
    return self._ring_submit()

  def _rlc_autoload_cmd(self):
    self._prep_ring_cmd(am.GFX_CMD_ID_AUTOLOAD_RLC)
    return self._ring_submit()
