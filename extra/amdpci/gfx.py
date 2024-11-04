import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class GFX_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c

  def __init__(self, adev):
    self.adev = adev
    self.eop_gpu_addr = self.adev.vmm.alloc_vram(0x1000, "eop")
    
    self.setup()
    pass

  def wait_for_rlc_autoload(self):
    # return # skip when load with amdgpu driver

    while True:
      cp_status = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_STAT, amdgpu_gc_11_0_0.regCP_STAT_BASE_IDX)

      # TODO: some exceptions here for other gpus
      bootload_status = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_RLCS_BOOTLOAD_STATUS, amdgpu_gc_11_0_0.regRLC_RLCS_BOOTLOAD_STATUS_BASE_IDX)

      print("cp_status", hex(cp_status), "bootload_status", hex(bootload_status))
      RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK = 0x80000000
      if cp_status == 0 and ((bootload_status & RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) == RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK):
        print("rlc_autoload_complete")
        break

  def gb_addr_config(self):
    return self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGB_ADDR_CONFIG, amdgpu_gc_11_0_0.regGB_ADDR_CONFIG_BASE_IDX)

  def init_golden_registers(self):
    val = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regTCP_CNTL, amdgpu_gc_11_0_0.regTCP_CNTL_BASE_IDX)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regTCP_CNTL, amdgpu_gc_11_0_0.regTCP_CNTL_BASE_IDX, val | 0x20000000)

  def init_compute_vmid(self):
    pass

  def constants_init(self):
    # WREG32_FIELD15_PREREG(GC, 0, GRBM_CNTL, READ_TIMEOUT, 0xff);
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGRBM_CNTL, amdgpu_gc_11_0_0.regGRBM_CNTL_BASE_IDX, 0xff)

    # TODO: Read configs here
    for i in range(16):
      self.adev.soc21_grbm_select(0, 0, 0, i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regSH_MEM_CONFIG, amdgpu_gc_11_0_0.regSH_MEM_CONFIG_BASE_IDX, self.DEFAULT_SH_MEM_CONFIG)

      tmp = (self.adev.vmm.private_aperture_start >> 48) | ((self.adev.vmm.shared_aperture_start >> 48) << 16)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regSH_MEM_BASES, amdgpu_gc_11_0_0.regSH_MEM_BASES_BASE_IDX, tmp)

      # We do not enable trap for each kfd vmid...

    self.adev.soc21_grbm_select(0, 0, 0, 0)

    for i in range(1, 16):
      # Initialize all compute VMIDs to have no GDS, GWS, or OA acccess. These should be enabled by FW for target VMIDs (?)
      # TODO: Check if we need to enable them for each VMID.
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGDS_VMID0_BASE, amdgpu_gc_11_0_0.regGDS_VMID0_BASE_BASE_IDX, 0, offset=2*i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGDS_VMID0_SIZE, amdgpu_gc_11_0_0.regGDS_VMID0_SIZE_BASE_IDX, 0, offset=2*i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGDS_GWS_VMID0, amdgpu_gc_11_0_0.regGDS_GWS_VMID0_BASE_IDX, 0, offset=i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGDS_OA_VMID0, amdgpu_gc_11_0_0.regGDS_OA_VMID0_BASE_IDX, 0, offset=i)

  def cp_set_doorbell_range(self):
    g_base = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_RB_DOORBELL_RANGE_LOWER, amdgpu_gc_11_0_0.regCP_RB_DOORBELL_RANGE_LOWER_BASE_IDX)
    cp_base = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_LOWER, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_LOWER_BASE_IDX)

    print("doorbell range", hex(g_base), hex(cp_base))

    # amdgpu_wreg(0x305a, 0x458) # ring0
    # amdgpu_wreg(0x305b, 0x7f8) 
    # amdgpu_wreg(0x305c, 0x0) # kiq
    # amdgpu_wreg(0x305d, 0x450)

  def cp_compute_enable(self):
    v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX)
    print("compute enabled", hex(v)) # 0x3c000000

    # Enable it
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x3c000000)

  def init_kiq_regs(self, ring): # kiq_init_registers
    print("sw", ring.me, ring.pipe, ring.queue)
    self.adev.soc21_grbm_select(ring.me, ring.pipe, ring.queue, 0)
    # v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX)
    # print("compute enabled", hex(v)) # 0x3c000000

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR_BASE_IDX, self.eop_gpu_addr& 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR_HI, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR_HI_BASE_IDX, (self.eop_gpu_addr >> 32) & 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_EOP_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_EOP_CONTROL_BASE_IDX, 12)

    act = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX)
    if act:
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST_BASE_IDX, 1)
      while act:
        act = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX)
      print("q deactivated")

    # print("is act?", )
    # print("is rptr", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_BASE_IDX))

  #   if (RREG32_SOC15(GC, 0, regCP_HQD_ACTIVE) & 1) {
	# 	WREG32_SOC15(GC, 0, regCP_HQD_DEQUEUE_REQUEST, 1);
	# 	for (j = 0; j < adev->usec_timeout; j++) {
	# 		if (!(RREG32_SOC15(GC, 0, regCP_HQD_ACTIVE) & 1))
	# 			break;
	# 		udelay(1);
	# 	}
	# 	WREG32_SOC15(GC, 0, regCP_HQD_DEQUEUE_REQUEST,
	# 	       mqd->cp_hqd_dequeue_request);
	# 	WREG32_SOC15(GC, 0, regCP_HQD_PQ_RPTR,
	# 	       mqd->cp_hqd_pq_rptr);
	# 	WREG32_SOC15(GC, 0, regCP_HQD_PQ_WPTR_LO,
	# 	       mqd->cp_hqd_pq_wptr_lo);
	# 	WREG32_SOC15(GC, 0, regCP_HQD_PQ_WPTR_HI,
	# 	       mqd->cp_hqd_pq_wptr_hi);
	# }

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_VMID, amdgpu_gc_11_0_0.regCP_HQD_VMID_BASE_IDX, ring.mqd.cp_hqd_vmid)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL_BASE_IDX, 0x0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_BASE_IDX, ring.mqd.cp_mqd_base_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_HI, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_HI_BASE_IDX, ring.mqd.cp_mqd_base_addr_hi)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_CONTROL, amdgpu_gc_11_0_0.regCP_MQD_CONTROL_BASE_IDX, ring.mqd.cp_mqd_control)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE_BASE_IDX, ring.mqd.cp_hqd_pq_base_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE_HI, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE_HI_BASE_IDX, ring.mqd.cp_hqd_pq_base_hi)
    
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR_BASE_IDX, ring.mqd.cp_hqd_pq_rptr_report_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR_HI_BASE_IDX, ring.mqd.cp_hqd_pq_rptr_report_addr_hi)

    # assert ring.mqd.cp_hqd_pq_control == 0xd8308011
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_CONTROL_BASE_IDX, ring.mqd.cp_hqd_pq_control)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_PQ_WPTR_POLL_CNTL, amdgpu_gc_11_0_0.regCP_PQ_WPTR_POLL_CNTL_BASE_IDX, 0x80000000)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR_BASE_IDX, ring.mqd.cp_hqd_pq_wptr_poll_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR_HI, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR_HI_BASE_IDX, ring.mqd.cp_hqd_pq_wptr_poll_addr_hi)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL_BASE_IDX, ring.mqd.cp_hqd_pq_doorbell_control)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PERSISTENT_STATE, amdgpu_gc_11_0_0.regCP_HQD_PERSISTENT_STATE_BASE_IDX, ring.mqd.cp_hqd_persistent_state)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX, ring.mqd.cp_hqd_active)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_PQ_STATUS, amdgpu_gc_11_0_0.regCP_PQ_STATUS_BASE_IDX, 0x2)

    # print("act?", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_LOWER, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_LOWER_BASE_IDX))
    # print("act?", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_UPPER, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_UPPER_BASE_IDX))

    self.adev.soc21_grbm_select(0, 0, 0, 0)

  def wdoorbell64(self, index, val):
    # for i in range(0, 0x10000):
    #   self.adev.doorbell[i] = val
    self.adev.doorbell64[index] = val
    self.adev.doorbell[index] = val

  def test_ring(self):
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, 0)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, (512 << 20) - 1)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0, self.adev.vmm.pdb0_base & 0xffffffff)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.adev.vmm.pdb0_base >> 32) & 0xffffffff)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_LO32, 0, 0)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_HI32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_LO32, 0, (512 << 20) - 1)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_LO32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_LO32, 0, self.adev.vmm.pdb0_base & 0xffffffff)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.adev.vmm.pdb0_base >> 32) & 0xffffffff)

    # v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_CNTL, 0)
    # print(hex(v), "CC")
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_CNTL, 0, 0x1fffe03)

    # v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_CNTL, 0)
    # print(hex(v), "CC")
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_CNTL, 0, 0x1fffe03)

    # mec0 is me1
    kiq = AMDRing(self.adev, me=1, pipe=0, queue=0, vmid=0, doorbell_index=22)
    kiq.init_compute_mqd()
    self.init_kiq_regs(kiq)

    from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
    PACKET3_SET_UCONFIG_REG_START = 0x0000c000
    self.adev.wreg(0x1fc00, 0x0) # hdp flush
    self.adev.wreg(0xc040, 0xcafedead)

    kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_UCONFIG_REG, 1))
    kiq.write(0x40) # uconfreg
    kiq.write(0xdeadc0de)

    print("PFS", self.adev.vmm.collect_pfs())
    # print("is mec alive?", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_EXCEPTION_STATUS, amdgpu_gc_11_0_0.regCP_MEC_RS64_EXCEPTION_STATUS_BASE_IDX))

    # print(hex(kiq.next_ptr))
    self.wdoorbell64(kiq.doorbell_index, kiq.next_ptr)

    while True:
      # print(kiq.rptr[0], kiq.wptr[0])

      # print("now", hex(self.adev.rreg(0xc040)))
      # print("PFS", self.adev.vmm.collect_pfs())
      if self.adev.rreg(0xc040) == 0xdeadc0de:
        break

  def config_gfx_rs64(self):
    regCP_MEC_RS64_PRGRM_CNTR_START = 0xc900 # adev->reg_offset[GC_HWIP][0][1] + 0x2900
    regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0xc938 # adev->reg_offset[GC_HWIP][0][1] + 0x2938

    for pipe in range(4):
      self.adev.soc21_grbm_select(1, pipe, 0, 0)
      self.adev.wreg(regCP_MEC_RS64_PRGRM_CNTR_START, 0x3000)
      self.adev.wreg(regCP_MEC_RS64_PRGRM_CNTR_START_HI, 0x70000)
    self.adev.soc21_grbm_select(0, 0, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x400f0000)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x40000000)

  def cp_gfx_enable(self):
    self.adev.wreg(0xa803, 0x1000000)
    for i in range(1000000):
      val = self.adev.rreg(0x21a0)
      if val == 0: return
    raise Exception('gfx_v11_0_cp_gfx_enable timeout')

  def setup(self):
    self.wait_for_rlc_autoload()
    assert self.gb_addr_config() == 0x545 # gfx11 is the same

    self.config_gfx_rs64()
    self.init_golden_registers()
    self.constants_init()

    self.cp_set_doorbell_range()
    self.cp_compute_enable()
    self.cp_gfx_enable()

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, 0)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, (1 << 30) - 1)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0, self.adev.vmm.pdb0_base & 0xffffffff)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.adev.vmm.pdb0_base >> 32) & 0xffffffff)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_LO32, 0, 0)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_START_ADDR_HI32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_LO32, 0, (1 << 30) - 1)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_END_ADDR_LO32, 0, 0)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_LO32, 0, self.adev.vmm.pdb0_base & 0xffffffff)
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.adev.vmm.pdb0_base >> 32) & 0xffffffff)

    # v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_CNTL, 0)
    # print(hex(v), "CC")
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT13_CNTL, 0, 0x1fffe05)

    # v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_CNTL, 0)
    # print(hex(v), "CC")
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_CNTL, 0, 0x1fffe05)
    
    # from extra.amdpci.mes import MES_IP
    # self.mes = MES_IP(self.adev)

    self.test_ring()
