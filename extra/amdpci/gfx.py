import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class GFX_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c
  AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 0x003

  def __init__(self, adev):
    self.adev = adev
    self.eop_gpu_vaddr = self.adev.vmm.alloc_vram(0x1000, "eop")

    self.clear_state_size = 0x10000
    self.clear_state_gpu_vaddr = self.adev.vmm.alloc_vram(self.clear_state_size, "clear_state")

  def soc21_grbm_select(self, me, pipe, queue, vmid):
    regGRBM_GFX_CNTL = 0xa900 # (adev->reg_offset[GC_HWIP][0][1] + 0x0900)
    GRBM_GFX_CNTL__PIPEID__SHIFT=0x0
    GRBM_GFX_CNTL__MEID__SHIFT=0x2
    GRBM_GFX_CNTL__VMID__SHIFT=0x4
    GRBM_GFX_CNTL__QUEUEID__SHIFT=0x8

    grbm_gfx_cntl = (me << GRBM_GFX_CNTL__MEID__SHIFT) | (pipe << GRBM_GFX_CNTL__PIPEID__SHIFT) | (vmid << GRBM_GFX_CNTL__VMID__SHIFT) | (queue << GRBM_GFX_CNTL__QUEUEID__SHIFT)
    self.adev.wreg(regGRBM_GFX_CNTL, grbm_gfx_cntl)

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
      self.soc21_grbm_select(0, 0, 0, i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regSH_MEM_CONFIG, amdgpu_gc_11_0_0.regSH_MEM_CONFIG_BASE_IDX, self.DEFAULT_SH_MEM_CONFIG)

      tmp = (self.adev.vmm.private_aperture_start >> 48) | ((self.adev.vmm.shared_aperture_start >> 48) << 16)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regSH_MEM_BASES, amdgpu_gc_11_0_0.regSH_MEM_BASES_BASE_IDX, tmp)

      # We do not enable trap for each kfd vmid...

    self.soc21_grbm_select(0, 0, 0, 0)

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
    self.soc21_grbm_select(ring.me, ring.pipe, ring.queue, 0)

    v = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX)
    print("compute enabled 2", hex(v)) # 0x3c000000

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR_BASE_IDX, self.eop_gpu_addr & 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR_HI, amdgpu_gc_11_0_0.regCP_HQD_EOP_BASE_ADDR_HI_BASE_IDX, (self.eop_gpu_addr >> 32) & 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_EOP_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_EOP_CONTROL_BASE_IDX, 12)

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX, 1)

    act = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX)
    if act and False:
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST_BASE_IDX, 1)
      while act:
        # print(act)
        act = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX)
      print("q deactivated")

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

    self.soc21_grbm_select(0, 0, 0, 0)

  def wdoorbell64(self, index, val):
    # for i in range(0, 0x10000):
    #   self.adev.doorbell[i] = val
    self.adev.doorbell64[index] = val
    self.adev.doorbell[index] = val

  def init_csb(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_HI, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_HI_BASE_IDX, self.clear_state_gpu_vaddr >> 32)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_LO, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_LO_BASE_IDX, self.clear_state_gpu_vaddr  & 0xfffffffc)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CSIB_LENGTH, amdgpu_gc_11_0_0.regRLC_CSIB_LENGTH_BASE_IDX, self.clear_state_size)

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
    kiq = AMDRing(self.adev, me=1, pipe=0, queue=0, vmid=0, doorbell_index=0)
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
      self.soc21_grbm_select(1, pipe, 0, 0)
      self.adev.wreg(regCP_MEC_RS64_PRGRM_CNTR_START, 0x3000)
      self.adev.wreg(regCP_MEC_RS64_PRGRM_CNTR_START_HI, 0x70000)
    self.soc21_grbm_select(0, 0, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x400f0000)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x40000000)

  def cp_gfx_enable(self):
    self.adev.wreg(0xa803, 0x1000000)
    for i in range(1000000):
      val = self.adev.rreg(0x21a0)
      if val == 0: return
    raise Exception('gfx_v11_0_cp_gfx_enable timeout')

  def kcq_init_queue(self, ring):
    self.soc21_grbm_select(ring.me, ring.pipe, ring.queue, 0)
    self.compute_init_mqd(ring)
    self.soc21_grbm_select(0, 0, 0, 0)

  def kcq_resume(self):
    self.kcq_ring = AMDRing(self.adev, size=0x2000, me=1, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MEC_RING0 << 1))
    self.kcq_init_queue(self.kcq_ring)
    # self.amdgpu_gfx_enable_kcq()

  def cp_resume(self):
    self.cp_set_doorbell_range()
    self.cp_compute_enable()

    self.adev.mes.kiq_hw_init()
    self.kcq_resume()

  def init(self):
    print("GFX: init")

    self.wait_for_rlc_autoload()
    assert self.gb_addr_config() == 0x545 # gfx11 is the same

    self.config_gfx_rs64()
    self.adev.vmm.init_gfxhub()
    self.init_golden_registers()
    self.constants_init()

    self.init_csb()
    self.cp_resume()

  def compute_init_mqd(self, ring):
    ring.mqd.header = 0xC0310800
    ring.mqd.compute_pipelinestat_enable = 0x00000001
    ring.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
    ring.mqd.compute_misc_reserved = 0x00000007

    eop_base_addr = ring.eop_gpu_vaddr >> 8
    ring.mqd.cp_hqd_eop_base_addr_lo = eop_base_addr & 0xffffffff
    ring.mqd.cp_hqd_eop_base_addr_hi = (eop_base_addr >> 32) & 0xffffffff
    ring.mqd.cp_hqd_eop_control = 0x8

    ring.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (ring.doorbell_index << 2)

    # disable the queue if it's active
    ring.mqd.cp_hqd_dequeue_request = 0
    ring.mqd.cp_hqd_pq_rptr = 0
    ring.mqd.cp_hqd_pq_wptr_lo = 0
    ring.mqd.cp_hqd_pq_wptr_hi = 0

    ring.mqd.cp_mqd_base_addr_lo = ring.mqd_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_mqd_base_addr_hi = (ring.mqd_gpu_vaddr >> 32) & 0xffffffff

    ring.mqd.cp_mqd_control = 0x100 ## ??

    hqd_gpu_addr = ring.ring_gpu_vaddr >> 8
    ring.mqd.cp_hqd_pq_base_lo = hqd_gpu_addr & 0xffffffff
    ring.mqd.cp_hqd_pq_base_hi = (hqd_gpu_addr >> 32) & 0xffffffff
    assert ring.ring_size in {0x2000}
    ring.mqd.cp_hqd_pq_control = 0xd030890a

    ring.mqd.cp_hqd_pq_rptr_report_addr_lo = ring.rptr_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_hqd_pq_rptr_report_addr_hi = (ring.rptr_gpu_vaddr >> 32) & 0xffff

    ring.mqd.cp_hqd_pq_wptr_poll_addr_lo = ring.wptr_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_hqd_pq_wptr_poll_addr_hi = (ring.wptr_gpu_vaddr >> 32) & 0xffff

    ring.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (ring.doorbell_index << 2)
    ring.mqd.cp_hqd_vmid = 0
    ring.mqd.cp_hqd_active = 1

    ring.mqd.cp_hqd_persistent_state = 0xbe05501
    ring.mqd.cp_hqd_ib_control = 0x300000 # 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT
    ring.mqd.cp_hqd_iq_timer = 0x0
    ring.mqd.cp_hqd_quantum = 0x0

    self.adev.vmm.paddr_to_cpu_mv(ring.mqd_gpu_paddr, len(ring.mqd_mv))[:] = ring.mqd_mv
