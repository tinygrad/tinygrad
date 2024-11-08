import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
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
    self.clear_state_gpu_paddr = self.adev.vmm.vaddr_to_paddr(self.clear_state_gpu_vaddr)
    self.clear_state_gpu_mc_addr = self.adev.vmm.paddr_to_mc(self.clear_state_gpu_paddr)

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
    # g_base = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_RB_DOORBELL_RANGE_LOWER, amdgpu_gc_11_0_0.regCP_RB_DOORBELL_RANGE_LOWER_BASE_IDX)
    # cp_base = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_LOWER, amdgpu_gc_11_0_0.regCP_MEC_DOORBELL_RANGE_LOWER_BASE_IDX)
    # print("doorbell range", hex(g_base), hex(cp_base))

    self.adev.wreg(0x305a, 0x458) # ring0
    self.adev.wreg(0x305b, 0x7f8) 
    self.adev.wreg(0x305c, 0x0) # kiq
    self.adev.wreg(0x305d, 0x450)

  def cp_compute_enable(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x3c000000)
    time.sleep(0.5)

  def nbio_v4_3_gc_doorbell_init(self):
    self.adev.wreg(0x507a40, 0x30000007)
    self.adev.wreg(0x507a43, 0x3000000d)

  def wdoorbell64(self, index, val): self.adev.doorbell64[index//2] = val

  def init_csb(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_HI, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_HI_BASE_IDX, self.clear_state_gpu_mc_addr >> 32)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_LO, amdgpu_gc_11_0_0.regRLC_CSIB_ADDR_LO_BASE_IDX, self.clear_state_gpu_mc_addr  & 0xfffffffc)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CSIB_LENGTH, amdgpu_gc_11_0_0.regRLC_CSIB_LENGTH_BASE_IDX, self.clear_state_size)

  def config_gfx_rs64(self):
    print(hex(self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_STAT, amdgpu_gc_11_0_0.regCP_STAT_BASE_IDX)))

    for pipe in range(2):
      self.soc21_grbm_select(0, pipe, 0, 0)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_PFP_PRGRM_CNTR_START, amdgpu_gc_11_0_0.regCP_PFP_PRGRM_CNTR_START_BASE_IDX, 0xc00)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_PFP_PRGRM_CNTR_START_HI, amdgpu_gc_11_0_0.regCP_PFP_PRGRM_CNTR_START_HI_BASE_IDX, 0x1c000)
    self.soc21_grbm_select(0, 0, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_CNTL, amdgpu_gc_11_0_0.regCP_ME_CNTL_BASE_IDX, 0x153c0000)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_CNTL, amdgpu_gc_11_0_0.regCP_ME_CNTL_BASE_IDX, 0x15300000)

    for pipe in range(2):
      self.soc21_grbm_select(0, pipe, 0, 0)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_PRGRM_CNTR_START, amdgpu_gc_11_0_0.regCP_ME_PRGRM_CNTR_START_BASE_IDX, 0xc00)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_PRGRM_CNTR_START_HI, amdgpu_gc_11_0_0.regCP_ME_PRGRM_CNTR_START_HI_BASE_IDX, 0x1c000)
    self.soc21_grbm_select(0, 0, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_CNTL, amdgpu_gc_11_0_0.regCP_ME_CNTL_BASE_IDX, 0x15300000)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_CNTL, amdgpu_gc_11_0_0.regCP_ME_CNTL_BASE_IDX, 0x15000000)

    for pipe in range(4):
      self.soc21_grbm_select(1, pipe, 0, 0)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_PRGRM_CNTR_START, amdgpu_gc_11_0_0.regCP_MEC_RS64_PRGRM_CNTR_START_BASE_IDX, 0xc00)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_PRGRM_CNTR_START_HI, amdgpu_gc_11_0_0.regCP_MEC_RS64_PRGRM_CNTR_START_HI_BASE_IDX, 0x1c000)
    self.soc21_grbm_select(0, 0, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x400f0000)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL, amdgpu_gc_11_0_0.regCP_MEC_RS64_CNTL_BASE_IDX, 0x40000000)

  def cp_gfx_enable(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_ME_CNTL, amdgpu_gc_11_0_0.regCP_ME_CNTL_BASE_IDX, 0x1000000)
    for i in range(100):
      val = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_STAT, amdgpu_gc_11_0_0.regCP_STAT_BASE_IDX)
      if val == 0: return
      time.sleep(0.00001)
    raise Exception('gfx_v11_0_cp_gfx_enable timeout')

  def kcq_init_queue(self, ring):
    self.soc21_grbm_select(ring.me, ring.pipe, ring.queue, 0)
    self.compute_init_mqd(ring)
    self.soc21_grbm_select(0, 0, 0, 0)

  def kcq_resume(self):
    self.kcq_ring = AMDRing(self.adev, size=0x100000, me=1, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MEC_RING0 << 1))
    self.kcq_init_queue(self.kcq_ring)

    self.adev.vmm.flush_hdp()
    self.adev.mes.kiq_set_resources(0xffffffffffffffff) # full mask

    # Directly map kcq with kiq, no MES map_legacy_queue.
    self.adev.mes.kiq_map_queue(self.kcq_ring, is_compute=True)

    # test kcq
    self.adev.wreg(0xc040, 0xcafedead)

    self.kcq_ring.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_UCONFIG_REG, 1))
    self.kcq_ring.write(0x40) # uconfreg
    self.kcq_ring.write(0xdeadc0de)

    # self.kcq_ring.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_UCONFIG_REG, 1))
    # self.kcq_ring.write(0x40) # uconfreg
    # self.kcq_ring.write(0xdeadc0de)

    self.wdoorbell64(self.kcq_ring.doorbell_index, self.kcq_ring.next_ptr)

    while True:
      self.adev.vmm.collect_pfs()
      if self.adev.rreg(0xc040) == 0xdeadc0de:
        break

    print("GFX: kcq test done")

  def cp_resume(self):
    self.cp_set_doorbell_range()
    self.cp_gfx_enable()
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
    self.nbio_v4_3_gc_doorbell_init()

    # self.init_csb()
    self.cp_resume()

  def compute_init_mqd(self, ring):
    ring.mqd.header = 0xC0310800
    ring.mqd.compute_pipelinestat_enable = 0x00000001
    ring.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
    ring.mqd.compute_misc_reserved = 0x00000007

    eop_base_addr = ring.eop_gpu_mc_addr >> 8
    ring.mqd.cp_hqd_eop_base_addr_lo = eop_base_addr & 0xffffffff
    ring.mqd.cp_hqd_eop_base_addr_hi = (eop_base_addr >> 32) & 0xffffffff
    ring.mqd.cp_hqd_eop_control = 0x8

    ring.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (ring.doorbell_index << 2)

    # disable the queue if it's active
    ring.mqd.cp_hqd_dequeue_request = 0
    ring.mqd.cp_hqd_pq_rptr = 0
    ring.mqd.cp_hqd_pq_wptr_lo = 0
    ring.mqd.cp_hqd_pq_wptr_hi = 0

    ring.mqd.cp_mqd_base_addr_lo = ring.mqd_gpu_mc_addr & 0xfffffffc
    ring.mqd.cp_mqd_base_addr_hi = (ring.mqd_gpu_mc_addr >> 32) & 0xffffffff

    ring.mqd.cp_mqd_control = 0x100

    hqd_gpu_addr = ring.ring_gpu_mc_addr >> 8
    ring.mqd.cp_hqd_pq_base_lo = hqd_gpu_addr & 0xffffffff
    ring.mqd.cp_hqd_pq_base_hi = (hqd_gpu_addr >> 32) & 0xffffffff
    assert ring.ring_size in {0x100000}
    ring.mqd.cp_hqd_pq_control = 0xd0308911

    ring.mqd.cp_hqd_pq_rptr_report_addr_lo = ring.rptr_gpu_mc_addr & 0xfffffffc
    ring.mqd.cp_hqd_pq_rptr_report_addr_hi = (ring.rptr_gpu_mc_addr >> 32) & 0xffff

    ring.mqd.cp_hqd_pq_wptr_poll_addr_lo = ring.wptr_gpu_mc_addr & 0xfffffffc
    ring.mqd.cp_hqd_pq_wptr_poll_addr_hi = (ring.wptr_gpu_mc_addr >> 32) & 0xffff

    ring.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (ring.doorbell_index << 2)
    ring.mqd.cp_hqd_vmid = 0

    ring.mqd.cp_hqd_pq_rptr = 0 # ??

    ring.mqd.cp_hqd_persistent_state = 0xbe05501
    ring.mqd.cp_hqd_ib_control = 0x300000 # 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT
    ring.mqd.cp_hqd_pipe_priority = 0x0
    ring.mqd.cp_hqd_queue_priority = 0x0
    ring.mqd.cp_hqd_active = 0

    self.adev.vmm.paddr_to_cpu_mv(ring.mqd_gpu_paddr, len(ring.mqd_mv))[:] = ring.mqd_mv
    self.adev.vmm.flush_hdp()
