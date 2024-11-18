import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
from tinygrad.helpers import to_mv, mv_address, getenv
from tinygrad.runtime.support.am.amring import AMRing

class GFX_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c
  AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 0x003

  def __init__(self, adev):
    self.adev = adev

    # self.clear_state_pm = self.adev.mm.palloc(0x10000)
    # self.eop_pm = self.adev.mm.palloc(0x1000)

  def init(self):
    print("GFX: init")

    self.wait_for_rlc_autoload()
    assert self.gb_addr_config() == 0x545 # gfx11 is the same

    self.config_gfx_rs64()
    self.adev.gmc.init_gfxhub()
    self.init_golden_registers()
    self.constants_init()
    self.nbio_v4_3_gc_doorbell_init()

    # self.init_csb()
    self.cp_resume()

  def soc21_grbm_select(self, me, pipe, queue, vmid):
    self.adev.regGRBM_GFX_CNTL.write((me << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__MEID__SHIFT) |
      (pipe << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__PIPEID__SHIFT) | (vmid << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__VMID__SHIFT) |
      (queue << amdgpu_gc_11_0_0.GRBM_GFX_CNTL__QUEUEID__SHIFT))

  def wait_for_rlc_autoload(self):
    while True:
      bootload_ready = (self.adev.regRLC_RLCS_BOOTLOAD_STATUS.read() & amdgpu_gc_11_0_0.RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) != 0
      if self.adev.regCP_STAT.read() == 0 and bootload_ready: break

  def gb_addr_config(self): return self.adev.regGB_ADDR_CONFIG.read()
  def init_golden_registers(self): self.adev.regTCP_CNTL.write(self.adev.regTCP_CNTL.read() | 0x20000000)

  def constants_init(self):
    self.adev.regGRBM_CNTL.write(0xff)

    # TODO: Read configs here
    for i in range(8, 16):
      self.soc21_grbm_select(0, 0, 0, i)
      self.adev.regSH_MEM_CONFIG.write(self.DEFAULT_SH_MEM_CONFIG)
      self.adev.regSH_MEM_BASES.write((self.adev.gmc.private_aperture_base >> 48) | ((self.adev.gmc.shared_aperture_base >> 48) << 16))
    self.soc21_grbm_select(0, 0, 0, 0)

    for i in range(8, 16):
      # Initialize all compute VMIDs to have no GDS, GWS, or OA acccess. These should be enabled by FW for target VMIDs (?)
      getattr(self.adev, f"regGDS_VMID{i}_BASE").write(0)
      getattr(self.adev, f"regGDS_VMID{i}_SIZE").write(0)
      getattr(self.adev, f"regGDS_GWS_VMID{i}").write(0)
      getattr(self.adev, f"regGDS_OA_VMID{i}").write(0)

  def cp_set_doorbell_range(self):
    self.adev.regCP_RB_DOORBELL_RANGE_LOWER.write(0x458)
    self.adev.regCP_RB_DOORBELL_RANGE_UPPER.write(0x7f8)
    self.adev.regCP_MEC_DOORBELL_RANGE_LOWER.write(0x0)
    self.adev.regCP_MEC_DOORBELL_RANGE_UPPER.write(0x0)

  def cp_compute_enable(self):
    self.adev.regCP_MEC_RS64_CNTL.write(0x3c000000)
    time.sleep(0.5)

  def nbio_v4_3_gc_doorbell_init(self):
    self.adev.regS2A_DOORBELL_ENTRY_0_CTRL.write(0x30000007)
    self.adev.regS2A_DOORBELL_ENTRY_3_CTRL.write(0x3000000d)

  # def init_csb(self):
  #   self.adev.regRLC_CSIB_ADDR_HI.write(self.clear_state_pm.mc_addr() >> 32)
  #   self.adev.regRLC_CSIB_ADDR_LO.write(self.clear_state_pm.mc_addr() & 0xfffffffc)
  #   self.adev.regRLC_CSIB_LENGTH.write(self.clear_state_pm.size)

  def config_gfx_rs64(self):
    for pipe in range(2):
      self.soc21_grbm_select(0, pipe, 0, 0)
      self.adev.regCP_PFP_PRGRM_CNTR_START.write(0xc00)
      self.adev.regCP_PFP_PRGRM_CNTR_START_HI.write(0x1c000)

    # TODO: write up!
    self.soc21_grbm_select(0, 0, 0, 0)
    self.adev.regCP_ME_CNTL.write(0x153c0000)
    self.adev.regCP_ME_CNTL.write(0x15300000)

    for pipe in range(2):
      self.soc21_grbm_select(0, pipe, 0, 0)
      self.adev.regCP_ME_PRGRM_CNTR_START.write(0xc00)
      self.adev.regCP_ME_PRGRM_CNTR_START_HI.write(0x1c000)

    self.soc21_grbm_select(0, 0, 0, 0)
    self.adev.regCP_ME_CNTL.write(0x15300000)
    self.adev.regCP_ME_CNTL.write(0x15000000)

    for pipe in range(4):
      self.soc21_grbm_select(1, pipe, 0, 0)
      self.adev.regCP_MEC_RS64_PRGRM_CNTR_START.write(0xc00)
      self.adev.regCP_MEC_RS64_PRGRM_CNTR_START_HI.write(0x1c000)

    self.soc21_grbm_select(0, 0, 0, 0)
    self.adev.regCP_MEC_RS64_CNTL.write(0x400f0000)
    self.adev.regCP_MEC_RS64_CNTL.write(0x40000000)

  def cp_gfx_enable(self):
    self.adev.regCP_ME_CNTL.write(0x1000000)
    self.adev.wait_reg(self.adev.regCP_STAT, value=0x0)

  def hqd_load(self, ring):
    self.soc21_grbm_select(1, ring.pipe, ring.queue, 0)

    mqd_mv = ring.mqd_mv.cast('I')
    for i, reg in enumerate(range(self.adev.regCP_MQD_BASE_ADDR.regoff, self.adev.regCP_HQD_PQ_WPTR_HI.regoff + 1)):
      self.adev.wreg(reg, mqd_mv[0x80 + i])

    self.adev.regCP_HQD_PQ_BASE.write(ring.mqd.cp_hqd_pq_base_lo)
    self.adev.regCP_HQD_PQ_BASE_HI.write(ring.mqd.cp_hqd_pq_base_hi)

    self.adev.regCP_HQD_PQ_RPTR_REPORT_ADDR.write(ring.mqd.cp_hqd_pq_rptr_report_addr_lo)
    self.adev.regCP_HQD_PQ_RPTR_REPORT_ADDR_HI.write(ring.mqd.cp_hqd_pq_rptr_report_addr_hi)

    self.adev.regCP_HQD_PQ_WPTR_POLL_ADDR.write(ring.mqd.cp_hqd_pq_wptr_poll_addr_lo)
    self.adev.regCP_HQD_PQ_WPTR_POLL_ADDR_HI.write(ring.mqd.cp_hqd_pq_wptr_poll_addr_hi)

    self.adev.regCP_PQ_WPTR_POLL_CNTL1.write(1 << (ring.pipe * 4 + ring.queue)) # queue mask

    self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.write(ring.mqd.cp_hqd_pq_doorbell_control)
    self.adev.regCP_HQD_ACTIVE.write(0x1)

    self.soc21_grbm_select(0, 0, 0, 0)

  def kcq_init(self):
    self.kcq_ring = AMRing(self.adev, size=0x100000, me=1, pipe=0, queue=0, vmid=0, doorbell_index=((self.AMDGPU_NAVI10_DOORBELL_MEC_RING0) << 1))
    self.hqd_load(self.kcq_ring)

    # self.adev.mes.kiq_set_resources(0xffffffffffffffff) # full mask

    # Directly map kcq with kiq, no MES map_legacy_queue.
    # self.adev.mes.map_legacy_queue(self.kcq_ring)
    # self.adev.mes.kiq_map_queue(self.kcq_ring, is_compute=True)

    # test kcq
    # self.adev.wreg(0xc040, 0xcafedead)

    # self.kcq_ring.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_UCONFIG_REG, 1))
    # self.kcq_ring.write(0x40) # uconfreg
    # self.kcq_ring.write(0xdeadc0de)

    # self.adev.wdoorbell64(self.kcq_ring.doorbell_index, self.kcq_ring.next_ptr)

    # while True:
    #   # self.adev.vmm.collect_pfs()
    #   if self.adev.rreg(0xc040) == 0xdeadc0de:
    #     break

    # print("GFX: kcq test done")
    # for i in range(3):
    #   print("Cool down", i)
    time.sleep(1)

  def cp_resume(self):
    self.cp_set_doorbell_range()
    self.cp_compute_enable()
    # self.cp_gfx_enable()
    # self.adev.mes.kiq_hw_init()

    self.kcq_init()
