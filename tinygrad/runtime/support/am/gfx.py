import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
from tinygrad.helpers import to_mv, mv_address, getenv

from extra.amdpci.amdring import AMDRing

class GFX_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c
  AMDGPU_NAVI10_DOORBELL_MEC_RING0 = 0x003

  def __init__(self, adev):
    self.adev = adev

    self.clear_state_pm = self.adev.mm.palloc(0x10000)
    self.eop_pm = self.adev.mm.palloc(0x1000)

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
    for i in range(1, 16):
      self.soc21_grbm_select(0, 0, 0, i)
      self.adev.regSH_MEM_CONFIG.write(self.DEFAULT_SH_MEM_CONFIG)
      self.adev.regSH_MEM_BASES.write((self.adev.gmc.private_aperture_base >> 48) | ((self.adev.gmc.shared_aperture_base >> 48) << 16))
    self.soc21_grbm_select(0, 0, 0, 0)

    for i in range(1, 16):
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
    self.adev.wreg(0x507a40, 0x30000007)
    self.adev.wreg(0x507a43, 0x3000000d)

  def init_csb(self):
    self.adev.regRLC_CSIB_ADDR_HI.write(self.clear_state_pm.mc_addr() >> 32)
    self.adev.regRLC_CSIB_ADDR_LO.write(self.clear_state_pm.mc_addr() & 0xfffffffc)
    self.adev.regRLC_CSIB_LENGTH.write(self.clear_state_pm.size)

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
    self.kcq_ring = AMDRing(self.adev, size=0x100000, me=1, pipe=0, queue=0, vmid=1, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MEC_RING0 << 1))
    self.kcq_init_queue(self.kcq_ring)
    self.adev.vmm.flush_hdp()

    if not getenv("NO_KIQ_HW"):
      self.adev.mes.kiq_set_resources(0xffffffffffffffff) # full mask

      # Directly map kcq with kiq, no MES map_legacy_queue.
      self.adev.mes.kiq_map_queue(self.kcq_ring, is_compute=True)
      # self.adev.mes.map_legacy_queue(self.kcq_ring)
    else:
      self.adev.mes.kiq_setting(self.kcq_ring)

    # test kcq
    self.adev.wreg(0xc040, 0xcafedead)

    self.kcq_ring.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_UCONFIG_REG, 1))
    self.kcq_ring.write(0x40) # uconfreg
    self.kcq_ring.write(0xdeadc0de)

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
    self.adev.gmc.init_gfxhub()
    self.init_golden_registers()
    self.constants_init()
    self.nbio_v4_3_gc_doorbell_init()

    self.init_csb()
    self.cp_resume()

  def compute_init_mqd(self, ring):
    ring.mqd.header = 0xC0310800
    ring.mqd.compute_pipelinestat_enable = 0x00000001
    ring.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se4 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se5 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se6 = 0xffffffff
    ring.mqd.compute_static_thread_mgmt_se7 = 0xffffffff
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

    ring.mqd.cp_mqd_control = 0x100

    hqd_gpu_addr = ring.ring_gpu_vaddr >> 8
    ring.mqd.cp_hqd_pq_base_lo = hqd_gpu_addr & 0xffffffff
    ring.mqd.cp_hqd_pq_base_hi = (hqd_gpu_addr >> 32) & 0xffffffff
    assert ring.ring_size in {0x100000}
    # ring.mqd.cp_hqd_pq_control = 0xd0308911
    ring.mqd.cp_hqd_pq_control = 5 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT | \
			(1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__UNORD_DISPATCH__SHIFT) | \
      (0 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__TUNNEL_DISPATCH__SHIFT) | \
      (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__PRIV_STATE__SHIFT) | \
      (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__KMD_QUEUE__SHIFT) | 0x11 # size

    ring.mqd.cp_hqd_pq_rptr_report_addr_lo = ring.rptr_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_hqd_pq_rptr_report_addr_hi = (ring.rptr_gpu_vaddr >> 32) & 0xffff

    ring.mqd.cp_hqd_pq_wptr_poll_addr_lo = ring.wptr_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_hqd_pq_wptr_poll_addr_hi = (ring.wptr_gpu_vaddr >> 32) & 0xffff

    ring.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (ring.doorbell_index << 2)
    ring.mqd.cp_hqd_vmid = ring.vmid

    ring.mqd.cp_hqd_pq_rptr = 0 # ??

    ring.mqd.cp_hqd_persistent_state = 0xbe05501
    ring.mqd.cp_hqd_ib_control = 0x300000 # 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT
    ring.mqd.cp_hqd_pipe_priority = 0x2
    ring.mqd.cp_hqd_queue_priority = 0xf
    ring.mqd.cp_hqd_active = 1

    self.adev.vmm.paddr_to_cpu_mv(ring.mqd_gpu_paddr, len(ring.mqd_mv))[:] = ring.mqd_mv
    self.adev.vmm.flush_hdp()
