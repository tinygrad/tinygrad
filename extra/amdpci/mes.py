import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class MES_IP:
  AMDGPU_NAVI10_DOORBELL_MES_RING0 = 0x00B
  AMDGPU_NAVI10_DOORBELL_MES_RING1 = 0x00C

  def __init__(self, adev):
    self.adev = adev

    self.sch_ctx_gpu_vaddr = self.adev.vmm.alloc_vram(0x1000, "sch_ctx")
    self.sch_ctx_gpu_paddr = self.adev.vmm.vaddr_to_paddr(self.sch_ctx_gpu_vaddr)
    self.sch_ctx_gpu_mc_addr = self.adev.vmm.paddr_to_mc(self.sch_ctx_gpu_paddr)

    self.query_status_fence_vaddr = self.adev.vmm.alloc_vram(0x1000, "query_status_fence")
    self.query_status_fence_paddr = self.adev.vmm.vaddr_to_paddr(self.query_status_fence_vaddr)
    self.query_status_fence_mc_addr = self.adev.vmm.paddr_to_mc(self.query_status_fence_paddr)
    self.query_status_fence_cpu_view = self.adev.vmm.paddr_to_cpu_mv(self.query_status_fence_paddr, 0x1000).cast('I')

    self.write_fence_vaddr = self.adev.vmm.alloc_vram(0x1000, "w_fence")
    self.write_fence_paddr = self.adev.vmm.vaddr_to_paddr(self.write_fence_vaddr)
    self.write_fence_mc_addr = self.adev.vmm.paddr_to_mc(self.write_fence_paddr)
    self.write_fence_cpu_view = self.adev.vmm.paddr_to_cpu_mv(self.write_fence_paddr, 0x1000).cast('I')
    self.write_fence_cpu_view[0] = 0x0
    self.next_write_fence = 1

  def mes_enable(self):
    # print("MES enable / mes_v11_0_enable()")
    val = (1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE0_RESET__SHIFT) | (1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE1_RESET__SHIFT)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX, val)

    ucodes = [
      self.adev.psp.mes_fw.header.mes_uc_start_addr_lo | (self.adev.psp.mes_fw.header.mes_uc_start_addr_hi << 32),
      self.adev.psp.mes_kiq_fw.header.mes_uc_start_addr_lo | (self.adev.psp.mes_kiq_fw.header.mes_uc_start_addr_hi << 32),
    ]

    for pipe in range(2):
      self.adev.gfx.soc21_grbm_select(3, pipe, 0, 0)

      ucode_addr = ucodes[pipe] >> 2
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_PRGRM_CNTR_START, amdgpu_gc_11_0_0.regCP_MES_PRGRM_CNTR_START_BASE_IDX, ucode_addr & 0xffffffff)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_PRGRM_CNTR_START_HI, amdgpu_gc_11_0_0.regCP_MES_PRGRM_CNTR_START_HI_BASE_IDX, ucode_addr >> 32)
    self.adev.gfx.soc21_grbm_select(0, 0, 0, 0)

    # unhalt mes, and activate it
    val = (1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE0_ACTIVE__SHIFT) | (1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE1_ACTIVE__SHIFT)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX, val)

  def kiq_setting(self, ring):
    tmp = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CP_SCHEDULERS, amdgpu_gc_11_0_0.regRLC_CP_SCHEDULERS_BASE_IDX)
    tmp &= 0xffffff00
    tmp |= (ring.me << 5) | (ring.pipe << 3) | (ring.queue)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CP_SCHEDULERS, amdgpu_gc_11_0_0.regRLC_CP_SCHEDULERS_BASE_IDX, tmp)
    tmp |= 0x80
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regRLC_CP_SCHEDULERS, amdgpu_gc_11_0_0.regRLC_CP_SCHEDULERS_BASE_IDX, tmp)

  def kiq_ring_test_helper(self):
    self.adev.wreg(sc_reg:=0xc040, 0xcafedead) # sratch reg
    self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_WRITE_DATA, 3))
    self.mes_kiq.write(1 << 16)
    self.mes_kiq.write(sc_reg) 
    self.mes_kiq.write(0x0)
    self.mes_kiq.write(0xdeadc0de)

    self.wdoorbell64(self.mes_kiq.doorbell_index, self.mes_kiq.next_ptr)
    while self.adev.rreg(sc_reg) != 0xdeadc0de: pass

  def kiq_map_queue(self, ring_to_enable, is_compute=False):
    if is_compute:
      me = 1
      eng_sel = 0
    else:
      # MES queue
      me = 2
      eng_sel = 5

    self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_MAP_QUEUES, 5))
    self.mes_kiq.write(
      amd_gpu.PACKET3_MAP_QUEUES_QUEUE_SEL(0) |
      amd_gpu.PACKET3_MAP_QUEUES_VMID(0) |
      amd_gpu.PACKET3_MAP_QUEUES_QUEUE(ring_to_enable.queue) |
      amd_gpu.PACKET3_MAP_QUEUES_PIPE(ring_to_enable.pipe) |
      amd_gpu.PACKET3_MAP_QUEUES_ME((me)) |
      amd_gpu.PACKET3_MAP_QUEUES_QUEUE_TYPE(0) | # queue_type: normal compute queue 
      amd_gpu.PACKET3_MAP_QUEUES_ALLOC_FORMAT(0) | # alloc format: all_on_one_pipe
      amd_gpu.PACKET3_MAP_QUEUES_ENGINE_SEL(eng_sel) |
      amd_gpu.PACKET3_MAP_QUEUES_NUM_QUEUES(1)
    )
    self.mes_kiq.write(amd_gpu.PACKET3_MAP_QUEUES_DOORBELL_OFFSET(ring_to_enable.doorbell_index))
    self.mes_kiq.write(ring_to_enable.mqd_gpu_vaddr & 0xffffffff)
    self.mes_kiq.write(ring_to_enable.mqd_gpu_vaddr >> 32)
    self.mes_kiq.write(ring_to_enable.wptr_gpu_vaddr & 0xffffffff)
    self.mes_kiq.write(ring_to_enable.wptr_gpu_vaddr >> 32)

    # Just to test if command is executed
    return self.kiq_ring_test_helper()

  def kiq_set_resources(self, queue_mask):
    self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_RESOURCES, 6))
    self.mes_kiq.write(amd_gpu.PACKET3_SET_RESOURCES_VMID_MASK(0) |
          amd_gpu.PACKET3_SET_RESOURCES_UNMAP_LATENTY(0xa) | # 1s
          amd_gpu.PACKET3_SET_RESOURCES_QUEUE_TYPE(0))
    self.mes_kiq.write(queue_mask & 0xffffffff)
    self.mes_kiq.write(queue_mask >> 32)
    self.mes_kiq.write(0)
    self.mes_kiq.write(0)
    self.mes_kiq.write(0)
    self.mes_kiq.write(0)

    # Just to test if command is executed
    return self.kiq_ring_test_helper()

  def kiq_enable_queue(self, ring_to_enable): self.kiq_map_queue(ring_to_enable)

  def queue_init(self, ring, is_kiq):
    self.mes_v11_0_mqd_init(ring)

    if is_kiq:
      self.init_mes_regs(ring)
    else:
      self.kiq_enable_queue(ring)

  def kiq_hw_init(self):
    self.mes_kiq = AMDRing(self.adev, size=0x100000, me=3, pipe=1, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING1 << 1))

    self.mes_enable()
    self.kiq_setting(self.mes_kiq)
    self.queue_init(self.mes_kiq, is_kiq=True)

    # self.mes_hw_init()

  def mes_hw_init(self):
    self.mes_ring = AMDRing(self.adev, size=0x2000, me=3, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1))
    self.queue_init(self.mes_ring, is_kiq=False)
    self.set_hw_resources()
    self.query_sched_status()

  def mes_v11_0_mqd_init(self, ring):
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

    # disable the queue if it's active
    ring.mqd.cp_hqd_pq_rptr = 0
    ring.mqd.cp_hqd_pq_wptr_lo = 0
    ring.mqd.cp_hqd_pq_wptr_hi = 0

    ring.mqd.cp_mqd_base_addr_lo = ring.mqd_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_mqd_base_addr_hi = (ring.mqd_gpu_vaddr >> 32) & 0xffffffff

    ring.mqd.cp_mqd_control = 0x100

    hqd_gpu_addr = ring.ring_gpu_vaddr >> 8
    ring.mqd.cp_hqd_pq_base_lo = hqd_gpu_addr & 0xffffffff
    ring.mqd.cp_hqd_pq_base_hi = (hqd_gpu_addr >> 32) & 0xffffffff

    ring.mqd.cp_hqd_pq_rptr_report_addr_lo = ring.rptr_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_hqd_pq_rptr_report_addr_hi = (ring.rptr_gpu_vaddr >> 32) & 0xffffffff

    ring.mqd.cp_hqd_pq_wptr_poll_addr_lo = ring.wptr_gpu_vaddr & 0xfffffffc
    ring.mqd.cp_hqd_pq_wptr_poll_addr_hi = (ring.wptr_gpu_vaddr >> 32) & 0xffffffff

    assert ring.ring_size in {0x100000, 0x2000}
    ring.mqd.cp_hqd_pq_control = 0xd8308011 if ring.ring_size == 0x100000 else 0xd830800a

    ring.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (ring.doorbell_index << 2)
    ring.mqd.cp_hqd_vmid = 0
    ring.mqd.cp_hqd_active = 1

    ring.mqd.cp_hqd_persistent_state = 0xbe05501
    ring.mqd.cp_hqd_ib_control = 0x300000 # 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT
    ring.mqd.cp_hqd_iq_timer = 0x0
    ring.mqd.cp_hqd_quantum = 0x0

    self.adev.vmm.paddr_to_cpu_mv(ring.mqd_gpu_paddr, len(ring.mqd_mv))[:] = ring.mqd_mv

  def init_mes_regs(self, ring): # init_mes_regs
    self.adev.gfx.soc21_grbm_select(3, ring.pipe, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_VMID, amdgpu_gc_11_0_0.regCP_HQD_VMID_BASE_IDX, ring.mqd.cp_hqd_vmid)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL_BASE_IDX, 0x0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_BASE_IDX, ring.mqd.cp_mqd_base_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_HI, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_HI_BASE_IDX, ring.mqd.cp_mqd_base_addr_hi)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_CONTROL, amdgpu_gc_11_0_0.regCP_MQD_CONTROL_BASE_IDX, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE_BASE_IDX, ring.mqd.cp_hqd_pq_base_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE_HI, amdgpu_gc_11_0_0.regCP_HQD_PQ_BASE_HI_BASE_IDX, ring.mqd.cp_hqd_pq_base_hi)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR_BASE_IDX, ring.mqd.cp_hqd_pq_rptr_report_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, amdgpu_gc_11_0_0.regCP_HQD_PQ_RPTR_REPORT_ADDR_HI_BASE_IDX, ring.mqd.cp_hqd_pq_rptr_report_addr_hi)

    assert ring.mqd.cp_hqd_pq_control == 0xd8308011
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_CONTROL_BASE_IDX, ring.mqd.cp_hqd_pq_control)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR_BASE_IDX, ring.mqd.cp_hqd_pq_wptr_poll_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR_HI, amdgpu_gc_11_0_0.regCP_HQD_PQ_WPTR_POLL_ADDR_HI_BASE_IDX, ring.mqd.cp_hqd_pq_wptr_poll_addr_hi)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL_BASE_IDX, ring.mqd.cp_hqd_pq_doorbell_control)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PERSISTENT_STATE, amdgpu_gc_11_0_0.regCP_HQD_PERSISTENT_STATE_BASE_IDX, ring.mqd.cp_hqd_persistent_state)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX, ring.mqd.cp_hqd_active)

    self.adev.gfx.soc21_grbm_select(0, 0, 0, 0)

  def wdoorbell64(self, index, val):
    self.adev.doorbell64[index//2] = val # this should be correct

  def submit_pkt_and_poll_completion(self, pkt):
    self.write_fence_cpu_view[0] = 0
    pkt.api_status.api_completion_fence_addr = self.write_fence_mc_addr
    pkt.api_status.api_completion_fence_value = 1

    for i in range(ctypes.sizeof(pkt) // 4): self.mes_ring.write(pkt.max_dwords_in_api[i])

    self.adev.vmm.flush_hdp()
    self.wdoorbell64(self.mes_ring.doorbell_index, self.mes_ring.next_ptr)

    while self.write_fence_cpu_view[0] != 1:
      self.adev.vmm.collect_pfs()

    return 0

  def query_sched_status(self):
    mes_sch_status_pkt = mes_v11_api_def.union_MESAPI__QUERY_MES_STATUS()
    mes_sch_status_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
    mes_sch_status_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_QUERY_SCHEDULER_STATUS
    mes_sch_status_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

    return self.submit_pkt_and_poll_completion(mes_sch_status_pkt)

  def set_hw_resources(self):
    mes_set_hw_res_pkt = mes_v11_api_def.union_MESAPI_SET_HW_RESOURCES()
    mes_set_hw_res_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
    mes_set_hw_res_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_SET_HW_RSRC
    mes_set_hw_res_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

    mes_set_hw_res_pkt.vmid_mask_mmhub = 0xffffff00
    mes_set_hw_res_pkt.vmid_mask_gfxhub = 0xffffff00
    mes_set_hw_res_pkt.gds_size = 0x1000
    mes_set_hw_res_pkt.paging_vmid = 0
    mes_set_hw_res_pkt.g_sch_ctx_gpu_mc_ptr = self.sch_ctx_gpu_mc_addr
    mes_set_hw_res_pkt.query_status_fence_gpu_mc_ptr = self.query_status_fence_mc_addr

    for i,v in enumerate([0xc, 0xc, 0xc, 0xc, 0x0, 0x0, 0x0, 0x0]): mes_set_hw_res_pkt.compute_hqd_mask[i] = v
    for i,v in enumerate([0xfffffffe, 0x0]): mes_set_hw_res_pkt.gfx_hqd_mask[i] = v
    for i,v in enumerate([0xfc, 0xfc]): mes_set_hw_res_pkt.sdma_hqd_mask[i] = v
    for i,v in enumerate([2048, 2050, 2052, 2054, 2056]): mes_set_hw_res_pkt.aggregated_doorbells[i] = v

    for i in range(5):
      mes_set_hw_res_pkt.gc_base[i] = self.adev.ip_base("GC", 0, i)
      mes_set_hw_res_pkt.mmhub_base[i] = self.adev.ip_base("MMHUB", 0, i)
      mes_set_hw_res_pkt.osssys_base[i] = self.adev.ip_base("OSSSYS", 0, i)

    mes_set_hw_res_pkt.disable_reset = 1
    mes_set_hw_res_pkt.disable_mes_log = 1
    mes_set_hw_res_pkt.use_different_vmid_compute = 1
    mes_set_hw_res_pkt.enable_reg_active_poll = 1
    mes_set_hw_res_pkt.enable_level_process_quantum_check = 1
    mes_set_hw_res_pkt.oversubscription_timer = 50

    return self.submit_pkt_and_poll_completion(mes_set_hw_res_pkt)

  def map_legacy_queue(self, ring, qtype):
    mes_add_queue_pkt = mes_v11_api_def.union_MESAPI__ADD_QUEUE()

    mes_add_queue_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
    mes_add_queue_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_ADD_QUEUE
    mes_add_queue_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

    mes_add_queue_pkt.pipe_id = ring.pipe
    mes_add_queue_pkt.queue_id = ring.queue
    mes_add_queue_pkt.doorbell_offset = ring.doorbell_index
    mes_add_queue_pkt.mqd_addr = ring.mqd_gpu_mc_addr
    mes_add_queue_pkt.wptr_addr = ring.wptr_gpu_mc_addr
    mes_add_queue_pkt.queue_type = qtype # mes_v11_api_def.MES_QUEUE_TYPE_COMPUTE
    mes_add_queue_pkt.map_legacy_kq = 1

    return self.submit_pkt_and_poll_completion(mes_add_queue_pkt)
