import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class MES_IP:
  AMDGPU_NAVI10_DOORBELL_MES_RING0 = 0x00B
  AMDGPU_NAVI10_DOORBELL_MES_RING1 = 0x00C

  def __init__(self, adev):
    self.adev = adev
    
    self.sch_ctx_gpu_addr = self.adev.vmm.alloc_vram(0x1000, "sch_ctx")
    self.query_status_fence_gpu_mc_ptr = self.adev.vmm.alloc_vram(0x1000, "query_status_fence")
    self.write_fence_addr = self.adev.vmm.alloc_vram(0x1000, "w_fence")
    
    self.kiq_hw_init()
    
    # self.mes_queue = AMDRing(self.adev, me=3, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1))
    # self.mes_queue.init_mes_mqd()
    # self.setup()

  def mes_enable(self):
    print("MES enable / mes_v11_0_enable()")
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

  def mes_v11_0_queue_init(self, ring, is_kiq):
    self.mes_v11_0_mqd_init(ring)

    if is_kiq:
      self.init_mes_regs(ring)
    else:
      self.mes_v11_0_kiq_enable_queue(ring)

  def kiq_hw_init(self):
    self.mes_kiq = AMDRing(self.adev, size=0x100000, me=3, pipe=1, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING1 << 1))

    self.mes_enable()
    self.kiq_setting(self.mes_kiq)
    self.mes_v11_0_queue_init(self.mes_kiq, is_kiq=True)

    # test kiq queue
    from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
    PACKET3_SET_UCONFIG_REG_START = 0x0000c000
    self.adev.wreg(0x1fc00, 0x0) # hdp flush
    self.adev.wreg(0xc040, 0xcafedead)

    self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_WRITE_DATA, 3))
    self.mes_kiq.write(1 << 16)
    self.mes_kiq.write(0xc040) 
    self.mes_kiq.write(0x0)
    self.mes_kiq.write(0xdeadc0de)

    print("PFS", self.adev.vmm.collect_pfs())

    print("reg before", hex(self.adev.rreg(0xc040)))
    self.wdoorbell64(self.mes_kiq.doorbell_index, self.mes_kiq.next_ptr)

    while True:
      if self.adev.rreg(0xc040) == 0xdeadc0de:
        break

    print("WOO, reg changed", hex(self.adev.rreg(0xc040)))
    
    # self.mes_hw_init()

  def mes_hw_init(self):
    self.mes_ring = AMDRing(self.adev, size=0x2000, me=3, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1))
    self.mes_v11_0_queue_init(self.mes_ring, is_kiq=False)

  def mes_v11_0_kiq_enable_queue():
    pass

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
    print("ringing", index, val)
    self.adev.doorbell64[index//2] = val # this should be correct

  def submit_pkt_and_poll_completion(self, pkt):
    ndw = ctypes.sizeof(pkt) // 4

    pkt.api_status.api_completion_fence_addr = self.write_fence_addr
    pkt.api_status.api_completion_fence_value = 0xdead

    self.fence_view = self.adev.vmm.vram_to_cpu_mv(self.write_fence_addr, 0x1000).cast('I')
    self.fence_view[0] = 0
    print(self.fence_view[0])
    
    for i in range(ndw):
      # print(hex(pkt.max_dwords_in_api[i]))
      self.mes_queue.write(pkt.max_dwords_in_api[i])

    self.wdoorbell64(self.mes_queue.doorbell_index, self.mes_queue.next_ptr)

    # print(self.fence_view[0])
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX, 1)
    self.adev.gfx.soc21_grbm_select(3, 0, 0, 0)
    print("act?", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX))
    self.adev.gfx.soc21_grbm_select(0, 0, 0, 0)
    while self.fence_view[0] == 0:
      # print(self.mes_queue.rptr[0], self.mes_queue.wptr[0])
      # print("PFS", self.adev.vmm.collect_pfs())
      pass

    print("fence", self.fence_view[0])
  
  def set_hw_resources(self):
    mes_set_hw_res_pkt = mes_v11_api_def.union_MESAPI_SET_HW_RESOURCES()
    mes_set_hw_res_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
    mes_set_hw_res_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_SET_HW_RSRC
    mes_set_hw_res_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

    mes_set_hw_res_pkt.vmid_mask_mmhub = 0xffffff00
    mes_set_hw_res_pkt.vmid_mask_gfxhub = 0xffffff00
    mes_set_hw_res_pkt.gds_size = 0x1000
    mes_set_hw_res_pkt.paging_vmid = 0
    mes_set_hw_res_pkt.g_sch_ctx_gpu_mc_ptr = self.sch_ctx_gpu_addr
    mes_set_hw_res_pkt.query_status_fence_gpu_mc_ptr = self.query_status_fence_gpu_mc_ptr

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

    self.submit_pkt_and_poll_completion(mes_set_hw_res_pkt)
  
  def mes_init(self):
    mes_v11_0_queue_init

  def setup(self):
    self.init_mes_regs(self.mes_queue)

    self.adev.gfx.soc21_grbm_select(3, 0, 0, 0)
    self.sched_version = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_GP3_LO, amdgpu_gc_11_0_0.regCP_MES_GP3_LO_BASE_IDX)
    print("MES API v", (self.sched_version & 0x00fff000) >> 12)

    # # reset mes
    # mes_cntr = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX)
    # if (mes_cntr & 0x40000000): print("MES HALTED")

    # mes_cntr_reset = mes_cntr | (0x00010000) # pipe0 reset
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX, mes_cntr_reset)

    # start_mes = (0x04000000) # pipe0 active
    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX, start_mes)

    # mes_cntr = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX)
    # if (mes_cntr & 0x40000000): print("MES HALTED")
    # print("MES STATUS", hex(mes_cntr))    

    # mes_ip = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_PRGRM_CNTR_START, amdgpu_gc_11_0_0.regCP_MES_PRGRM_CNTR_START_BASE_IDX)
    # print("mes_ip", hex(mes_ip))

    # mes_ip = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_MCAUSE_LO, amdgpu_gc_11_0_0.regCP_MES_MCAUSE_LO_BASE_IDX)
    # print("regCP_MES_MCAUSE", hex(mes_ip))

    # mes_ip = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_MBADADDR_LO, amdgpu_gc_11_0_0.regCP_MES_MBADADDR_LO_BASE_IDX)
    # print("regCP_MES_MBADADDR_LO", hex(mes_ip))

    # mes_ip = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_MIP_LO, amdgpu_gc_11_0_0.regCP_MES_MIP_LO_BASE_IDX)
    # print("regCP_MES_MIP_LO", hex(mes_ip))

    # mes_ip = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_MTIME_LO, amdgpu_gc_11_0_0.regCP_MES_MTIME_LO_BASE_IDX)
    # print("regCP_MES_MTIME_LO", hex(mes_ip))

    # while True:
    #   mes_ip = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_INSTR_PNTR, amdgpu_gc_11_0_0.regCP_MES_INSTR_PNTR_BASE_IDX)
    #   print("regCP_MES_INSTR_PNTR", hex(mes_ip))

    # self.adev.gfx.soc21_grbm_select(0, 0, 0, 0)

    # print(self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX))

    # self.set_hw_resources()
