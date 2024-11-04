import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
from tinygrad.helpers import to_mv, mv_address

from extra.amdpci.amdring import AMDRing

class MES_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c
  AMDGPU_NAVI10_DOORBELL_MES_RING0 = 0x00B

  def __init__(self, adev):
    self.adev = adev
    
    self.sch_ctx_gpu_addr = self.adev.vmm.alloc_vram(0x1000, "sch_ctx")
    self.query_status_fence_gpu_mc_ptr = self.adev.vmm.alloc_vram(0x1000, "query_status_fence")
    self.write_fence_addr = self.adev.vmm.alloc_vram(0x1000, "w_fence")
    
    self.mes_queue = AMDRing(self.adev, me=3, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1))
    self.mes_queue.init_mes_mqd()
    self.setup()

  def init_mes_regs(self, ring): # mes_v11_0_hw_init
    self.adev.soc21_grbm_select(3, ring.pipe, 0, 0)

    is_act = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX)
    if (is_act & 1) == 1:
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST_BASE_IDX, 1)
      while (is_act & 1) == 1:
        # print(hex(is_act))
        # print(self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST, amdgpu_gc_11_0_0.regCP_HQD_DEQUEUE_REQUEST_BASE_IDX))
        is_act = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX)
      print("q deactivated")

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX, 0)
    
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_VMID, amdgpu_gc_11_0_0.regCP_HQD_VMID_BASE_IDX, ring.mqd.cp_hqd_vmid)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL, amdgpu_gc_11_0_0.regCP_HQD_PQ_DOORBELL_CONTROL_BASE_IDX, 0x0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_BASE_IDX, ring.mqd.cp_mqd_base_addr_lo)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_HI, amdgpu_gc_11_0_0.regCP_MQD_BASE_ADDR_HI_BASE_IDX, ring.mqd.cp_mqd_base_addr_hi)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MQD_CONTROL, amdgpu_gc_11_0_0.regCP_MQD_CONTROL_BASE_IDX, ring.mqd.cp_mqd_control)

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

    print("act?", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX))

    # def dump_struct(st):
    #   print("\t", st.__class__.__name__, end=" { ")
    #   for v in type(st)._fields_: print(f"{v[0]}={hex(getattr(st, v[0]))}", end=" ")
    #   print("}")
    # dump_struct(ring.mqd)

    self.adev.soc21_grbm_select(0, 0, 0, 0)

  def wdoorbell64(self, index, val):
    print("calling", index, val)
    for i in range(len(self.adev.doorbell)):
      self.adev.doorbell[index] = val
    self.adev.doorbell64[index] = val
    self.adev.doorbell64[index//2] = val

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
    self.adev.soc21_grbm_select(3, 0, 0, 0)
    print("act?", self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE, amdgpu_gc_11_0_0.regCP_HQD_ACTIVE_BASE_IDX))
    self.adev.soc21_grbm_select(0, 0, 0, 0)
    while self.fence_view[0] == 0:
      # print(self.mes_queue.rptr[0], self.mes_queue.wptr[0])
      # print("PFS", self.adev.vmm.collect_pfs())
      pass
  
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
      mes_set_hw_res_pkt.gc_base[i] = self.adev.reg_off("GC", 0, i)
      mes_set_hw_res_pkt.mmhub_base[i] = self.adev.reg_off("MMHUB", 0, i)
      mes_set_hw_res_pkt.osssys_base[i] = self.adev.reg_off("OSSSYS", 0, i)

    mes_set_hw_res_pkt.disable_reset = 1
    mes_set_hw_res_pkt.disable_mes_log = 1
    mes_set_hw_res_pkt.use_different_vmid_compute = 1
    mes_set_hw_res_pkt.enable_reg_active_poll = 1
    mes_set_hw_res_pkt.enable_level_process_quantum_check = 1
    mes_set_hw_res_pkt.oversubscription_timer = 50

    self.submit_pkt_and_poll_completion(mes_set_hw_res_pkt)

  def setup(self):
    self.init_mes_regs(self.mes_queue)

    self.adev.soc21_grbm_select(3, 0, 0, 0)
    self.sched_version = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_GP3_LO, amdgpu_gc_11_0_0.regCP_MES_GP3_LO_BASE_IDX)
    print("MES API v", (self.sched_version & 0x00fff000) >> 12)

    # reset mes
    mes_cntr = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX)
    if (mes_cntr & 0x40000000): print("MES HALTED")

    mes_cntr_reset = mes_cntr | (0x00010000) # pipe0 reset
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX, mes_cntr_reset)

    start_mes = (0x04000000) # pipe0 active
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX, start_mes)

    mes_cntr = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX)
    if (mes_cntr & 0x40000000): print("MES HALTED")
    print("MES STATUS", hex(mes_cntr))

    

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

    # self.adev.soc21_grbm_select(0, 0, 0, 0)

    # print(self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regCP_MES_CNTL, amdgpu_gc_11_0_0.regCP_MES_CNTL_BASE_IDX))

    self.set_hw_resources()
