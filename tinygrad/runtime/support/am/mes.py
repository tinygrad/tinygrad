# import os, ctypes, time
# from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, mes_v11_api_def
# from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
# from tinygrad.helpers import to_mv, mv_address, getenv
# from tinygrad.runtime.support.am.amring import AMRing

# class MES_IP:
#   AMDGPU_NAVI10_DOORBELL_MES_RING0 = 0x00B
#   AMDGPU_NAVI10_DOORBELL_MES_RING1 = 0x00C

#   def __init__(self, adev):
#     self.adev = adev

#     # self.sch_ctx_vm = self.adev.mm.valloc(0x1000, uncached=True)
#     # self.query_status_fence_vm = self.adev.mm.valloc(0x1000, uncached=True)
#     # self.write_fence_vm = self.adev.mm.valloc(0x1000, uncached=True)

#   def mes_enable(self):
#     self.adev.regCP_MES_CNTL.write((1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE0_RESET__SHIFT) | (1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE1_RESET__SHIFT))

#     ucodes = [
#       self.adev.fw.mes_fw.header.mes_uc_start_addr_lo | (self.adev.fw.mes_fw.header.mes_uc_start_addr_hi << 32),
#       self.adev.fw.mes_kiq_fw.header.mes_uc_start_addr_lo | (self.adev.fw.mes_kiq_fw.header.mes_uc_start_addr_hi << 32),
#     ]

#     for pipe in range(2):
#       self.adev.gfx.soc21_grbm_select(3, pipe, 0, 0)
#       self.adev.regCP_MES_PRGRM_CNTR_START.write((ucodes[pipe] >> 2) & 0xffffffff)
#       self.adev.regCP_MES_PRGRM_CNTR_START_HI.write((ucodes[pipe] >> 34) & 0xffffffff)

#     self.adev.gfx.soc21_grbm_select(0, 0, 0, 0)

#     # unhalt mes, and activate it
#     self.adev.regCP_MES_CNTL.write((1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE0_ACTIVE__SHIFT) | (1 << amdgpu_gc_11_0_0.CP_MES_CNTL__MES_PIPE1_ACTIVE__SHIFT))

#   def kiq_setting(self, ring):
#     val = (self.adev.regRLC_CP_SCHEDULERS.read() & 0xffffff00) | (ring.me << 5) | (ring.pipe << 3) | (ring.queue)
#     self.adev.regRLC_CP_SCHEDULERS.write(val)
#     self.adev.regRLC_CP_SCHEDULERS.write(val | 0x80)

#   def kiq_ring_test_helper(self):
#     self.adev.wreg(sc_reg:=0xc040, 0xcafedead) # sratch reg
#     self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_WRITE_DATA, 3))
#     self.mes_kiq.write(1 << 16)
#     self.mes_kiq.write(sc_reg) 
#     self.mes_kiq.write(0x0)
#     self.mes_kiq.write(0xdeadc0de)

#     self.adev.wdoorbell64(self.mes_kiq.doorbell_index, self.mes_kiq.next_ptr)
#     while self.adev.rreg(sc_reg) != 0xdeadc0de: pass

#   def kiq_map_queue(self, ring_to_enable, is_compute=False):
#     if is_compute:
#       me = 1
#       eng_sel = 0
#     else:
#       # MES queue
#       me = 2
#       eng_sel = 5

#     self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_MAP_QUEUES, 5))
#     self.mes_kiq.write(
#       amd_gpu.PACKET3_MAP_QUEUES_QUEUE_SEL(0) |
#       amd_gpu.PACKET3_MAP_QUEUES_VMID(ring_to_enable.vmid) |
#       amd_gpu.PACKET3_MAP_QUEUES_QUEUE(ring_to_enable.queue) |
#       amd_gpu.PACKET3_MAP_QUEUES_PIPE(ring_to_enable.pipe) |
#       amd_gpu.PACKET3_MAP_QUEUES_ME(me) |
#       amd_gpu.PACKET3_MAP_QUEUES_QUEUE_TYPE(0) | # queue_type: normal compute queue 
#       amd_gpu.PACKET3_MAP_QUEUES_ALLOC_FORMAT(0x0) | # alloc format: all_on_one_pipe
#       amd_gpu.PACKET3_MAP_QUEUES_ENGINE_SEL(eng_sel) |
#       amd_gpu.PACKET3_MAP_QUEUES_NUM_QUEUES(1)
#     )
#     self.mes_kiq.write(amd_gpu.PACKET3_MAP_QUEUES_DOORBELL_OFFSET(ring_to_enable.doorbell_index))
#     self.mes_kiq.write(ring_to_enable.mqd_vm.vaddr & 0xffffffff)
#     self.mes_kiq.write(ring_to_enable.mqd_vm.vaddr >> 32)
#     self.mes_kiq.write(ring_to_enable.wptr_vm.vaddr & 0xffffffff)
#     self.mes_kiq.write(ring_to_enable.wptr_vm.vaddr >> 32)

#     # Just to test if command is executed
#     return self.kiq_ring_test_helper()

#   def kiq_set_resources(self, queue_mask):
#     self.mes_kiq.write(amd_gpu.PACKET3(amd_gpu.PACKET3_SET_RESOURCES, 6))
#     self.mes_kiq.write(amd_gpu.PACKET3_SET_RESOURCES_VMID_MASK(0) |
#           amd_gpu.PACKET3_SET_RESOURCES_UNMAP_LATENTY(0xa) | # 1s
#           amd_gpu.PACKET3_SET_RESOURCES_QUEUE_TYPE(0))
#     self.mes_kiq.write(queue_mask & 0xffffffff)
#     self.mes_kiq.write(queue_mask >> 32)
#     self.mes_kiq.write(0)
#     self.mes_kiq.write(0)
#     self.mes_kiq.write(0)
#     self.mes_kiq.write(0)

#     # Just to test if command is executed
#     return self.kiq_ring_test_helper()

#   def kiq_hw_init(self):
#     self.mes_kiq = AMRing(self.adev, size=0x100000, me=3, pipe=1, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING1 << 1))

#     self.mes_enable()
#     self.kiq_setting(self.mes_kiq)
#     self.init_mes_regs(self.mes_kiq)

#     self.mes_hw_init()
#     print("MES done")

#   def mes_hw_init(self):
#     self.mes_ring = AMRing(self.adev, size=0x100000, me=3, pipe=0, queue=0, vmid=0, doorbell_index=(self.AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1))
#     # self.queue_init(self.mes_ring, is_kiq=False)
#     self.kiq_map_queue(self.mes_ring, is_compute=False) # this is mes queue
#     self.set_hw_resources()
#     self.query_sched_status()

#   def init_mes_regs(self, ring):
#     self.adev.gfx.soc21_grbm_select(3, ring.pipe, 0, 0)

#     self.adev.regCP_HQD_VMID.write(ring.mqd.cp_hqd_vmid)
#     # self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.write(0x0)

#     self.adev.regCP_MQD_BASE_ADDR.write(ring.mqd.cp_mqd_base_addr_lo)
#     self.adev.regCP_MQD_BASE_ADDR_HI.write(ring.mqd.cp_mqd_base_addr_hi)

#     self.adev.regCP_MQD_CONTROL.write(0)

#     self.adev.regCP_HQD_PQ_BASE.write(ring.mqd.cp_hqd_pq_base_lo)
#     self.adev.regCP_HQD_PQ_BASE_HI.write(ring.mqd.cp_hqd_pq_base_hi)

#     self.adev.regCP_HQD_PQ_RPTR_REPORT_ADDR.write(ring.mqd.cp_hqd_pq_rptr_report_addr_lo)
#     self.adev.regCP_HQD_PQ_RPTR_REPORT_ADDR_HI.write(ring.mqd.cp_hqd_pq_rptr_report_addr_hi)

#     # assert ring.mqd.cp_hqd_pq_control == 0xd8308011
#     self.adev.regCP_HQD_PQ_CONTROL.write(ring.mqd.cp_hqd_pq_control)

#     self.adev.regCP_HQD_PQ_WPTR_POLL_ADDR.write(ring.mqd.cp_hqd_pq_wptr_poll_addr_lo)
#     self.adev.regCP_HQD_PQ_WPTR_POLL_ADDR_HI.write(ring.mqd.cp_hqd_pq_wptr_poll_addr_hi)

#     self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.write(ring.mqd.cp_hqd_pq_doorbell_control)
#     self.adev.regCP_HQD_PERSISTENT_STATE.write(ring.mqd.cp_hqd_persistent_state)
#     self.adev.regCP_HQD_ACTIVE.write(ring.mqd.cp_hqd_active)

#     self.adev.gfx.soc21_grbm_select(0, 0, 0, 0)

#   def submit_pkt_and_poll_completion(self, pkt):
#     self.write_fence_vm.cpu_view().cast('I')[0] = 0
#     pkt.api_status.api_completion_fence_addr = self.write_fence_vm.mc_addr()
#     pkt.api_status.api_completion_fence_value = 1

#     for i in range(ctypes.sizeof(pkt) // 4): self.mes_ring.write(pkt.max_dwords_in_api[i])

#     self.adev.gmc.flush_hdp()
#     self.adev.wdoorbell64(self.mes_ring.doorbell_index, self.mes_ring.next_ptr)

#     while self.write_fence_vm.cpu_view().cast('I')[0] != 1:
#       self.adev.gmc.collect_pfs()

#     return 0

#   def query_sched_status(self):
#     mes_sch_status_pkt = mes_v11_api_def.union_MESAPI__QUERY_MES_STATUS()
#     mes_sch_status_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
#     mes_sch_status_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_QUERY_SCHEDULER_STATUS
#     mes_sch_status_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

#     return self.submit_pkt_and_poll_completion(mes_sch_status_pkt)

#   def set_hw_resources(self):
#     mes_set_hw_res_pkt = mes_v11_api_def.union_MESAPI_SET_HW_RESOURCES()
#     mes_set_hw_res_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
#     mes_set_hw_res_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_SET_HW_RSRC
#     mes_set_hw_res_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

#     mes_set_hw_res_pkt.vmid_mask_mmhub = 0xffffff00
#     mes_set_hw_res_pkt.vmid_mask_gfxhub = 0xffffff00
#     mes_set_hw_res_pkt.gds_size = 0x1000
#     mes_set_hw_res_pkt.paging_vmid = 0
#     mes_set_hw_res_pkt.g_sch_ctx_gpu_mc_ptr = self.sch_ctx_vm.mc_addr()
#     mes_set_hw_res_pkt.query_status_fence_gpu_mc_ptr = self.query_status_fence_vm.mc_addr()

#     for i,v in enumerate([0xc, 0xc, 0xc, 0xc, 0x0, 0x0, 0x0, 0x0]): mes_set_hw_res_pkt.compute_hqd_mask[i] = v
#     for i,v in enumerate([0xfffffffe, 0x0]): mes_set_hw_res_pkt.gfx_hqd_mask[i] = v
#     for i,v in enumerate([0xfc, 0xfc]): mes_set_hw_res_pkt.sdma_hqd_mask[i] = v
#     for i,v in enumerate([2048, 2050, 2052, 2054, 2056]): mes_set_hw_res_pkt.aggregated_doorbells[i] = v

#     for i in range(5):
#       mes_set_hw_res_pkt.gc_base[i] = self.adev.ip_base("GC", 0, i)
#       mes_set_hw_res_pkt.mmhub_base[i] = self.adev.ip_base("MMHUB", 0, i)
#       mes_set_hw_res_pkt.osssys_base[i] = self.adev.ip_base("OSSSYS", 0, i)

#     mes_set_hw_res_pkt.disable_reset = 1
#     mes_set_hw_res_pkt.disable_mes_log = 1
#     mes_set_hw_res_pkt.use_different_vmid_compute = 1
#     mes_set_hw_res_pkt.enable_reg_active_poll = 1
#     mes_set_hw_res_pkt.enable_level_process_quantum_check = 1
#     mes_set_hw_res_pkt.oversubscription_timer = 50

#     return self.submit_pkt_and_poll_completion(mes_set_hw_res_pkt)

#   def map_legacy_queue(self, ring, qtype=mes_v11_api_def.MES_QUEUE_TYPE_COMPUTE):
#     mes_add_queue_pkt = mes_v11_api_def.union_MESAPI__ADD_QUEUE()

#     mes_add_queue_pkt.header.type = mes_v11_api_def.MES_API_TYPE_SCHEDULER
#     mes_add_queue_pkt.header.opcode = mes_v11_api_def.MES_SCH_API_ADD_QUEUE
#     mes_add_queue_pkt.header.dwsize = mes_v11_api_def.API_FRAME_SIZE_IN_DWORDS

#     mes_add_queue_pkt.pipe_id = ring.pipe
#     mes_add_queue_pkt.queue_id = ring.queue
#     mes_add_queue_pkt.doorbell_offset = ring.doorbell_index
#     mes_add_queue_pkt.mqd_addr = ring.mqd_vm.vaddr
#     mes_add_queue_pkt.wptr_addr = ring.wptr_vm.vaddr
#     mes_add_queue_pkt.queue_type = qtype
#     mes_add_queue_pkt.map_legacy_kq = 1

#     return self.submit_pkt_and_poll_completion(mes_add_queue_pkt)

