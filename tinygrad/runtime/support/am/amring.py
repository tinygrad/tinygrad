import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address

class AMRing:
  def __init__(self, adev, size, pipe, queue, me, vmid, doorbell_index):
    self.adev, self.size, self.pipe, self.queue, self.me, self.vmid, self.doorbell_index = adev, size, pipe, queue, me, vmid, doorbell_index

    self.mqd_mv = memoryview(bytearray(ctypes.sizeof(amdgpu_2.struct_v11_compute_mqd)))
    self.mqd = amdgpu_2.struct_v11_compute_mqd.from_address(mv_address(self.mqd_mv))

    self.eop_vm = self.adev.mm.valloc(0x8000)
    self.mqd_vm = self.adev.mm.valloc(len(self.mqd_mv))
    self.rptr_vm = self.adev.mm.valloc(0x1000, uncached=True)
    self.wptr_vm = self.adev.mm.valloc(0x1000, uncached=True)
    self.ring_vm = self.adev.mm.valloc(self.size, uncached=True)

    self.rptr = self.rptr_vm.cpu_view().cast('Q')
    self.wptr = self.wptr_vm.cpu_view().cast('Q')

    self.rptr[0] = 0x0
    self.wptr[0] = 0x0

    self.next_ptr = 0

    self.fill_mqd()

  def fill_mqd(self):
    # self.mqd.header = 0xC0310800
    # self.mqd.compute_pipelinestat_enable = 0x00000001
    # self.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se4 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se5 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se6 = 0xffffffff
    # self.mqd.compute_static_thread_mgmt_se7 = 0xffffffff
    # self.mqd.compute_misc_reserved = 0x00000007

    self.mqd.cp_mqd_base_addr_lo = self.mqd_vm.vaddr & 0xfffffffc
    self.mqd.cp_mqd_base_addr_hi = (self.mqd_vm.vaddr >> 32) & 0xffffffff
    self.mqd.cp_hqd_active = 0
    self.mqd.vmid = self.vmid
    self.mqd.cp_hqd_persistent_state = 0x5501
    self.mqd.cp_hqd_pipe_priority = 0x2
    self.mqd.cp_hqd_queue_priority = 0xf
    self.mqd.cp_hqd_quantum = 0x111
    self.mqd.cp_hqd_pq_base_lo = (self.ring_vm.vaddr >> 8) & 0xffffffff
    self.mqd.cp_hqd_pq_base_hi = (self.ring_vm.vaddr >> 40) & 0xffffffff
    self.mqd.cp_hqd_pq_rptr = 0
    self.mqd.cp_hqd_pq_rptr_report_addr_lo = self.rptr_vm.vaddr & 0xfffffffc
    self.mqd.cp_hqd_pq_rptr_report_addr_hi = (self.rptr_vm.vaddr >> 32) & 0xffffffff
    self.mqd.cp_hqd_pq_wptr_poll_addr_lo = self.wptr_vm.vaddr & 0xfffffffc
    self.mqd.cp_hqd_pq_wptr_poll_addr_hi = (self.wptr_vm.vaddr >> 32) & 0xffffffff
    self.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (self.doorbell_index << 2)
    # self.mqd.reserved_144 = 0
    self.mqd.cp_hqd_pq_control = 0x10008511
    # self.mqd.cp_hqd_ib_base_addr_lo = 0
    # self.mqd.cp_hqd_ib_base_addr_hi = 0
    # self.mqd.cp_hqd_ib_rptr = 0
    self.mqd.cp_hqd_ib_control = 0x300000
    # self.mqd.cp_hqd_iq_timer = 0
    # self.mqd.cp_hqd_iq_rptr = 0
    # self.mqd.cp_hqd_dequeue_request = 0
    # self.mqd.cp_hqd_dma_offload = 0
    # self.mqd.cp_hqd_sema_cmd = 0
    # self.mqd.cp_hqd_msg_type = 0
    # self.mqd.cp_hqd_atomic0_preop_lo = 0
    # self.mqd.cp_hqd_atomic0_preop_hi = 0
    # self.mqd.cp_hqd_atomic1_preop_lo = 0
    # self.mqd.cp_hqd_atomic1_preop_hi = 0
    self.mqd.cp_hqd_hq_status0 = 0x60004040 # wtf?
    # self.mqd.cp_hqd_hq_control0 = 0
    self.mqd.cp_mqd_control = 0x100
    # self.mqd.cp_hqd_hq_status1 = 0
    # self.mqd.cp_hqd_hq_control1 = 0
    self.mqd.cp_hqd_eop_base_addr_lo = (self.eop_vm.vaddr >> 8) & 0xffffffff
    self.mqd.cp_hqd_eop_base_addr_hi = (self.eop_vm.vaddr >> 40) & 0xffffffff
    self.mqd.cp_hqd_eop_control = 0x9
    self.mqd.cp_hqd_eop_rptr = 1 << amdgpu_gc_11_0_0.CP_HQD_EOP_RPTR__INIT_FETCHER__SHIFT

    # self.mqd.cp_hqd_eop_rptr = 0x400002a0
    # self.mqd.cp_hqd_eop_wptr = 0x3ff82a0
    # self.mqd.cp_hqd_eop_done_events; // offset: 170  (0xAA)
    # self.mqd.cp_hqd_ctx_save_base_addr_lo; // offset: 171  (0xAB)
    # self.mqd.cp_hqd_ctx_save_base_addr_hi; // offset: 172  (0xAC)
    # self.mqd.cp_hqd_ctx_save_control; // offset: 173  (0xAD)
    # self.mqd.cp_hqd_cntl_stack_offset; // offset: 174  (0xAE)
    # self.mqd.cp_hqd_cntl_stack_size; // offset: 175  (0xAF)
    # self.mqd.cp_hqd_wg_state_offset; // offset: 176  (0xB0)
    # self.mqd.cp_hqd_ctx_save_size; // offset: 177  (0xB1)
    # self.mqd.cp_hqd_gds_resource_state; // offset: 178  (0xB2)
    # self.mqd.cp_hqd_error; // offset: 179  (0xB3)
    # self.mqd.cp_hqd_eop_wptr_mem; // offset: 180  (0xB4)
    # self.mqd.cp_hqd_aql_control; // offset: 181  (0xB5)
    # self.mqd.cp_hqd_pq_wptr_lo; // offset: 182  (0xB6)
    # self.mqd.cp_hqd_pq_wptr_hi; // offset: 183  (0xB7)

    # self.mqd.cp_hqd_eop_base_addr_lo = (self.eop_vm.vaddr >> 8) & 0xffffffff
    # self.mqd.cp_hqd_eop_base_addr_hi = (self.eop_vm.vaddr >> 40) & 0xffffffff
    # self.mqd.cp_hqd_eop_control = 0x11

    # self.mqd.cp_mqd_control = amdgpu_gc_11_0_0.CP_MQD_CONTROL__PRIV_STATE_MASK

    # # print(hex(self.ring_vm.vaddr))
    # self.mqd.cp_hqd_pq_base_lo = (self.ring_vm.vaddr >> 8) & 0xffffffff
    # self.mqd.cp_hqd_pq_base_hi = (self.ring_vm.vaddr >> 40) & 0xffffffff

    # self.mqd.cp_hqd_pq_rptr_report_addr_lo = self.rptr_vm.vaddr & 0xfffffffc
    # self.mqd.cp_hqd_pq_rptr_report_addr_hi = (self.rptr_vm.vaddr >> 32) & 0xffffffff

    # self.mqd.cp_hqd_pq_wptr_poll_addr_lo = self.wptr_vm.vaddr & 0xfffffffc
    # self.mqd.cp_hqd_pq_wptr_poll_addr_hi = (self.wptr_vm.vaddr >> 32) & 0xffffffff

    # assert self.size in {0x100000}
    # self.mqd.cp_hqd_pq_control = 5 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT | \
		# 	(1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__QUEUE_FULL_EN__SHIFT) | \
    #   (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__NO_UPDATE_RPTR__SHIFT) | \
    #   (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__UNORD_DISPATCH__SHIFT) | \
    #   (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__TUNNEL_DISPATCH__SHIFT) | \
    #   (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__PRIV_STATE__SHIFT) | \
    #   (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__KMD_QUEUE__SHIFT) | 0x11 # size

    # self.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (self.doorbell_index << 2)
    # self.mqd.cp_hqd_vmid = self.vmid

    # self.mqd.cp_hqd_persistent_state = 0xbe05501
    # self.mqd.cp_hqd_ib_control = 0x300000 # 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT
    # self.mqd.cp_hqd_iq_timer = 0x0
    # self.mqd.cp_hqd_quantum = 0x0

    # self.mqd.cp_hqd_pipe_priority = 0x2
    # self.mqd.cp_hqd_queue_priority = 0xf
    # self.mqd.cp_hqd_active = 1

    self.mqd_vm.cpu_view()[:len(self.mqd_mv)] = self.mqd_mv
    self.adev.gmc.flush_hdp()

  def write(self, value):
    self.ring_vm.cpu_view().cast('I')[self.next_ptr % (self.size // 4)] = value
    self.next_ptr += 1
    self.wptr[0] = self.next_ptr
