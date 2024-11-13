import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address

class AMRing:
  def __init__(self, adev, size, pipe, queue, me, vmid, doorbell_index):
    self.adev, self.size, self.pipe, self.queue, self.me, self.vmid, self.doorbell_index = adev, size, pipe, queue, me, vmid, doorbell_index

    self.mqd_mv = memoryview(bytearray(ctypes.sizeof(amdgpu_2.struct_v11_compute_mqd)))
    self.mqd = amdgpu_2.struct_v11_compute_mqd.from_address(mv_address(self.mqd_mv))

    self.eop_pm = self.adev.mm.palloc(0x8000)
    self.mqd_pm = self.adev.mm.palloc(len(self.mqd_mv))
    self.rptr_pm = self.adev.mm.palloc(0x1000)
    self.wptr_pm = self.adev.mm.palloc(0x1000)
    self.ring_pm = self.adev.mm.palloc(self.size)

    self.rptr = self.rptr_pm.cpu_view().cast('Q')
    self.wptr = self.wptr_pm.cpu_view().cast('Q')

    self.rptr[0] = 0x0
    self.wptr[0] = 0x0

    self.next_ptr = 0

    self.fill_mqd()

  def fill_mqd(self):
    self.mqd.header = 0xC0310800
    self.mqd.compute_pipelinestat_enable = 0x00000001
    self.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se4 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se5 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se6 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se7 = 0xffffffff
    self.mqd.compute_misc_reserved = 0x00000007

    self.mqd.cp_hqd_eop_base_addr_lo = (self.eop_pm.mc_addr() >> 8) & 0xffffffff
    self.mqd.cp_hqd_eop_base_addr_hi = (self.eop_pm.mc_addr() >> 40) & 0xffffffff
    self.mqd.cp_hqd_eop_control = 0x8

    # init it only once, fine to skip this.
    # disable the queue if it's active
    # self.mqd.cp_hqd_pq_rptr = 0
    # self.mqd.cp_hqd_pq_wptr_lo = 0
    # self.mqd.cp_hqd_pq_wptr_hi = 0

    self.mqd.cp_mqd_base_addr_lo = self.mqd_pm.mc_addr() & 0xfffffffc
    self.mqd.cp_mqd_base_addr_hi = (self.mqd_pm.mc_addr() >> 32) & 0xffffffff

    self.mqd.cp_mqd_control = amdgpu_gc_11_0_0.CP_MQD_CONTROL__PRIV_STATE_MASK

    self.mqd.cp_hqd_pq_base_lo = (self.ring_pm.mc_addr() >> 8) & 0xffffffff
    self.mqd.cp_hqd_pq_base_hi = (self.ring_pm.mc_addr() >> 40) & 0xffffffff

    self.mqd.cp_hqd_pq_rptr_report_addr_lo = self.rptr_pm.mc_addr() & 0xfffffffc
    self.mqd.cp_hqd_pq_rptr_report_addr_hi = (self.rptr_pm.mc_addr() >> 32) & 0xffffffff

    self.mqd.cp_hqd_pq_wptr_poll_addr_lo = self.wptr_pm.mc_addr() & 0xfffffffc
    self.mqd.cp_hqd_pq_wptr_poll_addr_hi = (self.wptr_pm.mc_addr() >> 32) & 0xffffffff

    assert self.size in {0x100000}
    self.mqd.cp_hqd_pq_control = 5 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT | \
			(1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__UNORD_DISPATCH__SHIFT) | \
      (0 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__TUNNEL_DISPATCH__SHIFT) | \
      (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__PRIV_STATE__SHIFT) | \
      (1 << amdgpu_gc_11_0_0.CP_HQD_PQ_CONTROL__KMD_QUEUE__SHIFT) | 0x11 # size

    self.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (self.doorbell_index << 2)
    self.mqd.cp_hqd_vmid = self.vmid

    self.mqd.cp_hqd_persistent_state = 0xbe05501
    self.mqd.cp_hqd_ib_control = 0x300000 # 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT
    self.mqd.cp_hqd_iq_timer = 0x0
    self.mqd.cp_hqd_quantum = 0x0

    self.mqd.cp_hqd_pipe_priority = 0x2
    self.mqd.cp_hqd_queue_priority = 0xf
    self.mqd.cp_hqd_active = 1

    self.mqd_pm.cpu_view()[:len(self.mqd_mv)] = self.mqd_mv
    self.adev.gmc.flush_hdp()

  def write(self, value):
    self.ring_pm.cpu_view().cast('I')[self.next_ptr % (self.size // 4)] = value
    self.next_ptr += 1
    self.wptr[0] = self.next_ptr
