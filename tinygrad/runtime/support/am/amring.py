import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address

class AMRegister:
  def __init__(self, adev, regoff): self.adev, self.regoff = adev, regoff
  def write(self, value, inst=0): return self.adev.wreg(self.regoff, value)
  def read(self, inst=0): return self.adev.rreg(self.regoff)

class AMRing:
  def __init__(self, adev, size, pipe, queue, me, vmid, doorbell_index):
    self.adev, self.size, self.pipe, self.queue, self.me, self.vmid, self.doorbell_index = adev, size, pipe, queue, me, vmid, doorbell_index

    self.mqd_mv = memoryview(bytearray(ctypes.sizeof(amdgpu_2.struct_v11_compute_mqd)))
    self.mqd = amdgpu_2.struct_v11_compute_mqd.from_address(mv_address(self.mqd_mv))

    self.eop_vm = self.adev.mm.valloc(0x1000)
    self.mqd_vm = self.adev.mm.valloc(0x1000)
    self.ring_vm = self.adev.mm.valloc(self.size + 0x1000, uncached=True)
    self.wptr_vm = self.ring_vm.offset(self.size)
    self.rptr_vm = self.ring_vm.offset(self.size + 0x8)

    self.rptr = self.rptr_vm.cpu_view().cast('Q')
    self.wptr = self.wptr_vm.cpu_view().cast('Q')

    self.rptr[0] = 0x0
    self.wptr[0] = 0x0

    self.next_ptr = 0

    self.fill_mqd()

  def fill_mqd(self):
    self.mqd.header = 0xC0310800
    self.mqd.cp_mqd_base_addr_lo = self.mqd_vm.vaddr & 0xfffffffc
    self.mqd.cp_mqd_base_addr_hi = (self.mqd_vm.vaddr >> 32) & 0xffffffff
    self.mqd.cp_hqd_active = 0
    self.mqd.vmid = self.vmid
    self.mqd.cp_hqd_persistent_state = 0x5501
    self.mqd.cp_hqd_pipe_priority = 0x2
    self.mqd.cp_hqd_queue_priority = 0xf
    self.mqd.cp_hqd_quantum = 0x0
    self.mqd.cp_hqd_pq_base_lo = (self.ring_vm.vaddr >> 8) & 0xffffffff
    self.mqd.cp_hqd_pq_base_hi = (self.ring_vm.vaddr >> 40) & 0xffffffff
    self.mqd.cp_hqd_pq_rptr_report_addr_lo = self.rptr_vm.vaddr & 0xfffffffc
    self.mqd.cp_hqd_pq_rptr_report_addr_hi = (self.rptr_vm.vaddr >> 32) & 0xffffffff
    self.mqd.cp_hqd_pq_wptr_poll_addr_lo = self.wptr_vm.vaddr & 0xfffffffc
    self.mqd.cp_hqd_pq_wptr_poll_addr_hi = (self.wptr_vm.vaddr >> 32) & 0xffffffff
    self.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (self.doorbell_index << 2)
    self.mqd.cp_hqd_pq_control = 0x10000511
    self.mqd.cp_hqd_ib_control = 0x300000
    self.mqd.cp_hqd_hq_status0 = 0x20004000
    self.mqd.cp_mqd_control = 0x100
    self.mqd.cp_hqd_eop_base_addr_lo = (self.eop_vm.vaddr >> 8) & 0xffffffff
    self.mqd.cp_hqd_eop_base_addr_hi = (self.eop_vm.vaddr >> 40) & 0xffffffff
    self.mqd.cp_hqd_eop_control = 0x9
    self.mqd.cp_hqd_eop_rptr = 0

    self.mqd_vm.cpu_view()[:len(self.mqd_mv)] = self.mqd_mv
    self.adev.gmc.flush_hdp()
