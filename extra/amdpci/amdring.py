import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2
from tinygrad.helpers import to_mv, mv_address

class AMDRing:
  def __init__(self, adev, vmid):
    self.adev = adev

    self.vmid = vmid

    self.pipe = 0
    self.mqd_mv = memoryview(bytearray(ctypes.sizeof(amdgpu_2.struct_v11_compute_mqd)))
    self.mqd = amdgpu_2.struct_v11_compute_mqd.from_address(mv_address(self.mqd_mv))

    self.eop_gpu_addr = self.adev.vmm.alloc_vram(0x1000, "eop")
    self.mqd_gpu_addr = self.adev.vmm.alloc_vram(len(self.mqd_mv), "mqd")

    self.rptr_gpu_addr = self.adev.vmm.alloc_vram(len(self.mqd_mv), "w/rptr")
    self.wptr_gpu_addr = self.rptr_gpu_addr + 8

    self.ring_gpu_addr = self.adev.vmm.alloc_vram(0x1000, "ring")
    self.ring_view = self.adev.vmm.vram_to_cpu_mv(self.ring_gpu_addr, 0x1000)

    self.doorbell_index = 16
    self.next_ptr = 0
    self.init_compute_mqd()

  def init_compute_mqd(self):
    self.mqd.header = 0xC0310800 # some magic?
    self.mqd.compute_pipelinestat_enable = 0x00000001
    self.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
    self.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
    self.mqd.compute_misc_reserved = 0x00000007

    eop_base_addr = self.eop_gpu_addr >> 8
    self.mqd.cp_hqd_eop_base_addr_lo = eop_base_addr & 0xffffffff
    self.mqd.cp_hqd_eop_base_addr_hi = (eop_base_addr >> 32) & 0xffffffff
    self.mqd.cp_hqd_eop_control = 0x8

    # disable the queue if it's active
    self.mqd.cp_hqd_pq_rptr = 0
    self.mqd.cp_hqd_pq_wptr_lo = 0
    self.mqd.cp_hqd_pq_wptr_hi = 0

    self.mqd.cp_mqd_base_addr_lo = self.mqd_gpu_addr & 0xfffffffc
    self.mqd.cp_mqd_base_addr_hi = (self.mqd_gpu_addr >> 32) & 0xffffffff
    self.mqd.cp_mqd_control = 0x40000060 # set MQD vmid to 0

    self.mqd.cp_hqd_pq_base_lo = self.ring_gpu_addr & 0xffffffff
    self.mqd.cp_hqd_pq_base_hi = (self.ring_gpu_addr >> 32) & 0xffffffff

    self.mqd.cp_hqd_pq_rptr_report_addr_lo = self.rptr_gpu_addr & 0xfffffffc
    self.mqd.cp_hqd_pq_rptr_report_addr_hi = (self.rptr_gpu_addr >> 32) & 0xffffffff

    self.mqd.cp_hqd_pq_wptr_report_addr_lo = self.wptr_gpu_addr & 0xfffffffc
    self.mqd.cp_hqd_pq_wptr_report_addr_hi = (self.wptr_gpu_addr >> 32) & 0xffffffff
    self.mqd.cp_hqd_pq_control = 0xd8308011

    self.mqd.cp_hqd_pq_doorbell_control = (1 << 0x1e) | (self.doorbell_index << 2)

    self.mqd.cp_hqd_pq_doorbell_control = 0x40000060
    self.mqd.cp_hqd_vmid = self.vmid
    self.mqd.cp_hqd_active = 1

    self.mqd.cp_hqd_persistent_state = 0xbe05501
    self.mqd.cp_hqd_ib_control = 0x300000
    self.mqd.cp_hqd_iq_timer = 0x0
    self.mqd.cp_hqd_quantum = 0x0

    self.adev.vmm.vram_to_cpu_mv(self.mqd_gpu_addr, len(self.mqd_mv))[:] = self.mqd_mv

  def write(self, value):
    self.adev.vmm.vram_to_cpu_mv(self.ring_gpu_addr, 0x1000).cast('I')[self.next_ptr] = value
    self.adev.vmm.vram_to_cpu_mv(self.wptr_gpu_addr, 8).cast('I')[0] = self.next_ptr
    self.next_ptr += 4
