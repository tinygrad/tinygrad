from typing import Optional, Union
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, amdgpu_mmhub_3_0_0
from tinygrad.helpers import to_mv, mv_address, round_up

class GMC_IP:
  def __init__(self, adev):
    self.adev = adev

    self.mc_base = 0x8000000000 # TODO: read from register
    self.mc_end = 0x85feffffff

    self.vm_base = 0x7F0000000000
    self.vm_end = self.vm_base + (512 * (1 << 30)) - 1

    self.shared_aperture_base = 0x2000000000000000
    self.shared_aperture_end = self.shared_aperture_base + (4 << 30) - 1
    self.private_aperture_base = 0x1000000000000000
    self.private_aperture_end = self.private_aperture_base + (4 << 30) - 1

    self.memscratch_pm = self.adev.mm.palloc(0x1000)
    self.dummy_page_pm = self.adev.mm.palloc(0x1000)

  def init(self, root_pt):
    print("GMC init")
    self.root_pt = root_pt
    self.vm_config = 0x1fffe05 # 2 level, 1 gb huge pages
    self.init_mmhub()

  def init_aperture_regs(self, block:Union["MM", "GC"]):
    getattr(self.adev, f"reg{block}VM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32").write((self.vm_base >> 12) & 0xffffffff)
    getattr(self.adev, f"reg{block}VM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32").write(self.vm_base >> 44)

    getattr(self.adev, f"reg{block}VM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32").write((self.vm_end >> 12) & 0xffffffff)
    getattr(self.adev, f"reg{block}VM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32").write(self.vm_end >> 44)

    getattr(self.adev, f"reg{block}VM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32").write((self.root_pt.pmem.paddr & 0xffffffff) | 1)
    getattr(self.adev, f"reg{block}VM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32").write((self.root_pt.pmem.paddr >> 32) & 0xffffffff)

    getattr(self.adev, f"reg{block}VM_CONTEXT8_PAGE_TABLE_START_ADDR_LO32").write((self.vm_base >> 12) & 0xffffffff)
    getattr(self.adev, f"reg{block}VM_CONTEXT8_PAGE_TABLE_START_ADDR_HI32").write(self.vm_base >> 44)

    getattr(self.adev, f"reg{block}VM_CONTEXT8_PAGE_TABLE_END_ADDR_LO32").write((self.vm_end >> 12) & 0xffffffff)
    getattr(self.adev, f"reg{block}VM_CONTEXT8_PAGE_TABLE_END_ADDR_HI32").write(self.vm_end >> 44)

    getattr(self.adev, f"reg{block}VM_CONTEXT8_PAGE_TABLE_BASE_ADDR_LO32").write((self.root_pt.pmem.paddr & 0xffffffff) | 1)
    getattr(self.adev, f"reg{block}VM_CONTEXT8_PAGE_TABLE_BASE_ADDR_HI32").write((self.root_pt.pmem.paddr >> 32) & 0xffffffff)

  def init_system_aperture_regs(self, block:Union["MM", "GC"]):
    getattr(self.adev, f"reg{block}MC_VM_AGP_BASE").write(0)
    getattr(self.adev, f"reg{block}MC_VM_AGP_BOT").write(0xffffffffffff >> 24) # disable AGP
    getattr(self.adev, f"reg{block}MC_VM_AGP_TOP").write(0)
    getattr(self.adev, f"reg{block}MC_VM_SYSTEM_APERTURE_LOW_ADDR").write(self.mc_base >> 18)
    getattr(self.adev, f"reg{block}MC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(self.mc_end >> 18)

    getattr(self.adev, f"reg{block}MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB").write(self.memscratch_pm.paddr >> 12)
    getattr(self.adev, f"reg{block}MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB").write(self.memscratch_pm.paddr >> 44)

    getattr(self.adev, f"reg{block}VM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32").write(self.dummy_page_pm.paddr >> 12)
    getattr(self.adev, f"reg{block}VM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32").write(self.dummy_page_pm.paddr >> 44)

    getattr(self.adev, f"reg{block}VM_L2_PROTECTION_FAULT_CNTL2").write(0x000E0000) # TODO: write up!

  def init_tlb_regs(self, block:Union["MM", "GC"]):
    getattr(self.adev, f"reg{block}MC_VM_MX_L1_TLB_CNTL").write(0x00001859) # TODO: write up!

  def init_cache_regs(self, block:Union["MM", "GC"]):
    getattr(self.adev, f"reg{block}VM_L2_CNTL").write(0x80e01)
    getattr(self.adev, f"reg{block}VM_L2_CNTL2").write(0x3)
    getattr(self.adev, f"reg{block}VM_L2_CNTL3").write(0x80130009)
    getattr(self.adev, f"reg{block}VM_L2_CNTL4").write(0x1)
    getattr(self.adev, f"reg{block}VM_L2_CNTL5").write(0x3fe0)

  def enable_vm(self, block:Union["MM", "GC"]):
    # TODO: take from PTEs
    getattr(self.adev, f"reg{block}VM_CONTEXT0_CNTL").write(self.vm_config) # 2 level
    getattr(self.adev, f"reg{block}VM_CONTEXT8_CNTL").write(self.vm_config) # 2 level

  def disable_identity_aperture(self, block:Union["MM", "GC"]):
    getattr(self.adev, f"reg{block}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32").write(0xffffffff)
    getattr(self.adev, f"reg{block}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32").write(0xf)

    getattr(self.adev, f"reg{block}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32").write(0x0)
    getattr(self.adev, f"reg{block}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32").write(0x0)

    getattr(self.adev, f"reg{block}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32").write(0x0)
    getattr(self.adev, f"reg{block}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32").write(0x0)

  def program_invalidation(self, block:Union["MM", "GC"]):
    for i in range(18):
      getattr(self.adev, f"reg{block}VM_INVALIDATE_ENG{i}_ADDR_RANGE_LO32").write(0xffffffff)
      getattr(self.adev, f"reg{block}VM_INVALIDATE_ENG{i}_ADDR_RANGE_HI32").write(0x1f)

  def init_mmhub(self):
    print("MMHUB init")
    self.init_aperture_regs("MM")
    self.init_system_aperture_regs("MM")
    self.init_tlb_regs("MM")
    self.init_cache_regs("MM")

    self.enable_vm("MM")
    self.disable_identity_aperture("MM")
    self.program_invalidation("MM")

  def init_gfxhub(self):
    print("GFXHUB init")
    self.init_aperture_regs("GC")
    self.init_system_aperture_regs("GC")
    self.init_tlb_regs("GC")
    self.init_cache_regs("GC")

    self.enable_vm("GC")
    self.disable_identity_aperture("GC")
    self.program_invalidation("GC")

  def flush_hdp(self): self.adev.wreg(0x1fc00, 0x0) # TODO: write up!
  def flush_tlb_gfxhub(self, vmid, vmhub, flush_type):
    assert vmid == 0 and vmhub == 0 and flush_type == 0

    self.flush_hdp()

    self.adev.wreg(0x291c, 0xf80001)
    while self.adev.rreg(0x292e) != 1: pass

  def flush_tlb_mmhub(self, vmid, vmhub, flush_type):
    assert vmid == 0 and vmhub == 0 and flush_type == 0

    self.flush_hdp()

    self.adev.wreg(0x1a774, 0xf80001)
    while self.adev.rreg(0x1a786) != 1: pass

    self.adev.wreg(0x1a762, 0x0)
    while self.adev.rreg(0x1a786) != 1: pass

    self.adev.wreg(0x1a71b, 0x12104010)

  def collect_pfs(self):
    gfx = self.adev.regGCVM_L2_PROTECTION_FAULT_STATUS.read()

    if gfx != 0:
      addr = self.adev.regGCVM_L2_PROTECTION_FAULT_ADDR_LO32.read()
      addr |= self.adev.regGCVM_L2_PROTECTION_FAULT_ADDR_HI32.read() << 32
      addr <<= 12

      client_mappings = ["CB/DB",
        "Reserved",
        "GE1",
        "GE2",
        "CPF",
        "CPC",
        "CPG",
        "RLC",
        "TCP",
        "SQC (inst)",
        "SQC (data)",
        "SQG",
        "Reserved",
        "SDMA0",
        "SDMA1",
        "GCR",
        "SDMA2",
        "SDMA3"
      ]
      cid = (gfx & amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__CID_MASK) >> amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__CID__SHIFT
      more_faults = (gfx & amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__MORE_FAULTS_MASK) >> amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__MORE_FAULTS__SHIFT
      rw = (gfx & amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__RW_MASK) >> amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__RW__SHIFT
      vmid = (gfx & amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__VMID_MASK) >> amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__VMID__SHIFT
      mapping_error = (gfx & amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__MAPPING_ERROR_MASK) >> amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__MAPPING_ERROR__SHIFT
      permission_faults = (gfx & amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__PERMISSION_FAULTS_MASK) >> amdgpu_gc_11_0_0.GCVM_L2_PROTECTION_FAULT_STATUS__PERMISSION_FAULTS__SHIFT
      raise RuntimeError(f"GFX FAULT: {client_mappings[cid]} {addr=:X}: {more_faults=}, {rw=}, {vmid=}, {mapping_error=} {permission_faults=}")

    mmhub = self.adev.regMMVM_L2_PROTECTION_FAULT_STATUS.read()
    return gfx, mmhub