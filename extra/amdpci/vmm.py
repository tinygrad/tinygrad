from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0
from tinygrad.helpers import to_mv, mv_address, round_up

class VMM:
  def __init__(self, adev):
    self.adev = adev

    self.mappings = []
    self.next_va = 0x1000000

    self.pdb0_size = 0x1000
    self.pdb0_base = self.alloc_vram(self.pdb0_size, "pdb0")
    self.pdb0_cpu_addr = self.adev.vram_cpu_addr + self.pdb0_base
    self.pdb0_view = to_mv(self.pdb0_cpu_addr, self.pdb0_size)

    self.memscratch_size = 0x1000
    self.memscratch_base = self.alloc_vram(self.memscratch_size, "memscratch")
    self.dummy_page_addr = self.alloc_vram(0x1000, "dummy_page")

    self.shared_aperture_start = 0x2000000000000000
    self.shared_aperture_end = self.shared_aperture_start + (4 << 30) - 1
    self.private_aperture_start = 0x1000000000000000
    self.private_aperture_end = self.private_aperture_start + (4 << 30) - 1

    regGCVM_CONTEXT0_CNTL = 0x28e8
    print(hex(self.adev.rreg(regGCVM_CONTEXT0_CNTL)))

    self.amdgpu_gmc_init_pdb0()

  def amdgpu_gmc_set_pte_pde(self, pdb_ptr, gpu_page_idx, addr, flags):
    value = addr & 0x0000FFFFFFFFF000
    value |= flags
    to_mv(pdb_ptr + (gpu_page_idx * 8), 8).cast('Q')[0] = value

  def alloc_vram(self, size, tag=None, align=0x1000):
    addr = round_up(self.next_va, align)
    self.next_va = addr + size
    self.mappings.append((addr, size, tag))
    return addr

  def vram_to_cpu_addr(self, addr, size=0): return self.adev.vram_cpu_addr + addr
  def vram_to_cpu_mv(self, addr, size): return to_mv(self.adev.vram_cpu_addr + addr, size)

  def collect_pfs(self):
    gfx = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_STATUS, 0)
    mmhub = self.adev.rreg_ip("MMHUB", 0, 0x070c, 0) ## MMVM_L2_PROTECTION_FAULT_STATUS

    # self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_STATUS, 0, 0)
    # self.adev.wreg_ip("MMHUB", 0, 0x070c, 0, 0)
    # mmhub = self.adev.rreg_ip("MMHUB", 0, 0x070c, 0) ## MMVM_L2_PROTECTION_FAULT_STATUS
    return gfx, mmhub

  def amdgpu_gmc_init_pdb0(self):
    # idenity mapping for the whole vram, not sure how we can map system pages, we need dma_addresses (are they just cpu's physical addr?)
    vmid0_page_table_block_size = 0
    
    MTYPE_UC = 3
    flags = amdgpu_2.AMDGPU_PTE_MTYPE_NV10(0, MTYPE_UC) | amdgpu_2.AMDGPU_PTE_EXECUTABLE
    flags |= amdgpu_2.AMDGPU_PTE_VALID | amdgpu_2.AMDGPU_PTE_READABLE
    flags |= amdgpu_2.AMDGPU_PTE_WRITEABLE
    flags |= amdgpu_2.AMDGPU_PTE_SNOOPED
    flags |= amdgpu_2.AMDGPU_PTE_FRAG((vmid0_page_table_block_size + 9*1))
    flags |= amdgpu_2.AMDGPU_PDE_PTE

    # pde0_page_size = (1<<vmid0_page_table_block_size)<<21 # 2mb
    pde0_page_size = 1 << 30 # 1gb

    vram_base = 0
    # vram_size = 512 << 20
    # vram_end = vram_base + vram_size

    # Each PDE0 (used as PTE) covers (2^vmid0_page_table_block_size)*2M
    for i in range(0, 512):
      addr = vram_base + i * pde0_page_size
      self.amdgpu_gmc_set_pte_pde(self.pdb0_cpu_addr, i, addr, flags)

  def flush_hdp(self): self.adev.wreg(0x1fc00, 0x0)
  def flush_tlb(self, vmid, vmhub, flush_type):
    assert vmid == 0 and vmhub == 0 and flush_type == 0

    self.adev.wreg(0x291c, 0xf80001)
    while self.adev.rreg(0x292e) != 1: pass

  def gfxhub_v3_0_init_gart_aperture_regs(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, 0)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, (1 << 30) - 1)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, 0)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0, self.adev.vmm.pdb0_base & 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.adev.vmm.pdb0_base >> 32) & 0xffffffff)

  def gfxhub_v3_0_init_system_aperture_regs(self):
    # disabled
    agp_start = 0xffffffffffff
    agp_end = 0

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BASE, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BASE_BASE_IDX, 0)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BOT, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BOT_BASE_IDX, agp_start >> 24)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_AGP_TOP, amdgpu_gc_11_0_0.regGCMC_VM_AGP_TOP_BASE_IDX, agp_end >> 24)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX, agp_start >> 18)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX, agp_end >> 18)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX, self.memscratch_base >> 12)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX, (self.memscratch_base >> 44))

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32, 0, self.dummy_page_addr >> 12)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32, 0, (self.dummy_page_addr >> 44))

  def gfxhub_v3_0_init_tlb_regs(self):
    # TODO: write up
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_MX_L1_TLB_CNTL, amdgpu_gc_11_0_0.regGCMC_VM_MX_L1_TLB_CNTL_BASE_IDX, 0x1859)

  def gfxhub_v3_0_init_cache_regs(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CNTL, amdgpu_gc_11_0_0.regGCVM_L2_CNTL_BASE_IDX, 0x80e01)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CNTL2, amdgpu_gc_11_0_0.regGCVM_L2_CNTL2_BASE_IDX, 0x3)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CNTL3, amdgpu_gc_11_0_0.regGCVM_L2_CNTL3_BASE_IDX, 0x80130009)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CNTL4, amdgpu_gc_11_0_0.regGCVM_L2_CNTL4_BASE_IDX, 0x1)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CNTL5, amdgpu_gc_11_0_0.regGCVM_L2_CNTL5_BASE_IDX, 0x3fe0)

  def gfxhub_v3_0_enable_system_domain(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_CNTL, 0, 0x1fffe05)

  def gfxhub_v3_0_program_invalidation(self):
    eng_addr_distance = 2

    for i in range(18):
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + eng_addr_distance * i, amdgpu_gc_11_0_0.regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32_BASE_IDX, 0xffffffff)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 + eng_addr_distance * i, amdgpu_gc_11_0_0.regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32_BASE_IDX, 0x1f)

  def init_gfxhub(self):
    self.gfxhub_v3_0_init_gart_aperture_regs()
    self.gfxhub_v3_0_init_system_aperture_regs()
    self.gfxhub_v3_0_init_tlb_regs()
    self.gfxhub_v3_0_init_cache_regs()

    self.gfxhub_v3_0_enable_system_domain()
    # // gfxhub_v3_0_disable_identity_aperture(adev);
	  # // gfxhub_v3_0_setup_vmid_config(adev);
    self.gfxhub_v3_0_program_invalidation()
