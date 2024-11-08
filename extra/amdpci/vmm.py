import array
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, amdgpu_mmhub_3_0_0
from tinygrad.helpers import to_mv, mv_address, round_up

class VMM:
  def __init__(self, adev):
    self.adev = adev

    self.mappings = []

    self.vm_base = 0x7F0000000000
    self.vm_end = self.vm_base + (512 * (1 << 30)) - 1

    self.next_va = self.vm_base

    self.pdb0_size = 0x1000
    self.pdb0_vaddr = self.alloc_vram(self.pdb0_size, "pdb0")
    self.pdb0_paddr = self.vaddr_to_paddr(self.pdb0_vaddr)
    self.pdb0_cpu_addr = self.paddr_to_cpu_addr(self.pdb0_paddr)
    self.pdb0_view = self.paddr_to_cpu_mv(self.pdb0_paddr, self.pdb0_size)

    self.memscratch_size = 0x1000
    self.memscratch_vaddr = self.alloc_vram(self.memscratch_size, "memscratch")
    self.memscratch_paddr = self.vaddr_to_paddr(self.memscratch_vaddr)
    self.dummy_page_vaddr = self.alloc_vram(0x1000, "dummy_page")
    self.dummy_page_paddr = self.vaddr_to_paddr(self.dummy_page_vaddr)
    self.dummy_page_mc_addr = self.paddr_to_mc(self.dummy_page_paddr)
    # self.vram_to_cpu_mv(self.memscratch_vaddr, 0x1000).cast('I')[:] = array.array('I', [0xdeadbee1 for _ in range(0x1000 // 4)])
    self.paddr_to_cpu_mv(self.dummy_page_paddr, 0x1000).cast('I')[:] = array.array('I', [0xdeadbeef for _ in range(0x1000 // 4)])

    self.shared_aperture_start = 0x2000000000000000
    self.shared_aperture_end = self.shared_aperture_start + (4 << 30) - 1
    self.private_aperture_start = 0x1000000000000000
    self.private_aperture_end = self.private_aperture_start + (4 << 30) - 1

  def init(self):
    print("VMM init")
    self.amdgpu_gmc_init_pdb0()
    self.init_mmhub()

  def amdgpu_gmc_set_pte_pde(self, pdb_ptr, gpu_page_idx, addr, flags):
    value = addr & 0x0000FFFFFFFFF000
    value |= flags
    to_mv(pdb_ptr + (gpu_page_idx * 8), 8).cast('Q')[0] = value

  def alloc_vram(self, size, tag=None, align=0x1000):
    addr = round_up(self.next_va, align)
    self.next_va = addr + size
    assert self.next_va <= (self.vm_end + 1)
    self.mappings.append((addr, size, tag))
    return addr

  def vaddr_to_paddr(self, vaddr):
    assert self.vm_base <= vaddr <= self.vm_end, hex(vaddr)
    return vaddr - self.vm_base
  def paddr_to_cpu_addr(self, addr, size=0, allow_high=False):
    assert allow_high or addr < (20 << 30), hex(addr)
    return self.adev.vram_cpu_addr + addr
  def paddr_to_cpu_mv(self, addr, size, allow_high=False): 
    assert allow_high or addr < (20 << 30), hex(addr)
    return to_mv(self.paddr_to_cpu_addr(addr, allow_high=allow_high), size)
  def paddr_to_mc(self, addr): 
    return addr + 0x0000008000000000

  def collect_pfs(self):
    gfx = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_STATUS, 0)

    if gfx != 0:
      addr = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, 0)
      addr |= (self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, 0) << 32)

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

    mmhub = self.adev.rreg_ip("MMHUB", 0, 0x070c, 0) ## MMVM_L2_PROTECTION_FAULT_STATUS
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

    pde0_page_size = 1 << 30
    vram_base = 0

    # Each PDE0 (used as PTE) covers (2^vmid0_page_table_block_size)*2M
    for i in range(0, 512):
      addr = (vram_base + i * pde0_page_size) #% (32 << 30)
      self.amdgpu_gmc_set_pte_pde(self.pdb0_cpu_addr, i, addr, flags)
    
    self.vm_config = 0x1fffe05 # 2 level, 1 gb huge pages

  def flush_hdp(self): self.adev.wreg(0x1fc00, 0x0)
  def flush_tlb(self, vmid, vmhub, flush_type):
    assert vmid == 0 and vmhub == 0 and flush_type == 0

    self.flush_hdp()

    self.adev.wreg(0x291c, 0xf80001)
    while self.adev.rreg(0x292e) != 1: pass

  def mmhub_flush_tlb(self, vmid, vmhub, flush_type):
    assert vmid == 0 and vmhub == 0 and flush_type == 0

    self.flush_hdp()

    self.adev.wreg(0x1a774, 0xf80001)
    while self.adev.rreg(0x1a786) != 1: pass

    self.adev.wreg(0x1a762, 0x0)
    while self.adev.rreg(0x1a786) != 1: pass

    self.adev.wreg(0x1a71b, 0x12104010)

  def gfxhub_v3_0_init_gart_aperture_regs(self):
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, (self.vm_base >> 12) & 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, self.vm_base >> 44)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, (self.vm_end >> 12) & 0xffffffff)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32, 0, self.vm_end >> 44)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0, (self.pdb0_paddr & 0xffffffff) | 1)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.pdb0_paddr >> 32) & 0xffffffff)

  def mmhub_v3_0_init_gart_aperture_regs(self):
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, 0, (self.vm_base >> 12) & 0xffffffff)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, 0, self.vm_base >> 44)

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, 0, (self.vm_end >> 12) & 0xffffffff)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32, 0, self.vm_end >> 44)

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, 0, (self.pdb0_paddr & 0xffffffff) | 1)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, 0, (self.pdb0_paddr >> 32) & 0xffffffff)

  def gfxhub_v3_0_init_system_aperture_regs(self):
    # disabled
    agp_start = 0xffffffffffff
    agp_end = 0

    fb_start = 0x8000000000
    fb_end = 0x85feffffff

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BASE, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BASE_BASE_IDX, 0)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BOT, amdgpu_gc_11_0_0.regGCMC_VM_AGP_BOT_BASE_IDX, agp_start >> 24)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_AGP_TOP, amdgpu_gc_11_0_0.regGCMC_VM_AGP_TOP_BASE_IDX, agp_end >> 24)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX, fb_start >> 18)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX, fb_end >> 18)

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX, self.memscratch_paddr >> 12)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB, amdgpu_gc_11_0_0.regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX, (self.memscratch_paddr >> 44))

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32, 0, self.dummy_page_mc_addr >> 12)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32, 0, (self.dummy_page_mc_addr >> 44))

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_CNTL2, 0, 0x000E0000)

  def mmhub_v3_0_init_system_aperture_regs(self):
    # disabled
    agp_start = 0xffffffffffff
    agp_end = 0

    fb_start = 0x8000000000
    fb_end = 0x85feffffff

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_AGP_BASE, amdgpu_mmhub_3_0_0.regMMMC_VM_AGP_BASE_BASE_IDX, 0)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_AGP_BOT, amdgpu_mmhub_3_0_0.regMMMC_VM_AGP_BOT_BASE_IDX, agp_start >> 24)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_AGP_TOP, amdgpu_mmhub_3_0_0.regMMMC_VM_AGP_TOP_BASE_IDX, agp_end >> 24)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX, fb_start >> 18)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX, fb_end >> 18)

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX, self.memscratch_paddr >> 12)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB, amdgpu_mmhub_3_0_0.regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX, (self.memscratch_paddr >> 44))

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32, 0, self.dummy_page_mc_addr >> 12)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32, 0, (self.dummy_page_mc_addr >> 44))

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_PROTECTION_FAULT_CNTL2, 0, 0x000E0000)

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
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_CONTEXT0_CNTL, 0, self.vm_config)

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
    
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32, 0, 0xFFFFFFFF)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32, 0, 0x0000000F)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0, 0)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0, 0)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0, 0)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0, 0)

    self.gfxhub_v3_0_program_invalidation()

    self.flush_hdp()

    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0.regGCVM_L2_PROTECTION_FAULT_CNTL, 0, 0x3FFFFFFC)

    self.flush_tlb(0, 0, 0)

  def mmhub_v3_0_init_tlb_regs(self):
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMMC_VM_MX_L1_TLB_CNTL, amdgpu_mmhub_3_0_0.regMMMC_VM_MX_L1_TLB_CNTL_BASE_IDX, 0x00001859)

  def mmhub_v3_0_init_cache_regs(self):
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL_BASE_IDX, 0x80e01)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL2, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL2_BASE_IDX, 0x3)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL3, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL3_BASE_IDX, 0x80130009)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL4, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL4_BASE_IDX, 0x1)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL5, amdgpu_mmhub_3_0_0.regMMVM_L2_CNTL5_BASE_IDX, 0x3fe0)

  def mmhub_v3_0_enable_system_domain(self):
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_CONTEXT0_CNTL, 0, self.vm_config)

  def mmhub_v3_0_program_invalidation(self):
    eng_addr_distance = 2

    for i in range(18):
      self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + eng_addr_distance * i, amdgpu_mmhub_3_0_0.regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32_BASE_IDX, 0xffffffff)
      self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 + eng_addr_distance * i, amdgpu_mmhub_3_0_0.regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32_BASE_IDX, 0x1f)
  
  def init_mmhub(self):
    self.mmhub_v3_0_init_gart_aperture_regs()
    self.mmhub_v3_0_init_system_aperture_regs()
    self.mmhub_v3_0_init_tlb_regs()
    self.mmhub_v3_0_init_cache_regs()

    self.mmhub_v3_0_enable_system_domain()

    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32, 0, 0xFFFFFFFF)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32, 0, 0x0000000F)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0, 0)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0, 0)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0, 0)
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0, 0)
    self.mmhub_v3_0_program_invalidation()

    self.flush_hdp()

    # MMVM_L2_PROTECTION_FAULT_CNTL: 0x3FFFFFFC
    self.adev.wreg_ip("MMHUB", 0, amdgpu_mmhub_3_0_0.regMMVM_L2_PROTECTION_FAULT_CNTL, 0, 0x3FFFFFFC)

    self.mmhub_flush_tlb(0, 0, 0)
