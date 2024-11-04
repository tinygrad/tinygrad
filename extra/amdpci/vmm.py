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

  def alloc_vram(self, size, tag=None):
    size = round_up(size, 4096)
    addr = self.next_va
    self.next_va += size
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
