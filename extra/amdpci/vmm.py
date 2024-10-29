from tinygrad.runtime.autogen import libpciaccess, amdgpu_2
from tinygrad.helpers import to_mv, mv_address, round_up

class VMM:
  def __init__(self, adev):
    self.adev = adev

    self.next_va = 0
    self.mappings = {
      'pbd0': 0x1000000,
    }

    self.pbd0_base = self.mappings['pbd0']
    self.pbd0_size = 0x10000000 # this is dumb.
    self.pbd0_cpu_addr = self.adev.vram_cpu_addr + self.pbd0_base
    self.pbd0_view = to_mv(self.pbd0_cpu_addr, self.pbd0_size)

    self.amdgpu_gmc_init_pdb0()

  # def set_pte(self, pte, addr, flags):
  #   pte[addr >> 12] = (addr & 0xfffff000) | flags

  def amdgpu_gmc_set_pte_pde(self, pdb_ptr, gpu_page_idx, addr, flags):
    value = addr & 0x0000FFFFFFFFF000
    value |= flags
    to_mv(pdb_ptr + (gpu_page_idx * 8), 8).cast('Q')[0] = value

  def amdgpu_gmc_init_pdb0(self):
    # idenity mapping for the whole vram, not sure how we can map system pages, we need dma_addresses (are they just cpu's physical addr?)
    vmid0_page_table_block_size = 0
    
    flags |= AMDGPU_PTE_MTYPE_NV10(0, MTYPE_UC) | AMDGPU_PTE_EXECUTABLE
    flags |= AMDGPU_PTE_VALID | AMDGPU_PTE_READABLE
    flags |= AMDGPU_PTE_WRITEABLE
    flags |= AMDGPU_PTE_SNOOPED
    flags |= AMDGPU_PTE_FRAG((vmid0_page_table_block_size + 9*1))
    flags |= AMDGPU_PDE_PTE_FLAG(adev)

    pde0_page_size = (1<<vmid0_page_table_block_size)<<21 # 2mb

    vram_base = 0
    vram_size = (24 << 30)
    vram_end = vram_base + vram_size

    # Each PDE0 (used as PTE) covers (2^vmid0_page_table_block_size)*2M
    for i, addr in enumerate(range(vram_base, vram_end, pde0_page_size)):
      self.amdgpu_gmc_set_pte_pde(self.pbd0_cpu_addr, i, addr, flags)

    # TODO: should we map gart PT too?


  def alloc(self, size):
    pass

  