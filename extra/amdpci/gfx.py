import os, ctypes, time
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0_offset
from tinygrad.helpers import to_mv, mv_address

class GFX_IP:
  # ((SH_MEM_ADDRESS_MODE_64 << SH_MEM_CONFIG__ADDRESS_MODE__SHIFT) | \
	#  (SH_MEM_ALIGNMENT_MODE_UNALIGNED << SH_MEM_CONFIG__ALIGNMENT_MODE__SHIFT) | \
	#  (3 << SH_MEM_CONFIG__INITIAL_INST_PREFETCH__SHIFT))
  DEFAULT_SH_MEM_CONFIG = 0xc00c

  def __init__(self, adev):
    self.adev = adev
    self.setup()
    pass

  def wait_for_rlc_autoload(self):
    while True:
      cp_status = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regCP_STAT, amdgpu_gc_11_0_0_offset.regCP_STAT_BASE_IDX)

      # TODO: some exceptions here for other gpus
      bootload_status = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regRLC_RLCS_BOOTLOAD_STATUS, amdgpu_gc_11_0_0_offset.regRLC_RLCS_BOOTLOAD_STATUS_BASE_IDX)

      print("cp_status", hex(cp_status), "bootload_status", hex(bootload_status))
      RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK = 0x80000000
      if cp_status == 0 and ((bootload_status & RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) == RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK):
        print("rlc_autoload_complete")
        break

  def gb_addr_config(self):
    return self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regGB_ADDR_CONFIG, amdgpu_gc_11_0_0_offset.regGB_ADDR_CONFIG_BASE_IDX)

  def init_golden_registers(self):
    val = self.adev.rreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regTCP_CNTL, amdgpu_gc_11_0_0_offset.regTCP_CNTL_BASE_IDX)
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regTCP_CNTL, amdgpu_gc_11_0_0_offset.regTCP_CNTL_BASE_IDX, val | 0x20000000)

  def init_compute_vmid(self):
    pass

  def constants_init(self):
    # WREG32_FIELD15_PREREG(GC, 0, GRBM_CNTL, READ_TIMEOUT, 0xff);
    self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regGRBM_CNTL, amdgpu_gc_11_0_0_offset.regGRBM_CNTL_BASE_IDX, 0xff)

    # TODO: Read configs here
    for i in range(16):
      self.adev.soc21_grbm_select(0, 0, 0, i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regSH_MEM_CONFIG, amdgpu_gc_11_0_0_offset.regSH_MEM_CONFIG_BASE_IDX, self.DEFAULT_SH_MEM_CONFIG)

      tmp = (self.adev.vmm.private_aperture_start >> 48) | ((self.adev.vmm.shared_aperture_start >> 48) << 16)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regSH_MEM_BASES, amdgpu_gc_11_0_0_offset.regSH_MEM_BASES_BASE_IDX, tmp)

      # We do not enable trap for each kfd vmid...

    self.adev.soc21_grbm_select(0, 0, 0, 0)

    for i in range(1, 16):
      # Initialize all compute VMIDs to have no GDS, GWS, or OA acccess. These should be enabled by FW for target VMIDs (?)
      # TODO: Check if we need to enable them for each VMID.
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regGDS_VMID0_BASE, amdgpu_gc_11_0_0_offset.regGDS_VMID0_BASE_BASE_IDX, 0, offset=2*i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regGDS_VMID0_SIZE, amdgpu_gc_11_0_0_offset.regGDS_VMID0_SIZE_BASE_IDX, 0, offset=2*i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regGDS_GWS_VMID0, amdgpu_gc_11_0_0_offset.regGDS_GWS_VMID0_BASE_IDX, 0, offset=i)
      self.adev.wreg_ip("GC", 0, amdgpu_gc_11_0_0_offset.regGDS_OA_VMID0, amdgpu_gc_11_0_0_offset.regGDS_OA_VMID0_BASE_IDX, 0, offset=i)

  def setup(self):
    self.wait_for_rlc_autoload()
    assert self.gb_addr_config() == 0x545 # gfx11 is the same

    self.init_golden_registers()
    self.constants_init()