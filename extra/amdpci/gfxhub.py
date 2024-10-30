
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x2953
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x2954
def gfxhub_v3_0_setup_vm_pt_regs(adev, pt_base, vmid):
  assert vmid == 0, "not suppoer any other"
  adev.wreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, pt_base & 0xffffffff)
  adev.wreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, (pt_base >> 32) & 0xffffffff)

regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x2973
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x2974
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x2993
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x2994
def gfxhub_v3_0_init_gart_aperture_regs(adev):
  gfxhub_v3_0_setup_vm_pt_regs(adev.vmm.pdb0_base, 0)
  adev.wreg(regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, gart_base & 0xffffffff)
  adev.wreg(regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, (gart_base >> 32) & 0xffffffff)
  adev.wreg(regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, (gart_base + gart_size) & 0xffffffff)
  adev.wreg(regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32, ((gart_base + gart_size) >> 32) & 0xffffffff)

agp_start = (0xffffff << 24)
agp_end = (0x0 << 24)
fb_start = (0x200000 << 18)
fb_end = (0x217fbf << 18)

regGCMC_VM_AGP_BASE = 0x28dc
regGCMC_VM_AGP_BOT = 0x28db
regGCMC_VM_AGP_TOP = 0x28da
regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x28dd
regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x28de
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x2808
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x2809
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x282b
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x282c
regGCVM_L2_PROTECTION_FAULT_CNTL2 = 0x2825
def gfxhub_v3_0_init_system_aperture_regs(adev):
  adev.wreg(regGCMC_VM_AGP_BASE, 0)
  adev.wreg(regGCMC_VM_AGP_BOT, agp_start >> 24)
  adev.wreg(regGCMC_VM_AGP_TOP, agp_end >> 24)
  adev.wreg(regGCMC_VM_AGP_BOT, min(fb_start, agp_start) >> 18)
  adev.wreg(regGCMC_VM_AGP_TOP, min(fb_end, agp_end) >> 18)
  adev.wreg(regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB, 0x5feaff) # gfxhub_v3_0_init_system_aperture_regs:178:((adev->reg_offset[GC_HWIP][0][0] + 0x15a8))
  adev.wreg(regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB, 0x0) # gfxhub_v3_0_init_system_aperture_regs:180:((adev->reg_offset[GC_HWIP][0][0] + 0x15a9))
  adev.wreg(regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32, 0x3008e) # gfxhub_v3_0_init_system_aperture_regs:184:((adev->reg_offset[GC_HWIP][0][0] + 0x15cb))
  adev.wreg(regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32, 0x0) # gfxhub_v3_0_init_system_aperture_regs:186:((adev->reg_offset[GC_HWIP][0][0] + 0x15cc))
  val = adev.rreg(regGCVM_L2_PROTECTION_FAULT_CNTL2) # gfxhub_v3_0_init_system_aperture_regs:189:(adev->reg_offset[GC_HWIP][0][0] + 0x15c5)
  adev.wreg(regGCVM_L2_PROTECTION_FAULT_CNTL2, 0x60000) # gfxhub_v3_0_init_system_aperture_regs:189:(adev->reg_offset[GC_HWIP][0][0] + 0x15c5)


regGCMC_VM_MX_L1_TLB_CNTL = 0x28df
def gfxhub_v3_0_init_tlb_regs(adev):
  # tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL, ENABLE_L1_TLB, 1);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL, SYSTEM_ACCESS_MODE, 3);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL,
	# 		    ENABLE_ADVANCED_DRIVER_MODEL, 1);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL,
	# 		    SYSTEM_APERTURE_UNMAPPED_ACCESS, 0);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL, ECO_BITS, 0);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL,
	# 		    MTYPE, MTYPE_UC); /* UC, uncached */
  adev.wreg(regGCMC_VM_MX_L1_TLB_CNTL, 0x1859)

def gfxhub_v3_0_init_cache_regs(adev):
  # TODO: enable cache
  pass

regGCVM_CONTEXT0_CNTL = 0x28e8
def gfxhub_v3_0_enable_system_domain(adev):
  # ENABLE_CONTEXT=1, PAGE_TABLE_DEPTH=0, RETRY_PERMISSION_OR_INVALID_PAGE_FAULT=0
  adev.wreg(regGCVM_CONTEXT0_CNTL, 0x1fffe01)

regGCVM_CONTEXT1_CNTL = 0x28e9
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x2975
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 = 0x2976
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x2995
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 = 0x2996
def gfxhub_v3_0_setup_vmid_config(adev):
  ctx_distance = 1
  ctx_addr_distance = 2

  for vmid in range(15):
    adev.wreg(regGCVM_CONTEXT1_CNTL + ctx_distance * vmid, 0x1fffe07) # gfxhub_v3_0_setup_vmid_config:332:((adev->reg_offset[GC_HWIP][0][0] + 0x1689) + i * hub->ctx_distance)
    adev.wreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 + ctx_addr_distance * vmid, 0x0) # gfxhub_v3_0_setup_vmid_config:334:((adev->reg_offset[GC_HWIP][0][0] + 0x1715) + i * hub->ctx_addr_distance)
    adev.wreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 + ctx_addr_distance * vmid, 0x0) # gfxhub_v3_0_setup_vmid_config:336:((adev->reg_offset[GC_HWIP][0][0] + 0x1716) + i * hub->ctx_addr_distance)
    adev.wreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 + ctx_addr_distance * vmid, 0xffffffff) # gfxhub_v3_0_setup_vmid_config:338:((adev->reg_offset[GC_HWIP][0][0] + 0x1735) + i * hub->ctx_addr_distance)
    adev.wreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 + ctx_addr_distance * vmid, 0xf) # gfxhub_v3_0_setup_vmid_config:341:((adev->reg_offset[GC_HWIP][0][0] + 0x1736) + i * hub->ctx_addr_distance)

regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x292f
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x2930
def gfxhub_v3_0_program_invalidation(adev):
  eng_addr_distance = 2

  for i in range(18):
    adev.wreg(regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + eng_addr_distance * i, 0xffffffff)
    adev.wreg(regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 + eng_addr_distance * i, 0x1f)

def gfxhub_v3_0_set_fault_enable_default(adev):
  val = adev.rreg(0x307f) # gfxhub_v3_0_set_fault_enable_default:425:(adev->reg_offset[GC_HWIP][0][0] + 0x1e1f)
  adev.wreg(0x307f, 0x408000) # gfxhub_v3_0_set_fault_enable_default:427:((adev->reg_offset[GC_HWIP][0][0] + 0x1e1f))
  val = adev.rreg(0x2824) # gfxhub_v3_0_set_fault_enable_default:435:(adev->reg_offset[GC_HWIP][0][0] + 0x15c4)
  # assert val == 0x3ffffffc
  adev.wreg(0x2824, 0x3ffffffc) # gfxhub_v3_0_set_fault_enable_default:465:((adev->reg_offset[GC_HWIP][0][0] + 0x15c4))
  adev.wreg(0x1fc00, 0x0) # hdp_v6_0_flush_hdp:38:((adev->rmmio_remap.reg_offset + KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL) >> 2)
  gmc_v11_0_flush_gpu_tlb()

def gfxhub_v3_0_gart_enable(adev):
  gfxhub_v3_0_init_gart_aperture_regs(adev)
  gfxhub_v3_0_init_system_aperture_regs(adev)
  gfxhub_v3_0_init_tlb_regs(adev)
  gfxhub_v3_0_init_cache_regs(adev)

  gfxhub_v3_0_enable_system_domain(adev)
  # gfxhub_v3_0_disable_identity_aperture()
  gfxhub_v3_0_setup_vmid_config(adev)
  gfxhub_v3_0_program_invalidation(adev)
  gfxhub_v3_0_set_fault_enable_default(adev)
  print("done gfxhub_v3_0_gart_enable")
