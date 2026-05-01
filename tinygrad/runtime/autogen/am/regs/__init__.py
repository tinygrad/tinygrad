import pathlib, re, itertools
from tinygrad.runtime.autogen import load
from tinygrad.runtime.autogen.am import am_src

# 9060xt
__all__ = ["gc_12_0_0", "hdp_7_0_0", "mmhub_4_1_0", "mp_11_0", "mp_14_0_2", "nbif_6_3_1", "osssys_7_0_0"]

patterns = {
  # FIXME: CDNA4 naming
  "gc": ["GCVM", "GCMC_VM", "CP_(HQD|MQD|MEC|ME_CNTL|PERFMON|RB_WPTR_POLL_CNTL|INT_CNTL|STAT|PFP_PRGRM|ME_PRGRM|COHER_START)", "COMPUTE_",
         "(SQ|GL2C)_PERFCOUNTER", "SQ_THREAD_TRACE", "SPI_(CONFIG_CNTL|COMPUTE_QUEUE_RESET)", "GRBM", "SH_MEM", "RLC", "TCP", "GB_ADDR_CONFIG",
         "SDMA[01]_(WATCHDOG_CNTL|UTCL1_(CNTL|PAGE)|MCU_CNTL|F32_CNTL|CNTL($|_BASE_IDX$)|QUEUE0_|RLC_CGCG_CTRL)", "SCRATCH_REG[67]"],
  # FIXME: CDNA4 naming
  "mmhub": ["MMVM", "MMMC_VM", "MM_ATC_L2_MISC_CG"],
  "nbif": ["BIF_BX_PF[01]_GPU_HDP_FLUSH", "BIF_BX_PF0_RSMU", "BIF_BX0_(REMAP_HDP_MEM_FLUSH_CNTL|BIF_DOORBELL_INT_CNTL|PCIE_INDEX2|PCIE_DATA2)",
           "BIFC_(DOORBELL_ACCESS_EN_PF|GFX_INT_MONITOR_MASK)", "XCC_DOORBELL_FENCE", "DOORBELL0_CTRL_ENTRY", "GDC_S2A0_S2A_DOORBELL_ENTRY",
           "S2A_DOORBELL_ENTRY", "RCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", "RCC_DEV0_EPF2_STRAP2($|_BASE_IDX$)"],
  "mp": ["MP([01]|ASP)_SMN_C2PMSG"],
  "hdp": ["HDP_MEM_POWER_CTRL"],
  "oss": ["IH_"]
}

def __getattr__(nm):
  prefix = {"osssys": "oss"}.get(x:=nm.split("_", 1)[0], x)
  def genreg(name, files, patterns=[], **kwargs):
    def extract(lines, pat): return ((m.group(1), int(m.group(2), 0)) for l in lines if (m:=re.match(pat, l)))
    def clean_name(name): return name[next((i for i,c in enumerate(name) if c.isupper()), len(name)):]

    offset, sh_mask = pathlib.Path(f"{files[0]}_offset.h").read_text().splitlines(), pathlib.Path(f"{files[0]}_sh_mask.h").read_text().splitlines()
    defs = {k:v for k,v in extract(offset, r'#define\s+((?:mm|reg)\S+)\s+(0x[\da-fA-F]+|\d+)') if any(re.match("(mm|reg)" + p, k) for p in patterns)}
    fields = {reg: {name.split('__')[1].lower(): ((mask & -mask).bit_length() - 1, mask.bit_length() - 1) for name, mask in fields}
              for reg, fields in itertools.groupby(extract(sh_mask, r'#define\s+(\S+)_MASK\s+(0x[\da-fA-F]+|\d+)'), lambda x: x[0].split('__')[0])}

    regs = {reg: (off, defs[f"{reg}_BASE_IDX"], fields.get(clean_name(reg), {})) for reg,off in defs.items() if not reg.endswith("_BASE_IDX") and f"{reg}_BASE_IDX" in defs}
    print(f"defined {len(regs)} registers for {name}")
    return "\n".join(["regs = {"] + [f"{k!r}: {v!r}," for k,v in regs.items()] + ["}"])

  assert prefix in patterns, f"Failed to load ASIC registers for {prefix.upper()}"
  return load(f"am/regs/{nm}", [f"../amdgpu-33970e1351f5e511029602454979f3de7e22260f/drivers/gpu/drm/amd/include/asic_reg/{prefix}/{nm}"],
                           patterns=patterns[prefix], gen=genreg)
