#!/usr/bin/env python3
"""
every AM card's power state in one line: the AM boot flags (SCRATCH_REG7/6), the SMU's running features (named bits off that should be on),
and the metrics table's socket power vs total board power, clocks, rail voltages and currents, fan, pcie, temperatures.
read only (smu queries: GetRunningSmuFeatures, TransferTableSmu2Dram when an AM session set the table address). run with NO AM process
on the box (the smu mailbox is not shared between processes). usage: sudo PYTHONPATH=. python3 extra/amdpci/am_state.py [pci bus]
"""
import sys, os, glob, subprocess
from tinygrad.runtime.support.am.amdev import AMDev
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from am_smi import AMSMI, SMICtx  # noqa: E402

BITS = ("DPM_GFXCLK", "DPM_UCLK", "DPM_FCLK", "DPM_SOCCLK", "DPM_LINK", "VMEMP_SCALING", "VDDIO_MEM_SCALING", "DS_GFXCLK", "DS_SOCCLK", "DS_FCLK",
        "DS_LCLK", "DS_UCLK", "GFX_ULV", "FW_DSTATE", "GFXOFF", "BACO", "SOC_MPCLK_DS", "THROTTLERS", "FW_CTF", "FAN_CONTROL", "GFX_DCS",
        "SOC_CG", "DF_CSTATE", "ATHUB_MMHUB_PG", "MEM_TEMP_READ", "BACO_CG")

class AMState(AMSMI):
  def _init_from_d0(self):
    self._run_discovery()
    self._build_regs()
    self.am_version, self.am_err = self.reg("regSCRATCH_REG7").read(), self.reg("regSCRATCH_REG6").read()
    self.is_booting = True
    self.init_sw(smi_dev=True)
    self.partial_boot = True

def main():
  want = sys.argv[1] if len(sys.argv) > 1 else None
  ctx = SMICtx(dev_filter=want)
  buses = sorted({f[8:-5] for f in glob.glob("/tmp/am_*.lock") if "*" not in f and not f[8:].startswith("usb")} |
                 {l.split()[0] if l.split()[0].count(":") == 2 else "0000:" + l.split()[0]
                  for l in subprocess.check_output(["lspci", "-d", "1002:", "-n"]).decode().splitlines()
                  if any(f" {k}: " in l for k in ("0300", "0380", "1200"))})
  for bus in buses:
    if want is not None and not bus.endswith(want): continue
    if bus not in ctx.opened_pci_resources:
      ctx._open_am_device(bus)  # (as am_smi: its BARs mapped; AMSMI raises on a card AM didn't boot: AMState below doesn't)
    if bus not in ctx.opened_pci_resources:
      print(f"{bus}: not opened")
      continue
    try: d = AMState(bus, *ctx.opened_pci_resources[bus])
    except Exception as e:
      print(f"{bus}: {type(e).__name__}: {e}")
      continue
    line = f"{bus} pci={d.pci_state}"
    if d.pci_state != "D0":
      print(line)
      continue
    m = d.smu.smu_mod
    booted = d.am_version & ~(3 << 16) == AMDev.Version
    line += f" am_ver={d.am_version:#x}{'' if booted else '(not AM)'} am_err={d.am_err}"
    if not booted:
      # (no smu message to a card no AM booted: its firmware's driver mailbox may not run, a query times out; this stays a pure read)
      print(line + f" mp1_resp={d.mmMP1_SMN_C2PMSG_90.read():#x} mp1_msg={d.mmMP1_SMN_C2PMSG_66.read():#x}", flush=True)
      continue
    try:
      lo = d.smu._send_msg(m.PPSMC_MSG_GetRunningSmuFeaturesLow, 0, read_back_arg=True, timeout=1000)
      hi = d.smu._send_msg(m.PPSMC_MSG_GetRunningSmuFeaturesHigh, 0, read_back_arg=True, timeout=1000)
      feats = (hi << 32) | lo
      off = [b for b in BITS if hasattr(m, f"FEATURE_{b}_BIT") and not feats >> getattr(m, f"FEATURE_{b}_BIT") & 1]
      line += f" features={feats:#018x} off={','.join(off) or '-'}"
    except Exception as e: line += f" features=? ({type(e).__name__})"
    if booted:
      try:
        t = d.smu.read_table(m.SmuMetricsExternal_t, getattr(m, "TABLE_SMU_METRICS", getattr(m, "SMU_TABLE_SMU_METRICS", 0))).SmuMetrics
        line += (f" socketW={t.AverageSocketPower} boardW={t.AverageTotalBoardPower} gfx={t.AverageGfxclkFrequencyPreDs}/"
                 f"{t.AverageGfxclkFrequencyPostDs} uclk={t.AverageMemclkFrequencyPreDs}/{t.AverageMemclkFrequencyPostDs} "
                 f"fclk={t.AverageFclkFrequencyPreDs}/{t.AverageFclkFrequencyPostDs} gfx%={t.AverageGfxActivity} "
                 f"V={list(t.AvgVoltage)} A={list(t.AvgCurrent)} fan={t.AvgFanRpm}rpm/{t.AvgFanPwm}% pcie={t.PcieRate}x{t.PcieWidth} "
                 f"T={list(t.AvgTemperature)[:6]} d3hot={list(t.D3HotEntryCountPerMode)}")
      except Exception as e: line += f" metrics=? ({type(e).__name__}: {e})"
    print(line, flush=True)

if __name__ == "__main__": main()
