import time, mmap, sys, shutil, os, glob
from tinygrad.helpers import to_mv
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.autogen import libc
from tinygrad.helpers import to_mv, mv_address, getenv, round_up, DEBUG, temp, colored, ansilen
from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0, smu_v13_0_0
from tinygrad.runtime.support.allocator import TLSFAllocator
from tinygrad.runtime.support.am.ip import AM_SOC21, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA

AM_VERSION = 0xA0000002

def bold(s): return f"\033[1m{s}\033[0m"

def color_temp(temp):
  if temp >= 87: return colored(f"{temp:>4}", "red")
  elif temp >= 80: return colored(f"{temp:>4}", "yellow")
  return colored(f"{temp:>4}", "white")

def color_voltage(voltage): return colored(f"{voltage/1000:>5.3f}V", "cyan")

def draw_bar(percentage, width=40, fill='█', empty='░'):
  filled_width = int(width * percentage)
  bar = fill * filled_width + empty * (width - filled_width)
  return f'[{bar}] {percentage*100:.1f}%'

def same_line(strs:list[list[str]], split=8) -> list[str]:
  ret = []
  max_width_in_block = [max(ansilen(line) for line in block) for block in strs]
  max_height = max(len(block) for block in strs)
  for i in range(max_height):
    line = []
    for bid, block in enumerate(strs):
      if i < len(block): line.append(block[i] + ' ' * (split + max_width_in_block[bid] - ansilen(block[i])))
      else: line.append(' ' * (split + max_width_in_block[bid]))
    ret.append(' '.join(line))
  return ret

class AMSMI(AMDev):
  def __init__(self, pcibus):
    self.pcibus = pcibus
    self.bar_fds = {bar: os.open(f"/sys/bus/pci/devices/{self.pcibus}/resource{bar}", os.O_RDWR | os.O_SYNC) for bar in [0, 2, 5]}
    self.bar_size = {0: (32 << 30), 2: os.fstat(self.bar_fds[2]).st_size, 5: os.fstat(self.bar_fds[5]).st_size}

    assert self.bar_size[0] > (1 << 30), "Large BAR is not enabled"

    self.vram = self._map_pci_range(0)
    self.mmio = self._map_pci_range(5).cast('I')

    self._run_discovery()
    self._build_regs()

    if self.reg("regSCRATCH_REG7").read() != AM_VERSION:
      raise Exception(f"Unsupported AM version: {self.reg('regSCRATCH_REG7').read():x}")

    self.is_booting, self.smi_dev = True, True
    self.partial_boot = True # do not init anything
    self.mm = AMMemoryManager(self, self.vram_size)

    # Initialize IP blocks
    self.soc21:AM_SOC21 = AM_SOC21(self)
    self.gmc:AM_GMC = AM_GMC(self)
    self.ih:AM_IH = AM_IH(self)
    self.psp:AM_PSP = AM_PSP(self)
    self.smu:AM_SMU = AM_SMU(self)

  def _map_pci_range(self, bar, off=0, addr=0, size=None):
    fd, sz = self.bar_fds[bar], self.bar_size[bar]
    return to_mv(libc.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), fd, off), sz)

class SMICtx:
  def __init__(self):
    self.devs = []
    self.opened_pcidevs = []
    self.prev_lines_cnt = 0

  def _open_am_device(self, pcibus):
    try:
      self.devs.append(AMSMI(pcibus))
    except Exception as e:
      if DEBUG >= 2: print(f"Failed to open AM device {pcibus}: {e}")

    self.opened_pcidevs.append(pcibus)
    if DEBUG >= 2: print(f"Opened AM device {pcibus}")

  def rescan_devs(self):
    pattern = os.path.join('/tmp', 'am_*.lock')
    for d in [f[8:-5] for f in glob.glob(pattern)]: 
      if d not in self.opened_pcidevs:
        self._open_am_device(d)

    for d in self.devs:
      if d.reg("regSCRATCH_REG7").read() != AM_VERSION:
        self.devs.remove(d)
        self.opened_pcidevs.remove(d.pcibus)
        if DEBUG >= 2: print(f"Removed AM device {d.pcibus}")

  def collect(self): return {d: d.smu.read_metrics() for d in self.devs}

  def draw(self):
    terminal_width, _ = shutil.get_terminal_size()

    dev_metrics = self.collect()
    dev_content = []
    for dev, metrics in dev_metrics.items():
      device_line = [f"PCIe device: {bold(dev.pcibus)}"] + [""]
      activity_line = [f"GFX Activity  {draw_bar(metrics.SmuMetrics.AverageGfxActivity / 100, 50)}"] \
                    + [f"UCLK Activity {draw_bar(metrics.SmuMetrics.AverageUclkActivity / 100, 50)}"] + [""]

      # draw_metrics_table(metrics, dev)
      temps_keys = [(k, name) for k, name in smu_v13_0_0.c__EA_TEMP_e__enumvalues.items() if k < smu_v13_0_0.TEMP_COUNT]
      temps_table = ["=== Temps (C) ==="] + [f"{name:<15}: {color_temp(metrics.SmuMetrics.AvgTemperature[k])}" for k, name in temps_keys]

      voltage_keys = [(k, name) for k, name in smu_v13_0_0.c__EA_SVI_PLANE_e__enumvalues.items() if k < smu_v13_0_0.SVI_PLANE_COUNT]
      voltages_table = ["=== Voltages ==="] + [f"{name:<24}: {color_voltage(metrics.SmuMetrics.AvgVoltage[k])}" for k, name in voltage_keys] \
                    + ["", "=== Power ==="] \
                    + [f"AVG Socket Power: {metrics.SmuMetrics.AverageTotalBoardPower}W / {metrics.SmuMetrics.dGPU_W_MAX}W"] \
                    + [draw_bar(metrics.SmuMetrics.AverageSocketPower / metrics.SmuMetrics.dGPU_W_MAX, 30)] \

      frequency_table = ["=== Frequencies ===",
        f"GFXCLK Target : {metrics.SmuMetrics.AverageGfxclkFrequencyTarget} MHz",
        f"GFXCLK PreDs  : {metrics.SmuMetrics.AverageGfxclkFrequencyPreDs} MHz",
        f"GFXCLK PostDs : {metrics.SmuMetrics.AverageGfxclkFrequencyPostDs} MHz",
        f"FCLK PreDs    : {metrics.SmuMetrics.AverageFclkFrequencyPreDs} MHz",
        f"FCLK PostDs   : {metrics.SmuMetrics.AverageFclkFrequencyPostDs} MHz",
        f"MCLK PreDs    : {metrics.SmuMetrics.AverageMemclkFrequencyPreDs} MHz",
        f"MCLK PostDs   : {metrics.SmuMetrics.AverageMemclkFrequencyPostDs} MHz",
        f"VCLK0         : {metrics.SmuMetrics.AverageVclk0Frequency} MHz",
        f"DCLK0         : {metrics.SmuMetrics.AverageDclk0Frequency} MHz",
        f"VCLK1         : {metrics.SmuMetrics.AverageVclk1Frequency} MHz",
        f"DCLK1         : {metrics.SmuMetrics.AverageDclk1Frequency} MHz"]

      dev_content.append(device_line + activity_line + same_line([temps_table, voltages_table, frequency_table]))

    raw_text = 'AM Monitor'.center(terminal_width) + "\n" + "=" * terminal_width + "\n\n"
    for i in range(0, len(dev_content), 2):
      if i + 1 < len(dev_content): raw_text += '\n'.join(same_line([dev_content[i], dev_content[i+1]]))
      else: raw_text += '\n'.join(dev_content[i])
      if i + 2 < len(dev_content): raw_text += "\n" + "=" * terminal_width + "\n\n"

    sys.stdout.write(f'\033[{self.prev_lines_cnt}A')
    sys.stdout.flush()
    print(raw_text)

    self.prev_lines_cnt = len(raw_text.splitlines()) + 2

if __name__ == "__main__":
  try:
    os.system('clear')
    smi_ctx = SMICtx()
    while True:
      smi_ctx.rescan_devs()
      smi_ctx.draw()
      time.sleep(1) # Update every second
  except KeyboardInterrupt: print("\nExiting...")
