import time, mmap
import shutil
import os
import glob
from tinygrad.helpers import to_mv
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.autogen import libc
from tinygrad.helpers import to_mv, mv_address, getenv, round_up, DEBUG, temp, colored
from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0, smu_v13_0_0
from tinygrad.runtime.support.allocator import TLSFAllocator
from tinygrad.runtime.support.am.ip import AM_SOC21, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA

def color_temp(temp):
  if temp >= 87: return colored(f"{temp:>4}", "red")
  elif temp >= 80: return colored(f"{temp:>4}", "yellow")
  return colored(f"{temp:>4}", "white")

def color_voltage(voltage): return colored(f"{voltage/1000:>5.3f}V", "cyan")

COLORS = {
  'red': '\033[91m',
  'green': '\033[92m',
  'yellow': '\033[93m',
  'blue': '\033[94m',
  'reset': '\033[0m'
}

from dataclasses import dataclass
from typing import List, Any, Callable
from enum import Enum

def create_bar(value, max_val=100, width=15):
   filled = int(width * value / max_val)
   return f"{'█' * filled}{'░' * (width - filled)}"

# def draw_bar(percentage, width=40, fill='█', empty='░', color='green'):
#   filled_width = int(width * percentage / 100)
#   bar = fill * filled_width + empty * (width - filled_width)
#   return f'{COLORS[color]}[{bar}]{COLORS["reset"]} {percentage:.1f}%'

def clear_screen():
  os.system('cls' if os.name == 'nt' else 'clear')

def draw_bar(percentage, width=40, fill='█', empty='░'):
  filled_width = int(width * percentage / 100)
  bar = fill * filled_width + empty * (width - filled_width)
  return f'[{bar}] {percentage:.1f}%'

# typedef struct {
#   uint32_t CurrClock[PPCLK_COUNT];

#   uint16_t AverageGfxclkFrequencyTarget;
#   uint16_t AverageGfxclkFrequencyPreDs;
#   uint16_t AverageGfxclkFrequencyPostDs;
#   uint16_t AverageFclkFrequencyPreDs;
#   uint16_t AverageFclkFrequencyPostDs;
#   uint16_t AverageMemclkFrequencyPreDs  ; // this is scaled to actual memory clock
#   uint16_t AverageMemclkFrequencyPostDs  ; // this is scaled to actual memory clock
#   uint16_t AverageVclk0Frequency  ;
#   uint16_t AverageDclk0Frequency  ;
#   uint16_t AverageVclk1Frequency  ;
#   uint16_t AverageDclk1Frequency  ;
#   uint16_t PCIeBusy;
#   uint16_t dGPU_W_MAX;
#   uint16_t padding;

#   uint32_t MetricsCounter;

#   uint16_t AvgVoltage[SVI_PLANE_COUNT];
#   uint16_t AvgCurrent[SVI_PLANE_COUNT];

#   uint16_t AverageGfxActivity    ;
#   uint16_t AverageUclkActivity   ;
#   uint16_t Vcn0ActivityPercentage  ;
#   uint16_t Vcn1ActivityPercentage  ;

#   uint32_t EnergyAccumulator;
#   uint16_t AverageSocketPower;
#   uint16_t AverageTotalBoardPower;

#   uint16_t AvgTemperature[TEMP_COUNT];
#   uint16_t AvgTemperatureFanIntake;

#   uint8_t  PcieRate               ;
#   uint8_t  PcieWidth              ;

#   uint8_t  AvgFanPwm;
#   uint8_t  Padding[1];
#   uint16_t AvgFanRpm;


#   uint8_t ThrottlingPercentage[THROTTLER_COUNT];
#   uint8_t VmaxThrottlingPercentage;
#   uint8_t Padding1[3];

#   //metrics for D3hot entry/exit and driver ARM msgs
#   uint32_t D3HotEntryCountPerMode[D3HOT_SEQUENCE_COUNT];
#   uint32_t D3HotExitCountPerMode[D3HOT_SEQUENCE_COUNT];
#   uint32_t ArmMsgReceivedCountPerMode[D3HOT_SEQUENCE_COUNT];

#   uint16_t ApuSTAPMSmartShiftLimit;
#   uint16_t ApuSTAPMLimit;
#   uint16_t AvgApuSocketPower;

#   uint16_t AverageUclkActivity_MAX;

#   uint32_t PublicSerialNumberLower;
#   uint32_t PublicSerialNumberUpper;

# } SmuMetrics_t;

def same_line(strs:list[list[str]]) -> list[str]:
  ret = []
  max_width_in_block = max(len(line) for block in strs for line in block)
  max_height = max(len(block) for block in strs)
  for i in range(max_height):
    line = []
    for block in strs:
      if i < len(block):
        line.append(block[i].ljust(max_width_in_block))
      else:
        line.append(' ' * max_width_in_block)
    ret.append(' '.join(line))
  return ret

def draw_am_metrics(metrics_info):
  """Draw all system metrics in a formatted console UI"""
  # Get terminal size
  terminal_width, _ = shutil.get_terminal_size()

  # Create the header
  print('=' * terminal_width)
  print('AM Monitor'.center(terminal_width))
  print('=' * terminal_width)
  print()

  for dev, metrics in metrics_info.items():
    # draw_metrics_table(metrics, dev)
    temps_keys = [(k, name) for k, name in smu_v13_0_0.c__EA_TEMP_e__enumvalues.items() if k < smu_v13_0_0.TEMP_COUNT]
    temps_table = ["=== Temps (°C) ==="] + [f"{name}: {color_temp(metrics.SmuMetrics.AvgTemperature[k])}" for k, name in temps_keys]
    
    voltage_keys = [(k, name) for k, name in smu_v13_0_0.c__EA_SVI_PLANE_e__enumvalues.items() if k < smu_v13_0_0.SVI_PLANE_COUNT]
    voltages_table = ["=== Voltages ==="] + [f"{name}: {color_voltage(metrics.SmuMetrics.AvgVoltage[k])}" for k, name in voltage_keys] \
                   + ["\n=== Power ==="] \
                   + [f"AVG Socket Power: {metrics.SmuMetrics.AverageTotalBoardPower}W / {metrics.SmuMetrics.dGPU_W_MAX}W"] \
                   + [draw_bar(metrics.SmuMetrics.AverageSocketPower / metrics.SmuMetrics.dGPU_W_MAX, 30)] \
                   + [f"AVG BoardPower: {metrics.SmuMetrics.AverageSocketPower}W"] \
                   + [draw_bar(metrics.SmuMetrics.AverageSocketPower / metrics.SmuMetrics.dGPU_W_MAX, 30)] \

    tables = same_line([temps_table, voltages_table])
    print(tables)
    print('\n'.join(tables))

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

    if self.reg("regSCRATCH_REG7").read() != (am_version:=0xA0000002):
      raise Exception(f"Unsupported AM version: {am_version:x}")

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

  def collect(self): return {d: d.smu.read_metrics() for d in self.devs}

# def find_am_lock_files(path='/tmp'):
#   pattern = os.path.join(path, 'am_*.lock')
#   return glob.glob(pattern)

# opened_devs = []
# def update_device_list():
#   files = find_am_lock_files()
#   devs = [f[8:-5] for f in files]
  

# def update_smu_info():
#   files = find_am_lock_files()
#   print([f[8:-5] for f in files])

def main():
  smi_ctx = SMICtx()
  while True:
    smi_ctx.rescan_devs()
    clear_screen()
    draw_am_metrics(smi_ctx.collect())
    time.sleep(2)  # Update every second

if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\nExiting...")
