import pathlib, re, ctypes, mmap, collections, functools, copy
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import from_mv, mv_address
from test.mockgpu.driver import PCIDesc, PCIRegion, VirtDriver, VirtFileDesc, TextFileDesc, DirFileDesc, VirtFile
from test.mockgpu.amd.amdgpu import AMDGPU, gpu_props
from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0
from tinygrad.runtime.support.am.amdev import AMRegister

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

class MMEmulator:
  def __init__(self, adev):
    self.adev = adev
  def flush_tlb(self): pass

class AMDevice:
  def __init__(self, driver, vram_size):
    self.driver = driver
    self.vram_size = vram_size
    self.vram = memoryview(bytearray(vram_size))
    self.mmio = memoryview(bytearray(1 << 20))
    self.doorbells = memoryview(bytearray(2 << 20))
    self.reg_state = {}

    self.mm = MMEmulator(self)
    self._emu_boot()

  def _emu_boot(self):
    mmRCC_CONFIG_MEMSIZE = 0xde3
    self.reg_state[mmRCC_CONFIG_MEMSIZE] = self.vram_size >> 20

    # Fillup discovery info from file in this dir
    discovery_bin = pathlib.Path(__file__).parent / "discovery.bin"
    self.vram[self.vram_size-(64 << 10):self.vram_size-(54 << 10)] = discovery_bin.read_bytes()
    self._build_regs()

  def ip_base(self, ip:str, inst:int, seg:int) -> int: return self.regs_offset[am.__dict__[f"{ip}_HWIP"]][inst][seg]

  def _build_regs(self):
    self.regs_offset = {13: {0: [3072, 37784576]}, 28: {0: [93184, 37754880], 1: [201327616, 201461760], 2: [209716224, 209850368], 3: [218104832, 218238976], 4: [226493440, 226627584], 5: [234882048, 235016192], 6: [243270656, 243404800]}, 21: {0: [28672, 12582912, 37795840, 130023424, 306184192], 1: [201326592, 201463808, 201465856, 204210176, 204472320], 2: [209715200, 209852416, 209854464, 212598784, 212860928], 3: [218103808, 218241024, 218243072, 220987392, 221249536], 4: [226492416, 226629632, 226631680, 229376000, 229638144], 5: [234881024, 235018240, 235020288, 237764608, 238026752], 6: [243269632, 243406848, 243408896, 246153216, 246415360]}, 22: {0: [18, 192, 13504, 36864, 37764096]}, 1: {0: [4704, 40960, 114688, 37760000]}, 2: {0: [3872, 37790720]}, 11: {0: [70656, 38103040]}, 12: {0: [106496, 37783552]}, 15: {0: [90112, 14417920, 14680064, 14942208, 38009856]}, 16: {0: [90112, 14417920, 14680064, 14942208, 38009856]}, 14: {0: [0, 20, 3360, 66560, 37859328, 67371008]}, 26: {0: [0, 20, 3360, 66560, 37859328, 67371008]}, 23: {0: [4256, 37789696]}, 33: {0: [0, 20, 3360, 66560, 37859328, 67371008]}, 25: {0: []}, 3: {0: [4704, 40960, 114688, 37760000]}, 4: {0: [4704, 40960, 114688, 37760000]}, 24: {0: [92160, 92672, 37752832, 54788096]}, 27: {0: [91648, 37751808], 1: [201339904, 201458176], 2: [209728512, 209846784], 3: [218117120, 218235392], 4: [226505728, 226624000], 5: [234894336, 235012608], 6: [243282944, 243401216]}, 29: {0: [201342976, 201344000, 205520896, 205537280], 1: [209731584, 209732608, 213909504, 213925888], 2: [218120192, 218121216, 222298112, 222314496], 3: [226508800, 226509824, 230686720, 230703104], 4: [234897408, 234898432, 239075328, 239091712], 5: [243286016, 243287040, 247463936, 247480320]}, 17: {0: [30720, 32256], 1: [31488, 73728]}}
    self.reg_to_name = {}
    mods = [("MP0", mp_13_0_0), ("MP1", mp_11_0), ("NBIO", nbio_4_3_0), ("MMHUB", mmhub_3_0_0), ("GC", gc_11_0_0), ("OSSSYS", osssys_6_0_0)]
    for base, module in mods:
      rpref = "mm" if base == "MP1" else "reg" # MP1 regs starts with mm
      reg_names: set[str] = set(k[len(rpref):] for k in module.__dict__.keys() if k.startswith(rpref) and not k.endswith("_BASE_IDX"))
      reg_fields: dict[str, dict[str, tuple]] = collections.defaultdict(dict)
      for k, val in module.__dict__.items():
        if k.endswith("_MASK") and ((rname:=k.split("__")[0]) in reg_names):
          reg_fields[rname][k[2+len(rname):-5].lower()] = (val, module.__dict__.get(f"{k[:-5]}__SHIFT", val.bit_length() - 1))

      for k, regval in module.__dict__.items():
        if k.startswith(rpref) and not k.endswith("_BASE_IDX") and (base_idx:=getattr(module, f"{k}_BASE_IDX", None)) is not None:
          setattr(self, k, AMRegister(self, self.ip_base(base, 0, base_idx) + regval, reg_fields.get(k[len(rpref):], {})))
          self.reg_to_name[self.ip_base(base, 0, base_idx) + regval] = k

  def mmio_write(self, mv, addr, value):
    self.reg_state[addr] = value

    match addr:
      # GMC
      case self.regMMVM_INVALIDATE_ENG17_REQ.reg_off:
        self.mm.flush_tlb()
        self.reg_state[self.regMMVM_INVALIDATE_ENG17_ACK.reg_off] = 1
      case self.regGCVM_INVALIDATE_ENG17_REQ.reg_off:
        self.mm.flush_tlb()
        self.reg_state[self.regGCVM_INVALIDATE_ENG17_ACK.reg_off] = 1

      # SMU
      case self.mmMP1_SMN_C2PMSG_66.reg_off:
        # Emulate SMU answer ready.
        self.reg_state[self.mmMP1_SMN_C2PMSG_90.reg_off] = 1

      case self.regRLC_SAFE_MODE.reg_off:
        self.reg_state[self.regRLC_SAFE_MODE.reg_off] = 0 # accept command, status code 0

  def mmio_read(self, mv, addr):
    return_value = self.reg_state.get(addr, 0)

    match addr:
      case self.mmMP1_SMN_C2PMSG_90.reg_off:
        return_value = 1 # SMU answer ready
      case self.regMMVM_INVALIDATE_ENG17_ACK.reg_off:
        self.reg_state[self.regMMVM_INVALIDATE_ENG17_ACK.reg_off] = 0
      case self.regMMVM_INVALIDATE_ENG17_SEM.reg_off:
        return_value = 1 # semaphore available
    
    print("mmio read", self.reg_to_name.get(addr, f"unknown {addr}"), hex(return_value))
    return return_value

class BarDesc(VirtFileDesc):
  def __init__(self, amdevice, bar, dev, sz, rcb, wcb, fd):
    super().__init__(fd)
    self.amdevice, self.bar, self.dev, self.sz, self.rcb, self.wcb = amdevice, bar, dev, sz, rcb, wcb

  def mmap(self, start, sz, prot, flags, fd, offset):
    start = mv_address({0: self.amdevice.vram, 2: self.amdevice.doorbells, 5: self.amdevice.mmio}[self.bar])
    if self.rcb is not None: self.amdevice.driver.track_address(start, start + self.sz, self.rcb, self.wcb)
    return start

class AMDriver(VirtDriver):
  def __init__(self, gpus=6):
    super().__init__()

    regions = {0: PCIRegion(4 << 30), 2: PCIRegion(1 << 20), 5: PCIRegion(2 << 20)}
    self.pci_devs = [PCIDesc(self, vendor=0x1002, device=0x744c, domain=0, bus=i, slot=0, func=0, regions=regions) for i in range(gpus)]
    self.am_decs = [AMDevice(self, 4 << 30) for _ in range(gpus)]
    for a,d in zip(self.am_decs, self.pci_devs): self._prepare_pci(a, d)

    # Fake fw files
    self.tracked_files += [
      VirtFile(f'/lib/firmware/amdgpu/psp_13_0_0_sos.bin', functools.partial(TextFileDesc, text="psp_13_0_0_sos.bin"))
    ]

    self.next_fd = (1 << 30)

  def probe(self, dev): pass
  def enable(self, dev): pass
  def cfg_read(self, dev, cmd): raise NotImplementedError()
  def cfg_write(self, dev, cmd, value): raise NotImplementedError()

  def _prepare_pci(self, am_dev, pci_dev):
    # self.doorbells[gpu_id] = memoryview(bytearray(0x2000))
    # self.gpus[gpu_id] = AMDGPU(gpu_id)

    pcibus = f"{pci_dev.domain:04x}:{pci_dev.bus:02x}:{pci_dev.slot:02x}.{pci_dev.func:d}"

    self.tracked_files += [
      VirtFile(f'/sys/bus/pci/devices/{pcibus}/resource0', functools.partial(BarDesc, am_dev, 0, pci_dev, (4 << 30), None, None)),
      VirtFile(f'/sys/bus/pci/devices/{pcibus}/resource2', functools.partial(BarDesc, am_dev, 2, pci_dev, (1 << 20),
        functools.partial(self._on_doorbell_read_pci, am_dev), functools.partial(self._on_doorbell_write_pci, am_dev))),
      VirtFile(f'/sys/bus/pci/devices/{pcibus}/resource5', functools.partial(BarDesc, am_dev, 5, pci_dev, (2 << 20),
        functools.partial(self._on_mmio_read_pci, am_dev), functools.partial(self._on_mmio_write_pci, am_dev))),
    ]

  def _alloc_fd(self):
    my_fd = self.next_fd
    self.next_fd = self.next_fd + 1
    return my_fd

  def _on_doorbell_read_pci(self, am_dev, mv, index):
    print("doorbell read", am_dev, mv, index)
  def _on_doorbell_write_pci(self, am_dev, mv, index):
    print("doorbell write", am_dev, mv, index)
  def _on_mmio_read_pci(self, am_dev, mv, index): return am_dev.mmio_read(mv, index)
  def _on_mmio_write_pci(self, am_dev, mv, index): am_dev.mmio_write(mv, index, mv[index])

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

  def anon_mmap(self, start, sz, prot, flags, offset):
    print(start, sz, prot, flags, offset)
    assert False
    return libc.mmap(start, sz, prot, flags, -1, offset)
