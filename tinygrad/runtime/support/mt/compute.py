from __future__ import annotations
import os, struct, time
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.mtdev import MTDev, MT_SOCIF_OFFSET
from tinygrad.runtime.support.mt.mmu import MTMMU4

FW_BASE, ROOT_BASE, INIT_SIZE, FW_DEVVA = 0x20000000, 0x30000000, 0x2000000, 0xe1c0000000
FWIF_OFF, META_BASE = 0xc0000, 0x90000000
SUB_PA, SUB_VA = FW_BASE + INIT_SIZE, FW_DEVVA + INIT_SIZE
DEFAULT_FIRMWARE = '/tmp/mt-research/musa-driver-3.3.0/lib/firmware/mthreads/mtfw-gen1.bin'

class MTFirmwareLayout:
  def __init__(self, core_count:int=8):
    self.reg_dump_buffer_size = (core_count * 40 + 24 + 8) * 64
    self.vgpu_shared = (0xfffed2e0 + 0x97730 + core_count * 40) & 0xffffffff
    self.trace = (0xfffed2e0 + 0x26e10 + 0x40ee0 + FWIF_OFF) & 0xffffffff
    self.perf = FWIF_OFF + ((0xfffed2e0 + 0x26e10 + 0x70a50) & 0xffffffff)
    self.fault = self.perf + 0xe6018
    self.dump = self.fault + 0x88218
    self.regcfg = self.fault + 0xee240
    self.regbuf = self.regcfg + 0x30
    self.item = self.regbuf + self.reg_dump_buffer_size
    self.gpu = self.item + self.vgpu_shared
    self.watchdog = self.gpu + 0x4c0
    self.ccb = FWIF_OFF + 0x40ee0

class MTFirmware:
  def __init__(self, dev:MTDev, path:str|None=None, core_count:int=8, prepare:bool=True):
    self.dev, self.path, self.core_count = dev, path or os.environ.get('MT_FIRMWARE', DEFAULT_FIRMWARE), core_count
    self.layout = MTFirmwareLayout(core_count)
    self.v = self.dev.pci_dev.map_bar(2, off=FW_BASE, size=INIT_SIZE, fmt='B')
    self.mmu = MTMMU4(ROOT_BASE)
    if prepare: self._prepare()

  def _prepare(self) -> None:
    if os.environ.get('MT_TRACE'): print('MT prepare begin', flush=True)
    lay = self.layout
    self.dev.smc._write(0x1400000, struct.pack('<I', 1))
    self.dev.smc._write(0x1400004, struct.pack('<I', 1))
    with open(self.path, 'rb') as f: raw = f.read()
    if os.environ.get('MT_TRACE'): print('MT firmware read', flush=True)
    self.v[:0x500000] = bytes(0x500000)
    if os.environ.get('MT_TRACE'): print('MT firmware zeroed', flush=True)
    self.v[:len(raw)] = raw
    fw = mtfw.struct_MTFW_FWIF_TAG()
    fw.watchDog = META_BASE + lay.watchdog
    fw.ccbQueueArrayAddr, fw.ccbQueueArrayNum = META_BASE + lay.ccb, 5
    fw.traceBufAddr = META_BASE + lay.trace
    fw.metaPerfEventBufCtrlArray, fw.metaPerfEventBufCtrlNum = META_BASE + lay.perf, 1
    fw.faultDumpBuffer, fw.gpuUtilBufAddr = META_BASE + lay.fault, META_BASE + lay.gpu
    fw.socTimerFreq, fw.gpuFrequence = 25, 1600000000
    fw.multicoreClusterMask = 0
    cfg = fw.fwCfgs.cfg
    cfg.version, cfg.flags.value = 0, 0
    for name,count,base in [('flStateAddrsSCG',2,0),('flStateAddrsMCG2',4,0x200),('flStateAddrsMCG3',6,0x600),
                            ('flStateAddrsMCG4',8,0xc00),('flStateAddrsMCG5',10,0x1400),('flStateAddrsMCG6',12,0x1e00),
                            ('flStateAddrsMCG7',14,0x2a00),('flStateAddrsMCG8',16,0x3800)]:
      for i in range(count): getattr(cfg, name)[i] = 0xd9bfff0000 | base + i * 0x100
    cfg.yuvCoefficientAddr = 0xe0ffe00000
    cfg.pdsHeapBaseAddr, cfg.uscHeapBaseAddr = 0xda00000000, 0xe000000000
    cfg.fbcdcHeapBaseAddr, cfg.fbcdcLargeHeapBaseAddr = 0xec00000000, 0xec40000000
    cfg.texStateHeapBaseAddr, cfg.componentCtrlHeapBaseAddr = 0xb000000000, 0xeb00000000
    cfg.dmKillShaderOffset = 0xfff80000
    self.v[FWIF_OFF:FWIF_OFF+fw.SIZE] = bytes(fw)
    self.v[lay.trace:lay.trace+128] = struct.pack('<32I', *([0xffffffff] * 32))
    self.v[lay.trace+0x17d20:lay.trace+0x17d20+128] = struct.pack('<32I', *([1] * 32))
    self.v[lay.fault+0x8208:lay.fault+0x8210] = struct.pack('<Q', META_BASE + lay.dump)
    self.v[lay.fault+0x8210:lay.fault+0x8214] = struct.pack('<I', 1)
    for i in range(64):
      self.v[lay.fault+i*0x208+0x1f0:lay.fault+i*0x208+0x1f8] = struct.pack('<Q', META_BASE+lay.regbuf+i*(lay.reg_dump_buffer_size>>6))
      self.v[lay.fault+i*0x208+0x1f8:lay.fault+i*0x208+0x1fc] = struct.pack('<I', lay.reg_dump_buffer_size>>9)
    self.v[lay.dump+8:lay.dump+12] = struct.pack('<I', 1)
    rc = bytearray(16)
    struct.pack_into('<I', rc, 0, 0x208e150)
    struct.pack_into('<I', rc, 5, 0x1000000)
    struct.pack_into('<H', rc, 10, self.core_count * 2)
    self.v[lay.regcfg:lay.regcfg+16] = rc
    if os.environ.get('MT_TRACE'): print('MT FWIF written', flush=True)
    self.mmu.map_range(FW_DEVVA, FW_BASE, INIT_SIZE)
    if os.environ.get('MT_TRACE'): print('MT page tables built', flush=True)

  def _write(self, off:int, value:int, size:int=8) -> None: self.dev.smc._write(MT_SOCIF_OFFSET + off, value.to_bytes(size, 'little'))
  def _read(self, off:int, size:int=8) -> int: return int.from_bytes(self.dev.smc._read(MT_SOCIF_OFFSET + off, size), 'little')

  def upload_mmu(self) -> None:
    rv = self.dev.pci_dev.map_bar(2, off=ROOT_BASE, size=self.mmu.next, fmt='B')
    rv[:] = self.mmu.root[:self.mmu.next]

  def start(self, timeout:float=10.0) -> None:
    if os.environ.get('MT_TRACE'): print('MT start upload', flush=True)
    self.upload_mmu()
    if os.environ.get('MT_TRACE'): print('MT start clocks', flush=True)
    self._write(0xa100, 0, 4)
    self._read(0xa100, 4)
    for value in [0x1249249249249249,0x36db6db6db6db6db,0x7fffffffffffffff]:
      self._write(0x630, value)
      self._read(0x630)
    for off in [0x650,0x640]:
      self._write(off, 0xffffffff, 4)
      self._read(off, 4)
    for value in [0x386000000800,0x3860000088f2,0x20000080f2,0x2000000000]:
      self._write(0x100, value)
      self._read(0x100)
    for off in [0x640,0x650]:
      self._write(off, 0, 4)
      self._read(off, 4)
    for value in [0x36db6db6db6db6db,0x1249249249249249,0]:
      self._write(0x630, value)
      self._read(0x630)
    self._read(0x100)
    self._write(0xbf8, 1, 4)
    self._write(0xb50, 0)
    self._write(0, 0x5545010155151101)
    self._write(0x80, 0x5554151555455555)
    self._write(0x1158, 0xff, 4)
    if os.environ.get('MT_CLK_DEBUG'):
      print('MT PRECLK', hex(self._read(0)), hex(self._read(0x80)), hex(self._read(0x1158)), hex(self._read(0x1148)), flush=True)
    clock_deadline = time.monotonic() + min(timeout, 1.0)
    while self._read(0x1148) & 0xff != 0xff and time.monotonic() < clock_deadline: time.sleep(0.001)
    self._write(0, 0xaa8a0202aa2a2202)
    self._write(0x80, 0xaaa82a2aaa8aaaaa)
    self._write(0x80008, 0xff, 4)
    self._read(0x80008, 4)
    self._write(0x7080, 0x1000000000000000 | FW_BASE)
    self._write(0x7048, 0x3000000000000000 | (FW_BASE + 0x100000))
    self._write(0xb50, 0)
    self._write(0xe140, 0, 4)
    self._read(0xe140, 4)
    self._write(0xe148, ROOT_BASE >> 12, 4)
    for off,value in [(0xe350,0),(0xe358,0xfff80800),(0xe360,0),(0xe368,0xfff80000)]: self._write(off, value)
    self._write(0xe328, 0x0202020202020202)
    self._write(0xe320, 0x78ff2)
    self._write(0, (self._read(0) & 0xfcfffffff3fffcff) | 0x100000004000100)
    self._write(0x80, (self._read(0x80) & 0xffffffffcf3fffff) | 0x10400000)
    time.sleep(3e-6)
    self._write(0x100, 0)
    self._read(0x100)
    time.sleep(3e-6)
    self._write(0x93c0, 1, 4)
    self._write(0x70c0, 1, 4)
    if os.environ.get('MT_TRACE'): print('MT start released', flush=True)
    deadline = time.monotonic() + timeout
    while True:
      started = struct.unpack('<I', self.v[FWIF_OFF+4:FWIF_OFF+8])[0]
      if started & 0xff == 0x4d:
        self.started, self.build = started, bytes(self.v[FWIF_OFF+8:FWIF_OFF+16]).rstrip(b'\0')
        return
      if time.monotonic() > deadline: raise TimeoutError(f"MT firmware start timed out: {started:#x}, fault={self._read(0xe160):#x}")
      time.sleep(0.01)
