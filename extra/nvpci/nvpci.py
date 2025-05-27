import os, mmap, re
from tinygrad.helpers import fetch
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.autogen.nv import nv
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface

dev = None
for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
  vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
  device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
  if vendor == 0x10de and device == 0x2b85: dev = pcibus

pcibus = dev

if FileIOInterface.exists(f"/sys/bus/pci/devices/{pcibus}/driver"):
  FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/driver/unbind", os.O_WRONLY).write(pcibus)

cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in [0, 1, 3]}

bar_info = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource", os.O_RDONLY).read().splitlines()
bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

def _map_pci_range(bar, off=0, addr=0, size=None, fmt='B'):
  fd, sz = bar_fds[bar], size or (bar_info[bar][1] - bar_info[bar][0] + 1)
  libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
  assert loc != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(addr)}"
  return MMIOInterface(loc, sz, fmt=fmt)

regs = _map_pci_range(0, fmt='I')
fb = _map_pci_range(1)

class NVDev():
  def __init__(self, devfmt, mmio, vram):
    self.devfmt, self.mmio, self.vram = devfmt, mmio, vram
    self.defs, self.scanned_files = {}, set()

    self.include("src/nvidia/arch/nvalloc/common/inc/fsp/fsp_nvdm_format.h")

    self.kfsp_send_msg(self.NVDM_TYPE_CAPS_QUERY, bytes([self.NVDM_TYPE_CLOCK_BOOST]))

    # gb100+ are cot v2.
    cot = nv.NVDM_PAYLOAD_COT(version=2, size=ctypes.sizeof(nv.NVDM_PAYLOAD_COT), frtsSysmemOffset=0x0, frtsSysmemSize=0x0,
                             frtsVidmemOffset=0x1c00000, frtsVidmemSize=0x100000, gspBootArgsSysmemOffset=0xfffff000)


  def wreg(self, addr, value): self.mmio[addr // 4] = value
  def rreg(self, addr): return self.mmio[addr // 4]

  def gsp_image_setup(self):
    fwpath = "/lib/firmware/nvidia/570.133.20/gsp_ga10x.bin"
    fwbytes = FileIOInterface(fwpath, os.O_RDONLY).read(binary=True)
    assert len(fwbytes) == 63534832

    pass


    

  def include(self, file) -> str:
    if file in self.scanned_files: return
    self.scanned_files.add(file)

    url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/e8113f665d936d9f30a6d508f3bacd1e148539be/{file}"
    txt = fetch(url, subdir="defines").read_text()

    PARAM = re.compile(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)')
    CONST = re.compile(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)')
    BITFLD = re.compile(r'#define\s+(\w+)\s+(\d+):(\d+)')

    for raw in txt.splitlines():
      if raw.startswith("#define "):
        if (m := BITFLD.match(raw)):
          name, hi, lo = m.groups()
          self.defs[f"{name}_SHIFT"] = int(lo)
          self.defs[f"{name}_MASK"]  = ((1 << (int(hi) - int(lo) + 1)) - 1) << int(lo)
        elif (m := PARAM.match(raw)):
          name, param, expr = m.groups()
          expr = expr.strip().rstrip('\\').split('/*')[0].rstrip()
          assert self.__dict__.get(name) is None, f"Duplicate definition for {name} in {file}"
          self.__dict__[name] = eval(f"lambda {param}: {expr}")
        elif (m := CONST.match(raw)):
          name, value = m.groups()
          assert self.__dict__.get(name) is None, f"Duplicate definition for {name} in {file}"
          self.__dict__[name] = int(value, 0)

  def build_num(self, name, _val=0, **kwargs):
    for k, v in kwargs.items(): _val |= (v << self.defs[f"{name.upper()}_{k.upper()}_SHIFT"]) & self.defs[f"{name.upper()}_{k.upper()}_MASK"]
    return _val

  def read_num(self, name, val, *fields):
    if len(fields) > 1: return tuple(self.read_num(name, val, f) for f in fields)
    return (val & self.defs[f"{name}_{fields[0].upper()}_MASK"]) >> self.defs[f"{name}_{fields[0].upper()}_SHIFT"]

  def kfsp_send_msg(self, nvmd:int, buf:bytes):
    self.include("src/nvidia/arch/nvalloc/common/inc/fsp/fsp_mctp_format.h")
    self.include("src/common/inc/swref/published/hopper/gh100/dev_fsp_pri.h")
    self.include("src/nvidia/arch/nvalloc/common/inc/fsp/fsp_emem_channels.h")

    # All single-packets go to seid 0
    mctp_header = self.build_num("MCTP_HEADER", som=1, eom=1, seid=0, seq=0)
    nvdm_header = self.build_num("MCTP_MSG_HEADER", type=self.MCTP_MSG_HEADER_TYPE_VENDOR_PCI, nvdm_type=nvmd, vendor_id=0x10de)

    buf = int.to_bytes(mctp_header, 4, 'little') + int.to_bytes(nvdm_header, 4, 'little') + buf + (4 - (len(buf) % 4)) * b'\x00'
    assert len(buf) < 0x400, f"Message too long... {len(buf)} bytes, max 1024 bytes"

    self.wreg(self.NV_PFSP_EMEMC(self.FSP_EMEM_CHANNEL_RM), self.build_num("NV_PFSP_EMEMC", offs=0, blk=0, aincw=1, aincr=0))
    for i in range(0, len(buf), 4): self.wreg(self.NV_PFSP_EMEMD(self.FSP_EMEM_CHANNEL_RM), int.from_bytes(buf[i:i+4], 'little'))

    # Check offset
    reg = self.rreg(self.NV_PFSP_EMEMC(self.FSP_EMEM_CHANNEL_RM))
    offs, blk = self.read_num("NV_PFSP_EMEMC", self.rreg(self.NV_PFSP_EMEMC(self.FSP_EMEM_CHANNEL_RM)), "OFFS", "BLK")
    # print(offs, blk)

    self.wreg(self.NV_PFSP_QUEUE_TAIL(self.FSP_EMEM_CHANNEL_RM), len(buf) - 4) # TAIL points to the last DWORD written, so subtract 1
    self.wreg(self.NV_PFSP_QUEUE_HEAD(self.FSP_EMEM_CHANNEL_RM), 0)

    while True:
      head, tail = self.rreg(self.NV_PFSP_MSGQ_HEAD(self.FSP_EMEM_CHANNEL_RM)), self.rreg(self.NV_PFSP_MSGQ_TAIL(self.FSP_EMEM_CHANNEL_RM))
      if head != tail: break

    head, tail = self.rreg(self.NV_PFSP_MSGQ_HEAD(self.FSP_EMEM_CHANNEL_RM)), self.rreg(self.NV_PFSP_MSGQ_TAIL(self.FSP_EMEM_CHANNEL_RM))
    msg_len = tail - head + 4

    self.wreg(self.NV_PFSP_EMEMC(self.FSP_EMEM_CHANNEL_RM), self.build_num("NV_PFSP_EMEMC", offs=0, blk=0, aincw=0, aincr=1))

    msg = bytearray()
    for i in range(0, msg_len, 4): msg += int.to_bytes(self.rreg(self.NV_PFSP_EMEMD(self.FSP_EMEM_CHANNEL_RM)), 4, 'little')

    self.wreg(self.NV_PFSP_MSGQ_TAIL(self.FSP_EMEM_CHANNEL_RM), head)

    print(f"Received {len(msg)} bytes: {msg.hex()}")

    som, eom, seid, seq = self.read_num("MCTP_HEADER", int.from_bytes(msg[:4], 'little'), "SOM", "EOM", "SEID", "SEQ")
    typ, vendor_id = self.read_num("MCTP_MSG_HEADER", int.from_bytes(msg[4:8], 'little'), "TYPE", "VENDOR_ID")
    assert som == 1 and eom == 1, f"Invalid MCTP header: {som}, {eom}, {seid}, {seq}"
    assert typ == self.MCTP_MSG_HEADER_TYPE_VENDOR_PCI and vendor_id == 0x10de, f"Invalid NVDM header: {typ:x}, {vendor_id:x}"

nvdev = NVDev(pcibus, regs, fb)
# nvdev.kfsp_send_msg(0, b'\x00\x01\x02\x03\x04\x05\x06\x07')

# fwpath = "/lib/firmware/nvidia/570.133.20/gsp_ga10x.bin"
# fwbytes = FileIOInterface(fwpath, os.O_RDONLY).read(binary=True)
# assert len(fwbytes) == 63534832

# # def wreg(addr, value): regs[addr // 4] = value
# # def rreg(addr): return regs[addr // 4]

# # pmc_boot_1 = rreg(0x00000004)
# # pmc_boot_0 = rreg(0x00000000)
# # pmc_boot_42 = rreg(0x00000A00)

# def kfsp_send_msg(buf, ret=False):
#   # kfspSendMessage()
#   header_size = 2 * 4 # 2 dwords

#   pass

#   # kfspPollForResponse()
#   # kfspReadMessage()

# print(hex(pmc_boot_42))

# version=0x2, size=0x35c, gspFmcSysmemOffset=0xf7a80000
# frtsSysmemOffset=0x0, frtsSysmemSize=0x0
# frtsVidmemOffset=0x1c00000, frtsVidmemSize=0x100000
# gspBootArgsSysmemOffset=0xfffff000
# fsp

# prapare for bootstrap
# NV_PGSP_FALCON_ENGINE = 0x1103c0
# print(hex(rreg(NV_PGSP_FALCON_ENGINE)))
# exit(0)

# wreg(NV_PGSP_FALCON_ENGINE, rreg(NV_PGSP_FALCON_ENGINE) & ~0x1)
# while ((rreg(NV_PGSP_FALCON_ENGINE) >> 8) & 0b11) != 0b10:
#   print(hex(rreg(NV_PGSP_FALCON_ENGINE)))

# print("reset done")
# wreg(NV_PGSP_FALCON_ENGINE, rreg(NV_PGSP_FALCON_ENGINE) | 0x1)

# kfsp path

