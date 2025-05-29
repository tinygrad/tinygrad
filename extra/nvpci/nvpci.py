import os, mmap, re, array, gzip, struct, ctypes, time
from tinygrad.helpers import fetch, to_mv, round_up
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc, pci
from tinygrad.runtime.autogen.nv import nv
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from hexdump import hexdump

pagemap = FileIOInterface("/proc/self/pagemap", os.O_RDONLY)

MAP_LOCKED = 0x2000
def alloc_sysmem(size, contigous=False):
  size = round_up(size, mmap.PAGESIZE)
  
  assert not contigous or size <= (2 << 20), "Contiguous allocation is only supported for sizes <= 2 MiB"
  flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | MAP_LOCKED

  if contigous and size > 0x1000: flags |= libc.MAP_HUGETLB
  va = FileIOInterface.anon_mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, flags, 0)
  assert va != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(va)}"

  # Read pagemap to get the physical address of each page. The pages are locked.
  pagemap.seek(va // mmap.PAGESIZE * 8)
  return va, [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', pagemap.read(size//mmap.PAGESIZE*8, binary=True))]

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

cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
pci_cmd = int.from_bytes(cfg_fd.read(2, binary=True, offset=pci.PCI_COMMAND), byteorder='little') | pci.PCI_COMMAND_MASTER
cfg_fd.write(pci_cmd.to_bytes(2, byteorder='little'), binary=True, offset=pci.PCI_COMMAND)
# print('pci cfg', hex(pci_cmd))

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
    self.include("src/common/inc/swref/published/turing/tu102/dev_fb.h")
    self.include("src/common/inc/swref/published/hopper/gh100/dev_riscv_pri.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_riscv_pri.h")
    self.include("src/common/inc/swref/published/hopper/gh100/dev_falcon_v4.h")

    self.kfsp_send_msg(self.NVDM_TYPE_CAPS_QUERY, bytes([self.NVDM_TYPE_CLOCK_BOOST]))
    self.init_gsp()

  def wreg(self, addr, value): self.mmio[addr // 4] = value
  def rreg(self, addr): return self.mmio[addr // 4]

  def _alloc_boot_struct(self, typ):
    va, paddrs = alloc_sysmem(ctypes.sizeof(typ), contigous=True)
    return typ.from_address(va), paddrs[0]

  def init_boot_binary_image(self):
    text = self._download("src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmBoot_GB202.c")

    def _find_section(name):
      sl = text[text.find(name) + len(name) + 7:]
      return bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))

    image = _find_section("kgspBinArchiveGspRmBoot_GB202_ucode_image_prod_data")
    desc = _find_section("kgspBinArchiveGspRmBoot_GB202_ucode_desc_prod_data")
    desc = gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + desc)

    image_va, image_sysmem = alloc_sysmem(len(image), contigous=True)
    to_mv(image_va, len(image))[:] = image

    assert len(image) == 0x31000
    return image_sysmem[0], len(image), nv.RM_RISCV_UCODE_DESC.from_buffer_copy(desc)

  def init_gsp_fmc_image(self):
    text = self._download("src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmFmcGfwProdSigned_GB202.c")

    def _find_section(name):
      sl = text[text.find(name) + len(name) + 7:]
      return bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))

    image = _find_section("kgspBinArchiveGspRmFmcGfwProdSigned_GB202_ucode_image_data")
    ihash = _find_section("kgspBinArchiveGspRmFmcGfwProdSigned_GB202_ucode_hash_data")
    sig = _find_section("kgspBinArchiveGspRmFmcGfwProdSigned_GB202_ucode_sig_data")
    pkey = _find_section("kgspBinArchiveGspRmFmcGfwProdSigned_GB202_ucode_pkey_data") + b'\x00\x00\x00'

    print('fmc len:', hex(len(image)),  image[:16])
    image_va, image_sysmem = alloc_sysmem(len(image), contigous=True)
    to_mv(image_va, len(image))[:] = image

    return image_sysmem[0], len(image), memoryview(ihash).cast('I'), memoryview(sig).cast('I'), memoryview(pkey).cast('I')

  def init_gsp_image(self):
    fwpath = "/lib/firmware/nvidia/570.133.20/gsp_ga10x.bin"
    fwbytes = FileIOInterface(fwpath, os.O_RDONLY).read(binary=True)
    assert len(fwbytes) == 63534832

    _, sections, _ = elf_loader(fwbytes)
    image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == ".fwsignature_gb20x"))

    # radix3
    npages = [0] * 4
    offsets = [0] * 4
    npages[3] = (len(image) + nv.LIBOS_MEMORY_REGION_RADIX_PAGE_SIZE - 1) >> nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2
    for i in range(3, 0, -1): npages[i-1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1
    
    total_pages = 0
    for i in range(1, 4):
      total_pages += npages[i-1]
      offsets[i] = total_pages << nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2
    print(npages, offsets)

    alloc_size = total_pages << nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2
    sysmem_va, sysmem_paddrs = alloc_sysmem(alloc_size, contigous=False)

    page_off = 0
    for i in range(0, 2):
      page_off += npages[i]
      to_mv(sysmem_va + offsets[i], npages[i+1] * 8).cast('Q')[:] = array.array('Q', sysmem_paddrs[page_off:page_off + npages[i+1]])

    image_va, image_paddrs = alloc_sysmem(len(image), contigous=False)
    to_mv(image_va, len(image))[:] = image
    to_mv(sysmem_va + offsets[2], npages[3] * 8).cast('Q')[:] = array.array('Q', image_paddrs)

    radix3_head = sysmem_paddrs[0]

    sign_va, sign_sysmem = alloc_sysmem(len(signature), contigous=True)
    to_mv(sign_va, len(signature))[:] = signature

    return radix3_head, sign_sysmem, len(image)

  def init_rm_args(self):
    rm_args, rm_args_sysmem = self._alloc_boot_struct(nv.GSP_ARGUMENTS_CACHED)
    
    # queue messages init
    queue_sizes = 0x40000 + 0x40000 # cmd + stat
    pt_size = round_up((pt_entries:=(queue_sizes >> 12)) * 8, 0x1000)
    shared_buf_size = pt_size + queue_sizes

    shared_va, shared_sysmem = alloc_sysmem(shared_buf_size, contigous=False)
    for i,sysmem in enumerate(shared_sysmem): to_mv(shared_va + i * 0x8, 0x8).cast('Q')[0] = sysmem

    self.command_queue_va = shared_va + pt_size
    self.status_queue_va = shared_va + pt_size + 0x40000

    rm_args.messageQueueInitArguments.sharedMemPhysAddr = shared_sysmem[0]
    rm_args.messageQueueInitArguments.pageTableEntryCount = pt_entries
    rm_args.messageQueueInitArguments.cmdQueueOffset = pt_size
    rm_args.messageQueueInitArguments.statQueueOffset = pt_size + 0x40000
    rm_args.bDmemStack = True

    rm_args.srInitArguments.bInPMTransition = False
    rm_args.srInitArguments.oldLevel = 0
    rm_args.srInitArguments.flags = 0

    rm_args.gpuInstance = 0
    return rm_args_sysmem

  def init_libos_args(self):
    logbuf_va, logbuf_sysmem = alloc_sysmem((2 << 20), contigous=True)
    libos_init, libos_init_sysmem = alloc_sysmem(0x1000, contigous=True)

    off_sz = 0
    for i,(name, size) in enumerate([("INIT", 0x10000), ("INTR", 0x10000), ("RM", 0x10000), ("MNOC", 0x10000), ("KRNL", 0x10000)]):
      to_mv(logbuf_va + off_sz + 8, 8).cast('Q')[0] = logbuf_sysmem[0] + off_sz # this is radix3 pt address, we can ignore it, it's contig now.

      arg = nv.LibosMemoryRegionInitArgument.from_address(libos_init + i * ctypes.sizeof(nv.LibosMemoryRegionInitArgument))
      arg.kind = nv.LIBOS_MEMORY_REGION_CONTIGUOUS
      arg.loc = nv.LIBOS_MEMORY_REGION_LOC_SYSMEM
      arg.size = size
      arg.id8 = int.from_bytes(bytes("LOG" + name, 'utf-8'), 'big')
      arg.pa = logbuf_sysmem[0] + off_sz

      print(hex(arg.id8))

      off_sz += size

    # rm initargs
    rm_args_sysmem = self.init_rm_args()
    arg = nv.LibosMemoryRegionInitArgument.from_address(libos_init + 5 * ctypes.sizeof(nv.LibosMemoryRegionInitArgument))
    arg.kind = nv.LIBOS_MEMORY_REGION_CONTIGUOUS
    arg.loc = nv.LIBOS_MEMORY_REGION_LOC_SYSMEM
    arg.size = 0x1000
    arg.id8 = int.from_bytes(bytes("RMARGS", 'utf-8'), 'big')
    arg.pa = rm_args_sysmem

    return libos_init_sysmem[0]

  def init_gsp(self):
    bootload_ucode_sysmem, bootload_ucode_size, bootload_desc = self.init_boot_binary_image()
    radix3_sysmem, signature_sysmem, radix3_elf_size = self.init_gsp_image()

    print(hex(radix3_sysmem))

    fmc_boot, fmc_boot_sysmem = self._alloc_boot_struct(nv.GSP_FMC_BOOT_PARAMS)
    wpr_meta, wpr_meta_sysmem = self._alloc_boot_struct(nv.GspFwWprMeta)
    libos_init_sysmem = self.init_libos_args()

    # wpr meta fillup
    wpr_meta.vgaWorkspaceSize = 128 * 1024
    wpr_meta.pmuReservedSize = 0x1820000
    wpr_meta.sizeOfBootloader = bootload_ucode_size
    wpr_meta.sysmemAddrOfBootloader = bootload_ucode_sysmem

    wpr_meta.sizeOfRadix3Elf = radix3_elf_size
    wpr_meta.sysmemAddrOfRadix3Elf = radix3_sysmem
    print("radix3_sysmem:", hex(radix3_sysmem), "size:", radix3_elf_size)

    wpr_meta.bootloaderCodeOffset = bootload_desc.monitorCodeOffset
    wpr_meta.bootloaderDataOffset = bootload_desc.monitorDataOffset
    wpr_meta.bootloaderManifestOffset = bootload_desc.manifestOffset
    print("bootloader:", hex(wpr_meta.bootloaderCodeOffset), hex(wpr_meta.bootloaderDataOffset), hex(wpr_meta.bootloaderManifestOffset))

    wpr_meta.sysmemAddrOfSignature = signature_sysmem[0]
    wpr_meta.sizeOfSignature = 0x1000
    print("wpr signatue:", hex(signature_sysmem[0]), "size:", 0x1000)

    wpr_meta.nonWprHeapSize = 0x220000
    wpr_meta.gspFwHeapSize = 0x8700000
    wpr_meta.frtsSize = 0x100000
    wpr_meta.gspFwHeapVfPartitionCount = 0x0
    wpr_meta.revision = nv.GSP_FW_WPR_META_REVISION
    wpr_meta.magic = nv.GSP_FW_WPR_META_MAGIC

    fmc_boot.bootGspRmParams.gspRmDescOffset = wpr_meta_sysmem
    fmc_boot.bootGspRmParams.gspRmDescSize = ctypes.sizeof(nv.GspFwWprMeta)
    fmc_boot.bootGspRmParams.target = nv.GSP_DMA_TARGET_COHERENT_SYSTEM
    fmc_boot.bootGspRmParams.bIsGspRmBoot = True

    fmc_boot.gspRmParams.bootArgsOffset = libos_init_sysmem
    fmc_boot.gspRmParams.target = nv.GSP_DMA_TARGET_COHERENT_SYSTEM

    fmc_image_sysmem, fmc_image_len, ihash, sig, pkey = self.init_gsp_fmc_image()
    cot_payload = nv.NVDM_PAYLOAD_COT(version=0x2, size=0x35c, frtsVidmemOffset=0x1c00000, frtsVidmemSize=0x100000)
    cot_payload.gspBootArgsSysmemOffset = fmc_boot_sysmem
    cot_payload.gspFmcSysmemOffset = fmc_image_sysmem
    for i,x in enumerate(ihash): cot_payload.hash384[i] = x
    for i,x in enumerate(sig): cot_payload.signature[i] = x
    for i,x in enumerate(pkey): cot_payload.publicKey[i] = x

    print("COT")
    # hexdump(bytes(cot_payload))

    print(hex(self.rreg(self.NV_PFB_PRI_MMU_WPR2_ADDR_HI)))
    print(hex(self.rreg(self.NV_FALCON2_GSP_BASE + self.NV_PRISCV_RISCV_CPUCTL)))

    self.kfsp_send_msg(self.NVDM_TYPE_COT, bytes(cot_payload))

    time.sleep(5) # need normal wait here

    print(hex(self.rreg(self.NV_PFB_PRI_MMU_WPR2_ADDR_HI)))
    # print(hex(self.rreg(self.NV_FALCON2_GSP_BASE + self.NV_PFALCON_FALCON_MAILBOX0)))
    print(hex(self.rreg(self.NV_FALCON2_GSP_BASE + self.NV_PRISCV_RISCV_CPUCTL)))

    hexdump(to_mv(self.status_queue_va, 0x100))
    hexdump(to_mv(self.command_queue_va, 0x100))


  def gsp_rpc_wait(self, op):
    pass

  
  def _download(self, file) -> str:
    url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/e8113f665d936d9f30a6d508f3bacd1e148539be/{file}"
    return fetch(url, subdir="defines").read_text()

  def include(self, file) -> str:
    if file in self.scanned_files: return
    self.scanned_files.add(file)

    txt = self._download(file)

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
          assert self.__dict__.get(name) in [None, int(value, 0)], f"Duplicate definition for {name} in {file}"
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

