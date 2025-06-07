import ctypes, time, contextlib, importlib, functools, os, array, gzip, struct
from tinygrad.runtime.autogen.nv import nv
from tinygrad.helpers import to_mv, data64, lo32, hi32, DEBUG, round_up
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.support.nvd import alloc_sysmem
from tinygrad.runtime.support.elf import elf_loader
from hexdump import hexdump

class NV_FLCN:
  def __init__(self, nvdev): self.nvdev = nvdev

  def init_sw(self):
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_gsp.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_v4.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_v4_addendum.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_riscv_pri.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_fbif_v4.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_second_pri.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_sec_pri.h")
    self.nvdev.include("src/common/inc/swref/published/turing/tu102/dev_bus.h")
    self.nvdev.include("src/common/inc/swref/published/turing/tu102/dev_fb.h")

    self.prep_ucode_v3()

  def prep_ucode_v3(self):
    # TODO: these are hardcoded for now, need to be read from the ROM
    vbios_fd = FileIOInterface("/home/nimlgen/tinygrad/bios_4090.rom", os.O_RDONLY | os.O_SYNC | os.O_CLOEXEC)
    vbios_bytes = vbios_fd.read(size=0x98e00, binary=True, offset=0x9400)

    # hexdump(vbios_bytes[:0x1000])

    # # bit_addr = 0x1b0
    # # ucodeDescVersion = 0x3
    ucodeDescOffset, ucodeDescSize = 0x4515c, 0x32c
    self.desc_v3 = nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucodeDescOffset:ucodeDescOffset + ucodeDescSize])

    sig_total_size = ucodeDescSize - nv.FALCON_UCODE_DESC_V3_SIZE_44
    signature = vbios_bytes[ucodeDescOffset + nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
    image = vbios_bytes[0x45488:][:round_up(self.desc_v3.StoredSize, 256)]

    assert len(signature) == 0x300 and len(image) == 0x10300
    # self.image_va, self.image_sysmem = alloc_sysmem(len(image), contigous=True, data=image)
    # self.signature_va, self.signature_sysmem = alloc_sysmem(len(signature), contigous=True, data=signature)

    self.frts_offset = 0x5ff200000

    read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)

    frts_cmd = nv.FWSECLIC_FRTS_CMD()
    frts_cmd.frtsRegionDesc.version = 0x1
    frts_cmd.frtsRegionDesc.size = ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC)
    frts_cmd.frtsRegionDesc.frtsRegionOffset4K = self.frts_offset >> 12
    frts_cmd.frtsRegionDesc.frtsRegionSize = 0x100
    frts_cmd.frtsRegionDesc.frtsRegionMediaType = 2
    frts_cmd.readVbiosDesc = read_vbios_desc

    def __patch(cmd_id, cmd):
      data_offset = self.desc_v3.IMEMLoadSize
      # print(hex(self.desc_v3.IMEMLoadSize))
      dmem_mapper_offset = data_offset + 0xae0
      
      dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[dmem_mapper_offset:][:ctypes.sizeof(nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3)])
      dmem.init_cmd = cmd_id
      cmd_in_buffer_offset = data_offset + dmem.cmd_in_buffer_offset
      # print(hex(dmem.cmd_in_buffer_offset))

      cmd = bytes(cmd)
      if cmd_id == 0x15:
        x = memoryview(bytearray(bytes(cmd))).cast('I')
        x[11] = 0xffffffff
        cmd = bytes(x)

      patched_image = bytearray(image)
      patched_image[dmem_mapper_offset:dmem_mapper_offset+ctypes.sizeof(nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3)] = bytes(dmem)
      patched_image[cmd_in_buffer_offset:cmd_in_buffer_offset+len(cmd)] = bytes(cmd)
      # print(hex(cmd_in_buffer_offset), bytes(cmd))
      patched_image[data_offset+self.desc_v3.PKCDataOffset:data_offset+self.desc_v3.PKCDataOffset+0x180] = signature[0x180:]
      assert self.desc_v3.PKCDataOffset == 0xb24

      x = memoryview(patched_image).cast('I')
      checksum = 0
      for i in range(0x10300 // 4): checksum = (checksum + (x[i] * i)) % (int(1e9) + 7)
      print(f"Checksum: {checksum:08x}")
      return alloc_sysmem(len(patched_image), contigous=True, data=patched_image)

    self.frts_image_va, self.frts_image_sysmem = __patch(0x15, frts_cmd)
    self.sb_image_va, self.sb_image_sysmem = __patch(0x19, read_vbios_desc)

  def init_hw(self):
    self.nvdev.wreg(0x00100c40, 0x00000000)
    # self.nvdev.wreg(0x00100c10, 0x01637ea0)

    self.nvdev.wreg(0x00088080, 0x00002937)
    self.nvdev.wreg(0x00088080, 0x00002937)

    self.nvdev.wreg(0x00009410, 0x1846cedb)
    self.nvdev.wreg(0x00009400, 0x70dc2a48)

    self.falcon, self.sec2 = 0x00110000, 0x00840000

    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_sysmem[0])

    print(hex(self.nvdev.NV_PBUS_VBIOS_SCRATCH[0xe].read()))
    print(hex(self.nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read()))
    print(hex(self.nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_LO.read()))

    self.reset(self.falcon, riscv=True)

    # set up the mailbox
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_desc_sysmem))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_desc_sysmem))

    # booter
    print("---- booter -----")
    self.reset(self.sec2)

  def execute_dma(self, base, cmd, dest, mem_off, sysmem, size):
    while self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'] != 0: pass

    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE.with_base(base).write(lo32(sysmem >> 8))
    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE1.with_base(base).write(hi32(sysmem >> 8) & 0x1ff)

    xfered = 0
    while xfered < size:
      while self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'] != 0: pass

      self.nvdev.NV_PFALCON_FALCON_DMATRFMOFFS.with_base(base).write(dest + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFFBOFFS.with_base(base).write(mem_off + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).write(cmd)
      xfered += 256

    while self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['idle'] != 1: pass
  
  def execute_hs(self, base, image_sysmem):
    self.disable_ctx_req(base)

    self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma:=0].update(target=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM,
      mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=1, sec=1)
    self.execute_dma(base, cmd, dest=self.desc_v3.IMEMPhysBase, mem_off=self.desc_v3.IMEMVirtBase,
      sysmem=image_sysmem-self.desc_v3.IMEMVirtBase, size=self.desc_v3.IMEMLoadSize)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=0, sec=0)
    self.execute_dma(base, cmd, dest=self.desc_v3.DMEMPhysBase, mem_off=0,
      sysmem=image_sysmem+self.desc_v3.IMEMLoadSize, size=self.desc_v3.DMEMLoadSize)

    self.nvdev.NV_PFALCON2_FALCON_BROM_PARAADDR.with_base(base)[0].write(self.desc_v3.PKCDataOffset)
    self.nvdev.NV_PFALCON2_FALCON_BROM_ENGIDMASK.with_base(base).write(self.desc_v3.EngineIdMask)
    self.nvdev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.with_base(base).write(val=self.desc_v3.UcodeId)
    self.nvdev.NV_PFALCON2_FALCON_MOD_SEL.with_base(base).write(algo=self.nvdev.NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K)

    self.nvdev.NV_PFALCON_FALCON_BOOTVEC.with_base(base).write(self.desc_v3.IMEMVirtBase)

    # start cpu
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.NV_PFALCON_FALCON_CPUCTL.write(alias_startcpu=1)
    else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

    while self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'] == 0: pass

  def disable_ctx_req(self, base):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base, riscv=False):
    # print(hex(self.nvdev.NV_PFALCON_FALCON_HWCFG2.read()))
    # while not self.nvdev.NV_PFALCON_FALCON_HWCFG2.read_bitfields()['reset_ready']: time.sleep(0.1)
    time.sleep(1.5)

    engine_reg = self.nvdev.NV_PGSP_FALCON_ENGINE if base == self.falcon else self.nvdev.NV_PSEC_FALCON_ENGINE
    engine_reg.write(reset=1)
    time.sleep(0.1)
    engine_reg.write(reset=0)
    time.sleep(0.1)

    while self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['mem_scrubbing'] != 0: time.sleep(0.1)

    if riscv: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=1, valid=0, brfetch=1)
    elif self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['riscv'] == 1:
      self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=0)
      while self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).read_bitfields()['valid'] != 1: pass
      self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(0x192000a1) # TODO: put real chip id here

class NV_GSP:
  def __init__(self, nvdev): self.nvdev = nvdev

  def init_sw(self):
    self.libos_desc_sysmem = self.init_libos_args()
    self.init_wpr_meta()

  def init_rm_args(self):
    rm_args, rm_args_sysmem = self.nvdev._alloc_boot_struct(nv.GSP_ARGUMENTS_CACHED)

    # queue messages init
    queue_sizes = 0x40000 + 0x40000 # cmd + stat
    pt_size = round_up((pt_entries:=(queue_sizes >> 12)) * 8, 0x1000)
    shared_buf_size = pt_size + queue_sizes

    shared_va, shared_sysmem = alloc_sysmem(shared_buf_size, contigous=False)
    for i,sysmem in enumerate(shared_sysmem): to_mv(shared_va + i * 0x8, 0x8).cast('Q')[0] = sysmem

    rm_args.messageQueueInitArguments.sharedMemPhysAddr = shared_sysmem[0]
    rm_args.messageQueueInitArguments.pageTableEntryCount = pt_entries
    rm_args.messageQueueInitArguments.cmdQueueOffset = pt_size
    rm_args.messageQueueInitArguments.statQueueOffset = pt_size + 0x40000
    rm_args.bDmemStack = True

    rm_args.srInitArguments.bInPMTransition = False
    rm_args.srInitArguments.oldLevel = 0
    rm_args.srInitArguments.flags = 0

    rm_args.gpuInstance = 0

    self.command_queue_st_va = shared_va + pt_size
    self.status_queue_st_va = shared_va + pt_size + 0x40000

    self.command_queue_va = self.command_queue_st_va + 0x1000

    self.command_queue_tx = nv.msgqTxHeader.from_address(self.command_queue_st_va)
    self.command_queue_rx = nv.msgqRxHeader.from_address(self.command_queue_st_va + ctypes.sizeof(nv.msgqTxHeader))

    self.command_queue_tx.version = 0
    self.command_queue_tx.size = 0x40000
    self.command_queue_tx.entryOff = 0x1000
    self.command_queue_tx.msgSize = 0x1000
    self.command_queue_tx.msgCount = (0x40000 - 0x1000) // 0x1000
    self.command_queue_tx.writePtr = 0
    self.command_queue_tx.flags = 1
    self.command_queue_tx.rxHdrOff = ctypes.sizeof(nv.msgqTxHeader)

    self.status_queue_tx = nv.msgqTxHeader.from_address(shared_va + pt_size + 0x40000)
    return rm_args_sysmem

  def init_libos_args(self):
    logbuf_va, logbuf_sysmem = alloc_sysmem((2 << 20), contigous=True)
    libos_init, libos_init_sysmem = alloc_sysmem(0x1000, contigous=True)

    self.logs = {}

    off_sz = 0
    for i,(name, size) in enumerate([("INIT", 0x10000), ("INTR", 0x10000), ("RM", 0x10000), ("MNOC", 0x10000), ("KRNL", 0x10000)]):
      to_mv(logbuf_va + off_sz + 8, 8).cast('Q')[0] = logbuf_sysmem[0] + off_sz # this is radix3 pt address, we can ignore it, it's contig now.

      arg = nv.LibosMemoryRegionInitArgument.from_address(libos_init + i * ctypes.sizeof(nv.LibosMemoryRegionInitArgument))
      arg.kind = nv.LIBOS_MEMORY_REGION_CONTIGUOUS
      arg.loc = nv.LIBOS_MEMORY_REGION_LOC_SYSMEM
      arg.size = size
      arg.id8 = int.from_bytes(bytes("LOG" + name, 'utf-8'), 'big')
      arg.pa = logbuf_sysmem[0] + off_sz

      self.logs[name] = to_mv(logbuf_va + off_sz, size)
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

  def init_gsp_image(self):
    fwpath = "/lib/firmware/nvidia/560.35.05/gsp_ga10x.bin"
    fwbytes = FileIOInterface(fwpath, os.O_RDONLY).read(binary=True)
    assert len(fwbytes) == 0x288f630

    _, sections, _ = elf_loader(fwbytes)
    image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == (".fwsignature_ad10x")))

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

  def init_boot_binary_image(self):
    text = self.nvdev._download("src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmBoot_AD102.c")

    def _find_section(name):
      sl = text[text.find(name) + len(name) + 7:]
      return bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))

    image = _find_section("kgspBinArchiveGspRmBoot_AD102_ucode_image_prod_data")
    image = gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + image)
    desc = _find_section("kgspBinArchiveGspRmBoot_AD102_ucode_desc_prod_data")
    desc = gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + desc)

    image_va, image_sysmem = alloc_sysmem(len(image), contigous=True)
    to_mv(image_va, len(image))[:] = image

    # assert len(image) == 0x31000
    return image_sysmem[0], len(image), nv.RM_RISCV_UCODE_DESC.from_buffer_copy(desc)

  def init_wpr_meta(self):
    self.wpr_meta, self.wpr_meta_sysmem = self.nvdev._alloc_boot_struct(nv.GspFwWprMeta)
    self.gsp_radix3, self.gsp_signature, self.gsp_image_len = self.init_gsp_image()
    self.boot_sysmem, self.boot_image_len, self.boot_desc = self.init_boot_binary_image()

    self.wpr_meta.vgaWorkspaceOffset = 0x5ff300000
    self.wpr_meta.vgaWorkspaceSize = 0x100000

    self.wpr_meta.sizeOfRadix3Elf = self.gsp_image_len
    assert self.gsp_image_len == 0x2889000

    self.wpr_meta.gspFwWprEnd = 0x5ff300000

    self.wpr_meta.frtsSize = 0x100000
    self.wpr_meta.frtsOffset = 0x5ff200000

    self.wpr_meta.sizeOfBootloader = self.boot_image_len
    assert self.boot_image_len == 0x9000, f"Bootloader image size is {self.boot_image_len:x} (not 0x9000), check the boot binary"
    self.wpr_meta.bootBinOffset = 0x5ff1f7000

    self.wpr_meta.gspFwOffset = 0x5fc960000
    self.wpr_meta.gspFwHeapOffset = 0x5f4800000
    self.wpr_meta.gspFwHeapSize = 0x8100000

    self.wpr_meta.gspFwHeapVfPartitionCount = 0

    self.wpr_meta.gspFwWprStart = 0x5f4700000

    self.wpr_meta.nonWprHeapSize = 0x100000
    self.wpr_meta.nonWprHeapOffset = 0x5f4600000

    self.wpr_meta.gspFwRsvdStart = 0x5f4600000

    self.wpr_meta.sysmemAddrOfRadix3Elf = self.gsp_radix3
    self.wpr_meta.sysmemAddrOfBootloader = self.boot_sysmem

    self.wpr_meta.bootloaderCodeOffset = self.boot_desc.monitorCodeOffset
    self.wpr_meta.bootloaderDataOffset = self.boot_desc.monitorDataOffset
    self.wpr_meta.bootloaderManifestOffset = self.boot_desc.manifestOffset
    assert self.wpr_meta.bootloaderCodeOffset == 0x4800

    self.wpr_meta.sysmemAddrOfSignature = self.gsp_signature[0]
    self.wpr_meta.sizeOfSignature = 0x1000

    self.wpr_meta.revision = nv.GSP_FW_WPR_META_REVISION
    self.wpr_meta.magic = nv.GSP_FW_WPR_META_MAGIC

  def init_hw(self):
    pass
