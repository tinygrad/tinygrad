import ctypes, time, contextlib, importlib, functools, os, array, gzip, struct
from tinygrad.runtime.autogen.nv import nv
from tinygrad.helpers import to_mv, data64, lo32, hi32, DEBUG, round_up, mv_address
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.support.nvd import alloc_sysmem
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import nv_gpu
from tinygrad.device import CPUProgram
from hexdump import hexdump

class NVRpcQueue:
  def __init__(self, gsp, content_va, tx, rx):
    self.gsp, self.content_va, self.tx, self.rx, self.seq = gsp, content_va, tx, rx, 0
    self.queue_mv = to_mv(content_va, tx.msgSize * tx.msgCount)

  def _checksum(self, data):
    pad_len = (-len(data)) % 8
    if pad_len: data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8):
      (value,) = struct.unpack_from('Q', data, offset)
      checksum ^= value
    return ((checksum >> 32) & 0xFFFFFFFF) ^ (checksum & 0xFFFFFFFF)

  def send_rpc(self, func, msg, wait=False):
    assert len(msg) < 0xf00

    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3<<24), function=func, length=len(msg) + 0x20)

    # simple put rpc
    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=1, seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = bytes(phdr) + msg

    off = self.tx.writePtr * 0x1000
    self.queue_mv[off:off+len(msg)] = msg
    self.tx.writePtr += 1

    self.seq += 1
    self.gsp.nvdev.wreg(0x110c00, 0x0) # TODO

    if wait:
      while self.tx.writePtr != self.rx.readPtr: CPUProgram.atomic_lib.atomic_thread_fence(5)
      return to_mv(self.content_va + off + 0x50, header.length)
  
  def wait_resp(self, cmd):
    while True:
      CPUProgram.atomic_lib.atomic_thread_fence(5)
      
      if self.rx.readPtr == self.tx.writePtr: continue

      off = self.rx.readPtr * 0x1000
      x = nv.rpc_message_header_v.from_address(self.content_va + off + 0x30)

      # Special functions
      if x.function == 0x1002: self.gsp.run_cpu_seq(to_mv(self.content_va+off+0x50, x.length))
      # if x.function == 0x1020: hexdump(to_mv(self.content_va+off+0x50, x.length))

      self.rx.readPtr = (self.rx.readPtr + round_up(x.length, 0x1000) // 0x1000) % self.tx.msgCount
      CPUProgram.atomic_lib.atomic_thread_fence(5)

      print(f"RPC message: {x.function:x}, {x.length}, {x.rpc_result}, {x.rpc_result_private} {self.tx.writePtr} {self.rx.readPtr}")

      if x.function == cmd: return x.rpc_result, to_mv(self.content_va+off+0x50, x.length)

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
    self.prep_booter()

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
    # self.sb_image_va, self.sb_image_sysmem = __patch(0x19, read_vbios_desc)

  def prep_booter(self):
    text = self.nvdev._download("src/nvidia/generated/g_bindata_kgspGetBinArchiveBooterLoadUcode_AD102.c")

    def _find_section(name):
      sl = text[text.find(name) + len(name) + 7:]
      return bytes.fromhex(sl[:sl.find("};")].strip().replace("0x", "").replace(",", "").replace(" ", "").replace("\n", ""))

    image = _find_section("kgspBinArchiveBooterLoadUcode_AD102_image_prod_data")
    image = gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + image)

    header = _find_section("kgspBinArchiveBooterLoadUcode_AD102_header_prod_data")
    header = gzip.decompress(struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3) + header)

    sig = _find_section("kgspBinArchiveBooterLoadUcode_AD102_sig_prod_data")
    patch_loc = int.from_bytes(_find_section("kgspBinArchiveBooterLoadUcode_AD102_patch_loc_data"), 'little')
    patch_sig = _find_section("kgspBinArchiveBooterLoadUcode_AD102_patch_sig_data")
    patch_md = _find_section("kgspBinArchiveBooterLoadUcode_AD102_patch_meta_data")
    num_sigs = int.from_bytes(_find_section("kgspBinArchiveBooterLoadUcode_AD102_num_sigs_data"), 'little')

    sig_len = len(sig) // num_sigs

    patched_image = bytearray(image)
    patched_image[patch_loc:patch_loc+sig_len] = sig[:sig_len]

    self.booter_image_va, self.booter_image_sysmem = alloc_sysmem(len(patched_image), contigous=True, data=patched_image)
    

  def init_hw(self):
    self.nvdev.wreg(0x00100c40, 0x00000000)
    # self.nvdev.wreg(0x00100c10, 0x01637ea0)

    self.nvdev.wreg(0x00088080, 0x00002937)
    self.nvdev.wreg(0x00088080, 0x00002937)

    self.nvdev.wreg(0x00009410, 0x1846cedb)
    self.nvdev.wreg(0x00009400, 0x70dc2a48)

    self.falcon, self.sec2 = 0x00110000, 0x00840000

    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_sysmem[0], code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)

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
    mbx = self.execute_hs(self.sec2, self.booter_image_sysmem[0], code_off=256, data_off=32256,
      imemPa=0x0, imemVa=0x100, imemSz=32000,
      dmemPa=0x0, dmemVa=0x0, dmemSz=24576,
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.nvdev.gsp.wpr_meta_sysmem)
    assert mbx[0] == 0x0, f"Booster failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}"

    self.nvdev.NV_PFALCON_FALCON_OS.with_base(self.falcon).write(0x0)
    assert self.nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(self.falcon).read_bitfields()['active_stat'] == 1, "GSP Core is not active"

    print('--- falcon boot done ---')

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
  
  def execute_hs(self, base, img_sysmem, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz, pkc_off, engid, ucodeid, mailbox=None):
    self.disable_ctx_req(base)

    self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma:=0].update(target=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM,
      mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=1, sec=1)
    self.execute_dma(base, cmd, dest=imemPa, mem_off=imemVa, sysmem=img_sysmem+code_off-imemVa, size=imemSz)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=0, sec=0)
    self.execute_dma(base, cmd, dest=dmemPa, mem_off=dmemVa, sysmem=img_sysmem+data_off-dmemVa, size=dmemSz)

    self.nvdev.NV_PFALCON2_FALCON_BROM_PARAADDR.with_base(base)[0].write(pkc_off)
    self.nvdev.NV_PFALCON2_FALCON_BROM_ENGIDMASK.with_base(base).write(engid)
    self.nvdev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.with_base(base).write(val=ucodeid)
    self.nvdev.NV_PFALCON2_FALCON_MOD_SEL.with_base(base).write(algo=self.nvdev.NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K)

    self.nvdev.NV_PFALCON_FALCON_BOOTVEC.with_base(base).write(imemVa)

    if mailbox is not None:
      self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).write(lo32(mailbox))
      self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).write(hi32(mailbox))

    # start cpu
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.NV_PFALCON_FALCON_CPUCTL.write(alias_startcpu=1)
    else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

    while self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'] == 0: pass

    if mailbox is not None:
      return self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).read(), self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).read()

  def disable_ctx_req(self, base):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base, riscv=False):
    # print(hex(self.nvdev.NV_PFALCON_FALCON_HWCFG2.read()))
    # while not self.nvdev.NV_PFALCON_FALCON_HWCFG2.read_bitfields()['reset_ready']: time.sleep(0.1)
    time.sleep(1)

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
    self.next_handle = 0xcf000000

  def init_rm_args(self):
    rm_args, rm_args_sysmem = self.nvdev._alloc_boot_struct(nv.GSP_ARGUMENTS_CACHED)

    # queue messages init
    queue_sizes = 0x40000 + 0x40000 # cmd + stat
    num_ptes = queue_sizes >> 12
    num_ptes += round_up(num_ptes * 8, 0x1000) // 0x1000
    pt_size = round_up(num_ptes * 8, 0x1000)
    shared_buf_size = pt_size + queue_sizes

    shared_va, shared_sysmem = alloc_sysmem(shared_buf_size, contigous=False)
    for i,sysmem in enumerate(shared_sysmem): to_mv(shared_va + i * 0x8, 0x8).cast('Q')[0] = sysmem

    rm_args.messageQueueInitArguments.sharedMemPhysAddr = shared_sysmem[0]
    rm_args.messageQueueInitArguments.pageTableEntryCount = num_ptes
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

    self.command_q = NVRpcQueue(self, self.command_queue_va, self.command_queue_tx, None) # will be set later

    data = nv.GspSystemInfo(gpuPhysAddr=0xf2000000, gpuPhysFbAddr=0x38060000000, gpuPhysInstAddr=0x38070000000,
      pciConfigMirrorBase=0x88000, pciConfigMirrorSize=0x1000, nvDomainBusDeviceFunc=0x100, bIsPassthru=1,
      PCIDeviceID=0x268410de, PCISubDeviceID=0x13b3196e, PCIRevisionID=0xa1, maxUserVa=0x7ffffffff000)
    self.command_q.send_rpc(72, bytes(data))
    self.rpc_set_registry_table()

    return rm_args_sysmem

  def init_libos_args(self):
    logbuf_va, logbuf_sysmem = alloc_sysmem((2 << 20), contigous=True)
    libos_init, libos_init_sysmem = alloc_sysmem(0x1000, contigous=True)

    self.logs = {}

    off_sz = 0
    for i,(name, size) in enumerate([("INIT", 0x10000), ("INTR", 0x10000), ("RM", 0x10000), ("MNOC", 0x10000), ("KRNL", 0x10000)]):
      for poff in range(0, size, 0x1000):
        to_mv(logbuf_va + off_sz + 8 + (poff // 0x1000) * 8, 8).cast('Q')[0] = logbuf_sysmem[0] + off_sz + poff

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

  def rpc_set_registry_table(self):
    dt = {'RMForcePcieConfigSave': 0x1, 'RMSecBusResetEnable': 0x1}

    hdr_size = ctypes.sizeof(nv.PACKED_REGISTRY_TABLE)
    entries_size = ctypes.sizeof(nv.PACKED_REGISTRY_ENTRY) * len(dt)

    entries_bytes = b''
    data_bytes = b''

    for k,v in dt.items():
      entry = nv.PACKED_REGISTRY_ENTRY(nameOffset=hdr_size + entries_size + len(data_bytes),
        type=nv.REGISTRY_TABLE_ENTRY_TYPE_DWORD, data=v, length=4)
      entries_bytes += bytes(entry)
      data_bytes += k.encode('utf-8') + b'\x00'

    header = nv.PACKED_REGISTRY_TABLE(size=hdr_size + len(entries_bytes) + len(data_bytes), numEntries=len(dt))
    self.command_q.send_rpc(73, bytes(header) + entries_bytes + data_bytes)

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

    sign_va, sign_sysmem = alloc_sysmem(len(signature), contigous=True, data=signature)
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

    self.wpr_meta.fbSize = 0x5ff400000

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

  def _alloc_handle(self) -> int:
    self.next_handle += 1
    return self.next_handle

  def init_hw(self):
    while self.status_queue_tx.entryOff != 0x1000: pass

    self.status_queue_va = self.status_queue_st_va + self.status_queue_tx.entryOff
    self.status_queue_rx = nv.msgqRxHeader.from_address(self.status_queue_st_va + self.status_queue_tx.rxHdrOff)

    self.command_q.rx = self.status_queue_rx
    self.status_q = NVRpcQueue(self, self.status_queue_va, self.status_queue_tx, self.command_queue_rx)
    self.status_q.wait_resp(0x1001)
    print("GSP init done")

    self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
    self.nvdev.wreg(0x00B80000 + 0x00000F40, 0x0) # not sure this is needed.
    time.sleep(0.1)

    self.rpc_get_gsp_static_info()
    # self.nvdev.mm.valloc((2 << 20)) # TODO: forbid 0 allocs
    return # exit...

    gpfifo_area = self.nvdev.mm.valloc(2<<20, contigous=True)
    ramfc_alloc = self.nvdev.mm.valloc(2<<20, contigous=True)
    ringbuf = self.nvdev.mm.valloc(4<<20, contigous=True)

    print('gpfifo', gpfifo_area.paddrs, gpfifo_area.va_addr)

    self.rpc_rm_alloc(hParent=0x0, hObject=0xca000000, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())

    self.rpc_rm_alloc(hParent=self.client, hObject=0xcf000000, hClass=nv_gpu.NV01_DEVICE_0, params=nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=0x0, hClientShare=0xc1e00004,
      vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES))
    self.rpc_rm_alloc(hParent=0xcf000000, hObject=0xcf000001, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS(subDeviceId=0x0)) 

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(index=0, vaBase=0x0, vaSize=0x1fffffb000000-(512<<20), bigPageSize=64*1024,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ALLOW_ZERO_ADDRESS)
    x = self.rpc_rm_alloc(hParent=0xcf000000, hObject=0xcf000002, hClass=nv_gpu.FERMI_VASPACE_A, params=vaspace_params)

    self.rpc_set_page_directory(device=0xcf000000, hVASpace=0xcf000002, pdir_paddr=self.nvdev.mm.root_page_table.paddr)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.rpc_rm_alloc(hParent=0xcf000000, hObject=0xcf000003, hClass=nv_gpu.KEPLER_CHANNEL_GROUP_A, params=channel_params)

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=0xcf000002, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    self.rpc_rm_alloc(hParent=0xcf000003, hObject=0xcf000004, hClass=nv_gpu.FERMI_CONTEXT_SHARE_A, params=ctxshare_params)

    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x400 * 8, size=0x400, addressSpace=2, cacheAttrib=0)

    notifier_va, notifier_sysmem = alloc_sysmem(0x1000, contigous=True)
    notifier = nv_gpu.NV_MEMORY_DESC_PARAMS(base=notifier_sysmem[0], size=0x0000000ECC, addressSpace=1, cacheAttrib=0)

    ramfc = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
    instblock = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)

    method_va, method_sysmem = alloc_sysmem(0x5000, contigous=True)
    method_buffer = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_sysmem[0], size=0x5000, addressSpace=1, cacheAttrib=0)

    # tlb invalidation
    self.nvdev.wreg(0x00B80000 + 0x000030B0, (1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=0x0, hObjectBuffer=0x0,
      gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=0x400, hContextShare=0xcf000004, engineType=0x1, cid=8, hVASpace=0x0,
      userdOffset=(ctypes.c_uint64*8)(0x400 * 8), userdMem=userd, errorNotifierMem=notifier, instanceMem=instblock, ramfcMem=ramfc,
      mthdbufMem=method_buffer, internalFlags=0x1d, flags=0x201020, ProcessID=1, SubProcessID=1)
    self.rpc_rm_alloc(hParent=0xcf000003, hObject=0xcf000005, hClass=nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params=gg_params)

    self.rpc_rm_alloc(hParent=0xcf000005, hObject=0xcf000006, hClass=nv_gpu.ADA_COMPUTE_A, params=None)
    self.rpc_rm_alloc(hParent=0xcf000005, hObject=0xcf000007, hClass=nv_gpu.AMPERE_DMA_COPY_B, params=None)

    params = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    z = self.rpc_rm_control(hObject=0xcf000005, cmd=nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, params=params)
    dbell = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS.from_buffer_copy(z).workSubmitToken

    params = nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
    z = self.rpc_rm_control(hObject=0xcf000003, cmd=nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, params=params)
    nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS.from_buffer_copy(z)

    # Write into ring
    from tinygrad.runtime.ops_nvd import NVComputeQueue, NVCopyQueue, NVSignal
    nvq = NVComputeQueue().signal(None, 0xdea1beef, ringbuf.va_addr + 0xe0)
    cmd_bytes = bytes(array.array('I', nvq._q))
    self.nvdev.vram[ringbuf.paddrs[0][0]:ringbuf.paddrs[0][0] + len(cmd_bytes)] = cmd_bytes

    cmdq_addr = ringbuf.va_addr
    lenq = len(nvq._q)

    # Write simple command to execute
    self.nvdev.vram[gpfifo_area.paddrs[0][0]:gpfifo_area.paddrs[0][0]+8] = bytes(array.array('Q', [(cmdq_addr//4 << 2) | (lenq << 42) | (1 << 41)]))
    self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x400 * 8 + 0x8c] = 0x1 # move gpput
    self.nvdev.wreg(0x00B80000 + 0x30090, dbell)

    print("sleeping")
    time.sleep(1)

    print("notifier:")
    hexdump(to_mv(notifier_va, 0x20))

    print("signal ringed:")
    hexdump(self.nvdev.vram[ringbuf.paddrs[0][0]+0xe0:ringbuf.paddrs[0][0]+0xe0 + 0x10])

    exit(0)

  def run_cpu_seq(self, seq_buf):
    hdr = nv.rpc_run_cpu_sequencer_v17_00.from_address(mv_address(seq_buf))
    cmd_buf = seq_buf[ctypes.sizeof(nv.rpc_run_cpu_sequencer_v17_00):].cast('I')

    cmd_idx = 0
    while cmd_idx < hdr.cmdIndex:
      op_code = cmd_buf[cmd_idx]
      cmd_idx += 1

      if op_code == 0x0: # reg write
        addr, val = cmd_buf[cmd_idx:cmd_idx + 2]
        self.nvdev.wreg(addr, val)
        cmd_idx += 2
      elif op_code == 0x1: # reg modify
        addr, val, mask = cmd_buf[cmd_idx:cmd_idx + 3]
        self.nvdev.wreg(addr, (self.nvdev.rreg(addr) & ~mask) | (val & mask))
        cmd_idx += 3
      elif op_code == 0x2: # reg poll
        addr, mask, val, timeout, error = cmd_buf[cmd_idx:cmd_idx + 5]
        while (self.nvdev.rreg(addr) & mask) != val: pass # TODO: timeout
        cmd_idx += 5
      elif op_code == 0x3: # delay us
        delay_us = cmd_buf[cmd_idx]
        time.sleep(delay_us / 1e6)
        cmd_idx += 1
      elif op_code == 0x4:
        addr, index = cmd_buf[cmd_idx:cmd_idx + 2]
        hdr.regSaveArea[index] = self.nvdev.rreg(addr)
        cmd_idx += 2
      elif op_code == 0x5: # core reset
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon)
        self.nvdev.flcn.disable_ctx_req(self.nvdev.flcn.falcon)
      elif op_code == 0x6: # core start
        if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(self.nvdev.flcn.falcon).read_bitfields()['alias_en'] == 1:
          self.nvdev.NV_PFALCON_FALCON_CPUCTL.write(alias_startcpu=1)
        else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(self.nvdev.flcn.falcon).write(startcpu=1)
      elif op_code == 0x7: # wait halted
        while self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(self.nvdev.flcn.falcon).read_bitfields()['halted'] == 0: pass
      elif op_code == 0x8: # core resume
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon, riscv=True)

        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_desc_sysmem))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_desc_sysmem))

        if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(self.nvdev.flcn.sec2).read_bitfields()['alias_en'] == 1:
          self.nvdev.wreg(self.nvdev.flcn.sec2 + self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
          # self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS.with_base(self.nvdev.flcn.sec2).write(alias_startcpu=1)
        else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(self.nvdev.flcn.sec2).write(startcpu=1)

        print(self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields())
        while self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields()['boot_stage_3_handoff'] == 0: pass

        mailbox = self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(self.nvdev.flcn.sec2).read()
        assert mailbox == 0x0, f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}"

        print("core booted")

      else: raise ValueError(f"Unknown op code {op_code} at index {cmd_idx}")

  ### RPCs

  def rpc_get_gsp_static_info(self):
    # TODO: nv.GspStaticConfigInfo is parsed wrong, size does not match C impl.
    self.command_q.send_rpc(65, bytes(0x8b0))
    stat, resp = self.status_q.wait_resp(65)
    self.client = 0xc1e00004 # int.from_bytes(resp[0x888:0x888+4], 'little')
    assert stat == 0
    # print(hex(self.client))

  def rpc_rm_alloc(self, hParent, hClass, params) -> int:
    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=self.client, hParent=hParent, hObject=(obj:=self._alloc_handle()), hClass=hClass, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.command_q.send_rpc(103, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    stat, resp = self.status_q.wait_resp(103)
    assert stat == 0
    # hexdump(resp[:0x80])
    # return resp[len(bytes(alloc_args)):]
    if hClass == 0x0: return self.client # init root, return client
    if hClass == nv_gpu.FERMI_VASPACE_A:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr)
      
    if hClass == nv_gpu.KEPLER_CHANNEL_GROUP_A:
      fault_bufs_p = nv_gpu.NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS(numValidEntries=2)
      self.fault_bufs = []
      for i in range(2):
        method_va, method_sysmem = alloc_sysmem(0x5000, contigous=True)
        fault_bufs_p.methodBufferMemdesc[i] = nv_gpu.NV2080_CTRL_INTERNAL_MEMDESC_INFO(base=method_sysmem[0], size=0x5000, addressSpace=1, cpuCacheAttrib=0, alignment=1)
        self.fault_bufs.append(to_mv(method_va, 0x5000))
      self.rpc_rm_control(hObject=obj, cmd=nv_gpu.NVA06C_CTRL_CMD_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS, params=fault_bufs_p)
    return obj

  def rpc_rm_control(self, hObject, cmd, params):
    control_args = nv.rpc_gsp_rm_control_v(hClient=self.client, hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.command_q.send_rpc(76, bytes(control_args) + (bytes(params) if params is not None else b''))
    stat, resp = self.status_q.wait_resp(76)
    assert stat == 0
    return resp[len(bytes(control_args)):]

  def rpc_rm_dup(self, hParent, hObjectSrc, hObject):
    params = nv.struct_NVOS55_PARAMETERS_v03_00(hClient=self.client, hClientSrc=self.client, hParent=hParent, hObject=hObject, hObjectSrc=hObjectSrc, flags=0)
    alloc_args = nv.rpc_dup_object_v(params=params)
    self.command_q.send_rpc(21, bytes(alloc_args))
    stat, resp = self.status_q.wait_resp(21)
    assert stat == 0
  
  def rpc_set_page_directory(self, device, hVASpace, pdir_paddr):
    # UVM depth   HW level                            VA bits
    # 0           PDE3                                48:47
    # 1           PDE2                                46:38
    # 2           PDE1 (or 512M PTE)                  37:29
    # 3           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 4           PTE_64K / PTE_4K                    20:16 / 20:12
    # So, top level is 4 entries (?). Flags field is all channels, vid mem.
    # subDeviceId = ID+1, 0 for BC
    params = nv.struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(physAddress=pdir_paddr,
      numEntries=4, flags=0x8, hVASpace=hVASpace, pasid=0xffffffff, subDeviceId=1, chId=0)
    alloc_args = nv.rpc_set_page_directory_v(hClient=self.client, hDevice=device, pasid=0xffffffff, params=params)
    self.command_q.send_rpc(54, bytes(alloc_args))
    stat, resp = self.status_q.wait_resp(54)
    assert stat == 0
    # hexdump(resp[:0x80])
