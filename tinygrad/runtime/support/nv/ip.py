from __future__ import annotations
import ctypes, time, contextlib, importlib, functools, os, array, gzip, struct, itertools
from tinygrad.runtime.autogen.nv import nv
from tinygrad.helpers import to_mv, data64, lo32, hi32, DEBUG, round_up, round_down, mv_address
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.support.nvd import alloc_sysmem
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import nv_gpu
from tinygrad.device import CPUProgram
from hexdump import hexdump

class NVRpcQueue2:
  def __init__(self, gsp:NV_GSP, va:int, completion_q_va:int|None=None):
    self.tx = nv.msgqTxHeader.from_address(va)
    while self.tx.entryOff != 0x1000: pass # wait for the tx header to be initialized

    if completion_q_va is not None: self.rx = nv.msgqRxHeader.from_address(completion_q_va + nv.msgqTxHeader.from_address(completion_q_va).rxHdrOff)

    self.gsp, self.va, self.queue_va, self.seq = gsp, va, va + self.tx.entryOff, 0
    self.queue_mv = to_mv(self.queue_va, self.tx.msgSize * self.tx.msgCount)

  def _checksum(self, data):
    if (pad_len:=(-len(data)) % 8): data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from('Q', data, offset)[0]
    return ((checksum >> 32) & 0xFFFFFFFF) ^ (checksum & 0xFFFFFFFF)

  def send_rpc(self, func, msg, wait=False):
    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3<<24), function=func, length=len(msg) + 0x20)

    # simple put rpc
    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=round_up(len(msg), self.tx.msgSize) // self.tx.msgSize, seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = bytes(phdr) + msg

    off = self.tx.writePtr * self.tx.msgSize
    self.queue_mv[off:off+len(msg)] = msg
    self.tx.writePtr += round_up(len(msg), self.tx.msgSize) // self.tx.msgSize
    CPUProgram.atomic_lib.atomic_thread_fence(5)

    self.seq += 1
    self.gsp.nvdev.wreg(0x110c00, 0x0) # TODO

  def wait_resp(self, cmd) -> tuple[int, memoryview]:
    while True:
      CPUProgram.atomic_lib.atomic_thread_fence(5)
      if self.rx.readPtr == self.tx.writePtr: continue

      off = self.rx.readPtr * self.tx.msgSize
      x = nv.rpc_message_header_v.from_address(self.queue_va + off + 0x30)

      # Handling special functions
      if x.function == 0x1002: self.gsp.run_cpu_seq(self.queue_mv[off+0x50:off+0x50+x.length])
      # if x.function == 0x1020: hexdump(to_mv(self.content_va+off+0x50, x.length))

      self.rx.readPtr = (self.rx.readPtr + round_up(x.length, self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
      CPUProgram.atomic_lib.atomic_thread_fence(5)

      print(f"RPC message: {x.function:x}, {x.length}, {x.rpc_result}, {x.rpc_result_private} {self.tx.writePtr} {self.rx.readPtr}")

      if x.function == cmd: return x.rpc_result, self.queue_mv[off+0x50:off+0x50+x.length]

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

    self.prep_ucode()
    self.prep_booter()

  def prep_ucode(self):
    expansion_rom_off, bit_addr = 0x14e00, 0x1b0
    vbios_bytes = bytes(array.array('I', self.nvdev.mmio[0x00300000//4:(0x00300000+0x98e00)//4]))

    bit_header = nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[bit_addr:bit_addr + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
    assert bit_header.Signature == 0x00544942, f"Invalid BIT header signature {hex(bit_header.Signature)}"

    for i in range(bit_header.TokenEntries):
      bit = nv.BIT_TOKEN_V1_00.from_buffer_copy(vbios_bytes[bit_addr + bit_header.HeaderSize + i * bit_header.TokenSize:])
      if bit.TokenId != nv.BIT_TOKEN_FALCON_DATA or bit.DataVersion != 2 or bit.DataSize < nv.BIT_DATA_FALCON_DATA_V2_SIZE_4: continue

      falcon_data = nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(vbios_bytes[bit.DataPtr & 0xffff:])
      ucode_hdr = nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(vbios_bytes[(table_ptr:=expansion_rom_off + falcon_data.FalconUcodeTablePtr):])
      for j in range(ucode_hdr.EntryCount):
        ucode_entry = nv.FALCON_UCODE_TABLE_ENTRY_V1.from_buffer_copy(vbios_bytes[table_ptr + ucode_hdr.HeaderSize + j * ucode_hdr.EntrySize:])
        if ucode_entry.ApplicationID != nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue
        
        ucode_desc_hdr = nv.FALCON_UCODE_DESC_HEADER.from_buffer_copy(vbios_bytes[expansion_rom_off + ucode_entry.DescPtr:])
        ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr
        ucode_desc_size = ucode_desc_hdr.vDesc >> 16
    
    self.desc_v3 = nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucode_desc_off:ucode_desc_off + ucode_desc_size])

    sig_total_size = ucode_desc_size - nv.FALCON_UCODE_DESC_V3_SIZE_44
    signature = vbios_bytes[ucode_desc_off + nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
    image = vbios_bytes[ucode_desc_off + ucode_desc_size:][:round_up(self.desc_v3.StoredSize, 256)]
    assert ucode_desc_off + ucode_desc_size == 0x45488
    assert len(signature) == 0x300 and len(image) == 0x10300

    self.frts_offset = 0x5ff200000
    read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = nv.FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=self.frts_offset >> 12, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = nv.FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)

    def __patch(cmd_id, cmd):
      patched_image = bytearray(image)

      # Patch image
      dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset:=self.desc_v3.IMEMLoadSize+0xae0):])
      dmem.init_cmd = cmd_id
      patched_image[dmem_mapper_offset : dmem_mapper_offset+len(bytes(dmem))] = bytes(dmem)
      patched_image[(cmd_off:=self.desc_v3.IMEMLoadSize+dmem.cmd_in_buffer_offset) : cmd_off+len(cmd)] = cmd
      patched_image[(sig_off:=self.desc_v3.IMEMLoadSize+self.desc_v3.PKCDataOffset) : sig_off+0x180] = signature[0x180:]

      return alloc_sysmem(len(patched_image), contigous=True, data=patched_image)

    self.frts_image_va, self.frts_image_sysmem = __patch(0x15, bytes(frts_cmd))

  def prep_booter(self):
    text = self.nvdev._download("src/nvidia/generated/g_bindata_kgspGetBinArchiveBooterLoadUcode_AD102.c")

    image = self.nvdev.extract_fw(text, "kgspBinArchiveBooterLoadUcode_AD102_image_prod_data")
    sig = self.nvdev.extract_fw(text, "kgspBinArchiveBooterLoadUcode_AD102_sig_prod_data")
    patch_loc = int.from_bytes(self.nvdev.extract_fw(text, "kgspBinArchiveBooterLoadUcode_AD102_patch_loc_data"), 'little')
    sig_len = len(sig) // int.from_bytes(self.nvdev.extract_fw(text, "kgspBinArchiveBooterLoadUcode_AD102_num_sigs_data"), 'little')

    patched_image = bytearray(image)
    patched_image[patch_loc:patch_loc+sig_len] = sig[:sig_len]
    self.booter_image_va, self.booter_image_sysmem = alloc_sysmem(len(patched_image), contigous=True, data=patched_image)

  def init_hw(self):
    self.falcon, self.sec2 = 0x00110000, 0x00840000

    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_sysmem[0], code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)

    self.reset(self.falcon, riscv=True)

    # set up the mailbox
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_args_sysmem[0]))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_args_sysmem[0]))

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
    # time.sleep(1)

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
    self.handle_gen = itertools.count(0xcf000000)
    self.init_rm_args()
    self.init_libos_args()
    self.init_wpr_meta()

    # Prefill cmd queue with info for gsp to start.
    self.rpc_set_gsp_system_info()
    self.rpc_set_registry_table()

  def init_rm_args(self, queue_size=0x40000):
    # Alloc queues
    pte_cnt = ((queue_pte_cnt:=(queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    queues_va, queues_sysmem = alloc_sysmem(pt_size + queue_size * 2, contigous=False)

    # Fill up ptes
    for i, sysmem in enumerate(queues_sysmem): to_mv(queues_va + i * 0x8, 0x8).cast('Q')[0] = sysmem

    # Fill up arguments
    queue_args = nv.MESSAGE_QUEUE_INIT_ARGUMENTS(sharedMemPhysAddr=queues_sysmem[0], pageTableEntryCount=pte_cnt, cmdQueueOffset=pt_size,
      statQueueOffset=pt_size + queue_size)
    rm_args, self.rm_args_sysmem = self.nvdev._alloc_boot_struct_2(nv.GSP_ARGUMENTS_CACHED(bDmemStack=True, messageQueueInitArguments=queue_args))

    # Build command queue header
    self.cmd_q_va, self.stat_q_va = queues_va + pt_size, queues_va + pt_size + queue_size

    cmd_q_tx = nv.msgqTxHeader(version=0, size=queue_size, entryOff=0x1000, msgSize=0x1000, msgCount=(queue_size - 0x1000) // 0x1000,
      writePtr=0, flags=1, rxHdrOff=ctypes.sizeof(nv.msgqTxHeader))
    to_mv(self.cmd_q_va, ctypes.sizeof(nv.msgqTxHeader))[:] = bytes(cmd_q_tx)

    self.cmd_q = NVRpcQueue2(self, self.cmd_q_va, None)

  def init_libos_args(self):
    _, logbuf_sysmem = alloc_sysmem((2 << 20), contigous=True)
    libos_args_va, self.libos_args_sysmem = alloc_sysmem(0x1000, contigous=True)

    libos_structs = (nv.LibosMemoryRegionInitArgument * 6).from_address(libos_args_va)
    for i, name in enumerate(["INIT", "INTR", "RM", "MNOC", "KRNL"]):
      libos_structs[i] = nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x10000,
        id8=int.from_bytes(bytes(f"LOG{name}", 'utf-8'), 'big'), pa=logbuf_sysmem[0] + 0x10000 * i)

    libos_structs[5] = nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x1000,
        id8=int.from_bytes(bytes("RMARGS", 'utf-8'), 'big'), pa=self.rm_args_sysmem)

  def init_gsp_image(self):
    fwbytes = FileIOInterface("/lib/firmware/nvidia/560.35.05/gsp_ga10x.bin", os.O_RDONLY).read(binary=True)

    _, sections, _ = elf_loader(fwbytes)
    self.gsp_image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == (".fwsignature_ad10x")))

    # Build radix3
    npages = [0, 0, 0, round_up(len(self.gsp_image), 0x1000) // 0x1000]
    for i in range(3, 0, -1): npages[i-1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1

    offsets = [sum(npages[:i]) * 0x1000 for i in range(4)]
    radix_va, self.gsp_radix3_sysmem = alloc_sysmem(offsets[-1] + len(self.gsp_image), contigous=False)

    # Copy image
    to_mv(radix_va + offsets[-1], len(self.gsp_image))[:] = self.gsp_image

    # Copy level and image pages.
    for i in range(0, 3):
      cur_offset = sum(npages[:i+1])
      to_mv(radix_va + offsets[i], npages[i+1] * 8).cast('Q')[:] = array.array('Q', self.gsp_radix3_sysmem[cur_offset:cur_offset+npages[i+1]])

    # Copy signature
    self.gsp_signature_va, self.gsp_signature_sysmem = alloc_sysmem(len(signature), contigous=True, data=signature)

  def init_boot_binary_image(self):
    text = self.nvdev._download("src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmBoot_AD102.c")

    self.booter_image = self.nvdev.extract_fw(text, "kgspBinArchiveGspRmBoot_AD102_ucode_image_prod_data")
    self.booter_desc = nv.RM_RISCV_UCODE_DESC.from_buffer_copy(self.nvdev.extract_fw(text, "kgspBinArchiveGspRmBoot_AD102_ucode_desc_prod_data"))
    _, self.booter_sysmem = alloc_sysmem(len(self.booter_image), contigous=True, data=self.booter_image)

  def init_wpr_meta(self):
    self.init_gsp_image()
    self.init_boot_binary_image()

    m = nv.GspFwWprMeta(revision=nv.GSP_FW_WPR_META_REVISION, magic=nv.GSP_FW_WPR_META_MAGIC,
      fbSize=self.nvdev.vram_size, vgaWorkspaceSize=(vga_sz:=0x100000), vgaWorkspaceOffset=(vga_off:=self.nvdev.vram_size-vga_sz),
      sizeOfRadix3Elf=(radix3_sz:=len(self.gsp_image)), gspFwWprEnd=vga_off, frtsSize=(frts_sz:=0x100000), frtsOffset=(frts_off:=vga_off-frts_sz),
      sizeOfBootloader=(boot_sz:=len(self.booter_image)), bootBinOffset=(boot_off:=frts_off-boot_sz),
      gspFwOffset=(gsp_off:=round_down(boot_off-radix3_sz, 0x10000)), gspFwHeapSize=(gsp_heap_sz:=0x8100000),
      gspFwHeapOffset=(gsp_heap_off:=round_down(gsp_off-gsp_heap_sz, 0x100000)), gspFwWprStart=(wpr_start:=round_down(gsp_heap_off-0x1000, 0x100000)),
      nonWprHeapSize=(non_wpr_sz:=0x100000), nonWprHeapOffset=(non_wpr_off:=round_down(wpr_start-non_wpr_sz, 0x100000)), gspFwRsvdStart=non_wpr_off,
      sysmemAddrOfRadix3Elf=self.gsp_radix3_sysmem[0], sysmemAddrOfBootloader=self.booter_sysmem[0], sysmemAddrOfSignature=self.gsp_signature_sysmem[0],
      bootloaderCodeOffset=self.booter_desc.monitorCodeOffset, bootloaderDataOffset=self.booter_desc.monitorDataOffset,
      bootloaderManifestOffset=self.booter_desc.manifestOffset, sizeOfSignature=0x1000)
    self.wpr_meta, self.wpr_meta_sysmem = self.nvdev._alloc_boot_struct_2(m)

  def init_golden_image(self):
    self.root_class = self.rpc_rm_alloc(hParent=0x0, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())
    self.device = self.rpc_rm_alloc(hParent=self.client, hClass=nv_gpu.NV01_DEVICE_0, params=nv_gpu.NV0080_ALLOC_PARAMETERS(hClientShare=self.client))
    self.subdevice = self.rpc_rm_alloc(hParent=self.device, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS()) 
    self.vaspace = self.rpc_rm_alloc(hParent=self.device, hClass=nv_gpu.FERMI_VASPACE_A, params=nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS())

    # reserve 512MB for the reserved PDES
    resv = self.nvdev.mm.valloc(512 << 20, contigous=True, nomap=True)

    bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(pageSize=(512<<20), numLevelsToCopy=3,
      virtAddrLo=resv.va_addr, virtAddrHi=resv.va_addr+(512<<20)-1)
    bufs_p.levels[0] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=self.nvdev.mm.root_page_table.paddr, size=0x20, pageShift=47, aperture=1)
    bufs_p.levels[1] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=(8<<30)+0x1000, size=0x1000, pageShift=38, aperture=1)
    bufs_p.levels[2] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=(8<<30)+0x2000, size=0x1000, pageShift=29, aperture=1)
    self.rpc_rm_control(hObject=self.vaspace, cmd=0x90f10106, params=bufs_p)




  def init_hw(self):
    self.stat_q = NVRpcQueue2(self, self.stat_q_va, self.cmd_q_va)
    self.cmd_q.rx = nv.msgqRxHeader.from_address(self.stat_q.va + self.stat_q.tx.rxHdrOff)

    # self.status_q = NVRpcQueue(self, self.status_queue_va, self.status_queue_tx, self.command_queue_rx)
    self.stat_q.wait_resp(0x1001)
    print("GSP init done")

    self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
    self.rpc_get_gsp_static_info()

    self.init_golden_image()
    # self.nvdev.mm.valloc((1 << 20)) # TODO: forbid 0 allocs
    # return # exit...

    # self.root_class = self.rpc_rm_alloc(hParent=0x0, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())

    # self.device = self.rpc_rm_alloc(hParent=self.client, hClass=nv_gpu.NV01_DEVICE_0, params=nv_gpu.NV0080_ALLOC_PARAMETERS(hClientShare=self.client))
    # self.subdevice = self.rpc_rm_alloc(hParent=self.device, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS()) 
    # self.vaspace = self.rpc_rm_alloc(hParent=self.device, hClass=nv_gpu.FERMI_VASPACE_A, params=nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS())

    # resv = self.nvdev.mm.valloc(512 << 20, contigous=True, nomap=True) # reserve 512MB for the reserved PDES

    gpfifo_area = self.nvdev.mm.valloc(2<<20, contigous=True)
    ramfc_alloc = self.nvdev.mm.valloc(2<<20, contigous=True)
    ringbuf = self.nvdev.mm.valloc(4<<20, contigous=True)

    # set internally-owned page table
    # bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(pageSize=(512<<20), virtAddrLo=resv.va_addr, virtAddrHi=resv.va_addr+(512<<20)-1,
    #   numLevelsToCopy=3)
    # bufs_p.levels[0] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=self.nvdev.mm.root_page_table.paddr, size=0x20, pageShift=47, aperture=1)
    # bufs_p.levels[1] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=(8<<30)+0x1000, size=0x1000, pageShift=38, aperture=1)
    # bufs_p.levels[2] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=(8<<30)+0x2000, size=0x1000, pageShift=29, aperture=1)
    # self.rpc_rm_control(hObject=self.vaspace, cmd=0x90f10106, params=bufs_p)
    # self.rpc_set_page_directory(device=self.device, hVASpace=self.vaspace, pdir_paddr=self.nvdev.mm.root_page_table.paddr)

    # channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    # self.chgr = self.rpc_rm_alloc(hParent=self.device, hClass=nv_gpu.KEPLER_CHANNEL_GROUP_A, params=channel_params)

    # fault_bufs_p = nv_gpu.NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS(numValidEntries=2)
    # self.fault_bufs = []
    # for i in range(2):
    #   method_va, method_sysmem = alloc_sysmem(0x5000, contigous=True)
    #   fault_bufs_p.methodBufferMemdesc[i] = nv_gpu.NV2080_CTRL_INTERNAL_MEMDESC_INFO(base=method_sysmem[0], size=0x5000, addressSpace=1, cpuCacheAttrib=0, alignment=1)
    #   self.fault_bufs.append(to_mv(method_va, 0x5000))
    # self.rpc_rm_control(hObject=self.device, cmd=nv_gpu.NVA06C_CTRL_CMD_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS, params=fault_bufs_p)

    # ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=self.vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    # self.ctx_share = self.rpc_rm_alloc(hParent=self.chgr, hClass=nv_gpu.FERMI_CONTEXT_SHARE_A, params=ctxshare_params)

    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x400 * 8, size=0x400, addressSpace=2, cacheAttrib=0)

    notifier_va, notifier_sysmem = alloc_sysmem(0x1000, contigous=True)
    notifier = nv_gpu.NV_MEMORY_DESC_PARAMS(base=notifier_sysmem[0], size=0x0000000ECC, addressSpace=1, cacheAttrib=0)

    ramfc = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
    instblock = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)

    method_va, method_sysmem = alloc_sysmem(0x5000, contigous=True)
    method_buffer = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_sysmem[0], size=0x5000, addressSpace=1, cacheAttrib=0)

    # tlb invalidation
    # self.nvdev.wreg(0x00B80000 + 0x000030B0, (1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=0x0, hObjectBuffer=0x0, hPhysChannelGroup=0x0,
      gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=0x400, engineType=0x1, cid=3, hVASpace=self.vaspace, hContextShare=0x0,
      userdOffset=(ctypes.c_uint64*8)(0x400 * 8), userdMem=userd, errorNotifierMem=notifier, instanceMem=instblock, ramfcMem=ramfc,
      mthdbufMem=method_buffer, internalFlags=0x1a, flags=0x200320, ProcessID=1, SubProcessID=1)
    self.ch_gpfifo = self.rpc_rm_alloc(hParent=self.device, hClass=nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params=gg_params)

    bufs_info = [(0, 0x237000), (2, 24576), (3, 12288),  (4, 131072), (5, 39845888), (6, 524288), (9, 65536), (10, 524288), (11, 524288)]
    self.pro_bufs_info = []

    prom = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(bufs_info), engineType=0x1, hChanClient=self.client, hObject=self.ch_gpfifo)
    for i,(buf,size) in enumerate(bufs_info):
      x = self.nvdev.mm.valloc(size, contigous=True) # reserve buffers
      self.pro_bufs_info.append((buf, x))  # store for later use
      # x = wb[buf].va_addr if buf in wb else self.nvdev.mm.alloc_vaddr(round_up(size, 2<<20))
      prom.promoteEntry[i].gpuVirtAddr = x.va_addr if buf not in [10] else 0
      prom.promoteEntry[i].gpuPhysAddr = x.paddrs[0][0] if buf not in [3,4,5,6] else 0
      prom.promoteEntry[i].size = size if prom.promoteEntry[i].gpuPhysAddr != 0 else 0
      prom.promoteEntry[i].bufferId = buf
      prom.promoteEntry[i].physAttr = 0x4 if prom.promoteEntry[i].gpuPhysAddr != 0 else 0x0
      prom.promoteEntry[i].bInitialize = prom.promoteEntry[i].gpuPhysAddr != 0
      prom.promoteEntry[i].bNonmapped = (prom.promoteEntry[i].gpuPhysAddr != 0 and prom.promoteEntry[i].gpuVirtAddr == 0)

      print(f"Buffer {buf} - GPU Virt Addr: {hex(prom.promoteEntry[i].gpuVirtAddr)}, GPU Phys Addr: {hex(prom.promoteEntry[i].gpuPhysAddr)}, Size: {size}, Phys Attr: {prom.promoteEntry[i].physAttr}, bInitialize: {prom.promoteEntry[i].bInitialize}, bNonmapped: {prom.promoteEntry[i].bNonmapped}")
    self.rpc_rm_control(hObject=self.subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom)

    self.rpc_rm_alloc(hParent=self.ch_gpfifo, hClass=nv_gpu.ADA_COMPUTE_A, params=None)
    self.rpc_rm_alloc(hParent=self.ch_gpfifo, hClass=nv_gpu.AMPERE_DMA_COPY_B, params=None)

    # params = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    # z = self.rpc_rm_control(hObject=self.ch_gpfifo, cmd=nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, params=params)
    # dbell = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS.from_buffer_copy(z).workSubmitToken
    # print(hex(dbell))
    # dbell = 3

    # params = nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
    # z = self.rpc_rm_control(hObject=self.ch_gpfifo, cmd=nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, params=params)
    # nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS.from_buffer_copy(z)

    return

    # Write into ring
    from tinygrad.runtime.ops_nvd import NVComputeQueue, NVCopyQueue, NVSignal
    # nvq = NVComputeQueue().setup(compute_class=nv_gpu.ADA_COMPUTE_A)
    nvq = NVComputeQueue().signal_2(ringbuf.va_addr+0xe0, 0xdeadbeef)
    cmd_bytes = bytes(array.array('I', nvq._q))
    # self.nvdev.vram[ringbuf.paddrs[0][0]:ringbuf.paddrs[0][0] + len(cmd_bytes)] = cmd_bytes

    cmdq_addr = ringbuf.va_addr
    lenq = len(nvq._q)

    # Write simple command to execute
    # self.nvdev.vram[gpfifo_area.paddrs[0][0]:gpfifo_area.paddrs[0][0]+8] = bytes(array.array('Q', [(cmdq_addr//4 << 2) | (lenq << 42) | (1 << 41)]))
    self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x400 * 8 + 0x8c] = 0x1 # move gpput
    self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x8c] = 0x1 # move gpput
    self.nvdev.wreg(0x00B80000 + 0x30090, dbell)

    print("sleeping")
    time.sleep(1)

    print("notifier:")
    hexdump(to_mv(notifier_va, 0x20))

    print("signal ringed:")
    hexdump(self.nvdev.vram[ringbuf.paddrs[0][0]+0xe0:ringbuf.paddrs[0][0]+0xe0 + 0x10])

    print(self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x400 * 8 + 0x88])
    print(self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x400 * 8 + 0x8c])

    # print(self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x88])
    # print(self.nvdev.vram[gpfifo_area.paddrs[0][0] + 0x8c])

    # exit(0)

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

        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_args_sysmem[0]))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_args_sysmem[0]))

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
    self.cmd_q.send_rpc(65, bytes(0x8b0))
    stat, resp = self.stat_q.wait_resp(65)
    self.client = 0xc1e00004 # int.from_bytes(resp[0x888:0x888+4], 'little')
    assert stat == 0
    # print(hex(self.client))

  def rpc_rm_alloc(self, hParent, hClass, params) -> int:
    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=self.client, hParent=hParent, hObject=(obj:=next(self.handle_gen)), hClass=hClass, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(103, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    stat, resp = self.stat_q.wait_resp(103)
    assert stat == 0
    # hexdump(resp[:0x80])
    # return resp[len(bytes(alloc_args)):]
    if hClass == 0x0: return self.client # init root, return client
    if hClass == nv_gpu.FERMI_VASPACE_A and self.client == 0xdead0000:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr)

    #   bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(pageSize=(512<<20), virtAddrLo=512<<20, virtAddrHi=((512<<20)*2)-1,
    #     numLevelsToCopy=3)
    #   bufs_p.levels[0] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=0x3332000, size=0x20, pageShift=47, aperture=1)
    #   bufs_p.levels[1] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=0x3333000, size=0x1000, pageShift=38, aperture=1)
    #   bufs_p.levels[2] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=0x3334000, size=0x1000, pageShift=29, aperture=1)
    #   self.rpc_rm_control(hObject=obj, cmd=0x90f10106, params=bufs_p)
      
    if hClass == nv_gpu.NV20_SUBDEVICE_0: self.subdevice = obj # save subdevice handle
    if hClass == nv_gpu.KEPLER_CHANNEL_GROUP_A:
      self.channel_group = obj # save channel group handle
    #   fault_bufs_p = nv_gpu.NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS(numValidEntries=2)
    #   self.fault_bufs = []
    #   for i in range(2):
    #     method_va, method_sysmem = alloc_sysmem(0x5000, contigous=True)
    #     fault_bufs_p.methodBufferMemdesc[i] = nv_gpu.NV2080_CTRL_INTERNAL_MEMDESC_INFO(base=method_sysmem[0], size=0x5000, addressSpace=1, cpuCacheAttrib=0, alignment=1)
    #     self.fault_bufs.append(to_mv(method_va, 0x5000))
    #   self.rpc_rm_control(hObject=obj, cmd=nv_gpu.NVA06C_CTRL_CMD_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS, params=fault_bufs_p)
    # self.channel_group = obj # save channel group handle

    if hClass == nv_gpu.AMPERE_CHANNEL_GPFIFO_A and self.client == 0xdead0000:
      xx = {}
      bufs_info = [(0, 0x237000), (1, 0x18700), (2, 0x6000)]

      prom = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(bufs_info), engineType=0x1, hChanClient=self.client, hObject=obj)
      for i,(buf,size) in enumerate(bufs_info):
        x = self.nvdev.mm.valloc(size, contigous=True) # reserve buffers
        xx[buf] = x
        prom.promoteEntry[i].gpuVirtAddr = 0x0
        prom.promoteEntry[i].gpuPhysAddr = x.paddrs[0][0]
        prom.promoteEntry[i].size = size
        prom.promoteEntry[i].bufferId = buf
        prom.promoteEntry[i].bInitialize = 0x1
        prom.promoteEntry[i].bNonmapped = 0x1
        prom.promoteEntry[i].physAttr = 0x4
      self.rpc_rm_control(hObject=self.subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom)

      bufs_info = [(0, 0x237000), (1, 0x18700), (2, 24576), (3, 12288),  (4, 131072), (5, 39845888), (6, 524288), (9, 65536), (10, 524288), (11, 524288)]
      prom = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(bufs_info), engineType=0x1, hChanClient=self.client, hObject=obj)
      for i,(buf,size) in enumerate(bufs_info):
        x = xx[buf] if buf in xx else self.nvdev.mm.valloc(size, contigous=True) # reserve buffers

        prom.promoteEntry[i].gpuVirtAddr = x.va_addr
        prom.promoteEntry[i].gpuPhysAddr = 0x0
        prom.promoteEntry[i].size = 0x0
        prom.promoteEntry[i].bufferId = buf
      self.rpc_rm_control(hObject=self.subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom)

    return obj

  def rpc_rm_control(self, hObject, cmd, params):
    control_args = nv.rpc_gsp_rm_control_v(hClient=self.client, hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(76, bytes(control_args) + (bytes(params) if params is not None else b''))
    stat, resp = self.stat_q.wait_resp(76)
    assert stat == 0
    return resp[len(bytes(control_args)):]

  def rpc_rm_dup(self, hParent, hObjectSrc, hObject):
    params = nv.struct_NVOS55_PARAMETERS_v03_00(hClient=self.client, hClientSrc=self.client, hParent=hParent, hObject=hObject, hObjectSrc=hObjectSrc, flags=0)
    alloc_args = nv.rpc_dup_object_v(params=params)
    self.cmd_q.send_rpc(21, bytes(alloc_args))
    stat, resp = self.stat_q.wait_resp(21)
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
    self.cmd_q.send_rpc(54, bytes(alloc_args))
    stat, resp = self.stat_q.wait_resp(54)
    assert stat == 0
    # hexdump(resp[:0x80])

  def rpc_set_gsp_system_info(self):
    data = nv.GspSystemInfo(gpuPhysAddr=0xf2000000, gpuPhysFbAddr=0x38060000000, gpuPhysInstAddr=0x38070000000,
      pciConfigMirrorBase=0x88000, pciConfigMirrorSize=0x1000, nvDomainBusDeviceFunc=0x100, bIsPassthru=1,
      PCIDeviceID=0x268410de, PCISubDeviceID=0x13b3196e, PCIRevisionID=0xa1, maxUserVa=0x7ffffffff000)
    self.cmd_q.send_rpc(72, bytes(data))

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
    self.cmd_q.send_rpc(73, bytes(header) + entries_bytes + data_bytes)