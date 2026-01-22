from __future__ import annotations
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref
assert sys.platform != 'win32'
from typing import cast, ClassVar
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQProgram, HCQSignal, BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, MOCKGPU, hcq_filter_visible_devices, hcq_profile
from tinygrad.uop.ops import sint
from tinygrad.device import BufferSpec, CompilerPair, CompilerSet
from tinygrad.helpers import DEBUG, getenv, mv_address, round_up, data64, data64_le, prod, OSX, to_mv, hi32, lo32, NV_CC, NV_PTX, NV_NAK, PROFILE
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, PTXCompiler, NVPTXCompiler, NVCompiler
from tinygrad.runtime.autogen import nv_570, nv_580, pci, mesa
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import System, PCIIfaceBase, MAP_FIXED
from tinygrad.renderer.nir import NAKRenderer
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

nv_gpu = nv_570 # default to 570

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_ACCESS_TYPE_")}

def nv_iowr(fd:FileIOInterface, nr, args, cmd=None):
  ret = fd.ioctl(cmd or ((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF)), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev:NVDevice, addr:int|None=None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)

    # Init fields from module
    if (pref:="NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref)+1:]: dt for name,dt in nv_gpu.__dict__.items() if name.startswith(pref) and isinstance(dt, tuple)},
        **{name[len(pref)+1:]+f"_{i}": dt(i) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith(pref) and callable(dt)}}

    self.mv, self.pref = (memoryview(bytearray(self.sz * 4)) if addr is None else to_mv(addr, self.sz * 4)), pref
    if kwargs: self.write(**kwargs)

  def _rw_bits(self, hi:int, lo:int, value:int|None=None):
    mask = ((1 << (width:=hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo//8:hi//8+1], "little")

    if value is None: return (num & mask) >> (lo % 8)

    if value >= (1 << width): raise ValueError(f"{value:#x} does not fit.")
    self.mv[lo//8:hi//8+1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi//8 - lo//8 + 1), "little")

  def write(self, **kwargs):
    for k,val in kwargs.items(): self._rw_bits(*QMD.fields[self.pref][k.upper()], value=val) # type: ignore [misc]

  def read(self, k, val=0): return self._rw_bits(*QMD.fields[self.pref][k.upper()])

  def field_offset(self, k): return QMD.fields[self.pref][k.upper()][1] // 8

  def set_constant_buf_addr(self, i, addr):
    if self.ver < 4: self.write(**{f'constant_buffer_addr_upper_{i}':hi32(addr), f'constant_buffer_addr_lower_{i}':lo32(addr)})
    else: self.write(**{f'constant_buffer_addr_upper_shifted6_{i}':hi32(addr >> 6), f'constant_buffer_addr_lower_shifted6_{i}':lo32(addr >> 6)})

class NVCommandQueue(HWQueue[HCQSignal, 'NVDevice', 'NVProgram', 'NVArgsState']):
  def __init__(self):
    self.active_qmd = None
    super().__init__()

  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def nvm(self, subchannel, mthd, *args, typ=2): self.q((typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args)

  def setup(self, compute_class=None, copy_class=None, local_mem_window=None, shared_mem_window=None, local_mem=None, local_mem_tpc_bytes=None):
    if compute_class: self.nvm(1, nv_gpu.NVC6C0_SET_OBJECT, compute_class)
    if copy_class: self.nvm(4, nv_gpu.NVC6C0_SET_OBJECT, copy_class)
    if local_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(local_mem_window))
    if shared_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    if local_mem: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(local_mem))
    if local_mem_tpc_bytes: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(local_mem_tpc_bytes), 0xff)
    return self

  def wait(self, signal:HCQSignal, value:sint=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value), (3 << 0) | (1 << 24)) # ACQUIRE | PAYLOAD_SIZE_64BIT
    self.active_qmd = None
    return self

  def timestamp(self, signal:HCQSignal): return self.signal(signal, 0)

  def bind(self, dev:NVDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i, value in enumerate(self._q): hw_view[i] = value

    # From now on, the queue is on the device for faster submission.
    self._q = hw_view

  def _submit_to_gpfifo(self, dev:NVDevice, gpfifo:GPFifo):
    if dev == self.binded_device: cmdq_addr = self.hw_page.va_addr
    else:
      cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4, 16)
      cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
      dev.cmdq[cmdq_wptr : cmdq_wptr + len(self._q)] = array.array('I', self._q)

    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.gpput[0] = (gpfifo.put_value + 1) % gpfifo.entries_count

    System.memory_barrier()
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1

class NVComputeQueue(NVCommandQueue):
  def memory_barrier(self):
    self.nvm(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, (1 << 12) | (1 << 4) | (1 << 0))
    self.active_qmd:QMD|None = None
    return self

  def exec(self, prg:NVProgram, args_state:NVArgsState, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    self.bind_args_state(args_state)

    qmd_buf = args_state.buf.offset(round_up(prg.constbufs[0][1], 1 << 8))
    qmd_buf.cpu_view().view(size=prg.qmd.mv.nbytes, fmt='B')[:] = prg.qmd.mv
    assert qmd_buf.va_addr < (1 << 40), f"large qmd addr {qmd_buf.va_addr:x}"

    qmd = QMD(dev=prg.dev, addr=qmd_buf.cpu_view().addr) # Save qmd for later update

    self.bind_sints_to_mem(*global_size, mem=qmd_buf.cpu_view(), fmt='I', offset=qmd.field_offset('cta_raster_width' if qmd.ver<4 else 'grid_width'))
    self.bind_sints_to_mem(*(local_size[:2]), mem=qmd_buf.cpu_view(), fmt='H', offset=qmd.field_offset('cta_thread_dimension0'))
    self.bind_sints_to_mem(local_size[2], mem=qmd_buf.cpu_view(), fmt='B', offset=qmd.field_offset('cta_thread_dimension2'))
    qmd.set_constant_buf_addr(0, args_state.buf.va_addr)

    if self.active_qmd is None:
      self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
      # if PROFILE >= 2:
      #   # PM_TRIGGER before kernel for PC sampling
      #   self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
      #   # Use inline QMD for PC sampling compatibility - sends QMD data directly in command stream
      #   qmd_addr_shifted = qmd_buf.va_addr >> 8
      #   qmd_data = list(qmd_buf.cpu_view().view(size=qmd.sz*4, fmt='I'))
      #   self.nvm(1, nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A, (qmd_addr_shifted >> 32) & 0xff, qmd_addr_shifted & 0xffffffff, *qmd_data)
      #   # With inline QMD, signal() will use SET_REPORT_SEMAPHORE fallback since active_qmd stays None
      #   return self
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    else:
      self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)

    self.active_qmd, self.active_qmd_buf = qmd, qmd_buf
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    if self.active_qmd is not None:
      for i in range(2):
        if self.active_qmd.read(f'release{i}_enable') == 0:
          self.active_qmd.write(**{f'release{i}_enable': 1})

          addr_off = self.active_qmd.field_offset(f'release{i}_address_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_addr_lower')
          self.bind_sints_to_mem(signal.value_addr & 0xffffffff, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=addr_off)
          self.bind_sints_to_mem(signal.value_addr >> 32, mem=self.active_qmd_buf.cpu_view(), fmt='I', mask=0xf, offset=addr_off+4)

          val_off = self.active_qmd.field_offset(f'release{i}_payload_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_payload_lower')
          self.bind_sints_to_mem(value & 0xffffffff, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=val_off)
          self.bind_sints_to_mem(value >> 32, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=val_off+4)
          return self

    # if PROFILE >= 2:
    #   # PM_TRIGGER after kernel, then wait for idle and trigger again for PC sampling
    #   self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             (1 << 0) | (1 << 20) | (1 << 24) | (1 << 25)) # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    # if PROFILE >= 2:
    #   self.nvm(1, nv_gpu.NVC6C0_WAIT_FOR_IDLE, 0)
    #   self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
    self.active_qmd = None
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.compute_gpfifo)

class NVCopyQueue(NVCommandQueue):
  def __init__(self, queue_idx=0):
    self.queue_idx = queue_idx
    super().__init__()

  def copy(self, dest:sint, src:sint, copy_size:int):
    for off in range(0, copy_size, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src+off), *data64(dest+off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(copy_size-off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x182) # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x14)
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.dma_gpfifo)

class NVVideoQueue(NVCommandQueue):
  def decode_hevc_chunk(self, pic_desc:HCQBuffer, in_buf:HCQBuffer, out_buf:HCQBuffer, out_buf_pos:int, hist_bufs:list[HCQBuffer], hist_pos:list[int],
                        chroma_off:int, coloc_buf:HCQBuffer, filter_buf:HCQBuffer, intra_top_off:int, intra_unk_off:int|None, status_buf:HCQBuffer):
    self.nvm(4, nv_gpu.NVC9B0_SET_APPLICATION_ID, nv_gpu.NVC9B0_SET_APPLICATION_ID_ID_HEVC)
    self.nvm(4, nv_gpu.NVC9B0_SET_CONTROL_PARAMS, 0x52057)
    self.nvm(4, nv_gpu.NVC9B0_SET_DRV_PIC_SETUP_OFFSET, pic_desc.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_IN_BUF_BASE_OFFSET, in_buf.va_addr >> 8)
    for pos, buf in zip(hist_pos + [out_buf_pos], hist_bufs + [out_buf]):
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_LUMA_OFFSET0 + pos*4, buf.va_addr >> 8)
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_CHROMA_OFFSET0 + pos*4, buf.offset(chroma_off).va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_COLOC_DATA_OFFSET, coloc_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_NVDEC_STATUS_OFFSET, status_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_TILE_SIZES_OFFSET, pic_desc.offset(0x200).va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_FILTER_BUFFER_OFFSET, filter_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_INTRA_TOP_BUF_OFFSET, (filter_buf.va_addr + intra_top_off) >> 8)
    if intra_unk_off is not None: self.nvm(4, 0x4dc, (filter_buf.va_addr + intra_unk_off) >> 8)
    self.nvm(4, nv_gpu.NVC9B0_EXECUTE, 0)
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_D, (1 << 24) | (1 << 0))
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.vid_gpfifo)

class NVArgsState(CLikeArgsState):
  def __init__(self, buf:HCQBuffer, prg:NVProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    if MOCKGPU: prg.cbuf_0[80:82] = [len(bufs), len(vals)]
    super().__init__(buf, prg, bufs, vals=vals, prefix=prg.cbuf_0 or None)

class NVProgram(HCQProgram):
  def __init__(self, dev:NVDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[va_addr, size]]

    if (NAK:=isinstance(dev.renderer, NAKRenderer)):
      image, self.cbuf_0 = memoryview(bytearray(lib[ctypes.sizeof(info:=mesa.struct_nak_shader_info.from_buffer_copy(lib)):])), []
      self.regs_usage, self.shmem_usage, self.lcmem_usage = info.num_gprs, round_up(info.cs.smem_size, 128), round_up(info.slm_size, 16)
    elif MOCKGPU: image, sections, relocs = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), [], [] # type: ignore
    else: image, sections, relocs = elf_loader(self.lib, force_section_align=128)
    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.lib_gpu = self.dev.allocator.alloc(round_up((prog_sz:=image.nbytes), 0x1000) + 0x1000, buf_spec:=BufferSpec(nolru=True))
    prog_addr = self.lib_gpu.va_addr
    if not NAK:
      # For MOCKGPU, the lib is PTX code, so some values are emulated.
      self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0 if not MOCKGPU else 0x160
      for sh in sections: # pylint: disable=possibly-used-before-assignment
        if sh.name == f".nv.shared.{self.name}": self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
        if sh.name == f".text.{self.name}": prog_addr, prog_sz = self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size
        elif m:=re.match(r'\.nv\.constant(\d+)', sh.name):
          self.constbufs[int(m.group(1))] = (self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size)
        elif sh.name.startswith(".nv.info"):
          for typ, param, data in self._parse_elf_info(sh):
            if sh.name == f".nv.info.{name}" and param == 0xa: cbuf0_size = struct.unpack_from("IH", data)[1] # EIATTR_PARAM_CBANK
            elif sh.name == ".nv.info" and param == 0x12: self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240 # EIATTR_MIN_STACK_SIZE
            elif sh.name == ".nv.info" and param == 0x2f: self.regs_usage = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

      # Apply relocs
      for apply_image_offset, rel_sym_offset, typ, _ in relocs: # pylint: disable=possibly-used-before-assignment
        # These types are CUDA-specific, applying them here
        if typ == 2: image[apply_image_offset:apply_image_offset+8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset) # R_CUDA_64
        elif typ == 0x38: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
        elif typ == 0x39: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
        else: raise RuntimeError(f"unknown NV reloc {typ}")

      # Minimum cbuf_0 size for driver params: Blackwell needs index 223 (224 entries), older GPUs need index 11 (12 entries)
      min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
      self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)

    # Ensure device has enough local memory to run the program
    self.dev._ensure_has_local_memory(self.lcmem_usage)
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()

    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      if not NAK: self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
      qmd = {'qmd_major_version':5, 'qmd_type':nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'program_address_upper_shifted4':hi32(prog_addr>>4),
        'program_address_lower_shifted4':lo32(prog_addr>>4), 'register_count':self.regs_usage, 'shared_memory_size_shifted7':self.shmem_usage>>7,
        'shader_local_memory_high_size_shifted4':self.lcmem_usage>>4 if NAK else self.dev.slm_per_thread>>4}
    else:
      if not NAK: self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'program_address_upper':hi32(prog_addr), 'program_address_lower':lo32(prog_addr),
        'shared_memory_size':self.shmem_usage, 'register_count_v':self.regs_usage,
        **({'shader_local_memory_low_size':self.lcmem_usage} if NAK else {'shader_local_memory_high_size':self.dev.slm_per_thread})}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1

    self.qmd:QMD = QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(prog_sz>>8, 0x1ff),
      sass_version=dev.sass_version, program_prefetch_addr_upper_shifted=prog_addr>>40, program_prefetch_addr_lower_shifted=prog_addr>>8)

    for i,(addr,sz) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(i, addr)
      self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})

    # Registers allocation granularity per warp is 256, warp allocation granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(max(1, self.regs_usage) * 32, 256)) // 4) * 4 * 32

    # NV's kernargs is constbuffer, then arguments to the kernel follows. Kernargs also appends QMD at the end of the kernel.
    print("prog_addr", hex(prog_addr))
    super().__init__(NVArgsState, self.dev, self.name, kernargs_alloc_size=round_up(self.constbufs[0][1], 1 << 8) + (8 << 8))
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off+4:start_off+sz+4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if prod(local_size) > 1024 or self.max_threads < prod(local_size) or self.lcmem_usage > cast(NVDevice, self.dev).slm_per_thread:
      raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {self.max_threads=}")
    if any(cur > mx for cur,mx in zip(global_size, [2147483647, 65535, 65535])) or any(cur > mx for cur,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

class NVAllocator(HCQAllocator['NVDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host)

  def _do_free(self, opaque:HCQBuffer, options:BufferSpec): self.dev.iface.free(opaque)

  def _map(self, buf:HCQBuffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

  def _encode_decode(self, bufout:HCQBuffer, bufin:HCQBuffer, desc_buf:HCQBuffer, hist:list[HCQBuffer], shape:tuple[int,...], frame_pos:int):
    assert all(h.va_addr % 0x100 == 0 for h in hist + [bufin, bufout, desc_buf]), "all buffers must be 0x100 aligned"

    h, w = ((2 * shape[0]) // 3 if shape[0] % 3 == 0 else (2 * shape[0] - 1) // 3), shape[1]
    self.dev._ensure_has_vid_hw(w, h)

    q = NVVideoQueue().wait(self.dev.timeline_signal, self.dev.timeline_value - 1)
    with hcq_profile(self.dev, queue=q, desc="NVDEC", enabled=PROFILE):
      q.decode_hevc_chunk(desc_buf, bufin, bufout, frame_pos, hist, [(frame_pos-x) % (len(hist) + 1) for x in range(len(hist), 0, -1)],
                          round_up(w, 64)*round_up(h, 64), self.dev.vid_coloc_buf, self.dev.vid_filter_buf, self.dev.intra_top_off,
                          self.dev.intra_unk_off, self.dev.vid_stat_buf)
    q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)

@dataclass
class GPFifo:
  ring: MMIOInterface
  gpput: MMIOInterface
  entries_count: int
  token: int
  put_value: int = 0

class NVKIface:
  root = None
  fd_ctl: FileIOInterface
  fd_uvm: FileIOInterface
  gpus_info: list|ctypes.Array = []

  # TODO: Need a proper allocator for va addresses
  # 0x1000000000 - 0x2000000000, reserved for system/cpu mappings
  # VA space is 48bits.
  low_uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=0x1146780000, base=0x8000000000 if OSX else 0x1000000000, wrap=False)
  uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=(1 << 48) - 1, base=low_uvm_vaddr_allocator.base + low_uvm_vaddr_allocator.size, wrap=False)
  host_object_enumerator: int = 0x1000

  def __init__(self, dev, device_id):
    if NVKIface.root is None:
      global nv_gpu

      NVKIface.fd_ctl = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.fd_uvm = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      self.fd_uvm_2 = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.root = self.rm_alloc(0, nv_gpu.NV01_ROOT_CLIENT, None, root=0)

      drvver = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2, nv_gpu.NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS())
      if int(drvver.driverVersionBuffer.decode().split('.')[0], 10) >= 580: nv_gpu = nv_580

      self.uvm(nv_gpu.UVM_INITIALIZE, nv_gpu.UVM_INITIALIZE_PARAMS())

      # this error is okay, CUDA hits it too
      with contextlib.suppress(RuntimeError): self.uvm(nv_gpu.UVM_MM_INITIALIZE, nv_gpu.UVM_MM_INITIALIZE_PARAMS(uvmFd=self.fd_uvm.fd), self.fd_uvm_2)

      nv_iowr(NVKIface.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, gpus_info:=(nv_gpu.nv_ioctl_card_info_t*64)())
      NVKIface.gpus_info = hcq_filter_visible_devices(gpus_info)

    self.dev, self.device_id = dev, device_id
    if self.device_id >= len(NVKIface.gpus_info) or not NVKIface.gpus_info[self.device_id].valid:
      raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    self.fd_dev = self._new_gpu_fd()
    self.gpu_info = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
      nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS(gpuId=NVKIface.gpus_info[self.device_id].gpu_id))
    self.gpu_minor = NVKIface.gpus_info[self.device_id].minor_number
    self.gpu_instance = self.gpu_info.deviceInstance

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_ALLOC, made:=nv_gpu.NVOS21_PARAMETERS(hRoot=root if root is not None else self.root,
      hObjectParent=parent, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status == nv_gpu.NV_ERR_NO_MEMORY: raise MemoryError(f"rm_alloc returned {get_error_str(made.status)}")
    if made.status != 0: raise RuntimeError(f"rm_alloc returned {get_error_str(made.status)}")
    return made.hObjectNew

  def rm_control(self, obj, cmd, params=None, root=None):
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_CONTROL, made:=nv_gpu.NVOS54_PARAMETERS(hClient=root if root is not None else self.root, hObject=obj, cmd=cmd,
      paramsSize=ctypes.sizeof(params) if params is not None else 0, params=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status != 0: raise RuntimeError(f"rm_control returned {get_error_str(made.status)}")
    return params

  def uvm(self, cmd, params, fd=None):
    nv_iowr(fd or self.fd_uvm, None, params, cmd=cmd)
    if params.rmStatus != 0: raise RuntimeError(f"uvm returned {get_error_str(params.rmStatus)}")

  def setup_usermode(self):
    clsnum = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(numClasses=0))
    clsinfo = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(
      numClasses=clsnum.numClasses, classList=mv_address(classlist:=memoryview(bytearray(clsnum.numClasses * 4)).cast('I'))))
    self.nvclasses = {classlist[i] for i in range(clsinfo.numClasses)}
    self.usermode_class:int = next(c for c in [nv_gpu.HOPPER_USERMODE_A, nv_gpu.TURING_USERMODE_A] if c in self.nvclasses)
    self.gpfifo_class:int = next(c for c in [nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_CHANNEL_GPFIFO_A] if c in self.nvclasses)
    self.compute_class:int = next(c for c in [nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_COMPUTE_B] if c in self.nvclasses)
    self.dma_class:int = next(c for c in [nv_gpu.BLACKWELL_DMA_COPY_B, nv_gpu.AMPERE_DMA_COPY_B] if c in self.nvclasses)
    self.viddec_class:int|None = next((c for c in [nv_gpu.NVCFB0_VIDEO_DECODER, nv_gpu.NVC9B0_VIDEO_DECODER] if c in self.nvclasses), None)

    usermode = self.rm_alloc(self.dev.subdevice, self.usermode_class)
    return usermode, MMIOInterface(self._gpu_map_to_cpu(usermode, mmio_sz:=0x10000), mmio_sz, fmt='I')

  def setup_vm(self, vaspace):
    self.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, raw_uuid:=nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(
      flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16))
    self.gpu_uuid = nv_gpu.struct_nv_uuid(uuid=(ctypes.c_ubyte*16)(*[raw_uuid.data[i] for i in range(16)]))

    self.uvm(nv_gpu.UVM_REGISTER_GPU, nv_gpu.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=self.gpu_uuid))
    self.uvm(nv_gpu.UVM_REGISTER_GPU_VASPACE, nv_gpu.UVM_REGISTER_GPU_VASPACE_PARAMS(
      gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hVaSpace=vaspace))

    for dev in cast(list[NVDevice], [d for pg in HCQCompiled.peer_groups.values() for d in pg if isinstance(d, NVDevice) and not d.is_nvd()]):
      try: self.uvm(nv_gpu.UVM_ENABLE_PEER_ACCESS, nv_gpu.UVM_ENABLE_PEER_ACCESS_PARAMS(gpuUuidA=self.gpu_uuid, gpuUuidB=dev.iface.gpu_uuid))
      except RuntimeError as e: raise RuntimeError(f"{e}. Make sure GPUs #{self.gpu_minor} & #{dev.iface.gpu_minor} have P2P enabled.") from e

  def setup_gpfifo_vm(self, gpfifo):
    self.uvm(nv_gpu.UVM_REGISTER_CHANNEL, nv_gpu.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root,
      hChannel=gpfifo, base=self._alloc_gpu_vaddr(0x4000000, force_low=True), length=0x4000000))

  def _new_gpu_fd(self):
    fd_dev = FileIOInterface(f"/dev/nvidia{NVKIface.gpus_info[self.device_id].minor_number}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl.fd))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev.fd,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hMemory=memory_handle, length=size, flags=flags))
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {get_error_str(made.params.status)}")
    return fd_dev.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), 0)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flags=0, cpu_addr=None, **kwargs) -> HCQBuffer:
    # Uncached memory is "system". Use huge pages only for gpu memory.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (mmap.PAGESIZE if MOCKGPU else 4 << 10))
    size = round_up(size, page_size)
    va_addr = self._alloc_gpu_vaddr(size, alignment=page_size, force_low=cpu_access) if (alloced:=cpu_addr is None) else cpu_addr

    if host:
      if alloced: va_addr = FileIOInterface.anon_mmap(va_addr, size, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, 0)

      flags = (nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) \
            | (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30)

      NVKIface.host_object_enumerator += 1
      made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, flags=flags,
        hObjectNew=NVKIface.host_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, pMemory=va_addr, limit=size-1), fd=-1)
      nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)

      if made.params.status != 0: raise RuntimeError(f"host alloc returned {get_error_str(made.params.status)}")
      mem_handle = made.params.hObjectNew
    else:
      attr = ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contiguous else nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27) \
          | (nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE if page_size > 0x1000 else 0) << 23 | ((nv_gpu.NVOS32_ATTR_LOCATION_PCI if uncached else 0) << 25)

      attr2 = ((nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO if uncached else nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES) << 2) \
            | ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB if page_size > 0x1000 else 0) << 20) | nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC

      fl = nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nv_gpu.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE \
         | nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | (nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM if not uncached else 0)

      alloc_func = nv_gpu.NV1_MEMORY_SYSTEM if uncached else nv_gpu.NV1_MEMORY_USER
      alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, alignment=page_size, offset=0, limit=size-1, format=6, size=size,
        type=nv_gpu.NVOS32_TYPE_NOTIFIER if uncached else nv_gpu.NVOS32_TYPE_IMAGE, attr=attr, attr2=attr2, flags=fl)
      mem_handle = self.rm_alloc(self.dev.nvdevice, alloc_func, alloc_params)

      if cpu_access: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=uncached)

    return self._gpu_uvm_map(va_addr, size, mem_handle, has_cpu_mapping=cpu_access or host)

  def free(self, mem:HCQBuffer):
    if mem.meta.hMemory > NVKIface.host_object_enumerator: # not a host object, clear phys mem.
      made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, hObjectOld=mem.meta.hMemory)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
      if made.status != 0: raise RuntimeError(f"_gpu_free returned {get_error_str(made.status)}")

    self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=int(mem.va_addr), length=mem.size))
    if mem.view is not None: FileIOInterface.munmap(int(mem.va_addr), mem.size)

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True, has_cpu_mapping=False) -> HCQBuffer:
    if create_range:
      self.uvm(nv_gpu.UVM_CREATE_EXTERNAL_RANGE, nv_gpu.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size))
      made = nv_gpu.NVOS46_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hDma=self.dev.virtmem, hMemory=mem_handle, length=size,
        flags=(nv_gpu.NVOS46_FLAGS_PAGE_SIZE_4KB<<8)|(nv_gpu.NVOS46_FLAGS_CACHE_SNOOP_ENABLE<<4)|(nv_gpu.NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE<<15),
        dmaOffset=va_base)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY_DMA, made)
      if made.status != 0: raise RuntimeError(f"nv_sys_alloc 1 returned {get_error_str(made.status)}")
      assert made.dmaOffset == va_base, f"made.dmaOffset != va_base {made.dmaOffset=} {va_base=}"

    attrs = (nv_gpu.UvmGpuMappingAttributes*256)(nv_gpu.UvmGpuMappingAttributes(gpuUuid=self.gpu_uuid, gpuMappingType=1))

    self.uvm(nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION, uvm_map:=nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size,
      rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hMemory=mem_handle, gpuAttributesCount=1, perGpuAttributes=attrs, mapped_gpu_ids=[self.gpu_uuid]))
    return HCQBuffer(va_base, size, meta=uvm_map, view=MMIOInterface(va_base, size, fmt='B') if has_cpu_mapping else None, owner=self.dev)

  def map(self, mem:HCQBuffer):
    if mem.owner is not None and mem.owner._is_cpu():
      if not any(x.device.startswith("NV") for x in mem.mapped_devs): return self.alloc(mem.size, host=True, cpu_addr=mem.va_addr)
      mem = mem.mappings[next(x for x in mem.mapped_devs if x.device.startswith("NV"))]
    self._gpu_uvm_map(mem.va_addr, mem.size, mem.meta.hMemory, create_range=False)

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10), force_low=False):
    return NVKIface.low_uvm_vaddr_allocator.alloc(size, alignment) if force_low else NVKIface.uvm_vaddr_allocator.alloc(size, alignment)

class PCIIface(PCIIfaceBase):
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id):
    # PCIIface's MAP_FIXED mmap will overwrite UVM allocations made by NVKIface, so don't try PCIIface if kernel driver was already used.
    if NVKIface.root is not None: raise RuntimeError("Cannot use PCIIface after NVKIface has been initialized (would corrupt UVM memory)")
    super().__init__(dev, dev_id, vendor=0x10de, devices=[(0xff00, [0x2200, 0x2400, 0x2500, 0x2600, 0x2700, 0x2800, 0x2b00, 0x2c00, 0x2d00, 0x2f00])],
      base_class=0x03, bars=[0, 1, 3], vram_bar=1, va_start=NVMemoryManager.va_allocator.base, va_size=NVMemoryManager.va_allocator.size)
    if not OSX: System.reserve_hugepages(64)

    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.dev_impl:NVDev = NVDev(self.pci_dev)
    self.root, self.gpu_instance = 0xc1000000, 0
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # Setup classes for the GPU
    self.gpfifo_class, self.compute_class, self.dma_class = (gsp:=self.dev_impl.gsp).gpfifo_class, gsp.compute_class, gsp.dma_class
    self.viddec_class = None

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    # Force use of huge pages for large allocations. NVDev will attempt to use huge pages in any case,
    # but if the size is not aligned, the tail will be allocated with 4KB pages, increasing TLB pressure.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (4 << 10))
    return super().alloc(round_up(size, page_size), host=host, uncached=uncached, cpu_access=cpu_access, contiguous=contiguous, **kwargs)

  def setup_usermode(self): return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)
  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def rm_alloc(self, parent, clss, params=None, root=None) -> int: return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)
  def rm_control(self, obj, cmd, params=None): return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root)

  def device_fini(self): self.dev_impl.fini()

class NVDevice(HCQCompiled[HCQSignal]):
  def is_nvd(self) -> bool: return isinstance(self.iface, PCIIface)

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = self._select_iface(NVKIface, PCIIface)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.gf100_profiler = self.iface.rm_alloc(self.subdevice, nv_gpu.GF100_PROFILER, None)
    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL, nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1ffffffffffff))
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()

    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
      flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) | \
             (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)

    self.iface.setup_vm(vaspace)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    self.gpfifo_area = self.iface.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True,
      map_flags=(nv_gpu.NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED<<23))

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)

    self.compute_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0, entries=0x10000, compute=True)
    self.dma_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0x100000, entries=0x10000, compute=False)
    self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    self.cmdq_page:HCQBuffer = self.iface.alloc(0x200000, cpu_access=True)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=int(self.cmdq_page.va_addr), wrap=True)
    self.cmdq = self.cmdq_page.cpu_view().view(fmt='I')

    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info('num_gpcs',
      'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')

    # FIXME: no idea how to convert this for blackwells
    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)

    cucc, ptxcc = (CUDACompiler, PTXCompiler) if MOCKGPU else (NVCompiler, NVPTXCompiler)
    compilers = CompilerSet(ctrl_var=NV_CC, cset=[CompilerPair(functools.partial(NVRenderer, self.arch),functools.partial(cucc, self.arch)),
       CompilerPair(functools.partial(PTXRenderer, self.arch, device="NV"), functools.partial(ptxcc, self.arch), NV_PTX),
       CompilerPair(functools.partial(NAKRenderer, self.arch, self.max_warps_per_sm), None, NV_NAK)])
    super().__init__(device, NVAllocator(self), compilers, functools.partial(NVProgram, self), HCQSignal, NVComputeQueue, NVCopyQueue)

    # Initialize profiler if PROFILE >= 2
    if PROFILE >= 2: init_nv_profiler(self)
    
    self._setup_gpfifos()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False) -> GPFifo:
    notifier = self.iface.alloc(48 << 20, uncached=True)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem if video else gpfifo_area.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.meta.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset), engineType=19 if video else 0)
    gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)

    if compute:
      self.debug_compute_obj, self.debug_channel = self.iface.rm_alloc(gpfifo, self.iface.compute_class), gpfifo
      debugger_params = nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=self.iface.root, hClass3dObject=self.debug_compute_obj)
      self.debugger = self.iface.rm_alloc(self.nvdevice, nv_gpu.GT200_DEBUGGER, debugger_params)
    elif not video: self.iface.rm_alloc(gpfifo, self.iface.dma_class)
    else: self.iface.rm_alloc(gpfifo, self.iface.viddec_class)

    if channel_group == self.nvdevice:
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_BIND, nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=params.engineType))
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    ws_token_params = self.iface.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))
    if ctxshare != 0: self.iface.setup_gpfifo_vm(gpfifo)

    return GPFifo(ring=gpfifo_area.cpu_view().view(offset, entries*8, fmt='Q'), entries_count=entries, token=ws_token_params.workSubmitToken,
                  gpput=gpfifo_area.cpu_view().view(offset + entries*8 + getattr(nv_gpu.AmpereAControlGPFifo, 'GPPut').offset, fmt='I'))

  def _query_gpu_info(self, *reqs):
    nvrs = [getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_'+r.upper(), getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_LITTER_'+r.upper(), None)) for r in reqs]

    if self.is_nvd():
      x = self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
        nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS())
      return [x.engineInfo[0].infoList[nvr].data for nvr in nvrs]

    infos = (nv_gpu.NV2080_CTRL_GR_INFO*len(nvrs))(*[nv_gpu.NV2080_CTRL_GR_INFO(index=nvr) for nvr in nvrs])
    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO,
      nv_gpu.NV2080_CTRL_GR_GET_INFO_PARAMS(grInfoListSize=len(infos), grInfoList=ctypes.addressof(infos)))
    return [x.data for x in infos]

  def _setup_gpfifos(self):
    self.slm_per_thread, self.shader_local_mem = 0, None

    # Set windows addresses to not collide with other allocated buffers.
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000

    NVComputeQueue().setup(compute_class=self.iface.compute_class, local_mem_window=self.local_mem_window, shared_mem_window=self.shared_mem_window) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)

    NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                 .setup(copy_class=self.iface.dma_class) \
                 .signal(self.timeline_signal, self.next_timeline()).submit(self)

    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required: return

    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    self.shader_local_mem, ok = self._realloc(self.shader_local_mem, round_up(bytes_per_tpc*self.num_tpc_per_gpc*self.num_gpcs, 0x20000))

    # Realloc failed, restore the old value.
    if not ok: self.slm_per_thread = old_slm_per_thread

    cast(NVComputeQueue, NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1)) \
                                         .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                                         .signal(self.timeline_signal, self.next_timeline()).submit(self)

  def _ensure_has_vid_hw(self, w, h):
    if self.iface.viddec_class is None: raise RuntimeError(f"{self.device} Video decoder class not available.")

    coloc_size = round_up((round_up(h, 64) * round_up(h, 64)) + (round_up(w, 64) * round_up(h, 64) // 16), 2 << 20)
    self.intra_top_off = round_up(h, 64) * (608 + 4864 + 152 + 2000)
    intra_unk_size = ((2 << 20) if self.iface.viddec_class >= nv_gpu.NVCFB0_VIDEO_DECODER else 0)
    self.intra_unk_off = (round_up(self.intra_top_off, 0x10000) + (64 << 10)) if intra_unk_size > 0 else None
    filter_size = round_up(round_up(self.intra_top_off, 0x10000) + (64 << 10) + intra_unk_size, 2 << 20)

    if not hasattr(self, 'vid_gpfifo'):
      self.vid_gpfifo = self._new_gpu_fifo(self.gpfifo_area, 0, self.nvdevice, offset=0x200000, entries=2048, compute=False, video=True)
      self.vid_coloc_buf, self.vid_filter_buf = self.allocator.alloc(coloc_size), self.allocator.alloc(filter_size)
      self.vid_stat_buf = self.allocator.alloc(0x1000)
      NVVideoQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                    .setup(copy_class=self.iface.viddec_class) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)
    else:
      if coloc_size > self.vid_coloc_buf.size: self.vid_coloc_buf, _ = self._realloc(self.vid_coloc_buf, coloc_size, force=True)
      if filter_size > self.vid_filter_buf.size: self.vid_filter_buf, _ = self._realloc(self.vid_filter_buf, filter_size, force=True)

  def invalidate_caches(self):
    if self.is_nvd(): self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_BUS_FLUSH_WITH_SYSMEMBAR, None)
    else:
      self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE, nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS(
        flags=((nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES << 2) | (nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE_YES << 3) |
              (nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_FULL_CACHE << 4))))

  def on_device_hang(self):
    # Prepare fault report.
    # TODO: Restore the GPU using NV83DE_CTRL_CMD_CLEAR_ALL_SM_ERROR_STATES if needed.

    report = []
    sm_errors = self.iface.rm_control(self.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES,
      nv_gpu.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS(hTargetChannel=self.debug_channel, numSMsToRead=100))

    if sm_errors.mmuFault.valid:
      mmu = self.iface.rm_control(self.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO,
        nv_gpu.NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS())
      for i in range(mmu.count):
        pfinfo = mmu.mmuFaultInfoList[i]
        report += [f"MMU fault: 0x{pfinfo.faultAddress:X} | {NV_PFAULT_FAULT_TYPE[pfinfo.faultType]} | {NV_PFAULT_ACCESS_TYPE[pfinfo.accessType]}"]
    else:
      for i, e in enumerate(sm_errors.smErrorStateArray):
        if e.hwwGlobalEsr or e.hwwWarpEsr: report += [f"SM {i} fault: esr={e.hwwGlobalEsr} warp_esr={e.hwwWarpEsr:#x} warp_pc={e.hwwWarpEsrPc64:#x}"]

    raise RuntimeError("\n".join(report))

  def synchronize(self):
    super().synchronize()
    if nv_profiler is not None: nv_profiler.flush(wait=True)

# PC Sampling Profiler for NV devices (PROFILE=2)
class NVProfiler:
  """PC sampling profiler for NV devices using the kernel driver interface."""

  def __init__(self, dev:NVDevice):
    if not isinstance(dev.iface, NVKIface): raise RuntimeError("NVProfiler requires NVKIface (kernel driver)")
    self.dev, self.iface = dev, dev.iface

    # Create separate RM client for profiling (needed for proper permissions)
    self.root = self.iface.rm_alloc(0, nv_gpu.NV01_ROOT_CLIENT, None, root=0)

    # Create device/subdevice hierarchy under our profiler client
    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.root, vaMode=0)
    self.nvdevice = self.iface.rm_alloc(self.root, nv_gpu.NV01_DEVICE_0, device_params, root=self.root)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS(), root=self.root)

    # Allocate GF100_PROFILER like CUPTI does (capability probing)
    try: self.gf100_profiler = self.iface.rm_alloc(self.subdevice, nv_gpu.GF100_PROFILER, None, root=self.root)
    except RuntimeError: self.gf100_profiler = None

    # Create profiler targeting the app's channel group
    profiler_params = nv_gpu.NVB2CC_ALLOC_PARAMETERS(hClientTarget=self.iface.root, hContextTarget=dev.channel_group)
    self.profiler = self.iface.rm_alloc(self.subdevice, nv_gpu.MAXWELL_PROFILER_DEVICE, profiler_params, root=self.root)

    # Request power features (optional, may fail on some GPUs)
    try: self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES, nv_gpu.struct_NVB0CC_CTRL_POWER_REQUEST_FEATURES_PARAMS(controlMask=1349))
    except RuntimeError: pass

    # Allocate PMA buffers directly (system memory, not UVM)
    self.pma_size = 512 * 1024 * 1024
    pma_alloc = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.nvdevice, type=0, flags=0xC000, attr=0x2A800000, attr2=0x9,
                                                    format=6, size=self.pma_size, alignment=4096, limit=self.pma_size-1)
    self.pma_mem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV1_MEMORY_SYSTEM, pma_alloc, root=self.root)

    bytes_alloc = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.nvdevice, type=0, flags=0xC000, attr=0x2A800000, attr2=0x400009,
                                                      format=6, size=4096, alignment=4096, limit=4095)
    self.bytes_mem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV1_MEMORY_SYSTEM, bytes_alloc, root=self.root)

    # Setup PMA stream (pmaBufferVA=0x100000000 like CUPTI)
    pma_stream = nv_gpu.struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS(hMemPmaBuffer=self.pma_mem, pmaBufferSize=self.pma_size,
      hMemPmaBytesAvailable=self.bytes_mem, pmaBufferVA=0x100000000)
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM, pma_stream)
    print('pmaChannelIdx', hex(pma_stream.pmaChannelIdx))

    # CPU-map the buffers (flags 0x3008000 matches CUPTI's caching mode)
    self.pma_fd = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    self.pma_addr = self._map_memory(self.pma_fd, self.pma_mem, self.pma_size, 0x3008000)

    self.bytes_fd = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    self.bytes_addr = self._map_memory(self.bytes_fd, self.bytes_mem, 4096, 0x3008001)

    # Reserve PM resources and configure
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY, nv_gpu.struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS(ctxsw=0))
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER, None)
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_BIND_PM_RESOURCES, None)

    # Configure PC sampling hardware
    self._setup_pc_sampling()
    if DEBUG >= 1: print(f"NVProfiler: initialized for {dev.device}")

  def _rm_control(self, cmd, params): return self.iface.rm_control(self.profiler, cmd, params, root=self.root)

  def _map_memory(self, fd, mem, size, flags):
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd.fd,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.nvdevice, hMemory=mem, length=size, flags=flags))
    nv_iowr(self.iface.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"map_memory failed: {get_error_str(made.params.status)}")
    return fd.mmap(None, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, 0)

  def _setup_pc_sampling(self):
    """Configure PC sampling hardware registers."""
    # Query TPC masks for each GPC to determine which TPCs are enabled
    tpc_masks = []
    for gpc_id in range(self.dev.num_gpcs):
      params = nv_gpu.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=gpc_id)
      self.iface.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_TPC_MASK, params)
      tpc_masks.append(params.tpcMask)
    # Use first 11 GPCs (0-10) like CUPTI, even if some have no TPCs
    # The profiler API expects contiguous GPC indices
    enabled_gpcs = list(range(11))
    num_gpcs = 11
    if DEBUG >= 2: print(f"NVProfiler: using {num_gpcs} GPCs (0-10), masks={[hex(tpc_masks[i]) if i < len(tpc_masks) else '?' for i in range(11)]}")

    # Initial GR_CTX write
    self._reg_op(0x419bdc, 0x0, reg_type=1)

    # Clear GPC registers and setup HS credits (mask=0x100 to only modify bit 8 like CUPTI)
    self._reg_ops([(0x244000 + gpc * 0x200, 0x0) for gpc in enabled_gpcs], mask=0x100)

    # HS credits use logical chiplet indices (0 to num_gpcs-1), not physical GPC numbers
    hs_clear = nv_gpu.struct_NVB0CC_CTRL_HS_CREDITS_PARAMS(pmaChannelIdx=0, numEntries=num_gpcs + 7)
    for i in range(num_gpcs): hs_clear.creditInfo[i] = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO(chipletType=2, chipletIndex=i, numCredits=0)
    for i in range(6): hs_clear.creditInfo[num_gpcs+i] = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO(chipletType=1, chipletIndex=i, numCredits=0)
    hs_clear.creditInfo[num_gpcs+6] = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO(chipletType=3, chipletIndex=0, numCredits=0)
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_SET_HS_CREDITS, hs_clear)

    self._reg_op(0x24a620, 0x2000000, mask=0x2000000)
    self._reg_ops([(0x244000 + gpc * 0x200, 0x100) for gpc in enabled_gpcs], mask=0x100)

    # Get and allocate HS credits
    total_credits = nv_gpu.struct_NVB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS()
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_GET_TOTAL_HS_CREDITS, total_credits)
    if DEBUG >= 2: print(f"NVProfiler: total_credits={total_credits.numCredits}")

    # Allocate HS credits based on TPC count: 4 credits per enabled TPC (2 SMs * 2 credits)
    # For disabled GPCs (mask=0), still assign full credits (24) like CUPTI does
    hs_alloc = nv_gpu.struct_NVB0CC_CTRL_HS_CREDITS_PARAMS(pmaChannelIdx=0, numEntries=num_gpcs)
    for i, gpc in enumerate(enabled_gpcs):
      tpc_count = bin(tpc_masks[gpc]).count('1') if tpc_masks[gpc] else 6  # Treat disabled GPCs as having 6 TPCs
      hs_alloc.creditInfo[i] = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO(chipletType=2, chipletIndex=i, numCredits=tpc_count * 4)
    hs_alloc_x = self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_SET_HS_CREDITS, hs_alloc)
    print(f"NVProfiler: hs_alloc allocated credits={hs_alloc_x.statusInfo.status}")

    # Global registers
    self._reg_op(0x24a03c, 0x1)
    self._reg_op(0x24a62c, 0x0)
    self._reg_ops([
      (0x24a700, 0x0), (0x24a708, 0x0), (0x24a710, 0x0), (0x24a704, 0x0), (0x24a70c, 0x0),
      (0x24a714, 0x0), (0x24a718, 0x0), (0x24a71c, 0x0), (0x24a720, 0x0),
      (0x24a65c, 0xffffffff), (0x24a664, 0xffffffff), (0x24a66c, 0xffffffff),
      (0x24a660, 0xffffffff), (0x24a668, 0xffffffff), (0x24a670, 0xffffffff),
      (0x24a674, 0xffffffff), (0x24a67c, 0xffffffff), (0x24a684, 0xffffffff),
      (0x24a678, 0xffffffff), (0x24a680, 0xffffffff), (0x24a688, 0xffffffff),
      (0x24a6a0, 0xffffffff), (0x24a6a8, 0xffffffff), (0x24a6b0, 0xffffffff),
      (0x24a6a4, 0xffffffff), (0x24a6ac, 0xffffffff), (0x24a6b4, 0xffffffff),
      (0x24a6b8, 0x0), (0x24a6c0, 0x0), (0x24a6c8, 0x0), (0x24a6bc, 0x0), (0x24a6c4, 0x0), (0x24a6cc, 0x0),
    ])
    self._reg_ops([(0x24a010, 0xffffffff), (0x24a014, 0xffffffff)])
    self._reg_ops([(0x24a640, 0x40), (0x24a620, 0x2000007)])

    # Per-GPC configuration - CUPTI writes first set for ALL GPCs, then second set for ALL GPCs
    self._reg_ops([(0x180000 + gpc * 0x4000 + off, val) for gpc in enabled_gpcs for off, val in [(0x108, 0x0), (0x110, 0x0), (0x100, 0x0), (0x0ec, 0x1)]])
    self._reg_ops([(0x180000 + gpc * 0x4000 + off, val) for gpc in enabled_gpcs for off, val in [(0x308, 0x0), (0x310, 0x0), (0x300, 0x0), (0x2ec, 0x1)]])

    # Per-TPC configuration - skip disabled TPCs based on mask
    # TPC offset index % 6 maps to TPC number, indices 18-19 are always included
    # For fully disabled GPCs (mask=0x0), write to all TPCs anyway like CUPTI does
    tpc_offsets = (0x508, 0x708, 0x908, 0xb08, 0xd08, 0xf08, 0x1108, 0x1308, 0x1508, 0x1708, 0x1908, 0x1b08,
                   0x1d08, 0x1f08, 0x2108, 0x2308, 0x2508, 0x2708, 0x2908, 0x2b08)
    for gpc in enabled_gpcs:
      base, mask = 0x180000 + gpc * 0x4000, tpc_masks[gpc]
      for i, tpc_base in enumerate(tpc_offsets):
        # Skip if TPC is disabled: for indices 0-17, check TPC mask bit. But if mask is 0x0 (fully disabled GPC), write anyway
        if i < 18 and mask != 0 and not (mask & (1 << (i % 6))): continue
        self._reg_ops([(base + tpc_base + off, val) for off, val in [(0x0, 0x0), (0x8, 0x0), (-0x8, 0x0), (-0x1c, 0x1)]])

    self._reg_ops([(0x419b04, 0x0), (0x419b04, 0x80808a)])

    # TPC ID configuration (GPU-specific values from CUPTI trace)
    # Skip entries for disabled GPCs (based on tpc_masks)
    tpc_configs = [
      (0x180728, 0x403), (0x181328, 0x409), (0x180928, 0x404), (0x181528, 0x40a), (0x180b28, 0x405), (0x181728, 0x40b),
      (0x180d28, 0x406), (0x181928, 0x40c), (0x180f28, 0x407), (0x181b28, 0x40d), (0x184728, 0x423), (0x185328, 0x429),
      (0x184928, 0x424), (0x185528, 0x42a), (0x184b28, 0x425), (0x185728, 0x42b), (0x184d28, 0x426), (0x185928, 0x42c),
      (0x184f28, 0x427), (0x185b28, 0x42d), (0x188528, 0x442), (0x189128, 0x448), (0x188728, 0x443), (0x189328, 0x449),
      (0x188928, 0x444), (0x189528, 0x44a), (0x188b28, 0x445), (0x189728, 0x44b), (0x188d28, 0x446), (0x189928, 0x44c),
      (0x188f28, 0x447), (0x189b28, 0x44d), (0x18c528, 0x462), (0x18d128, 0x468), (0x18c728, 0x463), (0x18d328, 0x469),
      (0x18c928, 0x464), (0x18d528, 0x46a), (0x18cb28, 0x465), (0x18d728, 0x46b), (0x18cd28, 0x466), (0x18d928, 0x46c),
      (0x18cf28, 0x467), (0x18db28, 0x46d), (0x190528, 0x482), (0x191128, 0x488), (0x190728, 0x483), (0x191328, 0x489),
      (0x190928, 0x484), (0x191528, 0x48a), (0x190b28, 0x485), (0x191728, 0x48b), (0x190d28, 0x486), (0x191928, 0x48c),
      (0x190f28, 0x487), (0x191b28, 0x48d), (0x194528, 0x4a2), (0x195128, 0x4a8), (0x194728, 0x4a3), (0x195328, 0x4a9),
      (0x194928, 0x4a4), (0x195528, 0x4aa), (0x194b28, 0x4a5), (0x195728, 0x4ab), (0x194d28, 0x4a6), (0x195928, 0x4ac),
      (0x194f28, 0x4a7), (0x195b28, 0x4ad), (0x198528, 0x4c2), (0x199128, 0x4c8), (0x198728, 0x4c3), (0x199328, 0x4c9),
      (0x198928, 0x4c4), (0x199528, 0x4ca), (0x198b28, 0x4c5), (0x199728, 0x4cb), (0x198d28, 0x4c6), (0x199928, 0x4cc),
      (0x198f28, 0x4c7), (0x199b28, 0x4cd), (0x19c528, 0x502), (0x19d128, 0x508), (0x19c728, 0x503), (0x19d328, 0x509),
      (0x19c928, 0x504), (0x19d528, 0x50a), (0x19cb28, 0x505), (0x19d728, 0x50b), (0x19cd28, 0x506), (0x19d928, 0x50c),
      (0x19cf28, 0x507), (0x19db28, 0x50d), (0x1a0528, 0x522), (0x1a1128, 0x528), (0x1a0728, 0x523), (0x1a1328, 0x529),
      (0x1a0928, 0x524), (0x1a1528, 0x52a), (0x1a0b28, 0x525), (0x1a1728, 0x52b), (0x1a0d28, 0x526), (0x1a1928, 0x52c),
      (0x1a0f28, 0x527), (0x1a1b28, 0x52d), (0x1a4528, 0x542), (0x1a5128, 0x548), (0x1a4728, 0x543), (0x1a5328, 0x549),
      (0x1a4928, 0x544), (0x1a5528, 0x54a), (0x1a4b28, 0x545), (0x1a5728, 0x54b), (0x1a4d28, 0x546), (0x1a5928, 0x54c),
      (0x1a4f28, 0x547), (0x1a5b28, 0x54d), (0x1a8528, 0x562), (0x1a9128, 0x568), (0x1a8728, 0x563), (0x1a9328, 0x569),
      (0x1a8928, 0x564), (0x1a9528, 0x56a), (0x1a8b28, 0x565), (0x1a9728, 0x56b), (0x1a8d28, 0x566), (0x1a9928, 0x56c),
      (0x1a8f28, 0x567), (0x1a9b28, 0x56d),
    ]
    for addr, tpc_id in tpc_configs:
      gpc = (addr - 0x180000) >> 14
      if gpc >= 11: continue  # Only GPCs 0-10 (like CUPTI)
      base = addr - 0x128
      self._reg_ops([
        (base + 0x0ec, 0x1), (base + 0x06c, 0x2), (base + 0x108, 0x20), (base + 0x100, 0x0),
        (base + 0x0cc, 0x0), (base + 0x0d0, 0x0), (base + 0x0d4, 0x0), (base + 0x0d8, 0x0), (base + 0x0dc, 0x0),
        (base + 0x040, 0x0), (base + 0x048, 0x0), (base + 0x050, 0x0), (base + 0x044, 0x0), (base + 0x04c, 0x0), (base + 0x054, 0x0),
        (base + 0x040, 0x19181716), (base + 0x048, 0x1d1c1b1a), (base + 0x050, 0x1e001f), (base + 0x128, tpc_id), (base + 0x09c, 0x40005),
      ])

    # Enable PC sampling
    self._reg_op(0x419bdc, 0x1, reg_type=1)

  def _reg_op(self, offset, value, reg_type=0, mask=0xffffffff):
    """Execute single register write."""
    params = nv_gpu.struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS(regOpCount=1)
    params.regOps[0] = nv_gpu.struct_NV2080_CTRL_GPU_REG_OP(regOp=nv_gpu.NV2080_CTRL_GPU_REG_OP_WRITE_32, regType=reg_type, regOffset=offset, regValueLo=value, regAndNMaskLo=mask)
    try: self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_EXEC_REG_OPS, params)
    except RuntimeError as e:
      if DEBUG >= 3: print(f"NVProfiler: reg_op {offset:#x}={value:#x} type={reg_type} failed: {e}")

  def _reg_ops(self, ops, mask=0xffffffff):
    """Execute batch of register writes (max 124 per call). ops can be (offset, value) or (offset, value, mask)."""
    for i in range(0, len(ops), 124):
      chunk = ops[i:i + 124]
      params = nv_gpu.struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS(regOpCount=len(chunk))
      for j, op in enumerate(chunk):
        offset, value = op[0], op[1]
        op_mask = op[2] if len(op) > 2 else mask
        params.regOps[j] = nv_gpu.struct_NV2080_CTRL_GPU_REG_OP(regOp=nv_gpu.NV2080_CTRL_GPU_REG_OP_WRITE_32, regOffset=offset, regValueLo=value, regAndNMaskLo=op_mask)
      try: self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_EXEC_REG_OPS, params)
      except RuntimeError as e:
        if DEBUG >= 3: print(f"NVProfiler: reg_ops batch[{i}:{i+len(chunk)}] failed: {e}")

  def _parse_pc_samples(self, data: bytes):
    """Parse PC sampling data from PMA stream."""
    # PC sampling stall reasons - HW encoding to name mapping
    # HW bits[15:12] -> stall name (matching CUPTI semantics)
    STALL_REASONS = {0: "invalid", 1: "none", 2: "inst_fetch", 3: "exec_dep", 4: "mem_dep",
                     5: "texture", 6: "const_mem", 7: "pipe_busy", 8: "mem_throttle",
                     9: "not_selected", 10: "other", 11: "sleeping", 12: "sync",
                     13: "lg_throttle", 14: "branch", 15: "dispatch"}

    # Parse as 32-bit little-endian words
    words = struct.unpack(f'<{len(data)//4}I', data)

    if DEBUG >= 4:
      print("Raw words:")
      for i in range(0, min(len(words), 40), 2):
        print(f"  [{i:3d}] {words[i]:08x} {words[i+1]:08x}")

    # Helper functions for token classification
    def is_separator(w): return (w & 0x0000FFFF) == 0x421c and ((w >> 16) & 0xFFFF) in (0x0800, 0x1800)
    def is_meta(w): return (w >> 24) in (0x7F, 0x3F)

    # Collect stall statistics
    stall_counts: dict[int, int] = {}
    total_samples = 0
    pc_samples: dict[int, int] = {}  # PC address -> count

    # PMA PC sampling data format (8 bytes per sample):
    # - Separators: 0x1800421c, 0x0800421c (skip these)
    # - Meta tokens: 0x7Fxxxxxx or 0x3Fxxxxxx, stall_id = (meta >> 12) & 0xF
    # - PC tokens: (PC >> 4) truncated to 32 bits
    i = 0
    while i < len(words) - 1:
      w0, w1 = words[i], words[i + 1]

      # Skip separators
      if is_separator(w0):
        i += 1
        continue
      if is_separator(w1):
        i += 1
        continue

      # Determine which word is PC and which is metadata (can be in either order)
      if is_meta(w0) and not is_meta(w1) and not is_separator(w1):
        metadata, pc_token = w0, w1
        i += 2
      elif not is_meta(w0) and not is_separator(w0) and is_meta(w1):
        pc_token, metadata = w0, w1
        i += 2
      else:
        i += 1
        continue

      # Extract stall reason from metadata bits[15:12]
      stall = (metadata >> 12) & 0xF

      if pc_token != 0:
        stall_counts[stall] = stall_counts.get(stall, 0) + 1
        pc_samples[pc_token] = pc_samples.get(pc_token, 0) + 1
        total_samples += 1

    if total_samples > 0 and DEBUG >= 1:
      # Format output similar to CUPTI
      stall_str = " ".join(f"{STALL_REASONS.get(k, f'unk{k}')}:{v*100//total_samples}%"
                           for k, v in sorted(stall_counts.items(), key=lambda x: -x[1]) if v > 0)
      print(f"  NV PC sampling: {total_samples} samples | {stall_str}")
      if DEBUG >= 3:
        print(f"    PCs: {[(hex(pc), cnt) for pc, cnt in sorted(pc_samples.items(), key=lambda x: -x[1])[:10]]}")

  def flush(self, wait=False):
    """Read available profiling data from PMA stream."""
    # Query bytes available (bWait=1 blocks until data available, bWait=0 returns immediately)
    params = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bUpdateAvailableBytes=1, bWait=1)
    self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT, params)
    if DEBUG >= 2: print(f"NVProfiler: bytesAvailable={params.bytesAvailable}")

    if params.bytesAvailable > 0:
      # Read data from CPU-mapped PMA buffer
      pma_data = to_mv(self.pma_addr, params.bytesAvailable)
      if DEBUG >= 1: print(f"NVProfiler: got {params.bytesAvailable} bytes")

      # Acknowledge consumption
      ack = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bytesConsumed=params.bytesAvailable)
      self._rm_control(nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT, ack)

      # Parse PC sampling data
      self._parse_pc_samples(bytes(pma_data))
      return bytes(pma_data)
    return None

nv_profiler: NVProfiler|None = None

def init_nv_profiler(dev:NVDevice):
  global nv_profiler
  if nv_profiler is None and PROFILE >= 2:
    nv_profiler = NVProfiler(dev)
    # except Exception as e:
    #   print(f"  NVProfiler: Failed to initialize: {e}")
