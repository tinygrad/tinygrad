from __future__ import annotations
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref
assert sys.platform != 'win32'
from typing import cast, Union, ClassVar
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQProgram, HCQSignal, BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, MOCKGPU
from tinygrad.uop.ops import sint
from tinygrad.device import BufferSpec, CPUProgram
from tinygrad.helpers import getenv, mv_address, round_up, data64, data64_le, DEBUG, prod, OSX, to_mv, hi32, lo32
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, PTXCompiler, PTX, NVPTXCompiler, NVCompiler
from tinygrad.runtime.autogen import nv_gpu
from tinygrad.runtime.support.elf import elf_loader

import os, mmap, re, array, gzip, struct, ctypes, time, subprocess
from tinygrad.helpers import fetch, to_mv, round_up, getenv
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc, pci
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMapping
from tinygrad.runtime.support.system import System
from hexdump import hexdump

if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

@dataclass
class NVAllocationMeta: owner:AMDDevice; mapped_devs:list[AMDDevice]; mapping:NVMapping; has_cpu_mapping:bool # noqa: E702

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_ACCESS_TYPE_")}

# def nv_iowr(fd:FileIOInterface, nr, args):
#   ret = fd.ioctl((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF), args)
#   if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def rm_alloc(dev, clss, root, parent, params): return dev.gsp.rpc_rm_alloc(parent, clss, params, client=0xdead0000)

def rm_control(cmd, sttyp, dev, client, obj, **kwargs):
  params = sttyp(**kwargs)
  x = dev.gsp.rpc_rm_control(obj, cmd, params, client=0xdead0000)
  return type(params).from_buffer_copy(x)

def make_rmctrl_type():
  return type("NVRMCTRL", (object,), {name[name.find("_CTRL_CMD_")+10:].lower(): functools.partial(rm_control, dt, sttyp)
    for name,dt in nv_gpu.__dict__.items() if name.find("_CTRL_CMD_")>=0 and (sttyp:=getattr(nv_gpu, name.replace("_CTRL_CMD_", "_CTRL_")+"_PARAMS", \
      getattr(nv_gpu, name+"_PARAMS", getattr(nv_gpu, name.replace("_CTRL_CMD_", "_CTRL_DEBUG_")+"_PARAMETERS", None))))})
rmctrl = make_rmctrl_type()

def uvm_ioctl(cmd, sttyp, fd:FileIOInterface, **kwargs):
  ret = fd.ioctl(cmd, made:=sttyp(**kwargs))
  if ret != 0: raise RuntimeError(f"ioctl(uvm) returned {ret}")
  if made.rmStatus != 0: raise RuntimeError(f"uvm_ioctl returned {get_error_str(made.rmStatus)}")
  return made

def make_uvm_type():
  return type("NVUVM", (object,), {name.replace("UVM_", "").lower(): functools.partial(uvm_ioctl, dt, getattr(nv_gpu, name+"_PARAMS"))
                                   for name,dt in nv_gpu.__dict__.items() if name.startswith("UVM_") and nv_gpu.__dict__.get(name+"_PARAMS")})
uvm = make_uvm_type()

class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev:NVDDevice, addr:int|None=None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)

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
    for k,val in kwargs.items(): self._rw_bits(*QMD.fields[self.pref][k.upper()], value=val)

  def read(self, k, val=0): return self._rw_bits(*QMD.fields[self.pref][k.upper()])

  def field_offset(self, k): return QMD.fields[self.pref][k.upper()][1] // 8

  def set_constant_buf_addr(self, i, addr):
    if self.ver < 4: self.write(**{f'constant_buffer_addr_upper_{i}':hi32(addr), f'constant_buffer_addr_lower_{i}':lo32(addr)})
    else: self.write(**{f'constant_buffer_addr_upper_shifted6_{i}':hi32(addr >> 6), f'constant_buffer_addr_lower_shifted6_{i}':lo32(addr >> 6)})

class NVSignal(HCQSignal):
  def __init__(self, base_buf:HCQBuffer|None=None, **kwargs):
    super().__init__(base_buf, **kwargs, timestamp_divider=1000, dev_t=NVDDevice)

class NVCommandQueue(HWQueue[NVSignal, 'NVDDevice', 'NVProgram', 'NVArgsState']):
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

  def wait(self, signal:NVSignal, value:sint=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value), (3 << 0) | (1 << 24)) # ACQUIRE | PAYLOAD_SIZE_64BIT
    self.active_qmd = None
    return self

  def timestamp(self, signal:NVSignal): return self.signal(signal, 0)

  def bind(self, dev:NVDDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i, value in enumerate(self._q): hw_view[i] = value

    # From now on, the queue is on the device for faster submission.
    self._q = hw_view

  def _submit_to_gpfifo(self, dev:NVDDevice, gpfifo:GPFifo):
    if dev == self.binded_device: cmdq_addr = self.hw_page.va_addr
    else:
      cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4)
      cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
      dev.cmdq[cmdq_wptr : cmdq_wptr + len(self._q)] = array.array('I', self._q)

    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.controls.GPPut = (gpfifo.put_value + 1) % gpfifo.entries_count

    if CPUProgram.atomic_lib is not None: CPUProgram.atomic_lib.atomic_thread_fence(__ATOMIC_SEQ_CST:=5)
    # dev.gpu_mmio[0x90 // 4] = gpfifo.token
    dev.nvdev.wreg(0x00B80000 + 0x30090, gpfifo.token)

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

    qmd = QMD(dev=prg.dev, addr=cast(int, qmd_buf.va_addr)) # Save qmd for later update

    self.bind_sints_to_mem(*global_size, mem=qmd_buf.cpu_view(), fmt='I', offset=qmd.field_offset('cta_raster_width'))
    self.bind_sints_to_mem(*(local_size[:2]), mem=qmd_buf.cpu_view(), fmt='H', offset=qmd.field_offset('cta_thread_dimension0'))
    self.bind_sints_to_mem(local_size[2], mem=qmd_buf.cpu_view(), fmt='B', offset=qmd.field_offset('cta_thread_dimension2'))
    qmd.set_constant_buf_addr(0, args_state.buf.va_addr)

    if self.active_qmd is None:
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    else:
      self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)

    self.active_qmd, self.active_qmd_buf = qmd, qmd_buf
    return self

  # def signal_2(self, addr:int, value:sint=0):
  #   self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(addr), *data64_le(value),
  #            (1 << 0) | (1 << 20) | (1 << 24) | (1 << 25)) # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
  #   return self
  
  def signal(self, signal:NVSignal, value:sint=0):
    if self.active_qmd is not None:
      for i in range(2):
        if self.active_qmd.read(f'release{i}_enable') == 0:
          self.active_qmd.write(**{f'release{i}_enable': 1})
          self.bind_sints_to_mem(signal.value_addr, mem=self.active_qmd_buf.cpu_view(), fmt='Q', mask=0xfffffffff,
            offset=self.active_qmd.field_offset(f'release{i}_address_lower'))
          self.bind_sints_to_mem(value, mem=self.active_qmd_buf.cpu_view(), fmt='Q', offset=self.active_qmd.field_offset(f'release{i}_payload_lower'))
          return self

    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             (1 << 0) | (1 << 20) | (1 << 24) | (1 << 25)) # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit(self, dev:NVDDevice): self._submit_to_gpfifo(dev, dev.compute_gpfifo)

class NVCopyQueue(NVCommandQueue):
  def copy(self, dest:sint, src:sint, copy_size:int):
    for off in range(0, copy_size, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src+off), *data64(dest+off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(copy_size-off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x182) # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH
    return self

  # def signal(self, signal:NVSignal, value:sint=0, addr1=0x0, addr2=0x0):
  #   # self.nvm(4, nv_gpu.NVC6B5_NOP)

  #   # self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(addr1), *data64(addr2))
  #   # self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, 4)
  #   # self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x182) # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH

  #   self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(addr1), value)
  #   self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x14)
  #   return self
  def signal(self, signal:NVSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x14)
    return self

  def _submit(self, dev:NVDDevice): self._submit_to_gpfifo(dev, dev.dma_gpfifo)

class NVArgsState(CLikeArgsState):
  def __init__(self, buf:HCQBuffer, prg:NVProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    if MOCKGPU: prg.constbuffer_0[80:82] = [len(bufs), len(vals)]
    super().__init__(buf, prg, bufs, vals=vals, prefix=prg.constbuffer_0)

class NVProgram(HCQProgram):
  def __init__(self, dev:NVDDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib

    # For MOCKGPU, the lib is PTX code, so some values are emulated.
    cbuf0_size = 0 if not MOCKGPU else 0x160

    if MOCKGPU: image, sections, relocs = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), [], [] # type: ignore
    else: image, sections, relocs = elf_loader(self.lib, force_section_align=128)

    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000) + 0x1000, buf_spec:=BufferSpec(cpu_access=True))

    self.prog_addr, self.prog_sz, self.regs_usage, self.shmem_usage, self.lcmem_usage = self.lib_gpu.va_addr, image.nbytes, 0, 0x400, 0
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[va_addr, size]]
    for sh in sections:
      if sh.name == f".nv.shared.{self.name}": self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
      if sh.name == f".text.{self.name}": self.prog_addr, self.prog_sz = self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size
      elif m:=re.match(r'\.nv\.constant(\d+)', sh.name): self.constbufs[int(m.group(1))] = (self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size)
      elif sh.name.startswith(".nv.info"):
        for typ, param, data in self._parse_elf_info(sh):
          if sh.name == f".nv.info.{name}" and param == 0xa: cbuf0_size = struct.unpack_from("IH", data)[1] # EIATTR_PARAM_CBANK
          elif sh.name == ".nv.info" and param == 0x12: self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240 # EIATTR_MIN_STACK_SIZE
          elif sh.name == ".nv.info" and param == 0x2f: self.regs_usage = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

    # Ensure device has enough local memory to run the program
    # self.dev._ensure_has_local_memory(self.lcmem_usage)

    # Apply relocs
    for apply_image_offset, rel_sym_offset, typ, _ in relocs:
      # These types are CUDA-specific, applying them here
      if typ == 2: image[apply_image_offset:apply_image_offset+8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset) # R_CUDA_64
      elif typ == 0x38: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
      elif typ == 0x39: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
      else: raise RuntimeError(f"unknown NV reloc {typ}")

    ctypes.memmove(self.lib_gpu.va_addr, mv_address(image), image.nbytes)

    self.constbuffer_0 = [0] * (cbuf0_size // 4)

    if dev.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      self.constbuffer_0[188:192], self.constbuffer_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
      qmd = {'qmd_major_version':5, 'unknown_13':0x1, 'program_address_upper':hi32(self.prog_addr>>4),'program_address_lower':lo32(self.prog_addr>>4),
             'sass_version':0xA4}
    else:
      self.constbuffer_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'cwd_membar_type':nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
             'program_address_upper':hi32(self.prog_addr), 'program_address_lower':lo32(self.prog_addr), 'sass_version':0x89}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1

    self.qmd:QMD = QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      constant_buffer_invalidate_0=1, register_count_v=self.regs_usage, shader_local_memory_high_size=self.dev.slm_per_thread,
      min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a,
      shared_memory_size=self.shmem_usage, program_prefetch_size=min(self.prog_sz>>8, 0x1ff),
      program_prefetch_addr_upper_shifted=self.prog_addr>>40, program_prefetch_addr_lower_shifted=self.prog_addr>>8)

    for i,(addr,sz) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(i, addr)
      self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})

    # Registers allocation granularity per warp is 256, warp allocation granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(max(1, self.regs_usage) * 32, 256)) // 4) * 4 * 32

    # NV's kernargs is constbuffer, then arguments to the kernel follows. Kernargs also appends QMD at the end of the kernel.
    super().__init__(NVArgsState, self.dev, self.name, kernargs_alloc_size=round_up(self.constbufs[0][1], 1 << 8) + (8 << 8))
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off+4:start_off+sz+4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    # if prod(local_size) > 1024 or self.max_threads < prod(local_size) or self.lcmem_usage > cast(NVDDevice, self.dev).slm_per_thread:
    #   raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {self.max_threads=}")
    # if any(cur > mx for cur,mx in zip(global_size, [2147483647, 65535, 65535])) or any(cur > mx for cur,mx in zip(local_size, [1024, 1024, 64])):
    #   raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

class NVAllocator(HCQAllocator['NVDDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.host: return self.dev._gpu_alloc(size, host=True, tag="user host memory")
    return self.dev._gpu_alloc(size, cpu_access=options.cpu_access, tag=f"user memory ({options})")

  def _free(self, opaque:HCQBuffer, options:BufferSpec):
    self.dev.synchronize()
    self.dev._gpu_free(opaque)

  def map(self, buf:HCQBuffer): pass # self.dev._gpu_map(buf._base if buf._base is not None else buf)

@dataclass
class GPFifo:
  ring: MMIOInterface
  controls: nv_gpu.AmpereAControlGPFifo
  entries_count: int
  token: int
  put_value: int = 0

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVDDevice(HCQCompiled[NVSignal]):
  devices: ClassVar[list[HCQCompiled]] = []
  signal_pages: ClassVar[list[HCQBuffer]] = []
  signal_pool: ClassVar[list[HCQBuffer]] = []

  root = None
  gpus_info: Union[list, ctypes.Array] = []

  def _gpu_alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flags=0, tag="") -> HCQBuffer:
    # if host or (not getenv("NV_ALLOC_QUEUE_DEV_MEM", 1) and uncached and cpu_access): # host or gtt-like memory.
    #   vaddr = self.adev.mm.alloc_vaddr(size:=round_up(size, mmap.PAGESIZE), align=mmap.PAGESIZE)
    #   va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | MAP_LOCKED | MAP_FIXED, 0)
    #   assert va != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(vaddr)}"

    #   # Read pagemap to get the physical address of each page. The pages are locked.
    #   self.pagemap.seek(va // mmap.PAGESIZE * 8)
    #   paddrs = [((x & ((1<<55) - 1)) * mmap.PAGESIZE, mmap.PAGESIZE) for x in array.array('Q', self.pagemap.read(size//mmap.PAGESIZE*8, binary=True))]
    #   am_mapping = self.adev.mm.map_range(vaddr, size, paddrs, system=True, snooped=True, uncached=True)
    #   return HCQBuffer(vaddr, size, meta=NVAllocationMeta(self.dev, [self.dev], am_mapping, has_cpu_mapping=cpu_access),
    #     view=MMIOInterface(am_mapping.va_addr, size, fmt='B'))

    cpu_access = True
    nv_mapping = self.nvdev.mm.valloc(size:=round_up(size, 0x1000), uncached=uncached, contiguous=cpu_access)
    if cpu_access: self._map_pci_range(bar=1, off=nv_mapping.paddrs[0][0], addr=nv_mapping.va_addr, size=nv_mapping.size)
    return HCQBuffer(nv_mapping.va_addr, size, meta=NVAllocationMeta(self, [self], nv_mapping, has_cpu_mapping=cpu_access),
      view=MMIOInterface(nv_mapping.va_addr, size, fmt='B') if cpu_access else None)

  def _map_pci_range(self, bar, off=0, addr=0, size=None, fmt='B'):
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar][1] - self.bar_info[bar][0] + 1)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    assert loc != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(addr)}"
    return MMIOInterface(loc, sz, fmt=fmt)

  def _gpu_free(self, mem:HCQBuffer):
    pass

  def __init__(self, device:str=""):
    os.system("sudo sh -c 'echo 0 > /proc/sys/vm/compact_unevictable_allowed'")
    os.system("sudo sh -c 'echo 8 > /proc/sys/vm/nr_hugepages'")

    dev = None
    for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
      vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
      pdevice = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
      if vendor == 0x10de and pdevice == 0x2684: dev = pcibus

    pcibus = dev
    self.pcibus = pcibus

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/driver/unbind", os.O_WRONLY).write(pcibus)

    supported_sizes = int(FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource1_resize", os.O_RDONLY).read(), 16)
    try: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource1_resize", os.O_RDWR).write(str(supported_sizes.bit_length() - 1))
    except OSError as e: raise RuntimeError(f"Cannot resize BAR: {e}. Ensure the resizable BAR option is enabled on your system.") from e

    FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/enable", os.O_RDWR).write("1")

    cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    self.bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in [0, 1]}

    bar_info = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource", os.O_RDONLY).read().splitlines()
    self.bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

    cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    pci_cmd = int.from_bytes(cfg_fd.read(2, binary=True, offset=pci.PCI_COMMAND), byteorder='little') | pci.PCI_COMMAND_MASTER
    cfg_fd.write(pci_cmd.to_bytes(2, byteorder='little'), binary=True, offset=pci.PCI_COMMAND)

    regs = self._map_pci_range(0, fmt='I')
    fb = self._map_pci_range(1)

    # hexdump(bytes(array.array('I', regs[0x00300000:0x00310000])))
    # exit(0)

    self.nvdev = NVDev(pcibus, regs, fb)

    self.nvdev.gsp.client = 0xdead0000
    NVDDevice.root = rm_alloc(self.nvdev, nv_gpu.NV01_ROOT, 0, 0, nv_gpu.NV0000_ALLOC_PARAMETERS())
    # NVDDevice.root = self.nvdev.gsp.client

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=0x0, hClientShare=self.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.nvdevice = rm_alloc(self.nvdev, nv_gpu.NV01_DEVICE_0, self.root, self.root, device_params)

    subdevice_params = nv_gpu.NV2080_ALLOC_PARAMETERS(subDeviceId=0x0)
    self.subdevice = rm_alloc(self.nvdev, nv_gpu.NV20_SUBDEVICE_0, self.root, self.nvdevice, subdevice_params)

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x0, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED | \
            nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ALLOW_ZERO_ADDRESS)
    vaspace = rm_alloc(self.nvdev, nv_gpu.FERMI_VASPACE_A, self.root, self.nvdevice, vaspace_params)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = rm_alloc(self.nvdev, nv_gpu.KEPLER_CHANNEL_GROUP_A, self.root, self.nvdevice, channel_params)

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = rm_alloc(self.nvdev, nv_gpu.FERMI_CONTEXT_SHARE_A, self.root, channel_group, ctxshare_params)

    self.compute_class = nv_gpu.ADA_COMPUTE_A
    self.dma_class = nv_gpu.AMPERE_DMA_COPY_B

    gpfifo_area = self._gpu_alloc(0x200000, contiguous=True, cpu_access=True, tag="gpfifo")

    self.compute_gpfifo = self._new_gpu_fifo(gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, enable_debug=False)
    self.dma_gpfifo = self.compute_gpfifo # reuse for now.
    # self.dma_gpfifo = self._new_gpu_fifo(gpfifo_area, ctxshare, channel_group, offset=0x100000, entries=0x400)

    rmctrl.gpfifo_schedule(self.nvdev, self.root, channel_group, bEnable=1)

    self.cmdq_page:HCQBuffer = self._gpu_alloc(0x200000, cpu_access=True, tag="cmdq")
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=cast(int, self.cmdq_page.va_addr), wrap=True)
    self.cmdq = MMIOInterface(cast(int, self.cmdq_page.va_addr), 0x200000, fmt='I')

    self.arch = "sm_89"

    compiler_t = (PTXCompiler if PTX else CUDACompiler) if MOCKGPU else (NVPTXCompiler if PTX else NVCompiler)
    super().__init__(device, NVAllocator(self), PTXRenderer(self.arch, device="NVD") if PTX else NVRenderer(self.arch), compiler_t(self.arch),
                     functools.partial(NVProgram, self), NVSignal, NVComputeQueue, NVCopyQueue)

    self._setup_gpfifos()
    self.synchronize()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, enable_debug=False) -> GPFifo:
    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.meta.mapping.paddrs[0][0] + offset + entries * 8, size=0x400, addressSpace=2, cacheAttrib=0)

    # print(hex(gpfifo_area.meta.mapping.va_addr), hex(gpfifo_area.meta.mapping.paddrs[0][0]))

    notifier_va, notifier_sysmem = System.alloc_sysmem(0x1000, contiguous=True)
    notifier = nv_gpu.NV_MEMORY_DESC_PARAMS(base=notifier_sysmem[0], size=0x000000ECC, addressSpace=1, cacheAttrib=0)
    self.notifier = to_mv(notifier_va, 0x1000)

    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=0x0, hObjectBuffer=0x0,
      gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=entries, hContextShare=ctxshare, engineType=0x0, cid=0,
      userdOffset=(ctypes.c_uint64*8)(entries * 8), userdMem=userd, errorNotifierMem=notifier, internalFlags=0x1d, flags=0x201020)
    gpfifo = rm_alloc(self.nvdev, nv_gpu.AMPERE_CHANNEL_GPFIFO_A, self.root, channel_group, params)
    comp = rm_alloc(self.nvdev, nv_gpu.ADA_COMPUTE_A, self.root, gpfifo, None)
    rm_alloc(self.nvdev, nv_gpu.AMPERE_DMA_COPY_B, self.root, gpfifo, None)

    ws_token_params = rmctrl.gpfifo_get_work_submit_token(self.nvdev, self.root, gpfifo, workSubmitToken=-1)
    assert ws_token_params.workSubmitToken != -1

    print(hex(ws_token_params.workSubmitToken))

    return GPFifo(ring=MMIOInterface(gpfifo_area.va_addr + offset, entries*8, fmt='Q'), entries_count=entries, token=ws_token_params.workSubmitToken,
                  controls=nv_gpu.AmpereAControlGPFifo.from_address(gpfifo_area.va_addr + offset + entries * 8))

  def _query_gpu_info(self, *reqs):
    nvrs = [getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_'+r.upper(), getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_LITTER_'+r.upper(),None)) for r in reqs]
    infos = (nv_gpu.NV2080_CTRL_GR_INFO*len(nvrs))(*[nv_gpu.NV2080_CTRL_GR_INFO(index=nvr) for nvr in nvrs])
    rmctrl.gr_get_info(self.fd_ctl, self.root, self.subdevice, grInfoListSize=len(infos), grInfoList=ctypes.addressof(infos))
    return [x.data for x in infos]

  def _setup_gpfifos(self):
    self.slm_per_thread, self.shader_local_mem = 0, None

    # Set windows addresses to not collide with other allocated buffers.
    self.shared_mem_window = 0x729400000000 if self.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 0xfe000000
    self.local_mem_window = 0x729300000000 if self.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 0xff000000

    # NVComputeQueue().signal(self.timeline_signal, self.timeline_value).submit(self)
    # self.timeline_value += 1
    # time.sleep(0.4)
    # hexdump(self.notifier[:0x20])
    # self.synchronize()

    NVComputeQueue().setup(compute_class=self.compute_class, local_mem_window=self.local_mem_window, shared_mem_window=self.shared_mem_window) \
                    .signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1
    time.sleep(0.4)
    hexdump(self.notifier[:0x20])
    self.synchronize()

    # NVComputeQueue().setup(compute_class=self.compute_class) \
    #                 .signal(self.timeline_signal, self.timeline_value).submit(self)
    # time.sleep(0.4)
    # hexdump(self.notifier[:0x20])
    # self.synchronize()

    # # sys.exit(0)

    # self.synchronize()

    # # print("ok")

    cast(NVCopyQueue, NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1)) \
                                   .setup(copy_class=self.dma_class) \
                                   .signal(self.timeline_signal, self.timeline_value).submit(self)

    self.timeline_value += 1
    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required or ((maxlm:=getenv("NV_MAX_LOCAL_MEMORY_PER_THREAD")) > 0 and required >= maxlm): return

    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    self.shader_local_mem, ok = self._realloc(self.shader_local_mem, round_up(bytes_per_tpc*self.num_tpc_per_gpc*self.num_gpcs, 0x20000))

    # Realloc failed, restore the old value.
    if not ok: self.slm_per_thread = old_slm_per_thread

    cast(NVComputeQueue, NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1)) \
                                         .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                                         .signal(self.timeline_signal, self.next_timeline()).submit(self)

