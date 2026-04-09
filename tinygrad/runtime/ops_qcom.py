from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, sys, weakref, contextlib
assert sys.platform != 'win32'
from typing import cast
from tinygrad.device import BufferSpec, Device
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, HCQProgram, HCQCompiled, HCQAllocatorBase, HCQSignal, HCQArgsState, BumpAllocator
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.autogen import kgsl, mesa
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, ceildiv, prod, cpu_profile, lo32, suppress_finalizing
from tinygrad.helpers import flatten, PROFILE
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.adreno import qreg, pkt7_hdr, pkt4_hdr, parse_ir3_shader, compute_program_sizes
from tinygrad.runtime.support.adreno import build_a6xx_compute_pm4, build_a6xx_tex_descriptor
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.codegen import get_program
  buf, n = UOp(Ops.PARAM, dtypes.uint8.ptr(), arg=0), UOp(Ops.PARAM, dtypes.uint8.ptr(), arg=1)
  i = UOp.range(n.cast(dtypes.int), 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, dtypes.void, (buf.cast(dtypes.ulong) + i.cast(dtypes.ulong) * UOp.const(dtypes.ulong, 64),),
              arg='__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");')
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, dtypes.void, (), arg='__asm__ volatile("dsb sy" ::: "memory");'), arg=KernelInfo(name="dcache_flush"))
  ps = get_program(UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="CPU"), UOp(Ops.LINEAR, src=tuple(sink.toposort())))), Device["CPU"].renderer)
  return Device["CPU"].runtime(ps.function_name, ps.lib)

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class QCOMSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 19.2})

  def _sleep(self, time_spent_since_last_sleep_ms:int):
    # Sleep only for timeline signals. Do it immediately to free cpu.
    if self.is_timeline and self.owner is not None:
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.owner.fd, context_id=self.owner.ctx, timestamp=self.owner.last_cmd, timeout=0xffffffff)

class QCOMComputeQueue(HWQueue):
  def __init__(self, dev:QCOMDevice):
    self.dev = dev
    super().__init__()

  @suppress_finalizing
  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def cmd(self, opcode: int, *vals: int): self.q(pkt7_hdr(opcode, len(vals)), *vals)

  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    # TODO: 7xx support.
    if write_back: self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(self.dev.dummy_addr), 0) # dirty cache write-back.
    if invalidate: self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE) # invalidate cache lines (following reads from RAM).
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self):
    self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)
    return self

  def signal(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gpu_id[:2] < (7, 3):
      self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), *data64_le(signal.value_addr), lo32(value))
      self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    else:
      # TODO: support devices starting with 8 Gen 1. Also, 700th series have convenient CP_GLOBAL_TIMESTAMP and CP_LOCAL_TIMESTAMP
      raise RuntimeError('CP_EVENT_WRITE7 is not supported')
    return self

  def timestamp(self, signal:QCOMSignal):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    self.cmd(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True),*data64_le(signal.timestamp_addr))
    return self

  def wait(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),*data64_le(signal.value_addr),
             qreg.cp_wait_reg_mem_3(ref=value&0xFFFFFFFF), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))
    return self

  def _build_gpu_command(self, dev:QCOMDevice, hw_addr=None):
    to_mv((hw_page_addr:=hw_addr or dev.cmd_buf_allocator.alloc(len(self._q) * 4)), len(self._q) * 4).cast('I')[:] = array.array('I', self._q)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(self._q) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), numcmds=1, context_id=dev.ctx,
                                              cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    return submit_req, obj

  def bind(self, dev:QCOMDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    self.submit_req, self.obj = self._build_gpu_command(self.binded_device, self.hw_page.va_addr)
    # From now on, the queue is on the device for faster submission.
    self._q = to_mv(self.obj.gpuaddr, len(self._q) * 4).cast("I")

  def _submit(self, dev:QCOMDevice):
    if self.binded_device == dev: submit_req = self.submit_req
    else: submit_req, _ = self._build_gpu_command(dev)
    dev.last_cmd = kgsl.IOCTL_KGSL_GPU_COMMAND(dev.fd, __payload=submit_req).timestamp

  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    self.bind_args_state(args_state)
    build_a6xx_compute_pm4(self.cmd, self.reg, prg, int(args_state.buf.va_addr), prg.lib_gpu.va_addr, prg.dev._stack.va_addr,
                           prg.dev.border_color_buf.va_addr, global_size, local_size, nir=prg.NIR)
    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    return self

class QCOMArgsState(HCQArgsState):
  def __init__(self, buf:HCQBuffer, prg:QCOMProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(buf, prg, bufs, vals=vals)
    ctypes.memset(int(self.buf.va_addr), 0, prg.kernargs_alloc_size)

    ubos = [b for i,b in enumerate(bufs) for _,dt in prg.buf_dtypes[i] if not isinstance(dt, ImageDType)]
    uavs = [(dt,b) for i,b in enumerate(bufs) for _,dt in prg.buf_dtypes[i] if isinstance(dt, ImageDType)]
    # NIR can reorder images to different texture slots
    ibos, texs = uavs[:prg.ibo_cnt], [uavs[prg.ibo_cnt + (prg.tex_to_image[i] if prg.NIR else i)] for i in range(prg.tex_cnt)]
    for cnst_val,cnst_off,cnst_sz in prg.consts_info:
      to_mv(cast(int, self.buf.va_addr) + cnst_off, cnst_sz)[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')

    if prg.samp_cnt > 0: to_mv(int(self.buf.va_addr) + prg.samp_off, len(prg.samplers) * 4).cast('I')[:] = array.array('I', prg.samplers)
    if prg.NIR:
      self.bind_sints_to_buf(*[b.va_addr for b in ubos], buf=self.buf, fmt='Q', offset=prg.buf_off)
      self.bind_sints_to_buf(*vals, buf=self.buf, fmt='I', offset=prg.buf_off + len(ubos) * 8)
    else:
      for i, b in enumerate(ubos): self.bind_sints_to_buf(b.va_addr, buf=self.buf, fmt='Q', offset=prg.buf_offs[i])
      for i, v in enumerate(vals): self.bind_sints_to_buf(v, buf=self.buf, fmt='I', offset=prg.buf_offs[i+len(ubos)])

    self.bind_sints_to_buf(*flatten(build_a6xx_tex_descriptor(dt, b.va_addr) for dt,b in texs), buf=self.buf, fmt='I', offset=prg.tex_off)
    self.bind_sints_to_buf(*flatten(build_a6xx_tex_descriptor(dt, b.va_addr, ibo=True) for dt,b in ibos), buf=self.buf, fmt='I', offset=prg.ibo_off)

class QCOMProgram(HCQProgram):
  tex_to_image: list[int]; buf_off: int; hw_stack_offset: int; max_threads: int
  def __init__(self, dev: QCOMDevice, name: str, lib: bytes, buf_dtypes=[], **kwargs):
    self.dev: QCOMDevice = dev
    self.buf_dtypes, self.name, self.NIR = buf_dtypes, name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR: parse_ir3_shader(self, lib)
    else: self._parse_lib(lib)

    self.lib_gpu: HCQBuffer = self.dev.allocator.alloc(self.image_size, buf_spec:=BufferSpec(cpu_access=True, nolru=True))
    to_mv(self.lib_gpu.va_addr, self.image_size)[:] = self.image

    compute_program_sizes(self)
    dev._ensure_stack_size(self.hw_stack_offset * 4)

    super().__init__(QCOMArgsState, self.dev, self.name, kernargs_alloc_size=self.kernargs_alloc_size)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait=False, **kw):
    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

  def _parse_lib(self, lib):
    # Extract image binary
    self.image_size = _read_lib(lib, 0x100)
    self.image = bytearray(lib[(image_offset:=_read_lib(lib, 0xc0)):image_offset+self.image_size])

    # Parse image descriptors
    image_desc_off = _read_lib(lib, 0x110)
    self.prg_offset, self.brnchstck = _read_lib(lib, image_desc_off+0xc4), _read_lib(lib, image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(lib, image_desc_off+0xc8), _read_lib(lib, image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.consts_info = []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(lib, image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]
    else: self.samplers = []

    # Collect kernel arguments (buffers) info.
    bdoff, binfos = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file, []
    while bdoff + 32 <= len(lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("8I", lib[bdoff:bdoff+32])
      if length == 0: break
      binfos.append((offset_words * 4, typ))
      bdoff += length
    self.buf_offs = [off for off,typ in binfos if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(typ is BUFTYPE_TEX for _,typ in binfos), sum(typ is BUFTYPE_IBO for _,typ in binfos)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt

    if _read_lib(lib, 0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(lib, 0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", lib[cdoff:cdoff+4])[0], *struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

class QCOMTextureInfo:
  def __init__(self, pitch:int, real_stride:int, desc:list[int], ibo:list[int]):
    self.pitch, self.real_stride, self.desc, self.ibo = pitch, real_stride, desc, ibo

class QCOMAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, opts:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_map(opts.external_ptr, size) if opts.external_ptr else self.dev._gpu_alloc(size)

  def _do_copy(self, src_addr, dest_addr, size, prof_text):
    with cpu_profile(prof_text, f"{self.dev.device}:COPY"): ctypes.memmove(dest_addr, src_addr, size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self._do_copy(mv_address(src), dest.cpu_view().addr, src.nbytes, f"TINY -> {self.dev.device}")

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()

    self._do_copy(src.cpu_view().addr, mv_address(dest), src.size, f"{self.dev.device} -> TINY")

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.cpu_view().addr, src.size)

  def _do_free(self, opaque, options:BufferSpec): self.dev._gpu_free(opaque)

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class QCOMDevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)
    self.dummy_addr = int(self._gpu_alloc(0x1000).va_addr)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    self.cmd_buf = self._gpu_alloc(16 << 20)
    self.cmd_buf_allocator = BumpAllocator(size=self.cmd_buf.size, base=int(self.cmd_buf.va_addr), wrap=True)

    self.border_color_buf = self._gpu_alloc(0x1000, fill_zeroes=True)

    self.last_cmd:int = 0

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.gpu_id = (info.chip_id >> 24, (info.chip_id >> 16) & 0xFF, (info.chip_id >> 8) & 0xFF)

    # a7xx start with 730x or 'Cxxx', a8xx starts 'Exxx'
    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={info.chip_id:#x}")

    if PROFILE and self.gpu_id[:2] < (7, 3):
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    super().__init__(device, QCOMAllocator(self), [QCOMCLRenderer, IR3Renderer], functools.partial(QCOMProgram, self), QCOMSignal,
                     functools.partial(QCOMComputeQueue, self), arch="a%d%d%d" % self.gpu_id)

  def _gpu_alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> HCQBuffer:
    flags |= flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return HCQBuffer(va_addr=va_addr, size=size, meta=(alloc, True), view=MMIOInterface(va_addr, size, fmt='B'), owner=self)

  def _gpu_map(self, ptr:int, size:int) -> HCQBuffer:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ctypes.c_uint64(ceildiv(ptr + size - ptr_line_aligned, 64)))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return HCQBuffer(mi.gpuaddr + (ptr - ptr_aligned), size=size, meta=(mi, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
    except OSError as e:
      if e.errno == 14: return HCQBuffer(va_addr=ptr, size=size, meta=(None, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def _gpu_free(self, mem:HCQBuffer):
    if mem.meta[0] is None: return # external (gpu) ptr
    if not mem.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=mem.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.meta[0].id)
      FileIOInterface.munmap(mem.va_addr, mem.meta[0].mmapsize)

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self._gpu_alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self._gpu_free(self._stack)
      self._stack = self._gpu_alloc(sz)

  def _at_profile_finalize(self):
    super()._at_profile_finalize()
    with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")
