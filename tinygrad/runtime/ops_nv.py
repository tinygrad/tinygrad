from __future__ import annotations
import os, ctypes, pathlib, re, fcntl, functools, mmap, struct, tempfile, hashlib, subprocess, time, array
from typing import Tuple, List, Any, cast
from tinygrad.device import Compiled, Compiler, LRUAllocator, BufferOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up, to_char_p_p, DEBUG, prod
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.ops_cuda import check as cuda_check, _get_bytes, CUDACompiler
import tinygrad.runtime.autogen.cuda as cuda
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

if MOCKGPU:=getenv("MOCKGPU"):
  import extra.mockgpu.mockgpu  # noqa: F401
  libc.mmap = extra.mockgpu.mockgpu._mmap # type: ignore
  libc.munmap = extra.mockgpu.mockgpu._munmap # type: ignore

def nv_iowr(fd, nr, args):
  ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def rm_alloc(fd, clss, root, parant, params):
  made = nv_gpu.NVOS21_PARAMETERS(hRoot=root, hObjectParent=parant, hClass=clss,
                                  pAllocParms=ctypes.cast(ctypes.byref(params), ctypes.POINTER(None)) if params is not None else None) # type: ignore
  nv_iowr(fd, nv_gpu.NV_ESC_RM_ALLOC, made)
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def rm_control(fd, cmd, client, obj, params):
  made = nv_gpu.NVOS54_PARAMETERS(hClient=client, hObject=obj, cmd=cmd, paramsSize=ctypes.sizeof(params),
                                  params=ctypes.cast(ctypes.byref(params), ctypes.POINTER(None)) if params is not None else None) # type: ignore
  nv_iowr(fd, nv_gpu.NV_ESC_RM_CONTROL, made)
  if made.status != 0: raise RuntimeError(f"rm_control returned {made.status}")
  return made

def uvm_ioctl(cmd, sttyp, fd, **kwargs):
  ret = fcntl.ioctl(fd, cmd, made:=sttyp(**kwargs))
  if ret != 0: raise RuntimeError(f"uvm_ioctl returned {ret}")
  if made.rmStatus != 0: raise RuntimeError(f"uvm_ioctl struct returned {made.rmStatus}")
  return made

def make_uvm_type():
  fxns = {name.replace("UVM_", "").lower():
          functools.partial(uvm_ioctl, dt, getattr(nv_gpu, name+"_PARAMS"))
          for name,dt in nv_gpu.__dict__.items() if name.startswith("UVM_") and nv_gpu.__dict__.get(name+"_PARAMS")}
  return type("NVUVM", (object, ), fxns)
uvm = make_uvm_type()

def make_qmd_struct_type():
  fields = []
  bits = [(name,dt) for name,dt in nv_gpu.__dict__.items() if name.startswith("NVC6C0_QMDV03_00") and isinstance(dt, tuple)]
  bits += [(name+f"_{i}",dt(i)) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith("NVC6C0_QMDV03_00") and callable(dt)]
  bits = sorted(bits, key=lambda x: x[1][1])
  for i,(name, data) in enumerate(bits):
    if i > 0 and (gap:=(data[1] - bits[i-1][1][0] - 1)) != 0:  fields.append((f"_reserved{i}", ctypes.c_uint32, gap))
    fields.append((name.replace("NVC6C0_QMDV03_00_", "").lower(), ctypes.c_uint32, data[0]-data[1]+1))
  return init_c_struct_t(tuple(fields))
qmd_struct_t = make_qmd_struct_type()
assert ctypes.sizeof(qmd_struct_t) == 0x40 * 4

def nvmethod(subc, mthd, size, typ=2): return (typ << 28) | (size << 16) | (subc << 13) | (mthd >> 2)
def nvdata64(data): return (data >> 32, data & 0xFFFFFFFF)
def nvdata64_le(data): return (data & 0xFFFFFFFF, data >> 32)

class NVCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    #NVCompiler.compiler_opts = replace(NVCompiler.compiler_opts, has_tensor_cores=int(arch[3:]) >= 80)
    cuda_check(cuda.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.compile_options = [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_nv_{self.arch}")
  def compile(self, src:str) -> bytes:
    cuda_check(cuda.nvrtcCreateProgram(ctypes.byref(prog := cuda.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = cuda.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options]))

    if status != 0:
      raise RuntimeError(f"compile failed: {_get_bytes(prog, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, cuda_check).decode()}")
    return _get_bytes(prog, cuda.nvrtcGetCUBIN, cuda.nvrtcGetCUBINSize, cuda_check)

class HWComputeQueue:
  def __init__(self): self.q = []
  def ptr(self) -> int: return len(self.q)

  def copy_from_cpu(self, gpuaddr, data):
    self.q += [nvmethod(1, nv_gpu.NVC6C0_OFFSET_OUT_UPPER, 2), *nvdata64(gpuaddr)]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LINE_LENGTH_IN, 2), len(data)*4, 0x1]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LAUNCH_DMA, 1), 0x41]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LOAD_INLINE_DATA, len(data), typ=6)] + [x for x in data]
    return self

  def exec(self, prg, kernargs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1)):
    prg.qmd.cta_raster_width, prg.qmd.cta_raster_height, prg.qmd.cta_raster_depth = global_size
    prg.qmd.cta_thread_dimension0, prg.qmd.cta_thread_dimension1, prg.qmd.cta_thread_dimension2 = local_size
    prg.qmd.constant_buffer_addr_lower_0 = kernargs & 0xffffffff
    prg.qmd.constant_buffer_addr_upper_0 = kernargs >> 32
    self.q += [nvmethod(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1), (1 << 12) | (1 << 4) | (1 << 0)]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A, 0x42), *nvdata64((kernargs + round_up(prg.constbuf_0_size, 1 << 8)) >> 8)]
    self.q += [x for x in to_mv(ctypes.addressof(prg.qmd), ctypes.sizeof(prg.qmd)).cast("I")]
    return self

  def update_exec(self, cmd_ptr, global_size, local_size):
    # Patch the exec cmd with new launch dims
    assert self.q[cmd_ptr + 2] == nvmethod(1, nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A, 0x42),"The pointer does not point to a packet of this type"
    self.q[cmd_ptr + 5 + 12 : cmd_ptr + 5 + 15] = global_size
    self.q[cmd_ptr + 5 + 18] = (self.q[cmd_ptr + 5 + 18] & 0xffff) | ((local_size[0] & 0xffff) << 16)
    self.q[cmd_ptr + 5 + 19] = (local_size[1] & 0xffff) | ((local_size[2] & 0xffff) << 16)

  def wait(self, signal, value=0):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(ctypes.addressof(from_mv(signal))), *nvdata64_le(value),
               (3 << 0) | (1 << 12) | (1 << 24)] # ACQUIRE | ACQUIRE_SWITCH_TSG | PAYLOAD_SIZE_64BIT
    return self

  def signal(self, signal, value=0, timestamp=False):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(ctypes.addressof(from_mv(signal))), *nvdata64_le(value),
               (1 << 0) | (1 << 20) | (1 << 24) | ((1 << 25) if timestamp else 0)] # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.q += [nvmethod(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 1), 0x0]
    return self

  def submit(self, dev:NVDevice):
    if len(self.q) == 0: return
    assert len(self.q) < (1 << 21)
    dev.cmdq[dev.cmdq_wptr//4:dev.cmdq_wptr//4+len(self.q)] = array.array('I', self.q)
    fifo_entry = dev.compute_put_value % dev.compute_gpfifo_entries
    dev.compute_gpu_ring[fifo_entry] = ((dev.cmdq_page.base+dev.cmdq_wptr)//4 << 2) | (len(self.q) << 42) | (1 << 41)
    dev.compute_gpu_ring_controls.GPPut = (dev.compute_put_value + 1) % dev.compute_gpfifo_entries
    dev.compute_put_value += 1
    dev.gpu_mmio[0x90 // 4] = dev.compute_gpfifo_token
    dev.cmdq_wptr += len(self.q) * 4

class HWCopyQueue:
  def __init__(self): self.q = []

  def copy(self, dest, src, copy_size):
    self.q += [nvmethod(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, 4), *nvdata64(src), *nvdata64(dest)]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, 1), copy_size]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LAUNCH_DMA, 1), 0x182] # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH
    return self

  def wait(self, signal, value=0):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(ctypes.addressof(from_mv(signal))), *nvdata64_le(value),
               (3 << 0) | (1 << 12) | (1 << 24)] # ACQUIRE | ACQUIRE_SWITCH_TSG | PAYLOAD_SIZE_64BIT
    return self

  def signal(self, signal, value=0, timestamp=False):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(ctypes.addressof(from_mv(signal))), *nvdata64_le(value),
               (1 << 0) | (1 << 20) | (1 << 24) | ((1 << 25) if timestamp else 0)] # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.q += [nvmethod(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 1), 0x0]
    return self

  def submit(self, dev:NVDevice):
    if len(self.q) == 0: return
    dev.cmdq[dev.cmdq_wptr//4:dev.cmdq_wptr//4+len(self.q)] = array.array('I', self.q)
    fifo_entry = dev.dma_put_value % dev.dma_gpfifo_entries
    dev.dma_gpu_ring[fifo_entry] = ((dev.cmdq_page.base+dev.cmdq_wptr)//4 << 2) | (len(self.q) << 42)
    dev.dma_gpu_ring_controls.GPPut = (dev.dma_put_value + 1) % dev.dma_gpfifo_entries
    dev.dma_put_value += 1
    dev.gpu_mmio[0x90 // 4] = dev.dma_gpfifo_token
    dev.cmdq_wptr += len(self.q) * 4

SHT_PROGBITS, SHT_NOBITS, SHF_ALLOC, SHF_EXECINSTR = 0x1, 0x8, 0x2, 0x4
class NVProgram:
  def __init__(self, device:NVDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 6:
      try:
        fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
        with open(fn + ".cubin", "wb") as f: f.write(lib)
        print(subprocess.check_output(["nvdisasm", fn+".cubin"]).decode('utf-8'))
      except Exception as e: print("failed to disasm cubin", str(e))

    self.global_init, self.shmem_usage = None, 0
    constant_buffers_data = {}

    if MOCKGPU:
      self.program, self.registers_usage = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), 0x10
      constant_buffers_data[0] = memoryview(bytearray(0x190))
    else:
      _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
      sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]
      shstrtab = memoryview(bytearray(self.lib[sections[_shstrndx][4]:sections[_shstrndx][4]+sections[_shstrndx][5]]))
      for sh_name, sh_type, sh_flags, _, sh_offset, sh_size, _, sh_info, _ in sections:
        section_name = shstrtab[sh_name:].tobytes().split(b'\0', 1)[0].decode('utf-8')
        if sh_type == SHT_NOBITS and sh_flags & SHF_ALLOC: self.shmem_usage = sh_size
        elif sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC and sh_flags & SHF_EXECINSTR:
          self.program = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
          self.registers_usage = sh_info >> 24
        if match := re.match(r'\.nv\.constant(\d+)', section_name):
          constant_buffers_data[int(match.group(1))] = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
        if section_name == ".nv.global.init": self.global_init = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
        elif section_name == ".nv.info":
          section_data = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
          for i in range(sh_size // 12):
            if section_data[i * 3 + 0] & 0xffff == 0x1204 and section_data[i * 3 + 2] + 0x240 > self.device.slm_per_thread:
              raise RuntimeError("too high local memory")

    # Registers allocation granularity per warp is 256, warp allocaiton granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(self.registers_usage * 32, 256)) // 4) * 4 * 32

    # Load program and constant buffers (if any)
    self.lib_sz = round_up(round_up(self.program.nbytes, 128) + round_up(0 if self.global_init is None else self.global_init.nbytes, 128) +
                           sum([round_up(x.nbytes, 128) for i,x in constant_buffers_data.items()]), 0x1000)
    self.lib_gpu = self.device.allocator.alloc(self.lib_sz)
    for st in range(0, len(self.program), 4095):
      HWComputeQueue().copy_from_cpu(self.lib_gpu.base+st*4, self.program[st:st+4095]).submit(self.device)

    self.constbuffer_0 = [0] * 88
    self.constbuffer_0[6:12] = [*nvdata64_le(self.device.shared_mem_window), *nvdata64_le(self.device.local_mem_window), *nvdata64_le(0xfffdc0)]

    smem_config = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1
    self.qmd = qmd_struct_t(qmd_group_id=0x3f, sm_global_caching_enable=1, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
                            invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1,
                            cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, qmd_major_version=3,
                            shared_memory_size=max(0x400, round_up(self.shmem_usage, 0x100)), min_sm_config_shared_mem_size=smem_config,
                            max_sm_config_shared_mem_size=0x1a, register_count_v=self.registers_usage, target_sm_config_shared_mem_size=smem_config,
                            barrier_count=1, shader_local_memory_high_size=self.device.slm_per_thread, program_prefetch_size=0x10, sass_version=0x89,
                            program_address_lower=self.lib_gpu.base&0xffffffff, program_address_upper=self.lib_gpu.base>>32,
                            program_prefetch_addr_lower_shifted=self.lib_gpu.base>>8, program_prefetch_addr_upper_shifted=self.lib_gpu.base>>40,
                            constant_buffer_size_shifted4_0=0x190, constant_buffer_valid_0=1, constant_buffer_invalidate_0=1)

    # NV's kernargs is constbuffer (size 0x160), then arguments to the kernel follows. Kernargs also appends QMD at the end of the kernel.
    self.constbuf_0_size = constant_buffers_data[0].nbytes if 0 in constant_buffers_data else 0
    self.kernargs_segment_size = round_up(self.constbuf_0_size, 1 << 8) + (8 << 8)
    self.kernargs_offset = 0x160

    # constant buffer 0 is filled for each program, no need to copy it from elf (it's just zeroes)
    if 0 in constant_buffers_data: constant_buffers_data.pop(0)

    off = round_up(self.program.nbytes, 128)
    if self.global_init is not None:
      # Constbuffer 4 contains a pointer to nv.global.init, load section and set up the pointer.
      assert 4 in constant_buffers_data and constant_buffers_data[4].nbytes == 8
      HWComputeQueue().copy_from_cpu(load_addr:=(self.lib_gpu.base + off), self.global_init).submit(self.device)
      constant_buffers_data[4][0:2] = memoryview(struct.pack('Q', load_addr)).cast('I')
      off += round_up(self.global_init.nbytes, 128)

    for i,data in constant_buffers_data.items():
      self.qmd.__setattr__(f'constant_buffer_addr_upper_{i}', (self.lib_gpu.base + off) >> 32)
      self.qmd.__setattr__(f'constant_buffer_addr_lower_{i}', (self.lib_gpu.base + off) & 0xffffffff)
      self.qmd.__setattr__(f'constant_buffer_size_shifted4_{i}', data.nbytes)
      self.qmd.__setattr__(f'constant_buffer_valid_{i}', 1)

      HWComputeQueue().copy_from_cpu(self.lib_gpu.base + off, data).submit(self.device)
      off += round_up(data.nbytes, 128)

    HWComputeQueue().signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
    self.device.timeline_value += 1
    self.device.synchronize()

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device.allocator.free(self.lib_gpu, self.lib_sz)

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if prod(local_size) > 1024 or self.max_threads < prod(local_size): raise RuntimeError("Too many resources requsted for launch")
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))

    if self.device.kernargs_ptr >= (self.device.kernargs_page.base + self.device.kernargs_page.length - self.kernargs_segment_size):
      self.device.kernargs_ptr = self.device.kernargs_page.base

    # HACK: Save counts of args and vars to "unused" constbuffer for later extraction in mockgpu to pass into gpuocelot.
    if MOCKGPU: self.constbuffer_0[0:2] = [len(args), len(vals)]
    kernargs = [arg_half for arg in args for arg_half in nvdata64_le(arg.base)] + [val for val in vals]

    queue = HWComputeQueue()
    queue.wait(self.device.timeline_signal, self.device.timeline_value - 1)
    if wait: queue.signal(self.device.time_event_st, timestamp=True)
    queue.copy_from_cpu(self.device.kernargs_ptr, self.constbuffer_0 + kernargs)
    queue.exec(self, self.device.kernargs_ptr, global_size, local_size)
    if wait: queue.signal(self.device.time_event_en, timestamp=True)
    queue.signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
    self.device.timeline_value += 1
    self.device.kernargs_ptr += self.kernargs_segment_size

    if wait:
      self.device._wait_signal(self.device.timeline_signal, self.device.timeline_value - 1)
      return (self.device.time_event_en[1] - self.device.time_event_st[1]) / 1e9

class NVAllocator(LRUAllocator):
  def __init__(self, device:NVDevice):
    self.device = device
    self.b = [self.device._gpu_host_alloc(2 << 20) for _ in range(16)]
    self.b_timeline = [0] * len(self.b)
    self.b_next = 0
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    if options.host: return self.device._gpu_host_alloc(size)
    else: return self.device._gpu_alloc(size, map_to_cpu=options.cpu_access)

  def _free(self, gpumem, options:BufferOptions):
    NVDevice.synchronize_system()
    if options.host: self.device._gpu_host_free(gpumem)
    else: self.device._gpu_free(gpumem)

  def copyin(self, dest, src: memoryview):
    for i in range(0, src.nbytes, self.b[0].length):
      self.b_next = (self.b_next + 1) % len(self.b)
      NVDevice._wait_signal(self.device.timeline_signal, self.b_timeline[self.b_next])
      ctypes.memmove(self.b[self.b_next].va_addr, from_mv(src[i:]), lsize:=min(self.b[self.b_next].length, src.nbytes-i))
      HWCopyQueue().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                   .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      self.b_timeline[self.b_next] = self.device.timeline_value
      self.device.timeline_value += 1

  def copyout(self, dest:memoryview, src):
    NVDevice.synchronize_system()
    for i in range(0, dest.nbytes, self.b[0].length):
      HWCopyQueue().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                   .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].length, dest.nbytes-i)) \
                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      NVDevice._wait_signal(self.device.timeline_signal, self.device.timeline_value)
      self.device.timeline_value += 1

      ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def transfer(self, dest, src, sz:int, src_dev=None, dest_dev=None):
    src_dev._gpu_map(dest)
    HWCopyQueue().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                 .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                 .copy(dest.va_addr, src.va_addr, sz) \
                 .signal(src_dev.timeline_signal, src_dev.timeline_value).submit(src_dev)
    HWComputeQueue().wait(src_dev.timeline_signal, src_dev.timeline_value).submit(dest_dev)
    src_dev.timeline_value += 1

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVDevice(Compiled):
  root = None
  fd_ctl: int = -1
  fd_uvm: int = -1
  gpus_info = None
  signals_page:Any = None
  signal_number: int = 32
  uvm_vaddr: int = 0x1000000000
  host_object_enumerator: int = 0x1000
  devices: List[NVDevice] = []

  def _new_gpu_fd(self):
    fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.device, hMemory=memory_handle, length=size, flags=flags))
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {made.params.status}")
    return libc.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev, 0)

  def _gpu_alloc(self, size:int, contig=False, huge_page=False, va_addr=None, map_to_cpu=False, map_flags=0):
    size = round_up(size, align:=((4 << 10) if huge_page else (2 << 20))) # TODO: need hugepage option, any speedup?
    alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, alignment=align, offset=0, limit=size-1, format=6, size=size,
      attr=(((nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE << 23) if huge_page else 0) |
            ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contig else nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27)),
      attr2=((nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES << 2) |
             ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20) if huge_page else 0)),
      flags=(nv_gpu.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM | nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED |
             nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED))
    mem_handle = rm_alloc(self.fd_ctl, nv_gpu.NV1_MEMORY_USER, self.root, self.device, alloc_params).hObjectNew

    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size, alignment=align)
    if map_to_cpu: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags)
    return self._gpu_uvm_map(va_addr, size, mem_handle)

  def _gpu_system_alloc(self, size:int, va_addr=None, map_to_cpu=False, map_flags=0):
    alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, type=13,
      attr=(nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27) | (nv_gpu.NVOS32_ATTR_LOCATION_PCI << 25),
      attr2=(nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO << 2),
      flags=(nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED |
             nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED), format=6, size=size, alignment=(4<<10), offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nv_gpu.NV1_MEMORY_SYSTEM, self.root, self.device, alloc_params).hObjectNew

    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size)
    if map_to_cpu: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=True)

    return self._gpu_uvm_map(va_addr, size, mem_handle)

  def _gpu_host_alloc(self, size):
    va_base = self._alloc_gpu_vaddr(sz:=round_up(size, 4 << 10))
    libc.mmap(va_base, sz, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
    return self._map_to_gpu(va_base, sz)

  def _gpu_free(self, mem):
    made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.device, hObjectOld=mem.hMemory)
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
    if made.status != 0: raise RuntimeError(f"_gpu_free returned {made.status}")
    uvm.free(self.fd_uvm, base=mem.base, length=mem.length)

  def _gpu_host_free(self, mem):
    uvm.free(self.fd_uvm, base=mem.base, length=mem.length)
    libc.munmap(mem.base, mem.length)

  def _map_to_gpu(self, va_base, size):
    NVDevice.host_object_enumerator += 1
    flags = ((nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) |
             (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30))
    made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.device, flags=flags,
      hObjectNew=NVDevice.host_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, pMemory=va_base, limit=size-1), fd=-1)
    nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_map_to_gpu returned {made.params.status}")
    return self._gpu_uvm_map(va_base, size, made.params.hObjectNew)

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True) -> nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS:
    if create_range: uvm.create_external_range(self.fd_uvm, base=va_base, length=size)
    gpu_attrs = (nv_gpu.struct_c__SA_UvmGpuMappingAttributes*256)(
      nv_gpu.struct_c__SA_UvmGpuMappingAttributes(gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), gpuMappingType = 1))

    # NOTE: va_addr is set to make rawbufs compatable with AMD.
    return uvm.map_external_allocation(self.fd_uvm, base=va_base, length=size, rmCtrlFd=self.fd_ctl, hClient=self.root, hMemory=mem_handle,
                                       gpuAttributesCount=1, perGpuAttributes=gpu_attrs, va_addr=va_base)

  def _gpu_map(self, mem):
    if self.gpu_uuid in getattr(mem, "mapped_gpu_ids", []): return
    mem.__setattr__("mapped_gpu_ids", getattr(mem, "mapped_gpu_ids", []) + [self.gpu_uuid])
    return self._gpu_uvm_map(mem.base, mem.length, mem.hMemory, create_range=False)

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10)):
    NVDevice.uvm_vaddr = (res_va:=round_up(NVDevice.uvm_vaddr, alignment)) + size
    return res_va

  def __init__(self, device:str=""):
    if NVDevice.root is None:
      NVDevice.fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      fd_uvm_2 = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.root = rm_alloc(self.fd_ctl, nv_gpu.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew
      uvm.initialize(self.fd_uvm)
      try:
        uvm.mm_initialize(fd_uvm_2, uvmFd=self.fd_uvm)
      except RuntimeError:
        pass  # this error is okay, CUDA hits it too

      NVDevice.gpus_info = (nv_gpu.nv_ioctl_card_info_t*64)()
      nv_iowr(NVDevice.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, NVDevice.gpus_info)

    # TODO: Get classes from NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.fd_dev = self._new_gpu_fd()

    assert NVDevice.gpus_info[self.device_id].valid
    gpu_info = nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS(gpuId=NVDevice.gpus_info[self.device_id].gpu_id)
    rm_control(self.fd_ctl, nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, self.root, self.root, gpu_info)
    device_id = NVDevice.gpus_info[self.device_id].pci_info.device_id
    self.compute_type = nv_gpu.AMPERE_COMPUTE_B if device_id in [0x2204, 0x2206] else nv_gpu.ADA_COMPUTE_A

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=gpu_info.deviceInstance, hClientShare=self.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.device = rm_alloc(self.fd_ctl, nv_gpu.NV01_DEVICE_0, self.root, self.root, device_params).hObjectNew
    self.subdevice = rm_alloc(self.fd_ctl, nv_gpu.NV20_SUBDEVICE_0, self.root, self.device, None).hObjectNew
    self.usermode = rm_alloc(self.fd_ctl, nv_gpu.TURING_USERMODE_A, self.root, self.subdevice, None).hObjectNew
    gpu_mmio_ptr = self._gpu_map_to_cpu(self.usermode, 0x10000, flags=2)
    self.gpu_mmio = to_mv(gpu_mmio_ptr, 0x10000).cast("I")

    boost_params = nv_gpu.struct_NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff, flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | \
      (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX << 0)))
    rm_control(self.fd_ctl, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, self.root, self.subdevice, boost_params)

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = rm_alloc(self.fd_ctl, nv_gpu.FERMI_VASPACE_A, self.root, self.device, vaspace_params).hObjectNew

    gpu_uuid_params = nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
    rm_control(self.fd_ctl, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, self.root, self.subdevice, gpu_uuid_params)
    self.gpu_uuid = (ctypes.c_ubyte*16)(*[gpu_uuid_params.data[i] for i in range(16)])

    uvm.register_gpu(self.fd_uvm, rmCtrlFd=-1, gpu_uuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid))
    uvm.register_gpu_vaspace(self.fd_uvm, gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl,
                             hClient=self.root, hVaSpace=vaspace)

    for dev in self.devices:
      uvm.enable_peer_access(self.fd_uvm, gpuUuidA=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), gpuUuidB=nv_gpu.struct_nv_uuid(uuid=dev.gpu_uuid))

    if NVDevice.signals_page is None: NVDevice.signals_page = self._gpu_system_alloc(0x10000, map_to_cpu=True)
    else: self._gpu_map(NVDevice.signals_page)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = rm_alloc(self.fd_ctl, nv_gpu.KEPLER_CHANNEL_GROUP_A, self.root, self.device, channel_params).hObjectNew

    gpfifo = self._gpu_alloc(0x200000, contig=True, huge_page=True, map_to_cpu=True, map_flags=0x10d0000)

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = rm_alloc(self.fd_ctl, nv_gpu.FERMI_CONTEXT_SHARE_A, self.root, channel_group, ctxshare_params).hObjectNew

    self.compute_gpfifo_entries: int = 0x10000
    self.compute_gpfifo_token: int = self._gpu_fifo_setup(gpfifo, ctxshare, channel_group, offset=0, entries=self.compute_gpfifo_entries)
    self.compute_gpu_ring: memoryview = to_mv(gpfifo.base, self.compute_gpfifo_entries * 8).cast("Q")
    self.compute_gpu_ring_controls = nv_gpu.AmpereAControlGPFifo.from_address(gpfifo.base + self.compute_gpfifo_entries * 8)
    self.compute_put_value: int = 0

    self.dma_gpfifo_entries: int = 0x10000
    self.dma_gpfifo_token: int = self._gpu_fifo_setup(gpfifo, ctxshare, channel_group, offset=0x100000, entries=self.dma_gpfifo_entries)
    self.dma_gpu_ring: memoryview = to_mv(gpfifo.base + 0x100000, self.dma_gpfifo_entries * 8).cast("Q")
    self.dma_gpu_ring_controls = nv_gpu.AmpereAControlGPFifo.from_address(gpfifo.base + 0x100000 + self.dma_gpfifo_entries * 8)
    self.dma_put_value: int = 0

    en_fifo_params = nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
    rm_control(self.fd_ctl, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, self.root, channel_group, en_fifo_params)

    self.timeline_value: int = 1
    self.timeline_signal = NVDevice._get_signal(self.device_id * 2)
    self._shadow_timeline_signal = NVDevice._get_signal(self.device_id * 2 + 1)
    self.time_event_st, self.time_event_en = NVDevice._get_signal(), NVDevice._get_signal()

    self.cmdq_page: nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS = self._gpu_alloc(0x200000, map_to_cpu=True, huge_page=True)
    self.cmdq: memoryview = to_mv(self.cmdq_page.base, 0x200000).cast("I")
    self.cmdq_wptr: int = 0 # in bytes

    self.kernargs_page: nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS = self._gpu_alloc(0x4000000, map_to_cpu=True)
    self.kernargs_ptr: int = self.kernargs_page.base

    self.arch: str = "sm_89" if not MOCKGPU else "sm_35" # TODO: fix

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, NVAllocator(self), NVRenderer(self.arch), CUDACompiler(self.arch) if MOCKGPU else NVCompiler(self.arch),
                     functools.partial(NVProgram, self), functools.partial(HCQGraph, NVDevice, HWComputeQueue, HWCopyQueue))

    self._cmdq_setup_compute_gpfifo()
    self._cmdq_setup_dma_gpfifo()

    NVDevice.devices.append(self)

  def synchronize(self):
    NVDevice._wait_signal(self.timeline_signal, self.timeline_value - 1)
    self.cmdq_wptr = 0

    if self.timeline_value > (1 << 63):
      self.timeline_signal, self._shadow_timeline_signal = self._shadow_timeline_signal, self.timeline_signal
      self.timeline_signal[0], self.timeline_value = 0, 1
      cast(NVAllocator, self.allocator).b_timeline = [0] * len(cast(NVAllocator, self.allocator).b)

  @staticmethod
  def synchronize_system():
    for d in NVDevice.devices: d.synchronize()

  @classmethod
  def _set_signal(self, sig, value): sig[0] = value

  @classmethod
  def _get_signal(self, num=None, value=0) -> memoryview:
    if num is None:
      self.signal_number += 1
      if self.signals_page and self.signal_number * 16 >= self.signals_page.length: self.signal_number = 32
      num = self.signal_number
    sig = to_mv(self.signals_page.base + num * 16, 16).cast("Q")
    sig[0] = value
    return sig

  @classmethod
  def _wait_signal(self, signal, value=0, timeout=10000):
    start_time = time.time() * 1000
    sem_value = signal[0]
    while sem_value < value:
      sem_value = signal[0]
      if time.time() * 1000 - start_time > timeout: raise RuntimeError(f"wait_result: {timeout} ms TIMEOUT!")

  def _gpu_fifo_setup(self, gpfifo, ctxshare, channel_group, offset, entries=0x400):
    notifier = self._gpu_system_alloc(48 << 20)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=notifier.hMemory, hObjectBuffer=gpfifo.hMemory,
      gpFifoOffset=gpfifo.base+offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset))
    gpfifo = rm_alloc(self.fd_ctl, nv_gpu.AMPERE_CHANNEL_GPFIFO_A, self.root, channel_group, params).hObjectNew
    rm_alloc(self.fd_ctl, self.compute_type, self.root, gpfifo, None)
    rm_alloc(self.fd_ctl, nv_gpu.AMPERE_DMA_COPY_B, self.root, gpfifo, None)

    ws_token_params = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    rm_control(self.fd_ctl, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, self.root, gpfifo, ws_token_params)
    assert ws_token_params.workSubmitToken != -1

    channel_base = self._alloc_gpu_vaddr(0x4000000)
    uvm.register_channel(self.fd_uvm, gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl, hClient=self.root,
                         hChannel=gpfifo, base=channel_base, length=0x4000000)

    return ws_token_params.workSubmitToken

  def _cmdq_setup_compute_gpfifo(self):
    self.slm_per_thread = 0x900
    bytes_per_warp = round_up(self.slm_per_thread * 32, 0x200)
    bytes_per_tpc = round_up(bytes_per_warp * 48 * 2, 0x8000)
    self.shader_local_mem = self._gpu_alloc(round_up(bytes_per_tpc * 64, 0x20000), huge_page=True, contig=True).base

    # Set windows addresses to not collide with other allocated buffers.
    self.shared_mem_window, self.local_mem_window = 0xfe000000, 0xff000000

    queue = HWComputeQueue()
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_OBJECT, 1), self.compute_type]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 2), *nvdata64(self.shader_local_mem)]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3), *nvdata64(bytes_per_tpc), 0x40]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2), *nvdata64(self.local_mem_window)]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2), *nvdata64(self.shared_mem_window)]
    queue.signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1
    self.synchronize()

  def _cmdq_setup_dma_gpfifo(self):
    queue = HWCopyQueue()
    queue.q += [nvmethod(4, nv_gpu.NVC6C0_SET_OBJECT, 1), nv_gpu.AMPERE_DMA_COPY_B]
    queue.signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1
    self.synchronize()
