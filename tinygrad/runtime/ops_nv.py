from __future__ import annotations
import os, ctypes, pathlib, re, fcntl, functools, mmap, struct, tempfile, hashlib, subprocess
from typing import Tuple, Any
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions, CompilerOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up, to_char_p_p, DEBUG
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import to_mv, getenv, round_up
from tinygrad.runtime.ops_cuda import check as cuda_check, _get_bytes
import tinygrad.runtime.autogen.cuda as cuda
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl

libc = ctypes.CDLL("libc.so.6")
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_int]
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

def nv_iowr(fd, nr, args):
  ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def rm_alloc(fd, clss, root, parant, params):
  made = nv_gpu.NVOS21_PARAMETERS(hRoot=root, hObjectParent=parant, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)))
  nv_iowr(fd, nv_gpu.NV_ESC_RM_ALLOC, made)
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def rm_control(fd, cmd, client, obj, params):
  made = nv_gpu.NVOS54_PARAMETERS(hClient=client, hObject=obj, cmd=cmd, params=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)), paramsSize=ctypes.sizeof(params))
  nv_iowr(fd, nv_gpu.NV_ESC_RM_CONTROL, made)
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def uvm_ioctl(cmd, sttyp, fd, **kwargs):
  made = sttyp(**kwargs)
  ret = fcntl.ioctl(fd, cmd, made)
  if ret != 0: raise RuntimeError(f"uvm ioctl returned {ret}")
  if made.rmStatus != 0: raise RuntimeError(f"uvm ioctl struct returned {made.rmStatus}")
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
  compiler_opts = CompilerOptions("NV", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    NVCompiler.compiler_opts = NVCompiler.compiler_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    cuda_check(cuda.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.compile_options = [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_nv_{self.arch}")
  def render(self, name:str, uops) -> str: return CUDARenderer(name, uops)
  def compile(self, src:str) -> bytes:
    cuda_check(cuda.nvrtcCreateProgram(ctypes.byref(prog := cuda.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = cuda.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options]))

    if status != 0: raise RuntimeError(f"compile failed: {_get_bytes(prog, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, cuda_check).decode()}")
    return _get_bytes(prog, cuda.nvrtcGetCUBIN, cuda.nvrtcGetCUBINSize, cuda_check)

class HWComputeQueue:
  def __init__(self): self.q = []
  def copy_from_cpu(self, gpuaddr, data):
    self.q += [nvmethod(1, nv_gpu.NVC6C0_OFFSET_OUT_UPPER, 2), *nvdata64(gpuaddr)]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LINE_LENGTH_IN, 2), len(data)*4, 0x1]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LAUNCH_DMA, 1), 0x41]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_LOAD_INLINE_DATA, len(data), typ=6)] + [x for x in data]
    return self

  def exec(self, prg:NVProgram, qmd_ptr, kernargs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), completion_signal=None):
    prg.qmd.cta_raster_width, prg.qmd.cta_raster_height, prg.qmd.cta_raster_depth = global_size
    prg.qmd.cta_thread_dimension0, prg.qmd.cta_thread_dimension1, prg.qmd.cta_thread_dimension2 = local_size
    prg.qmd.constant_buffer_addr_lower_0 = kernargs & 0xffffffff
    prg.qmd.constant_buffer_addr_upper_0 = kernargs >> 32
    if completion_signal is not None:
      prg.qmd.release0_address_lower = (NVDevice.semaphores_page.base + completion_signal * 16) & 0xffffffff
      prg.qmd.release0_address_upper = (NVDevice.semaphores_page.base + completion_signal * 16) >> 32
      prg.qmd.release0_enable = 1
      prg.qmd.release0_reduction_enable = 1
      prg.qmd.release0_payload_lower = 1
      prg.qmd.release0_structure_size = 1
      prg.qmd.release0_membar_type = 1
    self.q += [nvmethod(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1), (1 << 12) | (1 << 4) | (1 << 0)]
    self.q += [nvmethod(1, nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A, 0x42), *nvdata64(qmd_ptr >> 8)]
    self.q += [x for x in to_mv(ctypes.addressof(prg.qmd), ctypes.sizeof(prg.qmd)).cast("I")]
    return self

  def wait(self, signal_id:int, value=0):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(NVDevice.semaphores_page.base + signal_id * 16), value, 0x0, (3 << 0) | (1 << 12) | (1 << 24)]
    return self

  def signal(self, signal_id:int, value=0):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(NVDevice.semaphores_page.base + signal_id * 16), 0x1, 0x0, (6 << 0) | (1 << 20) | (1 << 24) | (5 << 27)]
    self.q += [nvmethod(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 1), 0x0]
    return self

  def submit(self, dev:NVDevice):
    for i,packet in enumerate(self.q): dev.cmdq[dev.cmdq_wptr//4 + i] = packet
    fifo_entry = dev.compute_put_value % dev.compute_gpfifo_entries
    dev.compute_gpu_ring[fifo_entry] = ((dev.cmdq_page.base+dev.cmdq_wptr)//4 << 2) | (len(self.q) << 42) | (1 << 41) # | (1 << 63)
    dev.compute_gpu_ring_controls.GPPut = (dev.compute_put_value + 1) % dev.compute_gpfifo_entries
    dev.compute_put_value += 1
    dev.gpu_mmio[0x90 // 4] = dev.compute_gpfifo_token
    dev.cmdq_wptr += len(self.q) * 4

class HWCopyQueue:
  def __init__(self): self.q = []

  def copy(self, dest, src, copy_size):
    self.q += [nvmethod(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, 4), *nvdata64(src), *nvdata64(dest)]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, 1), copy_size]
    self.q += [nvmethod(4, nv_gpu.NVC6B5_LAUNCH_DMA, 1), 0x182]
    return self

  def wait(self, signal_id:int, value=0):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(NVDevice.semaphores_page.base + signal_id * 16), value, 0x0, (3 << 0) | (1 << 12) | (1 << 24)]
    return self

  def signal(self, signal_id:int):
    self.q += [nvmethod(0, nv_gpu.NVC56F_SEM_ADDR_LO, 5), *nvdata64_le(NVDevice.semaphores_page.base + signal_id * 16), 0x1, 0x0, (6 << 0) | (1 << 20) | (1 << 24) | (5 << 27)]
    self.q += [nvmethod(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 1), 0x0]
    return self

  def submit(self, dev:NVDevice):
    for i,packet in enumerate(self.q): dev.cmdq[dev.cmdq_wptr//4 + i] = packet
    fifo_entry = dev.dma_put_value % dev.dma_gpfifo_entries
    dev.dma_gpu_ring[fifo_entry] = ((dev.cmdq_page.base+dev.cmdq_wptr)//4 << 2) | (len(self.q) << 42) #| (1 << 41) # | (1 << 63)
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

    _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
    sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]
    shstrtab = memoryview(bytearray(self.lib[sections[_shstrndx][4]:sections[_shstrndx][4]+sections[_shstrndx][5]]))

    self.shmem_usage = 0
    self.registers_usage = 0
    self.constant_buffers = {}
    constant_buffers_data = {}
    for sh_name, sh_type, sh_flags, _, sh_offset, sh_size, _, sh_info, _ in sections:
      if sh_type == SHT_NOBITS and sh_flags & SHF_ALLOC: self.shmem_usage = sh_size
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC and sh_flags & SHF_EXECINSTR:
        self.program = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")
        self.registers_usage = sh_info >> 24
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC and not(sh_flags & SHF_EXECINSTR):
        section_name = shstrtab[sh_name:].tobytes().split(b'\0', 1)[0].decode('utf-8')
        if match := re.match(r'\.nv\.constant(\d+)', section_name):
          constant_buffers_data[int(match.group(1))] = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size])).cast("I")

    # constant buffer 0 is filled for each program, no need to copy it from elf (it's just zeroes)
    if 0 in constant_buffers_data: constant_buffers_data.pop(0)

    # Load program and constant buffers (if any)
    self.lib_sz = round_up(round_up(self.program.nbytes, 128) + sum([round_up(x.nbytes, 128) for i,x in constant_buffers_data.items()]), 0x1000)
    self.lib_gpu = self.device.allocator.alloc(self.lib_sz)

    for start in range(0, len(self.program), 1000): # split, cannot send more than 1024 packets (?)
      HWComputeQueue().copy_from_cpu(self.lib_gpu.base+start*4, self.program[start:start+1000]).signal(self.device.compute_signal).submit(self.device)

    off = round_up(self.program.nbytes, 128)
    for i,data in constant_buffers_data.items():
      self.constant_buffers[i] = (self.lib_gpu.base + off, data.nbytes)
      HWComputeQueue().copy_from_cpu(self.lib_gpu.base + off, data).signal(self.device.compute_signal).submit(self.device)
      off += round_up(data.nbytes, 128)
    self.device.synchronize() # TODO: remove once well tested

    self.constbuffer_0 = [0] * 88
    self.constbuffer_0[6:12] = [*nvdata64_le(self.device.shared_mem_window), *nvdata64_le(self.device.local_mem_window), *nvdata64_le(0xfffdc0)]

    smem_config = max(8 * 1024, min(96 * 1024, round_up(self.shmem_usage, 8 * 1024))) // 4096 + 1
    self.qmd = qmd_struct_t(qmd_group_id=0x3f, sm_global_caching_enable=1, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
                            invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1,
                            cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, qmd_major_version=3,
                            shared_memory_size=max(0x400, round_up(self.shmem_usage, 0x100)), min_sm_config_shared_mem_size=0x3,
                            max_sm_config_shared_mem_size=0x1a, register_count_v=self.registers_usage, target_sm_config_shared_mem_size=smem_config,
                            barrier_count=1, shader_local_memory_high_size=0x640, program_prefetch_size=0x10, sass_version=0x89,
                            program_address_lower=self.lib_gpu.base&0xffffffff, program_address_upper=self.lib_gpu.base>>32,
                            program_prefetch_addr_lower_shifted=self.lib_gpu.base>>8, program_prefetch_addr_upper_shifted=self.lib_gpu.base>>40,
                            constant_buffer_size_shifted4_0=0x190, constant_buffer_valid_0=1, constant_buffer_invalidate_0=1)

    for i,(addr,sz) in self.constant_buffers.items():
      self.qmd.__setattr__(f'constant_buffer_addr_upper_{i}', addr >> 32)
      self.qmd.__setattr__(f'constant_buffer_addr_lower_{i}', addr & 0xffffffff)
      self.qmd.__setattr__(f'constant_buffer_size_shifted4_{i}', sz)
      self.qmd.__setattr__(f'constant_buffer_valid_{i}', 1)

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device.allocator.free(self.lib_gpu, self.lib_sz)

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    kernargs_size = round_up(0x160 + len(args) * 8 + len(vals) * 4, 128)
    if self.device.kernargs_ptr >= (self.device.kernargs_page.base + self.device.kernargs_page.length - kernargs_size):
      self.device.kernargs_ptr = self.device.kernargs_page.base
    if self.device.qmd_ptr >= (self.device.qmd_page.base + self.device.qmd_page.length):
      self.device.qmd_ptr = self.device.qmd_page.base

    qmd_ptr = self.device.qmd_ptr
    self.device.qmd_ptr += 8 << 8

    kernargs_ptr = self.device.kernargs_ptr
    self.device.kernargs_ptr += kernargs_size

    kernargs = []
    for i in range(len(args)): kernargs += [*nvdata64_le(args[i].base)]
    for i in range(len(vals)): kernargs += [vals[i]]

    queue = HWComputeQueue()
    queue.wait(self.device.dma_signal, self.device.dma_put_value)
    queue.wait(self.device.compute_signal, self.device.compute_put_value).copy_from_cpu(kernargs_ptr, self.constbuffer_0 + kernargs)
    queue.exec(self, qmd_ptr, kernargs_ptr, global_size, local_size, completion_signal=self.device.compute_signal).submit(self.device)

class NVAllocator(LRUAllocator):
  def __init__(self, device:NVDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    if options.host: return self.device._gpu_host_alloc(size)
    else: return self.device._gpu_alloc2(size, map_to_all_gpus=True)

  def _free(self, gpumem, options:BufferOptions):
    if options.host: pass # TODO
    else: self.device._gpu_free(gpumem)

  def copyin(self, dest, src: memoryview):
    host_mem = self.alloc(src.nbytes, BufferOptions(host=True))
    self.device.pending_copyin.append((host_mem, src.nbytes, BufferOptions(host=True)))
    ctypes.memmove(host_mem, from_mv(src), src.nbytes)
    HWCopyQueue().copy(dest.base, host_mem, src.nbytes).signal(self.device.dma_signal).submit(self.device)
    self.device.synchronize()

  def copyout(self, dest:memoryview, src):
    self.device.synchronize()
    host_mem = self.alloc(dest.nbytes, BufferOptions(host=True))
    self.device.pending_copyin.append((host_mem, dest.nbytes, BufferOptions(host=True)))
    HWCopyQueue().copy(host_mem, src.base, dest.nbytes).signal(self.device.dma_signal).submit(self.device)
    self.device.synchronize()
    ctypes.memmove(from_mv(dest), host_mem, dest.nbytes)

  def transfer(self, dest, src, sz:int, src_dev=None, dest_dev=None):
    queue = HWCopyQueue()
    queue.wait(src_dev.dma_signal, src_dev.dma_put_value)
    queue.wait(src_dev.compute_signal, src_dev.compute_put_value)
    queue.copy(dest.base, src.base, sz).signal(src_dev.dma_signal).submit(src_dev)
    HWCopyQueue().wait(src_dev.dma_signal, src_dev.dma_put_value).signal(dest_dev.dma_signal).submit(dest_dev)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVDevice(Compiled):
  root = None
  fd_ctl:int = -1
  fd_uvm:int = -1
  fd_uvm_2:int = -1
  semaphores_page = None
  semaphores = None
  signal_number = -1
  devices = []

  def _new_gpu_fd(self):
    fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.device, hMemory=memory_handle, length=size, flags=flags))
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")
    return libc.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev, 0)

  def _gpu_alloc2(self, size:int, contig=False, huge_page=False, va_addr=None, map_to_cpu=False, map_to_all_gpus=False, map_flags=0):
    # TODO: need hugepage option, any speedup?
    size = round_up(size, alignment:=((4 << 10) if huge_page else (2 << 20)))
    attr = (((nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE << 23) if huge_page else 0) |
            ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contig else nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27))
    attr2 = ((nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES << 2) |
             ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20) if huge_page else 0))
    flags = (nv_gpu.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM | nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT |
             nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED)

    alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, flags=flags, attr=attr, attr2=attr2, format=6, size=size,
                                                     alignment=alignment, offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nv_gpu.NV1_MEMORY_USER, self.root, self.device, alloc_params).hObjectNew

    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size, cpu_mapping=map_to_cpu)
    if map_to_cpu: va_base = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags)
    else: va_base = va_addr

    handle = self._gpu_uvm_map2(va_base, size, mem_handle)
    if map_to_all_gpus:
      for dev in NVDevice.devices:
        if dev != self: dev._gpu_uvm_map2(handle.base, handle.length, handle.hMemory, create_range=False)
    return handle

  def _gpu_system_alloc(self, size:int, va_addr=None, map_to_cpu=False, map_flags=0):
    alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, type=13,
      flags=nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED,
      attr=(nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27) | (nv_gpu.NVOS32_ATTR_LOCATION_PCI << 25),
      attr2=(nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO << 2),
      format=6, size=size, alignment=(4<<10), offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nv_gpu.NV1_MEMORY_SYSTEM, self.root, self.device, alloc_params).hObjectNew
    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size, cpu_mapping=map_to_cpu)
    if map_to_cpu: va_base = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=True)
    else: va_base = va_addr

    return self._gpu_uvm_map2(va_base, size, mem_handle)

  def _gpu_host_alloc(self, size):
    size = round_up(size, 4 << 10)
    va_base = libc.mmap(self._alloc_gpu_vaddr(size, cpu_mapping=True), size, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0) # gpu address
    return self._gpu_map_to_gpu(va_base, size)

  def _gpu_map_to_gpu(self, va_base, size):
    fd_dev = self._new_gpu_fd()
    self.host_mem_object_enumerator += 1
    flags = ((nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) |
             (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30))
    made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.device,
      hObjectNew=self.host_mem_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, flags=flags, pMemory=va_base, limit=size-1), fd=-1)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_host_alloc returned {made.params.status}")
    return self._gpu_uvm_map2(va_base, size, made.params.hObjectNew).base

  def _gpu_uvm_map2(self, va_base, size, mem_handle, create_range=True):
    if create_range: uvm.create_external_range(self.fd_uvm, base=va_base, length=size)
    gpu_attrs = (nv_gpu.struct_c__SA_UvmGpuMappingAttributes*256)(
      nv_gpu.struct_c__SA_UvmGpuMappingAttributes(gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), gpuMappingType = 1))
    return uvm.map_external_allocation(self.fd_uvm, base=va_base, length=size, rmCtrlFd=self.fd_ctl, hClient=self.root, hMemory=mem_handle,
                                       gpuAttributesCount=1, perGpuAttributes=gpu_attrs)

  def _gpu_free(self, mem):
    made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.device, hObjectOld=mem.hMemory)
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
    if made.status != 0: raise RuntimeError(f"_gpu_host_alloc returned {made.status}")
    uvm.free(self.fd_uvm, base=mem.base, length=mem.length)

  def _alloc_gpu_vaddr(self, size, cpu_mapping=False):
    va_addr = self.next_mmaped_gpu_vaddr if cpu_mapping else self.next_gpu_vaddr
    if cpu_mapping: self.next_mmaped_gpu_vaddr += round_up(size, 2 << 20)
    else: self.next_gpu_vaddr += round_up(size, 2 << 20)
    return va_addr

  def __init__(self, device:str=""):
    if NVDevice.root is None:
      NVDevice.fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.fd_uvm_2 = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVDevice.root = rm_alloc(self.fd_ctl, nv_gpu.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew
      uvm.initialize(self.fd_uvm)
      uvm.mm_initialize(self.fd_uvm_2, uvmFd=self.fd_uvm)

      NVDevice.gpus_info = (nv_gpu.nv_ioctl_card_info_t*16)()
      nv_iowr(NVDevice.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, NVDevice.gpus_info)

    # TODO: Get classes from NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    self.host_mem_object_enumerator = 0x1000 + 0x400 * self.device_id
    self.next_gpu_vaddr = 0x8000000000 + 0x1000000000 * self.device_id
    self.next_mmaped_gpu_vaddr = 0x1000000000 + 0x1000000000 * self.device_id

    assert NVDevice.gpus_info[self.device_id].valid
    gpu_info = nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS(gpuId=NVDevice.gpus_info[self.device_id].gpu_id)
    rm_control(self.fd_ctl, nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, self.root, self.root, gpu_info)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=gpu_info.deviceInstance, hClientShare=self.root, vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.device = rm_alloc(self.fd_ctl, nv_gpu.NV01_DEVICE_0, self.root, self.root, device_params).hObjectNew
    self.subdevice = rm_alloc(self.fd_ctl, nv_gpu.NV20_SUBDEVICE_0, self.root, self.device, None).hObjectNew
    self.usermode = rm_alloc(self.fd_ctl, nv_gpu.TURING_USERMODE_A, self.root, self.subdevice, None).hObjectNew
    gpu_mmio_ptr = self._gpu_map_to_cpu(self.usermode, 0x10000, flags=2)
    self.gpu_mmio = to_mv(gpu_mmio_ptr, 0x10000).cast("I")

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = rm_alloc(self.fd_ctl, nv_gpu.FERMI_VASPACE_A, self.root, self.device, vaspace_params).hObjectNew

    gpu_uuid_params = nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
    rm_control(self.fd_ctl, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, self.root, self.subdevice, gpu_uuid_params)
    self.gpu_uuid = (ctypes.c_ubyte*16)()
    for i in range(16): self.gpu_uuid[i] = gpu_uuid_params.data[i]

    uvm.register_gpu(self.fd_uvm, rmCtrlFd=-1, gpu_uuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid))
    uvm.register_gpu_vaspace(self.fd_uvm, gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl,
                             hClient=self.root, hVaSpace=vaspace)

    for dev in self.devices:
      uvm.enable_peer_access(self.fd_uvm, gpuUuidA=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), gpuUuidB=nv_gpu.struct_nv_uuid(uuid=dev.gpu_uuid))

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = rm_alloc(self.fd_ctl, nv_gpu.KEPLER_CHANNEL_GROUP_A, self.root, self.device, channel_params).hObjectNew

    gpfifo = self._gpu_alloc2(0x200000, contig=True, huge_page=True, map_to_cpu=True, map_flags=0x10d0000)

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = rm_alloc(self.fd_ctl, nv_gpu.FERMI_CONTEXT_SHARE_A, self.root, channel_group, ctxshare_params).hObjectNew

    self.compute_gpfifo_entries = 0x400
    self.compute_gpfifo_token = self._gpu_fifo_setup(gpfifo, ctxshare, channel_group, offset=0, entries=self.compute_gpfifo_entries)
    self.compute_gpu_ring = to_mv(gpfifo.base, self.compute_gpfifo_entries * 8).cast("Q")
    self.compute_gpu_ring_controls = nv_gpu.AmpereAControlGPFifo.from_address(gpfifo.base + self.compute_gpfifo_entries * 8)
    self.compute_put_value = 0
    self.compute_signal = NVDevice._get_signal()

    self.dma_gpfifo_entries = 0x400
    self.dma_gpfifo_token = self._gpu_fifo_setup(gpfifo, ctxshare, channel_group, offset=0x100000, entries=self.dma_gpfifo_entries)
    self.dma_gpu_ring = to_mv(gpfifo.base + 0x100000, self.dma_gpfifo_entries * 8).cast("Q")
    self.dma_gpu_ring_controls = nv_gpu.AmpereAControlGPFifo.from_address(gpfifo.base + 0x100000 + self.dma_gpfifo_entries * 8)
    self.dma_put_value = 0
    self.dma_signal = NVDevice._get_signal()

    en_fifo_params = nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
    rm_control(self.fd_ctl, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, self.root, channel_group, en_fifo_params)

    self.cmdq_page = self._gpu_alloc2(0x200000, map_to_cpu=True, huge_page=True)
    self.cmdq = to_mv(self.cmdq_page.base, 0x200000).cast("I")
    self.cmdq_wptr = 0 # in bytes

    if NVDevice.semaphores_page is None:
      NVDevice.semaphores_page = self._gpu_system_alloc(0x10000, map_to_cpu=True)
      NVDevice.semaphores = to_mv(self.semaphores_page.base, 0x10000).cast("Q")
    else:
      self._gpu_uvm_map2(NVDevice.semaphores_page.base, NVDevice.semaphores_page.length, NVDevice.semaphores_page.hMemory, create_range=False)

    self.kernargs_page = self._gpu_alloc2(0x1000000)
    self.kernargs_ptr = self.kernargs_page.base

    self.qmd_page = self._gpu_alloc2(0x1000000)
    self.qmd_ptr = self.qmd_page.base

    self.arch = 'sm_89' # TODO: fix
    self.pending_copyin = []

    super().__init__(device, NVAllocator(self), NVCompiler(self.arch), functools.partial(NVProgram, self))

    self._cmdq_setup_compute_gpfifo()
    self._cmdq_setup_dma_gpfifo()

    NVDevice.devices.append(self)

  def synchronize(self):
    self._wait_signal(self.compute_signal, self.compute_put_value)
    self._wait_signal(self.dma_signal, self.dma_put_value)
    self.cmdq_wptr = 0

    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @classmethod
  def _get_signal(self) -> int:
    self.signal_number += 1
    if self.semaphores_page and self.signal_number * 16 >= self.semaphores_page.length:
      for i in range(16, NVDevice.signal_number): self.semaphores[i * 2] = 0
      self.signal_number = 16
    return self.signal_number

  @classmethod
  def _wait_signal(self, id, value=0):
    sem_value = self.semaphores[id * 2]
    while sem_value != value: sem_value = self.semaphores[id * 2]

  def _gpu_fifo_setup(self, gpfifo, ctxshare, channel_group, offset, entries=0x400):
    notifier = self._gpu_system_alloc(48<<20)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=notifier.hMemory, hObjectBuffer=gpfifo.hMemory, gpFifoOffset=gpfifo.base+offset,
      gpFifoEntries=entries, hContextShare=ctxshare, hUserdMemory=(ctypes.c_uint32*8)(gpfifo.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset))
    gpfifo = rm_alloc(self.fd_ctl, nv_gpu.AMPERE_CHANNEL_GPFIFO_A, self.root, channel_group, params).hObjectNew
    compute = rm_alloc(self.fd_ctl, nv_gpu.ADA_COMPUTE_A, self.root, gpfifo, None).hObjectNew
    dma = rm_alloc(self.fd_ctl, nv_gpu.AMPERE_DMA_COPY_B, self.root, gpfifo, None).hObjectNew

    ws_token_params = nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    rm_control(self.fd_ctl, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, self.root, gpfifo, ws_token_params)
    assert ws_token_params.workSubmitToken != -1

    channel_base = self._alloc_gpu_vaddr(0x4000000)
    uvm.register_channel(self.fd_uvm, gpuUuid=nv_gpu.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl, hClient=self.root,
                         hChannel=gpfifo, base=channel_base, length=0x4000000)

    return ws_token_params.workSubmitToken

  def _cmdq_setup_compute_gpfifo(self):
    self.local_mem_window = self._gpu_alloc2(256 << 20, huge_page=True, contig=True).base
    self.shader_local_mem = self._gpu_alloc2(256 << 20, huge_page=True, contig=True).base
    self.shared_mem_window = self._gpu_alloc2(256 << 20, huge_page=True, contig=True).base

    queue = HWComputeQueue()
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_OBJECT, 1), nv_gpu.ADA_COMPUTE_A]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 2), *nvdata64(self.shader_local_mem)]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3), *nvdata64(0x4b0000), 0x40]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2), *nvdata64(self.local_mem_window)]
    queue.q += [nvmethod(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2), *nvdata64(self.shared_mem_window)]
    queue.signal(self.compute_signal).submit(self)
    self.synchronize()

  def _cmdq_setup_dma_gpfifo(self):
    queue = HWCopyQueue()
    queue.q += [nvmethod(4, nv_gpu.NVC6C0_SET_OBJECT, 1), nv_gpu.AMPERE_DMA_COPY_B]
    queue.signal(self.dma_signal).submit(self)
    self.synchronize()
