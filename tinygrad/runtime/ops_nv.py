from __future__ import annotations
import os, ctypes, pathlib, re, fcntl, functools, mmap, time, tempfile, hashlib, subprocess, collections
from typing import Tuple, Any
from pathlib import Path
import os, fcntl, ctypes, functools, re, pathlib, mmap, struct, errno
from tinygrad.device import Compiled, LRUAllocator, Compiler, BufferOptions, CompilerOptions
from tinygrad.helpers import getenv, from_mv, init_c_struct_t, to_mv, round_up, to_char_p_p, DEBUG
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.helpers import to_mv, getenv, round_up
from extra.nv_gpu_driver import esc_ioctl as nvesc
from extra.nv_gpu_driver import class_ioctl as nvcls
from extra.nv_gpu_driver import ctrl_ioctl as nvctrl
from extra.nv_gpu_driver import uvm_ioctl as nvuvm
from extra.nv_gpu_driver import nv_qcmds as nvqcmd
from hexdump import hexdump
from tinygrad.runtime.ops_cuda import check as cuda_check, _get_bytes
import tinygrad.runtime.autogen.cuda as cuda
if getenv("IOCTL"):
  from extra.nv_gpu_driver import nv_ioctl
  from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo
else: _dump_gpfifo = lambda x: None

USE_DMA_FIFO = True

libc = ctypes.CDLL("libc.so.6")
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_char, ctypes.c_int]
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
libc.memcmp.restype = ctypes.c_int

def _IOWR(type, nr, size):
  return (3 << 30) | (size & 0x1FFF) << 16 | (type & 0xFF) << 8 | (nr & 0xFF)

def rm_alloc(fd, clss, root, parant, params):
  made = nvesc.NVOS21_PARAMETERS(hRoot=root, hObjectParent=parant, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_ALLOC, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def rm_control(fd, cmd, client, obj, params):
  made = nvesc.NVOS54_PARAMETERS(hClient=client, hObject=obj, cmd=cmd, params=ctypes.cast(ctypes.byref(params) if params else None, ctypes.POINTER(None)), paramsSize=ctypes.sizeof(params))
  ret = fcntl.ioctl(fd, _IOWR(ord('F'), nvesc.NV_ESC_RM_CONTROL, ctypes.sizeof(made)), made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  if made.status != 0: raise RuntimeError(f"rm_alloc returned {made.status}")
  return made

def uvm_ioctl(fd, cmd, params):
  ret = fcntl.ioctl(fd, cmd, params)
  if ret != 0: raise RuntimeError(f"ioctl (uvm_control) returned {ret}")
  if params.rmStatus != 0: raise RuntimeError(f"ioctl (uvm_control) returned {params.rmStatus}")

def set_bits_in_array(arr, end_bit, start_bit, value):
  for bt in range(start_bit, end_bit+1): arr[bt // 32] |= ((value >> (bt - start_bit)) & 0x1) << (bt % 32)

def cmd_compute(program_address, constant_address, constant_len, register_usage, shared_size, global_size, local_size):
  arr = (ctypes.c_uint32 * 0x40)()

  smem_config = max(8 * 1024, min(96 * 1024, round_up(shared_size, 8 * 1024))) // 4096 + 1

  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_QMD_GROUP_ID, 0x3F)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE, 1)

  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE, nvqcmd.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_API_VISIBLE_CALL_LIMIT, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SAMPLER_INDEX, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE, max(0x400, round_up(shared_size, 0x100)))
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE, 0x3)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1a)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_QMD_MAJOR_VERSION, 3)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_REGISTER_COUNT_V, register_usage)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE, smem_config)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_BARRIER_COUNT, 1)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE, 0x640)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE, 0x10)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_SASS_VERSION, 0x89)

  # print(smem_config, round_up(shared_size, 0x100), register_usage)

  # group
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, global_size[0])
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, global_size[1])
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, global_size[2])
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0, local_size[0])
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION1, local_size[1])
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION2, local_size[2])

  # program
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER, program_address)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER, program_address>>32)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, program_address>>8)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, program_address>>40)

  # args
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0), constant_address>>32)
  # set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0), constant_address)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0), constant_len)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(0), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE_TRUE)
  set_bits_in_array(arr, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID(0), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE)
  return arr

def cmdq_push_data(dev, data):
  dev.cmdq[dev.cmdq_wptr//4] = data
  dev.cmdq_wptr += 4
def cmdq_blit_data(dev, data, size):
  ctypes.memmove(dev.cmdq_page.base + dev.cmdq_wptr, data, size)
  dev.cmdq_wptr += size
def cmdq_push_data64(dev, data):
  cmdq_push_data(dev, data >> 32)
  cmdq_push_data(dev, data & 0xFFFFFFFF)
def cmdq_push_data64_le(dev, data):
  cmdq_push_data(dev, data & 0xFFFFFFFF)
  cmdq_push_data(dev, data >> 32)
def cmdq_push_method(dev, subc, mthd, size, typ=2): cmdq_push_data(dev, (typ << 28) | (size << 16) | (subc << 13) | (mthd >> 2)) 

class NVCompiler(Compiler):
  compiler_opts = CompilerOptions("NV", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    # NVCompiler.compiler_opts = NVCompiler.compiler_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
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

SHT_PROGBITS, SHT_NOBITS, SHF_ALLOC, SHF_EXECINSTR = 0x1, 0x8, 0x2, 0x4
class NVProgram:
  def __init__(self, device:NVDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
        with open(fn + ".cubin", "wb") as f: f.write(lib)
        print(subprocess.check_output(["nvdisasm", fn+".cubin"]).decode('utf-8'))
      except Exception as e: print("failed to disasm cubin", str(e))

    _phoff, _shoff, _flags, _ehsize, _phentsize, _phnum, _shentsize, _shnum, _shstrndx = struct.unpack_from("<QQIHHHHHH", self.lib, 0x20)
    sections = [struct.unpack_from("<IIQQQQIIQ", self.lib, _shoff + i * _shentsize) for i in range(_shnum)]

    lib_gpu_size = round_up(max(sh[5] for sh in sections if sh[1] == SHT_PROGBITS and sh[2] & SHF_EXECINSTR), 0x100)
    self.shmem_usage = 0
    self.registers_usage = 0
    self.constant_buffer = None
    self.lib_gpu = self.device.progs_ptr
    self.device.progs_ptr += lib_gpu_size
    assert self.device.progs_ptr < self.device.progs_page.base + self.device.progs_page.length

    cnt = 0
    lib_gpu_view = to_mv(self.lib_gpu, lib_gpu_size)
    for _, sh_type, sh_flags, sh_addr, sh_offset, sh_size, _, sh_info, _ in sections:
      if sh_type == SHT_NOBITS and sh_flags & SHF_ALLOC: self.shmem_usage = sh_size # sharedmem section
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC and sh_flags & SHF_EXECINSTR:
        lib_gpu_view[:sh_size] = self.lib[sh_offset:sh_offset+sh_size]
        self.registers_usage = sh_info >> 24
      if sh_type == SHT_PROGBITS and sh_flags & SHF_ALLOC and not(sh_flags & SHF_EXECINSTR):
        if sh_size < 360: self.constant_buffer = memoryview(bytearray(self.lib[sh_offset:sh_offset+sh_size]))
        cnt += 1
        # print(sh_info)
        # print(sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_info)
    # assert cnt <= 2

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))

    # ctypes.memset(self.device.kernargs_ptr, 0x0, 0x160)
    # args_st = self.args_struct_t.from_address(self.device.kernargs_ptr + 0x160)
    # for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i])
    # for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])

    # TODO: precalculate this
    qmd = cmd_compute(self.lib_gpu, 0x0, 0x190, self.registers_usage, self.shmem_usage, (1,1,1), (1,1,1))
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0), self.device.kernargs_ptr>>32)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0), self.device.kernargs_ptr&0xffffffff)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, global_size[0])
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, global_size[1])
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, global_size[2])
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0, local_size[0])
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION1, local_size[1])
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION2, local_size[2])
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_ADDRESS_LOWER, self.device.semaphores_page.base&0xffffffff)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_ADDRESS_UPPER, self.device.semaphores_page.base>>32)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_ENABLE, 0x1)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_REDUCTION_OP, 0x0)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_REDUCTION_ENABLE, 0x1)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER, 0x1)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE, 0x1)
    set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_RELEASE0_MEMBAR_TYPE, 0x1)

    # set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(1), self.device.kernargs_ptr>>32)
    # set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(1), self.device.kernargs_ptr&0xffffffff)
    # set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(1), 0x90)
    # set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID(1), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE)

    if self.constant_buffer is not None:
      ctypes.memmove(self.device.constbuf_ptr, from_mv(self.constant_buffer), self.constant_buffer.nbytes)
      set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(2), self.device.constbuf_ptr>>32)
      set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(2), self.device.constbuf_ptr&0xffffffff)
      set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(2), self.constant_buffer.nbytes)
      set_bits_in_array(qmd, *nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID(2), nvqcmd.NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE)

    cmdq_start_wptr = self.device.cmdq_wptr

    # COPY PROG
    # cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_OFFSET_OUT_UPPER, 2)
    # cmdq_push_data64(self.device, self.handle)
    # cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LINE_LENGTH_IN, 2)
    # cmdq_push_data(self.device, self.gpuprog.nbytes)
    # cmdq_push_data(self.device, 0x1)
    # cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LAUNCH_DMA, 1)
    # cmdq_push_data(self.device, 0x41)
    # print("BYTES", self.gpuprog.nbytes)
    # cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LOAD_INLINE_DATA, (self.gpuprog.nbytes + 3) // 4, typ=6)
    # for w in range((self.gpuprog.nbytes + 3) // 4): cmdq_push_data(self.device, self.gpuprog[w])

    # INV CACHE
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1)
    cmdq_push_data(self.device, (1 << 12) | (1 << 4) | (1 << 0))

    self.device.inc += 1

    # CONSTBUFFER FILLUP
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_OFFSET_OUT_UPPER, 2)
    cmdq_push_data64(self.device, self.device.kernargs_ptr)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LINE_LENGTH_IN, 2)
    cmdq_push_data(self.device, 0x160)
    cmdq_push_data(self.device, 0x1)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LAUNCH_DMA, 1)
    cmdq_push_data(self.device, 0x41)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LOAD_INLINE_DATA, 0x160 // 4, typ=6)
    cmdq_push_data(self.device, global_size[0])
    cmdq_push_data(self.device, global_size[1])
    cmdq_push_data(self.device, global_size[2])
    cmdq_push_data(self.device, local_size[0])
    cmdq_push_data(self.device, local_size[1])
    cmdq_push_data(self.device, local_size[2])
    cmdq_push_data64_le(self.device, self.device.NVC6C0_SHADER_SHARED_MEMORY_WINDOW)
    cmdq_push_data64_le(self.device, self.device.NVC6C0_SHADER_LOCAL_MEMORY_WINDOW)
    cmdq_push_data64_le(self.device, 0xfffdc0)
    cmdq_push_data64_le(self.device, self.device.inc) # TODO: change this
    cmdq_push_data64_le(self.device, self.device.unknown_buffer_1) # ???
    cmdq_push_data64_le(self.device, self.device.kernargs_ptr) # constant buffer 0
    cmdq_push_data64_le(self.device, 0x0) # cuda sets constant buffer 1, something uses it? what's there?
    for i in range(7): cmdq_push_data64_le(self.device, 0x0) # don't have any other const buffers
    cmdq_push_data64_le(self.device, 0x37a2f08) # always the same, flags?
    for i in range(2): cmdq_push_data64_le(self.device, 0x0)
    cmdq_push_data64_le(self.device, 0x1) # always the same
    for i in range(12): cmdq_push_data64_le(self.device, 0x0)
    cmdq_push_data(self.device, 0x0)
    cmdq_push_data(self.device, 0x80)
    cmdq_push_data(self.device, 0x0)
    cmdq_push_data(self.device, 0x400)
    for i in range(2): cmdq_push_data(self.device, 0xf)
    cmdq_push_data(self.device, 0x120)
    for i in range(2): cmdq_push_data(self.device, 0x0)
    cmdq_push_data(self.device, 0x160 + len(args)*8 + len(vals)*4)
    for i in range(2): cmdq_push_data(self.device, 0x0)
    cmdq_push_data64_le(self.device, self.device.unknown_buffer_2) # ???
    cmdq_push_data64_le(self.device, self.lib_gpu)
    cmdq_push_data64_le(self.device, 0x400) # shared mem
    for i in range(2): cmdq_push_data64_le(self.device, 0x0)

    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1)
    cmdq_push_data(self.device, (1 << 12) | (1 << 4) | (1 << 0))

    # # self.device._cmdq_insert_progress_semaphore(subc=1)
    # packets_written = (self.device.cmdq_wptr - cmdq_start_wptr) // 4
    # # hexdump(to_mv(self.device.cmdq_page.base, packets_written * 4))
    # self.device.compute_gpu_ring[self.device.compute_gpu_ring_controls.GPPut] = (((self.device.cmdq_page.base+cmdq_start_wptr)//4) << 2) | (packets_written << 42) | (1 << 63)
    # self.device.compute_gpu_ring_controls.GPPut += 1
    # # _dump_gpfifo(f"KERNEL_EXEC {self.device.kernargs_ptr}")
    # self.device._cmdq_ring_doorbell(self.device.compute_gpfifo_token)
    # # print("P1.5-WILL")
    # self.device.synchronize() # TODO: remove
    # # print("P1.5-DONE")
    # cmdq_start_wptr = self.device.cmdq_wptr

    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_OFFSET_OUT_UPPER, 2)
    cmdq_push_data64(self.device, self.device.kernargs_ptr + 0x160)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LINE_LENGTH_IN, 2)
    cmdq_push_data(self.device, len(args)*8 + len(vals)*4)
    cmdq_push_data(self.device, 0x1)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LAUNCH_DMA, 1)
    cmdq_push_data(self.device, 0x41)
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_LOAD_INLINE_DATA, len(args)*2 + len(vals), typ=6)
    for i in range(len(args)): cmdq_push_data64_le(self.device, args[i].base)
    for i in range(len(vals)): cmdq_push_data(self.device, vals[i])

    # INV CACHE
    # cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_INVALIDATE_SHADER_CACHES, 1)
    # cmdq_push_data(self.device, (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 12))
  
    cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_SET_INLINE_QMD_ADDRESS_A, 0x42)
    # print(hex(self.device.qmd))
    cmdq_push_data64(self.device, self.device.qmd_ptr>>8)
    cmdq_blit_data(self.device, qmd, 0x40 * 4)
    # for i in range(0x40): cmdq_push_data(self.device, qmd[i])

    # self.device.qmd_ptr += 8 << 8
    # if self.device.qmd_ptr - self.device.qmd_base_addr >= 0x800000:
    #   self.device.qmd_ptr = self.device.qmd_base_addr

    # self.device._cmdq_insert_progress_semaphore(subc=1)
    # cmdq_push_method(self.device, 1, nvqcmd.NVC6C0_SET_REPORT_SEMAPHORE_A, 4)
    # cmdq_push_data64(self.device, self.device.semaphores_addr)
    # cmdq_push_data(self.device, 0x1)
    # cmdq_push_data(self.device, (1 << 3))

    packets_written = (self.device.cmdq_wptr - cmdq_start_wptr) // 4
    self.device.compute_gpu_ring[self.device.compute_put_value % self.device.compute_gpfifo_entries] = ((self.device.cmdq_page.base+cmdq_start_wptr)//4 << 2) | (packets_written << 42) | (1 << 41) | (1 << 63)
    self.device.compute_put_value += 1
    self.device.compute_gpu_ring_controls.GPPut = self.device.compute_put_value % self.device.compute_gpfifo_entries
    self.device._cmdq_ring_doorbell(self.device.compute_gpfifo_token)
    # print("DAL", self.device.compute_put_value)
    # import time
    # time.sleep(0.09)
    self.device.synchronize() # TODO: remove
    # print("EXIT")
    # import time
    # time.sleep(1)

    # print("II")
    # for a in args:
    #   hexdump(to_mv(a, 0x20))
    #   print()
    # assert to_mv(args[0], 0x20)[0x10] != 0

class NVAllocator(LRUAllocator):
  def __init__(self, device:NVDevice):
    self.device = device
    # self.trasnf_buf = self.device._gpu_host_alloc((64 << 20))
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions):
    if options.host: return self.device._gpu_host_alloc(size)
    else: return self.device._gpu_alloc2(size)

  def _free(self, gpumem, options:BufferOptions):
    if options.host: pass # TODO
    else: self.device._gpu_free(gpumem)

  def copyin(self, dest, src: memoryview):
    host_mem = self.alloc(src.nbytes, BufferOptions(host=True))
    self.device.pending_copyin.append((host_mem, src.nbytes, BufferOptions(host=True)))
    ctypes.memmove(host_mem, from_mv(src), src.nbytes)
    self.device._cmdq_dma_copy(dest.base, host_mem, src.nbytes)

  def copyout(self, dest:memoryview, src):
    host_mem = self.alloc(dest.nbytes, BufferOptions(host=True))
    self.device.pending_copyin.append((host_mem, dest.nbytes, BufferOptions(host=True)))
    self.device._cmdq_dma_copy(host_mem, src.base, dest.nbytes)
    ctypes.memmove(from_mv(dest), host_mem, dest.nbytes)

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVDevice(Compiled):
  root = None
  fd_ctl:int = -1
  fd_uvm:int = -1
  fd_uvm_2:int = -1

  def _new_gpu_fd(self):
    fd_dev0 = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    made = nvesc.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl)
    ret = fcntl.ioctl(fd_dev0, _IOWR(ord('F'), nvesc.NV_ESC_REGISTER_FD, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    return fd_dev0
  
  def _gpu_alloc(self, size:int, coherent=False, huge_page=False, contig=False, system=False):
    attr, attr2, flags, alignment = 0, nvesc.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC, 0, 4<<10

    flags |= nvesc.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | nvesc.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nvesc.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED

    if coherent:
      attr |= nvesc.NVOS32_ATTR_LOCATION_PCI << 25
      attr2 |= nvesc.NVOS32_ATTR2_GPU_CACHEABLE_NO << 2
    else:
      attr2 |= nvesc.NVOS32_ATTR2_GPU_CACHEABLE_YES << 2
      flags |= nvesc.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM

    if contig: attr |= nvesc.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS << 27
    else: attr |= nvesc.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27

    if huge_page:
      attr |= nvesc.NVOS32_ATTR_PAGE_SIZE_HUGE << 23
      attr2 |= nvesc.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20
      flags |= nvesc.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE
      alignment = 2 << 20
    else:
      attr |= nvesc.NVOS32_ATTR_PAGE_SIZE_4KB << 23

    size = round_up(size, alignment)
    alloc_params = nvesc.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, flags=flags, attr=attr, attr2=attr2, format=6, size=size, alignment=alignment, offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nvcls.NV1_MEMORY_SYSTEM if system else nvcls.NV1_MEMORY_USER, self.root, self.device, alloc_params).hObjectNew
    return mem_handle

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0):
    fd_dev0 = self._new_gpu_fd()
    made = nvesc.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev0,
      params=nvesc.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.device, hMemory=memory_handle, length=size, flags=flags))
    ret = fcntl.ioctl(self.fd_ctl, _IOWR(ord('F'), nvesc.NV_ESC_RM_MAP_MEMORY, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.params.status != 0: raise RuntimeError(f"mmap_object returned {made.params.status}")
    return libc.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), fd_dev0, 0)

  def _gpu_alloc2(self, size:int, contig=False, huge_page=False, va_addr=None, map_to_cpu=False, map_flags=0):
    # TODO: need hugepage option?
    size = round_up(size, alignment:=((4 << 10) if huge_page else (2 << 20)))
    attr = (((nvesc.NVOS32_ATTR_PAGE_SIZE_HUGE << 23) if huge_page else 0) |
            ((nvesc.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contig else nvesc.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27))
    attr2 = ((nvesc.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (nvesc.NVOS32_ATTR2_GPU_CACHEABLE_YES << 2) |
             ((nvesc.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20) if huge_page else 0))
    flags = (nvesc.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | nvesc.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM | nvesc.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT |
             nvesc.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nvesc.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED)

    alloc_params = nvesc.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, flags=flags, attr=attr, attr2=attr2, format=6, size=size,
                                                     alignment=alignment, offset=0, limit=size-1)
    mem_handle = rm_alloc(self.fd_ctl, nvcls.NV1_MEMORY_USER, self.root, self.device, alloc_params).hObjectNew

    if va_addr is None: va_addr = self._alloc_gpu_vaddr(size, cpu_mapping=map_to_cpu)
    if map_to_cpu: va_base = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags)
    else: va_base = va_addr #libc.mmap(va_addr, size, 0, 34 | MAP_FIXED, -1, 0) # gpu address TODO: remove it?

    return self._gpu_uvm_map2(va_base, size, mem_handle)

  def _gpu_uvm_map(self, mem_handle, size:int, cpu_visible=False, fixed_address=None, map_flags=0):
    assert size % (2<<20) == 0
    if cpu_visible: va_base = self._gpu_map_to_cpu(mem_handle, size, target=fixed_address, flags=map_flags)
    else:
      va_base = libc.mmap(self.next_gpu_vaddr, size, 0, 34 | MAP_FIXED, -1, 0) # gpu address
      self.next_gpu_vaddr += size
    return self._gpu_uvm_map2(va_base, size, mem_handle).base

  def _gpu_host_alloc(self, size):
    size = round_up(size, 4 << 10)
    va_base = libc.mmap(self._alloc_gpu_vaddr(size, cpu_mapping=True), size, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0) # gpu address
    return self._gpu_map_to_gpu(va_base, size)

  def _gpu_map_to_gpu(self, va_base, size):
    fd_dev0 = self._new_gpu_fd()
    self.host_mem_object_enumerator += 1
    flags = ((nvesc.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nvesc.NVOS02_FLAGS_COHERENCY_CACHED << 12) |
             (nvesc.NVOS02_FLAGS_MAPPING_NO_MAP << 30))
    made = nvesc.nv_ioctl_nvos02_parameters_with_fd(params=nvesc.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.device,
      hObjectNew=self.host_mem_object_enumerator, hClass=nvcls.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, flags=flags, pMemory=va_base, limit=size-1), fd=-1)
    ret = fcntl.ioctl(fd_dev0, _IOWR(ord('F'), nvesc.NV_ESC_RM_ALLOC_MEMORY, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.params.status != 0: raise RuntimeError(f"_gpu_host_alloc returned {made.params.status}")
    return self._gpu_uvm_map2(va_base, size, made.params.hObjectNew).base

  def _gpu_uvm_map2(self, va_base, size, mem_handle):
    creat_range_params = nvuvm.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_CREATE_EXTERNAL_RANGE[2]), creat_range_params)
    if creat_range_params.rmStatus != 0: raise RuntimeError(f"_gpu_uvm_map returned {creat_range_params.rmStatus}")

    map_ext_params = nvuvm.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size, rmCtrlFd=self.fd_ctl, hClient=self.root, hMemory=mem_handle,
                                                              gpuAttributesCount=1)
    map_ext_params.perGpuAttributes[0].gpuUuid = nvuvm.struct_nv_uuid(uuid=self.gpu_uuid)
    map_ext_params.perGpuAttributes[0].gpuMappingType = 1
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_MAP_EXTERNAL_ALLOCATION[2]), map_ext_params)
    if map_ext_params.rmStatus != 0: raise RuntimeError(f"_gpu_uvm_map returned {map_ext_params.rmStatus}")

    return map_ext_params

  def _gpu_free(self, mem):
    made = nvesc.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.device, hObjectOld=mem.hMemory)
    ret = fcntl.ioctl(self.fd_ctl, _IOWR(ord('F'), nvesc.NV_ESC_RM_FREE, ctypes.sizeof(made)), made)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    if made.status != 0: raise RuntimeError(f"_gpu_host_alloc returned {made.status}")
    self._gpu_uvm_free(mem.base, mem.length)
  
  def _gpu_uvm_free(self, va_base, size):
    free_params = nvuvm.UVM_FREE_PARAMS(base=va_base, length=size)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_FREE[2]), free_params)
    if free_params.rmStatus != 0: raise RuntimeError(f"_gpu_uvm_free returned {free_params.rmStatus}")

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
      NVDevice.root = rm_alloc(self.fd_ctl, nvesc.NV01_ROOT_CLIENT, 0, 0, None).hObjectNew
      uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_INITIALIZE), nvuvm.UVM_INITIALIZE_PARAMS())
      uvm_ioctl(self.fd_uvm_2, int(nvuvm.UVM_MM_INITIALIZE[2]), nvuvm.UVM_MM_INITIALIZE_PARAMS(uvmFd=self.fd_uvm))

    # TODO: Get classes from NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.fd_dev = os.open(f"/dev/nvidia{self.device_id}", os.O_RDWR | os.O_CLOEXEC)
    self.host_mem_object_enumerator = 0x1000 + 0x400 * self.device_id # start 

    self.next_gpu_vaddr = 0x5000000000
    self.next_mmaped_gpu_vaddr = 0x1000000000

    device_params = nvcls.NV0080_ALLOC_PARAMETERS(deviceId=self.device_id, hClientShare=self.root, vaMode=nvesc.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.device = rm_alloc(self.fd_ctl, nvcls.NV01_DEVICE_0, self.root, self.root, device_params).hObjectNew
    self.subdevice = rm_alloc(self.fd_ctl, nvcls.NV20_SUBDEVICE_0, self.root, self.device, None).hObjectNew
    self.usermode = rm_alloc(self.fd_ctl, nvcls.TURING_USERMODE_A, self.root, self.subdevice, None).hObjectNew
    gpu_mmio_ptr = self._gpu_map_to_cpu(self.usermode, 0x10000, flags=2)

    vaspace_params = nvesc.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nvesc.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nvesc.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = rm_alloc(self.fd_ctl, nvcls.FERMI_VASPACE_A, self.root, self.device, vaspace_params).hObjectNew

    gpu_uuid_params = nvctrl.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(flags=nvctrl.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16)
    rm_control(self.fd_ctl, nvctrl.NV2080_CTRL_CMD_GPU_GET_GID_INFO, self.root, self.subdevice, gpu_uuid_params)
    self.gpu_uuid = (ctypes.c_ubyte*16)()
    for i in range(16): self.gpu_uuid[i] = gpu_uuid_params.data[i]

    register_gpu = nvuvm.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=nvuvm.struct_nv_uuid(uuid=self.gpu_uuid))
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_REGISTER_GPU[2]), register_gpu)

    register_vaspace = nvuvm.UVM_REGISTER_GPU_VASPACE_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=self.gpu_uuid), 
      rmCtrlFd=self.fd_ctl, hClient=self.root, hVaSpace=vaspace)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_REGISTER_GPU_VASPACE[2]), register_vaspace)

    channel_params = nvesc.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nvcls.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = rm_alloc(self.fd_ctl, nvcls.KEPLER_CHANNEL_GROUP_A, self.root, self.device, channel_params).hObjectNew

    gpfifo = self._gpu_alloc2(0x200000, contig=True, huge_page=True, va_addr=0x200400000 + 0x10000000 * self.device_id, map_to_cpu=True, map_flags=0x10d0000)

    ctxshare_params = nvesc.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nvesc.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = rm_alloc(self.fd_ctl, nvcls.FERMI_CONTEXT_SHARE_A, self.root, channel_group, ctxshare_params).hObjectNew

    self.compute_gpfifo_entries = 0x400
    self.compute_gpfifo_token = self._gpu_fifo_setup(gpfifo, ctxshare, channel_group, offset=0, entries=self.compute_gpfifo_entries)

    self.dma_gpfifo_entries = 0x400
    self.dma_gpfifo_token = self._gpu_fifo_setup(gpfifo, ctxshare, channel_group, offset=0x100000, entries=self.dma_gpfifo_entries)

    en_fifo_params = nvctrl.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)
    rm_control(self.fd_ctl, nvctrl.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, self.root, channel_group, en_fifo_params)

    self.compute_gpu_ring = to_mv(gpfifo.base, self.compute_gpfifo_entries * 8).cast("Q")
    self.compute_gpu_ring_controls = nvcls.AmpereAControlGPFifo.from_address(gpfifo.base + self.compute_gpfifo_entries * 8)
    self.compute_put_value = 0
    self.dma_gpu_ring = to_mv(gpfifo.base + 0x100000, self.dma_gpfifo_entries * 8).cast("Q")
    self.dma_gpu_ring_controls = nvcls.AmpereAControlGPFifo.from_address(gpfifo.base + 0x100000 + self.dma_gpfifo_entries * 8)
    self.dma_put_value = 0
    self.gpu_mmio = to_mv(gpu_mmio_ptr, 0x10000).cast("I")

    self.cmdq_page = self._gpu_alloc2(0x1a00000, map_to_cpu=True, huge_page=True)
    self.cmdq = to_mv(self.cmdq_page.base, 0x1a00000).cast("I")
    self.cmdq_wptr = 0 # in bytes

    self.semaphores_page = self._gpu_alloc2(0x4000, map_to_cpu=True)
    self.semaphores = to_mv(self.semaphores_page.base, 0x4000).cast("Q")

    self.kernargs_page = self._gpu_alloc2(0x200000, map_to_cpu=True)
    self.kernargs_ptr = self.kernargs_page.base

    self.constbuf_page = self._gpu_alloc2(0x200000, map_to_cpu=True)
    self.constbuf_ptr = self.constbuf_page.base

    self.qmd_page = self._gpu_alloc2(0x800000, map_to_cpu=True)
    self.qmd_ptr = self.qmd_page.base

    self.progs_page = self._gpu_alloc2(0x6000000, map_to_cpu=True)
    self.progs_ptr = self.progs_page.base

    self.arch = 'sm_89' # TODO: fix
    self.inc = 0
    self.pending_copyin = []

    super().__init__(device, NVAllocator(self), NVCompiler(self.arch), functools.partial(NVProgram, self))

    self._cmdq_setup_compute_gpfifo()
    self._cmdq_setup_dma_gpfifo()

  def synchronize(self):
    sem_value = self.semaphores[0]
    while sem_value != self.compute_put_value: sem_value = self.semaphores[0]
    sem_value = self.semaphores[4]
    while sem_value != (self.dma_put_value if USE_DMA_FIFO else self.compute_put_value): sem_value = self.semaphores[4]

    self.cmdq_wptr = 0

    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  def _gpu_fifo_setup(self, gpfifo, ctxshare, channel_group, offset, entries=0x400):
    notifier = self._gpu_alloc(50331648, coherent=True, system=True)

    gpfifo_params = nvesc.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=notifier, hObjectBuffer=gpfifo.hMemory, gpFifoOffset=gpfifo.base+offset,
      gpFifoEntries=entries, hContextShare=ctxshare, hUserdMemory=(ctypes.c_uint32*8)(gpfifo.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset))
    gpfifo = rm_alloc(self.fd_ctl, nvcls.AMPERE_CHANNEL_GPFIFO_A, self.root, channel_group, gpfifo_params).hObjectNew
    compute = rm_alloc(self.fd_ctl, nvcls.ADA_COMPUTE_A, self.root, gpfifo, None).hObjectNew
    dma = rm_alloc(self.fd_ctl, nvcls.AMPERE_DMA_COPY_B, self.root, gpfifo, None).hObjectNew

    ws_token_params = nvctrl.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    rm_control(self.fd_ctl, nvctrl.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, self.root, gpfifo, ws_token_params)
    assert ws_token_params.workSubmitToken != -1

    register_channel_params = nvuvm.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=nvuvm.struct_nv_uuid(uuid=self.gpu_uuid), rmCtrlFd=self.fd_ctl, hClient=self.root,
      hChannel=gpfifo, base=0x203600000, length=0x4000000)
    uvm_ioctl(self.fd_uvm, int(nvuvm.UVM_REGISTER_CHANNEL[2]), register_channel_params)

    return ws_token_params.workSubmitToken

  def _cmdq_setup_compute_gpfifo(self):
    shared_mem_handle = self._gpu_alloc(0x8000000, huge_page=True, contig=True)
    start = self._gpu_uvm_map(shared_mem_handle, 0x8000000)
    local_mem_window_handle = self._gpu_alloc(0x12c00000, huge_page=True, contig=True)
    self.NVC6C0_SHADER_LOCAL_MEMORY_WINDOW = self._gpu_uvm_map(local_mem_window_handle, 0x12c00000)
    local_mem_handle = self._gpu_alloc(0x12c00000, huge_page=True, contig=True)
    self.NVC6C0_SHADER_LOCAL_MEMORY = self._gpu_uvm_map(local_mem_handle, 0x12c00000)
    # print(hex(self.NVC6C0_SHADER_LOCAL_MEMORY_WINDOW))
    # start = 0x7f7d55000000
    # self.NVC6C0_SHADER_LOCAL_MEMORY_WINDOW = start
    self.NVC6C0_SHADER_SHARED_MEMORY_WINDOW = start + 0x2000000
    self.unknown_buffer_1 = 0xffffffffffffffff
    self.unknown_buffer_2 = 0xffffffffffffffff

    cmdq_start_wptr = self.cmdq_wptr
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_OBJECT, 1)
    cmdq_push_data(self, nvcls.ADA_COMPUTE_A)
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 1)
    cmdq_push_data(self, (self.NVC6C0_SHADER_LOCAL_MEMORY >> 32))
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_B, 1)
    cmdq_push_data(self, (self.NVC6C0_SHADER_LOCAL_MEMORY & 0xffffffff))
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 1)
    cmdq_push_data(self, (self.NVC6C0_SHADER_SHARED_MEMORY_WINDOW >> 32))
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_B, 1)
    cmdq_push_data(self, (self.NVC6C0_SHADER_SHARED_MEMORY_WINDOW & 0xffffffff))
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 1)
    cmdq_push_data(self, 0x0)
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B, 1)
    cmdq_push_data(self, 0x4b0000)
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 1)
    cmdq_push_data(self, (self.NVC6C0_SHADER_LOCAL_MEMORY_WINDOW >> 32))
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B, 1)
    cmdq_push_data(self, (self.NVC6C0_SHADER_LOCAL_MEMORY_WINDOW & 0xffffffff))

    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SPA_VERSION, 1)
    cmdq_push_data(self, 0x809)
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_RESERVED_SW_METHOD07, 1)
    cmdq_push_data(self, 0x1)
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_VALID_SPAN_OVERFLOW_AREA_A, 3)
    cmdq_push_data(self, 0x2)
    cmdq_push_data(self, 0x0)
    cmdq_push_data(self, 0x300000)
    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C, 1)
    cmdq_push_data(self, 0x40)
    cmdq_push_method(self, 4, nvqcmd.NVC6C0_SET_OBJECT, 1)
    cmdq_push_data(self, nvcls.AMPERE_DMA_COPY_B)

    cmdq_push_method(self, 1, nvqcmd.NVC6C0_SET_REPORT_SEMAPHORE_A, 4)
    cmdq_push_data64(self, self.semaphores_page.base)
    cmdq_push_data(self, 0x1)
    cmdq_push_data(self, (1 << 3))
    
    packets_written = (self.cmdq_wptr - cmdq_start_wptr) // 4
    self.compute_gpu_ring[self.compute_put_value % self.compute_gpfifo_entries] = ((self.cmdq_page.base+cmdq_start_wptr)//4 << 2) | (packets_written << 42) | (1 << 63)
    self.compute_put_value += 1
    self.compute_gpu_ring_controls.GPPut = self.compute_put_value % self.compute_gpfifo_entries
    self._cmdq_ring_doorbell(self.compute_gpfifo_token)
    self.synchronize() # TODO: remove
  
  def _cmdq_setup_dma_gpfifo(self):
    if not USE_DMA_FIFO: return
    cmdq_start_wptr = self.cmdq_wptr
    cmdq_push_method(self, 4, nvqcmd.NVC6C0_SET_OBJECT, 1)
    cmdq_push_data(self, nvcls.AMPERE_DMA_COPY_B)

    cmdq_push_method(self, 4, nvqcmd.NVC6B5_SET_SEMAPHORE_A, 3)
    cmdq_push_data64(self, self.semaphores_page.base+0x20)
    cmdq_push_data(self, 1)
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_LAUNCH_DMA, 1)
    cmdq_push_data(self, 0x14) # TODO: flags...

    packets_written = (self.cmdq_wptr - cmdq_start_wptr) // 4
    self.dma_gpu_ring[self.dma_put_value % self.dma_gpfifo_entries] = ((self.cmdq_page.base+cmdq_start_wptr)//4 << 2) | (packets_written << 42) | (1 << 63)
    self.dma_put_value += 1
    self.dma_gpu_ring_controls.GPPut = self.dma_put_value % self.dma_gpfifo_entries
    self._cmdq_ring_doorbell(self.dma_gpfifo_token)
    self.synchronize() # TODO: remove
    # pass
    # TODO: really need a spe queue?
    # self.dma_gpu_ring = self.compute_gpu_ring
    # self.dma_gpu_ring_controls = self.compute_gpu_ring_controls
    # self.dma_gpfifo_token = self.compute_gpfifo_token
  
  def _cmdq_dma_copy(self, dst, src, sz):
    cmdq_start_wptr = self.cmdq_wptr

    cmdq_push_method(self, 4, nvqcmd.NVC6B5_OFFSET_IN_UPPER, 4)
    cmdq_push_data64(self, src)
    cmdq_push_data64(self, dst)
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_LINE_LENGTH_IN, 1)
    cmdq_push_data(self, sz)
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_LAUNCH_DMA, 1)
    cmdq_push_data(self, (2 << 0) | (1 << 2) | (1 << 25) | (1 << 7) | (1 << 8)) # TODO: flags...
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_SET_SEMAPHORE_A, 3)
    cmdq_push_data64(self, self.semaphores_page.base+(0x20 if USE_DMA_FIFO else 0x0))
    cmdq_push_data(self, 0x1)
    cmdq_push_method(self, 4, nvqcmd.NVC6B5_LAUNCH_DMA, 1)
    cmdq_push_data(self, (1 << 19) | (5 << 14) | 0x14) # TODO: flags...

    packets_written = (self.cmdq_wptr - cmdq_start_wptr) // 4
    if USE_DMA_FIFO:
      self.dma_gpu_ring[self.dma_put_value % self.dma_gpfifo_entries] = ((self.cmdq_page.base+cmdq_start_wptr)//4 << 2) | (packets_written << 42) | (1 << 41) | (1 << 63)
      self.dma_put_value += 1
      self.dma_gpu_ring_controls.GPPut = self.dma_put_value % self.dma_gpfifo_entries
      self._cmdq_ring_doorbell(self.dma_gpfifo_token)
    else:
      self.compute_gpu_ring[self.compute_put_value % self.compute_gpfifo_entries] = ((self.cmdq_page.base+cmdq_start_wptr)//4 << 2) | (packets_written << 42) | (1 << 41) | (1 << 63)
      self.compute_put_value += 1
      self.compute_gpu_ring_controls.GPPut = self.compute_put_value % self.compute_gpfifo_entries
      self._cmdq_ring_doorbell(self.compute_gpfifo_token)
    # print("DMA", hex(src), hex(dst))
    self.synchronize() # TODO: remove
    # print("EXIT")

  def _cmdq_ring_doorbell(self, token): self.gpu_mmio[0x90 // 4] = token # TODO: this is bad...
