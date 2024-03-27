from __future__ import annotations
import subprocess, hashlib, tempfile, ctypes, ctypes.util, functools, re, io 
from pathlib import Path
from typing import Tuple, Optional, List
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import DEBUG, getenv, from_mv, to_char_p_p, init_c_var, init_c_struct_t, colored, cpu_time_execution, round_up, to_mv
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator, Compiler, BufferOptions
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.renderer.assembly import PTXRenderer
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])', lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers  # noqa: E501
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers  # noqa: E501
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

CUDACPU = getenv("CUDACPU") == 1
if CUDACPU:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501
  cuda.cuLaunchKernel = lambda src, gx, gy, gz, lx, ly, lz, shared, stream, unused_extra, args: gpuocelot_lib.ptx_run(src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), lx, ly, lz, gx, gy, gz, shared)  # type: ignore  # noqa: E501

def check(status):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()}")  # noqa: E501

def encode_args(args, vals) -> Tuple[ctypes.Structure, ctypes.Array]:
  c_args = init_c_struct_t(tuple([(f'f{i}', cuda.CUdeviceptr_v2) for i in range(len(args))] +
                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(c_args), ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(0))
  return c_args, vargs

def cu_time_execution(cb, enable=False) -> Optional[float]:
  if CUDACPU: return cpu_time_execution(cb, enable=enable)
  if not enable: return cb()
  evs = [init_c_var(cuda.CUevent(), lambda x: cuda.cuEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
  cuda.cuEventRecord(evs[0], None)
  cb()
  cuda.cuEventRecord(evs[1], None)
  check(cuda.cuEventSynchronize(evs[1]))
  cuda.cuEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: cuda.cuEventDestroy_v2(ev)
  return ret.value * 1e-3

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

class PTXCompiler(Compiler):
  linearizer_opts = LinearizerOptions("CUDA", suffix="PTX", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    self.version = "7.8" if arch >= "sm_89" else "7.5"
    PTXCompiler.linearizer_opts = PTXCompiler.linearizer_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    super().__init__(f"compile_ptx_{self.arch}")
  def render(self, name:str, uops) -> str: return PTXRenderer(name, uops).replace("TARGET", self.arch).replace("VERSION", self.version)
  def compile(self, src:str) -> bytes: return src.encode()

class CUDACompiler(Compiler):
  linearizer_opts = LinearizerOptions("CUDA", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    CUDACompiler.linearizer_opts = CUDACompiler.linearizer_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    check(cuda.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.compile_options = [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_cuda_{self.arch}")
  def render(self, name:str, uops) -> str: return CUDARenderer(name, uops)
  def compile(self, src:str) -> bytes:
    check(cuda.nvrtcCreateProgram(ctypes.byref(prog := cuda.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = cuda.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options]))

    if status != 0: raise RuntimeError(f"compile failed: {_get_bytes(prog, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, check).decode()}")
    return _get_bytes(prog, cuda.nvrtcGetPTX, cuda.nvrtcGetPTXSize, check)

def cuda_disassemble(lib, arch):
  try:
    fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn + ".ptx", "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn+".ptx"], check=True)
    print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
  except Exception as e: print("failed to generate SASS", str(e))

class CUDAProgram:
  def __init__(self, device:CUDADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))
    if DEBUG >= 6: cuda_disassemble(lib, device.arch)

    if CUDACPU: self.prg = lib
    else:
      check(cuda.cuCtxSetCurrent(self.device.context))
      # print("module created", self.device.device_id)
      self.module = cuda.CUmodule()
      check(cuda.cuModuleLoadData(ctypes.byref(self.module), lib))
      status = 0
      if status != 0:
        del self.module
        cuda_disassemble(lib, device.arch)
        raise RuntimeError("module load failed")
      check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
      self.prg = prg #type: ignore

  def __del__(self):
    if hasattr(self, 'module'): check(cuda.cuModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if CUDACPU: self.vargs = args+tuple(vals)
    else:
      check(cuda.cuCtxSetCurrent(self.device.context))
      if not hasattr(self, "vargs"):
        self.c_args, self.vargs = encode_args(args, vals) #type: ignore
      else:
        for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
        for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    # print("call", [hex(x) for x in args])
    return cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs)), enable=wait)

# class DeviceAlloc:
#   def __init__(self):
def round_to_next_power_of_2(n):
  if n < 1: return 1
  power_of_2 = 1
  while power_of_2 < n: power_of_2 *= 2
  return power_of_2

import collections
device_allocs = collections.defaultdict(int)

class SimpliestBestFit:
  def __init__(self, device:CUDADevice):
    self.device = device
    check(cuda.cuMemAddressReserve(ctypes.byref(va := cuda.CUdeviceptr()), 24 << 30, 2 << 20, 0x7000000000, 0))
    self.alloced = 0
    self.vaddr = va.value
    self.free_segments = [(self.vaddr, self.vaddr + (24 << 30))]
    self.size_from_start = {}
    self.mapped = {}
  
  def map_vaddr(self, vaddr):
    assert (vaddr & ((2 << 20) - 1)) == 0
    if vaddr in self.mapped: return
    # print(self.alloced / 1024 / 1024 / 1024, (len(self.mapped) * 2 << 20) / 1024 / 1024 / 1024)
    check(cuda.cuCtxSetCurrent(self.device.context))
    props = cuda.CUmemAllocationProp_v1(1, cuda.CU_MEM_HANDLE_TYPE_NONE, cuda.CUmemLocation(cuda.CU_MEM_LOCATION_TYPE_DEVICE, self.device.device_id), 0, cuda.struct_CUmemAllocationProp_st_allocFlags())
    check(cuda.cuMemCreate(ctypes.byref(paddr := cuda.CUmemGenericAllocationHandle()), 2 << 20, ctypes.byref(props), 0))
    # print("map", vaddr, paddr)
    check(cuda.cuMemMap(vaddr, 2 << 20, 0, paddr, 0))
    self.mapped[vaddr] = paddr

    # accs = []
    # for i in range(6):
    #   loc = props.location
    #   loc.id = 3
    #   accs.append(cuda.CUmemAccessDesc_v1(loc, cuda.CU_MEM_ACCESS_FLAGS_PROT_READWRITE))
    # c_acc = (cuda.CUmemAccessDesc_v1*6)(*accs)
    # print(c_acc[3].location.id)
    acc = cuda.CUmemAccessDesc_v1(props.location, cuda.CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
    check(cuda.cuMemSetAccess(vaddr, 2 << 20, ctypes.byref(acc), 1))
  
  def ensure_paddr_mapped(self, vaddr, size):
    page_start_vaddr = vaddr & ~((2 << 20) - 1)
    while page_start_vaddr < vaddr + size:
      self.map_vaddr(page_start_vaddr)
      # print("call mmap on", hex(page_start_vaddr), hex(vaddr), hex(vaddr + size - 1), size)
      page_start_vaddr += 2 << 20

  def reclaim(self):
    for seg in self.free_segments:
      # print(seg)
      vaddr = seg[0]
      vaddr_end = (seg[1] - 1) & ~((2 << 20) - 1)
      page_start_vaddr = round_up(vaddr, 2 << 20)
      while page_start_vaddr < vaddr_end:
        if page_start_vaddr not in self.mapped:
          page_start_vaddr += 2 << 20
          continue
        check(cuda.cuMemUnmap(page_start_vaddr, 2 << 20))
        check(cuda.cuMemRelease(self.mapped.pop(page_start_vaddr)))
        # print("call unmmap on", hex(page_start_vaddr), hex(vaddr), hex(vaddr_end))
        page_start_vaddr += 2 << 20

  def alloc(self, request_size):
    request_size = round_up(request_size, 64)
    best_fit_index = None
    best_fit_size = None

    # Find the best fit segment for the requested size.
    for i, segment in enumerate(self.free_segments):
      szsz = segment[1] - segment[0]
      if szsz >= request_size and (best_fit_size is None or szsz < best_fit_size):
        best_fit_index = i
        best_fit_size = szsz
            
    if best_fit_index is None: raise MemoryError("no gpu mem")

    allocated_segment = self.free_segments.pop(best_fit_index)
    allocated_start = allocated_segment[0]
    allocated_end = allocated_start + request_size
    szsz = allocated_segment[1] - allocated_segment[0]
    
    if szsz > request_size: self.free_segments.append((allocated_end, allocated_segment[1]))
    align_allocated_start = round_up(allocated_start, 64)
    self.size_from_start[align_allocated_start] = (allocated_start, request_size)
    self.ensure_paddr_mapped(allocated_start, request_size)
    self.alloced += request_size
    return align_allocated_start

  def free(self, aligned_start):
    start, size = self.size_from_start[aligned_start]
    new_seg = (start, start+size)
    while True:
      kick = None
      for i,seg in enumerate(self.free_segments):
        if seg[0] == new_seg[1] or seg[1] == new_seg[0]:
          kick = i
          break
      if kick is None: break
      seg = self.free_segments.pop(kick)
      new_seg = (min(seg[0], new_seg[0]), max(seg[1], new_seg[1]))
    self.free_segments.append(new_seg)
    self.alloced -= size

CHUNK_SIZE, PAGE_SIZE = 32*1024*1024, 0x1000
class CUDAAllocator(LRUAllocator):
  def __init__(self, device:CUDADevice):
    self.device = device
    self.dev_alloctor = SimpliestBestFit(device)
    super().__init__()
  def _alloc(self, size, options:BufferOptions):
    check(cuda.cuCtxSetCurrent(self.device.context))
    if options.host: return init_c_var(ctypes.c_void_p(), lambda x: check(cuda.cuMemHostAlloc(ctypes.byref(x), size, 0)))
    else: return self.dev_alloctor.alloc(size)
  def _free(self, opaque, options:BufferOptions):
    if options.host: return check(cuda.cuMemFreeHost(opaque))
    else: self.dev_alloctor.free(opaque) #check(cuda.cuMemFree_v2(opaque))
  def copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.device.context))
    # host_mem = self.alloc(len(src), BufferOptions(host=True))
    # self.device.pending_copyin.append((host_mem, len(src), BufferOptions(host=True)))
    # ctypes.memmove(host_mem, from_mv(src), len(src))
    # print(hex(dest))
    check(cuda.cuMemcpyHtoD_v2(dest, from_mv(src), len(src), None))
  def copy_from_fd(self, dest, fd, offset, size):
    check(cuda.cuCtxSetCurrent(self.device.context))
    if not hasattr(self, 'hb'):
      self.hb = [self._alloc(CHUNK_SIZE, BufferOptions(host=True)) for _ in range(2)]
      self.hb_events = [None, None]
      self.hb_polarity = 0
    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      if self.hb_events[self.hb_polarity] is not None:
        # NOTE: block doesn't work here because we modify the CPU memory
        check(cuda.cuEventSynchronize(self.hb_events[self.hb_polarity]))
        check(cuda.cuEventDestroy_v2(self.hb_events[self.hb_polarity]))
        self.hb_events[self.hb_polarity] = None
      fo.readinto(to_mv(self.hb[self.hb_polarity], local_size))
      check(cuda.cuMemcpyHtoDAsync_v2(dest + copied_in, self.hb[self.hb_polarity].value + minor_offset,
                               copy_size:=min(local_size-minor_offset, size-copied_in), None))
      self.hb_events[self.hb_polarity] = init_c_var(cuda.CUevent(), lambda x: check(cuda.cuEventCreate(ctypes.byref(x), 0)))
      check(cuda.cuEventRecord(self.hb_events[self.hb_polarity], None))
      copied_in += copy_size
      self.hb_polarity = (self.hb_polarity+1) % len(self.hb)
      minor_offset = 0 # only on the first
  def copyout(self, dest:memoryview, src):
    CUDADevice.synchronize_system()
    # check(cuda.cuCtxSetCurrent(self.device.context))
    # check(cuda.cuMemcpyDtoH_v2(from_mv(dest), src, len(dest)))
  def transfer(self, dest, src, sz:int, src_dev, dest_dev):
    pass
    # host_mem = self.alloc(sz, BufferOptions(host=True))
    # check(cuda.cuCtxSetCurrent(src_dev.context))
    # check(cuda.cuMemcpyDtoH_v2(host_mem, src, sz))
    # check(cuda.cuCtxSetCurrent(dest_dev.context))
    # check(cuda.cuMemcpyHtoD_v2(dest, host_mem, sz))
    # check(cuda.cuCtxSetCurrent(src_dev.context))
    # check(cuda.cuEventCreate(ctypes.byref(sync_event := cuda.CUevent()), 0))
    # check(cuda.cuMemcpyDtoD_v2(dest, src, sz, None))
    # check(cuda.cuEventRecord(sync_event, None))
    # check(cuda.cuCtxSetCurrent(dest_dev.context))
    # check(cuda.cuStreamWaitEvent(None, sync_event, 0)) # sync the default stream on the dest dev

class CUDADevice(Compiled):
  devices: List[CUDADevice] = []
  ctxs: List[Any] = []

  def __init__(self, device:str):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    if not CUDACPU:
      check(cuda.cuInit(0))
      if not CUDADevice.ctxs:
        for di in range(6):
          check(cuda.cuDeviceGet(ctypes.byref(cu_device := cuda.CUdevice()), di))
          CUDADevice.ctxs.append(init_c_var(cuda.CUcontext(), lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, cu_device))))
      self.context = CUDADevice.ctxs[self.device_id]
      check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), self.device_id))

    self.arch = f"sm_{major.value}{minor.value}" if not CUDACPU else "sm_35"
    self.pending_copyin: List[Tuple[int, int, Optional[BufferOptions]]] = []
    CUDADevice.devices.append(self)

    from tinygrad.runtime.graph.cuda import CUDAGraph
    super().__init__(device, CUDAAllocator(self) if not CUDACPU else MallocAllocator,
                     PTXCompiler(self.arch) if getenv("PTX") else CUDACompiler(self.arch),
                     functools.partial(CUDAProgram, self), graph=CUDAGraph if not CUDACPU else None)

  def synchronize(self):
    if CUDACPU: return
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in CUDADevice.devices: d.synchronize()
