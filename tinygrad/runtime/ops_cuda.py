from __future__ import annotations
import ctypes, ctypes.util, functools
from typing import Tuple, Optional, List
from tinygrad.helpers import DEBUG, getenv, from_mv, init_c_var, init_c_struct_t
from tinygrad.device import Compiled, BufferOptions, LRUAllocator
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.autogen import cuda
import os
from tinygrad.runtime.support.compiler_cuda import cuda_disassemble, pretty_ptx, CUDACompiler, PTXCompiler, PTX
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import

def check(status):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()}")  # noqa: E501

def encode_args(args, vals) -> Tuple[ctypes.Structure, ctypes.Array]:
  c_args = init_c_struct_t(tuple([(f'f{i}', cuda.CUdeviceptr_v2) for i in range(len(args))] +
                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(c_args), ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(0))
  return c_args, vargs

def cu_time_execution(cb, enable=False) -> Optional[float]:
  if not enable: return cb()
  evs = [init_c_var(cuda.CUevent(), lambda x: cuda.cuEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
  cuda.cuEventRecord(evs[0], None)
  cb()
  cuda.cuEventRecord(evs[1], None)
  check(cuda.cuEventSynchronize(evs[1]))
  cuda.cuEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: cuda.cuEventDestroy_v2(ev)
  return ret.value * 1e-3

class CUDAProgram:
  def __init__(self, device:CUDADevice, name:str, lib:bytes, smem:int=0):
    self.device, self.name, self.lib, self.smem = device, name, lib, smem
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))
    if DEBUG >= 6: cuda_disassemble(lib, device.arch)

    check(cuda.cuCtxSetCurrent(self.device.context))
    self.module = cuda.CUmodule()
    status = cuda.cuModuleLoadData(ctypes.byref(self.module), lib)
    if status != 0:
      del self.module
      cuda_disassemble(lib, device.arch)
      raise RuntimeError(f"module load failed with status code {status}: {cuda.cudaError_enum__enumvalues[status]}")
    check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
    self.prg = prg #type: ignore
    if self.smem > 0: check(cuda.cuFuncSetAttribute(self.prg, cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))

  def __del__(self):
    if hasattr(self, 'module'): check(cuda.cuModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    check(cuda.cuCtxSetCurrent(self.device.context))
    if not hasattr(self, "vargs"):
      self.c_args, self.vargs = encode_args(args, vals) #type: ignore
    else:
      for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    return cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, self.smem, None, None, self.vargs)), enable=wait)

def size_unit(size: str):
  for unit in ['bytes', 'KB', 'MB', 'GB']:
    if size < 1000 or unit == 'GB': break
    size /= 1000
  return float(size), unit

class CUDAAllocator(LRUAllocator):
  def __init__(self, device:CUDADevice, device_id: str):
    self.device = device
    self.name = device_id
    self.mem = 0
    self.mem_high = 0
    super().__init__()
  def reset_mem_high(self):
    self.mem_high = 0
    print(f"{self.name:8} Mem high reset")
  def mem_changed(self, mem):
    self.mem += mem
    self.mem_high = max(self.mem, self.mem_high)
    reset_color = "\u001b[39m"
    magenta = "\u001b[35m"
    blue = "\u001b[34m"
    cyan = "\u001b[36m"
    white = "\u001b[37m"
    color = magenta if self.name == "CUDA" else blue if self.name == "CUDA:1" else cyan if self.name == "CUDA:2" else white
    current_mem, mem_unit = size_unit(self.mem)
    mem_high, mem_high_unit = size_unit(self.mem_high)
    changed, changed_unit = size_unit(mem)
    if os.environ.get("DEBUG_MEM"):
      print(f"\n{color}{self.name:8} ALLOC {changed:10.2f} {changed_unit} current mem: {current_mem:.2f} {mem_unit}, highest: {mem_high:.2f} {mem_high_unit} {reset_color}")

  def _alloc(self, size, options:BufferOptions):
    self.mem_changed(size)
    check(cuda.cuCtxSetCurrent(self.device.context))
    if options.host: return init_c_var(ctypes.c_void_p(), lambda x: check(cuda.cuMemHostAlloc(ctypes.byref(x), size, 0x01)))
    return init_c_var(cuda.CUdeviceptr(), lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size)))
  def _free(self, opaque, options:BufferOptions, size: int):
    self.mem_changed(-1 * size)
    if options.host: check(cuda.cuMemFreeHost(opaque))
    else: check(cuda.cuMemFree_v2(opaque))
  def copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.device.context))
    host_mem = self.alloc(len(src), BufferOptions(host=True))
    self.device.pending_copyin.append((host_mem, len(src), BufferOptions(host=True)))
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(cuda.cuMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))
  def copyout(self, dest:memoryview, src):
    CUDADevice.synchronize_system()
    check(cuda.cuCtxSetCurrent(self.device.context))
    check(cuda.cuMemcpyDtoH_v2(from_mv(dest), src, len(dest)))
  def transfer(self, dest, src, sz:int, src_dev, dest_dev):
    check(cuda.cuCtxSetCurrent(src_dev.context))
    check(cuda.cuEventCreate(ctypes.byref(sync_event := cuda.CUevent()), 0))
    check(cuda.cuMemcpyDtoDAsync_v2(dest, src, sz, None))
    check(cuda.cuEventRecord(sync_event, None))
    check(cuda.cuCtxSetCurrent(dest_dev.context))
    check(cuda.cuStreamWaitEvent(None, sync_event, 0)) # sync the default stream on the dest dev
  def offset(self, buf, size:int, offset:int): return cuda.CUdeviceptr_v2(buf.value + offset)

class CUDADevice(Compiled):
  devices: List[CUDADevice] = []
  peer_access = False

  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    self.device_id = device
    check(cuda.cuInit(0))
    self.cu_device = init_c_var(cuda.CUdevice(), lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(cuda.CUcontext(), lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, self.cu_device)))
    check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), device_id))

    for dev in CUDADevice.devices:
      check(cuda.cuDeviceCanAccessPeer(ctypes.byref(val := ctypes.c_int()), self.cu_device, dev.cu_device))
      if val.value != 1: continue
      check(cuda.cuCtxSetCurrent(dev.context))
      check(cuda.cuCtxEnablePeerAccess(self.context, 0))
      check(cuda.cuCtxSetCurrent(self.context))
      check(cuda.cuCtxEnablePeerAccess(dev.context, 0))
      CUDADevice.peer_access = True

    self.arch = f"sm_{major.value}{minor.value}"
    self.pending_copyin: List[Tuple[int, int, Optional[BufferOptions]]] = []
    CUDADevice.devices.append(self)

    from tinygrad.runtime.graph.cuda import CUDAGraph
    super().__init__(device, CUDAAllocator(self, self.device_id), PTXRenderer(self.arch) if PTX else CUDARenderer(self.arch),
                     PTXCompiler(self.arch) if PTX else CUDACompiler(self.arch), functools.partial(CUDAProgram, self), graph=CUDAGraph)

  def synchronize(self):
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in CUDADevice.devices: d.synchronize()
