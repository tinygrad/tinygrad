from __future__ import annotations
import ctypes, functools
from tinygrad.helpers import DEBUG, getenv, mv_address, suppress_finalizing, CUDA_CC, CUDA_PTX
from tinygrad.device import Compiled, BufferSpec, LRUAllocator, CompilerPair, CompilerSet
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.autogen import cuda, cupti
from tinygrad.runtime.support.compiler_cuda import pretty_ptx, CUDACompiler, PTXCompiler, NVCCCompiler
from tinygrad.runtime.support.c import init_c_struct_t, init_c_var
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import

if getenv("IOCTL"):
  from extra.nv_gpu_driver.nv_ioctl import _dump_gpfifo
else:
  def _dump_gpfifo(*args, **kwargs): pass

if MOCKGPU:=getenv("MOCKGPU"): from test.mockgpu.cuda import cuda # type: ignore # pylint: disable=reimported

PROFILE = getenv("PROFILE", 0)

# PC sampling stall reason names
PC_STALL_REASONS = {
  0: "invalid", 1: "none", 2: "inst_fetch", 3: "exec_dep", 4: "mem_dep",
  5: "texture", 6: "sync", 7: "const_mem", 8: "pipe_busy", 9: "mem_throttle",
  10: "not_selected", 11: "other", 12: "sleeping"
}


# CUPTI profiling support
class CUPTIProfiler:
  def __init__(self):
    self.initialized = False
    self.pc_sampling_enabled = False
    self.buffers: list[ctypes.Array] = []
    self.kernel_stalls: dict[int, dict[int, int]] = {}  # correlationId -> {stall_reason: samples}

  def _check_cupti(self, status, soft=False):
    if status != cupti.CUPTI_SUCCESS:
      if soft: return False
      raise RuntimeError(f"CUPTI Error {status}")
    return True

  def init(self, ctx, device_id: int = 0):
    if self.initialized: return
    _dump_gpfifo("before cupti")
    # Initialize profiler API
    init_params = cupti.CUpti_Profiler_Initialize_Params()
    init_params.structSize = 16
    cupti.cuptiProfilerInitialize(ctypes.byref(init_params))

    # Register buffer callbacks for Activity API
    self._buf_req_cb = cupti.CUpti_BuffersCallbackRequestFunc(self._buffer_requested)
    self._buf_comp_cb = cupti.CUpti_BuffersCallbackCompleteFunc(self._buffer_completed)
    self._check_cupti(cupti.cuptiActivityRegisterCallbacks(self._buf_req_cb, self._buf_comp_cb))

    # PROFILE=1: kernel timing, PROFILE=2: PC sampling with stall reasons
    if PROFILE >= 2:
      # PC sampling for stall analysis (requires elevated privileges)
      if DEBUG >= 1: print("  CUPTI: PC sampling mode (before)")
      pc_status = cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING)
      if pc_status == cupti.CUPTI_SUCCESS:
        config = cupti.CUpti_ActivityPCSamplingConfig()
        config.size, config.samplingPeriod = 16, cupti.CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN
        cfg_status = cupti.dll.cuptiActivityConfigurePCSampling(ctx, ctypes.byref(config))
        if cfg_status == cupti.CUPTI_SUCCESS:
          if DEBUG >= 1: print("  CUPTI: PC sampling mode (before stall analysis)")
          cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO)
          self.pc_sampling_enabled = True
          if DEBUG >= 1: print("  CUPTI: PC sampling mode (stall analysis)")
        elif cfg_status == 35:
          if DEBUG >= 1: print("  CUPTI: PC sampling needs: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0'|sudo tee /etc/modprobe.d/nvidia.conf && sudo reboot")
      # Fall back to kernel timing if PC sampling setup failed
      if not self.pc_sampling_enabled:
        self._check_cupti(cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_KERNEL))
    else:
      # Kernel activity tracing for timing
      self._check_cupti(cupti.cuptiActivityEnable(cupti.CUPTI_ACTIVITY_KIND_KERNEL))

    _dump_gpfifo("after cupti")
    self.initialized = True

  def _buffer_requested(self, buffer, size, max_num_records):
    buf = (ctypes.c_uint8 * 1024 * 1024)()  # 1MB buffer
    self.buffers.append(buf)
    buffer[0] = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
    size[0] = ctypes.sizeof(buf)
    max_num_records[0] = 0

  def _buffer_completed(self, ctx, stream_id, buffer, size, valid_size):
    if valid_size > 0:
      record = ctypes.POINTER(cupti.CUpti_Activity)()
      while cupti.cuptiActivityGetNextRecord(buffer, valid_size, ctypes.byref(record)) == cupti.CUPTI_SUCCESS:
        kind = record.contents.kind
        if kind == cupti.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          kernel = ctypes.cast(record, ctypes.POINTER(cupti.CUpti_ActivityKernel9)).contents
          name = ctypes.string_at(kernel.name).decode() if kernel.name else "unknown"
          duration_us = (kernel.end - kernel.start) / 1000.0
          grid, block = (kernel.gridX, kernel.gridY, kernel.gridZ), (kernel.blockX, kernel.blockY, kernel.blockZ)
          print(f"  CUPTI: {name[:40]:40s} | {duration_us:10.2f} us | grid={grid} block={block} | regs={kernel.registersPerThread:3d} smem={kernel.staticSharedMemory + kernel.dynamicSharedMemory:6d}B")
        elif kind == cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING:
          pc = ctypes.cast(record, ctypes.POINTER(cupti.CUpti_ActivityPCSampling3)).contents
          cid = pc.correlationId
          if cid not in self.kernel_stalls: self.kernel_stalls[cid] = {}
          self.kernel_stalls[cid][pc.stallReason] = self.kernel_stalls[cid].get(pc.stallReason, 0) + pc.samples
        elif kind == cupti.CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
          info = ctypes.cast(record, ctypes.POINTER(cupti.CUpti_ActivityPCSamplingRecordInfo)).contents
          cid = info.correlationId
          if cid in self.kernel_stalls:
            stalls = self.kernel_stalls[cid]
            total = sum(stalls.values())
            if total > 0:
              top = sorted(stalls.items(), key=lambda x: -x[1])[:5]
              stall_str = " ".join(f"{PC_STALL_REASONS.get(r,'?')}:{100*c//total}%" for r,c in top if c > 0)
              print(f"  CUPTI stalls (corr={cid}): {total} samples | {stall_str}")
            del self.kernel_stalls[cid]

  def flush(self):
    if not self.initialized: return
    self._check_cupti(cupti.cuptiActivityFlushAll(0))

cupti_profiler = CUPTIProfiler() if PROFILE else None

def check(status):
  if status != 0:
    error = ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char), lambda x: cuda.cuGetErrorString(status, x))).decode()
    raise RuntimeError(f"CUDA Error {status}, {error}")

def encode_args(args, vals) -> tuple[ctypes.Structure, ctypes.Array]:
  c_args = init_c_struct_t(len(args) * 8 + len(vals) * 4, tuple([(f'f{i}', cuda.CUdeviceptr_v2, i*8) for i in range(len(args))] +
                                                                [(f'v{i}', ctypes.c_int, len(args)*8 + i*4) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(c_args), ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(0))
  return c_args, vargs

def cu_time_execution(cb, enable=False) -> float|None:
  if not enable: return cb()
  evs = [init_c_var(cuda.CUevent, lambda x: cuda.cuEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
  cuda.cuEventRecord(evs[0], None)
  cb()
  cuda.cuEventRecord(evs[1], None)
  check(cuda.cuEventSynchronize(evs[1]))
  cuda.cuEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: cuda.cuEventDestroy_v2(ev)
  return ret.value * 1e-3

class CUDAProgram:
  def __init__(self, dev:CUDADevice, name:str, lib:bytes, smem:int=0):
    self.dev, self.name, self.lib, self.smem = dev, name, lib, smem
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))

    check(cuda.cuCtxSetCurrent(self.dev.context))
    self.module = cuda.CUmodule()
    status = cuda.cuModuleLoadData(ctypes.byref(self.module), lib)
    if status != 0:
      del self.module
      raise RuntimeError(f"module load failed with status code {status}: {cuda.CUresult.get(status)}")
    check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
    self.prg = prg
    if self.smem > 0: check(cuda.cuFuncSetAttribute(self.prg, cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))

  @suppress_finalizing
  def __del__(self): check(cuda.cuModuleUnload(self.module))

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    _dump_gpfifo("before __call__")
    check(cuda.cuCtxSetCurrent(self.dev.context))
    if not hasattr(self, "vargs"):
      self.c_args, self.vargs = encode_args(args, vals)

      # HACK: For MOCKGPU send the args struct itself.
      if MOCKGPU: self.vargs = self.c_args # type: ignore[assignment]
    else:
      for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    x = cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, self.smem, None, None, self.vargs)), enable=wait)
    _dump_gpfifo("after __call__")
    return x

class CUDAAllocator(LRUAllocator['CUDADevice']):
  def _alloc(self, size, options:BufferSpec):
    check(cuda.cuCtxSetCurrent(self.dev.context))
    if options.external_ptr: return cuda.CUdeviceptr_v2(options.external_ptr)
    if options.host: return init_c_var(ctypes.c_void_p, lambda x: check(cuda.cuMemHostAlloc(ctypes.byref(x), size, 0x01)))
    return init_c_var(cuda.CUdeviceptr, lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size)))
  def _free(self, opaque, options:BufferSpec):
    try:
      if options.host: check(cuda.cuMemFreeHost(opaque))
      else: check(cuda.cuMemFree_v2(opaque))
    except (TypeError, AttributeError): pass
  def _copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.dev.context))
    host_mem = self.alloc(len(src), BufferSpec(host=True))
    self.dev.pending_copyin.append((host_mem, len(src), BufferSpec(host=True)))
    ctypes.memmove(host_mem, mv_address(src), len(src))
    check(cuda.cuMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))
  def _copyout(self, dest:memoryview, src):
    CUDADevice.synchronize_system()
    check(cuda.cuCtxSetCurrent(self.dev.context))
    check(cuda.cuMemcpyDtoH_v2(mv_address(dest), src, len(dest)))
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    check(cuda.cuCtxSetCurrent(src_dev.context))
    check(cuda.cuEventCreate(ctypes.byref(sync_event := cuda.CUevent()), 0))
    check(cuda.cuMemcpyDtoDAsync_v2(dest, src, sz, None))
    check(cuda.cuEventRecord(sync_event, None))
    check(cuda.cuCtxSetCurrent(dest_dev.context))
    check(cuda.cuStreamWaitEvent(None, sync_event, 0)) # sync the default stream on the dest dev
  def _offset(self, buf, size:int, offset:int): return cuda.CUdeviceptr_v2(buf.value + offset)

class CUDADevice(Compiled):
  devices: list[CUDADevice] = []
  peer_access = False

  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    check(cuda.cuInit(0))
    self.cu_device = init_c_var(cuda.CUdevice, lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(cuda.CUcontext, lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, self.cu_device)))
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
    self.pending_copyin: list[tuple[int, int, BufferSpec|None]] = []
    CUDADevice.devices.append(self)

    # Initialize CUPTI profiling if enabled
    if cupti_profiler: cupti_profiler.init(self.context, device_id)

    from tinygrad.runtime.graph.cuda import CUDAGraph
    compilers = CompilerSet([CompilerPair(functools.partial(CUDARenderer, self.arch), functools.partial(CUDACompiler, self.arch)),
                             CompilerPair(functools.partial(PTXRenderer, self.arch), functools.partial(PTXCompiler, self.arch), CUDA_PTX),
                             CompilerPair(functools.partial(CUDARenderer, self.arch), functools.partial(NVCCCompiler, self.arch))], ctrl_var=CUDA_CC)
    super().__init__(device, CUDAAllocator(self), compilers, functools.partial(CUDAProgram, self), None if MOCKGPU else CUDAGraph)
    _dump_gpfifo("after __init__")

  def synchronize(self):
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
    if cupti_profiler: cupti_profiler.flush()
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in CUDADevice.devices: d.synchronize()
