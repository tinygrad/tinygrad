from __future__ import annotations
import subprocess, pathlib, struct, ctypes, tempfile, functools, platform, weakref, threading, array, sys
from tinygrad.helpers import to_mv, round_up, cache_dir, unwrap, prod, dedup
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, Compiler, CompileError, MMIOInterface
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq2 import HWQueue, EncodeCtx, encode_submit, ccall, patch, layout_args
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
DLL("CoreGraphics", "CoreGraphics")

# FIXME: these need autogen to support objc categories
# https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjectiveC/Chapters/ocCategories.html
@functools.cache
def to_ns_str(s:str): return ctypes.cast(objc.msg("stringWithUTF8String:")(metal.NSString._objc_class_, s.encode()), metal.NSString).own()
def checked(fn, *args): # fn(*args, &error), raised if set
  ret = fn(*args, ctypes.byref(err:=metal.NSError()))
  if err.value is not None: raise RuntimeError(bytes(objc.msg("UTF8String", ctypes.c_char_p)(err.localizedDescription())).decode())
  return ret

class MetalCompiler(Compiler):
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  import tinygrad.runtime.autogen.llvm as _
  support = DLL("MTLCompiler", "MTLCompiler")
  support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

  def __init__(self):
    self.cgs = ctypes.c_void_p(MetalCompiler.support.MTLCodeGenServiceCreate(b"tinygrad"))
    super().__init__("compile_metal_direct")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    ret: Exception|bytes = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal ret
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        ret = CompileError(errorMessage.decode())

    # no changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old versions.
    macos_major = int(platform.mac_ver()[0].split('.')[0])
    metal_version = "metal4.0" if macos_major >= 26 else "metal3.1" if macos_major >= 14 else "metal3.0" if macos_major >= 13 else "macos-metal2.0"

    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}" -fno-caret-diagnostics'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    MetalCompiler.support.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(ret, Exception): raise ret
    assert ret[:4] == b"MTLB" and ret[-4:] == b"ENDT", f"Invalid Metal library. {ret!r}"
    return ret
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      proc = subprocess.Popen(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}",
                              stdout=subprocess.PIPE, shell=True, text=True, bufsize=1)
      for line in unwrap(proc.stdout): print(line, end="")
      ret = proc.wait()
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

# *****************
# queue

HANDLES = ("queue", "event", "fence", "resources", "count")
SELECTORS = ("commandBuffer", "computeCommandEncoder", "waitForFence:", "updateFence:", "encodeSignalEvent:value:", "endEncoding", "commit",
             "useResources:count:usage:", "executeCommandsInBuffer:withRange:", "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
             "setComputePipelineState:", "dispatchThreadgroups:threadsPerThreadgroup:", "signaledValue")
MSGSEND = {ret: metal.dll.bind(ret)(metal.dll.objc_msgSend) for ret in (None, ctypes.c_void_p)} # objc_msgSend by return type

def mtl_sel(dev, name:str) -> UOp:
  return UOp.placeholder((len(HANDLES) + len(SELECTORS),), dtypes.uint64, 0, device=dev, tag="mtl_sel").index((HANDLES + SELECTORS).index(name))
def mtl_cb(dev) -> UOp: return UOp.placeholder((1,), dtypes.uint64, 0, device=dev, volatile=True, tag="mtl_cb").index(0) # the command buffer
def mtl_enc(dev) -> UOp: return UOp.placeholder((1,), dtypes.uint64, 0, device=dev, volatile=True, tag="mtl_enc").index(0) # its encoder

def mtl_msg(h:UOp, target:UOp, sel:str, *args:UOp|int, ret=None) -> UOp: # objc_msgSend, after h
  obj = target.src[0].after(h).index(target.src[1]).load()
  return ccall(MSGSEND[ret], obj, mtl_sel(obj.device, sel).load(), *[UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args])

# the timeline is the event
def mtl_poll(tl:UOp) -> UOp: return mtl_msg(tl, mtl_sel(tl.device, "event"), "signaledValue", ret=ctypes.c_void_p)

class MetalQueue(HWQueue):
  dev:MetalDevice
  def __init__(self, ctx:EncodeCtx, submit:UOp):
    super().__init__(ctx, submit)
    self.rows, self.cmds, self.sizes, self.stamps, self.nbytes = list[tuple[int, UOp]](), list[tuple](), list[tuple[int, int]](), list[UOp](), 0

  def exec(self, call:UOp, prg:UOp):
    bufs, vals, obj = get_call_arg_uops(call), get_call_var_uops(call, prg), prg.to_elf()
    args = [bufs[i].getaddr(self.devs) for i in prg.arg.globals] + [v.ccast(var.dtype) for v, var in zip(vals, prg.arg.vars)]
    self.rows += (rows:=layout_args(args, off:=round_up(self.nbytes, 256)))
    self.nbytes = max([o + w.dtype.itemsize for o, w in rows], default=off + 8)

    # symbolic sizes, set on the command at run time
    dims = (*prg.arg.global_size, *prg.arg.local_size)
    if any(isinstance(d, UOp) for d in dims):
      self.sizes.append((len(self.cmds), at:=round_up(self.nbytes, 8)))
      self.rows += layout_args([d.cast(dtypes.uint64) if isinstance(d, UOp) else UOp.const(d, dtypes.uint64) for d in dims], at)
      self.nbytes = at + 48
    self.cmds.append((obj.lib, obj.name, tuple(1 if isinstance(d, UOp) else int(d) for d in dims), off))

  def wait(self, dst:UOp, val:UOp, eq=False): pass # the fence orders the queue
  def timestamp(self, dst:UOp): self.stamps.append(dst)
  def signal(self, dst:UOp, val:UOp): self.value = val

  def submit(self, cmdbuf:UOp) -> UOp:
    n, zero, pipes = len(self.cmds), round_up(self.nbytes, 8), dedup(c[:2] for c in self.cmds)
    buf = UOp.placeholder((zero + 24 + 8 * (1 + n + len(pipes)),), dtypes.uint8, device=self.devs, volatile=True,
                          tag=("mtl_icb", tuple(self.cmds), zero + 24))
    args = patch(buf, self.rows + [(zero + 8 * i, UOp.const(0, dtypes.uint64)) for i in range(3)])
    header, cb, enc, h = args.bitcast(dtypes.uint64)[zero // 8 + 3:], mtl_cb(self.devs), mtl_enc(self.devs), args

    # symbolic sizes
    for ci, off in self.sizes:
      h = mtl_msg(h, header.index(1 + ci), "concurrentDispatchThreadgroups:threadsPerThreadgroup:", args.index(off), args.index(off + 24))

    def run(h:UOp, first:UOp|int, count:int, last:bool) -> UOp: # a command buffer for the commands [first, first + count)
      h = cb.store(cbuf:=mtl_msg(h, mtl_sel(self.devs, "queue"), "commandBuffer", ret=ctypes.c_void_p))
      h = enc.store(mtl_msg(h, cb, "computeCommandEncoder", ret=ctypes.c_void_p))
      h = mtl_msg(h, enc, "waitForFence:", mtl_sel(self.devs, "fence").load())
      if self.dev.residency.value is None: # no residency set: declare the buffers
        h = mtl_msg(h, enc, "useResources:count:usage:", mtl_sel(self.devs, "resources").load(), mtl_sel(self.devs, "count").load(), 3)

      # before apple9 the encoder must use the pipelines
      if not self.dev.arch.startswith("Apple") or int(self.dev.arch[5:]) < 9:
        r = UOp.range(len(pipes), next(UOp.unique_num), dtype=dtypes.uint64)
        h = mtl_msg(h, enc, "setComputePipelineState:", header.index(1 + n + r).load())
        h = mtl_msg(h, enc, "dispatchThreadgroups:threadsPerThreadgroup:", args.index(zero), args.index(zero)).end(r)

      h = mtl_msg(h, enc, "executeCommandsInBuffer:withRange:", header.after(h).index(0).load(), first, count)
      h = mtl_msg(h, enc, "updateFence:", mtl_sel(self.devs, "fence").load())
      h = mtl_msg(h, enc, "endEncoding")

      # write meta to collect timestamps: [command buffer, 0] until synchronize reads its times
      # MTL4 solves that dance
      if self.stamps:
        slots = self.stamps[0].src[0] # [signal, timeline, [x, start, x, end]...]
        h = slots.after(h).index(7 + 4 * first).store(0)
        h = slots.after(h).index(5 + 4 * first).store(cbuf)

      if last: h = mtl_msg(h, cb, "encodeSignalEvent:value:", mtl_sel(self.devs, "event").load(), self.value)
      return mtl_msg(h, cb, "commit")

    # collect timestamps using cmdbuf metrics, so sep cmdbufs
    if not self.stamps: return run(h, 0, n, True)
    if n > 1: h = run(h.after(r:=UOp.range(n - 1, next(UOp.unique_num), dtype=dtypes.uint64)), r, 1, False).end(r)
    return run(h, n - 1, 1, True)

# *****************
# device

class MetalAllocator(Allocator['MetalDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    mtl = metal.MTLBuffer(options.external_ptr) if options.external_ptr else \
          self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    if mtl.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    self.dev.mark_resident(mtl, True)
    return BufferStorage(mtl.gpuAddress(), mtl, MMIOInterface(c, size) if (c:=mtl.contents()) else None)
  def _free(self, storage:BufferStorage, options:BufferSpec): # metal doesn't track what a kernel reaches
    self.dev.synchronize()
    self.dev.mark_resident(storage.meta, False)
    storage.meta.retain = False
    storage.meta.release()
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class MetalDevice(Compiled):
  has_copy_queue = False
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_metal_compute", name="submit"), lambda ctx, submit: encode_submit(MetalQueue(ctx, submit))),
  ])
  pm_lower = PatternMatcher([
    (UPat.var("tl").index(UPat(Ops.CONST, arg=0)).load(), lambda tl: mtl_poll(tl) if tl.without_after.tag == "timeline" else None),
  ])

  def __init__(self, device:str=""):
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()

    # queue allocation
    self.queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    if self.queue.value is None: raise RuntimeError("Cannot allocate a new command queue")

    # try to use residency set when supported
    rsd = ctypes.cast(objc.msg("new", clsmeth=True)(metal.MTLResidencySetDescriptor), metal.MTLResidencySetDescriptor)
    self.residency = self.sysdevice.newResidencySetWithDescriptor_error(rsd, None)
    if self.residency.value is not None: self.queue.addResidencySet(self.residency)

    self.resources, self.table = list[int](), (ctypes.c_uint64 * 1)()
    self.event, self.fence = self.sysdevice.newSharedEvent(), self.sysdevice.newFence()
    self.icbs, self.profile_slots = weakref.WeakKeyDictionary[Buffer, tuple](), weakref.WeakSet[Buffer]()

    # https://developer.apple.com/documentation/metal/mtlgpufamily
    def check_family(f): return next(filter(self.sysdevice.supportsFamily, reversed([v for v, nm in metal.enum_MTLGPUFamily.items() if f in nm])), 0)
    super().__init__(device, MetalAllocator(self), [MetalRenderer], None,
                     arch=metal.enum_MTLGPUFamily[check_family("Apple") or check_family("Mac")][12:])
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="mtl_sel"), lambda ctx: ctx.sels),
      (UPat(Ops.PARAM, tag="slots", name="b"), lambda ctx, b: ctx.new_slots(b.max_numel()) if b.max_numel() > 4 else None), # with stamps
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.new_icb(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == "mtl_icb" else None),
    ]) + self.pm_bufferize

  @functools.cached_property
  def sels(self) -> Buffer: # handles, then selectors
    vals = [self.queue.value, self.event.value, self.fence.value, ctypes.addressof(self.table), len(self.resources),
            *[objc.getsel(s.encode()).value for s in SELECTORS]]
    return Buffer(self.host, len(vals), dtypes.uint64, initial_value=struct.pack(f"{len(vals)}Q", *vals))

  def mark_resident(self, mtl:metal.MTLBuffer, add:bool):
    if self.residency.value is not None:
      objc.msg("addAllocation:" if add else "removeAllocation:", None, [objc.id_])(self.residency, mtl)
      return objc.msg("commit", None)(self.residency)

    self.resources.append(unwrap(mtl.value)) if add else self.resources.remove(unwrap(mtl.value))
    self.table = (ctypes.c_uint64 * max(len(self.resources), 1))(*self.resources)

    # update sels table
    if "sels" in self.__dict__: self.sels.host.view(fmt='Q')[3:5] = array.array('Q', [ctypes.addressof(self.table), len(self.resources)])

  def new_slots(self, n:int) -> Buffer:
    self.profile_slots.add(buf:=Buffer(self.host, n, dtypes.uint64, initial_value=bytes(8 * n)))
    return buf

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(lib, len(lib), None, None))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(library.newFunctionWithName(to_ns_str(name)))
    descriptor.setSupportIndirectCommandBuffers(True)
    return checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor, metal.MTLPipelineOptionNone, None)

  def new_icb(self, cmds:tuple[tuple[bytes, str, tuple[int, ...], int], ...], header:int) -> Buffer:
    pipes = dedup(c[:2] for c in cmds)
    buf = Buffer(self.device, header + 8 * (1 + len(cmds) + len(pipes)), dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
    desc = metal.MTLIndirectCommandBufferDescriptor.new()
    desc.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    desc.setMaxKernelBufferBindCount(1)
    icb = self.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(desc, max(len(cmds), 1), 0)
    if icb.value is None: raise RuntimeError("create indirect command buffer failed, does your system support this?")
    commands = [icb.indirectComputeCommandAtIndex(i).own() for i in range(len(cmds))]
    for cmd, (lib, name, dims, off) in zip(commands, cmds):
      cmd.setComputePipelineState(state:=self.pipeline(lib, name))
      if prod(dims[3:]) > (mx:=state.maxTotalThreadsPerThreadgroup()): raise RuntimeError(f"local size {dims[3:]} bigger than {mx}")
      cmd.setKernelBuffer_offset_atIndex(buf.get_storage().meta, off, 0)
      cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*dims[:3]), metal.MTLSize(*dims[3:]))
      cmd.setBarrier()
    buf.host.view(fmt='Q')[header // 8:] = array.array('Q', [icb.value, *[c.value for c in commands], *[self.pipeline(*p).value for p in pipes]])
    self.icbs[buf] = (icb, commands)
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if sys.is_finalizing(): return # the event doesn't wake at exit
    wait = objc.msg("waitUntilSignaledValue:timeoutMS:", ctypes.c_bool, [ctypes.c_uint64, ctypes.c_uint64])
    if not wait(self.event, value, int(self.wait_timeout_ms)): raise RuntimeError(f"{self.device} signal wait timed out")

  def synchronize(self, timeout:int|None=None):
    for buf in list(self.profile_slots): # pending: [command buffer, 0]
      slots = buf.host.view(fmt='Q')
      for start in range(5, buf.size, 4):
        if slots[start] and not slots[start + 2]:
          (cb:=metal.MTLCommandBuffer(slots[start])).waitUntilCompleted()
          slots[start], slots[start + 2] = int(cb.GPUStartTime() * 1e9), int(cb.GPUEndTime() * 1e9)

    super().synchronize(timeout)
    if (pool:=getattr(pools, "pool", None)) is not None: objc.lib.objc_autoreleasePoolPop(pool)
    pools.pool = objc.lib.objc_autoreleasePoolPush()

pools = threading.local()
