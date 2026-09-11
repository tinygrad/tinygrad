from __future__ import annotations
import subprocess, pathlib, struct, ctypes, tempfile, functools, platform, weakref, threading
from tinygrad.helpers import to_mv, round_up, cache_dir, unwrap, prod
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, Compiler, CompileError, MMIOInterface
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal, libc
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq2 import HWQueue, EncodeCtx, encode_submit, ccall, unwrap_view, HCQ_RUNTIME_DEV
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

pools = threading.local() # per thread, the autorelease pool the command buffers and encoders of its runs drain into at synchronize

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
# queue: the body is a chain of objc calls, the kernels run from an indirect command buffer

ICB_COUNT = 16
HANDLES = ("queue", "event", "fence")
SELECTORS = ("commandBuffer", "computeCommandEncoder", "executeCommandsInBuffer:withRange:", "endEncoding", "commit", "waitUntilCompleted",
  "setKernelBuffer:offset:atIndex:", "concurrentDispatchThreadgroups:threadsPerThreadgroup:", "GPUStartTime", "GPUEndTime",
  "encodeSignalEvent:value:", "signaledValue", "setSignaledValue:", "waitForFence:", "updateFence:")
def host_buf(*vals:int) -> Buffer:
  return Buffer(HCQ_RUNTIME_DEV.value, len(vals), dtypes.uint64, initial_value=struct.pack(f"{len(vals)}Q", *vals))

def mtl_const(name:str, devs:tuple[str, ...]) -> UOp: # a device's handles and the selectors: one host buffer the body loads from
  i = (HANDLES + SELECTORS).index(name)
  return UOp.placeholder((len(HANDLES) + len(SELECTORS),), dtypes.uint64, 0, device=devs, tag="mtl").index(i)

class MetalQueue(HWQueue):
  dev:MetalDevice

  def __init__(self, ctx:EncodeCtx, submit:UOp):
    super().__init__(ctx, submit)
    # a command per call: its pipeline and static sizes are set when the icb is made, the body binds the buffers before every run
    cmds = [(prg.src[3].arg, prg.arg.function_name, tuple(1 if isinstance(d, UOp) else int(d) for d in self.dims(prg)))
            for prg in [u.src[0] for u in self.lin.src if u.op is Ops.CALL]] # a serializable recipe, never an owned pipeline pointer
    self.stride = 4 + len(cmds) # each entry: completion value, icb, scalar buffer, scalar host address, commands
    self.pool = UOp.placeholder((1 + ICB_COUNT * self.stride,), dtypes.uint64, device=self.devs, volatile=True, tag=("icb", tuple(cmds)))
    self.slot = self.pool.index(0).load()
    self.icb = self.pool_ref(1)
    self.blob_buf = UOp.placeholder((8,), dtypes.uint8, device=self.devs) # stands in for the blob's buffer until submit
    handles = UOp.placeholder((2,), dtypes.uint64, device=self.devs, volatile=True, tag="mtl_handles") # [command buffer, open encoder]
    self.cb, self.enc, self.root = handles.index(0), handles.index(1), handles.after(self.blob_buf, self.slot)
    loop = UOp.loop(len(ctx.devs) + ctx.devs.index(self.devs[0])) # hcq_fence uses the first len(ctx.devs) loop IDs
    done = self.call(self.root.after(loop), mtl_const("event", self.devs), "signaledValue", restype=ctypes.c_void_p) # uint64 return ABI
    self.root = self.root.after(done.end(loop, done < self.pool_ref(0).load())) # only wait when reusing an in-flight icb
    self.tail, self.setups = self.root, list[UOp]() # the command buffer is encoded in order, the icb is written wide before the commit
    self.count, self.done, self.signals = 0, 0, list[tuple[UOp, UOp]]()
    self.start:UOp|None = None
    self.scalar_count = 0
    self.words(0) # empty kernels still need a nonempty host scratch buffer

  def pool_ref(self, i:int) -> UOp: return self.pool.index(1 + self.slot * self.stride + i)

  @staticmethod
  def dims(prg:UOp) -> tuple: return (*prg.arg.global_size, *prg.arg.local_size)
  def words(self, *ws:UOp|int) -> int: # append 64-bit words to the blob, returns the offset of the first
    return self.q(*[w.ccast(dtypes.uint64) if isinstance(w, UOp) else UOp.const(w, dtypes.uint64) for w in ws]) - 8 * len(ws)
  def ptr(self, off:int) -> UOp: return self.blob_buf.after(self.root).bitcast(dtypes.uint64).index(off // 8) # into the blob, after its patches
  def binding(self, buf:UOp) -> tuple[int, int]: # a buffer binds as its base's mtlbuffer: (the blob word holding it, the view's offset)
    base, off = unwrap_view(buf)
    if base.op is Ops.MSELECT:
      lane, lane_off = unwrap_view(base.src[0])
      base, off = lane.mselect(base.arg), off + lane_off
    return self.words(base.getaddr(self.devs)), off

  def call(self, after:UOp, target:UOp, sel:str, *args:UOp|int, result:UOp|None=None, restype=None) -> UOp:
    fn = metal.dll.bind(restype or (ctypes.c_void_p if result is not None else None))(metal.dll.objc_msgSend)
    cargs = [UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args]
    ret = ccall(fn, target.src[0].after(after).index(target.src[1]).load(), mtl_const(sel, self.devs).load(), *cargs)
    return result.src[0].after(after).index(result.src[1]).store(ret) if result is not None else ret
  def msg(self, target:UOp, sel:str, *args:UOp|int, result:UOp|None=None):
    self.tail = self.tail.after(self.call(self.tail, target, sel, *args, result=result))
  def setup(self, cmd:UOp, sel:str, *args:UOp|int): self.setups.append(self.call(self.root, cmd, sel, *args))

  def exec(self, call:UOp, prg:UOp):
    cmd, bufs, vals = self.pool_ref(4 + self.count), get_call_arg_uops(call), get_call_var_uops(call, prg)
    for i, arg in enumerate(prg.arg.globals):
      word, off = self.binding(bufs[arg])
      self.setup(cmd, "setKernelBuffer:offset:atIndex:", self.ptr(word).load(), off, i)
    for i, v in enumerate(vals):
      scalars = self.pool.after(self.root).index(1 + self.slot * self.stride + 3).load()
      self.setups.append(ccall(libc.memcpy, scalars + self.scalar_count * 8, self.ptr(self.words(v)), 8))
      self.setup(cmd, "setKernelBuffer:offset:atIndex:", self.pool_ref(2).load(), self.scalar_count * 8, len(prg.arg.globals) + i)
      self.scalar_count += 1
    if any(isinstance(d, UOp) for d in self.dims(prg)): # arm64 passes MTLSize by reference
      self.setup(cmd, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", self.ptr(sizes:=self.words(*self.dims(prg))), self.ptr(sizes + 24))
    self.count += 1

  def wait(self, dst:UOp, val:UOp): pass # the command buffer waits for the preceding batch on the GPU
  def timestamp(self, dst:UOp): # profiling isolates each kernel so command-buffer times measure that kernel
    if self.start is None: self.start = dst
    else:
      self.finish()
      self.msg(self.cb, "waitUntilCompleted")
      for slot, sel in ((self.start, "GPUStartTime"), (dst, "GPUEndTime")):
        tm = self.call(self.tail, self.cb, sel, restype=ctypes.c_double)
        self.tail = self.tail.after(slot.after(self.tail).index(1).store((tm * 1e9).cast(dtypes.uint64)))
      self.start = None
  def signal(self, dst:UOp, val:UOp): self.signals.append((dst, val))

  def finish(self, signal:UOp|None=None):
    if self.count == self.done: return
    self.msg(mtl_const("queue", self.devs), "commandBuffer", result=self.cb)
    self.msg(self.cb, "computeCommandEncoder", result=self.enc)
    self.msg(self.enc, "waitForFence:", mtl_const("fence", self.devs).load())
    self.msg(self.enc, "executeCommandsInBuffer:withRange:", self.icb.load(), self.done, self.count - self.done)
    self.msg(self.enc, "updateFence:", mtl_const("fence", self.devs).load())
    self.msg(self.enc, "endEncoding")
    self.tail = self.tail.after(*self.setups)
    if signal is not None: self.msg(self.cb, "encodeSignalEvent:value:", mtl_const("event", self.devs).load(), signal)
    self.msg(self.cb, "commit")
    self.done, self.setups = self.count, []

  def submit(self, cmdbuf:UOp) -> UOp:
    value = self.signals[-1][1]
    if self.count == self.done: self.msg(mtl_const("event", self.devs), "setSignaledValue:", value) # profiled kernels already completed
    else: self.finish(value)
    self.tail = self.tail.after(self.pool.after(self.tail).index(1 + self.slot * self.stride).store(value))
    self.tail = self.tail.after(self.pool.after(self.tail).index(0).store((self.slot + 1) % ICB_COUNT))
    # HCQ's scratch is reusable after encoding. Host accesses wait on the GPU event in _wait_signal.
    for dst, val in self.signals: self.tail = self.tail.after(dst.after(self.tail).index(0).store(val))
    return self.tail.substitute({self.blob_buf: cmdbuf, self.pool: self.pool.replace(tag=(*self.pool.tag, self.scalar_count))})

# *****************
# device

class MetalAllocator(Allocator['MetalDevice']):
  def __init__(self, dev:MetalDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    mtl = metal.MTLBuffer(options.external_ptr) if options.external_ptr else \
          self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    if mtl.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    self.dev.resident(mtl)
    return BufferStorage(mtl.value, mtl, MMIOInterface(c, size) if (c:=mtl.contents()) else None) # an external buffer may have no host side

  def do_free(self, storage:BufferStorage, options:BufferSpec): # the icb doesn't retain what it binds: the gpu must be done with a buffer first
    self.dev.synchronize()
    self.dev.resident(storage.meta, False)
    super().do_free(storage, options) # an external buffer only leaves the residency set
  def _free(self, storage:BufferStorage, options:BufferSpec): # released now, not when the storage is collected
    storage.meta.retain = False
    storage.meta.release()

  def _offset(self, buf:int, size:int, offset:int) -> int: return buf # a view binds its base's mtlbuffer, the offset rides with the Buffer

class MetalDevice(Compiled):
  has_copy_queue = False
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_metal_compute", name="submit"), lambda ctx, submit: encode_submit(MetalQueue(ctx, submit))),
  ])

  def __init__(self, device:str=""):
    if int(platform.mac_ver()[0].split('.')[0]) < 15: raise RuntimeError("METAL needs macOS 15 for residency sets")
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()
    self.queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    self.event = self.sysdevice.newSharedEvent()
    self.fence = self.sysdevice.newFence()
    if self.queue.value is None: raise RuntimeError("Cannot allocate a new command queue")

    # the buffers of an indirect command buffer must be resident: everything the device allocates is
    self.residency = checked(self.sysdevice.newResidencySetWithDescriptor_error, metal.MTLResidencySetDescriptor.new())
    self.queue.addResidencySet(self.residency)
    self.icbs:weakref.WeakKeyDictionary[Buffer, tuple] = weakref.WeakKeyDictionary() # an icb and its commands live as long as their words

    # https://developer.apple.com/documentation/metal/mtlgpufamily
    def check_family(f): return next(filter(self.sysdevice.supportsFamily, reversed([v for v, nm in metal.enum_MTLGPUFamily.items() if f in nm])), 0)
    super().__init__(device, MetalAllocator(self), [MetalRenderer], None,
                     arch=metal.enum_MTLGPUFamily[check_family("Apple") or check_family("Mac")][12:])
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="mtl"), lambda ctx: ctx.consts),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.new_icb(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == "icb" else None),
    ]) + self.pm_bufferize

  @functools.cached_property
  def consts(self) -> Buffer:
    return host_buf(*[unwrap(getattr(self, n).value) for n in HANDLES], *[unwrap(objc.getsel(s.encode()).value) for s in SELECTORS])

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(lib, len(lib), None, None))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(library.newFunctionWithName(to_ns_str(name)))
    descriptor.setSupportIndirectCommandBuffers(True)
    return checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor, metal.MTLPipelineOptionNone, None)

  def new_icb(self, cmds:tuple[tuple[bytes, str, tuple[int, ...]], ...], scalar_count:int) -> Buffer:
    descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    descriptor.setMaxKernelBufferBindCount(31)
    words, refs = [0], []
    for _ in range(ICB_COUNT):
      icb = self.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(descriptor, max(len(cmds), 1), 0)
      if icb.value is None: raise RuntimeError("create indirect command buffer failed, does your system support this?")
      objs = [icb.indirectComputeCommandAtIndex(i).own() for i in range(len(cmds))]
      for cmd, (lib, name, dims) in zip(objs, cmds):
        state = self.pipeline(lib, name)
        if prod(dims[3:]) > (mx:=state.maxTotalThreadsPerThreadgroup()): raise RuntimeError(f"local size {dims[3:]} bigger than {mx}")
        cmd.setComputePipelineState(state)
        cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*dims[:3]), metal.MTLSize(*dims[3:]))
        cmd.setBarrier() # the kernels of a batch run in order
      scalars = Buffer(self.device, max(scalar_count, 1), dtypes.uint64, options=BufferSpec(nolru=True), preallocate=True)
      words += [0, icb.value, scalars._buf, scalars.host.addr, *[c.value for c in objs]]
      refs.append((icb, objs, scalars))
    self.icbs[buf:=host_buf(*words)] = tuple(refs)
    return buf

  def resident(self, mtl:metal.MTLBuffer, add:bool=True):
    (self.residency.addAllocation if add else self.residency.removeAllocation)(ctypes.cast(mtl, metal.MTLAllocation))
    self.residency.commit()

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if not self.event.waitUntilSignaledValue_timeoutMS(value, timeout or int(self.wait_timeout_ms)):
      raise RuntimeError(f"{self.device} signal wait timed out")

  def synchronize(self, timeout:int|None=None):
    if not self.timeline.is_allocated(): return
    super().synchronize(timeout)
    # the gpu is done with every command buffer: drain them. a nested synchronize (a free during collection) finds no pool to pop
    if (pool:=getattr(pools, "pool", None)) is not None: objc.lib.objc_autoreleasePoolPop(pool)
    pools.pool = objc.lib.objc_autoreleasePoolPush()
