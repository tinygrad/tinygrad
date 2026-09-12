from __future__ import annotations
import subprocess, pathlib, struct, ctypes, tempfile, functools, platform, weakref, threading
from tinygrad.helpers import to_mv, round_up, cache_dir, unwrap, prod
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, Compiler, CompileError, MMIOInterface
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal, libc
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq2 import HWQueue, EncodeCtx, encode_submit, ccall, patch, unwrap_view, _is_input_addr, HCQ_RUNTIME_DEV
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

pools = threading.local() # command buffers and encoders live until synchronize

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
# UOps implementation

ICB_COUNT = 16
HANDLES = ("queue", "event", "fence")
SELECTORS = ("commandBuffer", "computeCommandEncoder", "executeCommandsInBuffer:withRange:", "endEncoding", "commit",
  "setKernelBuffer:offset:atIndex:", "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
  "encodeSignalEvent:value:", "signaledValue", "waitForFence:", "updateFence:", "useResources:count:usage:")
def host_buf(*vals:int) -> Buffer:
  return Buffer(HCQ_RUNTIME_DEV.value, len(vals), dtypes.uint64, initial_value=struct.pack(f"{len(vals)}Q", *vals))

def mtl_const(name:str, devs) -> UOp: # host handles and selectors
  i = (HANDLES + SELECTORS).index(name)
  return UOp.placeholder((len(HANDLES) + len(SELECTORS),), dtypes.uint64, 0, device=devs, tag="mtl")[i:i+1]

def mtl_call(h:UOp, target:UOp, sel:str, *args:UOp|int, restype=None) -> UOp:
  target, idx = (target.src[0], target.src[1]) if target.op is Ops.INDEX else (target, 0)
  return ccall(metal.dll.bind(restype)(metal.dll.objc_msgSend), target.after(h).index(idx).load(), mtl_const(sel, h.device).index(0).load(),
               *[UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args])

def mtl_msg(h:UOp, target:UOp, sel:str, *args:UOp|int, result:UOp|None=None) -> UOp:
  call = mtl_call(h, target, sel, *args, restype=ctypes.c_void_p if result is not None else None)
  return h.after(result.after(h).index(0).store(call) if result is not None else call)

def mtl_wait(h:UOp, value:UOp) -> UOp: # only wait when reusing an in-flight icb
  done = mtl_call(h.after(loop:=UOp.loop(next(UOp.unique_num))), mtl_const("event", h.device), "signaledValue", restype=ctypes.c_void_p)
  return done.end(loop, done < value)

def mtl_encode(h:UOp, cb:UOp, enc:UOp, icb:UOp, first:int, count:int, resources:list[UOp]) -> UOp:
  fence = mtl_const("fence", h.device).index(0).load()
  h = mtl_msg(h, mtl_const("queue", h.device), "commandBuffer", result=cb)
  h = mtl_msg(h, cb, "computeCommandEncoder", result=enc)
  h = mtl_msg(h, enc, "waitForFence:", fence)
  if resources:=list(dict.fromkeys(resources)): # virt metal does not support residency sets
    table = UOp.placeholder((8 * len(resources),), dtypes.uint8, device=h.device, volatile=True, tag=f"resources_{first}")
    h = mtl_msg(h, enc, "useResources:count:usage:", patch(table, [(8 * i, buf) for i, buf in enumerate(resources)]).index(0), len(resources),
                metal.MTLResourceUsageRead | metal.MTLResourceUsageWrite)
  h = mtl_msg(h, enc, "executeCommandsInBuffer:withRange:", icb, first, count)
  h = mtl_msg(h, enc, "updateFence:", fence)
  return mtl_msg(h, enc, "endEncoding")

# *****************
# queue

class MetalQueue(HWQueue):
  dev:MetalDevice

  def __init__(self, ctx:EncodeCtx, submit:UOp):
    super().__init__(ctx, submit)
    # serializable recipes: pipelines, static sizes and fixed bindings are set at link
    cmds = [(prg.src[3].arg, prg.arg.function_name, tuple(1 if isinstance(d, UOp) else int(d) for d in self.dims(prg)))
            for prg in [u.src[0] for u in self.lin.src if u.op is Ops.CALL]]
    stride = 4 + len(cmds) # [completion value, icb, scalar buffer, scalar host address, commands...]
    self.pool = UOp.placeholder((1 + ICB_COUNT * stride,), dtypes.uint64, device=self.devs, volatile=True, tag=("icb", tuple(cmds)))
    self.slot = self.pool.index(0).load() % ICB_COUNT
    self.offset = 1 + self.slot * stride
    self.blob_buf = UOp.placeholder((8,), dtypes.uint8, device=self.devs) # replaced by cmdbuf at submit
    handles = UOp.placeholder((2,), dtypes.uint64, device=self.devs, volatile=True, tag="mtl_handles") # [command buffer, open encoder]
    self.cb, self.enc, self.root = handles[:1], handles[1:2], handles.after(self.blob_buf, self.slot)
    self.tail = self.root = self.root.after(mtl_wait(self.root, self.pool_ref(0).load()))
    self.setups, self.resources = list[UOp](), list[UOp]()
    self.count, self.done, self.signals = 0, 0, list[tuple[UOp, UOp]]()
    self.bindings:list[tuple[int, int, Buffer|None, int]] = [] # command, argument, fixed buffer (None for scalars), offset
    self.ready = UOp(Ops.NOOP, tag=next(UOp.unique_num)) # all icb updates must precede the first commit
    self.start:UOp|None = None
    self.words(0) # nonempty scratch for kernels without arguments

  def pool_ref(self, i:int, *after:UOp) -> UOp: return self.pool.after(*after).index(self.offset + i)

  @staticmethod
  def dims(prg:UOp) -> tuple: return (*prg.arg.global_size, *prg.arg.local_size)
  def words(self, *ws:UOp|int) -> int: # append uint64 words, return byte offset
    return self.q(*[w.ccast(dtypes.uint64) if isinstance(w, UOp) else UOp.const(w, dtypes.uint64) for w in ws]) - 8 * len(ws)
  def ptr(self, off:int) -> UOp: return self.blob_buf.after(self.root).bitcast(dtypes.uint64).index(off // 8)
  def binding(self, buf:UOp, idx:int) -> UOp:
    base, off = unwrap_view(buf)
    if base.op is Ops.MSELECT:
      lane, lane_off = unwrap_view(base.src[0])
      base, off = lane.mselect(base.arg), off + lane_off
    resource = self.ptr(self.words(addr:=base.getaddr(self.devs))).load()
    if not _is_input_addr(addr) and isinstance(fixed:=getattr(base.arg, "buffer", None), Buffer):
      self.bindings.append((self.count, idx, fixed, off))
    else: self.setup("setKernelBuffer:offset:atIndex:", resource, off, idx)
    return resource

  def setup(self, sel:str, *args:UOp|int): self.setups.append(mtl_call(self.root, self.pool_ref(4 + self.count), sel, *args))

  def exec(self, call:UOp, prg:UOp):
    bufs, vals, dims = get_call_arg_uops(call), get_call_var_uops(call, prg), self.dims(prg)
    self.resources.extend(self.binding(bufs[i], idx) for idx, i in enumerate(prg.arg.globals))
    if vals:
      self.setups.append(ccall(libc.memcpy, self.pool_ref(3).load() + (off:=self.words(*vals)), self.ptr(off), 8 * len(vals)))
      self.bindings.extend((self.count, len(prg.arg.globals) + i, None, off + 8 * i) for i in range(len(vals)))
      self.resources.append(self.pool_ref(2).load())
    if any(isinstance(d, UOp) for d in dims): # arm64 passes MTLSize by reference
      self.setup("concurrentDispatchThreadgroups:threadsPerThreadgroup:", self.ptr(off:=self.words(*dims)), self.ptr(off + 24))
    self.count += 1

  def wait(self, dst:UOp, val:UOp): pass # the command buffer waits for the preceding batch on the GPU
  def timestamp(self, dst:UOp): # one command buffer per profiled kernel
    if self.start is None: self.start = dst
    else:
      self.finish()
      # start holds the command buffer until synchronize; zero end marks pending timestamps
      h = self.tail
      self.tail = h.after(self.start.after(h).index(1).store(self.cb.after(h).index(0).load()), dst.after(h).index(1).store(0))
      self.start = None
  def signal(self, dst:UOp, val:UOp): self.signals.append((dst, val))

  def finish(self):
    if self.count == self.done: return
    if self.done: self.tail = mtl_msg(self.tail.after(self.ready), self.cb, "commit")
    self.tail = mtl_encode(self.tail, self.cb, self.enc, self.pool_ref(1).load(), self.done, self.count - self.done, self.resources)
    self.done, self.resources = self.count, []

  def submit(self, cmdbuf:UOp) -> UOp:
    self.finish()
    h = mtl_msg(self.tail, self.cb, "encodeSignalEvent:value:", mtl_const("event", self.devs).index(0).load(), value:=self.signals[-1][1])
    h = mtl_msg(h.after(self.ready), self.cb, "commit")
    h = h.after(self.pool_ref(0, h).store(value))
    h = h.after(self.pool.after(h).index(0).store((self.slot + 1) % ICB_COUNT))
    # scratch is reusable after encoding; host copies wait on the GPU event
    for dst, val in self.signals: h = h.after(dst.after(h).index(0).store(val))
    return h.substitute({self.ready: self.root.after(*self.setups), self.blob_buf: cmdbuf,
                         self.pool: self.pool.replace(tag=(*self.pool.tag, len(self.blob), tuple(self.bindings)))})

# *****************
# device

class MetalAllocator(Allocator['MetalDevice']):
  def __init__(self, dev:MetalDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    mtl = metal.MTLBuffer(options.external_ptr) if options.external_ptr else \
          self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    if mtl.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return BufferStorage(mtl.value, mtl, MMIOInterface(c, size) if (c:=mtl.contents()) else None) # an external buffer may have no host side

  def do_free(self, storage:BufferStorage, options:BufferSpec): # the icb doesn't retain what it binds: the gpu must be done with a buffer first
    self.dev.synchronize()
    super().do_free(storage, options)
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
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()
    self.queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    self.event, self.fence = self.sysdevice.newSharedEvent(), self.sysdevice.newFence()
    if self.queue.value is None: raise RuntimeError("Cannot allocate a new command queue")

    self.icbs:weakref.WeakKeyDictionary[Buffer, tuple] = weakref.WeakKeyDictionary() # native objects live with their host buffer
    self.profile_slots:weakref.WeakSet[Buffer] = weakref.WeakSet()

    # https://developer.apple.com/documentation/metal/mtlgpufamily
    def check_family(f): return next(filter(self.sysdevice.supportsFamily, reversed([v for v, nm in metal.enum_MTLGPUFamily.items() if f in nm])), 0)
    super().__init__(device, MetalAllocator(self), [MetalRenderer], None,
                     arch=metal.enum_MTLGPUFamily[check_family("Apple") or check_family("Mac")][12:])
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="mtl"), lambda ctx: ctx.consts),
      (UPat(Ops.PARAM, tag="slots", name="b"), lambda ctx, b: ctx.new_slots(b.max_numel()) if b.max_numel() > 4 else None),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.new_icb(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == "icb" else None),
    ]) + self.pm_bufferize

  @functools.cached_property
  def consts(self) -> Buffer:
    return host_buf(*[unwrap(getattr(self, n).value) for n in HANDLES], *[unwrap(objc.getsel(s.encode()).value) for s in SELECTORS])

  def new_slots(self, size:int) -> Buffer:
    self.profile_slots.add(buf:=host_buf(*[0] * size))
    return buf

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(lib, len(lib), None, None))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(library.newFunctionWithName(to_ns_str(name)))
    descriptor.setSupportIndirectCommandBuffers(True)
    return checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor, metal.MTLPipelineOptionNone, None)

  def new_icb(self, cmds:tuple[tuple[bytes, str, tuple[int, ...]], ...], blob_size:int,
              bindings:tuple[tuple[int, int, Buffer|None, int], ...]) -> Buffer:
    descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    descriptor.setMaxKernelBufferBindCount(31)
    states = [(self.pipeline(lib, name), dims) for lib, name, dims in cmds]
    for state, dims in states:
      if prod(dims[3:]) > (mx:=state.maxTotalThreadsPerThreadgroup()): raise RuntimeError(f"local size {dims[3:]} bigger than {mx}")
    words, refs = [0], []
    for _ in range(ICB_COUNT):
      icb = self.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(descriptor, max(len(cmds), 1), 0)
      if icb.value is None: raise RuntimeError("create indirect command buffer failed, does your system support this?")
      objs = [icb.indirectComputeCommandAtIndex(i).own() for i in range(len(cmds))]
      for cmd, (state, dims) in zip(objs, states):
        cmd.setComputePipelineState(state)
        cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*dims[:3]), metal.MTLSize(*dims[3:]))
        cmd.setBarrier() # the kernels of a batch run in order
      scalars = Buffer(self.device, blob_size, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
      for ci, idx, buf, off in bindings:
        objs[ci].setKernelBuffer_offset_atIndex(metal.MTLBuffer((buf if buf is not None else scalars).get_buf(self.device)), off, idx)
      words += [0, icb.value, scalars._buf, scalars.host.addr, *[c.value for c in objs]]
      refs.append((icb, objs, scalars))
    self.icbs[buf:=host_buf(*words)] = tuple(refs)
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if not self.event.waitUntilSignaledValue_timeoutMS(value, timeout or int(self.wait_timeout_ms)):
      raise RuntimeError(f"{self.device} signal wait timed out")

  def synchronize(self, timeout:int|None=None):
    if "timeline" not in self.__dict__: return
    for buf in list(self.profile_slots):
      slots = buf.host.view(fmt='Q')
      for i in range(5, buf.size, 4): # two timeline slots, then a start/end pair per kernel
        if slots[i] and not slots[i + 2]:
          (cb:=metal.MTLCommandBuffer(slots[i])).waitUntilCompleted()
          slots[i], slots[i + 2] = int(cb.GPUStartTime() * 1e9), int(cb.GPUEndTime() * 1e9)
    super().synchronize(timeout)
    # release command buffers after collecting their timestamps
    if (pool:=getattr(pools, "pool", None)) is not None: objc.lib.objc_autoreleasePoolPop(pool)
    pools.pool = objc.lib.objc_autoreleasePoolPush()
