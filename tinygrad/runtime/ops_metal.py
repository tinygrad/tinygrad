from __future__ import annotations
import subprocess, pathlib, struct, ctypes, tempfile, functools, platform, weakref, threading
from tinygrad.helpers import to_mv, round_up, cache_dir, unwrap, prod, to_tuple
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Buffer, BufferSpec, Compiler, CompileError
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.runtime.support.hcq2 import HCQ2Compiled, HCQAllocator, HWQueue, EncodeCtx, encode_submit, ccall, patch, unwrap_view, timeline_value, \
  host_buf
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
DLL("CoreGraphics", "CoreGraphics")

# FIXME: these need autogen to support objc categories
# https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjectiveC/Chapters/ocCategories.html
@functools.cache
def to_ns_str(s: str): return ctypes.cast(objc.msg("stringWithUTF8String:")(metal.NSString._objc_class_, s.encode()), metal.NSString).own()
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

STRTAB = ("commandBuffer", "computeCommandEncoder", "executeCommandsInBuffer:withRange:", "updateFence:", "endEncoding", "commit",
  "setKernelBuffer:offset:atIndex:", "concurrentDispatchThreadgroups:threadsPerThreadgroup:", "encodeWaitForEvent:value:", "encodeSignalEvent:value:",
  "blitCommandEncoder", "waitForFence:", "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:")
strtab_buf = functools.cache(lambda: host_buf(*[unwrap(objc.getsel(s.encode()).value) for s in STRTAB]))

class MetalQueue(HWQueue):
  q_rewrite = PatternMatcher([
    (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), lambda ctx, call, prg: ctx.exec(call, prg)),
    (UPat(Ops.INS, arg=("wait", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val)),
    (UPat(Ops.INS, arg=("store", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.signal(dst, val)),
  ])

  def __init__(self, ctx:EncodeCtx, submit:UOp):
    super().__init__(ctx, submit)
    calls = tuple(u.src[0].call() for u in self.lin.src if u.op is Ops.CALL)
    self.icb = UOp.placeholder((1 + len(calls),), dtypes.uint64, device=self.devs, tag="icb").after(UOp(Ops.LINEAR, src=calls))
    self.args = UOp.placeholder((8,), dtypes.uint8, device=self.devs) # stands in for the blob's buffer until submit
    handles = UOp.placeholder((2,), dtypes.uint64, device=self.devs, volatile=True, tag="mtl_handles")
    self.cb, self.enc, self.fence = handles[:1], handles[1:2], self.handle("fence").index(0).load()
    self.root = self.tail = handles.after(self.args)
    self.setups, self.count, self.done = list[UOp](), 0, 0 # writes to the indirect command buffer, kernels queued, kernels handed to an encoder
    self.words(0) # word 0: the blob's own mtlbuffer, patched at submit
    self.msg(self.handle("queue"), "commandBuffer", result=self.cb)

  def handle(self, name:str, devs:tuple[str, ...]|None=None) -> UOp: return UOp.placeholder((1,), dtypes.uint64, device=devs or self.devs, tag=name)
  def words(self, *ws:UOp|int) -> int: # append 64-bit words to the blob, returns the offset of the first
    return self.q(*[w.ccast(dtypes.uint64) if isinstance(w, UOp) else UOp.const(w, dtypes.uint64) for w in ws]) - 8 * len(ws)
  def ptr(self, off:int) -> UOp: return self.args.after(self.root).bitcast(dtypes.uint64).index(off // 8) # into the blob, after its patches
  def binding(self, buf:UOp) -> tuple[int, int]: # a buffer's address is its mtlbuffer: (the word it's put in, the view's offset)
    base, off = unwrap_view(buf)
    if base.op is Ops.MSELECT: # a lane of a multi view
      lane, lane_off = unwrap_view(base.src[0])
      base, off = lane.mselect(base.arg), off + lane_off
    return self.words(base.getaddr(self.devs)), off

  def call(self, after:UOp, target:UOp, name:str, *args:UOp|int, result:UOp|None=None) -> UOp: # one objc_msgSend, its return stored into result
    strtab = UOp.placeholder((len(STRTAB),), dtypes.uint64, 0, device=self.devs, tag="strtab")
    fn = metal.dll.bind(ctypes.c_void_p if result is not None else None)(metal.dll.objc_msgSend)
    ret = ccall(fn, target.after(after).index(0).load(), strtab.index(STRTAB.index(name)).load(),
                *[UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args])
    return result.after(after).index(0).store(ret) if result is not None else ret
  def msg(self, target:UOp, name:str, *args:UOp|int, result:UOp|None=None): # the command buffer is encoded in order: a chain
    self.tail = self.tail.after(self.call(self.tail, target, name, *args, result=result))
  def setup(self, cmd:UOp, name:str, *args:UOp|int): # the indirect command buffer is written wide, only before the commit: deep chains sort slowly
    self.setups.append(self.call(self.root, cmd, name, *args))

  def event(self, dst:UOp, val:UOp) -> tuple[UOp, UOp]: # a timeline is its device's shared event, any other signal the cross-queue one
    devs = to_tuple((base:=unwrap_view(dst)[0]).device)
    if base.tag == "timeline": return self.handle("timeline_event", devs).index(0).load(), val.ccast(dtypes.uint64)
    return self.handle("event", devs).index(0).load(), (timeline_value(devs) << 32) | val.ccast(dtypes.uint64)

  def exec(self, call:UOp, prg:UOp):
    if self.count == self.done: self.msg(self.cb, "computeCommandEncoder", result=self.enc)
    cmd, bufs, vals = self.icb[1 + self.count:2 + self.count], get_call_arg_uops(call), get_call_var_uops(call, prg)
    binds = [self.binding(bufs[i]) for i in prg.arg.globals] + [(0, self.words(v)) for v in vals] # a variable binds the blob at its word
    for i, (word, off) in enumerate(binds): self.setup(cmd, "setKernelBuffer:offset:atIndex:", self.ptr(word).load(), off, i)
    dims = (*prg.arg.global_size, *prg.arg.local_size)
    if any(isinstance(d, UOp) for d in dims): # arm64 passes MTLSize by reference
      self.setup(cmd, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", self.ptr(sizes:=self.words(*dims)), self.ptr(sizes + 24))
    self.count += 1

  def flush(self): # the open encoder runs the kernels queued since the last one as one range of the indirect command buffer
    if self.count > self.done:
      self.msg(self.enc, "executeCommandsInBuffer:withRange:", self.icb.index(0).load(), self.done, self.count - self.done)
      self.msg(self.enc, "updateFence:", self.fence)
      self.msg(self.enc, "endEncoding")
      self.done = self.count

  def wait(self, dst:UOp, val:UOp):
    self.flush()
    self.msg(self.cb, "encodeWaitForEvent:value:", *self.event(dst, val))

  def signal(self, dst:UOp, val:UOp):
    self.flush()
    base, dst_off = unwrap_view(dst)
    if base.tag == "timeline": # the host polls the timeline: a blit behind the fence copies the value in after the kernels
      src = self.words(val, base.getaddr(self.devs))
      self.msg(self.cb, "blitCommandEncoder", result=self.enc)
      self.msg(self.enc, "waitForFence:", self.fence)
      self.msg(self.enc, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", self.ptr(0).load(), src, self.ptr(src+8).load(), dst_off, 8)
      self.msg(self.enc, "endEncoding")
    self.msg(self.cb, "encodeSignalEvent:value:", *self.event(dst, val))

  def submit(self, cmdbuf:UOp) -> UOp:
    self.flush()
    self.tail = self.tail.after(*self.setups)
    self.msg(self.cb, "commit")
    buf = unwrap_view(cmdbuf)[0] # word 0: the blob's own mtlbuffer, a link patch on the bare placeholder
    return self.tail.substitute({self.args: cmdbuf.after(patch(buf, [(0, buf.getaddr(self.devs))]))})

# *****************
# device

class MetalAllocator(HCQAllocator['MetalDevice']):
  def __init__(self, dev:MetalDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)

  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    mtl = metal.MTLBuffer(options.external_ptr) if options.external_ptr else \
          self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    if mtl.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    setattr(mtl, "retain", False) # released when freed, not when collected
    self.dev.resident(mtl)
    view = MMIOInterface(c, size, fmt='B') if (c:=mtl.contents()) else None # an external buffer may have no host side
    return HCQBuffer(unwrap(mtl.value), size, meta=mtl, view=view, owner=self.dev)

  def _offset(self, buf:HCQBuffer, size:int, offset:int) -> HCQBuffer:
    view = buf.view.view(offset, size) if buf.view is not None else None
    return HCQBuffer(buf.va_addr, size, meta=buf.meta, _base=buf.base, view=view, owner=buf.owner)

  def _free(self, buf:HCQBuffer, options:BufferSpec|None=None):
    super()._free(buf, options)
    if options is not None and options.external_ptr is not None: self.dev.resident(buf.meta, False) # hcq2 skips externals

  def _do_free(self, buf:HCQBuffer, options:BufferSpec):
    self.dev.resident(buf.meta, False)
    buf.meta.release()

  def _do_map(self, buf:HCQBuffer) -> HCQBuffer: # every metal device is the same gpu: a mapping only adds to this device's residency
    assert isinstance(buf.owner, MetalDevice), "metal maps only metal buffers"
    self.dev.resident(buf.meta)
    return buf
  def _do_unmap(self, mb:HCQBuffer): self.dev.resident(mb.meta, False)

class MetalDevice(HCQ2Compiled):
  has_copy_queue = False
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_metal_compute", name="submit"), lambda ctx, submit: encode_submit(MetalQueue(ctx, submit))),
  ])

  def __init__(self, device:str=""):
    if int(platform.mac_ver()[0].split('.')[0]) < 15: raise RuntimeError("METAL needs macOS 15 for residency sets")
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()
    self.queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    if self.queue.value is None: raise RuntimeError("Cannot allocate a new command queue")

    # the buffers of an indirect command buffer must be resident: everything the device allocates is
    self.residency = checked(self.sysdevice.newResidencySetWithDescriptor_error, metal.MTLResidencySetDescriptor.new())
    self.queue.addResidencySet(self.residency)
    self.event, self.timeline_event, self.fence = self.sysdevice.newSharedEvent(), self.sysdevice.newSharedEvent(), self.sysdevice.newFence()
    self.icbs:weakref.WeakKeyDictionary[Buffer, tuple] = weakref.WeakKeyDictionary()

    # https://developer.apple.com/documentation/metal/mtlgpufamily
    def check_family(f): return next(filter(self.sysdevice.supportsFamily, reversed([v for v, nm in metal.enum_MTLGPUFamily.items() if f in nm])), 0)
    super().__init__(device, MetalAllocator(self), [MetalRenderer], None,
                     arch=metal.enum_MTLGPUFamily[check_family("Apple") or check_family("Mac")][12:])
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="strtab"), lambda: strtab_buf()),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.handle(b.tag) if b.tag in ("queue", "event", "timeline_event", "fence") else None),
      (UPat(Ops.PARAM, tag="cmdbuf_compute_0", name="b"), # the queue binds the blob by its mtlbuffer: it can't be a view of the pool
       lambda ctx, b: Buffer(ctx.device, b.max_numel(), b.dtype, options=BufferSpec(nolru=True), preallocate=True)),
      (UPat(Ops.PARAM, tag="icb").after(UPat(Ops.LINEAR, name="linear")), lambda ctx, linear: ctx.new_icb(linear)),
    ]) + self.pm_bufferize

  @functools.cache
  def handle(self, name:str) -> Buffer: return host_buf(unwrap(getattr(self, name).value))

  def new_icb(self, linear:UOp) -> Buffer: # a command per call with its pipeline and static sizes, the queue binds the rest
    descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    descriptor.setMaxKernelBufferBindCount(31)
    icb = self.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(descriptor, max(len(linear.src), 1), 0)
    if icb.value is None: raise RuntimeError("create indirect command buffer failed, does your system support this?")
    cmds = [icb.indirectComputeCommandAtIndex(k).own() for k in range(len(linear.src))]
    for cmd, call in zip(cmds, linear.src):
      prg = call.src[0]
      state = self.pipeline(prg.src[3].arg, prg.arg.function_name)
      if prod(prg.arg.local_size) > (mx:=state.maxTotalThreadsPerThreadgroup()):
        raise RuntimeError(f"local size {prg.arg.local_size} bigger than {mx}")
      cmd.setComputePipelineState(state)
      dims = tuple(1 if isinstance(d, UOp) else int(d) for d in (*prg.arg.global_size, *prg.arg.local_size))
      cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*dims[:3]), metal.MTLSize(*dims[3:]))
      cmd.setBarrier()
    self.icbs[buf:=host_buf(icb.value, *[cmd.value for cmd in cmds])] = (icb, cmds) # the objects live as long as their words
    return buf

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(lib, len(lib), None, None))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(library.newFunctionWithName(to_ns_str(name)))
    descriptor.setSupportIndirectCommandBuffers(True)
    return checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor, metal.MTLPipelineOptionNone, None)

  def resident(self, mtl:metal.MTLBuffer, add:bool=True):
    (self.residency.addAllocation if add else self.residency.removeAllocation)(ctypes.cast(mtl, metal.MTLAllocation))
    self.residency.commit()

  def synchronize(self, timeout:int|None=None):
    super().synchronize(timeout)
    # the gpu is done with every command buffer: drain them. a nested synchronize (a free during collection) finds no pool to pop
    if (pool:=getattr(pools, "pool", None)) is not None: objc.lib.objc_autoreleasePoolPop(pool)
    pools.pool = objc.lib.objc_autoreleasePoolPush()
