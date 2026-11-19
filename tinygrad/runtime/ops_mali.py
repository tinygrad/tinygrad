from __future__ import annotations
import base64, ctypes, os, struct
from tinygrad.device import Compiled, BufferStorage, BufferSpec, HostAllocator, Program
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import suppress_finalizing, is_image_shape, from_mv, to_tuple
from tinygrad.runtime.support import c
from tinygrad.renderer import TinyELF
from tinygrad.renderer.nir import MALIRenderer
from tinygrad.runtime.autogen import panfrost as pf, mesa
from tinygrad.runtime.support.hcq2 import HWQueue, HCQ_RUNTIME_DEV, ccall, layout_args, pack_args, encode_cmdbuf
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, uopfunc
from tinygrad.engine.realize import get_call_kernel_args


def check(value, msg="MALI"):
  if value is None or (isinstance(value, int) and value == 0): raise RuntimeError(f"{msg} failed")
  return value

class MALIBuffer(int):
  resource: c.POINTER[pf.struct_pipe_resource]
  offset: int
  def __new__(cls, resource:c.POINTER[pf.struct_pipe_resource], offset:int=0) -> MALIBuffer:
    ret = int.__new__(cls, pf.panfrost_resource_address(resource) + offset)
    ret.resource, ret.offset = resource, offset
    return ret

class MALIAllocator(HostAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.host or options.cpu_access: return super()._alloc(size, options)
    tmpl = pf.struct_pipe_resource(target=pf.PIPE_BUFFER, format=pf.PIPE_FORMAT_R8_UINT, width0=size, height0=1,
      depth0=1, array_size=1, nr_samples=1, nr_storage_samples=1,
      bind=pf.PIPE_BIND_GLOBAL | pf.PIPE_BIND_SHADER_BUFFER | pf.PIPE_BIND_SAMPLER_VIEW | pf.PIPE_BIND_SHADER_IMAGE | pf.PIPE_BIND_LINEAR,
      usage=pf.PIPE_USAGE_DEFAULT)
    return BufferStorage(MALIBuffer(check(self.dev.screen.contents.resource_create(self.dev.screen, ctypes.byref(tmpl)), "MALI buffer allocation")))

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None:
      self.dev.synchronize()
      self.dev.screen.contents.resource_destroy(self.dev.screen, storage.buf.resource)

  def _copyin(self, dest, src:memoryview): self.dev.buffer_write(dest, src)
  def _copyout(self, dest:memoryview, src): self.dev.buffer_read(src, dest)
  def _offset(self, buf, size:int, offset:int): return MALIBuffer(buf.resource, offset) if isinstance(buf, MALIBuffer) else buf + offset

class MALIProgram(Program['MALIDevice']):
  def __init__(self, dev:MALIDevice, obj:TinyELF, links:dict[str,MALIProgram]):
    self.dev, self.name, self.signature = dev, obj.name, obj.signature
    self.kernel:int|None = None
    self.vendor_fau:list[int] = []
    src = base64.b64decode(obj.lib)
    self.args: list[tuple[int, DType, bool]] = []
    cursor = 0
    for i, (off, dt) in enumerate(TinyELF.iter_sig(obj.signature)):
      self.args.append((off, dt, obj.signature[i][1] is not AddrSpace.ALU))
      cursor = off + dt.itemsize
    self.arg_size = cursor
    self.buffer_slots = [i for i, (_, addr, _, _) in enumerate(obj.signature) if addr is not AddrSpace.ALU]
    self.image_info = [((shape[1], shape[0], dt.itemsize, 1) if is_image_shape(shape) else (0, 0, 0, 0))
                       for _, addr, dt, shape in obj.signature if addr is not AddrSpace.ALU]
    if src[:4] == b'MLV1':
      shared_size, self.num_samplers, self.num_images, barrier = struct.unpack_from('<IIII', src, 4)
      self.kernel = self.compile_vendor(src[20:], shared_size, bool(barrier))
    else:
      self.vendor_fau = []
      reader = mesa.struct_blob_reader()
      mesa.blob_reader_init(reader, src, len(src))
      nir = check(mesa.nir_deserialize(None, pf.pan_shader_get_compiler_options(10), reader), f"MALI deserialize {obj.name}")
      state = pf.struct_pipe_compute_state(ir_type=pf.PIPE_SHADER_IR_NIR, prog=ctypes.cast(nir, ctypes.c_void_p),
        static_shared_mem=nir.contents.info.shared_size)
      self.kernel = check(dev.ctx.contents.create_compute_state(dev.ctx, ctypes.byref(state)), f"MALI compile {obj.name}")
      self.num_samplers, self.num_images = nir.contents.info.num_textures, nir.contents.info.num_images

  def compile_vendor(self, src, shared_size, barrier):
    arg_size = self.arg_size
    ebin, code_off = src.find(b'EBIN'), src.find(b'OBJC')
    if ebin < 0 or code_off < 0: raise RuntimeError('invalid vendor MBS2')
    preload = struct.unpack_from('<H', src, ebin + 8 + 4)[0] << 48
    fau_count = struct.unpack_from('<H', src, ebin + 8 + 14)[0]
    tls_size = struct.unpack_from('<I', src, ebin + 8 + 16)[0]
    flags = struct.unpack_from('<I', src, ebin + 8 + 20)[0]
    fcst, pos = bytearray(), 0
    while (pos := src.find(b'FCST', pos, ebin)) >= 0:
      chunk_size = struct.unpack_from('<I', src, pos + 4)[0]
      size, offset = struct.unpack_from('<HH', src, pos + 8)
      assert chunk_size == 4 + ((size + 3) & ~3), (chunk_size, size, offset)
      fcst.extend(bytes(offset + size - len(fcst)))
      fcst[offset:offset + size] = src[pos + 12:pos + 12 + size]
      pos += chunk_size + 8
    fcst.extend(bytes(-len(fcst) % 4))
    self.vendor_fau = [0 if value[0] == 0x04000000 else value[0] for value in struct.iter_unpack('<I', fcst)]
    self.vendor_fau.extend([0] * (fau_count * 2 - len(self.vendor_fau)))
    code_size = struct.unpack_from('<I', src, code_off + 4)[0]
    self.code_buf = ctypes.create_string_buffer(src[code_off + 8:code_off + 8 + code_size])
    native = pf.struct_panfrost_native_shader()
    native.info.stage, native.info.work_reg_count = 5, 64
    native.info.wls_size, native.info.tls_size, native.info.preload = shared_size, tls_size, preload
    native.info.allow_merging_workgroups = False
    native.info.contains_barrier = barrier or shared_size > 0
    native.info.writes_global = bool(flags & 0x40000)
    native.info.ubo_count, native.info.ubo_mask, native.info.push_count = 1, 1, len(self.vendor_fau)
    for i in range(len(self.vendor_fau)): native.info.push_words[i].offset = arg_size + i * 4
    native.binary, native.binary_size = ctypes.cast(self.code_buf, ctypes.c_void_p), code_size
    state = pf.struct_pipe_compute_state(ir_type=pf.PIPE_SHADER_IR_NATIVE, prog=ctypes.c_void_p(ctypes.addressof(native)),
      static_shared_mem=shared_size)
    return check(self.dev.ctx.contents.create_compute_state(self.dev.ctx, ctypes.byref(state)), 'MALI vendor compile')

  def release(self):
    if self.kernel is not None:
      self.dev.synchronize()
      self.dev.ctx.contents.delete_compute_state(self.dev.ctx, self.kernel)
      self.kernel = None

  @suppress_finalizing
  def __del__(self): self.release()

@uopfunc
def mali_run(buf:UOp, cmds:tuple, ctx:UOp) -> UOp:
  for fn, args in cmds:
    out_off = args[-1][0] if fn.__name__ == 'panfrost_compute_create_sampler_view' else 0
    args = [buf.index(a[0]) if isinstance(a, tuple) else buf.bitcast(dtypes.u64).index(a).load() if isinstance(a, int) else a for a in args]
    if fn.__name__ == 'do_get_time_ms':
      buf = buf.after(fn(*args))
      continue
    ret = ccall(fn, ctx, *args)
    if fn.__name__ == 'panfrost_compute_create_sampler_view':
      buf = buf.after(buf.bitcast(dtypes.u64).index(out_off // 8).store(ret))
    else:
      ctx = ret
      buf = buf.after(ret)
  return buf.sink()

class MALIQueue(HWQueue):
  dev:MALIDevice
  def __init__(self, submit:UOp):
    super().__init__(submit)
    self.cmds:list[tuple] = []
    self.ctx = UOp.const(ctypes.addressof(self.dev.ctx.contents), dtypes.u64)

  def call(self, fn, *args): self.cmds.append((fn, args))
  def copy(self, dst:UOp, src:UOp, n:int):
    dst_mali, src_mali = to_tuple(dst.device)[0] == self.devs[0], to_tuple(src.device)[0] == self.devs[0]
    if dst_mali and src_mali:
      self.call(pf.panfrost_buffer_copy, dst.getaddr(self.devs), src.getaddr(self.devs), UOp.const(n, dtypes.u32))
    elif dst_mali:
      self.call(pf.panfrost_buffer_write, dst.getaddr(self.devs), src.getaddr(), UOp.const(n, dtypes.u32))
    elif src_mali:
      self.call(pf.panfrost_buffer_read, src.getaddr(self.devs), dst.getaddr(), UOp.const(n, dtypes.u32))

  def wait(self, dst:UOp, val:UOp|int, eq=False):
    self.call(pf.panfrost_compute_wait, dst.getaddr(HCQ_RUNTIME_DEV.device), UOp.const(val, dtypes.u64) if isinstance(val, int) else val)
  def memory_barrier(self): pass
  def exec_barrier(self): pass
  def signal(self, dst:UOp, val:UOp|int):
    self.call(pf.panfrost_compute_signal, dst.getaddr(HCQ_RUNTIME_DEV.device), UOp.const(val, dtypes.u64) if isinstance(val, int) else val)
  def timestamp(self, dst:UOp): self.call(pf.panfrost_compute_timestamp, dst.getaddr(HCQ_RUNTIME_DEV.device))

  def exec(self, call:UOp, prg:UOp):
    obj = TinyELF.from_program(prg)
    data = self.dev.program(obj)
    args = get_call_kernel_args(call, prg, self.devs)
    total_args = data.arg_size + 4 * len(data.vendor_fau)
    rows = layout_args(args) + [(data.arg_size + 4 * i, UOp.const(v, dtypes.u32)) for i, v in enumerate(data.vendor_fau)]
    arg_off = self.q(*pack_args(rows, total_args)) - total_args
    cb = pf.struct_pipe_constant_buffer(buffer_size=total_args)
    cb_off = self.q(UOp(Ops.BINARY, arg=bytes(cb))) - ctypes.sizeof(cb)
    grid = pf.struct_pipe_grid_info(work_dim=3)
    grid_off = self.q(UOp(Ops.BINARY, arg=bytes(grid))) - ctypes.sizeof(grid)
    self.patches += [(grid_off + off, w) for off, w in layout_args([UOp.const(x, dtypes.u32) for x in prg.arg.local_size], 8)]
    self.patches += [(grid_off + off, w) for off, w in layout_args([UOp.const(x, dtypes.u32) for x in prg.arg.global_size], 32)]

    self.call(pf.panfrost_compute_bind, UOp.const(check(data.kernel), dtypes.u64))
    self.call(pf.panfrost_compute_set_constants, (cb_off,), (arg_off,))

    sampler_offsets:list[int] = []
    image_addresses:list[UOp] = []
    image_blob = bytearray()
    handles_off = self.q(UOp(Ops.BINARY, arg=bytes(8 * data.num_samplers))) - 8 * data.num_samplers if data.num_samplers else 0
    for i, (width, height, itemsize, _) in enumerate(data.image_info):
      if not itemsize: continue
      resource = call.src[1 + data.buffer_slots[i]].getaddr(self.devs)
      fmt = pf.PIPE_FORMAT_R16G16B16A16_FLOAT if itemsize == 2 else pf.PIPE_FORMAT_R32G32B32A32_FLOAT
      tex2d = pf.struct_pipe_tex2d_from_buf(offset=0, row_stride=width, width=width, height=height)
      if len(sampler_offsets) < data.num_samplers:
        tmpl = pf.struct_pipe_sampler_view(format=fmt, target=pf.PIPE_TEXTURE_2D, is_tex2d_from_buf=True,
          swizzle_r=0, swizzle_g=1, swizzle_b=2, swizzle_a=3, texture=None, context=self.dev.ctx, u=tex2d)
        tmpl_off = self.q(UOp(Ops.BINARY, arg=bytes(tmpl))) - ctypes.sizeof(tmpl)
        handle_off = handles_off + 8 * len(sampler_offsets)
        self.call(pf.panfrost_compute_create_sampler_view, resource, (tmpl_off,), (handle_off,))
        sampler_offsets.append(handle_off)
      else:
        view = pf.struct_pipe_image_view(resource=None, format=fmt,
          access=pf.PIPE_IMAGE_ACCESS_READ_WRITE | pf.PIPE_IMAGE_ACCESS_TEX2D_FROM_BUFFER,
          shader_access=pf.PIPE_IMAGE_ACCESS_READ_WRITE, u=tex2d)
        image_addresses.append(resource)
        image_blob += bytes(view)

    if sampler_offsets:
      self.call(pf.panfrost_compute_set_samplers, UOp.const(self.dev.sampler_state, dtypes.u64), (handles_off,),
        UOp.const(len(sampler_offsets), dtypes.u32))
    if image_blob:
      assert len(image_addresses) <= 4
      image_off = self.q(UOp(Ops.BINARY, arg=bytes(image_blob))) - len(image_blob)
      self.call(pf.panfrost_compute_set_images, UOp.const(len(image_addresses), dtypes.u32), (image_off,),
        *[*image_addresses, *([UOp.const(0, dtypes.u64)] * (4 - len(image_addresses)))])

    self.call(pf.panfrost_compute_launch, (grid_off,))
    for handle_off in sampler_offsets: self.call(pf.panfrost_compute_release_sampler_view, handle_off)

  def encode(self) -> UOp: return mali_run(encode_cmdbuf(self, self.lin), tuple(self.cmds), self.ctx)

Compiled.pm_encode += PatternMatcher([(UPat.sink(arg=KernelInfo("mali_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"),
  lambda s: MALIQueue(s).encode()), (UPat.sink(arg=KernelInfo("mali_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"),
  lambda s: MALIQueue(s).encode())])

class MALIDevice(Compiled):
  renderers = [MALIRenderer]

  def __init__(self, device:str=""):
    self.fd = os.open("/dev/dri/renderD128", os.O_RDWR | os.O_CLOEXEC)
    self.option_info, self.option_cache = pf.struct_driOptionCache(), pf.struct_driOptionCache()
    specs = [
      (b"pan_afbc_tiled", pf.DRI_BOOL, True, 0, 0),
      (b"pan_force_afbc_packing", pf.DRI_BOOL, False, 0, 0),
      (b"pan_afbcp_reads_threshold", pf.DRI_INT, 10, 1, 255),
      (b"pan_afbcp_gpu_payload_sizes", pf.DRI_BOOL, False, 0, 0),
      (b"pan_csf_chunk_size", pf.DRI_INT, 2 * 1024 * 1024, 256 * 1024, 8 * 1024 * 1024),
      (b"pan_csf_initial_chunks", pf.DRI_INT, 5, 1, 65535),
      (b"pan_csf_max_chunks", pf.DRI_INT, 64, 1, 65535),
      (b"pan_compute_core_mask", pf.DRI_UINT64, (1 << 64) - 1, 0, 0),
      (b"pan_fragment_core_mask", pf.DRI_UINT64, (1 << 64) - 1, 0, 0),
      (b"pan_relax_afbc_yuv_imports", pf.DRI_BOOL, False, 0, 0),
    ]
    options = (pf.struct_driOptionDescription * (len(specs) + 2))()
    self.options, self._option_strings = options, []
    for i, spec in [(0, None), *enumerate(specs, start=1), (len(specs) + 1, None)]:
      options[i].info.type = pf.DRI_SECTION if spec is None else spec[1]
      if spec is None: continue
      name, typ, default, lo, hi = spec
      desc = ctypes.c_char_p(name)
      self._option_strings.append(desc)
      options[i].desc = ctypes.cast(desc, c.POINTER[ctypes.c_char])
      options[i].info.name = ctypes.cast(desc, c.POINTER[ctypes.c_char])
      if typ == pf.DRI_BOOL: options[i].value._bool = default
      elif typ == pf.DRI_INT: options[i].value._int = default
      else: options[i].value._uint64 = default
      if typ == pf.DRI_INT: options[i].info.range.start._int, options[i].info.range.end._int = lo, hi
    pf.driParseOptionInfo(self.option_info, options, len(options))
    self.config = pf.struct_pipe_screen_config(driver_name_is_inferred=False,
      options=ctypes.pointer(self.option_cache), options_info=ctypes.pointer(self.option_info))
    pf.glsl_type_singleton_init_or_ref()
    self.screen = check(pf.panfrost_drm_screen_create(self.fd, ctypes.byref(self.config)), "MALI panthor screen")
    flags = pf.PIPE_CONTEXT_COMPUTE_ONLY | pf.PIPE_CONTEXT_NO_LOD_BIAS
    self.ctx = check(self.screen.contents.context_create(self.screen, None, flags), "MALI compute context")
    ss = pf.struct_pipe_sampler_state(wrap_s=pf.PIPE_TEX_WRAP_CLAMP_TO_BORDER, wrap_t=pf.PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      wrap_r=pf.PIPE_TEX_WRAP_CLAMP_TO_BORDER, min_img_filter=pf.PIPE_TEX_FILTER_NEAREST,
      mag_img_filter=pf.PIPE_TEX_FILTER_NEAREST, unnormalized_coords=True)
    self.sampler_state = self.ctx.contents.create_sampler_state(self.ctx, ctypes.byref(ss))
    self.programs:dict[tuple, MALIProgram] = {}
    super().__init__(device, MALIAllocator(self), None, arch="g610")

  def program(self, obj:TinyELF) -> MALIProgram:
    key = (obj.lib, obj.name, obj.signature)
    if key not in self.programs: self.programs[key] = MALIProgram(self, obj, {})
    return self.programs[key]

  @property
  def has_copy_queue(self) -> bool: return True

  def buffer_write(self, dest:MALIBuffer, src:memoryview):
    box = pf.struct_pipe_box(x=dest.offset, y=0, z=0, width=len(src), height=1, depth=1)
    transfer = ctypes.c_void_p()
    ptr = check(self.ctx.contents.buffer_map(self.ctx, dest.resource, 0, pf.PIPE_MAP_WRITE, ctypes.byref(box), ctypes.byref(transfer)),
      "MALI buffer map write")
    ctypes.memmove(ptr, from_mv(src), src.nbytes)
    self.ctx.contents.buffer_unmap(self.ctx, transfer)

  def buffer_read(self, src:MALIBuffer, dest:memoryview):
    self.synchronize()
    box = pf.struct_pipe_box(x=src.offset, y=0, z=0, width=len(dest), height=1, depth=1)
    transfer = ctypes.c_void_p()
    ptr = check(self.ctx.contents.buffer_map(self.ctx, src.resource, 0, pf.PIPE_MAP_READ, ctypes.byref(box), ctypes.byref(transfer)),
      "MALI buffer map read")
    ctypes.memmove(from_mv(dest), ptr, dest.nbytes)
    self.ctx.contents.buffer_unmap(self.ctx, transfer)

  def flush(self):
    fence = ctypes.c_void_p()
    self.ctx.contents.flush(self.ctx, ctypes.byref(fence), 0)
    if fence.value:
      check(self.screen.contents.fence_finish(self.screen, None, fence, (1 << 64) - 1), "MALI fence wait")
      self.screen.contents.fence_reference(self.screen, ctypes.byref(fence), None)

  def _wait_signal(self, sig, value:int, timeout:int|None=None):
    self.flush()
    super()._wait_signal(sig, value, timeout)

  def finalize(self):
    super().finalize()
    for program in self.programs.values(): program.release()
    self.programs.clear()
    self.ctx.contents.delete_sampler_state(self.ctx, self.sampler_state)
    self.ctx.contents.destroy(self.ctx)
    self.screen.contents.destroy(self.screen)
    pf.driDestroyOptionCache(self.option_cache)
    pf.driDestroyOptionInfo(self.option_info)
    pf.glsl_type_singleton_decref()
    os.close(self.fd)
