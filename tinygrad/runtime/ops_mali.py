from __future__ import annotations
from typing import Any
import base64, ctypes, os, struct, time
from tinygrad.device import Compiled, BufferStorage, BufferSpec, HostAllocator, Program, MMIOInterface
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import suppress_finalizing, is_image_shape, from_mv, to_tuple, next_power2
from tinygrad.runtime.support import c
from tinygrad.renderer import TinyELF
from tinygrad.renderer.nir import MALIRenderer
from tinygrad.runtime.autogen import panfrost as pf, mesa, panthor as pt, libc
from tinygrad.runtime.support.hcq2 import HWQueue, HCQ_RUNTIME_DEV, ccall, layout_args, encode_cmdbuf, make_program, patch
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
    if options.host: return super()._alloc(size, options)
    tmpl = pf.struct_pipe_resource(target=pf.PIPE_BUFFER, format=pf.PIPE_FORMAT_R8_UINT, width0=size, height0=1,
      depth0=1, array_size=1, nr_samples=1, nr_storage_samples=1,
      bind=pf.PIPE_BIND_GLOBAL | pf.PIPE_BIND_SHADER_BUFFER | pf.PIPE_BIND_SAMPLER_VIEW | pf.PIPE_BIND_SHADER_IMAGE | pf.PIPE_BIND_LINEAR,
      usage=pf.PIPE_USAGE_DEFAULT)
    buf = MALIBuffer(check(self.dev.screen.contents.resource_create(self.dev.screen, ctypes.byref(tmpl)), "MALI buffer allocation"))
    box = pf.struct_pipe_box(x=0, y=0, z=0, width=size, height=1, depth=1)
    transfer = ctypes.c_void_p()
    ptr = check(self.dev.ctx.contents.buffer_map(self.dev.ctx, buf.resource, 0, pf.PIPE_MAP_READ | pf.PIPE_MAP_WRITE,
      ctypes.byref(box), ctypes.byref(transfer)), "MALI buffer map")
    return BufferStorage(buf, transfer, MMIOInterface(ptr, size) if options.cpu_access else None)

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None:
      self.dev.synchronize()
      self.dev.ctx.contents.buffer_unmap(self.dev.ctx, storage.meta)
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
    self.info = pf.struct_pan_shader_info()
    pf.panfrost_compute_shader_info(self.kernel, ctypes.byref(self.info))
    self.shader_desc = pf.panfrost_compute_shader_desc(self.kernel)

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

def mali_buffer(address, size): return struct.pack('<IIQIIII', 9 | (1 << 4), size, address, 0, 0, 0, 0)

# Direct Valhall CSF command queue.
class MALIRawQueue(HWQueue):
  dev:MALIDevice
  def __init__(self, submit:UOp):
    super().__init__(submit)
    self.cmds:list[tuple] = []
    self.ctx = UOp.const(ctypes.addressof(self.dev.ctx.contents), dtypes.u64)

  def move32(self, reg:int, value:Any):
    value = UOp.const(value, dtypes.u64) if isinstance(value, int) else value
    self.q(UOp(Ops.ADD, src=(UOp.const((2 << 56) | (reg << 48), dtypes.u64), value)))
  def move48(self, reg:int, value:Any):
    value = UOp.const(value, dtypes.u64) if isinstance(value, int) else value
    self.q(UOp(Ops.ADD, src=(UOp.const((1 << 56) | (reg << 48), dtypes.u64), value)))
  def wait(self, dst:UOp, val:Any, eq=False):
    self.cmds.append((pf.panfrost_compute_wait, (dst.getaddr(HCQ_RUNTIME_DEV.device), UOp.const(val, dtypes.u64) if isinstance(val, int) else val)))
  def memory_barrier(self): pass
  def exec_barrier(self): self.q(UOp.const((3 << 56) | (0xff << 16), dtypes.u64))
  def signal(self, dst:UOp, val:Any):
    self.cmds.append((pf.panfrost_compute_signal, (dst.getaddr(HCQ_RUNTIME_DEV.device), UOp.const(val, dtypes.u64) if isinstance(val, int) else val)))
  def timestamp(self, dst:UOp): self.cmds.append((pf.panfrost_compute_timestamp, (dst.getaddr(HCQ_RUNTIME_DEV.device),)))
  def copy(self, dst:UOp, src:UOp, n:int):
    dst_mali, src_mali = to_tuple(dst.device)[0] == self.devs[0], to_tuple(src.device)[0] == self.devs[0]
    if dst_mali and src_mali: self.cmds.append((pf.panfrost_buffer_copy, (dst.getaddr(self.devs), src.getaddr(self.devs), UOp.const(n, dtypes.u32))))
    elif dst_mali: self.cmds.append((pf.panfrost_buffer_write, (dst.getaddr(self.devs), src.getaddr(), UOp.const(n, dtypes.u32))))
    elif src_mali: self.cmds.append((pf.panfrost_buffer_read, (src.getaddr(self.devs), dst.getaddr(), UOp.const(n, dtypes.u32))))

  def write(self, address:UOp, value:UOp, size:int):
    self.cmds.append((pt.panthor_write, (address, value, UOp.const(size, dtypes.u32))))

  def exec(self, call:UOp, prg:UOp):
    obj = TinyELF.from_program(prg)
    data = self.dev.program(obj)
    args = get_call_kernel_args(call, prg, self.devs)
    def alloc(size:int, tag:str): return make_program(prg, size, self.devs[0], tag, call.key)
    total_args = data.arg_size + 4 * len(data.vendor_fau)
    rows = layout_args(args) + [(data.arg_size + 4 * i, UOp.const(v, dtypes.u32)) for i, v in enumerate(data.vendor_fau)]
    argbuf = patch(alloc(total_args, 'kernargs'), [], bytes(total_args))
    for off, word in rows: self.write(argbuf.getaddr(self.devs) + off, word, word.dtype.itemsize)
    ubodesc = alloc(32, 'ubo')
    ubodesc = patch(ubodesc, [(8, argbuf.getaddr(self.devs))], mali_buffer(0, (data.arg_size + 15) & ~15))
    sampler = (ctypes.c_ubyte * 32)()
    pf.panfrost_sampler_desc(self.dev.sampler_state, sampler)
    sampler_desc = alloc(32, 'sampler')
    sampler_desc = patch(sampler_desc, [], bytes(sampler))
    textures:list[tuple] = []
    images:list[tuple] = []
    for i, (width, height, itemsize, _) in enumerate(data.image_info):
      if not itemsize: continue
      dummy = self.dev.allocator.alloc(16, BufferSpec())
      fmt = pf.PIPE_FORMAT_R16G16B16A16_FLOAT if itemsize == 2 else pf.PIPE_FORMAT_R32G32B32A32_FLOAT
      tex2d = pf.struct_pipe_tex2d_from_buf(offset=0, row_stride=width, width=width, height=height)
      tmpl = pf.struct_pipe_sampler_view(format=fmt, target=pf.PIPE_TEXTURE_2D, is_tex2d_from_buf=True,
        swizzle_r=0, swizzle_g=1, swizzle_b=2, swizzle_a=3, texture=dummy.buf.resource, context=self.dev.ctx, u=tex2d)
      out = (ctypes.c_ubyte * 32)()
      pf.panfrost_make_texture_desc(self.dev.ctx, ctypes.byref(tmpl), out)
      resource = call.src[1 + data.buffer_slots[i]].getaddr(self.devs)
      (textures if len(textures) < data.num_samplers else images).append((out, resource))
      self.dev.allocator.free(dummy, 16, BufferSpec())
    resources = alloc(128, 'resources')
    resource_rows = [(0, ubodesc.getaddr(self.devs) | (1 << 56)), (8, UOp.const(32, dtypes.u64)),
                     (48, sampler_desc.getaddr(self.devs) | (1 << 56)), (56, UOp.const(32, dtypes.u64))]
    if textures:
      texture_desc = alloc(32 * len(textures), 'texture')
      texture_desc = patch(texture_desc, [], b''.join(map(bytes, textures)))
      for i, (_, resource) in enumerate(textures): self.write(texture_desc.getaddr(self.devs) + 16 + 32 * i, resource, 8)
      resource_rows += [(64, texture_desc.getaddr(self.devs) | (1 << 56)), (72, UOp.const(32 * len(textures), dtypes.u64))]
    if images:
      image_desc = alloc(32 * len(images), 'image')
      image_desc = patch(image_desc, [], b''.join(map(bytes, images)))
      for i, (_, resource) in enumerate(images): self.write(image_desc.getaddr(self.devs) + 16 + 32 * i, resource, 8)
      resource_rows += [(80, image_desc.getaddr(self.devs) | (1 << 56)), (88, UOp.const(32 * len(images), dtypes.u64))]
    resources = patch(resources, resource_rows, bytes(128))
    word_args = {}
    for off, word in rows:
      word_args[off] = word.cast(dtypes.u32)
      if word.dtype.itemsize == 8: word_args[off + 4] = (word >> 32).cast(dtypes.u32)
    fau = alloc(4 * data.info.push_count, 'fau')
    fau = patch(fau, [], bytes(4 * data.info.push_count))
    for i in range(data.info.push_count): self.write(fau.getaddr(self.devs) + 4 * i, word_args[data.info.push_words[i].offset], 4)

    if not getattr(self, 'resources_requested', False):
      self.q(UOp.const((34 << 56) | 0xf, dtypes.u64)) # REQ_RESOURCE: compute, fragment, tiler and IDVS
      self.resources_requested = True
    self.q(UOp.const((23 << 56) | 2, dtypes.u64)) # SET_SB_ENTRY endpoint=2, other=0
    self.move48(0, resources.getaddr(self.devs) | 8)
    fau_addr = fau.getaddr(self.devs) | ((data.info.push_count // 2) << 56)
    self.move32(8, fau_addr & 0xffffffff)
    self.move32(9, fau_addr >> 32)
    self.move48(16, data.shader_desc)
    local_size, global_size = prg.arg.local_size, prg.arg.global_size
    grid = pf.struct_pipe_grid_info(work_dim=3)
    grid.block[:], grid.grid[:] = local_size, global_size
    tls_info = (ctypes.c_uint32 * 5)()
    pf.panfrost_compute_tls_info(self.dev.ctx, data.kernel, ctypes.byref(grid), tls_info)
    tls_size, wls_size, instances, wls_total, tls_total = tls_info
    tls = alloc(tls_total, 'tls') if tls_total else None
    wls = alloc(wls_total, 'wls') if wls_total else None
    tls_shift = ((tls_size + 15) // 16 - 1).bit_length() if tls_size else 0
    wls_word = (instances.bit_length() - 1) | (next_power2(max(wls_size, 128)).bit_length() << 8) if wls_size else 31
    tsd = alloc(32, 'tsd')
    tsd_rows = ([(8, (tls.getaddr(self.devs) >> 8) | (1 << 60))] if tls is not None else [])
    tsd_rows += [(16, wls.getaddr(self.devs))] if wls is not None else []
    tsd = patch(tsd, tsd_rows, struct.pack('<IIQIIII', tls_shift, wls_word, 0, 0, 0, 0, 0))
    self.move48(24, tsd.getaddr(self.devs))
    self.move32(32, 0)
    wg = (local_size[0] - 1) | ((local_size[1] - 1) << 10) | ((local_size[2] - 1) << 20)
    if data.info.allow_merging_workgroups: wg |= 1 << 31
    self.move32(33, wg)
    for reg in (34, 35, 36): self.move32(reg, 0)
    for i, value in enumerate(global_size): self.move32(37 + i, value)

    threads = local_size[0] * local_size[1] * local_size[2]
    max_threads = pf.panfrost_max_threads(self.dev.screen, data.info.work_reg_count)
    axis, task_threads, increment = 0, threads, 0
    for i in range(3):
      if task_threads * global_size[i] >= max_threads:
        increment = max_threads // task_threads
        break
      if axis == 2:
        increment = global_size[i]
        break
      task_threads *= global_size[i]
      axis += 1
    self.q(UOp.const((4 << 56) | increment | (axis << 14), dtypes.u64)) # RUN_COMPUTE
    self.q(UOp.const((3 << 56) | (0xff << 16), dtypes.u64))
    self.move32(74, 0)
    self.q(UOp.const((36 << 56) | 1 | (1 << 4) | (2 << 8) | (74 << 40), dtypes.u64)) # clean L2/LSC, invalidate other caches
    self.q(UOp.const((3 << 56) | (1 << 16), dtypes.u64))

  def submit(self, cmdbuf:UOp) -> UOp:
    fd = UOp.const(self.dev.fd, dtypes.int32)
    return mali_run(cmdbuf, (*self.cmds, (pt.panthor_group_submit, (fd, UOp.const(self.dev.group_handle, dtypes.u32), cmdbuf.getaddr(self.devs),
      UOp.const(cmdbuf.max_numel(), dtypes.u32), UOp.const(self.dev.sync_handle, dtypes.u32)))), self.ctx)

  def encode(self) -> UOp:
    for u in self.lin.src: self.q_rewrite.rewrite(u, ctx=self)
    empty = not self.blob
    if empty: self.q(UOp.const(0, dtypes.u64))
    cmdbuf = encode_cmdbuf(self)
    if empty: return mali_run(cmdbuf, tuple(self.cmds), self.ctx) # host-only queues need no panthor submission
    return self.submit(cmdbuf)

def mali_encode(s:UOp) -> UOp: return MALIRawQueue(s).encode()
Compiled.pm_encode += PatternMatcher([(UPat.sink(arg=KernelInfo("mali_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"),
  mali_encode), (UPat.sink(arg=KernelInfo("mali_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), mali_encode)])

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
    self.group_handle = pf.panfrost_context_group_handle(self.ctx)
    sync = pt.struct_drm_syncobj_create()
    if libc.dll.ioctl(self.fd, pt.DRM_IOCTL_SYNCOBJ_CREATE, ctypes.byref(sync)) != 0:
      raise RuntimeError(f'MALI syncobj creation failed: {ctypes.get_errno()}')
    self.sync_handle = sync.handle
    pt.panthor_set_sync_handle(self.ctx, self.sync_handle)
    super().__init__(device, MALIAllocator(self), None, arch="g610")
    self.var_vals = {'panthor_fd': self.fd, 'panthor_group': self.group_handle, 'panthor_sync': self.sync_handle}

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
    pf.panfrost_buffer_read(self.ctx, int(src), from_mv(dest), len(dest))

  def flush(self):
    fence = ctypes.c_void_p()
    self.ctx.contents.flush(self.ctx, ctypes.byref(fence), 0)
    if fence.value:
      check(self.screen.contents.fence_finish(self.screen, None, fence, (1 << 64) - 1), "MALI fence wait")
      self.screen.contents.fence_reference(self.screen, ctypes.byref(fence), None)

  def _wait_signal(self, sig, value:int, timeout:int|None=None):
    self.flush()
    point = pt.panthor_context_point(self.ctx)
    handles, points = ctypes.c_uint32(self.sync_handle), ctypes.c_uint64(point)
    wait = pt.struct_drm_syncobj_timeline_wait(handles=ctypes.addressof(handles), points=ctypes.addressof(points),
      timeout_nsec=time.monotonic_ns() + 1000000000, count_handles=1)
    ret = libc.dll.ioctl(self.fd, pt.DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, ctypes.byref(wait))
    if ret != 0: raise RuntimeError('MALI timeline wait failed')
    super()._wait_signal(sig, value, timeout)

  def finalize(self):
    super().finalize()
    for program in self.programs.values(): program.release()
    self.programs.clear()
    libc.dll.ioctl(self.fd, pt.DRM_IOCTL_SYNCOBJ_DESTROY, ctypes.byref(pt.struct_drm_syncobj_destroy(handle=self.sync_handle)))
    self.ctx.contents.delete_sampler_state(self.ctx, self.sampler_state)
    self.ctx.contents.destroy(self.ctx)
    self.screen.contents.destroy(self.screen)
    pf.driDestroyOptionCache(self.option_cache)
    pf.driDestroyOptionInfo(self.option_info)
    pf.glsl_type_singleton_decref()
    os.close(self.fd)
