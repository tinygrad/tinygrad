from __future__ import annotations
import base64, ctypes, os, struct
from tinygrad.device import Compiled, BufferStorage, BufferSpec, HostAllocator, Program
from tinygrad.dtype import AddrSpace, DType
from tinygrad.helpers import suppress_finalizing, is_image_shape, unwrap, from_mv
from tinygrad.runtime.support import c
from tinygrad.renderer import TinyELF
from tinygrad.renderer.nir import MALIRenderer
from tinygrad.runtime.autogen import panfrost as pf, mesa


def check(value, msg="MALI"):
  if value is None or (isinstance(value, int) and value == 0): raise RuntimeError(f"{msg} failed")
  return value

class MALIBuffer:
  def __init__(self, resource, offset=0): self.resource, self.offset = resource, offset

class MALIAllocator(HostAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.host or options.cpu_access: return super()._alloc(size, options)
    tmpl = pf.struct_pipe_resource(target=pf.PIPE_BUFFER, format=pf.PIPE_FORMAT_R8_UINT, width0=size, height0=1,
      depth0=1, array_size=1, nr_samples=1, nr_storage_samples=1,
      bind=pf.PIPE_BIND_GLOBAL | pf.PIPE_BIND_SHADER_BUFFER | pf.PIPE_BIND_SAMPLER_VIEW | pf.PIPE_BIND_SHADER_IMAGE | pf.PIPE_BIND_LINEAR,
      usage=pf.PIPE_USAGE_DEFAULT)
    ret = MALIBuffer(check(self.dev.screen.contents.resource_create(self.dev.screen, ctypes.byref(tmpl)), "MALI buffer allocation"))
    return BufferStorage(ret)

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None:
      self.dev.synchronize()
      self.dev.screen.contents.resource_destroy(self.dev.screen, storage.buf.resource)

  def _copyin(self, dest, src:memoryview): self.dev.buffer_write(dest, src)
  def _copyout(self, dest:memoryview, src): self.dev.buffer_read(src, dest)
  def _offset(self, buf, size:int, offset:int):
    return MALIBuffer(buf.resource, offset)

class MALIProgram(Program['MALIDevice']):
  def __init__(self, dev:MALIDevice, obj:TinyELF, links:dict[str,MALIProgram]):
    self.dev, self.name, self.signature = dev, obj.name, obj.signature
    src = base64.b64decode(obj.lib)
    reader = mesa.struct_blob_reader()
    mesa.blob_reader_init(reader, src, len(src))
    nir = check(mesa.nir_deserialize(None, pf.pan_shader_get_compiler_options(10), reader), f"MALI deserialize {obj.name}")
    state = pf.struct_pipe_compute_state(ir_type=pf.PIPE_SHADER_IR_NIR, prog=ctypes.cast(nir, ctypes.c_void_p),
      static_shared_mem=nir.contents.info.shared_size)
    self.kernel = check(dev.ctx.contents.create_compute_state(dev.ctx, ctypes.byref(state)), f"MALI compile {obj.name}")
    self.num_samplers, self.num_images = nir.contents.info.num_textures, nir.contents.info.num_images
    self.args: list[tuple[int, DType, bool]] = []
    cursor = 0
    for i, (off, dt) in enumerate(TinyELF.iter_sig(obj.signature)):
      self.args.append((off, dt, obj.signature[i][1] is not AddrSpace.ALU))
      cursor = off + dt.itemsize
    self.arg_size = cursor
    self.offsets = [off for off, dt, is_buffer in self.args if is_buffer]
    self.image_info = [((shape[1], shape[0], dt.itemsize, 1) if is_image_shape(shape) else (0, 0, 0, 0))
                       for _, addr, dt, shape in obj.signature if addr is not AddrSpace.ALU]

  @suppress_finalizing
  def __del__(self):
    if hasattr(self, "kernel"):
      self.dev.synchronize()
      self.dev.ctx.contents.delete_compute_state(self.dev.ctx, self.kernel)

  def __call__(self, *raw_args, global_size=(1,1,1), local_size=(1,1,1), wait=False, timeout=None):
    args = ctypes.create_string_buffer(self.arg_size)
    buffers = (c.POINTER[pf.struct_pipe_resource] * len(self.offsets))()
    handles = (c.POINTER[ctypes.c_uint32] * len(self.offsets))()
    buf_idx, buffer_views = 0, []
    for value, (off, dt, is_buffer) in zip(raw_args, self.args):
      if is_buffer:
        buffers[buf_idx] = value.resource
        buffer_views.append(value)
        handles[buf_idx] = ctypes.cast(ctypes.addressof(args) + off, c.POINTER[ctypes.c_uint32])
        ctypes.c_uint64.from_address(ctypes.addressof(args) + off).value += value.offset
        buf_idx += 1
      else: struct.pack_into(unwrap(dt.fmt), args, off, value)
    cb = pf.struct_pipe_constant_buffer(buffer=None, buffer_offset=0, buffer_size=self.arg_size, user_buffer=ctypes.addressof(args))
    grid = pf.struct_pipe_grid_info(work_dim=3, block=local_size, grid=global_size, variable_shared_mem=0,
      globals=buffers, num_globals=len(buffers))
    self.dev.launch(self.kernel, cb, grid, buffers, handles, self.image_info, self.num_samplers, self.num_images, buffer_views)
    if wait: self.dev.synchronize()

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
      if typ == pf.DRI_INT:
        options[i].info.range.start._int, options[i].info.range.end._int = lo, hi
    pf.driParseOptionInfo(self.option_info, options, len(options))
    self.config = pf.struct_pipe_screen_config(driver_name_is_inferred=False,
      options=ctypes.pointer(self.option_cache), options_info=ctypes.pointer(self.option_info))
    pf.glsl_type_singleton_init_or_ref()
    self.screen = check(pf.panfrost_drm_screen_create(self.fd, ctypes.byref(self.config)), "MALI panthor screen")
    flags = pf.PIPE_CONTEXT_COMPUTE_ONLY | pf.PIPE_CONTEXT_NO_LOD_BIAS
    self.ctx = check(self.screen.contents.context_create(self.screen, None, flags), "MALI compute context")
    super().__init__(device, MALIAllocator(self), MALIProgram, arch="g610")

  @property
  def has_copy_queue(self) -> bool: return False

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

  def launch(self, kernel, cb, grid, buffers, handles, image_info, num_samplers, num_images, buffer_views):
    sviews: list[c.POINTER[pf.struct_pipe_sampler_view]] = []
    iviews: list[pf.struct_pipe_image_view] = []
    buffer_list = list(buffers)
    for i, (width, height, itemsize, _) in enumerate(image_info):
      if not itemsize: continue
      fmt = pf.PIPE_FORMAT_R16G16B16A16_FLOAT if itemsize == 2 else pf.PIPE_FORMAT_R32G32B32A32_FLOAT
      tex2d = pf.struct_pipe_tex2d_from_buf(offset=buffer_views[i].offset // (4 * itemsize),
        row_stride=width, width=width, height=height)
      if len(sviews) < num_samplers:
        tmpl = pf.struct_pipe_sampler_view(format=fmt, target=pf.PIPE_TEXTURE_2D, is_tex2d_from_buf=True,
          swizzle_r=0, swizzle_g=1, swizzle_b=2, swizzle_a=3, texture=buffer_list[i], context=self.ctx, u=tex2d)
        sviews.append(self.ctx.contents.create_sampler_view(self.ctx, buffer_list[i], ctypes.byref(tmpl)))
      else:
        iviews.append(pf.struct_pipe_image_view(resource=buffer_list[i], format=fmt,
          access=pf.PIPE_IMAGE_ACCESS_READ_WRITE | pf.PIPE_IMAGE_ACCESS_TEX2D_FROM_BUFFER,
          shader_access=pf.PIPE_IMAGE_ACCESS_READ_WRITE, u=tex2d))
    if num_samplers:
      ss = pf.struct_pipe_sampler_state(wrap_s=pf.PIPE_TEX_WRAP_CLAMP_TO_BORDER, wrap_t=pf.PIPE_TEX_WRAP_CLAMP_TO_BORDER,
        wrap_r=pf.PIPE_TEX_WRAP_CLAMP_TO_BORDER, min_img_filter=pf.PIPE_TEX_FILTER_NEAREST,
        mag_img_filter=pf.PIPE_TEX_FILTER_NEAREST, unnormalized_coords=True)
      sstate = ctypes.c_void_p(self.ctx.contents.create_sampler_state(self.ctx, ctypes.byref(ss)))
      self.ctx.contents.bind_sampler_states(self.ctx, 5, 0, 1, ctypes.byref(sstate))
      self.ctx.contents.delete_sampler_state(self.ctx, sstate)
      self.ctx.contents.set_sampler_views(self.ctx, 5, 0, len(sviews), 0, (c.POINTER[pf.struct_pipe_sampler_view] * len(sviews))(*sviews))
    if num_images > num_samplers:
      self.ctx.contents.set_shader_images(self.ctx, 5, 0, len(iviews), 0, (pf.struct_pipe_image_view * len(iviews))(*iviews))
    self.ctx.contents.bind_compute_state(self.ctx, kernel)
    self.ctx.contents.set_constant_buffer(self.ctx, 5, 0, False, ctypes.byref(cb) if cb.buffer_size else None)
    self.ctx.contents.set_global_binding(self.ctx, 0, len(buffers), buffers, handles)
    self.ctx.contents.launch_grid(self.ctx, ctypes.byref(grid))
    self.ctx.contents.set_global_binding(self.ctx, 0, len(buffers), None, None)
    for view in sviews: self.ctx.contents.sampler_view_release(self.ctx, view)

  def synchronize(self, timeout:int|None=None):
    fence = ctypes.c_void_p()
    self.ctx.contents.flush(self.ctx, ctypes.byref(fence), 0)
    if fence.value:
      check(self.screen.contents.fence_finish(self.screen, None, fence, (1 << 64) - 1), "MALI fence wait")
      self.screen.contents.fence_reference(self.screen, ctypes.byref(fence), None)

  def finalize(self):
    super().finalize()
    self.ctx.contents.destroy(self.ctx)
    self.screen.contents.destroy(self.screen)
    pf.driDestroyOptionCache(self.option_cache)
    pf.driDestroyOptionInfo(self.option_info)
    pf.glsl_type_singleton_decref()
    os.close(self.fd)
