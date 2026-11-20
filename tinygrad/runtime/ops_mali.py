from __future__ import annotations
from typing import Any
import base64, ctypes, os, struct, time, itertools, mmap, bisect
from tinygrad.device import Compiled, BufferStorage, BufferSpec, HostAllocator, Program, MMIOInterface
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import suppress_finalizing, is_image_shape, from_mv, to_tuple, next_power2, round_up, ceildiv, prod
from tinygrad.renderer import TinyELF
from tinygrad.renderer.glsl import MALIRenderer
from tinygrad.runtime.autogen import mali, panthor as pt, libc
from tinygrad.runtime.support.hcq2 import HWQueue, HCQ_RUNTIME_DEV, ccall, layout_args, encode_cmdbuf, make_program, patch, do_get_time_ms
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.support.system import FileIOInterface, MAP_FIXED
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, uopfunc
from tinygrad.engine.realize import get_call_kernel_args


class MALIBuffer(int):
  handle:int
  offset:int
  def __new__(cls, va:int, handle:int=0, offset:int=0) -> MALIBuffer:
    ret = int.__new__(cls, va + offset)
    ret.handle, ret.offset = handle, offset
    return ret

class MALIAllocator(HostAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.host: return super()._alloc(size, options)
    size = round_up(size, mmap.PAGESIZE)
    va = self.dev.va.alloc(size, mmap.PAGESIZE)
    bo = self.dev.ioctl(pt.DRM_IOCTL_PANTHOR_BO_CREATE, pt.struct_drm_panthor_bo_create(size=size, exclusive_vm_id=self.dev.vm_id))
    offset = self.dev.ioctl(pt.DRM_IOCTL_PANTHOR_BO_MMAP_OFFSET, pt.struct_drm_panthor_bo_mmap_offset(handle=bo.handle)).offset
    try:
      ptr = self.dev.fd.mmap(va, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, offset)
      if ptr != va: raise RuntimeError(f'MALI mmap address mismatch: {ptr:#x} != {va:#x}')
      bind_op = pt.struct_drm_panthor_vm_bind_op(bo_handle=bo.handle, va=va, size=size)
      self.dev.ioctl(pt.DRM_IOCTL_PANTHOR_VM_BIND, pt.struct_drm_panthor_vm_bind(vm_id=self.dev.vm_id,
        ops=pt.struct_drm_panthor_obj_array(stride=48, count=1, array=ctypes.addressof(bind_op))))
    except Exception:
      self.dev.ioctl(pt.DRM_IOCTL_GEM_CLOSE, pt.struct_drm_gem_close(handle=bo.handle))
      self.dev.va.free(va)
      raise
    buf = MALIBuffer(va, bo.handle)
    self.dev.allocations[va] = (bo.handle, size)
    return BufferStorage(buf, size, MMIOInterface(ptr, size) if options.cpu_access else None)

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None and isinstance(storage.buf, MALIBuffer):
      self.dev.synchronize()
      va, size = int(storage.buf), storage.meta
      unmap = pt.struct_drm_panthor_vm_bind_op(flags=pt.DRM_PANTHOR_VM_BIND_OP_TYPE_UNMAP, va=va, size=size)
      self.dev.ioctl(pt.DRM_IOCTL_PANTHOR_VM_BIND, pt.struct_drm_panthor_vm_bind(vm_id=self.dev.vm_id,
        ops=pt.struct_drm_panthor_obj_array(stride=48, count=1, array=ctypes.addressof(unmap))))
      FileIOInterface.munmap(va, size)
      self.dev.ioctl(pt.DRM_IOCTL_GEM_CLOSE, pt.struct_drm_gem_close(handle=storage.buf.handle))
      self.dev.allocations.pop(va, None)
      self.dev.va.free(va)

  def _copyin(self, dest, src:memoryview): self.dev.buffer_write(dest, src)
  def _copyout(self, dest:memoryview, src): self.dev.buffer_read(src, dest)
  def _offset(self, buf, size:int, offset:int):
    return MALIBuffer(int(buf) - buf.offset, buf.handle, buf.offset + offset) if isinstance(buf, MALIBuffer) else buf + offset

class MALIProgram(Program['MALIDevice']):
  def __init__(self, dev:MALIDevice, obj:TinyELF, links:dict[str,MALIProgram]):
    self.dev, self.name, self.signature = dev, obj.name, obj.signature
    self.vendor_fau:list[int] = []
    src = base64.b64decode(obj.lib)
    cursor = 0
    for off, dt in TinyELF.iter_sig(obj.signature): cursor = off + dt.itemsize
    self.arg_size = cursor
    self.buffer_slots = [i for i, (_, addr, _, _) in enumerate(obj.signature) if addr is not AddrSpace.ALU]
    self.image_info = [((shape[1], shape[0], dt.itemsize, 1) if is_image_shape(shape) else (0, 0, 0, 0))
                       for _, addr, dt, shape in obj.signature if addr is not AddrSpace.ALU]
    self.info = mali.struct_pan_shader_info()
    if src[:4] != b'MLV1': raise RuntimeError('invalid Mali OpenCL shader')
    shared_size, _, _, barrier = struct.unpack_from('<IIII', src, 4)
    self.compile_vendor(src[20:], shared_size, bool(barrier))

  def compile_vendor(self, src, shared_size, barrier):
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
    self.code = src[code_off + 8:code_off + 8 + code_size]
    self.info.stage, self.info.work_reg_count = 5, 64
    self.info.wls_size, self.info.tls_size, self.info.preload = shared_size, tls_size, preload
    self.info.allow_merging_workgroups = False
    self.info.contains_barrier = barrier or shared_size > 0
    self.info.writes_global = bool(flags & 0x40000)
    self.info.ubo_count, self.info.ubo_mask, self.info.push_count = 1, 1, len(self.vendor_fau)
    for i in range(len(self.vendor_fau)): self.info.push_words[i].offset = self.arg_size + i * 4

  def release(self): pass

@uopfunc
def mali_run(buf:UOp, cmds:tuple) -> UOp:
  prev = None
  for fn, args in cmds:
    args = [buf.index(a[0]) if isinstance(a, tuple) else buf.bitcast(dtypes.u64).index(a).load() if isinstance(a, int) else a for a in args]
    if isinstance(fn, str) and (fn.startswith('write') or fn == 'signal'):
      size = 8 if fn == 'signal' else int(fn[-1])
      dt = {1:dtypes.u8, 2:dtypes.u16, 4:dtypes.u32, 8:dtypes.u64}[size]
      value = patch(UOp.alloc((size,), dtypes.u8, device=HCQ_RUNTIME_DEV.device), [(0, args[1].cast(dt))], bytes(size))
      if prev is not None: args[0] = args[0].after(prev)
      ret = ccall(libc.memcpy, args[0], value.getaddr(), UOp.const(size, dtypes.u64))
      buf = buf.after(value, ret)
      prev = ret
    elif fn == 'timestamp':
      if prev is not None: args[0] = args[0].after(prev)
      ret = do_get_time_ms(args[0])
      buf = buf.after(ret)
      prev = ret
    else:
      if prev is not None: args[0] = args[0].after(prev)
      ret = ccall(fn, *args)
      buf = buf.after(ret)
      prev = ret
  return buf.sink()

def mali_buffer(address, size, flags=9 | (1 << 4)): return struct.pack('<IIQIIII', flags, size, address, 0, 0, 0, 0)

def mali_texture_desc(width:int, height:int, itemsize:int) -> bytes:
  return struct.pack('<IIIIQII', 0x37c00022 if itemsize == 2 else 0x2fc00022, (width - 1) | ((height - 1) << 16), 0x688, 0, 0, 0, 0)

# Direct Valhall CSF command queue.
RAW_QUEUE_NONCE = itertools.count()
class MALIRawQueue(HWQueue):
  dev:MALIDevice
  def __init__(self, submit:UOp):
    super().__init__(submit)
    self.cmds:list[tuple] = []
    self.nonce = next(RAW_QUEUE_NONCE)
    self.q(UOp.const(0, dtypes.u64)) # CSF queue bring-up requires a leading NOP

  def move32(self, reg:int, value:Any):
    value = UOp.const(value, dtypes.u64) if isinstance(value, int) else value
    self.q(UOp(Ops.ADD, src=(UOp.const((2 << 56) | (reg << 48), dtypes.u64), value)))
  def move48(self, reg:int, value:Any):
    value = UOp.const(value, dtypes.u64) if isinstance(value, int) else value
    self.q(UOp(Ops.ADD, src=(UOp.const((1 << 56) | (reg << 48), dtypes.u64), value)))
  def wait(self, dst:UOp, val:Any, eq=False): pass
  def memory_barrier(self): pass
  def exec_barrier(self): self.q(UOp.const((3 << 56) | (0xff << 16), dtypes.u64))
  def signal(self, dst:UOp, val:Any):
    self.cmds.append(('signal', (dst.getaddr(HCQ_RUNTIME_DEV.device), UOp.const(val, dtypes.u64) if isinstance(val, int) else val)))
  def timestamp(self, dst:UOp): self.cmds.append(('timestamp', (dst,)))
  def copy(self, dst:UOp, src:UOp, n:int):
    dst_mali, src_mali = to_tuple(dst.device)[0] == self.devs[0], to_tuple(src.device)[0] == self.devs[0]
    if dst_mali and src_mali: dst_addr, src_addr = dst.getaddr(self.devs), src.getaddr(self.devs)
    elif dst_mali: dst_addr, src_addr = dst.getaddr(self.devs), src.getaddr()
    elif src_mali: dst_addr, src_addr = dst.getaddr(), src.getaddr(self.devs)
    else: return
    self.cmds.append((self.dev.copy_cb, (dst_addr, src_addr, UOp.const(n, dtypes.u64), UOp.const((int(dst_mali) << 1) | int(src_mali), dtypes.i32))))

  def write(self, address:UOp, value:UOp, size:int):
    self.cmds.append((f'write{size}', (address, value)))
  def store64(self, address:UOp, value:UOp):
    self.move48(88, address)
    self.move48(90, value)
    self.q(UOp.const((21 << 56) | (3 << 16) | (88 << 40) | (90 << 48), dtypes.u64)) # STORE_MULTIPLE: 64 bits

  def exec(self, call:UOp, prg:UOp):
    obj = TinyELF.from_program(prg)
    data = self.dev.program(obj)
    args = get_call_kernel_args(call, prg, self.devs)
    def alloc(size:int, tag:str): return make_program(prg, size, self.devs[0], tag, call.key)
    total_args = max(16, data.arg_size + 4 * len(data.vendor_fau))
    rows = layout_args(args) + [(data.arg_size + 4 * i, UOp.const(v, dtypes.u32)) for i, v in enumerate(data.vendor_fau)]
    argbuf = patch(alloc(total_args, 'kernargs'), [], bytes(total_args))
    for off, word in rows: self.write(argbuf.getaddr(self.devs) + off, word, word.dtype.itemsize)
    ubodesc = alloc(32, 'ubo')
    ubodesc = patch(ubodesc, [(8, argbuf.getaddr(self.devs))], mali_buffer(0, (data.arg_size + 15) & ~15))
    shader = patch(alloc(len(data.code), 'shader'), [], data.code)
    reg_alloc = 2 if data.info.work_reg_count <= 32 else 0
    spd = patch(alloc(32, 'spd'), [], struct.pack('<IIQIIII', 8 | (1 << 4) | (reg_alloc << 30), data.info.preload >> 48, 0, 0, 0, 0, 0))
    self.write(spd.getaddr(self.devs) + 8, shader.getaddr(self.devs), 8)
    sampler_word = 1 | (11 << 8) | (11 << 12) | (11 << 16) | (1 << 21) | (1 << 23) | (1 << 24) | (1 << 26) | (1 << 27) | (1 << 28)
    sampler_desc = patch(alloc(32, 'sampler'), [], struct.pack('<IIQIIII', sampler_word, 0, 0, 0, 0, 0, 0))
    textures:list[tuple] = []
    images:list[tuple] = []
    image_params = sorted(enumerate(data.image_info), key=lambda x: (data.buffer_slots[x[0]] not in prg.arg.outs, x[0]))
    has_images = any(x[1][2] for x in image_params)
    for i, (width, height, itemsize, _) in image_params:
      if not itemsize: continue
      resource = args[data.buffer_slots[i]]
      payload = alloc(64, f'{self.nonce:x}_{i}')
      payload = patch(payload, [(0, UOp.const((3 if itemsize == 2 else 4) << 24 | 0x21a, dtypes.u32)),
        (4, UOp.const(width * height * itemsize * 4, dtypes.u32)), (16, UOp.const(width * itemsize * 4, dtypes.u32)),
        (24, UOp.const((width - 1) | ((height - 1) << 16), dtypes.u32))], bytes(64))
      self.store64(payload.getaddr(self.devs) + 8, resource)
      desc = mali_texture_desc(width, height, itemsize)
      is_texture = data.buffer_slots[i] not in prg.arg.outs
      if is_texture: textures.append((desc, payload, resource))
      images.append((desc, payload, resource))
    resources = alloc(192, 'resources')
    resource_rows = [(0, ubodesc.getaddr(self.devs) | (1 << 56)), (8, UOp.const(32, dtypes.u64)),
      (48, sampler_desc.getaddr(self.devs) | (1 << 56)), (56, UOp.const(32, dtypes.u64))]
    ssbo_params = [(i, info) for i, info in enumerate(data.image_info) if not info[2]]
    if ssbo_params:
      # Arm supports 16 bindings per set, reserves global descriptor location 8 once, and keeps offsets global across sets.
      locations = [i + (i >= 8) for i, _ in ssbo_params]
      descs = [bytes(32)] * (max(locations) + 1)
      for (i, _), location in zip(ssbo_params, locations):
        descs[location] = mali_buffer(0, round_up(prod(obj.signature[data.buffer_slots[i]][3]) * obj.signature[data.buffer_slots[i]][2].itemsize, 16))
      ssbo_desc = patch(alloc(32 * len(descs), 'ssbo'), [], b''.join(descs))
      for (i, _), location in zip(ssbo_params, locations): self.write(ssbo_desc.getaddr(self.devs) + 32 * location + 8, args[data.buffer_slots[i]], 8)
      for set_idx in sorted({i // 16 for i, _ in ssbo_params}):
        resource_rows += [((4 + set_idx) * 16, ssbo_desc.getaddr(self.devs) | (1 << 56)),
                          ((4 + set_idx) * 16 + 8, UOp.const(32 * len(descs), dtypes.u64))]
    if textures:
      texture_desc = alloc(32 * len(textures), 'texture')
      texture_desc = patch(texture_desc, [(16 + 32 * i, payload.getaddr(self.devs)) for i, (_, payload, _) in enumerate(textures)],
        b''.join(x[0] for x in textures))
      resource_rows += [(64, texture_desc.getaddr(self.devs) | (1 << 56)), (72, UOp.const(32 * len(textures), dtypes.u64))]
    if images:
      image_desc = alloc(32 * len(images), 'image')
      image_desc = patch(image_desc, [(16 + 32 * i, payload.getaddr(self.devs)) for i, (_, payload, _) in enumerate(images)],
        b''.join(x[0] for x in images))
      resource_rows += [(80, image_desc.getaddr(self.devs) | (1 << 56)), (88, UOp.const(32 * len(images), dtypes.u64))]
    resources = patch(resources, resource_rows, bytes(192))
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
    self.move48(16, spd.getaddr(self.devs))
    local_size, global_size = prg.arg.local_size, prg.arg.global_size
    tls_size, wls_size = data.info.tls_size, data.info.wls_size
    threads = local_size[0] * local_size[1] * local_size[2]
    wg_per_task = ceildiv(self.dev.max_threads // self.dev.max_tasks_per_core, threads)
    max_instances = next_power2(wg_per_task * self.dev.max_tasks_per_core)
    instances = min(next_power2(global_size[0]) * next_power2(global_size[1]) * next_power2(global_size[2]), max_instances)
    wls_total = next_power2(max(wls_size, 128)) * instances * self.dev.core_id_range if wls_size else 0
    tls_total = next_power2(round_up(tls_size, 16)) * self.dev.max_threads * self.dev.core_id_range if tls_size else 0
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

    max_threads = min(self.dev.thread_max_workgroup_size, self.dev.max_threads,
      self.dev.num_registers_per_core // (32 if data.info.work_reg_count <= 32 else 64))
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
    if has_images: self.q(UOp.const((3 << 56) | (1 << 16), dtypes.u64)) # wait for descriptor stores
    self.q(UOp.const((4 << 56) | increment | (axis << 14), dtypes.u64)) # RUN_COMPUTE
    self.q(UOp.const((3 << 56) | (0xff << 16), dtypes.u64))
    self.move32(74, 0)
    self.q(UOp.const((36 << 56) | 1 | (1 << 4) | (2 << 8) | (74 << 40), dtypes.u64)) # clean L2/LSC, invalidate other caches
    self.q(UOp.const((3 << 56) | (1 << 16), dtypes.u64))

  def submit(self, cmdbuf:UOp) -> UOp:
    sync = patch(UOp.alloc((16,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='sync_op'),
      [(0, UOp.const(pt.DRM_PANTHOR_SYNC_OP_SIGNAL | pt.DRM_PANTHOR_SYNC_OP_HANDLE_TYPE_SYNCOBJ, dtypes.u32)),
       (4, UOp.const(self.dev.sync_handle, dtypes.u32))], bytes(16))
    queue = patch(UOp.alloc((40,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='queue_submit'),
      [(4, UOp.const(cmdbuf.max_numel(), dtypes.u32)), (8, cmdbuf.getaddr(self.devs)), (24, UOp.const(16, dtypes.u32)),
       (28, UOp.const(1, dtypes.u32)), (32, sync.getaddr())], bytes(40))
    group = patch(UOp.alloc((24,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='group_submit'),
      [(0, UOp.const(self.dev.group_handle, dtypes.u32)), (8, UOp.const(40, dtypes.u32)), (12, UOp.const(1, dtypes.u32)),
       (16, queue.getaddr())], bytes(24))
    handle = patch(UOp.alloc((4,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='sync_handle'),
      [(0, UOp.const(self.dev.sync_handle, dtypes.u32))], bytes(4))
    reset = patch(UOp.alloc((16,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='sync_reset'),
      [(0, handle.getaddr()), (8, UOp.const(1, dtypes.u32))], bytes(16))
    fd = UOp.const(self.dev.fd.fd, dtypes.i32)
    wait = patch(UOp.alloc((40,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='sync_wait'),
      [(0, handle.getaddr()), (8, UOp.const(0x7fffffffffffffff, dtypes.u64)), (16, UOp.const(1, dtypes.u32))], bytes(40))
    wait_cmd = (libc.dll.ioctl, (fd, UOp.const(pt.DRM_IOCTL_SYNCOBJ_WAIT, dtypes.u32), wait.index(0)))
    reset_cmd = (libc.dll.ioctl, (fd, UOp.const(pt.DRM_IOCTL_SYNCOBJ_RESET, dtypes.u32), reset.index(0)))
    submit = (libc.dll.ioctl, (fd, UOp.const(pt.DRM_IOCTL_PANTHOR_GROUP_SUBMIT, dtypes.u32), group.index(0)))
    copy_cmds = [cmd for cmd in self.cmds if cmd[0] == libc.memcpy]
    other_cmds = [cmd for cmd in self.cmds if cmd[0] != libc.memcpy]
    return mali_run(cmdbuf, (*other_cmds, wait_cmd, *copy_cmds, reset_cmd, submit))

  def encode(self) -> UOp:
    for u in self.lin.src: self.q_rewrite.rewrite(u, ctx=self)
    empty = len(self.blob) <= 8  # the queue starts with one NOP for bring-up
    if empty: self.q(UOp.const(0, dtypes.u64))
    deps, self.deps = self.deps, ()
    cmdbuf = encode_cmdbuf(self)
    if empty:
      fd = UOp.const(self.dev.fd.fd, dtypes.i32)
      handle = patch(UOp.alloc((4,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='sync_handle'),
        [(0, UOp.const(self.dev.sync_handle, dtypes.u32))], bytes(4))
      wait = patch(UOp.alloc((40,), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name='sync_wait'),
        [(0, handle.getaddr()), (8, UOp.const(0x7fffffffffffffff, dtypes.u64)), (16, UOp.const(1, dtypes.u32))], bytes(40))
      wait_cmd = (libc.dll.ioctl, (fd, UOp.const(pt.DRM_IOCTL_SYNCOBJ_WAIT, dtypes.u32), wait.index(0)))
      ret = mali_run(cmdbuf, (wait_cmd, *self.cmds))
    else: ret = self.submit(cmdbuf)
    return UOp.sink(ret, *deps) if deps else ret

def mali_encode(s:UOp) -> UOp: return MALIRawQueue(s).encode()
Compiled.pm_encode += PatternMatcher([(UPat.sink(arg=KernelInfo("mali_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"),
  mali_encode), (UPat.sink(arg=KernelInfo("mali_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), mali_encode)])

class MALIDevice(Compiled):
  renderers = [MALIRenderer]

  def ioctl(self, cmd:int, req):
    if libc.dll.ioctl(self.fd.fd, cmd, ctypes.byref(req)) != 0: raise RuntimeError(f'MALI ioctl {cmd:#x} failed: {ctypes.get_errno()}')
    return req

  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/dri/renderD128', os.O_RDWR)
    info = pt.struct_drm_panthor_gpu_info()
    self.ioctl(pt.DRM_IOCTL_PANTHOR_DEV_QUERY, pt.struct_drm_panthor_dev_query(type=pt.DRM_PANTHOR_DEV_QUERY_GPU_INFO,
      size=ctypes.sizeof(info), pointer=ctypes.addressof(info)))
    self.gpu_id = info.gpu_id
    self.max_threads, self.thread_max_workgroup_size = info.max_threads, info.thread_max_workgroup_size
    self.max_tasks_per_core = info.thread_features >> 24
    self.num_registers_per_core = info.thread_features & 0x3fffff
    self.core_id_range = info.shader_present.bit_length()
    self.vm_id = self.ioctl(pt.DRM_IOCTL_PANTHOR_VM_CREATE, pt.struct_drm_panthor_vm_create()).id
    va_base = 0x100000000
    self.va = TLSFAllocator(1 << 40, base=va_base)
    queue = pt.struct_drm_panthor_queue_create(priority=8, ringbuf_size=0x10000)
    group = self.ioctl(pt.DRM_IOCTL_PANTHOR_GROUP_CREATE, pt.struct_drm_panthor_group_create(
      queues=pt.struct_drm_panthor_obj_array(stride=8, count=1, array=ctypes.addressof(queue)),
      max_compute_cores=info.shader_present.bit_count(), max_fragment_cores=info.shader_present.bit_count(), max_tiler_cores=1,
      priority=0, compute_core_mask=info.shader_present, fragment_core_mask=info.shader_present,
      tiler_core_mask=info.tiler_present, vm_id=self.vm_id))
    self.group_handle = group.group_handle
    self.sync_handle = self.ioctl(pt.DRM_IOCTL_SYNCOBJ_CREATE, pt.struct_drm_syncobj_create(flags=1)).handle # initially signaled
    self.programs:dict[tuple, MALIProgram] = {}
    self.allocations:dict[int, tuple[int, int]] = {}
    self.copy_cb = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int)(self._copy_sync)
    setattr(self.copy_cb, '__name__', 'mali_copy_sync')
    setattr(libc.dll, 'mali_copy_sync', self.copy_cb)
    super().__init__(device, MALIAllocator(self), None, arch="g610,IMAGE_PITCH_ALIGNMENT=64")
    self.var_vals = {'panthor_fd': self.fd.fd, 'panthor_group': self.group_handle, 'panthor_sync': self.sync_handle}

  def program(self, obj:TinyELF) -> MALIProgram:
    key = (obj.lib, obj.name, obj.signature)
    if key not in self.programs: self.programs[key] = MALIProgram(self, obj, {})
    return self.programs[key]

  @property
  def has_copy_queue(self) -> bool: return True

  def _sync_address(self, address:int, size:int, typ:int):
    vas = sorted(self.allocations)
    if (i:=bisect.bisect_right(vas, address)-1) < 0: return
    handle, alloc_size = self.allocations[vas[i]]
    if (offset:=address-vas[i]) < alloc_size: self._bo_sync([(handle, offset, min(size, alloc_size-offset))], typ)
  def _copy_sync(self, dst:int, src:int, n:int, flags:int):
    if flags & 1: self._sync_address(src, n, pt.DRM_PANTHOR_BO_SYNC_CPU_CACHE_FLUSH_AND_INVALIDATE)
    ctypes.memmove(dst, src, n)
    if flags & 2: self._sync_address(dst, n, pt.DRM_PANTHOR_BO_SYNC_CPU_CACHE_FLUSH)
  def _bo_sync(self, ops:list[tuple[int, int, int]], typ:int):
    if not ops: return
    arr = (pt.struct_drm_panthor_bo_sync_op * len(ops))(*[pt.struct_drm_panthor_bo_sync_op(handle=h, type=typ, offset=o, size=s) for h, o, s in ops])
    self.ioctl(pt.DRM_IOCTL_PANTHOR_BO_SYNC,
      pt.struct_drm_panthor_bo_sync(ops=pt.struct_drm_panthor_obj_array(stride=24, count=len(ops), array=ctypes.addressof(arr))))
  def buffer_write(self, dest:MALIBuffer, src:memoryview):
    ctypes.memmove(int(dest), from_mv(src), src.nbytes)
    self._bo_sync([(dest.handle, dest.offset, src.nbytes)], pt.DRM_PANTHOR_BO_SYNC_CPU_CACHE_FLUSH)
  def buffer_read(self, src:MALIBuffer, dest:memoryview):
    self.synchronize()
    self._bo_sync([(src.handle, src.offset, dest.nbytes)], pt.DRM_PANTHOR_BO_SYNC_CPU_CACHE_FLUSH_AND_INVALIDATE)
    ctypes.memmove(from_mv(dest), int(src), dest.nbytes)
  def flush(self): pass

  def _wait_signal(self, sig, value:int, timeout:int|None=None):
    handles = ctypes.c_uint32(self.sync_handle)
    wait = pt.struct_drm_syncobj_wait(handles=ctypes.addressof(handles), timeout_nsec=time.monotonic_ns() + 1000000000, count_handles=1)
    self.ioctl(pt.DRM_IOCTL_SYNCOBJ_WAIT, wait)
    super()._wait_signal(sig, value, timeout)

  def finalize(self):
    super().finalize()
    self.programs.clear()
    self.ioctl(pt.DRM_IOCTL_SYNCOBJ_DESTROY, pt.struct_drm_syncobj_destroy(handle=self.sync_handle))
    self.ioctl(pt.DRM_IOCTL_PANTHOR_GROUP_DESTROY, pt.struct_drm_panthor_group_destroy(group_handle=self.group_handle))
    self.ioctl(pt.DRM_IOCTL_PANTHOR_VM_DESTROY, pt.struct_drm_panthor_vm_destroy(id=self.vm_id))
    os.close(self.fd.fd)
