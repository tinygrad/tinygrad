from __future__ import annotations
import ctypes, dataclasses, os, struct, time
from typing import Any, cast
from tinygrad.device import Compiled, Allocator, BufferStorage, BufferSpec, Buffer
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.renderer.cstyle import CUDARenderer, CStyleLanguage
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, KernelInfo
from tinygrad.runtime.support.hcq2 import HWQueue, patch, get_time_ms, to_name, ccall
from tinygrad.runtime.support.usb import USB3, usb_link, usb_stream, usb_ctrl, usb_bulk, usb_stack, usb_fail, usb_poke, HOST_SIZE
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.runtime.autogen import mtfw, libc
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.compute import ROOT_BASE, META_BASE
from tinygrad.runtime.support.mt.ccb import MT_TOKEN_BASE, MT_HOST_PRIVATE
from tinygrad.runtime.support.mt.submission import SUBMISSION_VA
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.mt.launch import MTLaunch
from tinygrad.runtime.support.mt.generated import MTGeneratedLaunch
from tinygrad.runtime.support.mt.compiler import MTCompiler
from tinygrad.runtime.support.mt.elf import MTKernelImage


class MUSACompiler(Compiler):
  def __init__(self):
    super().__init__('mt_O3')
    self.mt = MTCompiler()
  def compile(self, src:str) -> bytes: return self.mt.compile_raw(src)

class MUSARenderer(CUDARenderer):
  kernel_typedef = 'extern "C" __global__ void'
  def __init__(self, target):
    CStyleLanguage.__init__(self, target)
    self.compiler, self.tensor_cores = MUSACompiler(), []
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    return super().render_kernel(function_name, kernel, bufs, uops, ['#include <musa.h>'] + (prefix or []))
  def supported_dtypes(self): return set(dtypes.all) - {dtypes.double}

@dataclasses.dataclass(frozen=True)
class MTBufferMeta:
  va: int
  pa: int
  size: int

class MTAllocator(Allocator['MTDevice']):
  def __init__(self, dev):
    super().__init__(dev)
    self.next_va, self.next_pa = 0x8001000000, 0x80000000
    self.dev.views = {}
    self.host_buffers:dict[int,Any] = {}
    self.free_blocks:list[tuple[int,int,int]] = []

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    size = round_up(size, 0x1000)
    if options.host:
      host = (ctypes.c_ubyte * size)()
      va = ctypes.addressof(host)
      view = MMIOInterface(va,size,fmt='B')
      meta = MTBufferMeta(va,0,size)
      self.host_buffers[va] = host
      self.dev.views[va] = (view,meta)
      return BufferStorage(va,meta,view)
    if self.free_blocks and self.free_blocks[-1][2] == size:
      va, pa, size = self.free_blocks.pop()
    else:
      va, pa = self.next_va, self.next_pa
      self.next_va += size
      self.next_pa += size
    view = self.dev.mt.pci_dev.map_bar(2, off=pa, size=size, fmt='B')
    meta = MTBufferMeta(va, pa, size)
    self.dev.map_buffer(va, pa, size)
    self.dev.views[va] = (view, meta)
    return BufferStorage(va, meta, view)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    meta = storage.meta
    self.dev.views.pop(meta.va, None)
    self.dev.buffer_maps.pop(meta.va, None)
    if meta.va in self.host_buffers:
      self.host_buffers.pop(meta.va)
    else: self.free_blocks.append((meta.va, meta.pa, meta.size))

  def _copyin(self, dest:int, src:memoryview):
    view, meta = self.dev._view(dest)
    off = dest-meta.va
    if off % 4 == 0 and src.nbytes % 4 == 0: view[off:off+src.nbytes] = src.cast('B')
    else:
      start, end = off & ~3, round_up(off+src.nbytes,4)
      data = bytearray(view[start:end])
      data[off-start:off-start+src.nbytes] = src.cast('B')
      view[start:end] = data

  def _copyout(self, dest:memoryview, src:int):
    view, meta = self.dev._view(src)
    off = src-meta.va
    start, end = off & ~3, round_up(off+dest.nbytes,4)
    data = view[start:end]
    dest[:] = data[off-start:off-start+dest.nbytes]

  def _map(self, buf:Buffer) -> BufferStorage:
    if buf.device == self.dev.device: return buf.get_storage()
    storage = self._alloc(buf.nbytes,buf.options)
    self._copyin(storage.buf,buf.as_memoryview(allow_zero_copy=True))
    return storage

  def _unmap(self, storage:BufferStorage): self._free(storage,BufferSpec())
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def mt_poll(link:UOp, addr:UOp, target:UOp) -> UOp:
  loop = UOp.range(UOp(Ops.NOOP).after(link,start:=get_time_ms(link)),next(UOp.unique_num),dtype=dtypes.void)
  slot = usb_stack(dtypes.u64)
  header = usb_ctrl(link.after(loop),0x40,0xF0,0x20|0x0F00,2,usb_stack(dtypes.uint64,addr,2).index(0),12,5000)
  read = usb_bulk(link.after(header),0x81,slot.index(0),8)
  response = slot.after(read,ccall(libc.usleep,100).sink()).index(0).load()
  done = (response >> 32).cast(dtypes.u64)
  poll = done.backedge(loop,(done < target) & (get_time_ms(done) - start < int(float(os.environ.get('MT_TIMEOUT','5'))*1000))).sink()
  status = response & 0xffffffff
  bad = ((done.after(poll) != target).cast(dtypes.u64) << 63) | (((status&0xffffffff) != 0).cast(dtypes.u64)*target << 32) | (status & 0xffffffff)
  return usb_fail(link.after(poll),bad)


class MTComputeQueue(HWQueue):
  dev:MTDevice
  def __init__(self, submit):
    super().__init__(submit)
    self.cacheable = (info:=getattr(submit.arg,'hcq',None)) is not None and info.cacheable
    self.submissions:list[tuple[MTGeneratedLaunch,UOp,bytes,dict[int,bytes],int,dict[int,bytes]|None,UOp|None]] = []
    self.signals:list[tuple[UOp,UOp]] = []
    self.timestamp_slots:list[list[UOp|None]] = []
    self.timestamp_open = False

  def encode(self) -> UOp:
    for u in self.lin.src:
      if u.op is Ops.END: raise NotImplementedError('MT queue loops are not supported')
      if u.op is not Ops.CALL: continue
      body = u.src[0]
      if body.op is Ops.PROGRAM: self.exec(u,body)
      elif body.op is Ops.CUSTOM_FUNCTION:
        if body.arg.name == 'store': self.signal(u.src[1],u.src[2])
        elif body.arg.name == 'timestamp': self.timestamp(u.src[1])
        elif body.arg.name not in ('memory_barrier','exec_barrier','wait','wait_eq'):
          raise NotImplementedError(f'MT queue operation {body.arg.name}')
      elif body.op is not Ops.STORE: raise NotImplementedError(f'MT queue operation {body.op}')
    return self.submit(UOp.sink())

  def memory_barrier(self): pass  # every MT submission is drained before the next
  def exec_barrier(self): pass
  def wait(self, signal:UOp, value:UOp): pass  # synchronous submissions make cross-queue waits implicit
  def signal(self, signal:UOp, value:UOp): self.signals.append((signal,value))
  def timestamp(self, signal:UOp):
    if self.timestamp_open:
      self.timestamp_slots[-1][1] = signal
    else:
      self.timestamp_slots.append([signal,None])
    self.timestamp_open = not self.timestamp_open

  def snapshot_mmu(self, launch:MTLaunch) -> dict[int,bytes]:
    pages:dict[int,bytes] = {}
    for va in launch.mmu.pages:
      if 0x8001000000 <= va < cast(MTAllocator,self.dev.allocator).next_va: continue
      pci,pdi,pti = (va>>30)&0x3ff,(va>>21)&0x1ff,(va>>12)&0x1ff
      if (pci,pdi) in launch.mmu.pt:
        offs = {(pci*4)&~0xfff,(launch.mmu.pd[pci]-launch.mmu.root_pa+pdi*8)&~0xfff,
                (launch.mmu.pt[pci,pdi]-launch.mmu.root_pa+pti*8)&~0xfff}
        for off in offs: pages[off] = bytes(launch.mmu.root[off:off+0x1000])
    return pages

  def exec(self, call:UOp, prg:UOp):
    if self.submissions and self.submissions[-1][5] is None:
      prev = self.submissions[-1]
      self.submissions[-1] = (*prev[:5],self.snapshot_mmu(prev[0]),prev[6])
    obj = TinyELF.from_program(prg)
    global_size = cast(tuple[int,int,int],tuple(x if isinstance(x,int) else 1 for x in prg.arg.global_size))
    local_size = cast(tuple[int,int,int],tuple(x if isinstance(x,int) else 1 for x in prg.arg.local_size))
    launch = self.dev.get_launch(obj,global_size,local_size)
    self.dev.activate_launch(launch,obj,global_size,local_size)
    if obj.profile_key is not None: self.dev.cached_launches[obj.profile_key] = (launch,obj,global_size,local_size)
    grid_deps = None
    if any(not isinstance(x,int) for x in (*prg.arg.global_size,*prg.arg.local_size)):
      grid_rows = [(off,(prg.arg.global_size[i]-1).cast(dtypes.u32)) for i,off in enumerate([0x2c,0x30,0x34])
                   if not isinstance(prg.arg.global_size[i],int)]
      local_rows = [(0x3c,prg.arg.local_size[2]-1,dtypes.u8),(0x3d,(prg.arg.local_size[1]-1)<<2,dtypes.u8),
                    (0x3e,(prg.arg.local_size[0]-1)<<4,dtypes.u16)]
      grid_rows += [(off,val.cast(dt)) for off,val,dt in local_rows if not isinstance(val,int)]
      grid_off = launch.backing_offsets[0x81e1400000]
      grid = UOp.alloc((0x1000,),dtypes.u8,device='CPU')
      grid_deps = patch(grid,grid_rows,launch.write_map.get(grid_off,bytes(0x1000)))
    const_off = launch.backing_offsets[0x81e1200000]
    args = UOp.alloc((0x1000,),dtypes.u8,device='CPU')
    rows = [(0x38+i*8,arg.cast(dtypes.u64)) for i,arg in enumerate(get_call_kernel_args(call,prg,self.devs))]
    state = (bytes(launch.packet),dict(launch.write_map),launch.generation,None)
    self.submissions.append((launch,patch(args,rows,launch.write_map.get(const_off,bytes(0x1000))),*state,grid_deps))

  def submit(self, cmdbuf:UOp) -> UOp:
    if len(self.submissions) > 1 and self.submissions[-1][5] is None:
      prev = self.submissions[-1]
      self.submissions[-1] = (*prev[:5],self.snapshot_mmu(prev[0]),prev[6])
    ops: list[Any] = []
    link = usb_link(self.dev.device)
    def add(op):
      nonlocal link
      ops.append(op)
      link = link.after(op)
    root_addr = self.dev.mt.pci_dev.bar_info(2)[0]+ROOT_BASE
    poll_in_hcq = not self.cacheable or len(self.submissions) > 1 or bool(self.timestamp_slots)
    for i,(launch,deps,packet,write_map,generation,mmu_pages,grid_deps) in enumerate(self.submissions):
      ccb = launch.ccb
      assert ccb is not None
      def write(addr:int|UOp, blob:bytes, rows=()):
        data = UOp.alloc((len(blob),),dtypes.u8,device='CPU')
        addr = addr if isinstance(addr,UOp) else UOp.const(addr,dtypes.u64)
        data = patch(data,list(rows),blob)
        if len(blob) > 0x200000: op = usb_stream(link,addr,data,UOp.const(len(blob),dtypes.int),True)
        else:
          header = usb_ctrl(link,0x40,0xF0,0x60|0x0F00,1,usb_stack(dtypes.uint64,addr,len(blob)//4).index(0),12,5000)
          op = usb_bulk(link.after(header),0x02,data.index(0),len(blob))
        add(op)
      previous = self.submissions[i-1][3] if i else self.dev.uploaded_write_map
      changed = sorted((off,data) for off,data in write_map.items() if off != launch.backing_offsets[0x81e1200000] and
                       (previous.get(off) != data))
      chunks:list[tuple[int,bytearray]] = []
      for off,data in changed:
        if chunks and off == chunks[-1][0]+len(chunks[-1][1]): chunks[-1][1].extend(data)
        else: chunks.append((off,bytearray(data)))
      for off,chunk in chunks: write(launch.backing.addr+off,bytes(chunk))
      if len(self.submissions) > 1:
        if i == 0: write(launch.sv.addr,packet)
        else:
          previous_packet = self.submissions[i-1][2]
          for off in range(0,len(packet),0x1000):
            packet_chunk = packet[off:off+0x1000]
            if previous_packet[off:off+len(packet_chunk)] != packet_chunk: write(launch.sv.addr+off,packet_chunk)
        previous_mmu = self.submissions[i-1][5] if i else self.submissions[-1][5]
        assert mmu_pages is not None and previous_mmu is not None
        for off,data in mmu_pages.items():
          if previous_mmu.get(off) != data: write(root_addr+off,data)
      idx = link.index(2).load()
      seq = link.index(3).load()
      token = (MT_TOKEN_BASE+seq).cast(dtypes.u64)
      ci = mtfw.MTFW_CCB_ITEM()
      ci.submissionCmd,ci.ccbItemToken,ci.hostPrivateData,ci.processId = 104,0,MT_HOST_PRIVATE,0
      sp = ci.ccbItemPara.submissionPara
      sp.pageTableRootAddr,sp.pageTableGeneration = ROOT_BASE,generation
      sp.submissionQueueId,sp.submissionAddr,sp.submissionSize,sp.qSempOutofOrder = 0,SUBMISSION_VA,len(packet),1
      item_size = ci.SIZE
      item = UOp.alloc((ci.SIZE,),dtypes.u8,device='CPU')
      item = patch(item,[(16,token)],bytes(ci))
      add(usb_stream(link.after(deps),UOp.const(launch.v.addr+ccb.item_off,dtypes.u64),item,UOp.const(item_size,dtypes.int),True))
      args_off = launch.backing_offsets[0x81e1200000]
      add(usb_stream(link,UOp.const(launch.backing.addr+args_off,dtypes.u64),deps,UOp.const(0x1000,dtypes.int),True))
      if grid_deps is not None:
        grid_off = launch.backing_offsets[0x81e1400000]
        add(usb_stream(link,UOp.const(launch.backing.addr+grid_off,dtypes.u64),grid_deps,UOp.const(0x1000,dtypes.int),True))
      next_wr = ((idx + 1) & 0xff).cast(dtypes.u32)
      write(launch.v.addr+ccb.queue_off+idx*8,struct.pack('<Q',META_BASE+ccb.item_off))
      add(usb_poke(link,launch.v.addr+ccb.queue_off+0x4000,next_wr))
      add(usb_poke(link,self.dev.mt.mmio.addr+0xc00b00,UOp.const(3,dtypes.u32)))
      if poll_in_hcq:
        add(mt_poll(link,launch.v.addr+ccb.queue_off+0x1004+idx*0x30,token))
        add(usb_poke(link,launch.v.addr+ccb.queue_off+0x4028,next_wr))
      if i < len(self.timestamp_slots):
        start_signal,end_signal = self.timestamp_slots[i]
        def read_timing(addr:int, dtype):
          slot = usb_stack(dtype)
          header = usb_ctrl(link,0x40,0xF0,0x20|0x0F00,2,usb_stack(dtypes.uint64,addr,dtype.itemsize//4).index(0),12,5000)
          read = usb_bulk(link.after(header),0x81,slot.index(0),dtype.itemsize)
          add(read)
          return slot.after(read).index(0).load()
        timestamp = read_timing(launch.v.addr+0x40a000,dtypes.u64)
        run_time = read_timing(launch.v.addr+0x409008,dtypes.u32).cast(dtypes.u64)
        if start_signal is not None: add(start_signal.index(1).store(timestamp))
        if end_signal is not None: add(end_signal.index(1).store(timestamp+run_time))
      self.dev.poll_in_hcq = poll_in_hcq
      add(link.index(2).store(next_wr.cast(dtypes.u64)))
      add(link.index(3).store(seq+1))
      self.dev.uploaded_write_map = write_map
    for signal,value in self.signals: add(signal.index(0).store(value))  # type: ignore[assignment]
    return UOp.sink(UOp.barrier(*self.deps,*ops))


def mt_lower_call(call:UOp) -> UOp:
  from tinygrad.runtime.support.hcq2 import lower_encoded_call
  for u in call.body.toposort():
    if (u.op is Ops.AFTER and u.src[0].op is Ops.CALL and u.src[0].src[0].op is Ops.SINK and
        u.src[0].src[1].op is Ops.PARAM and u.src[0].src[1].arg.name == 'COMPUTE:0'):
      queue = MTComputeQueue(u.replace(arg=call.arg)).encode().replace(arg=call.body.arg)
      return lower_encoded_call(call.replace(src=(queue,*call.src[1:])))
  raise RuntimeError('No MT compute submission found')

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("mt_compute_submit")).call(UPat()).after(allow_any_len=True,name="s"),lambda s:MTComputeQueue(s).encode()),
])

class MTDevice(Compiled):
  renderers = [MUSARenderer]
  timestamp_divider = 25
  serialize_hcq = True
  skip_nested_lift = True
  direct_hcq = True
  enqueue_copies = False
  def __init__(self, device:str=''):
    dev_id = int(device.split(':')[1]) if ':' in device else 0
    self.mt = MTDev.open_usb(dev_id)
    self.launch_order:list[MTGeneratedLaunch] = []
    self.launch_cache:dict[tuple[bytes,tuple[int,int,int],tuple[int,int,int]],MTGeneratedLaunch] = {}
    self.buffer_maps:dict[int,tuple[int,int]] = {}
    self.current_launch:MTGeneratedLaunch|None = None
    self.poll_in_hcq = True
    self.cached_launches:dict[bytes,tuple[MTGeneratedLaunch,TinyELF,tuple[int,int,int],tuple[int,int,int]]] = {}
    self.uploaded_write_map:dict[int,bytes] = {}
    self.started = False
    self.response_checked = 0
    self.views:dict[int,tuple[Any,MTBufferMeta]] = {}
    super().__init__(device, MTAllocator(self), None, arch='mp_21')
    usb_dev = cast(USBPCIDevice,self.mt.pci_dev).usb
    words = [ctypes.addressof(x.contents) for x in (usb_dev.usb.handle,USB3.ctx())] + [0,0,0]
    host = Buffer('CPU',HOST_SIZE,options=BufferSpec(nolru=True),initial_value=struct.pack('5Q',*words).ljust(HOST_SIZE,b'\0'))
    Compiled.link_bufs[to_name(self.device,'usb_host')] = host

  @property
  def has_copy_queue(self) -> bool: return False

  def get_launch(self, obj:TinyELF, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1)) -> MTGeneratedLaunch:
    key = (obj.lib, global_size, local_size)
    if key not in self.launch_cache:
      launch = MTGeneratedLaunch(self.mt)
      launch.kernel_lib = b''
      self.launch_cache[key] = launch
      self.launch_order.append(launch)
      if len(self.launch_order) > 512:
        old = self.launch_order.pop(0)
        for launch_key,cached_launch in list(self.launch_cache.items()):
          if cached_launch is old: del self.launch_cache[launch_key]
        for profile_key,cached in list(self.cached_launches.items()):
          if cached[0] is old: del self.cached_launches[profile_key]
      for va,(pa,size) in self.buffer_maps.items(): launch.map_buffer(va,pa,size)
    return self.launch_cache[key]

  def map_buffer(self, va:int, pa:int, size:int):
    self.buffer_maps[va] = (pa,size)
    if self.current_launch is not None: self.current_launch.map_buffer(va,pa,size)
    if self.started and self.current_launch is not None: self.current_launch.upload_mmu()

  def _fault(self) -> tuple[int, ...]:
    if self.current_launch is None: return ()
    launch = self.current_launch
    return (launch._read(0xe150),launch._read(0xe158),launch._read(0x130,4),launch._read(0x4a0))

  def synchronize(self, timeout:int|None=None):
    super().synchronize(timeout)
    host_buf = Compiled.link_bufs[to_name(self.device,'usb_host')]
    assert isinstance(host_buf,Buffer)
    host = host_buf.host.view(fmt='q')
    if host[5]: raise RuntimeError(f'MT USB error: {host[5]}, fault={self._fault()}')
    if self.poll_in_hcq:
      self.response_checked = host[3]  # every in-graph poll already validated the response value and token
      return
    seq = host[3]
    if self.current_launch is None: return
    ccb = self.current_launch.ccb
    assert ccb is not None
    for i in range(self.response_checked,seq):
      off = ccb.queue_off+0x1004+(i&0xff)*0x30
      deadline = time.monotonic() + (timeout if timeout is not None else float(os.environ.get('MT_TIMEOUT','5')))
      while True:
        response = struct.unpack('<Q',self.current_launch.v[off:off+8])[0]
        value, token = response & 0xffffffff, response >> 32
        if value: raise RuntimeError(f'MT submission {i} failed: value={value}, token={token:#x}, fault={self._fault()}')
        if token == MT_TOKEN_BASE+i: break
        if time.monotonic() >= deadline: raise TimeoutError(f'MT submission {i} response timed out: value={value}, token={token:#x}')
        time.sleep(0.0001)
      self.current_launch.v[ccb.queue_off+0x4028:ccb.queue_off+0x402c] = struct.pack('<I',(i+1)&0xff)
    self.response_checked = seq

  def hcq_runtime_key(self, input_uops:list[UOp]): return tuple((u.arg.buffer,u.arg.buffer._buf) for u in input_uops)
  def lower_hcq_call(self, call:UOp) -> UOp: return mt_lower_call(call)

  def post_run(self) -> None: self.synchronize()

  def post_link(self, input_uops:list[UOp]) -> None:
    for launch in self.launch_order:
      if launch.mmu_dirty:
        launch.upload_mmu()
        launch.mmu_dirty = False

  def activate_cached(self, profile_key:bytes|None) -> None:
    if profile_key is not None and (cached:=self.cached_launches.get(profile_key)) is not None:
      launch,obj,global_size,local_size = cached
      if launch.cache_activations and (self.current_launch is not launch or MTLaunch._loaded is not launch):
        self.activate_launch(launch,obj,global_size,local_size)
        for off,data in launch.write_map.items(): launch.backing[off:off+len(data)] = data
        launch.sv[:0x1000] = launch.packet[:0x1000]
      launch.cache_activations += 1

  def hcq_cacheable(self, linear:UOp) -> bool:
    for call in linear.toposort():
      if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM and call.src[0].arg.target.arch == 'mp_21':
        if MTKernelImage.from_elf(TinyELF.from_program(call.src[0]).lib).metadata.kernel['.has_barrier']: return False
    return True

  def ensure_started(self, launch:MTGeneratedLaunch):
    if self.current_launch is not launch:
      launch.start(timeout=float(os.environ.get('MT_BOOT_TIMEOUT','10')))
      self.current_launch = launch
      self.started = True

  def activate_launch(self, launch:MTGeneratedLaunch, obj:TinyELF, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> None:
    image = MTKernelImage.from_elf(obj.lib)
    if max(image.metadata.kernel['.private_memory_size'],image.metadata.kernel['.local_mem_used_in_cluster']):
      raise NotImplementedError('MT kernels requiring spill or local-cluster memory are not supported')
    unchanged = MTLaunch._loaded is launch and launch.kernel_lib == obj.lib
    self.ensure_started(launch)
    if not unchanged:
      launch.set_generated(image)
      launch.kernel_lib = obj.lib
      launch.generate_static_pds()
      launch.generate_pds(image)
    if launch.mmu_dirty:
      launch.upload_mmu()
      launch.mmu_dirty = False
    launch.set_grid(global_size,local_size)

  def _view(self, va:int):
    if va in self.views: return self.views[va]
    for base, (view, meta) in self.views.items():
      if base <= va < base + meta.size: return view, meta
    raise RuntimeError(f'Unknown MT buffer address {va:#x}')

  def finalize(self):
    super().finalize()
    self.mt.close()
