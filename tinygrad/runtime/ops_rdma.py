from __future__ import annotations
from typing import cast, Any
import functools, struct, operator, signal, sys
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, getenv, ceildiv, to_tuple
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.am.amdev import AMMemoryManager
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP, db_value, send_wqe, recv_wqe, WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU
from tinygrad.runtime.support.hcq2 import unwrap_view, nic_for
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

RDMA_CHUNK = 1 << 30 # a send length is 32 bits
NIC = (0x14e4, ((0xffff, (0x1760,)),), 0x02) # BCM57608: vendor, device ids, base class

class BNXTIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, index:int):
    super().__init__(dev, index, *NIC[:2], vram_bar=2, va_start=AMMemoryManager.va_allocator.base, va_size=AMMemoryManager.va_allocator.size,
      dev_impl_t=functools.partial(BNXTDev, ip=getenv("BNXT_IP", f"10.0.0.{index + 1}")), base_class=NIC[2])
    # a kill (timeout, pkill) still runs the atexit finalizers: a driver the firmware never forgot wedges the nic for every next driver
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL: signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

  def device_fini(self): self.dev_impl.fini()

  # nic memory as a buffer any gpu of the node maps: sysmem rings and counters, the doorbell page of the bar
  def buffer(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> Buffer:
    va = AMMemoryManager.alloc_vaddr(size:=round_up(mem.nbytes, 0x1000), 0x1000)
    mapping = VirtMapping(va, size, [(p, 0x1000) for p in paddrs], AddrSpace.SYS, uncached=True, snooped=snooped)
    return Buffer(self.dev.device, mem.nbytes, dtypes.uint8, opaque=BufferStorage(va, PCIAllocationMeta(mapping, True), mem))

  @functools.cached_property
  def doorbell(self) -> Buffer:
    off = self.dev_impl.db_off & ~0xfff
    return self.buffer(self.pci_dev.map_bar(2, off=off, size=0x1000), [self.pci_dev.bar_info(2)[0] + off], snooped=False)

class BNXTAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: raise RuntimeError("RDMA devices only map buffers")
  def _map(self, buf:Buffer) -> BufferStorage: # a memory region over the buffer's pages, keyed at the gpu virtual address
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError("RDMA requires memory on its node")
    mapping = buf.meta.mapping
    # the nic reaches vram over pcie: the bar, even where gpus reach each other over xgmi
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else PCIIfaceBase.p2p_paddrs(iface, mapping.paddrs)[0]
    align = buf._buf | functools.reduce(operator.or_, (p | s for p, s in paddrs)) # every address a multiple of the page: fewer pbl entries
    log_page = max(l for l in (12, 13, 16, 18, 20, 21, 22, 30) if not align & ((1 << l) - 1)) # the page sizes the nic has
    key = self.dev.iface.dev_impl.register_mem([p + off for p, size in paddrs for off in range(0, size, 1 << log_page)],
                                              mapping.size, log_page, va=buf._buf)
    return BufferStorage(key, key)
  def _offset(self, buf, size:int, offset:int): return buf
  def _unmap(self, storage:BufferStorage): self.dev.iface.dev_impl.unregister_mem(storage.meta)

class RDMADevice(Compiled):
  has_copy_queue = False
  ifaces = [BNXTIface]

  def __init__(self, device:str):
    self.iface = self._select_iface(device)
    self.qps:dict[tuple[str, str], BNXTQP] = {}
    self.bufs:dict[tuple[tuple[str, str], str], Buffer] = {}
    super().__init__(device, BNXTAllocator(self), [], None)
    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="b"),
      lambda ctx, b: ctx.bufs[b.tag[1:]] if isinstance(b.tag, tuple) and b.tag[0] == "rdma" else None)]) + self.pm_bufferize

  def synchronize(self, timeout:int|None=None):
    for d in {d for pair in self.qps for d in pair if Device[d].peer_group == self.peer_group}: Device[d].synchronize(timeout)

  def qp(self, pair:tuple[str, str], peer) -> BNXTQP: # one queue pair per gpu pair, on this nic and the peer's, with their rings and counters
    if pair not in self.qps:
      other = cast(RDMADevice, nic_for(peer))
      for nic in (self, other):
        nic.qps[pair] = q = BNXTQP(nic.iface.dev_impl)
        for name in ("sq", "rq", "scq", "rcq"): nic.bufs[pair, name] = nic.iface.buffer(getattr(q, name).ring, getattr(q, name).paddrs)
        for name in ("sq_seq", "rq_seq", "psn"): # zeroed sequence numbers in fresh sysmem
          nic.bufs[pair, name] = nic.iface.buffer(*nic.iface.pci_dev.alloc_sysmem(0x1000)).view(1, dtypes.uint64, 0).ensure_allocated()
        nic.bufs[pair, "db"] = nic.iface.doorbell
      for a, b in ((self, other), (other, self)): a.qps[pair].connect(b.qps[pair].qpn, b.iface.dev_impl.local_gid, b.iface.dev_impl.mac)
    return self.qps[pair]

  def arg(self, pair:tuple[str, str], name:str) -> UOp:
    b = self.bufs[pair, name]
    return UOp.placeholder((b.size,), b.dtype, 0, device=(self.device,), volatile=True, tag=("rdma", pair, name))

  # one side of a copy between nodes, on the queue of its gpu: write the wqe, ring the nic; wait for completion
  def copy(self, hq:Any, call:UOp):
    dst, src = get_call_arg_uops(call)
    recv = call.src[0].arg == "recv"
    buf, peer = (dst, Device[to_tuple(src.device)[0]]) if recv else (src, Device[to_tuple(dst.device)[0]])
    qp = self.qp(pair:=tuple(sorted((hq.dev.device, peer.device))), peer)
    ring, seq, cq = (self.arg(pair, n) for n in (("rq", "rq_seq", "rcq") if recv else ("sq", "sq_seq", "scq")))
    ring_addr, cq_addr = hq.rt(ring, hq.devs), hq.rt(cq, hq.devs)
    db = self.arg(pair, "db").getaddr(hq.devs) + (self.iface.dev_impl.db_off & 0xfff)
    key = unwrap_view(buf)[0].getaddr(self.device).cast(dtypes.uint32)
    for off in range(0, buf.nbytes(), RDMA_CHUNK):
      size = min(RDMA_CHUNK, buf.nbytes() - off)
      n = hq.bump(seq) # the operation's slot in the ring and, as every operation completes with one cqe, in the cq
      hdr = struct.unpack("<8I", (recv_wqe if recv else send_wqe)(0, 0, size)[:32])
      hq.write(ring_addr + (n % RING_ENTRIES) * WQE_SIZE, *hdr, buf.getaddr(hq.devs) + off, key, UOp.const(size, dtypes.uint32))
      if not recv: # the msn entry of the send
        psn = hq.bump(self.arg(pair, "psn"), packets:=max(1, ceildiv(size, MTU)))
        nxt = psn + packets
        hq.write(ring_addr + 0x1000 + (n % RING_ENTRIES) * 8, ((n % RING_ENTRIES) << 48) | ((nxt & 0xffffff) << 24) | (psn & 0xffffff))
      # the doorbell is a signal: an end of pipe write, after the data the nic reads is in memory. a plain cp write does not ring it
      hq.signal(db, db_value(qp.qpn, bnxt.DBC_DBC_TYPE_RQ if recv else bnxt.DBC_DBC_TYPE_SQ, (n + 1) % RING_ENTRIES, (n + 1) // RING_ENTRIES & 1))
      self.wait(hq, qp, recv, cq_addr, db, n)

  def wait(self, hq:Any, qp:BNXTQP, recv:bool, cq_addr:UOp, db:UOp, n:UOp): # the cqe: toggle of this pass, type (RES_RC for a receive), status 0
    hq.wait(cq_addr + (n % CQ_ENTRIES) * 32 + 24, (((n // CQ_ENTRIES) & 1) ^ 1 | (2 if recv else 0)).cast(dtypes.uint16), eq=True)
    hq.signal(db, db_value(qp.rcq_id if recv else qp.scq_id, bnxt.DBC_DBC_TYPE_CQ, (n + 1) % CQ_ENTRIES, (n + 1) // CQ_ENTRIES & 1))
    if recv: hq.memory_barrier() # the gpu caches see what the nic wrote
