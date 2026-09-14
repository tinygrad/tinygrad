from __future__ import annotations
from typing import cast
import functools, struct, operator
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, ceildiv, unwrap, to_tuple
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP, db_value, send_wqe, recv_wqe, WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU
from tinygrad.runtime.support.hcq2 import unwrap_view, patch
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping, MemoryManager
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, System
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

RDMA_CHUNK = 1 << 30 # a wqe length is 32 bits

class BNXTIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, index:int): super().__init__(dev, index, vram_bar=2, dev_impl_t=BNXTDev)
  def device_fini(self): self.dev_impl.fini()

  def storage(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> BufferStorage: # nic memory any gpu of the node maps
    va = MemoryManager.alloc_vaddr(size:=round_up(mem.nbytes, 0x1000), 0x1000)
    mapping = VirtMapping(va, size, [(p, 0x1000) for p in paddrs], AddrSpace.SYS, uncached=True, snooped=snooped)
    return BufferStorage(va, PCIAllocationMeta(mapping, True), mem)
  def buffer(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> Buffer:
    return Buffer(self.dev.device, mem.nbytes, dtypes.uint8, opaque=self.storage(mem, paddrs, snooped))

  @functools.cached_property
  def doorbell(self) -> Buffer:
    off = self.dev_impl.db_off & ~0xfff
    return self.buffer(self.pci_dev.map_bar(2, off=off, size=0x1000), [self.pci_dev.bar_info(2)[0] + off], snooped=False)

class BNXTAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: # sysmem: the counters, the batch slots, the timeline
    return self.dev.iface.storage(*self.dev.iface.pci_dev.alloc_sysmem(round_up(size, 0x1000)))
  def _copyin(self, dest:BufferStorage, src:memoryview): unwrap(dest.host)[:len(src)] = src
  def _map(self, buf:Buffer) -> BufferStorage: # a memory region over the buffer's pages, keyed at the gpu virtual address
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError("RDMA requires memory on its node")
    mapping = buf.meta.mapping # the nic reaches vram over pcie: the bar, even where gpus reach each other over xgmi
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else PCIIfaceBase.p2p_paddrs(iface, mapping.paddrs)[0]
    align = buf._buf | functools.reduce(operator.or_, (p | s for p, s in paddrs)) # every address a multiple of the page: fewer pbl entries
    log_page = max(l for l in (12, 13, 16, 18, 20, 21, 22, 30) if not align & ((1 << l) - 1)) # the page sizes the nic has
    key = self.dev.iface.dev_impl.register_mem([p + off for p, size in paddrs for off in range(0, size, 1 << log_page)],
                                              mapping.size, log_page, va=buf._buf)
    return BufferStorage(key, key)
  def _offset(self, buf, size:int, offset:int): return buf
  def _unmap(self, storage:BufferStorage): self.dev.iface.dev_impl.unregister_mem(storage.meta)

class RDMADevice(Compiled):
  ifaces = [BNXTIface]

  def __init__(self, device:str):
    self.iface = self._select_iface(device)
    self.qps:dict[tuple[str, str], BNXTQP] = {}
    self.bufs:dict[tuple[tuple[str, str], str], Buffer] = {} # a gpu pair's rings, cqs, counters and the doorbell
    self.words:dict[tuple[UOp, tuple[str, ...]], UOp] = {} # the runtime address words of the rings and cqs, per gpu
    super().__init__(device, BNXTAllocator(self), [], None)
    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="b"),
      lambda ctx, b: ctx.bufs[b.tag[1:]] if isinstance(b.tag, tuple) and b.tag[0] == "rdma" else None)]) + self.pm_bufferize

  def qp(self, pair:tuple[str, str], peer) -> BNXTQP: # one queue pair per gpu pair, on this nic and the peer's, with its rings and counters
    if pair not in self.qps:
      other = cast(RDMADevice, System.nic_for(peer))
      for nic in (self, other):
        nic.qps[pair] = q = BNXTQP(nic.iface.dev_impl)
        for name in ("sq", "rq", "scq", "rcq"): nic.bufs[pair, name] = nic.iface.buffer(getattr(q, name).ring, getattr(q, name).paddrs)
        for name in ("sq_seq", "rq_seq", "psn"): nic.bufs[pair, name] = Buffer(nic.device, 1, dtypes.uint64, initial_value=bytes(8))
        nic.bufs[pair, "db"] = nic.iface.doorbell
      for a, b in ((self, other), (other, self)): a.qps[pair].connect(b.qps[pair].qpn, b.iface.dev_impl.local_gid, b.iface.dev_impl.mac)
    return self.qps[pair]

  def arg(self, pair:tuple[str, str], name:str) -> UOp:
    b = self.bufs[pair, name]
    return UOp.placeholder((b.size,), b.dtype, 0, device=(self.device,), volatile=True, tag=("rdma", pair, name))

  def word(self, b:UOp, devs:tuple[str, ...]) -> UOp: # a ring or cq address as a gpu sees it, loaded at runtime: the counters are added to it
    word = UOp.placeholder((1,), dtypes.uint64, device=Device[devs[0]].host, volatile=True, tag="addr")
    return self.words.setdefault((b, devs), patch(word, [(0, b.getaddr(devs))]))

# *****************
# UOps implementation

def rdma_wire(call:UOp) -> UOp|None: # the wire of a copy between nodes: the operand on a nic
  if call.op is not Ops.CALL or call.src[0].op is not Ops.COPY: return None
  return next((b for b in get_call_arg_uops(call) if to_tuple(b.device)[0].startswith("RDMA")), None)
def is_rdma(call:UOp) -> bool: return rdma_wire(call) is not None

def queue_of(call:UOp) -> tuple[tuple[str, str], bool]: # the gpu pair and whether the call receives: the rq or the sq of their queue pair
  (dst, src), wire = get_call_arg_uops(call), unwrap(rdma_wire(call))
  gpu = to_tuple((dst if wire is src else src).device)[0]
  return (min(gpu, wire.tag), max(gpu, wire.tag)), wire is src

def ins(name:str, *src:UOp|int) -> UOp: # an op of the gpu's queue
  return UOp(Ops.INS, arg=(name, dtypes.void), src=tuple(UOp.const(s, dtypes.uint32) if isinstance(s, int) else s for s in src))

def rdma_copies(devs:tuple[str, ...], calls:list[UOp]) -> list[list[UOp]]: # the ops of each copy of a submit on one queue
  (pair, is_recv), nic = queue_of(calls[0]), cast(RDMADevice, Device[unwrap(rdma_wire(calls[0])).device])
  qp = nic.qp(pair, Device[next(p for p in pair if p != devs[0])])
  ring, cq, seq, psn = (nic.arg(pair, n) for n in (("rq", "rcq", "rq_seq", "psn") if is_recv else ("sq", "scq", "sq_seq", "psn")))
  bufs = [get_call_arg_uops(c)[0 if is_recv else 1] for c in calls]
  wqes, packets = sum(ceildiv(b.nbytes(), RDMA_CHUNK) for b in bufs), sum(ceildiv(b.nbytes(), MTU) for b in bufs)

  assert wqes <= min(RING_ENTRIES, CQ_ENTRIES), "a batch posts at most a ring of wqes per pair"

  # counters: loaded once, advanced once per submit
  n, p = seq.index(0).load(), psn.index(0).load()
  advances = [seq.index(0).store(n + wqes)] + ([] if is_recv else [psn.index(0).store(p + packets)])

  # addresses as the gpu sees them, the advances hang on the ring word
  ring_addr, cq_addr = nic.word(ring, devs).after(*advances).index(0).load(), nic.word(cq, devs).index(0).load()
  db = nic.arg(pair, "db").getaddr(devs) + (nic.iface.dev_impl.db_off & 0xfff)
  ring_db = db_value(qp.qpn, bnxt.DBC_DBC_TYPE_RQ if is_recv else bnxt.DBC_DBC_TYPE_SQ, 0, 0)
  cq_db = db_value(qp.rcq_id if is_recv else qp.scq_id, bnxt.DBC_DBC_TYPE_CQ, 0, 0)

  copies = []
  for buf in bufs:
    ops:list[UOp] = []
    for off in range(0, buf.nbytes(), RDMA_CHUNK): # a wqe per chunk, each completed
      size = min(RDMA_CHUNK, buf.nbytes() - off)

      # sdma fills in the wqe
      hdr, key = struct.unpack("<8I", (recv_wqe if is_recv else send_wqe)(0, 0, size)[:32]), unwrap_view(buf)[0].getaddr(nic.device)
      ops += [ins("write", ring_addr + (n % RING_ENTRIES) * WQE_SIZE, *hdr, buf.getaddr(devs) + off, key.cast(dtypes.uint32), size)]

      # a send also fills in its msn entry: the slot, the psn after it (a psn per packet), its first psn
      if not is_recv: ops += [ins("write", ring_addr + RING_ENTRIES * WQE_SIZE + (n % RING_ENTRIES) * 8,
                                  ((n % RING_ENTRIES) << 48) | (((p + ceildiv(size, MTU)) & 0xffffff) << 24) | (p & 0xffffff))]

      # rings the doorbell: the slot after the wqe and the epoch of its pass
      ops += [ins("store", db, ((n + 1) % RING_ENTRIES | ((n + 1) // RING_ENTRIES & 1) << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT) | ring_db)]

      # waits for the cqe, acks the cq
      ops += [ins("wait_eq", cq_addr + (n % CQ_ENTRIES) * 32 + 24, (n // CQ_ENTRIES & 1 ^ 1 | (2 if is_recv else 0)).cast(dtypes.uint16)),
              ins("store", db, ((n + 1) % CQ_ENTRIES | ((n + 1) // CQ_ENTRIES & 1) << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT) | cq_db)]
      n, p = n + 1, p + ceildiv(size, MTU)
    copies.append(ops + ([ins("barrier")] if is_recv else [])) # a receive invalidates the gpu caches
  return copies

# *****************
# encode rewrite

def rdma_submit(ctx, submit:UOp, lin:UOp) -> UOp|None: # the copies between nodes of a submit become its ops
  if not (calls:=[c for c in lin.src if is_rdma(c)]): return None
  posts:dict[int, list[UOp]] = {} # by position: a submit may repeat a call
  for k in dict.fromkeys(map(queue_of, calls)):
    idx = [i for i, c in enumerate(lin.src) if is_rdma(c) and queue_of(c) == k]
    posts |= dict(zip(idx, rdma_copies(lin.arg[0], [lin.src[i] for i in idx])))
  return submit.replace(src=(lin.replace(src=tuple(o for i, u in enumerate(lin.src) for o in posts.get(i, [u]))),))
pm_rdma_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), rdma_submit)])
