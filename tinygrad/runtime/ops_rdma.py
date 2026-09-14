from __future__ import annotations
from typing import cast
import functools, struct, operator
from dataclasses import dataclass
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, ceildiv, unwrap, flatten, to_tuple
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP, db_value, send_wqe, recv_wqe, WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU
from tinygrad.runtime.support.hcq2 import unwrap_view, is_rdma, rdma_wire, patch
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping, MemoryManager
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, System
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

RDMA_CHUNK = 1 << 30 # a send length is 32 bits

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
  def word(self, b:UOp, devs:tuple[str, ...]) -> UOp: # the runtime address word of a ring or cq, as a gpu sees it
    word = UOp.placeholder((1,), dtypes.uint64, device=Device[devs[0]].host, volatile=True, tag="addr")
    return self.words.setdefault((b, devs), patch(word, [(0, b.getaddr(devs))]))

# *****************
# UOps implementation

@dataclass(frozen=True)
class RDMAPair: nic:RDMADevice; qp:BNXTQP; recv:bool; devs:tuple[str, ...]; ring:UOp; cq:UOp; db:UOp # noqa: E702

# a side of a copy between nodes is a copy with the wire: a placeholder on the node's nic, tagged with the gpu pair
def chunks(buf:UOp) -> list[int]: return [min(RDMA_CHUNK, buf.nbytes() - off) for off in range(0, buf.nbytes(), RDMA_CHUNK)] # a wqe each

def packets(size:int) -> int: return max(1, ceildiv(size, MTU))

def ins(name:str, *src:UOp|int) -> UOp: # an op of the gpu's queue
  return UOp(Ops.INS, arg=(name, dtypes.void), src=tuple(UOp.const(s, dtypes.uint32) if isinstance(s, int) else s for s in src))

def side(call:UOp) -> tuple[tuple[str, str], UOp, bool]: # the gpu pair, the gpu's buffer and whether it receives
  (dst, src), wire = get_call_arg_uops(call), unwrap(rdma_wire(call))
  return wire.tag[1], (dst if wire is src else src), wire is src

def rdma_pair(devs:tuple[str, ...], calls:list[UOp]) -> tuple[RDMAPair, UOp, UOp]: # the side and its counters: the slot and the psn
  (pair, _, recv), nic = side(calls[0]), cast(RDMADevice, Device[to_tuple(unwrap(rdma_wire(calls[0])).device)[0]])
  qp = nic.qp(pair, Device[next(p for p in pair if Device.canonicalize(p) != devs[0])])
  ring, cq, seq, psn = (nic.arg(pair, n) for n in (("rq", "rcq", "rq_seq", "psn") if recv else ("sq", "scq", "sq_seq", "psn")))
  sizes = [s for c in calls for s in chunks(side(c)[1])]
  assert len(sizes) <= min(RING_ENTRIES, CQ_ENTRIES), "a batch posts at most a ring of wqes per pair"
  n, p = seq.index(0).load(), psn.index(0).load() # loaded once and advanced once: by the wqes of the submit, by every packet of its sends
  advances = [seq.index(0).store(n + len(sizes))] + ([] if recv else [psn.index(0).store(p + sum(map(packets, sizes)))])
  db = nic.arg(pair, "db").getaddr(devs) + (nic.iface.dev_impl.db_off & 0xfff)
  return RDMAPair(nic, qp, recv, devs, nic.word(ring, devs).after(*advances).index(0).load(), nic.word(cq, devs).index(0).load(), db), n, p

def rdma_copy(pc:RDMAPair, buf:UOp, n:UOp, p:UOp) -> tuple[list[UOp], list[UOp], UOp, UOp]: # the wqes and doorbells, the completion, n and p after
  key, typ = unwrap_view(buf)[0].getaddr(pc.nic.device).cast(dtypes.uint32), bnxt.DBC_DBC_TYPE_RQ if pc.recv else bnxt.DBC_DBC_TYPE_SQ
  ops:list[UOp] = []
  for off, size in zip(range(0, buf.nbytes(), RDMA_CHUNK), chunks(buf)):
    hdr = struct.unpack("<8I", (recv_wqe if pc.recv else send_wqe)(0, 0, size)[:32])
    ops.append(ins("write", pc.ring + (n % RING_ENTRIES) * WQE_SIZE, *hdr, buf.getaddr(pc.devs) + off, key, size))
    if not pc.recv: # the msn entry: the slot, the psn after the send, its first psn
      msn = ((n % RING_ENTRIES) << 48) | (((p + packets(size)) & 0xffffff) << 24) | (p & 0xffffff)
      ops.append(ins("write", pc.ring + RING_ENTRIES * WQE_SIZE + (n % RING_ENTRIES) * 8, msn))
      p = p + packets(size)
    ops.append(ins("store", pc.db, db_value(pc.qp.qpn, typ, (n + 1) % RING_ENTRIES, (n + 1) // RING_ENTRIES & 1))) # the doorbell, end of pipe
    n = n + 1
  last, cqe = n - 1, (n - 1) // CQ_ENTRIES & 1 ^ 1 | (2 if pc.recv else 0) # the last cqe: toggle of its pass, type, status 0. then the cq's index
  cq_id = pc.qp.rcq_id if pc.recv else pc.qp.scq_id
  done = [ins("wait_eq", pc.cq + (last % CQ_ENTRIES) * 32 + 24, cqe.cast(dtypes.uint16)),
          ins("store", pc.db, db_value(cq_id, bnxt.DBC_DBC_TYPE_CQ, (last + 1) % CQ_ENTRIES, (last + 1) // CQ_ENTRIES & 1))]
  return ops, done + ([ins("barrier")] if pc.recv else []), n, p # a receive's barrier: the gpu caches see what the nic wrote

# *****************
# encode rewrite

def rdma_submit(ctx, submit:UOp, lin:UOp) -> UOp|None: # the copies between nodes of a submit become its ops
  groups:dict[tuple, list[UOp]] = {}
  for c in filter(is_rdma, lin.src): groups.setdefault((side(c)[0], side(c)[2]), []).append(c)
  if not groups: return None
  pairs, ops, pending = {k: rdma_pair(lin.arg[0], cs) for k, cs in groups.items()}, list[UOp](), {}
  for u in lin.src: # what completes the posted copies is owed before a signal, and before an op of the queue that touches their buffers
    if is_rdma(u):
      key, buf = (side(u)[0], side(u)[2]), side(u)[1]
      posts, done, n, p = rdma_copy(*pairs[key][:1], buf, *pairs[key][1:])
      ops, pending[key], pairs[key] = ops + posts, (unwrap_view(buf)[0], done), (pairs[key][0], n, p)
      continue
    used = [unwrap_view(b)[0] for b in get_call_arg_uops(u)] if u.op is Ops.CALL else []
    if (u.op is Ops.INS and u.arg[0] == "store") or any(b in used for b, _ in pending.values()):
      ops, pending = ops + flatten(d for _, d in pending.values()), {}
    ops.append(u)
  return submit.replace(src=(lin.replace(src=tuple(ops + flatten(d for _, d in pending.values()))),))
pm_rdma_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), rdma_submit)])
