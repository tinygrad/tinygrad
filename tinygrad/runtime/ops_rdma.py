from __future__ import annotations
from typing import cast
import functools, struct, operator
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, ceildiv, unwrap, flatten, to_tuple
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP, db_value, send_wqe, recv_wqe, WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU
from tinygrad.runtime.support.hcq2 import unwrap_view, is_rdma, rdma_wire, patch, to_name, EncodeCtx
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping, MemoryManager
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, System
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

RDMA_CHUNK = 1 << 30 # a send length is 32 bits

class BNXTIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, index:int):
    super().__init__(dev, index, vram_bar=2, dev_impl_t=BNXTDev)

  def device_fini(self): self.dev_impl.fini()

  # nic memory any gpu of the node maps: sysmem rings, counters and batch slots, the doorbell page of the bar
  def storage(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> BufferStorage:
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
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: # host words: the queues' counters, the batch slots, the timeline
    return self.dev.iface.storage(*self.dev.iface.pci_dev.alloc_sysmem(round_up(size, 0x1000)))
  def _copyin(self, dest:BufferStorage, src:memoryview): unwrap(dest.host)[:len(src)] = src
  def _copyout(self, dest:memoryview, src:BufferStorage): dest[:] = unwrap(src.host)[:len(dest)]
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
    self.words:dict[tuple[UOp, tuple[str, ...]], UOp] = {} # the runtime address words of the rings and cqs, per gpu
    self.counters:dict[str, Buffer] = {} # the queues' slot and psn counters: state of the device, like the gpus' queue pointers
    super().__init__(device, BNXTAllocator(self), [], None)
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.bufs[b.tag[1:]] if isinstance(b.tag, tuple) and b.tag[0] == "rdma" else None),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.counter(b.tag) if isinstance(b.tag, str) and b.tag.startswith(("seq_", "psn_")) else None),
    ]) + self.pm_bufferize

  def counter(self, tag:str) -> Buffer: return self.counters.setdefault(tag, Buffer(self.device, 1, dtypes.uint64, initial_value=bytes(8)))

  def qp(self, pair:tuple[str, str], peer) -> BNXTQP: # one queue pair per gpu pair, on this nic and the peer's, with their rings
    if pair not in self.qps:
      other = cast(RDMADevice, System.nic_for(peer))
      for nic in (self, other):
        nic.qps[pair] = q = BNXTQP(nic.iface.dev_impl)
        for name in ("sq", "rq", "scq", "rcq"): nic.bufs[pair, name] = nic.iface.buffer(getattr(q, name).ring, getattr(q, name).paddrs)
        nic.bufs[pair, "db"] = nic.iface.doorbell
      for a, b in ((self, other), (other, self)): a.qps[pair].connect(b.qps[pair].qpn, b.iface.dev_impl.local_gid, b.iface.dev_impl.mac)
    return self.qps[pair]

  def word(self, b:UOp, devs:tuple[str, ...]) -> UOp: # the runtime address word of a ring or cq, as a gpu sees it
    word = UOp.placeholder((1,), dtypes.uint64, device=Device[devs[0]].host, volatile=True, tag="addr")
    return self.words.setdefault((b, devs), patch(word, [(0, b.getaddr(devs))]))

  def arg(self, pair:tuple[str, str], name:str) -> UOp:
    b = self.bufs[pair, name]
    return UOp.placeholder((b.size,), b.dtype, 0, device=(self.device,), volatile=True, tag=("rdma", pair, name))

def ins(name:str, *src:UOp|int) -> UOp: # an op of the gpu's queue
  return UOp(Ops.INS, arg=(name, dtypes.void), src=tuple(s if isinstance(s, UOp) else UOp.const(s, dtypes.uint32) for s in src))

def side(call:UOp) -> tuple[tuple[str, str], UOp, bool]: # the gpu pair (the wire's tag), the gpu's buffer and whether it receives
  (dst, src), wire = get_call_arg_uops(call), unwrap(rdma_wire(call))
  return wire.tag[1], (dst if wire is src else src), wire is src

def chunks(buf:UOp) -> list[int]: return [min(RDMA_CHUNK, buf.nbytes() - off) for off in range(0, buf.nbytes(), RDMA_CHUNK)] # a wqe each
def packets(size:int) -> int: return max(1, ceildiv(size, MTU))

class RDMASide: # a queue pair's side in a submit: the slot counter is loaded once and advanced once, the psn by every packet of the sends
  def __init__(self, nic:RDMADevice, pair:tuple[str, str], devs:tuple[str, ...], recv:bool, bufs:list[UOp]):
    self.nic, self.devs, self.recv = nic, devs, recv
    self.qp = self.nic.qp(pair, Device[next(p for p in pair if Device.canonicalize(p) != devs[0])])
    ring, cq = (self.nic.arg(pair, n) for n in (("rq", "rcq") if recv else ("sq", "scq")))
    self.cq_id, self.typ = (self.qp.rcq_id, bnxt.DBC_DBC_TYPE_RQ) if recv else (self.qp.scq_id, bnxt.DBC_DBC_TYPE_SQ)
    sizes = [s for b in bufs for s in chunks(b)]
    assert len(sizes) <= min(RING_ENTRIES, CQ_ENTRIES), "a batch posts at most a ring of wqes per pair"
    seq, psn = (UOp.placeholder((1,), dtypes.uint64, 0, device=(self.nic.device,), volatile=True, tag=to_name(n, *pair, "rq" if recv else "sq"))
                for n in ("seq", "psn"))
    self.n, self.p = seq.index(0).load(), psn.index(0).load()
    advances = [seq.index(0).store(self.n + len(sizes))] + ([] if recv else [psn.index(0).store(self.p + sum(map(packets, sizes)))])
    self.ring_addr, self.cq_addr = self.nic.word(ring, devs).after(*advances).index(0).load(), self.nic.word(cq, devs).index(0).load()
    self.db = self.nic.arg(pair, "db").getaddr(devs) + (self.nic.iface.dev_impl.db_off & 0xfff)

  def copy(self, buf:UOp) -> tuple[list[UOp], list[UOp]]: # write the wqes and ring the nic; then what completes it, owed at the next signal
    key, ops = unwrap_view(buf)[0].getaddr(self.nic.device).cast(dtypes.uint32), []
    for off, size in zip(range(0, buf.nbytes(), RDMA_CHUNK), chunks(buf)):
      hdr = struct.unpack("<8I", (recv_wqe if self.recv else send_wqe)(0, 0, size)[:32])
      ops.append(ins("write", self.ring_addr + (self.n % RING_ENTRIES) * WQE_SIZE, *hdr, buf.getaddr(self.devs) + off, key, size))
      if not self.recv: # the msn entry of the send: its slot, the psn after it and its first psn
        msn = ((self.n % RING_ENTRIES) << 48) | (((self.p + packets(size)) & 0xffffff) << 24) | (self.p & 0xffffff)
        ops.append(ins("write", self.ring_addr + RING_ENTRIES * WQE_SIZE + (self.n % RING_ENTRIES) * 8, msn))
        self.p = self.p + packets(size)
      # the doorbell is a signal: an end of pipe write, after the data the nic reads is in memory. a plain cp write does not ring it
      ops.append(ins("store", self.db, db_value(self.qp.qpn, self.typ, (self.n + 1) % RING_ENTRIES, (self.n + 1) // RING_ENTRIES & 1)))
      self.n = self.n + 1
    last, cqe = self.n - 1, (self.n - 1) // CQ_ENTRIES & 1 ^ 1 | (2 if self.recv else 0) # the last cqe: toggle of its pass, type, status 0
    done = [ins("wait_eq", self.cq_addr + (last % CQ_ENTRIES) * 32 + 24, cqe.cast(dtypes.uint16)),
            ins("store", self.db, db_value(self.cq_id, bnxt.DBC_DBC_TYPE_CQ, (last + 1) % CQ_ENTRIES, (last + 1) // CQ_ENTRIES & 1))]
    return ops, done + ([ins("barrier")] if self.recv else []) # the gpu caches see what the nic wrote

def rdma_submit(ctx:EncodeCtx, submit:UOp, lin:UOp) -> UOp|None: # the copies between nodes of a submit become its ops
  if not any(is_rdma(c) for c in lin.src): return None
  nic = cast(RDMADevice, Device[to_tuple(unwrap(next(map(rdma_wire, filter(is_rdma, lin.src)))).device)[0]]) # the node's nic
  sides = {(p, r): RDMASide(nic, p, lin.arg[0], r, [side(c)[1] for c in lin.src if is_rdma(c) and side(c)[0] == p and side(c)[2] == r])
           for p, _, r in map(side, filter(is_rdma, lin.src))}
  ops, pending = list[UOp](), {} # what completes the posted copies is owed before a signal, and before an op of this queue that touches their buffers
  for u in lin.src:
    if is_rdma(u):
      pair, buf, recv = side(u)
      posts, done = sides[(pair, recv)].copy(buf)
      ops, pending[(pair, recv)] = ops + posts, (unwrap_view(buf)[0], done)
      continue
    used = [unwrap_view(b)[0] for b in get_call_arg_uops(u)] if u.op is Ops.CALL else []
    if (u.op is Ops.INS and u.arg[0] == "store") or any(b in used for b, _ in pending.values()):
      ops, pending = ops + flatten(d for _, d in pending.values()), {}
    ops.append(u)
  return submit.replace(src=(lin.replace(src=tuple(ops + flatten(d for _, d in pending.values()))),))
pm_rdma_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), rdma_submit)])
