import unittest, struct
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad import Device, dtypes
from tinygrad.device import Buffer, BufferStorage
from tinygrad.runtime.ops_rdma import BNXTAllocator
from tinygrad.runtime.support import hcq2
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import send_wqe, recv_wqe, msn_entry, db_value, RING_ENTRIES, CQ_ENTRIES
from tinygrad.runtime import ops_rdma
from tinygrad.engine import realize
from tinygrad.runtime.support.memory import AddrSpace, VirtMapping
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, System
from tinygrad.uop.ops import UOp, Ops, ProgramInfo, KernelInfo, graph_rewrite

class TestBNXTAllocator(unittest.TestCase):
  def setUp(self):
    self.iface = PCIIfaceBase.__new__(PCIIfaceBase)
    self.iface.pci_dev, self.iface.vram_bar = SimpleNamespace(peer_group="node", bar_info=lambda bar: (0x100000000, 1 << 40)), 0 # the bar
    self.nic = Mock(peer_group="node", iface=SimpleNamespace(dev_impl=Mock()))
    self.nic.iface.dev_impl.register_mem.return_value = 0x1234
    self.nic.allocator = BNXTAllocator(self.nic)
    gpu = SimpleNamespace(iface=self.iface, allocator=Mock(_offset=lambda b, size, off: b + off))
    self.lookup = patch.object(type(Device), "__getitem__", lambda _, d: {"AMD": gpu, "RDMA": self.nic}[d])
    self.lookup.start()
    self.addCleanup(self.lookup.stop)

  def buffer(self, pages, va=0x200000, aspace=AddrSpace.PHYS):
    size = sum(s for _, s in pages)
    buf = Buffer("AMD", size, dtypes.uint8, opaque=BufferStorage(va, PCIAllocationMeta(VirtMapping(va, size, pages, aspace), False)))
    self.addCleanup(buf.deallocate)
    return buf

  def test_vram_huge_pages_and_views(self):
    buf = self.buffer([(0x400000, 0x400000)])
    view = buf.view(16, dtypes.uint8, 128).ensure_allocated()
    self.addCleanup(view.deallocate)
    self.assertEqual((view.get_buf("AMD"), view.get_buf("RDMA"), buf.get_buf("RDMA")), (0x200080, 0x1234, 0x1234))
    self.nic.iface.dev_impl.register_mem.assert_called_once_with([0x100400000, 0x100600000], 0x400000, 21, va=0x200000)
    self.nic.allocator._unmap(buf.get_storage("RDMA"))
    self.nic.iface.dev_impl.unregister_mem.assert_called_once_with(0x1234)

  def test_fragmented_pages(self):
    buf = self.buffer([(0x401000, 0x1000), (0x800000, 0x2000)])
    self.assertEqual(buf.get_buf("RDMA"), 0x1234)
    self.nic.iface.dev_impl.register_mem.assert_called_once_with([0x100401000, 0x100800000, 0x100801000], 0x3000, 12, va=0x200000)

  def test_sysmem_does_not_add_bar(self):
    self.buffer([(0x401000, 0x2000)], aspace=AddrSpace.SYS).get_buf("RDMA")
    self.nic.iface.dev_impl.register_mem.assert_called_once_with([0x401000, 0x402000], 0x2000, 12, va=0x200000)

  def test_reject_other_node(self):
    self.nic.peer_group = "other"
    with self.assertRaisesRegex(RuntimeError, "memory on its node"): self.buffer([(0x400000, 0x1000)]).get_buf("RDMA")
    self.nic.iface.dev_impl.register_mem.assert_not_called()

def copy(src, dst): return src.copy_to_device(dst.device).call(dst, src)
def buf(slot, device, size=16): return UOp.param(slot, dtypes.uint8, size, device)
def kernel(b, write=True): return UOp(Ops.PROGRAM, arg=ProgramInfo(outs=(0,) if write else (), ins=() if write else (0,))).call(b)

class TestRDMASchedule(unittest.TestCase):
  def setUp(self):
    self.devs = {d: SimpleNamespace(device=d, peer_group=g, host="CPU", has_copy_queue=True, pm_batch=None)
                 for d, g in (("AMD:1", "a"), ("AMD:2", "b"), ("AMD:3", "a"), ("RDMA:0", "a"), ("RDMA:1", "b"))}
    get_device = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), "__getitem__", lambda obj, d: self.devs[d] if d in self.devs else get_device(obj, d)))
    self.enterContext(patch.object(System, "nic_for", lambda dev: self.devs["RDMA:0" if dev.peer_group == "a" else "RDMA:1"]))

  def prepare(self, calls): return graph_rewrite(UOp(Ops.LINEAR, src=tuple(calls)), hcq2.pm_insert_copy_staging+realize.pm_flatten_linear)

  def test_split(self): # a copy between nodes: a send to the source's nic and a receive from the destination's, each on its gpu's queue
    src, dst = buf(0, "AMD:1"), buf(1, "AMD:2")
    send, recv = self.prepare([copy(src, dst)]).src
    self.assertTrue(ops_rdma.is_rdma(send) and ops_rdma.is_rdma(recv))
    self.assertEqual([ops_rdma.rdma_wire(c).device for c in (send, recv)], ["RDMA:0", "RDMA:1"])
    self.assertEqual([ops_rdma.queue_of(c) for c in (send, recv)], [(("AMD:1", "AMD:2"), False), (("AMD:1", "AMD:2"), True)])
    self.assertEqual([realize.get_call_arg_uops(c)[i] for c, i in ((send, 1), (recv, 0))], [src, dst]) # the gpu's end: a send's src, a receive's dst
    self.assertEqual((hcq2.get_enqueue_devs(send), hcq2.get_enqueue_devs(recv)), ("AMD:1", "AMD:2"))
    self.assertIsNone(hcq2.stage_copy((), send, *send.src[1:]))
    self.assertIsNone(hcq2.split_rdma(c:=copy(buf(2, "AMD:1"), buf(3, "AMD:3")), *c.src[1:])) # inside a node: not split
    with patch.object(System, "nic_for", return_value=None): # a node without a nic: not split
      self.assertIsNone(hcq2.split_rdma(c:=copy(buf(4, "AMD:1"), buf(5, "AMD:2")), *c.src[1:]))

  def test_each_node_has_its_side(self):
    src, dst = buf(0, "AMD:1"), buf(1, "AMD:2")
    batches = hcq2.sched_batches(self.prepare([copy(src, dst), copy(buf(2, "AMD:3"), dst)]), False).src
    self.assertEqual([(b.arg.aux.device, b.arg.aux.skip_wait) for b in batches], [(("AMD:1", "AMD:3"), True), (("AMD:2",), False)])

class TestBNXTCopy(unittest.TestCase):
  def test_words_replay(self): # a send and a receive as ops of the gpu queue, linked and run: rings and cqs wrap
    for recv in (False, True):
      rings = {n: Buffer("CPU", RING_ENTRIES * 128 + RING_ENTRIES * 8, dtypes.uint8, preallocate=True) for n in ("sq", "rq", "scq", "rcq", "db")}
      args = {n: UOp.from_buffer(b) for n, b in rings.items()} # addressed, never written here
      args |= {n: UOp.placeholder((1,), dtypes.uint64, 0, device="CPU", volatile=True, tag=n) for n in ("sq_seq", "rq_seq", "psn")}
      nic = SimpleNamespace(device="CPU", iface=SimpleNamespace(dev_impl=SimpleNamespace(db_off=0)),
                            qp=lambda pair, peer: SimpleNamespace(qpn=5, scq_id=6, rcq_id=7))
      self.enterContext(patch.object(ops_rdma, "rdma_mem", lambda nic, pair, name, *_: args[name]))
      src, dst = Buffer("CPU:1", 4096, dtypes.uint8, preallocate=True), Buffer("CPU", 4096, dtypes.uint8, preallocate=True) # "nodes" CPU:1 and CPU
      wire = UOp.placeholder((4096,), dtypes.uint8, 0, device="RDMA:0", tag="CPU:1" if recv else "CPU") # the far gpu
      call = (UOp.from_buffer(src).copy_to_device("CPU").call(UOp.from_buffer(dst), wire) if recv else
              UOp.from_buffer(src).copy_to_device("RDMA:0").call(wire, UOp.from_buffer(src)))
      signal = UOp(Ops.INS, arg=("store", dtypes.void), src=(args["db"], UOp.const(1, dtypes.uint64)))
      submit = hcq2.make_submit(call, signal, devs=("CPU",) if recv else ("CPU:1",), queue="COPY:0")
      get_device = type(Device).__getitem__
      with patch.object(type(Device), "__getitem__", lambda obj, d: nic if d == "RDMA:0" else get_device(obj, d)):
        ops = list(ops_rdma.rdma_submit(hcq2.EncodeCtx(("CPU",)), submit, submit.src[0]).src[0].src)
      self.assertIs(ops.pop(), signal) # the copy, then what completes it, then the signal
      words = [w for u in ops for w in u.src] # every word of the queue's writes, doorbells and waits, at its own width, read back as u64
      checks = UOp.placeholder((8 * len(words),), dtypes.uint8, device="CPU", tag="checks")
      out = hcq2.patch(checks, [(8 * i, w) for i, w in enumerate(words)], bytes(8 * len(words)))
      lowered = hcq2.lower_call(UOp.sink(out.index(0).load(), arg=KernelInfo("bnxt_copy_test"), tag=1).call(aux=hcq2.HCQInfo(("CPU",))))
      linked = hcq2.hcq_link(realize.lower_and_compile(UOp(Ops.LINEAR, src=(lowered,))), allow_cache=False)
      bufs = {p.tag: b.buffer for p, b in zip(lowered.without_after.src[1:], linked.src[0].without_after.src[1:])}
      ring, cq, data = ("rq", "rcq", dst) if recv else ("sq", "scq", src)
      seq = bufs[f"{ring}_seq"].host.view(fmt="Q")
      base, psn0 = seq[0], 0 if recv else bufs["psn"].host.view(fmt="Q")[0] # the counters persist across links
      for it in range(130):
        realize.run_linear(linked, jit=True)
        self.assertEqual(seq[0], base + it + 1)
        w, i = bufs["checks"].host.view(fmt="Q")[:], base + it
        # the wqe: the slot address, 8 header dwords, va, key, size; a send's msn entry; the doorbell
        self.assertEqual(w[0], rings[ring]._buf + i % RING_ENTRIES * 128)
        expect = (recv_wqe if recv else send_wqe)(data._buf, data._buf & 0xffffffff, 4096)
        self.assertEqual(struct.pack("<8I", *w[1:9]) + struct.pack("<QII", *w[9:12]), expect)
        w = w[12:]
        if not recv:
          self.assertEqual(w[:2], [rings[ring]._buf + RING_ENTRIES * 128 + i % RING_ENTRIES * 8, msn_entry(i, psn0 + i - base, 4096)[0]])
          w = w[2:]
        typ = bnxt.DBC_DBC_TYPE_RQ if recv else bnxt.DBC_DBC_TYPE_SQ
        self.assertEqual(w[1], db_value(5, typ, (i + 1) % RING_ENTRIES, (i + 1) // RING_ENTRIES & 1))
        # then the completion: the cqe and the cq doorbell
        self.assertEqual((w[2], w[3]), (rings[cq]._buf + i % CQ_ENTRIES * 32 + 24, (i // CQ_ENTRIES & 1) ^ 1 | (2 if recv else 0)))
        self.assertEqual(w[5], db_value(7 if recv else 6, bnxt.DBC_DBC_TYPE_CQ, (i + 1) % CQ_ENTRIES, (i + 1) // CQ_ENTRIES & 1))

  def test_two_sizes_one_pair(self): # sends of different sizes share the pair's slots: consecutive wqes, each completed
    rings = {n: Buffer("CPU", RING_ENTRIES * 128 + RING_ENTRIES * 8, dtypes.uint8, preallocate=True) for n in ("sq", "scq", "db")}
    args = {n: UOp.from_buffer(b) for n, b in rings.items()}
    args |= {n: UOp.placeholder((1,), dtypes.uint64, 0, device="CPU", volatile=True, tag=n) for n in ("sq_seq", "psn")}
    nic = SimpleNamespace(device="CPU", iface=SimpleNamespace(dev_impl=SimpleNamespace(db_off=0)),
                          qp=lambda pair, peer: SimpleNamespace(qpn=5, scq_id=6, rcq_id=7))
    self.enterContext(patch.object(ops_rdma, "rdma_mem", lambda nic, pair, name, *_: args[name]))
    srcs = [Buffer("CPU:1", n, dtypes.uint8, preallocate=True) for n in (4096, 2048)]
    wires = [UOp.placeholder((b.size,), dtypes.uint8, 0, device="RDMA:0", tag="CPU") for b in srcs] # the far gpu
    calls = [UOp.from_buffer(b).copy_to_device("RDMA:0").call(w, UOp.from_buffer(b)) for b, w in zip(srcs, wires)]
    signal = UOp(Ops.INS, arg=("store", dtypes.void), src=(args["db"], UOp.const(1, dtypes.uint64)))
    submit = hcq2.make_submit(*calls, signal, devs=("CPU:1",), queue="COPY:0")
    get_device = type(Device).__getitem__
    with patch.object(type(Device), "__getitem__", lambda obj, d: nic if d == "RDMA:0" else get_device(obj, d)):
      ops = list(ops_rdma.rdma_submit(hcq2.EncodeCtx(("CPU",)), submit, submit.src[0]).src[0].src)
    self.assertEqual([u.arg[0] for u in ops], ["write", "write", "store", "wait_eq", "store"] * 2 + ["store"]) # two completed posts, the signal
    words = [w for u in ops[:-1] for w in u.src]
    checks = UOp.placeholder((8 * len(words),), dtypes.uint8, device="CPU", tag="checks")
    out = hcq2.patch(checks, [(8 * i, w) for i, w in enumerate(words)], bytes(8 * len(words)))
    lowered = hcq2.lower_call(UOp.sink(out.index(0).load(), arg=KernelInfo("bnxt_two_sizes"), tag=1).call(aux=hcq2.HCQInfo(("CPU",))))
    linked = hcq2.hcq_link(realize.lower_and_compile(UOp(Ops.LINEAR, src=(lowered,))), allow_cache=False)
    bufs = {p.tag: b.buffer for p, b in zip(lowered.without_after.src[1:], linked.src[0].without_after.src[1:])}
    base = bufs["sq_seq"].host.view(fmt="Q")[0]
    realize.run_linear(linked, jit=True)
    w = bufs["checks"].host.view(fmt="Q")[:]
    self.assertEqual((w[0], w[20]), (rings["sq"]._buf + base % RING_ENTRIES * 128, rings["sq"]._buf + (base + 1) % RING_ENTRIES * 128))
    self.assertEqual(bufs["sq_seq"].host.view(fmt="Q")[0], base + 2)
    self.assertEqual((w[16], w[36]), tuple(rings["scq"]._buf + (base + i) % CQ_ENTRIES * 32 + 24 for i in range(2))) # each waits for its wqe

if __name__ == "__main__": unittest.main()
