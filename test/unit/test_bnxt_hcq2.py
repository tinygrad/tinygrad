import unittest, struct, types
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad import Device, dtypes
from tinygrad.device import Buffer, BufferStorage
from tinygrad.runtime.ops_rdma import BNXTAllocator
from tinygrad.runtime.support import hcq2
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import send_wqe, recv_wqe, msn_entry, db_value, RING_ENTRIES, CQ_ENTRIES
from tinygrad.runtime.ops_rdma import RDMADevice
from tinygrad.engine import realize
from tinygrad.runtime.support.memory import AddrSpace, VirtMapping
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta
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
    self.enterContext(patch.object(hcq2, "getenv", return_value=1))
    self.devs = {d: SimpleNamespace(device=d, peer_group=g, host="CPU", has_copy_queue=True, pm_batch=None)
                 for d, g in (("AMD:1", "a"), ("AMD:2", "b"), ("AMD:3", "a"))}
    get_device = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), "__getitem__", lambda obj, d: self.devs[d] if d in self.devs else get_device(obj, d)))

  def prepare(self, calls): return graph_rewrite(UOp(Ops.LINEAR, src=tuple(calls)), hcq2.pm_insert_copy_staging+realize.pm_flatten_linear)

  def test_split(self):
    src, dst = buf(0, "AMD:1"), buf(1, "AMD:2")
    send, recv = self.prepare([copy(src, dst)]).src
    self.assertEqual((send.src[0].arg, recv.src[0].arg), ("send", "recv"))
    self.assertEqual((hcq2.get_enqueue_devs(send), hcq2.get_enqueue_devs(recv)), ("AMD:1", "AMD:2"))
    self.assertIsNone(hcq2.stage_copy((), send, dst, src))
    # The ordinary-copy path is tested with real buffers elsewhere; this checks that these calls are not split.
    with patch.object(hcq2, "get_enqueue_devs", return_value=None):
      self.assertIsNone(hcq2.stage_copy((), copy(src, same:=buf(2, "AMD:3")), same, src))
      with patch.object(hcq2, "getenv", return_value=0): self.assertIsNone(hcq2.stage_copy((), copy(src, dst), dst, src))

  def test_dependencies_stay_on_each_node(self):
    src, dst = buf(0, "AMD:1"), buf(1, "AMD:2")
    send, recv = self.prepare([copy(src, dst)]).src
    calls = [(kernel(src), ("AMD:1",), "COPY:0"), (kernel(dst, False), ("AMD:2",), "COPY:0"),
             (send, ("AMD:1",), "COMPUTE:0"), (recv, ("AMD:2",), "COMPUTE:0"),
             (kernel(src), ("AMD:1",), "COPY:0"), (kernel(dst, False), ("AMD:2",), "COPY:0")]
    ctx = hcq2.BatchCtx(calls, False)
    waits = [hcq2._wait_ins(ctx, c, ds[0], q, i) for i, (c, ds, q) in enumerate(calls)]
    self.assertEqual([[w.src[1].val for w in ws] for ws in waits], [[], [], [1], [2], [3], [4]])
    batches = hcq2.sched_batches(self.prepare([copy(src, dst), copy(buf(2, "AMD:3"), dst)]), False).src
    self.assertEqual([(b.arg.aux.device, b.arg.aux.rdma) for b in batches], [(("AMD:1", "AMD:3"), True), (("AMD:2",), True)])

class TestBNXTCopy(unittest.TestCase):
  def test_words_replay(self): # the words of a send and a receive, linked and run: rings and cqs wrap, counters advance
    for recv in (False, True):
      rings = {n: Buffer("CPU", 8192, dtypes.uint8, preallocate=True) for n in ("sq", "rq", "scq", "rcq", "db")} # addressed, never written here
      counters = ("sq_seq", "rq_seq", "psn")
      args = {n: UOp.from_buffer(b) for n, b in rings.items()}
      args |= {n: UOp.placeholder((1,), dtypes.uint64, 0, device="CPU", volatile=True, tag=n) for n in counters}
      nic = SimpleNamespace(device="CPU", iface=SimpleNamespace(dev_impl=SimpleNamespace(db_off=0)), arg=lambda pair, n: args[n],
                            qp=lambda a, b: SimpleNamespace(qpn=5, scq_id=6, rcq_id=7))
      nic.wait = types.MethodType(RDMADevice.wait, nic)
      recorded:list = []
      hq = SimpleNamespace(dev=SimpleNamespace(device="CPU"), devs=("CPU",), ctx=SimpleNamespace(host="CPU"), counts={}, words={},
                           memory_barrier=lambda: None, write=lambda dst, *w: recorded.extend([dst, *w]),
                           signal=lambda dst, v: recorded.extend([dst, v]), wait=lambda a, v, eq: recorded.extend([a, v]))
      hq.rt, hq.bump = (types.MethodType(f, hq) for f in (hcq2.HWQueue.rt, hcq2.HWQueue.bump))
      src, dst = [Buffer("CPU", 4096, dtypes.uint8, preallocate=True) for _ in range(2)]
      call = UOp.from_buffer(src).copy_to_device("CPU").replace(arg="recv" if recv else "send").call(UOp.from_buffer(dst), UOp.from_buffer(src))
      RDMADevice.copy(nic, hq, call)
      # every recorded word at its own width, read back as u64
      checks = UOp.placeholder((8 * len(recorded),), dtypes.uint8, device="CPU", tag="checks")
      words = [w if isinstance(w, UOp) else UOp.const(w, dtypes.uint32) for w in recorded]
      out = hcq2.patch(checks, [(8 * i, w) for i, w in enumerate(words)], bytes(8 * len(words)))
      out = out.after(*[b.after(out).index(0).store(base + n) for b, (base, n) in hq.counts.items()])
      lowered = hcq2.lower_call(UOp.sink(out.index(0).load(), arg=KernelInfo("bnxt_copy_test"), tag=1).call(aux=hcq2.HCQInfo(("CPU",))))
      linked = hcq2.hcq_link(realize.lower_and_compile(UOp(Ops.LINEAR, src=(lowered,))), allow_cache=False)
      bufs = {p.tag: b.buffer for p, b in zip(lowered.without_after.src[1:], linked.src[0].without_after.src[1:])}
      ring, seq, cq = ("rq", "rq_seq", "rcq") if recv else ("sq", "sq_seq", "scq")
      data = dst if recv else src
      for i in range(130):
        realize.run_linear(linked, jit=True)
        words = bufs["checks"].host.view(fmt="Q")[:]
        wqe, rest = words[:12], words[12:] # the slot address, 8 header dwords, va, key, size
        self.assertEqual(wqe[0], rings[ring]._buf + i % RING_ENTRIES * 128)
        self.assertEqual(struct.pack("<8I", *wqe[1:9]) + struct.pack("<QII", *wqe[9:12]),
                         (recv_wqe if recv else send_wqe)(data._buf, data._buf & 0xffffffff, 4096))
        if not recv:
          self.assertEqual(rest[:2], [rings[ring]._buf + 0x1000 + i % RING_ENTRIES * 8, msn_entry(i, i, 4096)[0]])
          rest = rest[2:]
        doorbell = db_value(5, bnxt.DBC_DBC_TYPE_RQ if recv else bnxt.DBC_DBC_TYPE_SQ, (i + 1) % RING_ENTRIES, (i + 1) // RING_ENTRIES & 1)
        cq_doorbell = db_value(7 if recv else 6, bnxt.DBC_DBC_TYPE_CQ, (i + 1) % CQ_ENTRIES, (i + 1) // CQ_ENTRIES & 1)
        toggle = (i // CQ_ENTRIES & 1) ^ 1 | (2 if recv else 0)
        self.assertEqual((rest[1], rest[3], rest[5]), (doorbell, toggle, cq_doorbell)) # full width doorbell words
        self.assertEqual(rest[2], rings[cq]._buf + i % CQ_ENTRIES * 32 + 24)
        self.assertEqual(bufs[seq].host.view(fmt="Q")[0], i + 1)

if __name__ == "__main__": unittest.main()
