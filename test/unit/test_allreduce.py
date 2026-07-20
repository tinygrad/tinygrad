import unittest
from unittest.mock import patch
from tinygrad import Tensor, dtypes
from tinygrad.device import Buffer, Device, MultiBuffer
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.schedule.multi import multi_pm

def make_peer_tensor(values, devices):
  shards = [Tensor([value], device=device).realize() for value,device in zip(values, devices)]
  return Tensor(UOp.mstack(*(x.uop for x in shards)).multi(0), device=devices)

class TestRingAllReduce(unittest.TestCase):
  def test_schedule_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*96).shard(ds, axis=0).realize()
      linear = t.sum(0).linear_with_vars()[0]
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]
      # N*(N-1) scatter reduce, and N*(N-1) allgather
      self.assertEqual(len(pairs), N*(N-1)*2)
      # copy topology forms a ring
      self.assertEqual(len(set(pairs)), N)

  def test_schedule_all2all(self):
    with Context(ALL2ALL=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      linear = t.sum(0).mul(2.0).contiguous().linear_with_vars()[0]
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
      self.assertEqual(len(copies), 24)
      self.assertEqual(len(sinks), 25)

  @Context(RING=0, ALL2ALL=0)
  def test_schedule_naive(self):
    N = 4
    ds = tuple(f"NULL:{i}" for i in range(N))
    t = Tensor.empty(N, 4096).shard(ds, axis=0).realize()
    linear = t.sum(0).linear_with_vars()[0]

    copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
    sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
    pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]

    self.assertEqual(len(pairs), N*(N-1))
    self.assertEqual(len(sinks), 1)
    self.assertTrue(all(dst != src for dst, src in pairs))

  @Context(PEER_ALLREDUCE=1, RING=0, ALL2ALL=0)
  def test_schedule_peer(self):
    N = 4
    ds = tuple(f"NULL:{i}" for i in range(N))
    t = Tensor.empty(N).shard(ds, axis=0).realize()
    linear = t.sum(0).linear_with_vars()[0]
    copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
    peers = [si for si in linear.src if si.src[0].arg.name == "peer_allreduce_4_1"]
    broadcasts = [si for si in linear.src if si.src[0].arg.name == "peer_broadcast_1"]
    self.assertEqual(len(copies), 0)
    self.assertEqual(len(peers), 1)
    self.assertEqual(len(peers[0].src[1:]), N+1)
    self.assertEqual(len(broadcasts), N-1)
    self.assertTrue(all(src.op is Ops.MSELECT and src.src[0].op is Ops.BUFFER for si in peers+broadcasts for src in si.src[1:]))

    linear = Tensor.empty(N, 16).shard(ds, axis=0).realize().sum(0).linear_with_vars()[0]
    self.assertTrue(any(si.src[0].op is Ops.COPY for si in linear.src))
    self.assertFalse(any(si.src[0].arg.name.startswith("peer_") for si in linear.src if si.src[0].op is not Ops.COPY))

  def test_correct_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.ones(N, N*96).contiguous().shard(ds, axis=0).realize()
      out = t.sum(0)
      self.assertListEqual(out.tolist(), [4]*N*96)

  def test_correct_all2all(self):
    with Context(ALL2ALL=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.arange(N*N*100).reshape(N, N*100).contiguous().shard(ds, axis=0).realize()
      width = N*100
      expected = [N*i + width*N*(N-1)//2 for i in range(width)]
      self.assertListEqual(t.sum(0).tolist(), expected)

  def test_correct_peer(self):
    with Context(PEER_ALLREDUCE=1, RING=0, ALL2ALL=0):
      N = 4
      ds = tuple(f"CPU:{i+10}" for i in range(N))
      t = make_peer_tensor(range(N), ds)
      self.assertEqual(t.sum(0).item(), N*(N-1)//2)

  def test_peer_mapping_freed_once(self):
    src, target = Device["CPU:30"], Device["CPU:31"]
    with patch.object(target.allocator, "_unmap", wraps=target.allocator._unmap) as unmap:
      buf = Buffer(src.device, 1, dtypes.float).allocate()
      buf.get_buf(target.device)
      buf.deallocate()
      self.assertEqual(unmap.call_count, 0)
      src.allocator.free_cache()
      self.assertEqual(unmap.call_count, 1)

  def test_correct_all2all_stack(self):
    with Context(ALL2ALL=2):
      N = 8
      ds = tuple(f"CPU:{i}" for i in range(N))
      out = Tensor.ones(N, N*96).contiguous().shard(ds, axis=0).realize().sum(0).realize()
      mb = out.uop.buf_uop.buffer
      self.assertIsInstance(mb, MultiBuffer)
      for buf in mb.bufs:
        self.assertListEqual(list(buf.as_memoryview().cast("f")), [N]*N*96)

      cast_out = Tensor.ones(N, N*96, dtype=dtypes.bfloat16).contiguous().shard(ds, axis=0).realize().sum(0).cast(dtypes.bfloat16).realize()
      cast_mb = cast_out.uop.buf_uop.buffer
      self.assertIsInstance(cast_mb, MultiBuffer)
      for buf in cast_mb.bufs:
        self.assertListEqual(list(buf.as_memoryview().cast("H")), [0x4100]*N*96)

class TestAllreduceCast(unittest.TestCase):
  def _get_copy_dtypes(self, dtype, allreduce_cast):
    ds = tuple(f"CPU:{i}" for i in range(2))
    with Context(ALLREDUCE_CAST=allreduce_cast, RING=0, SCACHE=0):
      t = Tensor.empty(4, 4, dtype=dtype).shard(ds, axis=0)
      linear = t.sum(0).linear_with_vars()[0]
      return {si.src[1].buffer.dtype.scalar() for si in linear.src if si.src[0].op is Ops.COPY}

  def test_allreduce_cast_bf16(self):
    # with ALLREDUCE_CAST, allreduce copies stay in bfloat16 instead of promoting to float32
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=0))

  def test_allreduce_cast_half(self):
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=0))

  def test_allreduce_cast_float32_noop(self):
    # float32 should not be affected by ALLREDUCE_CAST (no promotion happens)
    dtypes_on = self._get_copy_dtypes(dtypes.float, allreduce_cast=1)
    dtypes_off = self._get_copy_dtypes(dtypes.float, allreduce_cast=0)
    self.assertEqual(dtypes_on, dtypes_off)

  def test_singleton_local_reduce_uses_original_dtype(self):
    ds = tuple(f"NULL:{i}" for i in range(8))
    with Context(ALLREDUCE_CAST=1):
      t = Tensor.empty(8, 16, dtype=dtypes.bfloat16, device="NULL").shard(ds, axis=0).realize()
      lowered = graph_rewrite(t.sum(0).uop, multi_pm)
    allreduce = next(u for u in lowered.toposort() if u.op is Ops.ALLREDUCE)
    self.assertNotIn(Ops.CAST, {u.op for u in allreduce.src[0].toposort()})

if __name__ == '__main__':
  unittest.main()
