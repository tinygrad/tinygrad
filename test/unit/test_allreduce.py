import unittest
from tinygrad import Tensor, dtypes
from tinygrad.device import MultiBuffer
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, KernelInfo, graph_rewrite
from tinygrad.schedule.multi import multi_pm

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

  def test_schedule_all2all_defers_copy_consumer(self):
    def named_copy(name):
      def copy(out, src):
        idx = UOp.range(src.shape[0], 0)
        return out[idx].store(src[idx]).end(idx).sink(arg=KernelInfo(name=name))
      return copy

    with Context(ALL2ALL=2):
      N = 4
      ds = tuple(f"NULL:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      allreduce = (t.sum(0)+1).contiguous()
      src = Tensor.ones(N*100, device="NULL").contiguous().realize()
      first = Tensor.custom_kernel(Tensor.empty_like(src), src, fxn=named_copy("local_first"))[0]
      second = Tensor.custom_kernel(Tensor.empty_like(src), first, fxn=named_copy("local_second"))[0]

      linear = Tensor.schedule_linear(allreduce, second)
      names = [call.src[0].arg.name if call.src[0].op is Ops.SINK else call.src[0].op.name for call in linear.src]
      first_allreduce_add = next(i for i,call in enumerate(linear.src)
                                 if call.src[0].op is Ops.SINK and isinstance(call.device, str) and len(call.src) == N+2)
      reduce_scatter_copies = [i for i,call in enumerate(linear.src[:first_allreduce_add]) if call.src[0].op is Ops.COPY]
      self.assertEqual(len(reduce_scatter_copies), N*(N-1))
      self.assertLess(max(reduce_scatter_copies), names.index("local_second"))
      self.assertLess(names.index("local_second"), first_allreduce_add)

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
