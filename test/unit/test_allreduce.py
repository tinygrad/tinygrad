import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from tinygrad.schedule.allreduce import create_allreduce_function

class TestRingAllReduce(unittest.TestCase):
  def test_precompiled_input_staged_once(self):
    devices = tuple(f"NULL:{i}" for i in range(4))
    buf = UOp.new_buffer(devices, 4096, dtypes.float) + 1
    ret = create_allreduce_function(buf, buf.allreduce(Ops.ADD, devices))
    assert ret is not None
    contiguous = [x for x in ret.toposort() if x.op is Ops.CONTIGUOUS]
    self.assertEqual(len(contiguous), 1)
    call = next(x for x in ret.toposort() if x.op is Ops.CALL)
    self.assertIs(call.src[-1], contiguous[0])

  def test_schedule_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      linear = t.sum(0).linear_with_vars()[0]
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]
      # N*(N-1) scatter reduce, and N*(N-1) allgather
      self.assertEqual(len(pairs), N*(N-1)*2)
      # copy topology forms a ring
      self.assertEqual(len(set(pairs)), N)

  def test_schedule_all2all(self):
    for width, expected_sinks in ((400, 23), (4096, 18)):
      with self.subTest(width=width), Context(ALL2ALL=2):
        N = 4
        ds = tuple(f"CPU:{i}" for i in range(N))
        t = Tensor.empty(N, width).shard(ds, axis=0).realize()
        linear = t.sum(0).mul(2.0).contiguous().linear_with_vars()[0]
        copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
        sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
        self.assertEqual(len(copies), 24)
        # source shards are staged once, then their physical slices feed SDMA directly
        self.assertEqual(len(sinks), expected_sinks)

  def test_correct_all2all_direct_slices(self):
    with Context(ALL2ALL=2):
      N, W = 4, 512
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = (Tensor.arange(N*W).reshape(N, W).shard(ds, axis=0) * 2 + 1).contiguous().realize()
      self.assertListEqual(t.sum(0).tolist(), [8*i + 4*W*3 + 4 for i in range(W)])

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
    self.assertEqual(len(sinks), 2)
    self.assertTrue(all(dst != src for dst, src in pairs))

  def test_symbolic_shape(self):
    rows = UOp.variable("rows", 1, 4).bind(3)
    t = Tensor.ones(4, 4).shard(("CPU:0", "CPU:1"), axis=1).realize()
    out = t[:rows].sum(1).realize()
    self.assertEqual(out.shape, (rows,))
    self.assertTrue((out == 4).all().item())

  def test_correct_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.ones(N, N*100).contiguous().shard(ds, axis=0).realize()
      out = t.sum(0)
      self.assertListEqual(out.tolist(), [4]*N*100)

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

if __name__ == '__main__':
  unittest.main()
