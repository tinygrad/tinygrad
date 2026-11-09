import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.device import Buffer
from tinygrad.llm.kernels.amd_qwen import partial_sum, _flag_allreduce, flag_allreduce, flag_ar_groups, reduce_scatter, all_gather, all_gather_q8
from tinygrad.llm.kernels.amd_qwen import _q8_pack_kernel

@unittest.skipUnless(Device.DEFAULT == "AMD", "AMD required")
class TestFlagAllreduce(unittest.TestCase):
  def test_chained_mixed_sizes_all_ranks(self):
    devs = tuple(Device.canonicalize(f"AMD:{i}") for i in range(getenv("TEST_GPUS", 2)))
    if any(Device[d].target[0] != 11 for d in devs): self.skipTest("experimental gfx11 peer mappings required")
    rng = np.random.default_rng(42)
    shapes = [2560, 12800, 81920, 330240]*3
    arrays = [[rng.uniform(-32, 32, (len(devs), 1, n)).astype(np.float32) for n in shapes] for _ in range(2)]
    inputs = [[Tensor(a).shard(devs, axis=0).contiguous().realize() for a in group] for group in arrays]
    refs = [[a.astype(np.float16).astype(np.float32).sum(0).ravel() for a in group] for group in arrays]
    with self.subTest(transport="gpu"):
      getenv.cache_clear()
      _flag_allreduce.cache_clear()
      @TinyJit
      def chain(*xs):
        ys = []
        for x in xs:
          y = partial_sum(x if not ys else (x + ys[-1].flatten()[0]*1e-30).contiguous())
          ys.append(y)
        Tensor.realize(*ys)
        return tuple(ys)  # realize returns only its first tensor, not the entire list
      for iteration in range(5):
        if iteration == 2:
          # Cross uint32 epoch wrap in both phases and in each independently sized pool.
          for groups in {flag_ar_groups(n//2) for n in shapes}:
            ar = flag_allreduce(devs, groups)
            for rank, buf in enumerate(ar.counter.uop.base.buffer.bufs):
              values = np.full(groups+1, 0xFFFFFFFC, dtype=np.uint32)
              values[-1] = rank
              buf.copy_from(Buffer("NPY", values.nbytes, opaque=memoryview(values).cast("B")))
          for d in devs: Device[d].synchronize()
        group = iteration % 2
        outputs = chain(*inputs[group])
        for d in devs: Device[d].synchronize()
        for j, (out, ref) in enumerate(zip(outputs, refs[group])):
          for rank, buf in enumerate(out.uop.base.buffer.bufs):
            actual = np.frombuffer(buf.as_memoryview(), dtype=np.float32)
            with self.subTest(iteration=iteration, call=j, rank=rank): np.testing.assert_array_equal(actual, ref)
      self._sequence_collectives(devs)
      self._packed_gather(devs)
      self._q8_reduce_scatter(devs)
      getenv.cache_clear()
      _flag_allreduce.cache_clear()

  def _sequence_collectives(self, devs):
    rng = np.random.default_rng(1)
    for dtype, npdtype in ((dtypes.float32, np.float32), (dtypes.half, np.float16)):
      @TinyJit
      def run(p):
        reduced = reduce_scatter(p, 1)
        gathered = all_gather(reduced, 1, dtype=dtype)
        Tensor.realize(reduced, gathered)
        return reduced, gathered
      for iteration in range(4):
        # Unaligned chunks exercise the fp16 fallback; aligned Q8 chunks are checked separately.
        data = rng.normal(size=(len(devs), 1, len(devs)*6, 264)).astype(np.float32)
        p = Tensor(data).shard(devs, axis=0).contiguous().realize()
        ref = data.astype(np.float16).astype(np.float32).sum(0)
        reduced, gathered = run(p)
        self.assertEqual(reduced.uop.axis, 1)
        for rank, buf in enumerate(reduced.uop.base.buffer.bufs):
          actual = np.frombuffer(buf.as_memoryview(), dtype=np.float32)
          np.testing.assert_array_equal(actual, np.split(ref, len(devs), axis=1)[rank].ravel())
        for buf in gathered.uop.base.buffer.bufs:
          np.testing.assert_array_equal(np.frombuffer(buf.as_memoryview(), dtype=npdtype), ref.astype(npdtype).ravel())
      # Pure half allgather must preserve special-value bits, not saturate as the reduction does.
      if dtype == dtypes.half:
        data = np.resize(np.array([-0., np.inf, -np.inf, np.nan, 1., -1.], dtype=np.float16), (1, 192, 256))
        x = Tensor(data).shard(devs, axis=1).contiguous().realize()
        gathered = all_gather(x, 1).realize()
        for buf in gathered.uop.base.buffer.bufs:
          np.testing.assert_array_equal(np.frombuffer(buf.as_memoryview(), dtype=np.uint16), data.ravel().view(np.uint16))

  def _q8_reduce_scatter(self, devs):
    rng, N = np.random.default_rng(93), len(devs)
    with self.subTest(transport="q8"):
      getenv.cache_clear()
      for rows, width in ((N, 256), (192, 256), (768, 256), (2112, 2560)):
        for npdtype in (np.float32, np.float16):
          @TinyJit
          def run(p): return reduce_scatter(p, 1).realize()
          for iteration in range(3):
            data = rng.normal(size=(N, 1, rows, width)).astype(npdtype)
            data.reshape(-1)[:256] = 0
            p = Tensor(data).shard(devs, axis=0).contiguous().realize()
            # Use the existing packer as the quantization reference (including its float32 rounding).
            pk = Tensor.empty(N, rows*width//32, 9, dtype=dtypes.uint32, device=devs, axis=0)
            pk = Tensor.custom_kernel(pk, p.reshape(N, rows*width), fxn=_q8_pack_kernel)[0].realize()
            decoded = []
            for b in pk.uop.base.buffer.bufs:
              raw = np.frombuffer(b.as_memoryview(), dtype=np.uint32).reshape(-1, 9)
              decoded.append(raw[:, :8].copy().view(np.int8).astype(np.float32)*raw[:, 8:].view(np.float32))
            decoded = np.stack(decoded).reshape(N, N, -1)
            if iteration == 1:
              groups = flag_ar_groups(rows*width//2)
              ar = flag_allreduce(devs, groups)
              for rank, buf in enumerate(ar.counter.uop.base.buffer.bufs):
                values = np.full(groups+1, 0xFFFFFFFF, dtype=np.uint32)
                values[-1] = rank
                buf.copy_from(Buffer("NPY", values.nbytes, opaque=memoryview(values).cast("B")))
              # Seed both previous epochs too: merely rewinding counters could alias a flag from an earlier test.
              flags = np.repeat(np.array([0xFFFFFFFE, 0xFFFFFFFF], dtype=np.uint32), N)
              for buf in ar.flags[0].uop.base.buffer.bufs:
                buf.copy_from(Buffer("NPY", flags.nbytes, opaque=memoryview(flags).cast("B")))
              for d in devs: Device[d].synchronize()
            out = run(p)
            self.assertEqual(out.uop.axis, 1)
            for rank, buf in enumerate(out.uop.base.buffer.bufs):
              ref = np.zeros(rows*width//N, dtype=np.float32)
              for s in range(N):
                if s != rank: ref += decoded[s, rank]
              ref += data.reshape(N, N, -1)[rank, rank].astype(np.float32)
              with self.subTest(q8=True, rows=rows, dtype=npdtype, iteration=iteration, rank=rank):
                np.testing.assert_allclose(np.frombuffer(buf.as_memoryview(), dtype=np.float32), ref, rtol=1e-6, atol=2e-6)
    getenv.cache_clear()

  def _packed_gather(self, devs):
    rng = np.random.default_rng(6)
    # Arbitrary packed bits (including float NaN encodings) must survive the raw-float transport unchanged.
    data = rng.integers(0, 2**32, size=(1, 192, 256), dtype=np.uint32)
    x = Tensor(data).shard(devs, axis=1).contiguous().realize()
    for buf in all_gather(x, 1).realize().uop.base.buffer.bufs:
      np.testing.assert_array_equal(np.frombuffer(buf.as_memoryview(), dtype=np.uint32), data.ravel())
    x = Tensor(rng.normal(size=(1, 192, 256)).astype(np.float32)).shard(devs, axis=1).contiguous().realize()
    with patch("tinygrad.llm.kernels.amd_qwen.flag_all_gather", return_value=None):
      getenv.cache_clear()
      reference = all_gather_q8(x, 1).realize()
    getenv.cache_clear()
    result = all_gather_q8(x, 1).realize()
    for actual, expected in zip(result.uop.base.buffer.bufs, reference.uop.base.buffer.bufs):
      np.testing.assert_array_equal(np.frombuffer(actual.as_memoryview(), dtype=np.uint16),
                                    np.frombuffer(expected.as_memoryview(), dtype=np.uint16))

if __name__ == "__main__": unittest.main()
