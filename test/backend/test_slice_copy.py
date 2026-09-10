import unittest
import numpy as np
from tinygrad import Tensor, Device, GlobalCounters, TinyJit, Variable

@unittest.skipUnless(Device.DEFAULT in {"CPU", "PYTHON", "CUDA", "NV", "AMD"}, "requires an offset-capable backend")
class TestSliceCopy(unittest.TestCase):
  @property
  def peer(self): return Device.DEFAULT if Device.DEFAULT in {"CUDA", "NV", "AMD"} else "CPU:1"
  @property
  def directions(self): return (("CPU", self.peer), (self.peer, "CPU"))

  def test_slice_layouts(self):
    cases = [
      ((20,), slice(10, 12), slice(13, 15), 1),
      ((20,), slice(19, 20), slice(0, 1), 1),
      ((20,), slice(0, 4), slice(16, 20), 1),
      ((20,), slice(0, 2), slice(18, 20), 1),
      ((20,), slice(18, 20), slice(0, 2), 1),
      ((20,), slice(10, 12), slice(13, 15), 1),
      ((4, 6), (slice(1, 2), slice(2, 5)), (slice(2, 3), slice(1, 4)), 1),
      ((4, 6), (slice(1, 3), slice(None)), (slice(0, 2), slice(None)), 1),
      ((2, 3, 4), (slice(1, 2), slice(1, 3), slice(None)), (slice(0, 1), slice(0, 2), slice(None)), 1),
      ((2, 1, 4), (slice(1, 2), slice(None), slice(1, 3)), (slice(0, 1), slice(None), slice(2, 4)), 1),
      ((2, 3, 1), (slice(0, 1), slice(1, 3), slice(None)), (slice(1, 2), slice(0, 2), slice(None)), 1),
      ((2, 3, 4), (slice(1, 2), slice(1, 2), slice(1, 2)), (slice(0, 1), slice(2, 3), slice(3, 4)), 1),
      ((4, 6), (slice(None), slice(1, 3)), (slice(None), slice(2, 4)), 3),
      ((2, 1, 3, 4), (slice(None), slice(None), slice(1, 3), slice(None)),
       (slice(None), slice(None), slice(0, 2), slice(None)), 3),
      ((20,), slice(2, 10, 2), slice(4, 8), None),
      ((20,), slice(4, 8), slice(2, 10, 2), None),
      ((4, 6), (slice(0, 0), slice(1, 3)), (slice(0, 0), slice(1, 3)), 0),
      ((4, 6), (slice(1, 3), slice(0, 0)), (slice(1, 3), slice(0, 0)), 0),
    ]
    for src, dst in self.directions:
      held = []
      for shape, di, si, count in cases:
        with self.subTest(src=src, dst=dst, shape=shape, di=di, si=si):
          expected, values = np.full(shape, -1., dtype=np.float32), np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
          a, b = Tensor(expected.copy(), device=dst).realize(), Tensor(values.copy(), device=src).realize()
          GlobalCounters.reset()
          a[di].assign(b[si].to(dst)).realize()
          if count is not None: self.assertEqual(GlobalCounters.kernel_count, count)
          expected[di] = values[si]
          np.testing.assert_array_equal(a.numpy(), expected)
          np.testing.assert_array_equal(b.numpy(), values)
          held.append((a, b, expected, values))
      for a, b, expected, values in held:
        np.testing.assert_array_equal(a.numpy(), expected)
        np.testing.assert_array_equal(b.numpy(), values)

  def test_copy_byte_boundaries(self):
    rng = np.random.default_rng(217)
    for src, dst in self.directions:
      for dtype in (np.uint8, np.float16, np.float32, np.int32, np.int64):
        itemsize = np.dtype(dtype).itemsize
        lengths = [1] + [boundary//itemsize+delta for boundary in (32, 64, 256, 4096) for delta in (-1, 0, 1)]
        for i, n in enumerate(lengths):
          do, so = ((1, 0), (3, 5), (19, 19))[i % 3]
          with self.subTest(src=src, dst=dst, dtype=dtype, n=n, do=do, so=so):
            av = np.frombuffer(rng.bytes((n+19)*itemsize), dtype=dtype).copy()
            bv = np.frombuffer(rng.bytes((n+19)*itemsize), dtype=dtype).copy()
            if dtype in (np.float16, np.float32):
              bits = np.uint16 if dtype == np.float16 else np.uint32
              patterns = [0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7d01, 1] if dtype == np.float16 else [
                0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001, 1]
              bv.view(bits)[so:so+min(n, len(patterns))] = patterns[:min(n, len(patterns))]
            expected = av.copy()
            expected.view(np.uint8)[do*itemsize:(do+n)*itemsize] = bv.view(np.uint8)[so*itemsize:(so+n)*itemsize]
            a, b = Tensor(av.copy(), device=dst).realize(), Tensor(bv.copy(), device=src).realize()
            GlobalCounters.reset()
            a[do:do+n].assign(b[so:so+n].to(dst)).realize()
            self.assertEqual(GlobalCounters.kernel_count, 1)
            self.assertEqual(a.numpy().tobytes(), expected.tobytes())
            self.assertEqual(b.numpy().tobytes(), bv.tobytes())

  def test_creation_copy_lifetime(self):
    for kind in ("copy", "consumer", "view", "shared"):
      with self.subTest(kind=kind):
        a, other = Tensor([1.]*20, device=self.peer).realize(), Tensor([0.]*20, device=self.peer).realize()
        b = Tensor([float(i) for i in range(20)], device="PYTHON").realize()
        copied = b[13:15].to(a.device)
        later = copied+1 if kind == "consumer" else copied[1:] if kind == "view" else copied
        if kind == "shared": other[4:6].assign(copied)
        GlobalCounters.reset()
        a[10:12].assign(copied).realize()
        self.assertEqual(GlobalCounters.kernel_count, 2)
        a[10:12].assign(Tensor([99., 99.], device=a.device)).realize()
        b[13:15].assign(Tensor([77., 77.], device="PYTHON")).realize()
        self.assertEqual(a.tolist(), [1.]*10 + [99., 99.] + [1.]*8)
        if kind == "shared": self.assertEqual(other.tolist(), [0.]*4 + [13., 14.] + [0.]*14)
        else: self.assertEqual(later.tolist(), [14., 15.] if kind == "consumer" else [14.] if kind == "view" else [13., 14.])

  def test_realized_copy_is_independent(self):
    for src, dst in self.directions:
      with self.subTest(src=src, dst=dst):
        a, b = Tensor([1.]*20, device=dst).realize(), Tensor([float(i) for i in range(20)], device=src).realize()
        copied = b[13:15].to(dst)
        a[10:12].assign(copied).realize()
        copied.realize()
        a[10:12].assign(Tensor([99., 99.], device=dst)).realize()
        b[13:15].assign(Tensor([77., 77.], device=src)).realize()
        self.assertEqual(copied.tolist(), [13., 14.])

  def test_pending_writes(self):
    for src, dst in self.directions:
      for first, second in (((1, 5), (3, 7)), ((3, 7), (1, 5)), ((1, 5), (5, 9)), ((1, 5), (9, 13))):
        for read_between in (False, True):
          with self.subTest(src=src, dst=dst, first=first, second=second, read_between=read_between):
            expected = np.full(16, -17., dtype=np.float32)
            bv, cv = np.arange(11, 15, dtype=np.float32), np.arange(31, 35, dtype=np.float32)
            a, b, c = Tensor(expected.copy(), device=dst), Tensor(bv.copy(), device=src), Tensor(cv.copy(), device=src)
            Tensor.realize(a, b, c)
            a[slice(*first)].assign(b.to(dst))
            mid = a[slice(*first)].sum().realize() if read_between else None
            a[slice(*second)].assign(c.to(dst)).realize()
            expected[slice(*first)], expected[slice(*second)] = bv, cv
            np.testing.assert_array_equal(a.numpy(), expected)
            np.testing.assert_array_equal(b.numpy(), bv)
            np.testing.assert_array_equal(c.numpy(), cv)
            if mid is not None: self.assertEqual(mid.item(), 50.)

  def test_jit_rebinds_new_buffers(self):
    for src, dst in (*self.directions, ("PYTHON", self.peer)):
      with self.subTest(src=src, dst=dst):
        @TinyJit
        def write(a, b): a[5:13].assign(b[7:15].to(a.device)).realize()
        held = []
        for i in range(6):
          av, bv = np.full(32, -100.-i, dtype=np.float32), np.arange(32, dtype=np.float32)+1000*i
          a, b = Tensor(av.copy(), device=dst).realize(), Tensor(bv.copy(), device=src).realize()
          write(a, b)
          av[5:13] = bv[7:15]
          held.append((a, b, av, bv))
        for a, b, av, bv in held:
          np.testing.assert_array_equal(a.numpy(), av)
          np.testing.assert_array_equal(b.numpy(), bv)

  def test_symbolic_rebindings(self):
    for src, dst in self.directions:
      for mode in ("offset", "length"):
        with self.subTest(src=src, dst=dst, mode=mode):
          @TinyJit
          def write(a, b, v):
            ds, ss = ((v, v+3), (9, 12)) if mode == "offset" else ((7, 7+v), (9, 9+v))
            a.shrink((ds,)).assign(b.shrink((ss,)).to(a.device)).realize()
          for i, value in enumerate((1, 4, 2, 1, 3, 4)):
            av, bv = np.full(20, -i-10., dtype=np.float32), np.arange(24, dtype=np.float32)+i*100
            a, b = Tensor(av.copy(), device=dst).realize(), Tensor(bv.copy(), device=src).realize()
            write(a, b, Variable("v", 1, 4).bind(value))
            start, n = (value, 3) if mode == "offset" else (7, value)
            av[start:start+n] = bv[9:9+n]
            np.testing.assert_array_equal(a.numpy(), av)
            np.testing.assert_array_equal(b.numpy(), bv)

  def test_computed_source_then_consumer(self):
    for src, dst in self.directions:
      with self.subTest(src=src, dst=dst):
        av, bv = np.arange(1047, dtype=np.float32), np.arange(1053, dtype=np.float32)
        a, b = Tensor(av.copy(), device=dst).realize(), Tensor(bv.copy(), device=src).realize()
        a[7:1032].assign((b[13:1038]*3-7).to(dst))
        result = (a*2+5).realize()
        av[7:1032] = bv[13:1038]*3-7
        np.testing.assert_array_equal(result.numpy(), av*2+5)
        np.testing.assert_array_equal(a.numpy(), av)
        np.testing.assert_array_equal(b.numpy(), bv)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "uses multiple logical CPU devices")
  def test_replicated_destination(self):
    devices = ("CPU", "CPU:1")
    a = Tensor.ones(4, 6, device="CPU").shard(devices).realize()
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    b = Tensor(values, device="CPU:2").shard(("CPU:2", "CPU:3")).realize()
    a[1:3, :].assign(b[:2, :].to(devices)).realize()
    expected = np.ones((4, 6), dtype=np.float32)
    expected[1:3, :] = values[:2, :]
    np.testing.assert_array_equal(a.numpy(), expected)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "uses multiple logical CPU devices")
  def test_explicit_reshard_before_assignment(self):
    devices = ("CPU", "CPU:1")
    a = Tensor.ones(4, 6, device="CPU").shard(devices, axis=0).realize()
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    b = Tensor(values, device="CPU:2").shard(("CPU:2", "CPU:3"), axis=0).realize()
    rhs = b[:, 2:4].to("CPU:2").shard(devices, axis=0)
    a[:, 1:3].assign(rhs).realize()
    expected = np.ones((4, 6), dtype=np.float32)
    expected[:, 1:3] = values[:, 2:4]
    np.testing.assert_array_equal(a.numpy(), expected)

if __name__ == '__main__':
  unittest.main()
