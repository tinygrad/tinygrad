import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEV, Context
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo

@unittest.skipUnless(Device.DEFAULT == "QCOM", "QCOM only")
class TestQCOMEmu(unittest.TestCase):
  def test_mod_const(self):
    a = np.array([-4, 7, -7, -9, 2**20 + 1], np.int32)
    np.testing.assert_equal((Tensor(a) % 3).numpy(), a % 3)

  def test_imul(self):
    a, b = np.array([2**17 + 1, -(2**16 + 1), -1, 2**31 - 1], np.int32), np.array([2**16 + 3, 2**16 + 1, -1, 3], np.int32)
    np.testing.assert_equal((Tensor(a) * Tensor(b)).numpy(), a * b)

  def test_int64_to_float(self):
    np.testing.assert_equal(Tensor([1, -2, 0, 2**40], dtype=dtypes.int64).cast(dtypes.float32).numpy(), np.array([1, -2, 0, 2**40], np.float32))

  def test_half_const(self):
    np.testing.assert_equal((Tensor([1.5, 2.5, 1.0], dtype=dtypes.half) - 1.0).numpy(), np.array([0.5, 1.5, 0.0], np.float16))

  def test_u8_lerp(self):
    a = np.random.default_rng(0).integers(0, 32, (1, 1, 8, 8)).astype(np.uint8)
    ref = Tensor(a, device="CPU").interpolate((4, 4), mode="linear").numpy()
    np.testing.assert_equal(Tensor(a).interpolate((4, 4), mode="linear").numpy(), ref)

  def test_where(self):
    a = np.array([1, -2, 3, -4], np.int32)
    np.testing.assert_equal((Tensor(a) > 0).where(Tensor(a * 10), Tensor(-a)).numpy(), np.where(a > 0, a * 10, -a))

  def test_bool_store(self):
    np.testing.assert_equal((Tensor([1.0, 5, 6]) < Tensor([2.0, 3, 6])).numpy(), np.array([True, False, False]))

  def test_mad(self): # not fused, denormal products flush
    for dt, eps in ((np.float32, 2**-12), (np.float16, 2**-6)):
      a, c = np.array([1 + eps, 1 + 2 * eps, 1 + 3 * eps, 1.5], dt), np.array([-1, -1, -1, -2.25], dt)
      np.testing.assert_equal((Tensor(a) * Tensor(a) + Tensor(c)).numpy(), a * a + c)
    for dt, a, c, bits in ((np.float32, 2.0**-65, 2.0**-126, [0x800000, 0x80800000]), (np.float16, 2.0**-8, 2.0**-14, [0x400, 0x8400])):
      out = (Tensor(np.array([a, a], dt)) * Tensor(np.array([a, a], dt)) + Tensor(np.array([c, -c], dt))).numpy()
      self.assertEqual(out.view(f"u{out.itemsize}").tolist(), bits)

  def test_dst_conv(self):
    a = Tensor(np.array([1 + 2**-11 + 2**-13, 1e5, 2**-15 + 2**-17], np.float32))
    out = (a * Tensor(np.ones(3, np.float32))).half().numpy()
    self.assertEqual(out.view(np.uint16).tolist(), [0x3c00, 0x7bff, 0x0])

  def test_cov(self):
    np.testing.assert_equal(Tensor([2**24 + 3, -(2**24 + 3), 2**31 - 1], dtype=dtypes.int32).cast(dtypes.float32).numpy(),
                            np.array([2**24 + 2, -(2**24 + 2), 2**31 - 2**7], np.float32))
    np.testing.assert_equal(Tensor([1 + 2**-11 + 2**-13, 1e5, -1e30, 2**-20], dtype=dtypes.float32).cast(dtypes.half).numpy(),
                            np.array([1, 65504, -65504, 0], np.float16))

  def test_denormals(self):
    np.testing.assert_equal((Tensor(np.array([1e-45, -1e-39, 1.0], np.float32)) * 1.5).numpy(), np.array([0, -0.0, 1.5], np.float32))
    np.testing.assert_equal(Tensor(np.array([1e-40], np.float32)).log2().numpy(), np.array([-np.inf], np.float32))
    np.testing.assert_equal((Tensor(np.array([1e-3, 2**-12], np.float16)) ** 2).numpy(), np.array([0, 0], np.float16))

  def test_nan(self):
    for dt, bits in ((np.float32, 0x7fc00000), (np.float16, 0x7e00)):
      out = (Tensor(np.array([1.0], dt)) - Tensor(np.array([np.nan], dt))).numpy()
      self.assertEqual(int(out.view(f"u{out.itemsize}")[0]), bits)

  def test_minmax_nan(self):
    a, b = Tensor(np.array([np.nan, 1, -0.0, 0.0], np.float32)), Tensor(np.array([1, np.nan, 0.0, -0.0], np.float32))
    self.assertEqual(a.maximum(b).numpy().view(np.uint32).tolist(), [0x3f800000, 0x3f800000, 0, 0])
    self.assertEqual(a.minimum(b).numpy().view(np.uint32).tolist(), [0x3f800000, 0x3f800000, 0x80000000, 0x80000000])

  def test_half_rcp(self):
    out = Tensor(np.array([894.0, 17.52, 2418.0], np.float16)).reciprocal().numpy()
    np.testing.assert_equal(out.view(np.uint16), np.array([0x1494, 0x2b4e, 0x0ec6], np.uint16))

  def test_predication(self):
    a, b = np.arange(15, dtype=np.float32).reshape(5, 3), np.arange(15, dtype=np.float32).reshape(3, 5) - 7
    with Context(IMAGE=1): np.testing.assert_equal((Tensor(a) @ Tensor(b)).numpy(), a @ b)

  def test_17_images(self):
    xs = [np.random.default_rng(i).random((4, 64)).astype(np.float32) for i in range(17)]
    with Context(IMAGE=1): out = sum([Tensor(x) for x in xs[1:]], Tensor(xs[0])).numpy()
    np.testing.assert_allclose(out, sum(xs), rtol=1e-6)

  def test_bar_in_loop(self):
    a = Tensor.arange(64 * 8).reshape(64, 8).float().contiguous().realize()
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(64, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOCAL)
      return C[i].store(A[i, j].reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    np.testing.assert_equal(Tensor.custom_kernel(Tensor.empty(64), a, fxn=kernel)[0].numpy(), a.sum(1).numpy())

  @unittest.skipUnless(DEV.interface.startswith("MOCK"), "MOCK only")
  def test_predf_first(self): # a zeroed register file would hide a skipped mov
    from test.mockgpu.qcom import emu
    init = emu.Threads.__init__
    def poisoned(t, d, n):
      init(t, d, n)
      t.r[:] = 0x77777777
      if isinstance(t.h, np.ndarray): t.h[:] = 0x7777
    a = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    with patch.object(emu.Threads, "__init__", poisoned), Context(IMAGE=1): np.testing.assert_equal(Tensor(a).triu(1).numpy(), np.triu(a, 1))

  @unittest.skipUnless(DEV.interface.startswith("MOCK"), "MOCK only")
  def test_alu_blocks(self):
    from test.mockgpu.qcom import emu
    run, dispatches = emu.run, []
    def capture(d):
      dispatches.append(d)
      run(d)
    w = np.random.default_rng(0).standard_normal((1, 1, 13, 13)).astype(np.float32)
    x, y = Tensor(np.arange(-40, 40, dtype=np.int32)), Tensor(np.arange(80, dtype=np.int32) * 7919)
    u = Tensor(np.arange(80, dtype=np.uint32) * 2654435761)
    with patch.object(emu, "run", capture), Context(IMAGE=1):
      floats = Tensor(w).conv2d(Tensor(w[..., :3, :3]), padding=1) + Tensor(w)
      ints = (x * y - x) ^ (y >> 3) | (x & ~y)
      selects = (x > 0).where(x, y)
      uints = ((u >> 5) & (u | 3)) - (u << 2)
      Tensor.realize(floats, ints, selects, uints)
    denormals, normals, specials = [1e-45, -1e-38], [1e-20, 0.1, 1.5, -2.0, -3.0, 7.0, 3e38], [-0.0, np.nan, np.inf, -np.inf]
    values = np.array(denormals + normals + specials, np.float32).view(np.uint32)
    checked = 0
    for d in dispatches:
      prog = emu.decode(d.image)
      targets = {i.target for i in prog if isinstance(i, emu.Cat0) and i.op in emu.JUMPS}
      for start, b in emu.alu_blocks(d.image, d.entry).items():
        self.assertFalse(targets & set(range(start + 1, b.end)))
        fast, slow = emu.Threads(d, 13), emu.Threads(d, 13)
        fast.r[:] = slow.r[:] = np.random.default_rng(start).choice(values, fast.r.shape)
        fast.mask = slow.mask = np.arange(13) % 3 != 0
        emu.exec_block(fast, b)
        with np.errstate(all="ignore"):
          for i in prog[start:b.end]:
            for k in range(i.iterations if type(i) in emu.EXEC else 0): emu.EXEC[type(i)](slow, i, k)
        np.testing.assert_equal(fast.r, slow.r)
        checked += 1
    self.assertGreater(checked, 0)

  @unittest.skipUnless(DEV.interface.startswith("MOCK"), "MOCK only")
  def test_unsupported(self):
    from tinygrad.engine.realize import lower_and_compile, run_linear
    from test.mockgpu.qcom.qcomdriver import EmulatorError
    linear = lower_and_compile((Tensor.empty(4) + 1).schedule_linear())
    binary = next(u for u in linear.toposort() if u.op is Ops.BINARY)
    lib = binary.arg.replace((mesa.OPC_END << 55).to_bytes(8, "little"), (mesa.OPC_KILL << 55).to_bytes(8, "little"), 1)
    with self.assertRaisesRegex(EmulatorError, "OPC_KILL is not emulated"):
      run_linear(linear.substitute({binary: binary.replace(arg=lib)}, enter_calls=True))
      Device[Device.DEFAULT].synchronize()

if __name__ == "__main__":
  unittest.main()
