import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEV, Context
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo

# each test pins one instruction semantic the mock infers from Mesa's ir3 compiler, so they also run on hardware with DEV=QCOM:IR3
@unittest.skipUnless(Device.DEFAULT == "QCOM", "QCOM only")
class TestQCOMEmu(unittest.TestCase):
  def test_mod_const(self): # a single madsh.m16 with the constant in src1: lo(src1) * hi(src2)
    a = np.array([-4, 7, -7, -9, 2**20 + 1], np.int32)
    np.testing.assert_equal((Tensor(a) % 3).numpy(), a % 3)

  def test_imul(self): # mull.u + madsh.m16 lower a 32-bit multiply
    a, b = np.array([-123456, 70000, -1, 2**31 - 1], np.int32), np.array([7891, -70000, -1, 3], np.int32)
    np.testing.assert_equal((Tensor(a) * Tensor(b)).numpy(), a * b)

  def test_int64_to_float(self): # the int64 -> float lowering finds the msb with clz.b, which is ~0 for 0
    np.testing.assert_equal(Tensor([1, -2, 0, 2**40], dtype=dtypes.int64).cast(dtypes.float32).numpy(), np.array([1, -2, 0, 2**40], np.float32))

  def test_half_const(self): # SP_MODE_CNTL.CONSTANT_DEMOTION_ENABLE: a half float op reading an f32 const converts it
    np.testing.assert_equal((Tensor([1.5, 2.5, 1.0], dtype=dtypes.half) - 1.0).numpy(), np.array([0.5, 1.5, 0.0], np.float16))

  def test_u8_lerp(self): # cov from u8 sign-extends, ir3 masks with and.b when it wants zero-extension
    a = np.random.default_rng(0).integers(0, 32, (1, 1, 8, 8)).astype(np.uint8)
    ref = Tensor(a, device="CPU").interpolate((4, 4), mode="linear").numpy()
    np.testing.assert_equal(Tensor(a).interpolate((4, 4), mode="linear").numpy(), ref)

  def test_where(self): # (rpt3)sel.b32 is src2 ? src1 : src3, with (r) stepping all three sources
    a = np.array([1, -2, 3, -4], np.int32)
    np.testing.assert_equal((Tensor(a) > 0).where(Tensor(a * 10), Tensor(-a)).numpy(), np.where(a > 0, a * 10, -a))

  def test_bool_store(self): # stg.u8 stores from a half register
    np.testing.assert_equal((Tensor([1.0, 5, 6]) < Tensor([2.0, 3, 6])).numpy(), np.array([True, False, False]))

  def test_mad_unfused(self): # (rpt3)mad.f32 and mad.f16 round the product before the add, unlike ffma
    for dt, eps in ((np.float32, 2**-12), (np.float16, 2**-6)):
      a, c = np.array([1 + eps, 1 + 2 * eps, 1 + 3 * eps, 1.5], dt), np.array([-1, -1, -1, -2.25], dt)
      np.testing.assert_equal((Tensor(a) * Tensor(a) + Tensor(c)).numpy(), a * a + c)

  def test_mad_flushes_product(self): # mad flushes a denormal product before the add, a, b and c are normal
    for dt, a, c, bits in ((np.float32, 2.0**-65, 2.0**-126, [0x800000, 0x80800000]), (np.float16, 2.0**-8, 2.0**-14, [0x400, 0x8400])):
      out = (Tensor(np.array([a, a], dt)) * Tensor(np.array([a, a], dt)) + Tensor(np.array([c, -c], dt))).numpy()
      self.assertEqual(out.view(f"u{out.itemsize}").tolist(), bits)

  def test_dst_conv_rounds_to_zero(self): # a float op writing a half register narrows like cov: toward zero, saturating, flushing
    a = Tensor(np.array([1 + 2**-11 + 2**-13, 1e5, 2**-15 + 2**-17], np.float32))
    out = (a * Tensor(np.ones(3, np.float32))).half().numpy()
    self.assertEqual(out.view(np.uint16).tolist(), [0x3c00, 0x7bff, 0x0])

  def test_cov_rounds_to_zero(self): # cov's round 0 isn't (even): int -> float and f32 -> f16 truncate, f16 overflow saturates
    np.testing.assert_equal(Tensor([16777219, -19794315, 2**31 - 1], dtype=dtypes.int32).cast(dtypes.float32).numpy(),
                            np.array([16777218, -19794314, 2147483520], np.float32))
    np.testing.assert_equal(Tensor([1 + 2**-11 + 2**-13, 1e5, -1e30, 2**-20], dtype=dtypes.float32).cast(dtypes.half).numpy(),
                            np.array([1, 65504, -65504, 0], np.float16))

  def test_denormals_flush(self): # float alu ops flush denormal sources and results, keeping the sign
    np.testing.assert_equal((Tensor(np.array([1e-45, -1e-39, 1.0], np.float32)) * 1.5).numpy(), np.array([0, -0.0, 1.5], np.float32))
    np.testing.assert_equal(Tensor(np.array([1e-40], np.float32)).log2().numpy(), np.array([-np.inf], np.float32))
    np.testing.assert_equal((Tensor(np.array([1e-3, 2**-12], np.float16)) ** 2).numpy(), np.array([0, 0], np.float16))

  def test_nan_is_positive(self): # a float op that makes or passes on a NaN returns the positive quiet NaN
    for dt, bits in ((np.float32, 0x7fc00000), (np.float16, 0x7e00)):
      out = (Tensor(np.array([5332.347], dt)) - Tensor(np.array([np.nan], dt))).numpy()
      self.assertEqual(int(out.view(f"u{out.itemsize}")[0]), bits)

  def test_minmax_nan(self): # min.f/max.f return the other source when one is NaN, and order -0 below +0
    a, b = Tensor(np.array([np.nan, 1, -0.0, 0.0], np.float32)), Tensor(np.array([1, np.nan, 0.0, -0.0], np.float32))
    self.assertEqual(a.maximum(b).numpy().view(np.uint32).tolist(), [0x3f800000, 0x3f800000, 0, 0])
    self.assertEqual(a.minimum(b).numpy().view(np.uint32).tolist(), [0x3f800000, 0x3f800000, 0x80000000, 0x80000000])

  def test_half_cat4_truncates(self): # a half cat4 result is the f32 one rounded toward zero
    out = Tensor(np.array([894.0, 17.52, 2418.0], np.float16)).reciprocal().numpy()
    np.testing.assert_equal(out.view(np.uint16), np.array([0x1494, 0x2b4e, 0x0ec6], np.uint16))

  def test_predication(self): # with images this matmul has a predt/predf block, each side runs for its threads
    a, b = np.arange(15, dtype=np.float32).reshape(5, 3), np.arange(15, dtype=np.float32).reshape(3, 5) - 7
    with Context(IMAGE=1): np.testing.assert_equal((Tensor(a) @ Tensor(b)).numpy(), a @ b)

  def test_local_reduce_in_loop(self): # a bar inside a loop that isn't unrolled, under the min-pc scheduler
    a = Tensor.arange(64 * 8).reshape(64, 8).float().contiguous().realize()
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(64, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOCAL)
      return C[i].store(A[i, j].reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    np.testing.assert_equal(Tensor.custom_kernel(Tensor.empty(64), a, fxn=kernel)[0].numpy(), a.sum(1).numpy())

  @unittest.skipUnless(DEV.interface.startswith("MOCK"), "poisons the mock's register file")
  def test_predf_first(self): # with images triu is predf; load; predt; mov 0; prede: the mov must run, a zeroed register file would hide it
    from test.mockgpu.qcom import emu
    init = emu.Threads.__init__
    def poisoned(t, d, n):
      init(t, d, n)
      t.r[:], t.h[:] = 0x7777, 0x7777
    a = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    with patch.object(emu.Threads, "__init__", poisoned), Context(IMAGE=1): np.testing.assert_equal(Tensor(a).triu(1).numpy(), np.triu(a, 1))

  @unittest.skipUnless(DEV.interface.startswith("MOCK"), "the mock raises on instructions it doesn't emulate")
  def test_unsupported_instruction_raises(self):
    from tinygrad.engine.realize import lower_and_compile, run_linear
    from test.mockgpu.qcom.qcomdriver import EmulatorError
    linear = lower_and_compile((Tensor.empty(4) + 1).schedule_linear())
    binary = next(u for u in linear.toposort() if u.op is Ops.BINARY)
    lib = binary.arg.replace((mesa.OPC_END << 55).to_bytes(8, "little"), (mesa.OPC_RET << 55).to_bytes(8, "little"), 1)
    with self.assertRaisesRegex(EmulatorError, "OPC_RET is not emulated"):
      run_linear(linear.substitute({binary: binary.replace(arg=lib)}, enter_calls=True))
      Device[Device.DEFAULT].synchronize()

if __name__ == "__main__":
  unittest.main()
