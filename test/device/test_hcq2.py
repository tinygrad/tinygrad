import unittest, struct, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor, UOp
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import getenv
from tinygrad.uop.ops import KernelInfo, Ops
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in

@unittest.skipUnless(getenv("HCQ2") and all_devices_in(Device.DEFAULT, HCQ_DEVS), "hcq2 device required")
class TestHCQ2(unittest.TestCase):
  def test_mixed_subword_args(self):
    a = UOp.param(0, dtypes.int8, (), name="a", addrspace=AddrSpace.ALU)
    out = UOp.param(1, dtypes.int, (1,))
    b = UOp.param(2, dtypes.int16, (), name="b", addrspace=AddrSpace.ALU)
    inp = UOp.param(3, dtypes.int, (1,))
    sink = out[0].store(inp[0].load() + a.cast(dtypes.int) + b.cast(dtypes.int)).sink(arg=KernelInfo("mixed_subword_args"))
    out_buf, inp_buf = Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=bytes(4)), \
                       Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=struct.pack("i", 5))
    call = sink.call(UOp.from_buffer(out_buf), UOp.from_buffer(inp_buf))
    run_linear(UOp(Ops.LINEAR, src=(call,)), var_vals={"a":-3, "b":2000}, update_stats=False)
    self.assertEqual(struct.unpack("i", out_buf.as_memoryview())[0], 2002)

  def test_copy_without_copy_queue(self):
    with patch.object(Device[Device.DEFAULT], "has_copy_queue", False):
      np.testing.assert_equal(Tensor(np.arange(61, dtype=np.float32)).to(Device.DEFAULT).contiguous().realize().numpy(), np.arange(61))

  @unittest.skipIf(Device.DEFAULT == "CPU", "staged copies need a non-CPU hcq2 device")
  def test_staged_copy_slot_reuse(self):
    # chunks of a staged copy rotate through the staging buffer slots, many rotations must stay bit-exact in both directions
    import tinygrad.runtime.support.hcq2 as hcq2
    buf = Buffer("CPU", 1 << 20, dtypes.uint8, preallocate=True)
    data = np.random.default_rng(42).integers(0, 256, (5 << 20) + 123, dtype=np.uint8)
    with patch.object(hcq2, "STAGING_SIZE", 1 << 20), patch.object(hcq2, "STAGING_SLOTS", 4), patch.object(hcq2, "_staging", lambda: buf):
      np.testing.assert_equal(Tensor(data).to(Device.DEFAULT).realize().numpy(), data)

  def test_overlapping_device_tuples(self):
    # an op on a wide device tuple followed by an op on an overlapping smaller tuple used to MMU-fault the smaller one
    d4, d2 = tuple(f"{Device.DEFAULT}:{i}" for i in range(4)), tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    ref = Tensor.arange(16).contiguous().realize()
    Tensor(ref.uop.copy_to_device(d4)).realize()
    out = Tensor.ones(8).shard(d2, axis=0).contiguous().realize()
    np.testing.assert_equal(out.numpy(), np.ones(8))

if __name__ == "__main__":
  unittest.main()
