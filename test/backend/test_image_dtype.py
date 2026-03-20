import unittest
import numpy as np
from tinygrad import Device, dtypes, Tensor, Context
from tinygrad.helpers import unwrap

IMAGE_SUPPORTED_DEVICES = ("QCOM", "CL")

@unittest.skipUnless(Device.DEFAULT in IMAGE_SUPPORTED_DEVICES, "Images not supported")
class TestImageDType(unittest.TestCase):
  # issue caused by: don't realize image to image casts. this is part of a larger problem
  #@unittest.expectedFailure
  # update: passing after tensor_map
  def test_lil_model(self):
    with Context(IMAGE=1):
      x = Tensor.zeros(1, 1)
      w1 = Tensor.zeros(1, 8, requires_grad=True)
      w2 = Tensor.zeros(8, 2)
      loss = x.image_dot(w1).image_dot(w2).float().max()
      loss.backward()
      sched = unwrap(w1.grad).schedule()
      for s in sched:
        s.run()
        if s.bufs[0].dtype == dtypes.float:
          lst = s.bufs[0].as_memoryview().cast("f").tolist()
          print(lst)
          assert not np.any(np.isnan(lst))
      # NOTE: the w1 grad must realize to a separate kernel
      assert w1.grad.uop.is_realized, f"never realized {w1.grad}"
      self.assertEqual(w1.grad.uop.base.buffer.dtype, dtypes.float32)
      self.assertEqual(len(sched), 9)

if __name__ == '__main__':
  unittest.main()
