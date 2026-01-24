import unittest
import numpy as np
from tinygrad import Device, dtypes, Tensor, Context
from tinygrad.device import LRUAllocator, is_dtype_supported
from tinygrad.dtype import ImageDType
from tinygrad.helpers import prod, unwrap
from test.helpers import REAL_DEV

IMAGE_SUPPORTED_DEVICES = ("QCOM", "CL", "METAL")

@unittest.skipUnless(REAL_DEV in IMAGE_SUPPORTED_DEVICES, "Images not supported")
class TestImageCopy(unittest.TestCase):
  def test_image_copyout_1x8(self, img_type=dtypes.imagef):
    it = Tensor.arange(32).cast(img_type((1,8,4))).realize()
    buf = it.uop.buffer
    out = buf.as_buffer()
    np.testing.assert_equal(out.cast(it.dtype.fmt).tolist(), np.arange(32))

  @unittest.skipUnless(is_dtype_supported(dtypes.half, device="PYTHON"), "need half")
  def test_imageh_copyout_1x8(self): self.test_image_copyout_1x8(img_type=dtypes.imageh)

  def test_image_numpy_1x8(self, img_type=dtypes.imagef):
    it = Tensor.arange(32).cast(img_type((1,8,4))).realize()
    np.testing.assert_equal(it.numpy(), np.arange(32))
  def test_imageh_numpy_1x8(self): self.test_image_numpy_1x8(img_type=dtypes.imageh)

  def test_image_copyout_2x4(self):
    it = Tensor.arange(2*4*4).cast(dtypes.imagef((2,4,4))).realize()
    buf = it.uop.buffer
    out = buf.as_buffer()
    np.testing.assert_equal(out.cast('f').tolist(), np.arange(2*4*4))

  def test_image_roundtrip(self):
    sz = (4,2,4)
    it = Tensor.rand(prod(sz)).cast(dtypes.imagef(sz)).realize()
    buf = it.uop.buffer
    out = buf.as_buffer()

    it2 = Tensor.rand(prod(sz)).cast(dtypes.imagef(sz)).realize()
    buf2 = it2.uop.buffer
    buf2.copyin(out)

    assert (it == it2).sum().item() == prod(sz)

@unittest.skipUnless(REAL_DEV in IMAGE_SUPPORTED_DEVICES, "Images not supported")
class TestImageDType(unittest.TestCase):
  def test_image_pitch(self):
    def __validate(imgdt, pitch_map):
      assert imgdt.pitch == pitch_map[REAL_DEV], f"Failed pitch for image: {imgdt}. Got 0x{imgdt.pitch:X}, expected 0x{pitch_map[REAL_DEV]:X}"

    __validate(dtypes.imageh((1, 201)), {"CL": 0x680, "QCOM": 0x680, "METAL": 0x800})
    __validate(dtypes.imageh((16, 216)), {"CL": 0x700, "QCOM": 0x700, "METAL": 0x800})
    __validate(dtypes.imageh((16, 9)), {"CL": 0x80, "QCOM": 0x80, "METAL": 0x800})
    __validate(dtypes.imageh((48, 64)), {"CL": 0x200, "QCOM": 0x200, "METAL": 0x800})
    __validate(dtypes.imageh((32, 128)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x800})
    __validate(dtypes.imageh((96, 128)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x800})
    __validate(dtypes.imageh((64, 256)), {"CL": 0x840, "QCOM": 0x840, "METAL": 0x800})
    __validate(dtypes.imageh((64, 9)), {"CL": 0x80, "QCOM": 0x80, "METAL": 0x800})
    __validate(dtypes.imageh((192, 256)), {"CL": 0x840, "QCOM": 0x840, "METAL": 0x800})
    __validate(dtypes.imageh((64, 768)), {"CL": 0x1840, "QCOM": 0x1840, "METAL": 0x1800})
    __validate(dtypes.imageh((256, 49)), {"CL": 0x1C0, "QCOM": 0x1C0, "METAL": 0x800})
    __validate(dtypes.imageh((128, 9)), {"CL": 0x80, "QCOM": 0x80, "METAL": 0x800})
    __validate(dtypes.imageh((16, 1024)), {"CL": 0x2080, "QCOM": 0x2080, "METAL": 0x2000})
    __validate(dtypes.imageh((64, 512)), {"CL": 0x1040, "QCOM": 0x1040, "METAL": 0x1000})
    __validate(dtypes.imageh((16, 512)), {"CL": 0x1080, "QCOM": 0x1080, "METAL": 0x1000})
    __validate(dtypes.imageh((132, 64)), {"CL": 0x200, "QCOM": 0x200, "METAL": 0x800})
    __validate(dtypes.imageh((4, 512)), {"CL": 0x1200, "QCOM": 0x1200, "METAL": 0x1000})
    __validate(dtypes.imageh((8, 512)), {"CL": 0x1100, "QCOM": 0x1100, "METAL": 0x1000})
    __validate(dtypes.imageh((128, 128)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x800})
    __validate(dtypes.imageh((32, 512)), {"CL": 0x1040, "QCOM": 0x1040, "METAL": 0x1000})
    __validate(dtypes.imageh((26, 64)), {"CL": 0x200, "QCOM": 0x200, "METAL": 0x800})
    __validate(dtypes.imageh((32, 516)), {"CL": 0x1040, "QCOM": 0x1040, "METAL": 0x1800})
    __validate(dtypes.imageh((32, 1024)), {"CL": 0x2040, "QCOM": 0x2040, "METAL": 0x2000})
    __validate(dtypes.imageh((16, 2048)), {"CL": 0x4080, "QCOM": 0x4080, "METAL": 0x4000})
    __validate(dtypes.imageh((8, 2048)), {"CL": 0x4100, "QCOM": 0x4100, "METAL": 0x4000})
    __validate(dtypes.imageh((4, 4096)), {"CL": 0x8200, "QCOM": 0x8200, "METAL": 0x8000})

    __validate(dtypes.imagef((16, 49)), {"CL": 0x380, "QCOM": 0x380, "METAL": 0x1000})
    __validate(dtypes.imagef((16, 1024)), {"CL": 0x4080, "QCOM": 0x4080, "METAL": 0x4000})
    __validate(dtypes.imagef((256, 64)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((64, 512)), {"CL": 0x2040, "QCOM": 0x2040, "METAL": 0x2000})
    __validate(dtypes.imagef((16, 512)), {"CL": 0x2080, "QCOM": 0x2080, "METAL": 0x2000})
    __validate(dtypes.imagef((132, 64)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((4, 512)), {"CL": 0x2200, "QCOM": 0x2200, "METAL": 0x2000})
    __validate(dtypes.imagef((4, 16)), {"CL": 0x200, "QCOM": 0x200, "METAL": 0x1000})
    __validate(dtypes.imagef((2, 16)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((8, 512)), {"CL": 0x2100, "QCOM": 0x2100, "METAL": 0x2000})
    __validate(dtypes.imagef((12, 64)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((3, 32)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((128, 128)), {"CL": 0x840, "QCOM": 0x840, "METAL": 0x1000})
    __validate(dtypes.imagef((32, 512)), {"CL": 0x2040, "QCOM": 0x2040, "METAL": 0x2000})
    __validate(dtypes.imagef((8, 3072)), {"CL": 0xC100, "QCOM": 0xC100, "METAL": 0xC000})
    __validate(dtypes.imagef((4, 2048)), {"CL": 0x8200, "QCOM": 0x8200, "METAL": 0x8000})
    __validate(dtypes.imagef((4, 1024)), {"CL": 0x4200, "QCOM": 0x4200, "METAL": 0x4000})
    __validate(dtypes.imagef((4, 4096)), {"CL": 0x10200, "QCOM": 0x10200, "METAL": 0x10000})
    __validate(dtypes.imagef((10, 384)), {"CL": 0x1900, "QCOM": 0x1900, "METAL": 0x2000})
    __validate(dtypes.imagef((24, 64)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((128, 12)), {"CL": 0xC0, "QCOM": 0xC0, "METAL": 0x1000})
    __validate(dtypes.imagef((10, 24)), {"CL": 0x200, "QCOM": 0x200, "METAL": 0x1000})
    __validate(dtypes.imagef((1, 129)), {"CL": 0x840, "QCOM": 0x840, "METAL": 0x1000})
    __validate(dtypes.imagef((1, 32)), {"CL": 0x200, "QCOM": 0x200, "METAL": 0x1000})
    __validate(dtypes.imagef((1, 64)), {"CL": 0x400, "QCOM": 0x400, "METAL": 0x1000})
    __validate(dtypes.imagef((1, 1239)), {"CL": 0x4D80, "QCOM": 0x4D80, "METAL": 0x5000})
    __validate(dtypes.imagef((1, 1)), {"CL": 0x40, "QCOM": 0x40, "METAL": 0x1000})

  def test_image_and_back(self):
    data = Tensor.randn(9*32*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,32,4))).contiguous().realize()
    assert isinstance(it.uop.base.realized.dtype, ImageDType)
    np.testing.assert_equal(tst, it.numpy())

  def test_image_cast_and_back_collapses(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    # the underlying UOp is identical
    self.assertIs(it.uop.base.realized, data.uop.base.realized)
    np.testing.assert_equal(tst, it.numpy())

  def test_image_and_back_wrong_shape(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,12,4))).realize()
    assert not isinstance(it.uop.base.realized.dtype, ImageDType)
    np.testing.assert_equal(tst, it.numpy())

  def test_image_add(self, img_type=dtypes.imagef, rtol=1e-6):
    data = Tensor.randn(1*4*4).realize()
    tst = data.numpy()
    with Context(IMAGE=2):
      it = data.cast(img_type((1,4,4))).contiguous().realize()
      out = (it + 1).realize()
    np.testing.assert_allclose(out.numpy(), tst + 1, rtol=rtol)
  def test_image_add_half(self): self.test_image_add(img_type=dtypes.imageh, rtol=1e-2)

  def test_shrink_load_float(self, img_type=dtypes.imagef):
    it = Tensor.randn(16).cast(img_type((1,4,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(imgv[0:2], it[0:2].numpy())
  def test_shrink_load_half(self): self.test_shrink_load_float(img_type=dtypes.imageh)

  def test_mul_stays_image(self):
    # NOTE: contiguous is needed otherwise this folds
    it = Tensor.randn(16).cast(dtypes.imagef((1,4,4))).contiguous().realize()
    out = (it*2).realize()
    assert isinstance(out.uop.base.realized.dtype, ImageDType)

  def test_sum(self):
    it = Tensor.rand(8).cast(dtypes.imagef((1,2,4))).realize()
    itn = it.numpy()
    np.testing.assert_allclose(np.sum(itn), it.sum().numpy(), rtol=1e-6)

  def test_shrink_max(self):
    it = Tensor.randn(16).cast(dtypes.imagef((1,4,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(np.maximum(imgv[0:3], 0), it[0:3].relu().numpy())

  def test_shrink_to_float(self):
    it = Tensor.randn(4, 4).cast(dtypes.imagef((1,4,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(np.maximum(imgv[:, 0], 0), it[:, 0].relu().numpy())

  @unittest.skipUnless(isinstance(Device.default.allocator, LRUAllocator), "Requires LRU")
  def test_lru_alloc(self):
    data = Tensor.randn(9*27*4).realize()
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    b1 = it.uop.base.realized._buf
    del it
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    assert it.uop.base.realized._buf == b1

  def test_no_lru_alloc(self):
    data = Tensor.randn(9*32*4).realize()
    it = data.cast(dtypes.imagef((9,32,4))).contiguous().realize()
    b1 = it.uop.base.realized._buf
    del it
    it = data.reshape(9,32,4).pad_to(10, None, None).cast(dtypes.imagef((10,32,4))).contiguous().realize()
    assert it.uop.base.realized._buf != b1

  def test_no_lru_alloc_dtype(self):
    data = Tensor.randn(9*32*4).realize()
    it = data.cast(dtypes.imagef((9,32,4))).contiguous().realize()
    b1 = it.uop.base.realized._buf
    del it
    it = data.cast(dtypes.imageh((9,32,4))).realize()
    assert it.uop.base.realized._buf != b1

  # issue caused by: don't realize image to image casts. this is part of a larger problem
  #@unittest.expectedFailure
  # update: passing after tensor_map
  def test_lil_model(self):
    with Context(IMAGE=2):
      x = Tensor.zeros(1, 1)
      w1 = Tensor.zeros(1, 8, requires_grad=True)
      w2 = Tensor.zeros(8, 2)
      loss = x.image_dot(w1).image_dot(w2).float().max()
      loss.backward()
      sched = unwrap(w1.grad).schedule()
      for s in sched:
        s.run()
        if s.bufs[0].dtype == dtypes.float:
          lst = s.bufs[0].as_buffer().cast("f").tolist()
          print(lst)
          assert not np.any(np.isnan(lst))
      # NOTE: the w1 grad must realize to a separate kernel
      assert w1.grad.uop.is_realized, f"never realized {w1.grad}"
      self.assertEqual(w1.grad.uop.base.buffer.dtype, dtypes.float32)
      self.assertEqual(len(sched), 9)

@unittest.skipUnless(REAL_DEV in IMAGE_SUPPORTED_DEVICES, "Images not supported")
class TestImageRealization(unittest.TestCase):
  def test_image_dtype_expand(self):
    data = Tensor.randn(9*32*4).realize()
    it = data.cast(dtypes.imagef((9,32,4))).contiguous().realize()
    self.assertEqual(it.dtype, dtypes.imagef((9,32,4)))
    it_expanded = it.reshape((9,32,4,1)).expand((9,32,4,4)).contiguous().realize()
    self.assertEqual(it_expanded.dtype, dtypes.float32)

  def test_image_dtype_expand_and_back(self):
    data = Tensor.randn(9*32*4).realize()
    it = data.cast(dtypes.imagef((9,32,4))).contiguous().realize()
    self.assertEqual(it.dtype, dtypes.imagef((9,32,4)))
    it_expanded = it.reshape((9,32,4,1)).expand((9,32,4,4))
    it2 = it_expanded.sum(3).realize()
    self.assertEqual(it2.dtype, dtypes.imagef((9,32,4)))

  def test_image_alu_children(self):
    data = Tensor.randn(9*32*4).realize()
    it = data.cast(dtypes.imagef((9,32,4))).contiguous().realize()
    self.assertEqual(it.dtype, dtypes.imagef((9,32,4)))
    it_expanded = it.reshape((9,32,4,1)).expand((9,32,4,4)).contiguous()
    alu1 = it_expanded+1
    alu2 = it_expanded.sum(3)
    it_expanded.realize()
    # NOTE: the parent becomes float, but the alu child will stay image until its output cannot fit the image
    self.assertEqual(alu1.dtype, dtypes.imagef((9,32,4)))
    alu1.realize()
    self.assertEqual(alu1.dtype, dtypes.float32)
    # alu2 is back in image because it fits the dtype again
    self.assertEqual(alu2.dtype, dtypes.imagef((9,32,4)))
    alu2.realize()
    self.assertEqual(alu2.dtype, dtypes.imagef((9,32,4)))

if __name__ == '__main__':
  unittest.main()
