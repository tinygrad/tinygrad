import unittest
import numpy as np
from tinygrad import Device, dtypes, Tensor, Context, nn
from tinygrad.dtype import ImageDType
from tinygrad.engine.realize import lower_schedule
from tinygrad.helpers import IMAGE, all_same, prod, unwrap
from tinygrad.ops import Ops

@unittest.skipIf(Device.DEFAULT not in ("QCOM", "GPU"), "only images on GPU")
class TestImageCopy(unittest.TestCase):
  def test_image_copyout_1x1(self, img_type=dtypes.imagef):
    it = Tensor.arange(4).cast(img_type((1,1,4))).realize()
    buf = it.lazydata.buffer
    out = buf.as_buffer()
    np.testing.assert_equal(out.cast(it.dtype.fmt).tolist(), np.arange(4))
  def test_imageh_copyout_1x1(self): self.test_image_copyout_1x1(img_type=dtypes.imageh)

  def test_image_numpy_1x1(self, img_type=dtypes.imagef):
    it = Tensor.arange(4).cast(img_type((1,1,4))).realize()
    np.testing.assert_equal(it.numpy(), np.arange(4))
  def test_imageh_numpy_1x1(self): self.test_image_numpy_1x1(img_type=dtypes.imageh)

  def test_image_copyout_2x3(self):
    it = Tensor.arange(2*3*4).cast(dtypes.imagef((2,3,4))).realize()
    buf = it.lazydata.buffer
    out = buf.as_buffer()
    np.testing.assert_equal(out.cast('f').tolist(), np.arange(2*3*4))

  def test_image_roundtrip(self):
    sz = (4,2,4)
    it = Tensor.rand(prod(sz)).cast(dtypes.imagef(sz)).realize()
    buf = it.lazydata.buffer
    out = buf.as_buffer()

    it2 = Tensor.rand(prod(sz)).cast(dtypes.imagef(sz)).realize()
    buf2 = it2.lazydata.buffer
    buf2.copyin(out)

    assert (it == it2).sum().item() == prod(sz)

@unittest.skipIf(Device.DEFAULT not in ("QCOM", "GPU"), "only images on GPU")
class TestImageDType(unittest.TestCase):
  def test_image_and_back(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,27,4))).contiguous().realize()
    assert isinstance(it.lazydata.base.realized.dtype, ImageDType)
    np.testing.assert_equal(tst, it.numpy())

  @unittest.expectedFailure # this isn't supported anymore, CAST to ImageDType stays ImageDType
  def test_image_cast_and_back_collapses(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    # the underlying UOp is identical
    self.assertIs(it.lazydata.base.realized, data.lazydata.base.realized)
    np.testing.assert_equal(tst, it.numpy())

  def test_image_and_back_wrong_shape(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,12,4))).realize()
    assert not isinstance(it.lazydata.base.realized.dtype, ImageDType)
    np.testing.assert_equal(tst, it.numpy())

  def test_shrink_load_float(self):
    it = Tensor.randn(4).cast(dtypes.imagef((1,1,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(imgv[0:2], it[0:2].numpy())

  def test_mul_stays_image(self):
    # NOTE: contiguous is needed otherwise this folds
    it = Tensor.randn(4).cast(dtypes.imagef((1,1,4))).contiguous().realize()
    out = (it*2).realize()
    assert isinstance(out.lazydata.base.realized.dtype, ImageDType)

  def test_sum(self):
    it = Tensor.rand(8).cast(dtypes.imagef((1,2,4))).realize()
    itn = it.numpy()
    np.testing.assert_allclose(np.sum(itn), it.sum().numpy(), rtol=1e-6)

  def test_shrink_max(self):
    it = Tensor.randn(8).cast(dtypes.imagef((1,2,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(np.maximum(imgv[0:3], 0), it[0:3].relu().numpy())

  def test_shrink_to_float(self):
    it = Tensor.randn(4, 4).cast(dtypes.imagef((1,4,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(np.maximum(imgv[:, 0], 0), it[:, 0].relu().numpy())

  def test_lru_alloc(self):
    data = Tensor.randn(9*27*4).realize()
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    b1 = it.lazydata.base.realized._buf
    del it
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    assert it.lazydata.base.realized._buf == b1

  def test_no_lru_alloc(self):
    data = Tensor.randn(9*27*4).realize()
    it = data.cast(dtypes.imagef((9,27,4))).contiguous().realize()
    b1 = it.lazydata.base.realized._buf
    del it
    it = data.cast(dtypes.imagef((10,27,4))).contiguous().realize()
    assert it.lazydata.base.realized._buf != b1

  def test_no_lru_alloc_dtype(self):
    data = Tensor.randn(9*27*4).realize()
    it = data.cast(dtypes.imagef((9,27,4))).contiguous().realize()
    b1 = it.lazydata.base.realized._buf
    del it
    it = data.cast(dtypes.imageh((9,27,4))).realize()
    assert it.lazydata.base.realized._buf != b1

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
      # NOTE: the w1 grad must realize to a seperate kernel
      assert w1.grad.lazydata.is_realized, f"never realized {w1.grad}"
      self.assertEqual(w1.grad.lazydata.base.buffer.dtype, dtypes.float32)
      self.assertEqual(len(sched), 10)
      for s,ei in zip(sched, lower_schedule(sched[:])):
        ei.run()
        if s.outputs[0].dtype == dtypes.float:
          lst = s.outputs[0].as_buffer().cast("f").tolist()
          print(lst)
          assert not np.any(np.isnan(lst))

@unittest.skipIf(Device.DEFAULT not in ("QCOM", "GPU"), "only images on GPU")
class TestImageRealization(unittest.TestCase):
  def setUp(self):
    self.old_image = IMAGE.value
    IMAGE.value = 2
  def tearDown(self):
    IMAGE.value = self.old_image

  def test_dtype_override(self):
    # this is, "make things that can't be images not images"
    a = Tensor.ones(9, 9).contiguous().cast(dtypes.imagef((3, 12, 4))).contiguous()
    # before realize, it's an image
    self.assertIsInstance(a.lazydata.dtype, ImageDType)
    # after realize, it becomes a float32
    # this is because we can't lower its ShapeTracker to index?
    a = a.realize()
    self.assertEqual(a.lazydata.dtype, dtypes.float)

  @unittest.expectedFailure
  def test_dtype_overriden_alu(self):
    # make something that can't be image
    a = Tensor.ones(9, 9).contiguous().cast(dtypes.imagef((3, 12, 4))).contiguous()
    # give it an ALU child
    add = a+2
    # realize the parent, it becomes a float32 BUFFER
    a.realize()
    # the child's dtype is still imagef, so the ADD becomes (float32)+(imagef)
    self.assertIsInstance(add.dtype, ImageDType)
    operand_dtypes = [x.dtype for x in add.lazydata.src]
    assert all_same(operand_dtypes), f"expected dtypes to be the same for {add.lazydata}"

  @unittest.expectedFailure
  def test_linear_grad_dtype(self):
    class TinyNet:
      def __init__(self):
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
      def __call__(self, x):
        return self.l2(self.l1(x).relu()).relu()
    with Tensor.train():
      model = TinyNet()
      # why is this needed?
      for param in nn.state.get_state_dict(model).values():
        if param.requires_grad is None: param.requires_grad = True
      # realize the loss, some BUFFERs change dtype
      X = Tensor.empty(32, 784)
      Y = Tensor.empty(32, 10)
      out = model(X)
      loss = (out*Y).mean()
      loss.backward()
      loss.realize()
      # then, try to realize the gradients
      grad_uop = model.l1.weight.grad.lazydata.base
      for x in grad_uop.toposort:
        operand_dtypes = [y.dtype for y in (x.src if x.op is not Ops.WHERE else x.src[1:])]
        assert all_same(operand_dtypes), f"expected dtypes to match for {x.op}, {operand_dtypes}"

if __name__ == '__main__':
  unittest.main()
