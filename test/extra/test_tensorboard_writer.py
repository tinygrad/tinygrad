import glob, os, unittest, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from extra.tensorboard.writer import TinySummaryWriter
from tinygrad.ops import LazyOp, BinaryOps, ReduceOps
from tinygrad.shape.shapetracker import MovementOps
from tinygrad.tensor import Tensor

def buf(*shp): return Tensor.ones(*shp, device="CPU").lazydata

class TestTinySummaryWriter(unittest.TestCase):
  def setUp(self):
    self.log_dir = './logs'
    self.writer, self.accumulator = TinySummaryWriter(self.log_dir), EventAccumulator(self.log_dir)
  def tearDown(self):
    self.writer.close()
    [os.remove(f) for f in glob.glob(os.path.join(self.log_dir, '*'))]
  def write_and_reload(self, func, *args, **kwargs):
    func(*args, **kwargs)
    self.writer.flush()
    self.accumulator.Reload()

  def base_test_scalar(self, test_value):
    self.write_and_reload(self.writer.add_scalar, 'test_scalar', test_value)
    scalar_events = self.accumulator.Scalars('test_scalar')
    self.assertEqual(len(scalar_events), 1)
    self.assertEqual(scalar_events[0].value, test_value)

  def test_scalar_tiny_tensor(self): self.base_test_scalar(Tensor(1.0))
  def test_scalar_np_array(self): self.base_test_scalar(np.array(1.0))
  def test_scalar_float(self): self.base_test_scalar(1.0)

  def base_test_h_gram(self, bins, test_values, expected_bucket):
    self.write_and_reload(self.writer.add_histogram, 'test_histogram', test_values, bins=bins)
    histogram_events = self.accumulator.Histograms('test_histogram')
    self.assertEqual(len(histogram_events), 1)
    histogram_value = histogram_events[0].histogram_value
    self.assertEqual(test_values.min(), histogram_value.min)
    self.assertEqual(test_values.max(), histogram_value.max)
    self.assertEqual(test_values.sum(), histogram_value.sum)
    self.assertEqual(test_values.pow(2), histogram_value.sum_squares)
    self.assertEqual(expected_bucket, histogram_value.bucket)

  def test_h_gram_4_bin_auto(self): self.base_test_h_gram('auto', Tensor([1.0, 2.0, 3.0, 4.0, 5.0]), [0, 1, 1, 1, 2])
  def test_h_gram_5_bin_auto(self): self.base_test_h_gram('auto', Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0]), [0, 2, 1, 1, 3])
  def test_h_gram_4_bin_2(self): self.base_test_h_gram(2, Tensor([1.0, 2.0, 3.0, 4.0, 5.0]), [0, 2, 3])
  def test_h_gram_4_bin_3(self): self.base_test_h_gram(3, Tensor([1.0, 2.0, 3.0, 4.0, 5.0]), [0, 2, 1, 2])

  def base_test_image(self, test_value):
    self.write_and_reload(self.writer.add_image, 'test_image', test_value)
    image_events = self.accumulator.Images('test_image')
    self.assertEqual(len(image_events), 1)
    self.assertEqual(image_events[0].height, test_value.shape[1])
    self.assertEqual(image_events[0].width, test_value.shape[2])

  def test_image_C1_H1_H1(self): self.base_test_image(Tensor.rand(1, 1, 1))
  def test_image_C1_H2_W1(self): self.base_test_image(Tensor.ones(1, 2, 1))
  def test_image_C1_H1_W2(self): self.base_test_image(Tensor.ones(1, 1, 2))
  def test_image_C3_H6_W9(self): self.base_test_image(Tensor.ones(3, 6, 9))

  def test_graph(self):
    a,b = buf(4,4), buf(1,1)
    op0 = LazyOp(MovementOps.RESHAPE, (b,), (4, 4))
    op1 = LazyOp(BinaryOps.ADD, (a,op0))
    ast = LazyOp(ReduceOps.SUM, (op1,), (1,1))
    self.writer.add_graph(ret=buf(1,1), ast=ast)
    self.writer.flush()

if __name__ == '__main__': unittest.main()
