import unittest
import numpy as np
from examples.mlperf.train_retinanet.train_retinanet import compute_overlap

class TestComputeOverlap(unittest.TestCase):
  def test_zero_overlap(self):
    result = compute_overlap(np.array([[0, 0, 1, 1]]), np.array([[2, 2, 3, 3]]))
    np.testing.assert_array_equal(result, np.array([[0.0]]))

  def test_full_overlap(self):
    result = compute_overlap(np.array([[0, 0, 1, 1]]), np.array([[0, 0, 1, 1]]))
    np.testing.assert_array_equal(result, np.array([[1.0]]))

  def test_partial_overlap(self):
    result = compute_overlap(np.array([[0, 0, 2, 2]]), np.array([[1, 1, 3, 3]]))
    np.testing.assert_array_almost_equal(result, np.array([[0.14]]), decimal=2)

  def test_multiple_boxes_and_query_boxes(self):
    boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
    query_boxes = np.array([[1, 1, 2, 2], [0, 0, 1, 1]])
    result = compute_overlap(boxes, query_boxes)
    np.testing.assert_array_almost_equal(result, np.array([[0.0, 1.0], [0.25, 0.25]]), decimal=2)

  def test_multiple_boxes_no_overlap(self):
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])
    query_boxes = np.array([[6, 6, 7, 7], [8, 8, 9, 9]])
    result = compute_overlap(boxes, query_boxes)
    np.testing.assert_array_equal(result, np.zeros((3, 2)))

if __name__ == '__main__':
  unittest.main()
