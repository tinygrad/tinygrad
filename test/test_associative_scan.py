import unittest
from tinygrad import Tensor

class TestAssociativeScan(unittest.TestCase):
  def test_add_lengths(self):
    for n in range(1, 33):
      x = Tensor(list(range(1, n+1)), device="PYTHON")
      self.assertEqual(x.associative_scan(lambda a,b: a+b).tolist(), [i*(i+1)//2 for i in range(1, n+1)])

  def test_axis(self):
    x = Tensor([[1,2,3,4],[5,6,7,8]], device="PYTHON")
    self.assertEqual(x.associative_scan(lambda a,b: a+b, axis=1).tolist(), [[1,3,6,10],[5,11,18,26]])

  def test_non_commutative(self):
    x = Tensor([[[1,1],[0,1]], [[1,0],[1,1]], [[2,0],[0,1]], [[1,2],[0,1]]], device="PYTHON")
    got = x.associative_scan(lambda a,b: a.matmul(b), axis=0).tolist()
    cur = Tensor([[1,0],[0,1]], device="PYTHON")
    expected=[]
    for i in range(4):
      cur = cur.matmul(x[i])
      expected.append(cur.tolist())
    self.assertEqual(got, expected)

  def test_singleton_and_empty(self):
    self.assertEqual(Tensor([7], device="PYTHON").associative_scan(lambda a,b:a+b).tolist(), [7])
    self.assertEqual(Tensor([], device="PYTHON").associative_scan(lambda a,b:a+b).tolist(), [])

if __name__ == "__main__": unittest.main()
