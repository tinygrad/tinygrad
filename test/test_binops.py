import unittest
from tinygrad.tensor import Tensor
import operator
import itertools

values = [float('inf'), float('-inf'), float('nan'), 0.0, -0.0, 5.0, -5.0]
tensors = [Tensor([value]) for value in values]
ops_dict = {
    "eq": operator.eq,
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
    "ne": operator.ne
}

class TestBinaryOperations(unittest.TestCase):

  def check_operation(self, op_func, op_name):
    for val_a, val_b in itertools.product(values, repeat=2):
      tensor_a, tensor_b = Tensor([val_a]), Tensor([val_b])
      actual = op_func(tensor_a, tensor_b).numpy()[0]
      desired = op_func(val_a, val_b)
      self.assertEqual(actual, desired, msg=f"For operation {op_name} with values {val_a} and {val_b}")
    
    def test_eq(self):        self.check_operation(ops_dict["eq"], "eq")
    def test_lt(self):        self.check_operation(ops_dict["lt"], "lt")    def test_gt(self):        self.check_operation(ops_dict["gt"], "gt")
    def test_le(self):        self.check_operation(ops_dict["le"], "le")
  
  def test_ge(self):        self.check_operation(ops_dict["ge"], "ge")
  def test_ne(self):
        self.check_operation(ops_dict["ne"], "ne")
    
if __name__ == '__main__':
    unittest.main()
