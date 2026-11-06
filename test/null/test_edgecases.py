import unittest
import torch
from tinygrad import Tensor, nn, Context

class TestEmptyTensorEdgeCases(unittest.TestCase):
  # we don't need more of these

  def test_max_empty(self):
    # Max on an empty tensor should also raise an error.
    with self.assertRaises(RuntimeError):
      torch.tensor([]).max()
    with self.assertRaises(RuntimeError):
      Tensor([]).max()
    with self.assertRaises(IndexError):
      torch.zeros(0, 3).max(0)
    with self.assertRaises(IndexError):
      Tensor.zeros(0, 3).max(0)

  def test_argmax_empty(self):
    # Argmax on an empty tensor should raise an error like torch does.
    with self.assertRaises(IndexError):
      torch.tensor([]).argmax()
    with self.assertRaises(IndexError):
      Tensor([]).argmax()

  def test_empty_identity_reductions(self):
    # reductions with a defined identity don't go through the max() guard
    assert not torch.tensor([]).any().item()
    assert not Tensor([]).any().item()
    assert Tensor.zeros(0, 3).softmax(0).shape == (0, 3)

class TestDropoutProbabilityEdgeCases(unittest.TestCase):
  # we don't need more of these

  def test_dropout_invalid_prob(self):
    with self.assertRaises(ValueError):
      torch.nn.functional.dropout(torch.ones(10), -0.1, True)
    with self.assertRaises(ValueError):
      with Context(TRAINING=1):
        Tensor.ones(10).dropout(-0.1)

class TestInputValidation(unittest.TestCase):
  # we don't need more of these, input validation bugs are not very interesting, many are WONTFIX

  def test_repeat_negative(self):
    # repeating with a negative value should error like PyTorch
    with self.assertRaises(RuntimeError):
      torch.tensor([1, 2, 3]).repeat(-1, 2)
    with self.assertRaises(RuntimeError):
      Tensor([1, 2, 3]).repeat(-1, 2)

  def test_negative_weight_decay(self):
    with self.assertRaises(ValueError):
      torch.optim.AdamW([torch.tensor([1.], requires_grad=True)], lr=0.1, weight_decay=-0.1)
    with self.assertRaises(ValueError):
      nn.optim.AdamW([Tensor([1.])], lr=0.1, weight_decay=-0.1)

  def test_negative_lr(self):
    with self.assertRaises(ValueError):
      torch.optim.SGD([torch.tensor([1.], requires_grad=True)], lr=-0.1)
    with self.assertRaises(ValueError):
      nn.optim.SGD([Tensor([1.])], lr=-0.1)

  def test_negative_momentum(self):
    with self.assertRaises(ValueError):
      torch.optim.SGD([torch.tensor([1.], requires_grad=True)], lr=0.1, momentum=-0.1)
    with self.assertRaises(ValueError):
      nn.optim.SGD([Tensor([1.])], lr=0.1, momentum=-0.1)

if __name__ == '__main__':
  unittest.main()
