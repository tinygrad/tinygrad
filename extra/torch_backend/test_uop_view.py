# simple view-spec tests
import unittest
import torch
from tinygrad.helpers import getenv
from extra.torch_backend.backend import unwrap
from extra.torch_backend.uop_view import view_spec_from_uop

if getenv("TINY_BACKEND2"):
  device = "cpu"
else:
  device = "tiny"

class TestUopView(unittest.TestCase):
  def test_view_spec_transpose(self):
    a = torch.arange(6, device=device).reshape(2, 3)
    spec = view_spec_from_uop(unwrap(a.transpose(0, 1)).uop)
    self.assertEqual(tuple(int(x) for x in spec.shape), (3, 2))
    self.assertEqual(tuple(int(x) for x in spec.strides), (1, 3))
    self.assertEqual(int(spec.offset), 0)

  def test_view_spec_shrink(self):
    a = torch.arange(6, device=device)
    spec = view_spec_from_uop(unwrap(a[2:5]).uop)
    self.assertEqual(tuple(int(x) for x in spec.shape), (3,))
    self.assertEqual(tuple(int(x) for x in spec.strides), (1,))
    self.assertEqual(int(spec.offset), 2)
