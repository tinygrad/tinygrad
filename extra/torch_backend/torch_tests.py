import sys
import unittest
import torch
import extra.torch_backend.backend

# just importing this is triggering lots of compute
#from torch.testing._internal import opinfo
from torch.testing._internal.common_utils import TestCase, is_privateuse1_backend_available
assert is_privateuse1_backend_available()
assert torch._C._get_privateuse1_backend_name() == "tiny"
from torch.testing._internal.common_device_type import ops, onlyOn, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import unary_ufuncs, op_db, python_ref_db

class TestTinyBackend(TestCase):
  @ops(unary_ufuncs, allowed_dtypes=[torch.float])
  def test_runs(self, device, dtype, op):
    samples = op.sample_inputs(device, dtype)
    for sample in samples:
      expected = op(sample.input, *sample.args, **sample.kwargs)

instantiate_device_type_tests(TestTinyBackend, globals(), only_for=["tiny"])

if __name__ == "__main__":
  unittest.main(verbosity=2)
