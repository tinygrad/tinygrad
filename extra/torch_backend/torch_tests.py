import sys
import unittest
import torch
import extra.torch_backend.backend

from torch.testing._internal.common_utils import TestCase, is_privateuse1_backend_available
assert is_privateuse1_backend_available() and torch._C._get_privateuse1_backend_name() == "tiny"
from torch.testing._internal.common_device_type import ops, onlyOn, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import unary_ufuncs, binary_ufuncs, op_db, python_ref_db

def to_cpu(arg): return arg.to(device="cpu") if isinstance(arg, torch.Tensor) else arg

class TestTinyBackend(TestCase):
  @ops([x for x in unary_ufuncs if not x.name.startswith("_refs") and not x.name.startswith("special")], allowed_dtypes=[torch.float])
  def test_compare(self, device, dtype, op):
    samples = op.sample_inputs(device, dtype)
    for sample in samples:
      tiny_results = op(sample.input, *sample.args, **sample.kwargs)
      tiny_results = sample.output_process_fn_grad(tiny_results)

      cpu_sample = sample.transform(to_cpu)
      cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
      cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

      self.assertEqual(tiny_results, cpu_results, atol=1e-3, rtol=1e-3)

instantiate_device_type_tests(TestTinyBackend, globals(), only_for=["tiny"])

if __name__ == "__main__":
  unittest.main(verbosity=2)
