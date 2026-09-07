"""Image results must come from compiler-generated instructions through HCQ2."""
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest
from tinygrad.helpers import DEV


def run_image_program(code, half_storage=False, required_ops=("isam", "stib.b")):
  # Preserve the real compiler, submission, decoder and executor. Recording the
  # executed instructions prevents a numerically correct buffer fallback from
  # silently satisfying the image contract.
  renderer = DEV.target("QCOM").renderer or "IR3"
  setup = """import numpy as np
from tinygrad import Device, Tensor
from test.mockgpu.qcom import emu
Device["QCOM"]
executed_image_ops = set()
executed_texture_counts = set()
executed_indirect_sampling = []
executed_direct_textures = set()
executed_component_bytes = set()
executed_compiler_modes = set()
executed_fences = 0
original_execute = emu.execute
def record_execution(image, *args, **kwargs):
  global executed_fences
  result = original_execute(image, *args, **kwargs)
  executed_compiler_modes.add(kwargs.get("opencl", False))
  instructions = emu.decode(image)
  executed_fences += sum(ins.op == "fence" for ins in instructions)
  executed_image_ops.update(ins.op for ins in instructions if ins.raw >> 61 in (5, 6))
  executed_texture_counts.add(len(kwargs["image_bindings"].textures))
  bindings = kwargs["image_bindings"]
  executed_component_bytes.update(resource.component_bytes for resource in bindings.textures + bindings.outputs)
  executed_indirect_sampling.extend(ins for ins in instructions if ins.op == "isam" and "INDICES" in ins.operands)
  executed_direct_textures.update(ins.fields["TEX"] for ins in instructions if ins.op == "isam" and "INDICES" not in ins.operands)
  return result
emu.execute = record_execution
"""
  finish = f"assert {set(required_ops)!r} <= executed_image_ops, executed_image_ops\n" + """
from test.mockgpu.mockgpu import drivers
assert drivers[0].error is None
assert all(ctx.retired == ctx.queued and not ctx.pending for ctx in drivers[0].contexts.values())
"""
  finish += f"assert {2 if half_storage else 4} in executed_component_bytes, executed_component_bytes\n"
  finish += f"assert executed_compiler_modes == {{{renderer == 'CL'!r}}}, executed_compiler_modes\n"
  with tempfile.TemporaryDirectory() as cache:
    environment = {
      **os.environ, "DEV": f"MOCK+QCOM:{renderer}", "HCQ_RUNTIME_DEV": "CPU", "PARALLEL": "0",
      "CACHELEVEL": "0", "OMP_NUM_THREADS": "1", "IMAGE": "1", "FLOAT16": str(int(half_storage)),
      "XDG_CACHE_HOME": cache, "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
      [sys.executable, "-c", setup + code + finish], cwd=pathlib.Path(__file__).parents[3],
      env=environment, capture_output=True, text=True, timeout=120,
    )
  assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("half_storage", [False, True])
def test_image_matmul_preserves_nonuniform_values_and_odd_channels(half_storage):
  # Wrong descriptor table selection, x/y order or channel padding changes these
  # independent NumPy results; an image-only allowlist change cannot pass.
  run_image_program("""
left = np.arange(15, dtype=np.float32).reshape(3, 5) / 8 - 1
right = np.arange(35, dtype=np.float32).reshape(5, 7) / 16 - 0.75
actual = (Tensor(left) @ Tensor(right)).numpy()
np.testing.assert_allclose(actual, left @ right, rtol=2e-3, atol=2e-3)
""", half_storage)


def test_seventeen_image_inputs_run_through_hcq2_without_truncating_the_table():
  run_image_program("""
values = [np.arange(768, dtype=np.float32).reshape(3, 64, 4) / 16 + index for index in range(17)]
inputs = [Tensor(value).realize() for value in values]
actual = sum(inputs).numpy()
np.testing.assert_array_equal(actual, np.sum(values, axis=0))
assert 17 in executed_texture_counts, executed_texture_counts
# Mesa uses a distinct sampler per texture and reaches the indirect form at
# sampler 16. QCOMCL shares a sampler and can encode texture 16 directly.
if Device["QCOM"].renderer.target.renderer == "IR3":
  assert executed_indirect_sampling
else:
  assert 16 in executed_direct_textures, executed_direct_textures
""")


@pytest.mark.parametrize("half_storage", [False, True])
@pytest.mark.parametrize("depthwise", [False, True])
def test_image_convolution_padding_and_groups(half_storage, depthwise):
  run_image_program(f"""
import torch
import torch.nn.functional as functional
torch.set_num_threads(1)
values = (np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) % 13 - 6) / 8
groups, output_channels = (3, 3) if {depthwise!r} else (1, 5)
weight_shape = (output_channels, 3 // groups, 3, 2)
weights = (np.arange(np.prod(weight_shape), dtype=np.float32).reshape(weight_shape) % 11 - 5) / 8
actual = Tensor(values).conv2d(Tensor(weights), groups=groups, padding=1).numpy()
expected = functional.conv2d(torch.tensor(values), torch.tensor(weights), groups=groups, padding=1).numpy()
np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
""", half_storage)


@pytest.mark.parametrize("half_storage", [False, True])
def test_image_convolution_backward_uses_real_image_results(half_storage):
  run_image_program("""
import torch
import torch.nn.functional as functional
torch.set_num_threads(1)
values = (np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4) % 9 - 4) / 8
weights = (np.arange(36, dtype=np.float32).reshape(3, 3, 2, 2) % 7 - 3) / 8
left, kernel = Tensor(values), Tensor(weights)
left_gradient, kernel_gradient = left.conv2d(kernel, padding=1).square().sum().gradient(left, kernel)
reference_left = torch.tensor(values, requires_grad=True)
reference_kernel = torch.tensor(weights, requires_grad=True)
functional.conv2d(reference_left, reference_kernel, padding=1).square().sum().backward()
np.testing.assert_allclose(left_gradient.numpy(), reference_left.grad.numpy(), rtol=4e-3, atol=4e-3)
np.testing.assert_allclose(kernel_gradient.numpy(), reference_kernel.grad.numpy(), rtol=4e-3, atol=4e-3)
""", half_storage)
