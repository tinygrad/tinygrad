"""Mutable images keep writable bindings and observe ordered image stores."""
import pytest

from tinygrad.helpers import DEV
from test.mockgpu.qcom.test_image_integration import run_image_program


def test_masked_view_assignment_retains_its_writable_image_binding():
  # Retain the ordinary assignment that exposed a texture-only declaration for
  # an argument which the generated shader also writes.
  run_image_program("""
a = Tensor.ones(4, 4).contiguous().realize()
b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2)
a.assign(a + b).realize()
np.testing.assert_array_equal(a.numpy(), [[2, 2, 3, 3]] * 4)
""", required_ops=("ldib.b", "stib.b"))


@pytest.mark.parametrize("half_storage", [False, True])
def test_initialized_mutable_image_reads_a_prior_store(half_storage):
  # The second row must consume the first row's updated pixels in the same
  # shader. Reorderable texture reads can retain the old values instead.
  run_image_program(f"""
from tinygrad.uop.ops import KernelInfo
def update(image):
  first = image[0].store(image[0] + 1)
  updated = image.after(first)
  second = updated[1].store(updated[0] * 2)
  return second.sink(arg=KernelInfo(name="image_store_then_load", opts_to_apply=()))
values = np.arange(32, dtype=np.float32).reshape(4, 8) / 8
values = values.astype(np.float16 if {half_storage!r} else np.float32)
image = Tensor(values).contiguous().realize()
actual = image.custom_kernel(fxn=update)[0].numpy()
expected = values.copy()
expected[0] += 1
expected[1] = expected[0] * 2
np.testing.assert_array_equal(actual, expected)
if Device["QCOM"].renderer.target.renderer == "IR3" and {{"ldg", "ldib.b", "stib.b"}} <= executed_image_ops:
  assert executed_fences > 0
""", half_storage, required_ops=("ldib.b", "stib.b"))


def test_read_only_texture_and_mutable_image_keep_separate_resource_slots():
  run_image_program("""
left = np.arange(16, dtype=np.float32).reshape(4, 4) / 8
right = np.arange(16, dtype=np.float32).reshape(4, 4) / 16 + 3
a, b = Tensor(left).realize(), Tensor(right).realize()
a.assign(a + b).realize()
np.testing.assert_array_equal(a.numpy(), left + right)
""", required_ops=("isam", "ldib.b", "stib.b"))


def test_read_only_image_inputs_do_not_get_mixed_view_fences():
  run_image_program("""
left = np.arange(768, dtype=np.float32).reshape(3, 64, 4) / 8
right = np.arange(768, dtype=np.float32).reshape(3, 64, 4) / 16 + 3
actual = (Tensor(left) + Tensor(right)).numpy()
np.testing.assert_array_equal(actual, left + right)
assert executed_fences == 0
""")


@pytest.mark.skipif(DEV.target("QCOM").renderer != "CL", reason="QCOMCL compiler metadata and image qualifiers")
def test_qcomcl_read_write_images_compile_with_writable_metadata_and_ordered_reads():
  run_image_program("""
from tinygrad.runtime.ops_qcom import _qcom_program_cache
from tinygrad.uop.ops import KernelInfo
compiled_sources = []
compiler = Device["QCOM"].renderer.compiler
original_compile = compiler.compile
def record_source(source):
  library = original_compile(source)
  compiled_sources.append(source)
  return library
compiler.compile = record_source
left = np.arange(16, dtype=np.float32).reshape(4, 4) / 8
right = left / 2 + 3
a, b = Tensor(left).realize(), Tensor(right).realize()
a.assign(a + b).realize()
np.testing.assert_array_equal(a.numpy(), left + right)
def update(image):
  first = image[0].store(image[0] + 1)
  updated = image.after(first)
  return updated[1].store(updated[0] * 2).sink(arg=KernelInfo(name="image_store_then_load", opts_to_apply=()))
values = np.arange(32, dtype=np.float32).reshape(4, 8) / 8
image = Tensor(values).contiguous().realize()
actual = image.custom_kernel(fxn=update)[0].numpy()
expected = values.copy()
expected[0] += 1
expected[1] = expected[0] * 2
np.testing.assert_array_equal(actual, expected)
mutable_sources = [source for source in compiled_sources if "read_write image2d_t" in source]
assert mutable_sources
# Guarded samplerless reads preserve border zero after image mask elimination.
assert all("(float4)(0.0f)" in source and "(uint)(" in source for source in mutable_sources)
assert any("atomic_work_item_fence(CLK_IMAGE_MEM_FENCE, memory_order_acq_rel, memory_scope_work_item)" in source
           for source in mutable_sources)
mutable_programs = [data for data, _ in _qcom_program_cache.values()
                    if not data.NIR and any(ins.op == "ldib.b" for ins in emu.decode(data.image))]
# Compiler type 3 must be an IBO, alongside the independent read-only texture.
assert any(data.ibo_cnt == 1 and data.tex_cnt == 1 and data.buf_offs == [] for data in mutable_programs)
assert any(ins.op == "ldib.b" and ins.raw & 1 for data in mutable_programs for ins in emu.decode(data.image))
""", required_ops=("isam", "ldib.b", "stib.b"))
