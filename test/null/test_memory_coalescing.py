import unittest

from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.codegen import to_program


def render_stores(offsets:list[int], dtype=dtypes.float) -> str:
  src = UOp.param(0, dtype.ptr(256))
  out = UOp.param(1, dtype.ptr(256))
  stores = []
  for off in offsets:
    idx = UOp.const(dtypes.int, off)
    val = src.index(idx, ptr=True).load()
    stores.append(out.index(idx, ptr=True).store(val + UOp.const(dtype, 1)))
  with Context(IMAGE=1):
    arch = "IMAGE_PITCH_ALIGNMENT=64" + (",cl_khr_fp16" if dtype == dtypes.half else "")
    prg = to_program(UOp.group(*stores).sink(arg=KernelInfo(opts_to_apply=())), OpenCLRenderer(Target("CL", arch=arch)))
  return prg.src[3].arg


class TestMemoryCoalescing(unittest.TestCase):
  def test_aligned_vec4_uses_images(self):
    src = render_stores([0, 1, 2, 3])
    self.assertIn("image2d_t", src)
    self.assertIn("read_imagef", src)
    self.assertIn("write_imagef", src)

  def test_aligned_half_vec4_uses_images(self):
    src = render_stores([0, 1, 2, 3], dtypes.half)
    self.assertIn("image2d_t", src)
    self.assertIn("read_imagef", src)
    self.assertIn("write_imagef", src)

  def test_partial_group_does_not_use_images(self):
    src = render_stores([0, 1, 2, 3, 4])
    self.assertNotIn("image2d_t", src)
    self.assertNotIn("read_imagef", src)
    self.assertNotIn("write_imagef", src)

  def test_misaligned_vec4_does_not_use_images(self):
    src = render_stores([1, 2, 3, 4])
    self.assertNotIn("image2d_t", src)
    self.assertNotIn("read_imagef", src)
    self.assertNotIn("write_imagef", src)


if __name__ == "__main__":
  unittest.main()
