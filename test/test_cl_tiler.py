#!/usr/bin/env python
import unittest
import torch
import numpy as np
from tinygrad.tensor import Device

def helper_test_tiler(x_shape, pts):
  from tinygrad.llops.ops_gpu import CLProgram
  from tinygrad.llops.ops_opencl import OpenCLBuffer
  torch.manual_seed(0)
  x = torch.randn(*x_shape).numpy()
  targets = np.array([(x[i, j] if (0 <= i < x.shape[0] and 0 <= j < x.shape[1]) else [0]*4) for (i, j) in pts])
  x_buffer, pts_buffer, out_buffer = OpenCLBuffer.fromCPU(x), OpenCLBuffer.fromCPU(np.flip(pts, -1)), OpenCLBuffer(targets.shape)
  x_buffer.image
  CLProgram("test_tiler", f"""
    __kernel void test_tiler(__global float4 *out, read_only image2d_t in, __global const float2 *pts) {{
      const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      int2 l_smp = convert_int2(pts[get_global_id(0)]);
      out[get_global_id(0)] = read_imagef(in, smp, {x_buffer._image.pos_to_sample_pos('l_smp')});
    }}
  """)((len(targets), 1), None, out_buffer.cl, x_buffer.image, pts_buffer.cl)
  out = out_buffer.toCPU()
  np.testing.assert_allclose(out, targets)

def get_pts(*boundary_coords):
  from tinygrad.llops.ops_gpu import CL
  c = sum(([i - 1, i, i + 1] for i in boundary_coords), start=[CL().cl_ctx.devices[0].image2d_max_width])
  return [[i, j] for i in c for j in c]

@unittest.skipUnless(hasattr(Device, "OPENCL"), "Test requires OpenCL")
class TestCLTiler(unittest.TestCase):
  """Test for CLImage tiling logic, which allows large tensors to fit within limited-size OpenCL images."""

  def test_small(self):
    helper_test_tiler((5, 6, 4), get_pts(0, 5, 6))
  def test_wide(self):
    helper_test_tiler((3, 40_000, 4), get_pts(0, 3, 40_000))
  def test_tall(self):
    helper_test_tiler((40_000, 3, 4), get_pts(0, 3, 40_000))