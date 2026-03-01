#!/usr/bin/env python
import time
import unittest
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device


MODEL_WIDTH = 512
MODEL_HEIGHT = 256
W, H = 1928, 1208


def warp_perspective_tinygrad(src, M_inv):
  w_dst, h_dst = (MODEL_WIDTH, MODEL_HEIGHT)
  w_src, h_src = (W,H)

  x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
  y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
  ones = Tensor.ones_like(x)
  dst_coords = x.reshape(1, -1).cat(y.reshape(1, -1)).cat(ones.reshape(1, -1))

  src_coords = M_inv @ dst_coords
  src_coords = src_coords / src_coords[2:3, :]
  x_src = src_coords[0].reshape(h_dst, w_dst)
  y_src = src_coords[1].reshape(h_dst, w_dst)

  x_nn_clipped = x_src.round().clip(0, w_src - 1).cast('int')
  y_nn_clipped = y_src.round().clip(0, h_src - 1).cast('int')

  idx = (y_nn_clipped * w_src + x_nn_clipped).reshape(-1)

  src_flat = src.reshape(h_src * w_src)
  sampled = src_flat[idx]
  return sampled

def warp_perspective_numpy(src, M_inv):
  w_dst, h_dst = (MODEL_WIDTH, MODEL_HEIGHT)
  h_src, w_src = src.shape[:2]
  xs, ys = np.meshgrid(np.arange(w_dst), np.arange(h_dst))
  dst_x = xs.reshape(-1)
  dst_y = ys.reshape(-1)

  ones = np.ones_like(xs)
  dst_hom = np.stack([xs, ys, ones], axis=0).reshape(3, -1)

  src_hom = M_inv @ dst_hom
  src_hom /= src_hom[2:3, :]

  src_x = np.clip(np.round(src_hom[0, :]).astype(int), 0, w_src - 1)
  src_y = np.clip(np.round(src_hom[1, :]).astype(int), 0, h_src - 1)

  dst = np.zeros((h_dst, w_dst), dtype=src.dtype)
  dst[dst_y, dst_x] = src[src_y, src_x]
  return dst.ravel()

class TestImageWarp(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Precision error")
  def test_image_warp_jit(self):
    update_img_jit = TinyJit(warp_perspective_tinygrad, prune=True)

    # run 20 times
    step_times = []
    for _ in range(20):
      inputs = [(32*Tensor.randn(H,W) + 128).cast(dtype='uint8').contiguous().realize(),
                    Tensor.randn(3,3).contiguous().realize()]
      Device.default.synchronize()
      inputs_np = [x.numpy() for x in inputs]

      # do warp
      st = time.perf_counter()
      out = update_img_jit(*inputs)
      mt = time.perf_counter()
      Device.default.synchronize()
      et = time.perf_counter()
      step_times.append((et-st)*1e3)
      print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {step_times[-1]:6.2f} ms")
      out_np = warp_perspective_numpy(*inputs_np)

      mismatch = np.abs(out.numpy() - out_np) > 0
      mismatch_percent = sum(mismatch.flatten()) / len(mismatch.flatten()) * 100
      mismatch_percent_tol = 1e-2
      assert mismatch_percent < mismatch_percent_tol, f"input mismatch percent {mismatch_percent} exceeds tolerance {mismatch_percent_tol}"


if __name__ == '__main__':
  unittest.main()