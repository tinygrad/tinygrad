import sys
import pytest
import numpy as np

if sys.platform.startswith("win"):
  pytest.skip("NVDEC NV12 tests are skipped on Windows CI runners", allow_module_level=True)

from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes
from tinygrad.runtime.cuvid import DecodedFrame
from tinygrad.helpers import getenv


pytestmark = pytest.mark.skipif(getenv("MOCKGPU"), reason="NV12 to RGB tests require a real CUDA device")


_COLOR_SPACE_COEFFS = {
  "bt601": (1.4020, -0.344136, -0.714136, 1.7720),
  "bt709": (1.5748, -0.187324, -0.468124, 1.8556),
  "bt2020": (1.4746, -0.164553, -0.571353, 1.8814),
}


def _require_cuda_device():
  try:
    return Device["CUDA"]
  except Exception as exc:  # pragma: no cover - depends on environment
    pytest.skip(f"CUDA device unavailable: {exc}")


def _buffer_pointer(buf: Buffer) -> int:
  base = buf._buf  # type: ignore[attr-defined]
  return base.value if hasattr(base, "value") else int(base)


def _create_nv12_surface(width: int, height: int, pitch: int, y_vals: list[int], uv_vals: list[int]) -> tuple[Buffer, bytearray]:
  total = pitch * height + pitch * (height // 2)
  buf = Buffer("CUDA", total, dtypes.uint8, preallocate=True)
  raw = bytearray(total)
  for row in range(height):
    start = row * pitch
    row_vals = y_vals[row * width:(row + 1) * width]
    raw[start:start + width] = bytes(row_vals)
  uv_offset = pitch * height
  for row in range(height // 2):
    start = uv_offset + row * pitch
    row_vals = uv_vals[row * width:(row + 1) * width]
    raw[start:start + width] = bytes(row_vals)
  buf.copyin(memoryview(raw))
  return buf, raw


def _reference_rgb(width: int, height: int, y_vals: list[int], uv_vals: list[int], coeffs) -> np.ndarray:
  out = np.zeros((height, width, 3), dtype=np.float32)
  coef_rv, coef_gu, coef_gv, coef_bu = coeffs
  for row in range(height):
    for col in range(width):
      y_val = float(y_vals[row * width + col])
      uv_index = (row // 2) * width + (col // 2) * 2
      u = float(uv_vals[uv_index]) - 128.0
      v = float(uv_vals[uv_index + 1]) - 128.0
      r = y_val + coef_rv * v
      g = y_val + coef_gu * u + coef_gv * v
      b = y_val + coef_bu * u
      out[row, col, 0] = min(255.0, max(0.0, r))
      out[row, col, 1] = min(255.0, max(0.0, g))
      out[row, col, 2] = min(255.0, max(0.0, b))
  return out


class _DummyDecoder:
  def _unmap(self, _ptr: int):
    pass


def test_decoded_frame_to_rgb_tensor_cuda():
  dev = _require_cuda_device()
  width, height, pitch = 4, 2, 4
  y_vals = [64, 80, 96, 112, 48, 32, 16, 0]
  uv_vals = [90, 240, 110, 16]
  buf, _ = _create_nv12_surface(width, height, pitch, y_vals, uv_vals)
  frame = None
  try:
    frame = DecodedFrame(_DummyDecoder(), 0, _buffer_pointer(buf), pitch, width, height, 1, 0, 0, 0)
    reference = _reference_rgb(width, height, y_vals, uv_vals, _COLOR_SPACE_COEFFS["bt601"])

    tensor = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.float32, normalize=False, color_space="bt601")
    dev.synchronize()
    rgb = tensor.to("CPU").numpy()
    np.testing.assert_allclose(rgb, reference, atol=1e-4)

    norm_tensor = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.float32, normalize=True, color_space="bt601")
    dev.synchronize()
    rgb_norm = norm_tensor.to("CPU").numpy()
    np.testing.assert_allclose(rgb_norm, reference / 255.0, atol=1e-5)

    fp16_tensor = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.float16, normalize=True, color_space="bt601")
    dev.synchronize()
    rgb_fp16 = fp16_tensor.to("CPU").numpy().astype(np.float32)
    np.testing.assert_allclose(rgb_fp16, reference / 255.0, atol=2e-3)

    u8_tensor = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.uint8, normalize=False, color_space="bt601")
    dev.synchronize()
    rgb_u8 = u8_tensor.to("CPU").numpy()
    expected_u8 = np.clip(np.round(reference), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(rgb_u8, expected_u8)

    with pytest.raises(ValueError):
      frame.to_rgb_tensor(device="CUDA", dtype=dtypes.uint8, normalize=True, color_space="bt601")
  finally:
    if frame is not None:
      frame.release()
    buf.deallocate()


def test_decoded_frame_color_space_variants():
  dev = _require_cuda_device()
  width, height, pitch = 4, 4, 4
  y_vals = [
    64, 64, 64, 64,
    128, 128, 128, 128,
    192, 192, 192, 192,
    255, 255, 255, 255,
  ]
  uv_vals = [128, 128] * 8
  buf, _ = _create_nv12_surface(width, height, pitch, y_vals, uv_vals)
  frame = None
  try:
    frame = DecodedFrame(_DummyDecoder(), 0, _buffer_pointer(buf), pitch, width, height, 1, 0, 0, 0)

    tensor_709 = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.float32, normalize=False, color_space="bt709")
    dev.synchronize()
    rgb_709 = tensor_709.to("CPU").numpy()
    reference_709 = _reference_rgb(width, height, y_vals, uv_vals, _COLOR_SPACE_COEFFS["bt709"])
    np.testing.assert_allclose(rgb_709, reference_709, atol=1e-4)

    tensor_2020 = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.float32, normalize=False, color_space="bt2020")
    dev.synchronize()
    rgb_2020 = tensor_2020.to("CPU").numpy()
    reference_2020 = _reference_rgb(width, height, y_vals, uv_vals, _COLOR_SPACE_COEFFS["bt2020"])
    np.testing.assert_allclose(rgb_2020, reference_2020, atol=1e-4)

    assert not np.allclose(reference_709, reference_2020)
  finally:
    if frame is not None:
      frame.release()
    buf.deallocate()


def test_decoded_frame_invalid_color_space():
  _require_cuda_device()
  width, height, pitch = 2, 2, 2
  y_vals = [0, 0, 0, 0]
  uv_vals = [128, 128]
  buf, _ = _create_nv12_surface(width, height, pitch, y_vals, uv_vals)
  frame = None
  try:
    frame = DecodedFrame(_DummyDecoder(), 0, _buffer_pointer(buf), pitch, width, height, 1, 0, 0, 0)
    with pytest.raises(ValueError):
      frame.to_rgb_tensor(color_space="unknown")
  finally:
    if frame is not None:
      frame.release()
    buf.deallocate()


def test_decoded_frame_custom_color_space_tuple():
  dev = _require_cuda_device()
  width, height, pitch = 2, 2, 2
  y_vals = [64, 64, 64, 64]
  uv_vals = [128, 128]
  coeffs = (1.6, -0.3, -0.6, 1.9)
  buf, _ = _create_nv12_surface(width, height, pitch, y_vals, uv_vals)
  frame = None
  try:
    frame = DecodedFrame(_DummyDecoder(), 0, _buffer_pointer(buf), pitch, width, height, 1, 0, 0, 0)
    tensor = frame.to_rgb_tensor(device="CUDA", dtype=dtypes.float32, normalize=False, color_space=coeffs)
    dev.synchronize()
    rgb = tensor.to("CPU").numpy()
    reference = _reference_rgb(width, height, y_vals, uv_vals, coeffs)
    np.testing.assert_allclose(rgb, reference, atol=1e-4)
  finally:
    if frame is not None:
      frame.release()
    buf.deallocate()


def test_decoded_frame_invalid_color_space_tuple_length():
  _require_cuda_device()
  width, height, pitch = 2, 2, 2
  y_vals = [0, 0, 0, 0]
  uv_vals = [128, 128]
  buf, _ = _create_nv12_surface(width, height, pitch, y_vals, uv_vals)
  frame = None
  try:
    frame = DecodedFrame(_DummyDecoder(), 0, _buffer_pointer(buf), pitch, width, height, 1, 0, 0, 0)
    with pytest.raises(TypeError):
      frame.to_rgb_tensor(color_space=(1.0, 2.0, 3.0))
  finally:
    if frame is not None:
      frame.release()
    buf.deallocate()