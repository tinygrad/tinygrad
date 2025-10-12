from __future__ import annotations

import os
from pathlib import Path
import ctypes
import types

import pytest

from tinygrad.device import Device
from tinygrad.runtime import cuvid as cuvid_module
from tinygrad.runtime.cuvid import CuvidUnavailable, CUDA_SUCCESS, CUVID_PKT_ENDOFSTREAM

SAMPLE_ENV = "TINYGRAD_NVDEC_SAMPLE"

# Minimal valid HEVC bitstream (16x16 I-frame) for testing
MINIMAL_HEVC = (
  b'\x00\x00\x00\x01\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\x00\x03\x00\x3c\x95' +  # VPS
  b'\x00\x00\x00\x01\x42\x01\x01\x01\x60\x00\x00\x03\x00\x00\x03\x00\x3c\xa0\x08\x08\x08\x20\x66\x59\xd5\x49\x6b' +  # SPS
  b'\x00\x00\x00\x01\x44\x01\xc1\x73\xd1\x89' +  # PPS
  b'\x00\x00\x00\x01\x26\x01\xaf\x08\x54\x65\x73\x74'  # IDR slice
)


class FakeCuvidLib:
  def __init__(self):
    self.seq_cb = None
    self.dec_cb = None
    self.disp_cb = None
    self.user = None
    self.decoder_handle = ctypes.c_void_p()
    self.decoder_destroyed = False
    self.parser_destroyed = False
    self.map_count = 0
    self.unmap_count = 0
    self.did_sequence = False

  def cuvidCreateVideoParser(self, parser_ptr, params_ptr):
    parser_ptr = ctypes.cast(parser_ptr, ctypes.POINTER(ctypes.c_void_p))
    params = ctypes.cast(params_ptr, ctypes.POINTER(cuvid_module.CUVIDPARSERPARAMS)).contents
    self.seq_cb = ctypes.cast(params.pfnSequenceCallback, cuvid_module.SEQUENCE_CALLBACK)
    self.dec_cb = ctypes.cast(params.pfnDecodePicture, cuvid_module.DECODE_CALLBACK)
    self.disp_cb = ctypes.cast(params.pfnDisplayPicture, cuvid_module.DISPLAY_CALLBACK)
    self.user = params.pUserData
    parser_ptr[0] = ctypes.c_void_p(1)
    return CUDA_SUCCESS

  def cuvidDestroyVideoParser(self, parser):
    self.parser_destroyed = True
    return CUDA_SUCCESS

  def cuvidParseVideoData(self, parser, packet_ptr):
    packet = ctypes.cast(packet_ptr, ctypes.POINTER(cuvid_module.CUVIDSOURCEDATAPACKET)).contents
    if packet.flags == CUVID_PKT_ENDOFSTREAM:
      return CUDA_SUCCESS
    if not self.did_sequence:
      fmt = cuvid_module.CUVIDEOFORMAT()
      fmt.codec = cuvid_module.cudaVideoCodec_HEVC
      fmt.chroma_format = cuvid_module.cudaVideoChromaFormat_420
      fmt.coded_width = 64
      fmt.coded_height = 32
      fmt.bit_depth_luma_minus8 = 0
      self.seq_cb(self.user, ctypes.byref(fmt))
      self.did_sequence = True
    pic = cuvid_module.CUVIDPICPARAMS()
    self.dec_cb(self.user, ctypes.byref(pic))
    disp = cuvid_module.CUVIDPARSERDISPINFO()
    disp.picture_index = 0
    disp.progressive_frame = 1
    disp.top_field_first = 1
    disp.timestamp = packet.timestamp
    self.disp_cb(self.user, ctypes.byref(disp))
    return CUDA_SUCCESS

  def cuvidCreateDecoder(self, decoder_ptr, info_ptr):
    decoder_ptr = ctypes.cast(decoder_ptr, ctypes.POINTER(ctypes.c_void_p))
    info = ctypes.cast(info_ptr, ctypes.POINTER(cuvid_module.CUVIDDECODECREATEINFO)).contents
    self.decoder_handle = ctypes.c_void_p(2)
    decoder_ptr[0] = self.decoder_handle
    self.decoder_info = info
    return CUDA_SUCCESS

  def cuvidDestroyDecoder(self, decoder):
    assert decoder.value == self.decoder_handle.value
    assert self.map_count == self.unmap_count
    self.decoder_destroyed = True
    return CUDA_SUCCESS

  def cuvidDecodePicture(self, decoder, pic_ptr):
    assert decoder.value == self.decoder_handle.value
    return CUDA_SUCCESS

  def cuvidMapVideoFrame(self, decoder, picture_index, devptr_ptr, pitch_ptr, proc_ptr):
    assert decoder.value == self.decoder_handle.value
    self.map_count += 1
    devptr = ctypes.cast(devptr_ptr, ctypes.POINTER(ctypes.c_ulonglong))
    pitch = ctypes.cast(pitch_ptr, ctypes.POINTER(ctypes.c_uint))
    devptr.contents.value = 0xABC000 + self.map_count * 0x1000
    pitch.contents.value = 128
    return CUDA_SUCCESS

  def cuvidUnmapVideoFrame(self, decoder, devptr):
    assert decoder.value == self.decoder_handle.value
    self.unmap_count += 1
    return CUDA_SUCCESS


def test_cuvid_surface_lifetime_without_hardware():
  fake = FakeCuvidLib()
  surfaces = cuvid_module.decode_annexb([b"\x00\x00"], _library=types.SimpleNamespace(lib=fake))
  assert len(surfaces) == 1
  surface = surfaces[0]
  assert surface.width == 64 and surface.height == 32
  assert surface.pitch == 128
  assert not fake.decoder_destroyed
  surface.release()
  assert fake.decoder_destroyed
  assert fake.unmap_count == 1
  surface.release()
  assert fake.unmap_count == 1  # idempotent
  assert fake.parser_destroyed


@pytest.mark.skipif(SAMPLE_ENV not in os.environ, reason=f"set {SAMPLE_ENV} to a raw HEVC Annex-B sample to run")
def test_nvdec_surfaces_release():
  sample_path = Path(os.environ[SAMPLE_ENV])
  data = sample_path.read_bytes()
  assert data, "sample bitstream must not be empty"

  try:
    dev = Device["NV"]
  except Exception as exc:  # noqa: BLE001
    pytest.skip(f"NV device unavailable: {exc}")

  try:
    surfaces = dev.decode_hevc_annexb(data)
  except CuvidUnavailable as exc:
    pytest.skip(f"libnvcuvid unavailable: {exc}")

  assert surfaces, "decoder returned no surfaces"

  for surface in surfaces:
    buf = surface.buffer
    assert buf.options and buf.options.external_ptr is not None
    assert buf.nbytes > 0
    assert surface.width > 0 and surface.height > 0 and surface.pitch > 0
    buf.deallocate()


@pytest.mark.skipif(not cuvid_module.is_available(), reason="libnvcuvid not available")
def test_decode_minimal_sample():
  """Test decoding minimal embedded HEVC sample"""
  try:
    surfaces = cuvid_module.decode_annexb(MINIMAL_HEVC)
    assert len(surfaces) >= 1, "Should decode at least one frame"
    assert surfaces[0].width == 16 and surfaces[0].height == 16, "Should be 16x16"
    for s in surfaces: s.release()
  except CuvidUnavailable:
    pytest.skip("CUVID hardware not available")
