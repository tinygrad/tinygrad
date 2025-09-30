from __future__ import annotations
import ctypes
import sys
import pytest

if sys.platform.startswith("win"):
  pytest.skip("NVDEC unit tests are skipped on Windows CI runners", allow_module_level=True)

@pytest.fixture(autouse=True)
def reset_cuvid_module_state():
  import tinygrad.runtime.cuvid as cuvid
  cuvid.reset_cuvid_cache_for_tests()
  yield
  cuvid.reset_cuvid_cache_for_tests()

class FakeFunction:
  def __init__(self, func):
    self._func = func
    self.argtypes = None
    self.restype = ctypes.c_int
  def __call__(self, *args):
    return self._func(*args)

class FakeLib:
  def __init__(self, func):
    self.cuvidGetDecoderCaps = FakeFunction(func)
    self.cuvidCreateVideoParser = FakeFunction(lambda *_: 0)
    self.cuvidDestroyVideoParser = FakeFunction(lambda *_: 0)
    self.cuvidParseVideoData = FakeFunction(lambda *_: 0)
    self.cuvidCreateDecoder = FakeFunction(lambda *_: 0)
    self.cuvidDestroyDecoder = FakeFunction(lambda *_: 0)
    self.cuvidDecodePicture = FakeFunction(lambda *_: 0)
    self.cuvidMapVideoFrame = FakeFunction(lambda *_: 0)
    self.cuvidUnmapVideoFrame = FakeFunction(lambda *_: 0)

def test_load_cuvid_missing_library(monkeypatch):
  import tinygrad.runtime.cuvid as cuvid
  def fake_loader():
    raise cuvid.CuvidUnavailable("missing")
  monkeypatch.setattr(cuvid, "_load_nvcuvid", fake_loader)
  with pytest.raises(cuvid.CuvidUnavailable):
    cuvid.load_cuvid()
  with pytest.raises(cuvid.CuvidUnavailable):
    cuvid.load_cuvid()

def test_get_decoder_caps_success(monkeypatch):
  import tinygrad.runtime.cuvid as cuvid
  def capture(caps_ptr):
    caps = ctypes.cast(caps_ptr, ctypes.POINTER(cuvid.CUVIDDECODECAPS)).contents
    assert caps.eCodecType == cuvid.cudaVideoCodec_HEVC
    assert caps.eChromaFormat == cuvid.cudaVideoChromaFormat_420
    caps.nMinWidth = 64
    caps.nMinHeight = 64
    caps.nMaxWidth = 3840
    caps.nMaxHeight = 2160
    caps.nMaxMBCount = 1620
    caps.nMaxSlices = 16
    caps.nMaxDecodeSurfaces = 20
    caps.nMaxOutputSurfaces = 4
    caps.bIsSupported = 1
    caps.bIsPartialSupported = 0
    return 0
  lib = cuvid.CuvidLibrary(FakeLib(capture))
  caps = lib.get_decoder_caps()
  assert caps.is_supported is True
  assert caps.is_partial_supported is False
  assert caps.max_width == 3840
  assert caps.max_height == 2160
  assert caps.max_decode_surfaces == 20


def test_get_decoder_caps_error(monkeypatch):
  import tinygrad.runtime.cuvid as cuvid
  def failing(_):
    return 719
  def boom(status, ptr):
    raise RuntimeError("boom")
  monkeypatch.setattr(cuvid.cuda, "cuGetErrorString", boom, raising=False)
  lib = cuvid.CuvidLibrary(FakeLib(failing))
  with pytest.raises(cuvid.CuvidError) as exc:
    lib.get_decoder_caps()
  assert "Unknown CUVID error" in str(exc.value)
