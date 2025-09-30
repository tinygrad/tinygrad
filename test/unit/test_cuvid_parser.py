from __future__ import annotations
import ctypes
import sys
import pytest

if sys.platform.startswith("win"):
  pytest.skip("NVDEC parser tests are skipped on Windows CI runners", allow_module_level=True)


from tinygrad.runtime.cuvid import (
  CUVIDDECODECREATEINFO,
  CUVIDPARSERDISPINFO,
  CUVIDPARSERPARAMS,
  CUVIDPICPARAMS,
  CUVIDSOURCEDATAPACKET,
  CUVIDEOFORMAT,
  CUVID_PKT_ENDOFSTREAM,
  DecodedFrame,
  NVVideoDecoder,
  NV12Plane,
  CuvidError,
  CuvidLibrary,
  CuvidUnavailable,
  _TimestampManager,
  nv12_frames_to_tensors,
)
from tinygrad.runtime.testing.cuvid_stub import StubCuvidLib, StubFunc, build_demo_annexb_stream, prepare_demo_callbacks
from tinygrad.dtype import dtypes
import tinygrad.runtime.cuvid as cuvid


def _header(nal_type: int) -> bytes:
  nal_type &= 0x3f
  nuh_layer_id = 0
  nuh_temporal_id_plus1 = 1
  first_byte = (nal_type << 1) | (nuh_layer_id >> 5)
  second_byte = ((nuh_layer_id & 0x1f) << 3) | (nuh_temporal_id_plus1 & 0x7)
  return bytes([first_byte, second_byte])

@pytest.fixture(autouse=True)
def reset_cache():
  import tinygrad.runtime.cuvid as cuvid
  cuvid.reset_cuvid_cache_for_tests()
  yield
  cuvid.reset_cuvid_cache_for_tests()

def _make_lib():
  stub = StubCuvidLib()
  stub.attach()
  return CuvidLibrary(stub)


def test_create_and_destroy_parser(monkeypatch):
  lib = _make_lib()
  params = CUVIDPARSERPARAMS()
  params.CodecType = 8
  params.ulMaxNumDecodeSurfaces = 4
  params.pfnSequenceCallback = ctypes.c_void_p(100)
  handle = lib.create_video_parser(params)
  assert isinstance(handle, ctypes.c_void_p)
  assert len(lib._lib.created_parsers) == 1
  stored_handle, stored_params = lib._lib.created_parsers[0]
  assert stored_handle.value == handle.value
  assert stored_params.CodecType == params.CodecType
  lib.destroy_video_parser(handle)
  assert lib._lib.destroyed_parsers[0].value == handle.value


def test_parse_video_data(monkeypatch):
  lib = _make_lib()
  params = CUVIDPARSERPARAMS()
  handle = lib.create_video_parser(params)
  packet = CUVIDSOURCEDATAPACKET()
  packet.flags = 1
  packet.payload_size = 5
  payload_buf = ctypes.create_string_buffer(b"abcde")
  packet.payload = ctypes.cast(payload_buf, ctypes.c_void_p)
  lib.parse_video_data(handle, packet)
  recorded = lib._lib.parsed_packets[0]
  assert recorded.flags == packet.flags
  assert recorded.payload_size == packet.payload_size


def test_decoder_lifecycle():
  lib = _make_lib()
  info = CUVIDDECODECREATEINFO()
  info.ulWidth = 1920
  info.ulHeight = 1080
  info.ulNumDecodeSurfaces = 8
  info.CodecType = 8
  decoder = lib.create_decoder(info)
  assert decoder.value != 0
  lib.destroy_decoder(decoder)
  assert lib._lib.destroyed_decoders[0].value == decoder.value


def test_decode_and_map():
  lib = _make_lib()
  decoder = lib.create_decoder(CUVIDDECODECREATEINFO())
  params = CUVIDPICPARAMS()
  params.CurrPicIdx = 3
  lib.decode_picture(decoder, params)
  assert lib._lib.decode_calls[0].CurrPicIdx == 3
  dev_ptr, pitch = lib.map_video_frame(decoder, 3, ctypes.c_void_p(0))
  assert dev_ptr in lib._lib.mapped_surfaces
  assert pitch > 0
  lib.unmap_video_frame(decoder, dev_ptr)
  assert lib._lib.unmap_calls[0][1].value == dev_ptr


def test_nvdecoder_sequence_creates_decoder():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib, max_decode_surfaces=4)
  fmt = CUVIDEOFORMAT()
  fmt.codec = cuvid.cudaVideoCodec_HEVC
  fmt.chroma_format = cuvid.cudaVideoChromaFormat_420
  fmt.bit_depth_luma_minus8 = 0
  fmt.coded_width = 1280
  fmt.coded_height = 720
  decoder._handle_sequence(fmt)
  assert len(stub.created_decoders) == 1
  _, info = stub.created_decoders[0]
  assert info.ulWidth == 1280
  assert info.ulNumDecodeSurfaces == 4
  decoder.close()


def test_nvdecoder_decode_and_display_queue():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  fmt = CUVIDEOFORMAT()
  fmt.codec = cuvid.cudaVideoCodec_HEVC
  fmt.chroma_format = cuvid.cudaVideoChromaFormat_420
  fmt.bit_depth_luma_minus8 = 0
  fmt.coded_width = 640
  fmt.coded_height = 360
  pic = CUVIDPICPARAMS()
  pic.CurrPicIdx = 1
  disp = CUVIDPARSERDISPINFO()
  disp.picture_index = 1
  disp.progressive_frame = 1
  disp.top_field_first = 0
  disp.repeat_first_field = 0
  disp.timestamp = 1234
  decoder._handle_sequence(fmt)
  decoder._handle_decode(pic)
  decoder._handle_display(disp)
  assert stub.decode_calls[0].CurrPicIdx == 1
  frame = decoder.acquire_frame()
  assert isinstance(frame, DecodedFrame)
  assert frame.picture_index == 1
  assert frame.device_pointer in stub.mapped_surfaces
  assert frame.width == 640 and frame.height == 360
  assert decoder.pending_frames() == 0
  base_ptr = frame.device_pointer
  frame.release()
  assert stub.unmap_calls[0][1].value == base_ptr
  decoder.close()


def test_nvdecoder_feed_packet_records_payload():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  decoder.feed_packet(b"abc", timestamp=5)
  assert stub.parsed_packets[0].payload_size == 3
  assert stub.parsed_packets[0].timestamp == 5
  decoder.close()

def test_nvdecoder_feed_annexb_stream():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  nalus = [
    _header(32) + b"VPS",
    _header(33) + b"SPS",
    _header(1) + b"\x80SLICE",
    _header(33) + b"SPS2",
    _header(1) + b"\x80SLICE2",
  ]
  data = b"".join([
    b"\x00\x00\x01" + nalus[0],
    b"\x00\x00\x00\x01" + nalus[1],
    b"\x00\x00\x01" + nalus[2],
    b"\x00\x00\x00\x01" + nalus[3],
    b"\x00\x00\x01" + nalus[4],
  ])
  count = decoder.feed_annexb_stream(data, timestamps=[100, 200])
  assert count == 2
  assert len(stub.packet_payloads) == 6
  first_payload = stub.packet_payloads[0]
  assert first_payload.startswith(b"\x00\x00\x01")
  assert first_payload[3:5] == _header(32)
  assert stub.packet_payloads[1].startswith(b"\x00\x00\x01")
  assert stub.packet_payloads[2].startswith(b"\x00\x00\x01")
  assert stub.packet_payloads[3].startswith(b"\x00\x00\x01")
  assert stub.packet_payloads[4].startswith(b"\x00\x00\x01")
  timestamps_recorded = [pkt.timestamp for pkt in stub.parsed_packets[:5]]
  assert timestamps_recorded == [100, 100, 100, 200, 200]
  assert stub.packet_payloads[5] == b""
  assert stub.parsed_packets[5].flags == CUVID_PKT_ENDOFSTREAM
  decoder.close()


def test_nvdecoder_feed_annexb_stream_timestamp_exhaustion():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  nalus = [
    _header(33) + b"SPS",
    _header(1) + b"\x80SLICE_A",
    _header(1) + b"\x00SLICE_B",
    _header(1) + b"\x80SLICE_C",
  ]
  data = b"".join([b"\x00\x00\x00\x01" + n for n in nalus])
  count = decoder.feed_annexb_stream(data, timestamps=[111])
  assert count == 2
  sent_ts = [pkt.timestamp for pkt in stub.parsed_packets[:4]]
  assert sent_ts == [111, 111, 111, 1]
  decoder.close()


def test_nvdecoder_feed_annexb_stream_no_timestamps():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  nalus = [
    _header(1) + b"\x80SLICE0",
    _header(1) + b"\x00SLICE_CONT",
    _header(33) + b"SPS",
    _header(1) + b"\x80SLICE1",
  ]
  data = b"".join([b"\x00\x00\x01" + nalus[0], b"\x00\x00\x01" + nalus[1], b"\x00\x00\x01" + nalus[2], b"\x00\x00\x01" + nalus[3]])
  count = decoder.feed_annexb_stream(data)
  assert count == 2
  sent_ts = [pkt.timestamp for pkt in stub.parsed_packets[:4]]
  assert sent_ts == [0, 0, 1, 1]
  decoder.close()


def test_timestamp_manager_with_iterable():
  mgr = _TimestampManager([50, 60])
  assert mgr.peek() == 50
  assert mgr.peek() == 50  # peek should be idempotent
  assert mgr.consume() == 50
  assert mgr.frames_used == 1
  assert mgr.peek() == 60
  assert mgr.consume() == 60
  assert mgr.frames_used == 2
  assert mgr.peek() == 2  # exhausted timestamps fall back to counter


def test_timestamp_manager_auto_sequence():
  mgr = _TimestampManager(None)
  assert [mgr.consume() for _ in range(4)] == [0, 1, 2, 3]
  assert mgr.peek() == 4


def test_prepare_demo_callbacks_enqueues_expected_events():
  stub = StubCuvidLib()
  stub.attach()
  prepare_demo_callbacks(stub, frames=3, timestamps=[0, 10, 20])
  events = [event for event, _ in stub.callback_events]
  assert events.count("sequence") == 1
  assert events.count("decode") == 3
  assert events.count("display") == 3


def test_decode_annexb_stream_with_stub_callbacks(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  data, timestamps = build_demo_annexb_stream(frames=3)
  prepare_demo_callbacks(stub, frames=3, timestamps=timestamps)

  def fake_nv12_frames_to_tensors(frames, **_):
    for frame in frames:
      try:
        yield (frame.timestamp, f"frame-{frame.picture_index}")
      finally:
        frame.release()

  monkeypatch.setattr(cuvid, "nv12_frames_to_tensors", fake_nv12_frames_to_tensors)

  out = list(decoder.decode_annexb_stream(data, timestamps=timestamps, include_timestamps=True,
                                          device="CUDA", dtype=dtypes.float32, normalize=False))
  assert out == [(ts, f"frame-{idx}") for idx, ts in enumerate(timestamps)]
  assert [call.CurrPicIdx for call in stub.decode_calls] == [0, 1, 2]
  assert len(stub.unmap_calls) == 3
  decoder.close()


def test_decode_annexb_stream_no_end_of_stream(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  data, timestamps = build_demo_annexb_stream(frames=2)
  prepare_demo_callbacks(stub, frames=2, timestamps=timestamps)

  def fake_nv12_frames_to_tensors(frames, **_):
    for frame in frames:
      try:
        yield (frame.timestamp, frame.picture_index)
      finally:
        frame.release()

  eos_calls: list[bool] = []
  monkeypatch.setattr(decoder, "feed_end_of_stream", lambda: eos_calls.append(True))
  monkeypatch.setattr(cuvid, "nv12_frames_to_tensors", fake_nv12_frames_to_tensors)

  out = list(decoder.decode_annexb_stream(data, timestamps=timestamps, include_timestamps=True,
                                          device="CUDA", dtype=dtypes.float32, normalize=False,
                                          end_of_stream=False))
  assert out == [(ts, idx) for idx, ts in enumerate(timestamps)]
  assert eos_calls == []
  decoder.close()


def _create_decoder_with_frame(width=192, height=112):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  fmt = CUVIDEOFORMAT()
  fmt.codec = cuvid.cudaVideoCodec_HEVC
  fmt.chroma_format = cuvid.cudaVideoChromaFormat_420
  fmt.bit_depth_luma_minus8 = 0
  fmt.coded_width = width
  fmt.coded_height = height
  decoder._handle_sequence(fmt)
  pic = CUVIDPICPARAMS()
  pic.CurrPicIdx = 0
  disp = CUVIDPARSERDISPINFO()
  disp.picture_index = 0
  disp.progressive_frame = 1
  disp.top_field_first = 0
  disp.repeat_first_field = 0
  disp.timestamp = 77
  decoder._handle_decode(pic)
  decoder._handle_display(disp)
  frame = decoder.acquire_frame()
  assert frame is not None
  return stub, decoder, frame


def test_decoded_frame_nv12_planes():
  stub, decoder, frame = _create_decoder_with_frame(width=190, height=118)
  base_ptr = frame.device_pointer
  assert base_ptr is not None
  y_plane = frame.luma_plane
  uv_plane = frame.chroma_plane
  assert isinstance(y_plane, NV12Plane)
  assert isinstance(uv_plane, NV12Plane)
  assert y_plane.pointer == base_ptr
  expected_pitch = frame.pitch
  assert y_plane.pitch == expected_pitch
  assert uv_plane.pitch == expected_pitch
  assert uv_plane.pointer == base_ptr + expected_pitch * frame.height
  assert y_plane.width == 190
  assert y_plane.height == 118
  assert uv_plane.width == max(190 // 2, 1)
  assert uv_plane.height == max(118 // 2, 1)
  expected_size = expected_pitch * frame.height + expected_pitch * ((frame.height + 1) // 2)
  assert frame.surface_size_bytes == expected_size
  assert frame.is_mapped
  frame.release()
  assert not frame.is_mapped
  assert stub.unmap_calls[-1][1].value == base_ptr
  decoder.close()


def test_decoded_frame_context_manager_unmaps():
  stub, decoder, frame = _create_decoder_with_frame(width=128, height=64)
  with frame as ctx:
    ptr = ctx.luma_plane.pointer
    assert ctx.is_mapped
  assert not frame.is_mapped
  assert stub.unmap_calls[-1][1].value == ptr
  decoder.close()


def test_acquire_rgb_tensor_uses_frame(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  class FakeFrame:
    def __init__(self):
      self.released = False
      self.called_kwargs = None
    def to_rgb_tensor(self, **kwargs):
      self.called_kwargs = kwargs
      return "rgb-tensor"
    def release(self):
      self.released = True
  fake = FakeFrame()
  monkeypatch.setattr(decoder, "acquire_frame", lambda: fake)
  result = decoder.acquire_rgb_tensor(device="CUDA", dtype=dtypes.float32, normalize=False)
  assert result == "rgb-tensor"
  assert fake.called_kwargs == {"device": "CUDA", "dtype": dtypes.float32, "normalize": False, "color_space": "bt709"}
  assert fake.released
  decoder.close()


def test_frames_to_tensors_generator(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)

  class FakeFrame:
    def __init__(self, ts, value):
      self.timestamp = ts
      self.value = value
      self.released = False
    def to_rgb_tensor(self, **_):
      return self.value
    def release(self):
      self.released = True

  frames = [FakeFrame(1, "a"), FakeFrame(2, "b")]
  monkeypatch.setattr(decoder, "drain_frames", lambda: iter(frames))

  out = list(decoder.frames_to_tensors(device="CUDA", dtype=dtypes.float32, normalize=True))
  assert out == [(1, "a"), (2, "b")]
  assert all(f.released for f in frames)

  frames = [FakeFrame(3, "c")]
  monkeypatch.setattr(decoder, "drain_frames", lambda: iter(frames))
  out = list(decoder.frames_to_tensors(device="CUDA", dtype=dtypes.float32, normalize=False, include_timestamps=False))
  assert out == ["c"]
  assert frames[0].released

  frame = FakeFrame(4, "d")
  out = list(nv12_frames_to_tensors([frame], device="CUDA", dtype=dtypes.float32,
                                    normalize=True, include_timestamps=True))
  assert out == [(4, "d")]
  assert frame.released


def test_nv12_frames_to_tensors_color_space(monkeypatch):
  class FakeFrame:
    def __init__(self):
      self.timestamp = 123
      self.kwargs = []
      self.released = False
    def to_rgb_tensor(self, **kwargs):
      self.kwargs.append(kwargs)
      return "tensor"
    def release(self):
      self.released = True

  frame = FakeFrame()
  out = list(nv12_frames_to_tensors([frame], device="CUDA", dtype=dtypes.float32,
                                    normalize=False, include_timestamps=True, color_space="bt2020"))
  assert out == [(123, "tensor")]
  assert frame.kwargs == [{"device": "CUDA", "dtype": dtypes.float32, "normalize": False, "color_space": "bt2020"}]
  assert frame.released


def test_decode_annexb_iter_color_space_forwarding(monkeypatch):
  captured: dict[str, object] = {}

  class FakeDecoder:
    def __init__(self, **_):
      captured["constructed"] = True
    def decode_annexb_stream(self, *_args, **kwargs):
      captured["stream_kwargs"] = kwargs
      yield (0, "frame")
    def close(self):
      captured["closed"] = True

  monkeypatch.setattr(cuvid, "NVVideoDecoder", FakeDecoder)
  result = list(cuvid.decode_annexb_iter(b"data", color_space="bt2020"))
  assert result == [(0, "frame")]
  assert captured["stream_kwargs"]["color_space"] == "bt2020"
  assert captured.get("closed") is True


def test_decode_annexb_to_tensors_auto_color_space(monkeypatch):
  class FakeDecoder:
    def __init__(self, **_):
      pass
    def decode_annexb_stream(self, *_args, **kwargs):
      assert kwargs["color_space"] == (1.0, 2.0, 3.0, 4.0)
      yield "tensor"
    def close(self):
      pass

  monkeypatch.setattr(cuvid, "NVVideoDecoder", FakeDecoder)
  tensors = cuvid.decode_annexb_to_tensors_auto(b"data", color_space=(1.0, 2.0, 3.0, 4.0))
  assert tensors == ["tensor"]


def test_decode_annexb_stream_generator(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)

  feed_calls: list[tuple[bytes, int, int]] = []
  eos_calls: list[bool] = []

  class FakeFrame:
    def __init__(self, timestamp, value):
      self.timestamp = timestamp
      self.value = value
      self.kwargs = []
      self.released = False
    def to_rgb_tensor(self, **kwargs):
      self.kwargs.append(kwargs)
      return self.value
    def release(self):
      self.released = True

  frames = [FakeFrame(100, "tensor-0"), FakeFrame(200, "tensor-1")]
  consumed: list[FakeFrame] = []

  def fake_feed_packet(payload, timestamp=0, flags=0):
    feed_calls.append((bytes(payload), timestamp, flags))
    if frames:
      raw = feed_calls[-1][0]
      start_len = 4 if raw.startswith(b"\x00\x00\x00\x01") else 3
      header = raw[start_len:start_len + 2]
      if len(header) == 2:
        nal_type = (header[0] >> 1) & 0x3F
        first_slice_flag = len(raw) > start_len + 2 and (raw[start_len + 2] & 0x80) != 0
        if nal_type <= 31 and first_slice_flag:
          frame = frames.pop(0)
          consumed.append(frame)
          decoder._frames.append(frame)

  monkeypatch.setattr(decoder, "feed_packet", fake_feed_packet)
  monkeypatch.setattr(decoder, "feed_end_of_stream", lambda: eos_calls.append(True))

  slice0 = _header(1) + b"\x80A"
  sps_next = _header(33) + b"SPSNEXT"
  slice1 = _header(1) + b"\x80B"
  source = [slice0, sps_next, slice1]
  gen = decoder.decode_annexb_stream(source, device="CUDA", dtype=dtypes.float32, normalize=False,
                                     include_timestamps=True, timestamps=[10, 20], long_start_code=True, color_space="bt2020")

  first = next(gen)
  assert first == (100, "tensor-0")
  assert consumed[0].kwargs == [{"device": "CUDA", "dtype": dtypes.float32, "normalize": False, "color_space": "bt2020"}]

  rest = list(gen)
  assert rest == [(200, "tensor-1")]
  assert all(frame.released for frame in consumed)
  assert consumed[1].kwargs == [{"device": "CUDA", "dtype": dtypes.float32, "normalize": False, "color_space": "bt2020"}]

  assert len(feed_calls) == 3
  assert feed_calls[0][1] == 10
  assert feed_calls[1][1] == 20
  assert feed_calls[2][1] == 20
  assert feed_calls[0][0].startswith(b"\x00\x00\x00\x01")
  assert eos_calls == [True]

  decoder.close()


def test_decode_annexb_stream_uint8_normalize_error():
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  with pytest.raises(ValueError):
    decoder.decode_annexb_stream([], dtype=dtypes.uint8, normalize=True)
  decoder.close()


def test_error_wrap(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  def failing(*_):
    return 719
  stub.cuvidCreateVideoParser = StubFunc(failing)
  lib = CuvidLibrary(stub)
  with pytest.raises(CuvidError):
    lib.create_video_parser(CUVIDPARSERPARAMS())


def test_decode_annexb_to_tensors_drains_frames(monkeypatch):
  stub = StubCuvidLib()
  stub.attach()
  lib = CuvidLibrary(stub)
  decoder = NVVideoDecoder(cuvid=lib)
  calls: list[dict[str, object]] = []

  streams = [
    [(100, "tensor-0"), (200, "tensor-1")],
    ["tensor-2"],
  ]

  def fake_stream(source, **kwargs):
    calls.append({"source": source, "kwargs": kwargs})
    data = streams.pop(0)
    for item in data:
      yield item

  monkeypatch.setattr(decoder, "decode_annexb_stream", fake_stream)

  outputs = decoder.decode_annexb_to_tensors(b"data", device="CUDA", dtype=dtypes.float32, normalize=False,
                                             include_timestamps=True, timestamps=[1, 2], chunk_size=128,
                                             long_start_code=True, end_of_stream=False)
  assert outputs == [(100, "tensor-0"), (200, "tensor-1")]
  assert calls[0]["source"] == b"data"
  assert calls[0]["kwargs"] == {
    "device": "CUDA",
    "dtype": dtypes.float32,
    "normalize": False,
    "include_timestamps": True,
    "chunk_size": 128,
    "timestamps": [1, 2],
    "long_start_code": True,
    "end_of_stream": False,
    "color_space": "bt709",
  }

  outputs_no_ts = decoder.decode_annexb_to_tensors(b"", device="CUDA", dtype=dtypes.float32, normalize=True,
                                                   include_timestamps=False, timestamps=None)
  assert outputs_no_ts == ["tensor-2"]
  assert calls[1]["source"] == b""
  assert calls[1]["kwargs"] == {
    "device": "CUDA",
    "dtype": dtypes.float32,
    "normalize": True,
    "include_timestamps": False,
    "chunk_size": 1 << 16,
    "timestamps": None,
    "long_start_code": False,
    "end_of_stream": True,
    "color_space": "bt709",
  }

  decoder.close()


def test_decode_annexb_iter_fallback(monkeypatch):
  class FailingDecoder:
    def __init__(self, **_):
      raise CuvidUnavailable("missing hardware")

  monkeypatch.setattr(cuvid, "NVVideoDecoder", FailingDecoder)
  captured: dict[str, object] = {}

  def fallback(err):
    captured["err"] = err
    return ["fallback-result"]

  result = cuvid.decode_annexb_iter(b"data", fallback=fallback)
  assert result == ["fallback-result"]
  assert isinstance(captured["err"], CuvidUnavailable)


def test_decode_annexb_iter_raises_without_fallback(monkeypatch):
  class FailingDecoder:
    def __init__(self, **_):
      raise CuvidUnavailable("no gpu")

  monkeypatch.setattr(cuvid, "NVVideoDecoder", FailingDecoder)

  with pytest.raises(CuvidUnavailable):
    list(cuvid.decode_annexb_iter(b"data"))
