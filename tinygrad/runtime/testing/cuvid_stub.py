from __future__ import annotations
import ctypes
from typing import Iterable, List, Sequence, Tuple

from tinygrad.runtime.cuvid import (
  CUVIDDECODECREATEINFO,
  CUVIDPARSERDISPINFO,
  CUVIDPARSERPARAMS,
  CUVIDPICPARAMS,
  CUVIDSOURCEDATAPACKET,
  CUVIDEOFORMAT,
  DECODE_CALLBACK,
  DISPLAY_CALLBACK,
  SEQUENCE_CALLBACK,
)

__all__ = [
  "StubFunc",
  "StubCuvidLib",
  "build_demo_annexb_stream",
  "prepare_demo_callbacks",
]


class StubFunc:
  """Lightweight callable used to wrap stubbed cuvid functions."""

  def __init__(self, value):
    self.value = value
    self.argtypes = None
    self.restype = ctypes.c_int

  def __call__(self, *args):
    return self.value(*args)


class StubCuvidLib:
  """In-memory simulation of the NVDEC CUVID interface for testing/demo."""

  def __init__(self):
    self.created_parsers: List[Tuple[ctypes.c_void_p, CUVIDPARSERPARAMS]] = []
    self.destroyed_parsers: List[ctypes.c_void_p] = []
    self.parsed_packets: List[CUVIDSOURCEDATAPACKET] = []
    self.packet_payloads: List[bytes] = []
    self.created_decoders: List[Tuple[ctypes.c_void_p, CUVIDDECODECREATEINFO]] = []
    self.destroyed_decoders: List[ctypes.c_void_p] = []
    self.decode_calls: List[CUVIDPICPARAMS] = []
    self.map_calls: List[Tuple[ctypes.c_void_p, int, ctypes.c_void_p]] = []
    self.unmap_calls: List[Tuple[ctypes.c_void_p, ctypes.c_void_p]] = []
    self.next_handle = 1
    self.parser_params: CUVIDPARSERPARAMS | None = None
    self.callback_events: List[Tuple[str, object]] = []
    self.decoder_infos: dict[int, CUVIDDECODECREATEINFO] = {}
    self.mapped_surfaces: dict[int, ctypes.Array] = {}

  # Attach helpers -----------------------------------------------------------------

  def attach(self):
    self.cuvidGetDecoderCaps = StubFunc(lambda caps_ptr: 0)
    self.cuvidCreateVideoParser = StubFunc(self._capture_parser)
    self.cuvidDestroyVideoParser = StubFunc(self._destroy_parser)
    self.cuvidParseVideoData = StubFunc(self._parse_video_data)
    self.cuvidCreateDecoder = StubFunc(self._create_decoder)
    self.cuvidDestroyDecoder = StubFunc(self._destroy_decoder)
    self.cuvidDecodePicture = StubFunc(self._decode_picture)
    self.cuvidMapVideoFrame = StubFunc(self._map_video_frame)
    self.cuvidUnmapVideoFrame = StubFunc(self._unmap_video_frame)

  # Handle stores ------------------------------------------------------------------

  def _allocate_handle(self):
    handle = ctypes.c_void_p(self.next_handle)
    self.next_handle += 1
    return handle

  def _capture_parser(self, handle_ptr, params_ptr):
    handle = self._allocate_handle()
    ctypes.cast(handle_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = handle
    ptr = ctypes.cast(params_ptr, ctypes.POINTER(CUVIDPARSERPARAMS))
    buf = ctypes.string_at(ptr, ctypes.sizeof(CUVIDPARSERPARAMS))
    params = CUVIDPARSERPARAMS.from_buffer_copy(buf)
    self.created_parsers.append((handle, params))
    self.parser_params = params
    return 0

  def _destroy_parser(self, handle):
    self.destroyed_parsers.append(handle)
    return 0

  def _parse_video_data(self, handle, packet_ptr):
    ptr = ctypes.cast(packet_ptr, ctypes.POINTER(CUVIDSOURCEDATAPACKET))
    raw = ptr.contents
    payload_bytes = b""
    if raw.payload and raw.payload_size:
      payload_bytes = ctypes.string_at(raw.payload, int(raw.payload_size))
    buf = ctypes.string_at(ptr, ctypes.sizeof(CUVIDSOURCEDATAPACKET))
    packet = CUVIDSOURCEDATAPACKET.from_buffer_copy(buf)
    self.parsed_packets.append(packet)
    self.packet_payloads.append(payload_bytes)
    if self.parser_params is not None:
      self.run_callbacks()
    return 0

  def _create_decoder(self, handle_ptr, info_ptr):
    handle = self._allocate_handle()
    ctypes.cast(handle_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = handle
    ptr = ctypes.cast(info_ptr, ctypes.POINTER(CUVIDDECODECREATEINFO))
    buf = ctypes.string_at(ptr, ctypes.sizeof(CUVIDDECODECREATEINFO))
    info = CUVIDDECODECREATEINFO.from_buffer_copy(buf)
    self.created_decoders.append((handle, info))
    self.decoder_infos[handle.value] = info
    return 0

  def _destroy_decoder(self, handle):
    self.destroyed_decoders.append(handle)
    self.decoder_infos.pop(handle.value, None)
    return 0

  def _decode_picture(self, decoder, params_ptr):
    if isinstance(params_ptr, CUVIDPICPARAMS):
      params = params_ptr
    else:
      params = ctypes.cast(params_ptr, ctypes.POINTER(CUVIDPICPARAMS)).contents
    buf = ctypes.string_at(ctypes.byref(params), ctypes.sizeof(CUVIDPICPARAMS))
    self.decode_calls.append(CUVIDPICPARAMS.from_buffer_copy(buf))
    return 0

  def _map_video_frame(self, decoder, pic_idx, dev_ptr, pitch_ptr, proc_params):
    info = self.decoder_infos.get(decoder.value)
    width = info.ulWidth if info is not None else 0
    height = info.ulHeight if info is not None else 0
    if width == 0 or height == 0:
      pitch = 2048
      height = max(height, 16)
    else:
      pitch = max(((width + 127) // 128) * 128, width)
    surface_bytes = pitch * height + pitch * max(height // 2, 1)
    buf = (ctypes.c_ubyte * max(surface_bytes, 1))()
    ptr_val = ctypes.addressof(buf)
    self.mapped_surfaces[ptr_val] = buf
    self.map_calls.append((decoder, pic_idx, proc_params))
    ctypes.cast(dev_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(ptr_val)
    ctypes.cast(pitch_ptr, ctypes.POINTER(ctypes.c_uint))[0] = ctypes.c_uint(pitch)
    return 0

  def _unmap_video_frame(self, decoder, dev_ptr):
    self.unmap_calls.append((decoder, dev_ptr))
    self.mapped_surfaces.pop(dev_ptr.value, None)
    return 0

  # Callback driver -----------------------------------------------------------------

  def enqueue_sequence(self, fmt: CUVIDEOFORMAT):
    self.callback_events.append(("sequence", fmt))

  def enqueue_decode(self, pic: CUVIDPICPARAMS):
    self.callback_events.append(("decode", pic))

  def enqueue_display(self, disp: CUVIDPARSERDISPINFO):
    self.callback_events.append(("display", disp))

  def run_callbacks(self):
    if self.parser_params is None:
      return
    params = self.parser_params
    user_data = params.pUserData
    while self.callback_events:
      event, payload = self.callback_events.pop(0)
      if event == "sequence" and params.pfnSequenceCallback:
        cb = ctypes.cast(params.pfnSequenceCallback, SEQUENCE_CALLBACK)
        cb(user_data, ctypes.pointer(payload))
      elif event == "decode" and params.pfnDecodePicture:
        cb = ctypes.cast(params.pfnDecodePicture, DECODE_CALLBACK)
        cb(user_data, ctypes.pointer(payload))
      elif event == "display" and params.pfnDisplayPicture:
        cb = ctypes.cast(params.pfnDisplayPicture, DISPLAY_CALLBACK)
        cb(user_data, ctypes.pointer(payload))


# Demo helpers -----------------------------------------------------------------------

def _demo_header(nal_type: int) -> bytes:
  nal_type &= 0x3F
  nuh_layer_id = 0
  nuh_temporal_id_plus1 = 1
  first_byte = (nal_type << 1) | (nuh_layer_id >> 5)
  second_byte = ((nuh_layer_id & 0x1F) << 3) | (nuh_temporal_id_plus1 & 0x7)
  return bytes([first_byte, second_byte])


def build_demo_annexb_stream(frames: int = 2) -> Tuple[bytes, Sequence[int]]:
  """Construct a tiny Annex B stream for demonstration purposes.

  The stream alternates between SPS and VCL slices to simulate two frames. The
  returned timestamps align with the number of frames.
  """
  nalus: List[bytes] = []
  timestamps: List[int] = []
  for idx in range(frames):
    nalus.append(_demo_header(33) + f"SPS{idx}".encode())
    nalus.append(_demo_header(1) + b"\x80" + f"F{idx}".encode())
    nalus.append(_demo_header(1) + b"\x00" + f"F{idx}_END".encode())
    timestamps.append(idx * 10)
  data = b"".join(b"\x00\x00\x00\x01" + n for n in nalus)
  return data, timestamps


def prepare_demo_callbacks(stub: StubCuvidLib, *, width: int = 640, height: int = 360, frames: int = 2,
                           timestamps: Iterable[int] | None = None):
  """Populate the stub with a simple decode/display sequence for the demo."""
  fmt = CUVIDEOFORMAT()
  fmt.codec = 8  # cudaVideoCodec_HEVC
  fmt.chroma_format = 1
  fmt.bit_depth_luma_minus8 = 0
  fmt.coded_width = width
  fmt.coded_height = height
  stub.enqueue_sequence(fmt)

  ts_iter = iter(timestamps or range(frames))
  for idx in range(frames):
    pic = CUVIDPICPARAMS()
    pic.CurrPicIdx = idx
    stub.enqueue_decode(pic)

    disp = CUVIDPARSERDISPINFO()
    disp.picture_index = idx
    disp.progressive_frame = 1
    disp.top_field_first = 0
    disp.repeat_first_field = 0
    try:
      disp.timestamp = next(ts_iter)
    except StopIteration:
      disp.timestamp = idx
    stub.enqueue_display(disp)