from __future__ import annotations
import ctypes, ctypes.util
from typing import Iterable
from tinygrad.helpers import suppress_finalizing

__all__ = [
  "CuvidUnavailable",
  "CuvidError",
  "Surface",
  "decode_annexb",
  "is_available",
  "cudaVideoCodec_HEVC",
  "cudaVideoChromaFormat_420",
  "CUVID_PKT_ENDOFSTREAM",
]

CUDA_SUCCESS = 0
cudaVideoCodec_HEVC = 8
cudaVideoChromaFormat_420 = 1
CUVID_PKT_ENDOFSTREAM = 0x02

_UInt = ctypes.c_uint
_ULong = ctypes.c_ulonglong
_VoidPtr = ctypes.c_void_p


class CuvidUnavailable(RuntimeError):
  pass


class CuvidError(RuntimeError):
  pass


class CUVIDPARSERPARAMS(ctypes.Structure):
  _fields_ = [
    ("CodecType", _UInt),
    ("ulMaxNumDecodeSurfaces", _UInt),
    ("ulClockRate", _UInt),
    ("ulErrorThreshold", _UInt),
    ("ulMaxDisplayDelay", _UInt),
    ("bAnnexB", _UInt),
    ("pvReserved1", _VoidPtr),
    ("pvReserved2", _VoidPtr),
    ("pUserData", _VoidPtr),
    ("pfnSequenceCallback", _VoidPtr),
    ("pfnDecodePicture", _VoidPtr),
    ("pfnDisplayPicture", _VoidPtr),
    ("reserved", _UInt * 4),
  ]


class CUVIDSOURCEDATAPACKET(ctypes.Structure):
  _fields_ = [
    ("flags", _UInt),
    ("payload_size", _UInt),
    ("payload", _VoidPtr),
    ("timestamp", _ULong),
  ]


class CUVIDDECODECREATEINFO(ctypes.Structure):
  _fields_ = [
    ("ulWidth", _UInt),
    ("ulHeight", _UInt),
    ("ulNumDecodeSurfaces", _UInt),
    ("CodecType", _UInt),
    ("ChromaFormat", _UInt),
    ("ulCreationFlags", _UInt),
    ("bitDepthMinus8", _UInt),
    ("ulIntraDecodeOnly", _UInt),
    ("Reserved1", _UInt * 2),
    ("ulNumOutputSurfaces", _UInt),
    ("ulTargetWidth", _UInt),
    ("ulTargetHeight", _UInt),
    ("ulNumDecodeStereoSurfaces", _UInt),
    ("Reserved2", _UInt * 3),
    ("pDeinterlaceFunc", _VoidPtr),
    ("vidLock", _VoidPtr),
  ]


class CUVIDPICPARAMS(ctypes.Structure):
  _fields_ = [
    ("PicWidthInMbs", _UInt),
    ("FrameHeightInMbs", _UInt),
    ("CurrPicIdx", ctypes.c_int),
    ("field_pic_flag", _UInt),
    ("bottom_field_flag", _UInt),
    ("second_field", _UInt),
    ("nBitstreamDataLen", _UInt),
    ("pBitstreamData", _VoidPtr),
    ("nNumSlices", _UInt),
    ("pSliceDataOffsets", ctypes.POINTER(_UInt)),
    ("ref_pic_flag", _UInt),
    ("intra_pic_flag", _UInt),
    ("Reserved", _UInt * 15),
  ]


class CUVIDPARSERDISPINFO(ctypes.Structure):
  _fields_ = [
    ("picture_index", ctypes.c_int),
    ("progressive_frame", ctypes.c_int),
    ("top_field_first", ctypes.c_int),
    ("repeat_first_field", ctypes.c_int),
    ("timestamp", ctypes.c_longlong),
    ("reserved", _UInt * 4),
  ]


class CUVIDEOFORMAT(ctypes.Structure):
  _fields_ = [
    ("codec", _UInt),
    ("chroma_format", _UInt),
    ("bit_depth_luma_minus8", _UInt),
    ("bit_depth_chroma_minus8", _UInt),
    ("nBitrate", _UInt),
    ("frame_rate_numerator", _UInt),
    ("frame_rate_denominator", _UInt),
    ("progressive_sequence", _UInt),
    ("coded_width", _UInt),
    ("coded_height", _UInt),
    ("display_area", _UInt * 4),
    ("aspect_ratio", _UInt * 2),
    ("video_signal_description", _UInt * 7),
    ("reserved", _UInt * 7),
  ]


class CUVIDPROCPARAMS(ctypes.Structure):
  _fields_ = [
    ("progressive_frame", _UInt),
    ("top_field_first", _UInt),
    ("reserved1", _UInt * 2),
    ("second_field_timestamp", ctypes.c_longlong),
    ("reserved2", _UInt * 4),
  ]


SEQUENCE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _VoidPtr, ctypes.POINTER(CUVIDEOFORMAT))
DECODE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _VoidPtr, ctypes.POINTER(CUVIDPICPARAMS))
DISPLAY_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _VoidPtr, ctypes.POINTER(CUVIDPARSERDISPINFO))


class Surface:
  __slots__ = ("pointer", "pitch", "width", "height", "timestamp", "_release", "_released")

  def __init__(self, pointer: int, pitch: int, width: int, height: int, timestamp: int, release):
    self.pointer = pointer
    self.pitch = pitch
    self.width = width
    self.height = height
    self.timestamp = timestamp
    self._release = release
    self._released = False

  def release(self) -> None:
    if self._released: return
    self._release()
    self._released = True

  @suppress_finalizing
  def __del__(self):  # pragma: no cover
    try:
      self.release()
    except Exception:  # noqa: BLE001
      pass


class _CuvidLibrary:
  def __init__(self):
    name = ctypes.util.find_library("nvcuvid") or "libnvcuvid.so"
    try:
      self.lib = ctypes.CDLL(name)
    except OSError as exc:
      raise CuvidUnavailable(str(exc)) from exc
    self._configure()

  def _configure(self) -> None:
    lib = self.lib
    lib.cuvidCreateVideoParser.argtypes = [ctypes.POINTER(_VoidPtr), ctypes.POINTER(CUVIDPARSERPARAMS)]
    lib.cuvidCreateVideoParser.restype = ctypes.c_int
    lib.cuvidDestroyVideoParser.argtypes = [_VoidPtr]
    lib.cuvidDestroyVideoParser.restype = ctypes.c_int
    lib.cuvidParseVideoData.argtypes = [_VoidPtr, ctypes.POINTER(CUVIDSOURCEDATAPACKET)]
    lib.cuvidParseVideoData.restype = ctypes.c_int
    lib.cuvidCreateDecoder.argtypes = [ctypes.POINTER(_VoidPtr), ctypes.POINTER(CUVIDDECODECREATEINFO)]
    lib.cuvidCreateDecoder.restype = ctypes.c_int
    lib.cuvidDestroyDecoder.argtypes = [_VoidPtr]
    lib.cuvidDestroyDecoder.restype = ctypes.c_int
    lib.cuvidDecodePicture.argtypes = [_VoidPtr, ctypes.POINTER(CUVIDPICPARAMS)]
    lib.cuvidDecodePicture.restype = ctypes.c_int
    lib.cuvidMapVideoFrame.argtypes = [_VoidPtr, ctypes.c_int, ctypes.POINTER(_ULong), ctypes.POINTER(_UInt), ctypes.POINTER(CUVIDPROCPARAMS)]
    lib.cuvidMapVideoFrame.restype = ctypes.c_int
    lib.cuvidUnmapVideoFrame.argtypes = [_VoidPtr, _ULong]
    lib.cuvidUnmapVideoFrame.restype = ctypes.c_int


_LIB: _CuvidLibrary | None = None


def _load_lib() -> _CuvidLibrary:
  global _LIB
  if _LIB is None:
    _LIB = _CuvidLibrary()
  return _LIB


class _DecoderContext:
  __slots__ = ("lib", "decoder", "format", "maps", "destroy_on_idle", "queue")

  def __init__(self, lib):
    self.lib = lib
    self.decoder = _VoidPtr()
    self.format: CUVIDEOFORMAT | None = None
    self.maps = 0
    self.destroy_on_idle = False
    self.queue: list[Surface] = []

  def configure(self, fmt: CUVIDEOFORMAT, codec: int, max_surfaces: int) -> None:
    if self.decoder:
      self.lib.cuvidDestroyDecoder(self.decoder)
      self.decoder = _VoidPtr()
    self.format = fmt
    height = max(int(fmt.coded_height), 1)
    tiles_per_col = max(height // 16, 2)  # HEVC tiles are 16px high
    surfaces = max(2, min(max_surfaces, tiles_per_col))  # Cap decode surfaces

    info = CUVIDDECODECREATEINFO()
    info.CodecType = codec or fmt.codec
    info.ChromaFormat = fmt.chroma_format
    info.ulWidth = fmt.coded_width
    info.ulHeight = fmt.coded_height
    info.ulTargetWidth = fmt.coded_width
    info.ulTargetHeight = fmt.coded_height
    info.ulNumDecodeSurfaces = surfaces
    info.ulNumOutputSurfaces = 2
    info.bitDepthMinus8 = fmt.bit_depth_luma_minus8
    status = self.lib.cuvidCreateDecoder(ctypes.byref(self.decoder), ctypes.byref(info))
    if status != CUDA_SUCCESS: raise CuvidError(f"cuvidCreateDecoder: {status}")

  def decode(self, pic: CUVIDPICPARAMS) -> None:
    if not self.decoder: return
    status = self.lib.cuvidDecodePicture(self.decoder, ctypes.byref(pic))
    if status != CUDA_SUCCESS: raise CuvidError(f"cuvidDecodePicture: {status}")

  def map(self, disp: CUVIDPARSERDISPINFO) -> None:
    if self.format is None or not self.decoder: raise CuvidError("decoder not ready")
    devptr, pitch, proc = _ULong(), _UInt(), CUVIDPROCPARAMS()
    proc.progressive_frame = max(disp.progressive_frame, 0)
    proc.top_field_first = max(disp.top_field_first, 0)
    status = self.lib.cuvidMapVideoFrame(self.decoder, disp.picture_index, ctypes.byref(devptr), ctypes.byref(pitch), ctypes.byref(proc))
    if status != CUDA_SUCCESS: raise CuvidError(f"cuvidMapVideoFrame: {status}")
    self.maps += 1
    released = False

    def _release() -> None:
      nonlocal released
      if released: return
      status_unmap = self.lib.cuvidUnmapVideoFrame(self.decoder, _ULong(devptr.value))
      if status_unmap != CUDA_SUCCESS: raise CuvidError(f"cuvidUnmapVideoFrame: {status_unmap}")
      released = True
      self.maps -= 1
      if self.destroy_on_idle and self.maps == 0: self.destroy()

    surface = Surface(pointer=int(devptr.value), pitch=int(pitch.value), width=int(self.format.coded_width),
      height=int(self.format.coded_height), timestamp=int(disp.timestamp), release=_release)
    self.queue.append(surface)

  def drain(self) -> list[Surface]:
    out, self.queue = self.queue, []
    return out

  def finish(self) -> None:
    self.destroy_on_idle = True
    if self.maps == 0: self.destroy()

  def destroy(self) -> None:
    if self.decoder:
      self.lib.cuvidDestroyDecoder(self.decoder)
      self.decoder = _VoidPtr()


def _packet_from(payload: bytes | bytearray | memoryview, timestamp: int, flags: int) -> tuple[CUVIDSOURCEDATAPACKET, ctypes.Array[ctypes.c_char]]:
  data = bytes(payload) if not isinstance(payload, (bytes, bytearray)) else payload
  buf = ctypes.create_string_buffer(data)
  packet = CUVIDSOURCEDATAPACKET()
  packet.flags = flags
  packet.payload_size = len(data)
  packet.payload = ctypes.cast(ctypes.addressof(buf), _VoidPtr)
  packet.timestamp = _ULong(timestamp)
  return packet, buf


def decode_annexb(stream: bytes | bytearray | memoryview | Iterable[bytes], *, codec: int = cudaVideoCodec_HEVC,
                  max_decode_surfaces: int = 8, _library: _CuvidLibrary | None = None) -> list[Surface]:
  lib = (_library or _load_lib()).lib
  ctx = _DecoderContext(lib)

  def _sequence(user, fmt_ptr):
    ctx.configure(fmt_ptr.contents, codec, max_decode_surfaces)
    return 1

  def _decode(user, pic_ptr):
    ctx.decode(pic_ptr.contents)
    return 1

  def _display(user, disp_ptr):
    ctx.map(disp_ptr.contents)
    return 1

  seq_cb = SEQUENCE_CALLBACK(_sequence)
  dec_cb = DECODE_CALLBACK(_decode)
  disp_cb = DISPLAY_CALLBACK(_display)
  token = ctypes.py_object(ctx)

  params = CUVIDPARSERPARAMS()
  params.CodecType = codec
  params.ulMaxNumDecodeSurfaces = max(2, max_decode_surfaces)
  params.ulClockRate = 0
  params.ulErrorThreshold = 0
  params.ulMaxDisplayDelay = 0
  params.bAnnexB = 1
  params.pUserData = ctypes.cast(ctypes.pointer(token), _VoidPtr)
  params.pfnSequenceCallback = ctypes.cast(seq_cb, _VoidPtr)
  params.pfnDecodePicture = ctypes.cast(dec_cb, _VoidPtr)
  params.pfnDisplayPicture = ctypes.cast(disp_cb, _VoidPtr)

  parser = _VoidPtr()
  status = lib.cuvidCreateVideoParser(ctypes.byref(parser), ctypes.byref(params))
  if status != CUDA_SUCCESS: raise CuvidError(f"cuvidCreateVideoParser: {status}")

  keepalive = [seq_cb, dec_cb, disp_cb, token]
  surfaces: list[Surface] = []
  try:
    if isinstance(stream, (bytes, bytearray, memoryview)):
      iterable: Iterable[bytes] = [bytes(stream)]
    else:
      iterable = stream

    for idx, chunk in enumerate(iterable):
      packet, buf = _packet_from(chunk, idx, 0)
      keepalive.append(buf)  # type: ignore[arg-type]
      status = lib.cuvidParseVideoData(parser, ctypes.byref(packet))
      if status != CUDA_SUCCESS: raise CuvidError(f"cuvidParseVideoData: {status}")
      surfaces.extend(ctx.drain())

    eos = CUVIDSOURCEDATAPACKET()
    eos.flags = CUVID_PKT_ENDOFSTREAM
    status = lib.cuvidParseVideoData(parser, ctypes.byref(eos))
    if status != CUDA_SUCCESS: raise CuvidError(f"cuvidParseVideoData: {status}")
    surfaces.extend(ctx.drain())
  except Exception:
    for s in surfaces + ctx.drain(): s.release()
    ctx.destroy()
    raise
  finally:
    lib.cuvidDestroyVideoParser(parser)
    ctx.finish()

  return surfaces


def is_available() -> bool:
  try:
    _load_lib()
  except CuvidUnavailable:
    return False
  return True
