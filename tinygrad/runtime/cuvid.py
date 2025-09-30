from __future__ import annotations
import ctypes, ctypes.util, threading, weakref
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Deque, Optional, Dict, Tuple, Callable, TYPE_CHECKING, cast
from tinygrad.helpers import init_c_var
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes, to_dtype
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.hevc_annexb import iter_annexb_nalus, with_annexb_start_code, parse_nal_header, AnnexBParserError

if TYPE_CHECKING:
  from tinygrad import Tensor

CUDA_SUCCESS = 0

cudaVideoCodec_HEVC = 8
cudaVideoChromaFormat_420 = 1

class CuvidUnavailable(RuntimeError):
  pass

class CuvidError(RuntimeError):
  pass

class CUVIDDECODECAPS(ctypes.Structure):
  _fields_ = [
    ("eCodecType", ctypes.c_uint),
    ("eChromaFormat", ctypes.c_uint),
    ("nBitDepthMinus8", ctypes.c_uint),
    ("reserved1", ctypes.c_uint * 3),
    ("nMinWidth", ctypes.c_uint),
    ("nMinHeight", ctypes.c_uint),
    ("nMaxWidth", ctypes.c_uint),
    ("nMaxHeight", ctypes.c_uint),
    ("nMaxMBCount", ctypes.c_uint),
    ("nMaxSlices", ctypes.c_uint),
    ("nMaxDecodeSurfaces", ctypes.c_uint),
    ("nMaxOutputSurfaces", ctypes.c_uint),
    ("bIsSupported", ctypes.c_ubyte),
    ("bIsPartialSupported", ctypes.c_ubyte),
    ("bReserved", ctypes.c_ubyte * 2),
    ("nReserved", ctypes.c_uint * 4),
  ]

@dataclass(frozen=True)
class DecoderCaps:
  codec_type: int
  chroma_format: int
  bit_depth_minus8: int
  min_width: int
  min_height: int
  max_width: int
  max_height: int
  max_mb_count: int
  max_slices: int
  max_decode_surfaces: int
  max_output_surfaces: int
  is_supported: bool
  is_partial_supported: bool

class CUVIDPARSERPARAMS(ctypes.Structure):
  _fields_ = [
    ("CodecType", ctypes.c_uint),
    ("ulMaxNumDecodeSurfaces", ctypes.c_uint),
    ("ulClockRate", ctypes.c_uint),
    ("ulErrorThreshold", ctypes.c_uint),
    ("ulMaxDisplayDelay", ctypes.c_uint),
    ("bAnnexB", ctypes.c_uint),
    ("pUserData", ctypes.c_void_p),
    ("pfnSequenceCallback", ctypes.c_void_p),
    ("pfnDecodePicture", ctypes.c_void_p),
    ("pfnDisplayPicture", ctypes.c_void_p),
    ("pvReserved1", ctypes.c_void_p),
    ("pvReserved2", ctypes.c_void_p),
    ("reserved", ctypes.c_uint * 4),
  ]

class CUVIDSOURCEDATAPACKET(ctypes.Structure):
  _fields_ = [
    ("flags", ctypes.c_uint),
    ("payload_size", ctypes.c_uint),
    ("payload", ctypes.c_void_p),
    ("timestamp", ctypes.c_ulonglong),
  ]

class CUVIDDECODECREATEINFO(ctypes.Structure):
  _fields_ = [
    ("ulWidth", ctypes.c_uint),
    ("ulHeight", ctypes.c_uint),
    ("ulNumDecodeSurfaces", ctypes.c_uint),
    ("CodecType", ctypes.c_uint),
    ("ChromaFormat", ctypes.c_uint),
    ("ulCreationFlags", ctypes.c_uint),
    ("bitDepthMinus8", ctypes.c_uint),
    ("ulIntraDecodeOnly", ctypes.c_uint),
    ("Reserved1", ctypes.c_uint * 2),
    ("ulNumOutputSurfaces", ctypes.c_uint),
    ("ulTargetWidth", ctypes.c_uint),
    ("ulTargetHeight", ctypes.c_uint),
    ("ulNumDecodeStereoSurfaces", ctypes.c_uint),
    ("Reserved2", ctypes.c_uint * 3),
    ("pDeinterlaceFunc", ctypes.c_void_p),
    ("vidLock", ctypes.c_void_p),
  ]

class CUVIDPICPARAMS(ctypes.Structure):
  _fields_ = [
    ("PicWidthInMbs", ctypes.c_uint),
    ("FrameHeightInMbs", ctypes.c_uint),
    ("CurrPicIdx", ctypes.c_int),
    ("field_pic_flag", ctypes.c_uint),
    ("bottom_field_flag", ctypes.c_uint),
    ("second_field", ctypes.c_uint),
    ("nBitstreamDataLen", ctypes.c_uint),
    ("pBitstreamData", ctypes.c_void_p),
    ("nNumSlices", ctypes.c_uint),
    ("pSliceDataOffsets", ctypes.POINTER(ctypes.c_uint)),
    ("ref_pic_flag", ctypes.c_uint),
    ("intra_pic_flag", ctypes.c_uint),
    ("Reserved", ctypes.c_uint * 15),
  ]

class CUVIDDISPLAYPARAMS(ctypes.Structure):
  _fields_ = [
    ("progressive_frame", ctypes.c_int),
    ("top_field_first", ctypes.c_int),
    ("repeat_first_field", ctypes.c_int),
    ("reserved", ctypes.c_int * 5),
    ("display_area", ctypes.c_int * 4),
    ("target_rect", ctypes.c_int * 4),
    ("pCurrFrame", ctypes.c_void_p),
  ]

class CUVIDEOFORMAT(ctypes.Structure):
  _fields_ = [
    ("codec", ctypes.c_uint),
    ("chroma_format", ctypes.c_uint),
    ("bit_depth_luma_minus8", ctypes.c_uint),
    ("bit_depth_chroma_minus8", ctypes.c_uint),
    ("nBitrate", ctypes.c_uint),
    ("frame_rate_numerator", ctypes.c_uint),
    ("frame_rate_denominator", ctypes.c_uint),
    ("progressive_sequence", ctypes.c_uint),
    ("coded_width", ctypes.c_uint),
    ("coded_height", ctypes.c_uint),
    ("display_area", ctypes.c_uint * 4),
    ("aspect_ratio", ctypes.c_uint * 2),
    ("video_signal_description", ctypes.c_uint * 7),
    ("reserved", ctypes.c_uint * 7),
  ]

class CUVIDPARSERDISPINFO(ctypes.Structure):
  _fields_ = [
    ("picture_index", ctypes.c_int),
    ("progressive_frame", ctypes.c_int),
    ("top_field_first", ctypes.c_int),
    ("repeat_first_field", ctypes.c_int),
    ("timestamp", ctypes.c_longlong),
    ("reserved", ctypes.c_uint * 4),
  ]

class CUVIDPROCPARAMS(ctypes.Structure):
  _fields_ = [
    ("progressive_frame", ctypes.c_uint),
    ("top_field_first", ctypes.c_uint),
    ("reserved1", ctypes.c_uint * 2),
    ("second_field_timestamp", ctypes.c_longlong),
    ("reserved2", ctypes.c_uint * 4),
  ]

CUVID_PKT_ENDOFSTREAM = 0x02

SEQUENCE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(CUVIDEOFORMAT))
DECODE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(CUVIDPICPARAMS))
DISPLAY_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(CUVIDPARSERDISPINFO))

class CuvidLibrary:
  def __init__(self, lib:ctypes.CDLL):
    self._lib = lib
    self._configure()

  def _configure(self):
    def _set_sig(name, argtypes, restype):
      fn = getattr(self._lib, name, None)
      if fn is None:
        return
      fn.argtypes = argtypes
      fn.restype = restype

    _set_sig("cuvidGetDecoderCaps", [ctypes.POINTER(CUVIDDECODECAPS)], ctypes.c_int)
    _set_sig("cuvidCreateVideoParser", [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUVIDPARSERPARAMS)], ctypes.c_int)
    _set_sig("cuvidDestroyVideoParser", [ctypes.c_void_p], ctypes.c_int)
    _set_sig("cuvidParseVideoData", [ctypes.c_void_p, ctypes.POINTER(CUVIDSOURCEDATAPACKET)], ctypes.c_int)
    _set_sig("cuvidCreateDecoder", [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUVIDDECODECREATEINFO)], ctypes.c_int)
    _set_sig("cuvidDestroyDecoder", [ctypes.c_void_p], ctypes.c_int)
    _set_sig("cuvidDecodePicture", [ctypes.c_void_p, ctypes.POINTER(CUVIDPICPARAMS)], ctypes.c_int)
    _set_sig(
      "cuvidMapVideoFrame",
      [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_void_p,
      ],
      ctypes.c_int,
    )
    _set_sig("cuvidUnmapVideoFrame", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int)

  def get_decoder_caps(self, codec: int = cudaVideoCodec_HEVC, chroma: int = cudaVideoChromaFormat_420,
                       bit_depth_minus8: int = 0) -> DecoderCaps:
    caps = CUVIDDECODECAPS()
    caps.eCodecType = codec
    caps.eChromaFormat = chroma
    caps.nBitDepthMinus8 = bit_depth_minus8
    check_cuvid(self._lib.cuvidGetDecoderCaps(ctypes.byref(caps)))
    return DecoderCaps(
      codec_type=caps.eCodecType,
      chroma_format=caps.eChromaFormat,
      bit_depth_minus8=caps.nBitDepthMinus8,
      min_width=caps.nMinWidth,
      min_height=caps.nMinHeight,
      max_width=caps.nMaxWidth,
      max_height=caps.nMaxHeight,
      max_mb_count=caps.nMaxMBCount,
      max_slices=caps.nMaxSlices,
      max_decode_surfaces=caps.nMaxDecodeSurfaces,
      max_output_surfaces=caps.nMaxOutputSurfaces,
      is_supported=bool(caps.bIsSupported),
      is_partial_supported=bool(caps.bIsPartialSupported),
    )

  def create_video_parser(self, params: CUVIDPARSERPARAMS) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    check_cuvid(self._lib.cuvidCreateVideoParser(ctypes.byref(handle), ctypes.byref(params)))
    return handle

  def destroy_video_parser(self, handle: ctypes.c_void_p):
    check_cuvid(self._lib.cuvidDestroyVideoParser(handle))

  def parse_video_data(self, handle: ctypes.c_void_p, packet: CUVIDSOURCEDATAPACKET):
    check_cuvid(self._lib.cuvidParseVideoData(handle, ctypes.byref(packet)))

  def create_decoder(self, info: CUVIDDECODECREATEINFO) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    check_cuvid(self._lib.cuvidCreateDecoder(ctypes.byref(handle), ctypes.byref(info)))
    return handle

  def destroy_decoder(self, handle: ctypes.c_void_p):
    check_cuvid(self._lib.cuvidDestroyDecoder(handle))

  def decode_picture(self, decoder: ctypes.c_void_p, params: CUVIDPICPARAMS):
    check_cuvid(self._lib.cuvidDecodePicture(decoder, ctypes.byref(params)))

  def map_video_frame(self, decoder: ctypes.c_void_p, pic_idx: int, proc_params: ctypes.c_void_p) -> tuple[int, int]:
    dev_ptr = ctypes.c_void_p()
    pitch = ctypes.c_uint()
    if proc_params is None:
      proc_ptr = None
    elif isinstance(proc_params, ctypes.c_void_p):
      proc_ptr = proc_params
    else:
      proc_ptr = ctypes.byref(proc_params)
    check_cuvid(self._lib.cuvidMapVideoFrame(decoder, pic_idx, ctypes.byref(dev_ptr), ctypes.byref(pitch), proc_ptr))
    return dev_ptr.value, pitch.value

  def unmap_video_frame(self, decoder: ctypes.c_void_p, dev_ptr: int):
    check_cuvid(self._lib.cuvidUnmapVideoFrame(decoder, ctypes.c_void_p(dev_ptr)))

  @property
  def raw(self) -> ctypes.CDLL:
    return self._lib

@dataclass(frozen=True)
class NV12Plane:
  pointer: int
  pitch: int
  width: int
  height: int
  channels: int

  @property
  def row_bytes(self) -> int:
    return self.pitch

  @property
  def size_bytes(self) -> int:
    return self.pitch * self.height


_COLOR_SPACE_COEFFS: Dict[str, Tuple[float, float, float, float]] = {
  "bt601": (1.4020, -0.344136, -0.714136, 1.7720),
  "bt709": (1.5748, -0.187324, -0.468124, 1.8556),
  "bt2020": (1.4746, -0.164553, -0.571353, 1.8814),
}


def _resolve_color_space(color_space: str | Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
  if isinstance(color_space, str):
    key = color_space.lower()
    coeffs = _COLOR_SPACE_COEFFS.get(key)
    if coeffs is None:
      available = ", ".join(sorted(_COLOR_SPACE_COEFFS))
      raise ValueError(f"Unsupported color_space '{color_space}'. Available: {available}")
    return coeffs
  if isinstance(color_space, tuple) and len(color_space) == 4:
    return tuple(float(x) for x in color_space)  # type: ignore[return-value]
  raise TypeError("color_space must be a string key or a tuple of four floats")


_NV12_TO_RGB_KERNEL_TEMPLATE = r"""
{includes}extern "C" __global__ void {kernel_name}(
  const unsigned char* __restrict__ y_plane,
  const unsigned char* __restrict__ uv_plane,
  {out_type}* __restrict__ out,
  int width, int height, int y_pitch, int uv_pitch, int out_stride, int normalize,
  float coef_rv, float coef_gu, float coef_gv, float coef_bu) {{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int y_index = y * y_pitch + x;
  const int uv_index = (y >> 1) * uv_pitch + (x & ~1);
  float Y = (float)y_plane[y_index];
  float U = (float)uv_plane[uv_index] - 128.0f;
  float V = (float)uv_plane[uv_index + 1] - 128.0f;
  float r = Y + coef_rv * V;
  float g = Y + coef_gu * U + coef_gv * V;
  float b = Y + coef_bu * U;
  r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
  g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
  b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);
  if (normalize) {{
    const float inv = 1.0f / 255.0f;
    r *= inv;
    g *= inv;
    b *= inv;
  }}
  const int out_index = y * out_stride + x * 3;
{store}
}}
"""

_NV12_KERNEL_CONFIGS: Dict[object, Dict[str, object]] = {
  dtypes.float32: {
    "suffix": "float32",
    "out_type": "float",
    "includes": "",
    "store": "\n".join([
      "  out[out_index + 0] = r;",
      "  out[out_index + 1] = g;",
      "  out[out_index + 2] = b;",
    ]),
  },
  dtypes.float16: {
    "suffix": "float16",
    "out_type": "__half",
    "includes": "#include <cuda_fp16.h>\n",
    "store": "\n".join([
      "  out[out_index + 0] = __float2half(r);",
      "  out[out_index + 1] = __float2half(g);",
      "  out[out_index + 2] = __float2half(b);",
    ]),
  },
  dtypes.uint8: {
    "suffix": "uint8",
    "out_type": "unsigned char",
    "includes": "",
    "store": "\n".join([
      "  out[out_index + 0] = (unsigned char)(r + 0.5f);",
      "  out[out_index + 1] = (unsigned char)(g + 0.5f);",
      "  out[out_index + 2] = (unsigned char)(b + 0.5f);",
    ]),
  },
}


def _as_cu_ptr(ptr: int) -> cuda.CUdeviceptr:
  if isinstance(ptr, (cuda.CUdeviceptr, cuda.CUdeviceptr_v2)):
    return ptr
  return cuda.CUdeviceptr_v2(int(ptr))


class _NV12ToRGBKernel:
  def __init__(self, device: str, dtype):
    self.device_name = Device.canonicalize(device)
    self.device = Device[self.device_name]
    if dtype not in _NV12_KERNEL_CONFIGS:
      raise NotImplementedError(f"Unsupported NV12 conversion dtype {dtype}")
    self.dtype = dtype
    self._program = None
    self._kernel_name = f"nv12_to_rgb_{_NV12_KERNEL_CONFIGS[self.dtype]['suffix']}"

  def _ensure_program(self):
    if self._program is not None:
      return
    cfg = _NV12_KERNEL_CONFIGS[self.dtype]
    src = _NV12_TO_RGB_KERNEL_TEMPLATE.format(
      includes=cfg["includes"],
      kernel_name=self._kernel_name,
      out_type=cfg["out_type"],
      store=cfg["store"],
    )
    lib = self.device.compiler.compile(src)
    self._program = self.device.runtime(self._kernel_name, lib)

  def launch(self, y_plane: NV12Plane, uv_plane: NV12Plane, out_ptr: int | cuda.CUdeviceptr,
       width: int, height: int, normalize: bool, coeffs: Tuple[float, float, float, float]):
    self._ensure_program()
    block_x, block_y = 16, 16
    grid_x = (width + block_x - 1) // block_x
    grid_y = (height + block_y - 1) // block_y
    args = (_as_cu_ptr(y_plane.pointer), _as_cu_ptr(uv_plane.pointer), _as_cu_ptr(out_ptr))
    def _pack(val: float) -> int:
      return ctypes.c_int.from_buffer_copy(ctypes.c_float(val)).value
    coeff_bits = tuple(_pack(c) for c in coeffs)
    vals = (width, height, y_plane.pitch, uv_plane.pitch, width * 3, 1 if normalize else 0, *coeff_bits)
    self._program(*args, global_size=(grid_x, grid_y, 1), local_size=(block_x, block_y, 1), vals=vals)


_nv12_kernel_cache: Dict[Tuple[str, object], _NV12ToRGBKernel] = {}


def _get_nv12_kernel(device: str, dtype) -> _NV12ToRGBKernel:
  dev = Device.canonicalize(device)
  key = (dev, dtype)
  if dtype not in _NV12_KERNEL_CONFIGS:
    raise NotImplementedError(f"NV12 conversion currently supports {', '.join(str(k) for k in _NV12_KERNEL_CONFIGS)} outputs")
  if key not in _nv12_kernel_cache:
    _nv12_kernel_cache[key] = _NV12ToRGBKernel(dev, dtype)
  return _nv12_kernel_cache[key]


class _TimestampManager:
  def __init__(self, timestamps: Optional[Iterable[int]]):
    self._iter = iter(timestamps) if timestamps is not None else None
    self._counter = 0
    self._peeked: Optional[int] = None
    self._frames_used = 0

  def _next_value(self) -> int:
    if self._iter is None:
      val = self._counter
    else:
      try:
        val = next(self._iter)
      except StopIteration:
        val = self._counter
    self._counter += 1
    return val

  def peek(self) -> int:
    if self._peeked is None:
      self._peeked = self._next_value()
    return self._peeked

  def consume(self) -> int:
    val = self.peek()
    self._peeked = None
    self._frames_used += 1
    return val

  @property
  def frames_used(self) -> int:
    return self._frames_used


def _nal_flags(nalu: bytes) -> Tuple[bool, bool]:
  if len(nalu) < 2:
    return False, False
  try:
    header = parse_nal_header(nalu)
  except AnnexBParserError:
    return False, False
  is_vcl = header.nal_unit_type <= 31
  is_first_slice = is_vcl and len(nalu) > 2 and (nalu[2] & 0x80) != 0
  return is_vcl, is_first_slice


def nv12_frames_to_tensors(frames: Iterable[DecodedFrame], *,
                           device: str = "CUDA", dtype=None, normalize: bool = True,
                           include_timestamps: bool = True,
                           color_space: str | Tuple[float, float, float, float] = "bt709") -> Iterator[Tuple[int, "Tensor"] | "Tensor"]:
  for frame in frames:
    try:
      tensor = frame.to_rgb_tensor(device=device, dtype=dtype, normalize=normalize, color_space=color_space)
      if include_timestamps:
        yield (frame.timestamp, tensor)
      else:
        yield tensor
    finally:
      frame.release()


@dataclass
class DecodedFrame:
  decoder: "NVVideoDecoder"
  picture_index: int
  device_pointer: Optional[int]
  pitch: int
  width: int
  height: int
  progressive_frame: int
  top_field_first: int
  repeat_first_field: int
  timestamp: int
  _proc_params: Optional[CUVIDPROCPARAMS] = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.release()

  @property
  def is_mapped(self) -> bool:
    return self.device_pointer is not None

  @property
  def luma_plane(self) -> NV12Plane:
    if self.device_pointer is None:
      raise RuntimeError("frame unmapped")
    return NV12Plane(self.device_pointer, self.pitch, self.width, self.height, 1)

  @property
  def chroma_plane(self) -> NV12Plane:
    if self.device_pointer is None:
      raise RuntimeError("frame unmapped")
    chroma_width = max(self.width // 2, 1)
    chroma_height = max(self.height // 2, 1)
    return NV12Plane(self.device_pointer + self.pitch * self.height, self.pitch, chroma_width, chroma_height, 2)

  @property
  def surface_size_bytes(self) -> int:
    return self.pitch * self.height + self.pitch * ((self.height + 1) // 2)

  def release(self):
    if self.device_pointer is None:
      return
    self.decoder._unmap(self.device_pointer)
    self.device_pointer = None
    self._proc_params = None

  def to_rgb_tensor(self, *, device: str = "CUDA", dtype=None, normalize: bool = True,
                    color_space: str | Tuple[float, float, float, float] = "bt709") -> "Tensor":
    from tinygrad.tensor import Tensor
    device_name = Device.canonicalize(device)
    if device_name != "CUDA":
      raise ValueError("NV12 conversion currently supports CUDA device only")
    if not self.is_mapped:
      raise RuntimeError("frame unmapped")
    dtype_obj = dtypes.float32 if dtype is None else to_dtype(dtype)
    if dtype_obj not in _NV12_KERNEL_CONFIGS:
      supported = ", ".join(sorted(str(k) for k in _NV12_KERNEL_CONFIGS))
      raise NotImplementedError(f"NV12 conversion currently supports output dtypes: {supported}")
    if dtype_obj == dtypes.uint8 and normalize:
      raise ValueError("NV12 conversion to uint8 does not support normalize=True")
    coeffs = _resolve_color_space(color_space)
    tensor = Tensor.empty(self.height, self.width, 3, device=device_name, dtype=dtype_obj)
    realized = tensor.realize()
    base_buffer = cast(Buffer, realized.uop.base.buffer).ensure_allocated()
    kernel = _get_nv12_kernel(device_name, dtype_obj)
    kernel.launch(self.luma_plane, self.chroma_plane, base_buffer._buf, self.width, self.height, normalize, coeffs)
    return tensor

class NVVideoDecoder:
  _registry: weakref.WeakValueDictionary[int, "NVVideoDecoder"] = weakref.WeakValueDictionary()

  def __init__(self, *, codec:int=cudaVideoCodec_HEVC, max_decode_surfaces:int=8, cuvid:Optional[CuvidLibrary]=None):
    self._lib = cuvid or load_cuvid()
    self._codec = codec
    self._max_decode_surfaces = max_decode_surfaces
    self._decoder_handle = ctypes.c_void_p()
    self._parser_handle = ctypes.c_void_p()
    self._format: Optional[CUVIDEOFORMAT] = None
    self._frames: Deque[DecodedFrame] = deque()
    self._lock = threading.RLock()
    self._callbacks: list = []
    self._user_token: int | None = id(self)
    NVVideoDecoder._registry[self._user_token] = self
    self._init_parser()

  def _release_pending_frames_locked(self):
    while self._frames:
      frame = self._frames.popleft()
      frame.release()

  def _init_parser(self):
    seq_cb = SEQUENCE_CALLBACK(self._sequence_trampoline)
    dec_cb = DECODE_CALLBACK(self._decode_trampoline)
    disp_cb = DISPLAY_CALLBACK(self._display_trampoline)
    self._callbacks = [seq_cb, dec_cb, disp_cb]
    params = CUVIDPARSERPARAMS()
    params.CodecType = self._codec
    params.ulMaxNumDecodeSurfaces = self._max_decode_surfaces
    params.ulClockRate = 0
    params.ulErrorThreshold = 0
    params.ulMaxDisplayDelay = 0
    params.bAnnexB = 1
    params.pUserData = ctypes.c_void_p(self._user_token)
    params.pfnSequenceCallback = ctypes.cast(seq_cb, ctypes.c_void_p)
    params.pfnDecodePicture = ctypes.cast(dec_cb, ctypes.c_void_p)
    params.pfnDisplayPicture = ctypes.cast(disp_cb, ctypes.c_void_p)
    self._parser_handle = self._lib.create_video_parser(params)

  @staticmethod
  def _from_user_data(user_data: ctypes.c_void_p) -> "NVVideoDecoder":
    if isinstance(user_data, int):
      token = user_data
    else:
      token = ctypes.cast(user_data, ctypes.c_void_p).value
    if token is None:
      raise CuvidError("Missing decoder context")
    decoder = NVVideoDecoder._registry.get(token)
    if decoder is None:
      raise CuvidError("Decoder context expired")
    return decoder

  @staticmethod
  def _sequence_trampoline(user_data: ctypes.c_void_p, format_ptr: ctypes.POINTER(CUVIDEOFORMAT)) -> int:
    decoder = NVVideoDecoder._from_user_data(user_data)
    return decoder._handle_sequence(ctypes.cast(format_ptr, ctypes.POINTER(CUVIDEOFORMAT)).contents)

  @staticmethod
  def _decode_trampoline(user_data: ctypes.c_void_p, pic_ptr: ctypes.POINTER(CUVIDPICPARAMS)) -> int:
    decoder = NVVideoDecoder._from_user_data(user_data)
    return decoder._handle_decode(ctypes.cast(pic_ptr, ctypes.POINTER(CUVIDPICPARAMS)).contents)

  @staticmethod
  def _display_trampoline(user_data: ctypes.c_void_p, disp_ptr: ctypes.POINTER(CUVIDPARSERDISPINFO)) -> int:
    decoder = NVVideoDecoder._from_user_data(user_data)
    return decoder._handle_display(ctypes.cast(disp_ptr, ctypes.POINTER(CUVIDPARSERDISPINFO)).contents)

  def _handle_sequence(self, fmt: CUVIDEOFORMAT) -> int:
    with self._lock:
      self._format = fmt
      if self._decoder_handle:
        self._release_pending_frames_locked()
        self._lib.destroy_decoder(self._decoder_handle)
        self._decoder_handle = ctypes.c_void_p()
      info = CUVIDDECODECREATEINFO()
      info.ulWidth = fmt.coded_width
      info.ulHeight = fmt.coded_height
      info.ulTargetWidth = fmt.coded_width
      info.ulTargetHeight = fmt.coded_height
      info.CodecType = fmt.codec
      info.ChromaFormat = fmt.chroma_format
      info.bitDepthMinus8 = fmt.bit_depth_luma_minus8
      info.ulNumDecodeSurfaces = max(self._max_decode_surfaces, 2)
      info.ulNumOutputSurfaces = 2
      self._decoder_handle = self._lib.create_decoder(info)
    return 1

  def _handle_decode(self, pic: CUVIDPICPARAMS) -> int:
    with self._lock:
      if not self._decoder_handle:
        return 0
      self._lib.decode_picture(self._decoder_handle, pic)
    return 1

  def _handle_display(self, disp: CUVIDPARSERDISPINFO) -> int:
    with self._lock:
      if not self._decoder_handle:
        return 0
      proc_params = CUVIDPROCPARAMS()
      proc_params.progressive_frame = disp.progressive_frame
      proc_params.top_field_first = disp.top_field_first
      dev_ptr, pitch = self._lib.map_video_frame(self._decoder_handle, disp.picture_index, proc_params)
      width = self._format.coded_width if self._format else 0
      height = self._format.coded_height if self._format else 0
      frame = DecodedFrame(self, disp.picture_index, dev_ptr, pitch, width, height,
                           disp.progressive_frame, disp.top_field_first, disp.repeat_first_field,
                           int(disp.timestamp), proc_params)
      self._frames.append(frame)
    return 1

  def feed_packet(self, payload: bytes | bytearray | memoryview, *, timestamp:int=0, flags:int=0):
    if not self._parser_handle:
      raise RuntimeError("Decoder not initialized")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
      raise TypeError("payload must be buffer-like")
    data = bytes(payload)
    packet = CUVIDSOURCEDATAPACKET()
    packet.flags = flags
    packet.payload_size = len(data)
    packet.timestamp = timestamp
    buf = None
    if packet.payload_size:
      buf = (ctypes.c_ubyte * packet.payload_size).from_buffer_copy(data)
      packet.payload = ctypes.cast(buf, ctypes.c_void_p)
    else:
      packet.payload = ctypes.c_void_p()
    packet._buffer = buf
    self._lib.parse_video_data(self._parser_handle, packet)

  def feed_end_of_stream(self):
    self.feed_packet(b"", flags=CUVID_PKT_ENDOFSTREAM)

  def feed_annexb_stream(self, source: bytes | bytearray | memoryview | Iterable[bytes] | object, *,
                         chunk_size: int = 1 << 16, timestamps: Optional[Iterable[int]] = None,
                         long_start_code: bool = False, end_of_stream: bool = True) -> int:
    timestamp_mgr = _TimestampManager(timestamps)
    current_timestamp: Optional[int] = None
    if isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray, memoryview)):
      nalu_iter = iter(source)
    else:
      nalu_iter = iter_annexb_nalus(source, chunk_size=chunk_size)
    for nalu in nalu_iter:
      is_vcl, first_slice = _nal_flags(nalu)
      if is_vcl:
        if first_slice or current_timestamp is None:
          current_timestamp = timestamp_mgr.consume()
      else:
        current_timestamp = None
      packet_ts = int(current_timestamp if current_timestamp is not None else timestamp_mgr.peek())
      packet_bytes = with_annexb_start_code(nalu, long_start_code=long_start_code)
      self.feed_packet(packet_bytes, timestamp=packet_ts)
    if end_of_stream:
      self.feed_end_of_stream()
    return timestamp_mgr.frames_used

  def acquire_frame(self) -> Optional[DecodedFrame]:
    with self._lock:
      if not self._frames:
        return None
      return self._frames.popleft()

  def pending_frames(self) -> int:
    with self._lock:
      return len(self._frames)

  def acquire_rgb_tensor(self, *, device: str = "CUDA", dtype=None, normalize: bool = True,
                         color_space: str | Tuple[float, float, float, float] = "bt709") -> Optional["Tensor"]:
    frame = self.acquire_frame()
    if frame is None:
      return None
    try:
      return frame.to_rgb_tensor(device=device, dtype=dtype, normalize=normalize, color_space=color_space)
    finally:
      frame.release()

  def drain_frames(self) -> Iterator[DecodedFrame]:
    while True:
      frame = self.acquire_frame()
      if frame is None:
        break
      yield frame

  def frames_to_tensors(self, *, device: str = "CUDA", dtype=None, normalize: bool = True,
                        include_timestamps: bool = True,
                        color_space: str | Tuple[float, float, float, float] = "bt709") -> Iterator[Tuple[int, "Tensor"] | "Tensor"]:
    return nv12_frames_to_tensors(self.drain_frames(), device=device, dtype=dtype,
                                  normalize=normalize, include_timestamps=include_timestamps,
                                  color_space=color_space)

  def decode_annexb_to_tensors(self, source: bytes | bytearray | memoryview | Iterable[bytes] | object, *,
                               device: str = "CUDA", dtype=None, normalize: bool = True,
                               include_timestamps: bool = True, chunk_size: int = 1 << 16,
                               timestamps: Optional[Iterable[int]] = None,
                               long_start_code: bool = False, end_of_stream: bool = True,
                               color_space: str | Tuple[float, float, float, float] = "bt709"):
    return list(self.decode_annexb_stream(source,
      device=device,
      dtype=dtype,
      normalize=normalize,
      include_timestamps=include_timestamps,
      chunk_size=chunk_size,
      timestamps=timestamps,
      long_start_code=long_start_code,
      end_of_stream=end_of_stream,
      color_space=color_space,
    ))

  def decode_annexb_stream(self, source: bytes | bytearray | memoryview | Iterable[bytes] | object, *,
                           device: str = "CUDA", dtype=None, normalize: bool = True,
                           include_timestamps: bool = True, chunk_size: int = 1 << 16,
                           timestamps: Optional[Iterable[int]] = None,
                           long_start_code: bool = False, end_of_stream: bool = True,
                           color_space: str | Tuple[float, float, float, float] = "bt709") -> Iterator[Tuple[int, "Tensor"] | "Tensor"]:
    dtype_obj = dtypes.float32 if dtype is None else to_dtype(dtype)
    if dtype_obj not in _NV12_KERNEL_CONFIGS:
      supported = ", ".join(sorted(str(k) for k in _NV12_KERNEL_CONFIGS))
      raise NotImplementedError(f"NV12 conversion currently supports output dtypes: {supported}")
    if dtype_obj == dtypes.uint8 and normalize:
      raise ValueError("NV12 conversion to uint8 does not support normalize=True")
    _resolve_color_space(color_space)
    timestamp_mgr = _TimestampManager(timestamps)
    current_timestamp: Optional[int] = None

    def _yield_available_frames() -> Iterator[DecodedFrame]:
      while True:
        frame = self.acquire_frame()
        if frame is None:
          break
        yield frame

    def _drain_to_tensors() -> Iterator[Tuple[int, "Tensor"] | "Tensor"]:
      return nv12_frames_to_tensors(_yield_available_frames(), device=device, dtype=dtype_obj,
                                    normalize=normalize, include_timestamps=include_timestamps,
                                    color_space=color_space)

    def _iterator() -> Iterator[Tuple[int, "Tensor"] | "Tensor"]:
      nonlocal current_timestamp
      if isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray, memoryview)):
        nalu_iter = iter(source)
      else:
        nalu_iter = iter_annexb_nalus(source, chunk_size=chunk_size)
      for nalu in nalu_iter:
        is_vcl, first_slice = _nal_flags(nalu)
        if is_vcl:
          if first_slice or current_timestamp is None:
            current_timestamp = timestamp_mgr.consume()
        else:
          current_timestamp = None
        packet_ts = int(current_timestamp if current_timestamp is not None else timestamp_mgr.peek())
        packet_bytes = with_annexb_start_code(nalu, long_start_code=long_start_code)
        self.feed_packet(packet_bytes, timestamp=packet_ts)
        yield from _drain_to_tensors()
      if end_of_stream:
        self.feed_end_of_stream()
        yield from _drain_to_tensors()

    return _iterator()

  def _unmap(self, device_pointer: int):
    if self._decoder_handle:
      self._lib.unmap_video_frame(self._decoder_handle, device_pointer)

  def close(self):
    with self._lock:
      self._release_pending_frames_locked()
      if self._decoder_handle:
        self._lib.destroy_decoder(self._decoder_handle)
        self._decoder_handle = ctypes.c_void_p()
      if self._parser_handle:
        self._lib.destroy_video_parser(self._parser_handle)
        self._parser_handle = ctypes.c_void_p()
      self._callbacks.clear()
      if self._user_token is not None:
        NVVideoDecoder._registry.pop(self._user_token, None)
        self._user_token = None

  def __del__(self):
    try:
      self.close()
    except Exception: # pragma: no cover - destructor safety
      pass

_lib_lock = threading.Lock()
_cached_lib: Optional[CuvidLibrary] = None
_cached_error: Optional[Exception] = None

def check_cuvid(status:int):
  if status == CUDA_SUCCESS:
    return
  try:
    msg_ptr = init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))
    msg = ctypes.string_at(msg_ptr).decode()
  except Exception: # pylint: disable=broad-except
    msg = f"Unknown CUVID error {status}"
  raise CuvidError(msg)

def _load_nvcuvid() -> ctypes.CDLL:
  search = ["nvcuvid", "libnvcuvid", "libnvcuvid.so.1", "nvcuvid.dll"]
  errors:list[str] = []
  for name in search:
    candidates = []
    libpath = ctypes.util.find_library(name)
    if libpath: candidates.append(libpath)
    if name not in candidates: candidates.append(name)
    for candidate in candidates:
      try:
        return ctypes.CDLL(candidate)
      except OSError as err: # pragma: no cover - platform specific messaging
        errors.append(f"{candidate}: {err}")
  raise CuvidUnavailable("Unable to locate NVDEC CUVID runtime: " + ", ".join(search) + (f"; errors: {errors[-1]}" if errors else ""))

def load_cuvid() -> CuvidLibrary:
  global _cached_lib, _cached_error
  with _lib_lock:
    if _cached_lib is not None:
      return _cached_lib
    if _cached_error is not None:
      raise CuvidUnavailable(str(_cached_error)) from _cached_error
    try:
      lib = _load_nvcuvid()
    except CuvidUnavailable as err:
      _cached_error = err
      raise
    _cached_lib = CuvidLibrary(lib)
    return _cached_lib

def is_available() -> bool:
  try:
    load_cuvid()
    return True
  except CuvidUnavailable:
    return False

def reset_cuvid_cache_for_tests():
  global _cached_lib, _cached_error
  with _lib_lock:
    _cached_lib = None
    _cached_error = None


def decode_annexb_iter(source: bytes | bytearray | memoryview | Iterable[bytes] | object, *,
                       device: str = "CUDA", dtype=None, normalize: bool = True,
                       include_timestamps: bool = True, chunk_size: int = 1 << 16,
                       timestamps: Optional[Iterable[int]] = None,
                       long_start_code: bool = False, end_of_stream: bool = True,
                       color_space: str | Tuple[float, float, float, float] = "bt709",
                       fallback: Optional[Callable[[CuvidUnavailable], object]] = None,
                       decoder_kwargs: Optional[Dict[str, object]] = None) -> object:
  decoder_kwargs = dict(decoder_kwargs or {})
  try:
    decoder = NVVideoDecoder(**decoder_kwargs)
  except CuvidUnavailable as err:
    if fallback is not None:
      return fallback(err)
    raise

  def _generator() -> Iterator[Tuple[int, "Tensor"] | "Tensor"]:
    try:
      yield from decoder.decode_annexb_stream(
        source,
        device=device,
        dtype=dtype,
        normalize=normalize,
        include_timestamps=include_timestamps,
        chunk_size=chunk_size,
        timestamps=timestamps,
        long_start_code=long_start_code,
        end_of_stream=end_of_stream,
        color_space=color_space,
      )
    finally:
      decoder.close()

  return _generator()


def decode_annexb_to_tensors_auto(source: bytes | bytearray | memoryview | Iterable[bytes] | object, *,
                                  device: str = "CUDA", dtype=None, normalize: bool = True,
                                  include_timestamps: bool = True, chunk_size: int = 1 << 16,
                                  timestamps: Optional[Iterable[int]] = None,
                                  long_start_code: bool = False, end_of_stream: bool = True,
                                  color_space: str | Tuple[float, float, float, float] = "bt709",
                                  fallback: Optional[Callable[[CuvidUnavailable], object]] = None,
                                  decoder_kwargs: Optional[Dict[str, object]] = None) -> object:
  result = decode_annexb_iter(
    source,
    device=device,
    dtype=dtype,
    normalize=normalize,
    include_timestamps=include_timestamps,
    chunk_size=chunk_size,
    timestamps=timestamps,
    long_start_code=long_start_code,
    end_of_stream=end_of_stream,
    color_space=color_space,
    fallback=fallback,
    decoder_kwargs=decoder_kwargs,
  )
  if isinstance(result, Iterator):
    return list(result)
  if isinstance(result, Iterable) and not isinstance(result, (bytes, bytearray, memoryview)):
    return list(result)
  return result

__all__ = [
  "CuvidUnavailable",
  "CuvidError",
  "CuvidLibrary",
  "DecoderCaps",
  "CUVIDDECODECAPS",
  "CUVIDPARSERPARAMS",
  "CUVIDSOURCEDATAPACKET",
  "CUVIDDECODECREATEINFO",
  "CUVIDPICPARAMS",
  "CUVIDDISPLAYPARAMS",
  "CUVIDEOFORMAT",
  "CUVIDPARSERDISPINFO",
  "CUVIDPROCPARAMS",
  "CUVID_PKT_ENDOFSTREAM",
  "SEQUENCE_CALLBACK",
  "DECODE_CALLBACK",
  "DISPLAY_CALLBACK",
  "DecodedFrame",
  "NV12Plane",
  "NVVideoDecoder",
  "decode_annexb_iter",
  "decode_annexb_to_tensors_auto",
  "nv12_frames_to_tensors",
  "load_cuvid",
  "is_available",
  "cudaVideoCodec_HEVC",
  "cudaVideoChromaFormat_420",
]
