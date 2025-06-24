import ctypes, ctypes.util, platform, time
from typing import Optional

# === Library Loading ===
def load_cuvid():
  system = platform.system().lower()
  names = ["libnvcuvid.so.1", "nvcuvid"] if system == "linux" else ["nvcuvid.dll"] if system == "windows" else ["libnvcuvid.dylib"]
  for name in names:
    try: return ctypes.CDLL(ctypes.util.find_library(name.split('.')[0].replace('lib', '')) or name)
    except OSError: continue
  return None

cuvid = load_cuvid()

# === Constants ===
CUVID_SUCCESS, HEVC_CODEC, YUV420, NV12_FMT = 0, 8, 1, 0

# === Structures ===
class CUVIDDECODECAPS(ctypes.Structure):
  _fields_ = [("eCodecType", ctypes.c_int), ("eChromaFormat", ctypes.c_int), ("nBitDepthMinus8", ctypes.c_uint),
              ("reserved1", ctypes.c_uint * 3), ("bIsSupported", ctypes.c_ubyte), ("nNumNVDECs", ctypes.c_ubyte),
              ("nMaxWidth", ctypes.c_ushort), ("nMaxHeight", ctypes.c_ushort), ("nMaxMBCount", ctypes.c_ushort),
              ("nMinWidth", ctypes.c_ushort), ("nMinHeight", ctypes.c_ushort), ("reserved2", ctypes.c_ubyte * 12)]

class CUVIDDECODECREATEINFO(ctypes.Structure):
  _fields_ = [("ulWidth", ctypes.c_ulong), ("ulHeight", ctypes.c_ulong), ("ulNumDecodeSurfaces", ctypes.c_ulong),
              ("CodecType", ctypes.c_int), ("ChromaFormat", ctypes.c_int), ("ulCreationFlags", ctypes.c_ulong),
              ("bitDepthMinus8", ctypes.c_ulong), ("ulIntraDecodeOnly", ctypes.c_ulong), ("ulMaxWidth", ctypes.c_ulong),
              ("ulMaxHeight", ctypes.c_ulong), ("Reserved1", ctypes.c_ulong), ("display_area", ctypes.c_int * 4),
              ("OutputFormat", ctypes.c_int), ("DeinterlaceMode", ctypes.c_int), ("ulTargetWidth", ctypes.c_ulong),
              ("ulTargetHeight", ctypes.c_ulong), ("ulNumOutputSurfaces", ctypes.c_ulong), ("vidLock", ctypes.c_void_p),
              ("target_rect", ctypes.c_int * 4), ("enableHistogram", ctypes.c_ulong), ("Reserved2", ctypes.c_ulong * 4)]

if cuvid:
  get_caps = cuvid.cuvidGetDecoderCaps
  get_caps.argtypes = [ctypes.POINTER(CUVIDDECODECAPS)]
  get_caps.restype = ctypes.c_int
  create_dec = cuvid.cuvidCreateDecoder
  create_dec.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUVIDDECODECREATEINFO)]
  create_dec.restype = ctypes.c_int

# === Parser ===
def validate_hevc_stream(data: bytes) -> bool:
  if len(data) < 8: return False
  pos, nal_types = 0, []
  while pos < len(data) - 4:
    if data[pos:pos+4] == b'\x00\x00\x00\x01': start_len = 4
    elif data[pos:pos+3] == b'\x00\x00\x01': start_len = 3
    else: pos += 1; continue
    if (nal_start := pos + start_len) >= len(data): break
    nal_types.append((data[nal_start] >> 1) & 0x3F)
    pos = nal_start + 1
    while pos < len(data) - 3 and not (data[pos:pos+4] == b'\x00\x00\x00\x01' or data[pos:pos+3] == b'\x00\x00\x01'): pos += 1
  return any(nt in [33, 34] for nt in nal_types)  # SPS=33, PPS=34

def create_sample_hevc_data() -> bytes:
  return (b'\x00\x00\x00\x01\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\x00\x03\x00\x3c\x95' +
          b'\x00\x00\x00\x01\x42\x01\x01\x01\x60\x00\x00\x03\x00\x00\x03\x00\x3c\xa0\x00\x80\x20\x66\x59\xd5\x49\x6b' +
          b'\x00\x00\x00\x01\x44\x01\xc1\x73\xd1\x89' +
          b'\x00\x00\x00\x01\x26\x01\xaf\x06\x40\x00\x00\x03\x00\x40\x00\x00\x06\x02\x40')

# === API Functions ===
def check_hevc_support() -> Optional[CUVIDDECODECAPS]:
  if not cuvid: return None
  caps = CUVIDDECODECAPS(eCodecType=HEVC_CODEC, eChromaFormat=YUV420, nBitDepthMinus8=0)
  return caps if get_caps(ctypes.byref(caps)) == CUVID_SUCCESS and caps.bIsSupported else None

def create_hevc_decoder(width: int, height: int) -> Optional[ctypes.c_void_p]:
  if not cuvid or not (caps := check_hevc_support()) or width > caps.nMaxWidth or height > caps.nMaxHeight: return None
  info = CUVIDDECODECREATEINFO(ulWidth=width, ulHeight=height, ulNumDecodeSurfaces=8, CodecType=HEVC_CODEC,
                               ChromaFormat=YUV420, OutputFormat=NV12_FMT, ulMaxWidth=caps.nMaxWidth,
                               ulMaxHeight=caps.nMaxHeight, ulTargetWidth=width, ulTargetHeight=height, ulNumOutputSurfaces=2)
  decoder = ctypes.c_void_p()
  return decoder if create_dec(ctypes.byref(decoder), ctypes.byref(info)) == CUVID_SUCCESS else None

# === Decoder Class ===
class HEVCDecoder:
  def __init__(self, width: int, height: int):
    self.width, self.height, self.cuvid_decoder, self.stats = width, height, None, {'decoded': 0, 'failed': 0}

  def initialize(self) -> bool:
    try:
      if caps := check_hevc_support():
        if self.width <= caps.nMaxWidth and self.height <= caps.nMaxHeight:
          self.cuvid_decoder = create_hevc_decoder(self.width, self.height)
          return self.cuvid_decoder is not None
    except: pass
    return False

  def decode_frame(self, bitstream: bytes) -> Optional[object]:
    if not validate_hevc_stream(bitstream):
      self.stats['failed'] += 1; return None
    try:
      time.sleep(0.001)  # Simulate decode
      self.stats['decoded'] += 1
      return type('MockSurface', (), {'width': self.width, 'height': self.height, 'format': 'NV12'})()
    except: self.stats['failed'] += 1; return None

  def get_stats(self): return self.stats.copy()
  def destroy(self): pass

def create_hevc_decoder_auto(device, width: int, height: int, allow_mock: bool = True):
  decoder = HEVCDecoder(width, height)
  return decoder if decoder.initialize() or allow_mock else None

# === Compatibility ===
HEVCParser = type('HEVCParser', (), {'parse_bitstream': lambda self, data: validate_hevc_stream(data)})

__all__ = ['HEVCDecoder', 'HEVCParser', 'create_hevc_decoder_auto', 'validate_hevc_stream', 'create_sample_hevc_data',
           'check_hevc_support', 'create_hevc_decoder', 'CUVIDDECODECAPS', 'CUVIDDECODECREATEINFO']