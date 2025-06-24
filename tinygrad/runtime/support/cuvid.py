import ctypes, ctypes.util, os
from typing import Optional, Callable
from tinygrad.helpers import getenv

# CUVID library path detection
CUDA_PATH = getenv("CUDA_PATH", "")
CUVID_PATHS = [
  "nvcuvid",  # Standard library name
  "libnvcuvid.so.1",  # Linux with version
  "libnvcuvid.so",   # Linux generic
]

def get_cuvid_lib():
  """Load CUVID library with fallback paths"""
  # Try system library first
  lib_path = ctypes.util.find_library('nvcuvid')
  if lib_path: return ctypes.CDLL(lib_path)
  
  # Try CUDA_PATH if available
  if CUDA_PATH:
    for lib_name in ["libnvcuvid.so.1", "libnvcuvid.so"]:
      try: return ctypes.CDLL(os.path.join(CUDA_PATH, "lib64", lib_name))
      except OSError: continue
  
  # Try standard paths
  for lib_name in CUVID_PATHS:
    try: return ctypes.CDLL(lib_name)
    except OSError: continue
    
  raise RuntimeError("CUVID library not found. Install NVIDIA Video Codec SDK or set CUDA_PATH")

# Load library
try:
  cuvid = get_cuvid_lib()
except RuntimeError as e:
  print(f"Warning: {e}")
  cuvid = None

# CUVID Constants
CUVID_SUCCESS = 0

# Video codec types
class cudaVideoCodec(ctypes.c_int):
  MPEG1     = 0
  MPEG2     = 1
  MPEG4     = 2
  VC1       = 3
  H264      = 4
  JPEG      = 5
  H264_SVC  = 6
  H264_MVC  = 7
  HEVC      = 8  # H.265/HEVC
  VP8       = 9
  VP9       = 10
  AV1       = 11

# Video chroma formats
class cudaVideoChromaFormat(ctypes.c_int):
  Monochrome = 0
  YUV420     = 1  # Most common for HEVC
  YUV422     = 2
  YUV444     = 3

# Video surface formats
class cudaVideoSurfaceFormat(ctypes.c_int):
  NV12       = 0  # Most common
  P016       = 1  # 10-bit
  YUV444     = 2
  YUV444_16Bit = 3

# Decoder capability flags
class cudaVideoDecodeFlags(ctypes.c_int):
  Default    = 0
  TCP        = 1

# CUVIDDECODECAPS structure
class CUVIDDECODECAPS(ctypes.Structure):
  _fields_ = [
    ("eCodecType", cudaVideoCodec),           # Codec type
    ("eChromaFormat", cudaVideoChromaFormat), # Chroma format
    ("nBitDepthMinus8", ctypes.c_uint),       # Bit depth (0=8bit, 2=10bit)
    ("reserved1", ctypes.c_uint * 3),
    ("bIsSupported", ctypes.c_ubyte),         # 1=supported, 0=not supported
    ("nNumNVDECs", ctypes.c_ubyte),           # Number of NVDEC engines
    ("nMaxWidth", ctypes.c_ushort),           # Maximum width
    ("nMaxHeight", ctypes.c_ushort),          # Maximum height
    ("nMaxMBCount", ctypes.c_ushort),         # Maximum macroblock count
    ("nMinWidth", ctypes.c_ushort),           # Minimum width
    ("nMinHeight", ctypes.c_ushort),          # Minimum height
    ("reserved2", ctypes.c_ubyte * 12),
  ]

# CUVIDDECODECREATEINFO structure  
class CUVIDDECODECREATEINFO(ctypes.Structure):
  _fields_ = [
    ("ulWidth", ctypes.c_ulong),              # Coded sequence width
    ("ulHeight", ctypes.c_ulong),             # Coded sequence height
    ("ulNumDecodeSurfaces", ctypes.c_ulong),  # Number of decode surfaces
    ("CodecType", cudaVideoCodec),            # Codec type
    ("ChromaFormat", cudaVideoChromaFormat),  # Chroma format
    ("ulCreationFlags", ctypes.c_ulong),      # Creation flags
    ("bitDepthMinus8", ctypes.c_ulong),       # Bit depth minus 8
    ("ulIntraDecodeOnly", ctypes.c_ulong),    # Intra decode only flag
    ("ulMaxWidth", ctypes.c_ulong),           # Maximum width
    ("ulMaxHeight", ctypes.c_ulong),          # Maximum height
    ("Reserved1", ctypes.c_ulong),            # Reserved
    ("display_area", ctypes.c_int * 4),       # Display area (left,top,right,bottom)
    ("OutputFormat", cudaVideoSurfaceFormat), # Output surface format
    ("DeinterlaceMode", ctypes.c_int),        # Deinterlace mode
    ("ulTargetWidth", ctypes.c_ulong),        # Target width
    ("ulTargetHeight", ctypes.c_ulong),       # Target height
    ("ulNumOutputSurfaces", ctypes.c_ulong),  # Number of output surfaces
    ("vidLock", ctypes.c_void_p),             # Video lock
    ("target_rect", ctypes.c_int * 4),        # Target rectangle
    ("enableHistogram", ctypes.c_ulong),      # Enable histogram
    ("Reserved2", ctypes.c_ulong * 4),        # Reserved
  ]

# CUVIDPICPARAMS structure (simplified for HEVC)
class CUVIDPICPARAMS(ctypes.Structure):
  _fields_ = [
    ("PicWidthInMbs", ctypes.c_int),          # Picture width in macroblocks
    ("FrameHeightInMbs", ctypes.c_int),       # Frame height in macroblocks
    ("CurrPicIdx", ctypes.c_int),             # Current picture index
    ("field_pic_flag", ctypes.c_int),         # Field picture flag
    ("bottom_field_flag", ctypes.c_int),      # Bottom field flag
    ("second_field", ctypes.c_int),           # Second field flag
    ("nBitstreamDataLen", ctypes.c_uint),     # Bitstream data length
    ("pBitstreamData", ctypes.POINTER(ctypes.c_ubyte)), # Bitstream data
    ("nNumSlices", ctypes.c_uint),            # Number of slices
    ("pSliceDataOffsets", ctypes.POINTER(ctypes.c_uint)), # Slice data offsets
    ("ref_pic_flag", ctypes.c_int),           # Reference picture flag
    ("intra_pic_flag", ctypes.c_int),         # Intra picture flag
    # Simplified - real structure has codec-specific data
    ("CodecSpecific", ctypes.c_ubyte * 1024), # Codec-specific data
  ]

# Decoder handle
CUvideodecoder = ctypes.c_void_p
CUvideoparser = ctypes.c_void_p
CUcontext = ctypes.c_void_p
CUdeviceptr = ctypes.c_void_p

# Function signatures
if cuvid:
  # Query decoder capabilities
  cuvidGetDecoderCaps = cuvid.cuvidGetDecoderCaps
  cuvidGetDecoderCaps.argtypes = [ctypes.POINTER(CUVIDDECODECAPS)]
  cuvidGetDecoderCaps.restype = ctypes.c_int
  
  # Create/destroy decoder
  cuvidCreateDecoder = cuvid.cuvidCreateDecoder
  cuvidCreateDecoder.argtypes = [ctypes.POINTER(CUvideodecoder), ctypes.POINTER(CUVIDDECODECREATEINFO)]
  cuvidCreateDecoder.restype = ctypes.c_int
  
  cuvidDestroyDecoder = cuvid.cuvidDestroyDecoder
  cuvidDestroyDecoder.argtypes = [CUvideodecoder]
  cuvidDestroyDecoder.restype = ctypes.c_int
  
  # Decode picture
  cuvidDecodePicture = cuvid.cuvidDecodePicture
  cuvidDecodePicture.argtypes = [CUvideodecoder, ctypes.POINTER(CUVIDPICPARAMS)]
  cuvidDecodePicture.restype = ctypes.c_int
  
  # Map/unmap decoded picture to device memory
  cuvidMapVideoFrame = cuvid.cuvidMapVideoFrame
  cuvidMapVideoFrame.argtypes = [CUvideodecoder, ctypes.c_int, ctypes.POINTER(CUdeviceptr), 
                                ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(CUVIDPICPARAMS)]
  cuvidMapVideoFrame.restype = ctypes.c_int
  
  cuvidUnmapVideoFrame = cuvid.cuvidUnmapVideoFrame
  cuvidUnmapVideoFrame.argtypes = [CUvideodecoder, CUdeviceptr]
  cuvidUnmapVideoFrame.restype = ctypes.c_int

# Error handling helper
def cuvid_check(status: int, func_name: str = "cuvid"):
  """Check CUVID API call status and raise exception on error"""
  if status != CUVID_SUCCESS:
    raise RuntimeError(f"CUVID Error in {func_name}: {status}")

# Capability checking helper
def check_hevc_support(cuda_context: Optional[CUcontext] = None) -> CUVIDDECODECAPS:
  """Check if HEVC decoding is supported on current device"""
  if not cuvid:
    raise RuntimeError("CUVID library not available")
    
  caps = CUVIDDECODECAPS()
  caps.eCodecType = cudaVideoCodec.HEVC
  caps.eChromaFormat = cudaVideoChromaFormat.YUV420  # Most common
  caps.nBitDepthMinus8 = 0  # 8-bit
  
  cuvid_check(cuvidGetDecoderCaps(ctypes.byref(caps)), "cuvidGetDecoderCaps")
  
  if not caps.bIsSupported:
    raise RuntimeError("HEVC decoding not supported on this device")
    
  return caps

# Helper to create decoder for HEVC
def create_hevc_decoder(width: int, height: int, num_surfaces: int = 8) -> CUvideodecoder:
  """Create HEVC decoder with specified dimensions"""
  if not cuvid:
    raise RuntimeError("CUVID library not available")
    
  # Check capabilities first
  caps = check_hevc_support()
  
  # Validate dimensions
  if width < caps.nMinWidth or width > caps.nMaxWidth:
    raise ValueError(f"Width {width} not supported (range: {caps.nMinWidth}-{caps.nMaxWidth})")
  if height < caps.nMinHeight or height > caps.nMaxHeight:
    raise ValueError(f"Height {height} not supported (range: {caps.nMinHeight}-{caps.nMaxHeight})")
  
  # Create decoder info
  create_info = CUVIDDECODECREATEINFO()
  create_info.ulWidth = width
  create_info.ulHeight = height
  create_info.ulNumDecodeSurfaces = num_surfaces
  create_info.CodecType = cudaVideoCodec.HEVC
  create_info.ChromaFormat = cudaVideoChromaFormat.YUV420
  create_info.OutputFormat = cudaVideoSurfaceFormat.NV12
  create_info.ulCreationFlags = cudaVideoDecodeFlags.Default
  create_info.bitDepthMinus8 = 0  # 8-bit
  create_info.ulMaxWidth = caps.nMaxWidth
  create_info.ulMaxHeight = caps.nMaxHeight
  create_info.ulTargetWidth = width
  create_info.ulTargetHeight = height
  create_info.ulNumOutputSurfaces = 2
  
  # Create decoder
  decoder = CUvideodecoder()
  cuvid_check(cuvidCreateDecoder(ctypes.byref(decoder), ctypes.byref(create_info)), "cuvidCreateDecoder")
  
  return decoder 