import ctypes, ctypes.util, os, sys, platform
from typing import Optional, Callable
from tinygrad.helpers import getenv

def get_cuvid_lib():
  """Load CUVID library with comprehensive platform detection"""
  # Environment variables
  cuda_path = getenv("CUDA_PATH", "")
  nvidia_sdk_path = getenv("NVIDIA_VIDEO_CODEC_SDK_PATH", "")
  
  # Platform-specific library names
  system = platform.system().lower()
  lib_names = []
  search_paths = []
  
  if system == "linux":
    lib_names = ["libnvcuvid.so.1", "libnvcuvid.so", "nvcuvid"]
    # Standard Linux paths
    search_paths.extend([
      "/usr/lib/x86_64-linux-gnu",
      "/usr/lib64",
      "/usr/local/lib",
      "/usr/local/cuda/lib64",
      "/opt/cuda/lib64",
    ])
  elif system == "windows":
    lib_names = ["nvcuvid.dll", "nvcuvid64.dll"]
    # Windows paths
    search_paths.extend([
      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin",
      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin", 
      "C:/Program Files/NVIDIA Corporation/NVIDIA Video Codec SDK",
    ])
  elif system == "darwin":  # macOS
    lib_names = ["libnvcuvid.dylib", "nvcuvid"]
    # macOS paths (if NVIDIA Web Drivers installed)
    search_paths.extend([
      "/usr/local/cuda/lib",
      "/Developer/NVIDIA/CUDA-12.0/lib",
      "/System/Library/Extensions/NVDANV50HalTesla.kext/Contents/MacOS",
    ])
  
  # Try system library detection first
  for lib_name in lib_names:
    lib_path = ctypes.util.find_library(lib_name.replace('lib', '').replace('.so', '').replace('.dll', '').replace('.dylib', ''))
    if lib_path:
      try:
        return ctypes.CDLL(lib_path)
      except OSError:
        continue
  
  # Add environment-specific paths
  if cuda_path:
    if system == "windows":
      search_paths.insert(0, os.path.join(cuda_path, "bin"))
    else:
      search_paths.insert(0, os.path.join(cuda_path, "lib64"))
      search_paths.insert(0, os.path.join(cuda_path, "lib"))
  
  # Add NVIDIA Video Codec SDK paths
  if nvidia_sdk_path:
    if system == "windows":
      search_paths.insert(0, os.path.join(nvidia_sdk_path, "Lib", "x64"))
      search_paths.insert(0, os.path.join(nvidia_sdk_path, "Lib", "Win32"))
    else:
      search_paths.insert(0, os.path.join(nvidia_sdk_path, "lib"))
      search_paths.insert(0, os.path.join(nvidia_sdk_path, "Lib", "linux", "stubs", "x86_64"))
  
  # Add standard Video Codec SDK installation paths
  if system == "linux":
    sdk_standard_paths = [
      "/opt/nvidia-video-codec-sdk/lib",
      "/opt/nvidia-video-codec-sdk/Lib/linux/stubs/x86_64",
      "/usr/local/nvidia-video-codec-sdk/lib",
      "/home/$USER/nvidia-video-codec-sdk/Lib/linux/stubs/x86_64"
    ]
    search_paths.extend(sdk_standard_paths)
  elif system == "windows":
    sdk_standard_paths = [
      "C:/Program Files/NVIDIA Corporation/NVIDIA Video Codec SDK/Lib/x64",
      "C:/NVIDIA Video Codec SDK/Lib/x64",
      "C:/nvidia-video-codec-sdk/Lib/x64"
    ]
    search_paths.extend(sdk_standard_paths)
  
  # Add common CUDA installation detection with version info
  cuda_locations = []
  detected_versions = []
  
  if system == "linux":
    cuda_locations = ["/usr/local/cuda", "/opt/cuda"]
    # Check for versioned CUDA installations
    for cuda_base in ["/usr/local", "/opt"]:
      if os.path.exists(cuda_base):
        try:
          for item in os.listdir(cuda_base):
            if item.startswith("cuda-"):
              version = item.replace("cuda-", "")
              cuda_locations.append(os.path.join(cuda_base, item))
              detected_versions.append(version)
              print(f"🔍 Found CUDA {version}: {os.path.join(cuda_base, item)}")
        except OSError:
          pass
          
  elif system == "windows":
    cuda_base = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    if os.path.exists(cuda_base):
      try:
        for version in sorted(os.listdir(cuda_base), reverse=True):
          cuda_locations.append(os.path.join(cuda_base, version))
          detected_versions.append(version) 
          print(f"🔍 Found CUDA {version}: {os.path.join(cuda_base, version)}")
      except OSError:
        pass
        
  elif system == "darwin":
    # Check Homebrew CUDA (if installed)
    brew_cuda = "/opt/homebrew/lib"
    if os.path.exists(brew_cuda):
      cuda_locations.append(brew_cuda)
      print(f"🔍 Found Homebrew CUDA path: {brew_cuda}")
  
  if detected_versions:
    print(f"🎯 Detected CUDA versions: {', '.join(detected_versions)}")
  
  # Add lib paths from detected CUDA installations
  for cuda_loc in cuda_locations:
    if system == "windows":
      search_paths.append(os.path.join(cuda_loc, "bin"))
    else:
      search_paths.append(os.path.join(cuda_loc, "lib64"))
      search_paths.append(os.path.join(cuda_loc, "lib"))
  
  # Try each combination of path and library name
  for search_path in search_paths:
    if not os.path.exists(search_path):
      continue
      
    for lib_name in lib_names:
      lib_full_path = os.path.join(search_path, lib_name)
      if os.path.exists(lib_full_path):
        try:
          print(f"🔍 Trying CUVID library: {lib_full_path}")
          return ctypes.CDLL(lib_full_path)
        except OSError as e:
          print(f"⚠️  Failed to load {lib_full_path}: {e}")
          continue
  
  # Try direct library names as last resort
  for lib_name in lib_names:
    try:
      print(f"🔍 Trying direct load: {lib_name}")
      return ctypes.CDLL(lib_name)
    except OSError:
      continue
  
  # Generate helpful error message
  error_msg = f"CUVID library not found on {system}. "
  if system == "linux":
    error_msg += "Install: apt-get install libnvidia-encode-515 (or latest driver)"
  elif system == "windows": 
    error_msg += "Install NVIDIA Video Codec SDK or CUDA Toolkit"
  elif system == "darwin":
    error_msg += "Install CUDA for macOS (if available) or use NVIDIA Web Drivers"
  
  error_msg += f"\n💡 Searched paths: {search_paths[:5]}..." if search_paths else ""
  error_msg += f"\n💡 Set CUDA_PATH or NVIDIA_VIDEO_CODEC_SDK_PATH environment variables"
  
  raise RuntimeError(error_msg)

# Load library with enhanced diagnostics
def load_cuvid_with_diagnostics():
  """Load CUVID library with detailed diagnostics"""
  try:
    print("🔍 Searching for CUVID library...")
    cuvid_lib = get_cuvid_lib()
    
    # Test basic functionality
    if hasattr(cuvid_lib, 'cuvidGetDecoderCaps'):
      print("✅ CUVID library loaded successfully")
      return cuvid_lib
    else:
      print("⚠️  CUVID library loaded but missing expected functions")
      return None
      
  except RuntimeError as e:
    print(f"❌ CUVID library loading failed: {e}")
    
    # Show diagnostic information
    system = platform.system().lower()
    print(f"\n🔧 Diagnostic Information:")
    print(f"   Platform: {system}")
    print(f"   Python: {sys.version}")
    
    # Check environment variables
    cuda_path = getenv("CUDA_PATH", "")
    sdk_path = getenv("NVIDIA_VIDEO_CODEC_SDK_PATH", "")
    print(f"   CUDA_PATH: {cuda_path if cuda_path else 'Not set'}")
    print(f"   NVIDIA_VIDEO_CODEC_SDK_PATH: {sdk_path if sdk_path else 'Not set'}")
    
    # Check common installation locations
    if system == "linux":
      common_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/x86_64-linux-gnu"]
      print(f"   Common CUDA paths:")
      for path in common_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"     {exists} {path}")
        
      # Check NVIDIA driver
      try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
          print(f"   ✅ NVIDIA driver detected")
        else:
          print(f"   ❌ NVIDIA driver not found or not working")
      except:
        print(f"   ❌ nvidia-smi not available")
        
    elif system == "windows":
      cuda_base = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
      exists = "✅" if os.path.exists(cuda_base) else "❌"
      print(f"   {exists} CUDA Toolkit: {cuda_base}")
      
    print(f"\n💡 Quick Setup:")
    if system == "linux":
      print(f"   sudo apt install nvidia-driver-535 nvidia-cuda-toolkit")
      print(f"   # Download SDK: https://developer.nvidia.com/nvidia-video-codec-sdk")
    elif system == "windows":
      print(f"   # Download CUDA: https://developer.nvidia.com/cuda-downloads")
      print(f"   # Download SDK: https://developer.nvidia.com/nvidia-video-codec-sdk") 
    elif system == "darwin":
      print(f"   # NVIDIA GPU not supported on recent macOS")
      print(f"   # Use VideoToolbox or Metal instead")
    
    return None

# Load library
cuvid = load_cuvid_with_diagnostics()

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