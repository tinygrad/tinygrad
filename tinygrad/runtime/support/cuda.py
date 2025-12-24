import ctypes.util, os, platform
from ctypes.util import find_library

CUDA_PATH: str | None
NVRTC_PATH: str | None
NVJITLINK_PATH: str | None

if platform.system() == "Windows":
  import glob, os.path
  def find_nv_dll(glob_pattern):
    cuda_bin = os.path.join(os.environ.get("CUDA_PATH", ""), "bin")
    matches = glob.glob(os.path.join(cuda_bin, glob_pattern))
    return matches[0] if matches else None  
  
  CUDA_PATH = find_library('nvcuda')
  NVRTC_PATH = find_nv_dll("nvrtc64_*.dll")
  NVJITLINK_PATH = find_nv_dll("nvJitLink_*.dll")
else:
  CUDA_PATH = find_library('cuda')
  NVRTC_PATH = find_library('nvrtc')
  NVJITLINK_PATH = find_library('nvJitLink')