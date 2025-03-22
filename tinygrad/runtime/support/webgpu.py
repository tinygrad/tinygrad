import ctypes, ctypes.util, os, subprocess
from tinygrad.helpers import OSX

if OSX:
  brew_prefix = subprocess.check_output(['brew', '--prefix', 'dawn']).decode().strip()

  if not os.path.exists(brew_prefix):
    raise FileNotFoundError('dawn library not found. Install it with `brew tap wpmed92/dawn && brew install dawn`')
  WEBGPU_PATH: str|None = os.path.join(brew_prefix, 'lib', 'libwebgpu_dawn.dylib')
else:
  WEBGPU_PATH = ctypes.util.find_library('webgpu_dawn')

  if WEBGPU_PATH is None:
    raise FileNotFoundError("dawn library not found. Install it with `sudo curl -L https://github.com/wpmed92/pydawn/releases/download/v0.1.6/libwebgpu_dawn.so -o /usr/lib/libwebgpu_dawn.so`")
