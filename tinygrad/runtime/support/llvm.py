import ctypes, ctypes.util, os, sys, subprocess
from tinygrad.helpers import DEBUG, OSX, getenv

if sys.platform == 'win32':
  # Windows llvm distribution doesn't seem to add itself to PATH or anywhere else where it can be easily retrieved from.
  # winget also doesn't have something like `brew --prefix llvm` so just hardcode default installation path with an option to override
  LLVM_PATH = getenv('LLVM_PATH', 'C:\\Program Files\\LLVM\\bin\\LLVM-C.dll')
  if not os.path.exists(LLVM_PATH):
    raise RuntimeError('LLVM not found, you can install it with `winget install LLVM.LLVM` or point at a custom dll with LLVM_PATH')
elif OSX and 'tinygrad.runtime.ops_metal' in sys.modules:
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  LLVM_PATH = ctypes.util.find_library('tinyllvm')
  if LLVM_PATH is None:
    raise RuntimeError("LLVM can't be opened in the same process with metal. You can install llvm distribution which supports that via `brew install uuuvn/tinygrad/tinyllvm`") # noqa: E501
elif OSX:
  brew_prefix = subprocess.check_output(['brew', '--prefix', 'llvm']).decode().strip()
  # `brew --prefix` will return even if formula is not installed
  if not os.path.exists(brew_prefix):
    raise RuntimeError('LLVM not found, you can install it with `brew install llvm`')
  LLVM_PATH = os.path.join(brew_prefix, 'lib', 'libLLVM.dylib')
else:
  LLVM_PATH = ctypes.util.find_library('LLVM')
  # use newer LLVM if possible
  for ver in reversed(range(14, 19+1)):
    if LLVM_PATH is not None: break
    LLVM_PATH = ctypes.util.find_library(f'LLVM-{ver}')
  if LLVM_PATH is None:
    raise RuntimeError("No LLVM library found on the system. Install it via your distro's package manager and ensure it's findable as 'LLVM'")

if DEBUG>=2: print(f'Using LLVM at {repr(LLVM_PATH)}')
