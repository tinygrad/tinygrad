import ctypes, ctypes.util, contextlib, os, sys, subprocess
from tinygrad.helpers import DEBUG, getenv

if sys.platform == 'win32':
  # Windows llvm distribution doesn't seem to add itself to PATH or anywhere else where it can be easily retrieved from.
  # winget also doesn't have something like `brew --prefix llvm` so just hardcode default installation path with an option to override
  LLVM_PATH = getenv('LLVM_PATH', 'C:\\Program Files\\LLVM\\bin\\LLVM-C.dll')
  if not os.path.exists(LLVM_PATH):
    raise RuntimeError('LLVM not found, you can install it with `winget install LLVM.LLVM`')
  major, minor, patch = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
  ctypes.CDLL(LLVM_PATH).LLVMGetVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
  if DEBUG >= 4: print(f'Using LLVM {major.value}.{minor.value}.{patch.value}')
  LLVM_VER = major.value
else:
  llvms: dict[int, str] = {}
  for ver in range(14, 19+1):
    # Try 1 (plain ctypes.util.find_library, works on ubuntu and fedora, doesn't work on macos)
    path = ctypes.util.find_library(f'LLVM-{ver}')
    if path is not None:
      llvms[ver] = path
      continue
    # Try 2 (macos homebrew)
    with contextlib.suppress(FileNotFoundError):
      brew_prefix = subprocess.check_output(['brew', '--prefix', f'llvm@{ver}']).decode().strip()
      os.stat(brew_prefix) # `brew --prefix` will return even if formula is not installed
      llvm_ver = int(subprocess.check_output([os.path.join(brew_prefix, 'bin', 'llvm-config'), '--version']).decode().strip().split('.')[0])
      if llvm_ver != ver: raise FileNotFoundError('homebrew bug workaround')
      llvms[ver] = subprocess.check_output([os.path.join(brew_prefix, 'bin', 'llvm-config'), '--libfiles']).decode().strip()
      continue

  if len(llvms) == 0: raise RuntimeError('No LLVM library found on the system')
  LLVM_VER = getenv('LLVM_VER', next(reversed(llvms.keys())))
  if LLVM_VER not in llvms: raise RuntimeError(f'LLVM {LLVM_VER} not found, availible llvms: {llvms}')
  LLVM_PATH = llvms[LLVM_VER]
  if DEBUG >= 4: print(f'Using LLVM {LLVM_VER}')
