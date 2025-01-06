import ctypes, ctypes.util, contextlib, os, subprocess
from tinygrad.helpers import DEBUG, getenv

LLVMS: dict[int, str] = {}
for ver in range(14, 19+1):
  # Try 1 (plain ctypes.util.find_library, works on ubuntu and fedora, doesn't work on macos)
  path = ctypes.util.find_library(f'LLVM-{ver}')
  if path is not None:
    LLVMS[ver] = path
    continue
  # Try 2 (macos homebrew)
  with contextlib.suppress(FileNotFoundError):
    brew_prefix = subprocess.check_output(['brew', '--prefix', f'llvm@{ver}']).decode().strip()
    os.stat(brew_prefix) # `brew --prefix` will return even if formula is not installed
    llvm_config_version = int(subprocess.check_output([os.path.join(brew_prefix, 'bin', 'llvm-config'), '--version']).decode().strip().split('.')[0])
    if llvm_config_version != ver: raise FileNotFoundError('homebrew bug workaround')
    LLVMS[ver] = subprocess.check_output([os.path.join(brew_prefix, 'bin', 'llvm-config'), '--libfiles']).decode().strip()
    continue

if len(LLVMS) == 0: raise RuntimeError('No LLVM library found on the system')
LLVM_VER = getenv('LLVM_VER', next(reversed(LLVMS.keys())))
if LLVM_VER not in LLVMS: raise RuntimeError(f'LLVM {LLVM_VER} not found, availible llvms: {LLVMS}')
if DEBUG >= 4: print(f'Using LLVM {LLVM_VER}')
