import ctypes, ctypes.util
from tinygrad.helpers import getenv
try: from tinygrad.runtime.support.llvm import LLVM_PATH
except (ImportError, FileNotFoundError): LLVM_PATH = ""

class dl_phdr_info(ctypes.Structure): _fields_ = [('padding', ctypes.c_void_p), ('name', ctypes.c_char_p)]

# FIXME
MESA_PATH = f"{getenv('MESA_PREFIX', '/usr')}/lib/x86_64-linux-gnu"

class LazyDLL(ctypes.CDLL):
  def __init__(self, path): self.path, self.loaded, self.error, self.mismatch = path, False, Exception(), False
  def __getattr__(self, name):
    if not self.loaded:
      try: super().__init__(self.path)
      except OSError as e:
        self.error = e
        raise AttributeError()
      self.loaded = True
      mesa_llvm = ""
      def cb(info):
        nonlocal mesa_llvm
        return 'libLLVM.so' in (mesa_llvm:=info.contents.name.decode()) and LLVM_PATH not in mesa_llvm
      if ctypes.CDLL(ctypes.util.find_library('c')).dl_iterate_phdr(ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(dl_phdr_info))(cb)):
        self.error, self.mismatch = RuntimeError(f"llvm version mismatch (mesa: {mesa_llvm[mesa_llvm.rfind('/')+1:]}, tinygrad: {LLVM_PATH})"), True
        raise self.error
    if self.mismatch: raise self.error
    try: return super().__getattr__(name)
    except AttributeError: self.error = AttributeError(f"{name} not visible in {self.path}, did you patch mesa?")
    except Exception as e: self.error = e
    raise self.error

nir = LazyDLL(MESA_PATH + "/libvulkan_lvp.so")
lvp = LazyDLL(MESA_PATH + "/libvulkan_lvp.so")
nak = LazyDLL(MESA_PATH + "/libvulkan_nouveau.so")
