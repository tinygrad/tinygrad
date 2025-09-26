import ctypes
from tinygrad.helpers import getenv

# FIXME
MESA_PATH = f"{getenv('MESA_PREFIX', '/usr')}/lib/x86_64-linux-gnu"

class LazyDLL(ctypes.CDLL):
  def __init__(self, path): self.path, self.loaded = path, False
  def __getattr__(self, name):
    if not self.loaded:
      try: super().__init__(self.path)
      except OSError: raise AttributeError()
      self.loaded = True
    return super().__getattr__(name)

nir = LazyDLL(MESA_PATH + "/libvulkan_lvp.so")
lvp = LazyDLL(MESA_PATH + "/libvulkan_lvp.so")
nak = LazyDLL(MESA_PATH + "/libvulkan_nouveau.so")
