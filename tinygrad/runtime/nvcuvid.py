from __future__ import annotations
import ctypes, ctypes.util

class NVHevcDecoder:
  def __init__(self, width:int, height:int):
    libpath = ctypes.util.find_library('nvcuvid')
    if libpath is None:
      raise RuntimeError('libnvcuvid not found')
    self.lib = ctypes.CDLL(libpath)
    # Only include fields needed for basic decoder init
    class CUVIDDECODECREATEINFO(ctypes.Structure):
      _fields_ = [
        ('ulWidth', ctypes.c_uint),
        ('ulHeight', ctypes.c_uint),
        ('ulNumDecodeSurfaces', ctypes.c_uint),
        ('CodecType', ctypes.c_int),
        ('ChromaFormat', ctypes.c_int),
        ('ulCreationFlags', ctypes.c_uint),
      ]
    self._info = CUVIDDECODECREATEINFO(ulWidth=width, ulHeight=height,
      ulNumDecodeSurfaces=1, CodecType=8, ChromaFormat=1, ulCreationFlags=0)
    self.decoder = ctypes.c_void_p()
    res = self.lib.cuvidCreateDecoder(ctypes.byref(self.decoder),
      ctypes.byref(self._info))
    if res != 0:
      raise RuntimeError(f'cuvidCreateDecoder failed: {res}')

  def decode(self, bitstream:bytes):
    raise NotImplementedError('Decoding not implemented')

  def close(self):
    if self.decoder:
      self.lib.cuvidDestroyDecoder(self.decoder)
      self.decoder = None
