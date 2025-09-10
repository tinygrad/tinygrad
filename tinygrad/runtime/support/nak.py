from typing import cast, Tuple
from tinygrad.device import Compiler
from tinygrad.helpers import round_up
import tinygrad.runtime.autogen.nak as nak
import tinygrad.runtime.autogen.nir as nir
import ctypes, pathlib, tempfile, hashlib, subprocess, functools

class NAKCompiler(Compiler):
  def __init__(self, dev, cache_key="nak"):
    self.arch = dev.arch
    self.cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=int(dev.arch[3:]), max_warps_per_mp=dev.max_warps_per_sm))
    super().__init__(f"compile_{cache_key}_{dev.arch}")
  def compile(self, src) -> bytes:
    shader = nak.struct_nir_shader.from_buffer(cast(bytes, src))
    nak.nak_preprocess_nir(shader, self.cc)
    return nak.nak_compile_shader(shader, False, self.cc, 0, None).contents
  def disassemble(self, lib: bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinynir_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn, "wb") as f: f.write(parse_nak_shader(lib)[0])
      print(subprocess.check_output(['nvdisasm', "-b", f"SM{self.arch[3:]}", fn]).decode('utf-8'))
    except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")
  @functools.cached_property
  def nir_options(self): return ctypes.cast(nak.nak_nir_options(self.cc), ctypes.POINTER(nir.nir_shader_compiler_options))

def parse_nak_shader(shader:bytes) -> Tuple[memoryview, int, int, int]:
  sb = nak.struct_nak_shader_bin.from_buffer(shader)
  return (memoryview(ctypes.cast(sb.code, ctypes.POINTER(ctypes.c_char * sb.code_size)).contents), sb.info.num_gprs,
          round_up(sb.info.cs.smem_size, 0x80), round_up(sb.info.slm_size, 0x10))

