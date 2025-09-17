from typing import Tuple
from tinygrad.device import Compiler
from tinygrad.helpers import round_up
import tinygrad.runtime.autogen.nak as nak
from tinygrad.runtime.autogen.nir import nir_shader_compiler_options
import ctypes, pathlib, tempfile, hashlib, subprocess

class NAKCompiler(Compiler):
  def __init__(self, dev, cache_key="nak"):
    self.arch = dev.arch
    self.cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=int(dev.arch[3:]), max_warps_per_mp=dev.max_warps_per_sm))
    self.nir_options = ctypes.cast(nak.nak_nir_options(self.cc), ctypes.POINTER(nir_shader_compiler_options))
    super().__init__(f"compile_{cache_key}_{dev.arch}")
  def compile(self, src) -> bytes:
    nak.glsl_type_singleton_init_or_ref() # TODO: call glsl_type_singleton_decref somewhere
    blobreader = nak.struct_blob_reader()
    nak.blob_reader_init(blobreader, src, len(src))
    shader = nak.nir_deserialize(None, ctypes.cast(self.nir_options, ctypes.POINTER(nak.nir_shader_compiler_options)), blobreader)
    nak.nak_preprocess_nir(shader, self.cc)
    return nak.nak_compile_shader(shader, False, self.cc, 0, None).contents
  def disassemble(self, lib: bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinynak_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn, "wb") as f: f.write(parse_nak_shader(lib)[0])
      print(subprocess.check_output(['nvdisasm', "-b", f"SM{self.arch[3:]}", fn]).decode('utf-8'))
    except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")

def parse_nak_shader(shader:bytes) -> Tuple[memoryview, int, int, int]:
  sb = nak.struct_nak_shader_bin.from_buffer(shader)
  return (memoryview(ctypes.cast(sb.code, ctypes.POINTER(ctypes.c_char * sb.code_size)).contents), sb.info.num_gprs,
          round_up(sb.info.cs.smem_size, 0x80), round_up(sb.info.slm_size, 0x10))

