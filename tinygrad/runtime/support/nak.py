from typing import cast, Tuple
from tinygrad.device import Compiler
from base64 import b64decode
from gzip import decompress
import tinygrad.runtime.autogen.nak as nak
import tinygrad.runtime.autogen.nir as nir
import ctypes, pathlib, tempfile, hashlib, subprocess

class NAKCompiler(Compiler):
  def __init__(self, dev, cache_key="nak"):
    self.arch = dev.arch
    self.cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=int(dev.arch[3:]), max_warps_per_mp=dev.max_warps_per_sm))
    super().__init__(f"compile_{cache_key}_{dev.arch}")
  def compile(self, src) -> bytes:
    shader = nak.struct_nir_shader.from_buffer(cast(bytes, src))
    # TODO: only "True" if we want to print sass
    nak.nak_preprocess_nir(shader, self.cc)
    out = nak.nak_compile_shader(shader, True, self.cc, 0, None).contents
    return out
  def disassemble(self, lib: bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn, "wb") as f: f.write(parse_nak_shader(lib)[0])
      print(subprocess.check_output(['nvdisasm', "-b", f"SM{self.arch[3:]}", fn]).decode('utf-8'))
    except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")

def parse_nak_shader(shader:bytes) -> Tuple[memoryview, int, int, int]:
  sb = nak.struct_nak_shader_bin.from_buffer(shader)
  return (memoryview(bytearray(ctypes.string_at(sb.code, sb.code_size))), sb.info.num_gprs, sb.info.cs.smem_size, sb.info.slm_size)

# TODO: just call nak_nir_options
nir_options = nir.nir_shader_compiler_options.from_buffer_copy(
  decompress(b64decode(b"H4sIAAAAAAAAA2NkAANGIABRIBZUAELBWWA2WB1cBSOqWohqEIQLoABGiBE4ABYpBRzS++sbGNzZIWx+fpgC7JYSAgBqun8JAAEAAA=="))
)

