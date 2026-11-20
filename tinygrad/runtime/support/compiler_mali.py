import base64, ctypes, re, struct, subprocess, tempfile, pathlib, sys
from tinygrad.renderer import Compiler, CompileError
from tinygrad.runtime.autogen import cmpbe

_TYPE_SIZE = {
  'bool': 1, 'char': 1, 'uchar': 1, 'int8_t': 1, 'uint8_t': 1, 'short': 2, 'ushort': 2, 'int16_t': 2, 'uint16_t': 2,
  'half': 2, 'int': 4, 'uint': 4, 'float': 4,
  'long': 8, 'ulong': 8, 'int64_t': 8, 'uint64_t': 8, 'float16_t': 2, 'f16vec4': 8, 'vec4': 16,
  'char2': 2, 'uchar2': 2, 'short2': 4, 'ushort2': 4, 'half2': 4,
  'int2': 8, 'uint2': 8, 'float2': 8, 'char4': 4, 'uchar4': 4, 'short4': 8, 'ushort4': 8,
  'half4': 8, 'int4': 16, 'uint4': 16, 'float4': 16,
}

class MALICompiler(Compiler):
  def __init__(self, arch='g610', isolated=False):
    self.arch, self.isolated = arch, isolated
    if not isolated: self.compiler_process = self.server(sys.executable, arch, True)
    super().__init__('compile_mali_glsl')
  def __reduce__(self): return MALICompiler, (self.arch, False)
  def __del__(self):
    if not self.isolated and hasattr(self, 'compiler_process'): self.compiler_process.kill()

  def compile(self, src:str) -> bytes:
    if not self.isolated: return self.compile_server(src, self.compiler_process)
    libc = ctypes.CDLL('libc.so.6')
    libc.malloc.argtypes, libc.malloc.restype = [ctypes.c_size_t], ctypes.c_void_p
    libc.free.argtypes, libc.free.restype = [ctypes.c_void_p], None
    alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)(libc.malloc)
    free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(libc.free)
    ctx = (ctypes.c_ubyte * 256)()
    cmpbe.put_ptr(ctx, 120, ctypes.cast(alloc, ctypes.c_void_p).value)
    cmpbe.put_ptr(ctx, 128, ctypes.cast(free, ctypes.c_void_p).value)
    with tempfile.TemporaryDirectory() as td:
      src_path, spv_path = pathlib.Path(td) / 'shader.comp', pathlib.Path(td) / 'shader.spv'
      src_path.write_text(src)
      proc = subprocess.run(['glslangValidator', '-S', 'comp', '--target-env', 'vulkan1.0', '-o', str(spv_path), str(src_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      if proc.returncode or not spv_path.exists(): raise CompileError(proc.stdout.decode(errors='replace') + '\n' + src)
      spirv = spv_path.read_bytes()
    ctx[216] = 1  # GLCompute SPIR-V, no OpenCL or kernel driver context
    entry = ctypes.c_char_p(b'main')
    variant = (ctypes.c_ubyte * 80)()
    cmpbe.cmpbe_v2_init_options(variant, 0)
    cmpbe.cmpbe_v2_set_option_value(variant, 12, 1)
    cmpbe.cmpbe_v2_set_option_value(variant, 16, 1)
    shader = (ctypes.c_ubyte * 40)()
    cmpbe.put_ptr(shader, 0, ctypes.cast(entry, ctypes.c_void_p).value)
    cmpbe.put_ptr(shader, 24, ctypes.addressof(variant))
    source = ctypes.create_string_buffer(spirv)
    result = ctypes.c_void_p()
    rc = cmpbe.cmpbe_v2_compile_single_shader(ctx, source, len(spirv), shader, 6, 7, 0, None, None, ctypes.byref(result))
    if rc != 0:
      log = ctypes.c_void_p.from_address(result.value + 8).value if result.value else None
      msg = ctypes.string_at(log).decode(errors='replace') if log else f'cmpbe_v2 error {rc}'
      if result.value: cmpbe.cmpbe_v2_free_compilation_result(result)
      raise CompileError(msg)
    if not result.value: raise CompileError('vendor compiler returned no result')
    mbs_ptr = ctypes.c_uint64.from_address(result.value + 16).value
    mbs_size = ctypes.c_uint64.from_address(result.value + 24).value
    if not mbs_ptr or not mbs_size:
      cmpbe.cmpbe_v2_free_compilation_result(result)
      raise CompileError('vendor compiler returned no MBS2')
    mbs2 = ctypes.string_at(mbs_ptr, mbs_size)
    cmpbe.cmpbe_v2_free_compilation_result(result)
    shared_size = 0
    for typ, _, count in re.findall(r'shared\s+(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;', src):
      if typ not in _TYPE_SIZE: raise CompileError(f'unknown local type {typ}')
      shared_size += _TYPE_SIZE[typ] * int(count)
    metadata = struct.pack('<4sIIII', b'MLV1', shared_size, 0, 0, int('barrier(' in src))
    return base64.b64encode(metadata + mbs2)
