import numpy as np
import functools
from tinygrad.runtime.lib import RawBufferCopyIn, LRUAllocator
from tinygrad.helpers import dtypes, DType
from tinygrad.ops import Compiled
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.glsl import GLSLLanguage
import moderngl
from array import array

ctx = moderngl.create_standalone_context()
dtype_map = { dtypes.half: "f2", dtypes.float: "f4", dtypes.int32: "i4", dtypes.uint8: "u1", dtypes.int8: "i1", dtypes.bool: "i1"}
class WebGLProgram:
  def __init__(self, name: str, prg: str):
    self.name, self.prg = name, ctx.program(
      vertex_shader="""
        #version 330

        in vec2 in_position;
        in vec2 in_uv;
        out vec2 uv;

        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_uv;
        }
      """, fragment_shader=prg)
  def __call__(self, *bufs, global_size, local_size, wait=False):
    self.vertices = ctx.buffer(array('f',[-1, 1, 0, 1, -1, -1, 0, 0, 1, 1, 1, 1, 1, -1, 1, 0]))
    self.quad = ctx.vertex_array(self.prg, [(self.vertices, '2f 2f', 'in_position', 'in_uv')])
    self.fbo = ctx.framebuffer(color_attachments=[bufs[0]._buf])

    for i, x in enumerate(bufs):
      if (i == 0): continue
      self.prg[f"data{i}"] = i
      x._buf.use(i)
    
    self.prg["w"].value = self.fbo.size[0]
    self.fbo.use()
    self.quad.render(mode=moderngl.TRIANGLE_STRIP)

    return
  
def reshape_texture(num, threshold):
  if num <= threshold: return (num, 1)
  
  for i in range(2, threshold + 1):
    if num % i == 0 and (num // i) <= threshold:
      return (num // i, i)
  
  return (num, 1)

class RawWebGLAllocator(LRUAllocator):
    def _do_alloc(self, size, dtype, device, **kwargs): 
      print(f"ALLOCATING SIZE={size}")
      return ctx.texture(reshape_texture(size, 4096), 1, dtype=dtype_map[dtype])
    def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize)
GLAlloc = RawWebGLAllocator()

class RawWebGLBuffer(RawBufferCopyIn):
  def __init__(self, size:int, dtype:DType):
    assert dtype not in [dtypes.int64, dtypes.uint64], f"dtype {dtype} not supported on WebGL"
    super().__init__(size, dtype, allocator=GLAlloc)
  def _copyin(self, x:np.ndarray): return self._buf.write(np.ascontiguousarray(x))
  def toCPU(self) -> np.ndarray: 
    backing = bytearray(self.size*self.dtype.itemsize)
    self._buf.read_into(backing)
    return np.frombuffer(backing, dtype=self.dtype.np)

renderer = functools.partial(uops_to_cstyle, GLSLLanguage())
WebGlBuffer = Compiled(RawWebGLBuffer, LinearizerOptions(device="WEBGL", supports_float4=False,supports_float4_alu=False, has_local=False), renderer, lambda x: x, WebGLProgram)
