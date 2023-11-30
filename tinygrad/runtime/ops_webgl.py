import numpy as np
import functools
from tinygrad.runtime.lib import RawBuffer, LRUAllocator
from tinygrad.helpers import dtypes, DType
from tinygrad.device import Compiled
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.glsl import GLSLLanguage
import moderngl
import os

CI = os.getenv("CI", "") != ""
ctx = moderngl.create_standalone_context()
dtype_map = { dtypes.float64: "f8", dtypes.float: "f4", dtypes.half: "f2", dtypes.int32: "i4", dtypes.uint32: "u4", dtypes.bool: "i1"}
class WebGLProgram:
  def __init__(self, name: str, prg: str): self.name, self.prg = name, ctx.program(vertex_shader="#version 330\nprecision highp float;\nin vec2 in_position;in vec2 in_uv;out vec2 uv;void main(){gl_Position=vec4(in_position,0.0,1.0);uv=in_uv;}", fragment_shader=prg)
  def __call__(self, *bufs, global_size, local_size, wait=False):
    vert, uv =ctx.buffer(np.asarray([-1, 1, -1, -1, 1, 1, 1, -1], dtype='f4').tobytes()), ctx.buffer(np.asarray([0, 1, 0, 0, 1, 1, 1, 0], dtype='f4').tobytes())
    self.vao = ctx.vertex_array(self.prg, [])
    self.vao.bind(0 if CI else self.prg["in_position"].location, buffer=vert, cls='f', fmt='2f4')
    self.vao.bind(1 if CI else self.prg["in_uv"].location, buffer=uv, cls='f', fmt='2f4')
    self.vao.vertices = vert.size//4//2
    self.fbo = ctx.framebuffer(color_attachments=[bufs[0]._buf])

    for i, x in enumerate(bufs):
      if (i == 0): continue
      if f"data{i}" in self.prg: 
        self.prg[f"data{i}"] = i
        x._buf.use(i)
    
    if ("w" in self.prg): self.prg["w"].value = self.fbo.size[0]
    ctx.viewport = (0, 0, self.fbo.size[0], self.fbo.size[1])
    self.fbo.use()
    self.vao.render(mode=moderngl.TRIANGLE_STRIP)
  
def reshape_texture(num, threshold):
  if num <= threshold: return (num, 1)
  
  for i in range(2, threshold + 1):
    if num % i == 0 and (num // i) <= threshold:
      return (num // i, i)
  
  return (num, 1)

class RawWebGLAllocator(LRUAllocator):
    def _do_alloc(self, size, dtype, device, **kwargs): 
      tex = ctx.texture(reshape_texture(size, 4096), 1, dtype=dtype_map[dtype])
      tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
      return tex
    def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize)
GLAlloc = RawWebGLAllocator()

class RawWebGLBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType):
    assert dtype not in [dtypes.int8,dtypes.uint8,dtypes.int64,dtypes.uint64], f"dtype {dtype} not supported on WebGL"
    super().__init__(size, dtype, allocator=GLAlloc)
  def _copyin(self, x:np.ndarray): return self._buf.write(np.ascontiguousarray(x))
  def toCPU(self) -> np.ndarray: 
    backing = bytearray(self.size*self.dtype.itemsize)
    self._buf.read_into(backing)
    return np.frombuffer(backing, dtype=self.dtype.np)

renderer = functools.partial(uops_to_cstyle, GLSLLanguage())
WebGlDevice = Compiled(RawWebGLBuffer, LinearizerOptions(device="WEBGL", supports_float4=False,supports_float4_alu=False, has_local=False), renderer, lambda x: x, WebGLProgram)
