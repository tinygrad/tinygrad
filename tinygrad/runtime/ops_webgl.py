import numpy as np
import functools
from tinygrad.helpers import dtypes, DType
from tinygrad.device import Compiled, Allocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.glsl import GLSLLanguage
import moderngl
import os

CI = os.getenv("CI", "") != ""
ctx = moderngl.create_standalone_context()
dtype_map = { dtypes.float64: "f8", dtypes.float: "f4", dtypes.half: "f2", dtypes.int32: "i4", dtypes.uint32: "u4", dtypes.bool: "i1"}
class WebGLProgram:
  def __init__(self, name: str, prg: str, bufs:int=0, vars:int=0): self.name, self.prg = name, ctx.program(vertex_shader="#version 330\nprecision highp float;\nin vec2 in_position;in vec2 in_uv;out vec2 uv;void main(){gl_Position=vec4(in_position,0.0,1.0);uv=in_uv;}", fragment_shader=prg)
  def __call__(self, *bufs, global_size, wait=False):
    vert, uv =ctx.buffer(np.asarray([-1, 1, -1, -1, 1, 1, 1, -1], dtype='f4').tobytes()), ctx.buffer(np.asarray([0, 1, 0, 0, 1, 1, 1, 0], dtype='f4').tobytes())
    self.vao = ctx.vertex_array(self.prg, [])
    self.vao.bind(0 if CI else self.prg["in_position"].location, buffer=vert, cls='f', fmt='2f4')
    self.vao.bind(1 if CI else self.prg["in_uv"].location, buffer=uv, cls='f', fmt='2f4')
    self.vao.vertices = vert.size//4//2
    self.fbo = ctx.framebuffer(color_attachments=[bufs[0]])

    for i, x in enumerate(bufs):
      if (i == 0): continue
      if f"data{i}" in self.prg:
        self.prg[f"data{i}"] = i
        x.use(i)

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

class RawWebGLAllocator(Allocator):
  def _alloc(self, size: int): return ctx.buffer(reserve=size)
  def _cast_image(self, buf:moderngl.Buffer, dtype:DType, size:int) -> moderngl.Texture:
    tex = ctx.texture(reshape_texture(size, 4096), 1, dtype=dtype_map[dtype])
    tex.write(buf)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return tex
  def copyin(self, dest:moderngl.Buffer, src: memoryview): dest.write(src)
  def copyout(self, dest:memoryview, src: moderngl.Buffer):
    src.read_into(dest)
    return dest

class WebGlDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(RawWebGLAllocator(), LinearizerOptions(device="WEBGL", supports_float4=False, supports_float4_alu=False, has_local=False, has_shared=False),
                     functools.partial(uops_to_cstyle, GLSLLanguage()), lambda x: x, WebGLProgram)
