from typing import List, Tuple
import functools, time
from wgpu.utils.device import get_default_device
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import UOp, UOps
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.cstyle import CStyleLanguage
import wgpu

class WGSLCompiler(Compiler):
  def compile(self, src): return src.encode()

class WGSLRenderer(CStyleLanguage):
  code_for_workitem = {"g": lambda x: f"i32(gindex.{'xyz'[int(x)]})", "l": lambda x: f"i32(lindex.{'xyz'[int(x)]})"}
  supports_float4 = False
  type_map = {dtypes.float: "f32", dtypes.half: "f16", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "f32"}

  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    if self.type_map[var_dtype]: return f"bitcast<{self.type_map[var_dtype]}>({x})" if bitcast else f"{self.type_map[var_dtype]}({x})"
    raise NotImplementedError(f"no cast for {var_dtype}")
  def render_dtype(self, dtype): return "var"
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:List[UOp], prefix=None) -> str:
    local_size = [u.arg[1] for u in uops if u.op is UOps.SPECIAL and u.arg[0][0] == 'l']
    if not local_size: local_size = [1]
    prekernel = []
    bind_it = iter(range(len(bufs)))
    prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\nfn inf(a: f32) -> f32 { return a/0.0; }\n"
    prg += "\n".join(prekernel+[f"@group(0) @binding({next(bind_it)}) {'var<storage,read_write>' if isinstance(dtype, PtrDType) else 'var<uniform>'} {name}: {f'array<{self.type_map[dtype]}>' if isinstance(dtype, PtrDType) else 'i32'};" for name,(dtype,rw) in bufs])  # noqa: E501
    prg += f"\n@compute @workgroup_size({','.join([str(x) for x in local_size])}) fn {function_name}(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"  # noqa: E501
    return prg

def create_uniform(wgpu_device, val: int) -> wgpu.GPUBuffer:
  buf = wgpu_device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
  wgpu_device.queue.write_buffer(buf, 0, val.to_bytes(4, "little"))
  return buf

class WebGPUProgram:
  def __init__(self, device, name:str, lib:bytes):
    self.device = device
    self.name, self.lib, self.prg = name, lib, self.device.create_shader_module(code=lib.decode())   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size, local_size, vals=(), wait=False):
    assert len(bufs) <= 8, "WEBGPU only supports 8 buffers"
    binding_layouts = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform if i >= len(bufs) else wgpu.BufferBindingType.storage }} for i in range(len(bufs)+len(vals))]  # noqa: E501
    bindings = [{"binding": i, "resource": {"buffer": create_uniform(self.device, x) if i >= len(bufs) else x, "offset": 0, "size": 4 if i >= len(bufs) else x.size}} for i,x in enumerate(bufs+vals)]  # noqa: E501
    bind_group_layout = self.device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = self.device.create_bind_group(layout=bind_group_layout, entries=bindings)
    compute_pipeline = self.device.create_compute_pipeline(layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name},)
    command_encoder = self.device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999) # last 2 not used
    compute_pass.dispatch_workgroups(*global_size)  # x y z
    compute_pass.end()
    st = time.perf_counter()
    self.device.queue.submit([command_encoder.finish()])
    return time.perf_counter() - st

class WebGpuAllocator(Allocator):
  def __init__(self, device): self.device = device
  def _alloc(self, size: int, options):
    return self.device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
  def copyin(self, dest, src: memoryview): self.device.queue.write_buffer(dest, 0, src)
  def copyout(self, dest, src: memoryview): dest[:] = self.device.queue.read_buffer(src, 0)    # TODO: remove this copy

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    wgpu_device = get_default_device()
    super().__init__(device, WebGpuAllocator(wgpu_device), WGSLRenderer(), WGSLCompiler(), functools.partial(WebGPUProgram, wgpu_device))
                     #CompilerOptions(device="WEBGPU", supports_float4=False, local_max=[256, 256, 64],
                     #                                     global_max=[65535, 65535, 65535]), WGSLRenderer, lambda x: x, WebGPUProgram)
