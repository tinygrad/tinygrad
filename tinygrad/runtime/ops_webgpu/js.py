# Compute ops
def create_layout(buf_types:list[str]) -> str: return "device.createBindGroupLayout({entries:" + \
  f'[{", ".join(f"{{binding: {i}, visibility: GPUShaderStage.COMPUTE, buffer: {{type: {btype}}} }}" for i, btype in enumerate(buf_types))}]}})'

def create_pipeline(layout:str, code:str) -> str: return f"""device.createComputePipelineAsync({{
  layout: device.createPipelineLayout({{bindGroupLayouts: [{layout}]}}),
  compute: {{ module: device.createShaderModule({{ code: {code} }}), entryPoint: "main" }}\n}})"""

def create_bind_group(layout:str, entries:str) -> str: return f"device.createBindGroup({{ layout: {layout}, entries: {entries} }})"

init_encoder = "device.createCommandEncoder()"
def begin_compute_pass(command_encoder:str, pipeline:str, bind_group:str, global_dims:str) -> list[str]:
  return [f"const passEncoder = {command_encoder}.beginComputePass();", f"passEncoder.setPipeline({pipeline});",
    f"passEncoder.setBindGroup(0, {bind_group});", f"passEncoder.dispatchWorkgroups(...{global_dims});", "passEncoder.end();"]

# Allocator ops
# TODO: handle 4-byte alignment
#if src.nbytes % 4: pad_src = f"const padded = new Uint8Array({src}.length + (4 - {src}.length % 4) % 4); padded.set({src});"
def alloc(size:str, usage:str) -> str: return f"device.createBuffer({{size: {size}, usage: {usage}}})"

# NOTE: dest/src names are resolved by tinygrad.renderer.graph.GraphRenderer
def copyin(dest:str, src:str) -> str: return f"device.queue.writeBuffer({dest}, 0, {src});"
def copyout(dest:str, src:str) -> list[str]:
  return [f'await {src}.mapAsync(GPUMapMode.READ);',
    f'{dest}.set(new {dest}.constructor({src}.getMappedRange()));',
    f'{src}.unmap();']
def copy(encoder:str, src:str, dest:str, size:str) -> str: return f"{encoder}.copyBufferToBuffer({src}, 0, {dest}, 0, {size});"

# Device ops
init_device = ['if (!navigator.gpu) throw new Error("WebGPU not supported.");',
  'const adapter = await navigator.gpu.requestAdapter();',
  'const { maxStorageBufferBindingSize, maxBufferSize, maxComputeInvocationsPerWorkgroup } = adapter.limits;',
  'const device = await adapter.requestDevice({',
  '  requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [], powerPreference: "high-performance",',
  '  requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxComputeInvocationsPerWorkgroup }',
  '});\n']