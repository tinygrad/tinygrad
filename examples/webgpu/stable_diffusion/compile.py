from os import path
from examples.compile_efficientnet import compile_net, jit_model
from examples.stable_diffusion import StableDiffusion, ClipTokenizer
from tinygrad.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
from extra.utils import download_file
from tinygrad.helpers import prod
import numpy as np
import os
import tempfile
from pathlib import Path
import math
from tqdm import tqdm
from tinygrad.helpers import dtypes, GlobalCounters

FILENAME = Path(__file__).parent.parent.parent.parent / "weights/sd-v1-4.ckpt"

def create_multipart_safetensor(fn, part_size):
  _, json_len, metadata = safe_load_metadata(fn)
  last_offset = 0
  part_start_offsets = []

  for k in metadata:
    offset = metadata[k]['data_offsets'][0]
    part_offset = offset - last_offset
    
    if (part_offset >= part_size):
      part_start_offsets.append(8+json_len+offset)
      last_offset = offset

  net_bytes = bytes(open(fn, 'rb').read())
  part_start_offsets.append(len(net_bytes))
  cur_pos = 0
  
  for i, end_pos in enumerate(part_start_offsets):
    with open(f'./net_part{i}.safetensors', "wb+") as f:
      f.write(net_bytes[cur_pos:end_pos])
      cur_pos = end_pos

  return part_start_offsets

if __name__ == "__main__":
  Device.DEFAULT = "WEBGPU"

  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  download_file('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', FILENAME)
  load_state_dict(model, torch_load(FILENAME)['state_dict'], strict=False)


  def text_model(model):
    def forward(prompt):
      return model.cond_stage_model.transformer.text_model(prompt).realize()
    
    return {'forward': forward, 'name': 'text_model'}
  
  def diffusor(model):
    def forward(latent, timestep, unconditional_context, context):
      latents = model.model.diffusion_model(latent.expand(2, *latent.shape[1:]), timestep.expand(2, *timestep.shape[1:]), unconditional_context.cat(context, dim=0))
      unconditional_latent, latent = latents[0:1], latents[1:2]

      unconditional_guidance_scale = 7.5
      e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
      return e_t

    return {'forward': forward, 'name': 'diffusor'}
  
  def get_x_prev_and_pred_x0():
    def forward(x, e_t, alphas, alphas_prev):
      a_t, a_prev = alphas, alphas_prev
      sigma_t = 0
      sqrt_one_minus_at = Tensor.sqrt(1-a_t)

      pred_x0 = (x - sqrt_one_minus_at * e_t) / Tensor.sqrt(a_t)

      dir_xt = Tensor.sqrt(1. - a_prev - sigma_t**2) * e_t

      x_prev = Tensor.sqrt(a_prev) * pred_x0 + dir_xt
    
      return Tensor.cat(x_prev, pred_x0, dim=0)
    
    return {'forward': forward, 'name': 'get_x_prev_and_pred_x0'}
  
  def decoder(model):
    def forward(latent):
      x = latent.reshape(1,4,64,64)
      x = model.first_stage_model.post_quant_conv(1/0.18215 * x)
      x = model.first_stage_model.decoder(x)

      # make image correct size and scale
      x = (x + 1.0) / 2.0
      return (x.reshape(3,512,512).permute(1,2,0).clip(0,1)*255)

    return {'forward': forward, 'name': 'decoder'}
  
  tokenizer = ClipTokenizer()
  prompt = Tensor([tokenizer.encode("Test test test")]).expand(1, 77)
  print(prompt)
  sub_steps = [
    {"input": prompt, "run": text_model(model)}, 
    {"input": [Tensor.rand(1,4,64,64), Tensor([1]), Tensor.rand(1, 77, 768), Tensor.rand(1, 77, 768)], "run": diffusor(model)}, 
    {"input": [Tensor.rand(1,4,64,64), Tensor.rand(1,4,64,64), Tensor.rand(1), Tensor.rand(1)], "run": get_x_prev_and_pred_x0()}, 
    {"input": Tensor.rand(1,4,64,64), "run": decoder(model)}
  ]
  
  prg = ""

  def compile_step(model, step_data, input):
    if hasattr(model, "forward"):
      delattr(model, 'forward')

    print(f'in compile step={input}')
    setattr(model, 'forward', step_data["forward"])
    run, special_names = jit_model(model, *input)
    print(f'special_names={special_names}')
    functions, statements, bufs, _ = compile_net(run, special_names)
    # Common part used by all substeps
    state = get_state_dict(model)
    weights = {id(x.lazydata.realized): name for name, x in state.items()}


    kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
    kernel_names = ', '.join([name for (name, _args, _global_size) in statements])
    kernel_calls = '\n    '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size) in enumerate(statements) ])
    bufs =  '\n    '.join([f"const {buf[0]} = " + (f"createEmptyBuf(device, {buf[1]});" if buf[2] not in weights  else f"createWeightBuf(device, {buf[1]}, getTensorBuffer(safetensor, metadata['{weights[buf[2]]}'], '{weights[buf[2]]}'))") + ";"  for buf in bufs.values()])
    gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if value != "outputs"])
    input_writer = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n    new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n    gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "outputs"])
    return f"""\n    var {step_data["name"]}Model = function() {{
    
    {kernel_code}

    return {{
      "setup": async (device, safetensor) => {{
        const metadata = getTensorMetadata(safetensor[0]);

        {bufs}

        
        {gpu_write_bufs}
        const gpuReadBuffer = device.createBuffer({{ size: outputs.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

        const kernels = [{kernel_names}];
        const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

        return async ({",".join([f'data{i}' for i,(k,v) in enumerate(special_names.items()) if v != "outputs"])}) => {{
            const commandEncoder = device.createCommandEncoder();

            {input_writer}

            {kernel_calls}
            commandEncoder.copyBufferToBuffer(outputs, 0, gpuReadBuffer, 0, outputs.size);
            const gpuCommands = commandEncoder.finish();
            device.queue.submit([gpuCommands]);

            await gpuReadBuffer.mapAsync(GPUMapMode.READ);
            const resultBuffer = new Float32Array(gpuReadBuffer.size);
            resultBuffer.set(new Float32Array(gpuReadBuffer.getMappedRange()));
            gpuReadBuffer.unmap();
            return resultBuffer;
        }}
      }} 
    }}
  }}
  """

  for step in sub_steps:
    run = step["run"]
    print(f'Executing step={run["name"]}')
    prg += compile_step(model, run, step["input"])
  

  state = get_state_dict(model)
  safe_save(state, path.join(path.dirname(__file__), "net.safetensors"))
  part_start_offsets = create_multipart_safetensor("./net.safetensors", 1073741824)
  safetensor_parts = '\n    '.join([f"parts.push(new Uint8Array(await (await fetch('./net_part{i}.safetensors')).arrayBuffer()));" for i, _ in enumerate(part_start_offsets)])

  prekernel = f"""const getTensorMetadata = (safetensorBuffer) => {{
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
    }};

  const getSafetensorParts = async () => {{
    let parts = [];

    {safetensor_parts}

    return parts;
  }}


  const getTensorBuffer = (safetensorParts, tensorMetadata, key) => {{
    let selectedPart = 0;
    let counter = 0;
    let partStartOffsets = {part_start_offsets};
    let correctedOffsets = tensorMetadata.data_offsets;
    let prev_offset = 0;

    for (let start of partStartOffsets) {{
      prev_offset = (counter == 0) ? 0 : partStartOffsets[counter-1];

      if (tensorMetadata.data_offsets[0] < start) {{
        selectedPart = counter;
        correctedOffsets = [correctedOffsets[0]-prev_offset, correctedOffsets[1]-prev_offset];
        break;
      }}

      counter++;
    }}

    return safetensorParts[selectedPart].subarray(...correctedOffsets);
  }}
    
  const createEmptyBuf = (device, size) => {{
      return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
  }};

  const createWeightBuf = (device, size, data) => {{
    const buf = device.createBuffer({{ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE }});
    new Uint8Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }};

  const addComputePass = (device, commandEncoder, pipeline, bufs, workgroup) => {{
    const bindGroup = device.createBindGroup({{layout: pipeline.getBindGroupLayout(0), entries: bufs.map((buffer, index) => ({{ binding: index, resource: {{ buffer }} }}))}});
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...workgroup);
    passEncoder.end();
  }};"""

  with open(path.join(path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prekernel + prg)
