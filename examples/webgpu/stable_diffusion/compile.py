from os import path
from examples.compile_efficientnet import compile_net, jit_model
from examples.stable_diffusion import StableDiffusion, ClipTokenizer
from tinygrad.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
from extra.utils import download_file
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

  tokenizer = ClipTokenizer()
  #Tensor([tokenizer.encode("The brain drawn in the style of Da Vinci sketch")]

  print(tokenizer.encode(""))
  run, special_names = jit_model(model, Tensor([tokenizer.encode("")]))
  functions, statements, bufs, _ = compile_net(run, special_names)
  
  state = get_state_dict(model)
  weights = {id(x.lazydata.realized): name for name, x in state.items()}
  safe_save(state, path.join(path.dirname(__file__), "net.safetensors"))
  part_start_offsets = create_multipart_safetensor("./net.safetensors", 1073741824)

  kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
  for (name, _args, _) in statements:
    if (len(_args) > 8):
      print(f'Too many storage buffers for a compute stage: {len(_args)}')
      print(name)

  kernel_names = ', '.join([name for (name, _args, _global_size) in statements])
  kernel_calls = '\n    '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size) in enumerate(statements) ])
  bufs =  '\n    '.join([f"const {buf[0]} = " + (f"createEmptyBuf(device, {buf[1]});" if buf[2] not in weights else f"createWeightBuf(device, {buf[1]}, getTensorBuffer(safetensor, metadata['{weights[buf[2]]}'], '{weights[buf[2]]}'))") + ";"  for buf in bufs.values()])
  safetensor_parts = '\n    '.join([f"parts.push(new Uint8Array(await (await fetch('./net_part{i}.safetensors')).arrayBuffer()));" for i, _ in enumerate(part_start_offsets)])

  prg = f"""const getTensorMetadata = (safetensorBuffer) => {{
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
  }};

  const getSafetensorParts = async () => {{
    let parts = [];

    {safetensor_parts}

    return parts;
  }};

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
  }};

  {kernel_code}
        
  const setupNet = async (device, safetensor) => {{
      const metadata = getTensorMetadata(safetensor[0]);

      {bufs}

      const gpuWriteBuffer = device.createBuffer({{size:input.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});
      const gpuReadBuffer = device.createBuffer({{ size: outputs.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

      const kernels = [{kernel_names}];
      const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

      return async (data) => {{
          await gpuWriteBuffer.mapAsync(GPUMapMode.WRITE);
          new Float32Array(gpuWriteBuffer.getMappedRange()).set(data);
          gpuWriteBuffer.unmap();

          const commandEncoder = device.createCommandEncoder();
          commandEncoder.copyBufferToBuffer(gpuWriteBuffer, 0, input, 0, gpuWriteBuffer.size);
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
  """

  with open(path.join(path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prg)
