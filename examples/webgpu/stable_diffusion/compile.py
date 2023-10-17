import os
from extra.export_model import compile_net, jit_model
from examples.stable_diffusion import StableDiffusion
from tinygrad.nn.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from extra.utils import download_file
from typing import NamedTuple, Any, List
from pathlib import Path

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

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  sub_steps = [
    Step(name = "textModel", input = [Tensor.randn(1, 77)], forward = model.cond_stage_model.transformer.text_model), 
    Step(name = "diffusor", input = [Tensor.randn(1, 77, 768), Tensor.randn(1, 77, 768), Tensor.randn(1,4,64,64), Tensor.rand(1), Tensor.randn(1), Tensor.randn(1), Tensor.randn(1)], forward = model),
    Step(name = "decoder", input = [Tensor.randn(1,4,64,64)], forward = model.decode)
  ]
  
  prg = ""

  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, _ = compile_net(run, special_names)
    state = get_state_dict(model)
    weights = {id(x.lazydata.realized): name for name, x in state.items()}

    kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
    kernel_names = ', '.join([name for (name, _, _, _) in statements])
    kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
    bufs =  '\n    '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weights else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weights[_key]}'], '{weights[_key]}'))") + ";"  for name,(size,dtype,_key) in bufs.items()])
    gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if value != "outputs"])
    input_writer = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n    new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n    gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "outputs"])
    return f"""\n    var {step.name} = function() {{
    
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
            const resultBuffer = new Float32Array(gpuReadBuffer.size/4);
            resultBuffer.set(new Float32Array(gpuReadBuffer.getMappedRange()));
            gpuReadBuffer.unmap();
            return resultBuffer;
        }}
      }} 
    }}
  }}
  """

  for step in sub_steps:
    print(f'Executing step={step.name}')
    prg += compile_step(model, step)

    if step.name == "diffusor":
      state = get_state_dict(model)
      safe_save(state, os.path.join(os.path.dirname(__file__), "net.safetensors"))
      part_start_offsets = create_multipart_safetensor("./net.safetensors", 1073741824)
      os.remove("net.safetensors") #don't need the original after splitting
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

    let allZero = true;
    let out = safetensorParts[selectedPart].subarray(...correctedOffsets);

    for (let i = 0; i < out.length; i++) {{
        if (out[i] !== 0) {{
            allZero = false;
            break;
        }}
    }}

    if (allZero) {{
        console.log("Error: weight '" + key + "' is all zero.");
    }}

    return safetensorParts[selectedPart].subarray(...correctedOffsets);
  }}

  const getWeight = (safetensors, key) => {{
    let uint8Data = getTensorBuffer(safetensors, getTensorMetadata(safetensors[0])[key], key);
    return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, uint8Data.byteLength / Float32Array.BYTES_PER_ELEMENT);
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

  with open(os.path.join(os.path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prekernel + prg)
