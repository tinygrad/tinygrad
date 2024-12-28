# based on ./examples/webgpu/stable_diffusion/compile.py

import os, sys
from extra.export_model import compile_net, jit_model, dtype_to_js_type
from examples.llama3 import build_transformer, Tokenizer
from tinygrad.nn.state import get_state_dict, safe_save, load_state_dict, safe_load, safe_load_metadata
from tinygrad.tensor import Tensor
from tinygrad import Device, GlobalCounters
from tinygrad.helpers import fetch
from typing import NamedTuple, Any, List

def split_safetensor(fn):
  _, data_start, metadata = safe_load_metadata(fn)
  chunk_size = 536870912
  last_offset = 0
  part_end_offsets = []

  for k in metadata:
    offset = metadata[k]['data_offsets'][0]
    part_offset = offset - last_offset

    if (part_offset >= chunk_size):
      part_end_offsets.append(data_start+offset)
      last_offset = offset

  net_bytes = bytes(open(fn, 'rb').read())
  part_end_offsets.append(len(net_bytes))
  cur_pos = 0

  for i, end_pos in enumerate(part_end_offsets):
    with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.safetensors'), "wb+") as f:
      f.write(net_bytes[cur_pos:end_pos])
      cur_pos = end_pos

  return part_end_offsets

if __name__=="__main__":
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  f32_fn = os.path.join(os.path.dirname(__file__), "llama3_1B_f32.safetensors")
  max_context=1024

  if not os.path.exists(f32_fn):
    # this is ugly, but wgpu adapter doesn't support f16 (they're working on it), throws exception on loading llama3 1B weights
    # the tinygrad llama code just converts the f16 to f32 anyway, we let that happen, then transfer the weights to WEBGPU device
    # TODO clean this up when wgpu supports f16, or maybe use dawn if it supports f16 (cc wpmed92)
    model = build_transformer(model_path, model_size=model_size, max_context=max_context)
    state_dict = get_state_dict(model)
    safe_save(state_dict, f32_fn)
    print(f"f32 weights saved to {f32_fn}, exiting to free idle GPU memory, restart program as-is to resume")
    # TODO force free all the currently used GPU memory after loading state_dict into the WEBGPU-initialized model
    #  this doesn't happen by default below, so we restart execution to clear ~3GB of GPU memory
    #  maybe extra.models.llama.Transformer.forward_jit is the culprit?
    sys.exit()

  Device.DEFAULT = "WEBGPU"
  device = Device.DEFAULT
  model = build_transformer(model_path, model_size=model_size, max_context=max_context, load_weights=False)
  state_dict = safe_load(f32_fn)
  load_state_dict(model, state_dict, consume=True)

  # For now, we initialize the tinygrad.extra.models.llama.Transformer with the first 3 tokens that the model sees in every first tinychat user prompt
  # Then in the start of tinychat, we will skip those 3 tokens
  # TODO: remove this complexity by getting a jit-compiled llama3 transformer with kv-cache ready but empty (or something like that)
  tokenizer_path = fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
  tokenizer = Tokenizer(str(tokenizer_path))
  # TODO: refactor to consolidate these encode functions with those in examples/llama3.py
  def encode_role(role: str):
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
  def encode_message(role: str, content: str):
    return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]
  toks = [tokenizer.bos_id] + encode_message("user", "hi")

  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0
  GlobalCounters.reset()
  model.forward(Tensor([[toks[0]]], device=device), 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  sub_steps = [
    Step(
      name = "transformer", 
      input = [
        Tensor([[toks[1]]], device=device), 1, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P,
        Tensor([[toks[2]]], device=device), 2, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P,
      ], 
      forward = model.forward),
  ]

  prg = ""

  def fixup_code(code, key):
    code = code.replace(key, 'main')\
      .replace("var<uniform> INFINITY : f32;\n", "fn inf(a: f32) -> f32 { return a/0.0; }\n")\
      .replace("@group(0) @binding(0)", "")\
      .replace("INFINITY", "inf(1.0)")

    for i in range(1,9): code = code.replace(f"binding({i})", f"binding({i-1})")
    return code

  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input, two_argsets_provided=True)
    functions, statements, bufs, _ = compile_net(run, special_names)
    state = get_state_dict(model)
    weights = {id(x.lazydata.base.realized): name for name, x in state.items()}
    kernel_code = '\n\n'.join([f"const {key} = `{fixup_code(code, key)}`;" for key, code in functions.items()])
    kernel_names = ', '.join([name for (name, _, _, _) in statements])
    input_names = [name for _,name in special_names.items() if "input" in name]
    output_names = [name for _,name in special_names.items() if "output" in name]
    input_buf_types = [dtype_to_js_type(bufs[inp_name][1]) for inp_name in input_names]
    output_buf_types = [dtype_to_js_type(bufs[out_name][1]) for out_name in output_names]
    kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
    exported_bufs =  '\n    '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weights else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weights[_key]}'], '{weights[_key]}'))") + ";"  for name,(size,dtype,_key) in bufs.items()])
    gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if "output" not in value])
    input_writer = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n    new {input_buf_types[i]}(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n    gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,_ in enumerate(input_names)])
    return f"""\n    var {step.name} = function() {{

    {kernel_code}

    return {{
      "setup": async (device, safetensor) => {{
        const metadata = safetensor ? getTensorMetadata(safetensor[0]) : null;

        {exported_bufs}

        {gpu_write_bufs}
        const gpuReadBuffer = device.createBuffer({{ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

        const kernels = [{kernel_names}];
        const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

        return async ({",".join([f'data{i}' for i,(k,v) in enumerate(special_names.items()) if v != "output0"])}) => {{
            const commandEncoder = device.createCommandEncoder();

            {input_writer}

            {kernel_calls}
            commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer, 0, output0.size);
            const gpuCommands = commandEncoder.finish();
            device.queue.submit([gpuCommands]);

            await gpuReadBuffer.mapAsync(GPUMapMode.READ);
            const resultBuffer = new {output_buf_types[0]}(gpuReadBuffer.size/{bufs[output_names[0]][1].itemsize});
            resultBuffer.set(new {output_buf_types[0]}(gpuReadBuffer.getMappedRange()));
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
    base_url="."

  # Since last safe_save, we added cache_kv, tok_embeddings.arange, maybe more
  # TODO: remove all the safe_save calls, and streamline original quantized weights --> partStartOffsets
  state_dict = get_state_dict(model)
  safe_save(state_dict, f32_fn)
  partStartOffsets = split_safetensor(f32_fn)
  #os.remove(f32_fn)

  prekernel = f"""
    window.MODEL_BASE_URL= "{base_url}";
    const getTensorMetadata = (safetensorBuffer) => {{
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
    }};

  const getTensorBuffer = (safetensorParts, tensorMetadata, key) => {{
    let selectedPart = 0;
    let counter = 0;
    let partStartOffsets = [{", ".join(str(i) for i in partStartOffsets)}];
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
