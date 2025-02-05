# based on ./examples/webgpu/stable_diffusion/compile.py
# TODO: merge with compile_clang.py

import os, sys, json, hashlib, math
from functools import partial
from extra.export_model import compile_net, jit_model, dtype_to_js_type, export_model, export_model_clang
from extra.f16_decompress import u32_to_f16
from examples.llama3 import build_transformer, Tokenizer, Int8Linear
from tinygrad.nn.state import get_state_dict, load_state_dict
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad import Device, GlobalCounters, Variable
from tinygrad.helpers import fetch, prod
from typing import NamedTuple, Any, List, Literal
from tiktoken.load import load_tiktoken_bpe, dump_tiktoken_bpe
from tinygrad.helpers import flatten
from tinygrad.ops import Ops
from collections import OrderedDict

# from nn.state.ggml_data_to_tensor
def q_to_uint8(t: Tensor, b: int) -> Tensor:
  # TODO: rewrite with arange?
  shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
  return t.unsqueeze(-1).expand((*t.shape,8//b)).idiv(shift_tensor).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

# adapted from nn.state.ggml_data_to_tensor
# uint8 (256 elements per 210 bytes) to float32
def q6k_to_f32(x: Tensor, target: Literal["WEBGPU", "CLANG"] = "WEBGPU") -> Tensor:
  blocks = x.reshape((-1, 210))
  xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
  scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
  if target == "WEBGPU":
    # hack to avoid fp16 which isn't supported by wgpu adapter
    d = u32_to_f16(blocks[:,-2:].cat(Tensor.zeros(blocks.shape[0], 2, dtype=dtypes.uint8), dim=1).bitcast(dtypes.uint32)).reshape(-1, 2)[:,0:1].expand((-1,256))
  elif target == "CLANG":
    d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
  return (d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales).cast(dtypes.float32).flatten()

def clang_export_q6k_to_f32():
  # won't work currently due to compile_net exceptions
  # works with commit 254ea814259465bffd157b10094a32e412ca8860
  Device.DEFAULT="CLANG"
  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  step = Step(
    name = "q6k_to_f32", 
    input = [Tensor.randn(430_080, dtype=dtypes.uint8).realize()], # llama-3.2-1B Q6_K weights are multiples of 430_080 bytes
    forward = partial(q6k_to_f32, target="CLANG"),
  )
  
  run, special_names = jit_model(step, *step.input)

  # hack the jit code to make clang export work
  # TODO: fix the export_model.compile_net code, which doesn't work by default here
  #   export_model doesn't work by default because there are ViewOp ExecItems (for casting) that don't have code,
  #   where extra.export_model.compile_net is looking for it
  Tensor.realize(*step.input)
  lbs = flatten([t.lazydata.lbs for t in step.input])
  input_buffers = [lb.base.realized for lb in lbs if lb.base.realized is not None]
  out = run.captured(input_buffers, {}, clear_inputs=False)
  functions, statements, bufs, bufs_to_save = compile_net(run.captured, special_names)
  input_names = [name for _,name in special_names.items() if "input" in name]
  output_names = [name for _,name in special_names.items() if "output" in name]
  prg = export_model_clang(functions, statements, bufs, bufs_to_save, input_names, output_names)

  with open(os.path.join(os.path.dirname(__file__), "q6k_to_f32.c"), "w") as text_file:
    text_file.write(prg)
    # we still need to manually edit this file: 
    #   manually fixed type declarations, deleted buf_6 and buf_7, cast buf_0 to buf_6, cast buf_4 to buf_7

def prepare_browser_chunks(model):
  # split weights into browser-friendly chunks
  state_dict = get_state_dict(model)
  del state_dict['output.weight'] # same as token_embeddings.weight
  del state_dict['output.scale'] # same as token_embeddings.scale
  chunk_size = 47 * 1024 * 1024 # small chunks based on iphone browser constraints
  metadata = {}
  t_infos = [(v.lazydata.base.realized.nbytes, k, v.dtype) for k,v in state_dict.items() if "cache_kv" not in k]
  empty_t_infos = [(v.lazydata.base.realized.nbytes, k, v.dtype) for k,v in state_dict.items() if "cache_kv" in k]

  split_t_infos = []
  for size, name, dtype in t_infos:
    if size <= chunk_size:
      split_t_infos.append((size, name, dtype, ()))
    else:
      for i in range(0, size, chunk_size):
        split_t_infos.append((min(chunk_size, size-i), f"{name}_part{math.ceil(i/chunk_size)}", dtype, (i, min(i+chunk_size, size))))

  files = []
  # FFD bin packing
  split_t_infos = sorted(split_t_infos, reverse=True)
  for info in split_t_infos:
      placed = False
      for file in files:
        if sum(i[0] for i in file) + info[0] <= chunk_size:
          file.append(info)
          placed = True
          break
      if not placed:
        files.append([info])

  tinygrad_dtypes = {dtypes.float32: "float32", dtypes.float16: "float16", dtypes.int8: "int8", dtypes.int32: "int32"} # TODO: still necessary?
  for i, file in enumerate(files):
    cursor = 0
    with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.chunk'), "wb+") as writer:
      for size, name, dtype, offsets in file:
        name, part_num = (name, 0) if "_part" not in name else (name.split("_part")[0], int(name.split("_part")[1]))
        default = {"parts": {}, "dtype": tinygrad_dtypes[dtype]}
        weight_metadata = metadata.get(name, default)
        weight_metadata["parts"][part_num] = {"file": i, "file_start_pos": cursor, "size": size}
        metadata[name] = weight_metadata
        data = bytes(state_dict[name].lazydata.base.realized.as_buffer())
        data = data if not offsets else data[offsets[0]:offsets[1]]
        writer.write(data)
        cursor += size

  metadata.update({name: {"parts": {0: {"empty": True, "size": size}}, "dtype": tinygrad_dtypes[dtype]} for size, name, dtype in empty_t_infos})

  for k in metadata:
    metadata[k]["parts"] = [part for part_num, part in sorted(metadata[k]["parts"].items(), key = lambda x: x[0])]
    cursor = 0
    for i, part in enumerate(metadata[k]["parts"]):
      metadata[k]["parts"][i]["target_start_pos"] = cursor
      cursor += part["size"]

  # compute hashes, which client app will check to determine whether to update with new weights and/or detect integrity issues
  state_dict_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
  metadata = {"state_dict": metadata, "state_dict_hash": state_dict_hash, "files": []}
  for i in range(len(files)):
    with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.chunk'), "rb") as reader:
      metadata["files"].append({"name": f'net_part{i}.chunk', "hash": hashlib.sha256(reader.read()).hexdigest()})
  metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
  metadata = {"metadata": metadata, "metadata_hash": metadata_hash}

  with open(os.path.join(os.path.dirname(__file__), f'./net_metadata.json'), "w") as writer: json.dump(metadata, writer, indent=4)
  return metadata

if __name__=="__main__":
  default_device = Device.DEFAULT
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf", "Llama-3.2-1B-Instruct-f16.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  f32_fn = os.path.join(os.path.dirname(__file__), "llama3_1B_f32.safetensors")
  max_context=1024

  Device.DEFAULT="CLANG" # may be necessary for quantization to work, from float16 original weights
  model = build_transformer(model_path, model_size=model_size, quantize="int8", device=Device.DEFAULT, max_context=1024)
  state_dict = get_state_dict(model)
  state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
  state_dict["output.scale"] = state_dict["tok_embeddings.scale"]
  Device.DEFAULT="WEBGPU"
  model = build_transformer(model_path, model_size=model_size, quantize="int8", device=Device.DEFAULT, max_context=1024, load_weights=False)
  load_state_dict(model, state_dict, consume=True)

  tokenizer_path = fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
  tokenizer = Tokenizer(str(tokenizer_path))
  # TODO: refactor to consolidate these encode functions with those in examples/llama3.py
  def encode_role(role: str):
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
  def encode_message(role: str, content: str):
    return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]
  toks = [tokenizer.bos_id] + encode_message("user", "hi")
  toks = toks + encode_role("assistant")

  # Export BPE data for use with tiktoken.js
  mergeable_ranks = load_tiktoken_bpe(str(tokenizer_path))  
  bpe_path = os.path.join(os.path.dirname(__file__), "llama3-2.tiktoken")
  dump_tiktoken_bpe(mergeable_ranks, bpe_path)

  # TODO: make these variables tunable by client?
  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0
  GlobalCounters.reset()

  # initialize stuff not included in downloaded weights: kv cache, freqs_cis, tok_embeddings.arange
  tok = 128000
  out = model.forward(Tensor([[tok]]), 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)

  # We waited to prepare the chunks until here, because model.freqs_cis and model.tok_embeddings.arange are only ready now
  metadata = prepare_browser_chunks(model)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None
    show_progress: bool = False

  start_pos = Variable("start_pos", 0, max_context).bind(0)
  sub_steps = [Step(name = "transformer", input = [Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P], forward = model.forward, show_progress=True),]

  prg = ""

  def fixup_code(code, key):
    code = code.replace(key, 'main')\
      .replace("var<uniform> INFINITY : f32;\n", "fn inf(a: f32) -> f32 { return a/0.0; }\n")\
      .replace("@group(0) @binding(0)", "")\
      .replace("INFINITY", "inf(1.0)")

    for i in range(1,9): code = code.replace(f"binding({i})", f"binding({i-1})")
    return code

  # TODO: refactor to merge with compile_clang.py
  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, _ = compile_net(run, special_names)
    state = get_state_dict(model)
    weights = {id(x.lazydata.base.realized): name for name, x in state.items()}
    kernel_code = '\n\n'.join([f"const {key} = `{fixup_code(code, key)}`;" for key, code in functions.items()])
    kernel_names = ', '.join([name for (name, _, _, _) in statements])
    input_names = [name for _,name in special_names.items() if "input" in name]
    output_names = [name for _,name in special_names.items() if "output" in name]
    input_buf_types = [dtype_to_js_type(bufs[inp_name][1]) for inp_name in input_names]
    output_buf_types = [dtype_to_js_type(bufs[out_name][1]) for out_name in output_names]

    # TODO: fix some of this stuff upstream
    symbolic_vars = OrderedDict()
    next_input_idx = max(int(name.split("input")[1]) for name in special_names.values() if "input" in name) + 1
    for i, (_, args, global_size, _) in enumerate(statements):
      for j, var in enumerate(args):
        if getattr(var, "op", None) is Ops.DEFINE_VAR and isinstance(getattr(var, "arg", None), tuple) and isinstance(var.arg[0], str):
          if var not in symbolic_vars:
            symbolic_vars[var] = f"input{next_input_idx}"
            next_input_idx += 1
            input_names.append(var.arg[0])
            input_buf_types.append(dtype_to_js_type(var.dtype))
            special_names[var.arg[0]] = symbolic_vars[var]
            bufs[symbolic_vars[var]] = (var.dtype.itemsize, var.dtype, var.arg[0])
          statements[i][1][j] = symbolic_vars[var]
      
      for j, dim in enumerate(global_size):
        if getattr(dim, "op", None) is Ops.ADD and len(dim.src) == 2:
          if {dim.src[0].op, dim.src[1].op} == {Ops.DEFINE_VAR, Ops.CONST}:
            name, val = dim.src if dim.src[1].op is Ops.CONST else reversed(dim.src)
            name, val = name.arg[0], val.arg
            # TODO: use something less tedious than repeated enumeration for input order canonicalization
            input_idx = list(i for i, (k,v) in enumerate((k,v) for (k,v) in special_names.items() if "output" not in v) if v == special_names[name])[0]
            global_size[j] = f"data{input_idx}[0] + {val}"

    kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], [{', '.join(str(x) for x in global_size)}]);" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
    # TODO: don't duplicate output.weight and tok_embeddings.weight
    buf_type = lambda x: "createUniformBuf" if x in set(uop.arg[0] for uop in symbolic_vars) else "createEmptyBuf"
    exported_bufs, prog, buf_end_prog = [], 40, 80
    for i, (name,(size,dtype,_key)) in enumerate(bufs.items()):
      if i > 1 and i % 20 == 1:
        exported_bufs.append(f"await new Promise(resolve => setTimeout(resolve, 0));") # prevent browser lag
        if step.show_progress: exported_bufs.append(f"progress({prog + int((buf_end_prog - prog) * i / len(bufs))}, 100, 'Launching WebGPU model:');")
      exported_bufs.append(f"const {name} = " + (f"{buf_type(_key)}(device, {size});" if _key not in weights else (f"createWeightBuf(device, {size}, state_dict['{weights[_key]}'])" if "cache_kv" not in weights[_key] else f"createEmptyBuf(device, {size})") + ";"))
    exported_bufs = '\n   '.join(exported_bufs)
    gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate((k,v) for (k,v) in special_names.items() if "output" not in v)])
    input_writer = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n    new {input_buf_types[i]}(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n    gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,_ in enumerate(input_names)])
    pipelines = f"""
        const piplines = [];
        for (let i=0; i<kernels.length; i++) {{
          const name = kernels[i];
          const pipeline = await device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}});
          piplines.push(pipeline);
          if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
          if (i / kernels.length > 0.33) {{progress({buf_end_prog + int((100-buf_end_prog)/3)}, 100, "Launching WebGPU model:");}}
          if (i / kernels.length > 0.66) {{progress({buf_end_prog + int((100-buf_end_prog)*2/3)}, 100, "Launching WebGPU model:");}}
        }}""" if step.show_progress else f'const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));'
    return f"""\n    var {step.name} = function() {{

    {kernel_code}

    return {{
      "setup": async (device{", state_dict" if state else ""}{f", progress" if step.show_progress else ""}) => {{

        {exported_bufs}

        {gpu_write_bufs}
        const gpuReadBuffer = device.createBuffer({{ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

        const kernels = [{kernel_names}];
        {pipelines}

        return async ({",".join([f'data{i}' for i,(k,v) in enumerate((k,v) for (k,v) in special_names.items() if "output" not in v)])}) => {{
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

  partStartOffsets=[]

  prekernel = f"""
  const createEmptyBuf = (device, size) => {{
      return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
  }};
  const createUniformBuf = (device, size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}})
  }}

  const createWeightBuf = (device, size, data) => {{
    const buf = device.createBuffer({{ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE }});
    data.bytes = buf;
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
