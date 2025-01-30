# based on ./examples/webgpu/stable_diffusion/compile.py
# TODO: merge with compile_clang.py

import os, sys, json, hashlib
from functools import partial
from extra.export_model import compile_net, jit_model, dtype_to_js_type, export_model, export_model_clang
from extra.models.llama import convert_from_gguf
from extra.f16_decompress import u32_to_f16
from examples.llama3 import build_transformer, Tokenizer, Int8Linear
from tinygrad.nn.state import get_state_dict, safe_save, load_state_dict, safe_load, safe_load_metadata, gguf_load
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

def prepare_browser_gguf_chunks(model_path, model):
  # split gguf file into browser-friendly chunks
  # export metadata JSON with offsets -- depends on tinygrad gguf metadata parser below
  chunk_size = 2**29 # 536_870_912
  metadata = {}

  gguf_tensor = Tensor.empty(os.stat(model_path).st_size, dtype=dtypes.uint8, device=f"disk:{model_path}").to(Device.DEFAULT)
  kv_data, state_dict, t_infos, data_start = gguf_load(gguf_tensor)

  # calculate byte size of each tensor
  for i, info in enumerate(t_infos[:-1]):
    size = t_infos[i+1][3] - t_infos[i][3]
    t_infos[i] = (size, info)
  assert (last_info_dtype:=t_infos[-1][2]) in (0, 14)
  last_info_size = {0: 4, 14: 8*210/256}.get(last_info_dtype) * prod(t_infos[-1][1])
  t_infos[-1] = (last_info_size, t_infos[-1])

  prerun_model = build_transformer(model_path, model_size=model_size, max_context=max_context, load_weights=False)
  # for llama, some model parameters used in inference are instantiated at runtime
  new_state_dict = get_state_dict(model)
  new_weights = set(new_state_dict.keys()) - set(convert_from_gguf(state_dict, prerun_model).keys())
  new_weights = {name: new_state_dict[name].contiguous().to("CLANG").realize() for name in new_weights}
  for k,v in new_weights.items():
    next_start_pos = t_infos[-1][1][3] + t_infos[-1][0]
    t_infos.append((v.lazydata.buffer.nbytes, (k, "not_from_gguf_file", {dtypes.float: 0, dtypes.int: 18}[v.dtype], next_start_pos)))

  chunks = []
  # FFD bin packing
  t_infos = sorted(t_infos, reverse=True)
  for info in t_infos:
      placed = False
      for chunk in chunks:
        if sum(i[0] for i in chunk) + info[0] <= chunk_size:
          chunk.append(info)
          placed = True
          break
      if not placed:
        chunks.append([info])

  gguf_dtypes = {0: "float32", 14: "Q6_K", 18: "int32"}
  new_weights = {k: {"tensor": v} for k,v in new_weights.items()}
  with open(model_path, 'rb') as reader:
    for i, chunk in enumerate(chunks):
      cursor = 0
      with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.gguf.chunk'), "wb+") as writer:
        for size, info in chunk:
          weight_metadata = {"chunk": i, "start_pos": cursor, "size": size, "dtype": gguf_dtypes[info[2]], "gguf_shape": info[1]}
          if (name:=info[0]) not in new_weights:
            reader.seek(data_start + info[3])
            data = reader.read(size)
            metadata[name] = weight_metadata
          elif name in new_weights:
            data = bytes(new_weights[name]["tensor"].lazydata.buffer.as_buffer())
            new_weights[name]["metadata"] = weight_metadata
          writer.write(data)
          cursor += size

  metadata = convert_from_gguf(metadata, model)
  for k,v in new_weights.items():
    metadata[k] = v["metadata"]

  # compute hashes, which client app will check to determine whether to update with new weights and/or detect integrity issues
  state_dict_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
  metadata = {"state_dict": metadata, "state_dict_hash": state_dict_hash, "chunks": []}
  for i in range(len(chunks)):
    with open(os.path.join(os.path.dirname(__file__), f'./net_part{i}.gguf.chunk'), "rb") as reader:
      metadata["chunks"].append({"name": f'net_part{i}.gguf.chunk', "hash": hashlib.sha256(reader.read()).hexdigest()})
  metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
  metadata = {"metadata": metadata, "metadata_hash": metadata_hash}

  with open(os.path.join(os.path.dirname(__file__), f'./net_metadata.json'), "w") as writer: json.dump(metadata, writer, indent=4)
  return metadata

if __name__=="__main__":
  default_device = Device.DEFAULT
  model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
  model_size="1B"
  Tensor.no_grad = True
  f32_fn = os.path.join(os.path.dirname(__file__), "llama3_1B_f32.safetensors")
  max_context=1024

  #clang_export_q6k_to_f32()
  #metadata = prepare_browser_gguf_chunks(str(model_path))

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

  Device.DEFAULT = "WEBGPU"
  model = build_transformer(model_path, model_size=model_size, quantize="int8", max_context=max_context, load_weights=False)
  state_dict = safe_load(f32_fn)
  for k,v in state_dict.items(): v.replace(v.to("WEBGPU").realize()) # will get "needs a renderer" exception if we don't do this
  state_dict = Int8Linear.quantize(state_dict, "WEBGPU")
  load_state_dict(model, state_dict, consume=True)

  # TODO: make these variables tunable by client?
  TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P = 0.95, 0, 0.0, 0.0, 0.0
  GlobalCounters.reset()

  # initialize stuff not included in downloaded weights: kv cache, freqs_cis, tok_embeddings.arange
  tok = 128000
  out = model.forward(Tensor([[tok]]), 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)

  # We waited to prepare the chunks until here, because model.freqs_cis and model.tok_embeddings.arange are only ready now
  metadata = prepare_browser_gguf_chunks(str(model_path), model)
  os.remove(f32_fn)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  start_pos = Variable("start_pos", 0, max_context).bind(0)
  sub_steps = [
    Step(name = "transformer", input = [Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P], forward = model.forward),
    # TODO: output transformer pipeline doesn't work unless we manually fix the start_pos instantiation and usage throughout entire pipeline
    # TODO: Fix compiler to not have above issue
    Step(name = "q6k_to_f32", input = [Tensor.randn(3_144_960, dtype=dtypes.uint8)], forward = q6k_to_f32), # throws exceptions with all larger sizes tried
    # TODO: output q6k_to_f32 pipeline doesn't work unless we manually fix the uint32 buffer size allocation
    # TODO: Fix compiler to not have above issue
  ]

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
    exported_bufs =  '\n    '.join([f"const {name} = " + (f"{buf_type(_key)}(device, {size});" if _key not in weights else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weights[_key]}']))") + ";"  for name,(size,dtype,_key) in bufs.items()])
    gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate((k,v) for (k,v) in special_names.items() if "output" not in v)])
    input_writer = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n    new {input_buf_types[i]}(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n    gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,_ in enumerate(input_names)])
    return f"""\n    var {step.name} = function() {{

    {kernel_code}

    return {{
      "setup": async (device, safetensor, metadata) => {{

        {exported_bufs}

        {gpu_write_bufs}
        const gpuReadBuffer = device.createBuffer({{ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

        const kernels = [{kernel_names}];
        const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

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
  const getTensorBuffer = (safetensorParts, t) => {{return safetensorParts[t.chunk].subarray(t.start_pos, t.start_pos + t.size)}}

  const createEmptyBuf = (device, size) => {{
      return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
  }};
  const createUniformBuf = (device, size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}})
  }}

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
