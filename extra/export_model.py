from typing import Tuple, Dict, List
from tinygrad.helpers import DType
from tinygrad.tensor import Device, Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import get_state_dict
import json

def compile_net(run:TinyJit, special_names:Dict[int,str]) -> Tuple[Dict[str,str],List[Tuple[str,List[str],List[int]]],Dict[str,Tuple[int,DType,int]],Dict[str,Tensor]]:
  functions, bufs, bufs_to_save, statements, bufnum = {}, {}, {}, [], 0
  for fxn,args,var_vals in run.jit_cache:
    assert not var_vals, "symbolic shape is not supported"
    functions[fxn.name] = fxn.prg   # NOTE: this assumes all with the same name are the same
    cargs = []
    for i,arg in enumerate(args):
      key = id(arg)
      if key not in bufs:
        if key in special_names:
          bufs[key] = (special_names[key], arg._memsz, arg.dtype, key)
        else:
          bufs[key] = (f"buf_{bufnum}", arg._memsz, arg.dtype, key)
          bufnum += 1
          if i > 0: bufs_to_save[bufs[key][0]] = arg   # if first usage of a buffer is not an output, and it's not a special name
      cargs.append(bufs[key][0])
    statements.append((fxn.name, cargs, fxn.global_size, fxn.local_size))

  return functions, statements, {name:(size, dtype, key) for (name,size,dtype,key) in bufs.values()}, bufs_to_save

def jit_model(model, the_input:Tensor) -> Tuple[TinyJit,Dict[int,str]]:
  assert hasattr(model, "forward") or callable(model), "model needs a forward function"
  @TinyJit
  def run(x): return (model.forward(x) if hasattr(model, "forward") else model(x)).realize()

  # twice to run the JIT
  for _ in range(2): the_output = run(the_input)

  # hack to put the inputs back
  assert len(run.input_replace) == 1, f"didn't get one input to replace {run.input_replace}"
  for (j,i),idx in run.input_replace.items():
    run.jit_cache[j][1][i] = the_input.lazydata.realized

  # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  special_names = {id(the_input.lazydata.realized): "input", id(the_output.lazydata.realized): "outputs"}
  return run, special_names

def export_model_clang(functions:Dict[str,str], statements:Dict[str,Tuple[str,int,int]], bufs:Dict[str,Tuple[str,int,int]], bufs_to_save:Dict[str,Tensor]) -> str:
  from tinygrad.runtime.ops_clang import CLANG_PROGRAM_HEADER
  cprog = [CLANG_PROGRAM_HEADER]

  for name,cl in bufs_to_save.items():
    weight = ''.join(["\\x%02X"%x for x in bytes(cl._buf)])
    cprog.append(f"unsigned char {name}_data[] = \"{weight}\";")

  cprog += [f"float {name}[{len}];" if name not in bufs_to_save else f"float *{name} = (float *){name}_data;" for name,(len,dtype,_key) in bufs.items() if name not in ['input', 'outputs']]
  cprog += list(functions.values())
  cprog += ["void net(float* input, float* outputs) {"] + [f"{name}({', '.join(args)});" for (name, args, _global_size, _local_size) in statements] + ["}"]
  return '\n'.join(cprog)

def export_model_webgpu(functions, statements, bufs, bufs_to_save, weight_names) -> Tuple[str,int,int]:
  kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
  kernel_names = ', '.join([name for (name, _args, _global_size, _local_size) in statements])
  kernel_calls = '\n    '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
  _bufs =  '\n    '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weight_names else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weight_names[_key]}']))") + ";"  for name,(size,dtype,_key) in bufs.items()])
  return f"""
const getTensorMetadata = (safetensorBuffer) => {{
  const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
  const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
  return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
}};

const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {{
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
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
    const metadata = getTensorMetadata(safetensor);

    {_bufs}

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
  """ + f"\n\nconst loadNet = async (device) => {{ return await fetch('net.safetensors').then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }}"

def export_model(model, input:Tensor, target:str):
  assert Device.DEFAULT in ["WEBGPU", "CLANG", "CUDA", "GPU", "METAL"], "only WEBGPU, CLANG, CUDA, GPU, METAL are supported"
  run,special_names = jit_model(model, input)
  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
  state = get_state_dict(model)
  weight_names = {id(x.lazydata.realized): name for name, x in state.items()}
  prg = ""
  if target == "clang":
    prg = export_model_clang(functions, statements, bufs, bufs_to_save)
  elif target == "webgpu":
    prg = export_model_webgpu(functions, statements, bufs, bufs_to_save, weight_names)
  else:
    prg = json.dumps({
      "backend": Device.DEFAULT,
      "input": {
        "size": bufs['input'][0],
        "dtype": bufs['input'][1].name
      },
      "output": {
        "size": bufs["outputs"][0],
        "dtype": bufs["outputs"][1].name
      },
      "functions": functions,
      "statements": [{
        "kernel": kernel,
        "args": args,
        "global_size": global_size,
        "local_size": local_size
      } for (kernel, args, global_size, local_size) in statements],
      "buffers": {
        name: {
          "size": size,
          "dtype": dtype.name,
          "id": weight_names[_key] if _key in weight_names else ""
        } for name, (size,dtype,_key) in bufs.items() if name not in ["input", "outputs"]
      }
    })

  return prg, bufs['input'][0], bufs['outputs'][0], state