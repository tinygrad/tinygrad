from os import path
from examples.compile_efficientnet import compile_net, jit_efficientnet

if __name__ == "__main__":
  run, special_names = jit_efficientnet()
  functions, statements, bufs, bufs_to_save = compile_net(run, special_names, lambda name, cargs, global_size: {'kernel': name, 'args': cargs, 'global_size': global_size})

  kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
  weight_values = '\n'.join([f'const data_{key} = "' + (''.join(["\\x%02X"%x for x in buf.toCPU().tobytes()])) + '";' for key, buf in bufs_to_save.items() ])
  kernel_calls = '\n  '.join([f"addComputePass(device, commandEncoder, {statement['kernel']}, [{', '.join(statement['args'])}], {statement['global_size']});" for statement in statements ])
  bufs =  '\n  '.join([f"const {buf[0]} = createEmptyBuf(device, {buf[1]});" if buf[0] not in bufs_to_save and buf[0] != 'input' else f"const {buf[0]} = createWeightBuf(device, {buf[1]}, {'inputData' if buf[0] == 'input' else f'str2buf(data_{buf[0]})'});" for buf in bufs.values() ])

  prg = f"""
const str2buf = (str) => new Float32Array(Uint8Array.from(str, c => c.charCodeAt(0)).buffer);

const createEmptyBuf = (device, size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }});
}};

const createWeightBuf = (device, size, dataBuf) => {{
  const buf = device.createBuffer({{ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE }});
  new Float32Array(buf.getMappedRange()).set(dataBuf);
  buf.unmap();
  return buf;
}};

const addComputePass = (device, commandEncoder, code, bufs, workgroup) => {{
  const computePipeline = device.createComputePipeline({{layout: "auto", compute: {{ module: device.createShaderModule({{ code }}), entryPoint: "main" }}}});
  const bindGroup = device.createBindGroup({{layout: computePipeline.getBindGroupLayout(0), entries: bufs.map((buffer, index) => ({{ binding: index, resource: {{ buffer }} }}))}});
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
}};

{kernel_code}

{weight_values}
      
const net = async (device, inputData) => {{
    const commandEncoder = device.createCommandEncoder();

    {bufs}

    {kernel_calls}

    const gpuReadBuffer = device.createBuffer({{ size: outputs.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});
    
    commandEncoder.copyBufferToBuffer(outputs, 0, gpuReadBuffer, 0, outputs.size);
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    return gpuReadBuffer.getMappedRange();
}}
"""

  with open(path.join(path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prg)
