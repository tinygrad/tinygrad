if (!navigator.gpu) throw new Error("WebGPU not supported.");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
	requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [], powerPreference: "high-performance",
  requiredLimits: {maxStorageBufferBindingSize: 262668288, maxBufferSize: 262668288,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup},
});

const test_wgsl = (() => {
  const createEmptyBuf = (size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  };
  const addComputePass = (commandEncoder, pipeline, bufs, workgroup) => {
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bufs.map((buffer, index) => ({ binding: index, resource: { buffer } }))
    });
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...workgroup);
    passEncoder.end();
  };

  // adapted from https://gist.github.com/wpmed92/c045e98fdb5916670c31383096706406
  const test_multiply_inf = `
fn INFINITY() -> f32 { let bits = 0x7F800000u; return bitcast<f32>(bits); }
@group(0) @binding(0) var<storage, read_write> data0: array<f32>;
@group(0) @binding(1) var<storage, read_write> data1: array<f32>;
@compute
@workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
    let i: u32 = index.x;
    data0[i] = data1[i] * INFINITY();
    //data0[i] = data1[i] * 7.0;
}}`

  const run = async () => {
    
    const buf_0 = createEmptyBuf(4);
    const buf_1 = createEmptyBuf(4);
    const read_buf = device.createBuffer({size:buf_1.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const kernels = [test_multiply_inf]
    const pipelines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({
      layout: "auto", compute: { module: device.createShaderModule({ code: name }), entryPoint: "main" }})));

    return async () => {
      const commandEncoder = device.createCommandEncoder();
      
      device.queue.writeBuffer(buf_0, 0, new Float32Array([5]));
      
      addComputePass(commandEncoder, pipelines[0], [buf_1, buf_0], [1, 1, 1]);
      commandEncoder.copyBufferToBuffer(buf_1, 0, read_buf, 0, 4);
      const gpuCommands = commandEncoder.finish();
      device.queue.submit([gpuCommands]);

      await read_buf.mapAsync(GPUMapMode.READ);
      const ret = new Float32Array(read_buf.size/4);
      ret.set(new Float32Array(read_buf.getMappedRange()));
      read_buf.unmap();
      //return ret;
      if (ret[0] !== Infinity) throw new Error("result is not Infinity")
    }
  };
  return { run };
})();
export default test_wgsl;
