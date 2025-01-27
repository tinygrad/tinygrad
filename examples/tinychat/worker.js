const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();
const q6k_to_f32_byteFactor = 1 / 210 * 256 * 4;
const q6k_chunk_size = 430080; // compiled kernel input size; GCD of all Q6_K weight sizes

self.addEventListener('message', async (event) => {
  const [k, v] = event.data; // k, v pair from state_dict
  await kernelsReady;
  const gguf_shape_to_decomposer = {
    "2048,2048": self.q6k_to_int8_2048_2048,
    "2048,512": self.q6k_to_int8_512_2048,
    "2048,8192": self.q6k_to_int8_8192_2048,
    "8192,2048": self.q6k_to_int8_2048_8192
  };
  let transferList;

  // this logic involving key name-checking is derived from tinygrad/examples/llama3.py Int8Linear.quantize
  if (k.includes("feed_forward") || k.includes("attention.w")) {
    if (!k.includes("weight")) {throw new Error(`Problem with model weight names: ${key} was expected to have 'weight' in its name.`)}

    const decompSetup = gguf_shape_to_decomposer[v.gguf_shape.join(",")];
    const decomp = await decompSetup();
    // the order of elements needs to be the same as the order of returned values from the exported (jitted) tinygrad function
    const [scale, decomped_bytes] = await decomp.run(v.bytes);
    v.dtype = "int8"
    v.bytes = decomped_bytes;
    v.size = decomped_bytes.length;
    v.scale = scale;
    transferList = [v.bytes.buffer, v.scale.buffer];
  } 
  else {
    // only tok_embeddings.weight and output.weight trigger this block when using int8 quant
    // byte size for each: 215470080
    const decomp = await self.q6k_to_f32();
    const wasm = decomp.wasm;
    const result = new Uint8Array(parseInt(v.bytes.length * q6k_to_f32_byteFactor));
    if (v.bytes.length % q6k_chunk_size !== 0) {throw new Error(`Expected buffer size: ${v.bytes.length} to be divisible by chunk size: ${q6k_chunk_size}`)}
    const num_chunks = v.bytes.length / q6k_chunk_size;
    
    // we could use decomp.run instead of the decomp.wasm handle to avoid malloc/free here, but that would cost an additional Uint8Array.set per run
    const inputPtr0 = wasm._malloc(q6k_chunk_size);
    const output_size = parseInt(q6k_chunk_size * q6k_to_f32_byteFactor);
    const outputPtr0 = wasm._malloc(output_size);

    for (let i = 0; i < num_chunks; i++) {
      const input_cursor = i * q6k_chunk_size;
      const output_cursor = i * output_size;
      wasm.HEAPU8.set(v.bytes.subarray(input_cursor, input_cursor + q6k_chunk_size), inputPtr0);
      wasm._net(outputPtr0, inputPtr0);
      result.set(wasm.HEAPU8.subarray(outputPtr0, outputPtr0 + output_size), output_cursor);
    }

    wasm._free(outputPtr0);
    wasm._free(inputPtr0);
    v.dtype = "float32";
    v.bytes = result;
    v.size = result.length;
    transferList = [v.bytes.buffer];
  }

  self.postMessage(v, transferList);
});