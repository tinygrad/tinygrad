const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();
const q6k_to_f32_byteFactor = 1 / 210 * 256 * 4;
const q6k_chunk_size = 430080; // compiled kernel input size; GCD of all Q6_K weight sizes

async function decompress(event) {
  const [k, v] = event.data; // k, v pair from state_dict
  await kernelsReady;

  // only tok_embeddings.weight triggers this block when using int8 quant
  // byte size: 215470080
  const decomp = await self.q6k_to_f32();
  const wasm = decomp.wasm;
  const result = new Uint8Array(parseInt(v.bytes.length * q6k_to_f32_byteFactor));
  if (v.bytes.length % q6k_chunk_size !== 0) {throw new Error(`Expected buffer size: ${v.bytes.length} to be divisible by chunk size: ${q6k_chunk_size}`)}
  const num_chunks = v.bytes.length / q6k_chunk_size;
    

  // we could use decomp.run instead of the decomp.wasm handle to avoid malloc/free here, but that incurs additional copy time + wasm buffer overhead
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

  self.postMessage(v, [v.bytes.buffer]);
}

async function setupTransformer(event) {
  await kernelsReady;
  self.model = await self.transformer(event.data);
  self.inputPtr = self.model.wasm._malloc(4);
  self.outputPtr = self.model.wasm._malloc(4);
  self.addEventListener("message", inference);
  self.removeEventListener("message", setupTransformer);
  self.postMessage("success");
}

function inference(event) {
  const [tok, start_pos] = event.data;
  const int32tok = new Int32Array([tok]);
  const uint8tok = new Uint8Array(int32tok.buffer);
  self.model.wasm.HEAPU8.set(uint8tok, self.inputPtr);
  self.model.wasm._net(self.outputPtr, self.inputPtr, start_pos);
  const uint8nextTok = self.model.wasm.HEAPU8.slice(self.outputPtr, self.outputPtr + 4);
  const int32nextTok = new Int32Array(uint8nextTok.buffer);
  self.postMessage(int32nextTok[0]);
}

async function setup(event) {
  if (event.data === "decompress") {self.addEventListener("message", decompress);}
  else if (event.data === "setup_transformer") {self.addEventListener("message", setupTransformer);}
  else {throw new Error(`initial message must be 'decompress' or 'setup_transformer', but was ${event.data}`);}
  self.removeEventListener("message", setup);
  self.postMessage("success");
}
self.addEventListener('message', setup);