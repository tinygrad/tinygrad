const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();

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

self.addEventListener('message', setupTransformer);