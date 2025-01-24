const kernelsReady = (async () => {
  // can't get browser to user updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();

self.addEventListener('message', async (event) => {
  const data = event.data;
  await kernelsReady;
  const decomp = await self.q6k_to_int8_8192_2048();
  const result = await decomp(data);
  const transferList = result.map(item => item.buffer);
  self.postMessage(result, transferList);
});