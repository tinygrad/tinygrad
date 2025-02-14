import transformerModule from './transformer.js'

var transformer = async function() {

  const wasm = await transformerModule();

  return {
    run: (input0,start_pos) => {
      const inputPtr0 = wasm._malloc(4);
      const outputPtr0 = wasm._malloc(4);
      wasm.HEAPU8.set(input0, inputPtr0);
      wasm._net(outputPtr0, inputPtr0, start_pos);
      const output0 = wasm.HEAPU8.slice(outputPtr0, outputPtr0 + 4);
      wasm._free(outputPtr0);
      wasm._free(inputPtr0);
      return [output0];
    },
    wasm: wasm
  }
}
export {transformer};