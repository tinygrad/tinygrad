import transformerModule from './transformer.js'

var transformer = async function(state_dict) {

  const wasm = await transformerModule();
  /*
  for (const [i, name] of weightNames.entries()) {
    const bufPtr = wasm._malloc(state_dict[name].size);
    state_dict[name].wasm_buf_start_pos = bufPtr;
    wasm._set_buf(i, bufPtr);
  }
    */

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
    wasm: wasm,
    state_dict: state_dict
  }
}
export {transformer};