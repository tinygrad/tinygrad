const modulePaths = Array.from({ length: 17 }, (_, i) => `./module${i}.js`);
const modulePromises = modulePaths.map(path => import(path));
const buf_map = [
  [0, 1, 2, 260],
  [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 20)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 36)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 52)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 68)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 84)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 100)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 116)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 132)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 148)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 164)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 180)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 196)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 212)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 228)],
  [8, ...Array.from({ length: 16 }, (_, i) => i + 244)],
]
const buf_map_sets = buf_map.map(innerArray => new Set(innerArray));

var transformer = async function(state_dict) {

  const modules = await Promise.all(modulePromises);
  const wasm = await Promise.all(modules.map(mod => mod.default()));
  const weightNames = ["tok_embeddings.arange", "tok_embeddings.weight", "tok_embeddings.scale", "layers.0.attention_norm.weight", "layers.0.attention.wk.weight", "layers.0.attention.wk.scale", "layers.0.attention.wv.weight", "layers.0.attention.wv.scale", "freqs_cis", "layers.0.attention.wq.weight", "layers.0.attention.wq.scale", "layers.0.attention.wo.weight", "layers.0.attention.wo.scale", "layers.0.ffn_norm.weight", "layers.0.feed_forward.w3.weight", "layers.0.feed_forward.w3.scale", "layers.0.feed_forward.w1.weight", "layers.0.feed_forward.w1.scale", "layers.0.feed_forward.w2.weight", "layers.0.feed_forward.w2.scale", "layers.1.attention_norm.weight", "layers.1.attention.wk.weight", "layers.1.attention.wk.scale", "layers.1.attention.wv.weight", "layers.1.attention.wv.scale", "layers.1.attention.wq.weight", "layers.1.attention.wq.scale", "layers.1.attention.wo.weight", "layers.1.attention.wo.scale", "layers.1.ffn_norm.weight", "layers.1.feed_forward.w3.weight", "layers.1.feed_forward.w3.scale", "layers.1.feed_forward.w1.weight", "layers.1.feed_forward.w1.scale", "layers.1.feed_forward.w2.weight", "layers.1.feed_forward.w2.scale", "layers.2.attention_norm.weight", "layers.2.attention.wk.weight", "layers.2.attention.wk.scale", "layers.2.attention.wv.weight", "layers.2.attention.wv.scale", "layers.2.attention.wq.weight", "layers.2.attention.wq.scale", "layers.2.attention.wo.weight", "layers.2.attention.wo.scale", "layers.2.ffn_norm.weight", "layers.2.feed_forward.w3.weight", "layers.2.feed_forward.w3.scale", "layers.2.feed_forward.w1.weight", "layers.2.feed_forward.w1.scale", "layers.2.feed_forward.w2.weight", "layers.2.feed_forward.w2.scale", "layers.3.attention_norm.weight", "layers.3.attention.wk.weight", "layers.3.attention.wk.scale", "layers.3.attention.wv.weight", "layers.3.attention.wv.scale", "layers.3.attention.wq.weight", "layers.3.attention.wq.scale", "layers.3.attention.wo.weight", "layers.3.attention.wo.scale", "layers.3.ffn_norm.weight", "layers.3.feed_forward.w3.weight", "layers.3.feed_forward.w3.scale", "layers.3.feed_forward.w1.weight", "layers.3.feed_forward.w1.scale", "layers.3.feed_forward.w2.weight", "layers.3.feed_forward.w2.scale", "layers.4.attention_norm.weight", "layers.4.attention.wk.weight", "layers.4.attention.wk.scale", "layers.4.attention.wv.weight", "layers.4.attention.wv.scale", "layers.4.attention.wq.weight", "layers.4.attention.wq.scale", "layers.4.attention.wo.weight", "layers.4.attention.wo.scale", "layers.4.ffn_norm.weight", "layers.4.feed_forward.w3.weight", "layers.4.feed_forward.w3.scale", "layers.4.feed_forward.w1.weight", "layers.4.feed_forward.w1.scale", "layers.4.feed_forward.w2.weight", "layers.4.feed_forward.w2.scale", "layers.5.attention_norm.weight", "layers.5.attention.wk.weight", "layers.5.attention.wk.scale", "layers.5.attention.wv.weight", "layers.5.attention.wv.scale", "layers.5.attention.wq.weight", "layers.5.attention.wq.scale", "layers.5.attention.wo.weight", "layers.5.attention.wo.scale", "layers.5.ffn_norm.weight", "layers.5.feed_forward.w3.weight", "layers.5.feed_forward.w3.scale", "layers.5.feed_forward.w1.weight", "layers.5.feed_forward.w1.scale", "layers.5.feed_forward.w2.weight", "layers.5.feed_forward.w2.scale", "layers.6.attention_norm.weight", "layers.6.attention.wk.weight", "layers.6.attention.wk.scale", "layers.6.attention.wv.weight", "layers.6.attention.wv.scale", "layers.6.attention.wq.weight", "layers.6.attention.wq.scale", "layers.6.attention.wo.weight", "layers.6.attention.wo.scale", "layers.6.ffn_norm.weight", "layers.6.feed_forward.w3.weight", "layers.6.feed_forward.w3.scale", "layers.6.feed_forward.w1.weight", "layers.6.feed_forward.w1.scale", "layers.6.feed_forward.w2.weight", "layers.6.feed_forward.w2.scale", "layers.7.attention_norm.weight", "layers.7.attention.wk.weight", "layers.7.attention.wk.scale", "layers.7.attention.wv.weight", "layers.7.attention.wv.scale", "layers.7.attention.wq.weight", "layers.7.attention.wq.scale", "layers.7.attention.wo.weight", "layers.7.attention.wo.scale", "layers.7.ffn_norm.weight", "layers.7.feed_forward.w3.weight", "layers.7.feed_forward.w3.scale", "layers.7.feed_forward.w1.weight", "layers.7.feed_forward.w1.scale", "layers.7.feed_forward.w2.weight", "layers.7.feed_forward.w2.scale", "layers.8.attention_norm.weight", "layers.8.attention.wk.weight", "layers.8.attention.wk.scale", "layers.8.attention.wv.weight", "layers.8.attention.wv.scale", "layers.8.attention.wq.weight", "layers.8.attention.wq.scale", "layers.8.attention.wo.weight", "layers.8.attention.wo.scale", "layers.8.ffn_norm.weight", "layers.8.feed_forward.w3.weight", "layers.8.feed_forward.w3.scale", "layers.8.feed_forward.w1.weight", "layers.8.feed_forward.w1.scale", "layers.8.feed_forward.w2.weight", "layers.8.feed_forward.w2.scale", "layers.9.attention_norm.weight", "layers.9.attention.wk.weight", "layers.9.attention.wk.scale", "layers.9.attention.wv.weight", "layers.9.attention.wv.scale", "layers.9.attention.wq.weight", "layers.9.attention.wq.scale", "layers.9.attention.wo.weight", "layers.9.attention.wo.scale", "layers.9.ffn_norm.weight", "layers.9.feed_forward.w3.weight", "layers.9.feed_forward.w3.scale", "layers.9.feed_forward.w1.weight", "layers.9.feed_forward.w1.scale", "layers.9.feed_forward.w2.weight", "layers.9.feed_forward.w2.scale", "layers.10.attention_norm.weight", "layers.10.attention.wk.weight", "layers.10.attention.wk.scale", "layers.10.attention.wv.weight", "layers.10.attention.wv.scale", "layers.10.attention.wq.weight", "layers.10.attention.wq.scale", "layers.10.attention.wo.weight", "layers.10.attention.wo.scale", "layers.10.ffn_norm.weight", "layers.10.feed_forward.w3.weight", "layers.10.feed_forward.w3.scale", "layers.10.feed_forward.w1.weight", "layers.10.feed_forward.w1.scale", "layers.10.feed_forward.w2.weight", "layers.10.feed_forward.w2.scale", "layers.11.attention_norm.weight", "layers.11.attention.wk.weight", "layers.11.attention.wk.scale", "layers.11.attention.wv.weight", "layers.11.attention.wv.scale", "layers.11.attention.wq.weight", "layers.11.attention.wq.scale", "layers.11.attention.wo.weight", "layers.11.attention.wo.scale", "layers.11.ffn_norm.weight", "layers.11.feed_forward.w3.weight", "layers.11.feed_forward.w3.scale", "layers.11.feed_forward.w1.weight", "layers.11.feed_forward.w1.scale", "layers.11.feed_forward.w2.weight", "layers.11.feed_forward.w2.scale", "layers.12.attention_norm.weight", "layers.12.attention.wk.weight", "layers.12.attention.wk.scale", "layers.12.attention.wv.weight", "layers.12.attention.wv.scale", "layers.12.attention.wq.weight", "layers.12.attention.wq.scale", "layers.12.attention.wo.weight", "layers.12.attention.wo.scale", "layers.12.ffn_norm.weight", "layers.12.feed_forward.w3.weight", "layers.12.feed_forward.w3.scale", "layers.12.feed_forward.w1.weight", "layers.12.feed_forward.w1.scale", "layers.12.feed_forward.w2.weight", "layers.12.feed_forward.w2.scale", "layers.13.attention_norm.weight", "layers.13.attention.wk.weight", "layers.13.attention.wk.scale", "layers.13.attention.wv.weight", "layers.13.attention.wv.scale", "layers.13.attention.wq.weight", "layers.13.attention.wq.scale", "layers.13.attention.wo.weight", "layers.13.attention.wo.scale", "layers.13.ffn_norm.weight", "layers.13.feed_forward.w3.weight", "layers.13.feed_forward.w3.scale", "layers.13.feed_forward.w1.weight", "layers.13.feed_forward.w1.scale", "layers.13.feed_forward.w2.weight", "layers.13.feed_forward.w2.scale", "layers.14.attention_norm.weight", "layers.14.attention.wk.weight", "layers.14.attention.wk.scale", "layers.14.attention.wv.weight", "layers.14.attention.wv.scale", "layers.14.attention.wq.weight", "layers.14.attention.wq.scale", "layers.14.attention.wo.weight", "layers.14.attention.wo.scale", "layers.14.ffn_norm.weight", "layers.14.feed_forward.w3.weight", "layers.14.feed_forward.w3.scale", "layers.14.feed_forward.w1.weight", "layers.14.feed_forward.w1.scale", "layers.14.feed_forward.w2.weight", "layers.14.feed_forward.w2.scale", "layers.15.attention_norm.weight", "layers.15.attention.wk.weight", "layers.15.attention.wk.scale", "layers.15.attention.wv.weight", "layers.15.attention.wv.scale", "layers.15.attention.wq.weight", "layers.15.attention.wq.scale", "layers.15.attention.wo.weight", "layers.15.attention.wo.scale", "layers.15.ffn_norm.weight", "layers.15.feed_forward.w3.weight", "layers.15.feed_forward.w3.scale", "layers.15.feed_forward.w1.weight", "layers.15.feed_forward.w1.scale", "layers.15.feed_forward.w2.weight", "layers.15.feed_forward.w2.scale", "norm.weight"];

  for (const [i, name] of weightNames.entries()) {
    for (const [j, buf_set] of buf_map_sets.entries()) {
      if (buf_set.has(i)) {
        const bufPtr = wasm[j]._malloc(state_dict[name].size);
        (state_dict[name].wasm_offsets ??= {})[j] = bufPtr;
        wasm[j]._set_buf(i, bufPtr);
      }
    }
  }

  const io_buf_sizes = [[4, 8192, 4]];
  for (let i=1; i <= 16; i++) {
    io_buf_sizes.push([8192, 8192]);
  }
  const ptrs = io_buf_sizes.map((innerArray, i) => innerArray.map(size => wasm[i]._malloc(size)));

  return {
    run: (input0,start_pos) => {

      let wasm_in = ptrs[0][0];
      let wasm_out = ptrs[0][1];
      wasm[0].HEAPU8.set(input0, wasm_in);
      console.log(wasm[0])
      wasm[0]._net0(wasm_out, wasm_in);

      let prev_out = wasm_out;
      let _net_out;

      // transformer blocks
      for (let i=1; i<17; i++) {
        wasm_in = ptrs[i][0];
        wasm[i].HEAPU8.set(wasm[i-1].HEAPU8.slice(prev_out, prev_out + 8192), wasm_in);
        wasm_out = ptrs[i][1];
        _net_out = wasm[i]._net(wasm_out, wasm_in, start_pos); // outputs a float* from last transformer block
        prev_out = wasm_out;
      }

      // output
      const buf_352 = ptrs[0][1];
      const buf_5 = ptrs[0][2];
      wasm_out = ptrs[0][0];

      wasm[0].HEAPU8.set(wasm[16].HEAPU8.slice(prev_out, prev_out + 8192), buf_352);
      wasm[0].HEAPU8.set(wasm[16].HEAPU8.slice(_net_out, _net_out + 4), buf_5);
      wasm_out = wasm[0]._malloc(4);
      wasm[0]._net1(wasm_out, buf_5, buf_352);
      const output0 = wasm[0].HEAPU8.slice(wasm_out, wasm_out + 4);

      return [output0];
    },
    wasm: wasm,
    state_dict: state_dict
  }
}
export {transformer};