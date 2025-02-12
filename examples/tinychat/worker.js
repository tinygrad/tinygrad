const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();

//const parts = [];
//const files = [];
// TODO: clean this up. It's all for stability of loading weights to WASM
// browser crash on mobile seems to be driven by frequent wasm.HEAPU8.set calls
// setting really big buffers to wasm is not zero-copy, and too much memory overhead will cause OOM
// Therefore we are striking a balance between copying buffers that are not too large (avoid OOM), and not too small (avoid frequent wasm.HEAPU8.set)

// This is how to glue together the current split files to reassemble tok_embeddings.weight, the rest of the weights are not split
const tok_embed_indices = [14, 13, 7, 6, 5, 4, 3, 2, 1, 0, 12, 11, 10, 9, 8, 63]; // then the rest in any order
let index_counter = 0;
const tok_embed_indices_set = new Set(tok_embed_indices);
const missingIndices = [...Array(74).keys()].filter(i => !tok_embed_indices.includes(i));
const file_order = [...tok_embed_indices, ...missingIndices];
const weightNames = ["tok_embeddings.arange", "tok_embeddings.weight", "tok_embeddings.scale", "layers.0.attention_norm.weight", "layers.0.attention.wk.weight", "layers.0.attention.wk.scale", "layers.0.attention.wv.weight", "layers.0.attention.wv.scale", "freqs_cis", "layers.0.attention.wq.weight", "layers.0.attention.wq.scale", "layers.0.attention.wo.weight", "layers.0.attention.wo.scale", "layers.0.ffn_norm.weight", "layers.0.feed_forward.w3.weight", "layers.0.feed_forward.w3.scale", "layers.0.feed_forward.w1.weight", "layers.0.feed_forward.w1.scale", "layers.0.feed_forward.w2.weight", "layers.0.feed_forward.w2.scale", "layers.1.attention_norm.weight", "layers.1.attention.wk.weight", "layers.1.attention.wk.scale", "layers.1.attention.wv.weight", "layers.1.attention.wv.scale", "layers.1.attention.wq.weight", "layers.1.attention.wq.scale", "layers.1.attention.wo.weight", "layers.1.attention.wo.scale", "layers.1.ffn_norm.weight", "layers.1.feed_forward.w3.weight", "layers.1.feed_forward.w3.scale", "layers.1.feed_forward.w1.weight", "layers.1.feed_forward.w1.scale", "layers.1.feed_forward.w2.weight", "layers.1.feed_forward.w2.scale", "layers.2.attention_norm.weight", "layers.2.attention.wk.weight", "layers.2.attention.wk.scale", "layers.2.attention.wv.weight", "layers.2.attention.wv.scale", "layers.2.attention.wq.weight", "layers.2.attention.wq.scale", "layers.2.attention.wo.weight", "layers.2.attention.wo.scale", "layers.2.ffn_norm.weight", "layers.2.feed_forward.w3.weight", "layers.2.feed_forward.w3.scale", "layers.2.feed_forward.w1.weight", "layers.2.feed_forward.w1.scale", "layers.2.feed_forward.w2.weight", "layers.2.feed_forward.w2.scale", "layers.3.attention_norm.weight", "layers.3.attention.wk.weight", "layers.3.attention.wk.scale", "layers.3.attention.wv.weight", "layers.3.attention.wv.scale", "layers.3.attention.wq.weight", "layers.3.attention.wq.scale", "layers.3.attention.wo.weight", "layers.3.attention.wo.scale", "layers.3.ffn_norm.weight", "layers.3.feed_forward.w3.weight", "layers.3.feed_forward.w3.scale", "layers.3.feed_forward.w1.weight", "layers.3.feed_forward.w1.scale", "layers.3.feed_forward.w2.weight", "layers.3.feed_forward.w2.scale", "layers.4.attention_norm.weight", "layers.4.attention.wk.weight", "layers.4.attention.wk.scale", "layers.4.attention.wv.weight", "layers.4.attention.wv.scale", "layers.4.attention.wq.weight", "layers.4.attention.wq.scale", "layers.4.attention.wo.weight", "layers.4.attention.wo.scale", "layers.4.ffn_norm.weight", "layers.4.feed_forward.w3.weight", "layers.4.feed_forward.w3.scale", "layers.4.feed_forward.w1.weight", "layers.4.feed_forward.w1.scale", "layers.4.feed_forward.w2.weight", "layers.4.feed_forward.w2.scale", "layers.5.attention_norm.weight", "layers.5.attention.wk.weight", "layers.5.attention.wk.scale", "layers.5.attention.wv.weight", "layers.5.attention.wv.scale", "layers.5.attention.wq.weight", "layers.5.attention.wq.scale", "layers.5.attention.wo.weight", "layers.5.attention.wo.scale", "layers.5.ffn_norm.weight", "layers.5.feed_forward.w3.weight", "layers.5.feed_forward.w3.scale", "layers.5.feed_forward.w1.weight", "layers.5.feed_forward.w1.scale", "layers.5.feed_forward.w2.weight", "layers.5.feed_forward.w2.scale", "layers.6.attention_norm.weight", "layers.6.attention.wk.weight", "layers.6.attention.wk.scale", "layers.6.attention.wv.weight", "layers.6.attention.wv.scale", "layers.6.attention.wq.weight", "layers.6.attention.wq.scale", "layers.6.attention.wo.weight", "layers.6.attention.wo.scale", "layers.6.ffn_norm.weight", "layers.6.feed_forward.w3.weight", "layers.6.feed_forward.w3.scale", "layers.6.feed_forward.w1.weight", "layers.6.feed_forward.w1.scale", "layers.6.feed_forward.w2.weight", "layers.6.feed_forward.w2.scale", "layers.7.attention_norm.weight", "layers.7.attention.wk.weight", "layers.7.attention.wk.scale", "layers.7.attention.wv.weight", "layers.7.attention.wv.scale", "layers.7.attention.wq.weight", "layers.7.attention.wq.scale", "layers.7.attention.wo.weight", "layers.7.attention.wo.scale", "layers.7.ffn_norm.weight", "layers.7.feed_forward.w3.weight", "layers.7.feed_forward.w3.scale", "layers.7.feed_forward.w1.weight", "layers.7.feed_forward.w1.scale", "layers.7.feed_forward.w2.weight", "layers.7.feed_forward.w2.scale", "layers.8.attention_norm.weight", "layers.8.attention.wk.weight", "layers.8.attention.wk.scale", "layers.8.attention.wv.weight", "layers.8.attention.wv.scale", "layers.8.attention.wq.weight", "layers.8.attention.wq.scale", "layers.8.attention.wo.weight", "layers.8.attention.wo.scale", "layers.8.ffn_norm.weight", "layers.8.feed_forward.w3.weight", "layers.8.feed_forward.w3.scale", "layers.8.feed_forward.w1.weight", "layers.8.feed_forward.w1.scale", "layers.8.feed_forward.w2.weight", "layers.8.feed_forward.w2.scale", "layers.9.attention_norm.weight", "layers.9.attention.wk.weight", "layers.9.attention.wk.scale", "layers.9.attention.wv.weight", "layers.9.attention.wv.scale", "layers.9.attention.wq.weight", "layers.9.attention.wq.scale", "layers.9.attention.wo.weight", "layers.9.attention.wo.scale", "layers.9.ffn_norm.weight", "layers.9.feed_forward.w3.weight", "layers.9.feed_forward.w3.scale", "layers.9.feed_forward.w1.weight", "layers.9.feed_forward.w1.scale", "layers.9.feed_forward.w2.weight", "layers.9.feed_forward.w2.scale", "layers.10.attention_norm.weight", "layers.10.attention.wk.weight", "layers.10.attention.wk.scale", "layers.10.attention.wv.weight", "layers.10.attention.wv.scale", "layers.10.attention.wq.weight", "layers.10.attention.wq.scale", "layers.10.attention.wo.weight", "layers.10.attention.wo.scale", "layers.10.ffn_norm.weight", "layers.10.feed_forward.w3.weight", "layers.10.feed_forward.w3.scale", "layers.10.feed_forward.w1.weight", "layers.10.feed_forward.w1.scale", "layers.10.feed_forward.w2.weight", "layers.10.feed_forward.w2.scale", "layers.11.attention_norm.weight", "layers.11.attention.wk.weight", "layers.11.attention.wk.scale", "layers.11.attention.wv.weight", "layers.11.attention.wv.scale", "layers.11.attention.wq.weight", "layers.11.attention.wq.scale", "layers.11.attention.wo.weight", "layers.11.attention.wo.scale", "layers.11.ffn_norm.weight", "layers.11.feed_forward.w3.weight", "layers.11.feed_forward.w3.scale", "layers.11.feed_forward.w1.weight", "layers.11.feed_forward.w1.scale", "layers.11.feed_forward.w2.weight", "layers.11.feed_forward.w2.scale", "layers.12.attention_norm.weight", "layers.12.attention.wk.weight", "layers.12.attention.wk.scale", "layers.12.attention.wv.weight", "layers.12.attention.wv.scale", "layers.12.attention.wq.weight", "layers.12.attention.wq.scale", "layers.12.attention.wo.weight", "layers.12.attention.wo.scale", "layers.12.ffn_norm.weight", "layers.12.feed_forward.w3.weight", "layers.12.feed_forward.w3.scale", "layers.12.feed_forward.w1.weight", "layers.12.feed_forward.w1.scale", "layers.12.feed_forward.w2.weight", "layers.12.feed_forward.w2.scale", "layers.13.attention_norm.weight", "layers.13.attention.wk.weight", "layers.13.attention.wk.scale", "layers.13.attention.wv.weight", "layers.13.attention.wv.scale", "layers.13.attention.wq.weight", "layers.13.attention.wq.scale", "layers.13.attention.wo.weight", "layers.13.attention.wo.scale", "layers.13.ffn_norm.weight", "layers.13.feed_forward.w3.weight", "layers.13.feed_forward.w3.scale", "layers.13.feed_forward.w1.weight", "layers.13.feed_forward.w1.scale", "layers.13.feed_forward.w2.weight", "layers.13.feed_forward.w2.scale", "layers.14.attention_norm.weight", "layers.14.attention.wk.weight", "layers.14.attention.wk.scale", "layers.14.attention.wv.weight", "layers.14.attention.wv.scale", "layers.14.attention.wq.weight", "layers.14.attention.wq.scale", "layers.14.attention.wo.weight", "layers.14.attention.wo.scale", "layers.14.ffn_norm.weight", "layers.14.feed_forward.w3.weight", "layers.14.feed_forward.w3.scale", "layers.14.feed_forward.w1.weight", "layers.14.feed_forward.w1.scale", "layers.14.feed_forward.w2.weight", "layers.14.feed_forward.w2.scale", "layers.15.attention_norm.weight", "layers.15.attention.wk.weight", "layers.15.attention.wk.scale", "layers.15.attention.wv.weight", "layers.15.attention.wv.scale", "layers.15.attention.wq.weight", "layers.15.attention.wq.scale", "layers.15.attention.wo.weight", "layers.15.attention.wo.scale", "layers.15.ffn_norm.weight", "layers.15.feed_forward.w3.weight", "layers.15.feed_forward.w3.scale", "layers.15.feed_forward.w1.weight", "layers.15.feed_forward.w1.scale", "layers.15.feed_forward.w2.weight", "layers.15.feed_forward.w2.scale", "norm.weight"];
const name_to_id = Object.fromEntries(weightNames.map((name, index) => [name, index]));
let first_malloc_size = 0;
const first_malloc_files = [];
let first_malloc_cursor = 0;

// in wasm, malloc one big chunk, the size of all the weights
// to each weight in state_dict, attach offset from malloc start
// then copyin one file at a time, as it arrives
// does order matter for setting? 
// does malloc size matter? we need to malloc at most once for each weight, otherwise there will be gaps within weights that are split. 
// real memory seems to be allocated lazily

async function initStateDict(event) {
  await kernelsReady;
  self.model = await self.transformer(event.data.state_dict);
  self.inputPtr = self.model.wasm._malloc(4);
  self.outputPtr = self.model.wasm._malloc(4);
  let files = event.data.files;


  // TODO: clean this up
  //let first_malloc_size = 0;
  for (const i of file_order) {
    first_malloc_size += files[i].size;
    if (i === 63) break;
  }
  //let cursor = self.model.wasm._malloc(event.data.totalSize);
  /*
  let cursor = self.model.wasm._malloc(first_malloc_size);
  for (const i of file_order) {
    files[i].wasm_offset = cursor;
    cursor += files[i].size;
    if (i === 63) break;
  }
    */

  files = file_order.map(idx => files[idx]);
  self.addEventListener("message", loadStateDict);
  self.removeEventListener("message", initStateDict);
  //self.postMessage(self.model.state_dict);
  //delete self.model.state_dict;
  self.postMessage(files);
}

function loadStateDict(event) {
  if (event.data === "done") {
    /*
    for (const file of files) {
      const ptr = self.model.wasm._malloc(file.bytes.length);
      self.model.wasm.HEAPU8.set(file.bytes, ptr);
    }
      */
    self.addEventListener("message", inference);
    self.removeEventListener("message", loadStateDict);
  }
  else {
    const file = event.data;
    //files.push(file);
    if (index_counter < 16) {
      //self.model.wasm.HEAPU8.set(file.bytes, file.wasm_offset);
      for (const part of file.parts) {
        if (part.target_start_pos === 0) {
          //self.model.wasm._set_buf(name_to_id[part.key], file.wasm_offset + part.file_start_pos);
          //self.model.wasm._set_buf(name_to_id[part.key], first_malloc_cursor + part.file_start_pos);
          part.wasm_offset = first_malloc_cursor + part.file_start_pos;
        }
      }
      first_malloc_files.push(file);
      index_counter += 1;
      first_malloc_cursor += file.size;

      if (index_counter === 16) {
        const coalesce = new Uint8Array(first_malloc_size);
        let coalesce_cursor = 0;
        const ptr = self.model.wasm._malloc(first_malloc_size);
        for (const f of first_malloc_files) {
          coalesce.set(f.bytes, coalesce_cursor);
          coalesce_cursor += f.size;
          f.bytes = null;

          for (const part of f.parts) {
            if (part.wasm_offset !== undefined) {
              self.model.wasm._set_buf(name_to_id[part.key], ptr + part.wasm_offset);
            }
          }
        }
        //const ptr = self.model.wasm._malloc(first_malloc_size);
        self.model.wasm.HEAPU8.set(coalesce, ptr);

      }
    }
    else {
      const malloc_ptr = self.model.wasm._malloc(file.size);
      self.model.wasm.HEAPU8.set(file.bytes, malloc_ptr);
      for (const part of file.parts) {
        if (part.target_start_pos === 0) {
          self.model.wasm._set_buf(name_to_id[part.key], malloc_ptr + part.file_start_pos);
        }
      }
      file.bytes = null;
    }

    /*
    self.model.wasm.HEAPU8.set(file.bytes, file.wasm_offset);
    for (const part of file.parts) {
      if (part.target_start_pos === 0) {
        self.model.wasm._set_buf(name_to_id[part.key], file.wasm_offset + part.file_start_pos);
      }
    }
      */
    //file.bytes = null;
  }
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

self.addEventListener('message', initStateDict);