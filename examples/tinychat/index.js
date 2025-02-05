window.TINYCHAT_ROOT = "/";
window.MODEL_BASE_URL= ".";
const queryParams = new URLSearchParams(window.location.search);
const normalizedParams = Object.fromEntries([...queryParams].map(([key, value]) => [key.toUpperCase(), value.toUpperCase()]));
window.BACKEND = (normalizedParams["BACKEND"] === "WASM") ? "WASM" : "WebGPU";

const tiktokenReady = (async () => {
  const { init, get_encoding, Tiktoken, load } = await import('./tiktoken.js');
  window.Tiktoken = Tiktoken;
  window.tiktokenInit = init;
  window.tiktokenGetEncoding = get_encoding;
  window.tiktokenLoad = load;
})();

const kernelsReady = (async () => {
  if (window.BACKEND === "WASM") {var exports = await import(`./net_clang.js?version=${Date.now()}`);} // TODO: is cache-busting necessary
  else if (window.BACKEND === "WebGPU") {var exports = await import(`./net.js?version=${Date.now()}`);}
  Object.assign(self, exports);
})();

// copied from examples/webgpu/stable_diffusion/index.html
const getDevice = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const requiredLimits = {};
  const maxBufferSizeInSDModel = 1073741824;
  requiredLimits.maxStorageBufferBindingSize = maxBufferSizeInSDModel;
  requiredLimits.maxBufferSize = maxBufferSizeInSDModel;
            
  return await adapter.requestDevice({
    requiredLimits
  });
};

// copied from examples/webgpu/stable_diffusion/index.html 
function initDb() {
  return new Promise((resolve, reject) => {
    let db;
    const request = indexedDB.open('tinydb', 1);
    request.onerror = (event) => {
      console.error('Database error:', event.target.error);
      resolve(null);
    };

    request.onsuccess = (event) => {
      db = event.target.result;
      console.log("Db initialized.");
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      db = event.target.result;
      if (!db.objectStoreNames.contains('tensors')) {
        db.createObjectStore('tensors', { keyPath: 'id' });
      }
    };
  });
}

// copied from examples/webgpu/stable_diffusion/index.html 
function readTensorFromDb(db, id) {
  return new Promise((resolve, reject) => {
    if (db == null) {
      resolve(null);
    }
            
      const transaction = db.transaction(['tensors'], 'readonly');
      const store = transaction.objectStore('tensors');
      const request = store.get(id);

      transaction.onabort = (event) => {
        console.log("Transaction error while reading tensor: " + event.target.error);
        resolve(null);
      };

      request.onsuccess = (event) => {
        const result = event.target.result;
        if (result) {
          resolve(result);
        } else {
          resolve(null);
        }
      };

      request.onerror = (event) => {
        console.error('Tensor retrieve failed: ', event.target.error);
        resolve(null);
      };
  });
}

function getAllKeysFromDb(db) {
  return new Promise((resolve, reject) => {
    if (db == null) {resolve([]);}
    const transaction = db.transaction(['tensors'], 'readonly');
    const store = transaction.objectStore('tensors');
    const request = store.getAllKeys();
    transaction.onabort = (event) => {
      console.log("Transaction error while reading IndexedDB keys: " + event.target.error);
      resolve([]);
    };
    request.onsuccess = function (event) {resolve(event.target.result);};
    request.onerror = (event) => {
      console.error('Retrieval of IndexedDB keys failed: ', event.target.error);
      resolve([]);
    };
  });
}

// modified from examples/webgpu/stable_diffusion/index.html 
function saveTensorToDb(db, id, tensor) {
  return readTensorFromDb(db, id).then((result) => {
    if (!result) {
      new Promise((resolve, reject) => {
        if (db == null) {
          resolve(null);
        }

        const transaction = db.transaction(['tensors'], 'readwrite');
        const store = transaction.objectStore('tensors');
        const request = store.put({ id: id, content: tensor });

        transaction.onabort = (event) => {
          console.log("Transaction error while saving tensor: " + event.target.error);
          resolve(null);
        };

        request.onsuccess = () => {
          console.log('Tensor saved successfully.');
          resolve();
        };

        request.onerror = (event) => {
          console.error('Tensor save failed:', event.target.error);
          resolve(null);
        };
      });
    } else {
      return null;
    }
  }).catch(()=> null);
}

function deleteTensorFromDb(db, id) {
  return new Promise((resolve, reject) => {
    if (db == null) {
      console.error("Database is not initialized.");
      resolve(null);
      return;
    }

    const transaction = db.transaction(['tensors'], 'readwrite');
    const store = transaction.objectStore('tensors');
    const request = store.delete(id);

    transaction.oncomplete = () => {
      console.log(`Tensor with ID '${id}' deleted successfully.`);
      resolve();
    };

    transaction.onerror = (event) => {
      console.error("Transaction error while deleting tensor:", event.target.error);
      resolve(null);
    };

    request.onerror = (event) => {
      console.error('Tensor deletion failed:', event.target.error);
      resolve(null);
    };

    request.onsuccess = () => {
      console.log(`Delete request for tensor with ID '${id}' succeeded.`);
    };
  });
}

async function hashBuffer(bytes) {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function getFreePipeline(pipelinePool) {
  for (;;) {
    const idx = pipelinePool.findIndex(obj => !obj.busy);
    if (idx >= 0) {
      pipelinePool[idx].busy = true;
      return pipelinePool[idx].pipeline;
    }
    await new Promise(r => setTimeout(r, 5));
  }
}

function releasePipeline(pipeline, pipelinePool) {
  const obj = pipelinePool.find(obj => obj.pipeline === pipeline);
  if (obj) obj.busy = false;
}

function sendMessageToWorker(worker, message) {
  return new Promise((resolve, reject) => {
    const onMessage = (event) => {
      resolve(event.data);
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
    };

    const onError = (error) => {
      reject(error);
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);

    if (message.header === "token") {worker.postMessage(message.data);}
    else if (message.header === "setup") {worker.postMessage(message.data);}
    // if message.data is a [k, v] from Object.entries(state_dict)
    else if (message.header === "k_v") {worker.postMessage(message.data, [message.data[1].bytes.buffer]);}
    // if message.data is the decompressed state_dict
    else if (message.header === "state_dict") {worker.postMessage(message.data, Object.values(message.data).flatMap(({ bytes }) => bytes ? [bytes.buffer] : []));}
  });
}


const load_state_dict = async (device, progress) => {
  let completed = 0;
  let totalLoaded = 0;
  let totalSize = 0;
  let partSize = {};

  const progressCallback = (part, loaded, total, message) => {
    totalLoaded += loaded;

    if (!partSize[part]) {
      totalSize += total;
      partSize[part] = true;
    }
                
    progress(totalLoaded, totalSize, message);
  };

  // modified from examples/webgpu/stable_diffusion/index.html getProgressDlForPart
  const loadPart = async (part, progressCallback) => {
      const response = await fetch(part);
      const contentLength = response.headers.get('content-length');
      const total = parseInt(contentLength, 10);

      const res = new Response(new ReadableStream({
          async start(controller) {
              const reader = response.body.getReader();
              for (;;) {
                  const { done, value } = await reader.read();
                  if (done) break;
                  progressCallback(part, value.byteLength, total, `Downloading model: ${completed}/29`);
                  controller.enqueue(value);
              }
                    
              controller.close();
          },
      }));
        
      return res.arrayBuffer();
  };

  const response = await fetch(`${window.MODEL_BASE_URL}/net_metadata.json`);
  // TODO: cache metadata (and everything else) so tinychat works offline
  const data = await response.json();
  const state_dict = data.metadata.state_dict;

  let db = await initDb();

  const getPart = async(filename, hash) => {
    let part = await readTensorFromDb(db, hash);

    if (part) {
      console.log(`Cache hit: ${filename}, hash: ${hash}`);
      totalLoaded += part.content.byteLength;
      totalSize += part.content.byteLength;
      progress(totalLoaded, totalSize, `Downloading model: ${completed}/29`)
      return Promise.resolve(part.content);
    } else {
      console.log(`Cache miss: ${filename}, hash: ${hash}`);
      return loadPart(`${window.MODEL_BASE_URL}/${filename}`, progressCallback);
    }
  }

  const correctHashes = data.metadata.files.map(file => file.hash)
  // delete unused cached buffers to free disk space -- if we update weights, user will otherwise have obsolete cached buffers
  const dbKeys = await getAllKeysFromDb(db);
  const correctHashesSet = new Set(correctHashes);
  const notInCorrectHashes = dbKeys.filter(key => !correctHashesSet.has(key));
  // await these right before starting to save new stuff
  const deletionPromises = notInCorrectHashes.map(async (hash) => deleteTensorFromDb(db, hash));
  //for (const hash of notInCorrectHashes) {deleteTensorFromDb(db, hash);}

  for (const [k,v] of Object.entries(state_dict)) {
    for (const part of v.parts) {
      if (part.empty) state_dict[k].empty = true; // assumes no other parts of this weight exist and are non-empty
      else {
        part.key = k;
        part.dtype = v.dtype;
        if (!data.metadata.files[part.file].parts) data.metadata.files[part.file].parts = [];
        data.metadata.files[part.file].parts.push(part);
      }
    }
  }

  const cachedFileHashes = new Set(dbKeys.filter(key => correctHashesSet.has(key)));
  const cachedFiles = data.metadata.files.filter(file => cachedFileHashes.has(file.hash));
  const toDownload = data.metadata.files.filter(file => !cachedFileHashes.has(file.hash));
  const downloaded = [];
  // to limit memory overhead, we pause downloads if we have this number of downloaded files waiting to be processed
  const numDownloaders = 5; // TODO: dynamically base this on DL file size?
  const chainDownload = async (file) => {
    loadPart(`${window.MODEL_BASE_URL}/${file.name}`, progressCallback) // triggers download
    .then(async (arraybuf) => { 
      downloaded.push({ ...file, bytes: new Uint8Array(arraybuf)});
      // pause downloads if further processing is a bottleneck
      while (toDownload.length && downloaded.length >= numDownloaders) await new Promise(resolve => setTimeout(resolve, 200));
      if (toDownload.length && downloaded.length < numDownloaders) chainDownload(toDownload.shift()); // start next download
    })
  }
  /*
  let totalLoaded = 0;
  let totalSize = Object.values(state_dict).filter(item => item.dtype === "Q6_K").reduce((sum, item) => sum + item.size, 0);
  const numCheckpoints = 90;
  let nextCheckpoint = totalSize / numCheckpoints;
  const decompProgressFraction = 0.90;
  totalSize = totalSize / decompProgressFraction; // extend progress bar for minor steps after decompression
  const t0 = performance.now();
  */
  for (let i=0; i<numDownloaders; i++) if (toDownload.length) chainDownload(toDownload.shift());

  await kernelsReady;
  // instantiates empty weight buffers on WebGPU, attaches buffers to state_dict
  const model = await transformer().setup(device, state_dict, progress);
  delete state_dict["output.weight"]; // uses same data as tok_embeddings.weight, TODO: make consistent with wasm loading
  delete state_dict["output.scale"]; // uses same data as tok_embeddings.weight, TODO: make consistent with wasm loading


  /*
  if (window.BACKEND === "WebGPU") {
    const num_decompressers = 8;
    // decompression time goes from 15sec to 10sec by scheduling GPU jobs like below, with Q6_K quantized llama-3.2-1B
    // TODO: can we get tinygrad to give us bigger kernels? currently throws exceptions when trying to compile them
    var pipelinePool = await Promise.all(
      Array.from({ length: num_decompressers }, () => q6k_to_f32().setup(device)).map(async (promise) => {
        return {pipeline: await promise, busy: false};
      })
    );
  }
  else if (window.BACKEND === "WASM") {
    // current source weights have everything int8 quantized or float32; only output.weight is decompressed from Q6_K. 
    // we could make this faster with more workers, but only takes 2-3 sec
    const num_decompressers = 1
    const workers = Array.from({ length: num_decompressers }, () => new Worker(`./worker.js?version=${Date.now()}`));
    const promises = workers.map(async (worker) => {
      await sendMessageToWorker(worker, {header: "setup", data: "decompress"}); // setup flag, worker can only do decompression now
      return {
        worker: worker,
        pipeline: (k_v_pair) => sendMessageToWorker(worker, {header: "k_v", data: k_v_pair}),
        busy: false
      };
    });
    var pipelinePool = await Promise.all(promises);
  }

  // Decompresses a tensor (or slice thereof), loading the result to the model's state_dict
  async function decompressToStateDict(part) {
    if (part.dtype !== "Q6_K") throw new Error("only Q6_K to float32 decompression is supported by tinychat")
    if (window.BACKEND === "WebGPU") {
      const gpuJobs = [];
      const inChunkSize = 3144960; // max size that tinygrad compiled without exceptions, that is divisible by 210; TODO base it on net.js
      const byteFactor = 1 / 210 * 256 * 4;

      function scheduleDequantizeJob(slice) {
        return (async () => {
          const decompress = await getFreePipeline(pipelinePool);
          const out = await decompress(slice.bytes); // local arraybuffer
          const decompBytes = new Uint8Array(out.buffer);
          const unpadded = (decompBytes.length === slice.output_size) ? decompBytes : decompBytes.subarray(0, slice.output_size); // in case we padded
          new Uint8Array(state_dict[slice.key].bytes.getMappedRange(slice.target_start_pos, slice.output_size)).set(unpadded);
          releasePipeline(decompress, pipelinePool);
        })();
      }

      for (let cursor = 0; cursor < part.size; cursor += inChunkSize) {
        const slice_end_pos = Math.min(cursor + inChunkSize, part.size);
        const slice = {
          key: part.key,
          bytes: part.bytes.slice(cursor, slice_end_pos),
          output_size: parseInt((slice_end_pos - cursor) * byteFactor), // needed in case we pad
          target_start_pos: parseInt((part.target_start_pos + cursor) * byteFactor)
          //target_end_pos: parseInt((part.target_start_pos + slice_end_pos) * byteFactor)
        }
        if (slice.bytes.length < inChunkSize) { // decompression kernel requires a constant input shape
          const padded = new Uint8Array(inChunkSize);
          padded.set(slice.bytes);
          slice.bytes = padded;
        }
        gpuJobs.push(scheduleDequantizeJob(slice));
      }
      await Promise.all(gpuJobs);
    } 
    else if (window.BACKEND === "WASM") {
      // TODO: this is probably broken with webgpu refactor, fix
      state_dict["output.weight"] = state_dict["tok_embeddings.weight"]; // buffer is the same; clang export code prioritized output.weight
      delete state_dict["tok_embeddings.weight"];

      function scheduleDequantizeJob(k, v) {
        // k, v are from the model's state_dict
        return (async () => {
          const pipeline = await getFreePipeline(pipelinePool);
          const new_v = await pipeline([k, v]);
          if (k.includes("feed_forward") || k.includes("attention.w")) {
            state_dict[k.replace("weight", "scale")] = {"dtype": "float32", "bytes": new_v.scale, "size": new_v.scale.length}
          }
          state_dict[k] = new_v;
          releasePipeline(pipeline, pipelinePool);
        })();
      }

      const cpuJobs = [];
      for (const [k, v] of Object.entries(state_dict)) {
        if (v.dtype === "Q6_K") {cpuJobs.push(scheduleDequantizeJob(k, v));}
      }
      await Promise.all(cpuJobs);
      pipelinePool.forEach(p => p.worker.terminate());

    } else {throw new Error(`window.BACKEND is ${window.BACKEND}, but must be WebGPU or WASM`)}

    //const t1 = performance.now();
    //console.log(`decompression elapsed seconds: ${(t1 - t0) / 1000}`)
  }
    */

  const valid_final_dtypes = new Set(["float32", "int8", "int32"]);
  const loadFileToStateDict = async(file) => {
    for (const part of file.parts) {
      if (part.empty) continue;
      part.bytes = (part.size === file.bytes.length) ? file.bytes : file.bytes.slice(part.file_start_pos, part.file_start_pos + part.size);
      if (valid_final_dtypes.has(part.dtype)) {
        new Uint8Array(state_dict[part.key].bytes.getMappedRange(part.target_start_pos, part.bytes.length)).set(part.bytes);
      }
      else throw new Error(`unexpected dtype: ${part.dtype} in file: ${file.name}`);
      part.bytes = null;
    }
    file.bytes = null;
    completed += 1;
  }

  while (completed < data.metadata.files.length) {
    // prioritize files from downloaded queue, so we can continue downloading more files
    if (downloaded.length) {
      const file = downloaded.shift();
      await Promise.all(deletionPromises); // maximize available IndexedDB cache; TODO: should we just await this once outside loop?
      saveTensorToDb(db, file.hash, file.bytes); // Promise, which we currently never await
      await loadFileToStateDict(file); // increments completed when done
    }
    else if (!downloaded.length && cachedFiles.length) {
      const file = cachedFiles.shift();
      file.bytes = await getPart(file.name, file.hash); // reads data from IndexedDB
      await loadFileToStateDict(file); // increments completed when done
    }
    await new Promise(resolve => setTimeout(resolve, 200));
  }

  for (const [k,v] of Object.entries(state_dict)) if (!v.empty) v.bytes.unmap();
  return model;
};

document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // loadingMessage updates the user on page load progress, including weights download and decompression
    // if loadingMessage is not '', then prompt box will be hidden: this is default behavior on page load
    loadingMessage: 'Loading...',
    // model
    nets: {},
    tokenizer: null,
    // TODO: implement context sliding; model currently outputs gibberish past max_context
    max_context: 1024,
    lastSeenToks: [],

    progress(loaded, total, message) {
      const percentage = total ? Math.trunc((loaded / total) * 100) : 0;
      document.querySelector('.progress').style.width = `${percentage}%`;
      document.getElementById('progress-percentage').textContent = `${percentage}%`;
      if (message) {
        this.loadingMessage = message;
        document.getElementById('loading-message').textContent = this.loadingMessage;
      }
    },

    async init() {
      var device = null;
      if (window.BACKEND === "WebGPU") {
        try {
          device = await getDevice();
          var modelPromise = load_state_dict(device, this.progress.bind(this));
          console.log("WebGPU device initialized");
        } catch (error) {
          this.progress(0, 100, "Failed to launch WebGPU. Loading WASM model instead...");
          window.BACKEND = "WASM";
          console.log(`error: ${error}\nFailed to launch WebGPU. Loading WASM model instead...`); // return;
        }
      }

      try {
        const placeholder = 1; // TODO: clean up this section, handle WASM
      } catch (error) {this.progress(0, 100, `Error decompressing model: ${error}`); console.log(error); return;}

      var p = 0;
      try {
        this.progress(p, 100, "Loading tokenizer:");
        const wasmResponse = await fetch(`${window.MODEL_BASE_URL}/tiktoken_bg.wasm`);
        p = 10; this.progress(p, 100, "Loading tokenizer:");
        const wasmBytes = await wasmResponse.arrayBuffer();
        await tiktokenReady;
        await window.tiktokenInit((imports) => WebAssembly.instantiate(wasmBytes, imports));
        p = 20; this.progress(p, 100, "Loading tokenizer:");

        this.tokenizer = await createTokenizer(`${window.MODEL_BASE_URL}/llama3-2.tiktoken`);
        const tokenizer_works = (new TextDecoder().decode(this.tokenizer.decode(this.tokenizer.encode("hello world"))) === "hello world");
        console.log("tokenizer works:", tokenizer_works)
        p = 30; this.progress(p, 100, "Loading tokenizer:");
      } catch (error) {this.progress(p, 100, `Error launching tokenizer: ${error}`); console.log(error); return;}

      try {
        p = 40; this.progress(p, 100, `Launching ${window.BACKEND} model:`);
        //await kernelsReady;
        if (window.BACKEND === "WebGPU") {
          //const model = await transformer().setup(device, state_dict, this.progress.bind(this));
          const model = await modelPromise;
          this.nets = {"transformer": model};
        }
        else if (window.BACKEND === "WASM") {
          const modelWorker = new Worker(`./worker.js?version=${Date.now()}`);
          let msg = await sendMessageToWorker(modelWorker, {header: "setup", data: "setup_transformer"});
          msg = await sendMessageToWorker(modelWorker, {header: "state_dict", data: state_dict});
          this.nets = {"transformer": async (tok, start_pos) => sendMessageToWorker(modelWorker, {header: "token", data: [tok, start_pos]})};
        }
        this.progress(100, 100, `Launching ${window.BACKEND} model:`);
        this.loadingMessage = ""; // Triggers removal of loading bar, display of prompt box
      } catch (error) {this.progress(p, 100, `Error launching model: ${error}`); console.log(error); return;}
    },

    // current state
    cstate: {
      time: null,
      messages: [],
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: `${window.location.origin}/v1`,

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    async handleSend() {
      const el = document.getElementById("input-form");
      const value = el.value.trim();
      if (!value) return;

      if (this.generating) return;
      // TODO: fix bug: if we switch to another chat session during generation, prompt bar locks up with "Generating..."
      this.generating = true;
      if (this.home === 0) this.home = 1;

      // ensure that going back in history will go back to home
      window.history.pushState({}, "", window.TINYCHAT_ROOT || "/");

      // add message to list
      this.cstate.messages.push({ role: "user", content: value });

      // clear textarea
      el.value = "";
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";

      // reset performance tracking
      const prefill_start = Date.now();
      let start_time = 0;
      let tokens = 0;
      this.tokens_per_second = 0;

      // start receiving server sent events
      let gottenFirstChunk = false;
      for await (
        const chunk of this.openaiChatCompletion(this.cstate.messages)
      ) {
        if (!gottenFirstChunk) {
          this.cstate.messages.push({ role: "assistant", content: "" });
          gottenFirstChunk = true;
        }

        // add chunk to the last message
        // TODO: handle errors with localStorage overflow
        //   possible example: this.cstate.messages[...] was undefined when trying to prompt within an old cstate (chat session)
        this.cstate.messages[this.cstate.messages.length - 1].content += chunk;

        // calculate performance tracking
        tokens += 1;
        this.total_tokens += 1;
        if (start_time === 0) {
          start_time = Date.now();
          this.time_till_first = start_time - prefill_start;
        } else {
          const diff = Date.now() - start_time;
          if (diff > 0) {
            this.tokens_per_second = tokens / (diff / 1000);
          }
        }
      }

      // update the state in histories or add it if it doesn't exist
      const index = this.histories.findIndex((cstate) => {
        return cstate.time === this.cstate.time;
      });
      this.cstate.time = Date.now();
      if (index !== -1) {
        // update the time
        this.histories[index] = this.cstate;
      } else {
        this.histories.push(this.cstate);
      }
      // update in local storage
      localStorage.setItem("histories", JSON.stringify(this.histories));

      this.generating = false;
    },

    async handleEnter(event) {
      // if shift is not pressed
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    updateTotalTokens(messages) {
      try {
        let toks = [this.tokenizer.bos_id];
        messages.forEach((message) => {
          if (!message.role || !message.content) {
            throw new Error("Each message must have a 'role' and 'content' property.");
          }
          toks = toks.concat(this.tokenizer.encodeMessage(message.role, message.content));

          if (messages.length > 0 && messages[messages.length - 1].role === "user") {
            toks = toks.concat(this.tokenizer.encodeRole("assistant"));
          }
          this.total_tokens = toks.length;
        });
      } catch (error) {
        console.error("Error updating total tokens:", error);
      }
    },

    async *openaiChatCompletion(messages) {
      let tokens = [this.tokenizer.bos_id];
      for (const message of messages) {
        tokens = tokens.concat(this.tokenizer.encodeMessage(message.role, message.content));
      }
      tokens = tokens.concat(this.tokenizer.encodeRole("assistant"));
      let startPos = 0
      let prefillToks = tokens.slice(0, -1);

      // Skip the largest possible sequence of tokens already represented at the beginning of the model's kv caches
      for (let i=0; i <= prefillToks.length; i++) {
        startPos = i;
        if (i == prefillToks.length) break;
        if (i == this.lastSeenToks.length) break;
        if (prefillToks[i] !== this.lastSeenToks[i]) break;
      }
      this.lastSeenToks = prefillToks;
      prefillToks = prefillToks.slice(startPos);

      for (const tok of prefillToks) {
        if (window.BACKEND === "WebGPU") {await this.nets["transformer"](new Int32Array([tok]), new Int32Array([startPos]));}
        else {await this.nets["transformer"](tok, startPos);}
        startPos += 1;
      }

      let lastTok = tokens[tokens.length - 1];
      while (true) {
        if (window.BACKEND === "WebGPU") {var tok = await this.nets["transformer"](new Int32Array([lastTok]), new Int32Array([startPos])); tok = tok[0];}
        else {var tok = await this.nets["transformer"](lastTok, startPos);}
        this.lastSeenToks.push(lastTok); // lets us skip prefilling with these tokens at the next prompt in this chain
        startPos += 1;
        lastTok = tok;
        if (this.tokenizer.stop_tokens.has(lastTok)) break;
        yield new TextDecoder().decode(this.tokenizer.decode([lastTok]));
      }
    },
  }));
});

const { markedHighlight } = globalThis.markedHighlight;
marked.use(markedHighlight({
  langPrefix: "hljs language-",
  highlight(code, lang, _info) {
    const language = hljs.getLanguage(lang) ? lang : "plaintext";
    return hljs.highlight(code, { language }).value;
  },
}));

// **** eventsource-parser ****
class EventSourceParserStream extends TransformStream {
  constructor() {
    let parser;

    super({
      start(controller) {
        parser = createParser((event) => {
          if (event.type === "event") {
            controller.enqueue(event);
          }
        });
      },

      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

function createParser(onParse) {
  let isFirstChunk;
  let buffer;
  let startingPosition;
  let startingFieldLength;
  let eventId;
  let eventName;
  let data;
  reset();
  return {
    feed,
    reset,
  };
  function reset() {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;
    eventId = void 0;
    eventName = void 0;
    data = "";
  }
  function feed(chunk) {
    buffer = buffer ? buffer + chunk : chunk;
    if (isFirstChunk && hasBom(buffer)) {
      buffer = buffer.slice(BOM.length);
    }
    isFirstChunk = false;
    const length = buffer.length;
    let position = 0;
    let discardTrailingNewline = false;
    while (position < length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") {
          ++position;
        }
        discardTrailingNewline = false;
      }
      let lineLength = -1;
      let fieldLength = startingFieldLength;
      let character;
      for (
        let index = startingPosition;
        lineLength < 0 && index < length;
        ++index
      ) {
        character = buffer[index];
        if (character === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (character === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (character === "\n") {
          lineLength = index - position;
        }
      }
      if (lineLength < 0) {
        startingPosition = length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }
      parseEventStreamLine(buffer, position, fieldLength, lineLength);
      position += lineLength + 1;
    }
    if (position === length) {
      buffer = "";
    } else if (position > 0) {
      buffer = buffer.slice(position);
    }
  }
  function parseEventStreamLine(lineBuffer, index, fieldLength, lineLength) {
    if (lineLength === 0) {
      if (data.length > 0) {
        onParse({
          type: "event",
          id: eventId,
          event: eventName || void 0,
          data: data.slice(0, -1),
          // remove trailing newline
        });

        data = "";
        eventId = void 0;
      }
      eventName = void 0;
      return;
    }
    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(
      index,
      index + (noValue ? lineLength : fieldLength),
    );
    let step = 0;
    if (noValue) {
      step = lineLength;
    } else if (lineBuffer[index + fieldLength + 1] === " ") {
      step = fieldLength + 2;
    } else {
      step = fieldLength + 1;
    }
    const position = index + step;
    const valueLength = lineLength - step;
    const value = lineBuffer.slice(position, position + valueLength).toString();
    if (field === "data") {
      data += value ? "".concat(value, "\n") : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!Number.isNaN(retry)) {
        onParse({
          type: "reconnect-interval",
          value: retry,
        });
      }
    }
  }
}
const BOM = [239, 187, 191];
function hasBom(buffer) {
  return BOM.every((charCode, index) => buffer.charCodeAt(index) === charCode);
}

const PAT_STR = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

async function createTokenizer(bpeUrl) {
  const num_base_tokens = 128000;
  const special_tokens = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009
  };
  const model = await window.tiktokenLoad({
        "load_tiktoken_bpe": bpeUrl,
        "special_tokens": special_tokens,
        "pat_str": PAT_STR
    });
  const tokenizer = new window.Tiktoken(model.bpe_ranks, model.special_tokens, model.pat_str)

  return {
    get bos_id() {
      return special_tokens["<|begin_of_text|>"];
    },

    get stop_tokens() {
      return new Set([
        special_tokens["<|end_of_text|>"],
        special_tokens["<|eot_id|>"],
      ]);
    },

    decode(toks) {
      const filtered = toks.filter((t) => t < num_base_tokens);
      return tokenizer.decode(filtered);
    },

    encode(text, allow_special = false) {
      const allowedSpecial = allow_special ? "all" : new Set();
      const disallowedSpecial = new Set();
      return tokenizer.encode(text, allowedSpecial, disallowedSpecial);
    },

    encodeRole(role) {
      const tokens = [];
      tokens.push(special_tokens["<|start_header_id|>"]);
      tokens.push(...this.encode(role));
      tokens.push(special_tokens["<|end_header_id|>"]);
      tokens.push(...this.encode("\n\n"));
      return tokens;
    },

    encodeMessage(role, content) {
      const roleTokens = this.encodeRole(role);
      const contentTokens = this.encode(content.trim());
      return [...roleTokens, ...contentTokens, special_tokens["<|eot_id|>"]];
    },
  };
}