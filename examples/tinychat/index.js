
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

const getAndDecompressGGUFChunks = async (decomp, progress) => {
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
                  progressCallback(part, value.byteLength, total, "Downloading model:");
                  controller.enqueue(value);
              }
                    
              controller.close();
          },
      }));
        
      return res.arrayBuffer();
  };

  const response = await fetch(`${window.MODEL_BASE_URL}/net_metadata.json`);
  // TODO: cache metadata
  const data = await response.json();
  const state_dict = data.metadata.state_dict;

  let db = await initDb();

  // TODO: add progress tracker
  const getPart = async(filename, hash) => {
    let part = await readTensorFromDb(db, hash);

    if (part) {
      console.log(`Cache hit: ${filename}, hash: ${hash}`);
      return Promise.resolve(part.content);
    } else {
      console.log(`Cache miss: ${filename}, hash: ${hash}`);
      return loadPart(`${window.MODEL_BASE_URL}/${filename}`, progressCallback);
    }
  }

  const correctHashes = data.metadata.chunks.map(chunk => chunk.hash)
  const compressedBuffers = await Promise.all(data.metadata.chunks.map(chunk => getPart(chunk.name, chunk.hash)));

  // delete unused cached buffers to free disk space -- if we update weights, user will otherwise have obsolete cached buffers
  const dbKeys = await getAllKeysFromDb(db);
  const correctHashesSet = new Set(correctHashes);
  const notInCorrectHashes = dbKeys.filter(key => !correctHashesSet.has(key));
  for (const hash of notInCorrectHashes) {deleteTensorFromDb(db, hash);}

  for (let i = 0; i < compressedBuffers.length; i++) {
    compressedBuffers[i] = new Uint8Array(compressedBuffers[i]);
    // check integrity of buffers, replace invalid cached buffers
    // may not be necessary, and takes time
    checked_hash = await hashBuffer(compressedBuffers[i]);
    if (checked_hash !== correctHashes[i]) {
      console.log(`Replacing invalid buffer with name: ${data.metadata.chunks[i].name}, expected hash: ${correctHashes[i]}, actual hash: ${checked_hash}`)
      await deleteTensorFromDb(db, correctHashes[i]);
      compressedBuffers[i] = await getPart(data.metadata.chunks[i].name, correctHashes[i]);
      compressedBuffers[i] = new Uint8Array(compressedBuffers[i]);
    }
    saveTensorToDb(db, correctHashes[i], compressedBuffers[i]);
  }

  console.log("Compressed model chunks loaded");

  totalLoaded = 0;
  totalSize = Object.values(state_dict).filter(item => item.dtype === "Q6_K").reduce((sum, item) => sum + item.size, 0) / 430080;
  const numCheckpoints = 100;
  let nextCheckpoint = totalSize / numCheckpoints;

  const dequantize = async(parent, decomp, BYTES_PER_CHUNK_IN, FLOATS_PER_CHUNK_OUT) => {
    if (parent.length % BYTES_PER_CHUNK_IN !== 0) {
      throw new Error("Parent length must be a multiple of BYTES_PER_CHUNK_IN bytes.");
    }
    const numChunks = parent.length / BYTES_PER_CHUNK_IN;
    const BYTES_PER_CHUNK_OUT = FLOATS_PER_CHUNK_OUT * 4;
    const inputPtr = decomp._malloc(BYTES_PER_CHUNK_IN);
    const outputPtr = decomp._malloc(BYTES_PER_CHUNK_OUT);
    const inputView = new Uint8Array(decomp.HEAPU8.buffer, inputPtr, BYTES_PER_CHUNK_IN);
    const outputViewF32 = new Float32Array(decomp.HEAPF32.buffer, outputPtr, FLOATS_PER_CHUNK_OUT);
    const outputViewU8 = new Uint8Array(outputViewF32.buffer, outputViewF32.byteOffset, outputViewF32.byteLength);
    const result = new Uint8Array(numChunks * BYTES_PER_CHUNK_OUT);

    for (let i = 0; i < numChunks; i++) {
      const start = i * BYTES_PER_CHUNK_IN;
      const end   = start + BYTES_PER_CHUNK_IN;
      inputView.set(parent.subarray(start, end));
      decomp._net(inputPtr, outputPtr);
      const offset = i * BYTES_PER_CHUNK_OUT;
      result.set(outputViewU8, offset);

      totalLoaded += 1;
      if (totalLoaded >= nextCheckpoint) {
        progress(totalLoaded, totalSize, "Decompressing model:");
        nextCheckpoint += totalSize / numCheckpoints;
      }

      // prevent browser lag 
      // TODO: use workers
      if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));
    }
    decomp._free(inputPtr);
    decomp._free(outputPtr);

    return result;
  }

  for (const [k, v] of Object.entries(state_dict)) {
    v.bytes = compressedBuffers[v.chunk].subarray(v.start_pos, v.start_pos + v.size);
    if (v.dtype === "Q6_K") {
      console.log(`decompressing ${k}`)
      v.bytes = await dequantize(v.bytes, decomp, 430080, 524288);
      v.dtype = "float32";
      v.size = v.bytes.byteLength;
    }
  }
  console.log("Decompression complete")

  // FFD bin packing
  const maxChunkSize = 1149173760; // byte size of float32 output.weight in llama-1B
  const chunks = [];
  const size_sorted_tensors = Object.entries(state_dict)
    .map(([key, value]) => ({name: key, size: value.size}))
    .sort((a, b) => b.size - a.size);
  for (const t of size_sorted_tensors) {
    let placed = false;
    for (const chunk of chunks) {
      const currentSum = chunk.reduce((sum, i) => sum + i.size, 0);
      if (currentSum + t.size <= maxChunkSize) {
        chunk.push(t);
        placed = true;
        break;
      }
    }
    if (!placed) chunks.push([t]);
  }
  console.log("binning complete");

  const decompressedBuffers = [];
  for (let i=0; i<chunks.length; i++) {
    const chunk = chunks[i];
    const chunkSize = chunk.reduce((sum, j) => sum + j.size, 0);
    decompressedBuffers.push(new Uint8Array(chunkSize));
    let cursor = 0;
    for (let j=0; j<chunk.length; j++) {
      const t = chunk[j];
      decompressedBuffers[i].set(state_dict[t.name].bytes, cursor);
      state_dict[t.name].bytes = null;
      state_dict[t.name].chunk = i;
      state_dict[t.name].start_pos = cursor;
      cursor += t.size;
      if (j % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0)); // prevent browser lag
    }
  }

  console.log("chunking complete");
  return {chunks: decompressedBuffers, metadata: state_dict};
};

document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // loadingMessage updates the user on page load progress, including weights download and decompression
    // if loadingMessage is not '', then prompt box will be hidden: this is default behavior on page load
    loadingMessage: 'Loading...',
    // model
    nets: {},
    tokenizer: null,
    max_context: 1024,

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
      try {
        const q6k_to_f32 = await Module();
        const tensorData = await getAndDecompressGGUFChunks(q6k_to_f32, this.progress.bind(this));

        const wasmResponse = await fetch(`${window.MODEL_BASE_URL}/tiktoken_bg.wasm`);
        const wasmBytes = await wasmResponse.arrayBuffer();
        await window.tiktokenInit((imports) => WebAssembly.instantiate(wasmBytes, imports));

        //this.tokenizer = await createTokenizer("./llama3-2.tiktoken");
        this.tokenizer = await createTokenizer(`${window.MODEL_BASE_URL}/llama3-2.tiktoken`);
        tokenizer_works = (new TextDecoder().decode(this.tokenizer.decode(this.tokenizer.encode("hello world"))) === "hello world");
        console.log("tokenizer works:", tokenizer_works)

        const device = await getDevice();
        console.log("WebGPU device initialized")

        let models = ["transformer"];
        this.nets = await Promise.all([
                transformer().setup(device, tensorData.chunks, tensorData.metadata),
            ]).then((loadedModels) => loadedModels.reduce((acc, model, index) => { acc[models[index]] = model; return acc; }, {}))
        console.log("Transformer setup without exceptions");
        this.loadingMessage = ""; // Triggers removal of loading bar, display of prompt box
      } catch (error) {
        console.error("Error initializing model:", error);
      }
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
      this.generating = true;
      if (this.home === 0) this.home = 1;

      // ensure that going back in history will go back to home
      window.history.pushState({}, "", "/");

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
      let start_pos = 0

      let prefill_toks = tokens.slice(0, -1);

      // TODO: implement whole llama3.py prefill method, here we are missing the prompt skipping logic
      for (const tok of prefill_toks) {
        const out = await this.nets["transformer"](new Float32Array([[tok]]), start_pos);
        start_pos += 1;
      }

      let last_tok = tokens[tokens.length - 1];
      while (true) {
        const tok = await this.nets["transformer"](new Float32Array([[last_tok]]), start_pos);
        start_pos += 1;
        last_tok = tok[0];
        if (this.tokenizer.stop_tokens.has(last_tok)) break;
        yield new TextDecoder().decode(this.tokenizer.decode([last_tok]));
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