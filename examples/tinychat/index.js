window.TINYCHAT_ROOT = "/";
window.MODEL_BASE_URL= ".";
const queryParams = new URLSearchParams(window.location.search);
const normalizedParams = Object.fromEntries([...queryParams].map(([key, value]) => [key.toUpperCase(), value.toUpperCase()]));
window.BACKEND = (normalizedParams["BACKEND"] === "WASM") ? "WASM" : "WebGPU";
const isMobileAgent = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const hasTouchScreen = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
window.isMobile = isMobileAgent || hasTouchScreen;
if (window.isMobile) document.documentElement.classList.add('mobile'); // prevent annoying auto-zoom when entering prompt on mobile

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
  const maxBufferSize = 322122544;
  requiredLimits.maxStorageBufferBindingSize = maxBufferSize;
  requiredLimits.maxBufferSize = maxBufferSize;
            
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

function makeProgress(total) {
  let acc = 0;
  const ret = function progress(amount, message) {
    if (amount >= 0) { // allow updating message only
      acc += amount;
      const percentage = total ? Math.trunc((acc / total) * 100) : 0;
      document.querySelector('.progress').style.width = `${percentage}%`;
      document.getElementById('progress-percentage').textContent = `${percentage}%`;
    }
    if (message) {
      this.loadingMessage = message;
      document.getElementById('loading-message').textContent = this.loadingMessage;
    }
  }.bind(this);
  ret.total = total;
  return ret;
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
    else if (message.header === "init_state_dict") worker.postMessage(Object.fromEntries(Object.entries(message.data).filter(([_, v]) => !v.empty)));
    else if (message.header === "load_part") worker.postMessage(message.data, message.data === "done" ? [] : [message.data.bytes.buffer]);
  });
}

async function load_state_dict (data, device, progress) {
  let state_dict = data.metadata.state_dict;
  let completed = 0;

  // modified from examples/webgpu/stable_diffusion/index.html getProgressDlForPart
  const loadPart = async (part) => {
      const response = await fetch(part);
      const res = new Response(new ReadableStream({
          async start(controller) {
              const reader = response.body.getReader();
              for (;;) {
                  const { done, value } = await reader.read();
                  if (done) break;
                  progress(value.byteLength, `Loading model:`);
                  controller.enqueue(value);
              }
              controller.close();
          },
      }));
        
      return res.arrayBuffer();
  };

  let db = await initDb();

  const getPart = async(filename, hash) => {
    let part = await readTensorFromDb(db, hash);

    if (part) {
      console.log(`Cache hit: ${filename}, hash: ${hash}`);
      progress(part.content.byteLength, `Loading model:`)
      return Promise.resolve(part.content);
    } else {
      console.log(`Cache miss: ${filename}, hash: ${hash}`);
      return loadPart(`${window.MODEL_BASE_URL}/${filename}`);
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

  const cachedFileHashes = new Set(dbKeys.filter(key => correctHashesSet.has(key)));
  const cachedFiles = data.metadata.files.filter(file => cachedFileHashes.has(file.hash));
  const toDownload = data.metadata.files.filter(file => !cachedFileHashes.has(file.hash));
  const downloaded = [];
  // to limit memory overhead, we pause downloads if we have this number of downloaded files waiting to be processed
  const numDownloaders = window.isMobile ? 2 : toDownload.length; // TODO: dynamically base this on DL file size? current assumption is 16 MiB chunks
  const chainDownload = async (file) => {
    loadPart(`${window.MODEL_BASE_URL}/${file.name}`) // triggers download
    .then(async (arraybuf) => { 
      downloaded.push({ ...file, bytes: new Uint8Array(arraybuf)});
      // pause downloads if further processing is a bottleneck
      while (toDownload.length && downloaded.length >= numDownloaders) await new Promise(resolve => setTimeout(resolve, 200));
      if (toDownload.length && downloaded.length < numDownloaders) chainDownload(toDownload.shift()); // start next download
    })
  }
  for (let i=0; i<numDownloaders; i++) if (toDownload.length) chainDownload(toDownload.shift());

  await kernelsReady;
  // instantiates empty weight buffers on WebGPU, attaches buffers to state_dict
  let model;
  if (window.BACKEND === "WebGPU") {
    model = await transformer().setup(device, state_dict, progress);
  }
  else if (window.BACKEND === "WASM") {
    progress(0.02 * progress.total, 'Loading model:');
    model = new Worker(`./worker.js?version=${Date.now()}`);
    progress(0.02 * progress.total, 'Loading model:');
    state_dict = await sendMessageToWorker(model, {header: "init_state_dict", data: state_dict});
    progress(0.11 * progress.total, 'Loading model:');
  }

  const valid_final_dtypes = new Set(["float32", "int8", "int32"]);
  const loadFileToStateDict = async(file) => {
    for (const part of file.parts) {
      if (part.empty) continue;
      part.bytes = (part.size === file.bytes.length) ? file.bytes : file.bytes.slice(part.file_start_pos, part.file_start_pos + part.size);
      if (valid_final_dtypes.has(part.dtype)) {
        if (window.BACKEND === "WebGPU") {
          device.queue.writeBuffer(state_dict[part.key].bytes, part.target_start_pos, part.bytes); // improves stability over mappedAtCreation writing
        }
        else if (window.BACKEND === "WASM") {
          part.target_start_pos = state_dict[part.key].wasm_buf_start_pos + part.target_start_pos
          const msg = await sendMessageToWorker(model, {header: "load_part", data: part});
        }
      }
      else throw new Error(`unexpected dtype: ${part.dtype} in file: ${file.name}`);
      part.bytes = null;
    }
    file.bytes = null;
    completed += 1;
  }

  const loadDelay = window.isMobile ? 100 : 20 // hoping to improve stability on mobile
  await Promise.all(deletionPromises);
  while (completed < data.metadata.files.length) {
    // prioritize files from downloaded queue, so we can continue downloading more files
    if (downloaded.length) {
      const file = downloaded.shift();
      await saveTensorToDb(db, file.hash, file.bytes); // prevent race between indexedDB and wasm
      await loadFileToStateDict(file); // increments completed when done
    }
    else if (!downloaded.length && cachedFiles.length) {
      const file = cachedFiles.shift();
      file.bytes = await getPart(file.name, file.hash); // reads data from IndexedDB
      await loadFileToStateDict(file); // increments completed when done
    }
    await new Promise(resolve => setTimeout(resolve, loadDelay));
  }

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

    progress: null,

    async init() {
      var device = null;
      if (window.BACKEND === "WebGPU") {
        try {
          device = await getDevice();
          console.log("WebGPU device initialized");
        } catch (error) {
          //this.progress(0, "Failed to launch WebGPU. Loading WASM model instead...");
          window.BACKEND = "WASM";
          console.log(`error: ${error}\nFailed to launch WebGPU. Loading WASM model instead...`); // return;
        }
      }

      const response = await fetch(`${window.MODEL_BASE_URL}/net_metadata.json`);
      // TODO: cache metadata (and everything else) so tinychat works offline
      const data = await response.json();
      const state_dict = data.metadata.state_dict;
      let totalSize = 0;
      for (let [k,v] of Object.entries(state_dict)) {
        for (const part of v.parts) {
          if (part.empty) state_dict[k].empty = true; // assumes no other parts of this weight exist and are non-empty
          else {
            totalSize += part.size;
            part.key = k;
            part.dtype = v.dtype;
            if (!data.metadata.files[part.file].parts) data.metadata.files[part.file].parts = [];
            data.metadata.files[part.file].parts.push(part);
          }
        }
      }
      totalSize = totalSize / 0.8; // give space in progress bar for initializing model bufs, and tokenizer
      this.progress = makeProgress.call(this, totalSize); // creates closure with totalSize

      try {
        this.progress(0.01 * totalSize, "Loading tokenizer:");
        const wasmResponse = await fetch(`${window.MODEL_BASE_URL}/tiktoken_bg.wasm`);
        this.progress(0.01 * totalSize, "Loading tokenizer:");
        const wasmBytes = await wasmResponse.arrayBuffer();
        await tiktokenReady;
        await window.tiktokenInit((imports) => WebAssembly.instantiate(wasmBytes, imports));
        this.progress(0.01 * totalSize, "Loading tokenizer:");

        this.tokenizer = await createTokenizer(`${window.MODEL_BASE_URL}/llama3-2.tiktoken`);
        const tokenizer_works = (new TextDecoder().decode(this.tokenizer.decode(this.tokenizer.encode("hello world"))) === "hello world");
        console.log("tokenizer works:", tokenizer_works)
        this.progress(0.01 * totalSize, "Loading tokenizer:");
      } catch (error) {this.progress(-1, `Error launching tokenizer: ${error}`); console.log(error); return;}

      try {
        const model = await load_state_dict(data, device, this.progress);

        if (window.BACKEND === "WebGPU") {
          this.nets = {"transformer": model};
        }
        else if (window.BACKEND === "WASM") {
          const msg = await sendMessageToWorker(model, {header: "load_part", data: "done"});
          this.nets = {"transformer": async (tok, start_pos) => sendMessageToWorker(model, {header: "token", data: [tok, start_pos]})};
        }
        this.progress(0.01 * totalSize, `Launching ${window.BACKEND} model:`);
        this.loadingMessage = ""; // Triggers removal of loading bar, display of prompt box
      } catch (error) {this.progress(-1, `Error launching model: ${error}`); console.log(error); return;}
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