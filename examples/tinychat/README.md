# How to reproduce tinychat in browser (WEBGPU)

These steps were done on ubuntu 24.04:

- with your tinygrad env, from repo root dir, run `PYTHONPATH=. python examples/tinychat/compile.py`
    - the first time this script is run, it will convert the quantized weights to float32 (float16 would be good, but the python wgpu adapter doesn't support float16 yet). The script then exits to free idly-consumed GPU memory
    - Run `PYTHONPATH=. python examples/tinychat/compile.py` a second time to load the float32 weights to WEBGPU and commence compilation
- the below files will be output to `examples/tinychat`:
    - float-32 llama v 3.2 1B weights, split into multiple such chunks: `net_part0.safetensors`, etc.
    - `llama3-2.tiktoken`: used for the tiktoken.js encoder (see below)
    - `net.js`: compiled llama model for webgpu in javascript
- obtain `tiktoken.js` and `tiktoken_bg.wasm` for encoding/decoding tokens in the browser:
    - run the `make_tiktoken_js.sh` script from `tinygrad/examples/tinychat`
    - this script depends on npm and webpack to download the tiktoken node module and bundle it, emitting `tiktoken.js` and `tiktoken_bg.wasm` which are later ingested by `index.html`
    - `tiktoken-export.js` and `webpack.config.js` are used during the above compilation
- launch browser with `google-chrome --enable-features=Vulkan --enable-unsafe-webgpu`
    - navigate to `chrome://gpu`, ensure that you see `Vulkan: Enabled` and `WebGPU: Hardware accelerated`
- serve the index.html: from `tinygrad/examples/tinychat`, activate a python env, run `python -m http.server 7776`
- in the webgpu-enabled chrome browser you launched above, navigate to `localhost:7776`
- to monitor loading, inspect the console (F12 on ubuntu):
    - `tokenizer works: true` will be output if tiktoken.js was set up properly above
    - `Transformer setup without exceptions` `true` will be output if webgpu loads the model without issues
- use app as normal


# Compiling wasm files

To compile `q6k_to_f32.js` and `q6k_to_f32.wasm`:

- install and activate emscripten
- from the tinychat dir run `emcc q6k_to_f32.c -o q6k_to_f32.js -s MODULARIZE=1 -s EXPORTED_FUNCTIONS='["_net", "_malloc", "_free"]' -s EXPORTED_RUNTIME_METHODS='["cwrap", "getValue", "setValue"]'`

# Hosting

The following files can be hosted at `window.MODEL_BASE_URL`, separately from the index.html location:

- `llama3-2.tiktoken`
- `tiktoken_bg.wasm`
- `net_partx.gguf.chunk` where x = 0, 1
- `net_metadata.json`
