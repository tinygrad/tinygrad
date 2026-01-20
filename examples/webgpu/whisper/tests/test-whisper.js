import { create, globals } from 'webgpu';
import { AudioContext, OfflineAudioContext } from 'node-web-audio-api';
import '../.node_env/node_modules/fake-indexeddb/auto/index.mjs';

const {default: mel} = await import("../mel.js");
const {default: encoder} = await import("../encoder.js");
const {default: decoder} = await import("../decoder.js");

Object.assign(globalThis, globals);

const navigator = { gpu: create([]) };

// #region imports
import {
  tensorStore,
  initDb,

  getDevice,

  fetchMonoFloat32Array,
  getProgressDlForPart,

  transcribeAudio,
  tokensToText,
  format_text,
  format_text_helper
} from "../whisper.js";
// #endregion imports

// #region limits_keys
const LIMITS_KEYS = ["maxTextureDimension1D",
"maxTextureDimension2D",
"maxTextureDimension3D",
"maxTextureArrayLayers",
"maxBindGroups",
"maxBindingsPerBindGroup",
"maxDynamicUniformBuffersPerPipelineLayout",
"maxDynamicStorageBuffersPerPipelineLayout",
"maxSampledTexturesPerShaderStage",
"maxSamplersPerShaderStage",
"maxStorageBuffersPerShaderStage",
"maxStorageTexturesPerShaderStage",
"maxUniformBuffersPerShaderStage",
"maxUniformBufferBindingSize",
"maxStorageBufferBindingSize",
"minUniformBufferOffsetAlignment",
"minStorageBufferOffsetAlignment",
"maxVertexBuffers",
"maxBufferSize",
"maxVertexAttributes",
"maxVertexBufferArrayStride",
"maxInterStageShaderVariables",
"maxColorAttachments",
"maxColorAttachmentBytesPerSample",
"maxComputeWorkgroupStorageSize",
"maxComputeInvocationsPerWorkgroup",
"maxComputeWorkgroupSizeX",
"maxComputeWorkgroupSizeY",
"maxComputeWorkgroupSizeZ",
"maxComputeWorkgroupsPerDimension"];
// #endregion limits_keys

const device = await getDevice(navigator.gpu);

const WASM_ARGSORT = false;

if (WASM_ARGSORT) {
  globalThis.argsort = await (async () => {
  // paste base64 string here (shortened in example)
  const wasmBase64 = "AGFzbQEAAAABBgFgAn9/AAIPAQNlbnYGbWVtb3J5AgACAwIBAAQFAXABAQEGCAF/AUGAiAQLBwsBB2FyZ3NvcnQAAArjBAHgBAIFfwF9QQAhAkEAIQMDQCABIAJqIgQgAzYCACAEQRxqIANBB2o2AgAgBEEYaiADQQZqNgIAIARBFGogA0EFajYCACAEQRBqIANBBGo2AgAgBEEMaiADQQNqNgIAIARBCGogA0ECajYCACAEQQRqIANBAWo2AgAgAkEgaiECIANBCGoiA0GYlQNHDQALQcvKASEDA0AgAyIFQQF0IgJBAXIhAyAAIAEgBUECdGooAgAiBkECdGoqAgAhByAFIQQDQCADIAQgByAAIAEgA0ECdGooAgBBAnRqKgIAXRshAwJAIAJBAmoiAkGXlQNKDQAgACABIANBAnRqKAIAQQJ0aioCACAAIAEgAkECdGooAgBBAnRqKgIAXUUNACACIQMLAkAgAyAERg0AIAEgBEECdGogASADQQJ0aiIEKAIANgIAIAQgBjYCACADIQQgA0EBdCICQQFyIgNBmJUDSA0BCwsgBUF/aiEDIAUNAAtBl5UDIQICQANAIAEgAkECdGoiAygCACEFIAMgASgCADYCACABIAU2AgAgAkECSQ0BIAAgBUECdGoqAgAhB0EAIQZBASEDQQAhBANAIAMgBCAHIAAgASADQQJ0aigCAEECdGoqAgBdGyEDAkAgBkECaiIGIAJODQAgACABIANBAnRqKAIAQQJ0aioCACAAIAEgBkECdGooAgBBAnRqKgIAXUUNACAGIQMLAkAgAyAERg0AIAEgBEECdGogASADQQJ0aiIEKAIANgIAIAQgBTYCACADIQQgA0EBdCIGQQFyIgMgAkgNAQsLIAJBf2ohAgwACwsLACUEbmFtZQEKAQAHYXJnc29ydAcSAQAPX19zdGFja19wb2ludGVyACYJcHJvZHVjZXJzAQxwcm9jZXNzZWQtYnkBBWNsYW5nBjE3LjAuMQAsD3RhcmdldF9mZWF0dXJlcwIrD211dGFibGUtZ2xvYmFscysIc2lnbi1leHQ="; // contents of wasm.b64

  const wasmBytes = Uint8Array.from(atob(wasmBase64), c => c.charCodeAt(0));
  const memory = new WebAssembly.Memory({ initial: 100 });
  const { instance } = await WebAssembly.instantiate(wasmBytes, { env: { memory } });
  // const argsort = instance.exports.argsort;
  // allocate offsets inside memory buffer (choose some offset, e.g. 0)
  // For production, maintain an allocator. Here: use offset 0 for inputs and N*4 for outputs.
  const N = 51864; // must match compile-time N
  const bytesPerFloat = 4, bytesPerInt = 4;
  //const buf = new Uint8Array(memory.buffer);

  const aOffset = 0;
  const outOffset = aOffset + N * bytesPerFloat;

  const f32 = new Float32Array(memory.buffer, aOffset, N);
  const i32 = new Int32Array(memory.buffer, outOffset, N);

  return function (array) {
    f32.set(array);
    instance.exports.argsort(aOffset, outOffset);
    return i32.slice().reverse();
  }
})();
}

const db = await initDb();

const modules = [
  ["mel", "./mel.js"],
  ["encoder", "./encoder.js"],
  ["decoder", "./decoder.js"],
];

// console.log(process.argv);

const BASE_URL = 'http://localhost:8000';
// const AUDIO_PATH = 'RED_16k.wav';
// const AUDIO_PATH = 'RED_60s.wav';
// const AUDIO_PATH = 'RED_you.wav';
// const AUDIO_PATH = 'test.wav';
// const AUDIO_PATH = 'test2.wav';
const AUDIO_PATH = process.argv[2] ? process.argv[2] : 'TINYCORP_MEETING_2025-07-07-DSWQCT9mypQ.mp3';
// const AUDIO_PATH = `${BASE_URL}/TINYCORP_MEETING_2025-08-25-KA0h9zmJtcs.mp3`;

const getPart = async (key) => {
  let full_url = `${BASE_URL}/${key}.safetensors`;
  const cached = await tensorStore(db).get(key);

  const download = await getProgressDlForPart(full_url, (() => { }), cached?.lastModified);

  if (download === null && cached) {
    console.log(`Cache hit: ${key}`);
    return cached.content;
  }
  {
    console.log(`Cache refresh: ${key}`);
    await tensorStore(db).put(key, download.buffer, download.lastModified);
    return download.buffer;
  }
}

let nets = {};
nets.mel = await mel.setupNet(device, new Uint8Array(await getPart("mel")));
// nets.mel = await mel.default.setupNet(device, new Uint8Array((await (await fetch(`${BASE_URL}/mel.safetensors`)).arrayBuffer())));
nets.encoder = await encoder.setupNet(device, new Uint8Array(await getPart("encoder")));
nets.decoder = await decoder.setupNet(device, new Uint8Array(await getPart("decoder")));

const mapping = await fetch(`${BASE_URL}/vocab.json`).then(res => res.json());
const model_metadata = await fetch(`${BASE_URL}/model_metadata.json`).then(res => res.json());
nets.mapping = mapping;
nets.model_metadata = model_metadata;

let currentCancel = null;


function onTranscriptionEvent(event, data) {
  if (event === "cancel") {

  } else if (event === "chunkUpdate") {
    // console.log(data.pendingTexts);
    // console.log(data.sequences);
    for (let [idx, seq] of data.sequences.entries()) {
      if (seq === undefined) continue;
      let text = format_text_helper(seq.tokens, nets.mapping, seq.seek, seq.seek_end);
      console.log(text);
      // console.log(tokensToText(seq.tokens))
    }
  } else if (event === "chunkDone") {
    // console.log(data.segment_cumlogprob.toFixed(2) + ' ' + data.pendingText);
  }
}

currentCancel = { cancelled: false };
await transcribeAudio(nets, async () => await fetchMonoFloat32Array(`${BASE_URL}/${AUDIO_PATH}`, AudioContext), currentCancel, onTranscriptionEvent, async () => {});
console.log("we're supposed to be done here");

delete globalThis.mel;
delete globalThis.encoder;
delete globalThis.decoder;
delete globalThis.nets;
globalThis.mel = null;
globalThis.encoder = null;
globalThis.decoder = null;
globalThis.nets = null;
nets = null;

await device.queue.onSubmittedWorkDone();
device.destroy?.();

db.destroy?.();
delete globalThis.db;
globalThis.db = null;

delete globalThis.device;
globalThis.device = null;
console.log("deleted everything but still here");
delete globalThis.globals;
globalThis.globals = null;
delete globalThis.navigator;
globalThis.navigator = null;
// console.log(process._getActiveRequests());
// console.log(process._getActiveHandles());