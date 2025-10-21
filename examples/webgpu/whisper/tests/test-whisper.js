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
  SAMPLES_PER_SEGMENT, MEL_SPEC_CHUNK_LENGTH, TOK_EOS,

  tensorStore,
  initDb,

  getDevice,

  fetchMonoFloat32Array,
  getProgressDlForPart,

  inferLoop
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

const BASE_URL = 'http://localhost:8000';
// const AUDIO_PATH = 'RED_16k.wav';
const AUDIO_PATH = 'RED_60s.wav';
// const AUDIO_PATH = 'test.wav';
// const AUDIO_PATH = 'test2.wav';
// const AUDIO_PATH = 'TINYCORP_MEETING_2025-07-07-DSWQCT9mypQ.mp3';
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

let currentCancel = null;
async function transcribeAudio(audioFetcher, cancelToken, onEvent, mapping) {
  let before = performance.now();
  // await loadAndInitializeModels();
  const { sampleRate, samples } = await audioFetcher();

  let log_specs_full = new Float32Array(Math.ceil(samples.length / SAMPLES_PER_SEGMENT) * MEL_SPEC_CHUNK_LENGTH);
  for (let i = 0; i < samples.length; i += SAMPLES_PER_SEGMENT) {
    let chunk = samples.slice(i, i + SAMPLES_PER_SEGMENT);
    if (chunk.length < SAMPLES_PER_SEGMENT) {
      const padded = new Float32Array(SAMPLES_PER_SEGMENT);
      const pad_length = SAMPLES_PER_SEGMENT - chunk.length;
      console.log(`padding last chunk by ${pad_length} samples (${Math.round(pad_length / SAMPLES_PER_SEGMENT * 100)}%)`);
      padded.set(chunk);
      chunk = padded;
    }
    let [mel_spec] = await nets.mel(chunk);
    log_specs_full.set(mel_spec, (MEL_SPEC_CHUNK_LENGTH) * (i / SAMPLES_PER_SEGMENT));
  }


  let pendingText = null, lastDisplayed = '', lastUpdateTime = 0, inferenceDone = false;

  console.log("begin new transcription");

  let previous_context = [];
  let temperature = 0;
  // for (let seek = 50 * MEL_SPEC_CHUNK_LENGTH; seek < 51*MEL_SPEC_CHUNK_LENGTH;) {
  for (let seek = 0; seek < log_specs_full.length;) {
    console.log("seek to " + (seek / MEL_SPEC_CHUNK_LENGTH * 30.0).toFixed(2));
    let log_spec = log_specs_full.slice(seek, seek + MEL_SPEC_CHUNK_LENGTH);
    if (seek + MEL_SPEC_CHUNK_LENGTH > log_specs_full.length) {
      let pad_length = seek + MEL_SPEC_CHUNK_LENGTH - log_specs_full.length;
      // TODO(irwin): possible double-pad, log_specs were already padded to multiple of MEL_SPEC_CHUNK_LENGTH
      // so this only triggers on custom seeks
      console.log(`must pad to ${pad_length}`);
      let padded = new Float32Array(MEL_SPEC_CHUNK_LENGTH);
      padded.set(log_specs_full.slice(seek));
      log_spec = padded;
    }
    const [audio_features] = await nets.encoder(log_spec);
    // const audio_features = audio_features_full.slice(576000 * (seek / MEL_SPEC_CHUNK_LENGTH), 576000 * ((seek / MEL_SPEC_CHUNK_LENGTH) + 1));
    function updateCallback(pd) {
      pendingText = pd;
      console.log(pendingText);
    }
    let [avg_logprob, segment_cumlogprob, context, offset] = await inferLoop(nets, mapping, log_specs_full, previous_context, temperature, audio_features, seek, cancelToken, updateCallback);
    if (cancelToken.cancelled) {
      console.log("Transcription cancelled");
      inferenceDone = true;
      onEvent("cancel");
      // currentTranscription.style.display = 'none';
      return;
    } else {
      if ((avg_logprob < -1) && temperature < 1) {
        temperature += 0.2;
        console.log(`decoding failed, raising temperature to ${temperature}, due to one of: avg_logprob: ${avg_logprob}, segment_cumlogprob: ${segment_cumlogprob}, tokens decoded: ${context.length - offset}`);
        continue;
      } else {
        temperature = 0;
      }
      previous_context = context.slice();

      onEvent("chunkDone", {avg_logprob, segment_cumlogprob, context, offset, pendingText});
      // const newChunk = document.createElement('div');
      // newChunk.className = 'transcription-chunk';
      // newChunk.innerText = segment_cumlogprob.toFixed(2) + ' ' + pendingText;
      // console.log(segment_cumlogprob.toFixed(2) + ' ' + pendingText);
      // transcriptionLog.appendChild(newChunk);
      pendingText = '';
      // currentTranscription.innerText = '';

      seek += MEL_SPEC_CHUNK_LENGTH;
    }
  }
  inferenceDone = true;
  // currentTranscription.style.display = 'none';

  let took = performance.now() - before;
  console.log("end transcription: " + took);
}

function onTranscriptionEvent(event, data) {
  if (event === "cancel") {

  } else if (event === "chunkDone") {
    console.log(data.segment_cumlogprob.toFixed(2) + ' ' + data.pendingText);
  }
}

currentCancel = { cancelled: false };
await transcribeAudio(async () => await fetchMonoFloat32Array(`${BASE_URL}/${AUDIO_PATH}`, AudioContext), currentCancel, onTranscriptionEvent, mapping);
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