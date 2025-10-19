import { create, globals } from 'webgpu';
import { AudioContext, OfflineAudioContext } from 'node-web-audio-api';
import '../.node_env/node_modules/fake-indexeddb/auto/index.mjs';

const {default: mel} = await import("../mel.js");
const {default: encoder} = await import("../encoder.js");
const {default: decoder} = await import("../decoder.js");

// (async () => {
//const { create, globals } = require('webgpu');
//import { create, globals } from 'webgpu';
//const { create, globals } = await import('webgpu');

//Object.assign(globalThis, globals);
//const navigator = { gpu: create([]) };

//const fs = require('node:fs');
//const { PNG } = require('pngjs');
//const { create, globals } = require('webgpu');

Object.assign(globalThis, globals);

const navigator = { gpu: create([]) };

// const SAMPLES_PER_SEGMENT = 480000;
// const MEL_SPEC_CHUNK_LENGTH = 80 * 3000;

// #region imports
import {
  SAMPLES_PER_SEGMENT, MEL_SPEC_CHUNK_LENGTH, TOK_EOS,
  TOK_BEGIN_TRANSCRIPTION,
  TOK_NO_TIMESTAMPS,
  TOK_STARTOFPREV,
  TOK_TRANSCRIBE,
  TOK_NOSPEECH,
  TOK_TS_FIRST,
  TOK_TS_LAST,
  MAX_TOKENS_TO_DECODE,

  MODEL_BATCH_SIZE_HARDCODED,

  NO_TIMESTAMPS,
  NO_CONTEXT,
  SUPPRESS_NONSPEECH_TOKENS,

  tensorStore,
  initDb,

  getDevice,

  argsort,
  logSoftmax,
  softmax,
  sample,
  normalize,

  format_seek,
  format_text,
  tokensToText,

  fetchMonoFloat32Array,
  fetchMonoFloat32ArrayFile,
  getProgressDlForPart,

  handle_timestamp_tokens,
  batch_double_helper,
  decoder_helper,
  rebuild_cache_tail_index,
  decodeOne
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


let currentCancel = null;
async function transcribeAudio(audioFetcher, cancelToken) {
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

  // console.log(log_specs_full.slice(0, 100));

  // let res = await fetch(`${BASE_URL}/RED_16k.mel_f32`);
  // log_specs_full = new Float32Array(await res.arrayBuffer());
  // const audio_features_full = new Float32Array(await fetch(`${BASE_URL}/RED_16k.features`).then((res) => res.arrayBuffer()));

  const mapping = await fetch(`${BASE_URL}/vocab.json`).then(res => res.json());

  let pendingText = null, lastDisplayed = '', lastUpdateTime = 0, inferenceDone = false;
  // const updateLoop = (now) => {
  //   if (pendingText !== null && pendingText !== lastDisplayed && now - lastUpdateTime >= 1000.0 / 30) {
  //     currentTranscription.innerText = pendingText;
  //     lastDisplayed = pendingText;
  //     lastUpdateTime = now;
  //     transcriptionContainer.scrollTop = transcriptionContainer.scrollHeight;
  //   }
  //   if (!inferenceDone) requestAnimationFrame(updateLoop);
  // };
  // requestAnimationFrame(updateLoop);
  // currentTranscription.style.display = 'block';

  async function inferLoop(previous_context, temperature, audio_features, seek, cancelToken) {
    let context = [];
    if (!NO_CONTEXT && previous_context.length > 0 && previous_context.at(-1) == TOK_EOS) {
      let prefix = [TOK_STARTOFPREV];
      let suffix = [TOK_BEGIN_TRANSCRIPTION];
      if (NO_TIMESTAMPS) suffix.push(TOK_NO_TIMESTAMPS);
      let max_context_to_take = MAX_TOKENS_TO_DECODE - prefix.length - suffix.length;
      context.push(...prefix);
      context.push(...previous_context.filter((tok) => tok < TOK_EOS /*|| (tok >= TOK_TS_FIRST && tok <= TOK_TS_LAST)*/).slice(-max_context_to_take));
      context.push(...suffix);
    } else {
      context = [TOK_BEGIN_TRANSCRIPTION];
      if (NO_TIMESTAMPS) context.push(TOK_NO_TIMESTAMPS);
    }
    console.log(context);

    const offset_DEADBEEF = context.length;
    if (offset_DEADBEEF > MAX_TOKENS_TO_DECODE) {
      console.error("Context length exceeds 224");
      return;
    }
    // var v = new Int32Array(51864);
    // v.fill(0);
    // v[0] = TOK_NO_TIMESTAMPS;

    let decoder_state = {
      last_index_DEADBEEF: undefined,
      context: []
    };

    const max_range_DEADBEEF = offset_DEADBEEF + MAX_TOKENS_TO_DECODE;

    let sequences = [];
    const default_sequence = {
      index: 0,
      max_range: 0,
      segment_cumlogprob: 0,
      avg_logprob: 0,
      last_eos_logprob: -10,
      context: undefined,
      logprobs: undefined,
      eos_logprobs: undefined
    };

    const BEST_OF = 5;
    const SEQUENCE_COUNT = temperature > 0 ? BEST_OF : 1;
    for (let i = 0; i < SEQUENCE_COUNT; ++i) {
      let sequence = Object.create(default_sequence);
      sequence.index = offset_DEADBEEF;
      sequence.max_range = max_range_DEADBEEF;
      sequence.context = context.slice();
      sequence.logprobs = [];
      sequence.eos_logprobs = [-10];
      sequences.push(sequence);
    }

    for (; sequences.some(c => c.context.at(-1) !== TOK_EOS);) {
      let updated = false;
      for (let idx = 0; idx < sequences.length; ++idx) {
        if (sequences[idx].context.at(-1) === TOK_EOS) continue;
        if (cancelToken.cancelled) return;

        let decode_result = await decodeOne(nets, sequences[idx], decoder_state, temperature, audio_features, offset_DEADBEEF);
        let keep = decode_result.keep;
        sequences[idx].context = decode_result.context;
        sequences[idx].avg_logprob = decode_result.avg_logprob;
        sequences[idx].segment_cumlogprob = decode_result.segment_cumlogprob;
        sequences[idx].last_eos_logprob = decode_result.last_eos_logprob;

        if (!updated) {
          pendingText = format_text(tokensToText(sequences[idx].context.slice(offset_DEADBEEF), mapping), sequences[idx].avg_logprob, seek, Math.min(seek + MEL_SPEC_CHUNK_LENGTH, log_specs_full.length));
          console.log(pendingText);
          updated = true;
        }

        if (!keep) break;
        ++sequences[idx].index;
      }
    }

    for (let seq of sequences) {
      console.log(seq.logprobs);
      console.log(seq.eos_logprobs);
      let cumlogprob = seq.logprobs.reduce((a, b) => a + b);
      console.log(cumlogprob);
      console.log(cumlogprob / seq.logprobs.length);
    }
    let segment_cumlogprobs = sequences.map(s => s.segment_cumlogprob);
    let idx = segment_cumlogprobs.indexOf(Math.min.apply(null, segment_cumlogprobs));

    return [sequences[idx].avg_logprob, sequences[idx].segment_cumlogprob, sequences[idx].context, offset_DEADBEEF];
  }

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


    let [avg_logprob, segment_cumlogprob, context, offset] = await inferLoop(previous_context, temperature, audio_features, seek, cancelToken);
    if (cancelToken.cancelled) {
      console.log("Transcription cancelled");
      inferenceDone = true;
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

      // const newChunk = document.createElement('div');
      // newChunk.className = 'transcription-chunk';
      // newChunk.innerText = segment_cumlogprob.toFixed(2) + ' ' + pendingText;
      console.log(segment_cumlogprob.toFixed(2) + ' ' + pendingText);
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

currentCancel = { cancelled: false };
await transcribeAudio(async () => await fetchMonoFloat32Array(`${BASE_URL}/${AUDIO_PATH}`, AudioContext), currentCancel);
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

// delete globalThis.navigator;
// }) ();