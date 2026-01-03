// #region constants
const SAMPLES_PER_SEGMENT = 480000;
const MEL_SPEC_CHUNK_LENGTH = 80 * 3000;

// TODO: rename to EOT - end of transcription
const TOK_EOS = 50256;
const TOK_BEGIN_TRANSCRIPTION = 50257;
const TOK_NO_TIMESTAMPS = 50362;
const TOK_STARTOFPREV = 50360;
const TOK_TRANSCRIBE = 50358;
const TOK_NOSPEECH = 50361;

const TOK_TS_FIRST = 50363;
const TOK_TS_LAST = 51863;

const MAX_CONTEXT_LENGTH = 448;
const MAX_TOKENS_TO_DECODE = 224;

const AUDIO_FEATURES_CACHE__REUSE = [1];
const AUDIO_FEATURES_CACHE__OVERWRITE = [0];

// #endregion constants

// #region audio
async function fetchMonoFloat32Array(url, AudioContextImplementation = globalThis.AudioContext) {
    const response = await fetch(url);
    return await fetchMonoFloat32ArrayFile(response, AudioContextImplementation);
}

async function fetchMonoFloat32ArrayFile(response, AudioContextImplementation = globalThis.AudioContext) {
    const arrayBuffer = await response.arrayBuffer();
    const audioCtx = new AudioContextImplementation({ sampleRate: 16000, sinkId: 'none', numberOfChannels: 1, length: 16000 });
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    await audioCtx.close?.();
    const mono = new Float32Array(audioBuffer.length);
    for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
        const data = new Float32Array(audioBuffer.length);
        audioBuffer.copyFromChannel(data, c);
        for (let i = 0; i < data.length; i++) mono[i] += data[i] / audioBuffer.numberOfChannels;
    }
    return { sampleRate: audioBuffer.sampleRate, samples: mono };
}
// #endregion audio

const getProgressDlForPart = async (part, progressCallback, lastModified) => {
    const serverLastModified = await fetch(part + '.version', {cache: 'no-cache'}).then(r => r.ok ? r.text() : '');
    if (lastModified) {
        if (serverLastModified === lastModified) return null; // not modified
    }
    const response = await fetch(part);

    const total = parseInt(response.headers.get('content-length'), 10);
    const newLastModified = serverLastModified;

    const res = new Response(new ReadableStream({
        async start(controller) {
            const reader = response.body.getReader();
            for (; ;) {
                const { done, value } = await reader.read();
                if (done) break;
                progressCallback(part, value.byteLength, total);
                controller.enqueue(value);
            }
            controller.close();
        },
    }));
    return { buffer: await res.arrayBuffer(), lastModified: newLastModified };
};

const tensorStore = (db) => ({
    get: (id) => new Promise(r => {
        const req = db.transaction('tensors').objectStore('tensors').get(id);
        req.onsuccess = () => r(req.result || null);
        req.onerror = () => r(null);
    }),
    put: (id, content, lastModified) => new Promise(r => {
        const req = db.transaction('tensors', 'readwrite')
            .objectStore('tensors').put({ id, content, lastModified });
        req.onsuccess = () => r();
        req.onerror = () => r(null);
    })
});

function initDb() {
    return new Promise((resolve, reject) => {
        let db;
        const request = indexedDB.open('tinywhisperdb', 2);
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
            if (event.oldVersion < 2 && db.objectStoreNames.contains("tensors")) db.deleteObjectStore?.('tensors');
            if (!db.objectStoreNames.contains('tensors')) {
                db.createObjectStore('tensors', { keyPath: 'id' });
            }
        };
    });
}

const getDevice = async (GPU) => {
    if (!GPU) return false;
    const adapter = await GPU.requestAdapter();
    if (!adapter) return false;
    let maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;

    const _2GB = 2 ** 31; // 2GB
    // safeguard against webgpu reporting nonsense value. some anti-fingerprinting measures?
    // TODO(irwin): use max_size_per_tensor_in_bytes to get actual required limit
    let maxBufferSize = Math.min(adapter.limits.maxBufferSize, _2GB);
    let maxComputeWorkgroupStorageSize = adapter.limits.maxComputeWorkgroupStorageSize;
    const params = {
        // requiredFeatures: ["shader-f16"],
        requiredLimits: { "maxStorageBufferBindingSize": maxStorageBufferBindingSize, "maxBufferSize": maxBufferSize, "maxComputeWorkgroupStorageSize": maxComputeWorkgroupStorageSize },
        powerPreference: "high-performance"
    };
    /** @type {GPUDevice} */
    const device = await adapter.requestDevice(params);
    return device;
};


function format_seek(seek) {
    return (seek / MEL_SPEC_CHUNK_LENGTH * 30.0).toFixed(2);
}

function format_text(text, seek, seek_end) {
    return `${format_seek(seek)} ---> ${format_seek(seek_end)} \n ${text}`;
}

function tokensToText(tokens, mapping) {
    return Array.from(tokens).filter((t) => ![TOK_EOS, TOK_NO_TIMESTAMPS].includes(t)).map(j => mapping[j]).join('');
}

function format_text_helper(tokens, mapping, seek, seek_end) {
    const detokenized = tokensToText(tokens, mapping);
    return format_text(detokenized, seek, seek_end);
}

// #region whisper

function batch_repeat_helper(array, bs) {
    let result = array.slice();
    for (let i = 0; i < bs-array.length; ++i) {
        result.push(array.at(-1));
    }
    return result;
}

async function decoder_decode_batch(nets, context_inputs, self_attention_kv_cache_cutoff) {
    const AUDIO_FEATURES_UPDATE_INDEX_STUB = [0];
    const AUDIO_FEATURES_STUB = [0];
    let decoder_output = await nets.decoder(context_inputs, AUDIO_FEATURES_STUB, [self_attention_kv_cache_cutoff], AUDIO_FEATURES_UPDATE_INDEX_STUB, AUDIO_FEATURES_CACHE__REUSE);
    return decoder_output;
}

async function decoder_upload_audio_features_item(nets, context_input, audio_features, batch_index_insert_point) {
    const SELF_ATTENTION_KV_CACHE_CUTOFF_STUB = [0];
    let decoder_output_discarded = await nets.decoder(context_input, audio_features, SELF_ATTENTION_KV_CACHE_CUTOFF_STUB, [batch_index_insert_point], AUDIO_FEATURES_CACHE__OVERWRITE);
}

async function decoder_upload_audio_features(nets, audio_features_batch) {
    let context_input = [TOK_BEGIN_TRANSCRIPTION];
    context_input = batch_repeat_helper(context_input, nets.model_metadata.decoder_batch_size);

    for (let i = 0; i < audio_features_batch.length; ++i) {
        await decoder_upload_audio_features_item(nets, context_input, audio_features_batch[i], i);
    }
}


/** @typedef {number} integer */
/** @typedef {number} float */

/**
 * @typedef Decode_Cursor
 * @type {object}
 * @property {integer} context_prompt_length
 * @property {integer} max_context_length
 * @property {Int32Array} tokens
 * @property {integer} length
 * @property {integer} valid_cache
 * @property {bool} done
 */


function initSequences(batch_size) {
    let buffer = new ArrayBuffer(batch_size * MAX_CONTEXT_LENGTH * 4);

    let context = [TOK_BEGIN_TRANSCRIPTION, TOK_NO_TIMESTAMPS];
    const context_prompt_length = context.length;
    const max_context_length = context_prompt_length + MAX_TOKENS_TO_DECODE;

    /** @type {Decode_Cursor[]} */
    let sequences = [];
    for (let i = 0; i < batch_size; ++i) {
        let sequence = {};
        sequence.context_prompt_length = context_prompt_length;
        sequence.max_context_length = max_context_length;
        let tokens = new Int32Array(buffer, i * MAX_CONTEXT_LENGTH * 4, MAX_CONTEXT_LENGTH);
        tokens.set(context, 0);
        sequence.tokens = tokens;
        sequence.length = context.length;
        sequence.valid_cache = 0;
        sequence.done = false;
        sequences.push(sequence);
    }

    return sequences;
}


async function transcribeAudio(nets, audioFetcher, cancelToken, onEvent, loadAndInitializeModels) {
    let before = performance.now();
    await loadAndInitializeModels();
    onEvent("audioDecode");
    const { sampleRate, samples } = await audioFetcher();

    let chunkCount = 0;
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

        ++chunkCount;
    }

    onEvent("inferenceBegin", {chunkCount, sampleCount: samples.length});
    console.log("begin new transcription");

    let seek_ranges = [];
    for (let seek = 0; seek < log_specs_full.length; seek += MEL_SPEC_CHUNK_LENGTH) {
        seek_ranges.push(seek);
    }

    const batch_size = nets.model_metadata.decoder_batch_size;
    for (let seek_index = 0; seek_index < seek_ranges.length;) {
        let audio_features_batch = [];
        onEvent("audioEncode");
        for (let i = 0; (i < batch_size) && (seek_index + i < seek_ranges.length); ++i) {
            let seek = seek_ranges[seek_index + i];

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
            audio_features_batch.push(audio_features);
        }

        let seeks_batch = [];
        for (let i = 0; (i < batch_size) && (seek_index + i < seek_ranges.length); ++i) {
            let seek = seek_ranges[seek_index + i];
            seeks_batch.push(seek);
        }

        function selectToken(tokens, policy) {
            console.assert(policy === "greedy", "only greedy policy implemented atm");
            return tokens[0];
        }

        // TODO: name misleading, actual sequence count equals to batch_size, sequence_count means actually how many batch slots we use
        const sequence_count = audio_features_batch.length;

        let sequences = initSequences(batch_size);
        for (let i = sequence_count; i < batch_size; ++i) sequences[i].done = true;

        await decoder_upload_audio_features(nets, audio_features_batch);

        let is_done = false;
        let currentTokenIndex = 0;
        while (!is_done) {
            if (currentTokenIndex < MAX_TOKENS_TO_DECODE && sequences.some(x => !x.done)) {
                if (cancelToken.cancelled) {
                    is_done = true;
                    break;
                }

                let chunkUpdate = {};

                // NOTE: pack batch inputs
                let context_inputs = [];
                for (let idx = 0; idx < sequence_count; ++idx) {
                    let ctx = sequences[idx].tokens;
                    context_inputs.push(ctx.at(sequences[idx].length-1));
                }
                const max_context_batch_length = Math.max.apply(null, sequences.map(x => x.length));
                let sorted;
                {
                    let context_inputs_padded = batch_repeat_helper(context_inputs, batch_size);
                    let context_last_token_index_absolute = max_context_batch_length - 1;
                    let self_attention_kv_cache_cutoff = context_last_token_index_absolute;
                    [sorted] = await decoder_decode_batch(nets, context_inputs_padded, self_attention_kv_cache_cutoff);

                    for (let i = 0; i < batch_size; ++i) {
                        sequences[i].valid_cache = sequences[i].length;
                    }
                }
                // NOTE: unpack batch results
                const indices_topk = nets.model_metadata.decoder_topk ? nets.model_metadata.decoder_topk : 10;
                let decode_results_topk = [];
                for (let i = 0; i < sequence_count; ++i) {
                    decode_results_topk.push(sorted.slice(i*indices_topk, (i+1)*indices_topk));
                }

                chunkUpdate.sequences = [];
                for (let idx = 0; idx < sequence_count; ++idx) {
                    let current_sequence = sequences[idx];

                    if (current_sequence.done) continue;
                    let seek = seeks_batch[idx];

                    let next_token = selectToken(decode_results_topk[idx], "greedy");
                    current_sequence.tokens[current_sequence.length] = next_token;
                    current_sequence.length += 1;

                    if (next_token === TOK_EOS) {
                        current_sequence.done = true;
                    }

                    let decoded_tokens_so_far = current_sequence.tokens.slice(current_sequence.context_prompt_length, current_sequence.length);
                    const seek_end = Math.min(seek + MEL_SPEC_CHUNK_LENGTH, log_specs_full.length);
                    chunkUpdate.sequences[idx] = {tokens: decoded_tokens_so_far, seek, seek_end};
                }

                onEvent("chunkUpdate", {sequences: chunkUpdate.sequences, currentTokenIndex, sequenceStatus: sequences.slice(0, sequence_count).map(x => x.done ? "done" : "running")});

                ++currentTokenIndex;
            } else {
                is_done = true;
                break;
            }
        }

        for (let i = 0; i < batch_size && seek_index + i < seek_ranges.length;) {
            if (cancelToken.cancelled) {
                console.log("Transcription cancelled");
                onEvent("cancel");
                return;
            } else {
                let sequence = sequences[i];
                let {tokens, context_prompt_length: offset} = sequence;

                onEvent("chunkDone", { context: tokens.slice(0, sequence.length), offset, index: i });

                ++i;
            }
        }

        seek_index += batch_size;
    }
    onEvent("inferenceDone");

    let took = performance.now() - before;
    console.log("end transcription: " + took);
}

// #endregion whisper

export {
    SAMPLES_PER_SEGMENT,
    MEL_SPEC_CHUNK_LENGTH,
    TOK_EOS,
    TOK_BEGIN_TRANSCRIPTION,
    TOK_NO_TIMESTAMPS,
    TOK_STARTOFPREV,
    TOK_TRANSCRIBE,
    TOK_NOSPEECH,
    TOK_TS_FIRST,
    TOK_TS_LAST,
    MAX_TOKENS_TO_DECODE,

    tensorStore,
    initDb,

    getDevice,

    tokensToText,
    format_text,
    format_text_helper,

    fetchMonoFloat32Array,
    fetchMonoFloat32ArrayFile,
    getProgressDlForPart,

    batch_repeat_helper,
    transcribeAudio
};