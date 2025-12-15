// #region constants
const SAMPLES_PER_SEGMENT = 480000;
const MEL_SPEC_CHUNK_LENGTH = 80 * 3000;

const TOK_EOS = 50256;
const TOK_BEGIN_TRANSCRIPTION = 50257;
const TOK_NO_TIMESTAMPS = 50362;
const TOK_STARTOFPREV = 50360;
const TOK_TRANSCRIBE = 50358;
const TOK_NOSPEECH = 50361;

const TOK_TS_FIRST = 50363;
const TOK_TS_LAST = 51863;

// TODO(irwin): remove or allow setting those from outside
const NO_TIMESTAMPS = true;
const NO_CONTEXT = true;
const SUPPRESS_NONSPEECH_TOKENS = true;

const MAX_TOKENS_TO_DECODE = 224;
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
        // const data = audioBuffer.getChannelData(c);
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
    // let limits = Object.fromEntries(LIMITS_KEYS.map(x => [x, adapter.limits[x]]));
    // let limits = adapter.limits;
    // console.log(limits);
    // console.log(Object.entries(adapter.limits));
    // console.log(Object.entries(adapter.features));
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
    return tokens.filter((t) => ![TOK_EOS, TOK_NO_TIMESTAMPS].includes(t)).map(j => mapping[j]).join('');
}

// #region whisper

function batch_repeat_helper(array, bs) {
    let result = array.slice();
    for (let i = 0; i < bs-array.length; ++i) {
        result.push(array.at(-1));
    }
    return result;
}

async function decoder_helper(nets, context_inputs, audio_features, context_last_token_index_absolute, decoder_state) {
    context_inputs = batch_repeat_helper(context_inputs, nets.model_metadata.decoder_batch_size);
    let [decoder_output, sorted] = await nets.decoder(context_inputs, audio_features, [context_last_token_index_absolute], [0], [1]);
    for (let i = 0; i < nets.model_metadata.decoder_batch_size; ++i) {
        decoder_state.contexts[i] = [...decoder_state.contexts[i].slice(0, context_last_token_index_absolute), context_inputs[i]];
    }

    return [decoder_output, sorted];
}

async function decoder_upload_audio_features(nets, audio_features_batch, decoder_state) {
    let context_input = [TOK_BEGIN_TRANSCRIPTION];
    context_input = batch_repeat_helper(context_input, nets.model_metadata.decoder_batch_size);

    for (let i = 0; i < audio_features_batch.length; ++i) {
        let decoder_output_discarded = await nets.decoder(context_input, audio_features_batch[i], [0], [i], [0]);
    }

    for (let i = 0; i < nets.model_metadata.decoder_batch_size; ++i) {
        decoder_state.contexts[i] = [context_input[i]];
    }
}

function rebuild_cache_tail_index(c1, c2) {
    let i = 0;
    for (; i < c1.length && i < c2.length; ++i) {
        if (c1[i] !== c2[i]) break;
    }
    return i;
}

const suppress = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361];

/** @typedef {number} integer */
/** @typedef {number} float */

/**
 * @typedef Decode_Sequence
 * @type {object}
 * @property {integer} index
 * @property {integer} context_prompt_length
 * @property {integer} max_context_length
 * @property {float} segment_cumlogprob
 * @property {float} avg_logprob
 * @property {integer[]} context
 * @property {float[]} logprobs
 */

/**
 * @typedef Decoder_State
 * @type {object}
 * @property {integer[][]} contexts
 */

/**
 * @typedef Decoder_Result
 * @type {object}
 * @property {integer[]} context
 * @property {number} avg_logprob
 * @property {number} segment_cumlogprob
 */

/**
 * @param {Decode_Sequence[]} decode_sequences
 * @param {Decoder_State} decoder_state
 * @param {Float32Array[]} audio_features_batch
 * @returns {Promise<Decoder_Result[]>}
 */
async function decodeOneBatch(nets, decode_sequences, decoder_state, audio_features_batch) {
    let audio_features = audio_features_batch[0];

    let results = [];
    let context_inputs = [];
    for (let i = 0; i < decode_sequences.length; ++i) {
        context_inputs.push(decode_sequences[i].context.at(-1));

        let result = {
            context: decode_sequences[i].context, // TODO(irwin): decide if we want to return the same context or a new copy  @ContextCopyOrReference
            avg_logprob: undefined,
            segment_cumlogprob: decode_sequences[i].segment_cumlogprob,
        }
        results.push(result);
    }
    const max_context_length = Math.max.apply(null, decode_sequences.map(x => x.context.length));

    let [sorted] = await decoder_helper(nets, context_inputs, audio_features, max_context_length - 1, decoder_state);

    let logprobs_topk = 10;
    let indices_topk = 10;
    for (let i = 0; i < results.length; ++i) {
        results[i].sorted = sorted.slice(i*indices_topk, (i+1)*indices_topk);
    }
    return results;
}

/**
 * @param {Decode_Sequence} decode_sequence
 * @param {Decoder_Result} decodeOne_result
 * @returns {Promise<Decoder_Result>}
 */
async function applyDecoderResults(decode_sequence, decodeOne_result) {
    const { context, context_prompt_length, max_context_length } = decode_sequence;

    let result = decodeOne_result;
    let decoder_output = decodeOne_result.decoder_output;

    let nextLogprobs;
    let nextTokens;

    nextLogprobs = decoder_output;
    nextTokens = decodeOne_result.sorted;

    let nextTokenIndex = 0;

    // NOTE(irwin): up until here, context is not modified  @ContextCopyOrReference
    context.push(nextTokens[nextTokenIndex]);

    const decoded_tokens_so_far = context.length - context_prompt_length;
    result.avg_logprob = result.segment_cumlogprob / (decoded_tokens_so_far - context_prompt_length);
    let nextLogprob = -0.1;
    decode_sequence.logprobs.push(nextLogprob);
    result.segment_cumlogprob += nextLogprob;

    if (nextTokens[nextTokenIndex] == TOK_EOS) {
        return result;
    } else if (context.length >= max_context_length) {
        context[context.length - 1] = TOK_EOS;
    }

    return result;
}

/** @returns {Promise<Decode_Sequence[]|undefined>} */
async function inferLoop(nets, log_specs_full, previous_context, audio_features_batch, seeks_batch, cancelToken, inferLoopContext) {
    if (inferLoopContext.state === "INIT") {
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

        const context_prompt_length = context.length;
        if (context_prompt_length > MAX_TOKENS_TO_DECODE) {
            console.error("Context prompt length exceeds 224");
            inferLoopContext.is_done = true;
            return;
        }

        const max_context_length = context_prompt_length + MAX_TOKENS_TO_DECODE;

        /** @type {Decode_Sequence[]} */
        let sequences = [];
        const default_sequence = {
            index: 0,
            context_prompt_length: 1,
            max_context_length: 0,
            segment_cumlogprob: 0,
            avg_logprob: 0,
            context: undefined,
            logprobs: undefined,
        };

        const SEQUENCE_COUNT = audio_features_batch.length;
        for (let i = 0; i < SEQUENCE_COUNT; ++i) {
            /** @type {Decode_Sequence} */
            let sequence = Object.create(default_sequence);
            sequence.index = context_prompt_length;
            sequence.context_prompt_length = context_prompt_length;
            sequence.max_context_length = max_context_length;
            sequence.context = context.slice();
            sequence.logprobs = [];
            sequences.push(sequence);
        }

        inferLoopContext.sequences = sequences;
        inferLoopContext.state = "DECODE_INIT";
        return;

    } else if (inferLoopContext.state === "DECODE_INIT") {
        let sequences = inferLoopContext.sequences;

        /** @type {Decoder_State} */
        let decoder_state = {
            contexts: []
        };

        for (let i = 0; i < nets.model_metadata.decoder_batch_size; ++i) {
            decoder_state.contexts.push([]);
        }

        await decoder_upload_audio_features(nets, audio_features_batch, decoder_state);


        let pendingTexts = [];
        for (let sequence_index = 0; sequence_index < sequences.length; ++sequence_index) {
            pendingTexts.push('');
        }

        inferLoopContext.decoder_state = decoder_state;
        inferLoopContext.pendingTexts = pendingTexts;
        inferLoopContext.currentTokenIndex = 0;
        inferLoopContext.state = "DECODE";
        return;

    } else if (inferLoopContext.state === "DECODE") {
        let pendingTexts = inferLoopContext.pendingTexts;
        let sequences = inferLoopContext.sequences;
        let decoder_state = inferLoopContext.decoder_state;
        if (inferLoopContext.currentTokenIndex < MAX_TOKENS_TO_DECODE && sequences.some(x => x.context.at(-1) !== TOK_EOS)) {
            if (cancelToken.cancelled) {
                inferLoopContext.is_done = true;
                return;
            }

            let decode_results = await decodeOneBatch(nets, sequences, decoder_state, audio_features_batch);
            for (let idx = 0; idx < sequences.length; ++idx) {
                if (cancelToken.cancelled) {
                    inferLoopContext.is_done = true;
                    return;
                }
                if (sequences[idx].context.at(-1) === TOK_EOS) continue;
                let seek = seeks_batch[idx];

                let decode_result = await applyDecoderResults(sequences[idx], decode_results[idx]);
                sequences[idx].context = decode_result.context;
                sequences[idx].avg_logprob = decode_result.avg_logprob;
                sequences[idx].segment_cumlogprob = decode_result.segment_cumlogprob;

                const detokenized = tokensToText(sequences[idx].context.slice(sequences[idx].context_prompt_length), nets.mapping);
                const seek_end = Math.min(seek + MEL_SPEC_CHUNK_LENGTH, log_specs_full.length);
                let pendingText = format_text(detokenized, seek, seek_end);
                pendingTexts[idx] = pendingText;
            }
            ++inferLoopContext.currentTokenIndex;
            return;

        } else {
            inferLoopContext.state = "POST_DECODE";
            return;
        }

    } else if (inferLoopContext.state === "POST_DECODE") {
        let sequences = inferLoopContext.sequences;

        for (let seq of sequences) {
            let cumlogprob = seq.logprobs.reduce((a, b) => a + b);
            console.log(cumlogprob);
            console.log(cumlogprob / seq.logprobs.length);
        }
        let segment_cumlogprobs = sequences.map(s => s.segment_cumlogprob);
        let idx = segment_cumlogprobs.indexOf(Math.min.apply(null, segment_cumlogprobs));

        inferLoopContext.is_done = true;
        inferLoopContext.state = "DONE";

        return sequences;
    } else if (inferLoopContext.state === "DONE") {
        return inferLoopContext.sequences;
    }
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

    let pendingTexts = [];

    onEvent("inferenceBegin", {chunkCount, sampleCount: samples.length});
    console.log("begin new transcription");

    let previous_context = [];
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

        let inferLoopContext = {
            state: "INIT",
            is_done: false
        };
        let sequences;
        while (!inferLoopContext.is_done) {
            sequences = await inferLoop(nets, log_specs_full, previous_context, audio_features_batch, seeks_batch, cancelToken, inferLoopContext);
            if (inferLoopContext.state === "INIT") {
            } else if (inferLoopContext.state === "DECODE_INIT") {
            } else if (inferLoopContext.state === "DECODE") {
                if (inferLoopContext.currentTokenIndex !== 0) {
                    // index was already incremented for the next decode iteration
                    let currentTokenIndex = inferLoopContext.currentTokenIndex - 1;
                    pendingTexts = inferLoopContext.pendingTexts.slice();
                    onEvent("chunkUpdate", {pendingTexts, currentTokenIndex, sequenceStatus: inferLoopContext.sequences.map(x => x.context.at(-1) === TOK_EOS ? "done" : "running")});
                }
            } else if (inferLoopContext.state === "POST_DECODE") {
            } else if (inferLoopContext.state === "DONE") {
            }
        }

        for (let i = 0; i < batch_size && seek_index + i < seek_ranges.length;) {
            if (cancelToken.cancelled) {
                console.log("Transcription cancelled");
                onEvent("cancel");
                return;
            } else {
                let sequence = sequences[i];
                let {avg_logprob, segment_cumlogprob, context, context_prompt_length: offset} = sequence;
                previous_context = context.slice();

                onEvent("chunkDone", { avg_logprob, segment_cumlogprob, context, offset, index: i, pendingText: pendingTexts[i] });
                pendingTexts[i] = '';

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

    NO_TIMESTAMPS,
    NO_CONTEXT,
    SUPPRESS_NONSPEECH_TOKENS,

    tensorStore,
    initDb,

    getDevice,

    tokensToText,

    fetchMonoFloat32Array,
    fetchMonoFloat32ArrayFile,
    getProgressDlForPart,

    batch_repeat_helper,
    decoder_helper,
    rebuild_cache_tail_index,
    decodeOneBatch,
    inferLoop,
    transcribeAudio
};