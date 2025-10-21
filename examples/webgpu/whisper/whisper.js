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

// TODO(irwin): should be read from model_metadata.json
const MODEL_BATCH_SIZE_HARDCODED = 1;

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
    const audioCtx = new AudioContextImplementation({ sampleRate: 16000, sinkId: 'none' });
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    await audioCtx.close();
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
    const response = await fetch(part, {
        headers: lastModified ? { "If-Modified-Since": lastModified } : {}
    });
    if (response.status === 304) return null; // not modified

    const total = parseInt(response.headers.get('content-length'), 10);
    const newLastModified = response.headers.get('Last-Modified');

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

// #region math
function argsort(array) {
    // arange
    const indices = new Uint32Array(array.length);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
    indices.sort((a, b) => array[b] - array[a]);
    return indices;
}

function logSoftmax(logits) {
    const max = Math.max.apply(null, logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const logSumExp = Math.log(sumExp);
    return [logits.map(x => x - max - logSumExp), max];
}

function softmax(logits) {
    const scaled = logits;
    const max = Math.max.apply(null, scaled); // prevent overflow
    const exps = scaled.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}


function sample(probs) {
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return probs.length - 1; // fallback for float imprecision
}

function normalize(probs) {
    const sum = probs.reduce((a, b) => a + b, 0);
    return probs.map(p => p / sum);
}
// #endregion math

function format_seek(seek) {
    return (seek / MEL_SPEC_CHUNK_LENGTH * 30.0).toFixed(2);
}

function format_text(text, segment_cumlogprob, seek, seek_end) {
    return (segment_cumlogprob).toFixed(2) + '\n' + `${format_seek(seek)} ---> ${format_seek(seek_end)} ` + text;
}

function tokensToText(tokens, mapping) {
    return tokens.filter((t) => ![TOK_EOS, TOK_NO_TIMESTAMPS].includes(t)).map(j => mapping[j]).join('');
}

// #region whisper
function handle_timestamp_tokens(nextTokens, context, token_count, last_token_index_DEADBEEF, one_before_last_token_index_DEADBEEF) {
    if (!NO_TIMESTAMPS) {
        if (token_count === 0) {
            nextTokens = nextTokens.filter((t) => t >= TOK_TS_FIRST && t <= TOK_TS_LAST);
        } else if (context[last_token_index_DEADBEEF] >= TOK_TS_FIRST) {
            if (context[one_before_last_token_index_DEADBEEF] >= TOK_TS_FIRST) {
                nextTokens = nextTokens.filter((t) => t < TOK_TS_FIRST);
            } else {
                nextTokens = nextTokens.filter((t) => t >= TOK_EOS);
            }
        }
    }

    return nextTokens;
}

function batch_double_helper(array) {
    // return [...array, ...array];
    return array;
}

async function decoder_helper(nets, context, context_input, audio_features, context_index_DEADBEEF, decoder_state) {
    context_input = batch_double_helper(context_input);
    let [decoder_output] = await nets.decoder(context_input, audio_features, [context_index_DEADBEEF]);
    decoder_state.last_index_DEADBEEF = context_index_DEADBEEF;
    decoder_state.context = [...decoder_state.context.slice(0, context_index_DEADBEEF * MODEL_BATCH_SIZE_HARDCODED), ...context_input];
    return decoder_output;
}

function rebuild_cache_tail_index(c1, c2) {
    let i_DEADBEEF = 0;
    for (; i_DEADBEEF < c1.length && i_DEADBEEF < c2.length; ++i_DEADBEEF) {
        if (c1[i_DEADBEEF] !== c2[i_DEADBEEF]) break;
    }
    return i_DEADBEEF;
}

const suppress = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361];

async function decodeOne(nets, decode_sequence, decoder_state, temperature, audio_features, offset_DEADBEEF) {
    const { index: i_DEADBEEF, max_range: max_range_DEADBEEF, last_eos_logprob } = decode_sequence;
    const last_index_DEADBEEF = decoder_state.last_index_DEADBEEF;
    let { context } = decode_sequence;
    let result = {
        keep: false,
        context,
        avg_logprob: undefined,
        segment_cumlogprob: decode_sequence.segment_cumlogprob,
        last_eos_logprob: undefined
    }
    if (i_DEADBEEF >= max_range_DEADBEEF) return result;

    let token_count = context.length - offset_DEADBEEF;
    let last_token_index_DEADBEEF = i_DEADBEEF;
    let one_before_last_token_index_DEADBEEF = i_DEADBEEF - 1;

    let tail_index_DEADBEEF = rebuild_cache_tail_index(context, decoder_state.context);
    if (tail_index_DEADBEEF < i_DEADBEEF - 1) {
        // NOTE(irwin): rebuild self attention kv cache
        // TODO(irwin): for batch==1 we can rebuild only the tail of the kv cache that should only differ by 1-2 tokens or so
        for (let build_cache_index_DEADBEEF = 0; build_cache_index_DEADBEEF < offset_DEADBEEF - 1; ++build_cache_index_DEADBEEF) {
            let context_input = context.slice(build_cache_index_DEADBEEF, build_cache_index_DEADBEEF + 1);
            await decoder_helper(nets, context, context_input, audio_features, build_cache_index_DEADBEEF, decoder_state);
        }
    }

    let context_input = context.slice(i_DEADBEEF - 1, i_DEADBEEF);
    // context_input = batch_double_helper(context_input);

    // let [decoder_output] = await nets.decoder(context_input, (audio_features), [i-1]);
    let decoder_output = await decoder_helper(nets, context, context_input, audio_features, i_DEADBEEF - 1, decoder_state);
    decoder_output = decoder_output.slice(0, decoder_output.length / MODEL_BATCH_SIZE_HARDCODED);
    decoder_state.last_index_DEADBEEF = i_DEADBEEF;
    // decoder_state.context = context;
    let nextLogprobs;
    let nextTokens;
    let max;

    if (SUPPRESS_NONSPEECH_TOKENS) {
        for (let token_index of suppress) {
            decoder_output[token_index] = -Infinity;
        }
    }
    if (temperature > 0) decoder_output = decoder_output.map(x => x / temperature);
    [nextLogprobs, max] = logSoftmax(decoder_output);
    nextTokens = argsort(nextLogprobs);

    // decoder_output = decoder_output.filter((t)=> ![TOK_NO_TIMESTAMPS, ...suppress].includes(t));
    nextTokens = handle_timestamp_tokens(nextTokens, context, token_count, last_token_index_DEADBEEF, one_before_last_token_index_DEADBEEF);
    let nextTokenIndex = 0;
    if (temperature > 0) {
        // let sortedSampledIndex = sample(normalize(nextLogprobs));
        let dist = normalize(softmax(decoder_output));
        nextTokenIndex = nextTokens.indexOf(sample(dist));
    }

    if (nextTokens[nextTokenIndex] == TOK_EOS && Math.abs(last_eos_logprob) - Math.abs(nextLogprobs[TOK_EOS]) > 8) {
        ++nextTokenIndex;
    }

    context.push(nextTokens[nextTokenIndex]);

    result.avg_logprob = result.segment_cumlogprob / (i_DEADBEEF - offset_DEADBEEF + 1);
    let nextLogprob = nextLogprobs[nextTokens[nextTokenIndex]];
    decode_sequence.logprobs.push(nextLogprob);
    decode_sequence.eos_logprobs.push(nextLogprobs[TOK_EOS]);
    result.segment_cumlogprob += nextLogprob;
    // pendingText = format_text(context.slice(offset).map(j => mapping[j]).join(''), avg_logprob, seek, Math.min(seek+MEL_SPEC_CHUNK_LENGTH, log_specs_full.length));
    result.last_eos_logprob = nextLogprobs[TOK_EOS];

    if (nextTokens[nextTokenIndex] == TOK_EOS) {
        return result;
    } else if (i_DEADBEEF + 1 >= max_range_DEADBEEF) {
        context[context.length - 1] = TOK_EOS;
    }

    result.keep = true;
    return result;
}

async function inferLoop(nets, mapping, log_specs_full, previous_context, temperature, audio_features, seek, cancelToken, updatedCallback) {
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
    // console.log(context);

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
                let pendingText = format_text(tokensToText(sequences[idx].context.slice(offset_DEADBEEF), mapping), sequences[idx].avg_logprob, seek, Math.min(seek + MEL_SPEC_CHUNK_LENGTH, log_specs_full.length));
                updatedCallback(pendingText);
                // console.log(pendingText);
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

async function transcribeAudio(nets, audioFetcher, cancelToken, onEvent, mapping, loadAndInitializeModels) {
    let before = performance.now();
    await loadAndInitializeModels();
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

    let pendingText = null;

    onEvent("inferenceBegin");
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
            onEvent("chunkUpdate", { pendingText });
        }
        let [avg_logprob, segment_cumlogprob, context, offset] = await inferLoop(nets, mapping, log_specs_full, previous_context, temperature, audio_features, seek, cancelToken, updateCallback);
        if (cancelToken.cancelled) {
            console.log("Transcription cancelled");
            onEvent("cancel");
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

            onEvent("chunkDone", { avg_logprob, segment_cumlogprob, context, offset, pendingText });
            pendingText = '';

            seek += MEL_SPEC_CHUNK_LENGTH;
        }
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
    decodeOne,
    inferLoop,
    transcribeAudio
};