# thanks to https://github.com/openai/whisper for a good chunk of MIT licensed code

import sys, base64, multiprocessing, itertools, collections, json
from typing import Optional, Union, Literal, List

from tinygrad import Tensor, TinyJit, Variable, nn
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.helpers import getenv, DEBUG, fetch

import numpy as np
import librosa
from scipy.special import log_softmax, logsumexp

class MultiHeadAttention:
  def __init__(self, n_state, n_head, kv_caching: Literal['cross', 'self']=None, max_self_attn_cache_len=None):
    self.n_head = n_head
    self.query = nn.Linear(n_state, n_state)
    self.key = nn.Linear(n_state, n_state, bias=False)
    self.value = nn.Linear(n_state, n_state)
    self.out = nn.Linear(n_state, n_state)

    self.kv_caching = kv_caching
    self.max_self_attn_cache_len = max_self_attn_cache_len

  def __call__(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None, len: Union[Variable,int]=None):
    if self.kv_caching == 'cross':
      if xa is not None:
        k, v = self.key(xa), self.value(xa)
        if not hasattr(self, 'cache_k'):
          self.cache_k, self.cache_v = k, v
        else:
          self.cache_k.assign(k).realize()
          self.cache_v.assign(v).realize()
      else:
        k, v = self.cache_k, self.cache_v
    else:
      k, v = self.key(x), self.value(x)
      if self.kv_caching == 'self':
        if not hasattr(self, 'cache_k'):
          self.cache_k = Tensor.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2])
          self.cache_v = Tensor.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2])
        k = self.cache_k.shrink((None, (0, len), None)).cat(k, dim=1)
        v = self.cache_v.shrink((None, (0, len), None)).cat(v, dim=1)
        padding = self.max_self_attn_cache_len-len-x.shape[1]
        self.cache_k.assign(k.pad((None, (0, padding), None)).contiguous()).realize()
        self.cache_v.assign(v.pad((None, (0, padding), None)).contiguous()).realize()

    q = self.query(x)
    n_ctx = q.shape[1]
    assert(q.shape[-1] == k.shape[-1] == v.shape[-1])
    head_dim = q.shape[-1] // self.n_head
    q = q.reshape(*q.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(*k.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(*v.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    attn = Tensor.scaled_dot_product_attention(q, k, v, mask[:n_ctx,:n_ctx] if mask is not None else None)
    wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
    return self.out(wv)


class ResidualAttentionBlock:
  def __init__(self, n_state, n_head, is_decoder_block=False, max_self_attn_cache_len=None):
    self.attn = MultiHeadAttention(n_state, n_head, kv_caching='self' if is_decoder_block else None, max_self_attn_cache_len=max_self_attn_cache_len)
    self.attn_ln = nn.LayerNorm(n_state)

    self.cross_attn = MultiHeadAttention(n_state, n_head, kv_caching='cross') if is_decoder_block else None
    self.cross_attn_ln = nn.LayerNorm(n_state) if is_decoder_block else None

    self.mlp = [nn.Linear(n_state, n_state*4), Tensor.gelu, nn.Linear(n_state*4, n_state)]
    self.mlp_ln = nn.LayerNorm(n_state)

  def __call__(self, x, xa=None, mask=None, len: Union[Variable, int]=None):
    x = x + self.attn(self.attn_ln(x), mask=mask, len=len)
    if self.cross_attn: x = x + self.cross_attn(self.cross_attn_ln(x), xa)
    x = x + self.mlp_ln(x).sequential(self.mlp)
    return x.realize()

class AudioEncoder:
  def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, **_):
    self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
    self.blocks = [ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
    self.ln_post = nn.LayerNorm(n_audio_state)
    self.positional_embedding = Tensor.empty(n_audio_ctx, n_audio_state)
    self.encode = TinyJit(self.__call__)

  def __call__(self, x):
    x = self.conv1(x).gelu()
    x = self.conv2(x).gelu()
    x = x.permute(0, 2, 1)
    x = x + self.positional_embedding[:x.shape[1]]
    x = x.sequential(self.blocks)
    x = self.ln_post(x)
    return x.realize()

class TextDecoder:
  def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, **_):
    self.max_tokens_to_sample = n_text_ctx // 2
    self.max_self_attn_cache_len = n_text_ctx

    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
    self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head, is_decoder_block=True, max_self_attn_cache_len=self.max_self_attn_cache_len) for _ in range(n_text_layer)]
    self.ln = nn.LayerNorm(n_text_state)
    self.mask = Tensor.full((n_text_ctx, n_text_ctx), -np.inf).triu(1).realize()
    self.getjitted = collections.defaultdict(lambda: TinyJit(self.forward))

  def __call__(self, x: Tensor, pos: int, encoded_audio: Tensor):
    pos = Variable("self_attn_cache_len", 1, self.max_self_attn_cache_len-1).bind(pos) if pos else 0
    return self.getjitted[x.shape](x, pos, encoded_audio)

  def forward(self, x:Tensor, pos:Union[Variable, Literal[0]], encoded_audio:Tensor):
    seqlen = x.shape[-1]
    x = self.token_embedding(x) + self.positional_embedding.shrink(((pos, pos+seqlen), None, None))
    for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=pos)
    return self.output_tok(x)

  def output_tok(self, x):
    return (self.ln(x) @ self.token_embedding.weight.T).realize()

  def rearrange_kv_cache(self, beam_indices: List[int], length=None):
    for block in self.blocks:
      if block.attn.kv_caching != 'self': continue
      for cache_attr in ['cache_k', 'cache_v']:
        cache = getattr(block.attn, cache_attr, None)
        if cache is None: continue
        selected = [cache[i] for i in beam_indices]
        if isinstance(length, int): selected = [s[:, :length] for s in selected]
        selected_tensor = selected[0].stack(*selected[1:], dim=0)
        getattr(block.attn, cache_attr).assign(selected_tensor.contiguous()).realize()

class Whisper:
  def __init__(self, dims, batch_size=1):
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)
    self.is_multilingual = dims["n_vocab"] == 51865
    self.batch_size = batch_size


RATE = 16000
SEGMENT_SECONDS=30
SAMPLES_PER_SEGMENT = RATE * SEGMENT_SECONDS # 480000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH # 3000

def pad_or_trim(arr, target_len):
  diff = target_len - arr.shape[-1]
  if diff == 0: return arr
  elif diff > 0:
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, diff)
    return np.pad(arr, pad, mode='constant')
  else: return arr[(..., slice(0, target_len))]

def prep_audio(waveforms: List[np.ndarray], batch_size: int, truncate=False) -> np.ndarray:
  """
  :param waveforms: A list of possibly variable length 16000Hz audio samples
  :param batch_size: The batch_size associated with the Whisper model being used to transcribe the audio.
                     Used to prevent JIT mismatch errors since the encoder does not accept symbolic shapes
  :param truncate: If true, truncates (or pads) audio to exactly 30s for a single encoder pass
  :return: mel spectrogram of the given waveforms
  """

  max_len = SAMPLES_PER_SEGMENT if truncate else max(len(wav) for wav in waveforms)
  if (r := max_len % SAMPLES_PER_SEGMENT) > 0: max_len += SAMPLES_PER_SEGMENT - r
  waveforms = np.array(list(map(lambda w: pad_or_trim(w, max_len), waveforms)))
  assert waveforms.shape[0] <= batch_size
  if waveforms.shape[0] < batch_size:
    # we could have a symbolic batch_size dim instead of manually padding here if conv/layernorm supported symbolic shapes
    waveforms = np.pad(waveforms, pad_width=((0, batch_size - waveforms.shape[0]), (0, 0)))

  stft = librosa.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.csingle)
  magnitudes = np.absolute(stft[..., :-1]) ** 2
  mel_spec = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS) @ magnitudes

  log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
  log_spec = np.maximum(log_spec, log_spec.max((1,2), keepdims=True) - 8.0)
  log_spec = (log_spec + 4.0) / 4.0

  return log_spec

LANGUAGES = {
  "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
  "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
  "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
  "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu",
  "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
  "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
  "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian",
  "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole",
  "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy",
  "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese",
}

def get_encoding(encoding_name):
  with fetch(f"https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{encoding_name}.tiktoken").open() as f:
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
  n_vocab = len(ranks)
  specials = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
  ]
  special_tokens = dict(zip(specials, itertools.count(n_vocab)))
  n_vocab += len(specials)
  import tiktoken
  return tiktoken.Encoding(
    name=encoding_name,
    explicit_n_vocab=n_vocab,
    pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks=ranks,
    special_tokens=special_tokens)

MODEL_URLS = {
  "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
  "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
  "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
  "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
  "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
  "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
  "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
  "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
  "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
  "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
  "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}
def init_whisper(model_name="tiny.en", batch_size=1):
  assert MODEL_URLS[model_name] is not None

  filename = fetch(MODEL_URLS[model_name])
  state = torch_load(filename)
  model = Whisper(state['dims'], batch_size)
  load_state_dict(model, state['model_state_dict'], strict=False)
  enc = get_encoding("multilingual" if model.is_multilingual else "gpt2")
  return model, enc

def load_file_waveform(filename):
  waveform, _ = librosa.load(filename, sr=RATE)
  return waveform

def transcribe_file(model, enc, filename, **kwargs):
  return transcribe_waveform(model, enc, [load_file_waveform(filename)], **kwargs)

def transcribe_waveform(model: Whisper, enc, waveforms, use_beam=False, use_timestamps=False, truncate=False):
  """
  Expects an array of shape (N,S) where N is the number waveforms to transcribe in parallel and S is number of 16000Hz samples
  Returns the transcribed text if a single waveform is provided, or an array of transcriptions if multiple are provided
  """

  log_spec = prep_audio(waveforms, 1 if use_beam else model.batch_size, truncate)
  nsample = model.decoder.max_tokens_to_sample

  def apply_logit_rules(ctx, logits):
    special_ids = [enc._special_tokens[c] for c in ["<|startoftranscript|>", "<|translate|>", "<|transcribe|>","<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]]
    notimestamp = special_ids[-1]
    logits[:, special_ids] = -np.inf

    for i, row in enumerate(ctx):
      last_tok = row[-1]
      penult_tok = row[-2] if len(row) > 1 else None

      if last_tok == start_tokens[-1]: logits[i, enc.encode(' ') + [eot]] = -np.inf
      if not use_timestamps: continue

      ts_indices = np.where(row > notimestamp)[0]
      last_time_tok = row[ts_indices[-1]] if len(ts_indices) > 0 else notimestamp
      if last_tok == start_tokens[-1]:
        logits[i, :notimestamp] = -np.inf
        logits[i, enc._special_tokens['<|1.00|>'] + 1:] = -np.inf
      elif penult_tok == start_tokens[-1]: logits[i, notimestamp:] = -np.inf
      elif last_tok > notimestamp:
        if penult_tok is not None and penult_tok > notimestamp: logits[i, notimestamp:] = -np.inf
        else: logits[i, np.r_[:eot, notimestamp:last_time_tok]] = -np.inf
      else:
        logprobs = log_softmax(logits[i])
        ts_prob = logsumexp(logprobs[last_time_tok:])
        text_prob = np.max(logprobs[:eot])
        if ts_prob > text_prob: logits[i, np.r_[:eot, notimestamp:last_time_tok + 1]] = -np.inf

    return logits

  def sample(ctx, next_logits, sum_logprobs):
    next_logits = apply_logit_rules(ctx, next_logits)
    logprobs = log_softmax(next_logits, axis=-1)
    last_eot = (ctx[:, -1] == eot)
    if use_beam:
      bs, vs = model.batch_size, next_logits.shape[-1]
      mask = np.full_like(logprobs, -np.inf)
      mask[:, eot] = 0
      logprobs = logprobs[0, :].flatten() if ctx[0, -1] == start_tokens[-1] else (np.where(last_eot[:, None], mask, logprobs) + sum_logprobs.reshape(-1, 1)).flatten()
      top_idxs = np.argpartition(logprobs, -bs)[-bs:]
      top_idxs = top_idxs[np.argsort(-logprobs[top_idxs])]
      beam_idxs, tokens = top_idxs // vs, (top_idxs % vs).reshape(-1, 1)
      sum_logprobs, ctx = logprobs[top_idxs], ctx[beam_idxs]
      model.decoder.rearrange_kv_cache(beam_idxs.tolist())
    else:
      tokens = np.argmax(next_logits, axis=-1).astype(np.int32).reshape(-1, 1)
      sum_logprobs += logprobs[np.arange(logprobs.shape[0]), tokens[:, 0]].reshape(-1) * ~last_eot
    tokens[last_eot] = eot
    ctx = np.concatenate((ctx, tokens), axis=1)
    return tokens, ctx, ctx.shape[-1] - 1, sum_logprobs

  def get_ctx_lens(ctx): return [len(seq) - (r := np.argmax(seq[::-1] != eot)) - ((i := len(seq) - r) >= 2 and seq[i-2] > eot and seq[i-1] > eot) for seq in ctx]

  def rank(ctx_lens, logprobs, length_penalty=None):
    best_idx = int(np.argmax([logprob / (length if length_penalty is None else ((5+length)/6)**length_penalty) for logprob, length in zip(logprobs, ctx_lens)]))
    return best_idx

  def inferloop(ctx: Union[np.ndarray, List[np.ndarray]], encoded_audio):
    pos, next_tokens, sum_logprobs = 0, ctx, [0]*ctx.shape[0]
    for i in range(nsample-len(start_tokens)):
      next_logits = model.decoder(Tensor(next_tokens), pos, encoded_audio)[:, -1].numpy()
      next_tokens, ctx, pos, sum_logprobs = sample(ctx, next_logits, sum_logprobs)
      if (next_tokens == eot).all(): break
    if use_beam:
      ctx_lens = get_ctx_lens(ctx)
      idx = rank(ctx_lens, sum_logprobs)
      model.decoder.rearrange_kv_cache([idx]*model.batch_size, ctx_lens[idx])
      ctx = np.tile(ctx[idx], (model.batch_size, 1))
    return ctx

  notimestamp = enc._special_tokens["<|notimestamps|>"]
  def gettexttoks(line):
    toks = [tok for tok in line if tok < eot or tok > notimestamp]
    if len(toks)>1 and toks[-1]>notimestamp and toks[-2]>notimestamp: toks = toks[:-1]
    return toks[-nsample+len(start_tokens):]

  def ts2frame(tokens):
    token = tokens[tokens > notimestamp][-1] if np.any(tokens > notimestamp) else None
    return FRAMES_PER_SEGMENT if token is None or token<notimestamp else int(float(enc.decode([token])[2:-2])*RATE//HOP_LENGTH)

  def group_sequence(data: np.ndarray):
    data = np.insert(data, 0, 0)
    result, i, n = [], 0, len(data)
    if n >= 3: _, current_first, i = result.append([data[0], data[1], data[2]]), data[0], 3
    while i < n:
      if i + 1 < n and i>1 and data[i+1] == data[i] + 1 and data[i-1] == data[i] - 1: _, current_first, i = result.append([data[i], data[i+1], data[i+2]]), data[i], i+3
      else:
        if i + 1 < n: _, i = result.append([current_first, data[i], data[i+1]]), i+2
        else: break
    return result

  def ctx2segs(line):
    text = [tok for tok in line if tok<eot]
    outline = group_sequence(np.where(line > notimestamp)[0])
    segments, curr_seek, seek_sum = [], 0, 0
    for key_idxs in outline:
      time_tokens, text_tokens = line[key_idxs], line[key_idxs[1]+1:key_idxs[2]]
      seek_frame, start_frame, end_frame = tuple(map(ts2frame, time_tokens))
      if seek_frame!=curr_seek:
        seek_sum += seek_frame
        curr_seek = seek_frame
      segments.append({'start': round((seek_sum+start_frame)/100/0.2)*0.2, 'end': round((seek_sum+end_frame)/100/0.2)*0.2, 'text': enc.decode(text_tokens)})
    return {'text': enc.decode(text), 'segments': segments}

  start_tokens = [enc._special_tokens["<|startoftranscript|>"]]
  if model.is_multilingual:
    # TODO detect language
    language_token = enc._special_tokens["<|startoftranscript|>"] + 1 + tuple(LANGUAGES.keys()).index("en")
    start_tokens.append(language_token)
    start_tokens.append(enc._special_tokens["<|transcribe|>"])
  if not use_timestamps: start_tokens.append(notimestamp)

  eot = enc._special_tokens["<|endoftext|>"]

  ctx = np.tile(start_tokens, (model.batch_size,1))
  transcriptions = [[] for _ in waveforms]

  curr_frame, max_frames = 0, max(map(len, waveforms))//HOP_LENGTH - RATE//HOP_LENGTH

  while curr_frame<max_frames:
    encoded_audio = model.encoder.encode(Tensor(pad_or_trim(log_spec[:, :, curr_frame:curr_frame+FRAMES_PER_SEGMENT], FRAMES_PER_SEGMENT)))
    if all(len(c) == len(ctx[0]) for c in ctx): ctx = inferloop(np.array(ctx), encoded_audio)
    else: ctx = [inferloop((np.array([c]*model.batch_size)), encoded_audio)[i] for i,c in enumerate(ctx)]
    for i, (res, arr) in enumerate(zip(transcriptions, ctx)):
      if curr_frame*HOP_LENGTH <= len(waveforms[i]): res.extend(arr[np.where(arr == start_tokens[-1])[0][0]+1:eoti[0] if len (eoti:=np.where(arr == eot)[0]) else None])
    curr_frame += FRAMES_PER_SEGMENT if not use_timestamps else ts2frame(ctx[0, :])
    ctx = [[enc._special_tokens['<|startofprev|>']]+gettexttoks(cs)+start_tokens for cs in ctx]

  transcriptions = list(map(lambda tokens: enc.decode(tokens).strip() if not use_timestamps else ctx2segs(np.asarray(tokens)), transcriptions))
  if use_timestamps:
    with open('transcription.json', 'w') as f: json.dump(transcriptions[0], f)
    transcriptions = [t['text'] for t in transcriptions]
  return transcriptions if len(transcriptions) > 1 else transcriptions[0]

CHUNK = 1600
RECORD_SECONDS = 10

def listener(q):
  import pyaudio
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
  print("listening")
  for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    waveform = ((np.frombuffer(data, np.int16)/32768).astype(np.float32)*3)
    q.put(waveform)
  print("done listening")

if __name__ == "__main__":
  beam_size = getenv("BEAMSIZE", 1)
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en", batch_size=beam_size)

  if len(sys.argv) > 1:
    print(transcribe_file(model, enc, sys.argv[1], use_beam=beam_size>1, use_timestamps=bool(getenv('TIMESTAMPS', 0))))
  else:
    # online
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    total = None
    did_read = False
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      while not q.empty() or total is None:
        waveform = q.get()
        if total is None: total = waveform
        else: total = np.concatenate([total, waveform])
        did_read = True
      if did_read:
        log_spec = prep_audio(total.reshape(1, -1), model.batch_size, truncate=True)
        encoded_audio = model.encoder.encode(Tensor(log_spec))
      # pass the previously inferred tokens as 'prefix' - https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
      out = model.decoder(Tensor([lst]), 0, encoded_audio, streaming=True).realize()
      idx = int(out[0,-1].argmax().numpy().item())
      lst.append(idx)
      dec = enc.decode(lst)
      print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
      if dec.endswith("<|endoftext|>"):
        lst.pop()