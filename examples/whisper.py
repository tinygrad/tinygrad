# thanks to https://github.com/openai/whisper for a good chunk of MIT licensed code

import sys, base64, multiprocessing, itertools, collections, zlib, datetime, math, argparse, json
from scipy.special import log_softmax
from dataclasses import dataclass
from typing import Optional, Union, Literal, List
from tinygrad import Tensor, TinyJit, nn, Variable
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.helpers import fetch, trange
from tinygrad.uop.ops import UOp
import numpy as np
import librosa
import tiktoken
from tiktoken import Encoding
import functools
import os

RATE = 16000
SEGMENT_SECONDS = 30
SAMPLES_PER_SEGMENT = RATE * SEGMENT_SECONDS
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH
CHUNK = 1600
RECORD_SECONDS = 10

@functools.lru_cache(maxsize=16)
def get_mel_filters(n_mels=80, n_fft=400, sr=16000):
  filters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mel_filters_{n_mels}_{n_fft}_{sr}.npz")
  if os.path.exists(filters_path):
    return np.load(filters_path, allow_pickle=False)['mel_filters']
  filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
  np.savez_compressed(filters_path, mel_filters=filters)
  return filters

def prep_audio(waveforms: List[np.ndarray], batch_size: int, truncate=False) -> np.ndarray:
  max_len = SAMPLES_PER_SEGMENT if truncate else max(len(wav) for wav in waveforms)
  if (r := max_len % SAMPLES_PER_SEGMENT) > 0: max_len += SAMPLES_PER_SEGMENT - r
  waveforms = np.stack([np.pad(wav[:max_len], (0, max(0, max_len - len(wav))), 'constant') for wav in waveforms])
  if waveforms.shape[0] < batch_size:
    waveforms = np.pad(waveforms, ((0, batch_size - waveforms.shape[0]), (0, 0)))
  stft = librosa.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.csingle)
  magnitudes = np.abs(stft[..., :-1]) ** 2
  mel_filters_matrix = get_mel_filters(N_MELS, N_FFT, RATE)
  log_spec = np.log10(np.clip(mel_filters_matrix @ magnitudes, 1e-10, None))
  log_spec = np.maximum(log_spec, log_spec.max((1,2), keepdims=True) - 8.0)
  return (log_spec + 4.0) / 4.0

def compression_ratio(text) -> float:
  text_bytes = text.encode("utf-8")
  return len(text_bytes) / len(zlib.compress(text_bytes))

def argmax_sampling(logits: np.ndarray):
  logits = Tensor(logits)
  softmax = logits.log_softmax(axis=-1)
  idx = softmax.argmax(axis=-1)
  prob = softmax.max(axis=-1)
  return idx.numpy(), prob.numpy()

def multinomial_sampling(logits: Tensor, temperature: int):
  scaled = logits / temperature if temperature != 0 else logits
  probs = scaled.softmax(axis=-1)
  next_tokens = probs.multinomial(1)
  selected_probs = probs[Tensor.arange(logits.shape[0]), next_tokens.flatten()]
  return next_tokens.numpy().astype(np.int64), selected_probs.numpy().astype(np.int64)

class CacheDict(collections.OrderedDict):
  def __init__(self, *args, cache_len: int = 10, **kwargs):
    assert cache_len > 0
    self.cache_len = cache_len
    super().__init__(*args, **kwargs)
  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    super().move_to_end(key)
    while len(self) > self.cache_len:
      oldkey = next(iter(self))
      super().__delitem__(oldkey)
  def __getitem__(self, key):
    val = super().__getitem__(key)
    super().move_to_end(key)
    return val

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

  def rearrange_kv_cache(self, indices: list[int]):
    if hasattr(self, "cache_k") and indices != list(range(len(self.cache_k))):
      for i, idx in enumerate(indices):
        self.cache_k[i] = self.cache_k[idx]
        self.cache_v[i] = self.cache_v[idx]
        self.cache_k.realize()
        self.cache_v.realize()


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
    self.max_self_attn_cache_len = self.max_tokens_to_sample * 4 + 10  # extra buffer

    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
    self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head, is_decoder_block=True, max_self_attn_cache_len=self.max_self_attn_cache_len) for _ in range(n_text_layer)]
    self.ln = nn.LayerNorm(n_text_state)
    self.mask = Tensor.full((n_text_ctx, n_text_ctx), -np.inf).triu(1).realize()
    self.jit = CacheDict(cache_len=2)

  def get_jitted(self, shape): return self.jit[shape] if shape in self.jit else self.jit.setdefault(shape, TinyJit(self.forward))

  def __call__(self, x: Tensor, pos: int, encoded_audio: Tensor):
    max_pos = min(447, (self.max_self_attn_cache_len - x.shape[1]) // 2)
    clamped_pos = min(pos, max_pos) if pos else 0
    pos = UOp.variable("self_attn_cache_len", 1, max_pos).bind(clamped_pos) if clamped_pos else 0
    cache_key = (x.shape, x.dtype, encoded_audio.shape, encoded_audio.dtype)
    if cache_key not in self.jit:
      self.jit[cache_key] = TinyJit(self.forward)
    return self.jit[cache_key](x, pos, encoded_audio)

  def forward(self, x:Tensor, pos:Union[Variable, Literal[0]], encoded_audio:Tensor):
    seqlen = x.shape[-1]
    x = self.token_embedding(x) + self.positional_embedding.shrink(((pos, pos+seqlen), None, None))
    for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=pos)
    return self.output_tok(x)

  def output_tok(self, x):
    return (self.ln(x) @ self.token_embedding.weight.T).realize()

class Whisper:
  def __init__(self, dims, batch_size=1):
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)
    self.is_multilingual = dims["n_vocab"] == 51865
    self.batch_size = batch_size


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
  ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in fetch(f"https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{encoding_name}.tiktoken").open() if line)}
  specials = ["<|endoftext|>", "<|startoftranscript|>", *[f"<|{lang}|>" for lang in LANGUAGES.keys()], "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]]
  return Encoding(name=encoding_name, explicit_n_vocab=len(ranks) + len(specials), pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", mergeable_ranks=ranks, special_tokens=dict(zip(specials, itertools.count(len(ranks)))))

enc = get_encoding("gpt2")

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

def load_file_waveform(filename): return librosa.load(filename, sr=RATE)[0]
def transcribe_file(model, enc: Encoding, filename, output_fh=None, beam=False, no_speech_threshold=0.8):
  return transcribe_waveform(model, enc, [load_file_waveform(filename)], output_fh, beam=beam, no_speech_threshold=no_speech_threshold)

def timestamp_gen(logits: Tensor, enc: Encoding):
  notimestamps_token = enc._special_tokens["<|notimestamps|>"]
  logits[:, notimestamps_token] = -math.inf
  timestamp = enc._special_tokens["<|0.00|>"]
  log_probs: Tensor = logits.log_softmax(axis=-1)
  timestamp_probs = log_probs[:, timestamp:].logsumexp(axis=-1)
  text_probs = log_probs[:, :timestamp].logsumexp(axis=-1)
  mask = (timestamp_probs > text_probs).unsqueeze(-1).expand(-1, timestamp)
  logits[:, :timestamp] = logits[:, :timestamp].masked_fill(mask, -math.inf)

def timestamp_rules(logits: np.ndarray, enc: Encoding, ctx: np.ndarray, sample_begin: int):
  timestamp_begin = enc._special_tokens["<|0.00|>"]
  for k in range(ctx.shape[0]):
    seq = ctx[k, sample_begin:].tolist()
    last_was_timestamp = bool(seq and seq[-1] >= timestamp_begin)
    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin
    if last_was_timestamp: logits[k, slice(timestamp_begin if penultimate_was_timestamp else 0, None if penultimate_was_timestamp else enc._special_tokens["<|endoftext|>"])] = -np.inf
    timestamps = ctx[k, sample_begin:][ctx[k, sample_begin:] >= timestamp_begin]
    if timestamps.size > 0: logits[k, timestamp_begin:timestamps[-1] + (0 if last_was_timestamp and not penultimate_was_timestamp else 1)] = -np.inf
  if ctx.shape[1] == sample_begin:
    logits[:, :timestamp_begin] = -np.inf
    logits[:, timestamp_begin + 50 + 1:] = -np.inf

def non_speech_filter(logits: np.ndarray):
  tokens = np.array([1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50358, 50357, 50257, 50360, 50359, 50361])
  logits[:, tokens] = -math.inf

def suppress_blank(logits: np.ndarray, enc: Encoding):
  tokens = np.array([enc.encode(" ")[0], enc._special_tokens["<|endoftext|>"]])
  logits[:, tokens] = -math.inf

def replace_eot(tokens: np.ndarray, ctx: np.ndarray, eot: int): return np.where(ctx[:, -1] == eot, eot, tokens)

@dataclass
class BeamSampled:
  ctx: np.ndarray
  next_tokens: np.ndarray
  finished: np.ndarray
  probs: np.ndarray
  finished_prob: np.ndarray

def beamsearch_sampling(logits: np.ndarray, model: Whisper, sum_probs: np.ndarray, ctx: np.ndarray, beam_size: int=5, topk: int=5):
  eot = enc._special_tokens["<|endoftext|>"]
  log_probs: np.ndarray = log_softmax(logits, axis=-1)
  top_k_idx = np.argpartition(log_probs, -beam_size)[:, -beam_size:]
  top_k_probs = np.take_along_axis(log_probs, top_k_idx, axis=-1)
  _tokens = [enc.decode([idx]) for idx in top_k_idx.reshape(-1)]
  candidates = {
      (sum_probs[i] + top_k_probs[i][j]).tolist(): (i, j, top_k_probs[i][j])
      for i, j in itertools.product(range(beam_size), range(topk))
  }
  next_tokens, probs, indices, finished, finished_prob, selected_sequence = [], [], [], [], [], []
  saved = 0

  for prob, (new_i, j, _prob) in sorted(candidates.items(), key=lambda items: items[0], reverse=True):
    token = top_k_idx[new_i][j]
    sequence = np.concatenate((ctx[new_i], np.array([token])))
    if token == eot:
      finished.append(sequence)
      finished_prob.append(prob)
    else:
      selected_sequence.append(sequence)
      indices.append(new_i)
      next_tokens.append(token)
      probs.append(prob)
      saved += 1
    if saved >= 5:
      break

  next_tokens = np.array(next_tokens).reshape((-1, 1))
  ctx = np.array(selected_sequence) if indices != list(range(beam_size)) else np.concat((ctx, next_tokens), axis=1)

  for block in model.decoder.blocks:
    block.attn.rearrange_kv_cache(indices)
  return BeamSampled(ctx=ctx, next_tokens=next_tokens, probs=np.array(probs), finished=finished, finished_prob=finished_prob)

def inferloop(model: Whisper, ctx: np.ndarray, encoded_audio: Tensor, temperature: int, num_sample: int, enc: Encoding, beam=False) -> np.ndarray:
  eot, pos, next_tokens, sample_begin = enc._special_tokens["<|endoftext|>"], 0, ctx, ctx.shape[1]
  sum_probs, beam_sequence, beam_sequence_probs = np.zeros(ctx.shape[0]), [], []
  for i in trange(num_sample):
    logits = model.decoder(Tensor(next_tokens), pos, encoded_audio)[:, -1].contiguous().numpy()
    if i == 0: suppress_blank(logits, enc)
    non_speech_filter(logits), timestamp_rules(logits, enc, ctx, sample_begin), timestamp_gen(Tensor(logits), enc)
    if beam and temperature == 0:
      sampled = beamsearch_sampling(logits, model, sum_probs, ctx)
      ctx, next_tokens, sum_probs = sampled.ctx, sampled.next_tokens, sampled.probs
      if len(sampled.finished) > 0:
        beam_sequence.extend(sampled.finished), beam_sequence_probs.extend(sampled.finished_prob)
        if len(beam_sequence) >= 5: return beam_sequence, np.array(beam_sequence_probs)
    else:
      next_tokens, probs = (argmax_sampling(logits) if temperature == 0 else multinomial_sampling(Tensor(logits), temperature))
      if probs.shape[0] != sum_probs.shape[0]:
        probs = probs[:sum_probs.shape[0]]
      sum_probs += probs
      if (next_tokens := replace_eot(next_tokens.flatten(), ctx, eot)).all() == eot: break
      next_tokens = next_tokens.reshape((-1, 1))[:ctx.shape[0]]  # Match batch dimension
      ctx = np.concat((ctx, next_tokens), axis=1)
    pos = ctx.shape[-1] - 1
  return ctx, sum_probs

@dataclass
class Segment:
  start: int
  end: int
  text: str
  tokens: list[int]

def parse(timetoken: str): return float(timetoken[2:-2])
def format_time(seconds: int): return str(datetime.timedelta(seconds=seconds))
def segment_and_seek(tokens: list[int], enc: Encoding, timeoffset: int):
  timestamp_pos = [i for i, tok in enumerate(tokens) if tok > enc._special_tokens["<|notimestamps|>"]]
  if len(timestamp_pos) < 2:
    text = [tok for tok in tokens if tok < enc._special_tokens["<|notimestamps|>"]]
    yield Segment(timeoffset, timeoffset + 30, enc.decode(text), text)
  else:
    total_timestamps = len(timestamp_pos)
    for i in range(0, total_timestamps - total_timestamps % 2,2):
      s = timestamp_pos[i]
      e = timestamp_pos[i+1]
      start_time = parse(enc.decode([tokens[s]])) + timeoffset
      end_time = parse(enc.decode([tokens[e]])) + timeoffset
      selected = tokens[s+1:e]
      selected_with_time = tokens[s: e+1]
      yield Segment(start_time, end_time, enc.decode(selected), selected_with_time)

def get_start_tokens(enc: Encoding, is_multilingual: bool=False):
  start_tokens = [enc._special_tokens["<|startoftranscript|>"]]
  if is_multilingual:
    # TODO detect language
    language_token = enc._special_tokens["<|startoftranscript|>"] + 1 + tuple(LANGUAGES.keys()).index("en")
    start_tokens.append(language_token)
    start_tokens.append(enc._special_tokens["<|transcribe|>"])
  return start_tokens

def pad_log_spec_if_needed(log_spec, end_frame):
  return np.pad(log_spec, ((0, 0), (0, 0), (0, end_frame - log_spec.shape[2]))) if end_frame > log_spec.shape[2] else log_spec

def get_segment_tokens(selected, start_tokens, eot, enc):
  eoti = index[0] if (index := (np.where(selected == eot)[0])).size > 0 else None
  soti = np.where(selected == start_tokens[-1])[0][0] + 1
  tokens = selected[soti:eoti]
  text = enc.decode(tokens)
  return tokens, text

def should_skip_segment(selected_avg_prob, compression, no_speech_threshold):
  prob_threshold = -3.5
  is_no_speech = selected_avg_prob < prob_threshold or (compression < 0.5 and selected_avg_prob < -0.5 * no_speech_threshold)
  too_repetitive = compression >= 5.0
  return is_no_speech, too_repetitive

def get_valid_segments(tokens, enc, start_time):
  segments = list(segment_and_seek(tokens, enc, start_time))
  return [segment for segment in segments if segment.text.strip()]

def update_transcriptions(transcriptions, segment, batch_idx, waveforms, log_spec):
  idx = batch_idx if len(waveforms) > 1 and log_spec.shape[0] > 1 else 0
  while len(transcriptions) <= idx: transcriptions.append("")
  transcriptions[idx] += segment.text.strip() + " "

def print_and_write_segment(segment, output_fh):
  (output_fh.write(f"{text}\n") if output_fh else None) if (text := f"{format_time(int(segment.start))} -> {format_time(int(segment.end))}: {segment.text}") and print(f"\033[31m{text}\033[0m") else None

def get_next_context(context_for_next, nsample, start_tokens, enc):
  return np.array([enc._special_tokens['<|startofprev|>']]+context_for_next[-nsample+len(start_tokens):]+start_tokens) if context_for_next else np.array(start_tokens)

def process_temperature_loop(model, to_decode, encoded_audio, nsample, start_tokens, eot, enc, beam, no_speech_threshold, start_time, output_fh, transcribed, transcriptions, waveforms, log_spec):
  actual_audio_duration = max(len(wav) for wav in waveforms) / RATE

  for t in np.linspace(0, 1.0, 6):
    print(f"temperature {t:.1f}")
    infer_result = inferloop(model, to_decode, encoded_audio, t, (nsample-len(start_tokens))*2, enc, beam)
    if infer_result is None:
      print("infer result is None, resampling")
      continue
    inferred, sum_probs = infer_result
    sum_probs = sum_probs.tolist()
    avg_probs = [sum_probs[i] / len(sequence) for i, sequence in enumerate(inferred)]
    candidate_idx = avg_probs.index(max(avg_probs))
    print(f"\033[34m{avg_probs=} {candidate_idx=}\033[0m")
    selected: np.ndarray = inferred[candidate_idx]
    selected_avg_prob = avg_probs[candidate_idx]
    tokens, text = get_segment_tokens(selected, start_tokens, eot, enc)
    print(f"Decoder output: {text=}")
    compression = compression_ratio(text)
    print(f"\033[33m{compression=}, {selected_avg_prob=}\033[0m")

    is_post_audio = start_time >= actual_audio_duration - 1.0  # Within 1 second of audio end
    if is_post_audio and selected_avg_prob > -3.0:
      print(f"post-audio hallucination detected (start_time={start_time}, audio_duration={actual_audio_duration}), skipping segment")
      return True, start_time + 30, np.array(start_tokens)

    is_no_speech, too_repetitive = should_skip_segment(selected_avg_prob, compression, no_speech_threshold)
    if too_repetitive:
      print("Too repetitive, resample")
      continue
    if is_no_speech:
      print(f"No speech detected (avg_prob={selected_avg_prob}, threshold={-0.1 * no_speech_threshold}), skipping segment")
      return True, start_time + 30, np.array(start_tokens)
    valid_segments = get_valid_segments(tokens, enc, start_time)
    if not valid_segments:
      print("No speech segments found, skipping to next segment")
      return True, start_time + 30, np.array(start_tokens)
    context_for_next = []
    for segment in valid_segments:
      print_and_write_segment(segment, output_fh)
      context_for_next.extend(segment.tokens)
      transcribed["segments"].append({"text": segment.text, "start": segment.start, "end": segment.end})
      update_transcriptions(transcriptions, segment, 0, waveforms, log_spec)
    return True, valid_segments[-1].end if valid_segments else start_time + 30, get_next_context(context_for_next, nsample, start_tokens, enc)
  return False, start_time + 30, np.array(start_tokens)

def transcribe_waveform(model: Whisper, enc: tiktoken.Encoding, waveforms, output_fh=None, truncate=False, beam=False, no_speech_threshold=0.8):
  log_spec, nsample, eot, start_tokens = prep_audio(waveforms, model.batch_size, truncate), model.decoder.max_tokens_to_sample, enc._special_tokens["<|endoftext|>"], get_start_tokens(enc, model.is_multilingual)

  if len(waveforms) > 1:
    transcriptions = []
    for i, waveform in enumerate(waveforms):
      print(f"\n=== Processing waveform {i+1}/{len(waveforms)} ===")
      individual_result = transcribe_waveform(model, enc, [waveform], output_fh, truncate, beam, no_speech_threshold)
      transcriptions.append(individual_result)
    return transcriptions

  actual_audio_duration = len(waveforms[0]) / RATE
  ctx, curr_frame, start_time, total_time = np.array(start_tokens), 0, 0, actual_audio_duration
  transcribed, transcriptions = {"segments": []}, []
  while start_time < total_time:
    curr_frame, end_frame = int(start_time) * 100, int(start_time) * 100 + FRAMES_PER_SEGMENT
    print(f"\n{curr_frame=} {end_frame=} {start_time=}")
    encoded_audio = model.encoder.encode(Tensor(pad_log_spec_if_needed(log_spec, end_frame)[:, :, curr_frame:curr_frame + FRAMES_PER_SEGMENT]))
    print(f"\033[32mcontext to decoder: {enc.decode(ctx)=}\033[0m")
    to_decode = np.tile(ctx, (5, 1)) if beam else ctx.reshape(1, -1)
    processed, start_time, ctx = process_temperature_loop(model, to_decode, encoded_audio, nsample, start_tokens, eot, enc, beam, no_speech_threshold, start_time, output_fh, transcribed, transcriptions, waveforms, log_spec)
    if not processed: start_time, ctx = start_time + 30, np.array(start_tokens)
  return transcriptions[0].strip() if transcriptions else transcribed

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
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="tiny.en", help="name of model")
  parser.add_argument("--audio", type=str, required=True, help="path to an mp3 audio file")
  parser.add_argument("--beam", action=argparse.BooleanOptionalAction, default=True, help="Whether to use beam decoding")
  parser.add_argument("--no_speech_threshold", type=float, default=0.8, help="Threshold for determining no speech")
  args = parser.parse_args()
  model, enc = init_whisper(args.model, batch_size=1)

  if len(sys.argv) > 1:
    with open("transcribed.txt", "w") as output_fh:
      transcribed = transcribe_file(model, enc, args.audio, output_fh, beam=args.beam, no_speech_threshold=args.no_speech_threshold)
      with open("transcribed.json", "w") as output_fh2:
        json.dump(transcribed, output_fh2, indent=2)

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