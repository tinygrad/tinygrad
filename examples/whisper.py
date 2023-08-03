# thanks to https://github.com/openai/whisper for a good chunk of MIT licensed code

import sys
from pathlib import Path
import base64
import multiprocessing
import numpy as np
from typing import List, Optional
from extra.utils import download_file
from tinygrad.state import torch_load, load_state_dict
from tinygrad.helpers import getenv, dtypes
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
import itertools
import librosa
import tiktoken
import pyaudio
from dataclasses import dataclass

def available_models() -> List[str]:
  """Returns the names of available models"""
  return list(_MODELS.keys())

# TODO: you have written this fifteen times
class MultiHeadAttention:
  def __init__(self, n_state, n_head):
    self.n_head = n_head
    self.query = nn.Linear(n_state, n_state)
    self.key = nn.Linear(n_state, n_state, bias=False)
    self.value = nn.Linear(n_state, n_state)
    self.out = nn.Linear(n_state, n_state)

  def __call__(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None):
    q = self.query(x)
    k = self.key(xa or x)
    v = self.value(xa or x)
    wv, qk = self.qkv_attention(q, k, v, mask)
    # NOTE: we aren't returning qk
    return self.out(wv)

  def qkv_attention(self, q, k, v, mask=None):
    _, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.reshape(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.reshape(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.reshape(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    qk = q @ k
    if mask is not None: qk = qk + mask[:n_ctx, :n_ctx]
    w = qk.softmax(-1)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock:
  def __init__(self, n_state, n_head, cross_attention=False):
    self.attn = MultiHeadAttention(n_state, n_head)
    self.attn_ln = nn.LayerNorm(n_state)

    self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
    self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

    self.mlp = [nn.Linear(n_state, n_state*4), Tensor.gelu, nn.Linear(n_state*4, n_state)]
    self.mlp_ln = nn.LayerNorm(n_state)

  def __call__(self, x, xa=None, mask=None):
    x = x + self.attn(self.attn_ln(x), mask=mask)
    if self.cross_attn: x = x + self.cross_attn(self.cross_attn_ln(x), xa)
    x = x + self.mlp_ln(x).sequential(self.mlp)
    return x

class GreedyDecoder:
  def __init__(self, eot): self.eot = eot

  def update(self, tokens, logits, sum_logprobs):
    tokens = tokens.numpy()
    next_tokens = logits.numpy().argmax(-1)
    logprobs = logits.log_softmax(-1).numpy()
    current_logprobs = logprobs[np.arange(0, logprobs.shape[0]), next_tokens]
    sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)
    next_tokens[tokens[:, -1] == self.eot] = self.eot
    tokens = np.concatenate([tokens, next_tokens[:, None]], axis=-1)
    completed = (tokens[:, -1] == self.eot).all()
    return Tensor(tokens), completed
  
class MaximumLikelihoodRanker:
  def __init__(self, length_penalty): self.length_penalty = length_penalty

  def rank(self, tokens, sum_logprobs):
    def scores(logprobs, lengths):
      result = []
      for logprob, length in zip(logprobs, lengths):
        if self.length_penalty is None: penalty = length
        else: penalty = ((5 + length) / 6) ** self.length_penalty
        result.append(logprob / penalty)
      return result
    lengths = [[len(t) for t in s] for s in tokens]
    return [np.argmax(scores(p, l) for p, l in zip(sum_logprobs, lengths))]

class AudioEncoder:
  def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer):
    self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
    self.blocks = [ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
    self.ln_post = nn.LayerNorm(n_audio_state)
    self.positional_embedding = Tensor.empty(n_audio_ctx, n_audio_state)

  def __call__(self, x):
    x = self.conv1(x).gelu()
    x = self.conv2(x).gelu()
    x = x.permute(0, 2, 1)
    x = x + self.positional_embedding[:x.shape[1]]
    x = x.sequential(self.blocks)
    x = self.ln_post(x)
    return x

class TextDecoder:
  def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer):
    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
    self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head, cross_attention=True) for _ in range(n_text_layer)]
    self.ln = nn.LayerNorm(n_text_state)
    #mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)

  def __call__(self, x, xa):
    offset = 0
    x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]

    seqlen, start_pos = x.shape[1], 0

    mask = np.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=np.float32)
    mask = np.triu(mask, k=start_pos + 1)  # TODO: this is hard to do in tinygrad
    mask = Tensor(mask)

    for block in self.blocks: x = block(x, xa, mask)
    x = self.ln(x)
    return x @ self.token_embedding.weight.T

@dataclass
class ModelDimensions:
  n_mels: int
  n_audio_ctx: int
  n_audio_state: int
  n_audio_head: int
  n_audio_layer: int
  n_vocab: int
  n_text_ctx: int
  n_text_state: int
  n_text_head: int
  n_text_layer: int

class Whisper:
  def __init__(self, dims: ModelDimensions):
    self.dims = dims
    self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            )
    self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            )

  @property
  def is_multilingual(self):
    return self.dims.n_vocab == 51865
  
  def __call__(self, mel:Tensor, tokens:Tensor):
    return self.decoder(tokens, self.encoder(mel))
  
  def get_encoding(self):
    if self.is_multilingual:
      download_file("https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken", _BASE / "multilingual.tiktoken")
      ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(_BASE / "multilingual.tiktoken") if line)}
    else:
      download_file("https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken", _BASE / "gpt2.tiktoken")
      ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(_BASE / "gpt2.tiktoken") if line)}
    n_vocab = len(ranks)
    specials = [
      "<|endoftext|>",
      "<|startoftranscript|>",
      *[f"<|{lang}|>" for lang in _LANGUAGES.keys()],
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
    assert n_vocab == self.dims.n_vocab
    return tiktoken.Encoding(
      name="bob",
      explicit_n_vocab=n_vocab,
      pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
      mergeable_ranks=ranks,
      special_tokens=special_tokens)
  
  def load_state_dict(self, state_dict):
    load_state_dict(self, state_dict)

  def decode_segment(self, segment, initial_tokens, sample_len=224, n_audio=1, n_group=1):
    if segment.ndim == 2: segment = np.expand_dims(segment, axis=0)
    texts, sample_begin = [], len(initial_tokens)
    decoder = GreedyDecoder(eot=enc.eot_token)
    sequence_ranker = MaximumLikelihoodRanker(None)
    audio_features = model.encoder(Tensor(segment)).realize()
    tokens = Tensor([initial_tokens], dtype=dtypes.int64).repeat((segment.shape[0], 1))
    sum_logprobs = Tensor.zeros(audio_features.shape[0])
    no_speech_probs = [np.nan] * tokens.shape[0]
    for i in range(sample_len):
      logits = self.decoder(tokens, audio_features)
      probs_at_sot = logits[:, initial_tokens.index(enc._special_tokens["<|startoftranscript|>"])].softmax(axis=-1)
      no_speech_probs = probs_at_sot[:, enc._special_tokens["<|nospeech|>"]].numpy().tolist()
      logits = logits[:, -1]
      tokens, completed = decoder.update(tokens, logits, sum_logprobs)
      if completed: break
    tokens = tokens.reshape(n_audio, n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, n_group)
    tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)
    tokens = [[t[sample_begin:(t == enc.eot_token).nonzero()[0][0]] for t in s] for s in tokens]
    selected = sequence_ranker.rank(tokens, sum_logprobs)
    tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
    texts.extend([enc.decode(t[1:]).strip() for t in tokens])
    sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
    avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]
    return texts
    

_RATE = 16000
_CHUNK = 1600
_RECORD_SECONDS = 10
_HOP_LENGTH = 160
_CHUNK_LENGTH = 30
N_SAMPLES = _CHUNK_LENGTH * _RATE
N_FRAMES = N_SAMPLES // _HOP_LENGTH

def prep_audio(waveform=None, sr=_RATE) -> Tensor:
  N_FFT = 400
  HOP_LENGTH = 160
  N_MELS = 80
  if waveform is None: waveform = np.zeros(N_FFT, dtype=np.float32)
  stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.float32)
  magnitudes = stft[..., :-1] ** 2
  mel_spec = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS) @ magnitudes
  log_spec = np.log10(np.clip(mel_spec, 1e-10, mel_spec.max() + 1e8))
  log_spec = (log_spec + 4.0) / 4.0
  #print(waveform.shape, log_spec.shape)
  return log_spec

_LANGUAGES = {
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

_BASE = Path(__file__).parent.parent / "weights"

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large.en": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
}

def img(x):
  import matplotlib.pyplot as plt
  plt.imshow(x.numpy())
  plt.show()

def listener(q):
  prep_audio()
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=_RATE, input=True, frames_per_buffer=_CHUNK)
  print("listening")
  for _ in range(0, int(_RATE / _CHUNK * _RECORD_SECONDS)):
    data = stream.read(_CHUNK)
    waveform = ((np.frombuffer(data, np.int16)/32768).astype(np.float32)*3).reshape(1, -1)
    q.put(waveform)
  print("done listening")

def load_model(name: str = None):
  if not name:
    sizes = ["TINY", "BASE", "SMALL", "MEDIUM", "LARGE"]
    envs = [x for x in sizes if getenv(x)]
    if len(envs) == 0:
      name = "base.en"
    else:
      name = f"{envs[0].lower()}.en"
  if name not in available_models():
    raise ValueError(f"unknown model {name}")
  fn = _BASE / f"whisper-{name}.pt"
  download_file(_MODELS[name], fn)
  checkpoint = torch_load(fn)
  dims = ModelDimensions(**checkpoint["dims"])
  model = Whisper(dims)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.encoding = model.get_encoding()
  return model

if __name__ == "__main__":
  model = load_model()

  if len(sys.argv) > 1:
    # offline
    waveform, sample_rate = librosa.load(sys.argv[1], normalize=True)
    log_spec = prep_audio(waveform, sample_rate)
    lst = [model.encoding._special_tokens["<|startoftranscript|>"]]
    dat = model.encoder(Tensor(log_spec)).realize()
    for i in range(50):
      out = model.decoder(Tensor([lst]), dat)
      out.realize()
      idx = out[0,-1].numpy().argmax()
      lst.append(idx)
      print(model.encoding.decode(lst))
  else:
    # online
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [model.encoding._special_tokens["<|startoftranscript|>"]]
    total = None
    did_read = False
    for i in range(0, int(_RATE / _CHUNK * _RECORD_SECONDS)):
      while not q.empty() or total is None:
        waveform = q.get()
        if total is None: total = waveform
        else: total = np.concatenate([total, waveform], axis=1)
        did_read = True
      if did_read:
        last_total = total.shape[1]
        log_spec = prep_audio(waveform=Tensor(total).numpy(), sr=_RATE)
        encoded_audio = model.encoder(Tensor(log_spec)).realize()
      out = model.decoder(Tensor([lst]), encoded_audio).realize()
      idx = out[0,-1].numpy().argmax()
      lst.append(idx)
      dec = model.encoding.decode(lst)
      print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
      if dec.endswith("<|endoftext|>"):
        #total = total[:, 320*(len(lst)-1):]
        lst = [model.encoding._special_tokens["<|startoftranscript|>"]]
