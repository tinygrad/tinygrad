# thanks to https://github.com/openai/whisper for a good chunk of MIT licensed code

import sys
import pathlib
import base64
import multiprocessing
import numpy as np
from typing import Optional
from extra.utils import download_file
from tinygrad.state import torch_load, load_state_dict
from tinygrad.helpers import getenv
import tinygrad.nn as nn
from tinygrad.tensor import Tensor

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
    n_batch, n_ctx, n_state = q.shape
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

class AudioEncoder:
  def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, **_):
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
  def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, **_):
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

class Whisper:
  def __init__(self, dims):
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)

  def __call__(self, mel:Tensor, tokens:Tensor):
    return self.decoder(tokens, self.encoder(mel))

# TODO: this is tragic. remove this
import functools
import torch
import torchaudio
import librosa

@functools.lru_cache(None)
def get_filters(sample_rate, n_fft, n_mels):return torch.tensor(librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels))
@functools.lru_cache(None)
def get_window(n_fft): return torch.hann_window(n_fft)

def prep_audio(waveform, sample_rate) -> Tensor:
  N_FFT = 400
  HOP_LENGTH = 160
  N_MELS = 80
  stft = torch.stft(waveform, N_FFT, HOP_LENGTH, window=get_window(N_FFT), return_complex=True)
  magnitudes = stft[..., :-1].abs() ** 2
  mel_spec = get_filters(sample_rate, N_FFT, N_MELS) @ magnitudes
  log_spec = torch.clamp(mel_spec, min=1e-10).log10()
  log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
  #print(waveform.shape, log_spec.shape)
  return log_spec.numpy()

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

BASE = pathlib.Path(__file__).parent.parent / "weights"
def get_encoding(n_vocab_in):
  download_file("https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken", BASE / "gpt2.tiktoken")
  ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(BASE / "gpt2.tiktoken") if line)}
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
  special_tokens = {}
  for token in specials:
    special_tokens[token] = n_vocab
    n_vocab += 1
  assert n_vocab == n_vocab_in
  import tiktoken
  return tiktoken.Encoding(
    name="bob",
    explicit_n_vocab=n_vocab,
    pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks=ranks,
    special_tokens=special_tokens)

def img(x):
  import matplotlib.pyplot as plt
  plt.imshow(x.numpy())
  plt.show()

RATE = 16000
CHUNK = 1600
RECORD_SECONDS = 10

def listener(q):
  prep_audio(torch.zeros(300), RATE)
  import pyaudio
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
  print("listening")
  for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    waveform = ((np.frombuffer(data, np.int16)/32768).astype(np.float32)*3).reshape(1, -1)
    q.put(waveform)
  print("done listening")

if __name__ == "__main__":
  if getenv("SMALL"):
    fn = BASE / "whisper-small.en.pt"
    download_file("https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt", fn)
  else:
    fn = BASE / "whisper-tiny.en.pt"
    download_file("https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt", fn)
  state = torch_load(fn)
  model = Whisper(state['dims'])
  load_state_dict(model, state['model_state_dict'])
  enc = get_encoding(state['dims']['n_vocab'])

  if len(sys.argv) > 1:
    # offline
    waveform, sample_rate = torchaudio.load(sys.argv[1], normalize=True)
    log_spec = prep_audio(waveform, sample_rate)
    lst = [enc._special_tokens["<|startoftranscript|>"]]
    dat = model.encoder(Tensor(log_spec)).realize()
    for i in range(50):
      out = model.decoder(Tensor([lst]), dat)
      out.realize()
      idx = out[0,-1].numpy().argmax()
      lst.append(idx)
      print(enc.decode(lst))
  else:
    # online

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [enc._special_tokens["<|startoftranscript|>"]]
    total = None
    did_read = False
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      while not q.empty() or total is None:
        waveform = q.get()
        if total is None: total = waveform
        else: total = np.concatenate([total, waveform], axis=1)
        did_read = True
      if did_read:
        last_total = total.shape[1]
        log_spec = prep_audio(torch.Tensor(total), RATE)
        encoded_audio = model.encoder(Tensor(log_spec)).realize()
      out = model.decoder(Tensor([lst]), encoded_audio).realize()
      idx = out[0,-1].numpy().argmax()
      lst.append(idx)
      dec = enc.decode(lst)
      print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
      if dec.endswith("<|endoftext|>"):
        #total = total[:, 320*(len(lst)-1):]
        lst = [enc._special_tokens["<|startoftranscript|>"]]
