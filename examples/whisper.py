# thanks to https://github.com/openai/whisper for a good chunk of MIT licensed code
import argparse
import sys, math, string, difflib, base64, functools, itertools, multiprocessing
import subprocess as sp
from pathlib import Path
from typing import Optional
import librosa
import numpy as np
from tinygrad import nn
from tinygrad.state import torch_load, load_state_dict
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from extra.utils import download_file
from extra.datasets.librispeech import ci, BASEDIR
from examples.mlperf.metrics import word_error_rate

def mel_frequencies(fmin, fmax, n_mels):
    #use approximation, not exact, should be good enough, need to check
    hz_to_mel = lambda freq: 2595 * np.log10(1 + freq / 700)
    mel_to_hz = lambda mels: 700 * (10 ** (mels / 2595) - 1)
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels)
    hz: np.ndarray = mel_to_hz(mels)
    return hz

def mel(sr, n_fft, n_mels):
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_frequencies(0, float(sr)/2, n_mels + 2)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        print('Empty filters in mel frequency basis. Some channels will produce empty responses.')

    return weights

# audio hyperparameters
RATE = 16000
CHUNK = 1600
RECORD_SECONDS = 10
MAX_ITERS = 100
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH
N_MELS = 80

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

  def __call__(self, x, xa):
    offset = 0
    x = self.token_embedding(x) + self.positional_embedding[offset:offset+x.shape[-1]]

    seqlen, start_pos = x.shape[1], 0
    mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf")).triu(k=start_pos+1)

    for block in self.blocks: x = block(x, xa, mask)
    x = self.ln(x)
    return x @ self.token_embedding.weight.T

class Whisper:
  def __init__(self, dims):
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)

  def __call__(self, mel:Tensor, tokens:Tensor):
    return self.decoder(tokens, self.encoder(mel))

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

MODELS = {
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

BASE = Path(__file__).parent.parent / "weights"

def get_encoding(n_vocab_in):
  download_file("https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken", BASE / "multilingual.tiktoken")
  ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(BASE / "multilingual.tiktoken") if line)}
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

def load_wav(file):
  cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(RATE), "-"]
  try: out = sp.run(cmd, capture_output=True, check=True).stdout
  except sp.CalledProcessError as e: raise RuntimeError(f"Failed to load audio {e.stderr.decode()}") from e
  return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

@functools.lru_cache(None)
def get_filters(sample_rate, n_fft, n_mels): return mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
@functools.lru_cache(None)
def get_window(n_fft): return (1 - np.cos(2 * math.pi * np.arange(n_fft) / n_fft)) / 2

def prep_audio(audio, padding) -> Tensor:
  if padding > 0: audio = np.pad(audio, (0, padding))
  stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=get_window(N_FFT))
  magnitudes = np.abs(stft[..., :-1]) ** 2
  mel_spec = get_filters(RATE, N_FFT, N_MELS) @ magnitudes
  log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
  log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
  return log_spec

def listener(q):
    prep_audio(np.zeros(300), RATE)
    import sounddevice as sd
    print("listening")
    def callback(indata, frames, time, status):
        waveform = ((indata).astype(np.float32)).reshape(1, -1)
        q.put(waveform)
    with sd.InputStream(callback=callback, channels=1, samplerate=RATE, blocksize=CHUNK):
        sd.sleep(RECORD_SECONDS * 1000)
    print("done listening")

def pad_or_trim(array, length=N_SAMPLES, axis=-1):
  if array.shape[axis] > length:
    array = array.take(indices=range(length), axis=axis)
  if array.shape[axis] < length:
    pad_widths = [(0, 0)] * array.ndim
    pad_widths[axis] = (0, length - array.shape[axis])
    array = np.pad(array, pad_widths)
  return array

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

  def finalize(self, tokens, sum_logprobs):
    tokens = np.pad(tokens.numpy(), [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
    return tokens, sum_logprobs.numpy().tolist()

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run Whisper in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', type=str, default="tiny", help='model to use')
  parser.add_argument('--test', action='store_true', help='run tests')
  parser.add_argument('--file', type=str, default=None, help='file to transcribe')
  parser.add_argument('--lang', type=str, default="en", help='language to transcribe')
  parser.add_argument('--translate', action='store_true', help='translate to english')
  args = parser.parse_args()
  assert args.lang in LANGUAGES, f"Language {args.lang} not found. Please choose from the following:\n{list(LANGUAGES.keys())}"
  lang = "<|" + args.lang + "|>"
  task = "<|translate|>" if args.translate else "<|transcribe|>"
  if args.model not in MODELS:
    print("Model not found. Please choose from the following:")
    print(list(MODELS.keys()))
    sys.exit(1)
  fn = BASE / f"whisper-{args.model}.pt"
  download_file(MODELS[args.model], fn)
  state = torch_load(fn)
  model = Whisper(state['dims'])
  load_state_dict(model, state['model_state_dict'])
  enc = get_encoding(state['dims']['n_vocab'])

  def decode_segment(segment, initial_tokens, sample_len=224, n_audio=1, n_group=1):
    if segment.ndim == 2: segment = np.expand_dims(segment, axis=0)
    texts, sample_begin = [], len(initial_tokens)
    decoder = GreedyDecoder(eot=enc.eot_token)
    sequence_ranker = MaximumLikelihoodRanker(None)
    audio_features = model.encoder(Tensor(segment)).realize()
    tokens = Tensor([initial_tokens], dtype=dtypes.int64).repeat((segment.shape[0], 1))
    sum_logprobs = Tensor.zeros(audio_features.shape[0])
    [np.nan] * tokens.shape[0]
    for i in range(sample_len):
      logits = model.decoder(tokens, audio_features)
      probs_at_sot = logits[:, initial_tokens.index(enc._special_tokens["<|startoftranscript|>"])].softmax(axis=-1)
      probs_at_sot[:, enc._special_tokens["<|nospeech|>"]].numpy().tolist()
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
    [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]
    return texts

  def transcribe_wav(fn, logprob_threshold=-1.0, no_speech_threshold=0.6):
    mel = prep_audio(load_wav(fn), padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    initial_tokens = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens[lang], enc._special_tokens[task]]
    seek, texts = 0, []
    while seek < content_frames:
      mel_segment = mel[:, seek:seek+N_FRAMES]
      mel_segment = pad_or_trim(mel_segment, N_FRAMES)
      segment_size = min(N_FRAMES, content_frames - seek)
      texts += decode_segment(mel_segment, initial_tokens)
      seek += segment_size
    return "".join(texts)

  if args.test:
    diff = difflib.Differ()
    for c in ci:
      fn = BASEDIR / c["files"][0]["fname"]
      print("-" * 128, f"{fn.stem}\n", sep="\n")
      predicted = "".join(transcribe_wav(fn)).translate(str.maketrans("", "", string.punctuation)).lower()
      transcript = c["transcript"].translate(str.maketrans("", "", string.punctuation))
      sys.stdout.writelines(list(diff.compare([predicted + "\n"], [transcript + "\n"])))
      print(f"\nword error rate: {word_error_rate([predicted], [transcript])[0]:.4f}")
  elif args.file:
    print(transcribe_wav(args.file))
  else:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens[lang], enc._special_tokens[task]]
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
        log_spec = prep_audio(total, 0)
        encoded_audio = model.encoder(Tensor(log_spec)).realize()
      out = model.decoder(Tensor([lst]), encoded_audio).realize()
      idx = out[0,-1].numpy().argmax()
      lst.append(idx)
      dec = enc.decode(lst)
      print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
      if dec.endswith("<|endoftext|>"):
        #total = total[:, 320*(len(lst)-1):]
        lst = [enc._special_tokens["<|startoftranscript|>"]]
