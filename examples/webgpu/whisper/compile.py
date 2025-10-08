from pathlib import Path
import tempfile
from typing import Optional, Union, Literal, List
import collections

import numpy as np

from examples.webgpu.whisper.audio_helpers import stft, hann_window, make_stft_basis_buffers, mel
from tinygrad import Tensor, TinyJit, Device, Variable, nn
import json

from tinygrad.nn.state import safe_save, safe_load, torch_load, load_state_dict, get_state_dict
from tinygrad.helpers import getenv, fetch

from extra.export_model import export_model
from examples.whisper import MODEL_URLS, get_encoding
from examples.whisper import RATE, SAMPLES_PER_SEGMENT, N_FFT, HOP_LENGTH, N_MELS
import math

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
      k, v = self.key(xa), self.value(xa)
      # if xa is not None:
      #   k, v = self.key(xa), self.value(xa)
      #   if not hasattr(self, 'cache_k'):
      #     self.cache_k, self.cache_v = k, v
      #   else:
      #     self.cache_k.assign(k).realize()
      #     self.cache_v.assign(v).realize()
      # else:
      #   k, v = self.cache_k, self.cache_v
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

  if False:
    def forward(self, x:Tensor, pos:Union[Variable, Literal[0]], encoded_audio:Tensor):
      seqlen = x.shape[-1]
      x = self.token_embedding(x) + self.positional_embedding.shrink(((pos, pos+seqlen), None, None))
      for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=pos)
      return self.output_tok(x)
  else:
    def forward(self, x:Tensor, encoded_audio:Tensor, ctx):
      # seqlen = x.shape[-1]
      bs, seqlen = x.shape[0], 1
      encoded_audio = encoded_audio.repeat(bs, 1, 1)
      x = self.token_embedding(x)
      x += self.positional_embedding.shrink(((ctx, ctx+seqlen), None, None))
      for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=ctx)
      # NOTE(irwin): wrong output size w/o contiguous. TODO: check on latest tinygrad
      # logits = self.output_tok(x)[:, ctx-1].contiguous()
      # logits = self.output_tok(x)[:, ctx].contiguous()
      logits = self.output_tok(x).shrink((None, (ctx, ctx+seqlen), None))
      # logits = self.output_tok(x)
      # return logits.log_softmax(axis=-1).reshape(-1, 1)
      # return logits.softmax(axis=-1).sort(descending=True)[1].reshape(-1, 1)[0, :]
      # return logits.softmax(axis=-1).argmax()
      # print(logits.shape)
      return logits

  def output_tok(self, x):
    return (self.ln(x) @ self.token_embedding.weight.T).realize()

class Whisper:
  def __init__(self, dims, batch_size=1):
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)
    self.is_multilingual = dims["n_vocab"] == 51865
    self.batch_size = batch_size

def init_whisper(model_name="tiny.en", batch_size=1):
  assert MODEL_URLS[model_name] is not None

  filename = fetch(MODEL_URLS[model_name])
  state = torch_load(filename)
  model = Whisper(state['dims'], batch_size)
  load_state_dict(model, state['model_state_dict'], strict=False)
  enc = get_encoding("multilingual" if model.is_multilingual else "gpt2")
  return model, enc

# NOTE(irwin): change to True to export weights in float16.
# IMPORTANT(irwin): unfortunately this doesn't switch all computations to half precision yet
FLOAT16 = False

if __name__ == '__main__':
  def tofull(sd):
    return {k: v.float() for k,v in sd.items()}

  def tohalf(sd):
    return {k: v.half() for k,v in sd.items()}

  change_sd = tohalf if FLOAT16 else tofull

  def todevice(sd, device):
    return {k: v.replace(v.to(device=device).realize()) for k,v in sd.items()}

  def reload(model, change_sd=None):
    if change_sd is None:
      change_sd = lambda x: x
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.close()
      safe_save(change_sd(get_state_dict(model)), f.name)
      load_state_dict(model, safe_load(f.name))

  model, enc = init_whisper("tiny.en")

  dirname = Path(__file__).parent
  # NOTE(irwin): force export as f32 as it's a little easier to validate
  # exporting a model that's loaded from safetensors doesn't work without loading in from safetensors first
  # loading the state dict from a safetensor file changes the generated kernels
  safe_save(tofull(get_state_dict(model)), (dirname / "net.safetensors").as_posix())
  Device.DEFAULT = "WEBGPU"
  todevice(get_state_dict(model), "WEBGPU")
  load_state_dict(model, safe_load(str(dirname / "net.safetensors")))

  def export_audio_prep():
    class AudioPrep:
      def __init__(self, n_fft:int, stride:int, pad:tuple[int, int], window="hann", pad_mode="reflect"):
        assert window == "hann", "other window types not implemented yet"
        self.n_fft = n_fft
        self.stride = stride
        self.pad = pad
        self.pad_mode = pad_mode
        self.forward_basis_buffers = make_stft_basis_buffers(n_fft, hann_window(n_fft))
        self.mel = mel(sr=RATE, n_fft=self.n_fft, n_mels=N_MELS)

      def stft_full(self, x:Tensor) -> Tensor:
        res = stft(x, self.forward_basis_buffers, self.n_fft, self.stride, self.pad, self.pad_mode)
        return res

      def __call__(self, waveforms):
        return self.forward(waveforms)

      def forward(self, waveforms):
        spec = self.stft_full(waveforms.reshape(-1, waveforms.shape[-1]))
        magnitudes = (spec[..., :-1] ** 2)
        mel_spec = self.mel @ magnitudes

        def log10(x:Tensor):
          return x.log2() * (math.log(2) / math.log(10))

        log_spec = log10(mel_spec.clip(1e-10, None))
        log_spec = log_spec.maximum(log_spec.max((1,2), keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    prep_audio = AudioPrep(N_FFT, stride=HOP_LENGTH, pad=(200, 200))
    reload(prep_audio, change_sd=change_sd)

    prg, inp_sizes, out_sizes, state = export_model(prep_audio, Device.DEFAULT.lower(), Tensor.randn(1, SAMPLES_PER_SEGMENT), model_name="mel")
    (dirname / 'mel.js').write_text(prg)
    safe_save(state, (dirname / 'mel.safetensors'))
    return prg, inp_sizes, out_sizes, state

  def export_encoder():
    reload(model.encoder, change_sd=change_sd)
    prg, inp_sizes, out_sizes, state = export_model(model.encoder, Device.DEFAULT.lower(), Tensor.randn(1,80,3000), model_name="encoder")
    (dirname / 'encoder.js').write_text(prg)
    safe_save(state, (dirname / 'encoder.safetensors'))
    return prg, inp_sizes, out_sizes, state

  def export_decoder_2():
    reload(model.decoder, change_sd=change_sd)
    BS = 1
    embedding_dims = model.decoder.positional_embedding.shape[1]
    # x = Tensor.randint(model.decoder.max_tokens_to_sample*2, low=0, high=50256).to("WEBGPU").reshape(1, -1)
    x = Tensor.randint(BS, low=0, high=50256).to("WEBGPU").reshape(BS, -1)
    prg, inp_sizes, out_sizes, state = export_model(
      model.decoder,
      Device.DEFAULT.lower(),
      x,
      Tensor.rand(1, 1500, embedding_dims),
      Variable("ctx", 0, model.decoder.max_tokens_to_sample*2-1).bind(1),
      model_name="decoder"
    )
    # print(out_sizes)
    (dirname / 'decoder.js').write_text(prg)
    safe_save(state, (dirname / 'decoder.safetensors'))
    return prg, inp_sizes, out_sizes, state

  def export_vocab():
    d = enc.decode_batch(np.arange(enc.n_vocab).reshape(-1, 1))
    (dirname / "vocab.json").write_text(json.dumps(d), encoding="utf8")

  export_audio_prep()
  export_encoder()
  export_decoder_2()
  export_vocab()
