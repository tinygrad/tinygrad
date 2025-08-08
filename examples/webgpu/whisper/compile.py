from pathlib import Path
import tempfile

import numpy as np

from examples.webgpu.whisper.audio_helpers import stft, hann_window, make_stft_basis_buffers, mel
from tinygrad import Tensor, Device, Variable
import json

from tinygrad.nn.state import safe_save, safe_load, load_state_dict, get_state_dict
from extra.export_model import export_model
from examples.whisper import init_whisper, TextDecoder
from examples.whisper import RATE, SAMPLES_PER_SEGMENT, N_FFT, HOP_LENGTH, N_MELS
import math

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
    def forward(self, x:Tensor, encoded_audio:Tensor, ctx):
      seqlen = x.shape[-1]
      x = self.token_embedding(x)
      x += self.positional_embedding.shrink(((0, seqlen), None, None))
      for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=0)
      logits = self.output_tok(x)[:, ctx-1]
      # return logits.log_softmax(axis=-1).reshape(-1, 1)
      return logits.softmax(axis=-1).sort(descending=True)[1].reshape(-1, 1)[0, :]
    model.decoder.forward = forward.__get__(model.decoder, TextDecoder)

    embedding_dims = model.decoder.positional_embedding.shape[1]
    x = Tensor.randint(model.decoder.max_tokens_to_sample*2, low=0, high=50256).to("WEBGPU").reshape(1, -1)
    prg, inp_sizes, out_sizes, state = export_model(model.decoder, Device.DEFAULT.lower(),
      x, Tensor.rand(1, 1500, embedding_dims), Variable("ctx", 1, model.decoder.max_tokens_to_sample*2-1).bind(2), model_name="decoder")

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
