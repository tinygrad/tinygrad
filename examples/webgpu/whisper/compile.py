import os
# os.environ["DEFAULT_FLOAT"] = "FLOAT32"
from pathlib import Path

import tinygrad
import numpy as np
from tinygrad import Tensor, Device, Context, dtypes, Variable, TinyJit
import json

Device.DEFAULT = "WEBGPU"

from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_save, safe_load, load_state_dict, get_state_dict
from extra.export_model import export_model
from examples.whisper import MODEL_URLS, get_encoding, init_whisper, MultiHeadAttention, ResidualAttentionBlock, TextDecoder, AudioEncoder, Whisper


if __name__ == '__main__':
    def tofull(sd):
        return {k: v.float() for k,v in sd.items()}

    model, enc = init_whisper()

    dirname = Path(__file__).parent
    # NOTE(irwin): force export as f32 as it's a little easier to validate
    # exporting a model that's loaded from safetensors doesn't work without loading in from safetensors first
    # loading the state dict from a safetensor file changes the generated kernels
    safe_save(tofull(get_state_dict(model)), (dirname / "net.safetensors").as_posix())
    load_state_dict(model, safe_load(str(dirname / "net.safetensors")))

    def export_encoder():
        prg, inp_sizes, out_sizes, state = export_model(model.encoder, Device.DEFAULT.lower(), Tensor.randn(1,80,3000), model_name="encoder")
        (dirname / 'encoder.js').write_text(prg)
        safe_save(state, (dirname / 'encoder.safetensors'))
        return prg, inp_sizes, out_sizes, state


    def export_decoder_2():
        def forward(self, x:Tensor, encoded_audio:Tensor, ctx):
            seqlen = x.shape[-1]
            x = self.token_embedding(x)
            x += self.positional_embedding.shrink(((0, seqlen), None, None))
            for block in self.blocks: x = block(x, xa=encoded_audio, mask=self.mask, len=0)
            return self.output_tok(x)[:, ctx-1].argmax(axis=-1).reshape(-1, 1)

        model.decoder.forward = forward.__get__(model.decoder, TextDecoder)

        # forward_jitted = TinyJit(forward)

        # var = Variable("x", 1, model.decoder.max_tokens_to_sample-1)
        # x = Tensor([[50257, 50362]]).pad((None, (0, model.decoder.max_tokens_to_sample-2))).realize()
        x = Tensor.randint(model.decoder.max_tokens_to_sample*2, low=0, high=50256).to("WEBGPU").reshape(1, -1)
        prg, inp_sizes, out_sizes, state = export_model(model.decoder, Device.DEFAULT.lower(),
            x, Tensor.rand(1, 1500, 384), Variable("ctx", 1, model.decoder.max_tokens_to_sample*2-1).bind(2), model_name="decoder")

        (dirname / 'decoder.js').write_text(prg)
        safe_save(state, (dirname / 'decoder.safetensors'))
        return prg, inp_sizes, out_sizes, state

    def export_vocab():
        d = enc.decode_batch(np.arange(enc.n_vocab).reshape(-1, 1))
        (dirname / "vocab.json").write_text(json.dumps(d), encoding="utf8")

    export_encoder()
    export_decoder_2()
    export_vocab()