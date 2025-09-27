import unittest
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_parameters
from extra.models import clip
from examples.mlperf.initializers import gelu_erf

clip_params = {"dims": 1024, "n_heads": 16, "layers": 24, "return_pooled": False, "ln_penultimate": True, "clip_tokenizer_version": "sd_mlperf_v5_0"}
def get_cond_stage_model(GPUS:list[str]|None=None) -> clip.FrozenOpenClipEmbedder:
  model = clip.FrozenOpenClipEmbedder(**clip_params)
  if GPUS and len(GPUS) > 1:
    for p in get_parameters(model): p.to_(GPUS)
  return model

class TestFrozenOpenClip(unittest.TestCase):
  def test_tokenizer(self):
    prompt = "Beautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than complex.\nComplex is better than complicated."
    tokens = get_cond_stage_model().tokenizer.encode(prompt, pad_with_zeros=True)
    assert tokens == [49406, 1215, 533, 1539, 1126, 8159, 269, 33228, 533, 1539, 1126, 15269, 585, 269, 4129, 533, 1539, 1126, 6324, 269, 6324, 533,
                      1539, 1126, 16621, 269, 49407] + [0]*50

  def test_clip_gelu_init(self):
    clip.gelu = gelu_erf
    for resblock in get_cond_stage_model().model.transformer.resblocks:
      assert resblock.mlp.gelu == gelu_erf

  def test_multigpu_clip_embed(self):
    GPUS = [f"NULL:{i}" for i in range(8)]
    BS = 304
    model = get_cond_stage_model(GPUS)
    tokens = Tensor([0] * 77 * BS, dtype=dtypes.int32, device="NULL").reshape(-1, 77)
    embeds = model.embed_tokens(tokens.shard(GPUS, axis=0))
    assert embeds.shape == (BS, 77, 1024)

  def test_multigpu_clip_embed(self):
    GPUS = [f"NULL:{i}" for i in range(8)]
    BS = 304
    model = get_cond_stage_model(GPUS)
    tokens = Tensor([0] * 77 * BS, dtype=dtypes.int32, device="NULL").reshape(-1, 77)
    embeds = model.embed_tokens(tokens.shard(GPUS, axis=0))
    assert embeds.shape == (BS, 77, 1024)

if __name__=="__main__":
  unittest.main()