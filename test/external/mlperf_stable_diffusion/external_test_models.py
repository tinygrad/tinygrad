import unittest
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.state import get_parameters
from extra.models import clip
from examples.mlperf.initializers import gelu_erf
Device.DEFAULT="NULL"
GPUS = [f"NULL:{i}" for i in range(8)]

clip_params = {"dims": 1024, "n_heads": 16, "layers": 24, "return_pooled": False, "ln_penultimate": True, "clip_tokenizer_version": "sd_mlperf_v5_0"}
def get_cond_stage_model(GPUS:list[str]|None=None) -> clip.FrozenOpenClipEmbedder:
  clip.gelu = gelu_erf
  model = clip.FrozenOpenClipEmbedder(**clip_params)
  if GPUS and len(GPUS) > 1:
    for p in get_parameters(model): p.to_(GPUS)
  return model
def get_tokens(BS:int) -> Tensor: return Tensor([0] * 77 * BS, dtype=dtypes.int32).reshape(-1, 77)

class TestOpenClip(unittest.TestCase):
  def test_tokenizer(self):
    prompt = "Beautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than complex.\nComplex is better than complicated."
    model = get_cond_stage_model()
    tokens = model.tokenizer.encode(prompt, pad_with_zeros=True)
    expected = [49406, 1215, 533, 1539, 1126, 8159, 269, 33228, 533, 1539, 1126, 15269, 585, 269, 4129, 533, 1539, 1126, 6324, 269, 6324, 533,
                1539, 1126, 16621, 269, 49407] + [0]*50
    self.assertEqual(tokens, expected)

  def test_clip_gelu_init(self):
    for resblock in get_cond_stage_model().model.transformer.resblocks:
      self.assertEqual(resblock.mlp.gelu, gelu_erf)

  def test_multigpu_clip_embed(self):
    BS = 304
    model = get_cond_stage_model(GPUS)
    tokens = get_tokens(BS)
    embeds = model.embed_tokens(tokens.shard(GPUS, axis=0)).realize()
    self.assertEqual(embeds.shape, (BS, 77, 1024))
    self.assertEqual(embeds.dtype, dtypes.float32)

  def test_multigpu_clip_score(self):
    BS = 240
    vision_cfg = {'width': 1280, 'layers': 32, 'd_head': 80, 'image_size': 224, 'patch_size': 14}
    text_cfg = {'width': 1024, 'n_heads': 16, 'layers': 24, 'vocab_size': 49408, 'ctx_length': 77}
    clip.gelu = gelu_erf
    clip_encoder = clip.OpenClipEncoder(1024, text_cfg, vision_cfg)
    for p in get_parameters(clip_encoder): p.to_(GPUS)
    tokens = get_tokens(BS)
    imgs = Tensor.zeros(BS,3,224,224).contiguous()
    scores = clip_encoder.get_clip_score(tokens.shard(GPUS, axis=0), imgs.shard(GPUS, axis=0)).realize()
    self.assertEqual(scores.shape, (BS,))
    self.assertEqual(scores.dtype, dtypes.float32)

if __name__=="__main__":
  unittest.main()