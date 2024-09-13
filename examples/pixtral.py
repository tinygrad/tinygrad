from typing import Dict, Optional, Tuple
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, is_tekken
from extra.models.llama import Transformer, TransformerBlock, fix_bf16, sample, complex_mult
import argparse, sys, json
from pathlib import Path
from PIL import Image

from tinygrad import nn, Tensor
from tinygrad.device import Device
from tinygrad.helpers import Context, getenv
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.shape.symbolic import Variable

class VisionEncoder:
  def __init__(self, hidden_size, num_channels, image_size, patch_size, rope_theta, intermediate_size, num_hidden_layers, num_attention_heads):
    self.ln_pre = nn.RMSNorm(hidden_size, eps=1e-5)
    self.patch_conv = nn.Conv2d(num_channels, hidden_size, patch_size, stride=patch_size, bias=False)
    self.transformer = {'layers':[TransformerBlock(image_size, intermediate_size, num_attention_heads, num_attention_heads, 1e-5, 1024) for _ in range(num_hidden_layers)]}

class VisionLanguageAdapter:
  def __init__(self, dim, hidden_size):
    self.w_in = nn.Linear(hidden_size, dim)
    self.w_out = nn.Linear(dim, dim)

  def __call__(self, x):
    return self.w_out(self.w_in(x).gelu())

class VisionTransformer(Transformer):
  def __init__(self, vision_encoder:Dict, **kwargs):
    super().__init__(**{k:v for k,v in kwargs.items() if k in ['dim', 'hidden_dim', 'n_heads', 'n_layers', 'norm_eps', 'vocab_size', 'n_kv_heads', 'rope_theta', 'head_dim', 'jit']})
    self.vision_encoder = VisionEncoder(**{k:v for k,v in vision_encoder.items() if k != 'image_token_id'})
    self.vision_language_adapter = VisionLanguageAdapter(kwargs['dim'], vision_encoder['hidden_size'])
    self.image_token_id = vision_encoder['image_token_id']

  def merge_multimodal_embeddings(self, input_ids:Tensor, text_embeds:Tensor, image_embeds:Tensor, image_id: int):
    text_locations = input_ids != image_id
    image_locations = input_ids == image_id

    seq_len = input_ids.shape[0]

    N_txt = text_locations.sum().item()
    _, D_txt = text_embeds.shape
    N_img, D_img = image_embeds.shape

    assert (D_txt == D_img), f"Text features dim {D_txt} should be equal to image features dim {D_img}"
    assert (seq_len == N_txt + N_img), f"seq_len {seq_len} should be equal to N_txt + N_img {(N_txt, N_img, image_locations.sum().item())}"

    text_embeds[image_locations, :] = image_embeds
    return text_embeds

  def __call__(self, tokens:Tensor, image:Optional[Tensor], start_pos:Variable, temperature:float=0.0, top_k:int=0, top_p:float=0.8, alpha_f:float=0.0, alpha_p:float=0.0):
    # if there is no image, this is trivial
    if image is None: return super().__call__(tokens, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)
    # otherwise we need to compute both the text embeddings and the vision embeddings
    vision_embeddings = self.vision_language_adapter(self.vision_encoder(image))
    text_embeddings = self.tok_embeddings(tokens)
    # and merge them together
    h = self.merge_multimodal_embeddings(tokens, text_embeddings, vision_embeddings, self.image_token_id)
    _bsz, seqlen = tokens.shape
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos+seqlen),None,None,None))
    mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos+1).realize() if seqlen > 1 else None
    for layer in self.layers: h = layer(h, start_pos, freqs_cis, mask)
    logits = self.output(self.norm(h)).float()[:, -1, :]

    return sample(logits.flatten(), temperature, top_k, top_p, alpha_f, alpha_p).realize()

def build(model_dir, device=None):
  with open(model_dir / "params.json") as f: model_params = json.load(f)
  weights = safe_load(model_dir / "consolidated.safetensors")
  for v in weights.values(): v.to_(device[0] if isinstance(device, tuple) else device)
  model = VisionTransformer(**model_params, jit=False)

  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # shard
    if isinstance(device, tuple):
      for k,v in nn.state.get_state_dict(model).items():
        if 'scale' in k: v.shard_(device, axis=None)  # from quantized
        elif '.attention.' in k:
          if getenv("SHARD_KVCACHE") and ('.wq.' in k or '.wk.' in k or '.wv.' in k): v.shard_(device, axis=0)
          else: v.shard_(device, axis=-1)
        elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.' in k: v.shard_(device, axis=-1)
        elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
        elif 'output.weight' in k: v.shard_(device, axis=-1)
        #elif k.endswith('.weight'): v.shard_(device, axis=-1)
        #elif 'norm.' in k: v.shard_(device, axis=-1)
        else: v.shard_(device, axis=None)
        #print(k, v.shape, v.lazydata.axis)

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)

  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Pixtral in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model", type=Path, default=Path("/scratch/weights/pixtral-12b-240910"), help="folder with weights to load")
  parser.add_argument("--device", type=str, default=None, help="device(s) to run on (ie. NV:0,NV:1)")
  parser.add_argument("--count", type=int, default=10, help="number of tokens to generate")

  tokenizer = MistralTokenizer.v3(is_tekken=True, is_mm=True)
  tokenizer.instruct_tokenizer.tokenizer._special_token_policy = SpecialTokenPolicy.KEEP
  args = parser.parse_args()
  if args.device is None: device = Device.DEFAULT
  else: device = tuple(args.device.split(',')) if ',' in args.device else args.device

  model = build(args.model, device=device)

  image = Image.new('RGB', (64, 64))
  tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="Describe this image"),
                                                                                               ImageChunk(image=image)
                                                                                               ])], model='pixtral'))
  outputted = tokenized.text
  print(outputted, end='', flush=True)

  toks, start_pos = tokenized.tokens, 0
  tok_tensor: Optional[Tensor] = None

  for i in range(args.count):
    next_tok = Tensor([toks[start_pos:]], device=device) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1)
    tok_tensor = model(next_tok, None, start_pos, 0)
    tok = tok_tensor.item()

    start_pos = len(toks)

    toks.append(tok)

    cur = tokenizer.decode(toks)
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur
