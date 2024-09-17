from typing import Dict, Optional, List
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
from extra.models.llama import Transformer, TransformerBlock, fix_bf16, sample
import argparse, sys, json
from pathlib import Path
from functools import partial

from tinygrad import nn, Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context, getenv
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.shape.symbolic import Variable

# https://github.com/mistralai/mistral-inference/blob/4304e4f991a70050444cf4441ef0affb3caa8925/src/mistral_inference/rope.py#L26
def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, dtype=dtypes.half) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))

  h, w = Tensor.arange(height), Tensor.arange(width)

  freqs_h = h.unsqueeze(-1) @ freqs[::2].unsqueeze(0)
  freqs_w = w.unsqueeze(-1) @ freqs[1::2].unsqueeze(0)
  freqs_2d = Tensor.cat(freqs_h[:, None, :].repeat(1, width, 1), freqs_w[None, :, :].repeat(height, 1, 1),dim=-1)
  return Tensor.stack(freqs_2d.cos().cast(dtype), freqs_2d.sin().cast(dtype), dim=-1).reshape(height, width, 1, dim//2, 2)

def meshgrid(*args:Tensor): return tuple(t[*(None,)*i,:,*(None,)*(len(args)-1-i)].expand(*[t0.size(0) for t0 in args]) for i, t in enumerate(args))

# https://github.com/mistralai/mistral-inference/blob/4304e4f991a70050444cf4441ef0affb3caa8925/src/mistral_inference/vision_encoder.py#L12
def position_meshgrid(patched: List[Tensor]) -> Tensor:
  return Tensor.cat(*[Tensor.stack(*meshgrid(Tensor.arange(p.size(-2)), Tensor.arange(p.size(-1))), dim=-1).reshape(-1, 2) for p in patched])

class VisionTransformer:
  def __init__(self, hidden_size, num_channels, image_size, patch_size, rope_theta, intermediate_size, num_hidden_layers, num_attention_heads):
    self.ln_pre = nn.RMSNorm(hidden_size, eps=1e-5)
    self.patch_conv = nn.Conv2d(num_channels, hidden_size, patch_size, stride=patch_size, bias=False)
    self.transformer = {'layers':[TransformerBlock(hidden_size, intermediate_size, num_attention_heads, num_attention_heads,
                                                   1e-5, 256, head_dim=hidden_size // num_attention_heads) for _ in range(num_hidden_layers)]}

    self.freqs_cis = precompute_freqs_cis_2d(hidden_size // num_attention_heads, image_size // patch_size, image_size // patch_size, rope_theta)

  # TODO: Accept more than one image
  def __call__(self, img):
    h = self.patch_conv(img)

    pos = position_meshgrid([h])
    freqs_cis = self.freqs_cis[pos[:,0],pos[:,1]].unsqueeze(0)

    h = self.ln_pre(h.flatten(2).permute(0, 2, 1))
    for layer in self.transformer['layers']: h = layer(h, 0, freqs_cis, mask=None)
    return h.squeeze(0)

class VisionLanguageAdapter:
  def __init__(self, dim, hidden_size): self.w_in, self.w_out = nn.Linear(hidden_size, dim), nn.Linear(dim, dim)
  def __call__(self, x): return self.w_out(self.w_in(x).gelu())

def merge_multimodal_embeddings(input_ids:Tensor, text_embeds:Tensor, image_embeds:Tensor, image_id: int):
  image_locations = input_ids == image_id
  idxs = (image_locations.cumsum(axis=1)-1).relu()
  return image_locations.unsqueeze(-1).expand(text_embeds.shape).where(image_embeds[idxs], text_embeds)

class MultiModalTransformer(Transformer):
  def __init__(self, vision_encoder:Dict, jit=False, **kwargs):
    super().__init__(**{k:v for k,v in kwargs.items() if k in ['dim', 'hidden_dim', 'n_heads', 'n_layers', 'norm_eps',
                                                               'vocab_size', 'n_kv_heads', 'rope_theta', 'head_dim']})
    self.vision_encoder = VisionTransformer(**{k:v for k,v in vision_encoder.items() if k != 'image_token_id'})
    self.vision_language_adapter = VisionLanguageAdapter(kwargs['dim'], vision_encoder['hidden_size'])
    self.image_token_id = vision_encoder['image_token_id']
    self.transformer_jit = TinyJit(self.transformer) if jit else None

  def transformer(self, embds:Tensor, temperature:float, top_k:int, top_p:float, alpha_f:float, alpha_p:float, start_pos:Variable):
    _bsz, seqlen, _ = embds.shape
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos+seqlen),None,None,None))
    mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), dtype=embds.dtype, device=embds.device).triu(start_pos+1).realize() if seqlen > 1 else None
    for layer in self.layers: embds = layer(embds, start_pos, freqs_cis, mask)
    logits = self.output(self.norm(embds)).float()[:, -1, :]

    return sample(logits.flatten(), temperature, top_k, top_p, alpha_f, alpha_p).realize()

  def __call__(self, tokens:Tensor, image:Optional[Tensor], start_pos:Variable, temperature:float=0.0, top_k:int=0, top_p:float=0.8, alpha_f:float=0.0, alpha_p:float=0.0):
    if tokens.shape[0:2] == (1,1) and self.transformer_jit is not None:
      transformer_fxn = partial(self.transformer_jit, start_pos=Variable("start_pos", 0, self.max_context).bind(start_pos))
    else: transformer_fxn = partial(self.transformer, start_pos=start_pos)
    # if there is no image, this is trivial
    if image is None: return transformer_fxn(self.tok_embeddings(tokens), temperature, top_k, top_p, alpha_f, alpha_p)
    # otherwise we need to compute both the text embeddings and the vision embeddings
    vision_embeddings = self.vision_language_adapter(self.vision_encoder(image))
    text_embeddings = self.tok_embeddings(tokens)
    # and merge them together
    merged = merge_multimodal_embeddings(tokens, text_embeddings, vision_embeddings, self.image_token_id)#.realize()
    return transformer_fxn(merged, temperature, top_k, top_p, alpha_f, alpha_p)

def build(model_dir, device=None, jit=False):
  with open(model_dir / "params.json") as f: model_params = json.load(f)
  weights = safe_load(model_dir / "consolidated.safetensors")
  for v in weights.values(): v.to_(device[0] if isinstance(device, tuple) else device)
  model = MultiModalTransformer(**model_params, jit=jit)

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
  parser.add_argument("--jit", action=argparse.BooleanOptionalAction, help="enable the (imperfect) JIT")
  parser.add_argument("--prompt", type=str, default="Describe this image")
  parser.add_argument("--image", type=str, default=None, help="Image input")
  parser.add_argument("--temperature", type=float, default=0.35)

  tokenizer = MistralTokenizer.v3(is_tekken=True, is_mm=True)
  tokenizer.instruct_tokenizer.tokenizer._special_token_policy = SpecialTokenPolicy.KEEP
  args = parser.parse_args()
  if args.device is None: device = Device.DEFAULT
  else: device = tuple(args.device.split(',')) if ',' in args.device else args.device

  model = build(args.model, device=device, jit=args.jit)

  prompt_content = [TextChunk(text=args.prompt)]
  if args.image is not None: prompt_content.append(ImageURLChunk(image_url=args.image))
  tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=prompt_content)], model='pixtral'))
  outputted = tokenized.text

  toks, image, start_pos = tokenized.tokens, Tensor(tokenized.images[0], device=device, dtype=dtypes.half).unsqueeze(0) if args.image is not None else None, 0
  tok_tensor: Optional[Tensor] = None

  for i in range(args.count):
    next_tok = Tensor([toks[start_pos:]], device=device) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1).to(device)
    tok_tensor = model(next_tok, image if i == 0 else None, start_pos, temperature=args.temperature)
    tok = tok_tensor.item()
    if tok == tokenizer.instruct_tokenizer.tokenizer.eos_id: break

    start_pos = len(toks)

    toks.append(tok)

    cur = tokenizer.decode(toks)
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur
