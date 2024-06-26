# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
import tempfile
from pathlib import Path
import gzip, argparse, math, re
from functools import lru_cache
from collections import namedtuple

from PIL import Image
import numpy as np
from tinygrad import Device, GlobalCounters, dtypes, Tensor, TinyJit
from tinygrad.helpers import Timing, Context, getenv, fetch, colored, tqdm
from tinygrad.nn import Conv2d, Linear, GroupNorm, LayerNorm, Embedding
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict

class AttnBlock:
  def __init__(self, in_channels):
    self.norm = GroupNorm(32, in_channels)
    self.q = Conv2d(in_channels, in_channels, 1)
    self.k = Conv2d(in_channels, in_channels, 1)
    self.v = Conv2d(in_channels, in_channels, 1)
    self.proj_out = Conv2d(in_channels, in_channels, 1)

  # copied from AttnBlock in ldm repo
  def __call__(self, x):
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
    h_ = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
    return x + self.proj_out(h_)

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = GroupNorm(32, in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = GroupNorm(32, out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h

class Mid:
  def __init__(self, block_in):
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1 = AttnBlock(block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

  def __call__(self, x):
    return x.sequential([self.block_1, self.attn_1, self.block_2])

class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512,3, padding=1)
    self.mid = Mid(512)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": Conv2d(s[0], s[0], 3, padding=1)}
    self.up = arr

    self.norm_out = GroupNorm(32, 128)
    self.conv_out = Conv2d(128, 3, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)
    x = self.mid(x)

    for l in self.up[::-1]:
      print("decode", x.shape)
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
        x = l['upsample']['conv'](x)
      x.realize()

    return self.conv_out(self.norm_out(x).swish())

class Encoder:
  def __init__(self):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3,128,3, padding=1)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[0], s[1]),
         ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": Conv2d(s[1], s[1], 3, stride=2, padding=(0,1,0,1))}
    self.down = arr

    self.mid = Mid(512)
    self.norm_out = GroupNorm(32, 512)
    self.conv_out = Conv2d(512, 8, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)

    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)

    x = self.mid(x)
    return self.conv_out(self.norm_out(x).swish())

class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, 1)
    self.post_quant_conv = Conv2d(4, 4, 1)

  def __call__(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)

# not to be confused with ResnetBlock
class ResBlock:
  def __init__(self, channels, emb_channels, out_channels):
    self.in_layers = [
      GroupNorm(32, channels),
      Tensor.silu,
      Conv2d(channels, out_channels, 3, padding=1)
    ]
    self.emb_layers = [
      Tensor.silu,
      Linear(emb_channels, out_channels)
    ]
    self.out_layers = [
      GroupNorm(32, out_channels),
      Tensor.silu,
      lambda x: x,  # needed for weights loading code to work
      Conv2d(out_channels, out_channels, 3, padding=1)
    ]
    self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x

  def __call__(self, x, emb):
    h = x.sequential(self.in_layers)
    emb_out = emb.sequential(self.emb_layers)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = h.sequential(self.out_layers)
    ret = self.skip_connection(x) + h
    return ret

class CrossAttention:
  def __init__(self, query_dim, context_dim, n_heads, d_head):
    self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
    self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
    self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = [Linear(n_heads*d_head, query_dim)]

  def __call__(self, x, context=None):
    context = x if context is None else context
    q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
    q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
    attention = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2)
    h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
    return h_.sequential(self.to_out)

class GEGLU:
  def __init__(self, dim_in, dim_out):
    self.proj = Linear(dim_in, dim_out * 2)
    self.dim_out = dim_out

  def __call__(self, x):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * gate.gelu()

class FeedForward:
  def __init__(self, dim, mult=4):
    self.net = [
      GEGLU(dim, dim*mult),
      lambda x: x,  # needed for weights loading code to work
      Linear(dim*mult, dim)
    ]

  def __call__(self, x):
    return x.sequential(self.net)

class BasicTransformerBlock:
  def __init__(self, dim, context_dim, n_heads, d_head):
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff = FeedForward(dim)
    self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    self.norm3 = LayerNorm(dim)

  def __call__(self, x, context=None):
    x = self.attn1(self.norm1(x)) + x
    x = self.attn2(self.norm2(x), context=context) + x
    x = self.ff(self.norm3(x)) + x
    return x

class SpatialTransformer:
  def __init__(self, channels, context_dim, n_heads, d_head):
    self.norm = GroupNorm(32, channels)
    assert channels == n_heads * d_head
    self.proj_in = Conv2d(channels, n_heads * d_head, 1)
    self.transformer_blocks = [BasicTransformerBlock(channels, context_dim, n_heads, d_head)]
    self.proj_out = Conv2d(n_heads * d_head, channels, 1)

  def __call__(self, x, context=None):
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    x = x.reshape(b, c, h*w).permute(0,2,1)
    for block in self.transformer_blocks:
      x = block(x, context=context)
    x = x.permute(0,2,1).reshape(b, c, h, w)
    ret = self.proj_out(x) + x_in
    return ret

class Downsample:
  def __init__(self, channels):
    self.op = Conv2d(channels, channels, 3, stride=2, padding=1)

  def __call__(self, x):
    return self.op(x)

class Upsample:
  def __init__(self, channels):
    self.conv = Conv2d(channels, channels, 3, padding=1)

  def __call__(self, x):
    bs,c,py,px = x.shape
    x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
    return self.conv(x)

def timestep_embedding(timesteps, dim, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp()
  args = timesteps * freqs
  return Tensor.cat(args.cos(), args.sin()).reshape(1, -1)

class UNetModel:
  def __init__(self):
    self.time_embed = [
      Linear(320, 1280),
      Tensor.silu,
      Linear(1280, 1280),
    ]
    self.input_blocks = [
      [Conv2d(4, 320, kernel_size=3, padding=1)],
      [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [Downsample(320)],
      [ResBlock(320, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [ResBlock(640, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [Downsample(640)],
      [ResBlock(640, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(1280, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [Downsample(1280)],
      [ResBlock(1280, 1280, 1280)],
      [ResBlock(1280, 1280, 1280)]
    ]
    self.middle_block = [
      ResBlock(1280, 1280, 1280),
      SpatialTransformer(1280, 768, 8, 160),
      ResBlock(1280, 1280, 1280)
    ]
    self.output_blocks = [
      [ResBlock(2560, 1280, 1280)],
      [ResBlock(2560, 1280, 1280)],
      [ResBlock(2560, 1280, 1280), Upsample(1280)],
      [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(1920, 1280, 1280), SpatialTransformer(1280, 768, 8, 160), Upsample(1280)],
      [ResBlock(1920, 1280, 640), SpatialTransformer(640, 768, 8, 80)],  # 6
      [ResBlock(1280, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [ResBlock(960, 1280, 640), SpatialTransformer(640, 768, 8, 80), Upsample(640)],
      [ResBlock(960, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
    ]
    self.out = [
      GroupNorm(32, 320),
      Tensor.silu,
      Conv2d(320, 4, kernel_size=3, padding=1)
    ]

  def __call__(self, x, timesteps=None, context=None):
    # TODO: real time embedding
    t_emb = timestep_embedding(timesteps, 320)
    emb = t_emb.sequential(self.time_embed)

    def run(x, bb):
      if isinstance(bb, ResBlock): x = bb(x, emb)
      elif isinstance(bb, SpatialTransformer): x = bb(x, context)
      else: x = bb(x)
      return x

    saved_inputs = []
    for i,b in enumerate(self.input_blocks):
      #print("input block", i)
      for bb in b:
        x = run(x, bb)
      saved_inputs.append(x)
    for bb in self.middle_block:
      x = run(x, bb)
    for i,b in enumerate(self.output_blocks):
      #print("output block", i)
      x = x.cat(saved_inputs.pop(), dim=1)
      for bb in b:
        x = run(x, bb)
    return x.sequential(self.out)

class CLIPMLP:
  def __init__(self):
    self.fc1 = Linear(768, 3072)
    self.fc2 = Linear(3072, 768)

  def __call__(self, hidden_states):
    hidden_states = self.fc1(hidden_states)
    hidden_states = hidden_states.quick_gelu()
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class CLIPAttention:
  def __init__(self):
    self.embed_dim = 768
    self.num_heads = 12
    self.head_dim = self.embed_dim // self.num_heads
    self.k_proj = Linear(self.embed_dim, self.embed_dim)
    self.v_proj = Linear(self.embed_dim, self.embed_dim)
    self.q_proj = Linear(self.embed_dim, self.embed_dim)
    self.out_proj = Linear(self.embed_dim, self.embed_dim)

  def __call__(self, hidden_states, causal_attention_mask):
    bsz, tgt_len, embed_dim = hidden_states.shape
    q,k,v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
    q,k,v = [x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (q,k,v)]
    attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=causal_attention_mask)
    return self.out_proj(attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim))

class CLIPEncoderLayer:
  def __init__(self):
    self.self_attn = CLIPAttention()
    self.layer_norm1 = LayerNorm(768)
    self.mlp = CLIPMLP()
    self.layer_norm2 = LayerNorm(768)

  def __call__(self, hidden_states, causal_attention_mask):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states, causal_attention_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states

class CLIPEncoder:
  def __init__(self):
    self.layers = [CLIPEncoderLayer() for i in range(12)]

  def __call__(self, hidden_states, causal_attention_mask):
    for l in self.layers:
      hidden_states = l(hidden_states, causal_attention_mask)
    return hidden_states

class CLIPTextEmbeddings:
  def __init__(self):
    self.token_embedding = Embedding(49408, 768)
    self.position_embedding = Embedding(77, 768)

  def __call__(self, input_ids, position_ids):
    return self.token_embedding(input_ids) + self.position_embedding(position_ids)

class CLIPTextTransformer:
  def __init__(self):
    self.embeddings = CLIPTextEmbeddings()
    self.encoder = CLIPEncoder()
    self.final_layer_norm = LayerNorm(768)

  def __call__(self, input_ids):
    x = self.embeddings(input_ids, Tensor.arange(input_ids.shape[1]).reshape(1, -1))
    x = self.encoder(x, Tensor.full((1, 1, 77, 77), float("-inf")).triu(1))
    return self.final_layer_norm(x)

# Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
@lru_cache()
def default_bpe(): return fetch("https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz", "bpe_simple_vocab_16e6.txt.gz")

def get_pairs(word):
  """Return set of symbol pairs in a word.
  Word is represented as tuple of symbols (symbols being variable-length strings).
  """
  return set(zip(word, word[1:]))

def whitespace_clean(text):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def bytes_to_unicode():
  """
  Returns list of utf-8 byte and a corresponding list of unicode strings.
  The reversible bpe codes work on unicode strings.
  This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
  When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
  This is a significant percentage of your normal, say, 32K bpe vocab.
  To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
  And avoids mapping to whitespace/control characters the bpe code barfs on.
  """
  bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8+n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))

class ClipTokenizer:
  def __init__(self, bpe_path: str = default_bpe()):
    self.byte_encoder = bytes_to_unicode()
    merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
    merges = merges[1:49152-256-2+1]
    merges = [tuple(merge.split()) for merge in merges]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v+'</w>' for v in vocab]
    for merge in merges:
      vocab.append(''.join(merge))
    vocab.extend(['<|startoftext|>', '<|endoftext|>'])
    self.encoder = dict(zip(vocab, range(len(vocab))))
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
    self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""", re.IGNORECASE)

  def bpe(self, token):
    if token in self.cache:
      return self.cache[token]
    word = tuple(token[:-1]) + ( token[-1] + '</w>',)
    pairs = get_pairs(word)

    if not pairs:
      return token+'</w>'

    while True:
      bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
      if bigram not in self.bpe_ranks:
        break
      first, second = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except Exception:
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word)-1 and word[i+1] == second:
          new_word.append(first+second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      pairs = get_pairs(word)
    word = ' '.join(word)
    self.cache[token] = word
    return word

  def encode(self, text):
    bpe_tokens = []
    text = whitespace_clean(text.strip()).lower()
    for token in re.findall(self.pat, text):
      token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
      bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
    # Truncation, keeping two slots for start and end tokens.
    if len(bpe_tokens) > 75:
      bpe_tokens = bpe_tokens[:75]
    return [49406] + bpe_tokens + [49407] * (77 - len(bpe_tokens) - 1)

def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
  betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
  alphas = 1.0 - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  return Tensor(alphas_cumprod)

class StableDiffusion:
  def __init__(self):
    self.alphas_cumprod = get_alphas_cumprod()
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel())
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = CLIPTextTransformer()))

  def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
    temperature = 1
    sigma_t = 0
    sqrt_one_minus_at = (1-a_t).sqrt()
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0

  def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
    # put into diffuser
    latents = self.model.diffusion_model(latent.expand(2, *latent.shape[1:]), timestep, unconditional_context.cat(context, dim=0))
    unconditional_latent, latent = latents[0:1], latents[1:2]

    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  def decode(self, x):
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,512,512).permute(1,2,0).clip(0,1)*255
    return x.cast(dtypes.uint8) if Device.DEFAULT != "WEBGPU" else x

  def __call__(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
    e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
    x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    return x_prev.realize()

# ** ldm.models.autoencoder.AutoencoderKL (done!)
# 3x512x512 <--> 4x64x64 (16384)
# decode torch.Size([1, 4, 64, 64]) torch.Size([1, 3, 512, 512])
# section 4.3 of paper
# first_stage_model.encoder, first_stage_model.decoder

# ** ldm.modules.diffusionmodules.openaimodel.UNetModel
# this is what runs each time to sample. is this the LDM?
# input:  4x64x64
# output: 4x64x64
# model.diffusion_model
# it has attention?

# ** ldm.modules.encoders.modules.FrozenCLIPEmbedder
# cond_stage_model.transformer.text_model

if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=10, help="Number of steps in diffusion")
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to render")
  parser.add_argument('--out', type=str, default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--noshow', action='store_true', help="Don't show the image")
  parser.add_argument('--fp16', action='store_true', help="Cast the weights to float16")
  parser.add_argument('--timing', action='store_true', help="Print timing per step")
  parser.add_argument('--seed', type=int, help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=7.5, help="Prompt strength")
  args = parser.parse_args()

  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

  if args.fp16:
    for l in get_state_dict(model).values():
      l.replace(l.cast(dtypes.float16).realize())

  # run through CLIP to get context
  tokenizer = ClipTokenizer()
  prompt = Tensor([tokenizer.encode(args.prompt)])
  context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got CLIP context", context.shape)

  prompt = Tensor([tokenizer.encode("")])
  unconditional_context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got unconditional CLIP context", unconditional_context.shape)

  # done with clip model
  del model.cond_stage_model

  timesteps = list(range(1, 1000, 1000//args.steps))
  print(f"running for {timesteps} timesteps")
  alphas = model.alphas_cumprod[Tensor(timesteps)]
  alphas_prev = Tensor([1.0]).cat(alphas[:-1])

  # start with random noise
  if args.seed is not None: Tensor.manual_seed(args.seed)
  latent = Tensor.randn(1,4,64,64)

  @TinyJit
  def run(model, *x): return model(*x).realize()

  # this is diffusion
  with Context(BEAM=getenv("LATEBEAM")):
    for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
      GlobalCounters.reset()
      t.set_description("%3d %3d" % (index, timestep))
      with Timing("step in ", enabled=args.timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
        tid = Tensor([index])
        latent = run(model, unconditional_context, context, latent, Tensor([timestep]), alphas[tid], alphas_prev[tid], Tensor([args.guidance]))
        if args.timing: Device[Device.DEFAULT].synchronize()
    del run

  # upsample latent space to image with autoencoder
  x = model.decode(latent)
  print(x.shape)

  # save image
  im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
  print(f"saving {args.out}")
  im.save(args.out)
  # Open image.
  if not args.noshow: im.show()

  # validation!
  if args.prompt == default_prompt and args.steps == 10 and args.seed == 0 and args.guidance == 7.5:
    ref_image = Tensor(np.array(Image.open(Path(__file__).parent / "stable_diffusion_seed0.png")))
    distance = (((x - ref_image).cast(dtypes.float) / ref_image.max())**2).mean().item()
    assert distance < 3e-4, colored(f"validation failed with {distance=}", "red")
    print(colored(f"output validated with {distance=}", "green"))
