# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md

from pathlib import Path
import gzip, argparse, math, re
from functools import lru_cache
from collections import namedtuple

import numpy as np
from tqdm import tqdm

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear, GroupNorm, LayerNorm
from extra.utils import download_file
from tinygrad.state import torch_load, load_state_dict

# TODO: refactor AttnBlock, CrossAttention, CLIPAttention to share code

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
    q = q.reshape(b,c,h*w)
    q = q.permute(0,2,1)   # b,hw,c
    k = k.reshape(b,c,h*w) # b,c,hw
    w_ = q @ k
    w_ = w_ * (c**(-0.5))
    w_ = w_.softmax()

    # attend to values
    v = v.reshape(b,c,h*w)
    w_ = w_.permute(0,2,1)
    h_ = v @ w_
    h_ = h_.reshape(b,c,h,w)

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
    self.scale = d_head ** -0.5
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = [Linear(n_heads*d_head, query_dim)]

  def __call__(self, x, context=None):
    context = x if context is None else context
    q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
    q = q.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,1,3)  # (bs, num_heads, time, head_size)
    k = k.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,3,1)  # (bs, num_heads, head_size, time)
    v = v.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,1,3)  # (bs, num_heads, time, head_size)

    score = q.dot(k) * self.scale
    weights = score.softmax()                     # (bs, num_heads, time, time)
    attention = weights.dot(v).permute(0,2,1,3)   # (bs, time, num_heads, head_size)

    h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
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
  freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
  args = timesteps * freqs
  embedding = np.concatenate([np.cos(args), np.sin(args)])
  return Tensor(embedding).reshape(1, -1)

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
      x.realize()
    for bb in self.middle_block:
      x = run(x, bb)
    for i,b in enumerate(self.output_blocks):
      #print("output block", i)
      x = x.cat(saved_inputs.pop(), dim=1)
      for bb in b:
        x = run(x, bb)
      x.realize()
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
    self.scale = self.head_dim**-0.5
    self.k_proj = Linear(self.embed_dim, self.embed_dim)
    self.v_proj = Linear(self.embed_dim, self.embed_dim)
    self.q_proj = Linear(self.embed_dim, self.embed_dim)
    self.out_proj = Linear(self.embed_dim, self.embed_dim)

  def _shape(self, tensor, seq_len: int, bsz: int):
    return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

  def __call__(self, hidden_states, causal_attention_mask):
    bsz, tgt_len, embed_dim = hidden_states.shape

    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    src_len = key_states.shape[1]
    value_states = value_states.reshape(*proj_shape)

    attn_weights = query_states @ key_states.permute(0,2,1)

    attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = attn_weights.softmax()

    attn_output = attn_weights @ value_states

    attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.permute(0,2,1,3)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)
    return attn_output

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
    #self.position_ids = Tensor.empty(1, 77)  # what is this?
    self.token_embedding = {"weight": Tensor.empty(49408, 768)}
    self.position_embedding = {"weight": Tensor.empty(77, 768)}

  def __call__(self, input_ids, position_ids):
    # TODO: actually support batches
    inputs = np.zeros((1, len(input_ids), 49408), dtype=np.float32)
    positions = np.zeros((1, len(position_ids), 77), dtype=np.float32)
    for i,x in enumerate(input_ids): inputs[0][i][x] = 1
    for i,x in enumerate(position_ids): positions[0][i][x] = 1
    inputs_embeds = Tensor(inputs, device=self.token_embedding['weight'].device) @ self.token_embedding['weight']
    position_embeddings = Tensor(positions, device=self.position_embedding['weight'].device) @ self.position_embedding['weight']
    return inputs_embeds + position_embeddings

class CLIPTextTransformer:
  def __init__(self):
    self.embeddings = CLIPTextEmbeddings()
    self.encoder = CLIPEncoder()
    self.final_layer_norm = LayerNorm(768)

  def __call__(self, input_ids):
    x = self.embeddings(input_ids, list(range(len(input_ids))))
    causal_attention_mask = np.triu(np.ones((1,1,77,77), dtype=np.float32) * -np.inf, k=1)
    x = self.encoder(x, Tensor(causal_attention_mask, device=x.device))
    return self.final_layer_norm(x)

# Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
@lru_cache()
def default_bpe():
  return Path(__file__).parent.parent / "weights/bpe_simple_vocab_16e6.txt.gz"

def get_pairs(word):
  """Return set of symbol pairs in a word.
  Word is represented as tuple of symbols (symbols being variable-length strings).
  """
  pairs = set()
  prev_char = word[0]
  for char in word[1:]:
    pairs.add((prev_char, char))
    prev_char = char
  return pairs

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
  This is a signficant percentage of your normal, say, 32K bpe vocab.
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
    self.pat = self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""", re.IGNORECASE)

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

class StableDiffusion:
  def __init__(self):
    self.alphas_cumprod = Tensor.empty(1000)
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel())
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = CLIPTextTransformer()))

  # TODO: make __call__ run the model

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

# this is sd-v1-4.ckpt
#FILENAME = "/Users/kafka/fun/mps/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
#FILENAME = "/home/kafka/model.ckpt"
FILENAME = Path(__file__).parent.parent / "weights/sd-v1-4.ckpt"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=5, help="Number of steps in diffusion")
  parser.add_argument('--prompt', type=str, default="a horse sized cat eating a bagel", help="Phrase to render")
  parser.add_argument('--out', type=str, default="/tmp/rendered.png", help="Output filename")
  args = parser.parse_args()

  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  download_file('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', FILENAME)
  load_state_dict(model, torch_load(FILENAME)['state_dict'], strict=False)

  # run through CLIP to get context
  tokenizer = ClipTokenizer()
  prompt = tokenizer.encode(args.prompt)
  context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got CLIP context", context.shape)

  prompt = tokenizer.encode("")
  unconditional_context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got unconditional CLIP context", unconditional_context.shape)

  # done with clip model
  del model.cond_stage_model

  def get_model_output(latent, timesteps):
    # put into diffuser
    unconditional_latent = model.model.diffusion_model(latent, timesteps, unconditional_context).realize()
    latent = model.model.diffusion_model(latent, timesteps, context).realize()

    unconditional_guidance_scale = 7.5
    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  timesteps = list(np.arange(1, 1000, 1000//args.steps))
  print(f"running for {timesteps} timesteps")
  alphas = [model.alphas_cumprod.numpy()[t] for t in timesteps]
  alphas_prev = [1.0] + alphas[:-1]

  def get_x_prev_and_pred_x0(x, e_t, index):
    temperature = 1
    a_t, a_prev = alphas[index], alphas_prev[index]
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1-a_t)
    sqrt_one_minus_at = Tensor([sqrt_one_minus_at]).realize()  # don't constant fold this
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

    # direction pointing to x_t
    dir_xt = math.sqrt(1. - a_prev - sigma_t**2) * e_t
    noise = sigma_t * Tensor.randn(*x.shape) * temperature

    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt #+ noise
    return x_prev, pred_x0

  # start with random noise
  latent = Tensor.randn(1,4,64,64)

  # this is diffusion
  for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
    t.set_description("%3d %3d" % (index, timestep))
    e_t = get_model_output(latent, timestep)
    x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t, index)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    latent = x_prev
    latent.realize()

  # upsample latent space to image with autoencoder
  x = model.first_stage_model.post_quant_conv(1/0.18215 * latent)
  x = model.first_stage_model.decoder(x)

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,512,512).permute(1,2,0)
  dat = (x.detach().numpy().clip(0, 1)*255).astype(np.uint8)
  print(dat.shape)

  # save image
  from PIL import Image
  im = Image.fromarray(dat)
  print(f"saving {args.out}")
  im.save(args.out)
  # Open image.
  im.show()
