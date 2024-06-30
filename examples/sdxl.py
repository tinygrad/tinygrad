# This file incorporates code from the following:
# Github Name                    | License | Link
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn import Linear, Conv2d, GroupNorm, LayerNorm, Embedding
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch, trange, colored, THREEFRY
from examples.stable_diffusion import ClipTokenizer, ResnetBlock, Mid, Downsample, Upsample
import numpy as np

from typing import Dict, List, Union, Callable, Optional, Any, Set, Tuple
import math, argparse, tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image


# configs:
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_refiner.yaml
configs: Dict = {
  "SDXL_Base": {
    "model": {"adm_in_channels": 2816, "in_channels": 4, "out_channels": 4, "model_channels": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048},
    "conditioner": {"concat_embedders": ["original_size_as_tuple", "crop_coords_top_left", "target_size_as_tuple"]},
    "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
    "denoiser": {"num_idx": 1000},
  },
  "SDXL_Refiner": {
    "model": {"adm_in_channels": 2560, "in_channels": 4, "out_channels": 4, "model_channels": 384, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4, 4], "d_head": 64, "transformer_depth": [4, 4, 4, 4], "ctx_dim": [1280, 1280, 1280, 1280]},
    "conditioner": {"concat_embedders": ["original_size_as_tuple", "crop_coords_top_left", "aesthetic_score"]},
    "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
    "denoiser": {"num_idx": 1000},
  }
}


def tensor_identity(x:Tensor) -> Tensor:
  return x


class UNet:
  """
  Namespace for UNet model components.
  """

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L136
  class ResBlock:
    def __init__(self, channels:int, emb_channels:int, out_channels:int):
      self.in_layers = [
        GroupNorm(32, channels),
        Tensor.silu,
        Conv2d(channels, out_channels, 3, padding=1),
      ]
      self.emb_layers = [
        Tensor.silu,
        Linear(emb_channels, out_channels),
      ]
      self.out_layers = [
        GroupNorm(32, out_channels),
        Tensor.silu,
        lambda x: x,  # needed for weights loading code to work
        Conv2d(out_channels, out_channels, 3, padding=1),
      ]
      self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else tensor_identity

    def __call__(self, x:Tensor, emb:Tensor) -> Tensor:
      h = x.sequential(self.in_layers)
      emb_out = emb.sequential(self.emb_layers)
      h = h + emb_out.reshape(*emb_out.shape, 1, 1)
      h = h.sequential(self.out_layers)
      return self.skip_connection(x) + h


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L163
  class CrossAttention:
    def __init__(self, query_dim:int, ctx_dim:int, n_heads:int, d_head:int):
      self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
      self.to_k = Linear(ctx_dim,   n_heads*d_head, bias=False)
      self.to_v = Linear(ctx_dim,   n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = [Linear(n_heads*d_head, query_dim)]

    def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
      ctx = x if ctx is None else ctx
      q,k,v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
      q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
      attention = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2)
      h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
      return h_.sequential(self.to_out)


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L180
  class GEGLU:
    def __init__(self, dim_in:int, dim_out:int):
      self.proj = Linear(dim_in, dim_out * 2)
      self.dim_out = dim_out

    def __call__(self, x:Tensor) -> Tensor:
      x, gate = self.proj(x).chunk(2, dim=-1)
      return x * gate.gelu()


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L189
  class FeedForward:
    def __init__(self, dim:int, mult:int=4):
      self.net = [
        UNet.GEGLU(dim, dim*mult),
        lambda x: x,  # needed for weights loading code to work
        Linear(dim*mult, dim)
      ]

    def __call__(self, x:Tensor) -> Tensor:
      return x.sequential(self.net)


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L200
  class BasicTransformerBlock:
    def __init__(self, dim:int, ctx_dim:int, n_heads:int, d_head:int):
      self.attn1 = UNet.CrossAttention(dim, dim, n_heads, d_head)
      self.ff    = UNet.FeedForward(dim)
      self.attn2 = UNet.CrossAttention(dim, ctx_dim, n_heads, d_head)
      self.norm1 = LayerNorm(dim)
      self.norm2 = LayerNorm(dim)
      self.norm3 = LayerNorm(dim)

    def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
      x = x + self.attn1(self.norm1(x))
      x = x + self.attn2(self.norm2(x), ctx=ctx)
      x = x + self.ff(self.norm3(x))
      return x


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L215
  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/attention.py#L619
  class SpatialTransformer:
    def __init__(self, channels:int, n_heads:int, d_head:int, ctx_dim:Union[int,List[int]], depth:int=1):
      if isinstance(ctx_dim, int):
        ctx_dim = [ctx_dim]*depth
      else:
        assert isinstance(ctx_dim, list) and depth == len(ctx_dim)
      self.norm = GroupNorm(32, channels)
      assert channels == n_heads * d_head
      self.proj_in = Linear(channels, n_heads * d_head)
      self.transformer_blocks = [UNet.BasicTransformerBlock(channels, ctx_dim[d], n_heads, d_head) for d in range(depth)]
      self.proj_out = Linear(n_heads * d_head, channels)

    def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
      B, C, H, W = x.shape
      x_in = x
      x = self.norm(x)
      x = x.reshape(B, C, H*W).permute(0,2,1) # b c h w -> b c (h w) -> b (h w) c
      x = self.proj_in(x)
      for block in self.transformer_blocks:
        x = block(x, ctx=ctx)
      x = self.proj_out(x)
      x = x.permute(0,2,1).reshape(B, C, H, W) # b (h w) c -> b c (h w) -> b c h w
      return x + x_in


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/util.py#L207
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L251
def timestep_embedding(timesteps:Tensor, dim:int, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp()
  args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
  return Tensor.cat(args.cos(), args.sin(), dim=-1).cast(dtypes.float16)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/openaimodel.py#L472
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L257
class UNetModel:
  def __init__(self, adm_in_channels:int, in_channels:int, out_channels:int, model_channels:int, attention_resolutions:List[int], num_res_blocks:int, channel_mult:List[int], d_head:int, transformer_depth:List[int], ctx_dim:Union[int,List[int]]):
    self.in_channels = in_channels
    self.model_channels = model_channels
    self.out_channels = out_channels
    self.num_res_blocks = [num_res_blocks] * len(channel_mult)

    self.attention_resolutions = attention_resolutions
    self.dropout = 0.0
    self.channel_mult = channel_mult
    self.conv_resample = True
    self.num_classes = "sequential"
    self.use_checkpoint = False
    self.d_head = d_head

    time_embed_dim = model_channels * 4
    self.time_embed = [
      Linear(model_channels, time_embed_dim),
      Tensor.silu,
      Linear(time_embed_dim, time_embed_dim),
    ]

    self.label_emb = [
      [
        Linear(adm_in_channels, time_embed_dim),
        Tensor.silu,
        Linear(time_embed_dim, time_embed_dim),
      ]
    ]

    self.input_blocks = [
      [Conv2d(in_channels, model_channels, 3, padding=1)]
    ]
    input_block_channels = [model_channels]
    ch = model_channels
    ds = 1
    for idx, mult in enumerate(channel_mult):
      for _ in range(self.num_res_blocks[idx]):
        layers: List[Any] = [
          UNet.ResBlock(ch, time_embed_dim, model_channels*mult),
        ]
        ch = mult * model_channels
        if ds in attention_resolutions:
          n_heads = ch // d_head
          layers.append(UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))

        self.input_blocks.append(layers)
        input_block_channels.append(ch)

      if idx != len(channel_mult) - 1:
        self.input_blocks.append([
          Downsample(ch),
        ])
        input_block_channels.append(ch)
        ds *= 2

    n_heads = ch // d_head
    self.middle_block: List = [
      UNet.ResBlock(ch, time_embed_dim, ch),
      UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[-1]),
      UNet.ResBlock(ch, time_embed_dim, ch),
    ]

    self.output_blocks = []
    for idx, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(self.num_res_blocks[idx] + 1):
        ich = input_block_channels.pop()
        layers = [
          UNet.ResBlock(ch + ich, time_embed_dim, model_channels*mult),
        ]
        ch = model_channels * mult

        if ds in attention_resolutions:
          n_heads = ch // d_head
          layers.append(UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))

        if idx > 0 and i == self.num_res_blocks[idx]:
          layers.append(Upsample(ch))
          ds //= 2
        self.output_blocks.append(layers)

    self.out = [
      GroupNorm(32, ch),
      Tensor.silu,
      Conv2d(model_channels, out_channels, 3, padding=1),
    ]

  def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor, y:Tensor) -> Tensor:
    t_emb = timestep_embedding(tms, self.model_channels).cast(dtypes.float16)
    emb   = t_emb.sequential(self.time_embed)

    assert y.shape[0] == x.shape[0]
    emb = emb + y.sequential(self.label_emb[0])

    emb = emb.cast(dtypes.float16)
    ctx = ctx.cast(dtypes.float16)
    x   = x  .cast(dtypes.float16)

    def run(x:Tensor, bb) -> Tensor:
      if isinstance(bb, UNet.ResBlock): x = bb(x, emb)
      elif isinstance(bb, UNet.SpatialTransformer): x = bb(x, ctx)
      else: x = bb(x)
      return x

    saved_inputs = []
    for b in self.input_blocks:
      for bb in b:
        x = run(x, bb)
      saved_inputs.append(x)
    for bb in self.middle_block:
      x = run(x, bb)
    for b in self.output_blocks:
      x = x.cat(saved_inputs.pop(), dim=1)
      for bb in b:
        x = run(x, bb)

    return x.sequential(self.out)


class DiffusionModel:
  def __init__(self, *args, **kwargs):
    self.diffusion_model = UNetModel(*args, **kwargs)


class Embedder(ABC):
  input_key: str
  @abstractmethod
  def __call__(self, x:Tensor) -> Tensor:
    pass


class Closed:
  """
  Namespace for OpenAI CLIP model components.
  """

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L329
  class ClipMlp:
    def __init__(self):
      self.fc1 = Linear(768, 3072)
      self.fc2 = Linear(3072, 768)

    def __call__(self, h:Tensor) -> Tensor:
      h = self.fc1(h)
      h = h.quick_gelu()
      h = self.fc2(h)
      return h


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L340
  class ClipAttention:
    def __init__(self):
      self.embed_dim = 768
      self.num_heads = 12
      self.head_dim = self.embed_dim // self.num_heads
      self.k_proj = Linear(self.embed_dim, self.embed_dim)
      self.v_proj = Linear(self.embed_dim, self.embed_dim)
      self.q_proj = Linear(self.embed_dim, self.embed_dim)
      self.out_proj = Linear(self.embed_dim, self.embed_dim)

    def __call__(self, hidden_states:Tensor, causal_attention_mask:Tensor) -> Tensor:
      bsz, tgt_len, embed_dim = hidden_states.shape
      q,k,v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
      q,k,v = [x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (q,k,v)]
      attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=causal_attention_mask)
      return self.out_proj(attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim))


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L357
  class ClipEncoderLayer:
    def __init__(self):
      self.self_attn = Closed.ClipAttention()
      self.layer_norm1 = LayerNorm(768)
      self.mlp = Closed.ClipMlp()
      self.layer_norm2 = LayerNorm(768)

    def __call__(self, hidden_states:Tensor, causal_attention_mask:Tensor) -> Tensor:
      residual = hidden_states
      hidden_states = self.layer_norm1(hidden_states)
      hidden_states = self.self_attn(hidden_states, causal_attention_mask)
      hidden_states = residual + hidden_states

      residual = hidden_states
      hidden_states = self.layer_norm2(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states

      return hidden_states


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L386
  class ClipTextEmbeddings:
    def __init__(self):
      self.token_embedding    = Embedding(49408, 768)
      self.position_embedding = Embedding(77, 768)

    def __call__(self, input_ids:Tensor, position_ids:Tensor) -> Tensor:
      return self.token_embedding(input_ids) + self.position_embedding(position_ids)


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L377
  class ClipEncoder:
    def __init__(self, layer_count:int=12):
      self.layers = [Closed.ClipEncoderLayer() for _ in range(layer_count)]

    def __call__(self, x:Tensor, causal_attention_mask:Tensor, ret_layer_idx:Optional[int]=None) -> Tensor:
      # the indexing of layers is NOT off by 1, the original code considers the "input" as the first hidden state
      layers = self.layers if ret_layer_idx is None else self.layers[:ret_layer_idx]
      for l in layers:
        x = l(x, causal_attention_mask)
      return x


  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L394
  class ClipTextTransformer:
    def __init__(self, ret_layer_idx:Optional[int]=None):
      self.embeddings       = Closed.ClipTextEmbeddings()
      self.encoder          = Closed.ClipEncoder()
      self.final_layer_norm = LayerNorm(768)
      self.ret_layer_idx    = ret_layer_idx

    def __call__(self, input_ids:Tensor) -> Tensor:
      x = self.embeddings(input_ids, Tensor.arange(input_ids.shape[1]).reshape(1, -1))
      x = self.encoder(x, Tensor.full((1, 1, 77, 77), float("-inf")).triu(1), self.ret_layer_idx)
      return self.final_layer_norm(x) if (self.ret_layer_idx is None) else x

  class ClipTextModel:
    def __init__(self):
      self.text_model = Closed.ClipTextTransformer(ret_layer_idx=11)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L331
class FrozenClosedClipEmbedder(Embedder):
  def __init__(self):
    self.tokenizer   = ClipTokenizer()
    self.transformer = Closed.ClipTextModel()
    self.input_key   = "txt"

  def __call__(self, text:Tensor) -> Tensor:
    tokens = Tensor(self.tokenizer.encode(text))
    return self.transformer.text_model(tokens.reshape(1,-1))


class Open:
  """
  Namespace for OpenCLIP model components.
  """

  class MultiheadAttention:
    def __init__(self, dims:int, n_heads:int):
      self.dims     = dims
      self.n_heads  = n_heads
      self.d_head   = self.dims // self.n_heads

      self.in_proj_bias   = Tensor.empty(3*dims)
      self.in_proj_weight = Tensor.empty(3*dims, dims)
      self.out_proj = Linear(dims, dims)

    def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
      T,B,C = x.shape

      proj = x.linear(self.in_proj_weight.T, self.in_proj_bias)
      proj = proj.unflatten(-1, (3,C)).unsqueeze(0).transpose(0,-2)
      q,k,v = proj.chunk(3)

      q,k,v = [y.reshape(T, B*self.n_heads, self.d_head).transpose(0, 1).reshape(B, self.n_heads, T, self.d_head) for y in (q,k,v)]

      attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
      attn_output = attn_output.permute(2,0,1,3).reshape(B*T, C)

      attn_output = self.out_proj(attn_output)
      attn_output = attn_output.reshape(T, B, C)

      return attn_output


  class Mlp:
    def __init__(self, dims, hidden_dims):
      self.c_fc   = Linear(dims, hidden_dims)
      self.c_proj = Linear(hidden_dims, dims)

    def __call__(self, x:Tensor) -> Tensor:
      return x.sequential([self.c_fc, Tensor.gelu, self.c_proj])


  # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L210
  class ResidualAttentionBlocks:
    def __init__(self, dims:int, n_heads:int, mlp_ratio:float):
      self.ln_1 = LayerNorm(dims)
      self.attn = Open.MultiheadAttention(dims, n_heads)

      self.ln_2 = LayerNorm(dims)
      self.mlp  = Open.Mlp(dims, int(dims * mlp_ratio))

    def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
      x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
      x = x + self.mlp(self.ln_2(x))
      return x


  # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L317
  class ClipTransformer:
    def __init__(self, dims:int, layers:int, n_heads:int, mlp_ratio:float=4.0):
      self.resblocks = [
        Open.ResidualAttentionBlocks(dims, n_heads, mlp_ratio) for _ in range(layers)
      ]

    def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
      x = x.transpose(0, 1)
      for r in self.resblocks:
        x = r(x, attn_mask=attn_mask)
      x = x.transpose(0, 1)
      return x


  # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/model.py#L220
  # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L661
  class ClipTextTransformer:
    def __init__(self, dims:int, vocab_size:int=49408, n_heads:int=20, ctx_length:int=77, layers:int=32):
      self.token_embedding = Embedding(vocab_size, dims)
      self.positional_embedding = Tensor.empty(ctx_length, dims)
      self.transformer = Open.ClipTransformer(dims, layers, n_heads)
      self.ln_final = LayerNorm(dims)
      self.text_projection = Tensor.empty(dims, dims)

    @property
    def attn_mask(self) -> Tensor:
      if not hasattr(self, "_attn_mask"):
        self._attn_mask = Tensor.full((77, 77), float("-inf")).triu(1)
      return self._attn_mask


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L396
class FrozenOpenClipEmbedder(Embedder):
  def __init__(self, dims:int=1280):
    self.model = Open.ClipTextTransformer(dims)
    self.input_key = "txt"
    self.tokenizer = ClipTokenizer()

  def text_transformer_forward(self, x:Tensor, attn_mask:Optional[Tensor]=None):
    for r in self.model.transformer.resblocks:
      x, penultimate = r(x, attn_mask=attn_mask), x
    return x.permute(1,0,2), penultimate.permute(1,0,2)

  def __call__(self, text:Tensor) -> Tensor:
    tokens = Tensor(self.tokenizer.encode(text, pad_with_zeros=True), dtype=dtypes.int64).reshape(1,-1)

    x = self.model.token_embedding(tokens).add(self.model.positional_embedding).permute(1,0,2)
    x, penultimate = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    x = self.model.ln_final(x)
    pooled = x[Tensor.arange(x.shape[0]), tokens.argmax(axis=-1)] @ self.model.text_projection

    return penultimate, pooled


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L913
class ConcatTimestepEmbedderND(Embedder):
  def __init__(self, outdim:int, input_key:str):
    self.outdim = outdim
    self.input_key = input_key

  def __call__(self, x:Tensor):
    assert len(x.shape) == 2
    emb = timestep_embedding(x.flatten(), self.outdim)
    emb = emb.reshape((x.shape[0],-1))
    return emb


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class Conditioner:
  OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
  KEY2CATDIM      = {"vector": 1, "crossattn": 2, "concat": 1}
  embedders: List[Embedder]

  def __init__(self, concat_embedders:List[str]):
    self.embedders = [
      FrozenClosedClipEmbedder(),
      FrozenOpenClipEmbedder(),
    ]
    for input_key in concat_embedders:
      self.embedders.append(ConcatTimestepEmbedderND(256, input_key))

  def get_keys(self) -> Set[str]:
    return set(e.input_key for e in self.embedders)

  def __call__(self, batch:Dict, force_zero_embeddings:List=[]) -> Dict[str,Tensor]:
    output: Dict[str,Tensor] = {}

    for embedder in self.embedders:
      emb_out = embedder(batch[embedder.input_key])

      if isinstance(emb_out, Tensor):
        emb_out = [emb_out]
      else:
        assert isinstance(emb_out, (list, tuple))

      for emb in emb_out:
        if embedder.input_key in force_zero_embeddings:
          emb = Tensor.zeros_like(emb)

        out_key = self.OUTPUT_DIM2KEYS[len(emb.shape)]
        if out_key in output:
          output[out_key] = Tensor.cat(output[out_key], emb, dim=self.KEY2CATDIM[out_key])
        else:
          output[out_key] = emb

    return output


class FirstStage:
  """
  Namespace for First Stage Model components
  """

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L74
  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L102
  class Downsample:
    def __init__(self, dims:int):
      self.conv = Conv2d(dims, dims, kernel_size=3, stride=2, padding=(0,1,0,1))

    def __call__(self, x:Tensor) -> Tensor:
      return self.conv(x)


  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L58
  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L83
  class Upsample:
    def __init__(self, dims:int):
      self.conv = Conv2d(dims, dims, kernel_size=3, stride=1, padding=1)

    def __call__(self, x:Tensor) -> Tensor:
      B,C,Y,X = x.shape
      x = x.reshape(B, C, Y, 1, X, 1).expand(B, C, Y, 2, X, 2).reshape(B, C, Y*2, X*2)
      return self.conv(x)


  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L487
  class Encoder:
    def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
      self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
      in_ch_mult = (1,) + tuple(ch_mult)

      class BlockEntry:
        def __init__(self, block:List[ResnetBlock], downsample):
          self.block = block
          self.downsample = downsample
      self.down: List[BlockEntry] = []
      for i_level in range(len(ch_mult)):
        block = []
        block_in  = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult   [i_level]
        for _ in range(num_res_blocks):
          block.append(ResnetBlock(block_in, block_out))
          block_in = block_out

        downsample = tensor_identity if (i_level == len(ch_mult)-1) else FirstStage.Downsample(block_in)
        self.down.append(BlockEntry(block, downsample))

      self.mid = Mid(block_in)

      self.norm_out = GroupNorm(32, block_in)
      self.conv_out = Conv2d(block_in, 2*z_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, x:Tensor) -> Tensor:
      h = self.conv_in(x)
      for down in self.down:
        for block in down.block:
          h = block(h)
        h = down.downsample(h)

      h = h.sequential([self.mid.block_1, self.mid.attn_1, self.mid.block_2])
      h = h.sequential([self.norm_out,    Tensor.swish,    self.conv_out   ])
      return h


  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L604
  class Decoder:
    def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
      block_in = ch * ch_mult[-1]
      curr_res = resolution // 2 ** (len(ch_mult) - 1)
      self.z_shape = (1, z_ch, curr_res, curr_res)

      self.conv_in = Conv2d(z_ch, block_in, kernel_size=3, stride=1, padding=1)

      self.mid = Mid(block_in)

      class BlockEntry:
        def __init__(self, block:List[ResnetBlock], upsample:Callable[[Any],Any]):
          self.block = block
          self.upsample = upsample
      self.up: List[BlockEntry] = []
      for i_level in reversed(range(len(ch_mult))):
        block = []
        block_out = ch * ch_mult[i_level]
        for _ in range(num_res_blocks + 1):
          block.append(ResnetBlock(block_in, block_out))
          block_in = block_out

        upsample = tensor_identity if i_level == 0 else Upsample(block_in)
        self.up.insert(0, BlockEntry(block, upsample)) # type: ignore

      self.norm_out = GroupNorm(32, block_in)
      self.conv_out = Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z:Tensor) -> Tensor:
      h = z.sequential([self.conv_in, self.mid.block_1, self.mid.attn_1, self.mid.block_2])

      for up in self.up[::-1]:
        for block in up.block:
          h = block(h)
        h = up.upsample(h)

      h = h.sequential([self.norm_out, Tensor.swish, self.conv_out])
      return h


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L102
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L437
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L508
class FirstStageModel:
  def __init__(self, embed_dim:int=4, **kwargs):
    self.encoder = FirstStage.Encoder(**kwargs)
    self.decoder = FirstStage.Decoder(**kwargs)
    self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
    self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)

  def decode(self, z:Tensor) -> Tensor:
    return z.sequential([self.post_quant_conv, self.decoder])


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/discretizer.py#L42
class LegacyDDPMDiscretization:
  def __init__(self, linear_start:float=0.00085, linear_end:float=0.0120, num_timesteps:int=1000):
    self.num_timesteps = num_timesteps
    betas = np.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)

  def __call__(self, n:int, flip:bool=False) -> Tensor:
    if n < self.num_timesteps:
      timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
      alphas_cumprod = self.alphas_cumprod[timesteps]
    elif n == self.num_timesteps:
      alphas_cumprod = self.alphas_cumprod
    sigmas = Tensor((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    sigmas = Tensor.cat(Tensor.zeros((1,)), sigmas)
    return sigmas if flip else sigmas.flip(axis=0) # sigmas is "pre-flipped", need to do oposite of flag


def append_dims(x:Tensor, t:Tensor) -> Tensor:
  dims_to_append = len(t.shape) - len(x.shape)
  assert dims_to_append >= 0
  return x.reshape(x.shape + (1,)*dims_to_append)


@TinyJit
def run(model, x, tms, ctx, y, c_out, add):
  return (model(x, tms, ctx, y)*c_out + add).realize()


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL:
  def __init__(self, config:Dict):
    self.conditioner = Conditioner(**config["conditioner"])
    self.first_stage_model = FirstStageModel(**config["first_stage_model"])
    self.model = DiffusionModel(**config["model"])

    self.discretization = LegacyDDPMDiscretization()
    self.sigmas = self.discretization(config["denoiser"]["num_idx"], flip=True)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L173
  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float=5.0) -> Tuple[Dict,Dict]:
    batch_c : Dict = {
      "txt": pos_prompt,
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    batch_uc: Dict = {
      "txt": "",
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    return model.conditioner(batch_c), model.conditioner(batch_uc, force_zero_embeddings=["txt"])

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/denoiser.py#L42
  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

    def sigma_to_idx(s:Tensor) -> Tensor:
      dists = s - self.sigmas.unsqueeze(1)
      return dists.abs().argmin(axis=0).view(*s.shape)

    sigma = self.sigmas[sigma_to_idx(sigma)]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, x)

    c_out   = -sigma
    c_in    = 1 / (sigma**2 + 1.0) ** 0.5
    c_noise = sigma_to_idx(sigma.reshape(sigma_shape))

    def prep(*tensors:Tensor):
      return tuple(t.cast(dtypes.float16).realize() for t in tensors)

    return run(self.model.diffusion_model, *prep(x*c_in, c_noise, cond["crossattn"], cond["vector"], c_out, x))

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L543
  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1.0 / 0.13025 * x)


class VanillaCFG:
  def __init__(self, scale:float):
    self.scale = scale

  def prepare_inputs(self, x:Tensor, s:float, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor,Tensor]:
    c_out = {}
    for k in c:
      assert k in ["vector", "crossattn", "concat"]
      c_out[k] = Tensor.cat(uc[k], c[k], dim=0)
    return Tensor.cat(x, x), Tensor.cat(s, s), c_out

  def __call__(self, x:Tensor, sigma:float) -> Tensor:
    x_u, x_c = x.chunk(2)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L21
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L287
class DPMPP2MSampler:
  def __init__(self, cfg_scale:float):
    self.discretization = LegacyDDPMDiscretization()
    self.guider = VanillaCFG(cfg_scale)

  def sampler_step(self, old_denoised:Optional[Tensor], prev_sigma:Optional[Tensor], sigma:Tensor, next_sigma:Tensor, denoiser, x:Tensor, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor]:
    denoised = denoiser(*self.guider.prepare_inputs(x, sigma, c, uc))
    denoised = self.guider(denoised, sigma)

    t, t_next = sigma.log().neg(), next_sigma.log().neg()
    h = t_next - t
    r = None if prev_sigma is None else (t - prev_sigma.log().neg()) / h

    mults = [t_next.neg().exp()/t.neg().exp(), (-h).exp().sub(1)]
    if r is not None:
      mults.extend([1 + 1/(2*r), 1/(2*r)])
    mults = [append_dims(m, x) for m in mults]

    x_standard = mults[0]*x - mults[1]*denoised
    if (old_denoised is None) or (next_sigma.sum().numpy().item() < 1e-14):
      return x_standard, denoised

    denoised_d = mults[2]*denoised - mults[3]*old_denoised
    x_advanced = mults[0]*x        - mults[1]*denoised_d
    x = Tensor.where(append_dims(next_sigma, x) > 0.0, x_advanced, x_standard)
    return x, denoised

  def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int) -> Tensor:
    sigmas = self.discretization(num_steps)
    x *= Tensor.sqrt(1.0 + sigmas[0] ** 2.0)
    num_sigmas = len(sigmas)

    old_denoised = None
    for i in trange(num_sigmas - 1):
      x, old_denoised = self.sampler_step(
        old_denoised=old_denoised,
        prev_sigma=(None if i==0 else sigmas[i-1].reshape(x.shape[0])),
        sigma=sigmas[i].reshape(x.shape[0]),
        next_sigma=sigmas[i+1].reshape(x.shape[0]),
        denoiser=denoiser,
        x=x,
        c=c,
        uc=uc,
      )
      x.realize()
      old_denoised.realize()

    return x


if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description="Run SDXL", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps',    type=int,   default=10, help="The number of diffusion steps")
  parser.add_argument('--prompt',   type=str,   default=default_prompt, help="Description of image to generate")
  parser.add_argument('--out',      type=str,   default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--seed',     type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=6.0, help="Prompt strength")
  parser.add_argument('--width',    type=int,   default=1024, help="The output image width")
  parser.add_argument('--height',   type=int,   default=1024, help="The output image height")
  parser.add_argument('--weights',  type=str,   help="Custom path to weights")
  parser.add_argument('--noshow',   action='store_true', help="Don't show the image")
  args = parser.parse_args()

  Tensor.no_grad = True
  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  model = SDXL(configs["SDXL_Base"])

  default_weight_url = 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors'
  weights = args.weights if args.weights else fetch(default_weight_url, 'sd_xl_base_1.0.safetensors')
  load_state_dict(model, safe_load(weights), strict=False)

  N = 1
  C = 4
  F = 8

  assert args.width  % F == 0, f"img_width must be multiple of {F}, got {args.width}"
  assert args.height % F == 0, f"img_height must be multiple of {F}, got {args.height}"

  c, uc = model.create_conditioning(args.prompt, args.width, args.height)
  del model.conditioner
  for v in c .values(): v.realize()
  for v in uc.values(): v.realize()
  print("created batch")

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L101
  shape = (N, C, args.height // F, args.width // F)
  randn = Tensor.randn(shape)

  sampler = DPMPP2MSampler(args.guidance)
  z = sampler(model.denoise, randn, c, uc, args.steps)
  print("created samples")
  x = model.decode(z).realize()
  print("decoded samples")

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,args.height,args.width).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)
  print(x.shape)

  im = Image.fromarray(x.numpy())
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()

  # validation!
  if args.prompt == default_prompt and args.steps == 10 and args.seed == 0 and args.guidance == 6.0 and args.width == args.height == 1024 \
    and not args.weights and THREEFRY:
    ref_image = Tensor(np.array(Image.open(Path(__file__).parent / "sdxl_seed0.png")))
    distance = (((x - ref_image).cast(dtypes.float) / ref_image.max())**2).mean().item()
    assert distance < 3e-4, colored(f"validation failed with {distance=}", "red")
    print(colored(f"output validated with {distance=}", "green"))
