from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, Conv2d, GroupNorm, LayerNorm
from tinygrad.device import is_dtype_supported
from tinygrad.dtype import DType
from typing import Optional, Union, List, Any, Tuple, Callable
from examples.mlperf.helpers import gelu_erf
import math
from contextlib import contextmanager
import globvars as gv

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/util.py#L207
def timestep_embedding(timesteps:Tensor, dim:int, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half, device=timesteps.device) / half).exp()
  args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
  out = Tensor.cat(args.cos(), args.sin(), dim=-1)
  return out.cast(dtypes.float16) if is_dtype_supported(dtypes.float16) else out

@contextmanager
def default_float(dtype:DType):
  original = dtypes.default_float
  dtypes.default_float = dtype
  try: yield
  finally: dtypes.default_float = original

class ResBlock:
  def __init__(self, channels:int, emb_channels:int, out_channels:int, num_groups:int=32, norm_dtype:DType=dtypes.default_float):
    with default_float(norm_dtype):
      self.in_layers = [GroupNorm(num_groups, channels)]
    self.in_layers += [
      Tensor.silu,
      Conv2d(channels, out_channels, 3, padding=1),
    ]
    self.emb_layers = [
      Tensor.silu,
      Linear(emb_channels, out_channels),
    ]
    with default_float(norm_dtype):
      self.out_layers = [GroupNorm(num_groups, out_channels)]
    self.out_layers += [
      Tensor.silu,
      lambda x: x,  # needed for weights loading code to work
      Conv2d(out_channels, out_channels, 3, padding=1),
    ]
    self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else (lambda x: x)

  def __call__(self, x:Tensor, emb:Tensor) -> Tensor:
    # to match behavior of torch automatic mixed precision with fp16, where GroupNorm weights stay in fp32
    h = x.sequential(self.in_layers[0:2]).cast(self.in_layers[2].weight.dtype)
    h = self.in_layers[2](h)
    emb_out = emb.sequential(self.emb_layers)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = h.sequential(self.out_layers[0:3]).cast(self.out_layers[3].weight.dtype)
    h = self.out_layers[3](h)
    return self.skip_connection(x) + h

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

class GEGLU:
  def __init__(self, dim_in:int, dim_out:int, gelu_approx:str="tanh"):
    self.proj = Linear(dim_in, dim_out * 2)
    assert gelu_approx in {"tanh", "erf"}
    self.gelu = Tensor.gelu if gelu_approx == "tanh" else gelu_erf
    assert gelu_approx=="erf"
    self.dim_out = dim_out

  def __call__(self, x:Tensor) -> Tensor:
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * self.gelu(gate)

class FeedForward:
  def __init__(self, dim:int, mult:int=4, gelu_approx:str="tanh"):
    self.net: tuple[GEGLU, Callable, Linear] = (
      GEGLU(dim, dim*mult, gelu_approx=gelu_approx),
      lambda x: x,  # needed for weights loading code to work
      Linear(dim*mult, dim)
    )

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(list(self.net))

class BasicTransformerBlock:
  def __init__(self, dim:int, ctx_dim:int, n_heads:int, d_head:int, norm_dtype:DType=dtypes.default_float, gelu_approx:str="tanh"):
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff    = FeedForward(dim, gelu_approx=gelu_approx)
    self.attn2 = CrossAttention(dim, ctx_dim, n_heads, d_head)
    with default_float(norm_dtype):
      self.norm1 = LayerNorm(dim)
      self.norm2 = LayerNorm(dim)
      self.norm3 = LayerNorm(dim)

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    # casts are to match behavior of torch automatic mixed precision with fp16
    x = x + self.attn1(self.norm1(x.cast(self.norm1.weight.dtype)).cast(self.attn1.to_q.weight.dtype))
    x = x + self.attn2(self.norm2(x.cast(self.norm2.weight.dtype)).cast(self.attn2.to_q.weight.dtype), ctx=ctx)
    x = x + self.ff(self.norm3(x.cast(self.norm3.weight.dtype)).cast(self.ff.net[0].proj.weight.dtype))
    return x

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/attention.py#L619
class SpatialTransformer:
  def __init__(self, channels:int, n_heads:int, d_head:int, ctx_dim:Union[int,List[int]], use_linear:bool, depth:int=1,
               norm_dtype=dtypes.default_float, norm_eps:float=1e-5, gelu_approx:str="tanh"):
    if isinstance(ctx_dim, int):
      ctx_dim = [ctx_dim]*depth
    else:
      assert isinstance(ctx_dim, list) and depth == len(ctx_dim)
    with default_float(norm_dtype):
      self.norm = GroupNorm(32, channels, eps=norm_eps)
    assert channels == n_heads * d_head
    self.proj_in  = Linear(channels, channels) if use_linear else Conv2d(channels, channels, 1)
    self.transformer_blocks = [BasicTransformerBlock(channels, ctx_dim[d], n_heads, d_head, norm_dtype, gelu_approx) for d in range(depth)]
    self.proj_out = Linear(channels, channels) if use_linear else Conv2d(channels, channels, 1)
    self.use_linear = use_linear

  def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
    b, c, h, w = x.shape
    x_in = x
    # to match behavior of torch automatic mixed precision with fp16, where GroupNorm weights stay in fp32
    x = self.norm(x).cast(self.proj_in.weight.dtype)
    ops = [ (lambda z: z.reshape(b, c, h*w).permute(0,2,1)), (lambda z: self.proj_in(z)) ]
    x = x.sequential(ops if self.use_linear else ops[::-1])
    for block in self.transformer_blocks:
      x = block(x, ctx=ctx)
    ops = [ (lambda z: self.proj_out(z)), (lambda z: z.permute(0,2,1).reshape(b, c, h, w)) ]
    x = x.sequential(ops if self.use_linear else ops[::-1])
    return x + x_in

class Downsample:
  def __init__(self, channels:int):
    self.op = Conv2d(channels, channels, 3, stride=2, padding=1)

  def __call__(self, x:Tensor) -> Tensor:
    return self.op(x)

class Upsample:
  def __init__(self, channels:int):
    self.conv = Conv2d(channels, channels, 3, padding=1)

  def __call__(self, x:Tensor) -> Tensor:
    bs,c,py,px = x.shape
    z = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
    return self.conv(z)

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/openaimodel.py#L472
class UNetModel:
  def __init__(self, adm_in_ch:Optional[int], in_ch:int, out_ch:int, model_ch:int, attention_resolutions:List[int], num_res_blocks:int,
               channel_mult:List[int], transformer_depth:List[int], ctx_dim:Union[int,List[int]], use_linear:bool=False, d_head:Optional[int]=None,
               n_heads:Optional[int]=None, num_groups:int=32, norm_dtype:DType=dtypes.default_float, st_norm_eps:float=1e-5,
               gelu_approx="tanh"):
    self.model_ch = model_ch
    self.num_res_blocks = [num_res_blocks] * len(channel_mult)

    self.attention_resolutions = attention_resolutions
    self.d_head  = d_head
    self.n_heads = n_heads
    def get_d_and_n_heads(dims:int) -> Tuple[int,int]:
      if self.d_head is None:
        assert self.n_heads is not None, f"d_head and n_heads cannot both be None"
        return dims // self.n_heads, self.n_heads
      else:
        assert self.n_heads is None, f"d_head and n_heads cannot both be non-None"
        return self.d_head, dims // self.d_head

    time_embed_dim = model_ch * 4
    self.time_embed = [
      Linear(model_ch, time_embed_dim),
      Tensor.silu,
      Linear(time_embed_dim, time_embed_dim),
    ]

    if adm_in_ch is not None:
      self.label_emb = [
        [
          Linear(adm_in_ch, time_embed_dim),
          Tensor.silu,
          Linear(time_embed_dim, time_embed_dim),
        ]
      ]

    self.input_blocks: List[Any] = [
      [Conv2d(in_ch, model_ch, 3, padding=1)]
    ]
    input_block_channels = [model_ch]
    ch = model_ch
    ds = 1
    for idx, mult in enumerate(channel_mult):
      for _ in range(self.num_res_blocks[idx]):
        layers: List[Any] = [
          ResBlock(ch, time_embed_dim, model_ch*mult, num_groups, norm_dtype),
        ]
        ch = mult * model_ch
        if ds in attention_resolutions:
          d_head, n_heads = get_d_and_n_heads(ch)
          layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[idx], norm_dtype=norm_dtype,
                                           norm_eps=st_norm_eps, gelu_approx=gelu_approx))

        self.input_blocks.append(layers)
        input_block_channels.append(ch)

      if idx != len(channel_mult) - 1:
        self.input_blocks.append([
          Downsample(ch),
        ])
        input_block_channels.append(ch)
        ds *= 2

    d_head, n_heads = get_d_and_n_heads(ch)
    self.middle_block: List = [
      ResBlock(ch, time_embed_dim, ch, num_groups, norm_dtype),
      SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[-1], norm_dtype=norm_dtype, norm_eps=st_norm_eps,
                         gelu_approx=gelu_approx),
      ResBlock(ch, time_embed_dim, ch, num_groups, norm_dtype),
    ]

    self.output_blocks = []
    for idx, mult in list(enumerate(channel_mult))[::-1]:
      for i in range(self.num_res_blocks[idx] + 1):
        ich = input_block_channels.pop()
        layers = [
          ResBlock(ch + ich, time_embed_dim, model_ch*mult, num_groups, norm_dtype),
        ]
        ch = model_ch * mult

        if ds in attention_resolutions:
          d_head, n_heads = get_d_and_n_heads(ch)
          layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, use_linear, depth=transformer_depth[idx], norm_dtype=norm_dtype,
                                           norm_eps=st_norm_eps, gelu_approx=gelu_approx))

        if idx > 0 and i == self.num_res_blocks[idx]:
          layers.append(Upsample(ch))
          ds //= 2
        self.output_blocks.append(layers)

    with default_float(norm_dtype):
      self.out = [GroupNorm(num_groups, ch)]
    self.out += [
      Tensor.silu,
      Conv2d(model_ch, out_ch, 3, padding=1),
    ]

  def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor, y:Optional[Tensor]=None) -> Tensor:
    t_emb = timestep_embedding(tms, self.model_ch)
    emb   = t_emb.sequential(self.time_embed)

    if y is not None:
      assert y.shape[0] == x.shape[0]
      emb = emb + y.sequential(self.label_emb[0])

    if is_dtype_supported(dtypes.float16):
      emb = emb.cast(dtypes.float16)
      ctx = ctx.cast(dtypes.float16)
      x   = x  .cast(dtypes.float16)

    def run(x:Tensor, bb) -> Tensor:
      if isinstance(bb, ResBlock): x = bb(x, emb)
      elif isinstance(bb, SpatialTransformer): x = bb(x, ctx)
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
    # to match behavior of torch automatic mixed precision with fp16, where GroupNorm weights stay in fp32
    gv.md(gv.d["pre_groupnorm"], x)
    # diff.abs().mean(): 0.09301980584859848
    # a.abs().mean(): 16.88650894165039
    # diff.abs().max(): 0.6484375
    # (0.09301980584859848, 16.88650894165039, 0.6484375)
    x = x.cast(self.out[0].weight.dtype).sequential(self.out[0:2]).cast(self.out[2].weight.dtype)
    return self.out[2](x)
