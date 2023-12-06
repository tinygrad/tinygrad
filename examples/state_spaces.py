import json
import math
import torch

from collections import namedtuple
from dataclasses import dataclass, field
from extra.models.llama import RMSNorm
from functools import partial
from tinygrad.helpers import dtypes, fetch
from tinygrad.nn import Linear, Conv1d, LayerNorm, Embedding
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.tensor import Tensor
from typing import Optional
from transformers import AutoTokenizer

try:
    # NOTE: pip install causal-conv1d
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

class Mamba:
  def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
               dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
               dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,  # Fused kernel options
               layer_idx=None):
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(self.expand * self.d_model)
    self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
    self.use_fast_path = use_fast_path
    self.layer_idx = layer_idx

    self.in_proj = Linear(self.d_model, self.d_inner*2, bias=bias)
    self.conv1d = Conv1d(self.d_inner, self.d_inner, self.d_conv, padding=self.d_conv-1, groups=self.d_inner)
    self.x_proj = Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
    self.dt_proj = Linear(self.dt_rank, self.d_inner, bias=True)
    self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)

    self.activation = "silu"

    # TODO: might not need this for inference since we're loading pretrained weights
    # Initialize special dt projection to preserve variance at initialization
    # dt_init_std = self.dt_rank**-0.5 * dt_scale
    # if dt_init == "constant":
    #   self.dt_proj.weight = Tensor.ones_like(self.dt_proj.weight) * dt_init_std
    # elif dt_init == "random":
    #   self.dt_proj.weight = Tensor.uniform(self.dt_proj.weight.shape, low=-dt_init_std, high=dt_init_std)
    # else:
    #   raise NotImplementedError
    
    # TODO: might not need this for inference since we're loading pretrained weights
    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    # dt = (Tensor.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).exp().maximum(dt_init_floor)
    # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    # neg_dt = torch.from_numpy(-dt.numpy())
    # inv_dt = dt + Tensor(torch.expm1(neg_dt).numpy()).log()
    # with Tensor.train(val=False):
    #   self.dt_proj.bias.assign(inv_dt)

    # S4D real initialization
    A = Tensor.arange(1, stop=self.d_state + 1, dtype=dtypes.float32).repeat((self.d_inner, self.d_state))
    self.A_log = A.log()
    self.A_log.requires_grad = True

    # D "skip" parameter
    self.D = Tensor.ones(self.d_inner, requires_grad=True)

  def __call__(self, hidden_states, inference_params=None):
    batch = hidden_states.shape[0]

    conv_state, ssm_state = None, None
    if inference_params is not None:
      conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
      if inference_params.seqlen_offset > 0:
        out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        return out
      
  def step(self, hidden_states, conv_state, ssm_state):
    assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
    xz = self.in_proj(hidden_states.squeeze(1))
    x, z = xz.chunk(2, dim=-1)

    if causal_conv1d_update is None:
      conv_state.assign(torch.roll(torch.from_numpy(conv_state.numpy()), shifts=-1, dims=-1).numpy())
      conv_state[:, :, -1] = x
      x = (x * self.conv1d.weight.squeeze(1)).sum(axis=-1)
      if self.conv1d.bias is not None:
        x = x + self.conv1d.bias
      x = x.silu()
    else:
      x = causal_conv1d_update(x, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)

    x_db = self.x_proj(x)
    dt, B, C = split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt = dt.linear(self.dt_proj.weight)
    A = self.A_log.float().exp()

    dt = Tensor.softplus(dt + self.dt_proj.bias)
    dA = Tensor(torch.einsum("bd,dn->bdn", torch.from_numpy(dt.numpy()), A).numpy()).exp()
    dB = Tensor(torch.einsum("bd,bn->bdn", torch.from_numpy(dt.numpy()), B).numpy())
    ssm_state.assign(ssm_state * dA + x.unsqueeze(-1) * dB)
    y = Tensor(torch.einsum("bdn,bn->bd", torch.from_numpy(ssm_state.numpy()), C).numpy())
    y = y + self.D * x
    y = y * z.silu()

    out = self.out_proj(y)
    return out.unsqueeze(1), conv_state, ssm_state
  
  def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
    assert self.layer_idx is not None
    if self.layer_idx not in inference_params.key_value_memory_dict:
        conv_state = Tensor.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=self.conv1d.weight.device,
            dtype=self.conv1d.weight.dtype,
        )
        ssm_state = Tensor.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=self.dt_proj.weight.device,
            dtype=self.dt_proj.weight.dtype,
            # dtype=torch.float32,
        )
        inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
    else:
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        # TODO: What if batch size changes between generation, and we reuse the same states?
        if initialize_states:
            conv_state = Tensor.zeros_like(conv_state)
            ssm_state = Tensor.zeros_like(ssm_state)
    return conv_state, ssm_state

class Block:
  def __init__(
      self, dim, mixer_cls, norm_cls=LayerNorm, residual_in_fp32=False
  ):
    """
    Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    [Ref: https://arxiv.org/abs/2002.04745]
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    """
    self.residual_in_fp32 = residual_in_fp32
    self.mixer = mixer_cls(dim)
    self.norm = norm_cls(dim)

  def __call__(
      self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
  ):
    r"""Pass the input through the encoder layer.

    Args:
        hidden_states: the sequence to the encoder layer (required).
        residual: hidden_states = Mixer(LN(residual))
    """
    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
    if self.residual_in_fp32:
        residual = residual.to(dtypes.float32)
    hidden_states = self.mixer(hidden_states, inference_params=inference_params)
    return hidden_states, residual

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class MixerModel:
  def __init__(
      self,
      d_model: int,
      n_layer: int,
      vocab_size: int,
      ssm_cfg=None,
      norm_epsilon: float = 1e-5,
      rms_norm: bool = False,
      initializer_cfg=None,
      residual_in_fp32=False,
  ) -> None:
      super().__init__()
      self.residual_in_fp32 = residual_in_fp32
      self.embedding = Embedding(vocab_size, d_model)
      self.layers = [
        create_block(
            d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            layer_idx=i,
        )
        for i in range(n_layer)
      ]

      self.norm_f = (LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)

      # NOTE: might not need to initialize here
      # self.apply(
      #     partial(
      #         _init_weights,
      #         n_layer=n_layer,
      #         **(initializer_cfg if initializer_cfg is not None else {}),
      #     )
      # )

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
      return {
          i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
          for i, layer in enumerate(self.layers)
      }

  def __call__(self, input_ids, inference_params=None):
      hidden_states = self.embedding(input_ids)
      residual = None
      for layer in self.layers:
          hidden_states, residual = layer(
              hidden_states, residual, inference_params=inference_params
          )
      residual = (hidden_states + residual) if residual is not None else hidden_states
      hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
      return hidden_states
  
class MambaLMHeadModel:
  def __init__(
      self,
      d_model: int,
      n_layer: int,
      vocab_size: int,
      initializer_cfg=None,
      pad_vocab_size_multiple: int = 1,
      **backbone_kwargs,
  ) -> None:
      if vocab_size % pad_vocab_size_multiple != 0:
          vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
      del backbone_kwargs["fused_add_norm"]
      self.backbone = MixerModel(
          d_model=d_model,
          n_layer=n_layer,
          vocab_size=vocab_size,
          initializer_cfg=initializer_cfg,
          **backbone_kwargs
      )
      self.lm_head = Linear(d_model, vocab_size, bias=False)

      # NOTE: might not need to initialize here
      # Initialize weights and apply final processing
      # self.apply(
      #     partial(
      #         _init_weights,
      #         n_layer=n_layer,
      #         **(initializer_cfg if initializer_cfg is not None else {}),
      #     )
      # )
      self.tie_weights()

  def tie_weights(self):
      self.lm_head.weight = self.backbone.embedding.weight

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
      return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

  def __call__(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
      """
      "position_ids" is just to be compatible with Transformer generation. We don't use it.
      num_last_tokens: if > 0, only return the logits for the last n tokens
      """
      hidden_states = self.backbone(input_ids, inference_params=inference_params)
      if num_last_tokens > 0:
          hidden_states = hidden_states[:, -num_last_tokens:]
      return self.lm_head(hidden_states)
  
  @classmethod
  def from_pretrained(cls, cfg_fn, weights_fn):
     cfg = json.load(cfg_fn.open())
     weights = torch_load(weights_fn)
     model = cls(**cfg)
     load_state_dict(model, weights, strict=False)
     return model

@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    layer_idx=None
):
  if ssm_cfg is None:
      ssm_cfg = {}
  mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
  norm_cls = partial(LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
  block = Block(
      d_model,
      mixer_cls,
      norm_cls=norm_cls,
      residual_in_fp32=residual_in_fp32,
  )
  block.layer_idx = layer_idx
  return block

def split(tensor, split_sizes, dim=0):  # if split_sizes is an integer, convert it to a tuple of size split_sizes elements
  if isinstance(split_sizes, int): split_sizes = (split_sizes,) * (tensor.shape[dim] // split_sizes)
  assert sum(split_sizes) == tensor.shape[
    dim], "Sum of split_sizes must equal the dimension size of tensor along the given dimension."
  start, slices = 0, []
  for size in split_sizes:
    slice_range = [(start, start + size) if j == dim else None for j in range(len(tensor.shape))]
    slices.append(slice_range)
    start += size
  return [tensor.slice(s) for s in slices]


if __name__ == "__main__":
  # TODO: make this nice and support different weights and configs
  weights_fn = fetch("https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin")
  cfg_fn = fetch("https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json")

  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
  model = MambaLMHeadModel.from_pretrained(cfg_fn, weights_fn)

