from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass, field
from collections import namedtuple
from functools import partial
from tinygrad import Tensor, nn
import torch
import json
import math


# TODO: use Tensor.reshape for einops.rearrange

class Mamba:
  def __init__(
    self, 
    d_model=2560, 
    d_state=16, 
    d_conv=4, 
    expand=2, 
    dt_rank="auto", 
    conv_bias=True, 
    bias=False, 
    layer_idx=None):
        
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(self.expand * self.d_model)
    self.dt_rank = math.ceil(self.d_model/16) if dt_rank == "auto" else dt_rank
    self.layer_index = layer_idx

    self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
    self.conv2d = nn.Conv2d(
      in_channels=self.d_inner,
      out_channels=self.d_inner, 
      kernel_size=d_conv, 
      groups=self.d_inner, 
      padding=d_conv-1,
      bias=conv_bias
    )
    self.x_proj = nn.Linear(self.d_inner, self.dt_rank+self.d_state*2, bias=False)
    self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
    # self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
    self.A_log = Tensor.empty(self.d_inner, self.d_state)
    self.A = None
    # self.D = nn.Parameter(torch.empty(self.d_inner))
    self.D = torch.empty(self.d_inner)
    self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

  def __call__(self, hidden_states: List[int], inference_params=None):
    assert len(hidden_states) == 2, "invalid shape for hidden states. should be [l, d]"
    seq_length, dim = hidden_states
    assert seq_length == 1, "too many tokens"
    conv_state, ssm_state = self._get_states(inference_params)
    xz = self.in_proj(hidden_states)
    x, z = xz.chunk(2, dim=1)
    # TODO: convert to tinygrad
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (d w)
    conv_state[:, -1] = x
    # x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
    x = (conv_state * self.conv2d.weight).sum()
    x += self.conv2d.bias
    x.silu()

    x_db = self.x_proj(x)
    dt, B, C = x_db.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt = self.dt_proj(dt)
    dt.softplus()

    self.A = self.A = -1 * self.A_log.exp() if self.A is None else self.A

    dA = (dt.einsum("d,dn->dn", self.A)).exp()
    dB = dt.einsum("d,dn->dn", self.B)
    # TODO: convert to tinygrad
    # ssm_state.copy_(ssm_state * dA + rearrange(x, "d -> d 1") * dB)
    ssm_state.copy_(ssm_state * dA + x.reshape() * dB)
    y = ssm_state.einsum("dn,n->d", C)
    y *= z.silu()
    out = self.out_proj(y)
    return out

  def _get_states(self, inference_params: Dict[Tuple[int]]) -> Tuple[Tensor]:
    assert self.layer_index is not None, "must pass layer_index param to Mamba"
    if self.layer_index not in inference_params: # inference_params.key_value_memory_dict
      conv_s = Tensor.zeros(self.d_inner, self.d_conv)
      ssm_s = Tensor.zeros(self.d_inner, self.d_state)
      inference_params[self.layer_index] = (conv_s, ssm_s)
    else: 
      conv_s, ssm_s = inference_params[self.layer_index]
    return conv_s, ssm_s


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, layer_idx=None):
    if ssm_cfg is None: ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls)
    block.layer_idx = layer_idx
    return block

class Block:
  def __init__(self, dim, mixer_cls, norm_cls):
    self.mixer = mixer_cls(dim)
    self.norm = norm_cls(dim)

  def __call__(self, hidden_states: Tensor, inference_params=None):
    residual = hidden_states
    hidden_states = self.norm(hidden_states)
    hidden_states = self.mixer(hidden_states, inference_params=inference_params)
    hidden_states += residual
    return hidden_states


class RMSNorm:
  def __init__(self, hidden_size, eps=1e-5):
    self.eps = eps
    self.weight = Tensor.empty(hidden_size)

  def __call__(self, x:Tensor):
    rstd = 1.0 / (x.square().mean(axis=-1, keepdim=True) + self.eps).sqrt()
    return x * rstd * self.weight

class MixerModel:
  def __init__(
    self,
    d_model: int,
    num_layers: int,
    vocab_size: int,
    ssm_cfg=None,
    n_eps: float = 1e-5
  ):
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.layers: List[Block] = [
      create_block(d_model, ssm_cfg=ssm_cfg, norm_epsilon=n_eps, layer_idx=i) 
      for i in range(num_layers)
    ]
    self.norm_f = RMSNorm(d_model, eps=n_eps)

  def __call__(self, input_ids, inference_params=None):
    hidden_states = self.embedding(input_ids)
    for layer in self.layers:
      hidden_states = layer(hidden_states, inference_params=inference_params)
    hidden_states = self.norm_f(hidden_states)
    return hidden_states

# class MambaConfig:
#     d_model: int = 2560
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = field(default_factory=dict)
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8


from mixin import GenerationMixin
class MambaLMHeadModel(GenerationMixin):
  def __init__(
    self,
    d_model: int = 2560,
    num_layers: int = 64,
    vocab_size: int = 50277,
    ssm_cfg: Dict = dict(),
    # rms_norm: bool = True,
    # residual_in_fp32: bool = True,
    # fused_add_norm: bool = True,
    pad_vocab_size: int = 8
  ):
    # TODO: do we need to set these in the class? do we use them later?
    self.d_model = d_model
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.pad_vocab_size_multiple = pad_vocab_size
    self.ssm_cfg = ssm_cfg
    if vocab_size % pad_vocab_size != 0:
      self.vocab_size += pad_vocab_size - (vocab_size % pad_vocab_size)

    self.backbone = MixerModel(
      d_model=d_model,
      num_layers=num_layers,
      vocab_size=vocab_size,
      ssm_cfg=ssm_cfg
    )

    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    self.lm_head.weight = self.backbone.embedding.weight

  def __call__(self, input_ids, inference_params=None):
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    lm_logits = self.lm_head(hidden_states)
    CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
    return CausalLMOutput(logits=lm_logits)
  
  @classmethod
  def from_pretrained(cls, model_name, device=None, dtype=None):
    pass

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
#         config_data = load_config_hf(pretrained_model_name)
#         config = MambaConfig(**config_data)
#         model = cls(config, **kwargs)
#         model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
#         return model

def main():
  mamba = MixerModel()

if __name__ == "__main__":
  main()




# from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
# from transformers.utils.hub import cached_file
# from dataclasses import dataclass, field
# from collections import namedtuple
# from functools import partial
# import torch.nn as nn
# import torch
# import json

# from mamba import Mamba, Block
# from mixin import GenerationMixin

# @dataclass
# class MambaConfig:
#     d_model: int = 2560
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = field(default_factory=dict)
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8


# def load_config_hf(model_name):
#     resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
#     return json.load(open(resolved_archive_file))

# def load_state_dict_hf(model_name, device=None, dtype=None):
#     resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
#     return torch.load(resolved_archive_file, map_location="cpu")


# def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, layer_idx=None):
#     if ssm_cfg is None:
#         ssm_cfg = {}
#     mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
#     norm_cls = partial(RMSNorm, eps=norm_epsilon)
#     block = Block(d_model, mixer_cls, norm_cls=norm_cls)
#     block.layer_idx = layer_idx
#     return block


# class RMSNorm(torch.nn.Module):
#     def __init__(self, hidden_size, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = torch.nn.Parameter(torch.empty(hidden_size))

#     def forward(self, x):
#         rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
#         return x * rstd * self.weight


# class MixerModel(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         n_layer: int,
#         vocab_size: int,
#         ssm_cfg=None,
#         norm_epsilon: float = 1e-5,
#     ) -> None:
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.layers = nn.ModuleList(
#             [
#                 create_block(d_model, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, layer_idx=i)
#                 for i in range(n_layer)
#             ]
#         )
#         self.norm_f = RMSNorm(d_model, eps=norm_epsilon)

#     def forward(self, input_ids, inference_params=None):
#         hidden_states = self.embedding(input_ids)
#         for layer in self.layers:
#             hidden_states = layer(hidden_states, inference_params=inference_params)
#         hidden_states = self.norm_f(hidden_states)
#         return hidden_states


# class MambaLMHeadModel(nn.Module, GenerationMixin):

#     def __init__(self, config: MambaConfig) -> None:
#         self.config = config
#         d_model = config.d_model
#         n_layer = config.n_layer
#         vocab_size = config.vocab_size
#         pad_vocab_size_multiple = config.pad_vocab_size_multiple
#         ssm_cfg = config.ssm_cfg

#         super().__init__()

#         if vocab_size % pad_vocab_size_multiple != 0:
#             vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

#         self.backbone = MixerModel(
#             d_model=d_model,
#             n_layer=n_layer,
#             vocab_size=vocab_size,
#             ssm_cfg=ssm_cfg,
#         )
#         self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
#         self.lm_head.weight = self.backbone.embedding.weight  # tie with the embedding weights

#     def forward(self, input_ids, inference_params=None):
#         hidden_states = self.backbone(input_ids, inference_params=inference_params)
#         lm_logits = self.lm_head(hidden_states)
#         CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
#         return CausalLMOutput(logits=lm_logits)

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
#         config_data = load_config_hf(pretrained_model_name)
#         config = MambaConfig(**config_data)
#         model = cls(config, **kwargs)
#         model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
#         return model
  





# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Mamba(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=16,
#         d_conv=4,
#         expand=2,
#         dt_rank="auto",
#         conv_bias=True,
#         bias=False,
#         layer_idx=None,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#         self.layer_idx = layer_idx

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
#         self.conv1d = nn.Conv1d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=self.d_inner,
#             padding=d_conv - 1,
#         )
#         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
#         self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
#         self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
#         self.A = None
#         self.D = nn.Parameter(torch.empty(self.d_inner))
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

#     def forward(self, hidden_states, inference_params=None):
#         """
#         hidden_states: (L, D)
#         Returns: same shape as hidden_states
#         """
#         seqlen, dim = hidden_states.shape
#         assert seqlen == 1, "Can decode only 1 token at a time"

#         conv_state, ssm_state = self._get_states_from_cache(inference_params)

#         xz = self.in_proj(hidden_states)  # (l 2d)
#         x, z = xz.chunk(2, dim=-1)  # (l d)

#         # Conv step
#         conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (d w)
#         conv_state[:, -1] = x
#         # x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (d)
#         x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
#         x = x + self.conv1d.bias
#         x = F.silu(x)

#         x_db = self.x_proj(x)  # (dt_rank+2*d_state)
#         dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         dt = self.dt_proj(dt)  # (d_inner)
#         dt = F.softplus(dt)

#         # Initialize A only once per layer
#         if self.A is None:
#             self.A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

#         # SSM step
#         # Discretize A and B
#         dA = torch.exp(torch.einsum("d,dn->dn", dt, self.A))
#         dB = torch.einsum("d,n->dn", dt, B)
#         ssm_state.copy_(ssm_state * dA + rearrange(x, "d -> d 1") * dB)
#         y = torch.einsum("dn,n->d", ssm_state, C)
#         y = y + self.D * x
#         y = y * F.silu(z)  # (d)

#         out = self.out_proj(y)
#         return out

#     def _get_states_from_cache(self, inference_params):
#         assert self.layer_idx is not None
#         if self.layer_idx not in inference_params.key_value_memory_dict:
#             conv_state = torch.zeros(self.d_inner, self.d_conv)
#             ssm_state  = torch.zeros(self.d_inner, self.d_state)
#             inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
#         else:
#             conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
#         return conv_state, ssm_state

# class Block(nn.Module):
#     def __init__(self, dim, mixer_cls, norm_cls):
#         """ Simple block wrapping a mixer class with RMSNorm and residual connection """
#         super().__init__()
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)

#     def forward(self, hidden_states: Tensor, inference_params=None):
#         """ Pass the input through the encoder layer """
#         residual = hidden_states
#         hidden_states = self.norm(hidden_states)
#         hidden_states = self.mixer(hidden_states, inference_params=inference_params)
#         hidden_states += residual
#         return hidden_states
