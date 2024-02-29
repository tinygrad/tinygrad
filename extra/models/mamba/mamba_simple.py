from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
from dataclasses import dataclass, field
from collections import namedtuple
from functools import partial
from tinygrad import Tensor, nn
from typing import List, Callable
import json

from mamba import Mamba, Block
from mixin import GenerationMixin

def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, layer_idx=None):
    if ssm_cfg is None: ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls)
    block.layer_idx = layer_idx
    return block

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