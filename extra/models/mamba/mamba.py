from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import SampleDecoderOnlyOutput, TextStreamer
from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass, field
from collections import namedtuple
from functools import partial
from tinygrad import Tensor, nn, dtypes
import torch
import json
import math

# TODO should we add dtype and device support for torchload
def load_state_dict_hf(path, device=None, dtype=None):
    resolved_archive_file = cached_file(path, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return nn.state.torch_load(resolved_archive_file)

def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, layer_idx=None):
    if ssm_cfg is None: ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls)
    block.layer_idx = layer_idx
    return block

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
    self.A_log = nn.state.Parameter(Tensor.empty(self.d_inner, self.d_state))
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


@dataclass
class Params:
  d_model: int = 2560
  n_layer: int = 64
  vocab_size: int = 50277
  ssm_cfg = field(default_factory=dict)
  rms_norm: bool = True
  residual_in_fp32: bool = True
  fused_add_norm: bool = True
  pad_vocab_size_multiple: int = 8



@dataclass
class InferenceParams:
  seqlen_offset: int = 0
  key_value_memory_dict: dict = field(default_factory=dict)
  lengths_per_sample: Optional[Tensor] = None
  def reset(self):
    self.seqlen_offset = 0
    if self.lengths_per_sample is not None:
      self.lengths_per_sample.zero_()


# TODO make part of MambaLMHeadModel class
class GenerationMixin:
  def generate(
    self,
    input_ids,
    max_length,
    output_scores,
    temperature,
    repetition_penalty,
    **kwargs
  ):
    output = self.decode(input_ids, self, max_length, output_scores, repetition_penalty, temperature=temperature, **kwargs)
    return output.sequence

  def decode(
    self,
    input_ids: Tensor,
    model,
    max_length,
    device="CPU",
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    vocab_size=None,
    streamer: Optional[TextStreamer] = None
    ):
    if streamer is not None: streamer.put(input_ids.to(device))
    # sequence_length = input_ids.shape
    inference_params = InferenceParams()
    def get_logits(input_id):
      logits = model(input_id, inference_params=inference_params).logits.squeeze(dim=1)
      return logits[..., :vocab_size] if vocab_size is not None else logits

    def should_stop(curr_tok):
      if inference_params.seqlen_offset == 0: return False
      if eos_token_id is not None and (curr_tok == eos_token_id).all(): return True
      if inference_params.seqlen_offset >= max_length - 1: return True
      return False
    
    scores, sequence = [], [input_ids[0]]
    seq = input_ids
    while not should_stop(sequence[-1]):
      logits = get_logits(sequence[-1])
      inference_params.seqlen_offset += 1
      if inference_params.seqlen_offset < input_ids.shape[0]:
        sampled_tok = input_ids[inference_params.seqlen_offset:inference_params.seqlen_offset+1]
      elif repetition_penalty == 1.0:
        sampled_tok = self.sample(logits, temperature=temperature)
      else:
        logits = self.adjust_logits_rep_penalty(logits.clone(), seq, repetition_penalty)
        sampled_tok = self.sample(logits, temperature=temperature)
        seq = seq.cat(sampled_tok, dim=0)
      sequence.append(sampled_tok)
      scores.append(logits)
      if streamer is not None: streamer.put(sampled_tok.to(device))
    if streamer is not None: streamer.end()
    output = SampleDecoderOnlyOutput
    return output(sequences=sequence.cat(dim=0), scores=tuple(scores))

  def sample(self, logits, topk=1, topp=0.0, temperature=1.0):
    if topk == 1: return logits.argmax(dim=-1)
    else:
      if topp > 0.0: assert topp <= 1.0, "top-p should be in range (0, 1]."
      if topk > 0.0:
        topk = min(topk, logits.size(-1))
        # TODO: tinygrad this
        logits_top, indices = torch.topk(logits, topk, dim=-1)
        if temperature != 1.0: logits_top /= temperature
        self.adjust_logits_p(logits_top, topp)
        return indices[
          Tensor.arange(indices.shape[0], device=indices.device),
          Tensor.multinomial(logits_top.softmax(dim=-1), num_samples=1).squeeze(dim=-1)
        ]
      else:
        logits_top = logits / temperature if temperature != 1.0 else logits.clone()
        self.adjust_logits_p(logits_top, topp)
        Tensor.multinomial(logits_top.softmax(dim=-1), num_samples=1).squeeze(dim=-1)

  # def adjust_logits_k(self, logits, topk):
  #   remove_indices = logits < torch.topk(logits, topk)[0][..., -1, None]
  #   logits.masked_fill_(remove_indices, float("-nf"))

  def adjust_logits_p(self, logits, topp):
    if topp <= 0.0 or topp >= 1.0: return
    sorted_logits, sorted_indices = logits.sort(logits, descending=False)
    probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove_indices = probs <= (1 - topp)
    remove = remove_indices.scatter(0, sorted_indices, remove_indices)
    logits.masked_fill_(remove, float("-inf"))

  def adjust_logits_rep_penalty(self, logits, prev, repetition_penalty=1.0):
    if repetition_penalty == 1.0: return logits
    score = torch.gather(logits, 0, prev)
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    logits.scatter_(0, prev, score)
    return logits



class MambaLMHeadModel(GenerationMixin):
  def __init__(self, config: Params):
    # TODO: do we need to set these in the class? do we use them later?
    self.config = config
    d_model = config.d_model
    num_layers = config.num_layers
    vocab_size = config.vocab_size
    pad_vocab_size_multiple = config.pad_vocab_size_multiple
    ssm_cfg = config.ssm_cfg
    if vocab_size % pad_vocab_size_multiple != 0:
      self.vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

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
  def from_pretrained(cls, path: str, device=None, dtype=None, **kwargs):
    config = Params()
    print(config.d_model)
    model = cls(config, **kwargs) 
    # TODO: do we even need a serperate function for this?
    # from GenerationMixin
    model.load_state_dict(load_state_dict_hf(path, device=device, dtype=dtype))
    return model

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
    #     config_data = load_config_hf(pretrained_model_name)
    #     config = MambaConfig(**config_data)
    #     model = cls(config, **kwargs)
    #     model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
    #     return model



def main():
  weights_path = ""
  temperature = 1.0
  rep_penalty = 1.0
  device = "CUDA"
  dtype = dtypes.float32
  prompt = ""
  prompt_length = len(prompt)
  gen_length = 100
  print(f"Loading model from {weights_path}")
  tokenizer = AutoTokenizer.from_pretrained(weights_path, device=device, dtype=dtype)
  mamba = MambaLMHeadModel.from_pretrained(weights_path, device=device, dtype=dtype)
  print(f"Number of parameters: {sum(p.numel() for p in mamba.parameters() if p.requires_grad)}")
  while 1:
    prompt = str(input("prompt: "))
    toks = tokenizer(prompt, return_tensors="pt")
    # TODO: what?
    input_ids = toks.input_ids.to(device=device)
    attention_mask = toks.attention.mask.to(device=device)
    max_len = input_ids.shape[1] + gen_length

    input_ids = input_ids[0]
    out = mamba.generate(
      input_ids=input_ids,
      max_length=max_len,
      output_scores=True,
      temperature=temperature,
      repetition_penalty=rep_penalty
    )
    print("".join(tokenizer.batch_decode(out.sequences)))



if __name__ == "__main__":
  main()



# from collections import namedtuple
# from dataclasses import dataclass, field
# from typing import Optional
# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer
# import argparse
# import time
# import json
# import torch
# import torch.nn.functional as F
# from einops import rearrange
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from mamba_ssm.mixer_seq_simple import MambaLMHeadModel
