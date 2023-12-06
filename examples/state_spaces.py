import argparse
import json
import math
import torch

from collections import namedtuple
from dataclasses import dataclass, field
from extra.models.llama import RMSNorm
from functools import partial
from pathlib import Path
from tinygrad.helpers import dtypes, fetch
from tinygrad.nn import Linear, Conv1d, LayerNorm, Embedding
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.tensor import Tensor
from typing import Optional, Tuple
from transformers import AutoTokenizer

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

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
    A = Tensor.arange(1, stop=self.d_state + 1, dtype=dtypes.float32).repeat((self.d_inner, 1))
    self.A_log = A.log()
    self.A_log.requires_grad = True

    # D "skip" parameter
    self.D = Tensor.ones(self.d_inner, requires_grad=True)

  def __call__(self, hidden_states, inference_params=None):
    batch, seqlen, _ = hidden_states.shape

    conv_state, ssm_state = None, None
    if inference_params is not None:
      # conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
      if inference_params.seqlen_offset > 0:
        out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        return out

    # We do matmul and transpose BLH -> HBL at the same time
    xz = self.in_proj.weight @ hidden_states.reshape(hidden_states.shape[-1], -1)
    xz = xz.reshape(-1, xz.shape[0], seqlen)
    if self.in_proj.bias is not None: xz = xz + self.in_proj.bias.unsqueeze(-1)

    A = self.A_log.float().exp()  # (d_inner, d_state) 
    x, z = xz.chunk(2, dim=1)
    # Compute short convolution
    # if conv_state is not None:
    #   conv_state.assign(x[:, :, -self.d_conv :])  # Update state (B D W)
    # if causal_conv1d_fn is None:
    x = self.conv1d(x)[..., :seqlen].silu()
    # else:
    #   assert self.activation in ["silu", "swish"]
    #   x = causal_conv1d_fn(
    #       x.numpy(),
    #       self.conv1d.weight.squeeze(1),
    #       self.conv1d.bias,
    #       self.activation
    #   )

    # We're careful here about the layout, to avoid extra transposes.
    # We want dt to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = self.x_proj(x.reshape(-1, x.shape[1]))  # (bl d)
    dt, B, C = torch.split(torch.from_numpy(x_dbl.numpy()), [self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt, B, C = Tensor(dt.numpy()), Tensor(B.numpy()), Tensor(C.numpy())
    dt = self.dt_proj.weight @ dt.transpose()
    dt = dt.reshape(-1, dt.shape[0], seqlen)
    B = B.reshape(-1, B.shape[1], seqlen)
    C = C.reshape(-1, C.shape[1], seqlen)
    assert self.activation in ["silu", "swish"]
    y = selective_scan_fn(
        torch.from_numpy(x.numpy()).to(torch.device("cuda:0")),
        torch.from_numpy(dt.numpy()).to(torch.device("cuda:0")),
        torch.from_numpy(A.numpy()).to(torch.device("cuda:0")),
        torch.from_numpy(B.numpy()).to(torch.device("cuda:0")),
        torch.from_numpy(C.numpy()).to(torch.device("cuda:0")),
        torch.from_numpy(self.D.float().numpy()).to(torch.device("cuda:0")),
        z=torch.from_numpy(z.numpy()).to(torch.device("cuda:0")),
        delta_bias=torch.from_numpy(self.dt_proj.bias.float().numpy()).to(torch.device("cuda:0")),
        delta_softplus=True,
        return_last_state=ssm_state is not None,
    )
    if ssm_state is not None:
        y, last_state = y
        ssm_state.assign(last_state)
    y = y.transpose(1, 2)
    out = self.out_proj(y)
      
  def step(self, hidden_states, conv_state, ssm_state):
    assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
    xz = self.in_proj(hidden_states.squeeze(1))
    x, z = xz.chunk(2, dim=-1)

    if causal_conv1d_update is None:
      # conv_state.assign(torch.roll(torch.from_numpy(conv_state.numpy()), shifts=-1, dims=-1).numpy())
      # conv_state[:, :, -1] = x
      x = (x * self.conv1d.weight.squeeze(1)).sum(axis=-1)
      if self.conv1d.bias is not None:
        x = x + self.conv1d.bias
      x = x.silu()
    else:
      x = causal_conv1d_update(x, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)

    x_db = self.x_proj(x)
    dt, B, C = torch.split(torch.from_numpy(x_db.numpy()), [self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt, B, C = Tensor(dt.numpy()), Tensor(B.numpy()), Tensor(C.numpy())
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
          dtype=self.conv1d.weight.dtype
        )
        ssm_state = Tensor.zeros(
          batch_size,
          self.d_model * self.expand,
          self.d_state,
          device=self.dt_proj.weight.device,
          dtype=self.dt_proj.weight.dtype
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
    hidden_states = self.norm(residual.cast(self.norm.weight.dtype))
    if self.residual_in_fp32:
        residual = residual.cast(dtypes.float32)
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
    hidden_states = self.norm_f(residual.cast(self.norm_f.weight.dtype))
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
  
  def generate(
      self,
      input_ids,
      max_length,
      top_k=1,
      top_p=0.0,
      temperature=1.0,
      return_dict_in_generate=False,
      output_scores=False,
      **kwargs,
  ):
    output = decode(
        input_ids, self, max_length, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs
    )
    if not output_scores:
        output.scores = None
    return output if return_dict_in_generate else output.sequences
  
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

def modify_logits_for_top_p_filtering(logits, top_p):
  """Set the logits for none top-p values to -inf. Done in-place."""
  if top_p <= 0.0 or top_p >= 1.0:
    return
  # First sort and calculate cumulative sum of probabilities.
  sorted_logits, sorted_indices = torch.sort(torch.from_numpy(logits.numpy()), descending=False)
  sorted_logits, sorted_indices = Tensor(sorted_logits), Tensor(sorted_indices)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(axis=-1)
  # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
  sorted_indices_to_remove = torch.from_numpy(cumulative_probs.numpy()) <= (1 - top_p)
  # scatter sorted tensors to original indexing
  indices_to_remove = sorted_indices_to_remove.scatter(
      1, sorted_indices, sorted_indices_to_remove
  )
  logits.masked_fill_(indices_to_remove, float("-inf"))
  return Tensor(logits.numpy())

def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
  """Sample from top-k logits.
  Arguments:
      logits: Tensor of shape (batch_size, vocab_size)
  """
  if top_k == 1:  # Short-circuit for greedy decoding
    return logits.argmax(axis=-1)
  else:
    if top_p > 0.0:
      assert top_p <= 1.0, "top-p should be in (0, 1]."
    if top_k > 0:
      top_k = min(top_k, logits.size(-1))  # Safety check
      logits_top, indices = torch.topk(logits, top_k, dim=-1)
      if temperature != 1.0:
          logits_top /= temperature
      logits_top = modify_logits_for_top_p_filtering(logits_top, top_p)
      return indices[
          torch.arange(indices.shape[0], device=indices.device),
          torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
      ]
    else:
      # Clone so that when we modify for top_p we don't change the original logits
      logits_top = logits / temperature if temperature != 1.0 else logits.clone()
      logits_top = modify_logits_for_top_p_filtering(logits_top, top_p)
      return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(
          dim=-1
      )

def decode(
  input_ids,
  model,
  max_length,
  top_k=1,
  top_p=0.0,
  temperature=1.0,
  eos_token_id=None,
  teacher_outputs=None,
  vocab_size=None,
  tensor_parallel=1,
  enable_timing=False,
):
  """Decoding, either greedy or with top-k or top-p sampling.
  If top-k = 0, don't limit the number of candidates (pure sampling).
  Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
  then top-p.
  We assume that all sequences in the same batch have the same length.

  Arguments:
      input_ids: (batch, seq_len)
      max_length: int
      teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
          logits, the next token is taken from the teacher_outputs. Useful for testing.
  Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
      sequences: (batch, max_length)
      scores: tuples of (batch, vocab_size)
  """
  batch_size, _ = input_ids.shape
  teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
  inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

  def get_logits(input_ids, inference_params):
    decoding = inference_params.seqlen_offset > 0
    if decoding:
      position_ids = Tensor.ones((batch_size, 1)) * inference_params.seqlen_offset
    else:
      position_ids = None
    if not decoding:
      logits = model(
          input_ids,
          position_ids=position_ids,
          inference_params=inference_params,
          num_last_tokens=1,
      ).squeeze(dim=1)
    return logits[..., :vocab_size] if vocab_size is not None else logits

  def sample_tokens(logits, inference_params):
    if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
      token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
    else:
      token = teacher_outputs[:, inference_params.seqlen_offset]
    # return rearrange(token, "b -> b 1")
    return token.unsqueeze(1).realize()

  def should_stop(current_token, inference_params):
    if inference_params.seqlen_offset == 0:
        return False
    if eos_token_id is not None and (current_token == eos_token_id).all():
        return True
    if inference_params.seqlen_offset >= max_length - 1:
        return True
    return False

  scores, sequences = [], [input_ids]
  while not should_stop(sequences[-1], inference_params):
    scores.append(get_logits(sequences[-1], inference_params))
    inference_params.seqlen_offset += sequences[-1].shape[1]
    sequences.append(sample_tokens(scores[-1], inference_params))
  return Tensor.cat(sequences, dim=1), tuple(scores)

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

def load_weights_and_config(model_name: str) -> Tuple[Path, ...]:
  weights_fn = fetch(f"https://huggingface.co/state-spaces/{args.model_name}/resolve/main/pytorch_model.bin")
  cfg_fn = fetch(f"https://huggingface.co/state-spaces/{args.model_name}/resolve/main/config.json")
  return weights_fn, cfg_fn


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run state space models in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model_name", type=str, default="mamba-130m", help="the state spaces model to use")
  parser.add_argument("--prompt", type=str, default="Hello.", help="the prompt to use for generation")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--gen_len", type=int, default=100)
  parser.add_argument("--top_k", type=int, default=1)
  parser.add_argument("--top_p", type=float, default=1.0)
  args = parser.parse_args()

  Tensor.training = False
  Tensor.manual_seed(0)

  weights_fn, cfg_fn = load_weights_and_config(args.model_name)

  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
  model = MambaLMHeadModel.from_pretrained(cfg_fn, weights_fn)
  repeats = 3

  if args.prompt is None:
     pass
  else:
     tokens = tokenizer(args.prompt, return_tensors="np")
     input_ids = Tensor(tokens.input_ids)
     attn_mask = Tensor(tokens.attention_mask)

  max_length = input_ids.shape[1] + args.gen_len
  fn = lambda: model.generate(
      input_ids=input_ids,
      max_length=max_length,
      return_dict_in_generate=False,
      output_scores=True,
      enable_timing=False,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
  )

  out = fn()
  if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

  for _ in range(repeats): fn()
  print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
