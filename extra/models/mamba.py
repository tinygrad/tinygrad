"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""

from transformers.utils.hub import cached_file
from tinygrad import Tensor, Device, nn, dtypes
from transformers import AutoTokenizer
from typing import Union, Dict, List
from dataclasses import dataclass
from tqdm import trange
import numpy as np
import math
import json


@dataclass
class ModelArgs:
  d_model: int
  n_layer: int
  vocab_size: int
  d_state: int = 16
  expand: int = 2
  dt_rank: Union[int, str] = "auto"
  d_conv: int = 4 
  pad_vocab_size_multiple: int = 8
  conv_bias: bool = True
  bias: bool = False
    
  def __post_init__(self):
    self.d_inner = int(self.expand * self.d_model)
    if self.dt_rank == "auto": self.dt_rank = math.ceil(self.d_model / 16)
    if self.vocab_size % self.pad_vocab_size_multiple != 0:
      self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba():
  def __init__(self, args: ModelArgs):
    self.args = args
    self.embedding = nn.Embedding(args.vocab_size, args.d_model)
    # TODO: does nn.state.get_parameters catch these?
    self.layers: List[ResidualBlock] = [ResidualBlock(args) for _ in range(args.n_layer)]
    self.norm_f = RMSNorm(args.d_model)
    self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
    self.lm_head.weight = self.embedding.weight

  def __call__(self, input_ids):
    """
    Args:
        input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        logits: shape (b, l, vocab_size)

    Official Implementation:
        class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

    """
    x = self.embedding(input_ids)
    for layer in self.layers: x = layer(x)    
    x = self.norm_f(x)
    logits = self.lm_head(x)
    return logits
  
  @staticmethod
  def from_pretrained(pretrained_model_name: str):
    """Load pretrained weights from HuggingFace into model.
    Args:
        pretrained_model_name: One of
            * "state-spaces/mamba-2.8b-slimpj"
            * "state-spaces/mamba-2.8b"
            * "state-spaces/mamba-1.4b"
            * "state-spaces/mamba-790m"
            * "state-spaces/mamba-370m"
            * "state-spaces/mamba-130m"
                        
    Returns:
        model: Mamba model with weights loaded

    """

    def load_config_hf(model_name: str):
      resolved_archive_file = cached_file(model_name, "config.json", _raise_exceptions_for_missing_entries=False)
      return json.load(open(resolved_archive_file))
    
    def load_state_dict_hf(model_name: str) -> Dict[str, Tensor]:
      resolved_archive_file = cached_file(model_name, "pytorch_model.bin", _raise_exceptions_for_missing_entries=False)
      return nn.state.torch_load(resolved_archive_file)
    
    config_data = load_config_hf(pretrained_model_name)
    args = ModelArgs(
      d_model=config_data["d_model"],
      n_layer=config_data["n_layer"],
      vocab_size=config_data["vocab_size"]
    )
    
    model = Mamba(args)
    state_dict = load_state_dict_hf(pretrained_model_name)
    # print(f"len state_dict: {len(state_dict)}")
    # print(f"len model.get_parameters: {len(nn.state.get_parameters(model))}")
    nn.state.load_state_dict(model, state_dict, strict=False)
    return model


class ResidualBlock():
  def __init__(self, args: ModelArgs):
    """Simple block wrapping Mamba block with normalization and residual connection."""

    self.args = args
    self.mixer = MambaBlock(args)
    self.norm = RMSNorm(args.d_model)

  def __call__(self, x):
    """
    Args:
        x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        output: shape (b, l, d)

    Official Implementation:
        Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
        
        Note: the official repo chains residual blocks that look like
            [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
        where the first Add is a no-op. This is purely for performance reasons as this
        allows them to fuse the Add->Norm.

        We instead implement our blocks as the more familiar, simpler, and numerically equivalent
            [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
        
    """
    output = self.mixer(self.norm(x)) + x
    return output
            

class MambaBlock():
  def __init__(self, args: ModelArgs):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
    self.args = args
    self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
    self.conv1d = nn.Conv1d(
      in_channels=args.d_inner,
      out_channels=args.d_inner,
      bias=args.conv_bias,
      kernel_size=args.d_conv,
      groups=args.d_conv,
      padding=args.d_conv - 1
    )
    # x_proj takes in `x` and outputs the input-specific Δ, B, C
    self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
    # dt_proj projects Δ from dt_rank to d_in
    self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
    A = Tensor.arange(1, args.d_state + 1).repeat([args.d_inner, 1])
    self.A_log = A.log()
    self.D = Tensor.ones(args.d_inner)
    self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    
  def __call__(self, x: Tensor):
    """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

    Args:
        x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        output: shape (b, l, d)
    
    Official Implementation:
        class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        
    """
    (b, l, d) = x.shape
    x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
    (x, res) = x_and_res.split(sizes=[self.args.d_inner, self.args.d_inner], dim=-1)
    x = x.reshape((x.shape[0], x.shape[2], x.shape[1]))
    x = self.conv1d(x)[:, :, :l]
    x = x.reshape((x.shape[0], x.shape[2], x.shape[1]))
    x = x.silu()
    y = self.ssm(x)
    y = y * res.silu()
    output = self.out_proj(y)
    return output
  
  def ssm(self, x):
    """Runs the SSM. See:
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    Args:
        x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        
    """
    (d_in, n) = self.A_log.shape

    # Compute ∆ A B C D, the state space parameters.
    #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn"t selective)
    #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    #                                  and is why Mamba is called **selective** state spaces)
    
    A = -self.A_log.float().exp() # shape (d_in, n)
    D = self.D.float()
    x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
    (delta, B, C) = x_dbl.split(sizes=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
    delta = self.dt_proj(delta).softplus()  # (b, l, d_in)
    y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
    return y

    
  def selective_scan(self, u, delta, A, B, C, D):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn"t match exactly.
        
    """
    (b, l, d_in) = u.shape
    n = A.shape[1]
    
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn"t change much with the simplification on B"
    deltaA = Tensor.einsum("bld,dn->bldn", delta, A).exp()
    deltaB_u = Tensor.einsum("bld,bln,bld->bldn", delta, B, u)
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    x = Tensor.zeros((b, d_in, n), device=deltaA.device)
    ys = []    
    for i in range(l):
      x = deltaA[:, i] * x + deltaB_u[:, i]
      y = Tensor.einsum("bdn,bn->bd", x, C[:, i, :])
      ys.append(y)
    y = Tensor.stack(ys, dim=1)  # shape (b, l, d_in)
    y = y + u * D
    return y


class RMSNorm():
  def __init__(self, d_model: int, eps: float = 1e-5):
    self.eps = eps
    self.weight = Tensor.ones(d_model)

  def __call__(self, x: Tensor):
    output = x * (x.pow(2).mean(axis=-1, keepdim=True) + self.eps).rsqrt() * self.weight
    return output


def generate(model, tokenizer, prompt: str, gen_length: int = 20, sample: bool = True, top_k: int = 40):
  inp = Tensor(tokenizer(prompt, max_length=gen_length, truncation=True, return_tensors="np")["input_ids"])
  for tok in trange(gen_length):
    indices = inp
    next_logits = model(indices)[:, -1]   
    probs = next_logits.softmax(axis=-1)
    (batch, vocab_size) = probs.shape     
    if top_k is not None:
      # TODO: do not convert to np - Tensor probably has something that can do this
      probs = probs.numpy()
      target_indices = np.argpartition(probs, -top_k, axis=1)[:, -top_k:]
      values = np.take_along_axis(probs, target_indices, axis=1)
      probs[probs < values[:, -1: None]] = 0
      probs = Tensor(probs)
      probs /= probs.sum(axis=1, keepdim=True)

    nxt = probs.multinomial(num_samples=1) if sample else probs.argmax(axis=-1)[:, None] 
    # print(nxt.shape)
    # print(nxt.numpy())
    inp = inp.cat(nxt, dim=1)
    # print(inp.shape)

  out = [tokenizer.decode(output.numpy().tolist()) for output in inp][0]
#   print(f"len output {len(out)}")
  return out


def main():
    # TODO: add device=device support
    # One of:
    #     "state-spaces/mamba-2.8b-slimpj"
    #     "state-spaces/mamba-2.8b"
    #     "state-spaces/mamba-1.4b"
    #     "state-spaces/mamba-790m"
    #     "state-spaces/mamba-370m"
    #     "state-spaces/mamba-130m"
    pretrained_model_name = "state-spaces/mamba-370m"

    model = Mamba.from_pretrained(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(generate(model, tokenizer, "Mamba is the"))

if __name__ == "__main__":
  main()