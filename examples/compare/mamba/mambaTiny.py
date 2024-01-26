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
from __future__ import annotations
from typing import Any, Union, Optional
from dataclasses import dataclass, field
import math
import json
import time
from tinygrad import Tensor
import tinygrad.nn as nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import get_state_dict, load_state_dict, torch_load

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from tqdm import tqdm
MODELS = {
  "130m": {
    "dim": 768,
    "n_layers": 24,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8,
  },
  "370m": {
    "dim": 1024,
    "n_layers": 48,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8,
  },
  "790m": {
    "dim": 1536,
    "n_layers": 48,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8,
  },
  "1.4b": {
    "dim": 2048,
    "n_layer": 48,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8,
  },
  "2.8b": {
    "dim": 2560,
    "n_layer": 64,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8,
  },
}

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba:
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        # super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [ResidualBlock(args) for _ in range(args.n_layer)]
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits
    def __call__(self, input_ids) -> Any:
        return self.forward(input_ids)

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        def fetch_weights(model_name: str):
            downloaded = fetch(
                f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin?download=true"
            )
            weights = torch_load(downloaded)
            return weights
        
        
      
        config_data = MODELS[pretrained_model_name]
        args = ModelArgs(
            d_model=config_data['dim'],
            n_layer=config_data['n_layers'],
            vocab_size=config_data['vocab_size']
        )
        weights_pre = fetch_weights(pretrained_model_name)
        model = Mamba(args)
        
        # state_dict = load_state_dict_hf(pretrained_model_name)
        weights = {}
        for key in weights_pre:
            new_key = key.replace('backbone.', '')
            weights[new_key] = weights_pre[key]
        load_state_dict(model, weights)
        # model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock:
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        # super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
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
    def __call__(self, x):
        return self.forward(x)
            

class MambaBlock:
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        # super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(Tensor.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        # self.A_log = nn.Parameter(torch.log(A))
        self.A_log = A.log()
        # self.D = nn.Parameter(torch.ones(args.d_inner))
        self.D = Tensor.ones(args.d_inner)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
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

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        # x = F.silu(x)
        x = x.silu()
        y = self.ssm(x)
        # y = y * F.silu(res)
        y = y * res.silu()
        output = self.out_proj(y)
        return output
    def __call__(self,x):
        return self.forward(x)

    
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
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -Tensor.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(sizes=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        # delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
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
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        
        deltaA = Tensor.einsum('bld,dn->bldn', delta, A).exp()
        # deltaA = Tensor.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = Tensor.einsum( 'bld,bln,bld->bldn', delta, B, u)
        # deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = Tensor.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = Tensor.einsum('bdn,bn->bd', x, C[:, i, :])
            # y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = Tensor.stack(ys, dim=1)  # shape (b, l, d_in)
        y = y + u * D
        return y


class RMSNorm:
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        # super().__init__()
        self.eps = eps
        self.weight = Tensor.ones(d_model)


    def forward(self, x):
        output = x * Tensor.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    def __call__(self,x):
        return self.forward(x)
    
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
def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 10,
             sample: bool = False,
             top_k: int = None):
    inference_params = InferenceParams(max_seqlen=1, max_batch_size=1, seqlen_offset=0)
    toks = tokenizer(prompt)["input_ids"]
    for token_n in tqdm(range(n_tokens_to_gen)):
        next_token_logits = model(Tensor([toks]))[:, -1]
        probs = next_token_logits.softmax()
        next_indicies = probs.argmax(axis=-1).item()
        toks.append(next_indicies)
    output_completions = ''.join([tokenizer.decode(output) for output in toks])
    return output_completions
        
if __name__ == '__main__':
    model = Mamba.from_pretrained('370m')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    prompt = 'The sky is blue '
    s = time.time()
    print(generate(model, tokenizer, prompt))
    print('TIME: ', time.time() - s)
