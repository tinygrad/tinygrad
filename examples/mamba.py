import os, sys, math, argparse
sys.path.append(os.getcwd())
from typing import Any, Optional

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import get_state_dict, load_state_dict, torch_load

from extra.models.llama import RMSNorm
from transformers import AutoTokenizer
from einops import rearrange, repeat

MODELS = {
  "130m": {
    "dim": 768,
    "n_layers": 24,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8
  },
  "370m": {
    "dim": 1024,
    "n_layers": 48,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8
  },
  "790m": {
    "dim": 1536,
    "n_layers": 48,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8
  },
  "1.4b": {
    "dim": 2048,
    "n_layer": 48,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8
  },
  "2.8b": {
    "dim": 2560,
    "n_layer": 64,
    "vocab_size": 50277,
    "pad_vocab_size_multiple": 8
  },
}

def fetch_weights(model_name: str):
  if model_name not in MODELS.keys(): raise Exception(f"Requested unknown mamba model: {model_name}")
  downloaded = fetch(f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin?download=true")
  weights = torch_load(downloaded)
  return weights


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = delta.softplus()
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = len(B.shape) >= 3
    is_variable_C = len(C.shape) >= 3
    # if A.is_complex():
    #     if is_variable_B:
    #         B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
    #     if is_variable_C:
    #         C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    # else:
    B = B.float()
    C = C.float()
    
    x = Tensor.zeros(batch, dim, dstate)
    ys = []
    deltaA = Tensor.einsum('bdl,dn->bdln', delta, A).exp()
    if not is_variable_B:
        deltaB_u = Tensor.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if len(B.shape) == 3:
            deltaB_u = Tensor.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = Tensor.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and len(C.shape) == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = Tensor.einsum('bdn,dn->bd', x, C)
        else:
            if len(C.shape) == 3:
                y = Tensor.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = Tensor.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        # if y.is_complex():
        #     y = y.real * 2
        ys.append(y)
    y = Tensor.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * z.silu()
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaMixer:
  def __init__(
    self,
    dim,
    d_state=16,
    d_conv=4,
    expand=2,
    dt_rank="auto",
    dt_min=0.001,
    dt_max=0.1,
    dt_init="random",
    dt_scale=1.0,
    dt_init_floor=1e-4,
    conv_bias=True,
    bias=False,
    layer_idx=None,
  ):
    self.dim = dim
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(self.expand * self.dim)
    self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
    self.layer_idx = layer_idx

    self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias)

    self.conv1d = nn.Conv1d(
      in_channels=self.d_inner,
      out_channels=self.d_inner,
      bias=conv_bias,
      kernel_size=d_conv,
      groups=self.d_inner,
      padding=d_conv - 1,
    )

    self.x_proj = nn.Linear(
      self.d_inner, self.dt_rank + self.d_state * 2, bias=False
    )
    self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = self.dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
      self.dt_proj.weight = Tensor.full(self.dt_proj.weight.shape, dt_init_std)
    elif dt_init == "random":
      self.dt_proj.weight = Tensor.uniform(self.dt_proj.weight.shape, low=-dt_init_std, high=dt_init_std)
    else:
      raise NotImplementedError

    dt = (
      Tensor.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
      + math.log(dt_min)
    ).exp().maximum(dt_init_floor)
    inv_dt = dt + (-((-dt).exp() - Tensor.ones(*dt.shape))).log() # TODO: implement torch.expm1?
     
    self.dt_proj.bias.assign(inv_dt)

    # S4D real initialization
    self.A_log = Tensor.arange(1, self.d_state + 1).repeat([self.d_inner, 1]).contiguous().log()

    print(f"self.A_log: {self.A_log.shape}")

    # D "skip" parameter
    self.D = Tensor.ones(self.d_inner)  # Keep in fp32

    self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias)


  def __call__(self, hidden_states: Tensor, inference_params=None):
    batch, seqlen, dim = hidden_states.shape
    
    conv_state, ssm_state = None, None
    if inference_params is not None:
      conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
      if inference_params.seqlen_offset > 0:
        # The states are updated inplace
        out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        return out
    
    xz = rearrange(
      self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
      "d (b l) -> b d l",
      l=seqlen,
    )
    
    if self.in_proj.bias is not None:
      xz = xz + rearrange(self.in_proj.bias, "d -> d 1")
    
    A = -self.A_log.exp()
    print(f"A: {A.shape}")
    x, z = xz.chunk(2, dim=1)
    # Compute short convolution
    if conv_state is not None:
      conv_state.assign(x[:, :, -self.d_conv:]) # Update state (B D W)
      x = self.conv1d(x)[..., :seqlen].swish()
      
    x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt = x_dbl[:,:self.dt_rank]
    B = x_dbl[:,self.dt_rank:self.d_state]
    C = x_dbl[:,(self.dt_rank + self.d_state):]
    dt = self.dt_proj.weight @ dt.T
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    
    y = selective_scan_ref( # TODO: actually implement selective_scan_fn
      x,
      dt,
      A,
      B,
      C,
      self.D,
      z=z,
      delta_bias=self.dt_proj.bias,
      delta_softplus=True,
      return_last_state=ssm_state is not None,
    )
    
    if ssm_state is not None:
      y, last_state = y
      ssm_state.assign(last_state)
      
    y = rearrange(y, "b d l -> b l d")
    out = self.out_proj(y)
    
    return out

  def step(self, hidden_states, conv_state, ssm_state):
    assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
    xz = self.in_proj(hidden_states.squeeze(1)) # (B 2D)
    x, z = xz.chunk(2, dim=-1) # (B D)
    
    # Conv step
    conv_state = conv_state[:,:,1:].cat(x.unsqueeze(-1), dim=-1)
    x = (conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w")).sum(-1)
    if self.conv1d.bias is not None:
      x = x + self.conv1d.bias
    x = x.swish()
    
    x_db = self.x_proj(x) # (B dt_rank+2*d_state)
    dt = x_db[:,:self.dt_rank]
    B = x_db[:,self.dt_rank:self.d_state]
    C = x_db[:,(self.dt_rank + self.d_state):]
    # Don't add dt_bias here
    dt = self.dt_proj.weight @ dt
    A = -self.A_log.exp()
    
    # SSM step
    dt = (dt + self.dt_proj.bias).softplus()
    # TODO: Tensor.einsum?
    dA = Tensor.einsum("bdn,bn->bdn", dt, A).exp()
    dB = Tensor.einsum("bd,bn->bdn", dt, B)
    ssm_state.assign(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
    y = Tensor.einsum("bdn,bn->bd", ssm_state, C)
    y = y + self.D * x
    y = y * z.swish() # (B D)
    
    out = self.out_proj(y)
    return out.unsqueeze(1), conv_state, ssm_state
  
  def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
    assert self.layer_idx is not None
    if self.layer_idx not in inference_params.key_value_memory_dict:
      batch_shape = (batch_size,)
      conv_state = Tensor.zeros(batch_size, self.dim * self.expand, self.d_conv).realize()
      ssm_state = Tensor.zeros(batch_size, self.dim * self.expand, self.d_state).realize()
      inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
    else:
      conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
    return conv_state, ssm_state

class MambaBlock:
  def __init__(self, dim: int, norm_eps: float = 1e-5, rms_norm: bool = True, layer_idx: Optional[int] = None):
    self.mixer = MambaMixer(dim, layer_idx=layer_idx)
    if rms_norm: self.norm = RMSNorm(dim, norm_eps)
    else: raise NotImplementedError
  
  def __call__(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = self.norm(residual)
    hidden_states = self.mixer(hidden_states, inference_params=inference_params)
    return hidden_states, residual


class MambaBackbone:
  def __init__(self, dim: int, n_layers: int, vocab_size: int, rms_norm: bool = True, norm_eps: float = 1e-5):
    self.embedding = nn.Embedding(vocab_size, dim)
    self.layers = [MambaBlock(dim, rms_norm=rms_norm, layer_idx=i) for i in range(n_layers)]
    if rms_norm: self.norm_f = RMSNorm(dim, norm_eps)
  
  def __call__(self, input_ids: Tensor, inference_params=None) -> Any:
    hidden_states = self.embedding(input_ids)
    residual = None
    for layer in self.layers: hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
      
    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = self.norm_f(residual)
    
    return hidden_states

class Mamba:
  def __init__(self, dim: int, n_layers: int, vocab_size: int, pad_vocab_size_multiple: int = 1):
    if vocab_size % pad_vocab_size_multiple != 0:
      vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

    self.backbone = MambaBackbone(dim, n_layers, vocab_size)
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)
  
  def __call__(self, input_ids, inference_params=None, num_last_tokens=0):
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
      hidden_states = hidden_states[:, -num_last_tokens:]
    return self.lm_head(hidden_states)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Mamba in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--prompt", type=str, default=None, help="Prompt for LLM completion")
  parser.add_argument("--size", type=str, default="130m", help=f"Size of model to use [{', '.join([k for k in MODELS.keys()])}]")
  args = parser.parse_args()
  
  weights = fetch_weights(args.size)
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
  
  model = Mamba(**MODELS[args.size])
  load_state_dict(model, weights)
  
  for k,v in get_state_dict(model).items(): print(f"{k}: {v.shape}")
  
  tks = tokenizer("Hello world")["input_ids"]
  print(f"\n{len(tks)} tokens")
  
  model(Tensor([tks]))