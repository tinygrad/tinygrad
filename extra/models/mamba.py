# https://arxiv.org/pdf/2312.00752.pdf
# https://medium.com/ai-insights-cobet/building-mamba-from-scratch-a-comprehensive-code-walkthrough-5db040c28049

from tinygrad import Tensor, nn, dtypes

#TODO: put these params in main and pass to classes
device = "CUDA"

DIFF_H = 0
d_model = 8
state_size = 128  
seq_len = 100  
batch_size = 256 
last_batch_size = 81 # TODO: len(data_set) % batch_size
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None


class S6Layer():
  def __init__(self, seq_len, d_model, state_size, device):
    self.fc1 = nn.Linear(d_model, d_model)
    self.fc2 = nn.Linear(d_model, state_size)
    self.fc3 = nn.Linear(d_model, state_size)

    self.seq_len = seq_len
    self.d_model = d_model
    self.state_size = state_size

    # NOTE: this is the same as functional.normalize(torch.ones)
    self.A = Tensor.full((d_model, state_size), 0.378, device=device)
    self.B = Tensor.zeros(batch_size, self.seq_len, self.state_size, device=device)
    self.C = Tensor.zeros(batch_size, self.seq_len, self.state_size, device=device)
    self.delta = Tensor.zeros(batch_size, self.seq_len, self.d_model, device=device)
    self.dA = Tensor.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
    self.dB = Tensor.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
    self.h = Tensor.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
    self.y = Tensor.zeros(batch_size, self.seq_len, self.d_model, device=device)

  def discretization(self):
    # https://arxiv.org/pdf/2312.00752.pdf (p. 28)
    self.dB = Tensor.einsum("bld,bln->bldn", (self.delta, self.B))
    self.dA = Tensor.einsum("bld,dn->bldn", (self.delta, self.A)).exp()
    return self.dA, self.dB

  def __call__(self, x):
    self.B = self.fc2(x)
    self.C = self.fc3(x)
    self.delta = self.fc1(x).softplus()
    self.discretization()

    if DIFF_H:  
      global current_batch_size
      current_batch_size = x.shape[0]
      if self.h.shape[0] != current_batch_size:
        different_batch_size = True
        h_new = self.dA.einsum("bldn,bldn->bldn", self.h[:current_batch_size, ...]) + x.reshape(current_batch_size, x.shape[1], 1) * self.dB
      else:
        different_batch_size = False
        h_new = self.dA.einsum("bldn,bldn->bldn", self.h) + x.reshape(current_batch_size, x.shape[1], x.shape[2], 1)
      self.y = self.C.einsum("bln,bldn->bld", h_new)
      global temp_buffer
      # TODO: do we need to clone/copy?
      temp_buffer = h_new.detach() if not self.h.requires_grad else h_new()
      return self.y
    else: 
      h = Tensor.zeros(x.shape[0], self.seq_len, self.d_model, self.state_size, device=x.device)
      y = Tensor.zeros_like(x)
      h = Tensor.einsum("bldn,bldn->bldn", (self.dA, h)) + x.reshape(current_batch_size, x.shape[1], x.shape[2], 1) * self.dB
      y = Tensor.einsum("bln,bldn->bld", (self.C, h))
      return y


class Block():
  def __init__(self, seq_len, d_model, state_size, device):
    self.inp_proj = nn.Linear(d_model, 2*d_model)
    self.out_proj = nn.Linear(2*d_model, d_model)
    self.D = nn.Linear(d_model, 2*d_model)

    self.out_proj.bias = Tensor.ones(self.out_proj.bias)
    self.S6 = S6Layer(seq_len, 2*d_model, state_size, device)
    self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1)
    self.conv_linear = nn.Linear(2*d_model, 2*d_model)
    self.norm = RMSNorm(d_model, device=device)

  def __call__(self, x):
    x = self.norm(x)
    x_proj = self.inp_proj(x)
    x_conv = self.conv(x_proj)
    x_conv_act = x_conv.silu()
    x_conv_out = self.conv_linear(x_conv_act)
    x_ssm = self.S6(x_conv_out)
    x_act = x_ssm.silu()  
    x_residual = self.D(x).silu()
    x_combined = x_act * x_residual
    x_out = self.out_proj(x_combined)
    return x_out
    
    
class Mamba():
  def __init__(self, seq_len, d_model, state_size, device):
    self.mb1 = Block(seq_len, d_model, state_size, device)
    self.mb2 = Block(seq_len, d_model, state_size, device)
    self.mb3 = Block(seq_len, d_model, state_size, device)

  def __call__(self, x):
    x = self.mb1(x)
    x = self.mb2(x)
    x = self.mb3(x)
    return x
    
class RMSNorm():
  def __init__(self, d_model: int, eps:float=1e-5, device="CUDA"):
    self.eps = eps
    # TODO: does this get counted in nn.state.get_parameters()
    self.weight = Tensor.ones(d_model, device=device)

  def __call__(self, x):
    output = x * ((x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).rsqrt()
    return output

def main():
  x = Tensor.rand(batch_size, seq_len, d_model, device=device)
  mamba = Mamba(seq_len, d_model, state_size, device)
  rmsnorm = RMSNorm(d_model)
  x = rmsnorm(x)
  output = mamba(x)
  print(f"shape: {output.shape}")


if __name__ == "__main__":
  main()

