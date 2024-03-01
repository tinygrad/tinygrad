# https://medium.com/ai-insights-cobet/building-mamba-from-scratch-a-comprehensive-code-walkthrough-5db040c28049
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torch.nn import functional as F
# from einops import rearrange
# from tqdm import tqdm

# import math
# import os
# import urllib.request
# from zipfile import ZipFile

# from transformers import AutoTokenizer

# torch.autograd.set_detect_anomaly(True)

from tinygrad import Tensor, nn, dtypes
import torch.nn as nnn

USE_MAMBA = 1
DIFF_H = 0

#TODO: put these params in main and pass as arguments
device = "CUDA"

d_model = 8
state_size = 128  # Example state size
seq_len = 100  # Example sequence length
batch_size = 256  # Example batch size
last_batch_size = 81  # only for the very last batch of the dataset
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

    # self.A = Tensor.ones(d_model, state_size, device=device)
    # NOTE: this is the same as functional.normalize(torch.ones)
    self.A = Tensor.full((d_model, state_size), 0.378, device=device)
    # TODO: is this needed?
    # Tensor.glorot_uniform((d_model, state_size), device=device)
    # nnn.init.xavier_uniform_(self.A)

    self.B = Tensor.zeros(batch_size, self.seq_len, self.state_size, device=device)
    self.C = Tensor.zeros(batch_size, self.seq_len, self.state_size, device=device)

    self.delta = Tensor.zeros(batch_size, self.seq_len, self.d_model, device=device)
    self.dA = Tensor.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
    self.dB = Tensor.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

    self.h = Tensor.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
    self.y = Tensor.zeros(batch_size, self.seq_len, self.d_model, device=device)


  def discretization(self):
    # https://arxiv.org/pdf/2312.00752.pdf (p. 28)

    # inverse() only supports square matrix
    #dB = torch.matmul(torch.inverse(A * delta), torch.matmul(dA - torch.eye(A.shape[0]), B))
    self.dB = Tensor.einsum("bld,bln->bldn", (self.delta, self.B))
    # https://github.com/state-spaces/mamba/blob/0131c1e94a46fc9f70bcfc9d57962963bb2f0b9e/mamba_ssm/modules/mamba_simple.py#L240
    #dA = torch.matrix_exp(A * delta)  # matrix_exp() only supports square matrix
    self.dA = Tensor.einsum("bld,dn->bldn", (self.delta, self.A)).exp()
    #print(f"self.dA.shape = {self.dA.shape}")
    #print(f"self.dA.requires_grad = {self.dA.requires_grad}")

    return self.dA, self.dB

  def __call__(self, x):
    self.B = self.fc2(x)
    self.C = self.fc3(x)
    self.delta = self.fc1(x).softplus()
    self.discretization()

    if DIFF_H:  # this will trigger in-place runtime error if without using `h_new`
      global current_batch_size
      current_batch_size = x.shape[0]
      if self.h.shape[0] != current_batch_size:
        #print("Adjusting h_new for the different batch size of input data `x`")
        different_batch_size = True
        h_new = self.dA.einsum("bldn,bldn->bldn", self.h[:current_batch_size, ...]) + x.reshape(current_batch_size, x.shape[1], 1) * self.dB
      else:
        different_batch_size = False
        h_new = self.dA.einsum("bldn,bldn->bldn", self.h) + x.reshape(current_batch_size, x.shape[1], x.shape[2], 1)

      self.y = self.C.einsum("bln,bldn->bld", h_new)
      # Update self.h with the detached state of h_new
      # Only do this if retaining gradients for self.h is not necessary for backprop
      # Otherwise, store h_new in a temporary list and update self.h after the loop
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

    # self.out_proj.bias._no_weight_decay = True
    self.out_proj.bias = Tensor.ones(self.out_proj.bias)
    self.S6 = S6Layer(seq_len, 2*d_model, state_size, device)
    self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1)
    self.conv_linear = nn.Linear(2*d_model, 2*d_model)
    self.norm = RMSNorm(d_model, device=device)

  def __call__(self, x):
    """
    x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
    x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
    x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
    """
    x = self.norm(x)
    x_proj = self.inp_proj(x)
    #print(f"x_proj.shape = {x_proj.shape}")
    # Add 1D convolution with kernel size 3
    x_conv = self.conv(x_proj)
    #print(f"x_conv.shape = {x_conv.shape}")
    x_conv_act = x_conv.silu()
    #print(f"x_conv_act.shape = {x_conv_act.shape}")
    # Add linear layer for conv output
    x_conv_out = self.conv_linear(x_conv_act)
    #print(f"x_conv_out.shape = {x_conv_out.shape}")
    x_ssm = self.S6(x_conv_out)
    x_act = x_ssm.silu()  # Swish activation can be implemented as x * sigmoid(x)
    #print(f"x_act.shape = {x_act.shape}")
    # residual skip connection with nonlinearity introduced by multiplication
    x_residual = self.D(x).silu()
    #print(f"x_residual.shape = {x_residual.shape}")
    x_combined = x_act * x_residual
    #print(f"x_combined.shape = {x_combined.shape}")
    x_out = self.out_proj(x_combined)
    #print(f"x_out.shape = {x_out.shape}")
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
    # self.weight = nn.Parameter(torch.ones(d_model, device=device))
    self.weight = Tensor.ones(d_model, device=device)

  def __call__(self, x):
    output = x * ((x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).rsqrt()
    return output

def main():
  x = Tensor.rand(batch_size, seq_len, d_model, device=device)
  mamba = Mamba(seq_len, d_model, state_size, device)
  norm = RMSNorm(d_model)
  x = norm(x)
  output = mamba(x)
  print(f"test_output.shape = {output.shape}")


if __name__ == "__main__":
  main()

