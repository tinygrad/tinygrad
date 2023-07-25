import json, logging, math, os, re, sys, time, wave, argparse, numpy as np
from functools import reduce
from pathlib import Path
from typing import List
from extra.utils import download_file
from tinygrad import nn
from tinygrad.helpers import dtypes
from tinygrad.state import torch_load
from tinygrad.tensor import Tensor
from unidecode import unidecode


class ResidualCouplingBlock:
  def __init__(self,
               channels,
               hidden_channels,
               kernel_size,
               dilation_rate,
               n_layers,
               n_flows=4,
               gin_channels=0,
               share_parameter=False
               ):
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = []

    self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
      self.flows.append(Flip())


class WN:
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    assert (kernel_size % 2 == 1)
    self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels, self.p_dropout = hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout
    self.in_layers, self.res_skip_layers = [], []
    if gin_channels != 0: self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
    for i in range(n_layers):
      dilation = dilation_rate ** i
      self.in_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation,
                                      padding=int((kernel_size * dilation - dilation) / 2)))
      self.res_skip_layers.append(
        nn.Conv1d(hidden_channels, 2 * hidden_channels if i < n_layers - 1 else hidden_channels, 1))

  def forward(self, x, x_mask, g=None, **kwargs):
    output = Tensor.zeros_like(x)
    if g is not None: g = self.cond_layer(g)
    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
      else:
        g_l = Tensor.zeros_like(x_in)
      acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
        output = output + res_skip_acts[:, self.hidden_channels:, :]
      else:
        output = output + res_skip_acts
    return output * x_mask


class ResidualCouplingLayer:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.mean_only = channels, hidden_channels, kernel_size, dilation_rate, n_layers, mean_only
    self.half_channels = channels // 2
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = split(x, [self.half_channels] * 2, 1)
    stats = self.post(self.enc.forward(self.pre(x0) * x_mask, x_mask, g=g)) * x_mask
    if not self.mean_only:
      m, logs = split(stats, [self.half_channels] * 2, 1)
    else:
      m = stats
      logs = Tensor.zeros_like(m)
    if not reverse: return x0.cat((m + x1 * logs.exp() * x_mask), dim=1)
    return x0.cat(((x1 - m) * (-logs).exp() * x_mask), dim=1)
