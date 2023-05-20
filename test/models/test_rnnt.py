#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from models.rnnt import LSTM
import torch

class TestRNNT(unittest.TestCase):
  def test_lstm(self):
    BS, SQ, IS, HS, L = 8, 24, 512, 1024, 2

    # create in torch
    with torch.no_grad():
      torch_layer = torch.nn.LSTM(IS, HS, L)

    # create in tinygrad
    layer = LSTM(IS, HS, L, 0.0)

    # copy weights
    with torch.no_grad():
      layer.cells[0].weights_ih.assign(Tensor(torch_layer.weight_ih_l0.numpy()))
      layer.cells[0].weights_hh.assign(Tensor(torch_layer.weight_hh_l0.numpy()))
      layer.cells[0].bias_ih.assign(Tensor(torch_layer.bias_ih_l0.numpy()))
      layer.cells[0].bias_hh.assign(Tensor(torch_layer.bias_hh_l0.numpy()))
      layer.cells[1].weights_ih.assign(Tensor(torch_layer.weight_ih_l1.numpy()))
      layer.cells[1].weights_hh.assign(Tensor(torch_layer.weight_hh_l1.numpy()))
      layer.cells[1].bias_ih.assign(Tensor(torch_layer.bias_ih_l1.numpy()))
      layer.cells[1].bias_hh.assign(Tensor(torch_layer.bias_hh_l1.numpy()))

    # test
    x = Tensor.randn(SQ, BS, IS)
    z, tg_h, tg_c = layer(x, None, None)
    torch_x = torch.tensor(x.cpu().numpy())
    torch_z, pt_hidden = torch_layer(torch_x)
    np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-3, rtol=5e-3)

    # test passing hidden
    x = Tensor.randn(SQ, BS, IS)
    z, tg_h, tg_c = layer(x, tg_h, tg_c)
    torch_x = torch.tensor(x.cpu().numpy())
    torch_z, pt_hidden = torch_layer(torch_x, pt_hidden)
    np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-3, rtol=5e-3)

if __name__ == '__main__':
  unittest.main()
