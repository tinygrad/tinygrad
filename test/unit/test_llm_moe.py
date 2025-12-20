import unittest
import numpy as np
from tinygrad import Tensor

class TestMoEFeedForward(unittest.TestCase):
  def test_moe_feed_forward(self):
    from tinygrad.apps.llm import ExpertWeights
    num_experts, dim, hidden, k = 4, 8, 16, 2

    # set up weights: gate scales by (expert_id+1), up/down are identity-ish
    gate = ExpertWeights(num_experts, dim, hidden)
    up = ExpertWeights(num_experts, dim, hidden)
    down = ExpertWeights(num_experts, hidden, dim)
    gate.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    up.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    down.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])

    # run with known selection: experts 0 and 2 with equal probs
    x = Tensor.ones(1, 1, dim)
    sel = Tensor([[0, 2]])
    probs = Tensor([[0.5, 0.5]])

    # compute: gate[0]=1*I, gate[2]=3*I, so output is 0.5*silu(1) + 0.5*silu(3)
    x_in = x.squeeze(0).unsqueeze(1)
    out = down(sel, (gate(sel, x_in).silu() * up(sel, x_in)))
    result = (out * probs.unsqueeze(-1)).sum(axis=1)

    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(result.numpy()[0, 0], expected, rtol=1e-5)

if __name__ == '__main__':
  unittest.main()
