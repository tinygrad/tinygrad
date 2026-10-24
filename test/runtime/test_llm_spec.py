import unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig

class TestSpeculativeDecoding(unittest.TestCase):
  def _models(self, drafts:int, max_context:int) -> tuple[Transformer, Transformer]:
    ssm = SSMConfig(conv_kernel=4, state_size=8, group_count=1, time_step_rank=2, inner_size=16)
    config = TransformerConfig(num_blocks=2, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=8, head_dim=8,
                               rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=max_context, ssm_layers=(True, False), ssm=ssm,
                               spec_drafts=drafts)
    spec, plain = Transformer(config), Transformer(replace(config, spec_drafts=0))
    rng = np.random.default_rng(0)
    state_dict = nn.state.get_state_dict(spec)
    for p in state_dict.values(): p.replace(Tensor(rng.normal(size=p.shape).astype(np.float32)*0.5).cast(p.dtype)).realize()
    nn.state.load_state_dict(plain, {k:v for k,v in state_dict.items() if not k.startswith("mtp.")}, verbose=False)
    return spec, plain

  def test_matches_greedy(self):
    # drafts are accepted and rejected, and a second turn continues from the cached tokens until max_context
    spec, plain = self._models(2, 32)
    outs = []
    for model in (spec, plain):
      first = [t for _,t in zip(range(10), model.generate(list(range(8))))]
      outs.append(first + list(model.generate(list(range(8)) + first + [2])))
    self.assertEqual(outs[0], outs[1])

if __name__ == "__main__":
  unittest.main()
