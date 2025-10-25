import unittest
import torch
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv

from icecream import ic

def set_equal_weights(model, torch_model):
  from tinygrad.nn.state import get_state_dict
  from tinygrad.dtype import _from_torch_dtype
  from tinygrad.apps.llm2 import get_keymap

  # map hf to tinygrad model state keys
  keymap = {v: k for k, v in get_keymap(len(model.blk)).items()}
  def fix_mxfp4_keymap(s): return s.replace('_blocks', '').replace('_scales', '')
  keymap = {fix_mxfp4_keymap(k): fix_mxfp4_keymap(v) for k, v in keymap.items()}

  state, torch_state = get_state_dict(model), torch_model.state_dict()
  assert len(state) == len(torch_state), f"State Mismatch: tinygrad model contains {len(state)} state objects but torch model contains {len(torch_state)} state objects"
  for k, v in state.items():
    torch_k = keymap[k]
    torch_v = torch_state[torch_k]
    assert torch_k in torch_state, f"State Mismatch: {k} in tinygrad model but {torch_k} not in torch model"
    torch_v.copy_(torch.from_numpy(v.numpy()))
    np.testing.assert_allclose(v.numpy(), torch_v.numpy(), strict=True) # check dtype, shape, and value
  torch_model.eval()

class TestGPTOSS(unittest.TestCase):
  def test_model(self):
    from tinygrad.apps.llm2 import Transformer as GptOss
    from transformers import GptOssForCausalLM as TorchGptOss
    from transformers import GptOssConfig

    # small params
    params = {"dim": 2, "hidden_dim": 12, "head_dim": 2, "n_heads": 2, "n_kv_heads": 1, "num_blocks": 2, "n_experts": 3, "n_active_experts": 2,
              "norm_eps": 1e-5, "vocab_size": 24, "sliding_window": 2, "max_context": 128,
              "rope_params": {"base": 150000, "scale": 32.0, "ntk_alpha": 1.0, "ntk_beta": 32.0, "initial_context_length": 4096},
              }
    torch_params = {"hidden_size": 2, "intermediate_size": 12, "head_dim": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 2, "num_local_experts": 3, "num_experts_per_tok": 2,
                    "norm_eps": 1e-5, "vocab_size": 24, "sliding_window": 2, "max_context": 128,
                    "rope_theta": 150000, "rope_scaling": {"factor": 32.0, "beta_slow": 1.0, "beta_fast": 32.0, "rope_type": "yarn", "original_max_position_embeddings": 4096},
                    }

    # Create in tinygrad
    Tensor.manual_seed(1337)
    model = GptOss(**params)

    # Create in torch
    with torch.no_grad():
      torch_model = TorchGptOss(GptOssConfig(**torch_params))

    # set weights and check each weight has the same shape, dtype
    set_equal_weights(model, torch_model)

    # forward pass
    seeds = (1337, 3141)
    bsz, seq_len = 2, 5
    for seed in seeds:
      np.random.seed(seed)
      input_ids = np.random.randint(torch_model.vocab_size, size=(bsz, seq_len))

      out = model(Tensor(input_ids))
      with torch.no_grad():
        torch_logits = torch_model.forward(torch.from_numpy(input_ids).long()).logits
        torch_out = torch_logits[:, -1, :].softmax(-1).argmax(-1, keepdim=True)
      np.testing.assert_allclose(out.numpy(), torch_out.detach().numpy(), atol=5e-4, rtol=5e-4)

if __name__ == '__main__':
  unittest.main()
