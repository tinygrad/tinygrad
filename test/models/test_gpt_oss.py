import unittest
import torch
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv, tqdm

from icecream import ic


params = {"dim": 2880, "hidden_dim": 2880, "head_dim": 64,
          "n_heads": 64, "n_kv_heads": 8, "num_blocks": 24,
          "n_experts": 32, "n_active_experts": 4,
          "norm_eps": 1e-5, "vocab_size": 201088, "sliding_window": 2, "max_context": 4096,
          "rope_params": {"base": 150000, "scale": 32.0, "ntk_alpha": 1.0, "ntk_beta": 32.0, "initial_context_length": 4096},
          }
torch_params = {"hidden_size": 2880, "intermediate_size": 2880, "head_dim": 64,
                "num_attention_heads": 64, "num_key_value_heads": 8, "num_hidden_layers": 24,
                "num_local_experts": 32, "num_experts_per_tok": 4,
                "norm_eps": 1e-5, "vocab_size": 201088, "sliding_window": 2, "initial_context_length": 4096,
                "rope_theta": 150000, "rope_scaling": {"factor": 32.0, "beta_slow": 1.0, "beta_fast": 32.0, "rope_type": "yarn", "original_max_position_embeddings": 4096},
                }

# small params
small_params = {"dim": 2, "hidden_dim": 12, "head_dim": 2,
                "n_heads": 2, "n_kv_heads": 1, "num_blocks": 2,
                "n_experts": 3, "n_active_experts": 2,
                "norm_eps": 1e-5, "vocab_size": 24, "sliding_window": 2, "max_context": 128,
                "rope_params": {"base": 150000, "scale": 32.0, "ntk_alpha": 1.0, "ntk_beta": 32.0, "initial_context_length": 4096},
                }
small_torch_params = {"hidden_size": 2, "intermediate_size": 12, "head_dim": 2,
                      "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 2,
                      "num_local_experts": 3, "num_experts_per_tok": 2,
                      "norm_eps": 1e-5, "vocab_size": 24, "sliding_window": 2, "initial_context_length": 128,
                      "rope_theta": 150000, "rope_scaling": {"factor": 32.0, "beta_slow": 1.0, "beta_fast": 32.0, "rope_type": "yarn", "original_max_position_embeddings": 4096},
                      }

def set_equal_weights(model, torch_model, keymap, fakeweights):
  from tinygrad.nn.state import get_state_dict
  from tinygrad.dtype import _from_torch_dtype
  from tinygrad.apps.llm2 import get_keymap

  # map hf to tinygrad model state keys
  keymap = {v: k for k, v in get_keymap(params["num_blocks"]).items()}
  def fix_mxfp4_keymap(s): return s.replace('_blocks', '').replace('_scales', '')
  keymap = {fix_mxfp4_keymap(k): fix_mxfp4_keymap(v) for k, v in keymap.items()}
  ic(keymap)

  state, torch_state = get_state_dict(model), torch_model.state_dict()
  assert len(state) == len(torch_state), f"State Mismatch: tinygrad model contains {len(state)} state objects but torch model contains {len(torch_state)} state objects"
  for k, v in tqdm(state.items(), desc='Model State Dict'):
    torch_k = keymap[k]
    torch_v = torch_state[torch_k]
    assert torch_k in torch_state, f"Key Mismatch: {k} in tinygrad model but {torch_k} not in torch model"
    if fakeweights: torch_v.copy_(torch.from_numpy(v.numpy()))
    np.testing.assert_allclose(v.numpy(), torch_v.cpu().numpy(), strict=True) # check dtype, shape, and value
  torch_model.eval()
  print("Weights are equal!")

class TestGPTOSS(unittest.TestCase):
  def test_model(self):
    from tinygrad.apps.llm2 import Transformer as GptOss, download_weights, MODELS
    from transformers import GptOssForCausalLM as TorchGptOss
    from transformers import GptOssConfig

    if getenv("SMALL"):
      params["num_blocks"] = torch_params["num_hidden_layers"] = 2
      params["max_context"] = torch_params["initial_context_length"] = 32

    # Create in tinygrad
    Tensor.manual_seed(1337)
    model = GptOss(**params)

    # Create in torch
    with torch.no_grad():
      torch_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
      torch_model = TorchGptOss(GptOssConfig(**torch_params)).to(torch_device)

    # set weights
    if not getenv("FAKEWEIGHTS"):
      model_path = download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])
      model = GptOss.from_pretrained(model_path, MODELS["20B"]["params"])
      print("set tinygrad model weights")
      torch_model = torch_model.from_pretrained(model_path, local_files_only=True, cache_dir=model_path, device_map=torch_device)
      print("set torch model weights")

    # set weights and check each weight has the same shape, dtype
    set_equal_weights(model, torch_model, keymap)

    # forward pass
    seeds = (1337, 3141)
    bsz, seq_len = 2, 5
    for seed in seeds:
      np.random.seed(seed)
      input_ids = np.random.randint(torch_model.vocab_size, size=(bsz, seq_len))

      out = model(Tensor(input_ids))
      with torch.no_grad():
        torch_logits = torch_model.forward(torch.from_numpy(input_ids).long().to(torch_device)).logits
        torch_out = torch_logits[:, -1, :].softmax(-1).argmax(-1, keepdim=True)
      np.testing.assert_allclose(out.numpy(), torch_out.cpu().numpy(), atol=5e-4, rtol=5e-4)

if __name__ == '__main__':
  unittest.main()
