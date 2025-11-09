import unittest
import torch
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv, tqdm

from icecream import ic

def set_equal_weights(model, torch_model, fakeweights, num_blocks):
  from tinygrad.nn.state import get_state_dict
  from tinygrad.dtype import _from_torch_dtype
  from tinygrad.apps.llm2 import get_keymap

  # map tinygrad to hf state_dict keys
  def mxfp4_keymap(s): return s.replace('_blocks', '').replace('_scales', '')
  keymap = {mxfp4_keymap(tg_key): mxfp4_keymap(hf_key) for hf_key, tg_key in get_keymap(num_blocks).items()}

  state, torch_state = get_state_dict(model), torch_model.state_dict()
  assert len(state) == len(torch_state), f"State Mismatch: tinygrad model contains {len(state)} state objects but torch model contains {len(torch_state)} state objects:"
  for k, v in tqdm(state.items(), desc='Model State Dict'):
    torch_k = keymap[k]
    torch_v = torch_state[torch_k]
    assert torch_k in torch_state, f"Key Mismatch: {k} in tinygrad model but {torch_k} not in torch model"
    if fakeweights: torch_v.copy_(torch.from_numpy(v.numpy()))
    # check dtype, shape, and value
    np.testing.assert_allclose(v.numpy(), torch_v.cpu().numpy(), strict=True,
                               err_msg=f"tinygrad: {k=}, {v.dtype=}, {v.shape=}\npytorch:k={torch_k}, v.dtype={torch_v.dtype}, v.shape={torch_v.shape}\n{v.numpy()=}\n{torch_v.cpu().numpy()=}")
    del k, v, torch_k, torch_v # todo: remove
  torch_model.eval()
  print("Weights are equal!")

def equal_state_dicts(model, torch_model, num_blocks):
  from tinygrad.nn.state import get_state_dict
  from tinygrad.apps.llm2 import get_keymap

  # map tinygrad to hf state_dict keys
  def mxfp4_keymap(s): return s.replace('_blocks', '').replace('_scales', '')
  keymap = {mxfp4_keymap(tg_key): mxfp4_keymap(hf_key) for hf_key, tg_key in get_keymap(num_blocks).items()}

  state, torch_state = get_state_dict(model), torch_model.state_dict()
  assert len(state) == len(torch_state), f"State Mismatch: tinygrad model contains {len(state)} state objects but torch model contains {len(torch_state)} state objects:"
  for k, v in tqdm(state.items(), desc='Model State Dict'):
    torch_k = keymap[k]
    torch_v = torch_state[torch_k]
    assert torch_k in torch_state, f"Key Mismatch: {k} in tinygrad model but {torch_k} not in torch model"
    # check dtype, shape, and value
    err_msg = f"tinygrad:\tk={k}\tv.dtype={v.dtype}\tv.shape={v.shape}\npytorch:\tk={torch_k}\tv.dtype={torch_v.dtype}\tv.shape={torch_v.shape}\n{v.numpy()=}\n{torch_v.cpu().numpy()=}"
    np.testing.assert_allclose(v.numpy(), torch_v.cpu().numpy(), strict=True, err_msg=err_msg)
    del k, v, torch_k, torch_v # todo: remove
  print("Weights are equal!")

class TestGPTOSS(unittest.TestCase):
  def test_model(self):
    from tinygrad.apps.llm2 import Transformer as GptOss, download_weights, MODELS
    from transformers import GptOssForCausalLM as TorchGptOss
    from transformers import GptOssConfig

    Tensor.manual_seed(42)
    np.random.seed(42)

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

    if getenv("SMALL") == 1:
      params["num_blocks"] = torch_params["num_hidden_layers"] = 2 # fewer layers
      params["max_context"] = torch_params["initial_context_length"] = 32 # reduce kv cache
    elif getenv("SMALL") == 2:
      params, torch_params = small_params, small_torch_params

    # Create in tinygrad
    if getenv("TINY"):
      model = GptOss(**params)
      print(f"loaded tinygrad model on {Device.DEFAULT}")

    # Create in torch
    if getenv("TORCH"):
      with torch.no_grad():
        torch_device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        torch_config = GptOssConfig(**torch_params)
        torch_model = TorchGptOss(torch_config).to(torch_device)
        print(f"loaded torch model on {torch_device}")

    # set weights
    if not getenv("FAKEWEIGHTS"):
      model_path = download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])
      if getenv("TINY"):
        model = GptOss.from_pretrained(model_path, params)
        print("loaded tinygrad weights")
      if getenv("TORCH"):
        torch_model = torch_model.from_pretrained(model_path, config=torch_config, ignore_mismatched_sizes=True, local_files_only=True, cache_dir=model_path, device_map=torch_device)
        print("loaded torch weights")

    if getenv("TORCH") and getenv("TINY"):
      # set weights and check each weight has the same shape, dtype
      set_equal_weights(model, torch_model, getenv("FAKEWEIGHTS", False), params["num_blocks"])

    # forward pass
    seeds = (1337, 3141)
    bsz, seq_len = 2, 5
    for seed in seeds:
      np.random.seed(seed)
      input_ids = np.random.randint(params["vocab_size"], size=(bsz, seq_len))

      if getenv("TINY"): out = model(Tensor(input_ids))
      if getenv("TORCH"):
        with torch.no_grad():
          torch_logits = torch_model.forward(torch.from_numpy(input_ids).long().to(torch_device)).logits
          torch_out = torch_logits[:, -1, :].softmax(-1).argmax(-1, keepdim=True)
      if getenv("TORCH") and getenv("TINY"):
        np.testing.assert_allclose(out.numpy(), torch_out.cpu().numpy(), atol=5e-4, rtol=5e-4)

  def test_mxfp4_weights(self):
    import json, math
    from pathlib import Path
    from tinygrad import dtypes, Tensor, Device
    from safetensors import safe_open
    from tinygrad.nn.state import safe_load, ggml_data_to_tensor, load_state_dict
    from tinygrad.apps.llm2 import Transformer as GptOss, download_weights, MODELS
    from transformers import GptOssForCausalLM as TorchGptOss
    from transformers import GptOssConfig
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    Tensor.manual_seed(42)
    np.random.seed(42)

    # load model weights
    model_path = download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])
    weight_path = str(model_path / "model-00000-of-00002.safetensors")
    block_key, scale_key = 'model.layers.0.mlp.experts.gate_up_proj_blocks', 'model.layers.0.mlp.experts.gate_up_proj_scales'

    # load mxfp4 weights in tinygrad
    weight_map = safe_load(weight_path)
    weight_map = {"blocks": weight_map[block_key], "scales": weight_map[scale_key]}
    class Proj:
      def __init__(self): self.blocks, self.scales = Tensor.empty(32, 5760, 90, 16, dtype=dtypes.uchar), Tensor.empty(32, 5760, 90, dtype=dtypes.uchar)
    proj = Proj()
    load_state_dict(proj, weight_map)
    blocks, scales = proj.blocks, proj.scales
    ic(blocks.shape, scales.shape)

    # load mxfp4 weights in torch
    with safe_open(weight_path, framework="pt", device="cpu") as f: torch_blocks = f.get_tensor(block_key)
    with safe_open(weight_path, framework="pt", device="cpu") as f: torch_scales = f.get_tensor(scale_key)

    # check we are loading the same weights
    assert scales.shape == torch_scales.shape
    assert blocks.shape == torch_blocks.shape
    np.testing.assert_allclose(blocks.numpy(), torch_blocks.cpu().numpy(), strict=True)
    np.testing.assert_allclose(scales.numpy(), torch_scales.cpu().numpy(), strict=True)

    # dequantize in torch
    torch_out = convert_moe_packed_tensors(torch_blocks, torch_scales)

    # dequantize
    MXFP4_ID = 39
    block_size, n_blocks = scales.shape[:2]
    assert blocks.shape[:-1] == scales.shape and blocks.shape[-1] == 16
    data = scales.unsqueeze(-1).cat(blocks, dim=-1).flatten()
    out = ggml_data_to_tensor(data, scales.numel() * 32, MXFP4_ID)
    out = out.reshape(*scales.shape, 2, -1).permute(0, 2, 4, 3, 1).reshape(block_size, -1, n_blocks)

    # compare outputs
    out, torch_out = out.squeeze(), torch_out.squeeze()
    np.testing.assert_allclose(out.float().numpy(), torch_out.float().detach().cpu().numpy(), atol=1e-6, rtol=1e-6)

  def test_mxfp4_weights2(self):
    from safetensors import safe_open
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    from tinygrad import dtypes
    from tinygrad.apps.llm2 import download_weights, MODELS, fix_mxfp4
    from tinygrad.nn.state import safe_load, load_state_dict

    Tensor.manual_seed(42)
    np.random.seed(42)

    # load model weights
    model_path = download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])
    weight_path = str(model_path / "model-00000-of-00002.safetensors")
    block_key, scale_key = 'model.layers.0.mlp.experts.gate_up_proj_blocks', 'model.layers.0.mlp.experts.gate_up_proj_scales'

    # load mxfp4 weights in torch
    with safe_open(weight_path, framework="pt", device="cpu") as f: torch_blocks = f.get_tensor(block_key)
    with safe_open(weight_path, framework="pt", device="cpu") as f: torch_scales = f.get_tensor(scale_key)

    # load mxfp4 weights in tinygrad
    weight_map = safe_load(weight_path)
    weight_map = {"blocks": weight_map[block_key], "scales": weight_map[scale_key]}
    class Proj:
      def __init__(self): self.blocks, self.scales = Tensor.empty(32, 5760, 90, 16, dtype=dtypes.uchar), Tensor.empty(32, 5760, 90, dtype=dtypes.uchar)
    proj = Proj()
    load_state_dict(proj, weight_map)
    blocks, scales = proj.blocks, proj.scales
    blocks = blocks.to(Device.DEFAULT).realize()
    scales = scales.to(Device.DEFAULT).realize()

    # check we are loading the same weights
    assert scales.shape == torch_scales.shape
    assert blocks.shape == torch_blocks.shape
    np.testing.assert_allclose(blocks.numpy(), torch_blocks.cpu().numpy(), strict=True)
    np.testing.assert_allclose(scales.numpy(), torch_scales.cpu().numpy(), strict=True)

    # dequantize
    weight_map = {'blk.0.ffn_gate_up_proj_blocks': blocks, 'blk.0.ffn_gate_up_proj_scales': scales}
    dequantized_dict = fix_mxfp4(weight_map, num_blocks=1)
    weight_map = {'out': list(dequantized_dict.values())[0]}
    class Proj:
      def __init__(self): self.out = Tensor.empty(32, 2880, 5760, dtype=dtypes.float32)
    proj = Proj()
    load_state_dict(proj, weight_map)
    out = proj.out

    # dequantize in torch
    torch_out = convert_moe_packed_tensors(torch_blocks, torch_scales)

    # compare outputs
    out, torch_out = out.squeeze(), torch_out.squeeze()
    np.testing.assert_allclose(out.float().numpy(), torch_out.float().detach().cpu().numpy(), atol=1e-6, rtol=1e-6)


  def test_mxfp4_weights3(self):
    from safetensors import safe_open
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    from tinygrad.apps.llm2 import Transformer as GptOSS, download_weights, MODELS

    Tensor.manual_seed(42)
    np.random.seed(42)

    # load model weights
    model_path = download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])

    # dequantize in torch
    weight_path = str(model_path / "model-00000-of-00002.safetensors")
    block_key, scale_key = 'model.layers.0.mlp.experts.gate_up_proj_blocks', 'model.layers.0.mlp.experts.gate_up_proj_scales'
    with safe_open(weight_path, framework="pt", device="cpu") as f: torch_blocks = f.get_tensor(block_key)
    with safe_open(weight_path, framework="pt", device="cpu") as f: torch_scales = f.get_tensor(scale_key)
    torch_out = convert_moe_packed_tensors(torch_blocks, torch_scales)

    # dequantize in tinygrad (override with small params)
    params = MODELS["20B"]["params"] | {'num_blocks': 1, 'max_context': 4}
    model = GptOSS.from_pretrained(model_path, params)
    out = model.blk[0].ffn_gate_up_proj

    # compare outputs
    ic(out.numpy(), torch_out.float().detach().cpu().numpy())
    np.testing.assert_allclose(out.float().numpy(), torch_out.float().detach().cpu().numpy(), atol=1e-6, rtol=1e-6)

  @torch.no_grad()
  def test_mxfp4_weights4(self):
    from tinygrad.apps.llm2 import Transformer as GptOSS, download_weights, MODELS
    from transformers import GptOssForCausalLM as TorchGptOss, GptOssConfig

    Tensor.manual_seed(42)
    np.random.seed(42)

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
    small_params = params | {'num_blocks': 1, 'max_context': 4}
    torch_small_params = torch_params | {'num_hidden_layers': 1, 'initial_context_length': 4}

    # load model weights
    model_path = download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])

    # dequantize in tinygrad (override with small params)
    model = GptOSS.from_pretrained(model_path, small_params)
    out = model.blk[0].ffn_gate_up_proj

    # dequantize in torch
    torch_device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    torch_config = GptOssConfig(**torch_small_params)
    torch_model = TorchGptOss(torch_config).to(torch_device)
    torch_model = torch_model.from_pretrained(model_path, config=torch_config, ignore_mismatched_sizes=True, local_files_only=True, cache_dir=model_path, device_map=torch_device)
    torch_model.eval()

    equal_state_dicts(model, torch_model, num_blocks=1)


if __name__ == '__main__':
  unittest.main()
