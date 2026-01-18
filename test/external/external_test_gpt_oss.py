import unittest, pathlib
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict
from tinygrad.helpers import getenv, tqdm
from tinygrad.apps.llm2 import Transformer as GptOss, download_weights, get_keymap, MODELS

# https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
TORCH_PARAMS = {
  "20B": {
    "params": {
      "architectures": ["GptOssForCausalLM"], "attention_bias": True, "attention_dropout": 0.0, "eos_token_id": 200002, "experts_per_token": 4,
      "head_dim": 64, "hidden_act": "silu", "hidden_size": 2880, "initial_context_length": 4096, "initializer_range": 0.02,
      "intermediate_size": 2880, "layer_types": [
        "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention","full_attention", "sliding_attention",
        "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention",
        "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention",
        "full_attention", "sliding_attention", "full_attention"], "max_position_embeddings": 131072, "model_type": "gpt_oss",
      "num_attention_heads": 64, "num_experts_per_tok": 4, "num_hidden_layers": 24, "num_key_value_heads": 8, "num_local_experts": 32,
      "output_router_logits": False, "pad_token_id": 199999, "quantization_config": {"modules_to_not_convert":
        ["model.layers.*.self_attn","model.layers.*.mlp.router", "model.embed_tokens", "lm_head"], "quant_method": "mxfp4"}, "rms_norm_eps": 1e-05,
      "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096, "rope_type": "yarn",
                       "truncate": False}, "rope_theta": 150000, "router_aux_loss_coef": 0.9, "sliding_window": 128, "swiglu_limit": 7.0,
      "tie_word_embeddings": False, "transformers_version": "4.57.0", "use_cache": True, "vocab_size": 201088},
  }
}

# map tinygrad to hf state_dict keys
num_blocks = getenv("GPT_OSS_LAYERS", 1)
def mxfp4_keymap(s): return s.replace('_blocks', '').replace('_scales', '')
keymap = {mxfp4_keymap(tg_key): mxfp4_keymap(hf_key) for hf_key, tg_key in get_keymap(num_blocks).items()}

def compare_state_dicts(model, torch_model):
  state, torch_state = get_state_dict(model), torch_model.state_dict()
  assert len(state) == len(torch_state), f"State Mismatch: tinygrad has {len(state)} objects in state dict but torch has {len(torch_state)} objects"
  for k, v in tqdm(state.items(), desc='Compare State Dicts'):
    torch_k = keymap[k]
    torch_v = torch_state[torch_k]
    assert torch_k in torch_state, f"Key Mismatch: {k} in tinygrad model but {torch_k} not in torch model"
    # check dtype, shape, and value
    np.testing.assert_allclose(v.numpy(), torch_v.float().cpu().numpy(), strict=True, atol=1e-6,
                               err_msg=f"tinygrad:\tk={k}\tv.dtype={v.dtype}\tv.shape={v.shape}\npytorch:\tk={torch_k}\tv.dtype={torch_v.dtype}\tv.shape={torch_v.shape}\n{v.numpy()=}\n{torch_v.float().cpu().numpy()=}")

class TestGptOss(unittest.TestCase):
  def get_model(self, fakeweights:bool):
    import torch
    from transformers import logging as hf_logging, GptOssForCausalLM as TorchGptOss, GptOssConfig

    print(f"Loading {'fake' if fakeweights else 'real'} weights for GptOss.")
    Tensor.manual_seed(42)
    np.random.seed(42)
    hf_logging.set_verbosity_error() # Suppress warning from loading smaller params

    # load model weights
    model_path = pathlib.Path('') if fakeweights else download_weights(MODELS["20B"]["model"], MODELS["20B"]["total_num_weights"])

    # override params with smaller values for testing
    params = MODELS["20B"]["params"] | {'num_blocks': getenv("GPT_OSS_LAYERS", 1), 'max_context': 8}
    torch_params = TORCH_PARAMS["20B"]["params"] | {'num_hidden_layers': getenv("GPT_OSS_LAYERS", 1), 'initial_context_length': 8}
    torch_params["layer_types"] = torch_params["layer_types"][:torch_params["num_hidden_layers"]]

    # tinygrad model
    model = GptOss(**params) if fakeweights else GptOss.from_pretrained(model_path, params, fakeweights)

    # torch model
    torch_device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    torch_model = TorchGptOss(GptOssConfig(**torch_params)) if fakeweights else TorchGptOss.from_pretrained(
      model_path, config=GptOssConfig(**torch_params), ignore_mismatched_sizes=True, local_files_only=True, cache_dir=model_path)
    torch_model = torch_model.to(torch_device).eval()

    # set fakeweights equal to each other
    if fakeweights:
      state, torch_state = get_state_dict(model), torch_model.state_dict()
      for k, v in tqdm(state.items(), desc='Set fakeweights'):
        torch_k = keymap[k]
        torch_v = torch_state[torch_k]
        torch_v.copy_(torch.from_numpy(v.numpy()))

    return model, torch_model

  def test_model(self):
    # compare model architecture and weights (shape, dtype, values)
    # if fakeweights, only compare model architecture and weight shapes (dtype, values will always be the same)
    model, torch_model = self.get_model(getenv("FAKEWEIGHTS", 1))
    compare_state_dicts(model, torch_model)

  def test_forward(self):
    import torch

    B, T = 2, 5
    model, torch_model = self.get_model(getenv("FAKEWEIGHTS", 1))
    input_ids = np.random.randint(torch_model.vocab_size, size=(B, T))

    # two forward passes
    for _ in range(2):
      out = model(Tensor(input_ids))
      with torch.no_grad(): torch_logits = torch_model.forward(torch.from_numpy(input_ids).long().to(torch_model.device)).logits
      torch_out = torch_logits[:, -1:, :].argmax(-1)
      np.testing.assert_allclose(out.numpy(), torch_out.cpu().numpy(), atol=5e-4, rtol=5e-4)
      input_ids = np.concatenate((input_ids, out.numpy()), axis=-1)

if __name__ == '__main__':
  unittest.main()
