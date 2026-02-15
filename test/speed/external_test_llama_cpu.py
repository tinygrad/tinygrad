#!/usr/bin/env python3
import os, unittest, time
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = os.environ["CPU_COUNT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1

from transformers import LlamaConfig, LlamaForCausalLM
from tinygrad import Tensor, Device, nn
from tinygrad.helpers import getenv, colorize_float
from tinygrad.apps.llm import Transformer
from tinygrad.uop.ops import UOp

# Llama 3.2 1B config (random weights, no download needed)
LLAMA_CONFIG = dict(dim=2048, hidden_dim=8192, n_heads=32, n_kv_heads=8, num_blocks=16, vocab_size=128256,
                    norm_eps=1e-5, rope_theta=500000, head_dim=64, max_context=512)

class TestLlamaCPU(unittest.TestCase):
  def test_llama_1b_decode(self):
    N, PREFILL = getenv("CNT", 8), list(range(1, 9))

    # HuggingFace torch.compile(inductor) baseline
    hf_config = LlamaConfig(hidden_size=LLAMA_CONFIG["dim"], intermediate_size=LLAMA_CONFIG["hidden_dim"],
      num_attention_heads=LLAMA_CONFIG["n_heads"], num_key_value_heads=LLAMA_CONFIG["n_kv_heads"],
      num_hidden_layers=LLAMA_CONFIG["num_blocks"], vocab_size=LLAMA_CONFIG["vocab_size"],
      max_position_embeddings=LLAMA_CONFIG["max_context"], rms_norm_eps=LLAMA_CONFIG["norm_eps"],
      rope_theta=LLAMA_CONFIG["rope_theta"], use_cache=True)
    torch_model = torch.compile(LlamaForCausalLM(hf_config).eval(), backend="inductor", mode="max-autotune-no-cudagraphs")

    # tinygrad llm.py
    tiny_model = Transformer(**LLAMA_CONFIG)
    for v in nn.state.get_state_dict(tiny_model).values(): v.realize()
    v_start_pos = UOp.variable("start_pos", 1, LLAMA_CONFIG["max_context"]-1)

    # warmup both (prefill + 3 decode)
    with torch.no_grad():
      out = torch_model(torch.tensor([PREFILL]), use_cache=True)
      for i in range(3): out = torch_model(torch.tensor([[1]]), past_key_values=out.past_key_values, use_cache=True)
    tiny_model(Tensor([PREFILL]), start_pos=0)
    for i in range(3):
      tiny_model(Tensor([[1]]), start_pos=v_start_pos.bind(len(PREFILL)+i))
      Device[Device.DEFAULT].synchronize()

    # benchmark torch decode
    with torch.no_grad():
      out = torch_model(torch.tensor([PREFILL]), use_cache=True)
      torch_times = []
      for i in range(N):
        st = time.perf_counter()
        out = torch_model(torch.tensor([[1]]), past_key_values=out.past_key_values, use_cache=True)
        torch_times.append(time.perf_counter() - st)

    # benchmark tinygrad decode
    tiny_model(Tensor([PREFILL]), start_pos=0)
    Device[Device.DEFAULT].synchronize()
    tiny_times = []
    for i in range(N):
      st = time.perf_counter()
      tiny_model(Tensor([[1]]), start_pos=v_start_pos.bind(len(PREFILL)+i))
      Device[Device.DEFAULT].synchronize()
      tiny_times.append(time.perf_counter() - st)

    et_torch, et_tiny = min(torch_times) * 1000, min(tiny_times) * 1000
    print(f"\nllama 1B decode: torch {et_torch:.2f}ms ({1000/et_torch:.2f} tok/s), tinygrad {et_tiny:.2f}ms ({1000/et_tiny:.2f} tok/s), " +
          f"{colorize_float(et_tiny/et_torch)} {'faster' if et_torch > et_tiny else 'slower'}")
    assert et_tiny <= et_torch * 1.05, f"tinygrad {et_tiny:.2f}ms slower than torch {et_torch:.2f}ms"

if __name__ == '__main__':
  unittest.main()
