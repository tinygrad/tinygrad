import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import unittest
import time
import numpy as np
import torch
torch.set_num_threads(1)
np.set_printoptions(linewidth=160)
from transformers import LlamaForCausalLM, LlamaConfig
from tinygrad.apps.llm import Transformer as TinygradTransformer, models
from tinygrad import Tensor, Device, GlobalCounters, UOp, fetch
from tinygrad.helpers import colorize_float, getenv, CI

TORCHCOMPILE, FAKEWEIGHTS, CNT, WARMUP = getenv("TORCHCOMPILE", 1), getenv("FAKEWEIGHTS", 1), getenv("CNT", 10), 10
MAX_CONTEXT = WARMUP + CNT
# Llama 3.2 1B config
LLAMA_1B = {"dim": 2048, "hidden_dim": 8192, "n_heads": 32, "n_kv_heads": 8, "num_blocks": 16,
            "vocab_size": 128256, "norm_eps": 1e-5, "rope_theta": 500000.0, "head_dim": 64, "max_context": MAX_CONTEXT}

def benchmark_hf(model, start_tok, warmup=WARMUP, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  device = next(model.parameters()).device
  toks = []

  with torch.no_grad():
    past_key_values = None
    input_ids = torch.tensor([[start_tok]], device=device)
    for i in range(warmup):
      outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]
      next_tok = logits.argmax(dim=-1).item()
      input_ids = torch.tensor([[next_tok]], device=device)

  times = []
  with torch.no_grad():
    for i in range(iters):
      st = time.perf_counter()
      outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]
      next_tok = logits.argmax(dim=-1).item()
      times.append(time.perf_counter() - st)
      input_ids = torch.tensor([[next_tok]], device=device)
      toks.append(next_tok)
  return times, toks

def benchmark_tinygrad(model, start_tok, warmup=WARMUP, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  v_start_pos = UOp.variable("start_pos", 0, model.max_context-1)
  toks = []
  tok_tensor = Tensor([[start_tok]]).realize()
  for i in range(warmup):
    last_tok = model(tok_tensor, v_start_pos.bind(i)).item()
    tok_tensor.assign(Tensor([[last_tok]])).realize()

  times = []
  mems = []
  for i in range(iters):
    GlobalCounters.reset()
    st = time.perf_counter()
    last_tok = model(tok_tensor, v_start_pos.bind(warmup + i)).item()
    elapsed = time.perf_counter() - st
    times.append(elapsed)
    mems.append(getattr(GlobalCounters, "global_mem", 0))
    tok_tensor.assign(Tensor([[last_tok]])).realize()
    toks.append(last_tok)
  return times, toks, mems

class TestLlamaBenchmark(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print(f"\nTinygrad device: {Device.DEFAULT}, FAKEWEIGHTS={FAKEWEIGHTS}")
    print(f"Config: {LLAMA_1B}")

    # with Context(BEAM=0):  # don't search for faster copy kernels
    if FAKEWEIGHTS:
      print("Creating tinygrad model with random weights...")
      cls.tiny_model = TinygradTransformer(**LLAMA_1B)
      print("Creating HuggingFace model with random weights...")
      hf_config = LlamaConfig(vocab_size=LLAMA_1B["vocab_size"], hidden_size=LLAMA_1B["dim"],
        intermediate_size=LLAMA_1B["hidden_dim"], num_hidden_layers=LLAMA_1B["num_blocks"],
        num_attention_heads=LLAMA_1B["n_heads"], num_key_value_heads=LLAMA_1B["n_kv_heads"],
        rms_norm_eps=LLAMA_1B["norm_eps"], rope_theta=LLAMA_1B["rope_theta"],
        max_position_embeddings=MAX_CONTEXT * 2, use_cache=True)
      cls.hf_model = LlamaForCausalLM(hf_config)
    else:
      gguf_path = fetch(models["llama3.2:1b"])
      print(f"Loading pretrained models from GGUF: {gguf_path}")
      cls.tiny_model, _ = TinygradTransformer.from_gguf(Tensor(gguf_path), max_context=MAX_CONTEXT)
      cls.hf_model = LlamaForCausalLM.from_pretrained(gguf_path.parent, gguf_file=gguf_path.name, dtype=torch.float16)

    cls.hf_model.eval()
    cls.start_tok = 1  # starting token for generation

  def reset_kv(self):
    # reset tinygrad kv cache (llm.py uses blk[i].cache_kv)
    for block in self.tiny_model.blk:
      if hasattr(block, 'cache_kv'):
        delattr(block, 'cache_kv')

  def test_benchmark(self):
    self.reset_kv()

    if TORCHCOMPILE:
      print("\nCompiling torch model...")
      hf_model = torch.compile(self.hf_model)
    else:
      hf_model = self.hf_model

    print("Running benchmarks...")

    hf_times, hf_toks = benchmark_hf(hf_model, self.start_tok)

    self.reset_kv()
    tiny_times, tiny_toks, tiny_mems = benchmark_tinygrad(self.tiny_model, self.start_tok)

    hf_mean, hf_std = np.mean(hf_times)*1000, np.std(hf_times)*1000
    tiny_mean, tiny_std = np.mean(tiny_times)*1000, np.std(tiny_times)*1000
    tiny_gbps = [(m * 1e-9) / t if t > 0 else 0.0 for m, t in zip(tiny_mems, tiny_times)]

    print("\ngreedy decoding:")
    for i in range(len(hf_times)):
      ratio = tiny_times[i]/hf_times[i]
      desc = "faster" if ratio < 1 else "slower"
      print(("\r" if not CI else "")+f"tok {i+1:3d}  {hf_times[i]*1000:7.2f} ms torch, {tiny_times[i]*1000:7.2f} ms tinygrad, {colorize_float(ratio)} {desc}")  # noqa: E501
    avg_ratio = colorize_float(tiny_mean/hf_mean)
    avg_desc = "faster" if tiny_mean < hf_mean else "slower"
    print(f"\naverage: {hf_mean:7.2f}±{hf_std:.2f} ms torch, {tiny_mean:7.2f}±{tiny_std:.2f} ms tinygrad, {avg_ratio} {avg_desc}")
    print(f"tinygrad mem bw: {np.mean(tiny_gbps):.2f}±{np.std(tiny_gbps):.2f} GB/s")

    if not FAKEWEIGHTS:
      self.assertEqual(hf_toks, tiny_toks, f"token mismatch: hf={hf_toks} vs tiny={tiny_toks}")

if __name__ == "__main__":
  unittest.main(verbosity=2)
