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
from transformers import LlamaForCausalLM, LlamaConfig, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
from extra.models.llama import Transformer as TinygradTransformer, convert_from_huggingface
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.nn.state import load_state_dict
from tinygrad.helpers import colorize_float, getenv, CI

TORCHCOMPILE = bool(int(getenv("TORCHCOMPILE", 1)))
CNT = getenv("CNT", 10)
WARMUP = 10

# llama 1B config
LLAMA_CONFIG = {
  'dim': 2048,
  'n_heads': 32,
  'n_kv_heads': 8,
  'n_layers': 16,
  'hidden_dim': 8192,
  'vocab_size': 128256,
  'norm_eps': 1e-5,
  'rope_theta': 500000,
  'max_context': WARMUP + CNT
}

# sampling parameters
TEMPERATURE = getenv("TEMPERATURE", 0.0)
TOP_K = 5
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

def create_hf_config():
  return LlamaConfig(
    vocab_size=LLAMA_CONFIG['vocab_size'],
    hidden_size=LLAMA_CONFIG['dim'],
    intermediate_size=LLAMA_CONFIG['hidden_dim'],
    num_hidden_layers=LLAMA_CONFIG['n_layers'],
    num_attention_heads=LLAMA_CONFIG['n_heads'],
    num_key_value_heads=LLAMA_CONFIG['n_kv_heads'],
    rms_norm_eps=LLAMA_CONFIG['norm_eps'],
    rope_theta=LLAMA_CONFIG['rope_theta'],
    max_position_embeddings=LLAMA_CONFIG['max_context'] * 2,
    use_cache=True,
  )

def benchmark_hf(model, start_tok, warmup=WARMUP, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  device = next(model.parameters()).device
  toks = []

  logits_processor = LogitsProcessorList()
  if TEMPERATURE > 0: logits_processor.append(TemperatureLogitsWarper(TEMPERATURE))
  if TOP_K > 0: logits_processor.append(TopKLogitsWarper(TOP_K))
  if TOP_P > 0: logits_processor.append(TopPLogitsWarper(TOP_P))

  def sample_hf(logits, input_ids):
    if TEMPERATURE < 1e-6: return logits.argmax(dim=-1).item()
    scores = logits_processor(input_ids, logits)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, 1).item()

  with torch.no_grad():
    past_key_values = None
    input_ids = torch.tensor([[start_tok]], device=device)
    for i in range(warmup):
      outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]
      next_tok = sample_hf(logits, input_ids)
      input_ids = torch.tensor([[next_tok]], device=device)

  times = []
  with torch.no_grad():
    for i in range(iters):
      st = time.perf_counter()
      outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]
      next_tok = sample_hf(logits, input_ids)
      times.append(time.perf_counter() - st)
      input_ids = torch.tensor([[next_tok]], device=device)
      toks.append(next_tok)
  return times, toks

def benchmark_tinygrad(model, start_tok, warmup=WARMUP, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  toks = []
  tok_tensor = Tensor([[start_tok]]).realize()
  for i in range(warmup):
    last_tok = model(tok_tensor, i, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
    tok_tensor.assign(Tensor([[last_tok]])).realize()

  times = []
  mems = []
  for i in range(iters):
    GlobalCounters.reset()
    st = time.perf_counter()
    last_tok = model(tok_tensor, warmup + i, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
    elapsed = time.perf_counter() - st
    times.append(elapsed)
    mems.append(getattr(GlobalCounters, "global_mem", 0))
    tok_tensor.assign(Tensor([[last_tok]])).realize()
    toks.append(last_tok)
  return times, toks, mems

def copy_weights_hf_to_tinygrad(hf_model, tiny_model):
  hf_state = {k: Tensor(v.cpu().numpy()) for k, v in hf_model.state_dict().items()}
  tiny_weights = convert_from_huggingface(hf_state, LLAMA_CONFIG['n_layers'], LLAMA_CONFIG['n_heads'], LLAMA_CONFIG['n_kv_heads'])
  load_state_dict(tiny_model, tiny_weights, strict=False)

class BaseLlamaTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print(f"\nTinygrad device: {Device.DEFAULT}")
    print(f"Config: {LLAMA_CONFIG}")
    print(f"Temperature: {TEMPERATURE} ({'sampling' if TEMPERATURE > 0 else 'greedy decoding'})")
    print("Using HuggingFace transformers LlamaForCausalLM")

    hf_config = create_hf_config()
    print("Creating HuggingFace model with random weights...")
    cls.hf_model = LlamaForCausalLM(hf_config)
    cls.hf_model.eval()

    print("Creating tinygrad model...")
    cls.tiny_model = TinygradTransformer(**LLAMA_CONFIG, jit=True)

    print("Copying weights from HuggingFace to tinygrad...")
    copy_weights_hf_to_tinygrad(cls.hf_model, cls.tiny_model)

    cls.start_tok = 1  # starting token for generation

  def reset_kv(self):
    # reset tinygrad kv cache
    for layer in self.tiny_model.layers:
      if hasattr(layer.attention, 'cache_kv'):
        delattr(layer.attention, 'cache_kv')

class TestLlamaBenchmark(BaseLlamaTest):
  @unittest.skipIf(TEMPERATURE > 0, "Skipping correctness test when sampling (TEMPERATURE > 0)")
  def test_correctness(self):
    self.reset_kv()

    # run a short sequence through both models and compare logits
    test_tokens = [1, 100, 200, 300]
    input_ids = torch.tensor([test_tokens])

    with torch.no_grad():
      hf_outputs = self.hf_model(input_ids, use_cache=False)
      hf_logits = hf_outputs.logits[0, -1, :].numpy()

    tiny_logits = self.tiny_model.forward(
      Tensor([test_tokens]).reshape(1, -1), 0, temperature=float('nan'), top_k=0, top_p=0, alpha_f=0, alpha_p=0
    )[0, -1, :].numpy()

    correlation = np.corrcoef(hf_logits.flatten(), tiny_logits.flatten())[0, 1]
    print(f"\nLogits correlation: {correlation:.6f}")

    self.assertGreater(correlation, 0.99, f"Logits correlation too low: {correlation}")

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

    if TEMPERATURE == 0 and hf_toks != tiny_toks:
      print("\nWarning: Generated sequences differ")
      print(f"  HF tokens:      {hf_toks[:10]}...")
      print(f"  tinygrad tokens: {tiny_toks[:10]}...")

    hf_mean, hf_std = np.mean(hf_times)*1000, np.std(hf_times)*1000
    tiny_mean, tiny_std = np.mean(tiny_times)*1000, np.std(tiny_times)*1000
    tiny_gbps = [(m * 1e-9) / t if t > 0 else 0.0 for m, t in zip(tiny_mems, tiny_times)]

    print(f"\n{'sampling' if TEMPERATURE > 0 else 'greedy decoding'}:")
    for i in range(len(hf_times)):
      ratio = tiny_times[i]/hf_times[i]
      desc = "faster" if ratio < 1 else "slower"
      print(("\r" if not CI else "")+f"tok {i+1:3d}  {hf_times[i]*1000:7.2f} ms torch, {tiny_times[i]*1000:7.2f} ms tinygrad, {colorize_float(ratio)} {desc}")  # noqa: E501
    avg_ratio = colorize_float(tiny_mean/hf_mean)
    avg_desc = "faster" if tiny_mean < hf_mean else "slower"
    print(f"\naverage: {hf_mean:7.2f}±{hf_std:.2f} ms torch, {tiny_mean:7.2f}±{tiny_std:.2f} ms tinygrad, {avg_ratio} {avg_desc}")
    print(f"tinygrad mem bw: {np.mean(tiny_gbps):.2f}±{np.std(tiny_gbps):.2f} GB/s")

if __name__ == "__main__":
  unittest.main(verbosity=2)
