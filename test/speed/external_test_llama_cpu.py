import os
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = os.environ["CPU_COUNT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from transformers import LlamaConfig, LlamaForCausalLM
from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp
from tinygrad.apps.llm import Transformer
from tinygrad.helpers import getenv

LLAMA_CONFIG = dict(dim=2048, hidden_dim=8192, n_heads=32, n_kv_heads=8, num_blocks=16, vocab_size=128256,
                    norm_eps=1e-5, rope_theta=500000, head_dim=64, max_context=512)

N_DECODE = getenv("N_DECODE", 15)
WARMUP = getenv("WARMUP", 5)
PREFILL = [1, 2, 3, 4, 5, 6, 7, 8]
BEAM = getenv("BEAM", 0)

def build_torch_model():
  config = LlamaConfig(hidden_size=LLAMA_CONFIG["dim"], intermediate_size=LLAMA_CONFIG["hidden_dim"],
                        num_attention_heads=LLAMA_CONFIG["n_heads"], num_key_value_heads=LLAMA_CONFIG["n_kv_heads"],
                        num_hidden_layers=LLAMA_CONFIG["num_blocks"], vocab_size=LLAMA_CONFIG["vocab_size"],
                        rms_norm_eps=LLAMA_CONFIG["norm_eps"], rope_theta=LLAMA_CONFIG["rope_theta"],
                        max_position_embeddings=LLAMA_CONFIG["max_context"])
  model = LlamaForCausalLM(config).float().eval()
  return torch.compile(model, backend="inductor", mode="max-autotune-no-cudagraphs")

def build_tiny_model():
  model = Transformer(**LLAMA_CONFIG)  # type: ignore[arg-type]
  # realize random weights
  from tinygrad import nn
  Tensor.realize(*nn.state.get_parameters(model))
  return model

def bench_torch(model):
  with torch.no_grad():
    out = model(torch.tensor([PREFILL]), use_cache=True)
    kv = out.past_key_values
    # warmup
    for i in range(WARMUP):
      out = model(torch.tensor([[1]]), past_key_values=kv, use_cache=True)
      kv = out.past_key_values
    # bench
    times = []
    for i in range(N_DECODE):
      st = time.perf_counter()
      out = model(torch.tensor([[1]]), past_key_values=kv, use_cache=True)
      kv = out.past_key_values
      times.append(time.perf_counter() - st)
  return times

def bench_tiny(model):
  v_start_pos = UOp.variable("start_pos", 1, LLAMA_CONFIG["max_context"] - 1)
  # prefill
  model(Tensor([PREFILL], dtype="int32"), start_pos=0)
  Device[Device.DEFAULT].synchronize()
  start_pos = len(PREFILL)
  # warmup
  for i in range(WARMUP):
    model(Tensor([[1]], dtype="int32"), start_pos=v_start_pos.bind(start_pos + i))
    Device[Device.DEFAULT].synchronize()
  # bench
  times = []
  for i in range(N_DECODE):
    Device[Device.DEFAULT].synchronize()
    st = time.perf_counter()
    model(Tensor([[1]], dtype="int32"), start_pos=v_start_pos.bind(start_pos + WARMUP + i))
    Device[Device.DEFAULT].synchronize()
    times.append(time.perf_counter() - st)
  return times

if __name__ == "__main__":
  print("building torch model...")
  torch_model = build_torch_model()
  print("building tinygrad model...")
  tiny_model = build_tiny_model()

  print(f"benchmarking torch ({N_DECODE} decode steps)...")
  torch_times = bench_torch(torch_model)
  print(f"benchmarking tinygrad ({N_DECODE} decode steps)...")
  tiny_times = bench_tiny(tiny_model)

  et_torch = min(torch_times) * 1000
  et_tiny = min(tiny_times) * 1000
  print(f"torch:   {et_torch:.2f} ms (min of {N_DECODE})")
  print(f"tinygrad: {et_tiny:.2f} ms (min of {N_DECODE})")
  print(f"ratio:   {et_tiny/et_torch:.3f}x")

  # with BEAM, tinygrad should be competitive with torch; without BEAM, just log results
  if BEAM:
    threshold = 1.15
    assert et_tiny <= et_torch * threshold, f"tinygrad {et_tiny:.2f}ms is more than {threshold:.2f}x torch {et_torch:.2f}ms (BEAM={BEAM})"
    print(f"PASS: tinygrad within {threshold:.1f}x of torch (BEAM={BEAM})")
  else:
    print(f"INFO: tinygrad/torch ratio = {et_tiny/et_torch:.3f}x (no BEAM, informational only)")
