# https://arxiv.org/pdf/2409.02060
import time
import numpy as np
np.set_printoptions(suppress=True, linewidth=1000)
import functools
from tinygrad import Tensor, nn, Device, GlobalCounters, Context
from tinygrad.helpers import Timing, getenv
from extra.models.llama import Transformer, convert_from_huggingface

# def topk(t:Tensor, k:int, dim:int=-1) -> tuple[Tensor, Tensor]:
#   from tinygrad import dtypes
#   counter, counter2 = Tensor.arange(t.numel(), device=t.device), Tensor.arange(t.numel() - 1, -1, -1, device=t.device)
#   output, output_indices = Tensor.zeros(k, device=t.device), Tensor.zeros(k, device=t.device, dtype=dtypes.int32)
#   for i in range(k):
#     t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
#     output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
#     output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
#     t = (counter == t_argmax).where(0, t)
#   return output, output_indices

class MixtureFeedForward:
  def __init__(self, num_experts:int, activated_experts:int, dim:int, hidden_dim:int, linear=nn.Linear):
    self.activated_experts = activated_experts
    self.gate = nn.Linear(dim, num_experts, bias=False)
    self.up_proj = Tensor.zeros(num_experts, hidden_dim, dim, dtype='bfloat16')
    self.down_proj = Tensor.zeros(num_experts, dim, hidden_dim, dtype='bfloat16')
    self.gate_proj = Tensor.zeros(num_experts, hidden_dim, dim, dtype='bfloat16')
  def __call__(self, x:Tensor) -> Tensor:
    assert x.shape[0] == 1, "only BS=1"
    assert x.shape[1] == 1, "only length=1"
    g = self.gate(x).float().softmax(-1)

    g = g.squeeze() # (BS, length, num_experts) -> (num_experts,)
    probs, sel = g.topk(self.activated_experts)

    # print(f"11111 mem_used: {GlobalCounters.global_mem/1e9:.2f} GB")
    with Context(FUSE_ARANGE=1):
      selected_gate_projs = self.gate_proj[sel]
      selected_up_projs = self.up_proj[sel]
      selected_down_projs = self.down_proj[sel]
      selected_gate_projs.realize(selected_up_projs, selected_down_projs)
    # print(f"222222 mem_used: {GlobalCounters.global_mem/1e9:.2f} GB")

    # run MoE
    x_up_gate = x.dot(selected_gate_projs.permute(0,2,1)).silu() * x.dot(selected_up_projs.permute(0,2,1))
    x_down = x_up_gate.dot(selected_down_projs.permute(0,2,1))
    ret = (x_down.float() * probs.reshape(self.activated_experts, 1, 1)).sum(axis=0)
    # print(f"333333 mem_used: {GlobalCounters.global_mem/1e9:.2f} GB")
    return ret

# model is bf16, 1.3B active, 6.9B total
# M3 Max is 400 GB/s, so 400/2.6 = ~154 tok/s

def fetch_weights() -> dict[str, Tensor]:
  # TODO: make this lazy so the 3 fetches can happen in parallel
  m1 = Tensor.from_url("https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00001-of-00003.safetensors").to(Device.DEFAULT)
  m2 = Tensor.from_url("https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00002-of-00003.safetensors").to(Device.DEFAULT)
  m3 = Tensor.from_url("https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00003-of-00003.safetensors").to(Device.DEFAULT)
  return {**nn.state.safe_load(m1), **nn.state.safe_load(m2), **nn.state.safe_load(m3)}

def filter_layers(state, layers):
  lay = [f"model.layers.{i}." for i in range(layers)]
  return {k: v for k, v in state.items() if not k.startswith("model.layers.") or any(k.startswith(l) for l in lay)}

if __name__ == "__main__":
  LAYERS = 2
  if getenv("TORCH"):
    from transformers import OlmoeForCausalLM, AutoTokenizer
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    inputs = tokenizer("Hello", return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    exit(0)

  with Timing("create model: "):
    model = Transformer(n_layers=LAYERS, dim=2048, hidden_dim=1024, n_heads=16, norm_eps=1e-5, qk_norm=1e-5, max_context=1024,
                        vocab_size=50304, feed_forward=functools.partial(MixtureFeedForward, 64, 8))
    model_state_dict = nn.state.get_state_dict(model)
    del model_state_dict['freqs_cis']

  with Timing("load weights to GPU: "):
    nhf_state = convert_from_huggingface(filter_layers(fetch_weights(), LAYERS), model, 16, 16)
    # NOTE: i'm not sure this actually needs float32, it may just change the type of things downstream from it. but doesn't match torch w/o this
    for needs_float32 in ['tok_embeddings.weight']: nhf_state[needs_float32] = nhf_state[needs_float32].float()
  print(f"ram used: {GlobalCounters.mem_used/1e9:.2f} GB")

  with Timing("unpack weights: "):
    nn.state.load_state_dict(model, nhf_state, verbose=False, strict=False, consume=True, realize=False)
    assert len(nhf_state) == 0
    Tensor.realize(*list(nn.state.get_state_dict(model).values()))
  print(f"ram used: {GlobalCounters.mem_used/1e9:.2f} GB")

  count = 30
  temperature = 0

  with Timing("load tokenizer: "):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

  toks = [12092]
  start_pos = 0
  timings = []
  for i in range(count):
    GlobalCounters.reset()
    st = time.perf_counter()
    tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).item()
    timings.append(time.perf_counter()-st)
    toks.append(tok)
    start_pos += 1
    print("AAAAAAAAAAAA")
    print("AAAAAAAAAAAA")
    print("AAAAAAAAAAAA")
    print("AAAAAAAAAAAA")
    print(toks)
    print(tokenizer.decode(toks))
    print(f"global_mem: {GlobalCounters.global_mem/1e9:.2f} GB")
    print(f"global_ops: {GlobalCounters.global_ops:,} ops")
    print(f"mem_used: {GlobalCounters.mem_used/1e9:.2f} GB")
    print(f"kernel_count: {GlobalCounters.kernel_count:,} kernels")
  print(f"fastest token {min(timings)*1e3:.2f} ms, {1/min(timings):.1f} tok/s")

  # if temperature == 0:
  #   # Hello, I am a newbie to this forum and I am trying to get a better understanding of the different types of data that can be stored in a
  #   assert toks == [12092, 13, 309, 717, 247, 747, 17782, 281, 436, 12209, 285, 309, 717, 2820, 281, 755,
  #                   247, 1805, 4685, 273, 253, 1027, 3510, 273, 941, 326, 476, 320, 7141, 275, 247], "BAD OUTPUT!"


# global_mem: 2.80 GB
# global_ops: 3,663,325,624 ops
# mem_used: 16.78 GB

# global_mem: 2.80 GB
# global_ops: 4,810,369,464 ops
# mem_used: 16.78 GB

# global_mem: 2.80 GB
# global_ops: 13,624,553,216 ops
# mem_used: 16.58 GB