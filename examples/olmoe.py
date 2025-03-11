# https://arxiv.org/pdf/2409.02060
import numpy as np
np.set_printoptions(suppress=True, linewidth=1000)
import functools, collections, json
from tinygrad import Tensor, nn, Device
from tinygrad.helpers import tqdm, CI, Profiling, Timing, fetch, getenv
from extra.models.llama import Transformer, Variable, convert_from_huggingface

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

    # TODO: don't go to CPU here
    choice = g.data().tolist()[0][0]
    top = sorted(enumerate(choice), key=lambda x: -x[1])[:self.activated_experts]
    sel, probs = Tensor([x[0] for x in top]), Tensor([x[1] for x in top])
    #print(sel.numpy(), probs.numpy())

    # run MoE
    x_up_gate = x.dot(self.gate_proj[sel].permute(0,2,1)).silu() * x.dot(self.up_proj[sel].permute(0,2,1))
    x_down = x_up_gate.dot(self.down_proj[sel].permute(0,2,1))
    return (x_down.float() * probs.reshape(self.activated_experts, 1, 1)).sum(axis=0)

# model is bf16, 1.3B active, 6.9B total
# M3 Max is 400 GB/s, so 400/2.6 = ~154 tok/s

def fetch_weights() -> dict[str, Tensor]:
  # TODO: make this lazy so the 3 fetches can happen in parallel
  m1 = Tensor.from_url("https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00001-of-00003.safetensors").to(Device.DEFAULT)
  m2 = Tensor.from_url("https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00002-of-00003.safetensors").to(Device.DEFAULT)
  m3 = Tensor.from_url("https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model-00003-of-00003.safetensors").to(Device.DEFAULT)
  return {**nn.state.safe_load(m1), **nn.state.safe_load(m2), **nn.state.safe_load(m3)}

if __name__ == "__main__":
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
    model = Transformer(n_layers=16, dim=2048, hidden_dim=1024, n_heads=16, norm_eps=1e-5, qk_norm=1e-5, max_context=1024,
                        vocab_size=50304, feed_forward=functools.partial(MixtureFeedForward, 64, 8), jit=False)
    model_state_dict = nn.state.get_state_dict(model)
    del model_state_dict['freqs_cis']

  with Timing("fetch and load weights: "):
    state = fetch_weights()
    nhf_state = convert_from_huggingface(state, model, 16, 16)
    # NOTE: i'm not sure this actually needs float32, it may just change the type of things downstream from it. but doesn't match torch w/o this
    for needs_float32 in ['tok_embeddings.weight']: nhf_state[needs_float32] = nhf_state[needs_float32].float()
    nn.state.load_state_dict(model, nhf_state, verbose=False, strict=False, consume=True, realize=False)
    assert len(nhf_state) == 0

  count = 30
  temperature = 0

  with Timing("load tokenizer: "):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

  toks = [12092]
  start_pos = 0
  for i in range(count):
    tok = model(Tensor([toks[start_pos:]]), 0 if start_pos == 0 else Variable("start_pos", 1, 1024).bind(start_pos), temperature).item()
    toks.append(tok)
    start_pos += 1
    print(toks)
    print(tokenizer.decode(toks))

  # Hello, I am a newbie to this forum and I am trying to get a better understanding of the different types of data that can be stored in a
  assert toks == [12092, 13, 309, 717, 247, 747, 17782, 281, 436, 12209, 285, 309, 717, 2820, 281, 755,
                  247, 1805, 4685, 273, 253, 1027, 3510, 273, 941, 326, 476, 320, 7141, 275, 247], "BAD OUTPUT!"

