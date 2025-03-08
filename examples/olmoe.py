# https://arxiv.org/pdf/2409.02060
import numpy as np
np.set_printoptions(suppress=True, linewidth=1000)
import functools, collections, json
from tinygrad import Tensor, nn, Device
from tinygrad.helpers import tqdm, CI, Profiling, Timing, fetch, getenv
from extra.models.llama import Transformer, Variable

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

    # TODO: should we renormalize the probs here? looks like it's norm_topk_prob, which is False
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
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924") #.to("mps")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    print(inputs)
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(generate_ids)
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    exit(0)

  #from transformers import PreTrainedTokenizerFast
  #tokenizer = json.loads(fetch("https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct/resolve/main/tokenizer.json").read_text())
  #print(tokenizer.keys())

  with Timing():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

  with Timing():
    model = Transformer(n_layers=16, dim=2048, hidden_dim=1024, n_heads=16, norm_eps=1e-5, qk_norm=1e-5, max_context=1024,
                        vocab_size=50304, feed_forward=functools.partial(MixtureFeedForward, 64, 8), jit=False)
  model_state_dict = nn.state.get_state_dict(model)
  del model_state_dict['freqs_cis']

  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

  experts = collections.defaultdict(dict)
  state = fetch_weights()
  for k in (t := tqdm(state, disable=CI)):
    rk = k.replace("model.", "")
    rk = rk.replace("mlp.gate.weight", "feed_forward.gate.weight")
    rk = rk.replace("input_layernorm.weight", "attention_norm.weight")
    rk = rk.replace("post_attention_layernorm.weight", "ffn_norm.weight")
    rk = rk.replace("self_attn.k_norm.weight", "attention.k_norm.weight")
    rk = rk.replace("self_attn.q_norm.weight", "attention.q_norm.weight")
    rk = rk.replace("self_attn.q_proj.weight", "attention.wq.weight")
    rk = rk.replace("self_attn.k_proj.weight", "attention.wk.weight")
    rk = rk.replace("self_attn.v_proj.weight", "attention.wv.weight")
    rk = rk.replace("self_attn.o_proj.weight", "attention.wo.weight")
    rk = rk.replace("embed_tokens.weight", "tok_embeddings.weight")
    rk = rk.replace("lm_head.weight", "output.weight") # is this right?
    if '.mlp.experts.' not in k:
      #print(k, rk, state[k].shape, model_state_dict[rk].shape)
      assert state[k].shape == model_state_dict[rk].shape
      v = state[k].float()
      n_heads = 16
      if 'q_proj' in k or 'k_proj' in k:
        v = permute(v, n_heads)
      elif 'q_norm' in k or 'k_norm' in k:
        v = v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, 1).transpose(1, 2).flatten()
      model_state_dict[rk].replace(v)
      del model_state_dict[rk]
    else:
      _, _, layer, _, _, expert, name, _ = k.split('.')
      # TODO: this is broken. it never updates the base tensor if you do this, it's updating a copy
      experts[f'layers.{layer}.feed_forward.{name}'][int(expert)] = state[k]

  for k,v in experts.items():
    assert len(v) == 64
    model_state_dict[k].replace(Tensor.stack(*[v[i] for i in range(len(v))]))
    del model_state_dict[k]

  assert len(model_state_dict) == 0, model_state_dict
  del state

  count = 30
  temperature = 0

  #toks = tokenizer.encode("what")# [tokenizer.bos_token_id]
  #print(toks)
  #toks = [1]
  #toks = [tokenizer.bos_token_id]
  toks = [12092]
  start_pos = 0
  for i in range(count):
    tok = model(Tensor([toks[start_pos:]]), 0 if start_pos == 0 else Variable("start_pos", 1, 1024).bind(start_pos), temperature).item()
    #tok = model(Tensor([toks]), 0, temperature).flatten()[-1].item()
    toks.append(tok)
    start_pos += 1
    print(toks)
    print(tokenizer.decode(toks))

