from typing import Union, Optional
from tqdm import tqdm
from tinygrad import Tensor, Variable, nn, Device, dtypes
from tinygrad.nn.state import torch_load, get_state_dict
from extra.models.llama import Attention, FeedForward, RMSNorm, Transformer

def fake_bfloat16_to_float16(x, device):
  return x.to("CLANG").contiguous().to(device).cast(dtypes.uint32).mul(1<<16).contiguous().bitcast(dtypes.float32).half()

class MixtureFeedForward:
  def __init__(self, num_experts:int, dim:int, hidden_dim:int, linear=nn.Linear):
    self.gate = nn.Linear(dim, num_experts, bias=False)
    self.experts = [FeedForward(dim, hidden_dim, linear) for _ in range(num_experts)]
  def __call__(self, x:Tensor) -> Tensor:
    assert x.shape[0] == 1, "only BS=1"
    g = self.gate(x).exp()
    choice = g.data().tolist()[0][0]
    top = sorted(enumerate(choice), key=lambda x: -x[1])
    norm = top[0][1] + top[1][1]
    e1 = self.experts[top[0][0]]
    e2 = self.experts[top[1][0]]
    e1_dev = e1.w1.weight.device
    e2_dev = e2.w1.weight.device
    #print(top[0][1]/norm, top[1][1]/norm)
    ret = e1(x.to(e1_dev)).to(x.device) * (top[0][1]/norm) + e2(x.to(e2_dev)).to(x.device) * (top[1][1]/norm)
    return ret

class MixtureTransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, max_context:int, linear=nn.Linear):
    self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)
    self.feed_forward = MixtureFeedForward(8, dim, hidden_dim, linear)

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Optional[Tensor]):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    return (h + self.feed_forward(self.ffn_norm(h))).realize()

if __name__ == "__main__":
  state = torch_load("/home/tiny/Downloads/mixtral-8x7b-32kseqlen/consolidated.00.pth.b")
  model = Transformer(n_layers=32, dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8, norm_eps=1e-5, vocab_size=32000, tf_block=MixtureTransformerBlock, jit=False)
  model_state_dict = get_state_dict(model)

  for k in (t := tqdm(state)):
    t.set_description(f"loading {k}")
    if 'feed_forward.experts.' in k:
      expert_no = int(k.split('feed_forward.experts.')[1].split('.')[0])
      device = Device.DEFAULT + ":" + str((expert_no//2)+1)
    else:
      device = Device.DEFAULT
    #print(k, state[k].shape, device)
    model_state_dict[k].assign(fake_bfloat16_to_float16(state[k], device)).realize()

  #for k,v in get_state_dict(model).items():
  #  if k in {'freqs_cis'}: continue
  #  v.assign(fake_bfloat16_to_float16(state[k]))

  from sentencepiece import SentencePieceProcessor
  spp = SentencePieceProcessor(model_file="/home/tiny/Downloads/mixtral-8x7b-32kseqlen/tokenizer.model")

  toks = [spp.bos_id()] # + spp.encode("hello")
  #print(toks)
  start_pos = 0
  temperature = 0 #.7

  for i in range(30):
    tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).multinomial().item()
    toks.append(tok)
    start_pos += 1
    print(spp.decode(toks))
