# LoRA (Low-Rank Adaptation) for Llama2 70B fine-tuning
# ref: https://arxiv.org/abs/2106.09685
from tinygrad import Tensor, nn

class LoRALinear:
  def __init__(self, inf, outf, r=16, alpha=32.0, bias=False):
    self.linear = nn.Linear(inf, outf, bias=bias)
    self.lora_A, self.lora_B = (nn.Linear(inf, r, bias=False), nn.Linear(r, outf, bias=False)) if r>0 else (None, None)
    if self.lora_B: self.lora_B.weight.assign(Tensor.zeros_like(self.lora_B.weight)) # init B to zero
    self.scale = alpha/r if r>0 else 0
  def __call__(self, x):
    out = self.linear(x)
    return out if self.lora_A is None else out + self.lora_B(self.lora_A(x)) * self.scale

def apply_lora(model, r=16, alpha=32.0, target=None, layers=None):
  target = target or ["wq", "wv", "wk", "wo"]
  for i,layer in enumerate(model.layers):
    if layers is not None and i not in layers: continue
    for name in target:
      if hasattr(layer.attention, name):
        lin = getattr(layer.attention, name)
        inf, outf, bias = lin.weight.shape[1], lin.weight.shape[0], lin.bias is not None
        lora = LoRALinear(inf, outf, r, alpha, bias)
        lora.linear.weight.assign(lin.weight.detach())
        lora.linear.weight.requires_grad = False
        if bias and lin.bias is not None:
          lora.linear.bias.assign(lin.bias.detach())
          lora.linear.bias.requires_grad = False
        setattr(layer.attention, name, lora)

def get_lora_params(model):
  ret = []
  for layer in model.layers:
    for name in ["wq", "wv", "wk", "wo"]:
      if hasattr(layer.attention, name) and isinstance(getattr(layer.attention, name), LoRALinear):
        m = getattr(layer.attention, name)
        if m.lora_A: ret += [m.lora_A.weight, m.lora_B.weight]
  return ret
