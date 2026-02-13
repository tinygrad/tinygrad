from tinygrad import Tensor


class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

def merge_gate_up_experts(block) -> None:
  """Merge ffn_gate_exps + ffn_up_exps into ffn_gate_up_exps. Halves expert gathers."""
  from tinygrad.nn.quantized import QuantizedExpertWeights
  gate, up = block.ffn_gate_exps, block.ffn_up_exps
  if isinstance(gate, QuantizedExpertWeights) and isinstance(up, QuantizedExpertWeights):
    gate._ensure_expert_blocks(gate.blocks.device)
    up._ensure_expert_blocks(up.blocks.device)
    merged_expert_blocks = gate._expert_blocks.cat(up._expert_blocks, dim=1)
    merged = QuantizedExpertWeights(merged_expert_blocks.flatten(end_dim=1),
                                    (gate.num_experts, gate.out_features + up.out_features, gate.in_features), gate.ggml_type)
    merged._expert_blocks = merged_expert_blocks
  else:
    merged = ExpertWeights(gate.weight.shape[0], gate.weight.shape[2], gate.weight.shape[1] + up.weight.shape[1])
    merged.weight = gate.weight.cat(up.weight, dim=1)
  block.ffn_gate_up_exps = merged
  del block.ffn_gate_exps, block.ffn_up_exps

def merge_gate_up_shared_expert(block) -> None:
  """Merge shared expert ffn_gate_shexp + ffn_up_shexp into ffn_gate_up_shexp."""
  if not hasattr(block, 'ffn_gate_shexp'): return
  from tinygrad.nn.quantized import QuantizedLinear
  gate, up = block.ffn_gate_shexp, block.ffn_up_shexp
  if isinstance(gate, QuantizedLinear) and isinstance(up, QuantizedLinear):
    block.ffn_gate_up_shexp = QuantizedLinear(gate.blocks.cat(up.blocks, dim=0),
                                              (gate.out_features + up.out_features, gate.in_features), gate.ggml_type)
  else:
    gate.weight = gate.weight.cat(up.weight, dim=0)
    block.ffn_gate_up_shexp = gate
  del block.ffn_up_shexp

def finalize_moe_weights(blocks) -> None:
  """Merge gate+up expert weights and cache router weights in float32."""
  for blk in blocks:
    if hasattr(blk, 'ffn_gate_exps'):
      merge_gate_up_experts(blk)
      merge_gate_up_shared_expert(blk)
      blk.ffn_gate_inp_f32 = blk.ffn_gate_inp.weight.float()
