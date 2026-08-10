from __future__ import annotations
import gc, json, pathlib
from dataclasses import replace
from collections import defaultdict
from typing import Callable
from tinygrad import Tensor, Device, nn
from tinygrad.nn.state import safe_load
from tinygrad.llm.kimi import load_kimi_tokenizer_data
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig

KIMI_K3_TOTAL_SIZE = 1_560_860_324_864
KIMI_K3_TEXT_SIZE = 1_559_945_066_624
KIMI_K3_TP8_BYTES_PER_GPU = 196_781_639_152
KIMI_K3_SHARDS = 96
KIMI_K3_EXPERTS = 896
KIMI_K3_LAYERS = 93
KIMI_K3_FULL_ATTN_LAYERS = (*range(3, KIMI_K3_LAYERS, 4), 92)
KIMI_K3_SSM_LAYERS = tuple(i not in KIMI_K3_FULL_ATTN_LAYERS for i in range(KIMI_K3_LAYERS))

def kimi_k3_config(max_context:int) -> TransformerConfig:
  """Official Kimi K3 text-tower configuration (zero-based full-attention layers)."""
  return TransformerConfig(num_blocks=93, dim=7168, hidden_dim=3072, n_heads=96, n_kv_heads=96, norm_eps=1e-5,
    vocab_size=163840, head_dim=192, rope_theta=10000.0, rope_dim=64, v_head_dim=128, max_context=max_context,
    q_lora_rank=1536, kv_lora_rank=512, num_experts=896, num_experts_per_tok=16, norm_topk_prob=True,
    shared_expert_dim=6144, leading_dense_blocks=1, dense_hidden_dim=33792, routed_scaling_factor=1.0,
    expert_bias=True, expert_mxfp4=True, bf16_activations=True, kda_split_qkv=True,
    ssm=SSMConfig(conv_kernel=4, state_size=128, group_count=96, time_step_rank=96, inner_size=12288, kda=True),
    ssm_layers=KIMI_K3_SSM_LAYERS, shared_expert_gate=False, attn_output_gate=True,
    activation_situ_beta=4.0, activation_situ_linear_beta=25.0, routed_expert_dim=3584, latent_moe_norm=True,
    route_weights_uncorrected=True, attn_res_block_size=12, kda_full_rank_gate=True, kda_gate_lower_bound=-5.0,
    recurrent_prefill_chunked=True, recurrent_prefill_chunk_size=8)

def kimi_k3_smoke_config(max_context:int=4) -> TransformerConfig:
  """Reduced K3 with every architectural feature retained for cheap compile/hardware admission tests."""
  return replace(kimi_k3_config(max_context), num_blocks=2, dim=32, hidden_dim=256, n_heads=8, n_kv_heads=8,
    vocab_size=64, head_dim=8, rope_dim=4, v_head_dim=4, q_lora_rank=16, kv_lora_rank=8, num_experts=512,
    num_experts_per_tok=2, shared_expert_dim=32, dense_hidden_dim=64, routed_expert_dim=32,
    ssm=SSMConfig(4, 4, 8, 8, 32, True), ssm_layers=(True, False), attn_res_block_size=1)

def _shard_kimi_k3(model:Transformer, devices:tuple[str, ...]) -> None:
  """Tensor parallel layout for K3. The official dimensions are divisible by TP8."""
  if len(devices) not in (1, 2, 4, 8): raise ValueError(f"Kimi K3 tensor parallelism requires 1, 2, 4, or 8 devices, got {len(devices)}")
  for name, value in nn.state.get_state_dict(model).items():
    axis = None
    if name in ("token_embd.weight", "output.weight"): axis = 0
    elif ".ffn_gate_exps.weight" in name or ".ffn_up_exps.weight" in name: axis = 1
    elif ".ffn_gate_exps.weight_scale" in name or ".ffn_up_exps.weight_scale" in name: axis = 1
    elif ".ffn_down_exps.weight" in name or ".ffn_down_exps.weight_scale" in name: axis = 2
    elif name.endswith((".ffn_gate.weight", ".ffn_up.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight")): axis = 0
    elif name.endswith((".ffn_down.weight", ".ffn_down_shexp.weight", ".ffn_routed_down.weight", ".ffn_routed_up.weight",
                        ".attn_output.weight", ".ssm_out.weight")): axis = 1
    elif name.endswith((".attn_q_b.weight", ".attn_k_b.weight", ".attn_v_b.weight", ".attn_gate.weight",
                        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".ssm_f_b.weight", ".ssm_g_full.weight", ".ssm_beta.weight")): axis = 0
    elif name.endswith((".ssm_q_conv1d.weight", ".ssm_k_conv1d.weight", ".ssm_v_conv1d.weight", ".ssm_a", ".ssm_dt.bias")): axis = 0
    value.shard_(devices, axis=axis)

def _validate_config(config:dict) -> None:
  text = config.get("text_config", config)
  expected = {"model_type":"kimi_linear", "hidden_size":7168, "num_hidden_layers":93, "num_attention_heads":96,
              "vocab_size":163840, "intermediate_size":33792, "num_experts":896, "num_experts_per_token":16,
              "moe_intermediate_size":3072, "num_shared_experts":2, "q_lora_rank":1536, "kv_lora_rank":512,
              "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128, "routed_expert_hidden_size":3584,
              "attn_res_block_size":12, "hidden_act":"situ", "mla_use_nope":True, "mla_use_output_gate":True,
              "activation_situ_beta":4.0, "activation_situ_linear_beta":25.0, "latent_moe_use_norm":True,
              "moe_renormalize":True, "first_k_dense_replace":1, "num_expert_group":1, "topk_group":1}
  bad = {k:(text.get(k), v) for k,v in expected.items() if text.get(k) != v}
  linear = text.get("linear_attn_config", {})
  linear_expected = {"head_dim":128, "num_heads":96, "short_conv_kernel_size":4, "use_full_rank_gate":True,
                     "gate_lower_bound":-5.0, "full_attn_layers":[i+1 for i in KIMI_K3_FULL_ATTN_LAYERS],
                     "kda_layers":[i+1 for i,x in enumerate(KIMI_K3_SSM_LAYERS) if x]}
  bad.update({f"linear_attn_config.{k}":(linear.get(k), v) for k,v in linear_expected.items() if linear.get(k) != v})
  quant = text.get("quantization_config", {})
  if quant.get("format") != "mxfp4-pack-quantized": bad["quantization_config.format"] = (quant.get("format"), "mxfp4-pack-quantized")
  if bad: raise ValueError(f"not the supported official Kimi K3 checkpoint: {bad}")

def audit_kimi_k3_checkpoint(model_dir:str|pathlib.Path, require_shards:bool=True) -> dict[str, int]:
  """Validate checkpoint metadata only. This never opens weight data and is safe on small hosts."""
  root = pathlib.Path(model_dir)
  _validate_config(json.loads((root / "config.json").read_text()))
  index = json.loads((root / "model.safetensors.index.json").read_text())
  weight_map, total = index.get("weight_map", {}), index.get("metadata", {}).get("total_size")
  language = [k for k in weight_map if k.startswith("language_model.")]
  experts = [k for k in language if ".block_sparse_moe.experts." in k]
  missing_files = {fn for fn in weight_map.values() if not (root / fn).is_file()}
  if total != KIMI_K3_TOTAL_SIZE: raise ValueError(f"unexpected checkpoint size {total}, expected {KIMI_K3_TOTAL_SIZE}")
  if len(set(weight_map.values())) != KIMI_K3_SHARDS: raise ValueError("official Kimi K3 must contain 96 safetensor shards")
  if len(experts) != 92 * KIMI_K3_EXPERTS * 3 * 2: raise ValueError(f"unexpected routed-expert tensor count {len(experts)}")
  if require_shards and missing_files: raise FileNotFoundError(f"missing {len(missing_files)} checkpoint shards, first: {sorted(missing_files)[0]}")
  return {"tensors":len(weight_map), "language_tensors":len(language), "expert_tensors":len(experts),
          "shards":len(set(weight_map.values())), "missing_shards":len(missing_files), "total_size":total}

def _layer_sources(i:int, is_kda:bool) -> dict[str, str]:
  src, dst = f"language_model.model.layers.{i}.", f"blk.{i}."
  out = {
    src+"input_layernorm.weight":dst+"attn_norm.weight", src+"post_attention_layernorm.weight":dst+"ffn_norm.weight",
    src+"self_attention_res_norm.weight":dst+"attn_res_norm.weight", src+"self_attention_res_proj.weight":dst+"attn_res_proj.weight",
    src+"mlp_res_norm.weight":dst+"mlp_res_norm.weight", src+"mlp_res_proj.weight":dst+"mlp_res_proj.weight",
  }
  if is_kda:
    for a,b in (("q_proj","attn_q"),("k_proj","attn_k"),("v_proj","attn_v"),("g_proj","ssm_g_full"),
                ("f_a_proj","ssm_f_a"),("f_b_proj","ssm_f_b"),("b_proj","ssm_beta"),("o_proj","ssm_out")):
      out[src+f"self_attn.{a}.weight"] = dst+b+".weight"
    for a,b in (("q_conv1d","ssm_q_conv1d"),("k_conv1d","ssm_k_conv1d"),("v_conv1d","ssm_v_conv1d")):
      out[src+f"self_attn.{a}.weight"] = dst+b+".weight"
    out[src+"self_attn.o_norm.weight"], out[src+"self_attn.dt_bias"], out[src+"self_attn.A_log"] = \
      dst+"ssm_norm.weight", dst+"ssm_dt.bias", dst+"ssm_a"
  else:
    for a,b in (("q_a_proj","attn_q_a"),("q_a_layernorm","attn_q_a_norm"),("q_b_proj","attn_q_b"),
                ("kv_a_proj_with_mqa","attn_kv_a_mqa"),("kv_a_layernorm","attn_kv_a_norm"),
                ("g_proj","attn_gate"),("o_proj","attn_output")):
      out[src+f"self_attn.{a}.weight"] = dst+b+".weight"
    # kv_b_proj is split into head-wise K and V tensors while loading.
    out[src+"self_attn.kv_b_proj.weight"] = dst+"attn_k_b.weight|"+dst+"attn_v_b.weight"
  if i == 0:
    for a,b in (("gate_proj","ffn_gate"),("up_proj","ffn_up"),("down_proj","ffn_down")): out[src+f"mlp.{a}.weight"] = dst+b+".weight"
  else:
    base = src+"block_sparse_moe."
    out[base+"gate.weight"], out[base+"gate.e_score_correction_bias"] = dst+"ffn_gate_inp.weight", dst+"exp_probs_b.bias"
    for a,b in (("gate_proj","ffn_gate_shexp"),("up_proj","ffn_up_shexp"),("down_proj","ffn_down_shexp"),
                ("routed_expert_down_proj","ffn_routed_down"),("routed_expert_up_proj","ffn_routed_up"),
                ("routed_expert_norm","ffn_routed_norm")):
      out[base+(f"shared_experts.{a}.weight" if a.endswith("_proj") and not a.startswith("routed_") else a+".weight")] = dst+b+".weight"
  return out

def _replace(dst:Tensor, src:Tensor) -> None:
  if dst.shape != src.shape: raise ValueError(f"shape mismatch: expected {dst.shape}, got {src.shape}")
  dst.replace(src if isinstance(src.device, tuple) else src.shard(dst.device, dst.uop.axis) if isinstance(dst.device, tuple) else src.to(dst.device))
  dst.realize()

def _load_nonexperts(root:pathlib.Path, weight_map:dict[str, str], model:Transformer, progress:Callable[[str], None]) -> set[str]:
  model_state, mappings = nn.state.get_state_dict(model), {
    "language_model.model.embed_tokens.weight":"token_embd.weight", "language_model.model.norm.weight":"output_norm.weight",
    "language_model.lm_head.weight":"output.weight", "language_model.model.output_attn_res_norm.weight":"output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight":"output_attn_res_proj.weight"}
  for i,is_kda in enumerate(KIMI_K3_SSM_LAYERS): mappings.update(_layer_sources(i, is_kda))
  by_file:dict[str, list[str]] = defaultdict(list)
  for source in mappings:
    if source not in weight_map: raise ValueError(f"missing Kimi K3 tensor {source}")
    by_file[weight_map[source]].append(source)
  consumed:set[str] = set()
  for filename, sources in sorted(by_file.items()):
    progress(f"loading non-expert tensors from {filename}")
    shard = safe_load(root / filename)
    for source in sources:
      value, targets = shard[source], mappings[source].split("|")
      if source.endswith("A_log"): value = -value.float().exp().reshape(96, 1)
      if source.endswith("conv1d.weight"): value = value.squeeze(1)
      if source.endswith("kv_b_proj.weight"):
        value = value.reshape(96, 256, 512)
        values:tuple[Tensor, ...] = (value[:, :128].transpose(1, 2), value[:, 128:])
      else: values = (value,)
      for target,tensor in zip(targets, values): _replace(model_state[target], tensor)
      consumed.add(source)
    del shard
    gc.collect()
  return consumed

def _load_experts(root:pathlib.Path, weight_map:dict[str, str], model:Transformer, progress:Callable[[str], None]) -> set[str]:
  model_state, consumed = nn.state.get_state_dict(model), set()
  for i in range(1, KIMI_K3_LAYERS):
    for wid,dst_name in (("w1","ffn_gate_exps"),("w3","ffn_up_exps"),("w2","ffn_down_exps")):
      base = f"language_model.model.layers.{i}.block_sparse_moe.experts"
      packed_keys = [f"{base}.{e}.{wid}.weight_packed" for e in range(KIMI_K3_EXPERTS)]
      scale_keys = [f"{base}.{e}.{wid}.weight_scale" for e in range(KIMI_K3_EXPERTS)]
      files = sorted({weight_map[k] for k in packed_keys+scale_keys})
      progress(f"loading layer {i}/92 {wid} routed experts from {', '.join(files)}")
      shards = {fn:safe_load(root / fn) for fn in files}
      _replace(model_state[f"blk.{i}.{dst_name}.weight"], Tensor.stack(*(shards[weight_map[k]][k] for k in packed_keys)))
      _replace(model_state[f"blk.{i}.{dst_name}.weight_scale"], Tensor.stack(*(shards[weight_map[k]][k] for k in scale_keys)))
      consumed.update(packed_keys+scale_keys)
      del shards
      gc.collect()
  return consumed

def load_kimi_k3(model_dir:str|pathlib.Path, max_context:int=4096, devices:int=8,
                 progress:Callable[[str], None]=print) -> Transformer:
  """Load the official native K3 checkpoint without ever materializing it in host RAM.

  Safetensor shards remain disk-backed, each destination is TP-sharded before transfer, and source
  mappings are discarded after every file/projection. Vision tensors are intentionally ignored.
  """
  root = pathlib.Path(model_dir)
  _validate_config(json.loads((root / "config.json").read_text()))
  index = json.loads((root / "model.safetensors.index.json").read_text())
  weight_map = index["weight_map"]
  if devices != 8: raise ValueError("official Kimi K3 currently requires --devices 8")
  if index.get("metadata", {}).get("total_size") != KIMI_K3_TOTAL_SIZE or len(set(weight_map.values())) != KIMI_K3_SHARDS:
    raise ValueError("checkpoint index does not match the official 96-shard Kimi K3 release")
  missing_files = {fn for fn in weight_map.values() if not (root / fn).is_file()}
  if missing_files: raise FileNotFoundError(f"missing {len(missing_files)} checkpoint shards, first: {sorted(missing_files)[0]}")
  model = Transformer(kimi_k3_config(max_context))
  _shard_kimi_k3(model, tuple(f"{Device.DEFAULT}:{i}" for i in range(devices)))
  consumed = _load_nonexperts(root, weight_map, model, progress)
  consumed.update(_load_experts(root, weight_map, model, progress))
  unused_language = {k for k in weight_map if k.startswith("language_model.")} - consumed
  if unused_language: raise ValueError(f"unmapped language tensors: {sorted(unused_language)[:20]}")
  return model

__all__ = ["KIMI_K3_FULL_ATTN_LAYERS", "KIMI_K3_SSM_LAYERS", "KIMI_K3_TEXT_SIZE", "KIMI_K3_TP8_BYTES_PER_GPU",
           "audit_kimi_k3_checkpoint", "kimi_k3_config", "kimi_k3_smoke_config", "load_kimi_k3", "load_kimi_tokenizer_data"]
