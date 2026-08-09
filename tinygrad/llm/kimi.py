from __future__ import annotations
import base64, gc, json, pathlib, shutil
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.nn.state import safe_load, safe_save
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.quant import quantize_mxfp4_cpu

KIMI_SSM_LAYERS = tuple(i not in (3, 7, 11, 15, 19, 23, 26) for i in range(27))
KIMI_TENSOR_COUNT, KIMI_LOGICAL_BYTES = 688, 29_051_930_368
KIMI_CHECKPOINT_FORMAT = "tinygrad-kimi-mxfp4-v2"

def kimi_config(max_context:int, expert_mxfp4:bool=True) -> TransformerConfig:
  return TransformerConfig(num_blocks=27, dim=2304, hidden_dim=1024, n_heads=32, n_kv_heads=32, norm_eps=1e-5,
    vocab_size=163840, head_dim=192, rope_theta=10000.0, rope_dim=64, v_head_dim=128, q_lora_rank=0, kv_lora_rank=512,
    num_experts=256, num_experts_per_tok=8, norm_topk_prob=True, shared_expert_dim=1024, leading_dense_blocks=1,
    dense_hidden_dim=9216, routed_scaling_factor=2.446, expert_bias=True, max_context=max_context, expert_mxfp4=expert_mxfp4,
    shared_expert_gate=False, bf16_activations=True, kda_split_qkv=True,
    ssm=SSMConfig(conv_kernel=4, state_size=128, group_count=32, time_step_rank=32, inner_size=4096, kda=True),
    ssm_layers=KIMI_SSM_LAYERS)

def _shard_kimi(model:Transformer, devices:tuple[str, ...]) -> None:
  """Tensor-parallel layout. Every GPU owns a slice of every expert (not a replicated expert set)."""
  for name, value in nn.state.get_state_dict(model).items():
    axis = None
    if name in ("token_embd.weight", "output.weight"): axis = 0
    elif ".ffn_gate_exps.weight" in name or ".ffn_up_exps.weight" in name: axis = 1
    elif ".ffn_gate_exps.weight_scale" in name or ".ffn_up_exps.weight_scale" in name: axis = 1
    elif ".ffn_down_exps.weight" in name or ".ffn_down_exps.weight_scale" in name: axis = 2
    elif name.endswith((".ffn_gate.weight", ".ffn_up.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight")): axis = 0
    elif name.endswith((".ffn_down.weight", ".ffn_down_shexp.weight", ".attn_output.weight", ".ssm_out.weight")): axis = 1
    elif name.endswith((".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_qkv.weight",
                        ".ssm_f_b.weight", ".ssm_g_b.weight", ".ssm_beta.weight")): axis = 0
    elif name.endswith((".ssm_conv1d.weight", ".ssm_q_conv1d.weight", ".ssm_k_conv1d.weight", ".ssm_v_conv1d.weight")): axis = 0
    elif name.endswith((".ssm_a", ".ssm_dt.bias")): axis = 0
    elif name.endswith((".attn_k_b.weight", ".attn_v_b.weight")): axis = 0
    value.shard_(devices, axis=axis)

def _validate_kimi_state(model:Transformer, state:dict[str, Tensor]) -> None:
  model_state = nn.state.get_state_dict(model)
  missing, unexpected = set(model_state)-set(state), set(state)-set(model_state)
  if missing or unexpected: raise ValueError(f"invalid Kimi tensor names: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
  for name, value in state.items():
    if value.shape != model_state[name].shape: raise ValueError(f"invalid shape for {name}: expected {model_state[name].shape}, got {value.shape}")
    expected_dtype = dtypes.uint8 if name.endswith((".weight_scale", "_exps.weight")) else dtypes.bfloat16
    if value.dtype != expected_dtype: raise ValueError(f"invalid dtype for {name}: expected {expected_dtype}, got {value.dtype}")
  if len(state) != KIMI_TENSOR_COUNT or (nbytes := sum(x.nbytes() for x in state.values())) != KIMI_LOGICAL_BYTES:
    raise ValueError(f"invalid Kimi checkpoint size: {len(state)} tensors, {nbytes} bytes")

def _load_converted_state(model_dir:pathlib.Path, files:list[str]) -> dict[str, Tensor]:
  state:dict[str, Tensor] = {}
  for filename in files:
    part = safe_load(model_dir / filename)
    if duplicates := set(state) & set(part): raise ValueError(f"duplicate Kimi tensors in {filename}: {sorted(duplicates)}")
    state.update(part)
  return state

def load_kimi(model_dir:str|pathlib.Path, max_context:int=4096, devices:int=4) -> Transformer:
  model_dir = pathlib.Path(model_dir)
  manifest = json.loads((model_dir / "tinygrad-kimi.json").read_text())
  if manifest.get("format") != KIMI_CHECKPOINT_FORMAT: raise ValueError("unsupported Kimi checkpoint format")
  if devices != 4: raise ValueError("Kimi-Linear MXFP4 checkpoint currently requires TP4 (--devices 4)")
  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(devices))
  model = Transformer(kimi_config(max_context, expert_mxfp4=True))
  state = _load_converted_state(model_dir, manifest["files"])
  _validate_kimi_state(model, state)
  _shard_kimi(model, devs)
  nn.state.load_state_dict(model, state, strict=True, consume=True, realize=True)
  if state: raise ValueError(f"unexpected Kimi tensors: {sorted(state)}")
  return model

def _load_hf_state(src:pathlib.Path) -> dict[str, Tensor]:
  index = json.loads((src / "model.safetensors.index.json").read_text())
  state:dict[str, Tensor] = {}
  for filename in sorted(set(index["weight_map"].values())): state.update(safe_load(src / filename))
  return state

def _layer_key(i:int, suffix:str) -> str: return f"model.layers.{i}.{suffix}"

def _convert_attention(sd:dict[str, Tensor], i:int, is_kda:bool, consume:bool=False) -> dict[str, Tensor]:
  p, out = f"blk.{i}.", {}
  def get(suffix:str) -> Tensor:
    key = _layer_key(i, suffix)
    return (sd.pop(key) if consume else sd[key]).to("CPU")
  if is_kda:
    for src_name, dst_name in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v")):
      out[p+dst_name+".weight"] = get(f"self_attn.{src_name}.weight")
    for src_name, dst_name in (("q_conv1d", "ssm_q_conv1d"), ("k_conv1d", "ssm_k_conv1d"), ("v_conv1d", "ssm_v_conv1d")):
      out[p+dst_name+".weight"] = get(f"self_attn.{src_name}.weight").squeeze(1)
    for src_name, dst_name in (("f_a_proj", "ssm_f_a"), ("f_b_proj", "ssm_f_b"), ("g_a_proj", "ssm_g_a"),
                               ("g_b_proj", "ssm_g_b"), ("b_proj", "ssm_beta"), ("o_proj", "ssm_out")):
      out[p+dst_name+".weight"] = get(f"self_attn.{src_name}.weight")
    out[p+"ssm_norm.weight"] = get("self_attn.o_norm.weight")
    out[p+"ssm_dt.bias"] = get("self_attn.dt_bias")
    out[p+"ssm_a"] = (-get("self_attn.A_log").float().exp()).reshape(32, 1)
  else:
    out[p+"attn_q.weight"] = get("self_attn.q_proj.weight")
    out[p+"attn_kv_a_mqa.weight"] = get("self_attn.kv_a_proj_with_mqa.weight")
    out[p+"attn_kv_a_norm.weight"] = get("self_attn.kv_a_layernorm.weight")
    kv_b = get("self_attn.kv_b_proj.weight").reshape(32, 256, 512)
    k_b, v_b = kv_b[:, :128], kv_b[:, 128:]
    out[p+"attn_k_b.weight"], out[p+"attn_v_b.weight"] = k_b.transpose(1, 2), v_b
    out[p+"attn_output.weight"] = get("self_attn.o_proj.weight")
  return out

def convert_kimi(src_dir:str|pathlib.Path, dst_dir:str|pathlib.Path) -> None:
  """Stream the official BF16 checkpoint into the tinygrad TP4 MXFP4/BF16 representation."""
  src, dst = pathlib.Path(src_dir), pathlib.Path(dst_dir)
  dst.mkdir(parents=True, exist_ok=True)
  config = json.loads((src / "config.json").read_text())
  expected = {"hidden_size":2304, "num_hidden_layers":27, "num_attention_heads":32, "num_key_value_heads":32,
              "vocab_size":163840, "intermediate_size":9216, "num_experts":256, "num_experts_per_token":8,
              "moe_intermediate_size":1024, "num_shared_experts":1, "qk_nope_head_dim":128, "qk_rope_head_dim":64,
              "v_head_dim":128, "kv_lora_rank":512, "first_k_dense_replace":1, "mla_use_nope":True}
  if any(config.get(k) != v for k,v in expected.items()): raise ValueError(f"not Kimi-Linear-48B-A3B: expected {expected}")
  sd, files = _load_hf_state(src), []

  common = {"token_embd.weight":sd.pop("model.embed_tokens.weight").to("CPU"), "output_norm.weight":sd.pop("model.norm.weight").to("CPU"),
            "output.weight":sd.pop("lm_head.weight").to("CPU")}
  safe_save(common, str(dst / "model-common.safetensors"))
  del common
  gc.collect()
  files.append("model-common.safetensors")
  for i in range(27):
    p = f"blk.{i}."
    layer = _convert_attention(sd, i, KIMI_SSM_LAYERS[i], consume=True)
    layer[p+"attn_norm.weight"] = sd.pop(_layer_key(i, "input_layernorm.weight")).to("CPU")
    layer[p+"ffn_norm.weight"] = sd.pop(_layer_key(i, "post_attention_layernorm.weight")).to("CPU")
    if i == 0:
      for src_name, dst_name in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
        layer[p+dst_name+".weight"] = sd.pop(_layer_key(i, f"mlp.{src_name}.weight")).to("CPU")
    else:
      base = _layer_key(i, "block_sparse_moe")
      layer[p+"ffn_gate_inp.weight"] = sd.pop(base+".gate.weight").to("CPU")
      # The official name is e_score_correction_bias; tolerate the early checkpoint spelling.
      bias_name = next(k for k in (base+".gate.e_score_correction_bias", base+".gate.e_score_correction") if k in sd)
      layer[p+"exp_probs_b.bias"] = sd.pop(bias_name).to("CPU")
      for src_name, dst_name in (("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")):
        layer[p+dst_name+".weight"] = sd.pop(base+f".shared_experts.{src_name}.weight").to("CPU")
    layer_file = f"model-layer-{i:02d}.safetensors"
    safe_save({k:v.cast(dtypes.bfloat16).contiguous() for k,v in layer.items()}, str(dst/layer_file))
    del layer
    gc.collect()
    files.append(layer_file)

    if i:
      base = _layer_key(i, "block_sparse_moe.experts")
      for wid, dst_name in (("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")):
        packed, scales = [], []
        for e in range(256):
          q, s = quantize_mxfp4_cpu(sd.pop(f"{base}.{e}.{wid}.weight").to("CPU"))
          packed.append(q)
          scales.append(s)
        expert_file = f"model-layer-{i:02d}-{wid}-mxfp4.safetensors"
        safe_save({p+dst_name+".weight":Tensor.stack(*packed), p+dst_name+".weight_scale":Tensor.stack(*scales)}, str(dst/expert_file))
        del packed, scales, q, s
        gc.collect()
        files.append(expert_file)

  if sd: raise ValueError(f"unconverted Kimi source tensors: {sorted(sd)}")
  for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json", "tiktoken.model", "chat_template.jinja"):
    if (src/name).exists(): shutil.copy2(src/name, dst/name)
  converted = _load_converted_state(dst, files)
  _validate_kimi_state(Transformer(kimi_config(max_context=1)), converted)
  manifest = {"format":KIMI_CHECKPOINT_FORMAT, "tensor_count":KIMI_TENSOR_COUNT,
              "logical_bytes":KIMI_LOGICAL_BYTES, "files":files}
  (dst / "tinygrad-kimi.json").write_text(json.dumps(manifest, indent=2)+"\n")

def load_kimi_tokenizer_data(model_dir:str|pathlib.Path) -> tuple[dict[str, int], dict[str, int], int, int]:
  """Return byte-encoded normal tokens and specials for SimpleTokenizer without transformers/tiktoken."""
  model_dir = pathlib.Path(model_dir)
  normal:dict[str, int] = {}
  bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
  byte_encoder = {b:chr(b) for b in bs} | {b:chr(256+i) for i,b in enumerate(b for b in range(256) if b not in bs)}
  for line in (model_dir / "tiktoken.model").read_bytes().splitlines():
    token, rank = line.split()
    normal["".join(byte_encoder[b] for b in base64.b64decode(token))] = int(rank)
  tc = json.loads((model_dir / "tokenizer_config.json").read_text())
  specials = {v["content"]:int(k) for k,v in tc.get("added_tokens_decoder", {}).items()}
  cfg = json.loads((model_dir / "config.json").read_text())
  return normal, specials, cfg["bos_token_id"], cfg["eos_token_id"]
