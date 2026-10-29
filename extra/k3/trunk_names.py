#!/usr/bin/env python3
# rewrite a kimi-k3 trunk gguf from the atomicchat llama.cpp converter into the naming tinygrad/llm/model.py loads (unsloth's):
# ffn_latent_* -> ffn_routed_*, kda attn_gate -> ssm_g, ssm_a = -exp(A_log), *_res_norm * *_res_proj -> *_res_score, renamed/added keys.
# python3 trunk_names.py in.gguf out.gguf [gguf-py path]
import re, sys
import numpy as np
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else "/home/nimlgen/k3q/atomic-llama/gguf-py")
import gguf

r = gguf.GGUFReader(sys.argv[1])
w = gguf.GGUFWriter(sys.argv[2], "kimi-k3")
A = "kimi-k3."
keys = {A + "moe_latent_size": A + "expert_latent_length", A + "attn_res_block_size": A + "attn_res.block_size",
        A + "situ_beta": A + "activation.situ_beta", A + "situ_linear_beta": A + "activation.situ_linear_beta"}
for f in r.fields.values():
  if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
  w.add_key_value(keys.get(f.name, f.name), f.contents(), f.types[0], sub_type=f.types[-1] if f.types[0] == gguf.GGUFValueType.ARRAY else None)
w.add_bool(A + "expert_weights_norm", True)  # the router renormalizes the selected weights

t = {x.name: x for x in r.tensors}
kda = {int(n.split(".")[1]) for n in t if n.endswith(".ssm_a")}
out: list[tuple[str, np.ndarray, gguf.GGMLQuantizationType]] = []
for name, x in t.items():
  if re.search(r"_res_proj\.weight$", name): continue
  data, typ = x.data, x.tensor_type
  if (m:=re.match(r"(.*_res)_norm\.weight$", name)):
    proj = np.asarray(t[f"{m.group(1)}_proj.weight"].data, np.float32).reshape(data.shape)
    name, data = f"{m.group(1)}_score.weight", np.asarray(data, np.float32) * proj
  name = name.replace("ffn_latent_", "ffn_routed_")
  if name.endswith(".attn_gate.weight") and int(name.split(".")[1]) in kda: name = name.replace(".attn_gate.", ".ssm_g.")
  if name.endswith(".ssm_a"): data = -np.asarray(data, np.float32)
  out.append((name, np.ascontiguousarray(data), typ))
for name, data, typ in out: w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=typ)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for _, data, _ in out: w.write_tensor_data(data)
w.close()
print(f"wrote {len(out)} tensors to {sys.argv[2]}")
