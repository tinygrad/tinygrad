"""FLUX.2-klein-4B text->image in tinygrad: generate + warm steady-state benchmark.

Runs on any tinygrad backend (set DEV=METAL, CUDA, ...): attention uses the ThunderMittens
Metal flash kernel on METAL and Tensor.scaled_dot_product_attention on every other backend.

  # generate one image (downloads weights from HuggingFace on first run):
  PYTHONPATH=. DEV=METAL python examples/flux2.py --prompt "a photo of a cat" --size 128

  # known-good ids, no tokenizer dependency:
  PYTHONPATH=. DEV=METAL python examples/flux2.py --input-ids /tmp/flux2_qwen_ref.safetensors

  # warm steady-state benchmark (JITBEAM autotunes the replayed kernels; gen 0 cold, gens 1.. warm):
  PYTHONPATH=. DEV=METAL JITBEAM=2 python examples/flux2.py --input-ids /tmp/flux2_qwen_ref.safetensors --gens 5

Ties together the three validated FLUX.2 pieces (Qwen3 encoder, DiT denoiser, conv VAE).
All three load once with optional int8 (q8) weights (~7GB resident, fits a 16GB box), stay
resident, and every stage is JIT-captured: gen 0 is cold (the JITs capture), gens 1.. are the
warm steady-state. The sampler / latent packing / schedule mirror mflux Flux2Klein.generate_image.
"""
from __future__ import annotations
import argparse, gc, tempfile, time
from pathlib import Path

from tinygrad import Tensor, Device, GlobalCounters, dtypes
from tinygrad.helpers import fetch
from tinygrad.nn import Linear
from tinygrad.nn.state import safe_load
from tinygrad.engine.jit import TinyJit

from extra.models.flux2 import HF_BASE
from extra.models.flux2.text_encoder import load_text_encoder, prepare_text_ids, OUT_LAYERS
from extra.models.flux2 import dit as dit_mod
from extra.models.flux2.dit import load_dit
from extra.models.flux2.vae import load_vae_decoder
from extra.models.flux2.pipeline import (make_schedule, prepare_packed_latents,
                                         unpack_to_decoder_layout, postprocess_image)
# q8: reuse examples/llama3.py's Int8Linear (int8 weight + per-out-channel f16 scale, dequant
# fused into the dot). llama3 quantizes a state_dict by name; here the models are already built,
# so the only addition is an in-place module-tree walk that swaps each bias-free Linear.
from examples.llama3 import Int8Linear

MAX_SEQUENCE_LENGTH = 512        # qwen3 prompt padded to this
TOKENIZER_BASE_URL = HF_BASE + "tokenizer/"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json", "chat_template.jinja")


def quantize_int8(model, scale_dtype=dtypes.float16) -> int:
  """Replace every bias-free nn.Linear reachable from `model` (attrs + list/tuple/dict
  containers) with a populated llama3 Int8Linear, in place. Returns the count quantized;
  the freed f16 weight has no other reference, so it is released on the next gc."""
  return _walk(model, set(), scale_dtype)

def _walk(obj, seen: set, scale_dtype) -> int:
  if id(obj) in seen or isinstance(obj, Tensor): return 0
  if not (hasattr(obj, "__dict__") or isinstance(obj, (list, tuple, dict))): return 0
  seen.add(id(obj))
  items = (list(enumerate(obj)) if isinstance(obj, (list, tuple)) else
           list(obj.items()) if isinstance(obj, dict) else list(vars(obj).items()))
  count = 0
  for key, child in items:
    if isinstance(child, Linear) and child.bias is None:
      out_f, in_f = child.weight.shape
      w = child.weight.cast(scale_dtype)
      scale = w.abs().max(axis=1, keepdim=True) / 127.0                 # (out, 1), per-out-channel
      q = Int8Linear(in_f, out_f)
      q.weight = (w / scale).round().clip(-127, 127).cast(dtypes.int8).realize()  # (out, in)
      q.scale = scale.reshape(-1).cast(scale_dtype).realize()                     # (out,)
      if isinstance(obj, (list, dict)): obj[key] = q
      else: setattr(obj, key, q)
      count += 1
    else:
      count += _walk(child, seen, scale_dtype)
  return count


def tokenize_prompt(prompt: str, tokenizer_dir: Path) -> tuple[Tensor, Tensor]:
  # transformers AutoTokenizer (needs the qwen3 chat template). Mirrors mflux LanguageTokenizer:
  # use_chat_template, enable_thinking=False, padding=max_length(512), truncation, add_special_tokens.
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
  text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
  enc = tok(text, padding="max_length", max_length=MAX_SEQUENCE_LENGTH, truncation=True,
            add_special_tokens=True, return_tensors="np")
  return Tensor(enc["input_ids"]).cast(dtypes.int32), Tensor(enc["attention_mask"]).cast(dtypes.int32)

def fetch_tokenizer_dir() -> Path:
  last = None
  for fn in TOKENIZER_FILES:
    last = fetch(TOKENIZER_BASE_URL + fn, fn, subdir="flux2-klein-4b/tokenizer")
  return last.parent

def load_input_ids(path: str) -> tuple[Tensor, Tensor]:
  sd = safe_load(path)
  return (sd["input_ids"].to(Device.DEFAULT).cast(dtypes.int32),
          sd["attention_mask"].to(Device.DEFAULT).cast(dtypes.int32))


def main():
  p = argparse.ArgumentParser(description="FLUX.2-klein text->image (generate + warm bench)",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument("--prompt", type=str, default="a photo of a cat", help="text prompt")
  p.add_argument("--input-ids", type=str, default=None, help="safetensors w/ input_ids+attention_mask (skips tokenizer)")
  p.add_argument("--steps", type=int, default=4, help="num inference steps (klein default 4)")
  p.add_argument("--size", type=int, default=128, help="square image size (snapped to /16)")
  p.add_argument("--seed", type=int, default=0, help="latent noise seed")
  p.add_argument("--gens", type=int, default=1, help="generations to run (gen 0 cold, rest warm steady-state)")
  p.add_argument("--out", type=str, default=str(Path(tempfile.gettempdir()) / "flux2.png"), help="output PNG")
  p.add_argument("--quantize", type=int, default=8, choices=[8], help="int8-quantize qwen+DiT Linears (fits 16GB)")
  p.add_argument("--flash", action="store_true", help="use the ThunderKittens flash kernel (METAL/CUDA) instead of SDPA")
  args = p.parse_args()
  if args.flash: dit_mod.USE_FLASH = True     # default: scaled_dot_product_attention
  use_flash = dit_mod.USE_FLASH and dit_mod.flash_attention is not None
  print(f"  attention: {'flash' if use_flash else 'sdpa'}", flush=True)

  height = width = 16 * (args.size // 16)              # snap to a multiple of 16 (mflux Config)
  image_seq_len = (height // 16) * (width // 16)
  q8 = args.quantize == 8
  def sync(): Device[Device.DEFAULT].synchronize()

  # ---- load all three models once, keep resident (q8 keeps the ~7GB footprint inside 16GB) ----
  t0 = time.perf_counter()
  text_model = load_text_encoder(n_layers=max(OUT_LAYERS))   # only depth max(OUT_LAYERS)=27 of 36
  if q8: print(f"  q8 qwen: {quantize_int8(text_model)} Linears -> int8", flush=True); gc.collect()
  dit = load_dit()
  if q8: print(f"  q8 DiT: {quantize_int8(dit)} Linears -> int8", flush=True); gc.collect()
  decoder = load_vae_decoder()                               # ~0.2GB conv VAE, not quantized
  sync(); print(f"  loaded qwen+DiT+VAE (q8={q8}) in {time.perf_counter()-t0:.1f}s", flush=True)

  # ---- fixed inputs (prompt + size are constant across gens) ----
  if args.input_ids: input_ids, attention_mask = load_input_ids(args.input_ids)
  else:
    input_ids, attention_mask = tokenize_prompt(args.prompt, fetch_tokenizer_dir())
    input_ids, attention_mask = input_ids.to(Device.DEFAULT), attention_mask.to(Device.DEFAULT)
  txt_ids = prepare_text_ids(input_ids.shape[1])[0]          # (512, 4); DiT wants 2D ids
  _, img_ids, lh, lw = prepare_packed_latents(args.seed, height, width)
  timesteps, sigmas = make_schedule(image_seq_len, args.steps)

  # the qwen output is identical every gen, so the DiT JIT closes over it through a cell that
  # each gen rebinds (so the captured graph reads fresh values).
  enc_cell: dict[str, Tensor] = {}
  jit_qwen = TinyJit(lambda ids, mask: text_model.get_prompt_embeds(ids, mask).realize())
  jit_dit = TinyJit(lambda h, ts: dit(hidden_states=h, encoder_hidden_states=enc_cell["enc"],
                                      timestep=ts, img_ids=img_ids, txt_ids=txt_ids).realize())
  jit_vae = TinyJit(lambda packed: decoder.decode_packed_latents(packed).realize())

  print(f"  size {height}x{width}  steps {args.steps}  gens {args.gens}", flush=True)
  img = None
  for g in range(args.gens):
    GlobalCounters.reset(); sync(); g0 = time.perf_counter()
    # 1. qwen encode.
    enc_cell["enc"] = jit_qwen(input_ids, attention_mask).cast("bfloat16").realize()
    sync(); t_qwen = time.perf_counter()
    # 2. DiT flow-match euler denoise (mflux: latents += (sigma[t+1]-sigma[t]) * model_pred).
    latents, _, _, _ = prepare_packed_latents(args.seed, height, width)
    for t in range(args.steps):
      noise = jit_dit(latents.cast("bfloat16"), Tensor([timesteps[t]]))
      latents = (latents + (sigmas[t + 1] - sigmas[t]) * noise.float()).realize()
    sync(); t_dit = time.perf_counter()
    # 3. VAE decode + post-process.
    decoded = jit_vae(unpack_to_decoder_layout(latents, lh, lw))
    img = postprocess_image(decoded, height, width).realize()
    sync(); t_vae = time.perf_counter()
    tag = "COLD" if g == 0 else "warm"
    print(f"  gen {g} [{tag}]: {(t_vae-g0)*1e3:7.0f} ms  | qwen {(t_qwen-g0)*1e3:6.0f}  "
          f"dit {(t_dit-t_qwen)*1e3:7.0f}  vae {(t_vae-t_dit)*1e3:5.0f} ms  "
          f"(kernels {GlobalCounters.kernel_count})", flush=True)

  try:
    from PIL import Image
    Image.fromarray(img.numpy()).save(args.out)
    print(f"device {Device.DEFAULT}  saved {args.out}", flush=True)
  except ImportError:
    print(f"device {Device.DEFAULT}  done (pip install pillow to save {args.out})", flush=True)


if __name__ == "__main__":
  main()
