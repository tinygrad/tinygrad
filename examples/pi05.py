# pi0.5 (Physical Intelligence): vision-language-action model with flow matching. https://arxiv.org/abs/2504.16054
# runs the lerobot/pi05_base checkpoint: images + task + robot state in, a chunk of 50 robot actions out.
# with --text it instead generates text from the VLM (subtask prediction / VQA).
import argparse
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch
from extra.models.pi05 import Pi05, load_pi05

def load_image(path:str) -> Tensor:
  # resize keeping aspect, pad to 224x224, scale to [-1, 1]
  from PIL import Image
  img = Image.open(path).convert("RGB")
  s = 224 / max(img.size)
  img = img.resize((round(img.size[0]*s), round(img.size[1]*s)))
  canvas = Image.new("RGB", (224, 224))
  canvas.paste(img, ((224-img.size[0])//2, (224-img.size[1])//2))
  x = Tensor(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1)
  return x.cast(dtypes.bfloat16)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="run pi0.5", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--weights", default=None, help="path to model.safetensors (default: fetch lerobot/pi05_base, 14.5GB)")
  parser.add_argument("--image", action="append", default=[], help="input image (repeat for multiple cameras)")
  parser.add_argument("--task", default="pick up the object", help="natural language task instruction")
  parser.add_argument("--state", default=",".join(["0"]*32), help="32 comma-separated state values, quantile-normalized to [-1,1]")
  parser.add_argument("--steps", type=int, default=10, help="flow matching euler steps")
  parser.add_argument("--text", default=None, help="generate text from this prompt instead of sampling actions")
  parser.add_argument("--count", type=int, default=20, help="tokens to generate in --text mode")
  parser.add_argument("--cpu-embedder", action="store_true", help="keep the 1GB embedding table in system ram (for ~6GB gpus)")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  Tensor.manual_seed(args.seed)
  weights = args.weights or fetch("https://huggingface.co/lerobot/pi05_base/resolve/main/model.safetensors", "pi05_base.safetensors")
  tokenizer = fetch("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", "paligemma_tokenizer.model")
  from sentencepiece import SentencePieceProcessor
  spp = SentencePieceProcessor(model_file=str(tokenizer))

  model = Pi05()
  load_pi05(model, str(weights), embedder_device="CPU" if args.cpu_embedder else None)
  images = [load_image(p) for p in args.image]

  if args.text is not None:
    # greedy decoding: the paligemma prefix (images + prompt) is bidirectional, generated tokens are causal
    toks = spp.encode(args.text, add_bos=True)
    x = Tensor.cat(*[model.embed_image(img) for img in images], model.llm.embed(Tensor([toks])).cast(dtypes.bfloat16), dim=1) \
        if images else model.llm.embed(Tensor([toks]))
    n = x.shape[1]
    (out,), cache = model.llm([x, None], Tensor.arange(n).reshape(1, n), Tensor.ones(1, n, n, dtype=dtypes.bool))
    tok, new_toks = model.llm.decode(out[:, -1]).argmax(-1).item(), []
    for i in range(args.count):
      if tok == spp.eos_id(): break
      new_toks.append(tok)
      (out,), cache = model.llm([model.llm.embed(Tensor([[tok]]))], Tensor([[n+i]]), Tensor.ones(1, 1, n+i+1, dtype=dtypes.bool), kv_cache=cache)
      tok = model.llm.decode(out[:, -1]).argmax(-1).item()
    print(spp.decode([t for t in new_toks if t < spp.piece_size()]))
  else:
    # pi05 prompt format: the (normalized) state is discretized into 256 bins and written into the text
    state = np.array([float(v) for v in args.state.split(",")])
    bins = np.digitize(np.clip(state, -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
    toks = spp.encode(f"Task: {args.task}, State: {' '.join(map(str, bins))};\nAction: ", add_bos=True)
    prefix = Tensor.cat(*[model.embed_image(img) for img in images], model.llm.embed(Tensor([toks])).cast(dtypes.bfloat16), dim=1) \
             if images else model.llm.embed(Tensor([toks]))
    actions = model.sample_actions(prefix.realize(), num_steps=args.steps)
    print(f"action chunk {actions.shape}: mean {actions.mean().item():+.4f}, std {actions.std().item():.4f}")
    for row in actions[0, :5].tolist(): print("  " + " ".join(f"{v:+.3f}" for v in row[:8]) + " ...")
