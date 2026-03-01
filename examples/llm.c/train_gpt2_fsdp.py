#!/usr/bin/env python3
import argparse, math, time, random
import numpy as np
from dataclasses import dataclass

from tinygrad import Tensor, nn, Device, GlobalCounters, dtypes, TinyJit
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch

@dataclass
class GPTConfig:
  block_size: int = 128
  vocab_size: int = 4096
  n_layer: int = 2
  n_head: int = 4
  n_embd: int = 256
  dropout: float = 0.0

class ShardedLinear:
  def __init__(self, in_dim:int, out_dim:int, devices:tuple[str, ...], bias:bool=True):
    assert out_dim % len(devices) == 0, "out_dim must be divisible by gpus"
    self.devices, self.out_dim = devices, out_dim
    weight_full = Tensor.randn(in_dim, out_dim)
    bias_full = Tensor.zeros(out_dim) if bias else None
    shard = out_dim // len(devices)
    self.w = [weight_full[:, i*shard:(i+1)*shard].to(dev) for i, dev in enumerate(devices)]
    self.b = [bias_full[i*shard:(i+1)*shard].to(dev) for i, dev in enumerate(devices)] if bias else None
    for w in self.w: w.requires_grad = True
    if self.b is not None:
      for b in self.b: b.requires_grad = True

  def gather(self, dev:str) -> tuple[Tensor, Tensor|None]:
    w_full = Tensor.cat(*[w.to(dev) for w in self.w], dim=1)
    b_full = Tensor.cat(*[b.to(dev) for b in self.b], dim=0) if self.b is not None else None
    return w_full, b_full

  def forward(self, x:Tensor, dev:str, gathered:list|None) -> Tensor:
    w_full, b_full = self.gather(dev)
    if gathered is not None:
      gathered.append((self, w_full, b_full))
    return x.dot(w_full) + (b_full if b_full is not None else 0)

  def scatter_grads(self, w_full:Tensor, b_full:Tensor|None):
    shard = w_full.shape[1] // len(self.devices)
    for i, dev in enumerate(self.devices):
      self.w[i].grad = w_full.grad[:, i*shard:(i+1)*shard].to(dev)
      if b_full is not None:
        self.b[i].grad = b_full.grad[i*shard:(i+1)*shard].to(dev)

class MLP:
  def __init__(self, cfg:GPTConfig, devices:tuple[str, ...]):
    self.c_fc = ShardedLinear(cfg.n_embd, 4 * cfg.n_embd, devices)
    self.c_proj = ShardedLinear(4 * cfg.n_embd, cfg.n_embd, devices)

  def __call__(self, x:Tensor, gathered:list, dev:str) -> Tensor:
    x = self.c_fc.forward(x.reshape(-1, x.shape[-1]), dev, gathered).gelu()
    return self.c_proj.forward(x, dev, gathered)

class FSDPCausalSelfAttention:
  def __init__(self, cfg:GPTConfig, devices:tuple[str, ...]):
    self.n_head = cfg.n_head
    self.n_embd = cfg.n_embd
    self.dropout_p = cfg.dropout
    self.c_attn = ShardedLinear(cfg.n_embd, 3 * cfg.n_embd, devices)
    self.c_proj = ShardedLinear(cfg.n_embd, cfg.n_embd, devices)
    self.bias = [Tensor.ones(1, 1, cfg.block_size, cfg.block_size, device=dev).tril() for dev in devices]
    for b in self.bias: b.requires_grad = False

  def __call__(self, x:Tensor, gathered:list, dev_idx:int) -> Tensor:
    B, T, C = x.shape
    dev = self.c_attn.devices[dev_idx]
    qkv = self.c_attn.forward(x.reshape(B*T, C), dev, gathered).reshape(B, T, 3, self.n_head, C // self.n_head)
    q = qkv[:, :, 0].transpose(1, 2)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[dev_idx][:, :, :T, :T] == 0, float("-inf"))
    att = att.softmax().dropout(self.dropout_p)
    y = att @ v
    y = y.transpose(1, 2).view(B, T, C)
    y = self.c_proj.forward(y.reshape(B*T, C), dev, gathered).reshape(B, T, C)
    return y

class FSDPBlock:
  def __init__(self, cfg:GPTConfig, devices:tuple[str, ...]):
    self.dropout_p = cfg.dropout
    self.ln_1 = [nn.LayerNorm(cfg.n_embd) for _ in devices]
    self.ln_2 = [nn.LayerNorm(cfg.n_embd) for _ in devices]
    for i, dev in enumerate(devices):
      self.ln_1[i].weight.to_(dev); self.ln_1[i].bias.to_(dev)
      self.ln_2[i].weight.to_(dev); self.ln_2[i].bias.to_(dev)
    self.attn = FSDPCausalSelfAttention(cfg, devices)
    self.mlp = MLP(cfg, devices)

  def __call__(self, x:Tensor, gathered:list, dev_idx:int) -> Tensor:
    x = x + self.attn(self.ln_1[dev_idx](x), gathered, dev_idx).dropout(self.dropout_p)
    x = x + self.mlp(self.ln_2[dev_idx](x), gathered, self.attn.c_attn.devices[dev_idx]).reshape(x.shape).dropout(self.dropout_p)
    return x

class FSDPGPT:
  def __init__(self, cfg:GPTConfig, devices:tuple[str, ...]):
    self.cfg, self.devices = cfg, devices
    self.wte = [nn.Embedding(cfg.vocab_size, cfg.n_embd) for _ in devices]
    self.wpe = [nn.Embedding(cfg.block_size, cfg.n_embd) for _ in devices]
    for i, dev in enumerate(devices):
      self.wte[i].weight.to_(dev)
      self.wpe[i].weight.to_(dev)
    self.h = [FSDPBlock(cfg, devices) for _ in range(cfg.n_layer)]
    self.ln_f = [nn.LayerNorm(cfg.n_embd) for _ in devices]
    for i, dev in enumerate(devices):
      self.ln_f[i].weight.to_(dev); self.ln_f[i].bias.to_(dev)
    self._replicate_params()

  def _replicate_params(self):
    for i in range(1, len(self.devices)):
      self.wte[i].weight.assign(self.wte[0].weight.to(self.devices[i]))
      self.wpe[i].weight.assign(self.wpe[0].weight.to(self.devices[i]))
      self.ln_f[i].weight.assign(self.ln_f[0].weight.to(self.devices[i]))
      self.ln_f[i].bias.assign(self.ln_f[0].bias.to(self.devices[i]))
    for blk in self.h:
      for i in range(1, len(self.devices)):
        blk.ln_1[i].weight.assign(blk.ln_1[0].weight.to(self.devices[i]))
        blk.ln_1[i].bias.assign(blk.ln_1[0].bias.to(self.devices[i]))
        blk.ln_2[i].weight.assign(blk.ln_2[0].weight.to(self.devices[i]))
        blk.ln_2[i].bias.assign(blk.ln_2[0].bias.to(self.devices[i]))

  def forward_device(self, idx:Tensor, dev_idx:int, gathered:list) -> Tensor:
    b, t = idx.shape
    dev = self.devices[dev_idx]
    pos = Tensor.arange(0, t, device=dev)
    x = self.wte[dev_idx](idx.to(dev)) + self.wpe[dev_idx](pos).reshape(1, t, self.cfg.n_embd)
    for blk in self.h:
      x = blk(x, gathered, dev_idx)
    x = self.ln_f[dev_idx](x)
    logits = x.reshape(b*t, self.cfg.n_embd).dot(self.wte[dev_idx].weight.T)
    return logits.reshape(b, t, self.cfg.vocab_size)

  def reduce_scatter_grads(self, gathered_per_dev:list[list[tuple[ShardedLinear, Tensor, Tensor|None]]]):
    if len(self.devices) < 2: return
    ref = gathered_per_dev[0]
    ndev = len(self.devices)
    for gi, (mod, _, _) in enumerate(ref):
      w_list = []
      b_list = []
      for di in range(ndev):
        mod_d, w_full, b_full = gathered_per_dev[di][gi]
        assert mod_d is mod
        w_list.append(w_full)
        b_list.append(b_full)
      if any(w.grad is None for w in w_list): continue
      g_sum = w_list[0].grad.to(self.devices[0])
      for g in w_list[1:]:
        g_sum = g_sum + g.grad.to(self.devices[0])
      g_avg = g_sum / ndev
      w_ref = w_list[0]
      w_ref.grad = g_avg
      b_ref = b_list[0] if b_list[0] is not None else None
      if b_ref is not None and all(b is not None for b in b_list):
        if any(b.grad is None for b in b_list): continue
        b_sum = b_list[0].grad.to(self.devices[0])
        for b in b_list[1:]:
          b_sum = b_sum + b.grad.to(self.devices[0])
        b_ref.grad = b_sum / ndev
      mod.scatter_grads(w_ref, b_ref)

  def allreduce_replicated_grads(self):
    if len(self.devices) < 2: return
    base = self.devices[0]
    def sync_param(param_list:list[Tensor]):
      grads = [p.grad for p in param_list]
      if any(g is None for g in grads): return
      g_sum = grads[0].to(base)
      for g in grads[1:]:
        g_sum = g_sum + g.to(base)
      g_avg = g_sum / len(grads)
      for i, dev in enumerate(self.devices):
        param_list[i].grad = g_avg.to(dev)
    sync_param([w.weight for w in self.wte])
    sync_param([w.weight for w in self.wpe])
    sync_param([ln.weight for ln in self.ln_f])
    sync_param([ln.bias for ln in self.ln_f])
    for blk in self.h:
      sync_param([ln.weight for ln in blk.ln_1])
      sync_param([ln.bias for ln in blk.ln_1])
      sync_param([ln.weight for ln in blk.ln_2])
      sync_param([ln.bias for ln in blk.ln_2])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpus", type=int, default=2)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--sequence_length", type=int, default=1024)
  parser.add_argument("--num_iterations", type=int, default=10)
  parser.add_argument("--no_jit", action="store_true", help="disable JIT")
  parser.add_argument("--eval_steps", type=int, default=0)
  parser.add_argument("--eval_only", action="store_true")
  parser.add_argument("--sample", action="store_true")
  parser.add_argument("--prompt", type=str, default="")
  parser.add_argument("--max_new_tokens", type=int, default=128)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=0)
  parser.add_argument("--n_layer", type=int, default=12)
  parser.add_argument("--n_head", type=int, default=12)
  parser.add_argument("--n_embd", type=int, default=768)
  parser.add_argument("--vocab_size", type=int, default=50257)
  parser.add_argument("--dropout", type=float, default=0.0)
  parser.add_argument("--load_gpt2", action="store_true")
  parser.add_argument("--preset", type=str, default="gpt2-small", choices=["", "gpt2-small", "gpt2-medium", "gpt2-large"])
  parser.add_argument("--data", type=str, default="tinyshakespeare_bin", help="Path to text data or 'tinyshakespeare'/'tinyshakespeare_bin'")
  parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["char", "gpt2"])
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  args = parser.parse_args()

  if args.preset == "gpt2-small":
    args.n_layer = 12
    args.n_head = 12
    args.n_embd = 768
    args.vocab_size = 50257
  elif args.preset == "gpt2-medium":
    args.n_layer = 24
    args.n_head = 16
    args.n_embd = 1024
    args.vocab_size = 50257
  elif args.preset == "gpt2-large":
    args.n_layer = 36
    args.n_head = 20
    args.n_embd = 1280
    args.vocab_size = 50257

  cfg = GPTConfig(block_size=args.sequence_length, vocab_size=args.vocab_size,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
  base = Device.DEFAULT
  devices = (base,) if args.gpus == 1 else tuple(f"{base}:{i}" for i in range(args.gpus))

  data_tokens = None
  data_is_tensor = False
  encode = None
  decode = None
  loaded_weights = None
  if args.data:
    if args.data == "tinyshakespeare_bin":
      tokens_bin = fetch("https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin")
      with open(tokens_bin, "rb") as f:
        f.seek(0x400)
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
      data_tokens = Tensor(tokens)
      data_is_tensor = True
      args.vocab_size = 50257 if args.preset in {"gpt2-small", "gpt2-medium"} else args.vocab_size
      cfg.vocab_size = args.vocab_size
    elif args.data == "tinyshakespeare":
      path = fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "tiny_shakespeare.txt")
      text = open(path).read()
    else:
      text = open(args.data).read()
    if data_tokens is None:
      if args.tokenizer == "gpt2":
        try:
          import tiktoken
        except Exception as e:
          raise RuntimeError("tiktoken is required for --tokenizer gpt2") from e
        enc = tiktoken.get_encoding("gpt2")
        encode = enc.encode
        decode = enc.decode
        data_tokens = encode(text)
      else:
        chars = sorted(list(set(text)))
        stoi = {ch:i for i,ch in enumerate(chars)}
        itos = {i:ch for ch,i in stoi.items()}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda t: "".join(itos[i] for i in t)
        data_tokens = encode(text)
        args.vocab_size = max(args.vocab_size, len(chars))
        cfg.vocab_size = args.vocab_size

  print(f"FSDP GPT2 custom: devices={devices} batch={args.batch} seq={args.sequence_length} dim={args.n_embd}")
  model = FSDPGPT(cfg, devices)

  if args.load_gpt2:
    if args.preset == "gpt2-large":
      ckpt_url = "https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin"
    elif args.preset == "gpt2-medium":
      ckpt_url = "https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin"
    else:
      ckpt_url = "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
    weights = torch_load(fetch(ckpt_url))
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    fixed = {}
    for k, v in weights.items():
      nk = k[12:] if k.startswith("transformer.") else k
      if nk.endswith(transposed):
        v = v.T.contiguous()
      fixed[nk] = v
    weights = fixed
    loaded_weights = weights

    def assign_replicated(params:list[Tensor], value:Tensor):
      value = value.to(None).contiguous()
      for i, dev in enumerate(devices):
        params[i].assign(value.to(dev).cast(params[i].dtype))

    def assign_sharded(sharded:ShardedLinear, w:Tensor, b:Tensor|None):
      in_dim = sharded.w[0].shape[0]
      out_dim = sharded.w[0].shape[1] * len(sharded.devices)
      w = w.to(None).contiguous()
      if w.shape == (out_dim, in_dim):
        w = w.T.contiguous()
      if w.shape != (in_dim, out_dim):
        raise RuntimeError(f"weight shape mismatch: got {w.shape} expected {(in_dim, out_dim)}")
      shard = w.shape[1] // len(sharded.devices)
      for i, dev in enumerate(sharded.devices):
        sharded.w[i].assign(w[:, i*shard:(i+1)*shard].to(dev))
        if b is not None:
          b = b.to(None).contiguous()
          if b.shape[0] != w.shape[1]:
            raise RuntimeError(f"bias shape mismatch: got {b.shape} expected {(w.shape[1],)}")
          sharded.b[i].assign(b[i*shard:(i+1)*shard].to(dev))

    # embeddings
    wte = weights["wte.weight"]
    wpe = weights["wpe.weight"]
    if cfg.n_embd != wte.shape[1]:
      raise RuntimeError(f"n_embd mismatch: cfg={cfg.n_embd} ckpt={wte.shape[1]}. Check --preset or --n_embd.")
    if cfg.vocab_size != wte.shape[0]:
      raise RuntimeError(f"vocab_size mismatch: cfg={cfg.vocab_size} ckpt={wte.shape[0]}. Check --preset or --vocab_size.")
    if cfg.block_size > wpe.shape[0]:
      raise RuntimeError(f"block_size too large: cfg={cfg.block_size} ckpt={wpe.shape[0]}. Use --sequence_length <= 1024.")
    assign_replicated([w.weight for w in model.wte], wte[:cfg.vocab_size])
    assign_replicated([w.weight for w in model.wpe], wpe[:cfg.block_size])
    assign_replicated([ln.weight for ln in model.ln_f], weights["ln_f.weight"])
    assign_replicated([ln.bias for ln in model.ln_f], weights["ln_f.bias"])
    for i, blk in enumerate(model.h):
      prefix = f"h.{i}."
      assign_replicated([ln.weight for ln in blk.ln_1], weights[prefix+"ln_1.weight"])
      assign_replicated([ln.bias for ln in blk.ln_1], weights[prefix+"ln_1.bias"])
      assign_replicated([ln.weight for ln in blk.ln_2], weights[prefix+"ln_2.weight"])
      assign_replicated([ln.bias for ln in blk.ln_2], weights[prefix+"ln_2.bias"])
      assign_sharded(blk.attn.c_attn, weights[prefix+"attn.c_attn.weight"], weights[prefix+"attn.c_attn.bias"])
      assign_sharded(blk.attn.c_proj, weights[prefix+"attn.c_proj.weight"], weights[prefix+"attn.c_proj.bias"])
      assign_sharded(blk.mlp.c_fc, weights[prefix+"mlp.c_fc.weight"], weights[prefix+"mlp.c_fc.bias"])
      assign_sharded(blk.mlp.c_proj, weights[prefix+"mlp.c_proj.weight"], weights[prefix+"mlp.c_proj.bias"])

  opts = []
  for i, dev in enumerate(devices):
    params = []
    for bi, blk in enumerate(model.h):
      params += [blk.attn.c_attn.w[i], blk.attn.c_attn.b[i], blk.attn.c_proj.w[i], blk.attn.c_proj.b[i]]
      params += [blk.mlp.c_fc.w[i], blk.mlp.c_fc.b[i], blk.mlp.c_proj.w[i], blk.mlp.c_proj.b[i]]
      params += [blk.ln_1[i].weight, blk.ln_1[i].bias, blk.ln_2[i].weight, blk.ln_2[i].bias]
    params += [model.wte[i].weight, model.wpe[i].weight, model.ln_f[i].weight, model.ln_f[i].bias]
    opts.append(nn.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay))

  def get_batch(state_key:str):
    if data_tokens is None:
      x = Tensor.randint(args.batch, args.sequence_length, low=0, high=cfg.vocab_size, dtype=dtypes.int32)
      return x, x.clone()
    if data_is_tensor:
      if not hasattr(main, state_key): setattr(main, state_key, 0)
      i = getattr(main, state_key)
      x = data_tokens[i:i+args.batch*args.sequence_length].view(args.batch, args.sequence_length)
      i += args.batch*args.sequence_length
      if i + args.batch*args.sequence_length + 1 >= len(data_tokens): i = 0
      setattr(main, state_key, i)
      return x, x.clone()
    max_start = len(data_tokens) - args.sequence_length - 1
    starts = [random.randint(0, max_start) for _ in range(args.batch)]
    x = [data_tokens[s:s+args.sequence_length] for s in starts]
    x = Tensor(x, dtype=dtypes.int32)
    return x, x.clone()

  def run_eval():
    Tensor.training = False
    total_loss = 0.0
    for _ in range(args.eval_steps):
      x, y = get_batch("eval_pos")
      logits = model.forward_device(x.to(devices[0]), 0, [])
      loss = logits[:, :-1, :].sparse_categorical_crossentropy(y[:, 1:].to(devices[0]))
      total_loss += loss.item()
    avg_loss = total_loss / max(1, args.eval_steps)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    print(f"eval loss={avg_loss:.4f} ppl={ppl:.4f}")

  def run_sample():
    if args.tokenizer == "gpt2":
      if encode is None:
        try:
          import tiktoken
        except Exception as e:
          raise RuntimeError("tiktoken is required for --tokenizer gpt2") from e
        enc = tiktoken.get_encoding("gpt2")
        local_encode, local_decode = enc.encode, enc.decode
      else:
        local_encode, local_decode = encode, decode
    else:
      if decode is None or encode is None:
        raise RuntimeError("char tokenizer requires --data with text content")
      local_encode, local_decode = encode, decode
    prompt_tokens = local_encode(args.prompt)
    if len(prompt_tokens) == 0:
      prompt_tokens = [0]
    Tensor.training = False
    out = list(prompt_tokens)
    for _ in range(args.max_new_tokens):
      idx = Tensor([out], dtype=dtypes.int32)
      logits = model.forward_device(idx.to(devices[0]), 0, [])
      last = logits[:, -1, :].numpy().astype(np.float64)[0]
      if args.temperature <= 0:
        next_id = int(last.argmax())
      else:
        last /= args.temperature
        if args.top_k > 0:
          top_k = min(args.top_k, last.shape[0])
          cutoff = np.partition(last, -top_k)[-top_k]
          last = np.where(last < cutoff, -np.inf, last)
        probs = np.exp(last - np.max(last))
        probs = probs / probs.sum()
        next_id = int(np.random.choice(len(probs), p=probs))
      out.append(next_id)
    print(local_decode(out))

  def split_batch(x:Tensor, y:Tensor) -> tuple[list[Tensor], list[Tensor]]:
    ndev = len(devices)
    if x.shape[0] % ndev != 0:
      raise RuntimeError(f"batch must be divisible by gpus for FSDP, batch={x.shape[0]} gpus={ndev}")
    per = x.shape[0] // ndev
    xs = [x[i*per:(i+1)*per].to(devices[i]) for i in range(ndev)]
    ys = [y[i*per:(i+1)*per].to(devices[i]) for i in range(ndev)]
    return xs, ys

  def step_fn(x:Tensor, y:Tensor) -> Tensor:
    for opt in opts: opt.zero_grad()
    xs, ys = split_batch(x, y)
    gathered_per_dev = [[] for _ in devices]
    losses = []
    for i, dev in enumerate(devices):
      logits = model.forward_device(xs[i], i, gathered_per_dev[i])
      loss = logits[:, :-1, :].sparse_categorical_crossentropy(ys[i][:, 1:].to(dev))
      loss.backward()
      losses.append(loss)
    model.reduce_scatter_grads(gathered_per_dev)
    model.allreduce_replicated_grads()
    for l in gathered_per_dev: l.clear()
    loss_out = losses[0].to(devices[0])
    for l in losses[1:]:
      loss_out = loss_out + l.to(devices[0])
    loss_out = loss_out / len(losses)
    return loss_out.realize(*[t for opt in opts for t in opt.schedule_step()])

  if not args.eval_only:
    if args.no_jit:
      Tensor.training = True
      step = step_fn
    else:
      step = TinyJit(Tensor.train()(step_fn))
    jit_logged = False
    for step_idx in range(args.num_iterations):
      GlobalCounters.reset()
      t0 = time.perf_counter()
      x, y = get_batch("train_pos")
      loss = step(x.contiguous(), y.contiguous())
      if not args.no_jit and not jit_logged and getattr(step, "captured", None) is not None:
        print("jit: captured")
        jit_logged = True
      for d in devices: Device[d].synchronize()
      dt = time.perf_counter() - t0
      tok_s = int(x.shape[0] * x.shape[1] / dt)
      print(f"iteration {step_idx}, loss: {loss.item():.6f}, time: {1000*dt:.3f}ms, {tok_s} tok/s, kernels={GlobalCounters.kernel_count}")

  if args.eval_steps > 0:
    run_eval()

  if args.sample:
    run_sample()

if __name__ == "__main__":
  main()
