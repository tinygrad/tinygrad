#!/usr/bin/env python3
import os, numpy as np
from tinygrad import Tensor, nn, Device, TinyJit, GlobalCounters, fetch
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.distributed import fsdp
from extra.models.transformer import Transformer

MODEL_CONFIG = {
  "syms": 50304,
  "maxlen": 1024,
  "layers": 24,
  "embed_dim": 2048,
  "num_heads": 32,
  "ff_dim": 8192
}

# Distributed setup
GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 4)))

@fsdp(GPUS)
class FSDPTransformer(Transformer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

if __name__ == "__main__":
  print(colored(f"Initializing FSDP model on {len(GPUS)} devices...", "cyan"))
  model = FSDPTransformer(**MODEL_CONFIG)
  
  parameters = nn.state.get_parameters(model)
  print(f"Model initialized with {len(parameters)} parameter tensors.")
  
  # Optimizer knows how to handle sharded parameters
  opt = nn.optim.Adam(parameters, lr=0.001)

  # Load Tiny Shakespeare tokens
  tokens_bin = fetch("https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin")
  with open(tokens_bin, "rb") as f:
    f.seek(0x400) # skip header
    tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
  tokens = Tensor(tokens)

  # Data generation from tokens
  def get_batch(bs=getenv("BS", 8)):
    T = MODEL_CONFIG["maxlen"]
    # Simple random sampling for demonstration
    idxs = np.random.randint(0, len(tokens) - T - 1, size=(bs,))
    x = Tensor.stack(*[tokens[int(i):int(i)+T] for i in idxs])
    y = Tensor.stack(*[tokens[int(i)+1:int(i)+T+1] for i in idxs])
    # Shard data on axis 0 (batch dimension)
    return x.shard(GPUS, axis=0), y.shard(GPUS, axis=0)

  @TinyJit
  def train_step(x, y):
    with Tensor.train():
      opt.zero_grad()
      # model(x) returns (bs, maxlen, syms)
      # we need to flatten for cross_entropy
      out = model.forward(x)
      loss = out.sparse_categorical_crossentropy(y)
      loss.backward()
      opt.step()
      return loss.realize()

  print(colored("Starting training loop...", "green"))
  for i in (t := trange(getenv("STEPS", 10))):
    GlobalCounters.reset()
    x, y = get_batch()
    loss = train_step(x, y)
    t.set_description(f"loss: {loss.item():.4f}")

  print(colored("Training complete!", "green"))
